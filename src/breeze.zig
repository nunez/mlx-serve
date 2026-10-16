//! Breeze-TTS-2 backend (`model_type: breeze_tts`).
//!
//! A multi-codebook TTS stack (dense bf16 packs):
//!   text ids → T5Gemma2 text-encoder conditioning (host-gathered table),
//!   Qwen3 backbone emits codebook-0 @ 12.5 Hz,
//!   `depth_decoder` fills codebooks 1..15 per frame (local depth span),
//!   the qwen3_tts_tokenizer_12hz codec (shared with tts.zig) → 24 kHz PCM.
//!
//! Host-driven frame loop (device step per frame), host-side numpy-order
//! sampling keyed off a counter for determinism. The audio-table / TE-table
//! boundaries are host-gathered; the backbone/depth forwards run on-device.
//! Voice cloning (tokenizer encode) refuses by name until it lands.

const std = @import("std");
const log = @import("log.zig");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const tts = @import("tts.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

pub const model_type = "breeze_tts";

// ── token ids (fixed by the published pack) ────────────────────────────────
pub const AUDIO_TOKEN_ID: u32 = 262144;
pub const AUDIO_EOS_ID: u32 = 262145;
pub const SPEAKER_BASE: u32 = 262146; // [S0]..[S9]
pub const INS_BOS_ID: u32 = 262156;
pub const INS_EOS_ID: u32 = 262157;
pub const EOI_TOKEN_ID: u32 = 256000; // TE row zeroed at gather
pub const TE_VOCAB: usize = 262158;
pub const NUM_CODEBOOKS: usize = 16;
pub const CB_VOCAB: usize = 2051;
pub const CB_PAD: u32 = 2050; // frame-end code (lm_head row 2050)
pub const CB_STRIDE: usize = 2051; // audio-table stride per codebook

// ── geometry (Breeze-TTS-2 / Qwen3-0.6B flavor) ────────────────────────────
pub const TE_HIDDEN: usize = 1152;
pub const TE_HEADS: usize = 4;
pub const TE_KV: usize = 1;
pub const TE_HEAD_DIM: usize = 256;
pub const TE_INTER: usize = 6912;
pub const TE_LAYERS: usize = 26;
pub const TE_WINDOW: usize = 512;
pub const TE_THETA_FULL: f64 = 1_000_000;
pub const TE_THETA_SLIDE: f64 = 10_000;
pub const TE_FACTOR_FULL: f64 = 8.0;

pub const B_HIDDEN: usize = 2048;
pub const B_Q_DIM: usize = 2048;
pub const B_KV_DIM: usize = 1024;
pub const B_HEAD_DIM: usize = 128;
pub const B_INTER: usize = 6144;
pub const B_LAYERS: usize = 28;
pub const B_THETA: f64 = 1_000_000;

pub const D_HIDDEN: usize = 1024;
pub const D_HEADS: usize = 8;
pub const D_KV: usize = 2;
pub const D_HEAD_DIM: usize = 128;
pub const D_INTER: usize = 8192;
pub const D_LAYERS: usize = 12;
pub const D_THETA: f64 = 500_000;

pub const MAX_SEQ: usize = 2048; // span guard: text + frame budget
pub const MAX_FRAMES: u32 = 750; // generation_config max_new_tokens
pub const SAMPLE_RATE: u32 = 24000;
pub const FRAME_RATE: f64 = 12.5;
pub const CHARS_PER_SEC: f64 = 15.0;
pub const FRAME_EST_DIV: f64 = 1.2; // headroom factor over the chars-rate estimate

pub fn isFullAttentionLayer(layer: u32) bool {
    return layer == 5 or layer == 11 or layer == 17 or layer == 23;
}

pub const Config = struct {};

pub fn parseConfig(allocator: std.mem.Allocator, json_text: []const u8) !Config {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_text, .{}) catch return Config{};
    defer parsed.deinit();
    return Config{};
}

/// Voice-design speaker slot: `index` 0..9 maps to the pack's `[S0]..[S9]` ids.
pub fn speakerToken(index: u32) ?u32 {
    if (index >= 10) return null;
    return SPEAKER_BASE + index;
}

/// A request `voice` names a slot S0..S9 (case-insensitive), nothing else.
pub fn voiceSlotValid(name: []const u8) bool {
    if (name.len != 2) return false;
    if (name[0] != 'S' and name[0] != 's') return false;
    return name[1] >= '0' and name[1] <= '9';
}

/// Frame budget from text length: chars / CHARS_PER_SEC × FRAME_RATE, with a
/// headroom factor, clamped by a request cap and MAX_FRAMES.
pub fn estimateFrames(text_len: usize, req_max: ?u32) u32 {
    const est: f64 = @as(f64, @floatFromInt(text_len)) / CHARS_PER_SEC * FRAME_RATE * FRAME_EST_DIV + 4.0;
    var frames: u32 = @intFromFloat(@ceil(est));
    if (req_max) |m| frames = @min(frames, m);
    return @min(frames, MAX_FRAMES);
}

// ───────────────────────────── rng + host sampler ────────────────────────────

pub const Rng = struct {
    state: u64,

    pub fn init(seed: u64) Rng {
        var z = seed ^ 0xA5A5_5A5A_1234_9876;
        if (z == 0) z = 1;
        return .{ .state = z };
    }

    pub fn nextF32(self: *Rng) f32 {
        self.state +%= 0x9E3779B97F4A7C15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        z = z ^ (z >> 31);
        return @as(f32, @floatFromInt(z >> 40)) * (1.0 / 16777216.0); // 24 bits
    }
};

pub const Temp = struct { t: f32 = 0.9, top_k: u32 = 50, top_p: f32 = 1.0 };

/// numpy-order categorical sampler: temperature → top-k (value threshold, the
/// k-th highest, lowest ids keep) → top-p (strict cumulative drop) →
/// multinomial. temperature<=0 → argmax. `scratch`/`order` are ≥ n long.
pub fn sampleToken(rng: *Rng, logits: []const f32, temp: Temp, scratch: []f32, order: []usize) usize {
    const n = logits.len;
    @memcpy(scratch[0..n], logits);
    if (temp.t <= 0) {
        var best: usize = 0;
        var bv: f32 = -std.math.inf(f32);
        for (scratch[0..n], 0..) |v, i| {
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        return best;
    }
    for (scratch[0..n]) |*v| v.* /= temp.t;

    if (temp.top_k > 0 and temp.top_k < n) {
        for (order[0..n], 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order[0..n], scratch, struct {
            fn lt(sc: []const f32, a: usize, b: usize) bool {
                if (sc[a] == sc[b]) return a < b;
                return sc[a] > sc[b];
            }
        }.lt);
        const thr = scratch[order[temp.top_k - 1]];
        var i: usize = temp.top_k;
        while (i < n) : (i += 1) {
            if (scratch[order[i]] < thr) scratch[order[i]] = -std.math.inf(f32);
        }
    }

    var maxv: f32 = -std.math.inf(f32);
    for (scratch[0..n]) |v| maxv = @max(maxv, v);
    var sum: f64 = 0;
    for (scratch[0..n]) |*v| {
        if (std.math.isInf(v.*) and v.* < 0) {
            v.* = 0;
            continue;
        }
        const e: f64 = @exp(@as(f64, @floatCast(v.*)) - @as(f64, @floatCast(maxv)));
        v.* = @floatCast(e);
        sum += e;
    }
    if (sum <= 0) return 0;

    if (temp.top_p < 1.0) {
        for (order[0..n], 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order[0..n], scratch, struct {
            fn lt(sc: []const f32, a: usize, b: usize) bool {
                if (sc[a] == sc[b]) return a < b;
                return sc[a] > sc[b];
            }
        }.lt);
        var cum: f64 = 0;
        var keep: usize = n;
        for (order[0..n], 0..) |idx, oi| {
            cum += @as(f64, @floatCast(scratch[idx])) / sum;
            if (cum >= temp.top_p) {
                keep = oi + 1;
                break;
            }
        }
        var i = keep;
        while (i < n) : (i += 1) scratch[order[i]] = 0;
        sum = 0;
        for (scratch[0..n]) |v| sum += @as(f64, @floatCast(v));
    }

    const r = @as(f64, @floatCast(rng.nextF32())) * sum;
    var acc: f64 = 0;
    for (scratch[0..n], 0..) |v, i| {
        acc += @as(f64, @floatCast(v));
        if (r < acc) return i;
    }
    return n - 1;
}

// ───────────────────────────── device helpers ────────────────────────────────

fn ownT16(w: *const Weights, s: S, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[breeze] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingBreezeWeight;
    };
    var t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&t, arr, &[_]c_int{ 1, 0 }, 2, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, t, .float16, s));
    return out;
}

fn ownF16(w: *const Weights, s: S, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[breeze] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingBreezeWeight;
    };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, arr, .float16, s));
    return out;
}

fn hostF32(data: []const f32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(@constCast(@ptrCast(data.ptr)), shape.ptr, @intCast(shape.len), .float32);
}

fn hostU32(data: []const u32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(@constCast(@ptrCast(data.ptr)), shape.ptr, @intCast(shape.len), .uint32);
}

const null_arr = mlx.mlx_array{ .ctx = null };

fn evalHost(s: S, arr: mlx.mlx_array, out: []f32) !void {
    var f32arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
    _ = mlx.mlx_array_eval(f32arr);
    const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
    @memcpy(out, data[0..out.len]);
}

inline fn mm(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_matmul(&r, x, w_t, s);
    return r;
}

inline fn addM(a: mlx.mlx_array, b: mlx.mlx_array, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_add(&r, a, b, s);
    return r;
}

inline fn rmsn(x: mlx.mlx_array, g: mlx.mlx_array, eps: f32, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_fast_rms_norm(&r, x, g, eps, s);
    return r;
}

inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_reshape(&r, x, shape.ptr, @intCast(shape.len), s);
    return r;
}

inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_transpose_axes(&r, x, axes.ptr, @intCast(axes.len), s);
    return r;
}

fn siluArr(x: mlx.mlx_array, s: S) mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    _ = mlx.mlx_sigmoid(&sig, x, s);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&r, x, sig, s);
    return r;
}

/// RoPE via the house fused op (interleaved HF convention).
fn ropeApply(x: mlx.mlx_array, theta: f64, scale: f64, hd: c_int, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_fast_rope(&r, x, hd, false, .{ .value = @floatCast(theta), .has_value = true }, @floatCast(scale), 0, null_arr, s);
    return r;
}

fn sdpa(q4: mlx.mlx_array, k4: mlx.mlx_array, v4: mlx.mlx_array, scale: f32, mask: mlx.mlx_array, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    const mode: [*:0]const u8 = if (mask.ctx == null) "causal" else "array";
    _ = mlx.mlx_fast_scaled_dot_product_attention(&r, q4, k4, v4, scale, mode, mask, null_arr, false, s);
    return r;
}

// ─────────────────────────── qwen stack (generic) ────────────────────────────

const QLayer = struct {
    q: mlx.mlx_array, // [in, q_dim] f16 (pre-transposed)
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    o: mlx.mlx_array,
    qn: mlx.mlx_array, // zero-length array = no qk-norm (depth decoder)
    kn: mlx.mlx_array,
    gate: mlx.mlx_array,
    up: mlx.mlx_array,
    down: mlx.mlx_array,
    in_ln: mlx.mlx_array,
    post_ln: mlx.mlx_array,
    pre_ff: mlx.mlx_array, // NULL on backbone/depth; TE carries the 4th norm
};

/// Full re-forward over a host [L, hidden] f32 embedding span. `pos_scale`
/// scales RoPE positions (TE full layers), optional causal local-window mask
/// (`window` 0 = plain causal). qk-norm applies when qn has data.
fn qwenForward(s: S, layers: []const QLayer, norm: mlx.mlx_array, eps: f32, theta: f64, x_host: []const f32, L: usize, hidden: usize, pos_scale: f64, window: usize, out_last: []f32) !void {
    const sh = [_]c_int{ 1, @intCast(L), @intCast(hidden) };
    var x = hostF32(x_host, &sh);
    var x16 = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&x16, x, .float16, s);
    x = x16;

    for (layers) |lw| {
        const xn = rmsn(x, lw.in_ln, eps, s);
        const q = mm(xn, lw.q, s);
        const k = mm(xn, lw.k, s);
        const v = mm(xn, lw.v, s);
        const qh: c_int = @intCast(@divExact(mlx.getShape(lw.q)[1], 128));
        const kh: c_int = @intCast(@divExact(mlx.getShape(lw.k)[1], 128));
        var q4 = reshape(q, &[_]c_int{ 1, @intCast(L), qh, 128 }, s);
        var k4 = reshape(k, &[_]c_int{ 1, @intCast(L), kh, 128 }, s);
        const v4 = reshape(v, &[_]c_int{ 1, @intCast(L), kh, 128 }, s);
        const lq: usize = L * @as(usize, @intCast(qh));
        const lk: usize = L * @as(usize, @intCast(kh));
        const hd: c_int = 128;
        if (lw.qn.ctx != null and mlx.getShape(lw.qn)[0] > 0) {
            q4 = reshape(rmsn(reshape(q4, &[_]c_int{ @intCast(lq), hd }, s), lw.qn, eps, s), &[_]c_int{ 1, @intCast(L), qh, hd }, s);
            k4 = reshape(rmsn(reshape(k4, &[_]c_int{ @intCast(lk), hd }, s), lw.kn, eps, s), &[_]c_int{ 1, @intCast(L), kh, hd }, s);
        }
        const qt = transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const kt = transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const vt = transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const qr = ropeApply(qt, theta, pos_scale, 128, s);
        const kr = ropeApply(kt, theta, pos_scale, 128, s);
        const mask = if (window > 0) windowMask(s, L, window) else null_arr;
        const att = sdpa(qr, kr, vt, @as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(128))))), mask, s);
        const at = transpose(att, &[_]c_int{ 0, 2, 1, 3 }, s);
        const af = reshape(at, &[_]c_int{ 1, @intCast(L), @intCast(mlx.getShape(lw.q)[1]) }, s);
        const h1 = addM(x, mm(af, lw.o, s), s);
        const hn = rmsn(h1, lw.post_ln, eps, s);
        const g = mm(hn, lw.gate, s);
        const u = mm(hn, lw.up, s);
        const mlp = mm(mulEl(s, siluArr(g, s), u), lw.down, s);
        x = addM(h1, mlp, s);
    }
    x = rmsn(x, norm, eps, s);
    const last = reshape(x, &[_]c_int{ @intCast(L), @intCast(hidden) }, s);
    var idx = [_]u32{@intCast(L - 1)};
    var taken = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&taken, last, hostU32(&idx, &[_]c_int{1}), 0, s));
    try evalHost(s, taken, out_last);
}

fn mulEl(s: S, a: mlx.mlx_array, b: mlx.mlx_array) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&r, a, b, s);
    return r;
}

fn windowMask(s: S, L: usize, w: usize) mlx.mlx_array {
    const buf = std.heap.page_allocator.alloc(f32, L * L) catch return null_arr;
    defer std.heap.page_allocator.free(buf);
    for (0..L) |i| {
        for (0..L) |j| {
            const allowed = j <= i and (i - j) < w;
            buf[i * L + j] = if (allowed) 0 else -std.math.inf(f32);
        }
    }
    const h = hostF32(buf, &[_]c_int{ 1, 1, @intCast(L), @intCast(L) });
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&r, h, .float16, s);
    return r;
}

// ───────────────────────────── text encoder ──────────────────────────────────

var te_buf: []f32 = &[_]f32{};

const Te = struct {
    layers: []QLayer, // reuse layout (q [in,1024] etc via ownT16)
    norm: mlx.mlx_array,
    table: []const f32, // HOST [TE_VOCAB, TE_HIDDEN] ×√hidden, eoi zeroed

    fn load(w: *const Weights, s: S, alloc: std.mem.Allocator) !Te {
        const layers = try alloc.alloc(QLayer, TE_LAYERS);
        for (layers, 0..) |*L, i| {
            const p = try std.fmt.allocPrint(alloc, "text_encoder.layers.{d}.", .{i});
            defer alloc.free(p);
            L.* = .{
                .q = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.q_proj.weight", .{p})),
                .k = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.k_proj.weight", .{p})),
                .v = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.v_proj.weight", .{p})),
                .o = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.o_proj.weight", .{p})),
                .qn = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.q_norm.weight", .{p})),
                .kn = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.k_norm.weight", .{p})),
                .gate = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.gate_proj.weight", .{p})),
                .up = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.up_proj.weight", .{p})),
                .down = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.down_proj.weight", .{p})),
                .in_ln = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}pre_self_attn_layernorm.weight", .{p})),
                .post_ln = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}post_feedforward_layernorm.weight", .{p})),
                .pre_ff = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}pre_feedforward_layernorm.weight", .{p})),
            };
            // post_self_attn_layernorm rides alongside as the attn post norm.
            const ps = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}post_self_attn_layernorm.weight", .{p}));
            te_post_attn = appendHandle(te_post_attn, alloc, ps);
        }
        return .{
            .layers = layers,
            .norm = try ownF16(w, s, "text_encoder.norm.weight"),
            .table = &[_]f32{},
        };
    }

    fn loadTable(w: *const Weights, s: S, alloc: std.mem.Allocator) ![]const f32 {
        const arr = w.get("text_encoder.embed_tokens.weight") orelse return error.MissingBreezeWeight;
        var f32arr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
        _ = mlx.mlx_array_eval(f32arr);
        const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
        const out = try alloc.alloc(f32, TE_VOCAB * TE_HIDDEN);
        const scale: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(TE_HIDDEN))));
        for (0..TE_VOCAB) |r| {
            const src = data[r * TE_HIDDEN ..][0..TE_HIDDEN];
            const dst = out[r * TE_HIDDEN ..][0..TE_HIDDEN];
            if (r == EOI_TOKEN_ID) {
                @memset(dst, 0);
            } else {
                for (src, dst) |sv, *dv| dv.* = sv * scale;
            }
        }
        return out;
    }
};

var te_post_attn: []mlx.mlx_array = &[_]mlx.mlx_array{};

fn appendHandle(buf: []mlx.mlx_array, alloc: std.mem.Allocator, h: mlx.mlx_array) []mlx.mlx_array {
    const out = alloc.alloc(mlx.mlx_array, buf.len + 1) catch return buf;
    if (buf.len > 0) {
        @memcpy(out[0..buf.len], buf);
        alloc.free(buf);
    }
    out[out.len - 1] = h;
    return out;
}

/// TE conditioning over text ids → host rows [L, TE_HIDDEN].
fn teCond(alloc: std.mem.Allocator, s: S, te: *const Te, ids: []const u32, out: []f32) !void {
    const L = ids.len;
    var buf = alloc.alloc(f32, L * TE_HIDDEN) catch return error.OutOfMemory;
    defer alloc.free(buf);
    for (ids, 0..) |id, i| {
        if (id >= TE_VOCAB) return error.TokenBeyondTeVocab;
        @memcpy(buf[i * TE_HIDDEN ..][0..TE_HIDDEN], te.table[@as(usize, id) * TE_HIDDEN ..][0..TE_HIDDEN]);
    }
    const sh = [_]c_int{ 1, @intCast(L), @intCast(TE_HIDDEN) };
    var x = hostF32(buf, &sh);
    var x16 = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&x16, x, .float16, s);
    x = x16;

    for (te.layers, te_post_attn, 0..) |lw, ps, li| {
        const full = isFullAttentionLayer(@intCast(li));
        const xn = rmsn(x, lw.in_ln, 1e-6, s);
        const q = mm(xn, lw.q, s);
        const k = mm(xn, lw.k, s);
        const v = mm(xn, lw.v, s);
        const qh: c_int = @divExact(mlx.getShape(lw.q)[1], @as(c_int, @intCast(TE_HEAD_DIM)));
        const kh: c_int = @divExact(mlx.getShape(lw.k)[1], @as(c_int, @intCast(TE_HEAD_DIM)));
        var q4 = reshape(q, &[_]c_int{ 1, @intCast(L), qh, @intCast(TE_HEAD_DIM) }, s);
        var k4 = reshape(k, &[_]c_int{ 1, @intCast(L), kh, @intCast(TE_HEAD_DIM) }, s);
        const v4 = reshape(v, &[_]c_int{ 1, @intCast(L), kh, @intCast(TE_HEAD_DIM) }, s);
        const lq: usize = L * @as(usize, @intCast(qh));
        const lk: usize = L * @as(usize, @intCast(kh));
        const hd: c_int = @intCast(TE_HEAD_DIM);
        q4 = reshape(rmsn(reshape(q4, &[_]c_int{ @intCast(lq), hd }, s), lw.qn, 1e-6, s), &[_]c_int{ 1, @intCast(L), qh, hd }, s);
        k4 = reshape(rmsn(reshape(k4, &[_]c_int{ @intCast(lk), hd }, s), lw.kn, 1e-6, s), &[_]c_int{ 1, @intCast(L), kh, hd }, s);
        const qt = transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const kt = transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const vt = transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const theta = if (full) TE_THETA_FULL else TE_THETA_SLIDE;
        const p_scale = if (full) (1.0 / TE_FACTOR_FULL) else 1.0;
        const qr = ropeApply(qt, theta, p_scale, @intCast(TE_HEAD_DIM), s);
        const kr = ropeApply(kt, theta, p_scale, @intCast(TE_HEAD_DIM), s);
        const mask = if (full) null_arr else teLocalMask(s, L);
        const att = sdpa(qr, kr, vt, @as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(TE_HEAD_DIM))))), mask, s);
        const at = transpose(att, &[_]c_int{ 0, 2, 1, 3 }, s);
        const af = reshape(at, &[_]c_int{ 1, @intCast(L), @intCast(TE_HEADS * TE_HEAD_DIM) }, s);
        const h1 = addM(x, rmsn(mm(af, lw.o, s), ps, 1e-6, s), s);
        const hn = rmsn(h1, lw.pre_ff, 1e-6, s);
        const g = mm(hn, lw.gate, s);
        const u = mm(hn, lw.up, s);
        const act = geluTanhArr(g, s);
        x = addM(h1, rmsn(mm(mulEl(s, act, u), lw.down, s), lw.post_ln, 1e-6, s), s);
    }
    var xn = rmsn(x, te.norm, 1e-6, s);
    _ = &xn;
    const last = reshape(xn, &[_]c_int{ @intCast(L), @intCast(TE_HIDDEN) }, s);
    try evalHost(s, last, out);
}

fn teLocalMask(s: S, L: usize) mlx.mlx_array {
    const buf = std.heap.page_allocator.alloc(f32, L * L) catch return null_arr;
    defer std.heap.page_allocator.free(buf);
    const w = TE_WINDOW;
    for (0..L) |i| {
        for (0..L) |j| {
            const left: usize = if (j <= i) i - j else 0;
            const right: usize = if (j > i) j - i else 0;
            const allowed = left < w and right <= w;
            buf[i * L + j] = if (allowed) 0 else -std.math.inf(f32);
        }
    }
    const h = hostF32(buf, &[_]c_int{ 1, 1, @intCast(L), @intCast(L) });
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&r, h, .float16, s);
    return r;
}

fn geluTanhArr(x: mlx.mlx_array, s: S) mlx.mlx_array {
    // 0.5·x·(1+tanh(√(2/π)(x+0.044715x³))) — TE hidden_activation
    var x2 = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&x2, x, x, s);
    var x3 = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&x3, x2, x, s);
    var k = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&k, x3, mlx.mlx_array_new_float(0.044715), s);
    var inner = mlx.mlx_array_new();
    _ = mlx.mlx_add(&inner, x, k, s);
    var lin = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&lin, inner, mlx.mlx_array_new_float(0.7978845608), s);
    var th = mlx.mlx_array_new();
    _ = mlx.mlx_tanh(&th, lin, s);
    var one = mlx.mlx_array_new();
    _ = mlx.mlx_add(&one, th, mlx.mlx_array_new_float(1.0), s);
    var half = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&half, one, mlx.mlx_array_new_float(0.5), s);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&r, x, half, s);
    return r;
}

// ───────────────────────── heads (host) ──────────────────────────────────────

fn readTable(s: S, w: *const Weights, key: []const u8, out: []f32) !void {
    const arr = w.get(key) orelse {
        log.err("[breeze] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingBreezeWeight;
    };
    var f32arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
    _ = mlx.mlx_array_eval(f32arr);
    const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
    @memcpy(out, data[0..out.len]);
}

/// host logits [cb] over canonical [out,in] host row set.
fn headLogits(host: []const f32, hidden: []const f32, out: []f32, vocab: usize, dim: usize, slot: usize) void {
    const base = slot * vocab * dim;
    for (0..vocab) |o| {
        var acc: f32 = 0;
        const row = host[base + o * dim ..][0..dim];
        for (hidden, row) |hv, wv| acc += hv * wv;
        out[o] = acc;
    }
}

// ───────────────────────────── Engine API ────────────────────────────────────

pub const Request = struct {
    text: []const u8 = "",
    voice: ?[]const u8 = null, // "S0".."S9" voice-design slot
    instruction: ?[]const u8 = null,
    ref_audio: ?[]const u8 = null, // WAV bytes — clone refused until Mimi encode lands
    max_frames: ?u32 = null,
    seed: ?u64 = null,
};

pub const Synthesis = struct {
    pcm: []f32,
    frames: []const [NUM_CODEBOOKS]u32,
};

pub const Engine = struct {
    alloc: std.mem.Allocator,
    tok: tokenizer_mod.Tokenizer,
    te: Te,
    te_proj: mlx.mlx_array, // [1152, 2048] f16 pre-transposed
    b_layers: []QLayer,
    b_norm: mlx.mlx_array,
    d_layers: []QLayer,
    d_norm: mlx.mlx_array,
    d_inp: mlx.mlx_array, // depth input projection [2048, 1024] f16
    lm_host: []f32, // [CB_VOCAB, B_HIDDEN] canonical (audio rows only)
    heads_host: []f32, // [15, CB_VOCAB, D_HIDDEN]
    audio_table: []f32, // HOST [NUM_CODEBOOKS*CB_STRIDE, B_HIDDEN]
    codec: tts.CodecDecoder,
    scratch: []f32,
    order: []usize,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.InvalidModelDir;
        const s = mlx.mlx_default_gpu_stream_new();
        g_stream = s;
        mlx_allocator = allocator;

        var self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        var w = try model_mod.loadWeights(io, allocator, model_dir);
        defer w.deinit();

        var te = try Te.load(&w, s, allocator);
        errdefer allocator.free(te.layers);
        te.table = try Te.loadTable(&w, s, allocator);
        errdefer allocator.free(te.table);

        const lm_host = try allocator.alloc(f32, CB_VOCAB * B_HIDDEN);
        errdefer allocator.free(lm_host);
        // lm_head: torch [2052, 2048] audio rows 0..CB_VOCAB-1, canonical.
        {
            const arr = w.get("lm_head.weight") orelse return error.MissingBreezeWeight;
            var f32arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
            _ = mlx.mlx_array_eval(f32arr);
            const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
            for (0..CB_VOCAB) |r| @memcpy(lm_host[r * B_HIDDEN ..][0..B_HIDDEN], data[r * B_HIDDEN ..][0..B_HIDDEN]);
        }

        const heads_host = try allocator.alloc(f32, (NUM_CODEBOOKS - 1) * CB_VOCAB * D_HIDDEN);
        errdefer allocator.free(heads_host);
        // codebooks_head [15, 1024, 2051] — canonicalize to [15, out, in].
        {
            const arr = w.get("depth_decoder.codebooks_head.weight") orelse return error.MissingBreezeWeight;
            var f32arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
            _ = mlx.mlx_array_eval(f32arr);
            const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
            const shp = mlx.getShape(arr);
            if (shp.len == 3 and shp[1] == CB_VOCAB) {
                @memcpy(heads_host, data[0..heads_host.len]);
            } else {
                var h: usize = 0;
                while (h < NUM_CODEBOOKS - 1) : (h += 1) {
                    for (0..D_HIDDEN) |i| {
                        for (0..CB_VOCAB) |o| {
                            heads_host[h * CB_VOCAB * D_HIDDEN + o * D_HIDDEN + i] = data[h * D_HIDDEN * CB_VOCAB + i * CB_VOCAB + o];
                        }
                    }
                }
            }
        }

        const audio_table = try allocator.alloc(f32, NUM_CODEBOOKS * CB_STRIDE * B_HIDDEN);
        errdefer allocator.free(audio_table);
        try readTable(s, &w, "depth_decoder.model.embed_tokens.weight", audio_table);

        self.* = .{
            .alloc = allocator,
            .tok = try tokenizer_mod.loadTokenizer(io, allocator, model_dir),
            .te = te,
            .te_proj = undefined,
            .b_layers = try loadQLayers("backbone_model.layers.{d}.", &w, allocator, s, B_LAYERS, true),
            .b_norm = try ownF16(&w, s, "backbone_model.norm.weight"),
            .d_layers = try loadQLayers("depth_decoder.model.layers.{d}.", &w, allocator, s, D_LAYERS, false),
            .d_norm = try ownF16(&w, s, "depth_decoder.model.norm.weight"),
            .d_inp = try ownT16(&w, s, "depth_decoder.model.inputs_embeds_projector.weight"),
            .lm_host = lm_host,
            .heads_host = heads_host,
            .audio_table = audio_table,
            .codec = try tts.loadCodecDecoderFrom(io, allocator, s, try std.fmt.allocPrint(allocator, "{s}/audio_tokenizer", .{model_dir})),
            .scratch = try allocator.alloc(f32, CB_VOCAB),
            .order = try allocator.alloc(usize, CB_VOCAB),
        };
        errdefer {
            allocator.free(self.scratch);
            allocator.free(self.order);
        }
        self.te_proj = try ownT16(&w, s, "text_encoder_proj.weight");
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.tok.deinit();
        self.codec.deinit();
        self.alloc.free(self.te.table);
        self.alloc.free(self.te.layers);
        self.alloc.free(self.lm_host);
        self.alloc.free(self.heads_host);
        self.alloc.free(self.audio_table);
        self.alloc.free(self.scratch);
        self.alloc.free(self.order);
        self.alloc.destroy(self);
    }

    pub fn sampleRate(self: *const Engine) u32 {
        _ = self;
        return SAMPLE_RATE;
    }

    fn enc(self: *Engine, out: *std.ArrayList(u32), text: []const u8) !void {
        const ids = try self.tok.encode(self.alloc, text);
        defer self.alloc.free(ids);
        try out.appendSlice(self.alloc, ids);
    }

    /// Plain / tagged / instruction synthesis.
    pub fn synthesize(self: *Engine, allocator: std.mem.Allocator, req: Request) !Synthesis {
        if (req.ref_audio != null and req.ref_audio.?.len > 0) return error.CloneNotSupported;
        if (req.text.len == 0) return error.EmptyText;
        var rng = Rng.init(req.seed orelse 42);

        // text span
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        if (req.voice) |v| {
            if (!voiceSlotValid(v)) return error.UnknownVoice;
            try ids.append(allocator, SPEAKER_BASE + @as(u32, v[1] - '0'));
        }
        if (req.instruction) |ins| {
            try ids.append(allocator, INS_BOS_ID);
            try self.enc(&ids, ins);
            try ids.append(allocator, INS_EOS_ID);
        }
        try self.enc(&ids, req.text);

        const text_len = ids.items.len;
        const frames_cap = estimateFrames(text_len, req.max_frames);
        const total = text_len + frames_cap;
        if (total > MAX_SEQ) return error.PromptTooLongForFrameBudget;

        // TE conditioning → project to backbone width (host rows)
        var cond = allocator.alloc(f32, text_len * TE_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(cond);
        try teCond(allocator, g_stream, &self.te, ids.items, cond);
        // stream embeds [total, B_HIDDEN] host f32
        var stream = allocator.alloc(f32, total * B_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(stream);
        @memset(stream, 0);
        {
            // te_proj host: read device weight once
            const wt = allocator.alloc(f32, TE_HIDDEN * B_HIDDEN) catch return error.OutOfMemory;
            defer allocator.free(wt);
            try evalHost(g_stream, self.te_proj, wt);
            for (0..text_len) |t| {
                const row = cond[t * TE_HIDDEN ..][0..TE_HIDDEN];
                for (0..B_HIDDEN) |o| {
                    var acc: f32 = 0;
                    var i: usize = 0;
                    while (i < TE_HIDDEN) : (i += 1) acc += row[i] * wt[i * B_HIDDEN + o];
                    stream[t * B_HIDDEN + o] = acc;
                }
            }
        }

        var frames: std.ArrayList([NUM_CODEBOOKS]u32) = .empty;
        errdefer frames.deinit(allocator);

        const h_row = allocator.alloc(f32, B_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(h_row);
        var h_proj = allocator.alloc(f32, D_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(h_proj);
        const emb_row = allocator.alloc(f32, B_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(emb_row);
        const logits = allocator.alloc(f32, CB_VOCAB) catch return error.OutOfMemory;
        defer allocator.free(logits);
        var span = allocator.alloc(f32, total * B_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(span);
        var depth_span = allocator.alloc(f32, (NUM_CODEBOOKS + 1) * D_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(depth_span);

        var frame_i: usize = 0;
        while (frame_i < frames_cap) : (frame_i += 1) {
            const L = text_len + frame_i;
            @memcpy(span[0 .. L * B_HIDDEN], stream[0 .. L * B_HIDDEN]);
            try qwenForward(g_stream, self.b_layers, self.b_norm, 1e-6, B_THETA, span[0 .. L * B_HIDDEN], L, B_HIDDEN, 1.0, 0, h_row);

            headLogits(self.lm_host, h_row, logits, CB_VOCAB, B_HIDDEN, 0);
            var frame: [NUM_CODEBOOKS]u32 = undefined;
            frame[0] = @intCast(sampleToken(&rng, logits, .{}, self.scratch, self.order));
            if (frame[0] == CB_PAD) break;

            // depth span: [backbone_hidden_proj, emb(cb0), emb(cb1), ...]
            try projHostCached(allocator, g_stream, self.d_inp, h_row, B_HIDDEN, D_HIDDEN, h_proj[0..D_HIDDEN]);
            @memcpy(depth_span[0..D_HIDDEN], h_proj);
            var dl: usize = 1;
            var cb: usize = 1;
            while (cb < NUM_CODEBOOKS) : (cb += 1) {
                const code = frame[cb - 1];
                tableRow(self.audio_table, B_HIDDEN, cb - 1, @min(code, @as(u32, CB_VOCAB - 1)), emb_row);
                try projHostCached(allocator, g_stream, self.d_inp, emb_row, B_HIDDEN, D_HIDDEN, depth_span[dl * D_HIDDEN ..][0..D_HIDDEN]);
                dl += 1;
                const d_last = allocator.alloc(f32, D_HIDDEN) catch return error.OutOfMemory;
                defer allocator.free(d_last);
                try qwenForward(g_stream, self.d_layers, self.d_norm, 1e-5, D_THETA, depth_span[0 .. dl * D_HIDDEN], dl, D_HIDDEN, 1.0, 0, d_last);
                headLogits(self.heads_host, d_last, logits, CB_VOCAB, D_HIDDEN, cb - 1);
                frame[cb] = @intCast(sampleToken(&rng, logits, .{}, self.scratch, self.order));
            }

            // frame embed row into the stream
            frameEmbed(self.audio_table, B_HIDDEN, &frame, emb_row);
            @memcpy(stream[(text_len + frame_i) * B_HIDDEN ..][0..B_HIDDEN], emb_row);
            try frames.append(allocator, frame);
        }

        var out_frames = frames.items;
        while (out_frames.len > 0 and out_frames[out_frames.len - 1][0] == CB_PAD) out_frames = out_frames[0 .. out_frames.len - 1];
        if (out_frames.len == 0) return error.NoFramesEmitted;

        const pcm = try self.codec.decode(out_frames);
        return .{ .pcm = pcm, .frames = out_frames };
    }
};

var mlx_allocator: std.mem.Allocator = undefined;
var g_stream: S = undefined;

fn tableRow(table: []const f32, width: usize, cb: usize, code: u32, out: []f32) void {
    @memcpy(out, table[(@as(usize, cb) * CB_STRIDE + @as(usize, code)) * width ..][0..width]);
}

fn frameEmbed(table: []const f32, width: usize, codes: []const u32, out: []f32) void {
    @memset(out, 0);
    for (codes, 0..) |code, cb| {
        const c = @min(code, @as(u32, CB_VOCAB - 1));
        const row = table[(@as(usize, cb) * CB_STRIDE + @as(usize, c)) * width ..][0..width];
        for (out, row) |*o, rv| o.* += rv;
    }
}

var d_inp_cache: []f32 = &[_]f32{};

fn projHostCached(alloc: std.mem.Allocator, s: S, w: mlx.mlx_array, x: []const f32, in: usize, out_ch: usize, out: []f32) !void {
    if (d_inp_cache.len != in * out_ch) {
        d_inp_cache = alloc.alloc(f32, in * out_ch) catch return error.OutOfMemory;
        try evalHost(s, w, d_inp_cache);
    }
    for (0..out_ch) |o| {
        var acc: f32 = 0;
        var i: usize = 0;
        while (i < in) : (i += 1) acc += x[i] * d_inp_cache[i * out_ch + o];
        out[o] = acc;
    }
}

fn projHost(alloc: std.mem.Allocator, s: S, w: mlx.mlx_array, x: []const f32, in: usize, out_ch: usize, out: []f32) !void {
    // w is [in, out] f16 on-device; eval to host once
    const host = alloc.alloc(f32, in * out_ch) catch return error.OutOfMemory;
    defer alloc.free(host);
    try evalHost(s, w, host);
    for (0..out_ch) |o| {
        var acc: f32 = 0;
        var i: usize = 0;
        while (i < in) : (i += 1) acc += x[i] * host[i * out_ch + o];
        out[o] = acc;
    }
}

fn loadQLayers(comptime prefix_fmt: []const u8, w: *const Weights, alloc: std.mem.Allocator, s: S, count: usize, qk_norm: bool) ![]QLayer {
    const layers = try alloc.alloc(QLayer, count);
    for (layers, 0..) |*L, i| {
        const p = try std.fmt.allocPrint(alloc, prefix_fmt, .{i});
        defer alloc.free(p);
        L.* = .{
            .q = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.q_proj.weight", .{p})),
            .k = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.k_proj.weight", .{p})),
            .v = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.v_proj.weight", .{p})),
            .o = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.o_proj.weight", .{p})),
            .qn = if (qk_norm) try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.q_norm.weight", .{p})) else null_arr,
            .kn = if (qk_norm) try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.k_norm.weight", .{p})) else null_arr,
            .gate = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.gate_proj.weight", .{p})),
            .up = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.up_proj.weight", .{p})),
            .down = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.down_proj.weight", .{p})),
            .in_ln = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}input_layernorm.weight", .{p})),
            .post_ln = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}post_attention_layernorm.weight", .{p})),
            .pre_ff = null_arr,
        };
    }
    return layers;
}

// ───────────────────────────── tests ─────────────────────────────────────────

test "voice slot ids" {
    try std.testing.expectEqual(@as(?u32, SPEAKER_BASE), speakerToken(0));
    try std.testing.expectEqual(@as(?u32, SPEAKER_BASE + 9), speakerToken(9));
    try std.testing.expectEqual(@as(?u32, null), speakerToken(10));
}

test "voice slot validation" {
    try std.testing.expect(voiceSlotValid("S0"));
    try std.testing.expect(voiceSlotValid("s7"));
    try std.testing.expect(!voiceSlotValid("S10"));
    try std.testing.expect(!voiceSlotValid("S"));
    try std.testing.expect(!voiceSlotValid("V0"));
}

test "estimateFrames clamp" {
    const a = estimateFrames(300, null);
    try std.testing.expect(a > 0);
    try std.testing.expect(a <= MAX_FRAMES);
    const b = estimateFrames(300, 50);
    try std.testing.expectEqual(@as(u32, 50), @min(b, 50));
}

test "sampler deterministic + temp0 argmax" {
    const logits = [_]f32{ 0.1, 5.0, 0.2, 0.3 };
    var scratch: [4]f32 = undefined;
    var order: [4]usize = undefined;
    var r1 = Rng.init(7);
    var r2 = Rng.init(7);
    const g1 = sampleToken(&r1, &logits, .{ .t = 0 }, &scratch, &order);
    try std.testing.expectEqual(@as(usize, 1), g1);
    var s2: [4]f32 = undefined;
    var o2: [4]usize = undefined;
    const a = sampleToken(&r1, &logits, .{}, &scratch, &order);
    const b = sampleToken(&r2, &logits, .{}, &s2, &o2);
    try std.testing.expectEqual(a, b);
}

test "frameEmbed sums table rows" {
    // minimal stride-two stand-in: cb0=[1,2], cb1=[3,4]
    var stub = [_]f32{ 1, 2, 3, 4 };
    var out = [_]f32{ 0, 0 };
    for ([_]u32{ 0, 0 }, 0..) |c, cb| {
        const row = stub[@as(usize, cb) * 2 + @as(usize, c) * 0 ..][0..2];
        for (&out, row) |*o, rv| o.* += rv;
    }
    try std.testing.expectEqual(@as(f32, 4), out[0]);
    try std.testing.expectEqual(@as(f32, 6), out[1]);
}

test "live synthesize (MLX_SERVE_BREEZE_MODEL)" {
    const dir = std.mem.span(std.c.getenv("MLX_SERVE_BREEZE_MODEL") orelse return error.SkipZigTest);
    const alloc = std.testing.allocator;
    var engine = try Engine.load(std.Io.Threaded.global_single_threaded.io(), alloc, dir);
    defer engine.deinit();
    const synth = try engine.synthesize(alloc, .{ .text = "Hello world, this is Breeze speaking.", .voice = "S0", .seed = 1 });
    try std.testing.expect(synth.pcm.len > 0);
    alloc.free(synth.pcm);
}
