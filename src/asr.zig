//! Voxtral-Mini-4B-Realtime ASR engine (fleet media lane, `/v1/audio/transcriptions`).
//!
//! Faithful port of the mlx-audio `voxtral_realtime` reference (read from the
//! installed omlx venv: `mlx_audio/stt/models/voxtral_realtime/*.py`). The
//! pipeline: WAV (any rate, mixed to mono, resampled to 16 kHz) → left/right
//! silence pad → log-mel spectrogram (Slaney filterbank, Hann window 400, hop
//! 160, clamp 1.5) → causal conv stem (128→1280 k3 s1, 1280→1280 k3 s2) →
//! 32-layer causal transformer (sliding window 750, interleaved RoPE θ=1e6,
//! fp biases on wq/wv/wo and w2) → 4× downsample → adapter MLP → decoder
//! prompt `[BOS] + [STREAMING_PAD]×(32 + n_delay)` added to the audio
//! embeddings → 26-layer GQA decoder (sliding window 8192, ada-RMS-Norm
//! t_cond, tied embeddings) → greedy decode to EOS / the audio span end →
//! Tekken decode → reply JSON `{"text","language","duration"}`.
//!
//! Ownership: TRANSFER. Every mlx op creates a fresh C wrapper owned by its
//! caller; a function returning an op array hands the wrapper to its caller
//! and must not free it (each mlx_array struct owns its own C++ wrapper, and
//! struct copies share it — only the owning handle may free).

const std = @import("std");
const mlx = @import("mlx.zig");
const laya = @import("laya.zig");
const wav_mod = @import("wav.zig");
const log = @import("log.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const none = mlx.mlx_array{ .ctx = null };
const free = laya.free;

fn managedFreeF32(p: ?*anyopaque) callconv(.c) void {
    if (p) |pp| {
        const ptr: [*]f32 = @ptrCast(@alignCast(pp));
        std.heap.page_allocator.free(ptr[0..0]);
    }
}

fn managedFreeF16(p: ?*anyopaque) callconv(.c) void {
    if (p) |pp| {
        const ptr: [*]f16 = @ptrCast(@alignCast(pp));
        std.heap.page_allocator.free(ptr[0..0]);
    }
}

fn managedFreeU16(p: ?*anyopaque) callconv(.c) void {
    if (p) |pp| {
        const ptr: [*]u16 = @ptrCast(@alignCast(pp));
        std.heap.page_allocator.free(ptr[0..0]);
    }
}

fn payloadNoop(_: ?*anyopaque) callconv(.c) void {}

const SAMPLE_RATE: f32 = 16000;
const FRAME_RATE: f32 = 12.5;
const RAW_AUDIO_LENGTH_PER_TOK: u32 = @intFromFloat(SAMPLE_RATE / FRAME_RATE); // 1280
const AUDIO_LENGTH_PER_TOK: u32 = RAW_AUDIO_LENGTH_PER_TOK / 160; // 8

// ── config ─────────────────────────────────────────────────────────────────

pub const AudioEncodingConfig = struct {
    sampling_rate: u32 = 16000,
    frame_rate: f32 = 12.5,
    num_mel_bins: u32 = 128,
    hop_length: u32 = 160,
    window_size: u32 = 400,
    global_log_mel_max: f32 = 1.5,
};

pub const EncoderConfig = struct {
    dim: u32 = 1280,
    n_layers: u32 = 32,
    n_heads: u32 = 32,
    head_dim: u32 = 64,
    hidden_dim: u32 = 5120,
    n_kv_heads: u32 = 32,
    norm_eps: f32 = 1e-5,
    rope_theta: f64 = 1_000_000.0,
    sliding_window: u32 = 750,
    causal: bool = true,
    use_biases: bool = true,
    downsample_factor: u32 = 4,
};

pub const DecoderConfig = struct {
    dim: u32 = 3072,
    n_layers: u32 = 26,
    n_heads: u32 = 32,
    n_kv_heads: u32 = 8,
    head_dim: u32 = 128,
    hidden_dim: u32 = 9216,
    vocab_size: u32 = 131072,
    norm_eps: f32 = 1e-5,
    rope_theta: f64 = 1_000_000.0,
    sliding_window: u32 = 8192,
    tied_embeddings: bool = true,
    ada_rms_norm_t_cond: bool = true,
    ada_rms_norm_t_cond_dim: u32 = 32,
};

pub const ModelConfig = struct {
    model_type: []const u8 = "voxtral_realtime",
    encoder_args: EncoderConfig = .{},
    decoder: DecoderConfig = .{},
    audio_encoding_args: AudioEncodingConfig = .{},
    transcription_delay_ms: u32 = 480,
    vocab_size: u32 = 131072,
    hidden_size: u32 = 3072,
    bos_token_id: u32 = 1,
    eos_token_id: u32 = 2,
    streaming_pad_token_id: u32 = 32,
    n_left_pad_tokens: u32 = 32,
    quant_group_size: u32 = 64,
    quant_bits: u32 = 4,

    fn deinit(self: *ModelConfig, a: Allocator) void {
        a.free(self.model_type);
    }

    const JsonShape = struct {
        model_type: []const u8 = "voxtral_realtime",
        encoder_args: ?struct {
            dim: u32 = 1280,
            n_layers: u32 = 32,
            n_heads: u32 = 32,
            head_dim: u32 = 64,
            hidden_dim: u32 = 5120,
            n_kv_heads: u32 = 32,
            norm_eps: f32 = 1e-5,
            rope_theta: f64 = 1_000_000.0,
            sliding_window: u32 = 750,
            causal: bool = true,
            use_biases: bool = true,
            downsample_factor: u32 = 4,
            audio_encoding_args: ?AudioEncodingConfig = null,
        } = null,
        decoder: ?DecoderConfig = null,
        audio_encoding_args: ?AudioEncodingConfig = null,
        transcription_delay_ms: u32 = 480,
        vocab_size: ?u32 = null,
        hidden_size: ?u32 = null,
        bos_token_id: ?u32 = null,
        eos_token_id: ?u32 = null,
        streaming_pad_token_id: ?u32 = null,
        n_left_pad_tokens: ?u32 = null,
        quantization: ?struct { group_size: u32 = 64, bits: u32 = 4, mode: []const u8 = "affine" } = null,
    };

    pub fn parse(a: Allocator, bytes: []const u8) !ModelConfig {
        const parsed = std.json.parseFromSlice(JsonShape, a, bytes, .{ .ignore_unknown_fields = true }) catch
            return error.AsrBadConfig;
        defer parsed.deinit();
        const j = parsed.value;
        if (!std.mem.eql(u8, j.model_type, "voxtral_realtime")) return error.AsrUnsupportedModel;
        var cfg: ModelConfig = .{ .model_type = try a.dupe(u8, j.model_type) };
        errdefer cfg.deinit(a);
        if (j.encoder_args) |e| {
            cfg.encoder_args = .{
                .dim = e.dim,
                .n_layers = e.n_layers,
                .n_heads = e.n_heads,
                .head_dim = e.head_dim,
                .hidden_dim = e.hidden_dim,
                .n_kv_heads = e.n_kv_heads,
                .norm_eps = e.norm_eps,
                .rope_theta = e.rope_theta,
                .sliding_window = e.sliding_window,
                .causal = e.causal,
                .use_biases = e.use_biases,
                .downsample_factor = e.downsample_factor,
            };
            if (e.audio_encoding_args) |ae| cfg.audio_encoding_args = ae;
        }
        if (j.audio_encoding_args) |ae| cfg.audio_encoding_args = ae;
        if (j.decoder) |d| cfg.decoder = d;
        cfg.transcription_delay_ms = j.transcription_delay_ms;
        cfg.vocab_size = j.vocab_size orelse cfg.decoder.vocab_size;
        cfg.hidden_size = j.hidden_size orelse cfg.decoder.dim;
        cfg.bos_token_id = j.bos_token_id orelse cfg.bos_token_id;
        cfg.eos_token_id = j.eos_token_id orelse cfg.eos_token_id;
        cfg.streaming_pad_token_id = j.streaming_pad_token_id orelse cfg.streaming_pad_token_id;
        cfg.n_left_pad_tokens = j.n_left_pad_tokens orelse cfg.n_left_pad_tokens;
        if (j.quantization) |q| {
            cfg.quant_group_size = q.group_size;
            cfg.quant_bits = q.bits;
        }
        return cfg;
    }

    pub fn numAudioTokens(samples: u64) u32 {
        const rem = samples % 160;
        const frames: u64 = if (rem != 0)
            @as(u64, @intFromFloat(@ceil(@as(f64, @floatFromInt(samples)) / 160.0))) - 1
        else
            samples / 160;
        return @intCast(@divFloor(frames + AUDIO_LENGTH_PER_TOK - 1, AUDIO_LENGTH_PER_TOK));
    }

    pub fn numDelayTokens(delay_ms: u32) u32 {
        const delay_len: u64 = @intFromFloat(@as(f64, @floatFromInt(delay_ms)) / 1000.0 * SAMPLE_RATE);
        return numAudioTokens(delay_len);
    }
};

// ── ops helpers (no ownership — transfer) ──────────────────────────────────

const Ops = struct {
    a: Allocator,
    s: S,
    group_size: u32 = 64,
    bits: u32 = 4,

    fn init(a: Allocator, s: S) Ops {
        return .{ .a = a, .s = s };
    }

    fn bin2(self: *Ops, comptime f: anytype, x: A, y: A) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(f(&out, x, y, self.s));
        return out;
    }

    fn un(self: *Ops, comptime f: anytype, x: A) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(f(&out, x, self.s));
        return out;
    }

    fn add(self: *Ops, x: A, y: A) !A {
        return self.bin2(mlx.mlx_add, x, y);
    }

    fn mul(self: *Ops, x: A, y: A) !A {
        return self.bin2(mlx.mlx_multiply, x, y);
    }

    fn div(self: *Ops, x: A, y: A) !A {
        return self.bin2(mlx.mlx_divide, x, y);
    }

    fn maximum(self: *Ops, x: A, y: A) !A {
        return self.bin2(mlx.mlx_maximum, x, y);
    }

    fn scalar(self: *Ops, like: A, v: f32) !A {
        _ = self;
        const no_shape = [_]c_int{};
        switch (mlx.mlx_array_dtype(like)) {
            .float16 => {
                const h = std.heap.page_allocator.alloc(f16, 1) catch return error.AsrOom;
                h[0] = @floatCast(v);
                return mlx.mlx_array_new_data_managed_payload(
                    h.ptr,
                    &no_shape,
                    0,
                    .float16,
                    @ptrCast(h.ptr),
                    managedFreeF16,
                );
            },
            .bfloat16 => {
                const b = std.heap.page_allocator.alloc(u16, 1) catch return error.AsrOom;
                b[0] = @truncate(@as(u32, @bitCast(v)) >> 16);
                return mlx.mlx_array_new_data_managed_payload(
                    b.ptr,
                    &no_shape,
                    0,
                    .bfloat16,
                    @ptrCast(b.ptr),
                    managedFreeU16,
                );
            },
            else => return mlx.mlx_array_new_float(v),
        }
    }

    fn reshape(self: *Ops, x: A, shape: []const c_int) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, self.s));
        return out;
    }

    fn transposeAxes(self: *Ops, x: A, axes: []const c_int) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_transpose_axes(&out, x, axes.ptr, axes.len, self.s));
        return out;
    }

    fn astype(self: *Ops, x: A, dt: mlx.mlx_dtype) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_astype(&out, x, dt, self.s));
        return out;
    }

    fn take(self: *Ops, x: A, idx: A, axis: c_int) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_take_axis(&out, x, idx, axis, self.s));
        return out;
    }

    fn sliceRows(self: *Ops, x: A, start: i64, end: i64) !A {
        const shape = mlx.getShape(x);
        var starts: [4]c_int = .{ 0, 0, 0, 0 };
        var stops: [4]c_int = undefined;
        @memcpy(stops[0..shape.len], shape);
        starts[0] = @intCast(start);
        stops[0] = @intCast(end);
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_slice(&out, x, &starts, shape.len, &stops, shape.len, &strides, shape.len, self.s));
        return out;
    }

    fn sliceAxis(self: *Ops, x: A, axis: c_int, start: i64, end: i64) !A {
        const shape = mlx.getShape(x);
        var starts: [4]c_int = .{ 0, 0, 0, 0 };
        var stops: [4]c_int = undefined;
        @memcpy(stops[0..shape.len], shape);
        starts[@intCast(axis)] = @intCast(start);
        stops[@intCast(axis)] = @intCast(end);
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_slice(&out, x, &starts, shape.len, &stops, shape.len, &strides, shape.len, self.s));
        return out;
    }

    fn sliceLastAxis(self: *Ops, x: A, start: i64, end: i64) !A {
        const shape = mlx.getShape(x);
        var starts: [4]c_int = .{ 0, 0, 0, 0 };
        var stops: [4]c_int = undefined;
        @memcpy(stops[0..shape.len], shape);
        starts[shape.len - 1] = @intCast(start);
        stops[shape.len - 1] = @intCast(end);
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_slice(&out, x, &starts, shape.len, &stops, shape.len, &strides, shape.len, self.s));
        return out;
    }

    fn cat(self: *Ops, xs: []const A, axis: c_int) !A {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, axis, self.s));
        return out;
    }

    fn own(self: *Ops, x: A) !A {
        _ = self;
        return x;
    }

    fn takeOut(self: *Ops, x: A) A {
        _ = self;
        return x;
    }

    fn arange(self: *Ops, start: i64, stop: i64) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_arange(&out, @floatFromInt(start), @floatFromInt(stop), 1.0, .int32, self.s));
        return out;
    }

    fn matmul(self: *Ops, x: A, w: A) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_matmul(&out, x, w, self.s));
        return out;
    }

    /// Left-only constant pad along `axis` with zeros.
    fn pad(self: *Ops, x: A, axis: c_int, dim_unused: u32, low: usize, high: usize) !A {
        _ = dim_unused;
        const value = mlx.mlx_array_new_float(0);
        defer free(value);
        const mode: [*:0]const u8 = "constant";
        const axes = [_]c_int{axis};
        const lows = [_]c_int{@intCast(low)};
        const highs = [_]c_int{@intCast(high)};
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_pad(&out, x, &axes, 1, &lows, 1, &highs, 1, value, mode, self.s));
        return out;
    }

    /// MLX conv1d over `[N, L, C]` input, weight `[out, k, in]`, padding 0.
    fn conv1d(self: *Ops, x: A, w: A, stride: c_int, padding: c_int) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_conv1d(&out, x, w, stride, padding, 1, 1, self.s));
        return out;
    }

    /// `[1, H, L, HD]` -> `[L, H*HD]`.
    fn mergeHeads(self: *Ops, x: A, seq: i64, dim: i64) !A {
        const tr = try self.transposeAxes(x, &[_]c_int{ 0, 2, 1, 3 });
        defer free(tr);
        return self.reshape(tr, &[_]c_int{ @intCast(seq), @intCast(dim) });
    }

    fn silu(self: *Ops, x: A) !A {
        const sig = try self.un(mlx.mlx_sigmoid, x);
        defer free(sig);
        return self.mul(x, sig);
    }

    fn rmsNorm(self: *Ops, x: A, w: A, eps: f32) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, self.s));
        return out;
    }

    /// Quantized affine linear `x @ w^T` (+ optional fp bias), w [out, in]
    /// packed u32, scales f16, group 64 bits 4. x is cast to the scales'
    /// dtype (MLX requires the match).
    fn qmm(self: *Ops, x_in: A, qw: *const QW, transpose: bool) !A {
        const sdt = mlx.mlx_array_dtype(qw.s);
        const x = if (mlx.mlx_array_dtype(x_in) == sdt) x_in else try self.astype(x_in, sdt);
        defer if (mlx.mlx_array_dtype(x_in) != sdt) free(x);
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_quantized_matmul(
            &out,
            x,
            qw.w,
            qw.s,
            qw.b,
            transpose,
            mlx.mlx_optional_int.some(@intCast(self.group_size)),
            mlx.mlx_optional_int.some(@intCast(self.bits)),
            "affine",
            self.s,
        ));
        if (qw.bias_fp) |b| {
            const with_b = try self.add(out, b);
            defer free(out);
            return with_b;
        }
        return out;
    }
};

/// Interleaved (GPT-J) RoPE via mlx fast_rope, traditional=true. `x` is
/// `[B, L, H, HD]` (rope dims = HD); `offset` = the absolute first position.
fn ropeInterleaved(ops: *Ops, x: A, head_dim: u32, theta: f64, offset: u32) !A {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_fast_rope(
        &out,
        x,
        @intCast(head_dim),
        true,
        mlx.mlx_optional_float.some(@floatCast(theta)),
        1.0,
        @intCast(offset),
        none,
        ops.s,
    ));
    return out;
}

fn sdpa(ops: *Ops, q: A, k: A, v: A, scale: f32, mask_mode: [*:0]const u8, mask: A) !A {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &out,
        q,
        k,
        v,
        scale,
        mask_mode,
        mask,
        none,
        false,
        ops.s,
    ));
    return out;
}

// ── weights ────────────────────────────────────────────────────────────────

pub const QW = struct {
    w: A = none,
    s: A = none,
    b: A = none,
    /// fp bias added after the matmul (encoder wq/wv/wo, FFN w2 — f32 in the
    /// checkpoint; added to the f16 matmul result like nn.QuantizedLinear).
    bias_fp: ?A = null,

    fn deinit(self: *QW) void {
        if (self.w.ctx != null) free(self.w);
        if (self.s.ctx != null) free(self.s);
        if (self.b.ctx != null) free(self.b);
        if (self.bias_fp) |b| free(b);
    }
};

const Weights = struct {
    map: std.StringHashMap(A),
    allocator: Allocator,

    fn init(a: Allocator) Weights {
        return .{ .map = std.StringHashMap(A).init(a), .allocator = a };
    }

    fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |kv| {
            free(kv.value_ptr.*);
            self.allocator.free(kv.key_ptr.*);
        }
        self.map.deinit();
    }

    fn get(self: *const Weights, name: []const u8) ?A {
        return self.map.get(name);
    }

    /// Borrowed weight handle (the map owns it for the engine's lifetime).
    fn w(self: *const Weights, name: []const u8) !A {
        return self.map.get(name) orelse error.AsrMissingWeight;
    }

    /// Set-copy: a NEW independent wrapper (the caller owns it and frees it;
    /// the map's own handle is untouched).
    fn copy(self: *const Weights, name: []const u8, out: *A) !void {
        const src = self.map.get(name) orelse return error.AsrMissingWeight;
        try mlx.check(mlx.mlx_array_set(out, src));
    }
};

/// Build a QW triplet for `{prefix}.weight/.scales/.biases` (+ `{prefix}.bias`
/// when the checkpoint ships the selective fp bias). Handles are set-copies
/// (independent wrappers owned by the returned QW).
fn buildQW(w: *const Weights, prefix: []const u8, want_fp_bias: bool) !QW {
    var q: QW = .{};
    errdefer q.deinit();
    var buf: [200]u8 = undefined;
    const parts = [_][]const u8{ "weight", "scales", "biases", "bias" };
    var handles: [4]A = .{ none, none, none, none };
    for (parts, 0..) |part, i| {
        if (i == 3 and !want_fp_bias) break;
        const name = std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, part }) catch return error.AsrKeyTooLong;
        try w.copy(name, &handles[i]);
    }
    q.w = handles[0];
    q.s = handles[1];
    q.b = handles[2];
    if (want_fp_bias) q.bias_fp = handles[3];
    return q;
}

// ── Tekken decode-only tokenizer ───────────────────────────────────────────

const Tekken = struct {
    allocator: Allocator,
    /// Decoded bytes per vocabulary entry (index = token_id - n_special).
    vocab: [][]u8 = &.{},
    n_special: u32 = 1000,
    special_ids: std.AutoHashMap(u32, void),

    fn init(a: Allocator, bytes: []const u8) !Tekken {
        const parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch return error.AsrBadTekken;
        defer parsed.deinit();
        const v = parsed.value;
        if (v != .object) return error.AsrBadTekken;
        const vocab_v = v.object.get("vocab") orelse return error.AsrBadTekken;
        if (vocab_v != .array) return error.AsrBadTekken;
        const n_special: u32 = blk: {
            if (v.object.get("config")) |c| {
                if (c == .object) {
                    if (c.object.get("default_num_special_tokens")) |n| {
                        if (n == .integer) break :blk @intCast(n.integer);
                    }
                }
            }
            break :blk 1000;
        };

        var special_ids = std.AutoHashMap(u32, void).init(a);
        errdefer special_ids.deinit();
        if (v.object.get("special_tokens")) |stv| {
            if (stv == .array) {
                for (stv.array.items) |st| {
                    if (st == .object) {
                        if (st.object.get("rank")) |r| {
                            if (r == .integer) try special_ids.put(@intCast(r.integer), {});
                        }
                    }
                }
            }
        }

        const n = vocab_v.array.items.len;
        const vocab = try a.alloc([]u8, n);
        var decoded: usize = 0;
        errdefer {
            for (vocab[0..decoded]) |entry| a.free(entry);
            a.free(vocab);
        }
        for (vocab_v.array.items, 0..) |entry, i| {
            var bytes_out: []u8 = &.{};
            if (entry == .object) {
                if (entry.object.get("token_bytes")) |tb| {
                    if (tb == .string and tb.string.len > 0) {
                        const dec = std.base64.standard.Decoder;
                        const out_len = dec.calcSizeForSlice(tb.string) catch 0;
                        if (out_len > 0) {
                            const buf = try a.alloc(u8, out_len);
                            errdefer a.free(buf);
                            dec.decode(buf, tb.string) catch return error.AsrBadTekken;
                            bytes_out = buf;
                        }
                    }
                }
            }
            vocab[i] = bytes_out;
            decoded = i + 1;
        }

        return .{
            .allocator = a,
            .vocab = vocab,
            .n_special = n_special,
            .special_ids = special_ids,
        };
    }

    fn deinit(self: *Tekken) void {
        for (self.vocab) |entry| self.allocator.free(entry);
        self.allocator.free(self.vocab);
        self.special_ids.deinit();
    }

    fn tokenBytes(self: *const Tekken, id: u32) []const u8 {
        if (id < self.n_special) return &.{};
        if (self.special_ids.contains(id)) return &.{};
        const idx = id - self.n_special;
        if (idx >= self.vocab.len) return &.{};
        return self.vocab[idx];
    }

    fn decode(self: *const Tekken, a: Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        for (ids) |id| {
            const b = self.tokenBytes(id);
            try out.appendSlice(a, b);
        }
        return out.toOwnedSlice(a);
    }
};

// ── Slaney mel filter bank (CPU, f32) ──────────────────────────────────────

fn slaneyMelFilters(a: Allocator, sample_rate: u32, n_fft: u32, n_mels: u32, f_min: f32, f_max_in: f32) ![]f32 {
    const n_freqs: usize = n_fft / 2 + 1;
    const f_max: f32 = if (f_max_in <= 0) @as(f32, @floatFromInt(sample_rate)) / 2 else f_max_in;

    const f_sp: f32 = 200.0 / 3.0;
    const min_log_hz: f32 = 1000.0;
    const min_log_mel: f32 = (min_log_hz - f_min) / f_sp;
    const logstep: f32 = @log(6.4) / 27.0;

    const hzToMel = struct {
        fn f(freq: f32, f_sp_v: f32, min_log_hz_v: f32, min_log_mel_v: f32, logstep_v: f32) f32 {
            if (freq < min_log_hz_v) return freq / f_sp_v;
            return min_log_mel_v + @log(freq / min_log_hz_v) / logstep_v;
        }
    }.f;
    const melToHz = struct {
        fn f(mel: f32, f_sp_v: f32, min_log_hz_v: f32, min_log_mel_v: f32, logstep_v: f32) f32 {
            if (mel >= min_log_mel_v) return min_log_hz_v * @exp(logstep_v * (mel - min_log_mel_v));
            return f_sp_v * mel;
        }
    }.f;

    const m_min = hzToMel(f_min, f_sp, min_log_hz, min_log_mel, logstep);
    const m_max = hzToMel(f_max, f_sp, min_log_hz, min_log_mel, logstep);

    const m_pts = try a.alloc(f32, n_mels + 2);
    defer a.free(m_pts);
    for (0..n_mels + 2) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_mels + 1));
        m_pts[i] = m_min + t * (m_max - m_min);
    }
    const f_pts = try a.alloc(f32, n_mels + 2);
    defer a.free(f_pts);
    for (m_pts, 0..) |m, i| f_pts[i] = melToHz(m, f_sp, min_log_hz, min_log_mel, logstep);

    const all_freqs = try a.alloc(f32, n_freqs);
    defer a.free(all_freqs);
    const top: f32 = @floatFromInt(sample_rate / 2);
    for (0..n_freqs) |i| {
        all_freqs[i] = top * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_freqs - 1));
    }

    const f_diff = try a.alloc(f32, n_mels + 1);
    defer a.free(f_diff);
    for (0..n_mels + 1) |i| f_diff[i] = f_pts[i + 1] - f_pts[i];

    const fb = try a.alloc(f32, n_mels * n_freqs);
    errdefer a.free(fb);
    for (0..n_mels) |m| {
        for (0..n_freqs) |f| {
            const down = (all_freqs[f] - f_pts[m]) / f_diff[m];
            const up = (f_pts[m + 2] - all_freqs[f]) / f_diff[m + 1];
            fb[m * n_freqs + f] = @max(0.0, @min(down, up));
        }
    }
    for (0..n_mels) |m| {
        const enorm = 2.0 / (f_pts[m + 2] - f_pts[m]);
        for (0..n_freqs) |f| {
            fb[m * n_freqs + f] *= enorm;
        }
    }
    return fb;
}

fn hannWindow(a: Allocator, n: u32) ![]f32 {
    const w = try a.alloc(f32, n);
    errdefer a.free(w);
    for (0..n) |i| {
        const t: f32 = @floatFromInt(i);
        w[i] = 0.5 * (1.0 - @cos(2.0 * std.math.pi * t / @as(f32, @floatFromInt(n))));
    }
    return w;
}

/// numpy `np.pad(x, (p, p), "reflect")`: mirror the interior WITHOUT
/// repeating the edge. For arrays shorter than the pad window the reflection
/// tiles (numpy's recursive mirror).
fn reflectPad(a: Allocator, x: []const f32, p: usize) ![]f32 {
    const n = x.len;
    const out = try a.alloc(f32, n + 2 * p);
    errdefer a.free(out);
    for (0..p) |i| {
        var k: i64 = @intCast(p - i);
        while (k >= @as(i64, @intCast(n))) k -= @intCast(n - 1);
        while (k < 1) k += @intCast(n - 1);
        out[i] = x[@intCast(k)];
    }
    @memcpy(out[p .. p + n], x);
    for (0..p) |i| {
        var k: i64 = @intCast(n - 2 - i);
        while (k < 1) k += @intCast(n - 1);
        while (k >= @as(i64, @intCast(n))) k -= @intCast(n - 1);
        out[p + n + i] = x[@intCast(k)];
    }
    return out;
}

// ── engine helpers ─────────────────────────────────────────────────────────

fn slaneyFreqBins(cfg: *const ModelConfig) usize {
    return cfg.audio_encoding_args.window_size / 2 + 1;
}

fn readFileAlloc(io: std.Io, a: Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
}

fn loadConfig(io: std.Io, a: Allocator, dir: []const u8) !ModelConfig {
    const path = try std.fmt.allocPrint(a, "{s}/config.json", .{dir});
    defer a.free(path);
    const bytes = try readFileAlloc(io, a, path);
    defer a.free(bytes);
    return ModelConfig.parse(a, bytes);
}

/// Raw safetensors load: every tensor stays exactly as stored (no dtype
/// rewriting) — f16 weights/scales, u32 packed, f32 norms/biases.
fn loadRawWeights(a: Allocator, dir: []const u8, out: *Weights) !void {
    // Load on the CPU stream — mlx_load_safetensors on a GPU stream leaves
    // the tensors as unevaluatable Load ops ("Load::eval_gpu Not implemented").
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
    defer a.free(path);
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path.ptr, s));
    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const key_str = std.mem.span(key.?);
        const owned_key = try a.dupe(u8, key_str);
        try out.map.put(owned_key, value);
    }
}

/// Upload a CPU f32 buffer as a managed array (the payload copies, so the
/// caller's buffer can be freed immediately).
fn uploadF32(data: []const f32, rows: usize, cols: usize) !A {
    const buf = try std.heap.page_allocator.dupe(f32, data);
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    return mlx.mlx_array_new_data_managed_payload(buf.ptr, &shape, 2, .float32, @ptrCast(buf.ptr), managedFreeF32);
}

/// Owned 2-D transpose of a borrowed weight (the adapter projections are
/// stored [out, in]; the matmuls need [in, out]).
fn castToT(src: A, dt: mlx.mlx_dtype, s: S) !A {
    var cast = mlx.mlx_array_new();
    errdefer free(cast);
    try mlx.check(mlx.mlx_astype(&cast, src, dt, s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_transpose_axes(&out, cast, &[_]c_int{ 1, 0 }, 2, s));
    free(cast);
    return out;
}

/// Owned cast of a borrowed weight.
fn castTo(src: A, dt: mlx.mlx_dtype, s: S) !A {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_astype(&out, src, dt, s));
    return out;
}

fn loadW(w: *const Weights, name: []const u8) !A {
    var out: A = none;
    try w.copy(name, &out);
    return out;
}

/// Set-copied norm weight (encoder norms stay f32).
fn normF32(w: *const Weights, prefix: []const u8, suffix: []const u8) !A {
    var buf: [200]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "{s}.{s}.weight", .{ prefix, suffix }) catch return error.AsrKeyTooLong;
    var out: A = none;
    try w.copy(name, &out);
    return out;
}

/// Reference compute_time_embedding: sinusoidal, theta 10000, [dim] f32.
fn timeEmbedding(a: Allocator, t: f32, dim: u32) ![]f32 {
    const half: usize = dim / 2;
    const out = try a.alloc(f32, dim);
    errdefer a.free(out);
    const theta_ln = @log(10000.0);
    for (0..half) |i| {
        const inv: f32 = @exp(-theta_ln * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        const e = t * inv;
        out[i] = @cos(e);
        out[half + i] = @sin(e);
    }
    return out;
}

/// AdaRMSNorm.compute_scale in f32 (the reference's f32 chain):
/// gelu(t_cond @ ada_down^T) @ ada_up^T -> [dim].
fn computeAdaScale(ops: *Ops, ada_down: A, ada_up: A, t_cond: A) !A {
    const down_t = try ops.transposeAxes(ada_down, &[_]c_int{ 1, 0 });
    defer free(down_t);
    const h = try ops.matmul(t_cond, down_t);
    defer free(h);
    const g = try laya.gelu(h, ops.s);
    defer free(g);
    const up_t = try ops.transposeAxes(ada_up, &[_]c_int{ 1, 0 });
    defer free(up_t);
    const scale = try ops.matmul(g, up_t);
    return scale;
}

// ── mel spectrogram ────────────────────────────────────────────────────────

const Mel = struct {
    /// `[n_mels, n_freqs]` f32 bank (CPU-owned; uploaded per engine load).
    fb: []f32,
    window: []f32,

    fn deinit(self: *Mel, a: Allocator) void {
        a.free(self.fb);
        a.free(self.window);
    }
};

/// CPU reflect-pad + framing + Hann multiply, then the GPU STFT→mel chain
/// (audio.py's compute_mel_spectrogram, byte-for-byte in f32). The returned
/// mel transfers to the caller.
fn computeMel(ops: *Ops, eng: *const Engine, samples: []const f32) !A {
    const aec = &eng.cfg.audio_encoding_args;
    const win: usize = aec.window_size;
    const hop: usize = aec.hop_length;
    const padded = try reflectPad(ops.a, samples, win / 2);
    defer ops.a.free(padded);

    const L = padded.len;
    if (L < win) return error.AsrAudioTooShort;
    const n_frames: usize = 1 + (L - win) / hop;

    // CPU framing: frames[i] = padded[i*hop .. i*hop+win] * window
    const frames_buf = try ops.a.alloc(f32, n_frames * win);
    defer ops.a.free(frames_buf);
    for (0..n_frames) |i| {
        const base = i * hop;
        for (0..win) |j| {
            frames_buf[i * win + j] = padded[base + j] * eng.mel.window[j];
        }
    }
    const shape = [_]c_int{ @intCast(n_frames), @intCast(win) };
    const frames = mlx.mlx_array_new_data_managed_payload(frames_buf.ptr, &shape, 2, .float32, null, payloadNoop);
    defer free(frames);

    // spectrum [n_frames, n_freqs] complex64
    var spectrum = mlx.mlx_array_new();
    defer free(spectrum);
    try mlx.check(mlx.mlx_fft_rfft(&spectrum, frames, @intCast(win), -1, 0, ops.s));
    _ = &spectrum;

    // power spectrum: |spectrum|^2, f32
    const mags0 = try ops.un(mlx.mlx_abs, spectrum);
    defer free(mags0);
    const mags = try ops.mul(mags0, mags0);
    defer free(mags);

    // drop last frame, transpose to [n_freqs, F-1]
    const dropped = try ops.sliceRows(mags, 0, @as(i64, @intCast(n_frames - 1)));
    defer free(dropped);
    const mags_t = try ops.transposeAxes(dropped, &[_]c_int{ 1, 0 });
    defer free(mags_t);

    // mel: fb [n_mels, n_freqs] @ mags_t [n_freqs, F-1] -> [n_mels, F-1]
    const mel = try ops.matmul(eng.mel_fb_arr, mags_t);
    defer free(mel);
    const mul_shape = mlx.getShape(mel);

    // log10(max(mel, 1e-10)), clamp at (global_log_mel_max - 8), then (x+4)/4
    const eps = try ops.scalar(mel, 1e-10);
    defer free(eps);
    const clamped = try ops.maximum(mel, eps);
    defer free(clamped);
    const logged = try ops.un(mlx.mlx_log, clamped);
    defer free(logged);
    const ln10_inv = try ops.scalar(logged, 1.0 / std.math.ln10);
    defer free(ln10_inv);
    const log10d = try ops.mul(logged, ln10_inv);
    defer free(log10d);
    const min_val: f32 = aec.global_log_mel_max - 8.0;
    const min_arr = try ops.scalar(log10d, min_val);
    defer free(min_arr);
    const floored = try ops.maximum(log10d, min_arr);
    defer free(floored);
    const four = try ops.scalar(floored, 4.0);
    defer free(four);
    const plus4 = try ops.add(floored, four);
    defer free(plus4);
    const quarter = try ops.scalar(plus4, 4.0);
    defer free(quarter);
    const scaled = try ops.div(plus4, quarter);

    // Materialize now: the CPU frames buffer (freed below by its own defer)
    // feeds this chain, so it must be evaluated before the buffer dies.
    {
        const v = vectorOf(scaled);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_eval(v));
    }

    // odd frames: drop the first column (reference: mel[:, 1:])
    const cols: i64 = @intCast(mul_shape[1]);
    if (@mod(cols, 2) != 0) {
        const trimmed = try ops.sliceLastAxis(scaled, 1, cols);
        {
            const v = vectorOf(trimmed);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_eval(v));
        }
        return trimmed;
    }
    return scaled;
}

// ── conv stem ──────────────────────────────────────────────────────────────

/// conv_stem(mel): [n_mels, F] -> [S, dim] (truncated to a multiple of the
/// downsample factor). Causal convs with explicit left-only padding. The
/// result transfers to the caller.
fn convStem(ops: *Ops, eng: *const Engine, mel: A) !A {
    const ec = &eng.cfg.encoder_args;
    const aec = &eng.cfg.audio_encoding_args;

    // mel [n_mels, F] -> [1, F, n_mels] (MLX conv1d NLC layout)
    const mel_t = try ops.transposeAxes(mel, &[_]c_int{ 1, 0 });
    defer free(mel_t);
    const f = mlx.getShape(mel_t)[0];
    const mel_b = try ops.reshape(mel_t, &[_]c_int{ 1, f, @intCast(aec.num_mel_bins) });
    defer free(mel_b);

    // conv 0: 128 -> dim, k 3, s 1, left pad 2, + bias, then gelu
    const padded0 = try ops.pad(mel_b, 1, @intCast(ec.dim), 2, 0);
    defer free(padded0);
    const c0 = try ops.conv1d(padded0, eng.conv0_w, 1, 0);
    defer free(c0);
    const b0 = try ops.add(c0, eng.conv0_b);
    defer free(b0);
    const g0 = try laya.gelu(b0, ops.s);
    defer free(g0);

    // conv 1: dim -> dim, k 3, s 2, left pad 1, + bias, then gelu
    const padded1 = try ops.pad(g0, 1, @intCast(ec.dim), 1, 0);
    defer free(padded1);
    const c1 = try ops.conv1d(padded1, eng.conv1_w, 2, 0);
    defer free(c1);
    const b1 = try ops.add(c1, eng.conv1_b);
    defer free(b1);
    const g1 = try laya.gelu(b1, ops.s);
    defer free(g1);

    // squeeze the batch dim: [1, S, dim] -> [S, dim]
    const g1_shape = mlx.getShape(g1);
    const seq_len: i64 = @intCast(g1_shape[1]);
    const out = try ops.reshape(g1, &[_]c_int{ @intCast(seq_len), @intCast(ec.dim) });

    // align to the downsample factor: drop `S % ds` leading rows
    const ds: i64 = @intCast(ec.downsample_factor);
    const trunc = @mod(seq_len, ds);
    if (trunc > 0) {
        const trimmed = try ops.sliceRows(out, trunc, seq_len);
        free(out);
        return trimmed;
    }
    return out;
}

// ── encoder ────────────────────────────────────────────────────────────────

const EncLayer = struct {
    attn_norm: A,
    q: QW,
    k: QW,
    v: QW,
    o: QW,
    ffn_norm: A,
    w1: QW,
    w3: QW,
    w2: QW,

    fn deinit(self: *EncLayer) void {
        if (self.attn_norm.ctx != null) free(self.attn_norm);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        if (self.ffn_norm.ctx != null) free(self.ffn_norm);
        self.w1.deinit();
        self.w3.deinit();
        self.w2.deinit();
    }
};

/// Rotating KV cache per encoder layer (reference: RotatingKVCache, keep=0).
const EncCache = struct {
    /// Roped keys/values, 4-D f16 `[1, H, kv_len, HD]`.
    k: A = none,
    v: A = none,
    /// Absolute position of `k[0]` (advances as old rows rotate out).
    start: u32 = 0,

    fn deinit(self: *EncCache) void {
        if (self.k.ctx != null) free(self.k);
        if (self.v.ctx != null) free(self.v);
    }
};

/// One encoder attention with an optional rotating cache. `x` is `[L, dim]`
/// f32; returns the attention residual `h` f32 (with the fp biases added).
/// When a cache is attached, k_t/v_t transfer into the cache.
fn encoderAttention(
    ops: *Ops,
    eng: *const Engine,
    layer: *const EncLayer,
    x: A,
    mask_mode: [*:0]const u8,
    mask: A,
    offset: u32,
    cache: ?*EncCache,
) !A {
    const ec = &eng.cfg.encoder_args;
    const h = try ops.rmsNorm(x, layer.attn_norm, ec.norm_eps);
    defer free(h);

    const q = try ops.qmm(h, &layer.q, true);
    defer free(q);
    const k = try ops.qmm(h, &layer.k, true);
    defer free(k);
    const v = try ops.qmm(h, &layer.v, true);
    defer free(v);

    const seq: i64 = @intCast(mlx.getShape(h)[0]);

    // The reference promotes q/k through the f32 cos/sin rope and runs the
    // SDPA in f32 — replicate: cast to f32 before the reshape/rope.
    const q32 = try ops.astype(q, .float32);
    defer free(q32);
    const k32 = try ops.astype(k, .float32);
    defer free(k32);
    const v32 = try ops.astype(v, .float32);
    defer free(v32);

    const q4 = try ops.reshape(q32, &[_]c_int{ 1, @intCast(seq), @intCast(ec.n_heads), @intCast(ec.head_dim) });
    defer free(q4);
    const k4 = try ops.reshape(k32, &[_]c_int{ 1, @intCast(seq), @intCast(ec.n_heads), @intCast(ec.head_dim) });
    defer free(k4);
    const v4 = try ops.reshape(v32, &[_]c_int{ 1, @intCast(seq), @intCast(ec.n_heads), @intCast(ec.head_dim) });
    defer free(v4);

    // transpose to [1, H, L, HD] — fast_rope's per-head layout — then rope;
    // the result is already in the SDPA's [1, H, L, HD] shape.
    const q_h = try ops.transposeAxes(q4, &[_]c_int{ 0, 2, 1, 3 });
    defer free(q_h);
    const k_h = try ops.transposeAxes(k4, &[_]c_int{ 0, 2, 1, 3 });
    defer free(k_h);
    const v_t = try ops.transposeAxes(v4, &[_]c_int{ 0, 2, 1, 3 });

    const q_t = try ropeInterleaved(ops, q_h, ec.head_dim, ec.rope_theta, offset);
    defer free(q_t);
    // k_t transfers into the rotating cache (or is freed by the no-cache
    // path below) — no local free here.
    const k_t = try ropeInterleaved(ops, k_h, ec.head_dim, ec.rope_theta, offset);

    var full_k: A = k_t;
    var full_v: A = v_t;
    if (cache) |c| {
        if (c.k.ctx != null) {
            const new_k = try ops.cat(&[_]A{ c.k, k_t }, 2);
            defer free(new_k);
            const new_v = try ops.cat(&[_]A{ c.v, v_t }, 2);
            defer free(new_v);
            const total: i64 = @intCast(mlx.getShape(new_k)[2]);
            const sw: i64 = @intCast(ec.sliding_window);
            if (total > sw) {
                const drop = total - sw;
                // 4-D [1,H,kv,HD] cache: trim along the KEY axis (2).
                // sliceRows would slice axis 0 = batch -> empty batch ->
                // sdpa "mismatching batch dimension (0,32,..)" (the
                // e332d74a fix; the merge lost these call sites).
                const kk = try ops.sliceAxis(new_k, 2, drop, total);
                const vv = try ops.sliceAxis(new_v, 2, drop, total);
                c.start += @intCast(drop);
                free(c.k);
                free(c.v);
                c.k = kk;
                c.v = vv;
            } else {
                free(c.k);
                free(c.v);
                c.k = k_t;
                c.v = v_t;
            }
        } else {
            c.k = k_t;
            c.v = v_t;
        }
        full_k = c.k;
        full_v = c.v;
    }

    const attn = try sdpa(ops, q_t, full_k, full_v, 1.0 / @sqrt(@as(f32, @floatFromInt(ec.head_dim))), mask_mode, mask);
    defer free(attn);
    if (cache == null) {
        free(k_t);
        free(v_t);
    }

    const merged = try ops.mergeHeads(attn, seq, @intCast(ec.n_heads * ec.head_dim));
    defer free(merged);

    const wo = try ops.qmm(merged, &layer.o, true);
    return wo;
}

fn encLayerForward(
    ops: *Ops,
    eng: *const Engine,
    layer: *const EncLayer,
    x: A,
    mask_mode: [*:0]const u8,
    mask: A,
    offset: u32,
    cache: ?*EncCache,
) !A {
    const ec = &eng.cfg.encoder_args;
    const h = try encoderAttention(ops, eng, layer, x, mask_mode, mask, offset, cache);
    defer free(h);
    const x1 = try ops.add(x, h);
    defer free(x1);

    const hn = try ops.rmsNorm(x1, layer.ffn_norm, ec.norm_eps);
    defer free(hn);
    const gate = try ops.qmm(hn, &layer.w1, true);
    defer free(gate);
    const gate_s = try ops.silu(gate);
    defer free(gate_s);
    const up = try ops.qmm(hn, &layer.w3, true);
    defer free(up);
    const prod = try ops.mul(gate_s, up);
    defer free(prod);
    const down = try ops.qmm(prod, &layer.w2, true);
    defer free(down);
    const x2 = try ops.add(x1, down);
    return x2;
}

/// Additive window mask `[chunk, kv_len]` f16: 0 where the key is in-window
/// and causal, -1e9 otherwise (the reference's mx.where mask). The array owns
/// its page-allocated buffer via the managed payload.
fn buildEncWindowMask(ops: *Ops, eng: *const Engine, chunk_len: u32, cache: *const EncCache, end_pos: u32) !A {
    _ = ops;
    const ec = &eng.cfg.encoder_args;
    const sw = ec.sliding_window;
    // The mask must describe the cache AS THE ATTENTION WILL SEE IT: the
    // layer rotates (cat + trim to sw) BEFORE sdpa, so columns are the
    // post-trim key span [end - final_len, end). Building from the
    // pre-update span [start, start+kv) shifts every column by `drop`
    // (10/26 F2 quality bug: early EOS / phrase repetition past 750 rows).
    const kv_pre: u32 = @intCast(mlx.getShape(cache.k)[2]);
    const final_len = @min(sw, kv_pre + chunk_len);
    const first_key: u32 = end_pos - final_len;
    const buf = try std.heap.page_allocator.alloc(f16, @as(usize, chunk_len) * final_len);
    var q_pos: u32 = end_pos - chunk_len;
    while (q_pos < end_pos) : (q_pos += 1) {
        var k_pos: u32 = first_key;
        while (k_pos < first_key + final_len) : (k_pos += 1) {
            const causal = k_pos <= q_pos;
            const in_window = k_pos + sw > q_pos;
            const idx = (q_pos - (end_pos - chunk_len)) * final_len + (k_pos - first_key);
            buf[@intCast(idx)] = if (causal and in_window) 0.0 else -1e9;
        }
    }
    const shape = [_]c_int{ @intCast(chunk_len), @intCast(final_len) };
    return mlx.mlx_array_new_data_managed_payload(buf.ptr, &shape, 2, .float16, @ptrCast(buf.ptr), managedFreeF16);
}

/// Chunked causal encode (the reference's encode_chunks): sliding-window
/// chunks with rotating per-layer caches. The concat result transfers.
fn encodeChunks(ops: *Ops, eng: *const Engine, conv_out: A) !A {
    const ec = &eng.cfg.encoder_args;
    const seq: i64 = @intCast(mlx.getShape(conv_out)[0]);
    const sw: i64 = @intCast(ec.sliding_window);
    const n_layers: usize = ec.n_layers;

    var caches = try ops.a.alloc(EncCache, n_layers);
    defer ops.a.free(caches);
    for (caches) |*c| c.* = .{};
    defer for (caches) |*c| c.deinit();

    var chunks: std.ArrayList(A) = .empty;
    defer {
        for (chunks.items) |c| free(c);
        chunks.deinit(ops.a);
    }

    var start: i64 = 0;
    while (start < seq) {
        const end = @min(start + sw, seq);
        const chunk_len = end - start;

        var x = try ops.sliceRows(conv_out, start, end);
        for (0..n_layers) |i| {
            const mask_mode: [*:0]const u8 = if (start == 0) "causal" else "array";
            var mask: A = none;
            if (start != 0) {
                mask = try buildEncWindowMask(ops, eng, @intCast(chunk_len), caches[i], @intCast(end));
            }
            defer if (mask.ctx != null) free(mask);
            const next = try encLayerForward(ops, eng, &eng.enc_layers[i], x, mask_mode, mask, @intCast(start), &caches[i]);
            free(x);
            x = next;
        }
        const normed = try ops.rmsNorm(x, eng.enc_norm, ec.norm_eps);
        free(x);
        try chunks.append(ops.a, normed);
        start = end;
    }

    const out = try ops.cat(chunks.items, 0);
    return out;
}

// ── downsample + adapter ───────────────────────────────────────────────────

/// downsample_and_project: [S, dim] -> [S/4, decoder_dim] f16. Transfers.
fn downsampleAndProject(ops: *Ops, eng: *const Engine, encoded: A) !A {
    const ec = &eng.cfg.encoder_args;
    const seq: i64 = @intCast(mlx.getShape(encoded)[0]);
    const ds: i64 = @intCast(ec.downsample_factor);
    const ds_len = @divFloor(seq, ds);
    if (ds_len == 0) return error.AsrAudioTooShort;

    const kept = ds_len * ds;
    const trimmed = if (kept != seq)
        try ops.sliceRows(encoded, 0, kept)
    else
        encoded;
    defer if (kept != seq) free(trimmed);

    const wide = try ops.reshape(trimmed, &[_]c_int{ @intCast(ds_len), @intCast(ec.dim * ec.downsample_factor) });
    defer free(wide);
    // The reference matmuls f32 x against the f16 projection — MLX promotes
    // to f32 — so the adapter output is f32, and the decoder stream starts f32.
    const p0 = try ops.matmul(wide, eng.adapter0_w);
    defer free(p0);
    const g = try laya.gelu(p0, ops.s);
    defer free(g);
    const p1 = try ops.matmul(g, eng.adapter2_w);
    return p1;
}

// ── decoder ────────────────────────────────────────────────────────────────

const DecLayer = struct {
    attn_norm: A,
    q: QW,
    k: QW,
    v: QW,
    o: QW,
    ffn_norm: A,
    ada_down: A,
    ada_up: A,
    w1: QW,
    w3: QW,
    w2: QW,
    /// Precomputed ada scale for the configured delay (f16 [dim]).
    ada_scale: ?A = null,

    fn deinit(self: *DecLayer) void {
        if (self.attn_norm.ctx != null) free(self.attn_norm);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        if (self.ffn_norm.ctx != null) free(self.ffn_norm);
        if (self.ada_down.ctx != null) free(self.ada_down);
        if (self.ada_up.ctx != null) free(self.ada_up);
        self.w1.deinit();
        self.w3.deinit();
        self.w2.deinit();
        if (self.ada_scale) |s| free(s);
    }
};

/// Decoder KV cache: 4-D f16 `[1, H, kv_len, HD]` + the absolute position of
/// row 0 (the reference concatenates then trims to the sliding window).
const DecCache = struct {
    k: A = none,
    v: A = none,
    start: u32 = 0,

    fn deinit(self: *DecCache) void {
        if (self.k.ctx != null) free(self.k);
        if (self.v.ctx != null) free(self.v);
    }
};

/// One decoder attention. `x` [L, dim] f16; positions `start_pos..+L`.
/// k_t/v_t transfer into the KV cache.
fn decoderAttention(
    ops: *Ops,
    eng: *const Engine,
    layer: *const DecLayer,
    x: A,
    start_pos: u32,
    mask_mode: [*:0]const u8,
    mask: A,
    cache: *DecCache,
) !A {
    const dc = &eng.cfg.decoder;
    const seq: i64 = @intCast(mlx.getShape(x)[0]);

    const h = try ops.rmsNorm(x, layer.attn_norm, dc.norm_eps);
    defer free(h);

    const q = try ops.qmm(h, &layer.q, true);
    defer free(q);
    const k = try ops.qmm(h, &layer.k, true);
    defer free(k);
    const v = try ops.qmm(h, &layer.v, true);
    defer free(v);

    const q32 = try ops.astype(q, .float32);
    defer free(q32);
    const k32 = try ops.astype(k, .float32);
    defer free(k32);
    const v32 = try ops.astype(v, .float32);
    defer free(v32);
    const q4 = try ops.reshape(q32, &[_]c_int{ 1, @intCast(seq), @intCast(dc.n_heads), @intCast(dc.head_dim) });
    defer free(q4);
    const k4 = try ops.reshape(k32, &[_]c_int{ 1, @intCast(seq), @intCast(dc.n_kv_heads), @intCast(dc.head_dim) });
    defer free(k4);
    const v4 = try ops.reshape(v32, &[_]c_int{ 1, @intCast(seq), @intCast(dc.n_kv_heads), @intCast(dc.head_dim) });
    defer free(v4);

    const q_h = try ops.transposeAxes(q4, &[_]c_int{ 0, 2, 1, 3 });
    defer free(q_h);
    const k_h = try ops.transposeAxes(k4, &[_]c_int{ 0, 2, 1, 3 });
    defer free(k_h);
    const v_t = try ops.transposeAxes(v4, &[_]c_int{ 0, 2, 1, 3 });

    const q_t = try ropeInterleaved(ops, q_h, dc.head_dim, dc.rope_theta, start_pos);
    defer free(q_t);
    const k_t = try ropeInterleaved(ops, k_h, dc.head_dim, dc.rope_theta, start_pos);

    var full_k: A = k_t;
    var full_v: A = v_t;
    {
        if (cache.k.ctx != null) {
            // new_k/new_v are owned by this fn until ownership moves into the
            // cache below — explicit frees (no defers, ownership transfers).
            const new_k = try ops.cat(&[_]A{ cache.k, k_t }, 2);
            const new_v = try ops.cat(&[_]A{ cache.v, v_t }, 2);
            const total: i64 = @intCast(mlx.getShape(new_k)[2]);
            const sw: i64 = @intCast(dc.sliding_window);
            if (total > sw) {
                const drop = total - sw;
                // axis-2 trim (see encoderAttention): never batch-slice a cache
                const kk = try ops.sliceAxis(new_k, 2, drop, total);
                const vv = try ops.sliceAxis(new_v, 2, drop, total);
                free(new_k);
                free(new_v);
                cache.start += @intCast(drop);
                free(cache.k);
                free(cache.v);
                cache.k = kk;
                cache.v = vv;
            } else {
                free(cache.k);
                free(cache.v);
                cache.k = new_k;
                cache.v = new_v;
            }
        } else {
            cache.k = k_t;
            cache.v = v_t;
        }
        full_k = cache.k;
        full_v = cache.v;
    }

    const attn = try sdpa(ops, q_t, full_k, full_v, 1.0 / @sqrt(@as(f32, @floatFromInt(dc.head_dim))), mask_mode, mask);
    defer free(attn);

    const merged = try ops.mergeHeads(attn, seq, @intCast(dc.n_heads * dc.head_dim));
    defer free(merged);

    const wo = try ops.qmm(merged, &layer.o, true);
    return wo;
}

fn decLayerForward(
    ops: *Ops,
    eng: *const Engine,
    layer: *const DecLayer,
    x: A,
    start_pos: u32,
    mask_mode: [*:0]const u8,
    mask: A,
    cache: *DecCache,
) !A {
    const dc = &eng.cfg.decoder;
    const h = try decoderAttention(ops, eng, layer, x, start_pos, mask_mode, mask, cache);
    defer free(h);
    const x1 = try ops.add(x, h);
    defer free(x1);

    var hn = try ops.rmsNorm(x1, layer.ffn_norm, dc.norm_eps);
    defer free(hn);
    if (layer.ada_scale) |scale| {
        const one = try ops.scalar(hn, 1.0);
        defer free(one);
        const plus = try ops.add(one, scale);
        defer free(plus);
        const modulated = try ops.mul(hn, plus);
        free(hn);
        hn = modulated;
    }
    const gate = try ops.qmm(hn, &layer.w1, true);
    defer free(gate);
    const gate_s = try ops.silu(gate);
    defer free(gate_s);
    const up = try ops.qmm(hn, &layer.w3, true);
    defer free(up);
    const prod = try ops.mul(gate_s, up);
    defer free(prod);
    const down = try ops.qmm(prod, &layer.w2, true);
    defer free(down);
    const x2 = try ops.add(x1, down);
    return x2;
}

// ── engine ─────────────────────────────────────────────────────────────────


fn asrDbg() bool {
    const raw = std.c.getenv("ASR_DEBUG") orelse return false;
    return raw[0] == '1';
}

pub const Engine = struct {
    allocator: Allocator,
    stream: S,
    cfg: ModelConfig,
    weights: Weights,
    mel: Mel,
    /// GPU f32 [n_mels, n_freqs] copy of `mel.fb`.
    mel_fb_arr: A,
    conv0_w: A,
    conv0_b: A,
    conv1_w: A,
    conv1_b: A,
    enc_layers: []EncLayer,
    enc_norm: A,
    adapter0_w: A,
    adapter2_w: A,
    dec_layers: []DecLayer,
    dec_norm: A,
    tok_emb: A,
    tok: Tekken,
    ops: Ops,
    /// delay tokens for the configured transcription_delay_ms.
    n_delay: u32,

    pub fn load(io: std.Io, a: Allocator, dir: []const u8) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        self.allocator = a;
        self.stream = mlx.mlx_default_gpu_stream_new();
        errdefer _ = mlx.mlx_stream_free(self.stream);
        self.cfg = try loadConfig(io, a, dir);
        errdefer self.cfg.deinit(a);
        self.weights = Weights.init(a);
        errdefer self.weights.deinit();
        try loadRawWeights(a, dir, &self.weights);
        self.mel = .{
            .fb = try slaneyMelFilters(a, self.cfg.audio_encoding_args.sampling_rate, self.cfg.audio_encoding_args.window_size, self.cfg.audio_encoding_args.num_mel_bins, 0, 8000),
            .window = try hannWindow(a, self.cfg.audio_encoding_args.window_size),
        };
        self.mel_fb_arr = try uploadF32(self.mel.fb, self.cfg.audio_encoding_args.num_mel_bins, slaneyFreqBins(&self.cfg));
        // conv weights: f16 in the checkpoint, cast to f32 (the reference's
        // f32 mel input promotes the conv to f32).
        self.conv0_w = try castTo(try loadW(&self.weights, "encoder.conv_layers_0_conv.conv.weight"), .float32, self.stream);
        self.conv0_b = try castTo(try loadW(&self.weights, "encoder.conv_layers_0_conv.conv.bias"), .float32, self.stream);
        self.conv1_w = try castTo(try loadW(&self.weights, "encoder.conv_layers_1_conv.conv.weight"), .float32, self.stream);
        self.conv1_b = try castTo(try loadW(&self.weights, "encoder.conv_layers_1_conv.conv.bias"), .float32, self.stream);

        self.ops = Ops.init(a, self.stream);
        self.ops.group_size = self.cfg.quant_group_size;
        self.ops.bits = self.cfg.quant_bits;

        self.enc_layers = try a.alloc(EncLayer, self.cfg.encoder_args.n_layers);
        var built_enc: usize = 0;
        errdefer {
            for (self.enc_layers[0..built_enc]) |*l| l.deinit();
            a.free(self.enc_layers);
        }
        for (0..self.cfg.encoder_args.n_layers) |i| {
            var buf: [128]u8 = undefined;
            self.enc_layers[i] = .{
                .attn_norm = try normF32(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}", .{i}) catch return error.AsrKeyTooLong, "attention_norm"),
                .q = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}.attention.wq", .{i}) catch return error.AsrKeyTooLong, true),
                .k = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}.attention.wk", .{i}) catch return error.AsrKeyTooLong, false),
                .v = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}.attention.wv", .{i}) catch return error.AsrKeyTooLong, true),
                .o = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}.attention.wo", .{i}) catch return error.AsrKeyTooLong, true),
                .ffn_norm = try normF32(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}", .{i}) catch return error.AsrKeyTooLong, "ffn_norm"),
                .w1 = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}.feed_forward_w1", .{i}) catch return error.AsrKeyTooLong, false),
                .w3 = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}.feed_forward_w3", .{i}) catch return error.AsrKeyTooLong, false),
                .w2 = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "encoder.transformer_layers.{d}.feed_forward_w2", .{i}) catch return error.AsrKeyTooLong, true),
            };
            built_enc += 1;
        }
        self.enc_norm = try normF32(&self.weights, "encoder", "transformer_norm");
        self.adapter0_w = try castToT(try loadW(&self.weights, "encoder.audio_language_projection_0.weight"), .float16, self.stream);
        self.adapter2_w = try castToT(try loadW(&self.weights, "encoder.audio_language_projection_2.weight"), .float16, self.stream);

        self.dec_layers = try a.alloc(DecLayer, self.cfg.decoder.n_layers);
        var built_dec: usize = 0;
        errdefer {
            for (self.dec_layers[0..built_dec]) |*l| l.deinit();
            a.free(self.dec_layers);
        }
        // t_cond (f32) from the delay; ada scales in f16.
        const t_cond_buf = try timeEmbedding(a, @floatFromInt(ModelConfig.numDelayTokens(self.cfg.transcription_delay_ms)), self.cfg.decoder.dim);
        defer a.free(t_cond_buf);
        const t_cond = try uploadF32(t_cond_buf, 1, self.cfg.decoder.dim);
        for (0..self.cfg.decoder.n_layers) |i| {
            var buf: [128]u8 = undefined;
            const attn_norm = try loadW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.attention_norm.weight", .{i}) catch return error.AsrKeyTooLong);
            const ffn_norm = try loadW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.ffn_norm.weight", .{i}) catch return error.AsrKeyTooLong);
            const ada_down = try loadW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.ada_rms_norm_t_cond.ada_down.weight", .{i}) catch return error.AsrKeyTooLong);
            const ada_up = try loadW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.ada_rms_norm_t_cond.ada_up.weight", .{i}) catch return error.AsrKeyTooLong);
            const ada_scale = try computeAdaScale(&self.ops, ada_down, ada_up, t_cond);
            self.dec_layers[i] = .{
                .attn_norm = attn_norm,
                .q = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.attention.wq", .{i}) catch return error.AsrKeyTooLong, false),
                .k = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.attention.wk", .{i}) catch return error.AsrKeyTooLong, false),
                .v = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.attention.wv", .{i}) catch return error.AsrKeyTooLong, false),
                .o = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.attention.wo", .{i}) catch return error.AsrKeyTooLong, false),
                .ffn_norm = ffn_norm,
                .ada_down = ada_down,
                .ada_up = ada_up,
                .w1 = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.feed_forward_w1", .{i}) catch return error.AsrKeyTooLong, false),
                .w3 = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.feed_forward_w3", .{i}) catch return error.AsrKeyTooLong, false),
                .w2 = try buildQW(&self.weights, std.fmt.bufPrint(&buf, "decoder.layers.{d}.feed_forward_w2", .{i}) catch return error.AsrKeyTooLong, false),
                .ada_scale = ada_scale,
            };
            built_dec += 1;
        }
        self.dec_norm = try loadW(&self.weights, "decoder.norm.weight");
        self.tok_emb = try loadW(&self.weights, "decoder.tok_embeddings.weight");

        const tekken_path = try std.fmt.allocPrint(a, "{s}/tekken.json", .{dir});
        defer a.free(tekken_path);
        const tekken_bytes = try readFileAlloc(io, a, tekken_path);
        defer a.free(tekken_bytes);
        self.tok = try Tekken.init(a, tekken_bytes);

        self.n_delay = ModelConfig.numDelayTokens(self.cfg.transcription_delay_ms);

        log.info("mlx asr engine loaded: {s} (encoder {d} layers, decoder {d} layers, delay {d} ms = {d} tokens)\n", .{
            dir,
            self.cfg.encoder_args.n_layers,
            self.cfg.decoder.n_layers,
            self.cfg.transcription_delay_ms,
            self.n_delay,
        });
        return self;
    }

    fn nDelayTokens(self: *const Engine) u32 {
        return ModelConfig.numDelayTokens(self.cfg.transcription_delay_ms);
    }

    pub fn deinit(self: *Engine) void {
        self.mel.deinit(self.allocator);
        _ = mlx.mlx_array_free(self.mel_fb_arr);
        _ = mlx.mlx_array_free(self.conv0_w);
        _ = mlx.mlx_array_free(self.conv0_b);
        _ = mlx.mlx_array_free(self.conv1_w);
        _ = mlx.mlx_array_free(self.conv1_b);
        for (self.enc_layers) |*l| l.deinit();
        self.allocator.free(self.enc_layers);
        _ = mlx.mlx_array_free(self.enc_norm);
        _ = mlx.mlx_array_free(self.adapter0_w);
        _ = mlx.mlx_array_free(self.adapter2_w);
        for (self.dec_layers) |*l| l.deinit();
        self.allocator.free(self.dec_layers);
        _ = mlx.mlx_array_free(self.dec_norm);
        _ = mlx.mlx_array_free(self.tok_emb);
        self.tok.deinit();
        self.weights.deinit();
        self.cfg.deinit(self.allocator);
        _ = mlx.mlx_stream_free(self.stream);
        self.allocator.destroy(self);
    }
};

// ── run context / chunk iterator / one-shot transcription ──────────────────

const RunCtx = struct {
    ops: *Ops,
    a: Allocator,
    caches: []DecCache,
    enc_caches: []EncCache,

    fn deinit(self: *RunCtx) void {
        for (self.caches) |*c| c.deinit();
        self.a.free(self.caches);
        for (self.enc_caches) |*c| c.deinit();
        self.a.free(self.enc_caches);
    }
};

pub const TranscribeOpts = struct {
    max_tokens: u32 = 4096,
    /// Polled between decode steps; true → error.Cancelled.
    cancelled: ?*const std.atomic.Value(bool) = null,
};

/// Reference `_pad_audio_streaming`: left n_left×1280 silence, align to a
/// multiple of 1280, right (align + n_right×1280) silence.
fn padAudioStreaming(a: Allocator, samples: []const f32, n_left: u32, n_right: u32) ![]f32 {
    const mult: usize = RAW_AUDIO_LENGTH_PER_TOK;
    const align_pad: usize = (mult - (@mod(samples.len, mult))) % mult;
    const right: usize = align_pad + n_right * mult;
    const left: usize = n_left * mult;
    const out = try a.alloc(f32, left + samples.len + right);
    errdefer a.free(out);
    @memset(out[0..left], 0);
    @memcpy(out[left .. left + samples.len], samples);
    @memset(out[left + samples.len ..], 0);
    return out;
}

/// Chunked encoder adapter iterator (replicates the reference's per-chunk
/// downsample grouping, which differs from a single full-array downsample).
const ChunkAdapterIter = struct {
    ops: *Ops,
    eng: *const Engine,
    conv_out: A,
    enc_caches: []EncCache,
    seq: i64,
    sw: i64,
    pos: i64 = 0,
    done: bool = false,

    fn next(self: *ChunkAdapterIter) !?A {
        if (self.done or self.pos >= self.seq) return null;
        const end = @min(self.pos + self.sw, self.seq);
        var x = try self.ops.sliceRows(self.conv_out, self.pos, end);
        const start_pos: u32 = @intCast(self.pos);
        for (0..self.eng.cfg.encoder_args.n_layers) |i| {
            const mask_mode: [*:0]const u8 = if (self.pos == 0) "causal" else "array";
            var mask: A = none;
            if (self.pos != 0) {
                mask = try buildEncWindowMask(self.ops, self.eng, @intCast(end - self.pos), &self.enc_caches[i], @intCast(end));
            }
            defer if (mask.ctx != null) free(mask);
            const next_layer = try encLayerForward(self.ops, self.eng, &self.eng.enc_layers[i], x, mask_mode, mask, start_pos, &self.enc_caches[i]);
            free(x);
            x = next_layer;
        }
        const normed = try self.ops.rmsNorm(x, self.eng.enc_norm, self.eng.cfg.encoder_args.norm_eps);
        free(x);
        const adapter = try downsampleAndProject(self.ops, self.eng, normed);
        free(normed);
        self.pos = end;
        if (self.pos >= self.seq) self.done = true;
        return adapter;
    }
};

/// Prefill the decoder over `prompt_len` rows; returns the LAST row's logits
/// (ownership transfers to the caller).
fn prefill(ops: *Ops, eng: *const Engine, embeds: A, prompt_len: u32, caches: []DecCache) !A {
    var h = embeds;
    for (0..eng.cfg.decoder.n_layers) |i| {
        const next = try decLayerForward(ops, eng, &eng.dec_layers[i], h, 0, "causal", none, &caches[i]);
        if (i > 0) free(h);
        h = next;
    }
    defer free(h);
    const normed = try ops.rmsNorm(h, eng.dec_norm, eng.cfg.decoder.norm_eps);
    defer free(normed);
    const last = try ops.sliceRows(normed, prompt_len - 1, prompt_len);
    defer free(last);
    const emb_t = try ops.transposeAxes(eng.tok_emb, &[_]c_int{ 1, 0 });
    defer free(emb_t);
    const logits = try ops.matmul(last, emb_t);
    return logits;
}

/// Greedy argmax over the [1, V] logits row.
fn sampleArgmax(ops: *Ops, logits: A) !u32 {
    var am = mlx.mlx_array_new();
    defer free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, logits, 1, false, ops.s));
    try mlx.check(mlx.mlx_array_eval(am));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, am));
    return @intCast(v);
}

/// Dump an array (any float dtype) as raw f32 + a shape sidecar, for the
/// parity bench. The IO Threaded is created once, lazily.
var dump_io: ?std.Io.Threaded = null;


fn vectorOf(x: A) mlx.mlx_vector_array {
    return mlx.mlx_vector_array_new_data(&[_]A{x}, 1);
}

/// Transcribe 16 kHz mono s16 WAV bytes (any WAV is mixed/resampled). The
/// reply JSON transfers to the caller.
pub fn transcribe(self: *Engine, a: Allocator, wav: []const u8, opts: TranscribeOpts) ![]u8 {
    const decoded = wav_mod.decode(a, wav) catch return error.AsrBadWav;
    defer a.free(decoded.pcm);

    // `samples` may alias the decoded buffer (mono 16 kHz) or be a fresh
    // buffer we own — only owned buffers are freed here; decoded.pcm is freed
    // once by its own defer.
    var samples: []f32 = decoded.pcm;
    var samples_owned = false;
    if (decoded.channels > 1) {
        const n = decoded.pcm.len / decoded.channels;
        const mono = try a.alloc(f32, n);
        for (0..n) |i| {
            var acc: f32 = 0;
            for (0..decoded.channels) |ch| acc += decoded.pcm[i * decoded.channels + ch];
            mono[i] = acc / @as(f32, @floatFromInt(decoded.channels));
        }
        samples = mono;
        samples_owned = true;
    }
    if (decoded.sample_rate != @as(u32, @intFromFloat(SAMPLE_RATE))) {
        const resampled = try wav_mod.resampleLinear(a, samples, 1, decoded.sample_rate, @intFromFloat(SAMPLE_RATE));
        if (samples_owned) a.free(samples);
        samples = resampled;
        samples_owned = true;
    }
    defer if (samples_owned) a.free(samples);
    const input_samples: f64 = @floatFromInt(decoded.pcm.len / @max(decoded.channels, 1));
    const duration_s = input_samples / @as(f64, @floatFromInt(decoded.sample_rate));

    var run = RunCtx{
        .ops = &self.ops,
        .a = a,
        .caches = try a.alloc(DecCache, self.cfg.decoder.n_layers),
        .enc_caches = try a.alloc(EncCache, self.cfg.encoder_args.n_layers),
    };
    defer run.deinit();
    for (run.caches) |*c| c.* = .{};
    for (run.enc_caches) |*c| c.* = .{};

    var ops = Ops.init(a, self.stream);
    ops.group_size = self.cfg.quant_group_size;
    ops.bits = self.cfg.quant_bits;
    run.ops = &ops;

    // 1) pad + mel + conv stem
    const padded = try padAudioStreaming(a, samples, self.cfg.n_left_pad_tokens, self.nDelayTokens() + 1 + 10);
    defer a.free(padded);

    if (asrDbg()) std.debug.print("ASRDBG pad ok\n", .{});
    const mel = try computeMel(&ops, self, padded);
    defer free(mel);
    if (asrDbg()) std.debug.print("ASRDBG mel shape {d}x{d}\n", .{mlx.getShape(mel)[0], mlx.getShape(mel)[1]});
    const conv_out = try convStem(&ops, self, mel);
    defer free(conv_out);
    if (asrDbg()) std.debug.print("ASRDBG conv {d}\n", .{mlx.getShape(conv_out)[0]});

    const seq: i64 = @intCast(mlx.getShape(conv_out)[0]);
    const ds: i64 = @intCast(self.cfg.encoder_args.downsample_factor);
    const n_audio: i64 = @divFloor(seq, ds);
    if (n_audio <= 0) return error.AsrAudioTooShort;
    const sw: i64 = @intCast(self.cfg.encoder_args.sliding_window);

    // 2) encode → adapter
    var adapter_out: A = undefined;
    var adapter_len: i64 = 0;
    var chunk_iter: ?ChunkAdapterIter = null;
    if (seq <= sw) {
        var x = conv_out;
        for (0..self.cfg.encoder_args.n_layers) |i| {
            const next = try encLayerForward(&ops, self, &self.enc_layers[i], x, "causal", none, 0, null);
            if (i > 0) free(x);
            x = next;
        }
        const normed = try ops.rmsNorm(x, self.enc_norm, self.cfg.encoder_args.norm_eps);
        if (x.ctx != conv_out.ctx) free(x);
        adapter_out = try downsampleAndProject(&ops, self, normed);
        free(normed);
        adapter_len = @intCast(mlx.getShape(adapter_out)[0]);
    } else {
        chunk_iter = .{
            .ops = &ops,
            .eng = self,
            .conv_out = conv_out,
            .enc_caches = run.enc_caches,
            .seq = seq,
            .sw = sw,
        };
        const prompt_len: i64 = @intCast(1 + self.cfg.n_left_pad_tokens + self.n_delay);
        var parts: std.ArrayList(A) = .empty;
        defer {
            for (parts.items) |p| free(p);
            parts.deinit(a);
        }
        while (adapter_len < prompt_len) {
            const chunk = (try chunk_iter.?.next()) orelse break;
            adapter_len += @intCast(mlx.getShape(chunk)[0]);
            try parts.append(a, chunk);
        }
        if (parts.items.len == 0) return error.AsrAudioTooShort;
        adapter_out = try ops.cat(parts.items, 0);
    }
    defer free(adapter_out);
    {
        const one_vec = vectorOf(adapter_out);
        defer _ = mlx.mlx_vector_array_free(one_vec);
        try mlx.check(mlx.mlx_eval(one_vec));
    }

    if (asrDbg()) std.debug.print("ASRDBG adapter {d} n_audio {d}\n", .{adapter_len, n_audio});
    // 3) prompt embeddings + prefill
    const prompt_len: i64 = @intCast(1 + self.cfg.n_left_pad_tokens + self.n_delay);
    const prompt_ids_buf = try a.alloc(u32, @intCast(prompt_len));
    defer a.free(prompt_ids_buf);
    prompt_ids_buf[0] = self.cfg.bos_token_id;
    for (1..@intCast(prompt_len)) |i| prompt_ids_buf[i] = self.cfg.streaming_pad_token_id;
    const pids_shape = [_]c_int{@intCast(prompt_len)};
    const prompt_ids = try ops.own(mlx.mlx_array_new_data(prompt_ids_buf.ptr, &pids_shape, 1, .int32));
    defer free(prompt_ids);

    const prompt_emb = try ops.take(self.tok_emb, prompt_ids, 0);
    defer free(prompt_emb);
    const adapter_prefix = try ops.sliceRows(adapter_out, 0, prompt_len);
    defer free(adapter_prefix);
    const prefix_embeds = try ops.add(adapter_prefix, prompt_emb);
    defer free(prefix_embeds);

    const logits = try prefill(&ops, self, prefix_embeds, @intCast(prompt_len), run.caches);
    defer free(logits);

    if (asrDbg()) std.debug.print("ASRDBG prefill logits ok\n", .{});
    // 4) decode loop: pos in [prompt_len, n_audio)
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(a);

    var next_tok: u32 = try sampleArgmax(&ops, logits);

    var stop = false;
    var pos: i64 = prompt_len;
    while (pos < n_audio and !stop) {
        if (opts.cancelled) |flag| {
            if (flag.load(.acquire)) return error.Cancelled;
        }
        generated.append(a, next_tok) catch return error.AsrOom;
        if (next_tok == self.cfg.eos_token_id or generated.items.len > opts.max_tokens) {
            stop = true;
            break;
        }
        if (chunk_iter != null and pos >= adapter_len) {
            const chunk_opt = try chunk_iter.?.next();
            if (chunk_opt) |ch| {
                defer free(ch);
                adapter_len += @intCast(mlx.getShape(ch)[0]);
                const old = adapter_out;
                adapter_out = try ops.cat(&[_]A{ old, ch }, 0);
                free(old);
            } else {
                chunk_iter = null;
            }
        }
        const tshape = [_]c_int{1};
        const tid = [_]u32{next_tok};
        const tok_arr = try ops.own(mlx.mlx_array_new_data(&tid, &tshape, 1, .int32));
        defer free(tok_arr);
        var embed = try ops.take(self.tok_emb, tok_arr, 0);
        if (pos < adapter_len) {
            const arow = try ops.sliceRows(adapter_out, pos, pos + 1);
            defer free(arow);
            const merged = try ops.add(arow, embed);
            free(embed);
            embed = merged;
        }

        var cur = embed;
        for (0..self.cfg.decoder.n_layers) |i| {
            const next = try decLayerForward(&ops, self, &self.dec_layers[i], cur, @intCast(pos), "causal", none, &run.caches[i]);
            if (i > 0) free(cur);
            cur = next;
        }
        const normed = try ops.rmsNorm(cur, self.dec_norm, self.cfg.decoder.norm_eps);
        free(cur);
        const emb_t = try ops.transposeAxes(self.tok_emb, &[_]c_int{ 1, 0 });
        defer free(emb_t);
        const logits2 = try ops.matmul(normed, emb_t);
        free(normed);
        next_tok = try sampleArgmax(&ops, logits2);
        pos += 1;
        if (generated.items.len % 256 == 0) _ = mlx.mlx_clear_cache();
    }
    if (!stop and pos >= n_audio and generated.items.len > 0 and
        generated.items[generated.items.len - 1] != next_tok)
    {
        generated.append(a, next_tok) catch return error.AsrOom;
    }

    if (asrDbg()) std.debug.print("ASRDBG decode done tokens={d}\n", .{generated.items.len});
    if (generated.items.len > 0 and generated.items[generated.items.len - 1] == self.cfg.eos_token_id) {
        _ = generated.pop();
    }

    const text = try self.tok.decode(a, generated.items);
    defer a.free(text);
    const trimmed = std.mem.trim(u8, text, " \t\r\n\x0b\x0c");

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "{\"text\":");
    try laya.wireString(a, &out, trimmed);
    try out.appendSlice(a, ",\"language\":null,\"duration\":");
    try laya.pyFloat(a, &out, duration_s);
    try out.appendSlice(a, "}");
    return out.toOwnedSlice(a);
}

// ── tests ──────────────────────────────────────────────────────────────────

test "slaney mel filter bank: librosa shape and structural sanity" {
    const a = testing.allocator;
    const fb = try slaneyMelFilters(a, 16000, 400, 128, 0, 8000);
    defer a.free(fb);
    const n_freqs: usize = 400 / 2 + 1;
    try testing.expectEqual(@as(usize, 128 * n_freqs), fb.len);
    var ok = true;
    for (fb) |v| if (v < 0 or v > 2.0 + 1e-5) {
        ok = false;
    };
    try testing.expect(ok);
}

test "hann window is periodic and peaks mid-window" {
    const a = testing.allocator;
    const w = try hannWindow(a, 400);
    defer a.free(w);
    try testing.expectEqual(@as(f32, 0.0), w[0]);
    const last: f32 = 0.5 * (1.0 - @cos(2.0 * std.math.pi / 400.0));
    try testing.expectApproxEqAbs(last, w[399], 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 1.0), w[200], 1e-6);
}

test "reflect pad mirrors numpy semantics" {
    const a = testing.allocator;
    const x = [_]f32{ 1, 2, 3, 4 };
    const p = try reflectPad(a, &x, 2);
    defer a.free(p);
    try testing.expectEqual(@as(usize, 8), p.len);
    try testing.expectEqual(@as(f32, 3), p[0]);
    try testing.expectEqual(@as(f32, 2), p[1]);
    try testing.expectEqual(@as(f32, 3), p[6]);
    try testing.expectEqual(@as(f32, 2), p[7]);
}

test "numAudioTokens: 480 ms delay is 6 tokens" {
    try testing.expectEqual(@as(u32, 6), ModelConfig.numAudioTokens(7680));
}

test "config parse: the shipped checkpoint parses" {
    const a = testing.allocator;
    const bytes =
        \\{"model_type":"voxtral_realtime","encoder_args":{"dim":1280,"n_layers":32,
        \\"audio_encoding_args":{"sampling_rate":16000,"frame_rate":12.5}},
        \\"decoder":{"dim":3072,"n_layers":26},"transcription_delay_ms":480,
        \\"quantization":{"group_size":64,"bits":4,"mode":"affine"}}
    ;
    var cfg = try ModelConfig.parse(a, bytes);
    defer cfg.deinit(a);
    try testing.expectEqual(@as(u32, 1280), cfg.encoder_args.dim);
    try testing.expectEqual(@as(u32, 26), cfg.decoder.n_layers);
    try testing.expectEqual(@as(u32, 480), cfg.transcription_delay_ms);
    try testing.expectEqual(@as(u32, 6), ModelConfig.numDelayTokens(480));
}

// Slow native test: loads the real checkpoint and transcribes a test clip.
// Gated on ASR_TEST_MODEL so ordinary `zig build test` stays hermetic.
test "asr engine: native transcription smoke (ASR_TEST_MODEL)" {
    mlx.dropLatchedErrorUnless(mlx.errorPending());
    const model_dir_raw = std.c.getenv("ASR_TEST_MODEL") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(model_dir_raw, 0);
    const wav_path_raw = std.c.getenv("ASR_TEST_WAV") orelse return error.SkipZigTest;
    const wav_path = std.mem.sliceTo(wav_path_raw, 0);

    const a = std.heap.c_allocator;
    var threaded = std.Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const eng = Engine.load(io, a, model_dir) catch |err| {
        std.debug.print("ASRSMOKE load failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer eng.deinit();

    const f = try std.Io.Dir.openFileAbsolute(io, wav_path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const wav = try rs.interface.allocRemaining(a, .limited(32 * 1024 * 1024));
    defer a.free(wav);

    var t0_tv: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t0_tv);
    const t0: f64 = @as(f64, @floatFromInt(t0_tv.sec)) * 1000.0 + @as(f64, @floatFromInt(t0_tv.nsec)) / 1_000_000.0;
    const reply = transcribe(eng, a, wav, .{}) catch |err| {
        std.debug.print("ASRSMOKE transcribe failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer a.free(reply);
    var t1_tv: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t1_tv);
    const t1: f64 = @as(f64, @floatFromInt(t1_tv.sec)) * 1000.0 + @as(f64, @floatFromInt(t1_tv.nsec)) / 1_000_000.0;
    std.debug.print("ASRSMOKE ms={d} reply={s}\n", .{ @as(i64, @intFromFloat(t1 - t0)), reply });
}

test "asr qmm isolation (ASR_TEST_MODEL)" {
    mlx.dropLatchedErrorUnless(mlx.errorPending());
    const model_dir_raw = std.c.getenv("ASR_TEST_MODEL") orelse return error.SkipZigTest;
    const a = std.heap.c_allocator;
    var threaded = std.Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const eng = try Engine.load(io, a, std.mem.sliceTo(model_dir_raw, 0));
    defer eng.deinit();

    var ops = Ops.init(a, eng.stream);
    var x = mlx.mlx_array_new();
    _ = &ops;
    defer _ = mlx.mlx_array_free(x);
    const shape = [_]c_int{ 448, 1280 };
    const buf = try a.alloc(f32, 448 * 1280);
    defer a.free(buf);
    for (buf) |*v| v.* = 0.1;
    x = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    var cache_mem: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache_mem);
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    var old_limit: usize = 0;
    _ = mlx.mlx_set_memory_limit(&old_limit, 400 * 1024 * 1024 * 1024);
    std.debug.print("ASRQMM active={d} cache={d} peak={d} old_limit={d}\n", .{active, cache_mem, peak, old_limit});
    const q = try ops.qmm(x, &eng.enc_layers[0].q, true);
    defer free(q);
    const v = vectorOf(q);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
    std.debug.print("ASRQMM qmm eval ok shape {any}\n", .{mlx.getShape(q)});
}

test "asr tok_emb row 32 (ASR_TEST_MODEL)" {
    const model_dir_raw = std.c.getenv("ASR_TEST_MODEL") orelse return error.SkipZigTest;
    const a = std.heap.c_allocator;
    var threaded = std.Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const eng = try Engine.load(io, a, std.mem.sliceTo(model_dir_raw, 0));
    defer eng.deinit();
    var ops = Ops.init(a, eng.stream);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    const tshape = [_]c_int{1};
    const tid = [_]u32{32};
    x = mlx.mlx_array_new_data(&tid, &tshape, 1, .int32);
    const row = try ops.take(eng.tok_emb, x, 0);
    defer free(row);
    const v = vectorOf(row);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
    const f32row = try ops.astype(row, .float32);
    defer free(f32row);
    try mlx.check(mlx.mlx_array_eval(f32row));
    const data = mlx.mlx_array_data_float32(f32row) orelse return error.AsrDumpFail;
    std.debug.print("TOKEMB32[:6] {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} absmax={d:.4}\n", .{
        data[0], data[1], data[2], data[3], data[4], data[5], blk: {
            var m: f32 = 0;
            for (0..3072) |i| { const av = @abs(data[i]); if (av > m) m = av; }
            break :blk m;
        },
    });
}

test "asr transpose semantics probe" {
    const a = std.heap.c_allocator;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var ops = Ops.init(a, cpu);
    _ = &ops;
    // in[b, s, h, d] = 100*s + 10*h + d; transpose [0,2,1,3] -> [1, 3, 2, 2]
    // out[0, h, s, d]: convention A gives 100*s+10*h+d (per-head rows);
    // the swapped reading gives 10*s+100*h+d.
    const src_buf = [_]f32{ 0, 1, 10, 11, 100, 101, 110, 111, 200, 201, 210, 211 };
    const sshape = [_]c_int{ 3, 4 };
    const src = mlx.mlx_array_new_data(&src_buf, &sshape, 2, .float32);
    defer _ = mlx.mlx_array_free(src);
    const r4 = try ops.reshape(src, &[_]c_int{ 1, 3, 2, 2 });
    defer free(r4);
    const tr = try ops.transposeAxes(r4, &[_]c_int{ 0, 2, 1, 3 });
    defer free(tr);
    // read the LOGICAL order: flatten to 1-D forces a materialized copy of a
    // transposed view, then reshape back.
    const flat = try ops.reshape(tr, &[_]c_int{12});
    defer free(flat);
    try mlx.check(mlx.mlx_array_eval(flat));
    const back = try ops.reshape(flat, &[_]c_int{ 1, 2, 3, 2 });
    defer free(back);
    try mlx.check(mlx.mlx_array_eval(back));
    const data = mlx.mlx_array_data_float32(back) orelse return error.AsrDumpFail;
    const sh = mlx.getShape(tr);
    try testing.expectEqual(@as(usize, 4), sh.len);
    try testing.expectEqual(@as(c_int, 1), sh[0]);
    try testing.expectEqual(@as(c_int, 2), sh[1]);
    try testing.expectEqual(@as(c_int, 3), sh[2]);
    try testing.expectEqual(@as(c_int, 2), sh[3]);
    // out shape [1, 3(s), 2(h), 2(d)] under convention A (out.axis i = in.axis
    // axes[i]): out[0, s, h, d] = 100*s + 10*h + d. Print the first 12.
    std.debug.print("TRSP out[0,0,0,0]={d:.0} [0,0,1,0]={d:.0} [0,1,0,0]={d:.0} [0,2,0,0]={d:.0}\n", .{
        data[0], data[2], data[6], data[12],
    });
}

test "asr sdpa gqa broadcast isolation" {
    const a = std.heap.c_allocator;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var ops = Ops.init(a, cpu);
    _ = &ops;
    // q [1, 4, 1, 2] distinct per head; kv [1, 2, 1, 2] distinct per head.
    // causal s=0: out[h] = v[kv(h)] exactly. Reveals MLX's head mapping.
    const qv = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const kvv = [_]f32{ 100, 101, 200, 201 };
    const qshape = [_]c_int{ 1, 4, 1, 2 };
    const kshape = [_]c_int{ 1, 2, 1, 2 };
    const q = mlx.mlx_array_new_data(&qv, &qshape, 4, .float32);
    defer _ = mlx.mlx_array_free(q);
    const k = mlx.mlx_array_new_data(&kvv, &kshape, 4, .float32);
    defer _ = mlx.mlx_array_free(k);
    const v = mlx.mlx_array_new_data(&kvv, &kshape, 4, .float32);
    defer _ = mlx.mlx_array_free(v);
    const out = try sdpa(&ops, q, k, v, 0.08838834764831843, "causal", none);
    defer free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const data = mlx.mlx_array_data_float32(out) orelse return error.AsrDumpFail;
    std.debug.print("GQAISO out per qhead: h0={d:.1} h1={d:.1} h2={d:.1} h3={d:.1}\n", .{ data[0], data[2], data[4], data[6] });
    std.debug.print("GQAISO expect block(h/4): h0=100 h1=100 h2=200 h3=200; interleaved(h%%2): h0=100 h1=200 h2=100 h3=200\n", .{});
}

test "asr sdpa causal isolation" {
    const a = std.heap.c_allocator;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var ops = Ops.init(a, cpu);
    _ = &ops;
    // q [1, 1, 2, 2], k/v [1, 1, 2, 2] with distinct values; causal mask
    const qb = [_]f32{ 1, 0, 0, 1 };
    const kb = [_]f32{ 1, 2, 3, 4 };
    const vb = [_]f32{ 10, 20, 30, 40 };
    const qshape = [_]c_int{ 1, 1, 2, 2 };
    const q = mlx.mlx_array_new_data(&qb, &qshape, 4, .float32);
    defer _ = mlx.mlx_array_free(q);
    const k = mlx.mlx_array_new_data(&kb, &qshape, 4, .float32);
    defer _ = mlx.mlx_array_free(k);
    const v = mlx.mlx_array_new_data(&vb, &qshape, 4, .float32);
    defer _ = mlx.mlx_array_free(v);
    const out = try sdpa(&ops, q, k, v, 0.08838834764831843, "causal", none);
    defer free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const data = mlx.mlx_array_data_float32(out) orelse return error.AsrDumpFail;
    // causal: row0 attends key0 only -> out[0] = v[0] = [10, 20];
    // row1 attends keys 0,1 -> softmax([q1.k0, q1.k1]*s) = softmax([3, 4])
    // ~= [0.269, 0.731] -> out[1] ~= 0.269*[10,20] + 0.731*[30,40]
    std.debug.print("SDPAISO out[0]=({d:.3},{d:.3}) out[1]=({d:.3},{d:.3})\n", .{ data[0], data[1], data[2], data[3] });
}
