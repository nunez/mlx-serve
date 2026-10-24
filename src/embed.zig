//! fleet embed — text/vision chat C ABI over the mlx-serve engine, for a BEAM
//! (Elixir) NIF host.
//!
//! Scope: chat-completions surface (text + Qwen images in, token stream out).
//! Image / video / music / speech generation engines stay standalone; media model dirs
//! are refused at init. The HTTP server is not used — the host drives the
//! scheduler directly.
//!
//! Fork delta vs upstream mlx-serve: this file + the `mac-lib` step in
//! build.zig. ABI handshake: `mlxembed_abi_version()`. Keep the exported C
//! surface append-only.

const std = @import("std");
// The gguf + sushi_exl3 submodules reach these through the host root
// (`mlx_host` == this module), so they must be `pub`, mirroring main.zig.
pub const mlx = @import("mlx.zig");
pub const io_util = @import("io_util.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const chat_mod = @import("chat.zig");
const gen_mod = @import("gen.zig");
const yue2_mod = @import("yue2.zig");
const breeze_mod = @import("breeze.zig");
const wav_mod = @import("wav.zig");
const sse_mod = @import("gen_sse.zig");
const scheduler_mod = @import("scheduler.zig");
const server = @import("server.zig");
const model_registry_mod = @import("model_registry.zig");
const transformer_mod = @import("transformer.zig");
pub const log = @import("log.zig");
const build_options = @import("build_options");

// ─────────────────────────────────────────────────────────────────────────────
// Event protocol (sent via the C callback).
//   TEXT  (1): utf8 bytes of decoded text. May split anywhere (byte-aligned).
//   DONE  (2): payload is DoneInfo (see below).
//   ERR   (3): payload is a NUL-terminated UTF-8 message.
// ─────────────────────────────────────────────────────────────────────────────

pub const ABI_VERSION: u32 = 8;

pub const Event = enum(u32) {
    text = 1,
    done = 2,
    err = 3,
    /// Media progress: payload "stage step/total" (stage is the engine's own
    /// label — "[semantic]", "[nar]", "[vae]", "denoise"…).
    progress = 4,
    /// Media result: PNG bytes.
    png = 5,
    /// Media result: WAV bytes (48 kHz stereo 16-bit for YuE2).
    wav = 6,
    /// Media result: raw HTTP response (status line + headers + JSON body)
    /// from gen.handleVideo — the canonical /v1/video/generations reply.
    video = 7,
    /// Media result: raw HTTP response from gen.prepareDecisions +
    /// gen.handleDecisions — the canonical /v1/decisions reply.
    decision = 8,
    /// Speech stream chunk: raw s16le mono PCM (engine sample rate).
    /// Prefix-exact with the one-shot WAV under the causal codec decoder.
    pcm = 9,
    /// ASR transcription: the full reply JSON {"text","language","duration"}.
    asr = 10,
};

pub const Callback = *const fn (
    ctx: ?*anyopaque,
    gen_id: u64,
    event: u32,
    data: ?[*]const u8,
    len: u32,
) callconv(.c) void;

/// finish_reason code. Mirrors the scheduler's string values.
pub const Reason = enum(u8) {
    stop = 0,
    length = 1,
    abort = 2,
    content_filtered = 3,
    model_context_window_exceeded = 4,
    other = 255,
};

pub const DoneInfo = extern struct {
    finish_reason: u8,
    _pad: [3]u8,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
};

pub const InitConfig = extern struct {
    model_dir: [*:0]const u8,
    mlx_cache_mb: u64, // 0 = mlx default
    max_concurrent: u32, // 0 = 1
    max_resident_models: u32, // 0 = 4
    idle_evict_secs: u32, // 0 = none
    warmup: u8, // eager warmup forward pass
    enable_pld: u8, // n-gram speculative decode
    enable_mtp: u8, // native MTP sidecar if shipped
};

pub const GenParams = extern struct {
    prompt: ?[*]const u8,
    prompt_len: usize,
    model_id: ?[*:0]const u8, // null/empty = default model
    max_tokens: u32,
    temperature: f32, // <= 0 => greedy
    top_p: f32, // <= 0 or >= 1 => 1.0
    top_k: u32, // 0 => disabled
    presence_penalty: f32,
    repeat_penalty: f32, // 0 => 1.0
    seed: u64, // 0 => unset (random)
    request_timeout_ms: u64, // 0 => none
};

pub const EncodedImage = extern struct {
    data: ?[*]const u8,
    len: usize,
};

/// Media generation config (image + YuE2 music). `model_dir` selects a model
/// previously loaded via `mlxembed_media_load` (null/empty = the last loaded).
pub const MediaGenConfig = extern struct {
    model_dir: ?[*:0]const u8,
    prompt: ?[*]const u8, // image: the prompt; music: the style tags
    prompt_len: usize,
    lyrics: ?[*]const u8, // music only ([Verse]-style section tags on own lines)
    lyrics_len: usize,
    width: u32, // image: 0 => 1024
    height: u32, // image: 0 => 1024
    steps: u32, // 0 => engine default
    guidance: f32, // image CFG; <=0 => engine default (1.0)
    seed: u64, // 0 => random
    duration_seconds: u32, // music upper bound; 0 => model default
    abc: ?[*]const u8, // music: an ABC melody score (requires cot != off)
    abc_len: usize,
    cot: u8, // music: 0=off 1=melody 2=full (default full)
    // Image instruction editing: encoded reference buffers (PNG/JPEG), source
    // first. Routed to edit_image_bytes on byte-preprocessing backends
    // (Qwen-Image, MageFlow). 0 = plain generation. Pointers only need to
    // live for the call; the request context copies.
    edit_image_ptrs: ?[*]const [*]const u8 = null,
    edit_image_lens: ?[*]const usize = null,
    edit_image_count: usize = 0,
    /// Video: the canonical /v1/video/generations JSON body, passed through to
    /// gen.handleVideo verbatim (prompt/width/height/num_frames/steps/seed/
    /// cfg_scale/negative_prompt/image refs/stream...). The response arrives
    /// as one VIDEO event carrying the full JSON reply.
    video_json: ?[*]const u8 = null,
    video_json_len: usize = 0,
    /// Typed decisions (/v1/decisions schema): the canonical request JSON,
    /// passed through gen.prepareDecisions + gen.handleDecisions over a
    /// socketpair; the full HTTP reply arrives as one DECISION event.
    decision_json: ?[*]const u8 = null,
    decision_json_len: usize = 0,
    /// Speech (Breeze-TTS / Qwen3-TTS /v1/audio/speech): voice slot text
    /// ("S0".."S9", empty => backend default), instruction text (nullable),
    /// frame cap (0 => budget from text length). prompt carries the text.
    voice: ?[*]const u8 = null,
    voice_len: usize = 0,
    instruction: ?[*]const u8 = null,
    instruction_len: usize = 0,
    max_frames: u32 = 0,
    ref_audio: ?[*]const u8 = null,
    ref_audio_len: usize = 0,
    ref_text: ?[*]const u8 = null,
    ref_text_len: usize = 0,
    /// Speech streaming: 0 = one-shot WAV event, 1 = PCM chunks via the
    /// PCM event (raw s16le mono at the engine sample rate).
    stream: u8 = 0,
    /// Streaming: emit every N completed frames (0 = engine default 12).
    stream_chunk_frames: u32 = 0,
    /// ASR (/v1/audio/transcriptions): the 16 kHz mono s16 WAV bytes. The
    /// engine decodes, pads, and transcribes; the reply JSON arrives as one
    /// ASR event. prompt is unused for ASR.
    audio: ?[*]const u8 = null,
    audio_len: usize = 0,
};

// ─────────────────────────────────────────────────────────────────────────────

const asr_mod = @import("asr.zig");

const Embed = struct {
    allocator: std.mem.Allocator,
    threaded: std.Io.Threaded,
    io: std.Io,
    registry: *model_registry_mod.ModelRegistry,
    scheduler: *scheduler_mod.Scheduler,
    enable_pld: bool,
    enable_mtp: bool,
    shutting_down: std.atomic.Value(bool) = .init(false),
    feeders: std.atomic.Value(i64) = .init(0),
};

const Feeder = struct {
    embed: *Embed,
    slot: *scheduler_mod.Slot,
    lm: *model_registry_mod.LoadedModel,
    gen_id: u64,
    cb: Callback,
    cb_ctx: ?*anyopaque,
    strip_leading_space: bool,
};

var g_embed: ?*Embed = null;
var g_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_shutdown_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_generations_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_generations: std.AutoHashMapUnmanaged(u64, *scheduler_mod.Slot) = .{};
var g_next_id: std.atomic.Value(u64) = .init(1);
threadlocal var g_last_err: [512:0]u8 = std.mem.zeroes([512:0]u8);

fn setErr(comptime fmt: []const u8, args: anytype) void {
    const msg = std.fmt.bufPrint(g_last_err[0 .. g_last_err.len - 1], fmt, args) catch {
        g_last_err[0] = 0;
        return;
    };
    g_last_err[msg.len] = 0;
}

fn lastErr() []const u8 {
    return std.mem.span(g_last_err[0..].ptr);
}

/// mlx-c's default error handler is printf + exit(-1): any MLX failure
/// (e.g. a Metal resource-limit) would kill the host BEAM. Trap it instead:
/// log it, record it in the thread-local last error (mlxembed_last_error
/// surfaces it on the ERR event), and let the caller's return code carry the
/// failure. Installed process-wide; the handler is not re-entrant into MLX.
fn trapMlxError(msg: [*:0]const u8, data: ?*anyopaque) callconv(.c) void {
    _ = data;
    const span = std.mem.span(msg);
    log.err("[mlx] {s}\n", .{span});
    setErr("{s}", .{span});
    mlxFailRecord(span);
}

fn installErrorTrap() void {
    mlx.mlx_set_error_handler(trapMlxError, null, null);
}

// Process-wide failure latch: the trap handler records the first MLX failure
// here. The metal resource limit is a process-wide GPU budget — once it is
// exhausted, engine state (KV caches, weight buffers) is corrupt and no
// engine can reload its way out; the host must restart the process. TTS
// fails fast on this instead of spinning on corrupt arrays; other engines
// keep serving until the pool recovers.
var g_mlx_fail_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_mlx_fail_msg: [512:0]u8 = std.mem.zeroes([512:0]u8);
var g_mlx_fail_len: usize = 0;
var g_mlx_fail_set: std.atomic.Value(bool) = .init(false);

fn mlxFailRecord(msg: []const u8) void {
    _ = std.c.pthread_mutex_lock(&g_mlx_fail_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mlx_fail_mutex);
    if (!g_mlx_fail_set.load(.acquire)) {
        const n = @min(msg.len, g_mlx_fail_msg.len - 1);
        @memcpy(g_mlx_fail_msg[0..n], msg[0..n]);
        g_mlx_fail_len = n;
        g_mlx_fail_set.store(true, .release);
        log.err("[mlx-poison] latched first failure\n", .{});
    }
}

/// Copy of the latched failure message, or null when no failure was recorded.
fn mlxFailPeek(dst: []u8) ?[]const u8 {
    _ = std.c.pthread_mutex_lock(&g_mlx_fail_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mlx_fail_mutex);
    if (!g_mlx_fail_set.load(.acquire)) return null;
    const n = @min(g_mlx_fail_len, dst.len);
    @memcpy(dst[0..n], g_mlx_fail_msg[0..n]);
    return dst[0..n];
}

fn reasonCode(r: []const u8) u8 {
    if (std.mem.eql(u8, r, "stop")) return @backingInt(Reason.stop);
    if (std.mem.eql(u8, r, "length")) return @backingInt(Reason.length);
    if (std.mem.eql(u8, r, "abort")) return @backingInt(Reason.abort);
    if (std.mem.eql(u8, r, "content_filtered")) return @backingInt(Reason.content_filtered);
    if (std.mem.eql(u8, r, "model_context_window_exceeded")) return @backingInt(Reason.model_context_window_exceeded);
    return @backingInt(Reason.other);
}

fn basename(path: []const u8) []const u8 {
    var p = path;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    if (p.len == 0) return p;
    if (std.mem.lastIndexOfScalar(u8, p, '/')) |slash_idx| return p[slash_idx + 1 ..];
    return p;
}

fn defaultMaxResidentMem() u64 {
    var dev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return 0;
    var max_rec: usize = 0;
    if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") != 0 or max_rec == 0) return 0;
    return @as(u64, max_rec) * 4 / 5;
}

/// Mirrors server.zig:906 (decodeTokens) for one token. Routes through the
/// ds4/llama engines when loaded; otherwise the MLX tokenizer.
fn decodeToken(alloc: std.mem.Allocator, lm: *model_registry_mod.LoadedModel, id: u32) ![]u8 {
    if (lm.ds4_engine) |eng| return chat_mod.decodeViaDs4(alloc, eng, &[_]u32{id});
    if (lm.llama_engine) |eng| return chat_mod.decodeViaLlama(alloc, eng, &[_]u32{id});
    const t = lm.tokenizer orelse return error.NoTokenizer;
    return t.decode(alloc, &[_]u32{id}, false);
}

/// server.zig:7455 utf8TrailingIncomplete.
fn utf8TrailingIncomplete(s: []const u8) usize {
    if (s.len == 0) return 0;
    var i: usize = s.len;
    var cont: usize = 0;
    while (cont < 3 and i > 0) {
        i -= 1;
        if (s[i] & 0xC0 != 0x80) break;
        cont += 1;
    }
    if (i >= s.len) return 0;
    const lead = s[i];
    const expected: usize = if (lead & 0x80 == 0) 1 else if (lead & 0xE0 == 0xC0) 2 else if (lead & 0xF0 == 0xE0) 3 else if (lead & 0xF8 == 0xF0) 4 else return 0;
    const actual = s.len - i;
    return if (actual < expected) actual else 0;
}

const FeederResult = union(enum) { ok, errored };

fn feederMain(f: *Feeder) void {
    const e = f.embed;
    const alloc = e.allocator;

    var carry: [4]u8 = undefined;
    var carry_len: usize = 0;
    var result: FeederResult = .ok;
    var final_reason: []const u8 = "stop";

    const flush = struct {
        fn run(feeder: *Feeder, buf: []const u8) void {
            if (buf.len > 0) {
                feeder.cb(feeder.cb_ctx, feeder.gen_id, @backingInt(Event.text), buf.ptr, @intCast(buf.len));
            }
        }
    }.run;

    defer {
        _ = std.c.pthread_mutex_lock(&g_generations_mutex);
        _ = g_generations.remove(f.gen_id);
        _ = std.c.pthread_mutex_unlock(&g_generations_mutex);
        // Emit DONE or ERR based on result, then release resources.
        if (result == .ok) {
            const info = DoneInfo{
                .finish_reason = reasonCode(final_reason),
                ._pad = .{ 0, 0, 0 },
                .prompt_tokens = f.slot.prompt_tokens,
                .cached_tokens = f.slot.cached_tokens,
                .completion_tokens = f.slot.completion_tokens,
                .prefill_ns = f.slot.prefill_ns,
                .decode_ns = f.slot.decode_ns,
            };
            const ptr: [*]const u8 = @ptrCast(&info);
            f.cb(f.cb_ctx, f.gen_id, @backingInt(Event.done), ptr, @sizeOf(DoneInfo));
        } else {
            f.cb(f.cb_ctx, f.gen_id, @backingInt(Event.err), null, 0);
        }
        e.scheduler.complete(f.slot);
        e.scheduler.release(f.lm);
        _ = e.feeders.fetchSub(1, .release);
        alloc.destroy(f);
    }

    while (true) {
        const nr = f.slot.waitNextTimeout(200) orelse {
            if (e.shutting_down.load(.acquire)) break;
            continue;
        };
        switch (nr) {
            .token => |tok_id| {
                if (f.slot.cancelled.load(.acquire)) {
                    final_reason = "abort";
                    break;
                }
                const raw = decodeToken(alloc, f.lm, tok_id) catch {
                    setErr("token decode failed", .{});
                    result = .errored;
                    break;
                };
                defer alloc.free(raw);
                // Fold any carry bytes + this token's bytes.
                var combined = raw;
                var owned = false;
                if (carry_len > 0) {
                    const merged = tryOrFree(alloc, carry[0..carry_len], raw);
                    if (merged) |m| {
                        combined = m;
                        owned = true;
                    } else {
                        setErr("oom: UTF-8 token carry", .{});
                        result = .errored;
                        break;
                    }
                }
                defer if (owned) alloc.free(combined);

                const tail = utf8TrailingIncomplete(combined);
                if (tail > 0) {
                    @memcpy(carry[0..tail], combined[combined.len - tail ..]);
                    carry_len = tail;
                } else {
                    carry_len = 0;
                }
                const emit_len = combined.len - tail;
                if (emit_len == 0) continue;

                flush(f, combined[0..emit_len]);
            },
            .done => {
                final_reason = if (f.slot.cancelled.load(.acquire))
                    "abort"
                else
                    f.slot.finish_reason;
                break;
            },
            .err => {
                setErr("generation: {s}", .{f.slot.error_code orelse "unknown error"});
                result = .errored;
                break;
            },
        }
    }
}

fn tryOrFree(alloc: std.mem.Allocator, a: []const u8, b: []const u8) ?[]u8 {
    const out = alloc.alloc(u8, a.len + b.len) catch return null;
    @memcpy(out[0..a.len], a);
    @memcpy(out[a.len..], b);
    return out;
}

fn embedOr(neg_code: i32) ?*Embed {
    const e = g_embed orelse {
        setErr("engine not initialized", .{});
        return null;
    };
    _ = neg_code;
    return e;
}

// ─────────────────────────────────────────────────────────────────────────────
// Exports
// ─────────────────────────────────────────────────────────────────────────────

export fn mlxembed_abi_version() callconv(.c) u32 {
    return ABI_VERSION;
}

export fn mlxembed_version() callconv(.c) [*:0]const u8 {
    return std.fmt.comptimePrint("{s}", .{build_options.version});
}

export fn mlxembed_last_error() callconv(.c) [*:0]const u8 {
    return g_last_err[0..].ptr;
}

export fn mlxembed_init(cfg: *const InitConfig) callconv(.c) i32 {
    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    if (g_embed != null) {
        setErr("already initialized", .{});
        return -1;
    }
    installErrorTrap();
    const alloc = std.heap.c_allocator;
    const model_dir = std.mem.span(cfg.model_dir);
    if (model_dir.len == 0) {
        setErr("empty model_dir", .{});
        return -2;
    }

    var threaded = std.Io.Threaded.init(alloc, .{});
    errdefer threaded.deinit();
    const io = threaded.io();

    // Device selection (mirrors ios_lib.zig / main.zig).
    var metal_avail: bool = false;
    _ = mlx.mlx_metal_is_available(&metal_avail);
    const force_cpu = if (std.c.getenv("MLXSERVE_DEVICE")) |d|
        std.mem.eql(u8, std.mem.span(d), "cpu")
    else
        false;
    if (force_cpu) {
        const cpu_dev = mlx.mlx_device_new_type(.cpu, 0);
        defer _ = mlx.mlx_device_free(cpu_dev);
        _ = mlx.mlx_set_default_device(cpu_dev);
    } else if (metal_avail) {
        const gpu_dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu_dev);
        _ = mlx.mlx_set_default_device(gpu_dev);
    }

    if (cfg.mlx_cache_mb > 0) {
        var old: usize = 0;
        _ = mlx.mlx_set_cache_limit(&old, cfg.mlx_cache_mb * 1024 * 1024);
    }

    // Media guard — embed serves text/chat only.
    if (gen_mod.detectModality(io, alloc, model_dir)) |modality| {
        setErr("media model ({s}) not supported by embed — text/chat only", .{@tagName(modality)});
        return -3;
    }

    // Config / tokenizer / chat config (mirrors main.zig 993-1085).
    const config_storage = alloc.create(model_mod.ModelConfig) catch {
        setErr("oom: ModelConfig", .{});
        return -4;
    };
    errdefer alloc.destroy(config_storage);
    config_storage.* = model_mod.parseConfig(io, alloc, model_dir) catch |err| {
        setErr("parseConfig: {s}", .{@errorName(err)});
        return -4;
    };

    const tok = alloc.create(tokenizer_mod.Tokenizer) catch {
        setErr("oom: Tokenizer", .{});
        return -5;
    };
    errdefer alloc.destroy(tok);
    tok.* = tokenizer_mod.loadTokenizer(io, alloc, model_dir) catch |err| {
        setErr("loadTokenizer: {s}", .{@errorName(err)});
        return -5;
    };

    const chat_config = alloc.create(chat_mod.ChatConfig) catch {
        setErr("oom: ChatConfig", .{});
        return -6;
    };
    errdefer alloc.destroy(chat_config);
    chat_config.* = chat_mod.loadChatConfig(io, alloc, model_dir) catch |err| {
        setErr("loadChatConfig: {s}", .{@errorName(err)});
        return -6;
    };

    // EOS merge (main.zig 1059-1081).
    if (chat_config.eos_token) |eos_str| {
        if (tok.special_tokens.get(eos_str)) |eos_id| {
            if (!config_storage.isEosToken(eos_id)) config_storage.addEosToken(eos_id);
        }
    }
    if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
        if (!config_storage.isEosToken(eot_id)) config_storage.addEosToken(eot_id);
    }
    if (tok.special_tokens.get("<pad>")) |pad_id| {
        if (pad_id > 0 and !config_storage.isEosToken(pad_id)) config_storage.addEosToken(pad_id);
    }

    // Registry.
    const model_id = basename(model_dir);
    const registry = model_registry_mod.ModelRegistry.init(
        alloc,
        io,
        null, // discovery_storage
        if (cfg.max_resident_models == 0) 4 else cfg.max_resident_models,
        defaultMaxResidentMem(),
        if (cfg.idle_evict_secs == 0) null else cfg.idle_evict_secs,
    ) catch |err| {
        setErr("ModelRegistry.init: {s}", .{@errorName(err)});
        return -7;
    };
    errdefer registry.deinit();

    const entry = registry.registerStub(model_id, model_dir, null) catch |err| {
        setErr("registerStub: {s}", .{@errorName(err)});
        return -7;
    };
    registry.setDefault(entry.id) catch |err| {
        setErr("setDefault: {s}", .{@errorName(err)});
        return -7;
    };

    // Scheduler — blocks until the model is loaded + warmed.
    const max_concurrent = if (cfg.max_concurrent == 0) 1 else cfg.max_concurrent;
    const params = scheduler_mod.LoadParams{
        .registry = registry,
        .entry = entry,
        .config = config_storage,
        .tok = tok,
        .chat_config = chat_config,
        .model_dir = model_dir,
        .load_vision = config_storage.has_vision,
        .warmup_eager = cfg.warmup != 0,
        .mtp_enabled = cfg.enable_mtp != 0,
        .prefix_cache_capacity = 1,
        .prefix_cache_mem_bytes = 256 * 1024 * 1024,
        .tokenize_cache_entries = 4,
    };
    const scheduler = scheduler_mod.Scheduler.init(alloc, io, params, max_concurrent) catch |err| {
        setErr("Scheduler.init: {s}", .{@errorName(err)});
        return -8;
    };
    errdefer scheduler.deinit();

    const e = alloc.create(Embed) catch {
        setErr("oom: Embed", .{});
        return -9;
    };
    e.* = .{
        .allocator = alloc,
        .threaded = threaded,
        .io = io,
        .registry = registry,
        .scheduler = scheduler,
        .enable_pld = cfg.enable_pld != 0,
        .enable_mtp = cfg.enable_mtp != 0,
    };
    // Ownership of config/tok/chat_config now sits on `entry` (installed by
    // the scheduler load); registry.deinit frees them. threaded moves into e.
    g_embed = e;

    log.info("mlx-serve embed {s} (MLX path) — text/chat ABI v{d}, model {s}\n", .{
        build_options.version,
        ABI_VERSION,
        model_id,
    });
    return 0;
}

export fn mlxembed_shutdown() callconv(.c) void {
    _ = std.c.pthread_mutex_lock(&g_shutdown_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_shutdown_mutex);
    _ = std.c.pthread_mutex_lock(&g_mutex);
    const e = g_embed orelse {
        _ = std.c.pthread_mutex_unlock(&g_mutex);
        return;
    };
    e.shutting_down.store(true, .release);
    e.scheduler.cancelAllInFlight();
    _ = std.c.pthread_mutex_unlock(&g_mutex);

    // Wait for feeder threads to drain (cancel triggers their terminal event).
    var spins: usize = 0;
    while (e.feeders.load(.acquire) > 0) : (spins += 1) {
        if (spins == 2000) log.warn("embed shutdown still draining cancelled generations\n", .{});
        const ts = std.c.timespec{ .sec = 0, .nsec = 5_000_000 };
        _ = std.c.nanosleep(&ts, null);
    }

    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    e.scheduler.deinit();
    e.registry.deinit();
    e.threaded.deinit();
    const alloc = e.allocator;
    alloc.destroy(e);
    g_embed = null;
    _ = std.c.pthread_mutex_lock(&g_generations_mutex);
    g_generations.deinit(alloc);
    g_generations = .{};
    _ = std.c.pthread_mutex_unlock(&g_generations_mutex);
}

export fn mlxembed_generate(cfg: *const GenParams, cb: Callback, cb_ctx: ?*anyopaque) callconv(.c) i64 {
    return generate(cfg, &.{}, cb, cb_ctx);
}

/// Encoded bytes are borrowed only until this call returns; preprocessing
/// is CPU-only and the scheduler runs the encoder on its inference thread.
export fn mlxembed_generate_images(cfg: *const GenParams, images: ?[*]const EncodedImage, image_count: usize, cb: Callback, cb_ctx: ?*anyopaque) callconv(.c) i64 {
    if (image_count > 0 and images == null) {
        setErr("images: missing encoded image array", .{});
        return -10;
    }
    return generate(cfg, if (images) |p| p[0..image_count] else &.{}, cb, cb_ctx);
}

fn generate(cfg: *const GenParams, images: []const EncodedImage, cb: Callback, cb_ctx: ?*anyopaque) i64 {
    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    const e = embedOr(-1) orelse return -1;
    if (e.shutting_down.load(.acquire)) {
        setErr("engine shutting down", .{});
        return -2;
    }
    const alloc = e.allocator;

    const model_id: []const u8 = if (cfg.model_id) |p| std.mem.span(p) else "";
    const prompt: []const u8 = if (cfg.prompt) |p| p[0..cfg.prompt_len] else "";
    if (prompt.len == 0) {
        setErr("empty prompt", .{});
        return -3;
    }

    const lm = e.scheduler.ensureLoaded(model_id) catch |err| {
        setErr("ensureLoaded: {s}", .{@errorName(err)});
        return -4;
    };
    // lm is refcounted; feeder releases it on terminal event. If we fail
    // after this point, release it here.

    const t = lm.tokenizer orelse {
        e.scheduler.release(lm);
        setErr("model has no tokenizer", .{});
        return -5;
    };
    var prompt_ids = t.encode(alloc, prompt) catch |err| {
        e.scheduler.release(lm);
        setErr("tokenize: {s}", .{@errorName(err)});
        return -5;
    };
    defer alloc.free(prompt_ids);

    const config = lm.config orelse {
        e.scheduler.release(lm);
        setErr("model has no config", .{});
        return -6;
    };

    var mm: server.MultimodalPrompt = .{};
    defer mm.deinit(alloc);
    if (config.qwen_vision and std.mem.count(u32, prompt_ids, &.{config.image_token_id}) != images.len) {
        e.scheduler.release(lm);
        setErr("images: prompt must contain one image placeholder per attached image", .{});
        return -10;
    }
    if (images.len > 0) {
        if (!config.qwen_vision or lm.vision_encoder == null) {
            e.scheduler.release(lm);
            setErr("images: model has no Qwen vision encoder", .{});
            return -10;
        }
        const decoded = alloc.alloc(chat_mod.ImageData, images.len) catch {
            e.scheduler.release(lm);
            setErr("oom: images", .{});
            return -11;
        };
        var decoded_count: usize = 0;
        defer {
            for (decoded[0..decoded_count]) |im| alloc.free(im.pixels);
            alloc.free(decoded);
        }
        const vp = server.visionPreprocFromConfig(config);
        for (images, 0..) |encoded, i| {
            if (encoded.data == null or encoded.len == 0 or encoded.len > std.math.maxInt(c_int)) {
                e.scheduler.release(lm);
                setErr("images: invalid encoded image {d}", .{i});
                return -10;
            }
            decoded[i] = server.decodeImageToPixels(alloc, encoded.data.?[0..encoded.len], vp) orelse {
                e.scheduler.release(lm);
                setErr("images: could not decode image {d}", .{i});
                return -10;
            };
            decoded_count += 1;
        }
        const messages = [_]chat_mod.Message{.{ .role = "user", .content = "", .images = decoded }};
        mm = server.prepareMultimodalPrompt(alloc, e.scheduler, lm, config, &messages, &prompt_ids) catch |err| {
            e.scheduler.release(lm);
            setErr("images: {s}", .{@errorName(err)});
            return if (err == error.MediaPlaceholderMismatch) -10 else -11;
        };
        if (mm.mrope.pos == null) {
            e.scheduler.release(lm);
            setErr("images: missing Qwen M-RoPE positions", .{});
            return -11;
        }
    }

    const enable_mtp = e.enable_mtp and (lm.mtp != null);
    const temperature: f32 = @max(0, cfg.temperature);
    const top_p: f32 = if (cfg.top_p <= 0 or cfg.top_p >= 1) 1.0 else cfg.top_p;
    const repeat_penalty: f32 = if (cfg.repeat_penalty == 0) 1.0 else cfg.repeat_penalty;
    const seed: ?u64 = if (cfg.seed == 0) null else cfg.seed;

    const params = scheduler_mod.SubmitParams{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = false,
        .sampling = .{
            .temperature = temperature,
            .top_p = top_p,
            .top_k = cfg.top_k,
            .repeat_penalty = repeat_penalty,
            .presence_penalty = cfg.presence_penalty,
            .seed = seed,
        },
        .eos_token_ids = config.eosTokenSlice(),
        .max_tokens = cfg.max_tokens,
        .timeout_ns = cfg.request_timeout_ms * std.time.ns_per_ms,
        .enable_pld = e.enable_pld and !enable_mtp,
        .enable_mtp = enable_mtp,
        .mtp = if (enable_mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .pld_draft_len = 5,
        .pld_key_len = 3,
        .logprobs_n = 0,
        .vision_embeddings = mm.embeddings,
        .media = mm.media,
        .mrope_pos = mm.mrope.pos,
        .mrope_total = mm.mrope.total,
        .mrope_delta = mm.mrope.delta,
    };
    // submit consumes embeddings/positions even on failure, as on the HTTP path.
    mm.embeddings = null;
    mm.mrope = .{};
    const slot = e.scheduler.submit(params) catch |err| {
        e.scheduler.release(lm);
        setErr("submit: {s}", .{@errorName(err)});
        return -7;
    };

    const gen_id = g_next_id.fetchAdd(1, .monotonic);
    const f = alloc.create(Feeder) catch {
        e.scheduler.complete(slot);
        e.scheduler.release(lm);
        setErr("oom: Feeder", .{});
        return -8;
    };
    f.* = .{
        .embed = e,
        .slot = slot,
        .lm = lm,
        .gen_id = gen_id,
        .cb = cb,
        .cb_ctx = cb_ctx,
        .strip_leading_space = false,
    };
    _ = std.c.pthread_mutex_lock(&g_generations_mutex);
    g_generations.put(alloc, gen_id, slot) catch {
        _ = std.c.pthread_mutex_unlock(&g_generations_mutex);
        e.scheduler.complete(slot);
        e.scheduler.release(lm);
        alloc.destroy(f);
        setErr("oom: generation registry", .{});
        return -8;
    };
    _ = std.c.pthread_mutex_unlock(&g_generations_mutex);
    _ = e.feeders.fetchAdd(1, .release);
    const th = std.Thread.spawn(.{}, feederMain, .{f}) catch {
        _ = std.c.pthread_mutex_lock(&g_generations_mutex);
        _ = g_generations.remove(gen_id);
        _ = std.c.pthread_mutex_unlock(&g_generations_mutex);
        _ = e.feeders.fetchSub(1, .release);
        e.scheduler.complete(slot);
        e.scheduler.release(lm);
        alloc.destroy(f);
        setErr("thread spawn failed", .{});
        return -9;
    };
    th.detach();
    return @intCast(gen_id);
}

export fn mlxembed_cancel(gen_id: u64) callconv(.c) i32 {
    _ = std.c.pthread_mutex_lock(&g_generations_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_generations_mutex);
    const slot = g_generations.get(gen_id) orelse {
        setErr("generation not found", .{});
        return -2;
    };
    slot.cancel();
    return 0;
}

export fn mlxembed_cancel_all() callconv(.c) i32 {
    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    const e = embedOr(-1) orelse return -1;
    e.scheduler.cancelAllInFlight();
    return 0;
}

export fn mlxembed_unload(model_id: [*:0]const u8) callconv(.c) i32 {
    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    const e = embedOr(-1) orelse return -1;
    const id = std.mem.span(model_id);
    e.scheduler.unloadModel(id) catch |err| {
        setErr("unloadModel: {s}", .{@errorName(err)});
        return -2;
    };
    return 0;
}

export fn mlxembed_load(model_dir: [*:0]const u8) callconv(.c) i32 {
    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    const e = embedOr(-1) orelse return -1;
    const dir = std.mem.span(model_dir);
    if (dir.len == 0) {
        setErr("empty model_dir", .{});
        return -2;
    }
    const id = basename(dir);
    if (e.registry.peekByPath(dir) == null and e.registry.peek(id) == null) {
        _ = e.registry.registerStub(id, dir, null) catch |err| {
            setErr("registerStub: {s}", .{@errorName(err)});
            return -3;
        };
    }
    const lm = e.scheduler.ensureLoaded(id) catch |err| {
        setErr("ensureLoaded: {s}", .{@errorName(err)});
        return -4;
    };
    e.scheduler.release(lm);
    return 0;
}

export fn mlxembed_tokenize(prompt: [*]const u8, prompt_len: usize) callconv(.c) i64 {
    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    const e = embedOr(-1) orelse return -1;
    const text: []const u8 = prompt[0..prompt_len];
    const lm = e.scheduler.ensureLoaded("") catch |err| {
        setErr("ensureLoaded: {s}", .{@errorName(err)});
        return -2;
    };
    defer e.scheduler.release(lm);
    const t = lm.tokenizer orelse return -3;
    const ids = t.encode(e.allocator, text) catch |err| {
        setErr("tokenize: {s}", .{@errorName(err)});
        return -3;
    };
    defer e.allocator.free(ids);
    return @intCast(ids.len);
}

export fn mlxembed_stats(out: ?[*]u64) callconv(.c) i32 {
    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    const e = embedOr(-1) orelse return -1;
    if (out) |o| {
        o[0] = e.scheduler.inflight_generated_tokens.load(.acquire);
        o[1] = e.scheduler.inflight_prefill_tokens.load(.acquire);
        o[2] = e.scheduler.requests_prefilling.load(.acquire);
        o[3] = @intCast(e.feeders.load(.acquire));
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Media sidecar (ABI v2, append-only): image + YuE2 music generation driven
// directly by the NIF host. A single dedicated worker thread owns model load
// and generation — MLX default streams are thread-local, so the thread that
// loads an engine is the thread that must run it (mirrors the server's
// scheduler). Media models live beside the chat engine and share the
// process-global MLX device; the host budgets residency (Fleet resource
// claims). One generation in flight at a time; cancellation via a
// per-generation flag polled by the engine's progress probe. Result bytes
// are borrowed for the duration of the callback — the host copies.
// ─────────────────────────────────────────────────────────────────────────────

pub const MediaEngine = union(enum) {
    image: *gen_mod.ImageEngine,
    yue2: *yue2_mod.Engine,
    tts: *gen_mod.AudioEngine, // speech backends (.breeze / .tts)
    video: *gen_mod.VideoEngine,
    decision: *gen_mod.DecisionEngine,
    asr: *asr_mod.Engine,
};

const MediaGenCtx = struct {
    cb: Callback,
    cb_ctx: ?*anyopaque,
    gen_id: u64,
    cancelled: *std.atomic.Value(bool),
    alloc: std.mem.Allocator,
    dir: []u8,
    prompt: []u8,
    lyrics: []u8,
    abc: []u8,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    seed: u64,
    duration_seconds: u32,
    cot: u8,
    voice: []u8,
    instruction: []u8,
    ref_audio: []u8,
    ref_text: []u8,
    audio: []u8,
    max_frames: u32,
    stream: u8,
    stream_chunk_frames: u32,
    edit_images: []const []const u8,
    failbuf: [512:0]u8 = std.mem.zeroes([512:0]u8),
    video_json: []u8,
    decision_json: []u8,
};

const MediaLoadReq = struct {
    dir: []u8,
    code: i32 = -99,
    done: bool = false,
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
};

const MediaJob = union(enum) {
    load: *MediaLoadReq,
    unload: *MediaLoadReq,
    generate: *MediaGenCtx,
    shutdown,
};

var g_media_threaded: ?std.Io.Threaded = null;
var g_media: std.StringHashMapUnmanaged(MediaEngine) = .{};
var g_media_default: []const u8 = "";
var g_media_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_media_gen_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_media_gen: std.AutoHashMapUnmanaged(u64, *std.atomic.Value(bool)) = .{};
var g_media_inflight: std.atomic.Value(i64) = .init(0);
var g_jobs: std.ArrayListUnmanaged(MediaJob) = .empty;
var g_jobs_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_jobs_cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER;
var g_worker: ?std.Thread = null;
var g_shutdown: std.atomic.Value(bool) = .init(false);

fn mediaDefaultBasename(dir: []const u8) []const u8 {
    var key = dir;
    if (std.mem.indexOfScalar(u8, dir, '/') != null) {
        var it = std.mem.splitScalar(u8, dir, '/');
        while (it.next()) |s| key = s;
    }
    return key;
}

fn mediaSetDefaultDir(alloc: std.mem.Allocator, key: []const u8) void {
    const duped = alloc.dupe(u8, key) catch return;
    // g_media_default starts as the static ""; free only the heap dupes.
    if (g_media_default.len > 0) alloc.free(@constCast(g_media_default));
    g_media_default = duped;
}

/// Detect the modality at `dir` and load the matching engine on the media
/// worker thread. Returns null (and sets the last error) on any failure.
/// Does NOT register the engine in g_media — the caller owns that.
/// One-time process-wide MLX cache limit for the media path. The text-engine
/// init sets its own via cfg.mlx_cache_mb; media-only nodes (fleet) never run
/// that path, so latch a sane bound here. Env: MLXSERVE_MEDIA_CACHE_MB (MB).
var g_media_cache_latched: bool = false;

fn ensureMediaCacheLimit() void {
    if (g_media_cache_latched) return;
    g_media_cache_latched = true;
    var mb: usize = 1024;
    if (std.c.getenv("MLXSERVE_MEDIA_CACHE_MB")) |v| {
        mb = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1024;
    }
    var old: usize = 0;
    _ = mlx.mlx_set_cache_limit(&old, mb * 1024 * 1024);
    log.info("[memstat] media cache limit set to {d}MB (was {d}B)\n", .{ mb, old });
}

/// Memory pressure snapshot — active/cache/peak MLX bytes. Cheap (mutex + ints).
fn logMemStat(tag: []const u8) void {
    var act: usize = 0;
    var cac: usize = 0;
    var pek: usize = 0;
    _ = mlx.mlx_get_active_memory(&act);
    _ = mlx.mlx_get_cache_memory(&cac);
    _ = mlx.mlx_get_peak_memory(&pek);
    log.info("[memstat] {s}: active={d}MB cache={d}MB peak={d}MB\n", .{
        tag,
        act / (1024 * 1024),
        cac / (1024 * 1024),
        pek / (1024 * 1024),
    });
}

fn mediaConstructEngine(io: std.Io, alloc: std.mem.Allocator, dir: []const u8) ?MediaEngine {
    ensureMediaCacheLimit();
    const modality = gen_mod.detectModality(io, alloc, dir) orelse {
        setErr("media: undetectable model at {s}", .{dir});
        return null;
    };
    return blk: {
        switch (modality) {
            .image => {
                const e = gen_mod.ImageEngine.load(io, alloc, dir) catch |err| {
                    setErr("media image load: {s}", .{@errorName(err)});
                    break :blk null;
                };
                break :blk MediaEngine{ .image = e };
            },
            .audio => {
                const ae = gen_mod.AudioEngine.load(io, alloc, dir) catch |err| {
                    setErr("media audio load: {s}", .{@errorName(err)});
                    break :blk null;
                };
                switch (ae.backend) {
                    .yue2 => |ye| break :blk MediaEngine{ .yue2 = ye },
                    .breeze => break :blk MediaEngine{ .tts = ae },
                    else => {
                        ae.deinit();
                        setErr("media audio: backend {s} not supported by embed", .{@tagName(ae.backend)});
                        break :blk null;
                    },
                }
            },
            .video => {
                const e = gen_mod.VideoEngine.load(io, alloc, dir) catch |err| {
                    setErr("media video load: {s}", .{@errorName(err)});
                    break :blk null;
                };
                break :blk MediaEngine{ .video = e };
            },
            .decision => {
                const e = gen_mod.DecisionEngine.load(io, alloc, dir) catch |err| {
                    setErr("media decision load: {s}", .{@errorName(err)});
                    break :blk null;
                };
                break :blk MediaEngine{ .decision = e };
            },
            .asr => {
                const e = asr_mod.Engine.load(io, alloc, dir) catch |err| {
                    setErr("media asr load: {s}", .{@errorName(err)});
                    break :blk null;
                };
                break :blk MediaEngine{ .asr = e };
            },
            else => {
                setErr("media: modality {s} not supported by embed", .{@tagName(modality)});
                break :blk null;
            },
        }
    };
}

fn doMediaLoad(req: *MediaLoadReq) void {
    const alloc = std.heap.c_allocator;
    const io = g_media_threaded.?.io();
    const dir = req.dir;
    _ = std.c.pthread_mutex_lock(&req.mutex);
    defer _ = std.c.pthread_mutex_unlock(&req.mutex);
    const key = mediaDefaultBasename(dir);
    _ = std.c.pthread_mutex_lock(&g_media_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_media_mutex);
    if (g_media.contains(key)) {
        req.code = 0;
        req.done = true;
        _ = std.c.pthread_cond_signal(&req.cond);
        return;
    }
    const engine = mediaConstructEngine(io, alloc, dir) orelse {
        req.code = -3;
        req.done = true;
        _ = std.c.pthread_cond_signal(&req.cond);
        return;
    };
    g_media.put(alloc, alloc.dupe(u8, key) catch key, engine) catch {
        setErr("oom: media map", .{});
        req.code = -5;
        req.done = true;
        _ = std.c.pthread_cond_signal(&req.cond);
        return;
    };
    mediaSetDefaultDir(alloc, key);
    log.info("mlx-serve embed media: {s} ({s}) ready\n", .{ key, @tagName(engine) });
    logMemStat("after-load");
    req.code = 0;
    req.done = true;
    _ = std.c.pthread_cond_signal(&req.cond);
}

/// Free a single media engine. Every engine's deinit() already destroys its
/// own struct (self.allocator.destroy(self)), so calling destroy again is a
/// double-free — that is what aborted the worker mid-lifecycle before the
/// fix. Do NOT destroy here.
fn mediaDeinitEngine(engine: MediaEngine) void {
    switch (engine) {
        .image => |e| e.deinit(),
        .yue2 => |e| e.deinit(),
        .video => |e| e.deinit(),
        .decision => |e| e.deinit(),
        .tts => |e| e.deinit(),
        .asr => |e| e.deinit(),
    }
}

/// Unload one resident media engine by dir (matched on basename), freeing its
/// weights. Serialized on the media worker queue, so it cannot race an
/// in-flight generation: any earlier `.generate` job finishes first.
fn doMediaUnload(req: *MediaLoadReq) void {
    const alloc = std.heap.c_allocator;
    const dir = req.dir;
    _ = std.c.pthread_mutex_lock(&req.mutex);
    defer _ = std.c.pthread_mutex_unlock(&req.mutex);
    const key = mediaDefaultBasename(dir);
    _ = std.c.pthread_mutex_lock(&g_media_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_media_mutex);
    const entry = g_media.getEntry(key) orelse {
        setErr("media: model not loaded: {s}", .{key});
        req.code = -2;
        req.done = true;
        _ = std.c.pthread_cond_signal(&req.cond);
        return;
    };
    const key_slice: []const u8 = entry.key_ptr.*;
    const engine = entry.value_ptr.*;
    // remove does not free the key string (unmanaged); free it explicitly.
    g_media.removeByPtr(entry.key_ptr);
    alloc.free(key_slice);
    mediaDeinitEngine(engine);
    // If this was the last-loaded default, drop it so a null-dir generate
    // fails cleanly instead of pointing at a dead engine.
    if (g_media_default.len > 0 and std.mem.eql(u8, g_media_default, key_slice)) {
        alloc.free(@constCast(g_media_default));
        g_media_default = "";
    }
    log.info("mlx-serve embed media: {s} unloaded\n", .{key});
    logMemStat("after-unload");
    req.code = 0;
    req.done = true;
    _ = std.c.pthread_cond_signal(&req.cond);
}

/// Runs gen.handleVideo over a socketpair and emits the complete raw HTTP
/// response (status line + headers + JSON body) as one VIDEO event. Elixir
/// splits the head; the body is the canonical /v1/video/generations reply.
/// Runs the /v1/decisions pipeline over a socketpair: gen.prepareDecisions
/// validates (answering errors itself), gen.handleDecisions predicts and
/// writes the HTTP reply. The captured bytes go out as one DECISION event.
fn runMediaDecision(c: *MediaGenCtx, eng: *gen_mod.DecisionEngine) void {
    const alloc = c.alloc;
    const io = g_media_threaded.?.io();

    var sv: [2]std.c.fd_t = undefined;
    // AF_UNIX, SOCK_STREAM
    if (std.c.socketpair(1, 1, 0, &sv) != 0) {
        const msg = "decision: socketpair failed";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    }

    var out: std.ArrayList(u8) = .empty;

    const Ctx = struct {
        fd: std.c.fd_t,
        out: *std.ArrayList(u8),
        fn run(x: *@This()) void {
            defer _ = std.c.close(x.fd);
            var buf: [65_536]u8 = undefined;
            while (true) {
                const n = std.posix.read(x.fd, &buf) catch break;
                if (n == 0) break;
                x.out.appendSlice(std.heap.c_allocator, buf[0..n]) catch break;
            }
        }
    };

    var cx = Ctx{ .fd = sv[1], .out = &out };
    const th = std.Thread.spawn(.{}, Ctx.run, .{&cx}) catch {
        _ = std.c.close(sv[0]);
        _ = std.c.close(sv[1]);
        out.deinit(alloc);
        const msg = "decision: spawn failed";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    };

    // prepare + predict on THIS thread; the engine's stream belongs here.
    {
        var conn: server.Conn = undefined;
        server.Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, io);
        var req_opt = gen_mod.prepareDecisions(alloc, &conn, c.decision_json, eng) catch |err| {
            _ = std.c.close(sv[0]);
            th.join();
            out.deinit(alloc);
            const msg = std.fmt.allocPrint(alloc, "decision: {s}", .{@errorName(err)}) catch "decision error";
            c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
            return;
        };
        if (req_opt) |*req| {
            defer req.deinit(alloc);
            var jobs = [_]gen_mod.DecisionJob{.{ .allocator = alloc, .conn = &conn, .req = req }};
            gen_mod.handleDecisions(eng, mediaDefaultBasename(c.dir), &jobs);
        }
        // else: prepareDecisions already answered (validation error) —
        // that response was captured like any other.
    }
    _ = std.c.close(sv[0]);
    th.join();

    const bytes = out.items;
    if (bytes.len == 0) {
        out.deinit(alloc);
        const msg = "decision: empty response";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    }
    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.decision), bytes.ptr, @intCast(bytes.len));
    mediaEmitDone(c, 0, 0);
    out.deinit(alloc);
}

fn runMediaAsr(c: *MediaGenCtx, eng: *asr_mod.Engine) void {
    if (c.audio.len == 0) {
        const msg = "asr: no audio bytes";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    }
    const reply = asr_mod.transcribe(eng, c.alloc, c.audio, .{
        .cancelled = c.cancelled,
    }) catch |err| {
        const msg = if (err == error.Cancelled) "cancelled" else @errorName(err);
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    };
    defer c.alloc.free(reply);
    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.asr), reply.ptr, @intCast(reply.len));
    mediaEmitDone(c, 0, 0);
}

fn runMediaVideo(c: *MediaGenCtx, eng: *gen_mod.VideoEngine) void {
    const alloc = c.alloc;
    const io = g_media_threaded.?.io();

    var sv: [2]std.c.fd_t = undefined;
    // AF_UNIX, SOCK_STREAM
    if (std.c.socketpair(1, 1, 0, &sv) != 0) {
        const msg = "video: socketpair failed";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    }

    const Ctx = struct {
        eng: *gen_mod.VideoEngine,
        io: std.Io,
        fd: std.c.fd_t,
        cancelled: *std.atomic.Value(bool),
        body: []const u8,
        out: *std.ArrayList(u8),
        fn run(x: *@This()) void {
            defer _ = std.c.close(x.fd);
            // Drain the socketpair; handleVideo runs on the caller thread so
            // its MLX streams stay on the thread that loaded the engine.
            var buf: [65_536]u8 = undefined;
            while (true) {
                const n = std.posix.read(x.fd, &buf) catch break;
                if (n == 0) break;
                x.out.appendSlice(std.heap.c_allocator, buf[0..n]) catch break;
            }
        }
    };

    var out: std.ArrayList(u8) = .empty;
    var cx = Ctx{ .eng = eng, .io = io, .fd = sv[1], .cancelled = c.cancelled, .body = c.video_json, .out = &out };
    const th = std.Thread.spawn(.{}, Ctx.run, .{&cx}) catch {
        _ = std.c.close(sv[0]);
        _ = std.c.close(sv[1]);
        const msg = "video: spawn failed";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    };

    // handleVideo on THIS thread (the media worker) — streams must match the
    // thread that loaded the engine.
    var conn: server.Conn = undefined;
    server.Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, io);
    gen_mod.handleVideo(io, alloc, &conn, c.video_json, eng) catch |err| {
        _ = std.c.close(sv[0]);
        th.join();
        out.deinit(alloc);
        const msg = std.fmt.allocPrint(alloc, "video: {s}", .{@errorName(err)}) catch "video error";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    };
    _ = std.c.close(sv[0]);
    th.join();

    const bytes = out.items;
    if (bytes.len == 0) {
        out.deinit(alloc);
        const msg = "video: empty response";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    }
    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.video), bytes.ptr, @intCast(bytes.len));
    mediaEmitDone(c, 0, 0);
    out.deinit(alloc);
}

fn runMediaGenerate(c: *MediaGenCtx) void {
    defer mediaFeederCleanup(c);
    const alloc = c.alloc;
    const s = mlx.mlx_default_gpu_stream_new();
    _ = mlx.mlx_set_default_stream(s);
    _ = mlx.mlx_stream_free(s);
    const progress = sse_mod.Progress{
        .ctx = @ptrCast(c),
        .cb = mediaProgress,
        .cancelled_cb = mediaCancelled,
    };

    const key = mediaDefaultBasename(c.dir);
    const io = g_media_threaded.?.io();
    _ = std.c.pthread_mutex_lock(&g_media_mutex);
    var engine = g_media.get(key);
    if (engine == null) {
        // On-demand: auto-load on first use. Serialized on this worker thread
        // so it cannot race another generation; freed weights (from a later
        // unload) return to the Metal allocator and lower the buffer count.
        const loaded = mediaConstructEngine(io, c.alloc, c.dir);
        if (loaded) |e| {
            g_media.put(c.alloc, c.alloc.dupe(u8, key) catch key, e) catch {
                setErr("oom: media map", .{});
            };
            mediaSetDefaultDir(c.alloc, key);
            log.info("mlx-serve embed media: {s} ({s}) ready (auto)\n", .{ key, @tagName(e) });
            engine = e;
        }
    }
    _ = std.c.pthread_mutex_unlock(&g_media_mutex);

    if (engine == null) {
        const msg = "media: model not loaded";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    }

    switch (engine.?) {
        .asr => |eng| {
            runMediaAsr(c, eng);
        },
        .video => |eng| {
            runMediaVideo(c, eng);
        },
        .decision => |eng| {
            runMediaDecision(c, eng);
        },
        .image => |eng| {
            const w: u32 = if (c.width == 0) 1024 else c.width;
            const h: u32 = if (c.height == 0) 1024 else c.height;
            const seed: u64 = c.seed;
            var opts = gen_mod.ImageGenOpts{};
            if (c.edit_images.len > 0) {
                if (!eng.supportsEdit()) {
                    const msg = "editing needs an edit-trained image model (Qwen-Image, FLUX.2, Mage-Flow-Edit)";
                    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                    return;
                }
                if (c.edit_images.len > gen_mod.editRefCap(eng.backend)) {
                    const msg = "too many reference images for this model";
                    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                    return;
                }
                opts.edit_image_bytes = c.edit_images;
            }
            const png = eng.generatePngOpts(alloc, c.prompt, w, h, seed, c.steps, opts, progress) catch |err| {
                const msg = if (err == error.Cancelled) "cancelled" else @errorName(err);
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            };
            defer alloc.free(png);
            c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.png), png.ptr, @intCast(png.len));
            mediaEmitDone(c, c.steps, 0);
        },
        .tts => |ae| {
            if (ae.backend != .breeze) {
                const msg = "speech: only breeze_tts is supported by embed";
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            }
            if (mlxFailPeek(c.failbuf[0..]) != null) {
                ttsMlxFailEvent(c);
                return;
            }
            const be = ae.backend.breeze;
            const req = breeze_mod.Request{
                .text = c.prompt,
                .voice = if (c.voice.len > 0) c.voice else null,
                .instruction = if (c.instruction.len > 0) c.instruction else null,
                .ref_audio = if (c.ref_audio.len > 0) c.ref_audio else null,
                .ref_text = if (c.ref_text.len > 0) c.ref_text else null,
                .max_frames = if (c.max_frames == 0) null else c.max_frames,
                .seed = if (c.seed == 0) null else c.seed,
            };
            if (c.stream != 0) {
                const synth = be.synthesizeStream(alloc, req, breeze_mod.StreamSpec{
                    .emit = ttsPcmEmit,
                    .emit_user = c,
                    .chunk_frames = c.stream_chunk_frames,
                    .cancel = mediaCancelled,
                    .cancel_user = c,
                }) catch |err| {
                    const msg = if (err == error.Cancelled) "cancelled" else @errorName(err);
                    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                    return;
                };
                // Stream mode: PCM left via ttsPcmEmit; Synthesis.pcm is a
                // static empty slice (do not free). On success the frames
                // buffer is ours (run's errdefer only fires on error).
                alloc.free(@constCast(synth.frames));
                if (mlxFailPeek(c.failbuf[0..]) != null) {
                    ttsMlxFailEvent(c);
                    return;
                }
                mediaEmitDone(c, 0, 0);
                logMemStat("after-tts");
                return;
            }
            const synth = be.synthesize(alloc, req) catch |err| {
                const msg = if (err == error.Cancelled) "cancelled" else @errorName(err);
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            };
            defer alloc.free(synth.pcm);
            // A Metal failure mid-synthesis (e.g. the resource limit) can leave
            // empty arrays behind; synthesize may still return "successfully"
            // with corrupt PCM. Drop it and fail the job instead.
            if (mlxFailPeek(c.failbuf[0..]) != null) {
                ttsMlxFailEvent(c);
                return;
            }
            const wav = wav_mod.encodePcm16Mono(alloc, synth.pcm, be.sampleRate()) catch |err| {
                const msg = if (err == error.Cancelled) "cancelled" else @errorName(err);
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            };
            defer alloc.free(wav);
            c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.wav), wav.ptr, @intCast(wav.len));
            mediaEmitDone(c, 0, 0);
            logMemStat("after-tts");
        },
        .yue2 => |eng| {
            const cot: yue2_mod.Cot = switch (c.cot) {
                0 => .off,
                1 => .melody,
                else => .full,
            };
            if (c.lyrics.len == 0) {
                const msg = "empty lyrics (YuE2 is lyric-conditioned)";
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            }
            if (c.abc.len > 0 and cot == .off) {
                const msg = "abc needs cot melody|full";
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            }
            const secs: u32 = std.math.clamp(if (c.duration_seconds == 0) yue2_mod.MAX_SECONDS else c.duration_seconds, yue2_mod.MIN_SECONDS, yue2_mod.MAX_SECONDS);
            const req = yue2_mod.Request{
                .style = c.prompt,
                .lyrics = c.lyrics,
                .cot = cot,
                .abc = if (c.abc.len > 0) c.abc else null,
                .seed = if (c.seed == 0) 831001 else c.seed,
                .cfg_scale = if (c.guidance > 0) @floatCast(c.guidance) else null,
                .steps = if (c.steps == 0) null else c.steps,
                .max_frames = secs * yue2_mod.FRAME_RATE,
            };
            const fits_ok = eng.fits(alloc, req) catch false;
            if (!fits_ok) {
                const msg = "style + lyrics + score leave no room in the context";
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            }
            const song = eng.generate(alloc, req, progress) catch |err| {
                const msg = if (err == error.Cancelled) "cancelled" else @errorName(err);
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            };
            defer {
                alloc.free(song.wav);
                if (song.abc) |a| alloc.free(a);
            }
            if (song.abc) |a| c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.text), a.ptr, @intCast(a.len));
            c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.wav), song.wav.ptr, @intCast(song.wav.len));
            mediaEmitDone(c, req.steps orelse yue2_mod.DEFAULT_STEPS, 0);
        },
    }
}

fn mediaWorkerLoop() void {
    installErrorTrap();
    const alloc = std.heap.c_allocator;
    g_media_threaded = std.Io.Threaded.init(alloc, .{});
    // Same device selection as mlxembed_init: Metal when available, else CPU.
    var metal_avail: bool = false;
    _ = mlx.mlx_metal_is_available(&metal_avail);
    const force_cpu = if (std.c.getenv("MLXSERVE_DEVICE")) |d|
        std.mem.eql(u8, std.mem.span(d), "cpu")
    else
        false;
    if (force_cpu) {
        const cpu_dev = mlx.mlx_device_new_type(.cpu, 0);
        defer _ = mlx.mlx_device_free(cpu_dev);
        _ = mlx.mlx_set_default_device(cpu_dev);
    } else if (metal_avail) {
        const gpu_dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu_dev);
        _ = mlx.mlx_set_default_device(gpu_dev);
    }

    while (true) {
        _ = std.c.pthread_mutex_lock(&g_jobs_mutex);
        while (g_jobs.items.len == 0 and !g_shutdown.load(.acquire)) {
            _ = std.c.pthread_cond_wait(&g_jobs_cond, &g_jobs_mutex);
        }
        if (g_jobs.items.len == 0 and g_shutdown.load(.acquire)) {
            _ = std.c.pthread_mutex_unlock(&g_jobs_mutex);
            break;
        }
        const job = g_jobs.orderedRemove(0);
        _ = std.c.pthread_mutex_unlock(&g_jobs_mutex);

        switch (job) {
            .load => |req| doMediaLoad(req),
            .unload => |req| doMediaUnload(req),
            .generate => |c| runMediaGenerate(c),
            .shutdown => {},
        }
    }

    mediaTeardown();
}

fn mediaEnsureWorker() void {
    if (g_worker != null) return;
    g_worker = std.Thread.spawn(.{}, mediaWorkerLoop, .{}) catch {
        setErr("media: worker spawn failed", .{});
        return;
    };
}

fn mediaEnqueue(job: MediaJob) void {
    mediaEnsureWorker();
    _ = std.c.pthread_mutex_lock(&g_jobs_mutex);
    g_jobs.append(std.heap.c_allocator, job) catch {
        _ = std.c.pthread_mutex_unlock(&g_jobs_mutex);
        return;
    };
    _ = std.c.pthread_cond_signal(&g_jobs_cond);
    _ = std.c.pthread_mutex_unlock(&g_jobs_mutex);
}

fn mediaProgress(ctx: *anyopaque, stage: []const u8, step: u32, total: u32) void {
    const c: *MediaGenCtx = @ptrCast(@alignCast(ctx));
    var buf: [256]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "{s} {d}/{d}", .{ stage, step, total }) catch return;
    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.progress), msg.ptr, @intCast(msg.len));
}

fn mediaCancelled(ctx: *anyopaque) bool {
    const c: *MediaGenCtx = @ptrCast(@alignCast(ctx));
    return c.cancelled.load(.acquire);
}

/// Streaming speech emit: f32 → s16le mono PCM with the wav.encodePcm16
/// scaling (clamp ±1, round to 32767), sent as one PCM event per chunk.
/// Runs on the media feeder thread; the callback copies the payload before
/// returning, so the buffer is safe to free afterwards.
fn ttsPcmEmit(user: *anyopaque, pcm: []const f32) void {
    const c: *MediaGenCtx = @ptrCast(@alignCast(user));
    if (pcm.len == 0) return;
    const out = c.alloc.alloc(u8, pcm.len * 2) catch {
        const msg = "speech: pcm encode failed";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    };
    defer c.alloc.free(out);
    var off: usize = 0;
    for (pcm) |s| {
        const clamped = std.math.clamp(s, -1.0, 1.0);
        const v: i16 = @intFromFloat(std.math.round(clamped * 32767.0));
        out[off] = @intCast(v & 0xFF);
        out[off + 1] = @intCast((v >> 8) & 0xFF);
        off += 2;
    }
    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.pcm), out.ptr, @intCast(out.len));
}

fn mediaFeederCleanup(c: *MediaGenCtx) void {
    const alloc = c.alloc;
    alloc.free(c.dir);
    alloc.free(c.prompt);
    alloc.free(c.lyrics);
    alloc.free(c.abc);
    if (c.voice.len > 0) alloc.free(c.voice);
    if (c.ref_audio.len > 0) alloc.free(c.ref_audio);
    if (c.ref_text.len > 0) alloc.free(c.ref_text);
    if (c.audio.len > 0) alloc.free(c.audio);
    if (c.instruction.len > 0) alloc.free(c.instruction);
    for (c.edit_images) |b| alloc.free(b);
    if (c.edit_images.len > 0) alloc.free(c.edit_images);
    if (c.video_json.len > 0) alloc.free(c.video_json);
    if (c.decision_json.len > 0) alloc.free(c.decision_json);
    _ = std.c.pthread_mutex_lock(&g_media_gen_mutex);
    _ = g_media_gen.remove(c.gen_id);
    _ = std.c.pthread_mutex_unlock(&g_media_gen_mutex);
    _ = g_media_inflight.fetchSub(1, .release);
    alloc.destroy(c);
}

/// ERR event for a TTS job aborted by a process-wide MLX failure; payload
/// carries "speech: mlx failure: <latched message>" (the host's send_err
/// prefers the last-error string; the payload is for the logs).
fn ttsMlxFailEvent(c: *MediaGenCtx) void {
    const prefix: []const u8 = "speech: mlx failure: ";
    var out: [600:0]u8 = std.mem.zeroes([600:0]u8);
    if (mlxFailPeek(c.failbuf[0..])) |m| {
        const n = @min(m.len, out.len - prefix.len);
        std.mem.copyForwards(u8, out[0..prefix.len], prefix);
        std.mem.copyForwards(u8, out[prefix.len .. prefix.len + n], m[0..n]);
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), out[0..].ptr, @intCast(prefix.len + n));
    } else {
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), out[0..].ptr, @intCast(prefix.len));
    }
}

fn mediaEmitDone(c: *MediaGenCtx, steps: u32, elapsed_ns: u64) void {
    var info = DoneInfo{
        .finish_reason = @backingInt(Reason.stop),
        ._pad = .{ 0, 0, 0 },
        .prompt_tokens = 0,
        .cached_tokens = 0,
        .completion_tokens = steps,
        .prefill_ns = 0,
        .decode_ns = elapsed_ns,
    };
    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.done), @ptrCast(&info), @sizeOf(DoneInfo));
}

export fn mlxembed_media_load(model_dir: [*:0]const u8) callconv(.c) i32 {
    const alloc = std.heap.c_allocator;
    const dir = std.mem.span(model_dir);
    if (dir.len == 0) {
        setErr("media: empty model_dir", .{});
        return -2;
    }
    const req = alloc.create(MediaLoadReq) catch return -5;
    req.* = .{ .dir = alloc.dupe(u8, dir) catch {
        alloc.destroy(req);
        return -5;
    } };
    mediaEnqueue(.{ .load = req });
    _ = std.c.pthread_mutex_lock(&req.mutex);
    defer _ = std.c.pthread_mutex_unlock(&req.mutex);
    while (!req.done) {
        _ = std.c.pthread_cond_wait(&req.cond, &req.mutex);
    }
    const code = req.code;
    alloc.free(req.dir);
    alloc.destroy(req);
    return code;
}

export fn mlxembed_media_unload(model_dir: [*:0]const u8) callconv(.c) i32 {
    const alloc = std.heap.c_allocator;
    const dir = std.mem.span(model_dir);
    if (dir.len == 0) {
        setErr("media: empty model_dir", .{});
        return -2;
    }
    const req = alloc.create(MediaLoadReq) catch return -5;
    req.* = .{ .dir = alloc.dupe(u8, dir) catch {
        alloc.destroy(req);
        return -5;
    } };
    mediaEnqueue(.{ .unload = req });
    _ = std.c.pthread_mutex_lock(&req.mutex);
    defer _ = std.c.pthread_mutex_unlock(&req.mutex);
    while (!req.done) {
        _ = std.c.pthread_cond_wait(&req.cond, &req.mutex);
    }
    const code = req.code;
    alloc.free(req.dir);
    alloc.destroy(req);
    return code;
}

export fn mlxembed_media_generate(cfg: *const MediaGenConfig, cb: Callback, cb_ctx: ?*anyopaque) callconv(.c) i64 {
    if (g_shutdown.load(.acquire)) {
        setErr("media: shutting down", .{});
        return -7;
    }
    const alloc = std.heap.c_allocator;
    const dir: []const u8 = if (cfg.model_dir) |p| std.mem.span(p) else g_media_default;
    if (dir.len == 0) {
        setErr("media: no model loaded", .{});
        return -1;
    }
    const prompt: []const u8 = if (cfg.prompt) |p| p[0..cfg.prompt_len] else "";
    const lyrics: []const u8 = if (cfg.lyrics) |p| p[0..cfg.lyrics_len] else "";
    const abc: []const u8 = if (cfg.abc) |p| p[0..cfg.abc_len] else "";
    if (prompt.len == 0 and cfg.audio_len == 0) {
        setErr("media: empty prompt", .{});
        return -3;
    }

    const c = alloc.create(MediaGenCtx) catch return -3;
    const cancelled = alloc.create(std.atomic.Value(bool)) catch {
        alloc.destroy(c);
        return -3;
    };
    cancelled.* = .init(false);
    c.* = .{
        .cb = cb,
        .cb_ctx = cb_ctx,
        .gen_id = 0,
        .cancelled = cancelled,
        .alloc = alloc,
        .dir = alloc.dupe(u8, dir) catch {
            alloc.destroy(c);
            alloc.destroy(cancelled);
            return -3;
        },
        .prompt = alloc.dupe(u8, prompt) catch {
            alloc.free(c.dir);
            alloc.destroy(c);
            alloc.destroy(cancelled);
            return -3;
        },
        .lyrics = alloc.dupe(u8, lyrics) catch {
            alloc.free(c.dir);
            alloc.free(c.prompt);
            alloc.destroy(c);
            alloc.destroy(cancelled);
            return -3;
        },
        .abc = alloc.dupe(u8, abc) catch {
            alloc.free(c.dir);
            alloc.free(c.prompt);
            alloc.free(c.lyrics);
            alloc.destroy(c);
            alloc.destroy(cancelled);
            return -3;
        },
        .width = cfg.width,
        .height = cfg.height,
        .steps = cfg.steps,
        .guidance = cfg.guidance,
        .seed = cfg.seed,
        
        .duration_seconds = cfg.duration_seconds,
        .cot = cfg.cot,
        .voice = if (cfg.voice) |p| alloc.dupe(u8, p[0..cfg.voice_len]) catch &.{} else &.{},
        .instruction = if (cfg.instruction) |p| alloc.dupe(u8, p[0..cfg.instruction_len]) catch &.{} else &.{},
        .ref_audio = if (cfg.ref_audio) |p| alloc.dupe(u8, p[0..cfg.ref_audio_len]) catch &.{} else &.{},
        .audio = if (cfg.audio) |p| alloc.dupe(u8, p[0..cfg.audio_len]) catch &.{} else &.{},
        .ref_text = if (cfg.ref_text) |p| alloc.dupe(u8, p[0..cfg.ref_text_len]) catch &.{} else &.{},
        .max_frames = cfg.max_frames,
        .stream = cfg.stream,
        .stream_chunk_frames = cfg.stream_chunk_frames,
        .edit_images = &.{},
        .video_json = if (cfg.video_json) |p| alloc.dupe(u8, p[0..cfg.video_json_len]) catch &.{} else &.{},
        .decision_json = if (cfg.decision_json) |p| alloc.dupe(u8, p[0..cfg.decision_json_len]) catch &.{} else &.{},
    };
    if (cfg.edit_image_count > 0 and cfg.edit_image_ptrs != null and cfg.edit_image_lens != null) {
        const n = cfg.edit_image_count;
        const cap = gen_mod.MAX_EDIT_IMAGES;
        if (n > cap) {
            mediaFeederCleanup(c);
            setErr("media: too many edit images", .{});
            return -3;
        }
        const slices = alloc.alloc([]const u8, n) catch {
            mediaFeederCleanup(c);
            return -3;
        };
        const ptrs = cfg.edit_image_ptrs.?;
        const lens = cfg.edit_image_lens.?;
        var i: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < i) : (j += 1) alloc.free(slices[j]);
            alloc.free(slices);
        }
        while (i < n) : (i += 1) {
            if (lens[i] == 0) {
                mediaFeederCleanup(c);
                setErr("media: empty edit image", .{});
                return -3;
            }
            slices[i] = alloc.dupe(u8, ptrs[i][0..lens[i]]) catch {
                mediaFeederCleanup(c);
                return -3;
            };
        }
        c.edit_images = slices;
    }
    const gen_id = g_next_id.fetchAdd(1, .monotonic);
    c.gen_id = gen_id;
    _ = std.c.pthread_mutex_lock(&g_media_gen_mutex);
    g_media_gen.put(alloc, gen_id, cancelled) catch {
        _ = std.c.pthread_mutex_unlock(&g_media_gen_mutex);
        mediaFeederCleanup(c);
        setErr("oom: media gen registry", .{});
        return -4;
    };
    _ = std.c.pthread_mutex_unlock(&g_media_gen_mutex);
    _ = g_media_inflight.fetchAdd(1, .release);
    mediaEnqueue(.{ .generate = c });
    return @intCast(gen_id);
}

export fn mlxembed_media_cancel(gen_id: u64) callconv(.c) i32 {
    _ = std.c.pthread_mutex_lock(&g_media_gen_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_media_gen_mutex);
    const flag = g_media_gen.get(gen_id) orelse {
        setErr("media: generation not found", .{});
        return -2;
    };
    flag.store(true, .release);
    return 0;
}

fn mediaTeardown() void {
    var it = g_media.iterator();
    while (it.next()) |entry| {
        mediaDeinitEngine(entry.value_ptr.*);
    }
    g_media.clearAndFree(std.heap.c_allocator);
    std.heap.c_allocator.free(@constCast(g_media_default));
    g_media_default = "";
    if (g_media_threaded) |*t| t.deinit();
    g_media_threaded = null;
}

export fn mlxembed_media_shutdown() callconv(.c) void {
    mediaEnqueue(.shutdown);
    g_shutdown.store(true, .release);
    _ = std.c.pthread_cond_broadcast(&g_jobs_cond);
    if (g_worker) |w| {
        w.join();
        g_worker = null;
    }
}


test "reason code mapping" {
    try std.testing.expectEqual(@backingInt(Reason.stop), reasonCode("stop"));
    try std.testing.expectEqual(@backingInt(Reason.length), reasonCode("length"));
    try std.testing.expectEqual(@backingInt(Reason.other), reasonCode("nope"));
}
