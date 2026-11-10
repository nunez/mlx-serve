//! fleet embed — the mlx-serve engine as a C ABI static lib for a BEAM (Elixir) NIF host.
//!
//! Scope: chat-completions (text + vision images in, token stream out), the
//! media sidecar (media-plugin routes + Breeze TTS + realtime ASR), and the
//! embeddings surface (ABI v9: a dedicated EmbeddingGemma-2 instance over the
//! scheduler's computeEmbeddings path). The HTTP server is not used — the host
//! drives the scheduler directly.
//!
//! Fork delta vs upstream mlx-serve: this file + `src/embed/` (the realtime
//! ASR engine) + the `mac-lib` step in `build/embed_lib.zig`. ABI handshake:
//! `mlxembed_abi_version()`. Keep the exported C surface append-only.

const std = @import("std");
// The gguf + exl3 SDK code reaches these through the host root (`mlx_host`
// == this module), so they must be `pub`, mirroring main.zig.
pub const mlx = @import("mlx/ffi.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const env = @import("env.zig");
const model_mod = @import("sdk/config/model.zig");
const tokenizer_mod = @import("sdk/tokenizer/tokenizer.zig");
const tokenizer_load = @import("sdk/tokenizer/load.zig");
const chat_mod = @import("host/chat/chat.zig");
const scheduler_mod = @import("host/gpu_thread/scheduler.zig");
const mlx_engine = @import("host/engines/mlx.zig");
const gpu_params = @import("host/gpu_thread/params.zig");
const gpu_requests = @import("host/gpu_thread/requests.zig");
const model_registry_mod = @import("host/registry/registry.zig");
const model_settings_mod = @import("host/registry/model_settings.zig");
const model_settings_apply = @import("host/registry/model_settings_apply.zig");
const media_encode = @import("host/media/encode.zig");
const media_decode = @import("host/media/decode.zig");
const media_prepare = @import("host/media/prepare.zig");
const media_placeholders = @import("host/media/placeholders.zig");
const request_media_mod = @import("host/media/request_media.zig");
const tokenize_api = @import("host/api/tokenize.zig");
const embeddings_api = @import("host/api/embeddings.zig");
const gen_mod = @import("host/media_jobs/gen.zig");
const plugin_served = @import("host/media_jobs/plugin/served.zig");
const plugin_engine = @import("host/media_jobs/plugin/engine.zig");
const plugin_http = @import("host/media_jobs/plugin/http.zig");
const http_server = @import("host/http/server.zig");
const http_conn = @import("host/http/conn.zig");
const breeze_mod = @import("plugins/breeze_tts/engine.zig");
const embed_asr = @import("embed/asr.zig");
const wav_mod = @import("sdk/media/wav.zig");
pub const log = @import("sdk/util/log.zig");
const build_options = @import("build_options");
// The generated plugin registry: its comptime block exports every in-tree
// plugin's table (`mlx_serve_plugin_<name>_v1`), which the NIF .so must carry
// for the media-plugin dlopen path. main.zig imports it the same way.
const plugin_registry = @import("plugins/registry.zig");
comptime {
    _ = plugin_registry.entries.len;
    // One compile unit (the embed lib never builds under -Dunits): the
    // engine's table is exported from here, as main.zig does for the exe.
    if (!build_options.units) _ = @import("engines/mlx/api.zig");
}

// ─────────────────────────────────────────────────────────────────────────────
// Event protocol (sent via the C callback).
//   TEXT  (1): utf8 bytes of decoded text. May split anywhere (byte-aligned).
//   DONE  (2): payload is DoneInfo (see below).
//   ERR   (3): payload is a NUL-terminated UTF-8 message.
// ─────────────────────────────────────────────────────────────────────────────

pub const ABI_VERSION: u32 = 9;

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
    /// — the canonical reply of the route that produced it.
    video = 7,
    /// Media result: raw HTTP response — the canonical /v1/decisions reply.
    decision = 8,
    /// Speech stream chunk: raw s16le mono PCM (engine sample rate).
    /// Prefix-exact with the one-shot WAV under the causal codec decoder.
    pcm = 9,
    /// ASR transcription: the full reply JSON {"text","language","duration"}.
    asr = 10,
    /// Embeddings ABI v9: every pooled vector of ONE embed request, packed
    /// little-endian — [count u32][dims u32][count*dims f32, row-major].
    /// DONE follows with the row count in completion_tokens and the compute
    /// duration in prefill_ns.
    embed_rows = 11,
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

/// Media generation config (media-plugin routes + Breeze TTS). `model_dir`
/// selects a model previously loaded via `mlxembed_media_load` (null/empty =
/// the last loaded).
pub const MediaGenConfig = extern struct {
    model_dir: ?[*:0]const u8,
    prompt: ?[*]const u8, // image/music: the prompt; speech: the text
    prompt_len: usize,
    lyrics: ?[*]const u8, // music only ([Verse]-style section tags on own lines)
    lyrics_len: usize,
    width: u32, // image: 0 => plugin default
    height: u32,
    steps: u32, // 0 => engine default
    guidance: f32, // image CFG; <=0 => engine default (1.0)
    seed: u64, // 0 => random
    duration_seconds: u32, // music upper bound; 0 => model default
    abc: ?[*]const u8, // music: an ABC melody score (requires cot != off)
    abc_len: usize,
    cot: u8, // music: 0=off 1=melody 2=full (default full)
    /// Image instruction editing: encoded reference buffers (PNG/JPEG), source
    /// first. Serialized into the request body as `ref_images`. 0 = plain
    /// generation. Pointers only need to live for the call; the request
    /// context copies.
    edit_image_ptrs: ?[*]const [*]const u8 = null,
    edit_image_lens: ?[*]const usize = null,
    edit_image_count: usize = 0,
    /// Video: the canonical /v1/video/generations JSON body, passed through
    /// the media plugin verbatim. The response arrives as one VIDEO event
    /// carrying the full raw HTTP reply.
    video_json: ?[*]const u8 = null,
    video_json_len: usize = 0,
    /// Typed decisions (/v1/decisions schema): the canonical request JSON.
    decision_json: ?[*]const u8 = null,
    decision_json_len: usize = 0,
    /// Speech (Breeze-TTS): voice slot text ("S0".."S9", empty => backend
    /// default), instruction text (nullable), frame cap (0 => budget from
    /// text length). prompt carries the text.
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
    /// ASR: the 16 kHz mono s16 WAV bytes. The engine decodes, pads, and
    /// transcribes; the reply JSON arrives as one ASR event. prompt is
    /// unused for ASR.
    audio: ?[*]const u8 = null,
    audio_len: usize = 0,
};

/// Embeddings ABI v9 (append-only): batched text and one-shot media embedding
/// over a dedicated embedder engine (booted via `mlxembed_embed_init`).
/// Pointers are borrowed only until the call returns — the request context
/// copies what it needs before the feeder thread starts.
pub const EmbedTextParams = extern struct {
    model_id: ?[*:0]const u8 = null, // null/empty = the embedder's default model
    texts: ?[*]const [*]const u8 = null, // `count` entries, parallel to `lens`
    lens: ?[*]const usize = null,
    count: usize = 0,
};

pub const EmbedMediaParams = extern struct {
    model_id: ?[*:0]const u8 = null, // null/empty = the embedder's default model
    text: ?[*]const u8 = null, // optional text part ("" / null = pure image(s))
    text_len: usize = 0,
    images: ?[*]const EncodedImage = null, // PNG/JPEG/WebP bytes, one soft-token run each
    image_count: usize = 0,
};

/// The `embed_rows` event payload header, then the packed f32 rows
/// (row-major, little-endian — the host stores them as-is).
pub const EmbedRowsInfo = extern struct {
    count: u32,
    dims: u32,
};

// ─────────────────────────────────────────────────────────────────────────────

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
    slot: *handles.Slot,
    /// Owns the stream release + engine-side complete (`Running.done`).
    running: mlx_engine.Running,
    lm: *model_registry_mod.LoadedModel,
    gen_id: u64,
    cb: Callback,
    cb_ctx: ?*anyopaque,
    strip_leading_space: bool,
};

const handles = @import("abi/mlx_handles.zig");

var g_embed: ?*Embed = null;
var g_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_shutdown_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_generations_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_generations: std.AutoHashMapUnmanaged(u64, *handles.Slot) = .{};
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
    @memcpy(dst[0..n], msg_dst(dst));
    return dst[0..n];
}

fn msg_dst(dst: []u8) []const u8 {
    _ = dst;
    return g_mlx_fail_msg[0..g_mlx_fail_len];
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

/// server-side decodeTokens for one token: routes through the embedded
/// engines when the model runs on one (GGUF vocab), else the MLX tokenizer.
fn decodeToken(alloc: std.mem.Allocator, lm: *model_registry_mod.LoadedModel, id: u32) ![]u8 {
    const t = lm.tokenizer orelse return error.NoTokenizer;
    return tokenize_api.decodeTokens(alloc, lm, t, &[_]u32{id}, false);
}

/// host/api/submit.zig utf8TrailingIncomplete.
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
        f.running.done();
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

/// Boots one engine instance into `slot` under `mu`. Shared by the chat
/// engine (`mlxembed_init`) and the embeddings engine (`mlxembed_embed_init`).
fn bootEngine(cfg: *const InitConfig, slot: *?*Embed, mu: *std.c.pthread_mutex_t, tag: []const u8) i32 {
    _ = std.c.pthread_mutex_lock(mu);
    defer _ = std.c.pthread_mutex_unlock(mu);
    if (slot.* != null) {
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

    // Device selection (mirrors main.zig / ios_lib.zig).
    var metal_avail: bool = false;
    _ = mlx.mlx_metal_is_available(&metal_avail);
    const force_cpu = if (env.get(.MLXSERVE_DEVICE)) |d|
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

    // Media guard — the chat/embedder engines serve text, images-in-prompt
    // and embeddings only; media generation rides the media sidecar.
    if (gen_mod.detectModality(io, alloc, model_dir)) |modality| {
        setErr("media model ({s}) not supported by this engine — use mlxembed_media_load", .{@tagName(modality)});
        return -3;
    }

    // Config / tokenizer / chat config (mirrors main.zig's CPU-state block).
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
    tok.* = tokenizer_load.loadTokenizerAny(io, alloc, model_dir) catch |err| {
        setErr("loadTokenizer: {s}", .{@errorName(err)});
        return -5;
    };

    const chat_config = alloc.create(chat_mod.ChatConfig) catch {
        setErr("oom: ChatConfig", .{});
        return -6;
    };
    errdefer alloc.destroy(chat_config);
    chat_config.* = @import("host/chat/template_load.zig").loadChatConfig(io, alloc, model_dir) catch |err| {
        setErr("loadChatConfig: {s}", .{@errorName(err)});
        return -6;
    };

    {
        var settings = model_settings_mod.overrideFor(alloc, io, model_dir);
        defer settings.deinit(alloc);
        model_settings_apply.applyModelSettings(config_storage, chat_config, &settings, cfg.enable_mtp != 0);
    }
    config_storage.applyTokenizer(tok, chat_config.eos_token);
    if (config_storage.radio_vision) chat_config.video_placeholder = "<img>";

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

    // Engine — blocks until the model is loaded + warmed on its GPU thread.
    const max_concurrent = if (cfg.max_concurrent == 0) 1 else cfg.max_concurrent;
    const params = gpu_params.LoadParams{
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
    const scheduler = mlx_engine.start(alloc, io, params, max_concurrent) catch |err| {
        setErr("Scheduler.init: {s}", .{@errorName(err)});
        return -8;
    };
    errdefer scheduler.deinit();
    // Vision/image prompts encode through `launch().global_scheduler`
    // (`media/encode.zig`); the boot that owns the latest engine wins, and
    // each media-embedding call site swaps it under `g_mm_mutex`.
    http_server.launch().global_scheduler = scheduler;

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
    // the engine load); registry.deinit frees them. threaded moves into e.
    slot.* = e;

    log.info("mlx-serve embed {s} (MLX path) — {s} ABI v{d}, model {s}\n", .{
        build_options.version,
        tag,
        ABI_VERSION,
        model_id,
    });
    return 0;
}

export fn mlxembed_init(cfg: *const InitConfig) callconv(.c) i32 {
    return bootEngine(cfg, &g_embed, &g_mutex, "text/chat");
}

var g_embedder: ?*Embed = null;
var g_embedder_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var g_embedder_shutdown_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;

export fn mlxembed_embed_init(cfg: *const InitConfig) callconv(.c) i32 {
    return bootEngine(cfg, &g_embedder, &g_embedder_mutex, "embeddings");
}

export fn mlxembed_embed_shutdown() callconv(.c) void {
    _ = std.c.pthread_mutex_lock(&g_embedder_shutdown_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_embedder_shutdown_mutex);
    _ = std.c.pthread_mutex_lock(&g_embedder_mutex);
    const e = g_embedder orelse {
        _ = std.c.pthread_mutex_unlock(&g_embedder_mutex);
        return;
    };
    g_embedder = null;
    _ = std.c.pthread_mutex_unlock(&g_embedder_mutex);
    runShutdown(e, "embedder shutdown still draining embed feeders");
}

// ─────────────────────────────────────────────────────────────────────────────
// Embeddings (ABI v9, append-only): a dedicated EmbeddingGemma-2 engine
// instance (`mlxembed_embed_init`) serves batched text embeddings and
// one-shot text+image embeddings. Each call spawns a feeder thread that
// routes the work through the scheduler's `computeEmbeddings` path — the
// SAME pooling / dense-head / L2-normalize pipeline the HTTP server's
// /v1/embeddings runs — so the vectors are byte-identical with what the HTTP
// surface would return for the same inputs. Results arrive as one packed
// `embed_rows` event followed by DONE.
// ─────────────────────────────────────────────────────────────────────────────

const EmbedFeeder = struct {
    engine: *Embed,
    lm: ?*model_registry_mod.LoadedModel = null,
    gen_id: u64,
    cb: Callback,
    cb_ctx: ?*anyopaque,
    media: bool = false,
    /// Owned copies of the borrowed C inputs (the feeder runs async).
    payload: []u8 = &.{},
    /// text path: byte length of each text, in order.
    text_lens: []usize = &.{},
    /// media path: byte length of each image, in order (after the text).
    image_lens: []usize = &.{},
    image_count: usize = 0,
    media_text_len: usize = 0,
    rows_count: usize = 0,
    total_tokens: usize = 0,
    compute_ns: u64 = 0,
};

fn embedderOr(neg_code: i32) ?*Embed {
    const e = g_embedder orelse {
        setErr("embedder engine not initialized", .{});
        return null;
    };
    _ = neg_code;
    return e;
}

/// Packs `rows` into one `embed_rows` event: [count][dims][f32 rows, LE,
/// row-major]. False = pack failure (last_error carries the reason).
fn sendEmbedRows(f: *EmbedFeeder, rows: []const []f32) bool {
    const alloc = f.engine.allocator;
    const dims: usize = if (rows.len > 0) rows[0].len else 0;
    var bytes: usize = @sizeOf(EmbedRowsInfo);
    for (rows) |r| bytes += r.len * @sizeOf(f32);
    const buf = alloc.alloc(u8, bytes) catch {
        setErr("oom: embed rows pack", .{});
        return false;
    };
    defer alloc.free(buf);
    const hdr = EmbedRowsInfo{
        .count = @intCast(@min(rows.len, std.math.maxInt(u32))),
        .dims = @intCast(@min(dims, std.math.maxInt(u32))),
    };
    @memcpy(buf[0 .. @sizeOf(EmbedRowsInfo)], std.mem.asBytes(&hdr));
    var off: usize = @sizeOf(EmbedRowsInfo);
    for (rows) |r| {
        const n = r.len * @sizeOf(f32);
        @memcpy(buf[off .. off + n], std.mem.sliceAsBytes(r));
        off += n;
    }
    f.cb(f.cb_ctx, f.gen_id, @backingInt(Event.embed_rows), buf.ptr, @intCast(bytes));
    return true;
}

fn embedTextBody(f: *EmbedFeeder, result: *FeederResult) void {
    const e = f.engine;
    const alloc = e.allocator;

    log.info("embed feeder: {d} text input(s)\n", .{f.text_lens.len});
    const lm = e.scheduler.ensureLoaded("") catch |err| {
        setErr("ensureLoaded: {s}", .{@errorName(err)});
        result.* = .errored;
        return;
    };
    f.lm = lm;
    const tok = lm.tokenizer orelse {
        setErr("embedder model has no tokenizer", .{});
        result.* = .errored;
        return;
    };
    const config = lm.config orelse {
        setErr("embedder model has no config", .{});
        result.* = .errored;
        return;
    };

    var seqs = std.ArrayList([]const u32).empty;
    defer {
        for (seqs.items) |s| alloc.free(s);
        seqs.deinit(alloc);
    }
    var total_tokens: usize = 0;
    var at: usize = 0;
    for (f.text_lens, 0..) |len, idx| {
        const text = f.payload[at .. at + len];
        at += len;
        if (std.mem.trim(u8, text, " \t\r\n").len == 0) {
            setErr("embed input {d} is empty", .{idx});
            result.* = .errored;
            return;
        }
        const raw = tok.encode(alloc, text) catch |err| {
            setErr("tokenize: {s}", .{@errorName(err)});
            result.* = .errored;
            return;
        };
        // Mirrors the HTTP surface's handleEmbeddings: wrap per the
        // checkpoint's declared id protocol (bidirectional BOS/EOS, last-token
        // append), then pool per the checkpoint's mode and normalize.
        const ids = embeddings_api.wrapEmbeddingIds(alloc, config, tok.appended_special, raw) catch |err| {
            setErr("wrap ids: {s}", .{@errorName(err)});
            result.* = .errored;
            return;
        };
        if (config.max_position_embeddings > 0 and ids.len > config.max_position_embeddings) {
            setErr("embed input {d} over limit: {d} tokens, {d} allowed", .{ idx, ids.len, config.max_position_embeddings });
            result.* = .errored;
            return;
        }
        seqs.append(alloc, ids) catch {
            alloc.free(ids);
            setErr("oom: embed seqs", .{});
            result.* = .errored;
            return;
        };
        total_tokens += ids.len;
    }

    var req = gpu_requests.EmbedRequest{
        .model = lm,
        .token_seqs = seqs.items,
        .allocator = alloc,
    };
    var sw = io_util.Stopwatch.init(e.io);
    const rows = e.scheduler.computeEmbeddings(&req) catch {
        setErr("embed: {s}", .{req.error_name orelse "compute failed"});
        result.* = .errored;
        return;
    };
    f.compute_ns = sw.read();
    f.rows_count = rows.len;
    f.total_tokens = total_tokens;
    log.info("embed feeder: {d} rows x {d} dims in {d} ms\n", .{
        rows.len,
        if (rows.len > 0) rows[0].len else 0,
        f.compute_ns / 1_000_000,
    });
    defer {
        for (rows) |r| alloc.free(r);
        alloc.free(rows);
    }
    if (rows.len != seqs.items.len) {
        setErr("embed: short response ({d} rows for {d} inputs)", .{ rows.len, seqs.items.len });
        result.* = .errored;
        return;
    }
    if (!sendEmbedRows(f, rows)) {
        result.* = .errored;
        return;
    }
}

fn embedMediaBody(f: *EmbedFeeder, result: *FeederResult) void {
    const e = f.engine;
    const alloc = e.allocator;

    const lm = e.scheduler.ensureLoaded("") catch |err| {
        setErr("ensureLoaded: {s}", .{@errorName(err)});
        result.* = .errored;
        return;
    };
    f.lm = lm;
    const tok = lm.tokenizer orelse {
        setErr("embedder model has no tokenizer", .{});
        result.* = .errored;
        return;
    };
    const config = lm.config orelse {
        setErr("embedder model has no config", .{});
        result.* = .errored;
        return;
    };
    if (!config.gv_budgeted) {
        setErr("media embedding requires a multimodal embedding model (EmbeddingGemma 2); the text path takes 'texts'", .{});
        result.* = .errored;
        return;
    }
    if (lm.vision_encoder == null) {
        setErr("media embedding: model has no vision encoder loaded", .{});
        result.* = .errored;
        return;
    }
    if (config.image_token_id == 0) {
        setErr("media embedding: model has no image token id", .{});
        result.* = .errored;
        return;
    }

    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(alloc);
    if (f.media_text_len > 0) {
        const text = f.payload[0 .. f.media_text_len];
        const enc = tok.encode(alloc, text) catch |err| {
            setErr("tokenize: {s}", .{@errorName(err)});
            result.* = .errored;
            return;
        };
        defer alloc.free(enc);
        ids.appendSlice(alloc, enc) catch {
            setErr("oom: media embed ids", .{});
            result.* = .errored;
            return;
        };
    }
    for (0..f.image_count) |_| {
        ids.append(alloc, config.image_token_id) catch {
            setErr("oom: media embed ids", .{});
            result.* = .errored;
            return;
        };
    }
    if (ids.items.len == 0) {
        setErr("media embed: empty prompt", .{});
        result.* = .errored;
        return;
    }

    var ids_var: []u32 = embeddings_api.wrapEmbeddingIds(
        alloc,
        config,
        tok.appended_special,
        alloc.dupe(u32, ids.items) catch {
            setErr("oom: media embed ids", .{});
            result.* = .errored;
            return;
        },
    ) catch |err| {
        setErr("wrap ids: {s}", .{@errorName(err)});
        result.* = .errored;
        return;
    };
    defer alloc.free(ids_var);

    // Decoded pixels must outlive `prepareMultimodalItems` (its encode runs
    // through the scheduler); free them after compute returns.
    const vp = media_prepare.visionPreprocFromConfig(config);
    const decoded = alloc.alloc(chat_mod.ImageData, f.image_count) catch {
        setErr("oom: media embed decode", .{});
        result.* = .errored;
        return;
    };
    var decoded_count: usize = 0;
    defer {
        for (decoded[0..decoded_count]) |im| alloc.free(im.pixels);
        alloc.free(decoded);
    }
    var items = alloc.alloc(media_placeholders.MediaItem, f.image_count) catch {
        setErr("oom: media embed items", .{});
        result.* = .errored;
        return;
    };
    defer alloc.free(items);
    var at: usize = f.media_text_len;
    for (0..f.image_count) |i| {
        const bytes = f.payload[at .. at + f.image_lens[i]];
        at += f.image_lens[i];
        decoded[i] = media_decode.decodeImageToPixels(alloc, bytes, vp) orelse {
            setErr("media embed: could not decode image {d}", .{i});
            result.* = .errored;
            return;
        };
        decoded_count += 1;
        items[i] = .{ .image = decoded[i .. i + 1] };
    }

    // Mirrors the HTTP surface's handleMessagesEmbedding: refuse an
    // over-limit prompt BEFORE the vision tower runs (the expansion length is
    // known from the preprocessed sizes alone).
    const expected = ids_var.len - f.image_count + media_placeholders.mediaRunTokens(items, config);
    if (config.max_position_embeddings > 0 and expected > config.max_position_embeddings) {
        setErr("media embedding prompt over limit: {d} tokens, {d} allowed", .{ expected, config.max_position_embeddings });
        result.* = .errored;
        return;
    }

    // The vision encode inside rides `launch().global_scheduler`; make it
    // THIS engine's for the duration (see bootEngine).
    const prev_scheduler = http_server.launch().global_scheduler;
    http_server.launch().global_scheduler = e.scheduler;
    defer http_server.launch().global_scheduler = prev_scheduler;
    var mm = media_encode.prepareMultimodalItems(alloc, lm, config, items, &ids_var) catch |err| {
        setErr("media embed: {s}", .{@errorName(err)});
        result.* = .errored;
        return;
    };
    defer mm.deinit(alloc);

    // The request owns the soft tokens: the compute frees them on every path
    // (success and failure alike).
    const vision = mm.embeddings;
    mm.embeddings = null;
    var seqs = [_][]const u32{ids_var};
    var req = gpu_requests.EmbedRequest{
        .model = lm,
        .token_seqs = &seqs,
        .vision_embeddings = vision,
        .allocator = alloc,
    };
    var sw = io_util.Stopwatch.init(e.io);
    const rows = e.scheduler.computeEmbeddings(&req) catch {
        setErr("media embed: {s}", .{req.error_name orelse "compute failed"});
        result.* = .errored;
        return;
    };
    f.compute_ns = sw.read();
    f.rows_count = rows.len;
    f.total_tokens = ids_var.len;
    defer {
        for (rows) |r| alloc.free(r);
        alloc.free(rows);
    }
    if (rows.len != 1) {
        setErr("media embed: expected one embedding, got {d}", .{rows.len});
        result.* = .errored;
        return;
    }
    if (!sendEmbedRows(f, rows)) {
        result.* = .errored;
        return;
    }
}

fn feederEmbedMain(f: *EmbedFeeder) void {
    const e = f.engine;
    const alloc = e.allocator;
    var result: FeederResult = .ok;

    defer {
        if (f.payload.len > 0) alloc.free(f.payload);
        if (f.text_lens.len > 0) alloc.free(f.text_lens);
        if (f.image_lens.len > 0) alloc.free(f.image_lens);
        if (f.lm) |lm| e.scheduler.release(lm);
        if (result == .ok) {
            const info = DoneInfo{
                .finish_reason = @backingInt(Reason.stop),
                ._pad = .{ 0, 0, 0 },
                .prompt_tokens = @intCast(@min(f.total_tokens, std.math.maxInt(u32))),
                .cached_tokens = 0,
                .completion_tokens = @intCast(@min(f.rows_count, std.math.maxInt(u32))),
                .prefill_ns = f.compute_ns,
                .decode_ns = 0,
            };
            const ptr: [*]const u8 = @ptrCast(&info);
            f.cb(f.cb_ctx, f.gen_id, @backingInt(Event.done), ptr, @sizeOf(DoneInfo));
        } else {
            f.cb(f.cb_ctx, f.gen_id, @backingInt(Event.err), null, 0);
        }
        _ = e.feeders.fetchSub(1, .release);
        alloc.destroy(f);
    }

    if (f.media) {
        embedMediaBody(f, &result);
    } else {
        embedTextBody(f, &result);
    }
}

export fn mlxembed_embed_texts(cfg: *const EmbedTextParams, cb: Callback, cb_ctx: ?*anyopaque) callconv(.c) i64 {
    _ = std.c.pthread_mutex_lock(&g_embedder_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_embedder_mutex);
    const e = embedderOr(-1) orelse return -1;
    if (e.shutting_down.load(.acquire)) {
        setErr("embedder engine shutting down", .{});
        return -2;
    }
    if (cfg.count == 0 or cfg.texts == null or cfg.lens == null) {
        setErr("embed_texts: 'texts'/'lens' missing or count 0", .{});
        return -3;
    }
    if (cfg.count > 512) {
        setErr("embed_texts: at most 512 inputs per request", .{});
        return -3;
    }
    const alloc = e.allocator;
    const texts = cfg.texts.?[0..cfg.count];
    const lens = cfg.lens.?[0..cfg.count];

    // Copy the borrowed inputs before the feeder starts.
    var total: usize = 0;
    for (lens) |l| total += l;
    const payload = alloc.alloc(u8, total) catch {
        setErr("oom: embed_texts copy", .{});
        return -4;
    };
    var at: usize = 0;
    for (texts, lens) |tp, l| {
        const src: []const u8 = tp[0..l];
        @memcpy(payload[at .. at + l], src);
        at += l;
    }
    const lens_copy = alloc.alloc(usize, cfg.count) catch {
        alloc.free(payload);
        setErr("oom: embed_texts lens copy", .{});
        return -4;
    };
    @memcpy(lens_copy, lens);

    const gen_id = g_next_id.fetchAdd(1, .monotonic);
    const f = alloc.create(EmbedFeeder) catch {
        alloc.free(payload);
        alloc.free(lens_copy);
        setErr("oom: embed feeder", .{});
        return -5;
    };
    f.* = .{
        .engine = e,
        .gen_id = gen_id,
        .cb = cb,
        .cb_ctx = cb_ctx,
        .payload = payload,
        .text_lens = lens_copy,
        .media = false,
    };
    _ = e.feeders.fetchAdd(1, .release);
    const th = std.Thread.spawn(.{}, feederEmbedMain, .{f}) catch {
        _ = e.feeders.fetchSub(1, .release);
        alloc.free(payload);
        alloc.free(lens_copy);
        alloc.destroy(f);
        setErr("thread spawn failed", .{});
        return -6;
    };
    th.detach();
    return @intCast(gen_id);
}

export fn mlxembed_embed_media(cfg: *const EmbedMediaParams, cb: Callback, cb_ctx: ?*anyopaque) callconv(.c) i64 {
    _ = std.c.pthread_mutex_lock(&g_embedder_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_embedder_mutex);
    const e = embedderOr(-1) orelse return -1;
    if (e.shutting_down.load(.acquire)) {
        setErr("embedder engine shutting down", .{});
        return -2;
    }
    if (cfg.images == null or cfg.image_count == 0) {
        setErr("embed_media: no image given", .{});
        return -3;
    }
    if (cfg.image_count > 64) {
        setErr("embed_media: at most 64 images per prompt", .{});
        return -3;
    }
    const alloc = e.allocator;
    const imgs = cfg.images.?[0..cfg.image_count];
    const text: []const u8 = if (cfg.text) |p| p[0..cfg.text_len] else "";

    var total: usize = text.len;
    for (imgs) |im| {
        if (im.data == null or im.len == 0) {
            setErr("embed_media: invalid image entry", .{});
            return -3;
        }
        total += im.len;
    }
    const payload = alloc.alloc(u8, total) catch {
        setErr("oom: embed_media copy", .{});
        return -4;
    };
    @memcpy(payload[0 .. text.len], text);
    var at: usize = text.len;
    const lens_copy = alloc.alloc(usize, cfg.image_count) catch {
        alloc.free(payload);
        setErr("oom: embed_media lens copy", .{});
        return -4;
    };
    for (imgs, 0..) |im, i| {
        @memcpy(payload[at .. at + im.len], im.data.?[0..im.len]);
        lens_copy[i] = im.len;
        at += im.len;
    }

    const gen_id = g_next_id.fetchAdd(1, .monotonic);
    const f = alloc.create(EmbedFeeder) catch {
        alloc.free(payload);
        alloc.free(lens_copy);
        setErr("oom: embed feeder", .{});
        return -5;
    };
    f.* = .{
        .engine = e,
        .gen_id = gen_id,
        .cb = cb,
        .cb_ctx = cb_ctx,
        .payload = payload,
        .image_lens = lens_copy,
        .image_count = cfg.image_count,
        .media_text_len = text.len,
        .media = true,
    };
    _ = e.feeders.fetchAdd(1, .release);
    const th = std.Thread.spawn(.{}, feederEmbedMain, .{f}) catch {
        _ = e.feeders.fetchSub(1, .release);
        alloc.free(payload);
        alloc.free(lens_copy);
        alloc.destroy(f);
        setErr("thread spawn failed", .{});
        return -6;
    };
    th.detach();
    return @intCast(gen_id);
}

// ─────────────────────────────────────────────────────────────────────────────
// Media sidecar (ABI v2..v8, append-only): image / music / video / decision
// generation driven directly by the NIF host, plus Breeze TTS (speech, with
// PCM streaming) and realtime ASR. A single dedicated worker thread owns
// model load and generation — MLX default streams are thread-local, so the
// thread that loads an engine is the thread that must run it (mirroring the
// server's scheduler). Media models live beside the chat engine and share
// the process-global MLX device; the host budgets residency (Fleet resource
// claims). One generation in flight at a time; cancellation via a
// per-generation flag. Result bytes are borrowed for the duration of the
// callback — the host copies.
//
// Image/music/video/mesh/decision packs load as MEDIA PLUGINS (`MediaTable`)
// and run the canonical route handler over a socketpair Conn — the reply is
// the route's own HTTP response, refusal status codes included. Speech and
// ASR keep their direct engines (PCM streaming + the realtime decoder are
// not table-shaped).
// ─────────────────────────────────────────────────────────────────────────────

const MediaEngine = union(enum) {
    /// A media-plugin pack, served through its table over a socketpair.
    plugin: *plugin_engine.Engine,
    /// Breeze-TTS: direct engine (synthesize / synthesizeStream).
    tts: *breeze_mod.Engine,
    /// Voxtral realtime ASR: direct engine (src/embed/asr.zig).
    asr: *embed_asr.Engine,
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

/// One-time process-wide MLX cache limit for the media path. The text-engine
/// init sets its own via cfg.mlx_cache_mb; media-only nodes (fleet) never run
/// that path, so latch a sane bound here. Env: MLXSERVE_MEDIA_CACHE_MB (MB).
var g_media_cache_latched: bool = false;

fn ensureMediaCacheLimit() void {
    if (g_media_cache_latched) return;
    g_media_cache_latched = true;
    var mb: usize = 1024;
    if (env.get(.MLXSERVE_MEDIA_CACHE_MB)) |v| {
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

/// Detect the modality at `dir` and load the matching engine on the media
/// worker thread. Returns null (and sets the last error) on any failure.
/// Does NOT register the engine in g_media — the caller owns that.
fn mediaConstructEngine(io: std.Io, alloc: std.mem.Allocator, dir: []const u8) ?MediaEngine {
    ensureMediaCacheLimit();
    const modality = gen_mod.detectModality(io, alloc, dir) orelse {
        setErr("media: undetectable model at {s}", .{dir});
        return null;
    };
    switch (modality) {
        // A Voxtral realtime pack has no media plugin on this tree; the
        // fork's own realtime decoder (src/embed/asr.zig) serves it.
        .diarization => {
            const peeked = gen_mod.peekModelType(io, alloc, dir) orelse "";
            defer if (peeked.len > 0) alloc.free(peeked);
            if (!std.mem.eql(u8, peeked, "voxtral_realtime")) {
                setErr("media: modality {s} not supported by embed", .{@tagName(modality)});
                return null;
            }
            const e = embed_asr.Engine.load(io, alloc, dir) catch |err| {
                setErr("media asr load: {s}", .{@errorName(err)});
                return null;
            };
            return .{ .asr = e };
        },
        else => {},
    }
    // Every other modality is a media plugin: the served table wins the pack
    // (two plugins at one rank is refused by name), and the table's identity
    // decides direct-engine vs plugin arms.
    const table = plugin_served.claimPath(io, dir) orelse {
        setErr("media: no media plugin claims the {s} pack at {s}", .{ @tagName(modality), dir });
        return null;
    };
    if (modality == .audio and std.mem.eql(u8, table.name.slice(), "breeze_tts")) {
        const e = breeze_mod.Engine.load(io, alloc, mlx.mlx_default_gpu_stream_new(), dir) catch |err| {
            setErr("media speech load: {s}", .{@errorName(err)});
            return null;
        };
        return .{ .tts = e };
    }
    const e = plugin_engine.Engine.load(io, alloc, dir, table, mlx.mlx_default_gpu_stream_new()) catch |err| {
        setErr("media {s} load: {s}", .{ @tagName(modality), @errorName(err) });
        return null;
    };
    return .{ .plugin = e };
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
        .plugin => |e| e.deinit(),
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

/// Captures everything written to the far end of a socketpair while the
/// canonical route handler runs on THIS thread (the media worker): MLX
/// streams must match the thread that loaded the engine.
fn runPluginRoute(c: *MediaGenCtx, eng: *plugin_engine.Engine, route: gen_mod.GenRoute, body: []const u8) void {
    const alloc = c.alloc;
    const io = g_media_threaded.?.io();

    var sv: [2]std.c.fd_t = undefined;
    // AF_UNIX, SOCK_STREAM
    if (std.c.socketpair(1, 1, 0, &sv) != 0) {
        const msg = "media: socketpair failed";
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
        const msg = "media: spawn failed";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    };

    var conn: http_conn.Conn = undefined;
    http_conn.Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, io);
    const model_id = mediaDefaultBasename(c.dir);
    plugin_http.handle(alloc, &conn, body, eng, route, model_id) catch |err| {
        log.warn("media {s} job failed: {s}\n", .{ @tagName(route), @errorName(err) });
    };
    _ = std.c.close(sv[0]);
    th.join();

    const bytes = out.items;
    if (bytes.len == 0) {
        out.deinit(alloc);
        const msg = "media: empty response";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    }
    mediaEmitRouteEvent(c, route, bytes);
    out.deinit(alloc);
}

/// Splits the captured reply at the head/body boundary and emits the event
/// the route's contract carries: PNG bytes for images, WAV bytes for audio,
/// the FULL raw HTTP reply for video + decisions (the host splits the head).
fn mediaEmitRouteEvent(c: *MediaGenCtx, route: gen_mod.GenRoute, reply: []const u8) void {
    const alloc = c.alloc;
    const body = blk: {
        const marker = "\r\n\r\n";
        if (std.mem.indexOf(u8, reply, marker)) |i| break :blk reply[i + marker.len ..];
        break :blk reply;
    };
    switch (route) {
        .image => {
            const png = decodeB64Field(alloc, body, "b64_json") orelse {
                // Not the success shape: a refusal — emit the whole reply.
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), reply.ptr, @intCast(reply.len));
                return;
            };
            defer alloc.free(png);
            c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.png), png.ptr, @intCast(png.len));
        },
        .music, .sound, .speech => {
            if (body.len == 0) {
                const msg = "media: empty wav body";
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            }
            c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.wav), body.ptr, @intCast(body.len));
        },
        .video, .mesh, .decisions => {
            c.cb(c.cb_ctx, c.gen_id, if (route == .decisions) @backingInt(Event.decision) else @backingInt(Event.video), reply.ptr, @intCast(reply.len));
        },
    }
    mediaEmitDone(c, c.steps, 0);
}

/// First base64 string under `field` in a JSON body (`{"data":[{"b64_json":
/// "…"}]}`-shaped; a cheap scan, not a JSON parser — the shape is ours).
fn decodeB64Field(alloc: std.mem.Allocator, body: []const u8, comptime field: []const u8) ?[]u8 {
    const needle = "\"" ++ field ++ "\":\"";
    const at = std.mem.indexOf(u8, body, needle) orelse return null;
    const start = at + needle.len;
    var end = start;
    while (end < body.len and body[end] != '"') end += 1;
    if (end == start) return null;
    const dec = std.base64.standard.Decoder;
    const n = dec.calcSizeForSlice(body[start..end]) catch return null;
    const out = alloc.alloc(u8, n) catch return null;
    dec.decode(out, body[start..end]) catch {
        alloc.free(out);
        return null;
    };
    return out;
}

/// Builds the canonical request body for a media-plugin route from the C
/// config. `stream:false` everywhere: the reply is ONE body, not SSE.
fn buildPluginBody(c: *MediaGenCtx, route: gen_mod.GenRoute) ?[]u8 {
    const alloc = c.alloc;
    var out: std.ArrayList(u8) = .empty;
    const w = &out;
    const P = struct {
        fn print(list: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) bool {
            list.print(alloc_for_print, fmt, args) catch return false;
            return true;
        }
        var alloc_for_print: std.mem.Allocator = undefined;
    };
    P.alloc_for_print = alloc;
    var ok = true;
    const putU32 = struct {
        fn go(list: *std.ArrayList(u8), name: []const u8, v: u32) bool {
            if (v == 0) return true;
            return P.print(list, ",\"{s}\":{d}", .{ name, v });
        }
    }.go;
    const putF32 = struct {
        fn go(list: *std.ArrayList(u8), name: []const u8, v: f32) bool {
            if (v <= 0) return true;
            return P.print(list, ",\"{s}\":{d:.4}", .{ name, v });
        }
    }.go;

    switch (route) {
        .image => {
            ok = P.print(w, "{{\"prompt\":{f},\"stream\":false", .{std.json.fmt(c.prompt, .{})});
            ok = putU32(w, "width", c.width) and ok;
            ok = putU32(w, "height", c.height) and ok;
            ok = putU32(w, "steps", c.steps) and ok;
            ok = putF32(w, "guidance_scale", c.guidance) and ok;
            if (c.seed != 0) ok = P.print(w, ",\"seed\":{d}", .{c.seed}) and ok;
            // source image first (`image`), further references as `ref_images`.
            for (c.edit_images, 0..) |img, i| {
                const b64 = encodeB64(alloc, img) catch break;
                defer alloc.free(b64);
                if (i == 0) {
                    ok = P.print(w, ",\"image\":{f}", .{std.json.fmt(b64, .{})}) and ok;
                } else {
                    if (i == 1) w.appendSlice(alloc, ",\"ref_images\":[") catch {};
                    if (i > 1) w.appendSlice(alloc, ",") catch {};
                    ok = P.print(w, "{f}", .{std.json.fmt(b64, .{})}) and ok;
                    if (i == c.edit_images.len - 1) w.appendSlice(alloc, "]") catch {};
                }
            }
            if (ok) w.appendSlice(alloc, "}") catch {};
        },
        .music, .sound => {
            ok = P.print(w, "{{\"prompt\":{f},\"lyrics\":{f},\"stream\":false", .{ std.json.fmt(c.prompt, .{}), std.json.fmt(c.lyrics, .{}) });
            if (c.abc.len > 0) ok = P.print(w, ",\"abc\":{f}", .{std.json.fmt(c.abc, .{})}) and ok;
            ok = putU32(w, "duration_seconds", c.duration_seconds) and ok;
            ok = putU32(w, "steps", c.steps) and ok;
            if (c.seed != 0) ok = P.print(w, ",\"seed\":{d}", .{c.seed}) and ok;
            if (ok) w.appendSlice(alloc, "}") catch {};
        },
        .video => return alloc.dupe(u8, c.video_json) catch null,
        .decisions => return alloc.dupe(u8, c.decision_json) catch null,
        else => return null,
    }
    if (!ok) {
        out.deinit(alloc);
        return null;
    }
    return out.toOwnedSlice(alloc) catch blk: {
        out.deinit(alloc);
        break :blk null;
    };
}

fn encodeB64(alloc: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const out = try alloc.alloc(u8, std.base64.standard.Encoder.calcSize(bytes.len));
    _ = std.base64.standard.Encoder.encode(out, bytes);
    return out;
}

fn runMediaAsr(c: *MediaGenCtx, eng: *embed_asr.Engine) void {
    if (c.audio.len == 0) {
        const msg = "asr: no audio bytes";
        c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
        return;
    }
    const reply = embed_asr.transcribe(eng, c.alloc, c.audio, .{
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

fn runMediaGenerate(c: *MediaGenCtx) void {
    defer mediaFeederCleanup(c);
    const alloc = c.alloc;
    const s = mlx.mlx_default_gpu_stream_new();
    _ = mlx.mlx_set_default_stream(s);
    _ = mlx.mlx_stream_free(s);
    const progress = sseProgress(c);

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
        .tts => |be| {
            if (mlxFailPeek(c.failbuf[0..]) != null) {
                ttsMlxFailEvent(c);
                return;
            }
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
        .plugin => |eng| {
            if (mlxFailPeek(c.failbuf[0..]) != null) {
                const msg = "media: mlx failure latched";
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            }
            const route: gen_mod.GenRoute = switch (eng.table.modality) {
                .image => .image,
                .audio => if (c.lyrics.len > 0) .music else .sound,
                .video => .video,
                .mesh => .mesh,
                .decision => .decisions,
                .diarization => {
                    const msg = "media: diarization packs have no embed route";
                    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                    return;
                },
            };
            const body: []const u8 = blk: {
                if (route == .video and c.video_json.len > 0) break :blk c.video_json;
                if (route == .decisions and c.decision_json.len > 0) break :blk c.decision_json;
                const built = buildPluginBody(c, route) orelse {
                    const msg = "media: request build failed";
                    c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                    return;
                };
                break :blk built;
            };
            defer if (body.ptr != c.video_json.ptr and body.ptr != c.decision_json.ptr) alloc.free(@constCast(body.ptr[0..body.len]));
            if (body.len == 0) {
                const msg = "media: empty request body";
                c.cb(c.cb_ctx, c.gen_id, @backingInt(Event.err), msg.ptr, @intCast(msg.len));
                return;
            }
            _ = progress;
            runPluginRoute(c, eng, route, body);
            logMemStat("after-gen");
        },
    }
}

/// A progress sink shaped for the direct engines (TTS/ASR never report;
/// plugin routes report through the SSE stream on their socketpair).
fn sseProgress(c: *MediaGenCtx) sse_mod.Progress {
    return .{
        .ctx = @ptrCast(c),
        .cb = mediaProgress,
        .cancelled_cb = mediaCancelled,
    };
}

const sse_mod = @import("host/media_jobs/sse.zig");

fn mediaWorkerLoop() void {
    installErrorTrap();
    const alloc = std.heap.c_allocator;
    g_media_threaded = std.Io.Threaded.init(alloc, .{});
    // Same device selection as mlxembed_init: Metal when available, else CPU.
    var metal_avail: bool = false;
    _ = mlx.mlx_metal_is_available(&metal_avail);
    const force_cpu = if (env.get(.MLXSERVE_DEVICE)) |d|
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

fn mediaTeardown() void {
    _ = std.c.pthread_mutex_lock(&g_media_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_media_mutex);
    const alloc = std.heap.c_allocator;
    var it = g_media.iterator();
    while (it.next()) |entry| {
        mediaDeinitEngine(entry.value_ptr.*);
        alloc.free(entry.key_ptr.*);
    }
    g_media.clearRetainingCapacity();
    if (g_media_default.len > 0) {
        alloc.free(@constCast(g_media_default));
        g_media_default = "";
    }
    g_media_threaded.?.deinit();
    g_media_threaded = null;
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

export fn mlxembed_media_cancel(gen_id: u64) callconv(.c) i32 {
    _ = std.c.pthread_mutex_lock(&g_media_gen_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_media_gen_mutex);
    const cancelled = g_media_gen.get(gen_id) orelse {
        setErr("media: generation not found", .{});
        return -2;
    };
    cancelled.store(true, .release);
    return 0;
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
    if (prompt.len == 0 and cfg.audio_len == 0 and (cfg.video_json_len == 0 or cfg.video_json == null) and (cfg.decision_json_len == 0 or cfg.decision_json == null)) {
        setErr("media: empty prompt", .{});
        return -3;
    }

    const c = alloc.create(MediaGenCtx) catch return -3;
    const cancelled = alloc.create(std.atomic.Value(bool)) catch {
        alloc.destroy(c);
        return -3;
    };
    cancelled.* = .init(false);
    var edit_images: []const []const u8 = &.{};
    var edit_ok = true;
    if (cfg.edit_image_count > 0 and cfg.edit_image_ptrs != null and cfg.edit_image_lens != null) {
        const bufs = alloc.alloc([]u8, cfg.edit_image_count) catch {
            alloc.destroy(cancelled);
            alloc.destroy(c);
            return -3;
        };
        var n: usize = 0;
        for (cfg.edit_image_ptrs.?[0..cfg.edit_image_count], cfg.edit_image_lens.?[0..cfg.edit_image_count]) |ptr, len| {
            const copy = alloc.alloc(u8, len) catch break;
            @memcpy(copy, ptr[0..len]);
            bufs[n] = copy;
            n += 1;
        }
        edit_ok = n == cfg.edit_image_count;
        edit_images = bufs[0..n];
    }
    if (!edit_ok) {
        alloc.destroy(cancelled);
        alloc.destroy(c);
        setErr("media: edit image copy failed", .{});
        return -3;
    }
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
        .edit_images = edit_images,
        .video_json = if (cfg.video_json) |p| alloc.dupe(u8, p[0..cfg.video_json_len]) catch &.{} else &.{},
        .decision_json = if (cfg.decision_json) |p| alloc.dupe(u8, p[0..cfg.decision_json_len]) catch &.{} else &.{},
    };

    const gen_id = g_next_id.fetchAdd(1, .monotonic);
    c.gen_id = gen_id;
    _ = std.c.pthread_mutex_lock(&g_media_gen_mutex);
    g_media_gen.put(alloc, gen_id, cancelled) catch {
        _ = std.c.pthread_mutex_unlock(&g_media_gen_mutex);
        mediaFeederCleanup(c);
        setErr("oom: media generation registry", .{});
        return -4;
    };
    _ = std.c.pthread_mutex_unlock(&g_media_gen_mutex);
    _ = g_media_inflight.fetchAdd(1, .release);
    mediaEnqueue(.{ .generate = c });
    return @intCast(gen_id);
}

export fn mlxembed_media_shutdown() callconv(.c) void {
    g_shutdown.store(true, .release);
    _ = std.c.pthread_mutex_lock(&g_jobs_mutex);
    _ = std.c.pthread_cond_signal(&g_jobs_cond);
    _ = std.c.pthread_mutex_unlock(&g_jobs_mutex);
    if (g_worker) |w| {
        w.join();
        g_worker = null;
    }
    g_shutdown.store(false, .release);
}

fn runShutdown(e: *Embed, drain_warn: []const u8) void {
    e.shutting_down.store(true, .release);
    e.scheduler.cancelAllInFlight();

    // Wait for feeder threads to drain (cancel triggers their terminal event).
    var spins: usize = 0;
    while (e.feeders.load(.acquire) > 0) : (spins += 1) {
        if (spins == 2000) log.warn("{s}\n", .{drain_warn});
        const ts = std.c.timespec{ .sec = 0, .nsec = 5_000_000 };
        _ = std.c.nanosleep(&ts, null);
    }

    e.scheduler.deinit();
    e.registry.deinit();
    e.threaded.deinit();
    const alloc = e.allocator;
    alloc.destroy(e);
}

export fn mlxembed_shutdown() callconv(.c) void {
    _ = std.c.pthread_mutex_lock(&g_shutdown_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_shutdown_mutex);
    _ = std.c.pthread_mutex_lock(&g_mutex);
    const e = g_embed orelse {
        _ = std.c.pthread_mutex_unlock(&g_mutex);
        return;
    };
    g_embed = null;
    _ = std.c.pthread_mutex_unlock(&g_mutex);
    runShutdown(e, "embed shutdown still draining cancelled generations");

    _ = std.c.pthread_mutex_lock(&g_generations_mutex);
    g_generations.deinit(std.heap.c_allocator);
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

    var mm: media_encode.MultimodalPrompt = .{};
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
        const vp = media_prepare.visionPreprocFromConfig(config);
        for (images, 0..) |encoded, i| {
            if (encoded.data == null or encoded.len == 0 or encoded.len > std.math.maxInt(c_int)) {
                e.scheduler.release(lm);
                setErr("images: invalid encoded image {d}", .{i});
                return -10;
            }
            decoded[i] = media_decode.decodeImageToPixels(alloc, encoded.data.?[0..encoded.len], vp) orelse {
                e.scheduler.release(lm);
                setErr("images: could not decode image {d}", .{i});
                return -10;
            };
            decoded_count += 1;
        }
        const messages = [_]chat_mod.Message{.{ .role = "user", .content = "", .images = decoded }};
        // The vision encode inside rides `launch().global_scheduler`; make
        // it THIS engine's for the duration (see bootEngine).
        const prev_scheduler = http_server.launch().global_scheduler;
        http_server.launch().global_scheduler = e.scheduler;
        defer http_server.launch().global_scheduler = prev_scheduler;
        mm = media_encode.prepareMultimodalPrompt(alloc, lm, config, &messages, &prompt_ids) catch |err| {
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

    const params = gpu_params.SubmitParams{
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
    const running = mlx_engine.Running.submit(e.scheduler, params) catch |err| {
        e.scheduler.release(lm);
        setErr("submit: {s}", .{@errorName(err)});
        return -7;
    };
    const slot = running.slot;

    const gen_id = g_next_id.fetchAdd(1, .monotonic);
    const f = alloc.create(Feeder) catch {
        running.done();
        e.scheduler.release(lm);
        setErr("oom: Feeder", .{});
        return -8;
    };
    f.* = .{
        .embed = e,
        .slot = slot,
        .running = running,
        .lm = lm,
        .gen_id = gen_id,
        .cb = cb,
        .cb_ctx = cb_ctx,
        .strip_leading_space = false,
    };
    _ = std.c.pthread_mutex_lock(&g_generations_mutex);
    g_generations.put(alloc, gen_id, slot) catch {
        _ = std.c.pthread_mutex_unlock(&g_generations_mutex);
        running.done();
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
        running.done();
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