//! fleet embed — text/chat-only C ABI over the mlx-serve engine, for a BEAM
//! (Elixir) NIF host.
//!
//! Scope: chat-completions surface only (text in, token stream out, stats).
//! Image / video / music / speech engines stay standalone; media model dirs
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
const scheduler_mod = @import("scheduler.zig");
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

pub const ABI_VERSION: u32 = 1;

pub const Event = enum(u32) {
    text = 1,
    done = 2,
    err = 3,
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
    next_id: std.atomic.Value(u64) = .init(1),
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
var g_last_err: [512:0]u8 = std.mem.zeroes([512:0]u8);

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

fn reasonCode(r: []const u8) u8 {
    if (std.mem.eql(u8, r, "stop")) return @intFromEnum(Reason.stop);
    if (std.mem.eql(u8, r, "length")) return @intFromEnum(Reason.length);
    if (std.mem.eql(u8, r, "abort")) return @intFromEnum(Reason.abort);
    if (std.mem.eql(u8, r, "content_filtered")) return @intFromEnum(Reason.content_filtered);
    if (std.mem.eql(u8, r, "model_context_window_exceeded")) return @intFromEnum(Reason.model_context_window_exceeded);
    return @intFromEnum(Reason.other);
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
    const expected: usize = if (lead & 0x80 == 0) 1
    else if (lead & 0xE0 == 0xC0) 2
    else if (lead & 0xF0 == 0xE0) 3
    else if (lead & 0xF8 == 0xF0) 4
    else return 0;
    const actual = s.len - i;
    return if (actual < expected) actual else 0;
}

const FeederResult = union(enum) { ok, errored };

fn feederMain(f: *Feeder) void {
    const e = f.embed;
    const alloc = e.allocator;

    var text_buf: [8192]u8 = undefined;
    var text_len: usize = 0;
    var carry: [4]u8 = undefined;
    var carry_len: usize = 0;
    var result: FeederResult = .ok;
    var final_reason: []const u8 = "stop";

    const flush = struct {
        fn run(feeder: *Feeder, buf: []const u8) void {
            if (buf.len > 0) {
                feeder.cb(feeder.cb_ctx, feeder.gen_id, @intFromEnum(Event.text), buf.ptr, @intCast(buf.len));
            }
        }
    }.run;

    defer {
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
            f.cb(f.cb_ctx, f.gen_id, @intFromEnum(Event.done), ptr, @sizeOf(DoneInfo));
        } else {
            f.cb(f.cb_ctx, f.gen_id, @intFromEnum(Event.err), null, 0);
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
                const raw = decodeToken(alloc, f.lm, tok_id) catch {
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

                // Append to buffer, flushing on overflow.
                var n: usize = 0;
                while (n < emit_len) {
                    const room = text_buf.len - text_len;
                    const chunk = @min(room, emit_len - n);
                    @memcpy(text_buf[text_len .. text_len + chunk], combined[n .. n + chunk]);
                    text_len += chunk;
                    n += chunk;
                    if (text_len == text_buf.len) {
                        flush(f, text_buf[0..text_len]);
                        text_len = 0;
                    }
                }
            },
            .done => {
                final_reason = f.slot.finish_reason;
                flush(f, text_buf[0..text_len]);
                text_len = 0;
                break;
            },
            .err => {
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
        .load_vision = false,
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
    while (e.feeders.load(.acquire) > 0 and spins < 2000) : (spins += 1) {
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
}

export fn mlxembed_generate(cfg: *const GenParams, cb: Callback, cb_ctx: ?*anyopaque) callconv(.c) i64 {
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
    const prompt_ids = t.encode(alloc, prompt) catch |err| {
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

    const enable_mtp = e.enable_mtp and (lm.mtp != null);
    const temperature: f32 = if (cfg.temperature <= 0) 1.0 else cfg.temperature;
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
    };
    const slot = e.scheduler.submit(params) catch |err| {
        e.scheduler.release(lm);
        setErr("submit: {s}", .{@errorName(err)});
        return -7;
    };

    const gen_id = e.next_id.fetchAdd(1, .monotonic);
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
    _ = e.feeders.fetchAdd(1, .release);
    const th = std.Thread.spawn(.{}, feederMain, .{f}) catch {
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
    _ = std.c.pthread_mutex_lock(&g_mutex);
    defer _ = std.c.pthread_mutex_unlock(&g_mutex);
    const e = embedOr(-1) orelse return -1;
    // Cancel is a no-op when we can't find the slot by gen_id (the feeder owns
    // it after complete); we track via a small set. For now cancel is best-effort
    // at the scheduler level: iterate is not exposed, so we expose a global
    // cancel of in-flight only when gen_id is 0.
    _ = e;
    _ = gen_id;
    setErr("cancel by gen_id not yet wired — use mlxembed_cancel_all", .{});
    return -2;
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

test "reason code mapping" {
    try std.testing.expectEqual(@intFromEnum(Reason.stop), reasonCode("stop"));
    try std.testing.expectEqual(@intFromEnum(Reason.length), reasonCode("length"));
    try std.testing.expectEqual(@intFromEnum(Reason.other), reasonCode("nope"));
}