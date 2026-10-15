//! Breeze-TTS-2 backend (`model_type: breeze_tts`).
//!
//! A multi-codebook TTS stack: a Qwen3-1B backbone emits codebook 0 at
//! 12.5 Hz, a small `breeze_depth_decoder_model` fills codebooks 1..15 per
//! step, a T5Gemma2 text encoder conditions the stream (speaker tokens
//! `[S0]..[S9]` + `<ins_bos>/<ins_eos>` instruction wrapping), and a Mimi
//! codec (`audio_tokenizer/`) turns 16 codebooks into 24 kHz PCM.
//!
//! Increment 1: config parse + pack validation + engine registration.
//! The forward (backbone loop, depth decoder, T5Gemma2 encoder, Mimi
//! decode) lands on top of `Config`; until then synthesis is a named 501.

const std = @import("std");
const log = @import("log.zig");

pub const model_type = "breeze_tts";

pub const Config = struct {
    // Audio-stream special ids (root config).
    audio_token_id: u32 = 262144,
    audio_eos_token_id: u32 = 262145,
    num_codebooks: u32 = 16,
    codebook_vocab: u32 = 2051,
    codebook_pad: u32 = 2050,
    codebook_eos: u32 = 0,
    audio_embed_size: u32 = 2048,

    // Backbone (Qwen3, `backbone_config`).
    b_hidden: u32 = 2048,
    b_inter: u32 = 6144,
    b_layers: u32 = 28,
    b_heads: u32 = 16,
    b_kv: u32 = 8,
    b_head_dim: u32 = 128,
    b_rms_eps: f32 = 1e-6,
    b_rope_theta: f32 = 1_000_000,
    b_text_vocab: u32 = 151936,
    b_tie_word_embeddings: bool = true,

    // Depth decoder (`depth_decoder_config`).
    d_hidden: u32 = 1024,
    d_inter: u32 = 8192,
    d_layers: u32 = 12,
    d_heads: u32 = 8,
    d_kv: u32 = 2,
    d_head_dim: u32 = 128,
    d_rms_eps: f32 = 1e-5,
    d_rope_theta: f32 = 500_000,
    d_rope_factor: f64 = 32.0,
    d_rope_low: f64 = 0.001953125,
    d_rope_high: f64 = 0.0078125,
    d_rope_orig_max: u32 = 16,
    d_backbone_hidden: u32 = 2048,
    d_max_position: u32 = 33,
    d_num_codebooks: u32 = 16,

    // Text encoder (T5Gemma2, `text_encoder_config`).
    te_hidden: u32 = 1152,
    te_inter: u32 = 6912,
    te_layers: u32 = 26,
    te_heads: u32 = 4,
    te_kv: u32 = 1,
    te_head_dim: u32 = 256,
    te_vocab: u32 = 262158,
    te_sliding_window: u32 = 512,
    te_rms_eps: f32 = 1e-6,
    te_full_rope_theta: f32 = 1_000_000,
    te_full_rope_factor: f64 = 8.0,
    te_sliding_rope_theta: f32 = 10_000,
    te_eoi_token: u32 = 256000,
    /// `full_attention` layer indices from `layer_types` (full = 1/te_full_every cadence).
    te_full_indices: [64]u32 = undefined,
    te_full_count: usize = 0,

    // Voice-design tokens (`text_encoder_special_tokens_config.token_ids`).
    speaker_base_token: u32 = 262146, // [S0]; [S1]..[S9] follow in order
    ins_bos_token: u32 = 262156,
    ins_eos_token: u32 = 262157,

    // Codec (Mimi, `codec_config`).
    sample_rate: u32 = 24000,
    codec_hidden: u32 = 512,
    codec_codebook_size: u32 = 2048,
    codec_codebook_dim: u32 = 256,
    codec_quantizers: u32 = 32,
    codec_compress: u32 = 2,
    codec_upsample: [4]u32 = .{ 8, 6, 5, 4 },

    /// Codec frames per second: sampling_rate / (prod(upsample) * compress).
    pub fn frameRate(self: Config) f64 {
        var prod: f64 = @floatFromInt(self.codec_compress);
        for (self.codec_upsample) |r| prod *= @floatFromInt(r);
        return @as(f64, @floatFromInt(self.sample_rate)) / prod;
    }
};

/// Parse the geometry + ids from a `breeze_tts` `config.json`. Omitted or
/// unparseable fields keep the struct defaults (the published pack's own
/// values), mirroring `tts.parseConfig`'s tolerance.
pub fn parseConfig(allocator: std.mem.Allocator, json_text: []const u8) !Config {
    var cfg = Config{};
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_text, .{}) catch return cfg;
    defer parsed.deinit();
    if (parsed.value != .object) return cfg;
    const obj = parsed.value.object;

    setU32(&cfg.audio_token_id, obj, "audio_token_id");
    setU32(&cfg.audio_eos_token_id, obj, "audio_eos_token_id");
    setU32(&cfg.num_codebooks, obj, "audio_num_codebooks");
    setU32(&cfg.codebook_vocab, obj, "audio_vocab_size");
    setU32(&cfg.codebook_pad, obj, "codebook_pad_token_id");
    setU32(&cfg.codebook_eos, obj, "codebook_eos_token_id");
    setU32(&cfg.audio_embed_size, obj, "audio_embed_size");
    setU32(&cfg.te_vocab, obj, "text_vocab_size");

    if (obj.get("backbone_config")) |v| {
        if (v == .object) {
            const b = v.object;
            setU32(&cfg.b_hidden, b, "hidden_size");
            setU32(&cfg.b_inter, b, "intermediate_size");
            setU32(&cfg.b_layers, b, "num_hidden_layers");
            setU32(&cfg.b_heads, b, "num_attention_heads");
            setU32(&cfg.b_kv, b, "num_key_value_heads");
            setU32(&cfg.b_head_dim, b, "head_dim");
            setF32(&cfg.b_rms_eps, b, "rms_norm_eps");
            setF32(&cfg.b_rope_theta, b, "rope_theta");
            setU32(&cfg.b_text_vocab, b, "vocab_size");
            setBool(&cfg.b_tie_word_embeddings, b, "tie_word_embeddings");
        }
    }

    if (obj.get("depth_decoder_config")) |v| {
        if (v == .object) {
            const d = v.object;
            setU32(&cfg.d_hidden, d, "hidden_size");
            setU32(&cfg.d_inter, d, "intermediate_size");
            setU32(&cfg.d_layers, d, "num_hidden_layers");
            setU32(&cfg.d_heads, d, "num_attention_heads");
            setU32(&cfg.d_kv, d, "num_key_value_heads");
            setU32(&cfg.d_head_dim, d, "head_dim");
            setF32(&cfg.d_rms_eps, d, "rms_norm_eps");
            setF32(&cfg.d_rope_theta, d, "rope_theta");
            setU32(&cfg.d_backbone_hidden, d, "backbone_hidden_size");
            setU32(&cfg.d_max_position, d, "max_position_embeddings");
            setU32(&cfg.d_num_codebooks, d, "num_codebooks");
            parseLlama3Scaling(&cfg.d_rope_factor, &cfg.d_rope_low, &cfg.d_rope_high, &cfg.d_rope_orig_max, d);
        }
    }

    if (obj.get("text_encoder_config")) |v| {
        if (v == .object) {
            const t = v.object;
            setU32(&cfg.te_hidden, t, "hidden_size");
            setU32(&cfg.te_inter, t, "intermediate_size");
            setU32(&cfg.te_layers, t, "num_hidden_layers");
            setU32(&cfg.te_heads, t, "num_attention_heads");
            setU32(&cfg.te_kv, t, "num_key_value_heads");
            setU32(&cfg.te_head_dim, t, "head_dim");
            setU32(&cfg.te_vocab, t, "vocab_size");
            setU32(&cfg.te_sliding_window, t, "sliding_window");
            setF32(&cfg.te_rms_eps, t, "rms_norm_eps");
            setU32(&cfg.te_eoi_token, t, "eoi_token_index");
            parseTextEncoderLayerTypes(&cfg, t);
            if (t.get("rope_parameters")) |rp| {
                if (rp == .object) {
                    if (rp.object.get("full_attention")) |fa| {
                        if (fa == .object) {
                            setF32(&cfg.te_full_rope_theta, fa.object, "rope_theta");
                            setF64(&cfg.te_full_rope_factor, fa.object, "factor");
                        }
                    }
                    if (rp.object.get("sliding_attention")) |sa| {
                        if (sa == .object) setF32(&cfg.te_sliding_rope_theta, sa.object, "rope_theta");
                    }
                }
            }
        }
    }

    if (obj.get("text_encoder_special_tokens_config")) |v| {
        if (v == .object and v.object.get("token_ids") != null and v.object.get("token_ids").? == .object) {
            const ids = v.object.get("token_ids").?.object;
            setU32(&cfg.speaker_base_token, ids, "[S0]");
            setU32(&cfg.ins_bos_token, ids, "<ins_bos>");
            setU32(&cfg.ins_eos_token, ids, "<ins_eos>");
        }
    }

    if (obj.get("codec_config")) |v| {
        if (v == .object) {
            const c = v.object;
            setU32(&cfg.sample_rate, c, "sampling_rate");
            setU32(&cfg.codec_hidden, c, "hidden_size");
            setU32(&cfg.codec_codebook_size, c, "codebook_size");
            setU32(&cfg.codec_codebook_dim, c, "codebook_dim");
            setU32(&cfg.codec_quantizers, c, "num_quantizers");
            setU32(&cfg.codec_compress, c, "compress");
            if (c.get("upsample_ratios")) |u| {
                if (u == .array and u.array.items.len == cfg.codec_upsample.len) {
                    for (u.array.items, 0..) |item, i| {
                        if (item == .integer and item.integer > 0) cfg.codec_upsample[i] = @intCast(item.integer);
                    }
                }
            }
        }
    }

    validate(&cfg) catch |err| {
        logBadConfig(err);
        return err;
    };
    return cfg;
}
/// Geometry the loader would otherwise misread as valid: a per-checkpoint
/// field that narrows unsafely or contradicts another one fails by NAME
/// (config-is-untrusted rule), never silently.
fn validate(cfg: *const Config) !void {
    if (cfg.num_codebooks == 0 or cfg.num_codebooks > 64) return error.BadCodebookCount;
    if (cfg.codebook_vocab <= cfg.codebook_pad) return error.BadCodebookVocab;
    if (cfg.b_heads == 0 or cfg.b_kv == 0 or cfg.b_heads % cfg.b_kv != 0) return error.BadBackboneGqa;
    if (cfg.b_head_dim == 0 or cfg.b_hidden % cfg.b_heads != 0) return error.BadBackboneHead;
    if (cfg.d_heads == 0 or cfg.d_kv == 0 or cfg.d_heads % cfg.d_kv != 0) return error.BadDepthGqa;
    if (cfg.d_num_codebooks != cfg.num_codebooks) return error.DepthCodebookMismatch;
    if (cfg.te_heads == 0 or cfg.te_kv == 0 or cfg.te_heads % cfg.te_kv != 0) return error.BadTextEncoderGqa;
    if (cfg.te_sliding_window == 0) return error.BadTextEncoderWindow;
    if (cfg.sample_rate == 0) return error.BadSampleRate;
    if (cfg.codec_compress == 0) return error.BadCodecCompress;
    for (cfg.codec_upsample) |r| {
        if (r == 0) return error.BadCodecUpsample;
    }
    if (cfg.audio_embed_size != cfg.b_hidden) return error.AudioEmbedMismatch;
}

fn logBadConfig(err: anyerror) void {
    log.err("[breeze] config rejected: {s}\n", .{@errorName(err)});
}

fn parseLlama3Scaling(factor: *f64, low: *f64, high: *f64, orig: *u32, obj: std.json.ObjectMap) void {
    const v = obj.get("rope_scaling") orelse return;
    if (v != .object) return;
    const s = v.object;
    const rt = s.get("rope_type") orelse return;
    if (rt != .string or !std.mem.eql(u8, rt.string, "llama3")) return;
    setF64(factor, s, "factor");
    setF64(low, s, "low_freq_factor");
    setF64(high, s, "high_freq_factor");
    setU32(orig, s, "original_max_position_embeddings");
}

fn parseTextEncoderLayerTypes(cfg: *Config, obj: std.json.ObjectMap) void {
    const v = obj.get("layer_types") orelse return;
    if (v != .array) return;
    cfg.te_full_count = 0;
    for (v.array.items, 0..) |item, i| {
        if (item != .string) continue;
        if (std.mem.eql(u8, item.string, "full_attention") and cfg.te_full_count < cfg.te_full_indices.len) {
            cfg.te_full_indices[cfg.te_full_count] = @intCast(i);
            cfg.te_full_count += 1;
        }
    }
}

pub fn isFullAttentionLayer(cfg: *const Config, layer: u32) bool {
    for (cfg.te_full_indices[0..cfg.te_full_count]) |idx| {
        if (idx == layer) return true;
    }
    return false;
}

fn setU32(dst: *u32, obj: std.json.ObjectMap, key: []const u8) void {
    const v = obj.get(key) orelse return;
    if (v == .integer and v.integer >= 0) dst.* = @intCast(v.integer);
}

fn setF32(dst: *f32, obj: std.json.ObjectMap, key: []const u8) void {
    const v = obj.get(key) orelse return;
    switch (v) {
        .integer => dst.* = @floatFromInt(v.integer),
        .float => dst.* = @as(f32, @floatCast(v.float)),
        else => {},
    }
}

fn setF64(dst: *f64, obj: std.json.ObjectMap, key: []const u8) void {
    const v = obj.get(key) orelse return;
    switch (v) {
        .integer => dst.* = @floatFromInt(v.integer),
        .float => dst.* = v.float,
        else => {},
    }
}

fn setBool(dst: *bool, obj: std.json.ObjectMap, key: []const u8) void {
    const v = obj.get(key) orelse return;
    if (v == .bool) dst.* = v.bool;
}

/// Voice-design speaker slot: `index` 0..9 maps to the pack's `[S0]..[S9]` ids.
pub fn speakerToken(cfg: *const Config, index: u32) ?u32 {
    if (index >= 10) return null;
    return cfg.speaker_base_token + index;
}

/// A request `voice` names a slot S0..S9 (case-insensitive), nothing else.
pub fn voiceSlotValid(name: []const u8) bool {
    if (name.len != 2) return false;
    if (name[0] != 'S' and name[0] != 's') return false;
    return name[1] >= '0' and name[1] <= '9';
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    cfg: Config,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        // openFileAbsolute asserts on relative/empty paths (ReleaseFast UB).
        if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.InvalidModelDir;
        const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
        defer allocator.free(path);
        const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.MissingConfig;
        defer file.close(io);
        var rb: [4096]u8 = undefined;
        var rs = file.reader(io, &rb);
        const text = try rs.interface.allocRemaining(allocator, .limited(8 * 1024 * 1024));
        defer allocator.free(text);
        const cfg = try parseConfig(allocator, text);
        // Mimi lives in `audio_tokenizer/` beside the LM shards — no codec, no waveform.
        const codec_path = try std.fmt.allocPrint(allocator, "{s}/audio_tokenizer/model.safetensors", .{model_dir});
        defer allocator.free(codec_path);
        if (!fileExists(io, codec_path)) return error.MissingCodec;

        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.* = .{ .allocator = allocator, .cfg = cfg };
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.allocator.destroy(self);
    }

    pub fn sampleRate(self: *const Engine) u32 {
        return self.cfg.sample_rate;
    }

    /// The synthesis forward (backbone loop + depth decoder + T5Gemma2 text
    /// conditioning + Mimi decode) is the next increment; refusal is named.
    pub fn synthesizeWav(self: *Engine, allocator: std.mem.Allocator, text: []const u8, ref_audio: ?[]const f32, instruction: ?[]const u8, seed: u64) ![]u8 {
        _ = text;
        _ = ref_audio;
        _ = instruction;
        _ = seed;
        _ = allocator;
        _ = self;
        return error.NotImplemented;
    }
};

fn fileExists(io: std.Io, path: []const u8) bool {
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

const pack_config =
    \\{
    \\"model_type": "breeze_tts",
    \\"audio_token_id": 262144,
    \\"audio_eos_token_id": 262145,
    \\"audio_num_codebooks": 16,
    \\"audio_vocab_size": 2051,
    \\"codebook_pad_token_id": 2050,
    \\"codebook_eos_token_id": 0,
    \\"audio_embed_size": 2048,
    \\"text_vocab_size": 262158,
    \\"backbone_config": {
    \\"model_type": "qwen3",
    \\"hidden_size": 2048, "intermediate_size": 6144,
    \\"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
    \\"head_dim": 128, "rms_norm_eps": 1e-06, "rope_theta": 1000000,
    \\"vocab_size": 151936, "tie_word_embeddings": true
    \\},
    \\"depth_decoder_config": {
    \\"model_type": "breeze_depth_decoder_model",
    \\"hidden_size": 1024, "intermediate_size": 8192,
    \\"num_hidden_layers": 12, "num_attention_heads": 8, "num_key_value_heads": 2,
    \\"head_dim": 128, "rms_norm_eps": 1e-05, "rope_theta": 500000,
    \\"vocab_size": 2051, "num_codebooks": 16,
    \\"backbone_hidden_size": 2048, "max_position_embeddings": 33,
    \\"rope_scaling": {"factor": 32.0, "low_freq_factor": 0.001953125, "high_freq_factor": 0.0078125, "original_max_position_embeddings": 16, "rope_type": "llama3"}
    \\},
    \\"text_encoder_config": {
    \\"model_type": "t5gemma2_text",
    \\"hidden_size": 1152, "intermediate_size": 6912,
    \\"num_hidden_layers": 26, "num_attention_heads": 4, "num_key_value_heads": 1,
    \\"head_dim": 256, "vocab_size": 262158, "sliding_window": 512,
    \\"rms_norm_eps": 1e-06, "eoi_token_index": 256000,
    \\"layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention"],
    \\"rope_parameters": {
    \\"full_attention": {"factor": 8.0, "rope_theta": 1000000, "rope_type": "linear"},
    \\"sliding_attention": {"rope_theta": 10000, "rope_type": "default"}
    \\}
    \\},
    \\"text_encoder_special_tokens_config": {
    \\"token_ids": {"[S0]": 262146, "[S1]": 262147, "<ins_bos>": 262156, "<ins_eos>": 262157}
    \\},
    \\"codec_config": {
    \\"model_type": "mimi",
    \\"sampling_rate": 24000, "hidden_size": 512,
    \\"codebook_size": 2048, "codebook_dim": 256,
    \\"num_quantizers": 32, "compress": 2,
    \\"upsample_ratios": [8, 6, 5, 4]
    \\}
    \\}
;

test "parseConfig: published Breeze-TTS-2 pack geometry" {
    const cfg = try parseConfig(testing.allocator, pack_config);
    try testing.expectEqual(@as(u32, 16), cfg.num_codebooks);
    try testing.expectEqual(@as(u32, 2051), cfg.codebook_vocab);
    try testing.expectEqual(@as(u32, 2050), cfg.codebook_pad);
    try testing.expectEqual(@as(u32, 262144), cfg.audio_token_id);
    try testing.expectEqual(@as(u32, 262145), cfg.audio_eos_token_id);
    try testing.expectEqual(@as(u32, 28), cfg.b_layers);
    try testing.expectEqual(@as(u32, 128), cfg.b_head_dim);
    try testing.expectEqual(@as(u32, 12), cfg.d_layers);
    try testing.expectEqual(@as(u32, 2), cfg.d_kv);
    try testing.expectEqual(@as(u32, 26), cfg.te_layers);
    try testing.expectEqual(@as(u32, 256), cfg.te_head_dim);
    try testing.expectEqual(@as(u32, 24000), cfg.sample_rate);
    // 12.5 Hz codec frames: 24000 / (8*6*5*4 * 2).
    try testing.expectApproxEqAbs(@as(f64, 12.5), cfg.frameRate(), 1e-4);
}

test "parseConfig: T5Gemma2 full-attention cadence is a layer predicate" {
    const cfg = try parseConfig(testing.allocator, pack_config);
    try testing.expectEqual(@as(usize, 4), cfg.te_full_count);
    try testing.expect(!isFullAttentionLayer(&cfg, 0));
    try testing.expect(isFullAttentionLayer(&cfg, 5));
    try testing.expect(isFullAttentionLayer(&cfg, 23));
    try testing.expect(!isFullAttentionLayer(&cfg, 24));
    try testing.expectApproxEqAbs(@as(f64, 8.0), cfg.te_full_rope_factor, 1e-9);
}

test "parseConfig: depth decoder llama3 scaling parsed, not the backbone default" {
    const cfg = try parseConfig(testing.allocator, pack_config);
    try testing.expectApproxEqAbs(@as(f64, 32.0), cfg.d_rope_factor, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.001953125), cfg.d_rope_low, 1e-12);
    try testing.expectEqual(@as(u32, 16), cfg.d_rope_orig_max);
    // Absent scaling keeps defaults rather than rejecting.
    const bare = try parseConfig(testing.allocator, "{}");
    try testing.expectApproxEqAbs(@as(f64, 32.0), bare.d_rope_factor, 1e-9);
}

test "parseConfig: voice-design tokens addressable as [S0]..[S9]" {
    const cfg = try parseConfig(testing.allocator, pack_config);
    try testing.expectEqual(@as(u32, 262146), speakerToken(&cfg, 0).?);
    try testing.expectEqual(@as(u32, 262155), speakerToken(&cfg, 9).?);
    try testing.expect(speakerToken(&cfg, 10) == null);
    try testing.expectEqual(@as(u32, 262156), cfg.ins_bos_token);
    try testing.expect(voiceSlotValid("S0") and voiceSlotValid("s9"));
    try testing.expect(!voiceSlotValid("S10"));
    try testing.expect(!voiceSlotValid("banana"));
}

test "parseConfig: contradictory geometry refused by name" {
    // A codebook count the depth decoder cannot match is a bad read, not a silent crop.
    try testing.expectError(error.DepthCodebookMismatch, parseConfig(testing.allocator,
        \\{"audio_num_codebooks": 16, "depth_decoder_config": {"num_codebooks": 8}}
    ));
    // audio_embed_size must equal the backbone width (the concat is the contract).
    try testing.expectError(error.AudioEmbedMismatch, parseConfig(testing.allocator,
        \\{"audio_embed_size": 1024, "backbone_config": {"hidden_size": 2048}}
    ));
    // GQA ratios that do not divide are refused at parse, not at first forward.
    try testing.expectError(error.BadBackboneGqa, parseConfig(testing.allocator,
        \\{"backbone_config": {"num_attention_heads": 16, "num_key_value_heads": 3}}
    ));
    try testing.expectError(error.BadCodebookVocab, parseConfig(testing.allocator,
        \\{"audio_vocab_size": 100, "codebook_pad_token_id": 2050}
    ));
}

test "Engine.load reads a pack dir and names a codec-less pack" {
    const io = testing.io;
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir_path = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir_path);

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = pack_config });
    // No audio_tokenizer/ yet → named refusal, not a fallback.
    try testing.expectError(error.MissingCodec, Engine.load(io, allocator, dir_path));

    try tmp.dir.createDirPath(io, "audio_tokenizer");
    try tmp.dir.writeFile(io, .{ .sub_path = "audio_tokenizer/model.safetensors", .data = "\x00\x00\x00\x01" });
    const engine = try Engine.load(io, allocator, dir_path);
    defer allocator.destroy(engine);
    try testing.expectEqual(@as(u32, 24000), engine.sampleRate());
}

test "Engine.load refuses relative and empty dirs before any open" {
    const io = testing.io;
    const allocator = testing.allocator;
    try testing.expectError(error.InvalidModelDir, Engine.load(io, allocator, ""));
    try testing.expectError(error.InvalidModelDir, Engine.load(io, allocator, "relative/dir"));
}
