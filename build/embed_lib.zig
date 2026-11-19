//! The fleet embed: `src/embed.zig` — the engine's C ABI for a BEAM NIF host —
//! as a macOS static lib (`zig build mac-lib` → zig-out/lib/libmlxserve_embed.a).
//! The module is the SERVER's module (same imports, same linked libs), so the
//! embed shares every seam main.zig uses; see src/embed.zig for the ABI.

const std = @import("std");
const macos = @import("macos.zig");

pub fn add(b: *std.Build, g: *const macos.Graph) void {
    const lib = b.addLibrary(.{
        .name = "mlxserve_embed",
        .root_module = g.embedModule(),
        .linkage = .static,
    });
    // Bundle compiler_rt: the NIF .so dlopens the static lib, and macOS's
    // compiler-rt excludes the quad/i128 soft-float symbols it references
    // (__addtf3 etc.). Without this the dlopen fails on `___addtf3`.
    lib.bundle_compiler_rt = true;
    const install = b.addInstallArtifact(lib, .{
        .dest_dir = .{ .override = .{ .custom = "lib" } },
    });
    const step = b.step("mac-lib", "Build the fleet embed static lib (libmlxserve_embed.a)");
    step.dependOn(&install.step);

    // Host-side unit tests of embed.zig (the metal-limit retry guard's pure helpers ride the
    // same module; the server's test step does not load embed.zig). The test
    // binary does not link the engine table (`mlx_serve_mlx_api`), so it runs
    // only when explicitly filtered; `mac-lib` itself just builds the lib.
    const embed_filter = b.option([]const u8, "embed-test-filter", "Only run embed tests whose name contains this substring");
    if (embed_filter) |f| {
        const embed_tests = b.addTest(.{
            .name = "embed-test",
            .root_module = g.embedModule(),
            .filters = &.{f},
        });
        embed_tests.root_module.addImport("mlx_embed_test_api", b.createModule(.{
            .root_source_file = b.path("src/engines/mlx/api.zig"),
        }));
        const run_embed_tests = b.addRunArtifact(embed_tests);
        run_embed_tests.has_side_effects = true;
        step.dependOn(&run_embed_tests.step);
    }
}