const builtin = @import("builtin");

test "fixed" {
    if (builtin.zig_backend == .stage2_spirv64) return error.SkipZigTest;

    const a: *void = undefined;
    const b: *[1]void = a;
    _ = b;
    const c: *[0]u8 = undefined;
    const d: []u8 = c;
    _ = d;
}
