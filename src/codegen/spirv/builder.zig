const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

const spec = @import("spec.zig");

pub const Word = spec.Word;
pub const DoubleWord = std.meta.Int(.unsigned, @bitSizeOf(Word) * 2);
pub const Log2Word = std.math.Log2Int(Word);

pub const Opcode = spec.Opcode;
pub const Instruction = spec.Instruction;

pub const Builder = struct {
    /// A general-purpose allocator which may be used to allocate temporary resources required for this
    /// builder and any sections.
    gpa: *Allocator,

    /// SPIR-V instructions return result-ids. This variable holds the module-wide counter for these.
    next_result_id: u32,

    pub fn init(gpa: *Allocator) Builder {
        return .{
            .gpa = gpa,
            .next_result_id = 1, // 0 is an invalid SPIR-V result ID.
        };
    }

    pub fn allocId(self: *Builder) spec.IdResult {
        defer self.next_result_id += 1;
        return .{.id = self.next_result_id};
    }

    pub fn idBound(self: Builder) Word {
        return self.next_result_id;
    }
};

/// Represents a section or subsection of instructions in a SPIR-V binary. Instructions can be append
/// to separate sections, which can then later be merged into the final binary.
pub const Section = struct {
    /// The instructions in this section. Memory is owned by the Builder
    /// externally associated to this Section.
    instructions: std.ArrayListUnmanaged(Word) = .{},

    pub fn deinit(section: *Section, builder: Builder) void {
        section.instructions.deinit(builder.gpa);
        section.* = undefined;
    }

    /// Clear the instructions in this section.
    pub fn reset(section: *Section) void {
        section.instructions.items.len = 0;
    }

    /// Write an instruction to this section.
    pub fn emit(
        section: *Section,
        builder: Builder,
        instruction: Instruction,
    ) !void {
        var word_count = instructionSize(instruction);
        try section.instructions.ensureUnusedCapacity(builder.gpa, word_count);
        section.instructions.appendAssumeCapacity((@intCast(Word, word_count << 16)) | @enumToInt(instruction));
        section.writeInstruction(instruction);
    }

    /// Write an instruction and size, operands are to be inserted manually.
    pub fn emitRaw(
        section: *Section,
        builder: Builder,
        opcode: Opcode,
        operands: usize, // opcode itself not included
    ) !void {
        const word_count = 1 + operands;
        try section.instructions.ensureUnusedCapacity(builder.gpa, word_count);
        section.instructions.appendAssumeCapacity((@intCast(Word, word_count << 16)) | @enumToInt(opcode));
    }

    /// Append the instructions from another section into this section.
    pub fn append(
        section: *Section,
        builder: Builder,
        other_section: Section
    ) !void {
        try section.instructions.appendSlice(builder.gpa, other_section.instructions.items);
    }

    pub fn writeInstruction(section: *Section, instruction: Instruction) void {
        const opcode = std.meta.activeTag(instruction);
        @setEvalBranchQuota(2000);
        inline for (@typeInfo(Instruction).Union.fields) |field| {
            if (@field(Instruction, field.name) == opcode) {
                section.writeOperands(field.field_type, @field(instruction, field.name));
                return;
            }
        }
        unreachable;
    }

    /// Write a sequence of operands to the instructions array. Note: Assumes sufficient capacity!
    pub fn writeOperands(section: *Section, comptime Operands: type, operands: Operands) void {
        const fields = switch (@typeInfo(Operands)) {
            .Pointer => |info| @typeInfo(info.child).Struct.fields,
            .Struct => |info| info.fields,
            .Void => return,
            else => unreachable,
        };

        inline for (fields) |field| {
            section.writeOperand(field.field_type, @field(operands, field.name));
        }
    }

    /// Write a single operand to the instructions array. Note: Assumes sufficient capacity!
    pub fn writeOperand(section: *Section, comptime Operand: type, operand: Operand) void {
        switch (Operand) {
            spec.IdResultType,
            spec.IdResult,
            spec.IdRef
            => section.writeWord(operand.id),
            spec.LiteralInteger => section.writeWord(operand),
            spec.LiteralString => section.writeString(operand),
            spec.LiteralContextDependentNumber => section.writeContextDependentNumber(operand),
            spec.LiteralExtInstInteger => section.writeWord(operand.inst),
            // TODO: Where this type is used (OpSpecConstantOp) is currently not correct in the spec json,
            // so it most likely needs to be altered into something that can actually describe the entire
            // instruction in which it is used.
            spec.LiteralSpecConstantOpInteger => section.writeWord(@enumToInt(operand.opcode)),
            spec.PairLiteralIntegerIdRef => section.writeWords(&.{ operand.value, operand.label.id }),
            spec.PairIdRefLiteralInteger => section.writeWords(&.{ operand.target.id, operand.member }),
            spec.PairIdRefIdRef => section.writeWords(&.{ operand[0].id, operand[1].id }),
            else => switch (@typeInfo(Operand)) {
                .Enum => section.writeWord(@enumToInt(operand)),
                .Optional => |info| if (operand) |inner| {
                    section.writeOperand(info.child, inner);
                },
                .Pointer => |info| {
                    std.debug.assert(info.size == .Slice); // Should be no other pointer types in the spec.
                    for (operand) |item| {
                        section.writeOperand(info.child, item);
                    }
                },
                .Struct => |info| {
                    if (info.layout == .Packed) {
                        section.writeWord(@bitCast(Word, operand));
                    } else {
                        section.writeExtendedMask(Operand, operand);
                    }
                },
                .Union => section.writeExtendedUnion(Operand, operand),
                else => unreachable,
            },
        }
    }

    fn writeWord(section: *Section, word: Word) void {
        section.instructions.appendAssumeCapacity(word);
    }

    fn writeWords(section: *Section, words: []const Word) void {
        section.instructions.appendSliceAssumeCapacity(words);
    }

    fn writeDoubleWord(section: *Section, dword: DoubleWord) void {
        section.writeWords(&.{
            @truncate(Word, dword),
            @truncate(Word, dword >> @bitSizeOf(Word)),
        });
    }

    fn writeString(section: *Section, str: []const u8) void {
        // TODO: Not actually sure whether this is correct for big-endian.
        // See https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#Literal
        const zero_terminated_len = str.len + 1;
        var i: usize = 0;
        while (i < zero_terminated_len) : (i += @sizeOf(Word)) {
            var word: Word = 0;

            var j: usize = 0;
            while (j < @sizeOf(Word) and i + j < str.len) : (j += 1) {
                word |= @as(Word, str[i + j]) << @intCast(Log2Word, j * std.meta.bitCount(u8));
            }

            section.instructions.appendAssumeCapacity(word);
        }
    }

    fn writeContextDependentNumber(section: *Section, operand: spec.LiteralContextDependentNumber) void {
        switch (operand) {
            .int32 => |int| section.writeWord(@bitCast(Word, int)),
            .uint32 => |int| section.writeWord(@bitCast(Word, int)),
            .int64 => |int| section.writeDoubleWord(@bitCast(DoubleWord, int)),
            .uint64 => |int| section.writeDoubleWord(@bitCast(DoubleWord, int)),
            .float32 => |float| section.writeWord(@bitCast(Word, float)),
            .float64 => |float| section.writeDoubleWord(@bitCast(DoubleWord, float)),
        }
    }

    fn writeExtendedMask(section: *Section, comptime Operand: type, operand: Operand) void {
        var mask: Word = 0;
        inline for (@typeInfo(Operand).Struct.fields) |field, bit| {
            switch (@typeInfo(field.field_type)) {
                .Optional => if (@field(operand, field.name) != null) {
                    mask |= 1 << @intCast(u5, bit);
                },
                .Bool => if (@field(operand, field.name)) {
                    mask |= 1 << @intCast(u5, bit);
                },
                else => unreachable,
            }
        }

        if (mask == 0) {
            return;
        }

        section.writeWord(0);

        inline for (@typeInfo(Operand).Struct.fields) |field| {
            switch (@typeInfo(field.field_type)) {
                .Optional => |info| if (@field(operand, field.name)) |child| {
                    section.writeOperands(info.child, child);
                },
                .Bool => {},
                else => unreachable,
            }
        }
    }

    fn writeExtendedUnion(section: *Section, comptime Operand: type, operand: Operand) void {
        const tag = std.meta.activeTag(operand);
        section.writeWord(@enumToInt(tag));

        inline for (@typeInfo(Operand).Union.fields) |field| {
            if (@field(Operand, field.name) == tag) {
                section.writeOperands(field.field_type, @field(operand, field.name));
                return;
            }
        }
        unreachable;
    }
};

fn instructionSize(instruction: Instruction) usize {
    const opcode = std.meta.activeTag(instruction);
    @setEvalBranchQuota(2000);
    inline for (@typeInfo(Instruction).Union.fields) |field| {
        if (@field(Instruction, field.name) == opcode) {
            // Add one for the instruction itself.
            return 1 + operandsSize(field.field_type, @field(instruction, field.name));
        }
    }
    unreachable;
}

fn operandsSize(comptime Operands: type, operands: Operands) usize {
    const fields = switch (@typeInfo(Operands)) {
        .Pointer => |info| @typeInfo(info.child).Struct.fields,
        .Struct => |info| info.fields,
        .Void => return 0,
        else => unreachable,
    };

    var total: usize = 0;
    inline for (fields) |field| {
        total += operandSize(field.field_type, @field(operands, field.name));
    }
    return total;
}

fn operandSize(comptime Operand: type, operand: Operand) usize {
    return switch (Operand) {
        void => 0,
        spec.IdResultType,
        spec.IdResult,
        spec.IdRef,
        spec.LiteralInteger,
        spec.LiteralExtInstInteger,
        => 1,
        spec.LiteralString => std.math.divCeil(usize, operand.len + 1, @sizeOf(Word)) catch unreachable, // Add one for zero-terminator.
        spec.LiteralContextDependentNumber => switch (operand) {
            .int32, .uint32, .float32 => @as(usize, 1),
            .int64, .uint64, .float64 => 2,
        },
        // TODO: Where this type is used (OpSpecConstantOp) is currently not correct in the spec json,
        // so it most likely needs to be altered into something that can actually describe the entire
        // instruction in which it is used.
        spec.LiteralSpecConstantOpInteger => 1,
        spec.PairLiteralIntegerIdRef,
        spec.PairIdRefLiteralInteger,
        spec.PairIdRefIdRef,
        => 2,
        else => switch (@typeInfo(Operand)) {
            .Enum => 1,
            .Optional => |info| if (operand) |child| operandSize(info.child, child) else 0,
            .Pointer => |info| blk: {
                std.debug.assert(info.size == .Slice); // Should be no other pointer types in the spec.
                var total: usize = 0;
                for (operand) |item| {
                    total += operandSize(info.child, item);
                }
                break :blk total;
            },
            .Struct => |info| if (info.layout == .Packed) 1 else extendedMaskSize(Operand, operand),
            .Union => extendedUnionSize(Operand, operand),
            else => unreachable,
        },
    };
}

fn extendedMaskSize(comptime Operand: type, operand: Operand) usize {
    var total: usize = 0;
    var any_set = false;
    inline for (@typeInfo(Operand).Struct.fields) |field| {
        switch (@typeInfo(field.field_type)) {
            .Optional => |info| if (@field(operand, field.name)) |child| {
                total += operandsSize(info.child, child);
                any_set = true;
            },
            .Bool => if (@field(operand, field.name)) {
                any_set = true;
            },
            else => unreachable,
        }
    }
    if (!any_set) {
        return 0;
    }
    return total + 1; // Add one for the mask itself.
}

fn extendedUnionSize(comptime Operand: type, operand: Operand) usize {
    const tag = std.meta.activeTag(operand);
    inline for (@typeInfo(Operand).Union.fields) |field| {
        if (@field(Operand, field.name) == tag) {
            // Add one for the tag itself.
            return 1 + operandsSize(field.field_type, @field(operand, field.name));
        }
    }
    unreachable;
}

test "SPIR-V Builder - no args" {
    var builder = Builder.init(std.testing.allocator);

    var section = Section{};
    defer section.deinit(builder);

    try section.emit(builder, .OpNop);

    try testing.expect(section.instructions.items[0] == (@as(Word, 1) << 16) | @enumToInt(Opcode.OpNop));
}

test "SPIR-V Builder - simple" {
    var builder = Builder.init(std.testing.allocator);

    var section = Section{};
    defer section.deinit(builder);

    try section.emit(builder, .{.OpUndef = &.{
        .id_result_type = .{.id = 0},
        .id_result = .{.id = 1},
    }});

    try testing.expectEqualSlices(Word, section.instructions.items, &.{
        (@as(Word, 3) << 16) | @enumToInt(Opcode.OpUndef),
        0,
        1,
    });
}

test "SPIR-V Builder - string" {
    var builder = Builder.init(std.testing.allocator);

    var section = Section{};
    defer section.deinit(builder);

    try section.emit(builder, .{.OpSource = &.{
        .source_language = .Unknown,
        .version = 123,
        .file = .{.id = 456},
        .source = "pub fn main() void {}",
    }});

    try testing.expectEqualSlices(Word, section.instructions.items, &.{
        (@as(Word, 10) << 16) | @enumToInt(Opcode.OpSource),
        @enumToInt(spec.SourceLanguage.Unknown),
        123,
        456,
        std.mem.bytesToValue(Word, "pub "),
        std.mem.bytesToValue(Word, "fn m"),
        std.mem.bytesToValue(Word, "ain("),
        std.mem.bytesToValue(Word, ") vo"),
        std.mem.bytesToValue(Word, "id {"),
        std.mem.bytesToValue(Word, "}\x00\x00\x00"),
    });
}

test "SPIR-V Builder - extended mask" {
    var builder = Builder.init(std.testing.allocator);

    var section = Section{};
    defer section.deinit(builder);

    try section.emit(builder, .{.OpLoopMerge = &.{
        .merge_block = .{.id = 10},
        .continue_target = .{.id = 20},
        .loop_control = .{
            .Unroll = true,
            .DependencyLength = .{
                .literal_integer = 2,
            },
        },
    }});

    try testing.expectEqualSlices(Word, section.instructions.items, &.{
        (@as(Word, 5) << 16) | @enumToInt(Opcode.OpLoopMerge),
        10,
        20,
        @bitCast(Word, spec.LoopControl{.Unroll = true, .DependencyLength = true}),
        2,
    });
}

test "SPIR-V Builder - extended union" {
    var builder = Builder.init(std.testing.allocator);

    var section = Section{};
    defer section.deinit(builder);

    try section.emit(builder, .{.OpExecutionMode = &.{
        .entry_point = .{.id = 888},
        .mode = .{
            .LocalSize = .{.x_size = 4, .y_size = 8, .z_size = 16},
        },
    }});

    try testing.expectEqualSlices(Word, section.instructions.items, &.{
        (@as(Word, 6) << 16) | @enumToInt(Opcode.OpExecutionMode),
        888,
        @enumToInt(spec.ExecutionMode.LocalSize),
        4,
        8,
        16,
    });
}
