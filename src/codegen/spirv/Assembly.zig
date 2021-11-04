//! SPIR-V inline assembly assembler.
//! Syntax is based on https://github.com/KhronosGroup/SPIRV-Tools/blob/master/docs/syntax.md,
//! with the modification that we do not support arbitrary integers.

const std = @import("std");
const assert = std.debug.assert;

const spec = @import("spec.zig");
const Opcode = spec.Opcode;
const Instruction = spec.Instruction;
const IdResult = spec.IdResult;

const DeclGen = @import("../spirv.zig").DeclGen;
const ZigModule = @import("../../Module.zig");
const LazySrcLoc = ZigModule.LazySrcLoc;

const Assembly = @This();

/// The declaration we are going to generate assembly for.
dg: *DeclGen,

/// A tokenizer used to loosely tokenize the assembly source into tokens.
tokenizer: Tokenizer,

/// An arena allocator that is used to store information that persists during the
/// entire assembling operation, like instruction operands.
arena: std.heap.ArenaAllocator,

/// Parsed sequence of instructions that appears in the assembly template.
/// Here, result-id's do not correspond with actual result-ids, but with indices into
/// the `id_map`. These are later resolved and patched in when the final assembly is emitted.
instructions: std.ArrayListUnmanaged(spec.Instruction) = .{},

/// A set of identifiers appearing in the assembly template. Indices may be referred to by index.
id_map: std.StringArrayHashMapUnmanaged(void) = .{},

const Error = error{
    OutOfMemory,
    CodegenFail,
};

const ParseOperandError = Error || error{
    Overflow,
    UnexpectedToken,
    InvalidCharacter,
};

/// Assemble a SPIR-V assembly template into a state that can be worked with.
/// Returns `error.CodegenFail` if something is not correct, after which `dg.error_msg` will be set.
pub fn assemble(dg: *DeclGen, template: []const u8) Error!Assembly {
    var as = Assembly{
        .dg = dg,
        .tokenizer = .{.source = template},
        .arena = std.heap.ArenaAllocator.init(dg.spv.gpa),
    };
    as.parse() catch |err| switch (err) {
        error.CodegenFail => return error.CodegenFail,
        else => return as.fail("Failed to parse", .{}),
    };
    return as;
}

pub fn deinit(self: *Assembly) void {
    self.arena.deinit();
    self.instructions.deinit(self.dg.spv.gpa);
    self.id_map.deinit(self.dg.spv.gpa);
    self.* = undefined;
}

fn fail(self: *Assembly, comptime format: []const u8, args: anytype) Error {
    @setCold(true);
    assert(self.dg.error_msg == null);
    const src: LazySrcLoc = .{ .node_offset = 0 };
    const src_loc = src.toSrcLoc(self.dg.decl);
    self.dg.error_msg = try ZigModule.ErrorMsg.create(self.dg.module.gpa, src_loc, "SPIR-V Assembler: " ++ format, args);
    return error.CodegenFail;
}

/// Parse the assembly template into the internal array of instructions.
fn parse(self: *Assembly) !void {
    _ = try self.parseInstruction();
}

fn parseInstruction(self: *Assembly) !Instruction {
    const token = self.tokenizer.next();
    if (token.id != .value)
        return error.UnexpectedToken;
    const text = self.tokenText(token);
    const opcode = EnumNameMap(Opcode).get(text) orelse return error.NoSuchOpcode;

    inline for (@typeInfo(Instruction).Union.fields) |field| {
        if (@field(Instruction, field.name) == opcode) {
            const operands = try self.parseOperands(field.field_type);
            return @unionInit(Instruction, field.name, operands);
        }
    }

    unreachable;
}

fn parseOperands(self: *Assembly, comptime Operands: type) ParseOperandError!Operands {
    const fields = switch (@typeInfo(Operands)) {
        .Pointer => |info| @typeInfo(info.child).Struct.fields,
        .Struct => |info| info.fields,
        .Void => return,
        else => unreachable,
    };

    var operands = switch (@typeInfo(Operands)) {
        .Pointer => |info| try self.arena.allocator.create(info.child),
        .Struct => @as(Operands, undefined),
        else => unreachable,
    };

    inline for (fields) |field| {
        @field(operands, field.name) = try self.parseOperand(field.field_type);
    }

    return operands;
}

fn parseOperand(self: *Assembly, comptime Operand: type) ParseOperandError!Operand {
    switch (Operand) {
        spec.LiteralInteger => {
            const token = self.tokenizer.next();
            if (token.id != .value)
                return error.UnexpectedToken;
            const text = self.tokenText(token);
            return try std.fmt.parseInt(spec.LiteralInteger, text, 0);
        },
        spec.IdResultType, spec.IdResult, spec.IdRef => {
            // TODO: Also allow names here instead of direct id references.
            const token = self.tokenizer.next();
            if (token.id != .identifier)
                return error.UnexpectedToken;
            const text = self.tokenText(token);
            const val = try std.fmt.parseInt(spec.LiteralInteger, text[1..], 0);
            return Operand{.id = val};
        },
        spec.PairIdRefIdRef => {
            const a = try self.parseOperand(spec.IdRef);
            const b = try self.parseOperand(spec.IdRef);
            return Operand{a, b};
        },
        else => return self.fail("TODO: Parse operand of type " ++ @typeName(Operand), .{}),
    }
}

fn tokenText(self: Assembly, token: Token) []const u8 {
    return self.tokenizer.source[token.start..token.end];
}

fn EnumNameMap(comptime Enum: type) type {
    const KV = std.meta.Tuple(&.{[]const u8, Enum});
    const fields = @typeInfo(Enum).Enum.fields;
    var pairs = [_]KV{undefined} ** fields.len;
    for (pairs) |*pair, i| {
        pair.* = .{fields[i].name, @field(Enum, fields[i].name)};
    }
    return std.ComptimeStringMap(Enum, pairs);
}

const Token = struct {
    id: Id,

    /// Offsets are relative to the assembly template string.
    start: usize,
    end: usize,

    const Id = enum {
        invalid,
        eof,

        /// A literal string. Supports the usual escape sequences, but note that the string literal
        /// used for the inline assembly also supports these escapes.
        /// Token includes quotes (first and last characters).
        string,

        /// A literal name, integer or float value. This is used for instructions, enums, masks, extended instructions,
        /// the opcode for OpSpecConstantOp, and constants. The way this value is interpreted depends on the context, and
        /// so this not done until later.
        /// Note that this is different from an identifier, which is prefixed with a %.
        /// Literals may consist of alphanumeric characters, '_' and '.'.
        /// Note: Contents are not validated.
        value,

        /// A %-prefixed literal name, which is used for identifiers.
        /// Token includes percent (first character).
        /// Note: May consist of only digits, outside the percent.
        /// Note: Contents are not validated.
        identifier,

        /// The '|' symbol, used to construct masks.
        pipe,

        /// The '=' symbol.
        equals,
    };
};

/// This struct is used to tokenize the assembly source into different tokens. Note that tokenization is
/// done in a very loose manner: Tokens in SPIR-V assembly are always separated by whitespace, and we
/// interpret any whitespace-delimited token without verifying the token's exact contents.
const Tokenizer = struct {
    source: []const u8,
    offset: usize = 0,

    fn next(self: *Tokenizer) Token {
        // First, skip any whitespace.
        while (self.offset < self.source.len) : (self.offset += 1) {
            if (!std.ascii.isSpace(self.source[self.offset]))
                break;
        }
        if (self.offset == self.source.len) {
            return Token{
                .id = .eof,
                .start = self.offset,
                .end = self.offset,
            };
        }
        // Tokens are always separated by whitespace in SPIR-V assembler,
        // so fetch the token's text by skipping non-whitespace.
        // Strings are an exception and need to be handled separately.
        if (self.source[self.offset] == '"') {
            return self.string();
        }
        const start = self.offset;
        while (self.offset < self.source.len) : (self.offset += 1) {
            if (std.ascii.isSpace(self.source[self.offset]))
                break;
        }

        // Note: length guaranteed > 0.
        var text = self.source[start .. self.offset];
        const id = switch (text[0]) {
            '%' => .identifier,
            '=' => if (text.len == 1) .equals else Token.Id.invalid,
            '|' => if (text.len == 1) .pipe else Token.Id.invalid,
            else => .value,
        };

        return .{
            .id = id,
            .start = start,
            .end = self.offset,
        };
    }

    fn string(self: *Tokenizer) Token {
        const start = self.offset;
        self.offset += 1; // skip over opening "
        // We're just going to skip over a single char when handling escapes,
        // they are handled properly later on, we just need it to not terminate when
        // encountering '\"'.
        var in_escape = false;
        while (self.offset < self.source.len) : (self.offset += 1) {
            const c = self.source[self.offset];
            switch (c) {
                '\\' => in_escape = true,
                '"' => if (in_escape) {
                    in_escape = false;
                } else return .{
                    // TODO: Verify that character after string is whitespace.
                    .id = .string,
                    .start = start,
                    .end = self.offset + 1,
                },
                else => in_escape = false,
            }
        }

        return .{
            .id = .invalid,
            .start = start,
            .end = self.offset,
        };
    }
};
