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

/// The node pointing to the assembly's asm() statement.
src_node: i32,

/// The assembly template of the asm() statement.
template: []const u8,

/// An arena allocator that is used to store information that persists during the
/// entire assembling operation, like instruction operands.
arena: std.heap.ArenaAllocator,

/// Parsed sequence of instructions that appears in the assembly template.
/// Here, result-id's do not correspond with actual result-ids, but with indices into
/// the `id_map`. These are later resolved and patched in when the final assembly is emitted.
instructions: std.ArrayListUnmanaged(spec.Instruction) = .{},

/// A set of identifiers appearing in the assembly template. Indices may be referred to by index.
id_map: std.StringArrayHashMapUnmanaged(void) = .{},

/// A reference to a token, which may be of any type.
const Token = struct {
    /// The starting offset of this token.
    start: u32,

    /// The ending offset of this token (exclusive).
    end: u32,
};

const AssembleError = error{
    OutOfMemory,
    CodegenFail,
};

/// Assemble a SPIR-V assembly template into a state that can be worked with.
/// Returns `error.CodegenFail` if something is not correct, after which `dg.error_msg` will be set.
pub fn assemble(dg: *DeclGen, src_node: i32, template: []const u8) AssembleError!Assembly {
    var as = Assembly{
        .dg = dg,
        .src_node = src_node,
        .template = template,
        .arena = std.heap.ArenaAllocator.init(dg.spv.gpa),
    };
    try as.parse();
    return as;
}

pub fn deinit(self: *Assembly) void {
    self.arena.deinit();
    self.instructions.deinit(self.dg.spv.gpa);
    self.id_map.deinit(self.dg.spv.gpa);
    self.* = undefined;
}

fn fail(self: *Assembly, comptime format: []const u8, args: anytype) AssembleError {
    @setCold(true);
    assert(self.dg.error_msg == null);
    const src: LazySrcLoc = .{ .node_offset_asm_source = self.src_node };
    const src_loc = src.toSrcLoc(self.dg.decl);
    self.dg.error_msg = try ZigModule.ErrorMsg.create(self.dg.module.gpa, src_loc, "SPIR-V Assembler: " ++ format, args);
    return error.CodegenFail;
}

fn parse(self: *Assembly) !void {
    var parser = Parser{ .assembly = self };
    while (try parser.nextInstruction()) |ins| {
        std.debug.print("Parsed instruction {}\n", .{ @as(Opcode, ins)});
    }
}

pub fn formatOperand(
    bytes: []const u8,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    _ = fmt;
    _ = options;
    try writer.writeAll(bytes);
}

fn fmtOperandName(comptime Operand: type) std.fmt.Formatter(formatOperand) {
    return .{
        // TODO: Generate better names for some of these
        .data = "<" ++ @typeName(Operand) ++ ">",
    };
}

fn EnumNameMap(comptime Enum: type) type {
    const KV = std.meta.Tuple(&.{ []const u8, Enum });
    const fields = @typeInfo(Enum).Enum.fields;
    var pairs = [_]KV{undefined} ** fields.len;
    for (pairs) |*pair, i| {
        pair.* = .{ fields[i].name, @field(Enum, fields[i].name) };
    }
    return std.ComptimeStringMap(Enum, pairs);
}

const Parser = struct {
    assembly: *Assembly,
    offset: u32 = 0,

    const ParseError = error{
        OutOfMemory,
        CodegenFail,
    };

    fn nextInstruction(self: *Parser) ParseError!?Instruction {
        const result_or_opcode_tok = (try self.nextToken()) orelse return null;

        // Cheeky check to see if we have a result-id on the LHS
        const maybe_result_id = if (result_or_opcode_tok[0] == '%') blk: {
            const result_id = try self.parseOperand(spec.IdResult, result_or_opcode_tok);
            try self.expectText("=");
            break :blk result_id;
        } else null;

        const opcode_tok = if (maybe_result_id == null) blk: {
            if (!looksLikeOpcode(result_or_opcode_tok)) {
                return self.assembly.fail("Expected {} or {}, at the start of an instruction, found '{s}'", .{
                    fmtOperandName(Opcode),
                    fmtOperandName(spec.IdResult),
                    result_or_opcode_tok,
                });
            }
            break :blk result_or_opcode_tok;
        } else if (try self.nextToken()) |tok|
            tok
        else
            return self.assembly.fail("Unexpected end of input, expected {}", .{fmtOperandName(Opcode)});

        const opcode = EnumNameMap(Opcode).get(opcode_tok) orelse {
            return self.assembly.fail("Invalid opcode '{s}'", .{opcode_tok});
        };

        inline for (@typeInfo(Instruction).Union.fields) |field| {
            if (@field(Instruction, field.name) == opcode) {
                const operands = try self.parseOperands(field.field_type, opcode, maybe_result_id);
                return @unionInit(Instruction, field.name, operands);
            }
        }

        unreachable;
    }

    fn looksLikeOpcode(text: []const u8) bool {
        const prefix = "Op";
        return text.len > prefix.len and
            std.mem.startsWith(u8, text, prefix) and
            std.ascii.isUpper(text[prefix.len]);
    }

    fn expectText(self: *Parser, text: []const u8) !void {
        if (try self.nextToken()) |tok| {
            if (!std.mem.eql(u8, tok, text)) {
                return self.assembly.fail("Unexpected '{s}', expected '{s}'", .{ tok, text });
            }
        } else {
            return self.assembly.fail("Unexpected end of input, expected '{s}'", .{text});
        }
    }

    fn parseOperands(
        self: *Parser,
        comptime Operands: type,
        opcode: Opcode,
        maybe_result_id: ?spec.IdResult,
    ) ParseError!Operands {
        const fields = switch (@typeInfo(Operands)) {
            .Pointer => |info| @typeInfo(info.child).Struct.fields,
            .Struct => |info| info.fields,
            .Void => return,
            else => unreachable,
        };

        var operands = switch (@typeInfo(Operands)) {
            .Pointer => |info| try self.assembly.arena.allocator.create(info.child),
            .Struct => @as(Operands, undefined),
            else => unreachable,
        };

        inline for (fields) |field| {
            if (field.field_type == spec.IdResult) {
                if (maybe_result_id == null) {
                    return self.assembly.fail("Instruction '{s}' expects {} on the left-hand side", .{
                        @tagName(opcode),
                        fmtOperandName(spec.IdResult),
                    });
                }
            }
        }

        inline for (fields) |field| {
            @field(operands, field.name) = if (field.field_type == spec.IdResult)
                maybe_result_id.?
            else
                try self.nextOperand(field.field_type);
        }

        return operands;
    }

    fn nextOperand(self: *Parser, comptime Operand: type) !Operand {
        if (try self.nextToken()) |tok| {
            return self.parseOperand(Operand, tok);
        } else {
            return self.assembly.fail("Unexpected end of input, expected operand {}", .{fmtOperandName(Operand)});
        }
    }

    fn parseOperand(self: *Parser, comptime Operand: type, token: []const u8) ParseError!Operand {
        switch (Operand) {
            spec.LiteralInteger => {
                return std.fmt.parseInt(spec.LiteralInteger, token, 0) catch {
                    return self.assembly.fail("Invalid integer literal '{s}'", .{token});
                };
            },
            spec.IdResultType, spec.IdResult, spec.IdRef => {
                // TODO: Also allow names here instead of direct id references.
                invalid_id: {
                    if (token[0] != '%')
                        break :invalid_id;

                    const id = std.fmt.parseInt(spec.LiteralInteger, token[1..], 0) catch break :invalid_id;
                    return Operand{ .id = id };
                }

                return self.assembly.fail("Invalid identifier '{s}'", .{token});
            },
            spec.PairIdRefIdRef => {
                const a = try self.nextOperand(spec.IdRef);
                const b = try self.nextOperand(spec.IdRef);
                return Operand{a, b};
            },
            else => return self.assembly.fail("TODO: Parse operand of type " ++ @typeName(Operand), .{}),
        }
    }

    /// The returned token spans at least one character.
    fn nextToken(self: *Parser) !?[]const u8 {
        // Tokens in spir-v are delimited by whitespace or eof on either side.
        // When a string appears in a token, we parse until the closing quote, skipping over
        // any quote escapes. The input `a"b c"d` will be regarded as a single token.
        self.skipWhitespace();

        var state: enum {
            start,
            text,
            string,
            escape,
        } = .start;

        const template = self.assembly.template;
        const start = self.offset;
        while (self.offset < template.len) : (self.offset += 1) {
            const c = template[self.offset];
            switch (state) {
                .start, .text => switch (c) {
                    '"' => state = .string,
                    else => if (std.ascii.isSpace(c)) {
                        break;
                    } else {
                        state = .text;
                    },
                },
                .string => switch (c) {
                    '\\' => state = .escape,
                    '"' => state = .text,
                    // Note: Strings in spir-v may include newlines.
                    else => {},
                },
                // Handle escape sequences by skipping over a single character after the
                // escape, and handle it properly when the token is interpreted.
                .escape => state = .text,
            }
        }

        return switch (state) {
            .start => null,
            .text => template[start..self.offset],
            .string, .escape => self.assembly.fail("Unterminated string", .{}),
        };
    }

    fn skipWhitespace(self: *Parser) void {
        const template = self.assembly.template;
        while (self.offset < template.len) : (self.offset += 1) {
            if (!std.ascii.isSpace(template[self.offset]))
                break;
        }
    }
};
