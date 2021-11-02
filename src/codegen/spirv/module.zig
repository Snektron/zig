const std = @import("std");
const Allocator = std.mem.Allocator;

const spec = @import("spec.zig");
const IdResult = spec.IdResult;
const IdRef = spec.IdRef;
const Word = spec.Word;

const Section = @import("Section.zig");
const SpvType = @import("type.zig").Type;

const SpvTypeCache = std.ArrayHashMapUnmanaged(SpvType, IdRef, SpvType.ShallowHashContext32, true);

pub const Module = struct {
    /// A general-purpose allocator which may be used to allocate resources for this module.
    gpa: *Allocator,

    /// Module layout, according to spec 2.4, Logical Layout of a Module.
    sections: struct {
        /// OpCapability instructions.
        capabilities: Section = .{},
        /// OpExtension instructions.
        extensions: Section = .{},
        // OpExtInstImport instructions - skip for now.
        // memory model defined by target, not required here.
        /// OpEntryPoint instructions.
        entry_points: Section = .{},
        // OpExecutionMode and OpExecutionModeId instructions - skip for now.
        /// OpString, OpSourcExtension, OpSource, OpSourceContinued.
        debug_strings: Section = .{},
        // OpName, OpMemberName - skip for now.
        // OpModuleProcessed - skip for now.
        /// Annotation instructions (OpDecorate etc).
        annotations: Section = .{},
        /// Type declarations, constants, global variables
        /// Below this section, OpLine and OpNoLine is allowed.
        types_globals_constants: Section = .{},
        // Functions without a body - skip for now.
        /// Regular function definitions.
        functions: Section = .{},
    },

    /// SPIR-V instructions return result-ids. This variable holds the module-wide counter for these.
    next_result_id: u32,

    /// SPIR-V type cache. Note that according to SPIR-V spec section 2.8, Types and Variables, non-pointer
    /// non-aggrerate types (which includes matrices and vectors) must have a _unique_ representation in
    /// the final binary.
    /// Note: Uses ArrayHashMap which is insertion ordered, so that we may refer to other types by index (SpvType.Ref).
    spv_type_cache: SpvTypeCache,

    pub fn init(gpa: *Allocator) Module {
        return .{
            .gpa = gpa,
            .sections = .{},
            .next_result_id = 1, // 0 is an invalid SPIR-V result ID.
            .types = SpvTypeCache.init(gpa),
        };
    }

    pub fn deinit(self: *Module) void {
        self.spv_type_cache.deinit(self.gpa);
        self.* = undefined;
    }

    pub fn allocId(self: *Module) spec.IdResult {
        defer self.next_result_id += 1;
        return .{.id = self.next_result_id};
    }

    pub fn idBound(self: Module) Word {
        return self.next_result_id;
    }

    /// Unconditionally emit a spir-v type into the appropriate section.
    /// Note: If this function is called with a type that is already generated, it may yield an invalid module
    /// as non-pointer non-aggregrate types must me unique!
    /// Note: This function does not attempt to perform any validation on the type.
    /// The type is emitted in a shallow fashion; any child types should already
    /// be emitted at this point.
    pub fn emitType(self: *Module, ty: SpvType) !IdRef {
        const result_id = self.allocId();
        const ref_id = result_id.toRef();
        const types = &self.sections.types_globals_constants;
        const annotations = &self.sections.annotations;
        const result_id_operand = .{.id_result = result_id};

        switch (ty.tag()) {
            .void => try types.emit(self.gpa, .{.OpTypeVoid = &result_id_operand}),
            .bool => try types.emit(self.gpa, .{.OpTypeBool = &result_id_operand}),
            .int =>  try types.emit(self.gpa, .{.OpTypeInt = &.{
                .id_result = result_id,
                .width = ty.payload(.int).width,
                .signedness = switch (ty.payload(.int).signedness) {
                    .unsigned => @as(spec.LiteralInteger, 0),
                    .signed => 1,
                },
            }}),
            .float => try types.emit(self.gpa, .{.OpTypeFloat = &.{
                .id_result = result_id,
                .width = ty.payload(.float).width,
            }}),
            .vector => try types.emit(self.gpa, .{.OpTypeVector = &.{
                .id_result = result_id,
                .component_type = self.getTypeByRef(ty.childType()),
                .component_count = ty.payload(.vector).component_count,
            }}),
            .matrix => try types.emit(self.gpa, .{.OpTypeMatrix = &.{
                .id_result = result_id,
                .column_type = self.getTypeByRef(ty.childType()),
                .column_count = ty.payload(.matrix).column_count,
            }}),
            .image => {
                const info = ty.payload(.image);
                try types.emit(self.gpa, .{.OpTypeImage = &.{
                    .id_result = result_id,
                    .sampled_type = self.getTypeByRef(ty.childType()),
                    .dim = info.dim,
                    .depth = @enumToInt(info.depth),
                    .arrayed = @boolToInt(info.arrayed),
                    .ms = @boolToInt(info.multisampled),
                    .sampled = @enumToInt(info.sampled),
                    .image_format = info.format,
                    .access_qualifier = info.access_qualifier,
                }});
            },
            .sampler => try types.emit(self.gpa, .{.OpTypeSampler = &result_id_operand}),
            .sampled_image => try types.emit(self.gpa, .{.OpTypeSampledImage = &.{
                .id_result = result_id,
                .image_type = self.getTypeByRef(ty.childType()),
            }}),
            .array => {
                const info = ty.payload(.array);
                try types.emit(self.gpa, .{.OpTypeArray = &.{
                    .id_result = result_id,
                    .element_type = self.getTypeByRef(ty.childType()),
                    .length = .{.id = 0}, // TODO: info.length must be emitted as constant!
                }});
                if (info.array_stride != 0) {
                    try annotations.decorate(self.gpa, ref_id, .{.ArrayStride = .{.array_stride = info.array_stride}});
                }
            },
            .runtime_array => {
                const info = ty.payload(.runtime_array);
                try types.emit(self.gpa, .{.OpTypeRuntimeArray = &.{
                    .id_result = result_id,
                    .element_type = self.getTypeByRef(ty.childType()),
                }});
                if (info.array_stride != 0) {
                    try annotations.decorate(self.gpa, ref_id, .{.ArrayStride = .{.array_stride = info.array_stride}});
                }
            },
            .@"struct" => {
                const info = ty.payload(.@"struct");
                try types.emitRaw(self.gpa, .OpTypeStruct, 1 + info.members.len);
                types.writeOperand(IdResult, result_id);
                for (info.members) |member| {
                    types.writeOperand(IdRef, self.getTypeByRef(member.ty));
                }
                try self.decorateStruct(ref_id, info);
            },
            .@"opaque" => try types.emit(self.gpa, .{.OpTypeOpaque = &.{
                .id_result = result_id,
                .literal_string = ty.payload(.@"opaque").name,
            }}),
            .pointer => {
                const info = ty.payload(.pointer);
                try types.emit(self.gpa, .{.OpTypePointer = &.{
                    .id_result = result_id,
                    .storage_class = info.storage_class,
                    .type = self.getTypeByRef(ty.childType()),
                }});
                if (info.array_stride != 0) {
                    try annotations.decorate(self.gpa, ref_id, .{.ArrayStride = .{.array_stride = info.array_stride}});
                }
                if (info.alignment) |alignment| {
                    try annotations.decorate(self.gpa, ref_id, .{.Alignment = .{.alignment = alignment}});
                }
                if (info.max_byte_offset) |max_byte_offset| {
                    try annotations.decorate(self.gpa, ref_id, .{.MaxByteOffset = .{.max_byte_offset = max_byte_offset}});
                }
            },
            .function => {
                const info = ty.payload(.function);
                try types.emitRaw(self.gpa, .OpTypeFunction, 2 + info.parameters.len);
                types.writeOperand(IdResult, result_id);
                types.writeOperand(IdRef, self.getTypeByRef(info.return_type));
                for (info.parameters) |parameter_type| {
                    types.writeOperand(IdRef, self.getTypeByRef(parameter_type));
                }
            },
            .event => try types.emit(self.gpa, .{.OpTypeEvent = &result_id_operand}),
            .device_event => try types.emit(self.gpa, .{.OpTypeDeviceEvent = &result_id_operand}),
            .reserve_id => try types.emit(self.gpa, .{.OpTypeReserveId = &result_id_operand}),
            .queue => try types.emit(self.gpa, .{.OpTypeQueue = &result_id_operand}),
            .pipe => try types.emit(self.gpa, .{.OpTypePipe = &.{
                .id_result = result_id,
                .qualifier = ty.payload(.pipe).qualifier,
            }}),
            .pipe_storage => try types.emit(self.gpa, .{.OpTypePipeStorage = &result_id_operand}),
            .named_barrier => try types.emit(self.gpa, .{.OpTypeNamedBarrier = &result_id_operand}),
        }

        return ref_id;
    }

    fn decorateStruct(self: *Module, target: IdRef, info: *SpvType.Payload.Struct) !void {
        const annotations = &self.sections.annotations;

        // Decorations for the struct type itself.
        if (info.decorations.block) {
            try annotations.decorate(self.gpa, target, .Block);
        }
        if (info.decorations.buffer_block) {
            try annotations.decorate(self.gpa, target, .BufferBlock);
        }
        if (info.decorations.glsl_shared) {
            try annotations.decorate(self.gpa, target, .GLSLShared);
        }
        if (info.decorations.glsl_packed) {
            try annotations.decorate(self.gpa, target, .GLSLPacked);
        }
        if (info.decorations.c_packed) {
            try annotations.decorate(self.gpa, target, .CPacked);
        }

        // Decorations for the struct members.
        const extra = info.member_decoration_extra;
        var extra_i: u32 = 0;
        for (info.members) |member, i| {
            const d = member.decorations;
            const index = @intCast(Word, i);
            switch (d.matrix_layout) {
                .row_major => try annotations.decorateMember(self.gpa, target, index, .RowMajor),
                .col_major => try annotations.decorateMember(self.gpa, target, index, .ColMajor),
                .none => {},
            }
            if (d.matrix_layout != .none) {
                try annotations.decorateMember(self.gpa, target, index, .{
                    .MatrixStride = .{.matrix_stride = extra[extra_i]},
                });
                extra_i += 1;
            }

            if (d.no_perspective)
                try annotations.decorateMember(self.gpa, target, index, .NoPerspective);
            if (d.flat)
                try annotations.decorateMember(self.gpa, target, index, .Flat);
            if (d.patch)
                try annotations.decorateMember(self.gpa, target, index, .Patch);
            if (d.centroid)
                try annotations.decorateMember(self.gpa, target, index, .Centroid);
            if (d.sample)
                try annotations.decorateMember(self.gpa, target, index, .Sample);
            if (d.invariant)
                try annotations.decorateMember(self.gpa, target, index, .Invariant);
            if (d.@"volatile")
                try annotations.decorateMember(self.gpa, target, index, .Volatile);
            if (d.coherent)
                try annotations.decorateMember(self.gpa, target, index, .Coherent);
            if (d.non_writable)
                try annotations.decorateMember(self.gpa, target, index, .NonWritable);
            if (d.non_readable)
                try annotations.decorateMember(self.gpa, target, index, .NonReadable);

            if (d.builtin) {
                try annotations.decorateMember(self.gpa, target, index, .{
                    .BuiltIn = .{.built_in = @intToEnum(spec.BuiltIn, extra[extra_i])},
                });
                extra_i += 1;
            }
            if (d.stream) {
                try annotations.decorateMember(self.gpa, target, index, .{
                    .Stream = .{.stream_number = extra[extra_i]},
                });
                extra_i += 1;
            }
            if (d.location) {
                try annotations.decorateMember(self.gpa, target, index, .{
                    .Location = .{.location = extra[extra_i]},
                });
                extra_i += 1;
            }
            if (d.component) {
                try annotations.decorateMember(self.gpa, target, index, .{
                    .Component = .{.component = extra[extra_i]},
                });
                extra_i += 1;
            }
            if (d.xfb_buffer) {
                try annotations.decorateMember(self.gpa, target, index, .{
                    .XfbBuffer = .{.xfb_buffer_number = extra[extra_i]},
                });
                extra_i += 1;
            }
            if (d.xfb_stride) {
                try annotations.decorateMember(self.gpa, target, index, .{
                    .XfbStride = .{.xfb_stride = extra[extra_i]},
                });
                extra_i += 1;
            }
            if (d.user_semantic) {
                const len = extra[extra_i];
                extra_i += 1;
                const semantic = @ptrCast([*]const u8, &extra[extra_i])[0..len];
                try annotations.decorateMember(self.gpa, target, index, .{
                    .UserSemantic = .{.semantic = semantic},
                });
                extra_i += std.math.divCeil(u32, extra_i, @sizeOf(u32)) catch unreachable;
            }
        }
    }

    /// Get the result-id of a particular type, by reference. Asserts type_ref is valid.
    pub fn getTypeByRef(self: Module, type_ref: SpvType.Ref) IdRef {
        return self.spv_type_cache.values()[type_ref];
    }
};
