// SPDX-License-Identifier: MIT
// Copyright (c) 2015-2021 Zig Contributors
// This file is part of [zig](https://ziglang.org/), which is MIT licensed.
// The MIT license requires this copyright notice to be included in all copies
// and substantial portions of the software.
const std = @import("../std.zig");
const CpuFeature = std.Target.Cpu.Feature;
const CpuModel = std.Target.Cpu.Model;

pub const Feature = enum {
    // SPIR-V core versions
    v1_1,
    v1_2,
    v1_3,
    v1_4,
    v1_5,

    // Extensions
    SPV_AMD_shader_explicit_vertex_parameter,
    SPV_AMD_shader_trinary_minmax,
    SPV_AMD_gcn_shader,
    SPV_KHR_shader_ballot,
    SPV_AMD_shader_ballot,
    SPV_AMD_gpu_shader_half_float,
    SPV_KHR_shader_draw_parameters,
    SPV_KHR_subgroup_vote,
    SPV_KHR_16bit_storage,
    SPV_KHR_device_group,
    SPV_KHR_multiview,
    SPV_NVX_multiview_per_view_attributes,
    SPV_NV_viewport_array2,
    SPV_NV_stereo_view_rendering,
    SPV_NV_sample_mask_override_coverage,
    SPV_NV_geometry_shader_passthrough,
    SPV_AMD_texture_gather_bias_lod,
    SPV_KHR_storage_buffer_storage_class,
    SPV_KHR_variable_pointers,
    SPV_AMD_gpu_shader_int16,
    SPV_KHR_post_depth_coverage,
    SPV_KHR_shader_atomic_counter_ops,
    SPV_EXT_shader_stencil_export,
    SPV_EXT_shader_viewport_index_layer,
    SPV_AMD_shader_image_load_store_lod,
    SPV_AMD_shader_fragment_mask,
    SPV_EXT_fragment_fully_covered,
    SPV_AMD_gpu_shader_half_float_fetch,
    SPV_GOOGLE_decorate_string,
    SPV_GOOGLE_hlsl_functionality1,
    SPV_NV_shader_subgroup_partitioned,
    SPV_EXT_descriptor_indexing,
    SPV_KHR_8bit_storage,
    SPV_KHR_vulkan_memory_model,
    SPV_NV_ray_tracing,
    SPV_NV_compute_shader_derivatives,
    SPV_NV_fragment_shader_barycentric,
    SPV_NV_mesh_shader,
    SPV_NV_shader_image_footprint,
    SPV_NV_shading_rate,
    SPV_INTEL_subgroups,
    SPV_INTEL_media_block_io,
    SPV_INTEL_device_side_avc_motion_estimation,
    SPV_EXT_fragment_invocation_density,
    SPV_KHR_no_integer_wrap_decoration,
    SPV_KHR_float_controls,
    SPV_EXT_physical_storage_buffer,
    SPV_INTEL_fpga_memory_attributes,
    SPV_NV_cooperative_matrix,
    SPV_INTEL_shader_integer_functions2,
    SPV_INTEL_fpga_loop_controls,
    SPV_EXT_fragment_shader_interlock,
    SPV_NV_shader_sm_builtins,
    SPV_KHR_shader_clock,
    SPV_INTEL_unstructured_loop_controls,
    SPV_EXT_demote_to_helper_invocation,
    SPV_INTEL_fpga_reg,
    SPV_INTEL_blocking_pipes,
    SPV_KHR_physical_storage_buffer,
    SPV_KHR_fragment_shading_rate,
    SPV_KHR_ray_query,
    SPV_KHR_ray_tracing,
    SPV_EXT_shader_image_int64,
    SPV_INTEL_function_pointers,
    SPV_INTEL_kernel_attributes,
    SPV_EXT_shader_atomic_float_add,

    // Capabilities
    Matrix,
    Shader,
    Geometry,
    Tessellation,
    Addresses,
    Linkage,
    Kernel,
    Vector16,
    Float16Buffer,
    Float16,
    Float64,
    Int64,
    Int64Atomics,
    ImageBasic,
    ImageReadWrite,
    ImageMipmap,
    Pipes,
    Groups,
    DeviceEnqueue,
    LiteralSampler,
    AtomicStorage,
    Int16,
    TessellationPointSize,
    GeometryPointSize,
    ImageGatherExtended,
    StorageImageMultisample,
    UniformBufferArrayDynamicIndexing,
    SampledImageArrayDynamicIndexing,
    StorageBufferArrayDynamicIndexing,
    StorageImageArrayDynamicIndexing,
    ClipDistance,
    CullDistance,
    ImageCubeArray,
    SampleRateShading,
    ImageRect,
    SampledRect,
    GenericPointer,
    Int8,
    InputAttachment,
    SparseResidency,
    MinLod,
    Sampled1D,
    Image1D,
    SampledCubeArray,
    SampledBuffer,
    ImageBuffer,
    ImageMSArray,
    StorageImageExtendedFormats,
    ImageQuery,
    DerivativeControl,
    InterpolationFunction,
    TransformFeedback,
    GeometryStreams,
    StorageImageReadWithoutFormat,
    StorageImageWriteWithoutFormat,
    MultiViewport,
    SubgroupDispatch,
    NamedBarrier,
    PipeStorage,
    GroupNonUniform,
    GroupNonUniformVote,
    GroupNonUniformArithmetic,
    GroupNonUniformBallot,
    GroupNonUniformShuffle,
    GroupNonUniformShuffleRelative,
    GroupNonUniformClustered,
    GroupNonUniformQuad,
    ShaderLayer,
    ShaderViewportIndex,
    FragmentShadingRateKHR,
    SubgroupBallotKHR,
    DrawParameters,
    SubgroupVoteKHR,
    StorageBuffer16BitAccess,
    StorageUniformBufferBlock16,
    UniformAndStorageBuffer16BitAccess,
    StorageUniform16,
    StoragePushConstant16,
    StorageInputOutput16,
    DeviceGroup,
    MultiView,
    VariablePointersStorageBuffer,
    VariablePointers,
    AtomicStorageOps,
    SampleMaskPostDepthCoverage,
    StorageBuffer8BitAccess,
    UniformAndStorageBuffer8BitAccess,
    StoragePushConstant8,
    DenormPreserve,
    DenormFlushToZero,
    SignedZeroInfNanPreserve,
    RoundingModeRTE,
    RoundingModeRTZ,
    RayQueryProvisionalKHR,
    RayQueryKHR,
    RayTraversalPrimitiveCullingKHR,
    RayTracingKHR,
    Float16ImageAMD,
    ImageGatherBiasLodAMD,
    FragmentMaskAMD,
    StencilExportEXT,
    ImageReadWriteLodAMD,
    Int64ImageEXT,
    ShaderClockKHR,
    SampleMaskOverrideCoverageNV,
    GeometryShaderPassthroughNV,
    ShaderViewportIndexLayerEXT,
    ShaderViewportIndexLayerNV,
    ShaderViewportMaskNV,
    ShaderStereoViewNV,
    PerViewAttributesNV,
    FragmentFullyCoveredEXT,
    MeshShadingNV,
    ImageFootprintNV,
    FragmentBarycentricNV,
    ComputeDerivativeGroupQuadsNV,
    FragmentDensityEXT,
    ShadingRateNV,
    GroupNonUniformPartitionedNV,
    ShaderNonUniform,
    ShaderNonUniformEXT,
    RuntimeDescriptorArray,
    RuntimeDescriptorArrayEXT,
    InputAttachmentArrayDynamicIndexing,
    InputAttachmentArrayDynamicIndexingEXT,
    UniformTexelBufferArrayDynamicIndexing,
    UniformTexelBufferArrayDynamicIndexingEXT,
    StorageTexelBufferArrayDynamicIndexing,
    StorageTexelBufferArrayDynamicIndexingEXT,
    UniformBufferArrayNonUniformIndexing,
    UniformBufferArrayNonUniformIndexingEXT,
    SampledImageArrayNonUniformIndexing,
    SampledImageArrayNonUniformIndexingEXT,
    StorageBufferArrayNonUniformIndexing,
    StorageBufferArrayNonUniformIndexingEXT,
    StorageImageArrayNonUniformIndexing,
    StorageImageArrayNonUniformIndexingEXT,
    InputAttachmentArrayNonUniformIndexing,
    InputAttachmentArrayNonUniformIndexingEXT,
    UniformTexelBufferArrayNonUniformIndexing,
    UniformTexelBufferArrayNonUniformIndexingEXT,
    StorageTexelBufferArrayNonUniformIndexing,
    StorageTexelBufferArrayNonUniformIndexingEXT,
    RayTracingNV,
    VulkanMemoryModel,
    VulkanMemoryModelKHR,
    VulkanMemoryModelDeviceScope,
    VulkanMemoryModelDeviceScopeKHR,
    PhysicalStorageBufferAddresses,
    PhysicalStorageBufferAddressesEXT,
    ComputeDerivativeGroupLinearNV,
    RayTracingProvisionalKHR,
    CooperativeMatrixNV,
    FragmentShaderSampleInterlockEXT,
    FragmentShaderShadingRateInterlockEXT,
    ShaderSMBuiltinsNV,
    FragmentShaderPixelInterlockEXT,
    DemoteToHelperInvocationEXT,
    SubgroupShuffleINTEL,
    SubgroupBufferBlockIOINTEL,
    SubgroupImageBlockIOINTEL,
    SubgroupImageMediaBlockIOINTEL,
    IntegerFunctions2INTEL,
    FunctionPointersINTEL,
    IndirectReferencesINTEL,
    SubgroupAvcMotionEstimationINTEL,
    SubgroupAvcMotionEstimationIntraINTEL,
    SubgroupAvcMotionEstimationChromaINTEL,
    FPGAMemoryAttributesINTEL,
    UnstructuredLoopControlsINTEL,
    FPGALoopControlsINTEL,
    KernelAttributesINTEL,
    FPGAKernelAttributesINTEL,
    BlockingPipesINTEL,
    FPGARegINTEL,
    AtomicFloat32AddEXT,
    AtomicFloat64AddEXT,
};

pub usingnamespace CpuFeature.feature_set_fns(Feature);

pub const all_features = blk: {
    const len = @typeInfo(Feature).Enum.fields.len;
    std.debug.assert(len <= CpuFeature.Set.needed_bit_count);
    var result: [len]CpuFeature = undefined;
    result[@enumToInt(Feature.v1_1)] = .{
        .llvm_name = null,
        .description = "SPIR-V core specification 1.1",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.v1_2)] = .{
        .llvm_name = null,
        .description = "SPIR-V core specification 1.2",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.v1_3)] = .{
        .llvm_name = null,
        .description = "SPIR-V core specification 1.3",
        .dependencies = featureSet(&[_]Feature{
            .v1_2,
        }),
    };
    result[@enumToInt(Feature.v1_4)] = .{
        .llvm_name = null,
        .description = "SPIR-V core specification 1.4",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
        }),
    };
    result[@enumToInt(Feature.v1_5)] = .{
        .llvm_name = null,
        .description = "SPIR-V core specification 1.5",
        .dependencies = featureSet(&[_]Feature{
            .v1_4,
        }),
    };
    result[@enumToInt(Feature.SPV_AMD_shader_explicit_vertex_parameter)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_AMD_shader_trinary_minmax)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_AMD_gcn_shader)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_KHR_shader_ballot)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_AMD_shader_ballot)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_AMD_gpu_shader_half_float)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_KHR_shader_draw_parameters)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_subgroup_vote)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_16bit_storage)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_device_group)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_multiview)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NVX_multiview_per_view_attributes)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_viewport_array2)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_stereo_view_rendering)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_viewport_array2,
        }),
    };
    result[@enumToInt(Feature.SPV_NV_sample_mask_override_coverage)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_geometry_shader_passthrough)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_AMD_texture_gather_bias_lod)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_KHR_storage_buffer_storage_class)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_variable_pointers)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_AMD_gpu_shader_int16)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_KHR_post_depth_coverage)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_shader_atomic_counter_ops)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_EXT_shader_stencil_export)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_EXT_shader_viewport_index_layer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_AMD_shader_image_load_store_lod)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
        }),
    };
    result[@enumToInt(Feature.SPV_AMD_shader_fragment_mask)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_2, // Reports being written against the 1.12.1 spec, but that doesn't exist yet
        }),
    };
    result[@enumToInt(Feature.SPV_EXT_fragment_fully_covered)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_AMD_gpu_shader_half_float_fetch)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_2,
        }),
    };
    result[@enumToInt(Feature.SPV_GOOGLE_decorate_string)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_GOOGLE_hlsl_functionality1)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_shader_subgroup_partitioned)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
        }),
    };
    result[@enumToInt(Feature.SPV_EXT_descriptor_indexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_8bit_storage)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_storage_buffer_storage_class,
        }),
    };
    result[@enumToInt(Feature.SPV_KHR_vulkan_memory_model)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
        }),
    };
    result[@enumToInt(Feature.SPV_NV_ray_tracing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_compute_shader_derivatives)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_fragment_shader_barycentric)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_mesh_shader)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_shader_image_footprint)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_shading_rate)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_INTEL_subgroups)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_INTEL_media_block_io)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_INTEL_device_side_avc_motion_estimation)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_EXT_fragment_invocation_density)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_no_integer_wrap_decoration)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_float_controls)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_EXT_physical_storage_buffer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
        }),
    };
    result[@enumToInt(Feature.SPV_INTEL_fpga_memory_attributes)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_cooperative_matrix)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_vulkan_memory_model,
        }),
    };
    result[@enumToInt(Feature.SPV_INTEL_shader_integer_functions2)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_INTEL_fpga_loop_controls)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_EXT_fragment_shader_interlock)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_NV_shader_sm_builtins)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
        }),
    };
    result[@enumToInt(Feature.SPV_KHR_shader_clock)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_INTEL_unstructured_loop_controls)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_EXT_demote_to_helper_invocation)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_INTEL_fpga_reg)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_INTEL_blocking_pipes)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_physical_storage_buffer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
        }),
    };
    result[@enumToInt(Feature.SPV_KHR_fragment_shading_rate)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_ray_query)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_KHR_ray_tracing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_4,
        }),
    };
    result[@enumToInt(Feature.SPV_EXT_shader_image_int64)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
        }),
    };
    result[@enumToInt(Feature.SPV_INTEL_function_pointers)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_INTEL_kernel_attributes)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.SPV_EXT_shader_atomic_float_add)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Matrix)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Shader)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Matrix,
        }),
    };
    result[@enumToInt(Feature.Geometry)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.Tessellation)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.Addresses)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Linkage)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Kernel)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Vector16)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Kernel,
        }),
    };
    result[@enumToInt(Feature.Float16Buffer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Kernel,
        }),
    };
    result[@enumToInt(Feature.Float16)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Float64)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Int64)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Int64Atomics)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Int64,
        }),
    };
    result[@enumToInt(Feature.ImageBasic)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Kernel,
        }),
    };
    result[@enumToInt(Feature.ImageReadWrite)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .ImageBasic,
        }),
    };
    result[@enumToInt(Feature.ImageMipmap)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .ImageBasic,
        }),
    };
    result[@enumToInt(Feature.Pipes)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Kernel,
        }),
    };
    result[@enumToInt(Feature.Groups)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_AMD_shader_ballot,
        }),
    };
    result[@enumToInt(Feature.DeviceEnqueue)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Kernel,
        }),
    };
    result[@enumToInt(Feature.LiteralSampler)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Kernel,
        }),
    };
    result[@enumToInt(Feature.AtomicStorage)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.Int16)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.TessellationPointSize)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Tessellation,
        }),
    };
    result[@enumToInt(Feature.GeometryPointSize)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Geometry,
        }),
    };
    result[@enumToInt(Feature.ImageGatherExtended)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.StorageImageMultisample)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.UniformBufferArrayDynamicIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.SampledImageArrayDynamicIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.StorageBufferArrayDynamicIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.StorageImageArrayDynamicIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ClipDistance)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.CullDistance)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ImageCubeArray)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SampledCubeArray,
        }),
    };
    result[@enumToInt(Feature.SampleRateShading)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ImageRect)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SampledRect,
        }),
    };
    result[@enumToInt(Feature.SampledRect)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.GenericPointer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Addresses,
        }),
    };
    result[@enumToInt(Feature.Int8)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.InputAttachment)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.SparseResidency)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.MinLod)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.Sampled1D)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.Image1D)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Sampled1D,
        }),
    };
    result[@enumToInt(Feature.SampledCubeArray)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.SampledBuffer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{}),
    };
    result[@enumToInt(Feature.ImageBuffer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SampledBuffer,
        }),
    };
    result[@enumToInt(Feature.ImageMSArray)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.StorageImageExtendedFormats)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ImageQuery)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.DerivativeControl)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.InterpolationFunction)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.TransformFeedback)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.GeometryStreams)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Geometry,
        }),
    };
    result[@enumToInt(Feature.StorageImageReadWithoutFormat)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.StorageImageWriteWithoutFormat)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Shader,
        }),
    };
    result[@enumToInt(Feature.MultiViewport)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .Geometry,
        }),
    };
    result[@enumToInt(Feature.SubgroupDispatch)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
            .DeviceEnqueue,
        }),
    };
    result[@enumToInt(Feature.NamedBarrier)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
            .Kernel,
        }),
    };
    result[@enumToInt(Feature.PipeStorage)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_1,
            .Pipes,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniform)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniformVote)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .GroupNonUniform,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniformArithmetic)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .GroupNonUniform,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniformBallot)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .GroupNonUniform,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniformShuffle)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .GroupNonUniform,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniformShuffleRelative)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .GroupNonUniform,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniformClustered)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .GroupNonUniform,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniformQuad)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .GroupNonUniform,
        }),
    };
    result[@enumToInt(Feature.ShaderLayer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
        }),
    };
    result[@enumToInt(Feature.ShaderViewportIndex)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
        }),
    };
    result[@enumToInt(Feature.FragmentShadingRateKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_fragment_shading_rate,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.SubgroupBallotKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_shader_ballot,
        }),
    };
    result[@enumToInt(Feature.DrawParameters)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_shader_draw_parameters,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.SubgroupVoteKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_subgroup_vote,
        }),
    };
    result[@enumToInt(Feature.StorageBuffer16BitAccess)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_16bit_storage,
        }),
    };
    result[@enumToInt(Feature.StorageUniformBufferBlock16)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_16bit_storage,
        }),
    };
    result[@enumToInt(Feature.UniformAndStorageBuffer16BitAccess)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_16bit_storage,
            .StorageBuffer16BitAccess,
            .StorageUniformBufferBlock16,
        }),
    };
    result[@enumToInt(Feature.StorageUniform16)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_16bit_storage,
            .StorageBuffer16BitAccess,
            .StorageUniformBufferBlock16,
        }),
    };
    result[@enumToInt(Feature.StoragePushConstant16)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_16bit_storage,
        }),
    };
    result[@enumToInt(Feature.StorageInputOutput16)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_16bit_storage,
        }),
    };
    result[@enumToInt(Feature.DeviceGroup)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_device_group,
        }),
    };
    result[@enumToInt(Feature.MultiView)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_multiview,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.VariablePointersStorageBuffer)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_variable_pointers,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.VariablePointers)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_3,
            .SPV_KHR_variable_pointers,
            .VariablePointersStorageBuffer,
        }),
    };
    result[@enumToInt(Feature.AtomicStorageOps)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_shader_atomic_counter_ops,
        }),
    };
    result[@enumToInt(Feature.SampleMaskPostDepthCoverage)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_post_depth_coverage,
        }),
    };
    result[@enumToInt(Feature.StorageBuffer8BitAccess)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_KHR_8bit_storage,
        }),
    };
    result[@enumToInt(Feature.UniformAndStorageBuffer8BitAccess)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_KHR_8bit_storage,
            .StorageBuffer8BitAccess,
        }),
    };
    result[@enumToInt(Feature.StoragePushConstant8)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_KHR_8bit_storage,
        }),
    };
    result[@enumToInt(Feature.DenormPreserve)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_4,
            .SPV_KHR_float_controls,
        }),
    };
    result[@enumToInt(Feature.DenormFlushToZero)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_4,
            .SPV_KHR_float_controls,
        }),
    };
    result[@enumToInt(Feature.SignedZeroInfNanPreserve)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_4,
            .SPV_KHR_float_controls,
        }),
    };
    result[@enumToInt(Feature.RoundingModeRTE)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_4,
            .SPV_KHR_float_controls,
        }),
    };
    result[@enumToInt(Feature.RoundingModeRTZ)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_4,
            .SPV_KHR_float_controls,
        }),
    };
    result[@enumToInt(Feature.RayQueryProvisionalKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_ray_query,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.RayQueryKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_ray_query,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.RayTraversalPrimitiveCullingKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_ray_query,
            .SPV_KHR_ray_tracing,
            .RayQueryKHR,
            .RayTracingKHR,
        }),
    };
    result[@enumToInt(Feature.RayTracingKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_ray_tracing,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.Float16ImageAMD)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_AMD_gpu_shader_half_float_fetch,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ImageGatherBiasLodAMD)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_AMD_texture_gather_bias_lod,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.FragmentMaskAMD)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_AMD_shader_fragment_mask,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.StencilExportEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_shader_stencil_export,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ImageReadWriteLodAMD)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_AMD_shader_image_load_store_lod,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.Int64ImageEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_shader_image_int64,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ShaderClockKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_shader_clock,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.SampleMaskOverrideCoverageNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_sample_mask_override_coverage,
            .SampleRateShading,
        }),
    };
    result[@enumToInt(Feature.GeometryShaderPassthroughNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_geometry_shader_passthrough,
            .Geometry,
        }),
    };
    result[@enumToInt(Feature.ShaderViewportIndexLayerEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_shader_viewport_index_layer,
            .MultiViewport,
        }),
    };
    result[@enumToInt(Feature.ShaderViewportIndexLayerNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_viewport_array2,
            .MultiViewport,
        }),
    };
    result[@enumToInt(Feature.ShaderViewportMaskNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_viewport_array2,
            .ShaderViewportIndexLayerNV,
        }),
    };
    result[@enumToInt(Feature.ShaderStereoViewNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_stereo_view_rendering,
            .ShaderViewportMaskNV,
        }),
    };
    result[@enumToInt(Feature.PerViewAttributesNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NVX_multiview_per_view_attributes,
            .MultiView,
        }),
    };
    result[@enumToInt(Feature.FragmentFullyCoveredEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_fragment_fully_covered,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.MeshShadingNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_mesh_shader,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ImageFootprintNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_shader_image_footprint,
        }),
    };
    result[@enumToInt(Feature.FragmentBarycentricNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_fragment_shader_barycentric,
        }),
    };
    result[@enumToInt(Feature.ComputeDerivativeGroupQuadsNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_compute_shader_derivatives,
        }),
    };
    result[@enumToInt(Feature.FragmentDensityEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_fragment_invocation_density,
            .SPV_NV_shading_rate,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ShadingRateNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_shading_rate,
            .SPV_EXT_fragment_invocation_density,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.GroupNonUniformPartitionedNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_shader_subgroup_partitioned,
        }),
    };
    result[@enumToInt(Feature.ShaderNonUniform)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ShaderNonUniformEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.RuntimeDescriptorArray)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.RuntimeDescriptorArrayEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.InputAttachmentArrayDynamicIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .InputAttachment,
        }),
    };
    result[@enumToInt(Feature.InputAttachmentArrayDynamicIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .InputAttachment,
        }),
    };
    result[@enumToInt(Feature.UniformTexelBufferArrayDynamicIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SampledBuffer,
        }),
    };
    result[@enumToInt(Feature.UniformTexelBufferArrayDynamicIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .SampledBuffer,
        }),
    };
    result[@enumToInt(Feature.StorageTexelBufferArrayDynamicIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .ImageBuffer,
        }),
    };
    result[@enumToInt(Feature.StorageTexelBufferArrayDynamicIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .ImageBuffer,
        }),
    };
    result[@enumToInt(Feature.UniformBufferArrayNonUniformIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.UniformBufferArrayNonUniformIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.SampledImageArrayNonUniformIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.SampledImageArrayNonUniformIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.StorageBufferArrayNonUniformIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.StorageBufferArrayNonUniformIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.StorageImageArrayNonUniformIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.StorageImageArrayNonUniformIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.InputAttachmentArrayNonUniformIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .InputAttachment,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.InputAttachmentArrayNonUniformIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .InputAttachment,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.UniformTexelBufferArrayNonUniformIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SampledBuffer,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.UniformTexelBufferArrayNonUniformIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .SampledBuffer,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.StorageTexelBufferArrayNonUniformIndexing)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .ImageBuffer,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.StorageTexelBufferArrayNonUniformIndexingEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_descriptor_indexing,
            .ImageBuffer,
            .ShaderNonUniform,
        }),
    };
    result[@enumToInt(Feature.RayTracingNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_ray_tracing,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.VulkanMemoryModel)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
        }),
    };
    result[@enumToInt(Feature.VulkanMemoryModelKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_KHR_vulkan_memory_model,
        }),
    };
    result[@enumToInt(Feature.VulkanMemoryModelDeviceScope)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
        }),
    };
    result[@enumToInt(Feature.VulkanMemoryModelDeviceScopeKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_KHR_vulkan_memory_model,
        }),
    };
    result[@enumToInt(Feature.PhysicalStorageBufferAddresses)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_physical_storage_buffer,
            .SPV_KHR_physical_storage_buffer,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.PhysicalStorageBufferAddressesEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .v1_5,
            .SPV_EXT_physical_storage_buffer,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ComputeDerivativeGroupLinearNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_compute_shader_derivatives,
        }),
    };
    result[@enumToInt(Feature.RayTracingProvisionalKHR)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_KHR_ray_tracing,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.CooperativeMatrixNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_cooperative_matrix,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.FragmentShaderSampleInterlockEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_fragment_shader_interlock,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.FragmentShaderShadingRateInterlockEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_fragment_shader_interlock,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.ShaderSMBuiltinsNV)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_NV_shader_sm_builtins,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.FragmentShaderPixelInterlockEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_fragment_shader_interlock,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.DemoteToHelperInvocationEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_demote_to_helper_invocation,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.SubgroupShuffleINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_subgroups,
        }),
    };
    result[@enumToInt(Feature.SubgroupBufferBlockIOINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_subgroups,
        }),
    };
    result[@enumToInt(Feature.SubgroupImageBlockIOINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_subgroups,
        }),
    };
    result[@enumToInt(Feature.SubgroupImageMediaBlockIOINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_media_block_io,
        }),
    };
    result[@enumToInt(Feature.IntegerFunctions2INTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_shader_integer_functions2,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.FunctionPointersINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_function_pointers,
        }),
    };
    result[@enumToInt(Feature.IndirectReferencesINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_function_pointers,
        }),
    };
    result[@enumToInt(Feature.SubgroupAvcMotionEstimationINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_device_side_avc_motion_estimation,
        }),
    };
    result[@enumToInt(Feature.SubgroupAvcMotionEstimationIntraINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_device_side_avc_motion_estimation,
        }),
    };
    result[@enumToInt(Feature.SubgroupAvcMotionEstimationChromaINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_device_side_avc_motion_estimation,
        }),
    };
    result[@enumToInt(Feature.FPGAMemoryAttributesINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_fpga_memory_attributes,
        }),
    };
    result[@enumToInt(Feature.UnstructuredLoopControlsINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_unstructured_loop_controls,
        }),
    };
    result[@enumToInt(Feature.FPGALoopControlsINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_fpga_loop_controls,
        }),
    };
    result[@enumToInt(Feature.KernelAttributesINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_kernel_attributes,
        }),
    };
    result[@enumToInt(Feature.FPGAKernelAttributesINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_kernel_attributes,
        }),
    };
    result[@enumToInt(Feature.BlockingPipesINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_blocking_pipes,
        }),
    };
    result[@enumToInt(Feature.FPGARegINTEL)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_INTEL_fpga_reg,
        }),
    };
    result[@enumToInt(Feature.AtomicFloat32AddEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_shader_atomic_float_add,
            .Shader,
        }),
    };
    result[@enumToInt(Feature.AtomicFloat64AddEXT)] = .{
        .llvm_name = null,
        .description = "",
        .dependencies = featureSet(&[_]Feature{
            .SPV_EXT_shader_atomic_float_add,
            .Shader,
        }),
    };
    break :blk result;
};
