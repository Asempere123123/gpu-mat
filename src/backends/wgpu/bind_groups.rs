use once_cell::sync::Lazy;
use std::num::NonZeroU64;
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferAddress, ComputePipeline, PipelineLayout,
    ShaderModule,
};

use super::globals::DEVICE_QUEUE;

static ABC_F32_BIND_GROUP_LAYOUT: Lazy<BindGroupLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: Some(unsafe {
                            NonZeroU64::new(core::mem::size_of::<
                                super::tensor_info::UniformTensorInfo,
                            >() as BufferAddress)
                            .unwrap_unchecked()
                        }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(4).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(4).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(unsafe { NonZeroU64::new(4).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        })
});

static ABC_F64_BIND_GROUP_LAYOUT: Lazy<BindGroupLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: Some(unsafe {
                            NonZeroU64::new(core::mem::size_of::<
                                super::tensor_info::UniformTensorInfo,
                            >() as BufferAddress)
                            .unwrap_unchecked()
                        }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(8).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(8).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(unsafe { NonZeroU64::new(8).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        })
});

static ABC_F16_BIND_GROUP_LAYOUT: Lazy<BindGroupLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: Some(unsafe {
                            NonZeroU64::new(core::mem::size_of::<
                                super::tensor_info::UniformTensorInfo,
                            >() as BufferAddress)
                            .unwrap_unchecked()
                        }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(2).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(2).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(unsafe { NonZeroU64::new(2).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        })
});

static ABC_F32_PIPELINE_LAYOUT: Lazy<PipelineLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&ABC_F32_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
});

static ABC_F64_PIPELINE_LAYOUT: Lazy<PipelineLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&ABC_F64_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
});

static ABC_F16_PIPELINE_LAYOUT: Lazy<PipelineLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&ABC_F16_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
});

static AB_F32_BIND_GROUP_LAYOUT: Lazy<BindGroupLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: Some(unsafe {
                            NonZeroU64::new(core::mem::size_of::<
                                super::tensor_info::UniformTensorInfo,
                            >() as BufferAddress)
                            .unwrap_unchecked()
                        }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(unsafe { NonZeroU64::new(4).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(4).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        })
});

static AB_F32_PIPELINE_LAYOUT: Lazy<PipelineLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&AB_F32_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
});

static AB_F64_BIND_GROUP_LAYOUT: Lazy<BindGroupLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: Some(unsafe {
                            NonZeroU64::new(core::mem::size_of::<
                                super::tensor_info::UniformTensorInfo,
                            >() as BufferAddress)
                            .unwrap_unchecked()
                        }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(unsafe { NonZeroU64::new(8).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(8).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        })
});

static AB_F64_PIPELINE_LAYOUT: Lazy<PipelineLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&AB_F64_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
});

static AB_F16_BIND_GROUP_LAYOUT: Lazy<BindGroupLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: Some(unsafe {
                            NonZeroU64::new(core::mem::size_of::<
                                super::tensor_info::UniformTensorInfo,
                            >() as BufferAddress)
                            .unwrap_unchecked()
                        }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(unsafe { NonZeroU64::new(2).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(unsafe { NonZeroU64::new(2).unwrap_unchecked() }),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        })
});

static AB_F16_PIPELINE_LAYOUT: Lazy<PipelineLayout> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&AB_F16_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
});

static ADD_F32_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/add_f32.wgsl"))
});

pub static ADD_F32_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&ABC_F32_PIPELINE_LAYOUT),
            module: &ADD_F32_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static ADD_F64_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/add_f64.wgsl"))
});

pub static ADD_F64_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&ABC_F64_PIPELINE_LAYOUT),
            module: &ADD_F64_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static ADD_F16_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/add_f16.wgsl"))
});

pub static ADD_F16_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&ABC_F16_PIPELINE_LAYOUT),
            module: &ADD_F16_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static MUL_F32_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/mul_f32.wgsl"))
});

pub static MUL_F32_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&ABC_F32_PIPELINE_LAYOUT),
            module: &MUL_F32_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static MUL_F16_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/mul_f16.wgsl"))
});

pub static MUL_F16_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&ABC_F16_PIPELINE_LAYOUT),
            module: &MUL_F16_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static MUL_F64_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/add_f64.wgsl"))
});

pub static MUL_F64_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&ABC_F64_PIPELINE_LAYOUT),
            module: &MUL_F64_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static INCREMENT_F32_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/increment_f32.wgsl"))
});

pub static INCREMENT_F32_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&AB_F32_PIPELINE_LAYOUT),
            module: &INCREMENT_F32_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static INCREMENT_F64_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/increment_f64.wgsl"))
});

pub static INCREMENT_F64_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&AB_F64_PIPELINE_LAYOUT),
            module: &INCREMENT_F64_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static INCREMENT_F16_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/increment_f16.wgsl"))
});

pub static INCREMENT_F16_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&AB_F16_PIPELINE_LAYOUT),
            module: &INCREMENT_F16_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static MUL_IN_PLACE_F32_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/mul_in_place_f32.wgsl"))
});

pub static MUL_IN_PLACE_F32_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&AB_F32_PIPELINE_LAYOUT),
            module: &MUL_IN_PLACE_F32_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static MUL_IN_PLACE_F16_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/mul_in_place_f16.wgsl"))
});

pub static MUL_IN_PLACE_F16_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&AB_F16_PIPELINE_LAYOUT),
            module: &MUL_IN_PLACE_F16_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

static MUL_IN_PLACE_F64_MODULE: Lazy<ShaderModule> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_shader_module(wgpu::include_wgsl!("../wgpu_shaders/mul_in_place_f64.wgsl"))
});

pub static MUL_IN_PLACE_F64_PIPELINE: Lazy<ComputePipeline> = Lazy::new(|| {
    DEVICE_QUEUE
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&AB_F64_PIPELINE_LAYOUT),
            module: &MUL_IN_PLACE_F64_MODULE,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
});

pub fn abc_f32_bind_group(info: &Buffer, a: &Buffer, b: &Buffer, c: &Buffer) -> BindGroup {
    DEVICE_QUEUE
        .0
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ABC_F32_BIND_GROUP_LAYOUT,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c.as_entire_binding(),
                },
            ],
        })
}

pub fn abc_f64_bind_group(info: &Buffer, a: &Buffer, b: &Buffer, c: &Buffer) -> BindGroup {
    DEVICE_QUEUE
        .0
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ABC_F64_BIND_GROUP_LAYOUT,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c.as_entire_binding(),
                },
            ],
        })
}

pub fn abc_f16_bind_group(info: &Buffer, a: &Buffer, b: &Buffer, c: &Buffer) -> BindGroup {
    DEVICE_QUEUE
        .0
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ABC_F16_BIND_GROUP_LAYOUT,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c.as_entire_binding(),
                },
            ],
        })
}

pub fn ab_f32_bind_group(info: &Buffer, a: &Buffer, b: &Buffer) -> BindGroup {
    DEVICE_QUEUE
        .0
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &AB_F32_BIND_GROUP_LAYOUT,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
            ],
        })
}

pub fn ab_f64_bind_group(info: &Buffer, a: &Buffer, b: &Buffer) -> BindGroup {
    DEVICE_QUEUE
        .0
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &AB_F64_BIND_GROUP_LAYOUT,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
            ],
        })
}

pub fn ab_f16_bind_group(info: &Buffer, a: &Buffer, b: &Buffer) -> BindGroup {
    DEVICE_QUEUE
        .0
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &AB_F16_BIND_GROUP_LAYOUT,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
            ],
        })
}
