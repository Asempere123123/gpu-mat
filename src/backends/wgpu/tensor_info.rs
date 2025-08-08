use wgpu::{Buffer, BufferAddress, BufferDescriptor};

use super::globals::DEVICE_QUEUE;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformTensorInfo {
    pub shape: [u32; 8],
    pub rank: u32,
    pub length: u32,
    pub _padding: [u32; 2],
}

impl UniformTensorInfo {
    pub fn new(shape: &Vec<u32>) -> Self {
        let mut shape_arr = [1; 8];
        shape_arr[..shape.len()].copy_from_slice(shape);
        Self {
            shape: shape_arr,
            rank: shape.len() as u32,
            length: shape.iter().product::<u32>(),
            _padding: [0; 2],
        }
    }
}

pub struct TensorInfo {
    buffer: Buffer,
}

impl TensorInfo {
    pub fn new() -> Self {
        Self {
            buffer: DEVICE_QUEUE.0.create_buffer(&BufferDescriptor {
                label: None,
                size: core::mem::size_of::<UniformTensorInfo>() as BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    pub fn set(&self, info: &UniformTensorInfo) {
        DEVICE_QUEUE
            .1
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(info));
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}
