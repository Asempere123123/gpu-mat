use wgpu::{Buffer, BufferAddress};

use super::{dtype::Dtype, globals::DEVICE_QUEUE};

pub struct DownloadGpuTensor {
    shape: Vec<u32>,
    buffer: Buffer,
    dtype: Dtype,
}

impl DownloadGpuTensor {
    pub fn new(size: BufferAddress, shape: Vec<u32>, dtype: Dtype) -> Self {
        let buffer = DEVICE_QUEUE.0.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            shape,
            dtype,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn shape(&self) -> &Vec<u32> {
        &self.shape
    }

    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }
}
