use wgpu::{Buffer, BufferAddress};

use super::{dtype::Dtype, globals::DEVICE_QUEUE};

pub struct DownloadGpuVec {
    buffer: Buffer,
    dtype: Dtype,
}

impl DownloadGpuVec {
    pub fn new(size: BufferAddress, dtype: Dtype) -> Self {
        let buffer = DEVICE_QUEUE.0.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self { buffer, dtype }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }
}
