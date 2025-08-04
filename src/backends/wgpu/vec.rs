use wgpu::{Buffer, BufferAddress, util::DeviceExt};

use crate::backends::wgpu::dtype::Dtype;

use super::{dtype::Dtyped, globals::DEVICE_QUEUE};

pub struct GpuVec {
    buffer: Buffer,
    dtype: Dtype,
}

impl GpuVec {
    pub fn new_init<F: Dtyped>(value: &[F]) -> Self {
        let buffer = DEVICE_QUEUE
            .0
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(value),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        Self {
            buffer,
            dtype: F::dtype(),
        }
    }

    pub fn new_uninit<F: Dtyped>(size: BufferAddress) -> Self {
        let buffer = DEVICE_QUEUE.0.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            dtype: F::dtype(),
        }
    }

    pub fn capacity(&self) -> BufferAddress {
        self.buffer.size()
    }

    pub fn capacity_elements(&self) -> BufferAddress {
        self.buffer.size() / self.dtype.size() as BufferAddress
    }

    pub fn set_dtype(&mut self, dtype: Dtype) {
        self.dtype = dtype;
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}
