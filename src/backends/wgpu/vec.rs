use wgpu::{Buffer, BufferAddress, util::DeviceExt};

use crate::backends::wgpu::dtype::Dtype;

use super::{
    bind_groups::{ADD_F32_PIPELINE, abc_f32_bind_group},
    bind_groups::{INCREMENT_F32_PIPELINE, ab_f32_bind_group},
    command_encoder::GlobalCommandEncoder,
    download_vec::DownloadGpuVec,
    dtype::Dtyped,
    globals::DEVICE_QUEUE,
    handle::{ComputeHandle, INTERMEDIATES_MAP},
};

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

    pub fn size(&self) -> BufferAddress {
        self.buffer.size()
    }

    pub fn save_intermediate(&self, name: &'static str) -> &Self {
        let intermediate_download_vec = DownloadGpuVec::new(self.size(), self.dtype);

        GlobalCommandEncoder::lock().get().copy_buffer_to_buffer(
            &self.buffer,
            0,
            &intermediate_download_vec.buffer(),
            0,
            self.size(),
        );

        INTERMEDIATES_MAP
            .lock()
            .insert(name, intermediate_download_vec);

        self
    }

    pub fn compute(&self) -> ComputeHandle {
        let output_download_vec = DownloadGpuVec::new(self.size(), self.dtype);

        let mut encoder = GlobalCommandEncoder::lock();
        encoder.get().copy_buffer_to_buffer(
            &self.buffer,
            0,
            &output_download_vec.buffer(),
            0,
            self.size(),
        );

        let command_buffer = encoder.finish();
        let idx = DEVICE_QUEUE.1.submit([command_buffer]);
        ComputeHandle::new(output_download_vec, idx)
    }

    pub fn set(&self, seter: GpuVecSetterFn<impl FnOnce(&GpuVec)>) {
        seter.0(self)
    }

    pub fn add(&self, lhs: &Self, rhs: &Self) -> &Self {
        assert!(lhs.size() == rhs.size() && rhs.size() == self.size());

        let mut encoder = GlobalCommandEncoder::lock();
        let mut compute_pass = encoder
            .get()
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

        compute_pass.set_pipeline(&ADD_F32_PIPELINE);
        compute_pass.set_bind_group(
            0,
            &abc_f32_bind_group(&lhs.buffer, &rhs.buffer, &self.buffer),
            &[],
        );

        let workgroup_count = self.size().div_ceil(64);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);

        return self;
    }

    pub fn increment(&self, by: &Self) -> &Self {
        assert!(self.size() == by.size());

        let mut encoder = GlobalCommandEncoder::lock();
        let mut compute_pass = encoder
            .get()
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

        compute_pass.set_pipeline(&INCREMENT_F32_PIPELINE);
        compute_pass.set_bind_group(0, &ab_f32_bind_group(&self.buffer, &by.buffer), &[]);

        let workgroup_count = self.size().div_ceil(64);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);

        return self;
    }
}

pub struct GpuVecSetterFn<Fn: FnOnce(&GpuVec)>(Fn);

impl core::ops::Add for &GpuVec {
    type Output = GpuVecSetterFn<impl FnOnce(&GpuVec)>;
    fn add(self, rhs: Self) -> Self::Output {
        GpuVecSetterFn(|target| {
            assert!(self.size() == rhs.size() && rhs.size() == target.size());

            let mut encoder = GlobalCommandEncoder::lock();
            let mut compute_pass = encoder
                .get()
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

            compute_pass.set_pipeline(&ADD_F32_PIPELINE);
            compute_pass.set_bind_group(
                0,
                &abc_f32_bind_group(&self.buffer, &rhs.buffer, &target.buffer),
                &[],
            );

            let workgroup_count = self.size().div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        })
    }
}

impl<'a, Fn: FnOnce(&GpuVec) + 'a> core::ops::Add<&'a GpuVec> for GpuVecSetterFn<Fn> {
    type Output = GpuVecSetterFn<impl FnOnce(&GpuVec) + 'a>;

    fn add(self, rhs: &'a GpuVec) -> Self::Output {
        GpuVecSetterFn(move |target: &GpuVec| {
            self.0(target);

            assert!(rhs.size() == target.size());

            let mut encoder = GlobalCommandEncoder::lock();
            let mut compute_pass = encoder
                .get()
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

            compute_pass.set_pipeline(&INCREMENT_F32_PIPELINE);
            compute_pass.set_bind_group(0, &ab_f32_bind_group(&target.buffer, &rhs.buffer), &[]);

            let workgroup_count = rhs.size().div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        })
    }
}
