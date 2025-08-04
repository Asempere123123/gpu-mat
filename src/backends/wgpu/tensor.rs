use wgpu::{Buffer, BufferAddress};

use super::{
    bind_groups::{
        ADD_F32_PIPELINE, INCREMENT_F32_PIPELINE, ab_f32_bind_group, abc_f32_bind_group,
    },
    command_encoder::GlobalCommandEncoder,
    download_vec::DownloadGpuTensor,
    dtype::Dtype,
    dtype::Dtyped,
    globals::DEVICE_QUEUE,
    handle::{ComputeHandle, INTERMEDIATES_MAP},
    vec::GpuVec,
};

pub struct GpuTensor {
    shape: Vec<BufferAddress>,
    buffer: GpuVec,
}

impl GpuTensor {
    pub fn new<F: Dtyped>(shape: Vec<BufferAddress>, values: &[F]) -> Self {
        assert!(shape.iter().product::<BufferAddress>() == values.len() as BufferAddress);

        Self {
            shape,
            buffer: GpuVec::new_init(values),
        }
    }

    pub fn with_capacity<F: Dtyped>(capacity: BufferAddress) -> Self {
        Self {
            shape: Vec::new(),
            buffer: GpuVec::new_uninit::<F>(capacity),
        }
    }

    pub fn capacity(&self) -> BufferAddress {
        self.buffer.capacity()
    }

    pub fn capacity_elements(&self) -> BufferAddress {
        self.buffer.capacity_elements()
    }

    fn buffer(&self) -> &Buffer {
        self.buffer.buffer()
    }

    pub fn dtype(&self) -> Dtype {
        self.buffer.dtype()
    }

    pub fn compute(&self) -> ComputeHandle {
        let output_download_vec =
            DownloadGpuTensor::new(self.capacity(), self.shape.clone(), self.dtype());

        let mut encoder = GlobalCommandEncoder::lock();
        encoder.get().copy_buffer_to_buffer(
            &self.buffer(),
            0,
            &output_download_vec.buffer(),
            0,
            self.capacity(),
        );

        let command_buffer = encoder.finish();
        let idx = DEVICE_QUEUE.1.submit([command_buffer]);
        ComputeHandle::new(output_download_vec, idx)
    }

    pub fn save_intermediate(&self, name: &'static str) -> &Self {
        let intermediate_download_vec =
            DownloadGpuTensor::new(self.capacity(), self.shape.clone(), self.dtype());

        GlobalCommandEncoder::lock().get().copy_buffer_to_buffer(
            &self.buffer(),
            0,
            &intermediate_download_vec.buffer(),
            0,
            self.capacity(),
        );

        INTERMEDIATES_MAP
            .lock()
            .insert(name, intermediate_download_vec);

        self
    }

    pub fn save_intermediate_mut(&mut self, name: &'static str) -> &mut Self {
        self.save_intermediate(name);
        self
    }

    pub fn set(&mut self, seter: GpuTensorSetterFn<impl FnOnce(&mut GpuTensor)>) -> &mut Self {
        seter.0(self);
        self
    }

    pub fn add(&mut self, lhs: &Self, rhs: &Self) -> &mut Self {
        assert!(lhs.dtype() == rhs.dtype());
        self.buffer.set_dtype(lhs.dtype());

        assert!(lhs.shape == rhs.shape);
        assert!(self.buffer.capacity_elements() >= lhs.shape.iter().product::<BufferAddress>());
        self.shape.clear();
        self.shape.extend_from_slice(&lhs.shape);

        let mut encoder = GlobalCommandEncoder::lock();
        let mut compute_pass = encoder
            .get()
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

        // Todo: match on dtype and use corresponding add pipeline
        compute_pass.set_pipeline(&ADD_F32_PIPELINE);
        compute_pass.set_bind_group(
            0,
            &abc_f32_bind_group(&lhs.buffer(), &rhs.buffer(), &self.buffer()),
            &[],
        );

        let workgroup_count = self.capacity_elements().div_ceil(64);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);

        self
    }

    pub fn increment(&mut self, by: &Self) -> &mut Self {
        assert!(self.dtype() == by.dtype());
        assert!(self.shape == by.shape);

        let mut encoder = GlobalCommandEncoder::lock();
        let mut compute_pass = encoder
            .get()
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

        // TODO: match on dtype
        compute_pass.set_pipeline(&INCREMENT_F32_PIPELINE);
        compute_pass.set_bind_group(0, &ab_f32_bind_group(&self.buffer(), &by.buffer()), &[]);

        let workgroup_count = self.capacity_elements().div_ceil(64);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);

        self
    }
}

pub struct GpuTensorSetterFn<Fn: FnOnce(&mut GpuTensor)>(Fn);

impl<Fn: FnOnce(&mut GpuTensor)> GpuTensorSetterFn<Fn> {
    pub fn save_intermediate(
        self,
        name: &'static str,
    ) -> GpuTensorSetterFn<impl FnOnce(&mut GpuTensor)> {
        GpuTensorSetterFn(move |target| {
            self.0(target);

            target.save_intermediate(name);
        })
    }
}

impl core::ops::Add for &GpuTensor {
    type Output = GpuTensorSetterFn<impl FnOnce(&mut GpuTensor)>;
    fn add(self, rhs: Self) -> Self::Output {
        GpuTensorSetterFn(|target| {
            target.add(self, rhs);
        })
    }
}

impl<'a, Fn: FnOnce(&mut GpuTensor) + 'a> core::ops::Add<&'a GpuTensor> for GpuTensorSetterFn<Fn> {
    type Output = GpuTensorSetterFn<impl FnOnce(&mut GpuTensor) + 'a>;

    fn add(self, rhs: &'a GpuTensor) -> Self::Output {
        GpuTensorSetterFn(move |target| {
            self.0(target);

            target.increment(rhs);
        })
    }
}

impl<'a, Fn: FnOnce(&mut GpuTensor) + 'a> core::ops::Add<GpuTensorSetterFn<Fn>> for &'a GpuTensor {
    type Output = GpuTensorSetterFn<impl FnOnce(&mut GpuTensor)>;

    fn add(self, rhs: GpuTensorSetterFn<Fn>) -> Self::Output {
        rhs + self
    }
}
