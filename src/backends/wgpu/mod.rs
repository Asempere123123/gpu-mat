mod bind_groups;
mod command_encoder;
mod download_vec;
mod dtype;
mod globals;
mod handle;
mod tensor;
mod vec;

pub use tensor::GpuTensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu() {
        let a = &tensor::GpuTensor::new(vec![32], &[1.; 32]);
        let b = &tensor::GpuTensor::new(vec![32], &[2.; 32]);
        let c = &tensor::GpuTensor::new(vec![32], &[0.5; 32]);
        let d = &mut tensor::GpuTensor::with_capacity::<f32>(a.capacity());

        panic!(
            "{:?}",
            d.set((a + b).save_intermediate("holi") + c)
                .compute()
                .join()
        )
        /*panic!(
            "{:?}",
            c.add(a, b)
                .save_intermediate_mut(":3")
                .increment(a)
                .compute()
                .join()
        );*/
    }
}
