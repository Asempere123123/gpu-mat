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
    fn add_f32() {
        env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

        type F = f32;

        let a = &tensor::GpuTensor::new::<F>(vec![32], &[1.; 32]);
        let b = &tensor::GpuTensor::new::<F>(vec![32], &[2.; 32]);
        let c = &tensor::GpuTensor::new::<F>(vec![32], &[0.5; 32]);
        let d = &mut tensor::GpuTensor::with_capacity(a.capacity());

        assert!(d.set(a + b + c).compute().join().0 == super::dtype::DtypeVec::F32(vec![3.5; 32]));
        assert!(d.increment(a).compute().join().0 == super::dtype::DtypeVec::F32(vec![4.5; 32]));
    }
}
