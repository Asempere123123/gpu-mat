mod bind_groups;
mod command_encoder;
mod download_vec;
mod dtype;
mod globals;
mod handle;
mod tensor;
mod tensor_info;
mod vec;

pub use tensor::GpuTensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_f32() {
        env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

        type F = f32;

        let a = &tensor::GpuTensor::new::<F>(vec![2, 2], &[1.; 4]);
        let b = &tensor::GpuTensor::new::<F>(vec![2, 2], &[2.; 4]);
        let c = &tensor::GpuTensor::new::<F>(vec![2, 2], &[0.5; 4]);
        let d = &mut tensor::GpuTensor::with_capacity(a.capacity());

        assert!(
            d.set(a + b + c).compute().join().0
                == super::dtype::DtypeVec::F32(ndarray::Array::from_elem(
                    ndarray::IxDyn(&[2, 2]),
                    3.5
                ))
        );
        assert!(
            d.increment(a).compute().join().0
                == super::dtype::DtypeVec::F32(ndarray::Array::from_elem(
                    ndarray::IxDyn(&[2, 2]),
                    4.5
                ))
        );
    }
}
