pub trait GpuTensor<F> {
    fn new(shape: &[u32], values: &[f32]) -> Self;
}
