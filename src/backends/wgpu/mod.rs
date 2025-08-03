mod bind_groups;
mod command_encoder;
mod download_vec;
mod dtype;
mod globals;
mod handle;
mod vec;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu() {
        let nums: [f32; 2] = [1., 2.];
        let other_nums: [f32; 2] = [2., 3.];
        let more_nums: [f32; 2] = [1., 1.];

        let a = &vec::GpuVec::new_init(&nums);
        let b = &vec::GpuVec::new_init(&other_nums);
        let c = &vec::GpuVec::new_init(&more_nums);
        let d = &vec::GpuVec::new_uninit::<f32>(a.size());

        d.add(a, b).save_intermediate(":3").increment(c);

        let result = d.compute().join();
        panic!("{:?}", result);
    }
}
