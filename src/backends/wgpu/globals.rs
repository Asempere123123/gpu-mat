use once_cell::sync::Lazy;
use pollster::FutureExt;
use wgpu::{Adapter, Device, Instance, Queue};

static INSTANCE: Lazy<Instance> =
    Lazy::new(|| wgpu::Instance::new(&wgpu::InstanceDescriptor::default()));

static ADAPTER: Lazy<Adapter> = Lazy::new(|| {
    let adapter = INSTANCE
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .block_on()
        .expect("GpuMat: Could not get a wgpu adapter");

    if !adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("GpuMat: wgpu adapter does not support compute shaders");
    }

    adapter
});

pub static DEVICE_QUEUE: Lazy<(Device, Queue)> = Lazy::new(|| {
    let features = ADAPTER
        .features()
        .intersection(wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_F16);
    if !features.contains(wgpu::Features::SHADER_F64) {
        log::warn!("f64 values are not suported on this device");
    }
    if !features.contains(wgpu::Features::SHADER_F16) {
        log::warn!("f16 values are not suported on this device");
    }

    ADAPTER
        .request_device(&wgpu::DeviceDescriptor {
            label: "GpuMat".into(),
            required_features: features,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .block_on()
        .expect("GpuMat: Failed to create wgpu device")
});
