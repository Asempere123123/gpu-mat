use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use wgpu::{CommandBuffer, CommandEncoder};

use super::globals::DEVICE_QUEUE;

static COMMAND_ENCODER: Lazy<GlobalCommandEncoder> = Lazy::new(GlobalCommandEncoder::new);

pub struct GlobalCommandEncoder {
    encoder: Mutex<Option<CommandEncoder>>,
}

impl GlobalCommandEncoder {
    fn new() -> Self {
        Self {
            encoder: Mutex::new(None),
        }
    }

    pub fn lock<'a>() -> GlobalCommandEncoderGuard<'a> {
        GlobalCommandEncoderGuard {
            guard: COMMAND_ENCODER.encoder.lock(),
        }
    }
}

pub struct GlobalCommandEncoderGuard<'a> {
    guard: MutexGuard<'a, Option<CommandEncoder>>,
}

impl<'a> GlobalCommandEncoderGuard<'a> {
    pub fn get(&mut self) -> &mut CommandEncoder {
        self.guard.get_or_insert_with(|| {
            DEVICE_QUEUE
                .0
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
        })
    }

    pub fn finish(&mut self) -> CommandBuffer {
        self.get();

        unsafe { self.guard.take().unwrap_unchecked().finish() }
    }
}
