use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use wgpu::SubmissionIndex;

use super::{download_vec::DownloadGpuVec, dtype::DtypeVec, globals::DEVICE_QUEUE};

pub static INTERMEDIATES_MAP: Lazy<Mutex<HashMap<&'static str, DownloadGpuVec>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub struct ComputeHandle {
    output: DownloadGpuVec,
    submission: SubmissionIndex,
}

impl ComputeHandle {
    pub fn new(output: DownloadGpuVec, submission: SubmissionIndex) -> Self {
        output
            .buffer()
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});

        INTERMEDIATES_MAP
            .lock()
            .iter()
            .map(|(_k, vec)| vec.buffer().slice(..))
            .for_each(|buffer_slice| buffer_slice.map_async(wgpu::MapMode::Read, |_| {}));

        Self { output, submission }
    }

    pub fn join(self) -> (DtypeVec, HashMap<&'static str, DtypeVec>) {
        DEVICE_QUEUE
            .0
            .poll(wgpu::PollType::WaitForSubmissionIndex(self.submission))
            .unwrap();

        let intermediates = INTERMEDIATES_MAP
            .lock()
            .drain()
            .map(|(k, vec)| {
                let data = vec.buffer().slice(..).get_mapped_range();
                let vec = vec.dtype().to_vec(&data);
                (k, vec)
            })
            .collect();
        let output_data = self.output.buffer().slice(..).get_mapped_range();
        (self.output.dtype().to_vec(&output_data), intermediates)
    }
}
