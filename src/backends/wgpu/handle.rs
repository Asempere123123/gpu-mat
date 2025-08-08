use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::atomic::AtomicBool,
    sync::{Arc, atomic::Ordering},
    task::Poll,
};
use wgpu::SubmissionIndex;

use super::{download_vec::DownloadGpuTensor, dtype::DtypeVec, globals::DEVICE_QUEUE};

pub static INTERMEDIATES_MAP: Lazy<Mutex<HashMap<&'static str, DownloadGpuTensor>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub struct ComputeHandle {
    output: DownloadGpuTensor,
    submission: SubmissionIndex,
    ready: Arc<AtomicBool>,
}

impl ComputeHandle {
    pub fn new(output: DownloadGpuTensor, submission: SubmissionIndex) -> Self {
        INTERMEDIATES_MAP
            .lock()
            .iter()
            .map(|(_k, vec)| vec.buffer().slice(..))
            .for_each(|buffer_slice| buffer_slice.map_async(wgpu::MapMode::Read, |_| {}));

        let ready = Arc::new(AtomicBool::new(false));
        let ready_clone = ready.clone();
        output
            .buffer()
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |_result| {
                ready_clone.store(true, Ordering::Relaxed);
            });

        Self {
            output,
            submission,
            ready,
        }
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
                let vec = vec.dtype().to_vec(&data, vec.shape());
                (k, vec)
            })
            .collect();
        let output_data = self.output.buffer().slice(..).get_mapped_range();
        (
            self.output
                .dtype()
                .to_vec(&output_data, self.output.shape()),
            intermediates,
        )
    }
}

impl Future for ComputeHandle {
    type Output = (DtypeVec, HashMap<&'static str, DtypeVec>);
    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> Poll<Self::Output> {
        if !self.ready.load(Ordering::Relaxed) {
            cx.waker().wake_by_ref();
            DEVICE_QUEUE.0.poll(wgpu::PollType::Poll).unwrap();
            return Poll::Pending;
        }

        let intermediates = INTERMEDIATES_MAP
            .lock()
            .drain()
            .map(|(k, vec)| {
                let data = vec.buffer().slice(..).get_mapped_range();
                let vec = vec.dtype().to_vec(&data, vec.shape());
                (k, vec)
            })
            .collect();
        let output_data = self.output.buffer().slice(..).get_mapped_range();
        Poll::Ready((
            self.output
                .dtype()
                .to_vec(&output_data, self.output.shape()),
            intermediates,
        ))
    }
}
