use std::fmt::Debug;

use bytemuck::{AnyBitPattern, NoUninit};
use wgpu::BufferView;

#[derive(Clone, Copy)]
pub enum Dtype {
    F32,
}

impl Dtype {
    pub fn to_vec(&self, data: &BufferView) -> DtypeVec {
        match self {
            Dtype::F32 => DtypeVec::F32(bytemuck::cast_slice(data).to_vec()),
        }
    }
}

#[derive(Debug)]
pub enum DtypeVec {
    F32(Vec<f32>),
}

pub trait Dtyped: NoUninit + AnyBitPattern + Debug {
    fn dtype() -> Dtype;
}

impl Dtyped for f32 {
    fn dtype() -> Dtype {
        Dtype::F32
    }
}
