use bytemuck::{AnyBitPattern, NoUninit};
use half::f16;
use ndarray::{Array, IxDyn};
use std::fmt::Debug;
use wgpu::BufferView;

#[derive(Clone, Copy, PartialEq)]
pub enum Dtype {
    F16,
    F32,
    F64,
}

impl Dtype {
    pub fn to_vec(&self, data: &BufferView, shape: &Vec<u32>) -> DtypeVec {
        match self {
            Dtype::F16 => DtypeVec::F16(
                Array::from_shape_vec(
                    IxDyn(&shape.iter().map(|&idx| idx as usize).collect::<Vec<_>>()),
                    bytemuck::cast_slice(data).to_vec(),
                )
                .unwrap(),
            ),
            Dtype::F32 => DtypeVec::F32(
                Array::from_shape_vec(
                    IxDyn(&shape.iter().map(|&idx| idx as usize).collect::<Vec<_>>()),
                    bytemuck::cast_slice(data).to_vec(),
                )
                .unwrap(),
            ),
            Dtype::F64 => DtypeVec::F64(
                Array::from_shape_vec(
                    IxDyn(&shape.iter().map(|&idx| idx as usize).collect::<Vec<_>>()),
                    bytemuck::cast_slice(data).to_vec(),
                )
                .unwrap(),
            ),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Dtype::F16 => core::mem::size_of::<f16>(),
            Dtype::F32 => core::mem::size_of::<f32>(),
            Dtype::F64 => core::mem::size_of::<f64>(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DtypeVec {
    F16(Array<f16, IxDyn>),
    F32(Array<f32, IxDyn>),
    F64(Array<f64, IxDyn>),
}

pub trait Dtyped: NoUninit + AnyBitPattern + Debug {
    fn dtype() -> Dtype;
}

impl Dtyped for f16 {
    fn dtype() -> Dtype {
        Dtype::F16
    }
}

impl Dtyped for f32 {
    fn dtype() -> Dtype {
        Dtype::F32
    }
}

impl Dtyped for f64 {
    fn dtype() -> Dtype {
        Dtype::F64
    }
}
