//! Tensor storage used by the native CPU runtime.

use super::arena::{Buffer, F32Storage, IntoF32Storage};
use anyhow::{Result, bail};
use std::sync::Arc;

pub(crate) trait IntoShape {
    fn into_shape(self) -> Vec<usize>;
}

impl IntoShape for usize {
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

macro_rules! tuple_shape {
    ($($name:ident),+) => {
        impl IntoShape for ($(tuple_shape!(@ty $name),)+) {
            #[allow(non_snake_case)]
            fn into_shape(self) -> Vec<usize> {
                let ($($name,)+) = self;
                vec![$($name),+]
            }
        }
    };
    (@ty $name:ident) => { usize };
}

tuple_shape!(A);
tuple_shape!(A, B);
tuple_shape!(A, B, C);
tuple_shape!(A, B, C, D);
tuple_shape!(A, B, C, D, E);

#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: TensorData,
}

#[derive(Clone, Debug)]
pub(crate) enum TensorData {
    F32(Arc<F32Storage>),
    I64(Arc<Vec<i64>>),
}

impl Tensor {
    pub fn from_f32(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        validate_len(&shape, data.len())?;
        Ok(Self {
            shape,
            data: TensorData::F32(Arc::new(F32Storage::unpooled(data))),
        })
    }

    pub fn from_i64(shape: impl Into<Vec<usize>>, data: Vec<i64>) -> Result<Self> {
        let shape = shape.into();
        validate_len(&shape, data.len())?;
        Ok(Self {
            shape,
            data: TensorData::I64(Arc::new(data)),
        })
    }

    pub(crate) fn new_f32(shape: Vec<usize>, data: impl IntoF32Storage) -> Self {
        let data = data.into_f32_storage();
        assert_eq!(element_count(&shape), Some(data.len()));
        Self {
            shape,
            data: TensorData::F32(Arc::new(data)),
        }
    }

    pub(crate) fn new_i64(shape: Vec<usize>, data: Vec<i64>) -> Self {
        assert_eq!(element_count(&shape), Some(data.len()));
        Self {
            shape,
            data: TensorData::I64(Arc::new(data)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        match &self.data {
            TensorData::F32(data) => data.len(),
            TensorData::I64(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> Result<&[f32]> {
        match &self.data {
            TensorData::F32(data) => Ok(data.values()),
            TensorData::I64(_) => bail!("expected an f32 tensor"),
        }
    }

    pub fn as_i64(&self) -> Result<&[i64]> {
        match &self.data {
            TensorData::I64(data) => Ok(data),
            TensorData::F32(_) => bail!("expected an i64 tensor"),
        }
    }

    pub(crate) fn f32_mut(&mut self) -> Result<&mut Vec<f32>> {
        match &mut self.data {
            TensorData::F32(data) => {
                if Arc::get_mut(data).is_none() {
                    let mut values = Buffer::for_overwrite(data.len());
                    values.copy_from_slice(data.values());
                    *data = Arc::new(values.into_f32_storage());
                }
                Ok(Arc::get_mut(data)
                    .expect("tensor storage was just made unique")
                    .values_mut())
            }
            TensorData::I64(_) => bail!("expected an f32 tensor"),
        }
    }

    pub(crate) fn into_f32(self) -> Result<Vec<f32>> {
        match self.data {
            TensorData::F32(data) => Ok(Arc::try_unwrap(data)
                .map(F32Storage::into_vec)
                .unwrap_or_else(|data| data.values().to_vec())),
            TensorData::I64(_) => bail!("expected an f32 tensor"),
        }
    }

    pub(crate) fn into_i64(self) -> Result<Vec<i64>> {
        match self.data {
            TensorData::I64(data) => {
                Ok(Arc::try_unwrap(data).unwrap_or_else(|data| (*data).clone()))
            }
            TensorData::F32(_) => bail!("expected an i64 tensor"),
        }
    }
}

pub(crate) fn element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |size, dim| size.checked_mul(*dim))
}

pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![1; shape.len()];
    for index in (1..shape.len()).rev() {
        result[index - 1] = result[index] * shape[index];
    }
    result
}

fn validate_len(shape: &[usize], len: usize) -> Result<()> {
    let expected = element_count(shape).ok_or_else(|| anyhow::anyhow!("tensor shape overflow"))?;
    if expected != len {
        bail!("tensor shape {shape:?} requires {expected} values, found {len}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::InferenceArena;

    #[test]
    fn validates_tensor_length() {
        assert!(Tensor::from_f32([2, 3], vec![0.0; 6]).is_ok());
        assert!(Tensor::from_f32([2, 3], vec![0.0; 5]).is_err());
    }

    #[test]
    fn computes_contiguous_strides() {
        assert_eq!(strides(&[2, 3, 4]), [12, 4, 1]);
    }

    #[test]
    fn returns_storage_only_after_the_last_tensor_reference() {
        let arena = InferenceArena::default();
        arena.scope(|| {
            let tensor = Tensor::new_f32(vec![1024], Buffer::zeroed(1024));
            let address = tensor.as_f32().unwrap().as_ptr();
            let clone = tensor.clone();
            drop(tensor);
            assert_eq!(arena.cached_buffers(), 0);
            drop(clone);
            assert_eq!(arena.cached_buffers(), 1);

            let reused = Tensor::new_f32(vec![1024], Buffer::zeroed(1024));
            assert_eq!(reused.as_f32().unwrap().as_ptr(), address);
        });
    }

    #[test]
    fn public_input_and_extracted_output_stay_unpooled() {
        let arena = InferenceArena::default();
        arena.scope(|| {
            drop(Tensor::from_f32([1024], vec![1.0; 1024]).unwrap());
            assert_eq!(arena.cached_buffers(), 0);

            let tensor = Tensor::new_f32(vec![1024], Buffer::zeroed(1024));
            let values = tensor.into_f32().unwrap();
            assert_eq!(arena.cached_buffers(), 0);
            drop(values);
            assert_eq!(arena.cached_buffers(), 0);
        });
    }
}
