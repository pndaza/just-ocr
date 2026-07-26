use super::tensor::{IntoShape, Tensor, element_count};
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, fs, path::Path};

pub(crate) struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("read model {}", path.display()))?;
        Self::from_bytes(&bytes)
            .with_context(|| format!("decode safetensors model {}", path.display()))
    }

    /// Deserialize weights directly from an in-memory safetensors buffer.
    /// Used by the host's `include_bytes!`-bundled loader.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let archive = SafeTensors::deserialize(bytes)?;
        let mut tensors = HashMap::with_capacity(archive.len());
        for (name, view) in archive.iter() {
            ensure!(
                view.dtype() == Dtype::F32,
                "tensor {name:?} uses {:?}; only F32 safetensors are supported",
                view.dtype()
            );
            let expected = element_count(view.shape()).context("safetensors shape overflow")?;
            let expected_bytes = expected
                .checked_mul(size_of::<f32>())
                .context("safetensors byte length overflow")?;
            ensure!(
                view.data().len() == expected_bytes,
                "tensor {name:?} has an invalid byte length"
            );
            let mut values = Vec::with_capacity(expected);
            for bytes in view.data().chunks_exact(size_of::<f32>()) {
                values.push(f32::from_le_bytes(
                    bytes.try_into().expect("f32 chunk has four bytes"),
                ));
            }
            tensors.insert(
                name.to_owned(),
                Tensor::from_f32(view.shape().to_vec(), values)
                    .with_context(|| format!("decode tensor {name:?}"))?,
            );
        }
        Ok(Self { tensors })
    }

    pub(crate) fn builder(&self) -> VarBuilder<'_> {
        VarBuilder {
            weights: self,
            prefix: String::new(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct VarBuilder<'a> {
    weights: &'a Weights,
    prefix: String,
}

impl<'a> VarBuilder<'a> {
    pub(crate) fn pp(&self, part: impl ToString) -> Self {
        let part = part.to_string();
        let prefix = if self.prefix.is_empty() {
            part
        } else {
            format!("{}.{}", self.prefix, part)
        };
        Self {
            weights: self.weights,
            prefix,
        }
    }

    pub(crate) fn get(&self, shape: impl IntoShape, name: &str) -> Result<Tensor> {
        let shape = shape.into_shape();
        let full_name = if self.prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        let tensor = self
            .weights
            .tensors
            .get(&full_name)
            .with_context(|| format!("missing tensor {full_name:?}"))?;
        ensure!(
            tensor.shape() == shape,
            "tensor {full_name:?} expects shape {shape:?}, found {:?}",
            tensor.shape()
        );
        Ok(tensor.clone())
    }
}
