//! Candle-based inference: forward pass + upsample + sigmoid.
//!
//! Alternative to `inference::run_inference` that uses the candle-core
//! `SegmentationModelCandle` instead of ONNX Runtime.

use anyhow::{Context, Result};
use candle_core::Tensor;
use ndarray::{s, Array3, Array4};

use crate::kraken::heatmap::Heatmap;
use crate::kraken::inference_helpers::{nearest_upsample_2d, sigmoid};
use crate::kraken::preprocess::Preprocessed;
use crate::kraken::segmentation_candle::SegmentationModelCandle;

/// Run inference using the candle-core backend.
///
/// Mirrors `inference::run_inference` but executes the forward pass via
/// candle instead of ONNX Runtime. Produces the same `Heatmap` output.
pub fn run_inference_candle(
    model: &SegmentationModelCandle,
    input: &Preprocessed,
) -> Result<Heatmap> {
    let padding = model_padding(model);
    let (pl, pr, pt, pb) = (
        padding[0] as usize,
        padding[1] as usize,
        padding[2] as usize,
        padding[3] as usize,
    );
    let cls_map = model.meta.class_mapping.clone();

    // Convert preprocessed ndarray to candle tensor.
    // input.tensor is (C, H, W), we need (1, C, H, W).
    let (c, h, w) = input.tensor.dim();
    let input_data: Vec<f32> = match input.tensor.as_slice() {
        Some(slice) => slice.to_vec(),
        None => input.tensor.iter().cloned().collect(),
    };
    let input_tensor = Tensor::from_vec(input_data, (1, c, h, w), &candle_core::Device::Cpu)?;

    // Forward pass via candle.
    let logits = model.forward(&input_tensor)?;

    // Extract logits to ndarray (1, C, H', W').
    let logit_dims = logits.dims();
    let n = logit_dims[1]; // num channels
    let logits_contig = logits.contiguous()?;
    let logits_data: Vec<f32> = logits_contig.flatten_all()?.to_vec1()?;
    let logits_4d = Array4::from_shape_vec(
        (logit_dims[0], logit_dims[1], logit_dims[2], logit_dims[3]),
        logits_data,
    )
    .context("failed to reshape candle logits")?;

    // Upsample to scal_im shape.
    let (scal_h, scal_w) = input.scal_im.dim();
    let upsampled = nearest_upsample_2d(&logits_4d, scal_h, scal_w);

    // Sigmoid.
    let probs_4d = upsampled.mapv(sigmoid);

    // Remove padding.
    let h_end = if pb > 0 { scal_h - pb } else { scal_h };
    let w_crop = if input.content_w > 0 && input.content_w < scal_w {
        input.content_w
    } else {
        scal_w
    };
    let w_end = if pr > 0 { w_crop - pr } else { w_crop };

    let probs_sliced = probs_4d.slice(s![0..1, 0..n, pt..h_end, pl..w_end]);
    let scal_im_sliced = input.scal_im.slice(s![pt..h_end, pl..w_end]).to_owned();

    let probs: Array3<f32> = probs_sliced
        .to_owned()
        .into_shape_with_order((n, h_end - pt, w_end - pl))
        .context("failed to reshape heatmap")?;

    Ok(Heatmap {
        probs,
        cls_map,
        scale: (1.0, 1.0),
        scal_im: scal_im_sliced,
    })
}

/// Get padding from the candle model's metadata.
fn model_padding(model: &SegmentationModelCandle) -> [i64; 4] {
    match model.meta.padding.len() {
        0 => [0, 0, 0, 0],
        1 => [model.meta.padding[0]; 4],
        2 => [
            model.meta.padding[0],
            model.meta.padding[0],
            model.meta.padding[1],
            model.meta.padding[1],
        ],
        4 => [
            model.meta.padding[0],
            model.meta.padding[1],
            model.meta.padding[2],
            model.meta.padding[3],
        ],
        _ => [0, 0, 0, 0],
    }
}
