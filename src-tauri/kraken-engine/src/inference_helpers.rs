//! Shared inference helpers reused by the candle backend.
//!
//! Extracted from kraken-rust's `inference.rs` (the rest of that file is the
//! ONNX forward pass, not vendored here).

use ndarray::Array4;

/// Sigmoid function.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Nearest-neighbor upsampling of a 4D array (1, C, H, W) to (1, C, OH, OW).
/// Matches PyTorch `F.interpolate(mode='nearest')`, which maps each output
/// coordinate back to an input coordinate via `floor(out * scale)` where
/// `scale = in_size / out_size`.
pub fn nearest_upsample_2d(input: &Array4<f32>, out_h: usize, out_w: usize) -> Array4<f32> {
    let (_, c, in_h, in_w) = input.dim();
    let mut out = Array4::<f32>::zeros((1, c, out_h, out_w));
    let scale_h = in_h as f64 / out_h as f64;
    let scale_w = in_w as f64 / out_w as f64;
    for oh in 0..out_h {
        let ih = ((oh as f64) * scale_h).floor() as usize;
        let ih = ih.min(in_h.saturating_sub(1));
        for ow in 0..out_w {
            let iw = ((ow as f64) * scale_w).floor() as usize;
            let iw = iw.min(in_w.saturating_sub(1));
            for ch in 0..c {
                out[[0, ch, oh, ow]] = input[[0, ch, ih, iw]];
            }
        }
    }
    out
}
