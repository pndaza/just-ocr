//! ppocr-engine: vendored PP-OCRv6 tiny text detector (DBNet).
//!
//! Slimmed subset of ppocr-rs (https://github.com/weidix/ppocr-rs): detector +
//! preprocess + DB postprocess only. The recognizer, GPU backend, model
//! download, and CLI were excluded. The tiny-det safetensors is bundled by the
//! host via `include_bytes!`; this crate exposes `Detector::load_from_buffer`.
//!
//! Public API:
//!   - [`Detector`] — loaded detector, reused across calls.
//!   - [`Detector::load_from_buffer`] — load the bundled tiny-det weights.
//!   - [`Detector::detect`] — image → quads in source-image pixel coords.

// The bulk of this crate is verbatim upstream code (tensor ops, kernels, the
// model graph, postprocess helpers) trimmed to detector-only. Trimming leaves
// dead code (recognizer-side helpers like `Linear`, `LayerNorm`, `argmax`) and
// unknown cfg values (`cpu-profile`, `gpu`) that upstream gates on but we don't
// declare as features. Both are expected from the vendor-and-trim approach and
// are not actionable without diverging from upstream. Silence them at the crate
// root so dev-build output stays readable; the host crate (`just-ocr`) itself
// remains warning-clean and any new hand-written code here will still surface.
#![allow(dead_code)]
#![allow(unexpected_cfgs)]

mod arena;
mod backend;
mod kernels;
mod model;
mod ops;
mod postprocess;
mod preprocess;
mod tensor;
mod weights;

pub use model::{CpuOptions, Detector};
pub use postprocess::{Detection, DetectorTransform, Point};
pub use tensor::Tensor;

/// Interleaved row-major RGB8 image view, built from the host's `DynamicImage`.
/// Confines the `image` crate to this type so the preprocess kernels operate
/// on a plain byte slice.
pub struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl RgbImage {
    /// Build from the host's `image::DynamicImage` (converts to RGB8).
    pub fn from_dynamic(img: &image::DynamicImage) -> Self {
        let rgb = img.to_rgb8();
        Self {
            width: rgb.width(),
            height: rgb.height(),
            pixels: rgb.into_raw(),
        }
    }

    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn height(&self) -> u32 {
        self.height
    }
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

use crate::postprocess::{DetectorInputPlan, DetectorPostprocessOptions};
use crate::preprocess::prepare_detector;
use anyhow::Result;

impl Detector {
    /// Run end-to-end detection: image → quads in source-image pixel coords.
    ///
    /// Resizes the input so its longest side is ≤ 736 (PaddleOCR default),
    /// aligned to 32-pixel multiples. Returns one `Detection` per text region
    /// (4-corner quad + score), with coords already mapped back to the source
    /// image via the transform baked into the input plan.
    pub fn detect(&self, img: &image::DynamicImage) -> Result<Vec<Detection>> {
        let rgb = RgbImage::from_dynamic(img);
        let plan = DetectorInputPlan::new(rgb.width(), rgb.height(), Some(736))?;
        let prepared = prepare_detector(&rgb, plan);
        let input = Tensor::from_f32(prepared.shape().to_vec(), prepared.data)?;
        let output = self.forward(input)?;
        // The output is [1, 1, H, W] — same H, W as the input (DB head preserves
        // spatial dims). extract_detections reads `values[y * width + x]`.
        let values: &[f32] = output.as_f32()?;
        let shape: &[usize] = output.shape();
        let opts = DetectorPostprocessOptions::default();
        crate::postprocess::extract_detections(values, shape, plan.transform(), opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loads the bundled tiny-det from the repo-root ppocr-models/ dir.
    /// Verifies the safetensors deserializes and the detector builds.
    #[test]
    fn load_from_buffer_builds_detector() {
        let bytes = std::fs::read("../../ppocr-models/tiny-det.safetensors")
            .expect("read bundled tiny-det (relative to crate manifest dir)");
        assert!(bytes.len() > 1_000_000, "tiny-det too small: {}", bytes.len());
        let det = Detector::load_from_buffer(&bytes)
            .expect("tiny-det loads from buffer");
        let _ = det; // constructed successfully
    }
}
