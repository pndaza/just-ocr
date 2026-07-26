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
