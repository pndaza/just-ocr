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
mod tensor;
mod weights;

pub use model::{CpuOptions, Detector};
pub use postprocess::{Detection, DetectorTransform, Point};
pub use tensor::Tensor;
