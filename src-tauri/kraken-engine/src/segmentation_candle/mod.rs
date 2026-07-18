//! Segmentation model backend using candle-core (native Rust).
//!
//! An alternative to the ONNX-based `SegmentationModel` that loads kraken
//! segmentation models directly from `.safetensors` and runs the forward pass
//! via candle-core. This avoids the ONNX Runtime dependency entirely.

pub mod model;

pub use model::SegmentationModelCandle;
