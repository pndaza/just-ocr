//! OCR text recognition: decode text from line image crops.
//!
//! Port of kraken's recognition pipeline (`kraken/lib/vgsl/rpred.py`).
//!
//! Loads a kraken recognition model directly from `.safetensors` (no ONNX),
//! runs the forward pass via candle-core, and decodes the output via greedy
//! CTC + the model's codec.
//!
//! Architecture overview:
//!   - [`model`] — builds the VGSL network from safetensors weights
//!   - [`preprocess`] — line image → normalized tensor
//!   - [`decode`] — greedy CTC best-path decoding
//!   - [`codec`] — label → grapheme mapping
//!   - [`meta`] — safetensors metadata parsing

pub mod codec;
pub mod decode;
pub mod meta;
pub mod model;
pub mod preprocess;

#[cfg(test)]

pub use codec::Codec;
pub use model::RecognitionModel;
pub use preprocess::preprocess_line;
