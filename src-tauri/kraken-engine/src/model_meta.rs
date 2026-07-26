//! Shared metadata types for kraken models.
//!
//! Extracted from kraken-rust's `model.rs` (which also holds the ONNX
//! `SegmentationModel` — not vendored here). These two structs are reused by
//! both the candle segmentation backend and the detection post-processing.

/// Class mapping: channel indices for aux / baselines / regions. Mirrors the
/// sidecar JSON written by `export_blla_onnx.py`. Iteration order matches a
/// `HashMap`, which is fine for the lookups the post-processing does.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassMapping {
    pub aux: std::collections::HashMap<String, usize>,
    pub baselines: std::collections::HashMap<String, usize>,
    pub regions: std::collections::HashMap<String, usize>,
}

/// Model metadata loaded from the sidecar JSON / safetensors header.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelMeta {
    pub class_mapping: ClassMapping,
    pub one_channel_mode: String,
    pub topline: bool,
    /// Padding as [left, right, top, bottom]. Defaults to [0,0,0,0].
    #[serde(default)]
    pub padding: Vec<i64>,
    pub bounding_regions: Option<Vec<String>>,
    /// VGSL input spec as [batch, channels, height, width] (NCHW). Height is
    /// parsed from the model's spec (1800 for the bundled bur_segment; 0 means
    /// variable-height upstream BLLA, which the Rust loader falls back to 1800
    /// for). Width is 0 (variable).
    #[serde(default)]
    pub input: Vec<i64>,
}
