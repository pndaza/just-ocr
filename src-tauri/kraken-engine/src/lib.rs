//! kraken-engine: vendored kraken-rust candle backend (layout segmentation +
//! line recognition), packaged as its own crate so the host Tauri app can
//! optimize it in dev builds via `[profile.dev.package."*"] opt-level = 3`.
//!
//! Source: /Users/pndaza/Projects/playground/kraken-rust (candle path only).
//! ONNX/ort modules, the CLI binary, and Python-fixture test harnesses were
//! excluded. `use crate::` paths resolve within this crate.
//!
//! Public API:
//!   - [`Engine`] — owns the loaded seg + recog models, reused across calls.
//!   - [`Engine::load`] — cold-load both models from safetensors paths.
//!   - [`Engine::segment`] — run layout detection on a page image.
//!   - [`Engine::recognize_line`] — recognize a single line crop.

pub mod boundaries;
pub mod config;
pub mod containers;
pub mod contours;
pub mod detect;
pub mod heatmap;
pub mod inference_candle;
pub mod inference_helpers;
pub mod model_meta;
pub mod ndimage;
pub mod polygon;
pub mod preprocess;
pub mod reading_order;
pub mod recognition;
pub mod segmentation_candle;
pub mod vectorize;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use image::DynamicImage;

pub use config::SegmentationConfig;
pub use containers::{BaselineLine, Region, Segmentation};
pub use detect::{detect_candle, postprocess};
pub use recognition::{crop::crop_polygon_white_bg, preprocess::preprocess_line, RecognitionModel};
pub use segmentation_candle::SegmentationModelCandle;

/// Loaded kraken models, reused across OCR calls. Both are `Send + Sync`
/// (pure candle tensors under `Arc<RwLock<Storage>>`), so one `Engine` can be
/// shared (e.g. wrapped in `Arc` or `tauri::State`) and called from any thread.
///
/// Models are loaded eagerly at construction so the first OCR call doesn't
/// pay the load cost. Load is fast regardless (~1-3 ms each for these
/// safetensors), but doing it once makes per-call timings honest.
pub struct Engine {
    seg: Arc<SegmentationModelCandle>,
    rec: Arc<RecognitionModel>,
}

impl Engine {
    /// Load both models from disk.
    ///
    /// Used when the user has supplied override models in the app data dir;
    /// for the default bundled-models path see [`Engine::load_from_buffers`].
    pub fn load(seg_path: &Path, rec_path: &Path) -> Result<Self> {
        log::info!(
            "Loading kraken models from disk: seg={}, rec={}",
            seg_path.display(),
            rec_path.display()
        );
        let seg = SegmentationModelCandle::load(&seg_path.to_string_lossy())
            .with_context(|| format!("Failed to load seg model: {}", seg_path.display()))?;
        let rec = RecognitionModel::load(&rec_path.to_string_lossy())
            .with_context(|| format!("Failed to load rec model: {}", rec_path.display()))?;
        Ok(Engine {
            seg: Arc::new(seg),
            rec: Arc::new(rec),
        })
    }

    /// Load both models from in-memory safetensors buffers.
    ///
    /// Used to bundle the models into the binary via `include_bytes!` so a
    /// fresh install works with zero setup. The bytes must remain valid for
    /// the lifetime of the engine — `&'static [u8]` from `include_bytes!`
    /// satisfies this naturally.
    pub fn load_from_buffers(seg_bytes: &[u8], rec_bytes: &[u8]) -> Result<Self> {
        log::info!("Loading bundled kraken models from binary");
        let seg = SegmentationModelCandle::load_from_buffer(seg_bytes)
            .context("Failed to load bundled seg model")?;
        let rec = RecognitionModel::load_from_buffer(rec_bytes)
            .context("Failed to load bundled rec model")?;
        Ok(Engine {
            seg: Arc::new(seg),
            rec: Arc::new(rec),
        })
    }

    /// Run layout segmentation on a page image. Returns the detected lines
    /// (each carries its baseline polyline + boundary polygon).
    pub fn segment(&self, img: &DynamicImage) -> Result<Vec<BaselineLine>> {
        let config = SegmentationConfig {
            text_direction: "horizontal-lr".to_string(),
        };
        let result =
            detect_candle(img, &self.seg, &config).context("Kraken segmentation failed")?;
        Ok(result.lines)
    }

    /// Recognize text from a single pre-cropped line image.
    ///
    /// When `binarize` is `Some(method)`, the crop is binarized before
    /// recognition (for models trained on 1-bit images).
    pub fn recognize_line(
        &self,
        crop: &DynamicImage,
        binarize: Option<recognition::Binarization>,
    ) -> Result<String> {
        let tensor =
            preprocess_line(crop, self.rec.height, self.rec.padding, binarize, self.rec.center_norm)?;
        self.rec
            .recognize(&tensor)
            .context("Kraken recognition failed")
    }

    /// Recognize text from a single line, dewarping it first.
    ///
    /// Runs the Stage-1 geometric dewarp (`extract_polygon_line`: polygon mask
    /// + baseline straightening) on the full page image, then preprocesses the
    /// flat strip (Stage-2 centerline normalization when the model requests it)
    /// and recognizes it. This is the faithful kraken path and supersedes
    /// [`recognize_line`](Self::recognize_line) for curved/tilted lines.
    ///
    /// Falls back to a plain masked bbox crop if the dewarp fails (degenerate
    /// baseline), so it never hard-errors on an unusual line.
    pub fn recognize_line_dewarped(
        &self,
        image: &DynamicImage,
        baseline: &[(f64, f64)],
        boundary: &[(f64, f64)],
        binarize: Option<recognition::Binarization>,
    ) -> Result<String> {
        let strip =
            recognition::dewarp::extract_polygon_line(image, baseline, boundary).unwrap_or_else(
                |_| {
                    // Fallback: masked bbox crop (no dewarp), as a GrayImage strip.
                    let crop = recognition::crop::crop_polygon_white_bg(image, boundary);
                    crop.to_luma8()
                },
            );
        let strip_dyn = DynamicImage::ImageLuma8(strip);
        self.recognize_line(&strip_dyn, binarize)
    }

    /// Borrow the recognition model directly (e.g. for rayon-parallel batch
    /// recognition — `RecognitionModel` is `Send + Sync`).
    pub fn recognizer(&self) -> &RecognitionModel {
        &self.rec
    }
}
