//! Vendored kraken-rust candle backend: layout segmentation + line recognition.
//!
//! Source: /Users/pndaza/Projects/playground/kraken-rust (candle path only).
//! ONNX/ort modules, the CLI binary, and Python-fixture test harnesses were
//! not vendored. `use crate::` paths in the subtree have been rewritten to
//! `use crate::kraken::` so the tree is self-contained.

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

use std::path::PathBuf;

use anyhow::{Context, Result};
use image::DynamicImage;
use once_cell::sync::OnceCell;

use crate::kraken::config::SegmentationConfig;
use crate::kraken::containers::BaselineLine;
use crate::kraken::detect::detect_candle;
use crate::kraken::recognition::preprocess::preprocess_line;
use crate::kraken::recognition::RecognitionModel;
use crate::kraken::segmentation_candle::SegmentationModelCandle;

/// Cached loaded models. Both are `Send + Sync` (pure candle tensors under
/// `Arc<RwLock<Storage>>`), so a single instance is shared across OCR calls
/// by shared reference without a mutex.
pub struct KrakenCache {
    seg: OnceCell<SegmentationModelCandle>,
    rec: OnceCell<RecognitionModel>,
}

impl KrakenCache {
    pub fn new() -> Self {
        KrakenCache { seg: OnceCell::new(), rec: OnceCell::new() }
    }

    /// Lazily load + cache the segmentation model.
    pub fn segmenter(&self, path: &str) -> Result<&SegmentationModelCandle> {
        self.seg.get_or_try_init(|| {
            log::info!("Loading kraken segmentation model: {path}");
            SegmentationModelCandle::load(path)
        })
    }

    /// Lazily load + cache the recognition model.
    pub fn recognizer(&self, path: &str) -> Result<&RecognitionModel> {
        self.rec.get_or_try_init(|| {
            log::info!("Loading kraken recognition model: {path}");
            RecognitionModel::load(path)
        })
    }
}

impl Default for KrakenCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve `(seg_path, rec_path)`. Looks in the app local data dir first, then
/// falls back to `kraken-models/` relative to the current working dir (the dev
/// workflow — the repo keeps these models at its root).
pub fn resolve_models(app: &tauri::AppHandle) -> Result<(PathBuf, PathBuf)> {
    use tauri::Manager;

    let candidates: Vec<PathBuf> = match app.path().app_local_data_dir() {
        Ok(d) => vec![d.join("kraken-models"), PathBuf::from("kraken-models")],
        Err(_) => vec![PathBuf::from("kraken-models")],
    };

    for dir in &candidates {
        let seg = dir.join("bur_segment.safetensors");
        let rec = dir.join("bur_recog.safetensors");
        if seg.exists() && rec.exists() {
            return Ok((seg, rec));
        }
    }
    anyhow::bail!(
        "Kraken models not found. Place bur_segment.safetensors and \
         bur_recog.safetensors in one of: {}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Run layout segmentation on a page image. Returns the detected lines (each
/// carries its baseline polyline + boundary polygon).
pub fn segment(
    app: &tauri::AppHandle,
    cache: &KrakenCache,
    img: &DynamicImage,
) -> Result<Vec<BaselineLine>> {
    use std::time::Instant;

    let (seg_path, _rec_path) = resolve_models(app)?;

    let t = Instant::now();
    let model = cache.segmenter(&seg_path.to_string_lossy())?;
    let model_load_ms = t.elapsed().as_secs_f64() * 1000.0;
    if model_load_ms > 1.0 {
        // Only log when the model actually had to load (cold start). Cached
        // calls return in microseconds and would just spam the log.
        log::info!("[kraken] loaded segmentation model in {:.0} ms", model_load_ms);
    }

    let config = SegmentationConfig {
        text_direction: "horizontal-lr".to_string(),
    };
    let t = Instant::now();
    let result = detect_candle(img, model, &config).context("Kraken segmentation failed")?;
    log::debug!(
        "[kraken] segment: {:.0} ms ({} lines)",
        t.elapsed().as_secs_f64() * 1000.0,
        result.lines.len()
    );
    Ok(result.lines)
}

/// Recognize text from a single pre-cropped line image.
pub fn recognize_line(
    app: &tauri::AppHandle,
    cache: &KrakenCache,
    crop: &DynamicImage,
) -> Result<String> {
    use std::time::Instant;

    let (_seg_path, rec_path) = resolve_models(app)?;

    let t = Instant::now();
    let model = cache.recognizer(&rec_path.to_string_lossy())?;
    let model_load_ms = t.elapsed().as_secs_f64() * 1000.0;
    if model_load_ms > 1.0 {
        log::info!("[kraken] loaded recognition model in {:.0} ms", model_load_ms);
    }

    let t = Instant::now();
    let tensor = preprocess_line(crop, model.height, model.padding)?;
    let text = model.recognize(&tensor).context("Kraken recognition failed")?;
    log::debug!(
        "[kraken] recognize_line: {:.1} ms ({}x{}px → {} chars)",
        t.elapsed().as_secs_f64() * 1000.0,
        crop.width(),
        crop.height(),
        text.chars().count()
    );
    Ok(text)
}
