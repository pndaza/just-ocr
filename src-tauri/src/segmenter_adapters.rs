//! Adapters that wrap each vendored engine behind the host's [`Segmenter`]
//! trait. Each adapter owns the type-shape conversion (engine-native line
//! type → [`DetectedLine`]) so the recognizer path stays uniform.

use crate::segmentation::{DetectedLine, Segmenter};
use image::DynamicImage;

/// Wraps a shared [`kraken_engine::Engine`] as a [`Segmenter`]. Kraken's
/// `BaselineLine` already carries both the baseline polyline and the boundary
/// polygon, so this is a 1:1 field copy.
pub struct KrakenSegmenter {
    engine: std::sync::Arc<kraken_engine::Engine>,
}

impl KrakenSegmenter {
    pub fn new(engine: std::sync::Arc<kraken_engine::Engine>) -> Self {
        Self { engine }
    }
    /// Borrow the underlying engine (for recog when seg=ppocr but recog=kraken).
    pub fn engine(&self) -> &kraken_engine::Engine {
        &self.engine
    }
}

impl Segmenter for KrakenSegmenter {
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String> {
        let lines = self.engine.segment(img).map_err(|e| e.to_string())?;
        Ok(lines
            .into_iter()
            .map(|l| DetectedLine {
                baseline: l.baseline,
                boundary: l.boundary,
            })
            .collect())
    }
    fn name(&self) -> &'static str {
        "kraken"
    }
}
