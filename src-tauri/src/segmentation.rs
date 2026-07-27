//! Engine-agnostic segmentation abstraction. Both Kraken and PP-OCR segmenters
//! implement [`Segmenter`] so `run_myanmar` can hold either behind
//! `Arc<dyn Segmenter>` and call `segment()` uniformly.
//!
//! `DetectedLine` carries only the two fields the recognizer path consumes
//! (verified at engine.rs:208–234): a baseline polyline (Kraken recog dewarp)
//! and a closed boundary polygon (bbox, Tesseract crop, overlay, dewarp
//! fallback). It is deliberately distinct from `kraken_engine::BaselineLine`
//! so the host doesn't depend on Kraken's container type for the abstraction.

use image::DynamicImage;
use serde::Serialize;

/// One detected text line in source-image pixel coordinates.
#[derive(Debug, Clone, Serialize)]
pub struct DetectedLine {
    /// Midline polyline (left → right). Used by Kraken recog for dewarp.
    /// For PP-OCR, synthesized as the quad's vertical midline. May be empty
    /// if only the boundary matters (e.g. Tesseract recog only).
    pub baseline: Vec<(f64, f64)>,
    /// Closed boundary polygon (≥ 3 points). Used for bbox, Tesseract crop,
    /// overlay, and Kraken dewarp fallback. For PP-OCR: 4 corners + repeat-first.
    pub boundary: Vec<(f64, f64)>,
}

/// A text-line segmenter. Both vendored engines implement this so the host
/// dispatches uniformly.
pub trait Segmenter: Send + Sync {
    /// Segment the page image into detected text lines (source-image coords).
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String>;
    /// Human-readable name for logs (e.g. "kraken", "ppocr-tiny").
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detected_line_serializes_with_baseline_and_boundary() {
        let line = DetectedLine {
            baseline: vec![(1.0, 2.0), (3.0, 2.0)],
            boundary: vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)],
        };
        let json = serde_json::to_string(&line).expect("serialize");
        assert!(json.contains("\"baseline\""), "missing baseline in: {json}");
        assert!(json.contains("\"boundary\""), "missing boundary in: {json}");
    }
}
