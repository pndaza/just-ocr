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

use ppocr_engine::{Detection, Point};

/// Close a polygon by repeating the first point at the end (if not already
/// closed). Matches Kraken's convention so `polygon_bbox` and point-in-polygon
/// behave identically across segmenters.
fn close_polygon(poly: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if poly.len() < 2 {
        return poly.to_vec();
    }
    let mut out = poly.to_vec();
    if out.first() != out.last() {
        out.push(out[0]);
    }
    out
}

/// Synthesize a baseline (midline) for a 4-corner quad by averaging the top
/// and bottom edges. Returns `n` samples along the text axis (left → right).
///
/// Assumes the quad is ordered counter-clockwise from the top-left corner:
///   `[top_left, top_right, bottom_right, bottom_left]` — the order PaddleOCR's
///   DB postprocess produces (verified in ppocr-rs `fit_rotated_box`).
/// If the quad is rotated, the midline tracks the rotation.
fn synth_midline(quad: &[(f64, f64); 4], n: usize) -> Vec<(f64, f64)> {
    let [tl, tr, br, bl] = [quad[0], quad[1], quad[2], quad[3]];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let u = if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
        // Top edge: tl → tr. Bottom edge: bl → br.
        let top_x = tl.0 + (tr.0 - tl.0) * u;
        let top_y = tl.1 + (tr.1 - tl.1) * u;
        let bot_x = bl.0 + (br.0 - bl.0) * u;
        let bot_y = bl.1 + (br.1 - bl.1) * u;
        out.push(((top_x + bot_x) / 2.0, (top_y + bot_y) / 2.0));
    }
    out
}

/// Wraps a shared [`ppocr_engine::Detector`] as a [`Segmenter`]. Converts each
/// PP-OCR detection quad into a [`DetectedLine`] (closed boundary + synthesized
/// baseline). The boundary feeds Tesseract recog + overlay; the baseline feeds
/// Kraken recog dewarp (with graceful fallback if dewarp rejects it).
pub struct PPOcrSegmenter {
    detector: std::sync::Arc<ppocr_engine::Detector>,
}

impl PPOcrSegmenter {
    pub fn new(detector: std::sync::Arc<ppocr_engine::Detector>) -> Self {
        Self { detector }
    }
}

impl Segmenter for PPOcrSegmenter {
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String> {
        let detections = self.detector.detect(img).map_err(|e| e.to_string())?;
        log::info!("[ocr] ppocr detections: {}", detections.len());
        Ok(detections
            .into_iter()
            .filter_map(|d| detection_to_line(&d))
            .collect())
    }
    fn name(&self) -> &'static str {
        "ppocr-tiny"
    }
}

/// Convert a PP-OCR `Detection` (4-corner quad) to a `DetectedLine`. Returns
/// `None` if the quad is degenerate (wrong corner count).
fn detection_to_line(d: &Detection) -> Option<DetectedLine> {
    let quad: [(f64, f64); 4] = [
        (d.polygon[0].0 as f64, d.polygon[0].1 as f64),
        (d.polygon[1].0 as f64, d.polygon[1].1 as f64),
        (d.polygon[2].0 as f64, d.polygon[2].1 as f64),
        (d.polygon[3].0 as f64, d.polygon[3].1 as f64),
    ];
    let boundary = close_polygon(&quad);
    let baseline = synth_midline(&quad, 8);
    Some(DetectedLine { baseline, boundary })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_polygon_repeats_first_point() {
        let quad = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let closed = close_polygon(&quad);
        assert_eq!(closed.len(), 5);
        assert_eq!(closed[0], closed[4]);
    }

    #[test]
    fn synth_midline_averages_top_and_bottom_edges() {
        // Axis-aligned rectangle: top edge y=0, bottom edge y=4.
        // Midline should be at y=2 along x=0..4.
        let quad = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let mid = synth_midline(&quad, 5);
        assert_eq!(mid.len(), 5);
        // First sample (u=0): midline at (0, 2).
        assert!((mid[0].0 - 0.0).abs() < 1e-6 && (mid[0].1 - 2.0).abs() < 1e-6);
        // Last sample (u=1): midline at (4, 2).
        assert!((mid[4].0 - 4.0).abs() < 1e-6 && (mid[4].1 - 2.0).abs() < 1e-6);
        // Middle sample (u=0.5): midline at (2, 2).
        assert!((mid[2].0 - 2.0).abs() < 1e-6 && (mid[2].1 - 2.0).abs() < 1e-6);
    }
}
