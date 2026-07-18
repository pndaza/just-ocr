//! OCR dispatcher: Kraken segmentation → per-line crop → recognizer.
//!
//! Both recognizers (Tesseract per-line, Kraken recognition) consume line
//! crops produced by the shared Kraken segmentation model and return text.
//! The result is a structured `OcrResult` carrying one `LineBox` per line.

use std::time::Instant;

use image::GenericImageView;
use serde::Serialize;
use tauri::Manager;

use crate::kraken::{self, KrakenCache};
use crate::OcrOpts;

/// One recognized line: an axis-aligned bbox (in source-image pixel space) and
/// the decoded text. The frontend overlays the bbox and joins `text` for display.
#[derive(Debug, Clone, Serialize)]
pub struct LineBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub text: String,
}

/// Structured OCR result. Text-mode and box-overlay rendering are both
/// projections of `lines`, so there is no separate "output mode".
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrResult {
    pub width: u32,
    pub height: u32,
    pub lines: Vec<LineBox>,
    /// Mean recognizer confidence in [0,100]. -1 when unknown (Kraken recog).
    pub confidence: i32,
    pub elapsed_ms: u64,
}

/// Entry point invoked by the `ocr_from_bytes` Tauri command.
pub fn run_ocr(
    app: &tauri::AppHandle,
    image_bytes: &[u8],
    opts: &OcrOpts,
) -> Result<OcrResult, String> {
    let started = Instant::now();

    let img = image::load_from_memory(image_bytes)
        .map_err(|e| format!("Failed to decode image: {e}"))?;
    let (w, h) = img.dimensions();

    let cache = app.state::<KrakenCache>();
    let lines = kraken::segment(app, &cache, &img)
        .map_err(|e| format!("Segmentation failed: {e}"))?;

    let mut boxes: Vec<LineBox> = Vec::with_capacity(lines.len());
    let mut conf_sum: i64 = 0;
    let mut conf_n: i32 = 0;

    for line in &lines {
        if line.boundary.len() < 3 {
            continue;
        }
        let (min_x, min_y, lw, lh) = match polygon_bbox((w, h), &line.boundary) {
            Some(b) => b,
            None => continue,
        };
        let crop = img.crop_imm(min_x, min_y, lw, lh);

        let (text, conf) = match opts.engine.as_str() {
            "tesseract" => crate::tesseract_line::recognize(
                &image::DynamicImage::ImageRgb8(crop.to_rgb8()),
                app,
                &opts.language,
                &opts.whitelist,
            )?,
            "kraken" => {
                let t = kraken::recognize_line(
                    app,
                    &cache,
                    &image::DynamicImage::ImageRgb8(crop.to_rgb8()),
                )
                .map_err(|e| format!("Recognition failed: {e}"))?;
                (t, -1)
            }
            other => return Err(format!("Unknown engine: {other}")),
        };

        if conf >= 0 {
            conf_sum += conf as i64;
            conf_n += 1;
        }

        boxes.push(LineBox {
            x0: min_x,
            y0: min_y,
            x1: min_x + lw,
            y1: min_y + lh,
            text,
        });
    }

    let confidence = if conf_n > 0 {
        (conf_sum / conf_n as i64) as i32
    } else {
        -1
    };

    Ok(OcrResult {
        width: w,
        height: h,
        lines: boxes,
        confidence,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// Axis-aligned bbox of a polygon, clamped to image bounds. Returns
/// `(min_x, min_y, width, height)` or `None` if the bbox is zero-area.
pub fn polygon_bbox(
    (img_w, img_h): (u32, u32),
    boundary: &[(f64, f64)],
) -> Option<(u32, u32, u32, u32)> {
    if boundary.is_empty() {
        return None;
    }
    let min_x = boundary
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min)
        .max(0.0) as u32;
    let min_y = boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min)
        .max(0.0) as u32;
    let max_x = boundary
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .min((img_w - 1) as f64) as u32;
    let max_y = boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .min((img_h - 1) as f64) as u32;
    let w = max_x.saturating_sub(min_x) + 1;
    let h = max_y.saturating_sub(min_y) + 1;
    if w == 0 || h == 0 {
        None
    } else {
        Some((min_x, min_y, w, h))
    }
}

#[cfg(test)]
mod tests {
    use super::polygon_bbox;

    #[test]
    fn polygon_bbox_basic() {
        let b = vec![(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)];
        assert_eq!(polygon_bbox((100, 100), &b), Some((10, 20, 21, 21)));
    }

    #[test]
    fn polygon_bbox_clamps_to_image() {
        // Points outside the image clamp inward at the high end.
        let b = vec![(-5.0, -5.0), (200.0, -5.0), (200.0, 200.0), (-5.0, 200.0)];
        assert_eq!(polygon_bbox((50, 50), &b), Some((0, 0, 50, 50)));
    }

    #[test]
    fn polygon_bbox_empty_returns_none() {
        assert_eq!(polygon_bbox((100, 100), &[]), None);
    }
}
