//! OCR dispatcher: Kraken segmentation → per-line crop → recognizer.
//!
//! Both recognizers (Tesseract per-line, Kraken recognition) consume line
//! crops produced by the shared Kraken segmentation model and return text.
//! The result is a structured `OcrResult` carrying one `LineBox` per line.

use std::path::PathBuf;
use std::time::Instant;

use image::GenericImageView;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::Serialize;
use tauri::Manager;

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

/// Process-wide lazily-loaded kraken engine. The first OCR call pays the
/// (fast, ~1-3 ms) model-load cost; subsequent calls reuse the instance.
/// `kraken_engine::Engine` is `Send + Sync`, so a `&Engine` is safe to share
/// across the blocking-thread calls Tauri spawns per OCR request.
static KRAKEN: OnceCell<kraken_engine::Engine> = OnceCell::new();

/// Borrow the shared kraken engine, loading it on first call.
fn kraken_engine(app: &tauri::AppHandle) -> Result<&kraken_engine::Engine, String> {
    KRAKEN.get_or_try_init(|| {
        let (seg, rec) = resolve_kraken_models(app)?;
        let t = Instant::now();
        let engine = kraken_engine::Engine::load(&seg, &rec)
            .map_err(|e| format!("Kraken model load failed: {e}"))?;
        log::info!("[kraken] models loaded in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        Ok(engine)
    })
}

/// Resolve `(seg_path, rec_path)`. Looks in the app local data dir first, then
/// falls back to `kraken-models/` relative to the current working dir (the dev
/// workflow — the repo keeps these models at its root).
pub fn resolve_kraken_models(app: &tauri::AppHandle) -> Result<(PathBuf, PathBuf), String> {
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
    Err(format!(
        "Kraken models not found. Place bur_segment.safetensors and \
         bur_recog.safetensors in one of: {}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Entry point invoked by the `ocr_from_bytes` Tauri command.
///
/// Dispatches on language:
///   - `"mya"` → Kraken segmentation (hidden from the user) + per-line
///     recognition by the chosen engine ("kraken" | "tesseract"). Kraken's
///     layout is far better than Tesseract's for Myanmar script, so we always
///     use it regardless of which recognizer the user picked.
///   - any other language → full-page Tesseract with the user's PSM. Tesseract
///     does its own layout + recognition here; Kraken is not involved.
pub fn run_ocr(
    app: &tauri::AppHandle,
    image_bytes: &[u8],
    opts: &OcrOpts,
) -> Result<OcrResult, String> {
    let started = Instant::now();
    log::info!(
        "[ocr] language={} engine={} psm={} image={} bytes",
        opts.language,
        opts.engine,
        opts.psm,
        image_bytes.len()
    );

    let t = Instant::now();
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| format!("Failed to decode image: {e}"))?;
    let (w, h) = img.dimensions();
    log::info!(
        "[ocr] decode image {}x{}: {:.1} ms",
        w,
        h,
        t.elapsed().as_secs_f64() * 1000.0
    );

    // Myanmar: Kraken-driven pipeline.
    if opts.language == "mya" {
        return run_myanmar(app, &img, opts, w, h, started);
    }

    // Everything else: full-page Tesseract with the user's PSM.
    run_tesseract_page(app, &img, opts, w, h, started)
}

/// Myanmar pipeline: Kraken segmentation → per-line recognition.
fn run_myanmar(
    app: &tauri::AppHandle,
    img: &image::DynamicImage,
    opts: &OcrOpts,
    w: u32,
    h: u32,
    started: Instant,
) -> Result<OcrResult, String> {
    let engine = kraken_engine(app)?;

    let t = Instant::now();
    let lines = engine
        .segment(img)
        .map_err(|e| format!("Segmentation failed: {e}"))?;
    log::info!(
        "[ocr] segmentation (kraken): {:.0} ms ({} lines)",
        t.elapsed().as_secs_f64() * 1000.0,
        lines.len()
    );

    // Recognize each detected line. The Kraken recognizer (pure candle
    // tensors under `Arc<RwLock<Storage>>`) is `Send + Sync` and runs on the
    // rayon pool — one shared model across worker threads, no weight
    // duplication. Tesseract wraps libtesseract (a C library that is NOT
    // thread-safe across concurrent calls), so the tesseract engine stays
    // serial: each call constructs a fresh `TesseractAPI`, but they must not
    // overlap.
    let recog_start = Instant::now();
    let engine_kind = opts.engine.as_str();

    // Build the (LineBox, conf) pairs from each non-degenerate line. The
    // closure captures shared refs to img + engine + (for tesseract) the app
    // handle and opts — all Send + Sync.
    let recognize = |line: &kraken_engine::BaselineLine| -> Result<Option<(LineBox, i32)>, String> {
        if line.boundary.len() < 3 {
            return Ok(None);
        }
        let (min_x, min_y, lw, lh) = match polygon_bbox((w, h), &line.boundary) {
            Some(b) => b,
            None => return Ok(None),
        };
        let crop_img =
            image::DynamicImage::ImageRgb8(img.crop_imm(min_x, min_y, lw, lh).to_rgb8());

        let (text, conf) = match engine_kind {
            "tesseract" => crate::tesseract_line::recognize(
                &crop_img,
                app,
                &opts.language,
                &opts.whitelist,
            )?,
            "kraken" => {
                let t = engine
                    .recognize_line(&crop_img)
                    .map_err(|e| format!("Recognition failed: {e}"))?;
                (t, -1)
            }
            other => return Err(format!("Unknown engine: {other}")),
        };

        Ok(Some((
            LineBox {
                x0: min_x,
                y0: min_y,
                x1: min_x + lw,
                y1: min_y + lh,
                text,
            },
            conf,
        )))
    };

    // Dispatch: parallel for kraken, serial for tesseract.
    let results: Vec<(LineBox, i32)> = match engine_kind {
        "kraken" => lines
            .par_iter()
            .map(|line| recognize(line))
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .flatten()
            .collect(),
        _ => lines
            .iter()
            .map(|line| recognize(line))
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .flatten()
            .collect(),
    };

    let recog_n = results.len();
    log::info!(
        "[ocr] recognition: {:.0} ms ({} lines, {:.1} ms/line avg, {})",
        recog_start.elapsed().as_secs_f64() * 1000.0,
        recog_n,
        if recog_n > 0 {
            recog_start.elapsed().as_secs_f64() * 1000.0 / recog_n as f64
        } else {
            0.0
        },
        if engine_kind == "kraken" {
            format!("rayon {} threads", rayon::current_num_threads())
        } else {
            "serial".to_string()
        }
    );

    // Tally confidences and split out the boxes the frontend needs.
    let mut boxes: Vec<LineBox> = Vec::with_capacity(recog_n);
    let mut conf_sum: i64 = 0;
    let mut conf_n: i32 = 0;
    for (b, conf) in results {
        if conf >= 0 {
            conf_sum += conf as i64;
            conf_n += 1;
        }
        boxes.push(b);
    }

    let confidence = if conf_n > 0 {
        (conf_sum / conf_n as i64) as i32
    } else {
        -1
    };

    log::info!(
        "[ocr] TOTAL: {:.0} ms (myanmar/{} recognizer, {} boxes)",
        started.elapsed().as_secs_f64() * 1000.0,
        opts.engine,
        boxes.len()
    );

    Ok(OcrResult {
        width: w,
        height: h,
        lines: boxes,
        confidence,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// Non-Myanmar pipeline: full-page Tesseract with the user's PSM.
fn run_tesseract_page(
    app: &tauri::AppHandle,
    img: &image::DynamicImage,
    opts: &OcrOpts,
    w: u32,
    h: u32,
    started: Instant,
) -> Result<OcrResult, String> {
    let t = Instant::now();
    let (boxes, confidence) = crate::tesseract_page::recognize(
        img,
        app,
        &opts.language,
        opts.psm,
        &opts.whitelist,
    )?;
    log::info!(
        "[ocr] tesseract full-page (psm={}): {:.0} ms ({} lines, {}% conf)",
        opts.psm,
        t.elapsed().as_secs_f64() * 1000.0,
        boxes.len(),
        confidence
    );

    log::info!(
        "[ocr] TOTAL: {:.0} ms (tesseract, {} boxes)",
        started.elapsed().as_secs_f64() * 1000.0,
        boxes.len()
    );

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
