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

/// One recognized line: an axis-aligned bbox (in source-image pixel space),
/// the decoded text, and — for the Kraken-segmented path — the true boundary
/// polygon. The frontend overlays the polygon when present and falls back to
/// the bbox otherwise.
#[derive(Debug, Clone, Serialize)]
pub struct LineBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub text: String,
    /// True boundary polygon from Kraken segmentation (source-image pixel
    /// space). Present only for the Kraken-segmented (Myanmar) path; `None`
    /// for the Tesseract full-page path, which produces bboxes, not polygons.
    /// Skipped on serialization when `None` so the Tesseract-path wire format
    /// is byte-identical to pre-polygon builds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<[f64; 2]>>,
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
    /// Per-stage timing for the Kraken-segmented (Myanmar) path, where seg and
    /// recog are distinct passes. `None` on the full-page Tesseract path, which
    /// does both in one `get_hocr_text` call — there is no separate measurement
    /// to report and we don't want to fabricate one. Skipped on serialization
    /// when `None` so the Tesseract-path wire format stays minimal.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segmentation_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recognition_ms: Option<u64>,
}

/// Bundled Kraken models, embedded in the binary at compile time via
/// `include_bytes!`. A fresh install works with zero setup — the user does
/// not need to place any model files. Bytes are `&'static`, so they outlive
/// any `Engine` that loads them.
///
/// Path is relative to `src-tauri/src/` (this file's directory). The
/// `kraken-models/` directory sits at the repo root, two levels up.
static BUNDLED_SEG: &[u8] = include_bytes!("../../kraken-models/bur_segment.safetensors");
static BUNDLED_REC: &[u8] = include_bytes!("../../kraken-models/bur_recog.safetensors");

/// Bundled PP-OCRv6 tiny detector. Same `include_bytes!` pattern as the
/// Kraken models. Path is relative to `src-tauri/src/` (this file's dir).
static BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/tiny-det.safetensors");

/// Process-wide lazily-loaded kraken engine, wrapped in `Arc` so it can be
/// shared with `KrakenSegmenter` (which holds `Arc<Engine>` to satisfy the
/// `'static` requirement of `Arc<dyn Segmenter>`). The first OCR call pays
/// the (fast, ~1-3 ms) model-load cost; subsequent calls reuse the instance.
/// `kraken_engine::Engine` is `Send + Sync`, so a `&Engine` is safe to share
/// across the blocking-thread calls Tauri spawns per OCR request.
static KRAKEN: OnceCell<std::sync::Arc<kraken_engine::Engine>> = OnceCell::new();

/// Process-wide lazily-loaded PP-OCR detector, same Arc-wrapped shape.
static PPOCR: OnceCell<std::sync::Arc<ppocr_engine::Detector>> = OnceCell::new();

/// Borrow the shared kraken engine, loading it on first call.
///
/// Resolution order:
///   1. **Override** — if the user has placed `bur_segment.safetensors` +
///      `bur_recog.safetensors` in the platform app-data dir, load those.
///      Lets power users swap models without an app rebuild.
///   2. **Bundled** — fall back to the models embedded in the binary via
///      `include_bytes!`. The default for fresh installs.
fn kraken_engine(app: &tauri::AppHandle) -> Result<&std::sync::Arc<kraken_engine::Engine>, String> {
    KRAKEN.get_or_try_init(|| {
        let t = Instant::now();
        let engine = match resolve_override_models(app) {
            Some((seg_path, rec_path)) => {
                log::info!(
                    "[kraken] using override models from {}",
                    seg_path.parent().unwrap_or(std::path::Path::new(".")).display()
                );
                kraken_engine::Engine::load(&seg_path, &rec_path)
                    .map_err(|e| format!("Kraken override load failed: {e}"))?
            }
            None => kraken_engine::Engine::load_from_buffers(BUNDLED_SEG, BUNDLED_REC)
                .map_err(|e| format!("Kraken bundled load failed: {e}"))?,
        };
        log::info!(
            "[kraken] models loaded in {:.0} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
        Ok(std::sync::Arc::new(engine))
    })
}

/// Look for user-supplied override models in the platform app-data dir.
/// Returns `Some((seg_path, rec_path))` only if BOTH files exist there —
/// partial overrides are ignored to avoid mixing model versions.
fn resolve_override_models(app: &tauri::AppHandle) -> Option<(PathBuf, PathBuf)> {
    let dir = app.path().app_local_data_dir().ok()?.join("kraken-models");
    let seg = dir.join("bur_segment.safetensors");
    let rec = dir.join("bur_recog.safetensors");
    if seg.exists() && rec.exists() {
        Some((seg, rec))
    } else {
        None
    }
}

/// Load the PP-OCR detector (bundled or override). Returns `Arc<Detector>`.
fn load_ppocr(app: &tauri::AppHandle) -> Result<std::sync::Arc<ppocr_engine::Detector>, String> {
    let t = Instant::now();
    let det = match resolve_override_ppocr(app) {
        Some(path) => {
            log::info!("[ppocr] using override model from {}", path.display());
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("PP-OCR override read failed: {e}"))?;
            ppocr_engine::Detector::load_from_buffer(&bytes)
                .map_err(|e| format!("PP-OCR override load failed: {e}"))?
        }
        None => ppocr_engine::Detector::load_from_buffer(BUNDLED_PPOCR_DET)
            .map_err(|e| format!("PP-OCR bundled load failed: {e}"))?,
    };
    log::info!("[ppocr] det loaded in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    Ok(std::sync::Arc::new(det))
}

/// User-supplied PP-OCR override: a single `tiny-det.safetensors` in the
/// platform app-data dir's `ppocr-models/` subdir. Returns `Some(path)` only
/// if the file exists. (Unlike kraken's two-file rule, PP-OCR is one file.)
fn resolve_override_ppocr(app: &tauri::AppHandle) -> Option<PathBuf> {
    let dir = app.path().app_local_data_dir().ok()?.join("ppocr-models");
    let det = dir.join("tiny-det.safetensors");
    if det.exists() { Some(det) } else { None }
}

/// Resolve the segmenter for this OCR call. Choices:
///   - `opts.segmenter == Some("kraken")` → `KrakenSegmenter` (lazy-loads Kraken)
///   - anything else (including `None`) → `PPOcrSegmenter` (lazy-loads PP-OCR det)
///
/// PP-OCR is the default (faster, generalizes well). Kraken remains available
/// as an explicit opt-in for cases where its baseline-aware segmentation
/// outperforms PP-OCR's quad detection. Unknown strings warn and fall back to
/// the PP-OCR default.
///
/// Returns `Arc<dyn Segmenter>` so `run_myanmar` holds a uniform type.
fn resolve_segmenter(
    app: &tauri::AppHandle,
    opts: &OcrOpts,
) -> Result<std::sync::Arc<dyn crate::segmentation::Segmenter>, String> {
    use crate::segmenter_adapters::{KrakenSegmenter, PPOcrSegmenter};
    match opts.segmenter.as_deref() {
        Some("kraken") => {
            let eng = KRAKEN.get_or_try_init(|| kraken_engine(app).cloned())?.clone();
            Ok(std::sync::Arc::new(KrakenSegmenter::new(eng)))
        }
        Some("ppocr") | None => {
            let det = PPOCR.get_or_try_init(|| load_ppocr(app))?.clone();
            Ok(std::sync::Arc::new(PPOcrSegmenter::new(det)))
        }
        Some(other) => {
            log::warn!("[ocr] unknown segmenter {other:?}, falling back to ppocr");
            let det = PPOCR.get_or_try_init(|| load_ppocr(app))?.clone();
            Ok(std::sync::Arc::new(PPOcrSegmenter::new(det)))
        }
    }
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
    let segmenter = resolve_segmenter(app, opts)?;
    let seg_name = segmenter.name();

    let t = Instant::now();
    let lines = segmenter
        .segment(img)
        .map_err(|e| format!("Segmentation failed: {e}"))?;
    let segmentation_ms = t.elapsed().as_millis() as u64;
    log::info!(
        "[ocr] segmentation ({}): {:.0} ms ({} lines)",
        seg_name,
        segmentation_ms as f64,
        lines.len()
    );

    // If recog is Kraken, we need a Kraken engine handle regardless of which
    // segmenter produced the lines. Lazy-load it (shares the OnceCell with
    // KrakenSegmenter — no double-load).
    let kraken_rec_engine: Option<&kraken_engine::Engine> = if opts.engine == "kraken" {
        Some(kraken_engine(app)?.as_ref())
    } else {
        None
    };

    // Recognize each detected line. The Kraken recognizer (pure candle
    // tensors under `Arc<RwLock<Storage>>`) is `Send + Sync` and runs on the
    // rayon pool — one shared model across worker threads, no weight
    // duplication. Tesseract wraps libtesseract (a C library that is NOT
    // thread-safe across concurrent calls), so the tesseract engine stays
    // serial: each call constructs a fresh `TesseractAPI`, but they must not
    // overlap.
    let recog_start = Instant::now();
    let engine_kind = opts.engine.as_str();

    // Parse the binarize option once (Myanmar/Kraken path only). Tesseract
    // does its own internal binarization and ignores this.
    let binarize = opts.binarize.as_deref().and_then(|s| match s {
        "otsu" => Some(kraken_engine::recognition::Binarization::Otsu),
        "sauvola" => Some(kraken_engine::recognition::Binarization::Sauvola),
        _ => None,
    });

    // Build the (LineBox, conf) pairs from each non-degenerate line. The
    // closure captures shared refs to img + engine + (for tesseract) the app
    // handle and opts — all Send + Sync.
    let recognize = |line: &crate::segmentation::DetectedLine| -> Result<Option<(LineBox, i32)>, String> {
        if line.boundary.len() < 3 {
            return Ok(None);
        }
        let (min_x, min_y, lw, lh) = match polygon_bbox((w, h), &line.boundary) {
            Some(b) => b,
            None => return Ok(None),
        };

        let (text, conf) = match engine_kind {
            "tesseract" => {
                // Tesseract operates on the masked bbox crop (no dewarp).
                let crop_img = kraken_engine::crop_polygon_white_bg(img, &line.boundary);
                crate::tesseract_line::recognize(
                    &crop_img,
                    app,
                    &opts.language,
                    &opts.whitelist,
                )?
            }
            // Kraken: dewarp (polygon mask + baseline straightening) then
            // recognize. extract_polygon_line operates on the full page image
            // + the line's baseline + boundary, producing a flat strip that
            // the Stage-2 centerline normalizer and LSTM consume. Falls back
            // to a masked bbox crop inside the engine if the dewarp fails.
            "kraken" => {
                // Safe unwrap: kraken_rec_engine is Some iff engine_kind == "kraken".
                let eng = kraken_rec_engine.expect("kraken engine loaded for kraken recog");
                let t = eng
                    .recognize_line_dewarped(img, &line.baseline, &line.boundary, binarize)
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
                polygon: Some(line.boundary.iter().map(|p| [p.0, p.1]).collect()),
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
    let recognition_ms = recog_start.elapsed().as_millis() as u64;
    log::info!(
        "[ocr] recognition: {:.0} ms ({} lines, {:.1} ms/line avg, {})",
        recognition_ms as f64,
        recog_n,
        if recog_n > 0 {
            recognition_ms as f64 / recog_n as f64
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
        segmentation_ms: Some(segmentation_ms),
        recognition_ms: Some(recognition_ms),
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
        // Tesseract full-page does layout + recognition in one call; there is
        // no per-stage measurement to surface. The status bar shows total only.
        segmentation_ms: None,
        recognition_ms: None,
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
    use super::{polygon_bbox, LineBox, BUNDLED_REC, BUNDLED_SEG};

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

    /// Confirm the bundled bytes were actually embedded (non-empty) and are
    /// valid safetensors by loading both models from them. This is the test
    /// that proves `include_bytes!` bundling works end-to-end.
    #[test]
    fn bundled_models_load_from_buffers() {
        // Sanity: bytes are present.
        assert!(BUNDLED_SEG.len() > 1_000_000, "seg model too small: {}", BUNDLED_SEG.len());
        assert!(BUNDLED_REC.len() > 1_000_000, "rec model too small: {}", BUNDLED_REC.len());

        let engine = kraken_engine::Engine::load_from_buffers(BUNDLED_SEG, BUNDLED_REC)
            .expect("bundled models should load from buffers");
        // Smoke: recognizer input height was parsed from the VGSL spec (not
        // hardcoded — the parser is the source of truth, exercised directly in
        // recognition::meta::tests). We only assert it's populated and sane so
        // this test doesn't spuriously fail when the bundled model is swapped
        // (e.g. bur_recog is 48, other models differ).
        assert!(
            engine.recognizer().height > 0,
            "recognizer height should be parsed from the VGSL spec"
        );
    }

    /// Confirm the bundled PP-OCR tiny-det bytes are non-empty and load into
    /// a `Detector`. Mirrors `bundled_models_load_from_buffers` for kraken.
    #[test]
    fn bundled_ppocr_det_loads_from_buffer() {
        assert!(
            super::BUNDLED_PPOCR_DET.len() > 1_000_000,
            "ppocr det too small: {}",
            super::BUNDLED_PPOCR_DET.len()
        );
        let det = ppocr_engine::Detector::load_from_buffer(super::BUNDLED_PPOCR_DET)
            .expect("bundled ppocr det loads from buffer");
        let _ = det;
    }

    #[test]
    fn linebox_without_polygon_omits_field() {
        let lb = LineBox {
            x0: 1,
            y0: 2,
            x1: 3,
            y1: 4,
            text: "hi".to_string(),
            polygon: None,
        };
        let json = serde_json::to_string(&lb).unwrap();
        assert!(
            !json.contains("polygon"),
            "polygon field must be absent when None, got: {json}"
        );
    }

    #[test]
    fn linebox_with_polygon_includes_field() {
        let lb = LineBox {
            x0: 1,
            y0: 2,
            x1: 3,
            y1: 4,
            text: "hi".to_string(),
            polygon: Some(vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        };
        let json = serde_json::to_string(&lb).unwrap();
        assert!(
            json.contains("\"polygon\":[[1.0,2.0],[3.0,4.0],[5.0,6.0]]"),
            "polygon field must serialize as array-of-pairs, got: {json}"
        );
    }
}
