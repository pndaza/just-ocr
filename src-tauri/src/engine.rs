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

/// Bundled PP-OCRv6 **small** detector (~9.9MB, 2.4M params). Wider channels
/// than tiny (PPLCNetV4-Large stem [48,96,192,384] vs [32,48,64,160], neck 96
/// vs 64) → a more accurate score map on dense/curved text. Measured: tiny
/// over-detects 44 vs small's correct 27 on heavy_curve_02 (ground truth 27),
/// so small is the recommended default for accuracy-sensitive use. Same
/// `include_bytes!` pattern as the Kraken models; path relative to
/// `src-tauri/src/` (this file's dir).
static BUNDLED_PPOCR_DET_SMALL: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");

/// Bundled PP-OCRv6 **tiny** detector (~1.8MB, 0.4M params). Narrower channels
/// ([32,48,64,160] stem, neck 64) → faster but noticeably less accurate on
/// dense/curved Burmese. Offered as a user-selectable variant for the
/// Myanmar/PP-OCR path (faster on low-end machines, or where small's accuracy
/// gain isn't needed). Loads via `DetectorConfig::tiny()`.
static BUNDLED_PPOCR_DET_TINY: &[u8] = include_bytes!("../../ppocr-models/tiny-det.safetensors");

/// Process-wide lazily-loaded kraken engine, wrapped in `Arc` so it can be
/// shared with `KrakenSegmenter` (which holds `Arc<Engine>` to satisfy the
/// `'static` requirement of `Arc<dyn Segmenter>`). The first OCR call pays
/// the (fast, ~1-3 ms) model-load cost; subsequent calls reuse the instance.
/// `kraken_engine::Engine` is `Send + Sync`, so a `&Engine` is safe to share
/// across the blocking-thread calls Tauri spawns per OCR request.
static KRAKEN: OnceCell<std::sync::Arc<kraken_engine::Engine>> = OnceCell::new();

/// PP-OCR detector variant selectable by the user from Settings. `Small` is
/// the accuracy-oriented default (wider backbone); `Tiny` is the fast/compact
/// alternative. The variant only matters for the PP-OCR segmenters; Kraken
/// ignores it. Both weights are bundled (see `BUNDLED_PPOCR_DET_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetVariant {
    Small,
    Tiny,
}

impl DetVariant {
    /// Parse the frontend-supplied option string. `None`/unrecognized →
    /// `Small` (the default) with a warning, so a stale/old frontend still
    /// gets the recommended variant rather than failing.
    fn from_opt_str(s: &Option<String>) -> Self {
        match s.as_deref() {
            Some("tiny") => DetVariant::Tiny,
            Some("small") => DetVariant::Small,
            Some(other) => {
                log::warn!("[ocr] unknown detVariant {other:?}, falling back to small");
                DetVariant::Small
            }
            None => DetVariant::Small,
        }
    }

    /// Bundled weights bytes for this variant.
    const fn bundled(&self) -> &'static [u8] {
        match self {
            DetVariant::Small => BUNDLED_PPOCR_DET_SMALL,
            DetVariant::Tiny => BUNDLED_PPOCR_DET_TINY,
        }
    }

    /// Matching `DetectorConfig` (channel widths) for this variant's weights.
    const fn config(&self) -> ppocr_engine::DetectorConfig {
        match self {
            DetVariant::Small => ppocr_engine::DetectorConfig::small(),
            DetVariant::Tiny => ppocr_engine::DetectorConfig::tiny(),
        }
    }

    /// Override filename in the app-data `ppocr-models/` dir.
    const fn override_filename(&self) -> &'static str {
        match self {
            DetVariant::Small => "small-det.safetensors",
            DetVariant::Tiny => "tiny-det.safetensors",
        }
    }

    const fn as_str(&self) -> &'static str {
        match self {
            DetVariant::Small => "small",
            DetVariant::Tiny => "tiny",
        }
    }
}

/// Process-wide lazily-loaded PP-OCR detectors, one per variant. A single
/// `OnceCell` is write-once, so to let the user switch variants at runtime
/// without an app restart we keep one cell per variant — each lazy-loaded on
/// first use, both able to coexist. The active variant is chosen per OCR call
/// from `opts.det_variant`; the other only loads if/when selected. Both the
/// quad and poly segmenters borrow whichever `Arc<Detector>` matches the
/// selected variant.
static PPOCR_SMALL: OnceCell<std::sync::Arc<ppocr_engine::Detector>> = OnceCell::new();
static PPOCR_TINY: OnceCell<std::sync::Arc<ppocr_engine::Detector>> = OnceCell::new();

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

/// Load the requested PP-OCR detector variant (bundled or override), built
/// with the matching [`DetectorConfig`] (`small()`/`tiny()`) so the backbone
/// widths match the weights. Returns `Arc<Detector>`, shared by both the quad
/// and poly segmenters via the variant's dedicated `OnceCell`.
///
/// Override resolution is per-variant: the override file is
/// `small-det.safetensors` / `tiny-det.safetensors` in the platform app-data
/// dir's `ppocr-models/` subdir (one-file override — unlike Kraken's
/// two-file rule). A user can drop in either or both.
fn load_ppocr(
    app: &tauri::AppHandle,
    variant: DetVariant,
) -> Result<std::sync::Arc<ppocr_engine::Detector>, String> {
    let t = Instant::now();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let det = match resolve_override_ppocr(app, variant) {
        Some(path) => {
            log::info!(
                "[ppocr] using {} override model from {}",
                variant.as_str(),
                path.display()
            );
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("PP-OCR override read failed: {e}"))?;
            ppocr_engine::Detector::load_from_buffer_with_config(
                &bytes,
                threads,
                variant.config(),
            )
            .map_err(|e| format!("PP-OCR override load failed: {e}"))?
        }
        None => ppocr_engine::Detector::load_from_buffer_with_config(
            variant.bundled(),
            threads,
            variant.config(),
        )
        .map_err(|e| format!("PP-OCR bundled load failed: {e}"))?,
    };
    log::info!(
        "[ppocr] {} det loaded in {:.0} ms",
        variant.as_str(),
        t.elapsed().as_secs_f64() * 1000.0
    );
    Ok(std::sync::Arc::new(det))
}

/// User-supplied PP-OCR override for a specific variant: a single
/// `small-det.safetensors` / `tiny-det.safetensors` in the platform app-data
/// dir's `ppocr-models/` subdir. Returns `Some(path)` only if the file
/// exists. (Unlike kraken's two-file rule, PP-OCR is one file per variant.)
fn resolve_override_ppocr(app: &tauri::AppHandle, variant: DetVariant) -> Option<PathBuf> {
    let dir = app.path().app_local_data_dir().ok()?.join("ppocr-models");
    let det = dir.join(variant.override_filename());
    if det.exists() { Some(det) } else { None }
}

/// Resolve the segmenter for this OCR call. Choices:
///   - `"kraken"` → `KrakenSegmenter` (lazy-loads Kraken; ignores `det_variant`)
///   - `"ppocr-poly"` → `PPOcrPolySegmenter` backed by the PP-OCR detector
///     selected via `opts.det_variant` + multi-point polygon postprocess
///     (contour → simplify → unclip)
///   - `"ppocr"` or `None` → `PPOcrSegmenter` backed by the PP-OCR detector
///     selected via `opts.det_variant` + rigid 4-corner quad postprocess
///
/// PP-OCR (quad) is the default segmenter; PP-OCR (poly) is the opt-in for
/// dense/curved Burmese where the polygon mask + curvature-gated dewarp help.
/// Both PP-OCR paths honor `det_variant` (`small` default / `tiny`) — each
/// variant has its own `OnceCell`, so switching variants at runtime lazily
/// loads the other without unloading the first. Unknown segmenter strings
/// warn and fall back to the PP-OCR quad default.
///
/// Returns `Arc<dyn Segmenter>` so `run_myanmar` holds a uniform type.
fn resolve_segmenter(
    app: &tauri::AppHandle,
    opts: &OcrOpts,
) -> Result<std::sync::Arc<dyn crate::segmentation::Segmenter>, String> {
    use crate::segmenter_adapters::{KrakenSegmenter, PPOcrPolySegmenter, PPOcrSegmenter};
    match opts.segmenter.as_deref() {
        Some("kraken") => {
            let eng = KRAKEN.get_or_try_init(|| kraken_engine(app).cloned())?.clone();
            Ok(std::sync::Arc::new(KrakenSegmenter::new(eng)))
        }
        Some("ppocr-poly") => {
            // Multi-point polygon postprocess + curvature-gated dewarp
            // (see recognize_line_poly). Detector variant is user-selectable.
            let det = ppocr_detector(app, opts)?;
            Ok(std::sync::Arc::new(PPOcrPolySegmenter::new(det)))
        }
        Some("ppocr") | None => {
            let det = ppocr_detector(app, opts)?;
            Ok(std::sync::Arc::new(PPOcrSegmenter::new(det)))
        }
        Some(other) => {
            log::warn!("[ocr] unknown segmenter {other:?}, falling back to ppocr");
            let det = ppocr_detector(app, opts)?;
            Ok(std::sync::Arc::new(PPOcrSegmenter::new(det)))
        }
    }
}

/// Borrow the PP-OCR detector for the variant selected in `opts.det_variant`,
/// lazy-loading (bundled or override) on first use of that variant. Each
/// variant lives in its own `OnceCell` so both can coexist and the active one
/// is chosen per call without rebuilding. Returns a fresh `Arc` clone (cheap).
fn ppocr_detector(
    app: &tauri::AppHandle,
    opts: &OcrOpts,
) -> Result<std::sync::Arc<ppocr_engine::Detector>, String> {
    let variant = DetVariant::from_opt_str(&opts.det_variant);
    let cell = match variant {
        DetVariant::Small => &PPOCR_SMALL,
        DetVariant::Tiny => &PPOCR_TINY,
    };
    Ok(cell.get_or_try_init(|| load_ppocr(app, variant))?.clone())
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

    // Recover column-aware reading order. Both segmenters emit a flat
    // y-sorted line list, which interleaves multi-column pages line by
    // line; this geometric pass (see `reading_order`) reorders them
    // column-wise and is a no-op on single-column pages.
    let t = Instant::now();
    let (lines, cuts) = crate::reading_order::sort_lines(lines, (w, h));
    log::info!(
        "[ocr] reading order: {:.2} ms ({} lines, {} column splits, {} band splits)",
        t.elapsed().as_secs_f64() * 1000.0,
        lines.len(),
        cuts.columns,
        cuts.bands
    );

    // Detected-line heights (bbox height in source pixels). Useful as a sanity
    // signal: a too-small or too-large average hints at over/under-segmentation,
    // and it sizes the resize scale the recognizer applies. Computed from the
    // same `polygon_bbox` the recog path uses, skipping degenerate lines
    // (boundary < 3 pts or zero-area bbox), so the stats reflect the lines
    // actually recognized.
    let line_heights: Vec<u32> = lines
        .iter()
        .filter_map(|line| polygon_bbox((w, h), &line.boundary).map(|(_, _, _, lh)| lh))
        .collect();
    if line_heights.is_empty() {
        log::info!("[ocr] avg line height: n/a (no valid lines)");
    } else {
        let avg = line_heights.iter().map(|&x| x as f64).sum::<f64>()
            / line_heights.len() as f64;
        log::info!(
            "[ocr] avg line height: {:.0} px (range {}-{} px)",
            avg,
            line_heights.iter().min().unwrap(),
            line_heights.iter().max().unwrap(),
        );
    }

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
                crate::tesseract_line::recognize(&crop_img, app, &opts.language)?
            }
            // Kraken: dewarp (polygon mask + baseline straightening) then
            // recognize. extract_polygon_line operates on the full page image
            // + the line's baseline + boundary, producing a flat strip that
            // the Stage-2 centerline normalizer and LSTM consume. Falls back
            // to a masked bbox crop inside the engine if the dewarp fails.
            //
            // PP-OCR seg → Kraken recog takes a direct path
            // (recognize_line_direct): it skips the baseline mesh warp, since
            // PP-OCR's rigid quads carry no curvature and the synth midline is
            // not a real baseline. It still masks + deskews (cheap, correct).
            "kraken" => {
                // Safe unwrap: kraken_rec_engine is Some iff engine_kind == "kraken".
                let eng = kraken_rec_engine.expect("kraken engine loaded for kraken recog");
                let t = if seg_name == "ppocr" {
                    // Tiny + quad: rigid 4-corner crop + deskew.
                    eng.recognize_line_direct(img, &line.boundary)
                } else if seg_name == "ppocr-poly" {
                    // Small + poly: crop masked to the multi-point boundary
                    // (tighter than a quad — masks neighbor-line ink), with
                    // curvature-gated dewarp when the line genuinely curves.
                    // Falls back to the direct path if the quad is missing
                    // (shouldn't happen for ppocr-poly — the poly segmenter
                    // always carries one from fit_min_area_quad).
                    match line.quad.as_ref() {
                        Some(q) => eng.recognize_line_poly(img, &line.boundary, q),
                        None => eng.recognize_line_direct(img, &line.boundary),
                    }
                } else {
                    eng.recognize_line_dewarped(img, &line.baseline, &line.boundary)
                }
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

    // Dispatch: parallel for kraken, serial for tesseract. When debug image
    // dumping is on (KRKN_DUMP_DIR set), force kraken serial too — the per-line
    // dump sequence numbers are allocated via an atomic counter, so parallel
    // workers grab them in scheduler order, not document order, and the dumped
    // files come out shuffled. Serial execution makes `0004_in.png` actually be
    // document line 4. Debug runs don't need the parallel speedup.
    let dump_enabled = std::env::var_os("KRKN_DUMP_DIR").is_some();
    let results: Vec<(LineBox, i32)> = match engine_kind {
        "kraken" if !dump_enabled => lines
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
    let (boxes, confidence) =
        crate::tesseract_page::recognize(img, app, &opts.language, opts.psm)?;
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

    /// Confirm the bundled PP-OCR detector bytes are non-empty and load into a
    /// `Detector` with the matching config, for BOTH variants. Uses
    /// `load_from_buffer_with_config` + the variant's config because the
    /// default `load_from_buffer` hard-codes the tiny architecture, which won't
    /// match the small weights. Mirrors `bundled_models_load_from_buffers` for
    /// kraken.
    #[test]
    fn bundled_ppocr_det_loads_from_buffer() {
        assert!(
            super::BUNDLED_PPOCR_DET_SMALL.len() > 1_000_000,
            "ppocr small-det too small: {}",
            super::BUNDLED_PPOCR_DET_SMALL.len()
        );
        let det = ppocr_engine::Detector::load_from_buffer_with_config(
            super::BUNDLED_PPOCR_DET_SMALL,
            1,
            ppocr_engine::DetectorConfig::small(),
        )
        .expect("bundled ppocr small-det loads with small config");
        let _ = det;

        assert!(
            super::BUNDLED_PPOCR_DET_TINY.len() > 1_000_000,
            "ppocr tiny-det too small: {}",
            super::BUNDLED_PPOCR_DET_TINY.len()
        );
        let det = ppocr_engine::Detector::load_from_buffer_with_config(
            super::BUNDLED_PPOCR_DET_TINY,
            1,
            ppocr_engine::DetectorConfig::tiny(),
        )
        .expect("bundled ppocr tiny-det loads with tiny config");
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
