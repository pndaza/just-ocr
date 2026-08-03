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
    /// Sauvola binarization is applied unconditionally inside
    /// [`preprocess_line`] (before resize); see its docs for why.
    pub fn recognize_line(&self, crop: &DynamicImage) -> Result<String> {
        let tensor =
            preprocess_line(crop, self.rec.height, self.rec.padding, self.rec.center_norm)?;
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
        self.recognize_line(&strip_dyn)
    }

    /// Recognize text from a line whose geometry is fully described by its
    /// boundary polygon — the PP-OCR direct path.
    ///
    /// Skips kraken's Stage-1 geometric dewarp (`extract_polygon_line`) entirely.
    /// That dewarp exists for kraken's *baselines*, which carry genuine
    /// curvature that the piecewise-affine mesh warp (`curved_dewarp`)
    /// straightens. PP-OCR's segmenter emits rigid quads (4 corners, no
    /// curvature) whose boundary is the only real geometry — there is no
    /// baseline-derived curve to straighten, and the synth 8-point midline we
    /// used to feed `recognize_line_dewarped` carries no curvature information
    /// either. Running the mesh warp on it is near-identity work for
    /// axis-aligned text (output ≈ input) plus an avoidable double-resample
    /// (mesh bilinear + Stage-2 `scale_to_h`) that softens edges.
    ///
    /// Pipeline:
    ///   1. `crop_polygon_white_bg` — mask outside-quad to white (255). For an
    ///      axis-aligned quad this is an exact no-op (quad == its AABB); for a
    ///      rotated quad it isolates the AABB corner triangles so neighbor ink
    ///      doesn't bleed into the strip. Correct and ~free either way.
    ///   2. Deskew when the quad's top edge is tilted above
    ///      [`DESKEW_THRESHOLD`] (1.5°): `angle = atan2(TR.y - TL.y, TR.x - TL.x)`.
    ///      This matters for the downstream `trim_neighbor_bleed`, whose
    ///      horizontal row-scan breaks on skewed text (the body crosses every
    ///      row, hiding the gap to neighbor-line bleed). Uses the cheap 2-point
    ///      [`recognition::dewarp::rotate_deskew`], never the mesh warp — a
    ///      rigid quad has no curve to straighten.
    ///   3. [`recognize_line`] — binarize → trim_neighbor_bleed →
    ///      center_norm/resize → pad → invert → forward → CTC.
    ///
    /// Assumes `boundary[0]`/`boundary[1]` are the quad's top edge `[TL, TR]`,
    /// as PaddleOCR DB's `fit_rotated_box` emits and `detection_to_line`
    /// preserves. Only call this for PP-OCR lines.
    pub fn recognize_line_direct(
        &self,
        image: &DynamicImage,
        boundary: &[(f64, f64)],
    ) -> Result<String> {
        // Mask the quad. Falls back to a plain bbox crop for degenerate
        // polygons (<3 pts) inside crop_polygon_white_bg.
        let mut crop = crop_polygon_white_bg(image, boundary);

        // Deskew only for genuinely tilted quads.
        if let Some(angle) = quad_deskew_angle(boundary) {
            if angle.abs() >= DESKEW_THRESHOLD {
                let strip = recognition::dewarp::rotate_deskew(
                    &crop.to_luma8(),
                    &[
                        (boundary[0].0, boundary[0].1),
                        (boundary[1].0, boundary[1].1),
                    ],
                    255,
                );
                crop = DynamicImage::ImageLuma8(strip);
            }
        }

        self.recognize_line(&crop)
    }

    /// Poly-segmenter recog path. Uses the **multi-point boundary polygon** to
    /// decide whether the line curves, then picks the cheapest correct dewarp:
    ///
    /// - **Curved** (sagitta / chord ≥ [`CURVATURE_THRESHOLD`]): synthesize a
    ///   curved centerline from the polygon (the polygon genuinely follows the
    ///   text shape — unlike the rigid quad) and run the full Stage-1 geometric
    ///   dewarp (`extract_polygon_line` → piecewise-affine mesh warp), which
    ///   straightens the curve before the Stage-2 center_norm. This is the path
    ///   kraken's own segmenter takes for curved baselines.
    /// - **Straight**: crop masked to the polygon (tighter than a quad bbox —
    ///   excludes neighbor-line ink) + deskew from the quad's TL→TR top edge,
    ///   same as `recognize_line_direct`. Cheap, no double-resample.
    ///
    /// `polygon` is the unclipped contour (used for the crop mask and, when
    /// curved, the dewarp boundary + centerline source). `quad` is the
    /// `[TL, TR, BR, BL]` 4-corner box (from `fit_min_area_quad`), used only
    /// for the straight-path deskew angle.
    pub fn recognize_line_poly(
        &self,
        image: &DynamicImage,
        polygon: &[(f64, f64)],
        quad: &[(f64, f64)],
    ) -> Result<String> {
        use image::GenericImageView;

        // Curvature gate: synthesize a centerline from the polygon and measure
        // its sagitta. Only genuinely curved lines pay for the mesh warp.
        let midline = recognition::dewarp::curved_midline(polygon, 16);
        let sagitta = recognition::dewarp::baseline_sagitta(&midline);
        if sagitta >= CURVATURE_THRESHOLD && midline.len() >= 3 {
            log::info!("[ocr] poly line curved (sag={:.3}) → geometric dewarp", sagitta);
            let strip =
                recognition::dewarp::extract_polygon_line(image, &midline, polygon).unwrap_or_else(
                    |_| {
                        // Fallback: masked bbox crop (no dewarp), as a GrayImage strip.
                        let crop = crop_polygon_white_bg(image, polygon);
                        crop.to_luma8()
                    },
                );
            return self.recognize_line(&DynamicImage::ImageLuma8(strip));
        }

        // Straight path: polygon mask + quad deskew.
        let mut crop = crop_polygon_white_bg(image, polygon);

        // Deskew from the quad's top edge (TL→TR), same gate as the direct path.
        if let Some(angle) = quad_deskew_angle(quad) {
            if angle.abs() >= DESKEW_THRESHOLD {
                // The deskew angle is in source-image coords; the crop's local
                // frame has the same orientation (just translated), so the angle
                // applies directly. Translate the quad's TL/TR to crop-local
                // coords (offset by the polygon bbox origin) for rotate_deskew.
                let (img_w, img_h) = image.dimensions();
                let min_x = polygon.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).max(0.0) as u32;
                let min_y = polygon.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(0.0) as u32;
                let _ = (img_w, img_h);
                let strip = recognition::dewarp::rotate_deskew(
                    &crop.to_luma8(),
                    &[
                        (quad[0].0 - min_x as f64, quad[0].1 - min_y as f64),
                        (quad[1].0 - min_x as f64, quad[1].1 - min_y as f64),
                    ],
                    255,
                );
                crop = DynamicImage::ImageLuma8(strip);
            }
        }

        self.recognize_line(&crop)
    }

    /// Borrow the recognition model directly (e.g. for rayon-parallel batch
    /// recognition — `RecognitionModel` is `Send + Sync`).
    pub fn recognizer(&self) -> &RecognitionModel {
        &self.rec
    }
}

// ── PP-OCR direct pipeline helpers ──────────────────────────────────

/// Quad deskew threshold in radians (~1.5°). PP-OCR quads below this angle are
/// left as-is; above it the crop is de-rotated before binarize+trim+resize.
///
/// Set low (1.5°) because `trim_neighbor_bleed` finds the text-body boundary
/// with a *horizontal* row scan: even ~1.5° of residual skew makes the text
/// body cross every row, defeating the gap detection and leaving neighbor-line
/// bleed untrimmed. Above this the scan is reliable and the bilinear resample
/// cost of deskewing is negligible.
///
/// **Empirically validated:** lowering this to 0.0 (deskew *every* quad, even
/// sub-degree) was A/B tested on thawzin_02 and *regressed* recognition on
/// ~17 of 34 lines. The double bilinear resample (deskew + height-resize)
/// softens edges enough to hurt the recognizer more than the sub-degree skew
/// it removes. 1.5° is the right gate — keep it.
const DESKEW_THRESHOLD: f64 = 1.5_f64.to_radians();

/// Curvature gate (sagitta / chord length) above which a PP-OCR poly line is
/// treated as curved and run through the full geometric dewarp
/// (`extract_polygon_line` → `curved_dewarp` piecewise-affine mesh warp) instead
/// of the cheap crop+deskew path.
///
/// 0.04 means the baseline's peak deviation reaches 4% of its length — for a
/// 300px-wide line that's ~12px of sag, which is clearly visible curvature and
/// starts to hurt the recognizer (column ink-centers drift, the height-normalize
/// resize smears the curve). Below this the straight-quad path is cheaper and
/// avoids a double bilinear resample that softens edges (see `DESKEW_THRESHOLD`
/// comment for the analogous regression on sub-threshold deskew).
const CURVATURE_THRESHOLD: f64 = 0.04;

/// The tilt of a quad's top edge, in radians. `Some(angle)` from
/// `atan2(boundary[1].y - boundary[0].y, boundary[1].x - boundary[0].x)`,
/// or `None` if `boundary` has fewer than 2 points.
///
/// Assumes `boundary[0]`/`boundary[1]` are the top edge `[TL, TR]` (the order
/// PaddleOCR's `fit_rotated_box` emits). A horizontal line has `angle == 0`;
/// clockwise tilt (top edge sloping down in image coords, y-down) is positive.
pub(crate) fn quad_deskew_angle(boundary: &[(f64, f64)]) -> Option<f64> {
    if boundary.len() < 2 {
        return None;
    }
    let (x0, y0) = boundary[0];
    let (x1, y1) = boundary[1];
    Some((y1 - y0).atan2(x1 - x0))
}

// ── Debug image dumping (KRKN_DUMP_DIR env var) ─────────────────────
//
// When `KRKN_DUMP_DIR` is set, intermediate images along the recognition
// pipeline are written as PNGs into that directory so you can eyeball exactly
// what the net sees. Files are named `<seq>_<stage>.png` where `<seq>` is a
// process-wide monotonic line counter (so the stages of one line share a
// prefix) and `<stage>` ∈ {in, trimmed, resized}. Off by default — no
// perf cost when unset (a single env read, cached in a OnceLock).

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

static DUMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Allocate the next line sequence number for dump filenames. Monotonic
/// across threads, so concurrent rayon workers don't clobber each other's
/// files (interleaving is fine; per-line collisions are not).
fn next_dump_seq() -> u64 {
    DUMP_SEQ.fetch_add(1, AtomicOrdering::Relaxed)
}

/// Dump an image to `{KRKN_DUMP_DIR}/{seq}_{stage}.png` if the env var is set.
/// Any write error is logged at warn level and swallowed — dumping is a debug
/// aid, never a functional failure.
pub(crate) fn dump_debug(image: &DynamicImage, stage: &str, seq: u64) {
    use std::sync::OnceLock;
    static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    let dir = DIR.get_or_init(|| {
        std::env::var_os("KRKN_DUMP_DIR").map(std::path::PathBuf::from)
    });
    let Some(dir) = dir else { return };
    let path = dir.join(format!("{seq:04}_{stage}.png"));
    if let Err(e) = image.save(&path) {
        log::warn!("[dump] failed to write {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_deskew_angle_axis_aligned_is_zero() {
        // Horizontal top edge TL(0,10) → TR(100,10): dy=0 → angle 0.
        let boundary = [(0.0, 10.0), (100.0, 10.0), (100.0, 40.0), (0.0, 40.0)];
        let angle = quad_deskew_angle(&boundary).unwrap();
        assert!(angle.abs() < 1e-9, "axis-aligned angle should be ~0, got {angle}");
    }

    #[test]
    fn quad_deskew_angle_rotated_quad() {
        // A ~10° clockwise tilt: top edge TL(0,0) → TR(100, tan(10°)*100≈17.6).
        let deg: f64 = 10.0;
        let dy = (deg.to_radians().tan() * 100.0).round();
        let boundary = [(0.0, 0.0), (100.0, dy), (100.0, dy + 30.0), (0.0, 30.0)];
        let angle = quad_deskew_angle(&boundary).unwrap().to_degrees();
        // Allow rounding from the integer dy.
        assert!(
            (angle - deg).abs() < 0.5,
            "rotated quad angle should be ~{deg}°, got {angle:.2}°"
        );
    }

    #[test]
    fn quad_deskew_angle_degenerate_returns_none() {
        assert!(quad_deskew_angle(&[]).is_none());
        assert!(quad_deskew_angle(&[(1.0, 2.0)]).is_none());
    }
}
