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
}

impl Segmenter for KrakenSegmenter {
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String> {
        let lines = self.engine.segment(img).map_err(|e| e.to_string())?;
        Ok(lines
            .into_iter()
            .map(|l| DetectedLine {
                baseline: l.baseline,
                boundary: l.boundary,
                quad: None,
            })
            .collect())
    }
    fn name(&self) -> &'static str {
        "kraken"
    }
}

use ppocr_engine::Detection;

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
/// Assumes the quad is ordered clockwise (in image coordinates, where y
///   increases downward) from the top-left corner:
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
        "ppocr"
    }
}

/// Convert a PP-OCR `Detection` (4-corner quad) to a `DetectedLine`. Returns
/// `None` if any quad coordinate is non-finite (corrupted detection).
fn detection_to_line(d: &Detection) -> Option<DetectedLine> {
    let quad: [(f64, f64); 4] = [
        (d.polygon[0].0 as f64, d.polygon[0].1 as f64),
        (d.polygon[1].0 as f64, d.polygon[1].1 as f64),
        (d.polygon[2].0 as f64, d.polygon[2].1 as f64),
        (d.polygon[3].0 as f64, d.polygon[3].1 as f64),
    ];
    // Reject corrupted detections with non-finite coords — they would produce
    // garbage bboxes downstream (NaN/inf → invalid `as u32` casts in polygon_bbox).
    if !quad.iter().all(|(x, y)| x.is_finite() && y.is_finite()) {
        return None;
    }
    let boundary = close_polygon(&quad);
    // 8 samples: matches typical Kraken baseline polyline resolution; enough for
    // dewarp without over-sampling.
    let baseline = synth_midline(&quad, 8);
    Some(DetectedLine { baseline, boundary, quad: None })
}

// ── ppocr-poly: multi-point polygon segmenter ──────────────────────
//
// A second PP-OCR segmenter that produces a multi-point boundary polygon
// (contour → Douglas-Peucker simplify → pyclipper-equivalent `unclip` + an
// anisotropic axis stretch) instead of the rigid 4-corner quad. The polygon
// follows curved/rotated text, so the overlay hugs the line and the masked
// recog crop excludes more neighbor-line ink than a quad bbox would.
//
// The pipeline mirrors PaddleOCR's `polygons_from_bitmap` (db_postprocess.py:59)
// and was validated against the Python reference via the `expand_compare_ppocr`
// example. Base thresholds start from the PP-OCRv6 det config but are TUNED for
// the bundled Burmese model — this is NOT a verbatim config match:
//   thresh=0.2, box_thresh=0.45, unclip_ratio=2.0 (v6 config is 1.4; 1.4–1.7
//   under-expands the poly-masked recog crop), poly_eps=0.002.
//
// NOTE on faithfulness: the `stretch_anisotropic` step (below) is a deliberate
// ADDITION with no equivalent in `polygons_from_bitmap` — PaddleOCR offsets and
// is done. The stretch grows top/bottom coverage (across ×1.10) while leaving
// the line ends at unclip's natural reach (along ×1.0). It is empirically tuned
// for the recognizer's ascender/descender margin, not derived from the
// reference; if matching PaddleOCR's output exactly is the goal, set both
// stretch factors to 1.0.

/// PP-OCRv6 det thresholds (DetectorPostprocessOptions::default / v6 config).
const POLY_THRESH: f32 = 0.2;
const POLY_BOX_THRESH: f32 = 0.45;
const POLY_MIN_AREA: usize = 3;
/// PaddleOCR `unclip_ratio`. db_postprocess.py:39 default is 2.0; the v6 det
/// config uses 1.4, but we found 1.4–1.7 under-expands for the poly-masked recog
/// crop (glyph tops/bottoms get clipped on dense lines where the contour hugs
/// the ink). 2.0 gives full vertical coverage — the recognizer needs ~15%+
/// margin top and bottom to avoid clipping ascenders/descenders after the
/// height-normalize resize.
const POLY_UNCLIP_RATIO: f64 = 2.0;
/// approxPolyDP epsilon as a fraction of arc length (db_postprocess.py:76).
const POLY_EPS: f64 = 0.002;
/// Min-area-rect smaller-side threshold for the post-unclip drop
/// (db_postprocess.py:97: `sside < min_size + 2`, min_size=3 → 5px). Detections
/// whose expanded polygon's shorter side is below this are noise/speckle.
const POLY_MIN_SIDE: f64 = 5.0;
/// Axis stretch about the centroid, applied after unclip. Anisotropic: the
/// perpendicular (`across` = top/bottom) and along-axis (`along` = line ends)
/// directions scale independently so the post-unclip polygon can be shaped.
///
/// - `POLY_STRETCH_ACROSS` > 1 grows top/bottom coverage (more vertical margin
///   for ascenders/descenders). `POLY_STRETCH_ALONG` = 1.0 leaves the line ends
///   at the unclip's natural horizontal reach — shrinking it (<1) clipped the
///   start/end glyphs, growing it (>1) pushed into neighboring lines. The
///   area-based unclip already extends the long axis adequately.
const POLY_STRETCH_ACROSS: f64 = 1.10;
const POLY_STRETCH_ALONG: f64 = 1.0;

/// PP-OCR segmenter that emits multi-point boundary polygons. Shares the same
/// lazy-loaded `Detector` as [`PPOcrSegmenter`] (no second model load); the
/// difference is purely the postprocess geometry applied to the score map.
pub struct PPOcrPolySegmenter {
    detector: std::sync::Arc<ppocr_engine::Detector>,
}

impl PPOcrPolySegmenter {
    pub fn new(detector: std::sync::Arc<ppocr_engine::Detector>) -> Self {
        Self { detector }
    }
}

impl Segmenter for PPOcrPolySegmenter {
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String> {
        let t = std::time::Instant::now();
        // Get the raw DB score map (forward pass only — no postprocess yet).
        let (values, ih, iw, transform) = self
            .detector
            .detect_raw(img)
            .map_err(|e| e.to_string())?;
        let cw = transform.content_width() as usize;
        let ch = transform.content_height() as usize;

        // Connected components over the binary mask (8-conn).
        let components = collect_components(&values, iw, cw, ch, POLY_THRESH, POLY_MIN_AREA);

        // Two-pass fit, mirroring ppocr-engine's postprocess: pass 1 fits every
        // component with free-angle PCA — elongated text lines have a stable
        // axis and vote for the page's text direction. Pass 2 (the loop below)
        // re-fits near-square components (page numbers, 1–3 glyphs) with the
        // axis locked to that consensus angle, because their own PCA axis is
        // numerically ill-conditioned (`atan2(2·cov_xy, cov_xx − cov_yy)` with
        // a noise-dominated denominator) and comes out rotated anywhere in
        // ±90° — false skew on the synthesized baseline and deskew quad.
        let fits: Vec<Option<MinAreaFit>> =
            components.iter().map(|px| fit_min_area_quad(px)).collect();
        let mut voter_angles: Vec<f64> = fits
            .iter()
            .flatten()
            .filter(|f| f.min_side() >= POLY_MIN_SIDE && f.aspect() >= PAGE_ANGLE_VOTER_ASPECT)
            .map(|f| f.axis_angle)
            .collect();
        voter_angles.sort_by(|a, b| a.total_cmp(b));
        let page_angle = median_angle(&voter_angles).unwrap_or(0.0);

        let mut out = Vec::with_capacity(components.len());
        for (px, fit) in components.iter().zip(&fits) {
            let fit = match fit {
                Some(f) if (SNAP_ASPECT_MIN..SNAP_ASPECT_MAX).contains(&f.aspect()) => {
                    fit_min_area_quad_at_angle(px, page_angle)
                }
                other => *other,
            };
            // 4-corner quad via PCA min-area box (same shape `fit_rotated_box`
            // produces in the quad segmenter). Carried on `DetectedLine.quad`
            // so `recognize_line_direct`'s deskew (which indexes [0]/[1] as the
            // top edge) keeps working — the multi-point boundary can't be
            // indexed that way.
            let fit = match fit {
                Some(f) => f,
                None => continue,
            };
            // Pre-unclip min-side guard: drop speckle whose min-area-rect
            // smaller side is already below the post-unclip threshold. The
            // Rust candle forward pass produces a slightly less bimodal score
            // map than PaddlePaddle (per-pixel values drift ±0.1 in the
            // transition band), so `box_score_fast` alone can't reject all
            // noise contours the way Python's does — the structural min-side
            // check on the component shape is a robust second gate that doesn't
            // depend on absolute score values. Real text lines have min-side
            // >10px at this stage; score-map speckle is <5px.
            if fit.min_side() < POLY_MIN_SIDE {
                continue;
            }
            let quad = fit.quad;

            // Ordered contour via Moore boundary trace (kraken_engine). Input
            // wants (y, x) tuples; returns Vec<Point> in (x, y) space.
            let yx: Vec<(usize, usize)> =
                px.iter().map(|&(x, y)| (y as usize, x as usize)).collect();
            let contour = kraken_engine::contours::boundary_trace(&yx);
            if contour.len() < 4 {
                continue;
            }

            // Douglas-Peucker simplify (≈ cv2.approxPolyDP, closed). Drop if
            // the simplified polygon has fewer than 4 vertices.
            let perimeter = closed_perimeter(&contour);
            let simplified = kraken_engine::polygon::simplify(&contour, POLY_EPS * perimeter);
            if simplified.len() < 4 {
                continue;
            }

            // box_score_fast filter (db_postprocess.py:189): mean of the score
            // map under the simplified polygon mask, over its bbox. Drops
            // low-confidence detections.
            if box_score_fast(&values, iw, &simplified) < POLY_BOX_THRESH {
                continue;
            }

            // pyclipper `unclip` (JT_ROUND, ET_CLOSEDPOLYGON).
            // distance = area * ratio / perimeter (db_postprocess.py:162).
            let area = shoelace_area(&simplified);
            let distance = area * POLY_UNCLIP_RATIO / perimeter.max(1e-9);
            let mut poly = match unclip_round(&simplified, distance) {
                Some(p) => p,
                None => continue,
            };
            if poly.len() < 4 {
                continue;
            }
            // Anisotropic stretch about the centroid: more top/bottom (across)
            // for glyph coverage, less at the line ends (along) so the unclip's
            // uniform push doesn't over-extend long lines into their neighbors.
            poly = stretch_anisotropic(&poly, POLY_STRETCH_ALONG, POLY_STRETCH_ACROSS);

            // Min-side filter — matches `polygons_from_bitmap`'s `get_mini_boxes`
            // drop (db_postprocess.py:96-98): discard detections whose post-unclip
            // min-area-rect smaller side is below `min_size + 2` (= 5px). Without
            // this, score-map speckle survives as spurious tiny detections — the
            // cause of the 39-vs-27 over-detection on heavy_curve_02.
            if min_area_side(&poly) < POLY_MIN_SIDE {
                continue;
            }

            // Map bitmap coords → source-image pixel space.
            let to_src = |p: (f64, f64)| -> (f64, f64) {
                (
                    transform.map_x_to_source(p.0 as f32) as f64,
                    transform.map_y_to_source(p.1 as f32) as f64,
                )
            };
            let mut boundary: Vec<(f64, f64)> = poly.into_iter().map(to_src).collect();
            // Ensure closed (first point repeated at the end) to match the
            // convention `polygon_bbox` / `crop_polygon_white_bg` expect.
            if boundary.first() != boundary.last() {
                boundary.push(boundary[0]);
            }
            let quad_src = quad.map(to_src);
            // Baseline synthesized from the quad (same as the quad segmenter) —
            // the polygon has no natural single midline.
            let baseline = synth_midline(&quad_src, 8);
            out.push(DetectedLine {
                baseline,
                boundary,
                quad: Some(quad_src),
            });
        }
        log::info!(
            "[ocr] ppocr-poly segmentation: {} lines from {} components in {:?} \
             (score map {}x{})",
            out.len(),
            components.len(),
            t.elapsed(),
            iw,
            ih,
        );
        Ok(out)
    }
    fn name(&self) -> &'static str {
        "ppocr-poly"
    }
}

/// 8-connectivity connected-component labeling. Returns one `Vec<(x,y)>` pixel
/// set per component (integer pixel coords). Mirrors `collect_component` in
/// ppocr-engine's postprocess (which is `pub(crate)` so unreachable here).
fn collect_components(
    values: &[f32],
    map_w: usize,
    cw: usize,
    ch: usize,
    thr: f32,
    min_area: usize,
) -> Vec<Vec<(i32, i32)>> {
    use std::collections::VecDeque;
    let mut visited = vec![false; cw * ch];
    let mut out = Vec::new();
    const N8: [(isize, isize); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ];
    for y in 0..ch {
        for x in 0..cw {
            let idx = y * cw + x;
            if visited[idx] || values[y * map_w + x] < thr {
                continue;
            }
            let mut q = VecDeque::new();
            q.push_back((x, y));
            visited[idx] = true;
            let mut pts = Vec::new();
            while let Some((cx, cy)) = q.pop_front() {
                pts.push((cx as i32, cy as i32));
                for (dx, dy) in N8 {
                    let nx = cx as isize + dx;
                    let ny = cy as isize + dy;
                    if nx < 0 || ny < 0 || nx >= cw as isize || ny >= ch as isize {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    let nidx = ny * cw + nx;
                    if !visited[nidx] && values[ny * map_w + nx] >= thr {
                        visited[nidx] = true;
                        q.push_back((nx, ny));
                    }
                }
            }
            if pts.len() >= min_area {
                out.push(pts);
            }
        }
    }
    out
}

/// Aspect (fitted width / height) at or above which a component's PCA axis is
/// trusted and votes for the page's text direction. Mirrors
/// `PAGE_ANGLE_VOTER_ASPECT` in ppocr-engine's postprocess.
const PAGE_ANGLE_VOTER_ASPECT: f64 = 3.0;
/// Half-open aspect band in which the PCA axis is ill-conditioned (near-square
/// components: page numbers) and gets re-fitted with the axis locked to the
/// page-consensus angle. Mirrors `SNAP_ASPECT_MIN/MAX` in ppocr-engine.
const SNAP_ASPECT_MIN: f64 = 0.5;
const SNAP_ASPECT_MAX: f64 = 2.0;

/// A component's rotated-box fit: quad corners `[TL, TR, BR, BL]` plus the raw
/// (pre-unclip) extents and axis angle needed for page-consensus
/// classification.
#[derive(Clone, Copy)]
struct MinAreaFit {
    quad: [(f64, f64); 4],
    width: f64,
    height: f64,
    axis_angle: f64,
}

impl MinAreaFit {
    fn aspect(&self) -> f64 {
        self.width / self.height.max(1e-9)
    }
    fn min_side(&self) -> f64 {
        self.width.min(self.height)
    }
}

/// Median of a pre-sorted, non-empty angle slice; `None` when empty.
fn median_angle(sorted: &[f64]) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some((sorted[mid - 1] + sorted[mid]) * 0.5)
    }
}

/// PCA min-area rotated box for a pixel set — same algorithm as
/// `fit_rotated_box` (ppocr-engine postprocess.rs), ported here so we can
/// compute the quad from the component pixels directly (the engine's version
/// is `pub(crate)`). `None` for degenerate (near-zero-area) components.
fn fit_min_area_quad(points: &[(i32, i32)]) -> Option<MinAreaFit> {
    if points.len() < 3 {
        return None;
    }
    let count = points.len() as f64;
    let (cx, cy) = {
        let (sx, sy): (f64, f64) = points.iter().fold((0.0, 0.0), |(sx, sy), &(x, y)| {
            (sx + x as f64, sy + y as f64)
        });
        (sx / count, sy / count)
    };
    let (sxx, sxy, syy) = points.iter().fold((0.0, 0.0, 0.0), |(sxx, sxy, syy), &(x, y)| {
        let (dx, dy) = (x as f64 - cx, y as f64 - cy);
        (sxx + dx * dx, sxy + dx * dy, syy + dy * dy)
    });
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    Some(fit_along_axis(points, angle))
}

/// Constrained variant: the reading direction is pinned to `angle` (the
/// page-consensus angle) instead of estimated from the pixels. Used for
/// near-square components whose own PCA axis is noise (see the two-pass
/// comment in `PPOcrPolySegmenter::segment`).
fn fit_min_area_quad_at_angle(points: &[(i32, i32)], angle: f64) -> Option<MinAreaFit> {
    if points.len() < 3 {
        return None;
    }
    Some(fit_along_axis(points, angle))
}

/// Project the pixels onto the given axis direction and build the rotated box.
fn fit_along_axis(points: &[(i32, i32)], angle: f64) -> MinAreaFit {
    let count = points.len() as f64;
    let (cx, cy) = {
        let (sx, sy): (f64, f64) = points.iter().fold((0.0, 0.0), |(sx, sy), &(x, y)| {
            (sx + x as f64, sy + y as f64)
        });
        (sx / count, sy / count)
    };
    let mut axis = (angle.cos(), angle.sin());
    if axis.0 < 0.0 || (axis.0.abs() < f64::EPSILON && axis.1 < 0.0) {
        axis = (-axis.0, -axis.1);
    }
    let normal = (-axis.1, axis.0);
    let (mut min_a, mut max_a, mut min_n, mut max_n) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in points {
        let (dx, dy) = (x as f64 - cx, y as f64 - cy);
        let along = dx * axis.0 + dy * axis.1;
        let across = dx * normal.0 + dy * normal.1;
        min_a = min_a.min(along);
        max_a = max_a.max(along);
        min_n = min_n.min(across);
        max_n = max_n.max(across);
    }
    let width = max_a - min_a + 1.0;
    let height = max_n - min_n + 1.0;
    // Center of the fitted box in (axis, normal) space, mapped back to xy.
    let a_center = (min_a + max_a) * 0.5;
    let n_center = (min_n + max_n) * 0.5;
    let bx = cx + axis.0 * a_center + normal.0 * n_center;
    let by = cy + axis.1 * a_center + normal.1 * n_center;
    let ha = width * 0.5;
    let hn = height * 0.5;
    // [TL, TR, BR, BL] — matches PaddleOCR's fit_rotated_box ordering, which
    // is what recognize_line_direct's deskew assumes for boundary[0]/[1].
    let tl = (bx + axis.0 * -ha + normal.0 * -hn, by + axis.1 * -ha + normal.1 * -hn);
    let tr = (bx + axis.0 * ha + normal.0 * -hn, by + axis.1 * ha + normal.1 * -hn);
    let br = (bx + axis.0 * ha + normal.0 * hn, by + axis.1 * ha + normal.1 * hn);
    let bl = (bx + axis.0 * -ha + normal.0 * hn, by + axis.1 * -ha + normal.1 * hn);
    MinAreaFit {
        quad: [tl, tr, br, bl],
        width,
        height,
        axis_angle: axis.1.atan2(axis.0),
    }
}

/// Smaller side of the min-area rotated rect for an arbitrary polygon
/// (the `min(bounding_box[1])` value `get_mini_boxes` returns,
/// db_postprocess.py:187). Used for the post-unclip min-side drop.
fn min_area_side(poly: &[(f64, f64)]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let n = poly.len() as f64;
    let cx = poly.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = poly.iter().map(|p| p.1).sum::<f64>() / n;
    let (sxx, sxy, syy) = poly.iter().fold((0.0, 0.0, 0.0), |(sxx, sxy, syy), &(x, y)| {
        let (dx, dy) = (x - cx, y - cy);
        (sxx + dx * dx, sxy + dx * dy, syy + dy * dy)
    });
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let axis = (angle.cos(), angle.sin());
    let normal = (-axis.1, axis.0);
    let (mut min_a, mut max_a, mut min_n, mut max_n) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in poly {
        let (dx, dy) = (x - cx, y - cy);
        let along = dx * axis.0 + dy * axis.1;
        let across = dx * normal.0 + dy * normal.1;
        min_a = min_a.min(along);
        max_a = max_a.max(along);
        min_n = min_n.min(across);
        max_n = max_n.max(across);
    }
    (max_a - min_a).min(max_n - min_n)
}

/// Perimeter of a closed polygon (closing edge included).
fn closed_perimeter(poly: &[kraken_engine::polygon::Point]) -> f64 {
    if poly.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        sum += ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
    }
    sum
}

/// Shoelace polygon area (absolute).
fn shoelace_area(poly: &[kraken_engine::polygon::Point]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        sum += a.x * b.y - b.x * a.y;
    }
    sum.abs() * 0.5
}

/// `box_score_fast` (db_postprocess.py:189): mean of the score map under the
/// polygon, masked by the rasterized polygon over its bbox. Rasterization uses
/// `imageproc::draw_polygon_mut` — the same `fillPoly`-equivalent that
/// `crop_polygon_white_bg` uses — so the mask matches cv2.fillPoly's coverage
/// (a hand-rolled `point_in_polygon` test diverges on tiny blobs by ~half a
/// pixel of vertex convention, which flips noise-contour scores past the
/// threshold).
fn box_score_fast(
    pred: &[f32],
    map_w: usize,
    poly: &[kraken_engine::polygon::Point],
) -> f32 {
    let (mut xmin, mut ymin, mut xmax, mut ymax) =
        (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in poly {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }
    let h = pred.len() / map_w;
    let x0 = xmin.floor().clamp(0.0, (map_w - 1) as f64) as i32;
    let x1 = xmax.ceil().clamp(0.0, (map_w - 1) as f64) as i32;
    let y0 = ymin.floor().clamp(0.0, (h - 1) as f64) as i32;
    let y1 = ymax.ceil().clamp(0.0, (h - 1) as f64) as i32;
    if x1 < x0 || y1 < y0 {
        return 0.0;
    }
    let cw = (x1 - x0 + 1) as u32;
    let ch = (y1 - y0 + 1) as u32;
    // Translate to crop-local integer coords. Dedup consecutive duplicates and
    // drop a closing point equal to the first — imageproc panics on first==last.
    let mut pts: Vec<imageproc::point::Point<i32>> = poly
        .iter()
        .map(|p| imageproc::point::Point::new(p.x as i32 - x0, p.y as i32 - y0))
        .collect();
    pts.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    if pts.len() >= 2 && pts.first() == pts.last() {
        pts.pop();
    }
    if pts.len() < 3 {
        // Degenerate — fall back to bbox mean (matches cv2.fillPoly empty-mask
        // behavior loosely; these are dropped by the score threshold anyway).
        let mut sum = 0.0f32;
        let mut cnt = 0u32;
        for y in y0..=y1 {
            for x in x0..=x1 {
                sum += pred[y as usize * map_w + x as usize];
                cnt += 1;
            }
        }
        return if cnt == 0 { 0.0 } else { sum / cnt as f32 };
    }
    let mut mask = image::GrayImage::new(cw, ch);
    imageproc::drawing::draw_polygon_mut(
        &mut mask,
        &pts,
        image::Luma([255u8]),
    );
    let mut sum = 0.0f32;
    let mut cnt = 0u32;
    for y in 0..ch {
        for x in 0..cw {
            if mask.get_pixel(x, y)[0] != 0 {
                sum += pred[(y as i32 + y0) as usize * map_w + (x as i32 + x0) as usize];
                cnt += 1;
            }
        }
    }
    if cnt == 0 { 0.0 } else { sum / cnt as f32 }
}

/// pyclipper-equivalent `unclip`: offset a closed polygon outward by `delta`
/// using Clipper2 with round joins (`JT_ROUND`, `ET_CLOSEDPOLYGON`). Returns
/// `None` if Clipper yields zero or more than one solution polygon (matches
/// `polygons_from_bitmap`'s drop rule, db_postprocess.py:88-89).
///
/// Uses the float (`PathD`) entry point [`inflate_paths_d`], which preserves
/// sub-pixel precision by internally scaling to int64 at `10^precision` before
/// offsetting and scaling back — mirroring pyclipper's precision handling.
/// The previous int (`Path64`) path rounded each vertex to an integer pixel
/// BEFORE offsetting, discarding the fractional part; since the polygon lives
/// in downsampled score-map space (each unit = several source pixels), that
/// coarsened the expanded shape well away from PaddleOCR's float output.
fn unclip_round(
    poly: &[kraken_engine::polygon::Point],
    delta: f64,
) -> Option<Vec<(f64, f64)>> {
    use clipper2_rust::clipper::inflate_paths_d;
    use clipper2_rust::offset::{EndType, JoinType};
    use clipper2_rust::{PathD, PathsD};
    let path: PathD = poly
        .iter()
        .map(|p| clipper2_rust::Point::new(p.x, p.y))
        .collect();
    let mut paths = PathsD::new();
    paths.push(path);
    // precision = 2 ⇒ internal scale ×100, matching pyclipper's default fixed
    // scaling factor. miter_limit=2.0 and arc_tolerance=0.0 mirror the
    // `ClipperOffset::new_default()` values the previous int path used.
    let result = inflate_paths_d(
        &paths,
        delta,
        JoinType::Round,
        EndType::Polygon,
        2.0,
        2,
        0.0,
    );
    if result.len() != 1 {
        return None;
    }
    let out = result.first()?;
    Some(out.iter().map(|p| (p.x, p.y)).collect())
}

/// Stretch a polygon about its centroid along its PCA principal axis. Unlike a
/// uniform `unclip` (which pushes outward equally in every direction), this is
/// **anisotropic**: the along-axis coordinate (text-reading direction = line
/// ends) and the across-axis coordinate (perpendicular = top/bottom) scale by
/// independent factors. The centroid stays fixed.
///
/// `along < 1` shrinks the line ends (counteracts unclip's over-extension on
/// long lines); `across > 1` grows top/bottom coverage (vertical margin for
/// ascenders/descenders the recognizer needs). Both 1.0 = identity.
fn stretch_anisotropic(poly: &[(f64, f64)], along: f64, across: f64) -> Vec<(f64, f64)> {
    if poly.len() < 2 || ((along - 1.0).abs() < 1e-9 && (across - 1.0).abs() < 1e-9) {
        return poly.to_vec();
    }
    let n = poly.len() as f64;
    let cx = poly.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = poly.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for &(x, y) in poly {
        let (dx, dy) = (x - cx, y - cy);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (ax, ay) = (angle.cos(), angle.sin());
    let (nx, ny) = (-ay, ax);
    poly.iter()
        .map(|&(x, y)| {
            let (dx, dy) = (x - cx, y - cy);
            let along_s = (dx * ax + dy * ay) * along;
            let across_s = (dx * nx + dy * ny) * across;
            (cx + along_s * ax + across_s * nx, cy + along_s * ay + across_s * ny)
        })
        .collect()
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

    #[test]
    fn unclip_round_offsets_square_outward() {
        // A 10×10 axis-aligned square. distance = area*ratio/perimeter.
        // With ratio 1.0: 100*1.0/40 = 2.5. Each edge moves out 2.5 → 15×15.
        // The four corners of the expanded polygon (whichever order Clipper
        // emits) must trace a 15×15 box from (-2.5,-2.5) to (12.5,12.5).
        let square = vec![
            kraken_engine::polygon::Point::new(0.0, 0.0),
            kraken_engine::polygon::Point::new(10.0, 0.0),
            kraken_engine::polygon::Point::new(10.0, 10.0),
            kraken_engine::polygon::Point::new(0.0, 10.0),
        ];
        let expanded = unclip_round(&square, 2.5).expect("square should offset cleanly");
        let xs: Vec<f64> = expanded.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = expanded.iter().map(|p| p.1).collect();
        let (xmin, xmax) = (xs.iter().cloned().fold(f64::INFINITY, f64::min), xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let (ymin, ymax) = (ys.iter().cloned().fold(f64::INFINITY, f64::min), ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        assert!((xmin - (-2.5)).abs() < 0.05, "xmin {xmin} ≈ -2.5");
        assert!((xmax - 12.5).abs() < 0.05, "xmax {xmax} ≈ 12.5");
        assert!((ymin - (-2.5)).abs() < 0.05, "ymin {ymin} ≈ -2.5");
        assert!((ymax - 12.5).abs() < 0.05, "ymax {ymax} ≈ 12.5");
    }

    #[test]
    fn unclip_round_preserves_subpixel_input() {
        // Regression for the int-rounding bug. A square with FRACTIONAL
        // vertices (e.g. shifted by 0.4 px) must offset from its true position,
        // not from the rounded-to-int position. Naive int-rounding would snap
        // 0.4→0 and 10.4→10, collapsing the shift entirely; the precision-
        // preserving float path keeps the 0.4 offset.
        //
        //   int-rounded input: square at [0,10]   → offset bbox xmin ≈ -2.5
        //   float input:       square at [0.4,10.4] → offset bbox xmin ≈ -2.1
        let shifted = vec![
            kraken_engine::polygon::Point::new(0.4, 0.4),
            kraken_engine::polygon::Point::new(10.4, 0.4),
            kraken_engine::polygon::Point::new(10.4, 10.4),
            kraken_engine::polygon::Point::new(0.4, 10.4),
        ];
        let expanded = unclip_round(&shifted, 2.5).expect("shifted square should offset");
        let xmin = expanded.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        // xmin should be ~0.4 - 2.5 = -2.1, NOT -2.5 (which would mean rounding
        // ate the 0.4 shift). Allow a small tolerance for the round-join curve.
        assert!(
            (xmin - (-2.1)).abs() < 0.1,
            "xmin {xmin} should be ≈ -2.1 (sub-pixel preserved); -2.5 would mean \
             the fractional vertex was rounded away (the int-rounding bug)"
        );
    }

    /// Pixels of a skewed bar: for each x in [x0, x0+len), rows around
    /// y0 + slope·(x − x0). 8-connectivity keeps it one component.
    fn bar_pixels(x0: i32, y0: i32, len: i32, thickness: i32, slope: f64) -> Vec<(i32, i32)> {
        let mut px = Vec::new();
        for x in x0..x0 + len {
            let cy = y0 as f64 + slope * (x - x0) as f64;
            for dy in 0..thickness {
                px.push((x, (cy + dy as f64).round() as i32));
            }
        }
        px
    }

    /// A near-square blob with strongly diagonal pixel mass — its raw PCA axis
    /// comes out diagonal (the false-skew failure mode on page numbers).
    fn diagonal_blob_pixels(x0: i32, y0: i32, size: i32) -> Vec<(i32, i32)> {
        let mut px = Vec::new();
        for y in 0..size {
            for x in 0..(size * 6 / 5) {
                if (x as f64) / 1.2 + (y as f64) < size as f64 {
                    px.push((x0 + x, y0 + y));
                }
            }
        }
        px
    }

    #[test]
    fn pca_fit_on_diagonal_blob_is_ill_conditioned() {
        // Sanity-checks the premise of the two-pass fix: a near-square blob's
        // free-angle PCA fit lands on a diagonal axis...
        let fit = fit_min_area_quad(&diagonal_blob_pixels(0, 0, 30)).expect("fit");
        let deg = fit.axis_angle.to_degrees();
        assert!(deg.abs() > 10.0, "PCA axis should be diagonal, got {deg:.1}°");
        // ...while the constrained fit at 0° is horizontal and wider than tall.
        let snapped = fit_min_area_quad_at_angle(&diagonal_blob_pixels(0, 0, 30), 0.0).expect("fit");
        assert!(snapped.axis_angle.abs() < 1e-6);
        assert!(snapped.quad[1].1 - snapped.quad[0].1 < 1.0, "top edge should be horizontal");
        assert!(snapped.width > snapped.height);
    }

    #[test]
    fn elongated_pca_fit_is_stable_under_skew() {
        // The voting population: a −3° bar must come back at ≈−3°.
        let fit = fit_min_area_quad(&bar_pixels(10, 50, 400, 12, -0.0524)).expect("fit");
        let deg = fit.axis_angle.to_degrees();
        assert!((deg + 3.0).abs() < 0.5, "expected ≈−3°, got {deg:.2}°");
        assert!(fit.aspect() >= PAGE_ANGLE_VOTER_ASPECT);
    }
}
