//! Stage 1 geometric line dewarp: polygon mask + baseline straightening.
//!
//! Faithful port of kraken's `extract_polygons`
//! (`kraken/lib/segmentation.py:1424`), specifically the non-legacy branches:
//!   - **straight baseline (2 points):** rotation deskew via `_rotate`
//!     (`segmentation.py:452, 1551-1566`)
//!   - **curved baseline (>2 points):** Bézier-bevelled piecewise-affine mesh
//!     warp onto an arc-length strip (`segmentation.py:1568-1628` +
//!     `_bevelled_warping_envelope` at `:1334`)
//!
//! What this replaces: the orchestrator previously fed recognition an
//! axis-aligned bounding-box crop of the boundary polygon with no mask and no
//! dewarp. kraken instead (1) masks the patch to the polygon, then (2) warps
//! it so the baseline becomes a straight horizontal line, producing a flat
//! strip that the Stage-2 normalizer ([`crate::recognition::lineest`]) and the
//! recognition model consume. The outside-polygon fill is white (255) to match
//! the black-on-white polarity the rest of the pipeline assumes (kraken uses
//! black cval=0 because it inverts earlier in its transform chain).
//!
//! Sub-port references:
//!   - `apply_polygonal_mask` / `make_polygonal_mask` — `segmentation.py:1398-1421`
//!   - `subdivide_polygon` (B-spline / Chaikin) — `skimage/measure/_polygon.py`
//!   - `approximate_polygon` (Douglas-Peucker) — reused via [`polygon::simplify`]

use anyhow::{anyhow, Result};
use image::{GrayImage, Luma, GenericImageView};

use crate::polygon::simplify as approximate_polygon;
use crate::polygon::Point;

pub(crate) type Pt = (f64, f64);

/// Extract a single line as a flat, dewarped grayscale strip.
///
/// Mirrors the per-line body of `extract_polygons` (non-legacy path). Returns
/// a `GrayImage` whose x-axis is arc-length along the baseline and whose y-axis
/// is signed perpendicular distance, with the curve straightened out. Outside
/// the boundary polygon is filled white (255), preserving the black-on-white
/// polarity the recognition pipeline expects.
///
/// Falls back to a plain masked bbox crop when the baseline is degenerate
/// (`<2` points or total length `<5px`), matching kraken's `LineString.length
/// < 5` guard.
pub fn extract_polygon_line(
    image: &image::DynamicImage,
    baseline: &[Pt],
    boundary: &[Pt],
) -> Result<GrayImage> {
    if boundary.len() < 3 {
        return Err(anyhow!("boundary polygon needs >= 3 points"));
    }

    // Crop the source patch to the boundary polygon's axis-aligned bbox.
    let (img_w, img_h) = image.dimensions();
    let (c_min, r_min, c_max, r_max) = polygon_bbox(boundary, img_w, img_h);
    let patch_w = c_max.saturating_sub(c_min) + 1;
    let patch_h = r_max.saturating_sub(r_min) + 1;
    if patch_w < 2 || patch_h < 2 {
        return Err(anyhow!("line patch too small: {patch_w}x{patch_h}"));
    }
    let gray = image.to_luma8();
    let patch = crop_luma(&gray, c_min, r_min, patch_w, patch_h);

    // Mask the patch: keep pixels inside the polygon, fill outside with the
    // page background color (255=white), matching the black-on-white polarity
    // the rest of the pipeline (binarize → pad(255) → invert) assumes.
    // (kraken's recognition path fills cval=0 here because it later works in
    // inverted ink-high space; our pipeline inverts only at the very end, so
    // we keep the white background consistent throughout.)
    let offset_polygon: Vec<Pt> = boundary
        .iter()
        .map(|&(x, y)| (x - c_min as f64, y - r_min as f64))
        .collect();
    let masked = apply_polygonal_mask(&patch, &offset_polygon, 255u8);

    // Decide dewarp strategy from the baseline shape.
    let baseline_len: f64 = polyline_length(baseline);
    if baseline.len() < 2 || baseline_len < 5.0 {
        // Degenerate: return the masked bbox crop unwarped.
        return Ok(masked);
    }

    if baseline.len() == 2 {
        // Straight baseline: rotation deskew.
        let offset_bl: Vec<Pt> = baseline
            .iter()
            .map(|&(x, y)| (x - c_min as f64, y - r_min as f64))
            .collect();
        Ok(rotate_deskew(&masked, &offset_bl, 255u8))
    } else {
        // Curved baseline: piecewise-affine mesh warp.
        let offset_bl: Vec<Pt> = baseline
            .iter()
            .map(|&(x, y)| (x - c_min as f64, y - r_min as f64))
            .collect();
        curved_dewarp(&masked, &offset_bl, &offset_polygon)
    }
}

// ───────────────────────── polygon mask ─────────────────────────

/// Mask an image to a polygon, filling the area outside with `cval`.
///
/// Port of `apply_polygonal_mask` (`segmentation.py:1414`): rasterize the
/// polygon to a 1-bit mask (inside=keep, outside=fill) and composite. Uses the
/// even-odd scanline fill convention (same as `boundaries.rs::rasterize_polygon_fill`).
pub fn apply_polygonal_mask(image: &GrayImage, polygon: &[Pt], cval: u8) -> GrayImage {
    let (w, h) = image.dimensions();
    let mut mask = vec![false; (w * h) as usize];
    fill_polygon_mask(&mut mask, polygon, w, h);
    let mut out = GrayImage::from_pixel(w, h, Luma([cval]));
    for y in 0..h {
        for x in 0..w {
            if mask[(y * w + x) as usize] {
                out.put_pixel(x, y, *image.get_pixel(x, y));
            }
        }
    }
    out
}

/// Even-odd scanline fill of `polygon` into a boolean mask of shape `(w,h)`.
fn fill_polygon_mask(mask: &mut [bool], polygon: &[Pt], w: u32, h: u32) {
    let n = polygon.len();
    if n < 3 {
        return;
    }
    for py in 0..h {
        let yc = py as f64 + 0.5;
        let mut xs: Vec<f64> = Vec::with_capacity(n);
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = polygon[i];
            let (xj, yj) = polygon[j];
            if (yi > yc) != (yj > yc) {
                let t = (yc - yi) / (yj - yi);
                xs.push(xi + t * (xj - xi));
            }
            j = i;
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut k = 0;
        while k + 1 < xs.len() {
            let x_start = xs[k].max(0.0);
            let x_end = xs[k + 1].min(w as f64);
            if x_end > x_start {
                let px_start = x_start.floor() as usize;
                let px_end = (x_end.ceil() as usize).min(w as usize);
                for px in px_start..px_end {
                    mask[(py as usize) * w as usize + px] = true;
                }
            }
            k += 2;
        }
    }
}

// ───────────────────────── straight-baseline deskew ─────────────────────────

/// Rotate a patch so a 2-point baseline becomes horizontal.
///
/// Port of the `len(baseline) == 2` branch + `_rotate` (`segmentation.py:452,
/// 1551`). `angle = atan2(dy, dx)`; the output is sized to fit the rotated
/// corners and sampled with **bilinear** interpolation (kraken uses BILINEAR
/// for `order=1`). Outside the rotated extent is filled `cval`.
pub(crate) fn rotate_deskew(image: &GrayImage, baseline: &[Pt], cval: u8) -> GrayImage {
    let (w, h) = image.dimensions();
    let (dx, dy) = (baseline[1].0 - baseline[0].0, baseline[1].1 - baseline[0].1);
    let angle = dy.atan2(dx);
    let (sin, cos) = angle.sin_cos();

    // Forward-map the 4 corners through the rotation to size the output.
    let corners = [
        (0.0_f64, 0.0_f64),
        (0.0, (h - 1) as f64),
        ((w - 1) as f64, (h - 1) as f64),
        ((w - 1) as f64, 0.0),
    ];
    let transformed: Vec<(f64, f64)> = corners
        .iter()
        .map(|&(x, y)| (cos * x + sin * y, -sin * x + cos * y))
        .collect();
    let min_x = transformed.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let min_y = transformed.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_x = transformed.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let max_y = transformed.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let out_w = ((max_x - min_x).round() as i64).max(1) as u32;
    let out_h = ((max_y - min_y).round() as i64).max(1) as u32;
    let tx = -min_x;
    let ty = -min_y;

    let mut out = GrayImage::from_pixel(out_w, out_h, Luma([cval]));
    let wf = w as f64;
    let hf = h as f64;
    for oy in 0..out_h {
        for ox in 0..out_w {
            // Inverse map: undo translation, undo rotation.
            let qx = ox as f64 - tx;
            let qy = oy as f64 - ty;
            let sx = cos * qx - sin * qy;
            let sy = sin * qx + cos * qy;
            if sx < 0.0 || sy < 0.0 || sx > wf - 1.0 || sy > hf - 1.0 {
                continue;
            }
            out.put_pixel(ox, oy, Luma([sample_bilinear(image, sx, sy)]));
        }
    }
    out
}

/// Bilinear sample of a GrayImage at continuous `(x, y)`. Pixels off the edge
/// are clamped to the nearest border sample.
fn sample_bilinear(image: &GrayImage, x: f64, y: f64) -> u8 {
    let (w, h) = image.dimensions();
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let clamp = |xi: i64, yi: i64| -> u8 {
        let cx = xi.clamp(0, (w - 1) as i64) as u32;
        let cy = yi.clamp(0, (h - 1) as i64) as u32;
        image.get_pixel(cx, cy)[0]
    };
    let v00 = clamp(x0, y0) as f64;
    let v10 = clamp(x1, y0) as f64;
    let v01 = clamp(x0, y1) as f64;
    let v11 = clamp(x1, y1) as f64;
    let top = v00 + (v10 - v00) * fx;
    let bot = v01 + (v11 - v01) * fx;
    (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8
}

// ───────────────────────── curved-baseline mesh warp ─────────────────────────

/// Piecewise-affine mesh warp that straightens a curved baseline onto a flat
/// arc-length strip. Port of the `else` (non-legacy, >2 baseline points)
/// branch of `extract_polygons` (`segmentation.py:1568-1628`).
fn curved_dewarp(patch: &GrayImage, baseline: &[Pt], polygon: &[Pt]) -> Result<GrayImage> {
    // 1. Simplify the polygon if very dense, then B-spline subdivide (Chaikin).
    let pl: Vec<Point> = if polygon.len() > 50 {
        let pts: Vec<Point> = polygon.iter().map(|&(x, y)| Point::new(x, y)).collect();
        approximate_polygon(&pts, 2.0)
            .iter()
            .map(|p| (p.x, p.y))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(x, y)| Point::new(x, y))
            .collect()
    } else {
        polygon.iter().map(|&(x, y)| Point::new(x, y)).collect()
    };
    let pl_tuples: Vec<Pt> = pl.iter().map(|p| (p.x, p.y)).collect();
    let full_polygon = subdivide_polygon(&pl_tuples, true);

    // 2. Baseline segment vectors, cumulative arc length, unit tangents/normals.
    let l_bl = baseline.len();
    let diff_bl: Vec<Pt> = (0..l_bl - 1)
        .map(|i| (baseline[i + 1].0 - baseline[i].0, baseline[i + 1].1 - baseline[i].1))
        .collect();
    let diff_bl_norms: Vec<f64> = diff_bl.iter().map(|(dx, dy)| (dx * dx + dy * dy).sqrt()).collect();
    let diff_bl_normed: Vec<Pt> = diff_bl
        .iter()
        .zip(diff_bl_norms.iter())
        .map(|((dx, dy), n)| (*dx / n, *dy / n))
        .collect();

    let mut cum_lens = vec![0.0f64; l_bl];
    for i in 1..l_bl {
        cum_lens[i] = cum_lens[i - 1] + diff_bl_norms[i - 1];
    }

    // 3. Project each polygon vertex onto its nearest baseline segment →
    //    destination (arc_len, signed_perp_dist). Mirrors the einsum + cross.
    let l_poly = full_polygon.len();
    let bl0 = baseline[0];
    let mut pol_dst: Vec<Pt> = Vec::with_capacity(l_poly);
    for p in &full_polygon {
        let (px, py) = (p.0, p.1);
        let mut best_k = 0usize;
        let mut best_dist = f64::INFINITY;
        let mut best_x = 0.0f64;
        let mut best_y = 0.0f64;
        for k in 0..l_bl - 1 {
            // diff = polygon - baseline[k]
            let ddx = px - baseline[k].0;
            let ddy = py - baseline[k].1;
            let (nx, ny) = diff_bl_normed[k];
            // scalar projection onto segment direction (einsum 'kpm,km->kp')
            let x = ddx * nx + ddy * ny;
            // distance-to-segment metric: max(-x, x - seg_len)
            let segdist = (-x).max(x - diff_bl_norms[k]);
            if segdist < best_dist {
                best_dist = segdist;
                best_k = k;
                best_x = x;
                // signed perpendicular distance: np.cross([nx,ny],[ddx,ddy])
                // = nx*ddy - ny*ddx (the z-component of the 2D cross product).
                best_y = nx * ddy - ny * ddx;
            }
        }
        let _ = best_k;
        let dst_x = bl0.0 + cum_lens[best_k] + best_x;
        let dst_y = bl0.1 + best_y;
        pol_dst.push((dst_x, dst_y));
    }

    // 4. Output shape from destination polygon bbox.
    let c_dst_min = pol_dst.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).floor() as i64;
    let c_dst_max = pol_dst.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).ceil() as i64;
    let r_dst_min = pol_dst.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).floor() as i64;
    let r_dst_max = pol_dst.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).ceil() as i64;
    let out_w = ((c_dst_max - c_dst_min + 1).max(1)) as usize;
    let out_h = ((r_dst_max - r_dst_min + 1).max(1)) as usize;

    // 5. Destination baseline start in output-local coords.
    let bl_dst0_x = bl0.0 - c_dst_min as f64;
    let bl_dst0_y = bl0.1 - r_dst_min as f64;
    let output_bl_start: Pt = (bl_dst0_x, bl_dst0_y);

    // 6. Build the bevelled warping envelope (source in patch coords, target
    //    in output coords).
    let (source_env, target_env) =
        bevelled_warping_envelope(baseline, output_bl_start, (out_h, out_w));

    // 7. Mesh warp via per-strip inverse affine + bilinear sampling.
    Ok(mesh_warp(patch, &source_env, &target_env, out_w, out_h, 255u8))
}

/// `_bevelled_warping_envelope` (`segmentation.py:1334`). Walks baseline joints,
/// rounding each corner with a quadratic Bézier and emitting top/bottom source
/// points (perpendicular offsets of the full strip height) and matching
/// axis-aligned target column points.
///
/// `output_shape` is `(rows, cols)` per numpy convention (height, width).
/// Returns `(source_envelope, target_envelope)` as flat `Vec<Pt>` organized in
/// top/bottom pairs per column.
fn bevelled_warping_envelope(
    baseline: &[Pt],
    output_bl_start: Pt,
    output_shape: (usize, usize),
) -> (Vec<Pt>, Vec<Pt>) {
    let (out_h, out_w) = output_shape;
    let envelope_dy = [-output_bl_start.1, out_h as f64 - output_bl_start.1];

    let diff_bl: Vec<Pt> = (0..baseline.len() - 1)
        .map(|i| (baseline[i + 1].0 - baseline[i].0, baseline[i + 1].1 - baseline[i].1))
        .collect();
    let diff_bl_norms: Vec<f64> = diff_bl.iter().map(|(dx, dy)| (dx * dx + dy * dy).sqrt()).collect();
    let diff_bl_normed: Vec<Pt> = diff_bl
        .iter()
        .zip(diff_bl_norms.iter())
        .map(|((dx, dy), n)| (*dx / n, *dy / n))
        .collect();
    let bl_seg_normals: Vec<Pt> = diff_bl_normed.iter().map(|(tx, ty)| (-ty, *tx)).collect();

    let mut cum_lens = vec![0.0f64; baseline.len()];
    for i in 1..baseline.len() {
        cum_lens[i] = cum_lens[i - 1] + diff_bl_norms[i - 1];
    }

    let as_int = |t: Pt| ((t.0 as i64) as f64, (t.1 as i64) as f64);

    let l_bl = baseline.len();
    let ini_point = (
        baseline[0].0 - diff_bl_normed[0].0 * output_bl_start.0,
        baseline[0].1 - diff_bl_normed[0].1 * output_bl_start.0,
    );
    let mut source_env: Vec<Pt> = vec![
        as_int((
            ini_point.0 + envelope_dy[0] * bl_seg_normals[0].0,
            ini_point.1 + envelope_dy[0] * bl_seg_normals[0].1,
        )),
        as_int((
            ini_point.0 + envelope_dy[1] * bl_seg_normals[0].0,
            ini_point.1 + envelope_dy[1] * bl_seg_normals[0].1,
        )),
    ];
    let mut target_env: Vec<Pt> = vec![(0.0, 0.0), (0.0, out_h as f64)];

    let max_bevel_width = out_h as f64 / 3.0;
    let bevel_step_width = max_bevel_width / 2.0;

    for k in 0..(l_bl - 2) {
        let pt = baseline[k + 1];
        let seg_prev = (baseline[k].0 - pt.0, baseline[k].1 - pt.1);
        let seg_next = (baseline[k + 2].0 - pt.0, baseline[k + 2].1 - pt.1);
        let len_prev_full = (seg_prev.0 * seg_prev.0 + seg_prev.1 * seg_prev.1).sqrt();
        let len_next_full = (seg_next.0 * seg_next.0 + seg_next.1 * seg_next.1).sqrt();
        let scale_prev = 1.0 / 2.0_f64.max(len_prev_full / max_bevel_width);
        let scale_next = 1.0 / 2.0_f64.max(len_next_full / max_bevel_width);
        let bevel_prev = (seg_prev.0 * scale_prev, seg_prev.1 * scale_prev);
        let bevel_next = (seg_next.0 * scale_next, seg_next.1 * scale_next);
        let l_prev = (bevel_prev.0 * bevel_prev.0 + bevel_prev.1 * bevel_prev.1).sqrt();
        let l_next = (bevel_next.0 * bevel_next.0 + bevel_next.1 * bevel_next.1).sqrt();
        let bevel_nsteps =
            (((l_prev + l_next) / bevel_step_width).round() as i64).max(1) as usize;

        for i in 0..=bevel_nsteps {
            let t = i as f64 / bevel_nsteps as f64;
            let omt = 1.0 - t;
            // Quadratic Bézier corner rounding.
            let tpt = (
                pt.0 + omt * omt * bevel_prev.0 + t * t * bevel_next.0,
                pt.1 + omt * omt * bevel_prev.1 + t * t * bevel_next.1,
            );
            let tx = output_bl_start.0 + cum_lens[k + 1] - omt * omt * l_prev + t * t * l_next;
            // Interpolated, renormalized normal.
            let nrm = (
                omt * bl_seg_normals[k].0 + t * bl_seg_normals[k + 1].0,
                omt * bl_seg_normals[k].1 + t * bl_seg_normals[k + 1].1,
            );
            let nlen = (nrm.0 * nrm.0 + nrm.1 * nrm.1).sqrt().max(1e-12);
            let tnormal = (nrm.0 / nlen, nrm.1 / nlen);
            let src_top = as_int((
                tpt.0 + envelope_dy[0] * tnormal.0,
                tpt.1 + envelope_dy[0] * tnormal.1,
            ));
            let src_bot = as_int((
                tpt.0 + envelope_dy[1] * tnormal.0,
                tpt.1 + envelope_dy[1] * tnormal.1,
            ));
            let tgt_top = (tx.trunc(), 0.0);
            let tgt_bot = (tx.trunc(), out_h as f64);
            // Dedup guard against singularities.
            if src_top == source_env[source_env.len() - 2]
                || src_bot == source_env[source_env.len() - 1]
                || tgt_top == target_env[target_env.len() - 2]
            {
                continue;
            }
            source_env.push(src_top);
            source_env.push(src_bot);
            target_env.push(tgt_top);
            target_env.push(tgt_bot);
        }
    }

    // End column: extend the last tangent to reach the output width.
    let last = baseline.len() - 1;
    let end_point = (
        baseline[last].0 + diff_bl_normed[last - 1].0 * (out_w as f64 - cum_lens[last] - output_bl_start.0),
        baseline[last].1 + diff_bl_normed[last - 1].1 * (out_w as f64 - cum_lens[last] - output_bl_start.0),
    );
    source_env.push(as_int((
        end_point.0 + envelope_dy[0] * bl_seg_normals[last - 1].0,
        end_point.1 + envelope_dy[0] * bl_seg_normals[last - 1].1,
    )));
    source_env.push(as_int((
        end_point.0 + envelope_dy[1] * bl_seg_normals[last - 1].0,
        end_point.1 + envelope_dy[1] * bl_seg_normals[last - 1].1,
    )));
    target_env.push((out_w as f64, 0.0));
    target_env.push((out_w as f64, out_h as f64));

    (source_env, target_env)
}

/// Piecewise-affine mesh warp. For each output pixel, locate the containing
/// target strip (by monotonic target x), apply that strip's inverse affine
/// (target→source, fit by least squares on the 4 corner correspondences), and
/// bilinearly sample the source patch. Matches PIL `Image.MESH` semantics.
///
/// Envelopes are organized as top/bottom pairs per column: indices `2c` (top)
/// and `2c+1` (bottom). Strip `c` spans columns `c` and `c+1`, using source
/// corners `[2c, 2c+1, 2c+2, 2c+3]` = (TL, BL, TR, BR).
fn mesh_warp(
    patch: &GrayImage,
    source_env: &[Pt],
    target_env: &[Pt],
    out_w: usize,
    out_h: usize,
    cval: u8,
) -> GrayImage {
    // Build per-strip target_x boundaries and per-strip affine (target→source).
    // Number of columns = env.len()/2; strips = columns - 1.
    let n_cols = target_env.len() / 2;
    if n_cols < 2 {
        return GrayImage::from_pixel(out_w as u32, out_h as u32, Luma([cval]));
    }
    // Target x per column = target_env[2c].0 (the top point's x).
    let target_x: Vec<f64> = (0..n_cols).map(|c| target_env[2 * c].0).collect();

    // Precompute affine params per strip: source = M · target + b for x and y.
    // Solve by least squares on the 4 corner correspondences (target rect →
    // source quad). Target corners: TL=(x0,0), BL=(x0,H), TR=(x1,0), BR=(x1,H).
    struct StripAffine {
        a: f64, b: f64, c: f64, // sx = a*tx + b*ty + c
        d: f64, e: f64, f: f64, // sy = d*tx + e*ty + f
    }
    let strips: Vec<StripAffine> = (0..n_cols - 1)
        .map(|c| {
            let tl_t = (target_x[c], 0.0);
            let bl_t = (target_x[c], out_h as f64);
            let tr_t = (target_x[c + 1], 0.0);
            let br_t = (target_x[c + 1], out_h as f64);
            let tl_s = source_env[2 * c];
            let bl_s = source_env[2 * c + 1];
            let tr_s = source_env[2 * c + 2];
            let br_s = source_env[2 * c + 3];
            let (a, b, cc) = fit_affine_1d(&[tl_t, bl_t, tr_t, br_t], &[tl_s.0, bl_s.0, tr_s.0, br_s.0]);
            let (d, e, f) = fit_affine_1d(&[tl_t, bl_t, tr_t, br_t], &[tl_s.1, bl_s.1, tr_s.1, br_s.1]);
            StripAffine { a, b, c: cc, d, e, f }
        })
        .collect();

    let mut out = GrayImage::from_pixel(out_w as u32, out_h as u32, Luma([cval]));
    let pw = patch.width() as f64;
    let ph = patch.height() as f64;
    for oy in 0..out_h {
        // The strip index must restart at the left edge each row (target_x is
        // monotonic in x, so we scan forward within the row).
        let mut strip = 0usize;
        for ox in 0..out_w {
            let oxf = ox as f64 + 0.5;
            // Advance strip index while ox is past this strip's right edge.
            while strip + 1 < strips.len() && oxf >= target_x[strip + 1] {
                strip += 1;
            }
            let s = &strips[strip];
            let tx = oxf;
            let ty = oy as f64 + 0.5;
            let sx = s.a * tx + s.b * ty + s.c;
            let sy = s.d * tx + s.e * ty + s.f;
            if sx < 0.0 || sy < 0.0 || sx > pw - 1.0 || sy > ph - 1.0 {
                continue;
            }
            out.put_pixel(ox as u32, oy as u32, Luma([sample_bilinear(patch, sx, sy)]));
        }
    }
    out
}

/// Least-squares fit of `out = a*x + b*y + c` to the 4 `(target_xy, out)`
/// correspondences. Solves the 3x3 normal equations (closed form). Used to fit
/// an affine target→source per mesh strip.
fn fit_affine_1d(targets: &[Pt; 4], outs: &[f64; 4]) -> (f64, f64, f64) {
    // Normal equations for [a, b, c] with rows [x, y, 1]:
    //   Σx²·a + Σxy·b + Σx·c = Σx·o
    //   Σxy·a + Σy²·b + Σy·c = Σy·o
    //   Σx·a  + Σy·b  + N·c   = Σo
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut so = 0.0;
    let mut sxo = 0.0;
    let mut syo = 0.0;
    for (t, &o) in targets.iter().zip(outs.iter()) {
        sxx += t.0 * t.0;
        sxy += t.0 * t.1;
        syy += t.1 * t.1;
        sx += t.0;
        sy += t.1;
        so += o;
        sxo += t.0 * o;
        syo += t.1 * o;
    }
    let n = outs.len() as f64;
    // Solve 3x3 system via Cramer's rule.
    let m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
    let rhs = [sxo, syo, so];
    let det = det3(&m);
    if det.abs() < 1e-12 {
        return (0.0, 0.0, outs[0]);
    }
    let a = det3(&[rhs, [sxy, syy, sy], [sx, sy, n]]) / det;
    let b = det3(&[[sxx, sxy, sx], rhs, [sx, sy, n]]) / det;
    let c = det3(&[[sxx, sxy, sx], [sxy, syy, sy], rhs]) / det;
    (a, b, c)
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ───────────────────────── shared helpers ─────────────────────────

/// B-spline polygon subdivision (degree 2), port of skimage
/// `subdivide_polygon(coords, degree=2, preserve_ends)`. For each edge `(a, b)`
/// emits Chaikin corner-cut points `(3a+b)/4` and `(a+3b)/4`; when
/// `preserve_ends` is set the original first/last vertices are re-attached.
/// (In scipy's `'valid'` convolve mode the `boundary='wrap'` is irrelevant, so
/// no wraparound is applied for open polylines.)
fn subdivide_polygon(coords: &[Pt], preserve_ends: bool) -> Vec<Pt> {
    let n = coords.len();
    if n < 2 {
        return coords.to_vec();
    }
    let mut out: Vec<Pt> = Vec::with_capacity(2 * n);
    if preserve_ends {
        out.push(coords[0]);
    }
    for i in 0..n - 1 {
        let (ax, ay) = coords[i];
        let (bx, by) = coords[i + 1];
        // Q = (3a + b)/4, R = (a + 3b)/4
        out.push(((3.0 * ax + bx) / 4.0, (3.0 * ay + by) / 4.0));
        out.push(((ax + 3.0 * bx) / 4.0, (ay + 3.0 * by) / 4.0));
    }
    if preserve_ends {
        out.push(coords[n - 1]);
    }
    out
}

/// Sum of segment lengths of a polyline.
fn polyline_length(pts: &[Pt]) -> f64 {
    pts.windows(2)
        .map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt())
        .sum()
}

/// Axis-aligned bbox of a polygon as `(min_x, min_y, max_x, max_y)`, clamped to
/// the image bounds.
fn polygon_bbox(boundary: &[Pt], img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
    let min_x = boundary.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).max(0.0) as u32;
    let min_y = boundary.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(0.0) as u32;
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
    (min_x, min_y, max_x, max_y)
}

/// Crop a `GrayImage` to `(min_x, min_y, w, h)`, returning an owned copy.
fn crop_luma(image: &GrayImage, min_x: u32, min_y: u32, w: u32, h: u32) -> GrayImage {
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            out.put_pixel(x, y, *image.get_pixel(min_x + x, min_y + y));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    fn img_with_horizontal_line(h: u32, w: u32, row: u32) -> GrayImage {
        let mut im = GrayImage::from_pixel(w, h, Luma([255]));
        for x in 0..w {
            im.put_pixel(x, row, Luma([0]));
        }
        im
    }

    #[test]
    fn test_subdivide_polygon_doubles_and_preserves_ends() {
        let coords = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
        let out = subdivide_polygon(&coords, true);
        // 2*(n-1) interior points + 2 endpoints = 2*2 + 2 = 6.
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], coords[0], "first endpoint preserved");
        assert_eq!(out[5], coords[2], "last endpoint preserved");
        // First Chaikin point on edge 0: (3*0+10)/4 = 2.5.
        assert!((out[1].0 - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_apply_polygonal_mask_fills_outside() {
        // 9x9 image all-white; mask to a 3x3 square in the center.
        let img = GrayImage::from_pixel(9, 9, Luma([200]));
        let poly = vec![(3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (3.0, 5.0)];
        let out = apply_polygonal_mask(&img, &poly, 0);
        // Center pixel kept.
        assert_eq!(out.get_pixel(4, 4)[0], 200);
        // Far corner zeroed.
        assert_eq!(out.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn test_rotate_deskew_horizontalizes_tilted_line() {
        // A patch containing a diagonal black line (slope 1). After deskew the
        // line's vertical span across columns should shrink.
        let (h, w) = (32u32, 64u32);
        let mut im = GrayImage::from_pixel(w, h, Luma([255]));
        for x in 0..w {
            let y = (x / 2) as u32; // diagonal
            if y < h {
                im.put_pixel(x, y, Luma([0]));
            }
        }
        // Baseline along the diagonal direction (angle ~ atan2(0.5,1)).
        let baseline = vec![(0.0, 0.0), (63.0, 31.0)];
        let out = rotate_deskew(&im, &baseline, 255);
        // After deskew, ink should occupy far fewer distinct rows than before.
        let mut rows_before = 0;
        for y in 0..h {
            if (0..w).any(|x| im.get_pixel(x, y)[0] < 128) {
                rows_before += 1;
            }
        }
        let mut rows_after = 0;
        for y in 0..out.height() {
            if (0..out.width()).any(|x| out.get_pixel(x, y)[0] < 128) {
                rows_after += 1;
            }
        }
        assert!(
            rows_after <= rows_before,
            "deskew should not increase row span: before={rows_before} after={rows_after}"
        );
    }

    #[test]
    fn test_curved_dewarp_straightens_arc() {
        // Synthesize a patch with a horizontal black band that sags in the
        // middle (parabolic), plus a matching curved baseline and a polygon
        // hugging the band. After curved_dewarp the band should be flatter.
        let (h, w) = (50usize, 200usize);
        let mut im = GrayImage::from_pixel(w as u32, h as u32, Luma([255]));
        // Band center row sags from 15 at the edges to 30 in the middle.
        let center = |xf: f64| 15.0 + 15.0 * ((xf - (w as f64) / 2.0) / (w as f64 / 2.0)).powi(2);
        for x in 0..w {
            let c = center(x as f64);
            for dy in 0..5 {
                let y = (c as usize) + dy;
                if y < h {
                    im.put_pixel(x as u32, y as u32, Luma([0]));
                }
            }
        }
        // Curved baseline following the band center.
        let baseline: Vec<Pt> = (0..=8)
            .map(|i| {
                let xf = (i as f64 / 8.0) * (w - 1) as f64;
                (xf, center(xf))
            })
            .collect();
        // Polygon hugging the band (top = center-6, bottom = center+11), as a
        // closed ring traced top-then-bottom-reversed.
        let mut polygon: Vec<Pt> = Vec::new();
        for i in 0..=8 {
            let xf = (i as f64 / 8.0) * (w - 1) as f64;
            polygon.push((xf, center(xf) - 6.0));
        }
        for i in (0..=8).rev() {
            let xf = (i as f64 / 8.0) * (w - 1) as f64;
            polygon.push((xf, center(xf) + 11.0));
        }
        let dyn_img = DynamicImage::ImageLuma8(im.clone());
        let out = extract_polygon_line(&dyn_img, &baseline, &polygon).unwrap();
        assert!(out.width() > 0 && out.height() > 0);

        // After dewarp the band should be ~horizontal: column-wise ink center
        // has low variance. Only count "band" columns (ink covers < 50% of the
        // column) to avoid counting the cval=0 mask fill as ink.
        let band_center_var = |im: &GrayImage| -> f64 {
            let (w, h) = im.dimensions();
            let mut centers = Vec::new();
            for x in 0..w {
                let mut rows = Vec::new();
                for y in 0..h {
                    if im.get_pixel(x, y)[0] < 128 {
                        rows.push(y as f64);
                    }
                }
                if !rows.is_empty() && (rows.len() as f64) < 0.5 * (h as f64) {
                    centers.push(rows.iter().sum::<f64>() / rows.len() as f64);
                }
            }
            if centers.len() < 2 {
                return f64::INFINITY;
            }
            let mean = centers.iter().sum::<f64>() / centers.len() as f64;
            centers.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / centers.len() as f64
        };
        let before = band_center_var(&im);
        let after = band_center_var(&out);
        assert!(
            after < before,
            "curved dewarp should flatten the band; ink-center variance before={before:.2} after={after:.2}"
        );
    }

    #[test]
    fn test_extract_polygon_line_degenerate_baseline_falls_back() {
        // A single-point baseline should fall back to the masked bbox crop
        // without panicking.
        let img = DynamicImage::ImageLuma8(img_with_horizontal_line(20, 40, 10));
        let boundary = vec![(0.0, 0.0), (39.0, 0.0), (39.0, 19.0), (0.0, 19.0)];
        let baseline = vec![(20.0, 10.0)]; // length < 2
        let out = extract_polygon_line(&img, &baseline, &boundary).unwrap();
        assert!(out.width() > 0 && out.height() > 0);
    }

    #[test]
    fn test_curved_dewarp_straightens_tilted_multipoint_baseline() {
        // A straight-but-tilted band (constant slope, no curvature) with a
        // multi-point colinear baseline — the geometry this repo's segmenter
        // actually emits for tilted text (it never produces 2-point baselines,
        // so the rotate_deskew fast path never fires; tilted lines go through
        // curved_dewarp). The mesh warp must deskew it to horizontal.
        let (h, w) = (60usize, 200usize);
        let mut im = GrayImage::from_pixel(w as u32, h as u32, Luma([255]));
        // Band slopes down 0.15 px/px: center row goes 15 → 45 across the width.
        let slope = 0.15f64;
        let center = |xf: f64| 15.0 + slope * xf;
        for x in 0..w {
            let c = center(x as f64);
            for dy in 0..6 {
                let y = (c as usize) + dy;
                if y < h {
                    im.put_pixel(x as u32, y as u32, Luma([0]));
                }
            }
        }
        // Colinear multi-point baseline along the band center (5 points).
        let baseline: Vec<Pt> = (0..=4)
            .map(|i| {
                let xf = (i as f64 / 4.0) * (w - 1) as f64;
                (xf, center(xf))
            })
            .collect();
        // Polygon hugging the band.
        let mut polygon: Vec<Pt> = Vec::new();
        for i in 0..=4 {
            let xf = (i as f64 / 4.0) * (w - 1) as f64;
            polygon.push((xf, center(xf) - 7.0));
        }
        for i in (0..=4).rev() {
            let xf = (i as f64 / 4.0) * (w - 1) as f64;
            polygon.push((xf, center(xf) + 13.0));
        }
        let dyn_img = DynamicImage::ImageLuma8(im.clone());
        let out = extract_polygon_line(&dyn_img, &baseline, &polygon).unwrap();
        assert!(out.width() > 0 && out.height() > 0);

        // After deskew the band is horizontal: column ink-center variance must
        // drop sharply vs the tilted input.
        let band_center_var = |im: &GrayImage| -> f64 {
            let (w, h) = im.dimensions();
            let mut centers = Vec::new();
            for x in 0..w {
                let mut rows = Vec::new();
                for y in 0..h {
                    if im.get_pixel(x, y)[0] < 128 {
                        rows.push(y as f64);
                    }
                }
                if !rows.is_empty() && (rows.len() as f64) < 0.5 * (h as f64) {
                    centers.push(rows.iter().sum::<f64>() / rows.len() as f64);
                }
            }
            if centers.len() < 2 {
                return f64::INFINITY;
            }
            let mean = centers.iter().sum::<f64>() / centers.len() as f64;
            centers.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / centers.len() as f64
        };
        let before = band_center_var(&im);
        let after = band_center_var(&out);
        assert!(
            after < before * 0.2,
            "curved_dewarp should deskew a tilted band; variance before={before:.2} after={after:.2} (should be <20% of before)"
        );
    }
}
