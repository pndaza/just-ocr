//! Rust port of PaddleOCR's `polygons_from_bitmap` postprocess — renders
//! pre-unclip (green) and post-unclip (red) poly-box polygons on the source
//! image, so the 1.4x expansion is visible. Mirrors clones/PaddleOCR's
//! `expand_compare.py` but runs entirely off the vendored `ppocr-engine`,
//! no Python env.
//!
//! Pipeline (faithful to ppocr/postprocess/db_postprocess.py:59 + 160):
//!   score_map → binary mask → connected components → ordered contour
//!   (kraken_engine::contours::boundary_trace) → Douglas-Peucker simplify
//!   (kraken_engine::polygon::simplify) → box_score_fast filter → pyclipper
//!   unclip via clipper2-rust (ClipperOffset + JoinType::Round) → scale to
//!   source → render.
//!
//! Run with:
//!
//!   cargo run --release --example expand_compare_ppocr -- <image> [out.png] [ratio]
//!
//! Defaults to ../sample_images/heavy_curve_01.png, ratio 1.4; writes
//! <stem>-expand-<ratio>.png. Green = pre-unclip, red = post-unclip.

use std::time::Instant;

use image::{GenericImageView, Rgba};
use ppocr_engine::{Detector, DetectorConfig};

/// Bundled small-det (path matches the other examples).
const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");
const DEFAULT_IMAGE: &str = "../sample_images/heavy_curve_01.png";

// Postprocess defaults — match DetectorPostprocessOptions::default() and the
// PaddleOCR PP-OCRv6 det config used by expand_compare.py.
const THRESH: f32 = 0.2;
const BOX_THRESH: f32 = 0.45;
const MIN_AREA: usize = 3;
/// approxPolyDP epsilon as a fraction of arc length (db_postprocess.py:76).
const POLY_EPS: f64 = 0.002;

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_IMAGE.to_string());
    let unclip_ratio: f64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.4);
    /// Axis stretch factor (4th CLI arg, optional). Scales each polygon about
    /// its centroid along its PCA principal axis AFTER the unclip offset, so
    /// the line ends extend further. 1.0 = no stretch (offset only). 1.05 →
    /// endpoints move out by 2.5% of the line's half-width each.
    let axis_stretch: f64 = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    println!("Loading image: {img_path}");
    let img = image::open(&img_path)?;
    let (w, h) = img.dimensions();
    println!("Image dimensions: {w}x{h}  unclip_ratio={unclip_ratio}");

    let t = Instant::now();
    let det = Detector::load_from_buffer_with_config(
        BUNDLED_PPOCR_DET,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        DetectorConfig::small(),
    )?;
    println!("  detector loaded in {:?}", t.elapsed());

    let t = Instant::now();
    let (values, ih, iw, transform) = det.detect_raw(&img)?;
    println!("  forward in {:?}: score map {}x{}", t.elapsed(), iw, ih);
    let cw = transform.content_width() as usize;
    let ch = transform.content_height() as usize;

    // --- Connected components (8-conn) over the binary mask. ---
    let components = collect_components(&values, iw, cw, ch, THRESH, MIN_AREA);
    println!("Connected components (min_area={MIN_AREA}): {}", components.len());

    // --- Per component: contour → simplify → score → unclip. ---
    let mut pre_polys: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut post_polys: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut dropped_score = 0u32;
    let mut dropped_unclip = 0u32;

    for px in &components {
        // boundary_trace wants (y, x) tuples.
        let yx: Vec<(usize, usize)> = px.iter().map(|&(x, y)| (y as usize, x as usize)).collect();
        let contour = kraken_engine::contours::boundary_trace(&yx);
        if contour.len() < 4 {
            continue;
        }

        // Douglas-Peucker: epsilon = POLY_EPS * perimeter (closed).
        let perimeter = closed_perimeter(&contour);
        let eps = POLY_EPS * perimeter;
        let simplified = kraken_engine::polygon::simplify(&contour, eps);
        if simplified.len() < 4 {
            continue;
        }

        // box_score_fast: mean of pred under the polygon mask, over its bbox.
        let score = box_score_fast(&values, iw, &simplified);
        if score < BOX_THRESH {
            dropped_score += 1;
            continue;
        }

        // unclip (pyclipper JT_ROUND, ET_CLOSEDPOLYGON). distance = area*ratio/perim.
        let area = shoelace_area(&simplified);
        let distance = area * unclip_ratio / perimeter.max(1e-9);
        let expanded = match unclip_round(&simplified, distance) {
            Some(p) => p,
            None => {
                dropped_unclip += 1;
                continue;
            }
        };
        if expanded.len() < 4 {
            dropped_unclip += 1;
            continue;
        }

        // Scale bitmap coords → source-image pixel space.
        let to_src = |p: &(f64, f64)| -> (f64, f64) {
            (
                transform.map_x_to_source(p.0 as f32) as f64,
                transform.map_y_to_source(p.1 as f32) as f64,
            )
        };
        let pre: Vec<(f64, f64)> = simplified
            .iter()
            .map(|p| (p.x, p.y))
            .map(|p| to_src(&p))
            .collect();
        // Optional axis stretch about the polygon's centroid, applied AFTER the
        // unclip offset so the line ends extend further (the middle stays put).
        let expanded_final = if (axis_stretch - 1.0).abs() > 1e-9 {
            stretch_along_axis(&expanded, axis_stretch)
        } else {
            expanded
        };
        let post: Vec<(f64, f64)> = expanded_final.iter().copied().map(|p| to_src(&p)).collect();
        pre_polys.push(pre);
        post_polys.push(post);
    }

    let pre_pts: usize = pre_polys.iter().map(|p| p.len()).sum();
    let post_pts: usize = post_polys.iter().map(|p| p.len()).sum();
    println!(
        "Polygons: {} kept (dropped {} on score, {} on unclip); pre pts {pre_pts}, post pts {post_pts}",
        post_polys.len(),
        dropped_score,
        dropped_unclip,
    );

    // --- Render: green pre-unclip, red post-unclip, on the source. ---
    let mut canvas = img.to_rgba8();
    let green = Rgba([0, 200, 0, 255]);
    let red = Rgba([255, 0, 0, 255]);
    for p in &pre_polys {
        draw_polyline_closed(&mut canvas, p, green);
    }
    for p in &post_polys {
        draw_polyline_closed(&mut canvas, p, red);
    }

    let out_path = std::env::args().nth(2).unwrap_or_else(|| {
        let p = std::path::Path::new(&img_path);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
        let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
        parent
            .join(format!("{stem}-expand-{unclip_ratio}.{ext}"))
            .to_string_lossy()
            .into_owned()
    });
    canvas.save(&out_path)?;
    println!(
        "\nWrote {}x{} overlay to: {out_path}\n  green: pre-unclip polygon\n  red:   post-unclip (ratio {unclip_ratio}, axis_stretch {axis_stretch})",
        canvas.width(),
        canvas.height(),
    );
    Ok(())
}

// === helpers ===

/// 8-connectivity CC labeling → one Vec<(x,y)> pixel set per component.
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

/// Stretch a polygon about its centroid along its PCA principal axis by
/// `factor`. Points off-axis move proportionally to their along-axis position:
/// endpoints (at ±W/2) move by (factor−1)·W/2, the centroid stays fixed, and
/// perpendicular (across-axis) position is preserved. So a horizontal line's
/// left/right ends extend further while top/bottom stay as-is.
///
/// Used after `unclip` to give the line ends extra reach beyond the uniform
/// offset, which is ~text-height-scaled and barely extends long lines.
fn stretch_along_axis(poly: &[(f64, f64)], factor: f64) -> Vec<(f64, f64)> {
    if poly.len() < 2 || (factor - 1.0).abs() < 1e-9 {
        return poly.to_vec();
    }
    let n = poly.len() as f64;
    let cx = poly.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = poly.iter().map(|p| p.1).sum::<f64>() / n;
    // PCA via covariance: principal axis = dominant eigenvector.
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for &(x, y) in poly {
        let (dx, dy) = (x - cx, y - cy);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (ax, ay) = (angle.cos(), angle.sin()); // principal axis unit vector
    let (nx, ny) = (-ay, ax); // perpendicular (normal)

    poly.iter()
        .map(|&(x, y)| {
            let (dx, dy) = (x - cx, y - cy);
            // Project onto axis (stretch) and normal (keep).
            let along = dx * ax + dy * ay;
            let across = dx * nx + dy * ny;
            let along = along * factor;
            (cx + along * ax + across * nx, cy + along * ay + across * ny)
        })
        .collect()
}

/// Perimeter of a closed polygon (closing edge included), polygon::Point space.
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

/// box_score_fast (db_postprocess.py:189): mean of `pred` over the polygon's
/// bbox, masked by the rasterized polygon. Polygon coords are in score-map
/// pixel space (f64); we sample pred at integer offsets.
fn box_score_fast(pred: &[f32], map_w: usize, poly: &[kraken_engine::polygon::Point]) -> f32 {
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in poly {
        xmin = xmin.min(p.x); xmax = xmax.max(p.x);
        ymin = ymin.min(p.y); ymax = ymax.max(p.y);
    }
    let h = pred.len() / map_w;
    let x0 = xmin.floor().clamp(0.0, (map_w - 1) as f64) as usize;
    let x1 = xmax.ceil().clamp(0.0, (map_w - 1) as f64) as usize;
    let y0 = ymin.floor().clamp(0.0, (h - 1) as f64) as usize;
    let y1 = ymax.ceil().clamp(0.0, (h - 1) as f64) as usize;
    if x1 < x0 || y1 < y0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut cnt = 0u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            if point_in_polygon(x as f64 + 0.5, y as f64 + 0.5, poly) {
                sum += pred[y * map_w + x];
                cnt += 1;
            }
        }
    }
    if cnt == 0 { 0.0 } else { sum / cnt as f32 }
}

/// Ray-cast point-in-polygon.
fn point_in_polygon(x: f64, y: f64, poly: &[kraken_engine::polygon::Point]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i].x, poly[i].y);
        let (xj, yj) = (poly[j].x, poly[j].y);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// pyclipper-equivalent unclip: offset a closed polygon outward by `delta`
/// using Clipper2 with round joins. Returns None if Clipper yields zero or
/// more than one solution polygon (matches polygons_from_bitmap's drop rule).
fn unclip_round(
    poly: &[kraken_engine::polygon::Point],
    delta: f64,
) -> Option<Vec<(f64, f64)>> {
    // Use Clipper2's float (PathD) entry point, which internally scales to
    // int64 at 10^precision and scales back — mirroring pyclipper's precision
    // handling. Kept in sync with segmenter_adapters::unclip_round; this copy
    // exists only because examples can't reach the app crate's private fn.
    use clipper2_rust::clipper::inflate_paths_d;
    use clipper2_rust::offset::{EndType, JoinType};
    use clipper2_rust::{PathD, PathsD};

    let path: PathD = poly
        .iter()
        .map(|p| clipper2_rust::Point::new(p.x, p.y))
        .collect();
    let mut paths = PathsD::new();
    paths.push(path);
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

/// Draw a closed polyline (thickened by neighbor stamping so it reads at full
/// source resolution).
fn draw_polyline_closed(canvas: &mut image::RgbaImage, pts: &[(f64, f64)], color: Rgba<u8>) {
    if pts.len() < 2 {
        return;
    }
    let cw = canvas.width() as isize;
    let ch = canvas.height() as isize;
    let mut stamp = |x: f64, y: f64| {
        for &(dx, dy) in &[
            (0isize, 0isize), (1, 0), (0, 1), (1, 1),
            (-1, 0), (0, -1), (-1, -1),
        ] {
            let px = (x.round() as isize + dx).clamp(0, cw - 1) as u32;
            let py = (y.round() as isize + dy).clamp(0, ch - 1) as u32;
            canvas.put_pixel(px, py, color);
        }
    };
    // Bresenham-ish per segment (integer stepping).
    for w in pts.windows(2) {
        let (x0, y0) = (w[0].0, w[0].1);
        let (x1, y1) = (w[1].0, w[1].1);
        let steps = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize).max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
        }
    }
    // Closing edge.
    let (x0, y0) = (pts[0].0, pts[0].1);
    let (x1, y1) = (pts[pts.len() - 1].0, pts[pts.len() - 1].1);
    let steps = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize).max(1);
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
    }
}
