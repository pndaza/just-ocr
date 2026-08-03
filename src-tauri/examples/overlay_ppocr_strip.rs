//! Centerline extraction for PP-OCR detections, working directly on each
//! connected component's pixel set in image space — no PCA, no projection.
//!
//! For each CC we slice its horizontal extent into N bands and find the
//! **vertical center** (median y of the component's pixels) in each band.
//! Connecting those N points gives a centerline that follows the text whether
//! it's straight or curved. This is the raw artifact; once it's visibly correct
//! we build the top/bottom extents and the polygon on top of it.
//!
//! Renders on the source image:
//!   - green  = the CC's axis-aligned bbox (reference)
//!   - yellow = the N-point centerline (dots + connecting segments)
//!
//! Run with:
//!
//!   cargo run --release --example overlay_ppocr_strip -- <image.png> [out.png]
//!
//! Defaults to `../sample_images/curve_lines_01.png`; writes `<stem>-strip.png`.

use std::time::Instant;

use image::{GenericImageView, Rgba};
use imageproc::drawing::draw_line_segment_mut;
use imageproc::point::Point as Ipt;
use ppocr_engine::{Detector, DetectorConfig};

type Point = Ipt<f32>;

/// Bundled small-det. Path is relative to this file's directory
/// (`src-tauri/examples/`): two levels up reaches the repo root, where
/// `ppocr-models/` lives. (Matches `smoke_ppocr.rs`.)
const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");

/// Default fixture (relative to the crate root `src-tauri/`).
const DEFAULT_IMAGE: &str = "../sample_images/curve_lines_01.png";

/// Match `DetectorPostprocessOptions::default()` (postprocess.rs).
const BINARY_THRESHOLD: f32 = 0.2;
const MIN_AREA: usize = 3;

/// Number of horizontal bands → centerline samples. 4 stations per the
/// current task; the polyline connecting them is the centerline. The polygon
/// built on top has 2*STATIONS vertices (4 top + 4 bottom).
const STATIONS: usize = 4;
/// PaddleOCR unclip ratio (db_postprocess.py:39 default 2.0; the Rust port's
/// `DetectorPostprocessOptions::default()` uses 1.4). Drives the per-edge
/// offset distance = area·ratio/perimeter — same formula both repos use.
const UNCLIP_RATIO: f32 = 1.4;

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_IMAGE.to_string());

    println!("Loading image: {img_path}");
    let img = image::open(&img_path)?;
    let (w, h) = img.dimensions();
    println!("Image dimensions: {w}x{h}");

    let t = Instant::now();
    println!("Loading PP-OCR small-det from bundled bytes...");
    let det = Detector::load_from_buffer_with_config(
        BUNDLED_PPOCR_DET,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        DetectorConfig::small(),
    )?;
    println!("  loaded in {:?}", t.elapsed());

    let t = Instant::now();
    let (values, ih, iw, transform) = det.detect_raw(&img)?;
    println!("\nForward in {:?}: score map {}x{}", t.elapsed(), iw, ih);
    let cw = transform.content_width() as usize;
    let ch = transform.content_height() as usize;

    // Collect components as pixel sets in input (network) space.
    let components = collect_components(&values, iw, cw, ch, BINARY_THRESHOLD, MIN_AREA);
    println!("Components (min_area={MIN_AREA}): {}", components.len());

    // === For each component, extract a centerline: split the CC's x-extent
    //     into STATIONS bands, take the vertical center (median y) in each,
    //     connect the points. All in input space; mapped to source coords
    //     only at draw time. ===
    let to_src = |x: f32, y: f32| -> Point {
        Point::new(transform.map_x_to_source(x), transform.map_y_to_source(y))
    };

    let mut canvas = img.to_rgba8();
    let green = Rgba([60, 200, 60, 255]);
    let yellow = Rgba([255, 220, 0, 255]);
    let magenta = Rgba([240, 60, 200, 255]);
    let red = Rgba([255, 60, 60, 255]);

    for pts in &components {
        // CC bounding box in input space.
        let (xmin, ymin, xmax, ymax) = bbox(pts);

        // Reference: draw the CC bbox (green) so the polygon is read against
        // the same rectangle the current quad-based path would use.
        let bx0 = to_src(xmin, ymin);
        let bx1 = to_src(xmax, ymax);
        draw_rect(&mut canvas, bx0, bx1, green);

        // Centerline: STATIONS samples across the x-extent (edges + interior),
        // each at the median y of CC pixels in its window.
        let cl = vertical_center_centerline(pts, xmin, xmax, STATIONS);

        // 8-point polygon: expand each centerline point up/down perpendicular
        // to its local tangent, measuring the CC's local top/bottom extents.
        let poly = build_centerline_polygon(pts, &cl, UNCLIP_RATIO);
        let poly_src: Vec<Point> = poly.iter().map(|&(x, y)| to_src(x, y)).collect();

        // Draw the polygon (magenta) + vertex dots (red).
        if poly_src.len() >= 2 {
            for w in poly_src.windows(2) {
                draw_thick_segment(&mut canvas, w[0], w[1], magenta);
            }
            // Close the loop.
            draw_thick_segment(&mut canvas, *poly_src.last().unwrap(), poly_src[0], magenta);
        }
        for p in &poly_src {
            for &(dx, dy) in &[(0isize, 0isize), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)] {
                let px = (p.x.round() as isize + dx).clamp(0, canvas.width() as isize - 1) as u32;
                let py = (p.y.round() as isize + dy).clamp(0, canvas.height() as isize - 1) as u32;
                canvas.put_pixel(px, py, red);
            }
        }

        // Centerline (yellow) on top so it stays visible.
        let cl_src: Vec<Point> = cl.iter().map(|&(x, y)| to_src(x, y)).collect();
        for w in cl_src.windows(2) {
            draw_thick_segment(&mut canvas, w[0], w[1], yellow);
        }
        for p in &cl_src {
            for &(dx, dy) in &[(0isize, 0isize), (1, 0), (0, 1), (1, 1)] {
                let px = (p.x.round() as isize + dx).clamp(0, canvas.width() as isize - 1) as u32;
                let py = (p.y.round() as isize + dy).clamp(0, canvas.height() as isize - 1) as u32;
                canvas.put_pixel(px, py, yellow);
            }
        }
    }

    let out_path = match std::env::args().nth(2) {
        Some(p) => p,
        None => {
            let p = std::path::Path::new(&img_path);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
            let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
            parent
                .join(format!("{stem}-strip.{ext}"))
                .to_string_lossy()
                .into_owned()
        }
    };
    canvas.save(&out_path)?;
    println!(
        "\nWrote overlay ({}x{}, {} components) to: {out_path}",
        canvas.width(),
        canvas.height(),
        components.len(),
    );
    println!("  green:   CC bbox (reference)");
    println!("  yellow:  {STATIONS}-point centerline (vertical center per x-band)");
    println!("  magenta: {}-point polygon (centerline ± perpendicular extents)", 2 * STATIONS);
    println!("  red:     polygon vertices");
    Ok(())
}

/// Collect components as pixel sets (input space coords). 8-connectivity,
/// matching the port's `collect_component`. Each pixel is recorded at its
/// integer coords (x, y) — no +0.5 center offset here, since the centerline
/// math is easier in integer pixel space.
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
            let mut queue = VecDeque::new();
            queue.push_back((x, y));
            visited[idx] = true;
            let mut pts = Vec::new();
            while let Some((cx, cy)) = queue.pop_front() {
                pts.push((cx as i32, cy as i32));
                for (dx, dy) in N8 {
                    let nx = cx as isize + dx;
                    let ny = cy as isize + dy;
                    if nx < 0 || ny < 0 || nx >= cw as isize || ny >= ch as isize {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nidx = ny * cw + nx;
                    if !visited[nidx] && values[ny * map_w + nx] >= thr {
                        visited[nidx] = true;
                        queue.push_back((nx, ny));
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

/// Axis-aligned bbox of a pixel set → (xmin, ymin, xmax, ymax) as f32.
fn bbox(pts: &[(i32, i32)]) -> (f32, f32, f32, f32) {
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for &(x, y) in pts {
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
    }
    (xmin as f32, ymin as f32, xmax as f32, ymax as f32)
}

/// Extract a centerline by finding the vertical center at STATIONS horizontal
/// positions sampled across the CC's x-extent.
///
/// For STATIONS=4 the sample x-fractions are `{0, 1/3, 2/3, 1}` — i.e. the two
/// line END edges plus two interior points. This is deliberately not 4 equal
/// bands: the end samples must sit at the actual line ends (so the centerline
/// reaches the tips), and the two interior samples detect curvature between
/// them. Each sample's y is the **median** y of component pixels within a
/// window centered on the sample's x (the window extends inward at the ends so
/// edge samples still gather enough pixels).
///
/// Median (not mean) is robust to descenders/ascenders and ink-density skew —
/// a few stray pixels at the top/bottom of a window won't drag the center.
fn vertical_center_centerline(
    pts: &[(i32, i32)],
    xmin: f32,
    xmax: f32,
    stations: usize,
) -> Vec<(f32, f32)> {
    let span = (xmax - xmin).max(1.0);
    // Sample fractions: 0, 1/(n-1), 2/(n-1), ..., 1 → endpoints inclusive.
    let mut out = Vec::with_capacity(stations);
    for s in 0..stations {
        let frac = if stations == 1 { 0.5 } else { s as f32 / (stations - 1) as f32 };
        let x_s = xmin + frac * span;

        // Window around x_s: half the inter-sample gap, centered. Clamped to
        // [xmin, xmax] so the end samples don't read past the CC boundary.
        let gap = span / (stations as f32).max(1.0);
        let win = gap * 0.5;
        let lo = x_s - win;
        let hi = x_s + win;
        let lo = lo.max(xmin);
        let hi = hi.min(xmax);

        let mut ys: Vec<i32> = pts
            .iter()
            .copied()
            .filter(|&(x, _)| {
                let x = x as f32;
                x >= lo && x <= hi
            })
            .map(|(_, y)| y)
            .collect();
        ys.sort_unstable();
        let y_s = if ys.is_empty() {
            // Empty window (gap in the component) — caller sees a kink, which
            // is honest. Could interpolate from neighbors later.
            0.0
        } else {
            ys[ys.len() / 2] as f32 // median
        };
        out.push((x_s, y_s));
    }
    out
}

/// Build a 2N-point polygon (N top + N bottom) by expanding each centerline
/// station up/down perpendicular to its local tangent, measuring the CC's
/// extents over each station's full Voronoi band.
///
/// Each pixel is assigned to its nearest station (by x, via Voronoi midpoints)
/// and its signed perpendicular distance to that STATION's centerline point +
/// normal is recorded. Per station we take max/min → the top/bottom extents.
/// Vertices are placed in that same station's frame, so measure-and-place use
/// one consistent coordinate frame per station (earlier versions mixed per-
/// segment measurement with per-station placement, which shrank the extents).
///
/// Every CC pixel contributes to exactly one station, so even the edge
/// stations (line ends) see their full share of pixels — the polygon covers
/// heads and legs, not just stroke centers. Top vertices left→right, then
/// bottoms right→left → closed clockwise loop (8 points for N=4).
fn build_centerline_polygon(
    pts: &[(i32, i32)],
    cl: &[(f32, f32)],
    unclip: f32,
) -> Vec<(f32, f32)> {
    let n = cl.len();
    if n < 2 {
        return Vec::new();
    }

    // CC bbox — fallback extent + the unclip base dimensions.
    let (xmin, ymin, xmax, ymax) = bbox(pts);
    let cc_h = (ymax - ymin).max(1.0);
    let cc_w = (xmax - xmin).max(1.0);
    // PaddleOCR's unclip distance (db_postprocess.py:162, postprocess.rs:365):
    //   distance = area * ratio / perimeter = W*H*ratio / (2*(W+H))
    // Applied as a uniform offset to EVERY edge (pyclipper JT_ROUND), not just
    // top/bottom. For a long text line (W≫H) this is ≈ ratio·H/2 per side —
    // substantially larger than a fixed fraction of height, which is why the
    // earlier margin was too tight.
    let margin = (cc_w * cc_h * unclip) / (2.0 * (cc_w + cc_h));

    // --- Per-station tangent + normal, computed once. The tangent at station i
    //     is the neighbor difference cl[i-1] → cl[i+1] (clamped at ends); the
    //     normal is its 90° rotation. Every pixel assigned to station i is
    //     measured AND placed in this single frame — no mixing. ---
    let mut tan = vec![(0.0f32, 0.0f32); n];
    let mut nrm = vec![(0.0f32, 0.0f32); n];
    for i in 0..n {
        let prev = cl[i.saturating_sub(1)];
        let next = cl[(i + 1).min(n - 1)];
        let tdx = next.0 - prev.0;
        let tdy = next.1 - prev.1;
        let tlen = (tdx * tdx + tdy * tdy).sqrt().max(1e-6);
        tan[i] = (tdx / tlen, tdy / tlen);
        nrm[i] = (-tan[i].1, tan[i].0);
    }

    // --- Voronoi band boundaries in x. Station i owns [bound[i], bound[i+1]],
    //     where bound[j] is the midpoint between cl[j-1] and cl[j]. The end
    //     bounds extend to the CC's x-extents so edge stations own all the way
    //     out to the line tips. ---
    let mut bound = Vec::with_capacity(n + 1);
    bound.push(cl[0].0); // left edge: station 0 owns from xmin
    for i in 1..n {
        bound.push((cl[i - 1].0 + cl[i].0) * 0.5);
    }
    bound.push(cl[n - 1].0); // right edge: station n-1 owns to xmax

    // --- Assign each pixel to its station (by x → Voronoi band) and measure
    //     its signed perpendicular distance in that station's frame. ---
    let mut band_top = vec![f32::NEG_INFINITY; n];
    let mut band_bot = vec![f32::INFINITY; n];
    for &(x, y) in pts {
        let xf = x as f32;
        let yf = y as f32;
        // Find the band containing xf. Bands are contiguous in x, so a linear
        // scan with the first upper-bound match is correct (n is tiny: 4).
        let mut station = 0;
        for i in 0..n {
            if xf <= bound[i + 1] {
                station = i;
                break;
            }
            station = i; // past the last bound → clamp to last station
        }
        let (cx, cy) = cl[station];
        let (nx, ny) = nrm[station];
        // Signed distance along the station's normal.
        let along_n = (xf - cx) * nx + (yf - cy) * ny;
        if along_n > band_top[station] {
            band_top[station] = along_n;
        }
        if along_n < band_bot[station] {
            band_bot[station] = along_n;
        }
    }

    let mut tops = Vec::with_capacity(n);
    let mut bots = Vec::with_capacity(n);
    for i in 0..n {
        // Empty band (gap in the component) → fall back to half the CC height.
        let (top, bot) = if band_top[i] == f32::NEG_INFINITY {
            (cc_h * 0.5, -cc_h * 0.5)
        } else {
            (band_top[i], band_bot[i])
        };
        // Add the unclip margin symmetrically (past the ink) so the polygon
        // covers ascenders/descenders and, under tight line spacing, can
        // overlap the adjacent line's polygon — which is expected and correct
        // (the recognizer crops to the polygon, so overlap just means neither
        // line clips the other's ink).
        let top = top + margin;
        let bot = bot - margin;

        // pyclipper offsets EVERY edge, including the line ends. Replicate by
        // shifting the end stations outward along the tangent by `margin`:
        // station 0 → −tangent, station n−1 → +tangent. Interior stations keep
        // their centerline x (the unclip is only at the tips).
        let (cx, cy) = cl[i];
        let (nx, ny) = nrm[i];
        let (tx, ty) = tan[i];
        let ext = if i == 0 {
            -margin
        } else if i == n - 1 {
            margin
        } else {
            0.0
        };
        let cx = cx + tx * ext;
        let cy = cy + ty * ext;
        tops.push((cx + nx * top, cy + ny * top));
        bots.push((cx + nx * bot, cy + ny * bot));
    }

    // Assemble clockwise: tops left→right, then bottoms right→left.
    let mut poly = Vec::with_capacity(2 * n);
    poly.extend(tops);
    bots.iter().rev().for_each(|&p| poly.push(p));
    poly
}

/// Draw an axis-aligned rectangle between two corners (inclusive).
fn draw_rect(canvas: &mut image::RgbaImage, p0: Point, p1: Point, color: Rgba<u8>) {
    let (x0, y0) = (p0.x.round() as i32, p0.y.round() as i32);
    let (x1, y1) = (p1.x.round() as i32, p1.y.round() as i32);
    let xmin = x0.min(x1).max(0);
    let xmax = x0.max(x1).min(canvas.width() as i32 - 1);
    let ymin = y0.min(y1).max(0);
    let ymax = y0.max(y1).min(canvas.height() as i32 - 1);
    // Top + bottom edges.
    for x in xmin..=xmax {
        canvas.put_pixel(x as u32, ymin as u32, color);
        canvas.put_pixel(x as u32, ymax as u32, color);
    }
    // Left + right edges.
    for y in ymin..=ymax {
        canvas.put_pixel(xmin as u32, y as u32, color);
        canvas.put_pixel(xmax as u32, y as u32, color);
    }
}

/// Draw a line segment thickened by stamping parallel neighbors (imageproc's
/// `draw_line_segment_mut` is 1px; the overlay needs ~2-3px to read at full
/// source resolution).
fn draw_thick_segment(canvas: &mut image::RgbaImage, p0: Point, p1: Point, color: Rgba<u8>) {
    for &(dx, dy) in &[
        (0isize, 0isize), (1, 0), (0, 1), (1, 1),
        (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1),
    ] {
        let a = (
            (p0.x.round() as isize + dx).clamp(0, canvas.width() as isize - 1) as f32,
            (p0.y.round() as isize + dy).clamp(0, canvas.height() as isize - 1) as f32,
        );
        let b = (
            (p1.x.round() as isize + dx).clamp(0, canvas.width() as isize - 1) as f32,
            (p1.y.round() as isize + dy).clamp(0, canvas.height() as isize - 1) as f32,
        );
        draw_line_segment_mut(canvas, a, b, color);
    }
}
