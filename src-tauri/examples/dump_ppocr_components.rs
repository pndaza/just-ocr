//! Visualize the intermediate stages of PP-OCR DB postprocess on the source
//! image: the **binary mask**, **connected components** (filled, colored per
//! label), and **traced contours** (boundary polygons) — i.e. everything
//! `collect_component`/`fit_rotated_box` compute internally but discard, and
//! the contour trace that PaddleOCR's `polygons_from_bitmap` does in Python
//! but the Rust port never implemented.
//!
//! All three are rendered as a 3-panel side-by-side montage, upsampled to
//! source resolution so they align with the input. Stage params mirror the
//! vendored `DetectorPostprocessOptions::default()`:
//!   binary_threshold = 0.2, min_area = 3.
//!
//! Run with:
//!
//!   cargo run --release --example dump_ppocr_components -- <image.png> [out.png]
//!
//! Defaults to `../sample_images/curve_lines_01.png`; writes
//! `<stem>-components.png` next to the input.
//!
//! Why: the contour panel answers "what would a multi-point polygon path
//! capture?" vs the current quad-only output (see overlay_ppocr.rs).

use std::time::Instant;

use image::{GenericImageView, Rgba, RgbaImage};
use ppocr_engine::{Detector, DetectorConfig};

/// Bundled small-det. Path is relative to this file's directory
/// (`src-tauri/examples/`): two levels up reaches the repo root, where
/// `ppocr-models/` lives. (Matches `smoke_ppocr.rs`.)
const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");

/// Default fixture (relative to the crate root `src-tauri/`).
const DEFAULT_IMAGE: &str = "../sample_images/curve_lines_01.png";

/// Match `DetectorPostprocessOptions::default()` (postprocess.rs).
const BINARY_THRESHOLD: f32 = 0.2;
const MIN_AREA: usize = 3;

/// Distinct colors for component fill. Repeats after N — fine for a debug tool.
const PALETTE: &[(u8, u8, u8)] = &[
    (231, 76, 60),   // red
    (46, 204, 113),  // green
    (52, 152, 219),  // blue
    (241, 196, 15),  // yellow
    (155, 89, 182),  // purple
    (26, 188, 156),  // teal
    (230, 126, 34),  // orange
    (149, 165, 166), // gray
];

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_IMAGE.to_string());

    println!("Loading image: {img_path}");
    let img = image::open(&img_path)?;
    let (sw, sh) = img.dimensions();
    println!("Image dimensions: {sw}x{sh}");

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
    println!(
        "\nForward in {:?}: score map {}x{}",
        t.elapsed(),
        iw,
        ih,
    );
    // Active region mirrors collect_component: [0, content_w) × [0, content_h).
    let cw = transform.content_width() as usize;
    let ch = transform.content_height() as usize;
    println!("Content region (active mask): {cw}x{ch}");

    // === Stage A: binary mask (score_map >= BINARY_THRESHOLD) ===
    let mut mask = vec![false; cw * ch];
    let mut fg = 0usize;
    for y in 0..ch {
        for x in 0..cw {
            if values[y * iw + x] >= BINARY_THRESHOLD {
                mask[y * cw + x] = true;
                fg += 1;
            }
        }
    }
    println!(
        "Binary mask: {fg} fg pixels ({:.1}% of content region)",
        100.0 * fg as f64 / (cw * ch) as f64
    );

    // === Stage B: connected components (4-connectivity — matches the Rust
    //     port's NEIGHBORS table semantics for grouping; avoids diagonal
    //     over-merge that would join adjacent diagonal text lines). ===
    let labels = label_components(&mask, cw, ch, MIN_AREA);
    let n_components = labels.iter().copied().filter(|&l| l > 0).max().unwrap_or(0);
    println!("Connected components (min_area={MIN_AREA}, 4-conn): {n_components}");

    // === Stage C: extract the boundary (contour) of each component. A pixel is
    //     on the contour if any of its 8-neighbors is background or a different
    //     label — i.e. the component's outer edge. This is the same pixel set
    //     cv2.findContours walks; cv2 additionally *orders* the points into a
    //     polyline (needed to build a polygon, but not to visualize the shape).
    //     Rendering this set shows exactly what a multi-point polygon path
    //     would trace before simplification (approxPolyDP). ===
    let contours = extract_boundaries(&labels, n_components, cw, ch);
    let total_pts: usize = contours.iter().map(|c| c.len()).sum();
    println!(
        "Extracted {} contour boundaries ({} edge pixels; mean {:.0} px/contour)",
        contours.len(),
        total_pts,
        total_pts as f64 / contours.len().max(1) as f64,
    );

    // === Render a single composite overlay at network resolution, upsampled
    //     to source dims: translucent component fill + crisp contour outlines
    //     on top of the original image. ===
    let src_rgb = img.to_rgba8();
    let out = render_overlay(&labels, &contours, cw, ch, &src_rgb, sw, sh);

    let out_path = match std::env::args().nth(2) {
        Some(p) => p,
        None => {
            let p = std::path::Path::new(&img_path);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
            let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
            parent
                .join(format!("{stem}-components.{ext}"))
                .to_string_lossy()
                .into_owned()
        }
    };
    out.save(&out_path)?;
    println!(
        "\nWrote overlay ({}x{}, {} contours / {} edge pts) to: {out_path}",
        out.width(),
        out.height(),
        contours.len(),
        total_pts,
    );
    println!("  fill: connected components (4-conn, min_area {MIN_AREA})");
    println!("  outline: contour boundaries (edge-pixel set)");
    Ok(())
}

/// 4-connectivity CC labeling via iterative BFS (same flood-fill shape as the
/// port's `collect_component`, but returns a label image instead of per-comp
/// point lists). Labels are 1..=N; 0 = background or components below
/// `min_area` (filtered after labeling).
fn label_components(mask: &[bool], w: usize, h: usize, min_area: usize) -> Vec<usize> {
    let mut labels = vec![0usize; w * h];
    let mut next_label = 1usize;
    // Per-label area; index 0 unused (background).
    let mut areas: Vec<usize> = vec![0];
    let mut queue: std::collections::VecDeque<(usize, usize)> = std::collections::VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !mask[idx] || labels[idx] != 0 {
                continue;
            }
            labels[idx] = next_label;
            let mut area = 1;
            queue.push_back((x, y));
            while let Some((cx, cy)) = queue.pop_front() {
                for (dx, dy) in [(0, !0), (0, 1), (!0, 0), (1, 0)] {
                    let nx = cx.wrapping_add(dx);
                    let ny = cy.wrapping_add(dy);
                    if nx >= w || ny >= h {
                        continue;
                    }
                    let nidx = ny * w + nx;
                    if mask[nidx] && labels[nidx] == 0 {
                        labels[nidx] = next_label;
                        area += 1;
                        queue.push_back((nx, ny));
                    }
                }
            }
            areas.push(area);
            next_label += 1;
        }
    }
    // Drop small components: relabel to 0 where area < min_area.
    for y in 0..h {
        for x in 0..w {
            let l = labels[y * w + x];
            if l != 0 && areas[l] < min_area {
                labels[y * w + x] = 0;
            }
        }
    }
    labels
}

/// Extract the boundary (edge) pixels of each labeled component. A pixel is
/// on the boundary if any of its 8-neighbors is background (label 0) or a
/// different label. Returns one `Vec<(x,y)>` per label `1..=n`, in raster
/// order (not ordered into a walkable polyline — this is the pixel set, not a
/// traced path).
///
/// This is the same set `cv2.findContours` walks around; cv2 additionally
/// orders the points into a closed polyline (needed to build a polygon, but
/// not to visualize the contour shape). The multi-point polygon port would
/// add an ordered trace + Douglas-Peucker simplification on top of this set.
fn extract_boundaries(
    labels: &[usize],
    n: usize,
    w: usize,
    h: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut out: Vec<Vec<(usize, usize)>> = (0..n).map(|_| Vec::new()).collect();
    for y in 0..h {
        for x in 0..w {
            let l = labels[y * w + x];
            if l == 0 {
                continue;
            }
            // A pixel is on the boundary if any 8-neighbor differs/is outside.
            let mut edge = false;
            'nb: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let nl = if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        0
                    } else {
                        labels[ny as usize * w + nx as usize]
                    };
                    if nl != l {
                        edge = true;
                        break 'nb;
                    }
                }
            }
            if edge {
                out[l - 1].push((x, y));
            }
        }
    }
    out
}

// === Rendering ===

fn upscale(src: &RgbaImage, sw: u32, sh: u32) -> RgbaImage {
    image::imageops::resize(src, sw, sh, image::imageops::FilterType::Nearest)
}

/// Single composite overlay: connected-component fill (translucent, colored per
/// label) with crisp contour outlines drawn on top, over the original image.
///
/// The labels/contours live in network-resolution space (`w × h`). The source
/// image is at `sw × sh`. To align them we first **downscale** the source to
/// `w × h` so the background shares the labels' coordinate space, composite,
/// then **upscale** the result back to source dims. Reading the full-res source
/// directly with network-space coords would only sample its top-left corner.
fn render_overlay(
    labels: &[usize],
    contours: &[Vec<(usize, usize)>],
    w: usize,
    h: usize,
    src: &RgbaImage,
    sw: u32,
    sh: u32,
) -> RgbaImage {
    // Downscale source → network resolution so pixel (x,y) means the same
    // location in both the background and the label/contour arrays.
    let bg = image::imageops::resize(src, w as u32, h as u32, image::imageops::FilterType::Triangle);
    let mut raw = RgbaImage::new(w as u32, h as u32);

    // Pass 1: component fill — blend a translucent palette color over the
    // source where a label is present. Background stays as the source pixel.
    for y in 0..h {
        for x in 0..w {
            let l = labels[y * w + x];
            let s = bg.get_pixel(x as u32, y as u32);
            if l == 0 {
                raw.put_pixel(x as u32, y as u32, *s);
            } else {
                let c = PALETTE[(l - 1) % PALETTE.len()];
                // 50/50 blend: tint is visible but the underlying text still reads.
                raw.put_pixel(
                    x as u32,
                    y as u32,
                    Rgba([
                        (s[0] as u32 / 2 + c.0 as u32 / 2) as u8,
                        (s[1] as u32 / 2 + c.1 as u32 / 2) as u8,
                        (s[2] as u32 / 2 + c.2 as u32 / 2) as u8,
                        255,
                    ]),
                );
            }
        }
    }

    // Pass 2: contour outlines — mark each edge pixel (and its right/down
    // neighbor, so the stroke survives nearest-upsampling as a 2px line).
    for (ci, contour) in contours.iter().enumerate() {
        let c = PALETTE[ci % PALETTE.len()];
        let color = Rgba([c.0, c.1, c.2, 255]);
        for &(x, y) in contour {
            for &(dx, dy) in &[(0isize, 0isize), (1, 0), (0, 1), (1, 1)] {
                let px = (x as isize + dx).clamp(0, w as isize - 1) as u32;
                let py = (y as isize + dy).clamp(0, h as isize - 1) as u32;
                raw.put_pixel(px, py, color);
            }
        }
    }

    upscale(&raw, sw, sh)
}
