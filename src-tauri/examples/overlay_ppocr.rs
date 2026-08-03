//! Debug overlay for the PP-OCR detector: load the bundled small-det, detect
//! text regions on a page image, draw each detection's quad onto the image,
//! and save the result. Used to eyeball detector quality on curved/rotated
//! text fixtures (e.g. `sample_images/curve_lines_01.png`).
//!
//! Run with:
//!
//!   cargo run --release --example overlay_ppocr -- <image.png> [out.png]
//!
//! Defaults to `../sample_images/curve_lines_01.png` if no image arg is given
//! and writes `<image>-overlay.png` next to it (or the given out path).
//!
//! Each detection's 4-corner quad is drawn as a hollow polygon (the same shape
//! the host's `PPOcrSegmenter` closes into a boundary, see
//! `segmenter_adapters.rs::detection_to_line`). Vertices are also marked with
//! small dots so corner ordering is visible.

use std::time::Instant;

use image::{GenericImageView, Rgba};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_polygon_mut};
use imageproc::point::Point;
use ppocr_engine::{Detector, DetectorConfig};

/// Bundled small-det. Path is relative to this file's directory
/// (`src-tauri/examples/`): two levels up reaches the repo root, where
/// `ppocr-models/` lives. (Matches `smoke_ppocr.rs`.)
const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");

/// Default fixture (relative to the crate root `src-tauri/`).
const DEFAULT_IMAGE: &str = "../sample_images/curve_lines_01.png";

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
    let detections = det.detect(&img)?;
    println!("\nDetection in {:?}: {} regions", t.elapsed(), detections.len());

    // Work on an RGBA copy so we can draw translucent strokes.
    let mut canvas = img.to_rgba8();

    for (i, d) in detections.iter().enumerate() {
        let poly = &d.polygon;
        println!(
            "  region {i:2} (score {:.2}): ({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0})",
            d.score, poly[0].0, poly[0].1, poly[1].0, poly[1].1,
            poly[2].0, poly[2].1, poly[3].0, poly[3].1,
        );

        // Sanity: all coords in source-image bounds.
        for p in poly {
            debug_assert!(
                p.0 >= 0.0 && p.0 <= w as f32 && p.1 >= 0.0 && p.1 <= h as f32,
                "detection {i} coord out of bounds: ({}, {})",
                p.0, p.1
            );
        }

        // Hollow polygon outline — the closed quad shape, same vertices the
        // host turns into a `DetectedLine::boundary`. imageproc's polygon
        // drawer takes `Point<f32>` (it rounds internally).
        let pts: Vec<Point<f32>> = poly
            .iter()
            .map(|p| Point::new(p.0, p.1))
            .collect();
        draw_hollow_polygon_mut(&mut canvas, &pts, Rgba([0, 200, 0, 255]));

        // Vertex dots so corner ordering (tl, tr, br, bl) is visible.
        for p in &pts {
            draw_filled_circle_mut(
                &mut canvas,
                (p.x.round() as i32, p.y.round() as i32),
                3,
                Rgba([255, 60, 60, 255]),
            );
        }
    }

    let out_path = match std::env::args().nth(2) {
        Some(p) => p,
        None => {
            // <stem>-overlay.png next to the input.
            let p = std::path::Path::new(&img_path);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
            let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
            parent
                .join(format!("{stem}-overlay.{ext}"))
                .to_string_lossy()
                .into_owned()
        }
    };

    canvas.save(&out_path)?;
    println!("\nWrote overlay with {} quads to: {out_path}", detections.len());
    Ok(())
}
