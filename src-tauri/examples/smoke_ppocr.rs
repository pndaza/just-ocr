//! Smoke test for the PP-OCR detector: load the bundled small-det, detect text
//! regions on a page image, and print the resulting quads. Run with:
//!
//!   cargo run --release --example smoke_ppocr -- <image.png>
//!
//! Defaults to /tmp/scan2_p1.png if no arg is given. Loads the bundled
//! small-det bytes via `include_bytes!` (same path the host app uses).

use std::time::Instant;

use image::GenericImageView;
use ppocr_engine::{Detector, DetectorConfig};

/// Bundled small-det. Path is relative to this file's directory
/// (`src-tauri/examples/`): two levels up reaches the repo root, where
/// `ppocr-models/` lives. (The host's `engine.rs` uses `../../ppocr-models/`
/// from `src-tauri/src/` — one level up from `src/` to `src-tauri/`, then one
/// more to the repo root. From `examples/` we need the same two levels.)
const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/scan2_p1.png".to_string());

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
                "detection {i} coord out of bounds: ({}, {})", p.0, p.1
            );
        }
    }

    println!("\nAll {} detections are within image bounds.", detections.len());
    Ok(())
}
