//! Smoke test for the PP-OCR-direct pipeline: load the bundled PP-OCR
//! small-det + bundled Kraken rec, detect text quads on a page image, and
//! recognize each line via `Engine::recognize_line_direct` (the path that
//! skips kraken's baseline mesh warp and just masks + deskews the quad).
//! Run with:
//!
//!   cargo run --release --example smoke_ppocr_recog -- <image.png>
//!
//! Defaults to ../sample_images/myanmar_01.png (committed at the repo root).
//! Loads both models via `include_bytes!` — zero external setup.

use std::time::Instant;

use image::GenericImageView;
use kraken_engine::Engine;
use ppocr_engine::{Detector, DetectorConfig};

// Bundled models. Paths are relative to this file (`src-tauri/examples/`):
// two levels up reaches the repo root. (The host's `engine.rs` uses the same
// two levels from `src-tauri/src/`.)
const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");
const BUNDLED_KRAKEN_SEG: &[u8] = include_bytes!("../../kraken-models/bur_segment.safetensors");
const BUNDLED_KRAKEN_REC: &[u8] = include_bytes!("../../kraken-models/bur_recog.safetensors");

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../sample_images/myanmar_01.png".to_string());

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
    println!("Loading Kraken rec (seg bytes are loaded but unused here)...");
    // recognize_line_direct only needs the recog model, but Engine bundles
    // seg+rec; load both from the bundled bytes (matches the host's path).
    let engine = Engine::load_from_buffers(BUNDLED_KRAKEN_SEG, BUNDLED_KRAKEN_REC)?;
    println!("  loaded in {:?}", t.elapsed());

    // Stage 1: PP-OCR detection → quads.
    let t = Instant::now();
    let detections = det.detect(&img)?;
    println!("\nPP-OCR detection in {:?}: {} regions", t.elapsed(), detections.len());

    // Stage 2: recognize each quad via the direct path.
    let t = Instant::now();
    let mut recognized = 0;
    let mut total_text = String::new();
    for (i, d) in detections.iter().enumerate() {
        let poly = &d.polygon;
        // Build the closed 5-point boundary exactly as the host's
        // `detection_to_line` does: the 4 corners + first repeated.
        let boundary: Vec<(f64, f64)> = vec![
            (poly[0].0 as f64, poly[0].1 as f64),
            (poly[1].0 as f64, poly[1].1 as f64),
            (poly[2].0 as f64, poly[2].1 as f64),
            (poly[3].0 as f64, poly[3].1 as f64),
            (poly[0].0 as f64, poly[0].1 as f64),
        ];

        // Report the deskew angle (computed inside recognize_line_direct; we
        // recompute it here for visibility).
        let (x0, y0) = (poly[0].0 as f64, poly[0].1 as f64);
        let (x1, y1) = (poly[1].0 as f64, poly[1].1 as f64);
        let angle_deg = (y1 - y0).atan2(x1 - x0).to_degrees();

        let min_x = boundary.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).max(0.0) as u32;
        let min_y = boundary.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(0.0) as u32;
        let max_x = boundary
            .iter()
            .map(|p| p.0)
            .fold(f64::NEG_INFINITY, f64::max)
            .min((w - 1) as f64) as u32;
        let max_y = boundary
            .iter()
            .map(|p| p.1)
            .fold(f64::NEG_INFINITY, f64::max)
            .min((h - 1) as f64) as u32;

        let text = match engine.recognize_line_direct(&img, &boundary) {
            Ok(t) => t,
            Err(e) => {
                println!(
                    "  region {i:2} (bbox {min_x},{min_y}..{max_x},{max_y}, {angle_deg:+.1}°): recognize failed: {e}"
                );
                continue;
            }
        };
        recognized += 1;
        println!(
            "  region {i:2} (bbox {min_x},{min_y}..{max_x},{max_y}, {angle_deg:+.1}°): {text}"
        );
        total_text.push_str(&text);
        total_text.push('\n');
    }
    println!(
        "\nRecognized {recognized}/{} lines in {:?}",
        detections.len(),
        t.elapsed()
    );

    print!("\n=== Full text ===\n{total_text}");
    Ok(())
}
