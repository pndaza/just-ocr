//! Dump the PP-OCR detector's raw DB score map as a heatmap PNG, upsampled
//! back to source-image resolution. Shows exactly what the network sees
//! *before* `extract_detections` collapses it to quads — useful for tuning
//! thresholds and for eyeballing whether a multi-point polygon path (tracing
//! the binary mask contour) would follow the text shape better than quads.
//!
//! Run with:
//!
//!   cargo run --release --example dump_ppocr_scoremap -- <image.png> [out.png]
//!
//! Defaults to `../sample_images/curve_lines_01.png`; writes `<stem>-scoremap.png`
//! next to the input. White = high text probability, black = background.
//! Also prints the value histogram (how much of the map is above the default
//! 0.2 binary / 0.4 box thresholds) so threshold tuning is evidence-based.

use std::time::Instant;

use image::{GenericImageView, Luma};
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
    let (values, ih, iw, _transform) = det.detect_raw(&img)?;
    println!(
        "\nForward in {:?}: raw score map {}x{} ({} values)",
        t.elapsed(),
        iw,
        ih,
        values.len()
    );

    // Histogram vs the postprocess defaults (see postprocess.rs defaults:
    // binary_threshold = 0.2, box_threshold = 0.4). Cheap threshold-tuning aid.
    let mut above_bin = 0u32; // >= 0.2
    let mut above_box = 0u32; // >= 0.4
    let mut lt02 = 0u32;
    let mut b02045 = 0u32;
    let mut ge045 = 0u32;
    for &v in &values {
        if v >= 0.2 { above_bin += 1; }
        if v >= 0.4 { above_box += 1; }
        if v < 0.2 { lt02 += 1; }
        else if v < 0.45 { b02045 += 1; }
        else { ge045 += 1; }
    }
    let total = values.len() as u32;
    println!(
        "  histogram: <0.2: {lt02}, 0.2-0.45: {b02045}, >=0.45: {ge045} (total {total})",
    );
    println!(
        "  histogram: {:.1}% >= 0.2 (binary thr), {:.1}% >= 0.4 (box thr)",
        100.0 * above_bin as f64 / total as f64,
        100.0 * above_box as f64 / total as f64,
    );
    if let Some(&mx) = values.iter().max_by(|a, b| a.total_cmp(b)) {
        println!("  max score: {mx:.3}");
    }

    // Render the raw map to a grayscale image at network resolution, then
    // nearest-neighbor upsample to source dims so it visually aligns with the
    // input for side-by-side comparison.
    let mut raw = image::GrayImage::new(iw as u32, ih as u32);
    for (i, &v) in values.iter().enumerate() {
        let x = i % iw;
        let y = i / iw;
        // Clamp to [0,1] — the sigmoid head keeps values in range, but be
        // defensive against tiny float drift.
        let clamped = v.clamp(0.0, 1.0);
        raw.put_pixel(x as u32, y as u32, Luma([(clamped * 255.0).round() as u8]));
    }
    let upsampled = image::imageops::resize(
        &raw,
        w,
        h,
        image::imageops::FilterType::Nearest,
    );

    let out_path = match std::env::args().nth(2) {
        Some(p) => p,
        None => {
            let p = std::path::Path::new(&img_path);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
            let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
            parent
                .join(format!("{stem}-scoremap.{ext}"))
                .to_string_lossy()
                .into_owned()
        }
    };

    upsampled.save(&out_path)?;
    println!("\nWrote scoremap ({iw}x{ih} → {w}x{h}) to: {out_path}");
    Ok(())
}
