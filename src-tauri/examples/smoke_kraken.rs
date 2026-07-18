//! Smoke test for the vendored kraken engine: load both models, segment a
//! page image, recognize each line, and print the results. Run with:
//!
//!   cargo run --release --example smoke_kraken -- <image.png> <models_dir>
//!
//! where `models_dir` contains bur_segment.safetensors + bur_recog.safetensors.
//! If args are omitted, defaults to /tmp/scan2_p1.png and ../kraken-models.

use std::time::Instant;

use image::GenericImageView;

use just_ocr_lib::kraken::{
    self, config::SegmentationConfig, detect::detect_candle,
    recognition::preprocess::preprocess_line, segmentation_candle::SegmentationModelCandle,
};

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/scan2_p1.png".to_string());
    let models_dir =
        std::env::args().nth(2).unwrap_or_else(|| "../kraken-models".to_string());

    let seg_path = format!("{models_dir}/bur_segment.safetensors");
    let rec_path = format!("{models_dir}/bur_recog.safetensors");

    println!("Loading image: {img_path}");
    let img = image::open(&img_path)?;
    let (w, h) = img.dimensions();
    println!("Image dimensions: {w}x{h}");

    let t = Instant::now();
    println!("Loading segmentation model: {seg_path}");
    let seg = SegmentationModelCandle::load(&seg_path)?;
    println!("  loaded in {:?}", t.elapsed());

    let t = Instant::now();
    println!("Loading recognition model: {rec_path}");
    let rec = just_ocr_lib::kraken::recognition::RecognitionModel::load(&rec_path)?;
    println!("  loaded in {:?}", t.elapsed());

    let t = Instant::now();
    let config = SegmentationConfig { text_direction: "horizontal-lr".to_string() };
    let segmentation = detect_candle(&img, &seg, &config)?;
    println!("\nSegmentation in {:?}: {} lines", t.elapsed(), segmentation.lines.len());

    let t = Instant::now();
    let mut recognized = 0;
    let mut total_text = String::new();
    for (i, line) in segmentation.lines.iter().enumerate() {
        if line.boundary.len() < 3 {
            continue;
        }
        let min_x = line.boundary.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).max(0.0) as u32;
        let min_y = line.boundary.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(0.0) as u32;
        let max_x = line.boundary.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).min((w - 1) as f64) as u32;
        let max_y = line.boundary.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).min((h - 1) as f64) as u32;
        let cw = max_x.saturating_sub(min_x) + 1;
        let ch = max_y.saturating_sub(min_y) + 1;
        if cw < 2 || ch < 2 {
            continue;
        }
        let crop = img.crop_imm(min_x, min_y, cw, ch);
        let tensor = match preprocess_line(&image::DynamicImage::ImageRgb8(crop.to_rgb8()), rec.height, rec.padding) {
            Ok(t) => t,
            Err(e) => {
                println!("  line {i}: preprocess failed: {e}");
                continue;
            }
        };
        let text = match rec.recognize(&tensor) {
            Ok(t) => t,
            Err(e) => {
                println!("  line {i}: recognize failed: {e}");
                continue;
            }
        };
        recognized += 1;
        println!("  line {i:2} (bbox {min_x},{min_y}..{max_x},{max_y}): {text}");
        total_text.push_str(&text);
        total_text.push('\n');
    }
    println!("\nRecognized {recognized}/{} lines in {:?}", segmentation.lines.len(), t.elapsed());

    // Smoke-test the public API path too (resolve_models via the cache wrapper).
    // We can't easily build a tauri::AppHandle here, so just confirm the
    // free functions are callable; the line above already proved segment+recognize.
    let _ = kraken::KrakenCache::new();
    println!("\nKrakenCache::new() OK");

    print!("\n=== Full text ===\n{total_text}");
    Ok(())
}
