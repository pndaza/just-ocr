//! Load the PP-OCRv6 **small** detector from disk and compare its score map /
//! detection count against the tiny model (the bundled default) and the
//! Python `PP-OCRv6_small_det_infer` reference. The small model should
//! produce a score map much closer to Python's (resolving the over-detection
//! on dense images like heavy_curve_02: tiny gave 31 lines, Python's small
//! gives 27).
//!
//!   cargo run --release --example smoke_ppocr_small -- <image.png>
//!
//! Defaults to ../sample_images/heavy_curve_02.png. Loads small-det.safetensors
//! from ../ppocr-models/ (must be downloaded — not bundled):
//!   curl -L -o ppocr-models/small-det.safetensors \
//!     "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_safetensors/resolve/main/model.safetensors"

use std::time::Instant;

use image::GenericImageView;
use ppocr_engine::{CpuOptions, Detector, DetectorConfig};

const SMALL_DET_PATH: &str = "../ppocr-models/small-det.safetensors";
const BUNDLED_TINY_DET: &[u8] = include_bytes!("../../ppocr-models/tiny-det.safetensors");
const DEFAULT_IMAGE: &str = "../sample_images/heavy_curve_02.png";

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_IMAGE.to_string());
    println!("Loading image: {img_path}");
    let img = image::open(&img_path)?;
    let (w, h) = img.dimensions();
    println!("Image dimensions: {w}x{h}");

    // --- Small model (from disk) ---
    let t = Instant::now();
    println!("Loading PP-OCRv6 small-det from {SMALL_DET_PATH}...");
    let small = Detector::load_with_config(
        SMALL_DET_PATH,
        CpuOptions::default(),
        DetectorConfig::small(),
    )?;
    println!("  small loaded in {:?}", t.elapsed());

    let t = Instant::now();
    let (sm_values, sm_ih, sm_iw, _) = small.detect_raw(&img)?;
    println!(
        "  small forward in {:?}: score map {}x{}",
        t.elapsed(),
        sm_iw,
        sm_ih,
    );

    // --- Tiny model (bundled, the default) ---
    let t = Instant::now();
    let tiny = Detector::load_from_buffer(BUNDLED_TINY_DET)?;
    println!("\nLoaded bundled tiny-det in {:?}", t.elapsed());
    let t = Instant::now();
    let (tn_values, tn_ih, tn_iw, _) = tiny.detect_raw(&img)?;
    println!(
        "  tiny forward in {:?}: score map {}x{}",
        t.elapsed(),
        tn_iw,
        tn_ih,
    );

    // --- Compare the two score maps ---
    assert_eq!(
        (sm_ih, sm_iw),
        (tn_ih, tn_iw),
        "small and tiny score maps differ in shape"
    );
    let n = sm_values.len();
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    for (a, b) in sm_values.iter().zip(tn_values.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        sum_diff += d as f64;
    }
    println!(
        "\nsmall vs tiny score map: max abs diff = {max_diff:.4}, mean abs diff = {:.5}",
        sum_diff / n as f64,
    );

    // --- Histograms (3-bucket) for both ---
    for (name, vals) in [("small", &sm_values[..]), ("tiny", &tn_values[..])] {
        let (mut lt02, mut mid, mut ge045) = (0u32, 0u32, 0u32);
        for &v in vals {
            if v < 0.2 {
                lt02 += 1;
            } else if v < 0.45 {
                mid += 1;
            } else {
                ge045 += 1;
            }
        }
        println!(
            "  {name} histogram: <0.2: {lt02}, 0.2-0.45: {mid}, >=0.45: {ge045} (total {})",
            vals.len()
        );
    }

    // --- Detection count via the segmenter (poly path) ---
    // Reuse the host's PPOcrPolySegmenter to count lines each model produces.
    use std::sync::Arc;
    use just_ocr_lib::{PPOcrPolySegmenter, Segmenter};
    for (name, det) in [("small", Arc::new(small)), ("tiny", Arc::new(tiny))] {
        let seg = PPOcrPolySegmenter::new(det);
        let lines = seg.segment(&img).map_err(|e| anyhow::anyhow!(e))?;
        println!("  {name} segmenter (poly): {} lines", lines.len());
    }
    Ok(())
}
