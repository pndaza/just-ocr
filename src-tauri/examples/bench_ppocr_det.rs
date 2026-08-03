//! Benchmark PP-OCR detector forward-pass time: tiny-det vs small-det.
//!
//! Both models share identical preprocess + postprocess — the ONLY difference
//! between them is the network architecture (LcNet channel widths + neck
//! kernel), so the timing gap between the two `detect_raw` runs is entirely the
//! forward-pass cost. That is what matters for the "drop tiny, ship small"
//! decision.
//!
//! `detect_raw` (preprocess + forward) is timed rather than `forward` alone
//! because `forward` is private; preprocess is identical between models, so it
//! cancels out of the comparison.
//!
//! Reports per-image: mean + min forward time over N iterations (after warmup),
//! plus detection count on the default-postprocess path for a sanity check that
//! both models find the text.
//!
//! Run with:
//!
//!   cargo run --release --example bench_ppocr_det -- [image1] [image2] ...
//!
//! Defaults to ../sample_images/thawzin_02.png and heavy_curve_02.png (one clean
//! scan, one dense/curved). Uses release builds — dev timings are meaningless.

use std::time::Instant;

use image::GenericImageView;
use ppocr_engine::{Detector, DetectorConfig};

/// Bundled tiny-det (matches the host's include_bytes! path).
const TINY: &[u8] = include_bytes!("../../ppocr-models/tiny-det.safetensors");
/// Small-det (currently untracked / opt-in on the ppocr-curve branch).
const SMALL: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");

const WARMUP: usize = 3;
const ITERS: usize = 10;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let images: Vec<String> = if args.is_empty() {
        vec![
            "../sample_images/thawzin_02.png".to_string(),
            "../sample_images/heavy_curve_02.png".to_string(),
        ]
    } else {
        args
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!(
        "PP-OCR detector benchmark — threads={}, warmup={}, iters={}",
        threads, WARMUP, ITERS
    );
    println!(
        "  tiny:  {:.1} MB (stage_channels [32,48,64,160], neck_kernel 5)",
        TINY.len() as f64 / 1e6
    );
    println!(
        "  small: {:.1} MB (stage_channels [48,96,192,384], neck_kernel 7)",
        SMALL.len() as f64 / 1e6
    );
    println!();

    let tiny = Detector::load_from_buffer_with_config(TINY, threads, DetectorConfig::tiny())?;
    let small = Detector::load_from_buffer_with_config(SMALL, threads, DetectorConfig::small())?;

    let mut tiny_total = 0.0f64;
    let mut small_total = 0.0f64;
    let mut tiny_n = 0u32;
    let mut small_n = 0u32;

    for path in &images {
        let img = image::open(path)?;
        let (w, h) = img.dimensions();
        println!("--- {path} ({w}x{h}) ---");

        // Tiny: warmup + timed.
        for _ in 0..WARMUP {
            let _ = tiny.detect_raw(&img)?;
        }
        let mut tiny_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            let (vals, oh, ow, _) = tiny.detect_raw(&img)?;
            let us = t.elapsed().as_secs_f64() * 1e3;
            tiny_times.push(us);
            let _ = (vals, oh, ow);
        }
        // Small: warmup + timed.
        for _ in 0..WARMUP {
            let _ = small.detect_raw(&img)?;
        }
        let mut small_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            let (vals, oh, ow, _) = small.detect_raw(&img)?;
            let us = t.elapsed().as_secs_f64() * 1e3;
            small_times.push(us);
            let _ = (vals, oh, ow);
        }

        // Detection counts (sanity: both find the text).
        let tiny_dets = tiny.detect(&img)?.len();
        let small_dets = small.detect(&img)?.len();

        report("tiny ", &tiny_times);
        report("small", &small_times);
        let ratio = median(&small_times) / median(&tiny_times).max(1e-9);
        println!(
            "  → small/tiny median = {:.2}x   (tiny found {}, small found {})",
            ratio, tiny_dets, small_dets
        );
        println!();

        tiny_total += median(&tiny_times);
        small_total += median(&small_times);
        tiny_n += 1;
        small_n += 1;
    }

    if tiny_n > 0 {
        println!(
            "=== across {} image(s): tiny mean {:.2} ms, small mean {:.2} ms, small/tiny = {:.2}x ===",
            tiny_n,
            tiny_total / tiny_n as f64,
            small_total / small_n as f64,
            (small_total / small_n as f64) / (tiny_total / tiny_n as f64).max(1e-9),
        );
    }
    Ok(())
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

fn report(label: &str, times: &[f64]) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  {label}: mean {mean:6.2} ms  min {min:6.2}  max {max:6.2}  ({} iters)",
        times.len()
    );
}
