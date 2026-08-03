//! Smoke test for the `ppocr-poly` segmenter: run the multi-point polygon
//! segmentation, render the polygons (green) on the source image, and — if
//! `KRKN_DUMP_DIR` is set — recognize each line via kraken's direct path so the
//! per-line recognizer-input crops (`{seq}_in.png`) are dumped.
//!
//!   cargo run --release --example smoke_ppocr_poly -- <img> [overlay_out]
//!
//! Default image: ../sample_images/heavy_curve_02.png. Overlay written to
//! <stem>-polyseg.png next to the input (or the given path).
//!
//! Set KRKN_DUMP_DIR=/tmp/poly_dumps to also dump recognizer crops.

use std::sync::Arc;
use std::time::Instant;

use image::{GenericImageView, Rgba};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_polygon_mut};
use imageproc::point::Point;
use just_ocr_lib::{PPOcrPolySegmenter, Segmenter};
use kraken_engine::Engine;
use ppocr_engine::{CpuOptions, DetectorConfig};

const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");
const BUNDLED_KRAKEN_SEG: &[u8] = include_bytes!("../../kraken-models/bur_segment.safetensors");
const BUNDLED_KRAKEN_REC: &[u8] = include_bytes!("../../kraken-models/bur_recog.safetensors");
const DEFAULT_IMAGE: &str = "../sample_images/heavy_curve_02.png";
const SMALL_DET_PATH: &str = "../ppocr-models/small-det.safetensors";

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_IMAGE.to_string());
    println!("Loading image: {img_path}");
    let img = image::open(&img_path)?;
    let (w, h) = img.dimensions();
    println!("Image dimensions: {w}x{h}");

    // Detector choice: default to the bundled small-det; set PPOCR_DET_MODEL=small
    // to load the small variant from disk (must be downloaded first — see
    // smoke_ppocr_small.rs). The small model matches PaddlePaddle's
    // PP-OCRv6_small_det reference and resolves the dense-image over-detection.
    let t = Instant::now();
    let det = match std::env::var("PPOCR_DET_MODEL").as_deref() {
        Ok("small") => {
            println!("Loading PP-OCRv6 small-det from {SMALL_DET_PATH}...");
            Arc::new(ppocr_engine::Detector::load_with_config(
                SMALL_DET_PATH,
                CpuOptions::default(),
                DetectorConfig::small(),
            )?)
        }
        _ => {
            println!("Loading bundled small-det...");
            Arc::new(ppocr_engine::Detector::load_from_buffer_with_config(
                BUNDLED_PPOCR_DET,
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
                DetectorConfig::small(),
            )?)
        }
    };
    println!("  detector loaded in {:?}", t.elapsed());

    // Stage 1: poly segmentation → multi-point DetectedLines.
    let seg = PPOcrPolySegmenter::new(det);
    let t = Instant::now();
    let lines = seg.segment(&img).map_err(|e| anyhow::anyhow!(e))?;
    println!(
        "\n[{}] segmentation in {:?}: {} lines",
        seg.name(),
        t.elapsed(),
        lines.len(),
    );

    // Render the polygons (green outlines + red vertex dots) on the source.
    // `boundary` is closed (first==last); imageproc's polygon drawer panics on
    // that, so drop the repeated closing vertex before drawing.
    let mut canvas = img.to_rgba8();
    let green = Rgba([0, 200, 0, 255]);
    let red = Rgba([255, 60, 60, 255]);
    for line in &lines {
        let mut pts: Vec<Point<f32>> = line
            .boundary
            .iter()
            .map(|&(x, y)| Point::new(x as f32, y as f32))
            .collect();
        if pts.len() >= 2 && pts.first() == pts.last() {
            pts.pop();
        }
        if pts.len() >= 3 {
            draw_hollow_polygon_mut(&mut canvas, &pts, green);
        }
        for p in &pts {
            draw_filled_circle_mut(
                &mut canvas,
                (p.x.round() as i32, p.y.round() as i32),
                3,
                red,
            );
        }
    }
    let out_path = std::env::args().nth(2).unwrap_or_else(|| {
        let p = std::path::Path::new(&img_path);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
        parent
            .join(format!("{stem}-polyseg.png"))
            .to_string_lossy()
            .into_owned()
    });
    canvas.save(&out_path)?;
    println!(
        "\nWrote {}-line polygon overlay ({}x{}) to: {out_path}",
        lines.len(),
        canvas.width(),
        canvas.height(),
    );

    // Stage 2 (optional): recognize each line if KRKN_DUMP_DIR is set, to dump
    // the recognizer's input crops. Pass line.quad to recognize_line_direct
    // (the multi-point boundary can't be indexed as TL/TR for deskew).
    let dump = std::env::var_os("KRKN_DUMP_DIR").is_some();
    if !dump {
        println!("\n(set KRKN_DUMP_DIR to also dump recognizer crops + run recognition)");
        return Ok(());
    }
    println!("KRKN_DUMP_DIR set — writing per-line crops and recognizing.");
    let t = Instant::now();
    let engine = Engine::load_from_buffers(BUNDLED_KRAKEN_SEG, BUNDLED_KRAKEN_REC)?;
    let mut recognized = 0;
    for (i, line) in lines.iter().enumerate() {
        if line.boundary.len() < 3 {
            continue;
        }
        let npts = line.boundary.len();
        // Report curvature so we can see which lines took the geometric-dewarp
        // branch inside recognize_line_poly.
        let mid = kraken_engine::recognition::dewarp::curved_midline(&line.boundary, 16);
        let sag = kraken_engine::recognition::dewarp::baseline_sagitta(&mid);
        let curved = sag >= 0.04 && mid.len() >= 3;
        let tag = if curved { "CURVED" } else { "straight" };
        // Crop with the multi-point boundary (tight polygon mask), deskew from
        // the quad — matches engine.rs's recognize_line_poly path.
        let result = match line.quad.as_ref() {
            Some(q) => engine.recognize_line_poly(&img, &line.boundary, q),
            None => engine.recognize_line_direct(&img, &line.boundary),
        };
        match result {
            Ok(text) => {
                recognized += 1;
                println!("  line {i:2} ({npts:2} pts, {tag} sag={sag:.3}): {text}");
            }
            Err(e) => println!("  line {i:2} ({npts} pts): recognize failed: {e}"),
        }
    }
    println!(
        "\nRecognized {recognized}/{} lines in {:?}",
        lines.len(),
        t.elapsed(),
    );
    Ok(())
}
