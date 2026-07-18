//! Benchmark the vendored kraken engine: per-stage segmentation timing +
//! per-line recognition timing (serial vs rayon-parallel), with stats over N
//! iterations.
//!
//! Usage:
//!   cargo run --release --example bench_kraken -- [image.png] [models_dir] [iters]
//!
//! Defaults: image=/tmp/scan2_p1.png, models_dir=../kraken-models, iters=5
//!
//! Reference (kraken-rust project, on similar hardware): segmentation ~3-4s,
//! recognition ~1-2s. This benchmark reports where the time goes so we can
//! compare and find optimization opportunities.

use std::time::{Duration, Instant};

use image::GenericImageView;
use rayon::prelude::*;

use just_ocr_lib::kraken::{
    self,
    config::SegmentationConfig,
    containers::Segmentation,
    detect::postprocess,
    inference_candle::run_inference_candle,
    preprocess::preprocess,
    recognition::{
        preprocess::preprocess_line, RecognitionModel,
    },
    segmentation_candle::SegmentationModelCandle,
};

#[derive(Clone, Copy)]
struct Stats {
    min: f64,
    mean: f64,
    median: f64,
    max: f64,
}

fn stats(durations: &[Duration]) -> Stats {
    let mut ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ms.len() as f64;
    let sum: f64 = ms.iter().sum();
    Stats {
        min: ms.first().copied().unwrap_or(0.0),
        mean: if n > 0.0 { sum / n } else { 0.0 },
        median: if ms.is_empty() {
            0.0
        } else if ms.len() % 2 == 1 {
            ms[ms.len() / 2]
        } else {
            (ms[ms.len() / 2 - 1] + ms[ms.len() / 2]) / 2.0
        },
        max: ms.last().copied().unwrap_or(0.0),
    }
}

fn fmt(s: Stats) -> String {
    format!(
        "min {:7.1}  mean {:7.1}  median {:7.1}  max {:7.1} ms",
        s.min, s.mean, s.median, s.max
    )
}

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/scan2_p1.png".to_string());
    let models_dir =
        std::env::args().nth(2).unwrap_or_else(|| "../kraken-models".to_string());
    let iters: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().unwrap_or(5))
        .unwrap_or(5);

    let seg_path = format!("{models_dir}/bur_segment.safetensors");
    let rec_path = format!("{models_dir}/bur_recog.safetensors");

    println!("=== kraken engine benchmark ===");
    println!("image:   {img_path}");
    println!("seg:     {seg_path}");
    println!("rec:     {rec_path}");
    println!("iters:   {iters} (per stage; 1 warmup discarded)");
    println!(
        "threads: rayon sees {} (set RAYON_NUM_THREADS to change)",
        rayon::current_num_threads()
    );
    println!();

    // ── Load image + models (untimed) ───────────────────────────────────────
    let img = image::open(&img_path)?;
    let (w, h) = img.dimensions();
    println!("image dimensions: {w}x{h}");

    let t = Instant::now();
    let seg = SegmentationModelCandle::load(&seg_path)?;
    println!("seg model load:  {:6.1} ms (one-time)", t.elapsed().as_secs_f64() * 1000.0);
    let t = Instant::now();
    let rec = RecognitionModel::load(&rec_path)?;
    println!("rec model load:  {:6.1} ms (one-time)", t.elapsed().as_secs_f64() * 1000.0);
    println!();

    // ── Segmentation: per-stage breakdown ───────────────────────────────────
    // We replicate detect_candle()'s stages here so each is individually timed:
    // preprocess → inference → scale fix → postprocess. postprocess is exposed
    // as pub so we can call it directly without re-running preprocess+inference.
    println!("── Segmentation (per-stage) ──────────────────────────────────────");
    let config = SegmentationConfig { text_direction: "horizontal-lr".to_string() };

    let padding = match seg.meta.padding.len() {
        0 => [0i64, 0, 0, 0],
        2 => [seg.meta.padding[0], seg.meta.padding[0],
              seg.meta.padding[1], seg.meta.padding[1]],
        4 => [seg.meta.padding[0], seg.meta.padding[1],
              seg.meta.padding[2], seg.meta.padding[3]],
        _ => [0, 0, 0, 0],
    };

    let mut t_preprocess = Vec::with_capacity(iters);
    let mut t_inference = Vec::with_capacity(iters);
    let mut t_postprocess = Vec::with_capacity(iters);
    let mut t_seg_total = Vec::with_capacity(iters);
    let mut last_seg: Option<Segmentation> = None;

    // 1 warmup + iters timed.
    for i in 0..=iters {
        let t_total = Instant::now();

        let t = Instant::now();
        let preprocessed = preprocess(&img, seg.height, &padding, 0)?;
        let preprocess_dur = t.elapsed();

        let t = Instant::now();
        let mut heatmap = run_inference_candle(&seg, &preprocessed)?;
        let inference_dur = t.elapsed();

        // Scale fix (mirrors detect_candle exactly).
        let (orig_w, orig_h) = img.dimensions();
        let (hm_h, hm_w) = (heatmap.probs.dim().1, heatmap.probs.dim().2);
        heatmap.scale = (orig_w as f64 / hm_w as f64, orig_h as f64 / hm_h as f64);

        let t = Instant::now();
        let result = postprocess(
            &heatmap,
            &config,
            seg.meta.topline,
            &seg.meta.bounding_regions,
        )?;
        let postprocess_dur = t.elapsed();

        let total_dur = t_total.elapsed();
        last_seg = Some(result);

        if i == 0 {
            println!("  (warmup discarded)");
            continue;
        }
        t_preprocess.push(preprocess_dur);
        t_inference.push(inference_dur);
        t_postprocess.push(postprocess_dur);
        t_seg_total.push(total_dur);
    }
    let seg = last_seg.expect("segmentation ran");

    let s_pre = stats(&t_preprocess);
    let s_inf = stats(&t_inference);
    let s_post = stats(&t_postprocess);
    let s_tot = stats(&t_seg_total);
    println!("  preprocess:     {}", fmt(s_pre));
    println!("  inference:      {}", fmt(s_inf));
    println!("  postprocess:    {}", fmt(s_post));
    println!("  ────────────────────────────────────────────");
    println!("  TOTAL:          {}", fmt(s_tot));
    println!(
        "  (stages sum at median: {:.1} ms — close to TOTAL median {:.1} ms)",
        s_pre.median + s_inf.median + s_post.median,
        s_tot.median
    );
    println!("  lines detected: {}", seg.lines.len());
    println!();

    // ── Recognition: serial vs parallel ─────────────────────────────────────
    // Build the line crops once (cropping is not what we're benchmarking).
    let crops: Vec<image::DynamicImage> = seg
        .lines
        .iter()
        .filter_map(|line| {
            if line.boundary.len() < 3 {
                return None;
            }
            let min_x = line.boundary.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).max(0.0) as u32;
            let min_y = line.boundary.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(0.0) as u32;
            let max_x = line.boundary.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).min((w - 1) as f64) as u32;
            let max_y = line.boundary.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).min((h - 1) as f64) as u32;
            let cw = max_x.saturating_sub(min_x) + 1;
            let ch = max_y.saturating_sub(min_y) + 1;
            if cw < 2 || ch < 2 {
                return None;
            }
            Some(image::DynamicImage::ImageRgb8(
                img.crop_imm(min_x, min_y, cw, ch).to_rgb8(),
            ))
        })
        .collect();
    println!("── Recognition ({} line crops) ──────────────────────────────────", crops.len());

    // Serial: per-line breakdown (preprocess / forward+decode) and total.
    let mut t_rec_pre_total = Vec::with_capacity(iters);
    let mut t_rec_fwd_total = Vec::with_capacity(iters);
    let mut t_rec_serial_total = Vec::with_capacity(iters);

    for i in 0..=iters {
        let t_total = Instant::now();
        let mut t_pre_acc = Duration::ZERO;
        let mut t_fwd_acc = Duration::ZERO;
        for crop in &crops {
            let t = Instant::now();
            let tensor = preprocess_line(crop, rec.height, rec.padding)?;
            t_pre_acc += t.elapsed();

            let t = Instant::now();
            let _text = rec.recognize(&tensor)?;
            t_fwd_acc += t.elapsed();
        }
        let total = t_total.elapsed();
        if i == 0 {
            println!("  (warmup discarded)");
            continue;
        }
        t_rec_pre_total.push(t_pre_acc);
        t_rec_fwd_total.push(t_fwd_acc);
        t_rec_serial_total.push(total);
    }

    println!("  SERIAL");
    println!("    preprocess (all lines): {}", fmt(stats(&t_rec_pre_total)));
    println!("    forward+decode (all):   {}", fmt(stats(&t_rec_fwd_total)));
    println!("    ────────────────────────────────────────────");
    println!("    TOTAL serial:            {}", fmt(stats(&t_rec_serial_total)));

    // Parallel: rayon over lines, single shared &RecognitionModel.
    let mut t_rec_par_total = Vec::with_capacity(iters);
    for i in 0..=iters {
        let t_total = Instant::now();
        let results: Vec<String> = crops
            .par_iter()
            .map(|crop| {
                let tensor = preprocess_line(crop, rec.height, rec.padding)?;
                rec.recognize(&tensor)
            })
            .collect::<Result<_, anyhow::Error>>()
            .map_err(|e| anyhow::anyhow!(e))?;
        let _ = results; // discard text; we're timing
        let total = t_total.elapsed();
        if i == 0 {
            println!("  (warmup discarded)");
            continue;
        }
        t_rec_par_total.push(total);
    }
    println!("  PARALLEL (rayon, {} threads)", rayon::current_num_threads());
    println!("    TOTAL parallel:          {}", fmt(stats(&t_rec_par_total)));

    let serial_median = stats(&t_rec_serial_total).median;
    let par_median = stats(&t_rec_par_total).median;
    if par_median > 0.0 {
        println!("    speedup vs serial:       {:.2}×", serial_median / par_median);
    }
    println!();

    // ── NOTES ───────────────────────────────────────────────────────────────
    println!("── NOTES ────────────────────────────────────────────────────────");
    println!("• Segmentation TOTAL is the honest preprocess + inference +");
    println!("  postprocess sum — matches what engine::run_ocr spends per image.");
    println!("• Reference (kraken-rust, similar HW): seg ~3-4s, recog ~1-2s.");
    println!("• RecognitionModel is Send+Sync (candle tensors under Arc), so the");
    println!("  parallel path shares one model — no weight duplication.");
    println!("• The engine in engine.rs currently calls recognize_line() serially.");
    println!("  If PARALLEL TOTAL is materially faster, switching to rayon in");
    println!("  engine.rs is a drop-in optimization.");

    // Silence unused-import warning for `kraken` re-export path.
    let _ = kraken::KrakenCache::new();
    Ok(())
}
