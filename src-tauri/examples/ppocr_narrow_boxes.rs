//! Quad-geometry dump for the PP-OCR detector: run small-det on an image and
//! print width / height / angle for every quad, flagging narrow ones
//! (top-edge width < 100 px). Run with:
//!
//!   cargo run --release --example ppocr_narrow_boxes -- <image.png>

use std::time::Instant;

use image::GenericImageView;
use ppocr_engine::{Detector, DetectorConfig};

const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../sample_images/p022.png".to_string());

    let img = image::open(&img_path)?;
    let (w, h) = img.dimensions();
    println!("Image: {img_path} ({w}x{h})");

    let det = Detector::load_from_buffer_with_config(
        BUNDLED_PPOCR_DET,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        DetectorConfig::small(),
    )?;

    let t = Instant::now();
    let detections = det.detect(&img)?;
    println!("Detection in {:?}: {} regions\n", t.elapsed(), detections.len());

    println!(
        "{:>4}  {:>7} {:>7} {:>8}  {:>7} {:>7} {:>8}  {:>8}  {}",
        "idx", "w_top", "h_left", "angle°", "w_rect", "h_rect", "angle_r", "score", "narrow"
    );
    let mut narrow = 0usize;
    for (i, d) in detections.iter().enumerate() {
        let p = &d.polygon;
        // Top edge p0→p1 = reading-direction width; left edge p0→p3 = height.
        let top = (p[1].0 - p[0].0, p[1].1 - p[0].1);
        let left = (p[3].0 - p[0].0, p[3].1 - p[0].1);
        let w_top = top.0.hypot(top.1);
        let h_left = left.0.hypot(left.1);
        let angle = top.1.atan2(top.0).to_degrees();

        // Min-area rect for cross-check: smaller/larger side + its angle.
        let quad = [(p[0].0, p[0].1), (p[1].0, p[1].1), (p[2].0, p[2].1), (p[3].0, p[3].1)];
        let (w_rect, h_rect, angle_r) = min_area_rect(&quad);

        let is_narrow = w_top < 100.0;
        if is_narrow {
            narrow += 1;
        }
        println!(
            "{i:>4}  {:>7.1} {:>7.1} {:>8.2}  {:>7.1} {:>7.1} {:>8.2}  {:>8.2}  {}",
            w_top,
            h_left,
            angle,
            w_rect,
            h_rect,
            angle_r,
            d.score,
            if is_narrow { "<-- width < 100" } else { "" }
        );
        if is_narrow {
            println!(
                "       quad: ({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0})",
                p[0].0, p[0].1, p[1].0, p[1].1, p[2].0, p[2].1, p[3].0, p[3].1
            );
        }
    }
    println!("\n{narrow} of {} quads have top-edge width < 100 px.", detections.len());
    Ok(())
}

/// Rotating-calipers-ish min-area rect via convex hull + edge iteration.
/// Returns (smaller side, larger side, angle of the larger side in degrees).
fn min_area_rect(p: &[(f32, f32); 4]) -> (f32, f32, f32) {
    // 4 points is already tiny; build hull by sorting then Andrew's monotone chain.
    let mut pts: Vec<(f32, f32)> = p.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    let mut hull: Vec<(f32, f32)> = Vec::with_capacity(4);
    for &pt in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0.0 {
            hull.pop();
        }
        hull.push(pt);
    }
    let lower = hull.len() + 1;
    for &pt in pts.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0.0 {
            hull.pop();
        }
        hull.push(pt);
    }
    hull.pop();

    let mut best: Option<(f32, f32, f32)> = None;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let e = (b.0 - a.0, b.1 - a.1);
        let elen = e.0.hypot(e.1);
        if elen < 1e-6 {
            continue;
        }
        let (ux, uy) = (e.0 / elen, e.1 / elen);
        let mut min1 = f32::MAX;
        let mut max1 = f32::MIN;
        let mut min2 = f32::MAX;
        let mut max2 = f32::MIN;
        for &q in &hull {
            let d1 = (q.0 - a.0) * ux + (q.1 - a.1) * uy;
            let d2 = -(q.0 - a.0) * uy + (q.1 - a.1) * ux;
            min1 = min1.min(d1);
            max1 = max1.max(d1);
            min2 = min2.min(d2);
            max2 = max2.max(d2);
        }
        let area = (max1 - min1) * (max2 - min2);
        if best.map_or(true, |(ba, _, _)| area < ba) {
            let s1 = max1 - min1;
            let s2 = max2 - min2;
            let (small, large) = if s1 <= s2 { (s1, s2) } else { (s2, s1) };
            // Angle of the larger side: if s1 is larger it runs along u, else
            // along u's normal.
            let ang = if s1 >= s2 { uy.atan2(ux) } else { (-uy).atan2(ux) };
            best = Some((small, large, ang.to_degrees()));
        }
    }
    best.unwrap_or((0.0, 0.0, 0.0))
}

fn cross(o: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}
