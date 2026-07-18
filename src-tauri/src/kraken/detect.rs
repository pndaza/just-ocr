//! Top-level detection orchestrator.
//!
//! Wires the full pipeline: preprocess → inference → vectorize → boundaries → reading order.
//! Source: kraken/lib/vgsl/spred.py:160-235 (_segmentation_pred)

use anyhow::Result;
use image::{DynamicImage, GenericImageView};

use crate::kraken::config::SegmentationConfig;
use crate::kraken::containers::{BaselineLine, Region, Segmentation};
use crate::kraken::heatmap::Heatmap;
use crate::kraken::preprocess::preprocess;
use crate::kraken::segmentation_candle::SegmentationModelCandle;
use crate::kraken::vectorize::{vectorize_regions, vectorize_lines};
use crate::kraken::boundaries::calculate_polygonal_environment;
use crate::kraken::reading_order::polygonal_reading_order;
use crate::kraken::ndimage::filters::{gaussian_filter, sobel};
use crate::kraken::polygon::{Point, Polygon, point_in_polygon};

/// Run detection using the candle-core backend (native Rust, no ONNX).
pub fn detect_candle(
    image: &DynamicImage,
    model: &SegmentationModelCandle,
    config: &SegmentationConfig,
) -> Result<Segmentation> {
    use crate::kraken::inference_candle::run_inference_candle;

    let padding = match model.meta.padding.len() {
        0 => [0i64, 0, 0, 0],
        2 => [model.meta.padding[0], model.meta.padding[0],
              model.meta.padding[1], model.meta.padding[1]],
        4 => [model.meta.padding[0], model.meta.padding[1],
              model.meta.padding[2], model.meta.padding[3]],
        _ => [0, 0, 0, 0],
    };

    // 1. Preprocess
    log::debug!("Preprocessing image...");
    let preprocessed = preprocess(image, model.height, &padding, 0)?;

    // 2. Inference (candle)
    log::debug!("Running inference (candle)...");
    let mut heatmap = run_inference_candle(model, &preprocessed)?;

    // Fix scale
    let (orig_w, orig_h) = image.dimensions();
    let (hm_h, hm_w) = (heatmap.probs.dim().1, heatmap.probs.dim().2);
    heatmap.scale = (orig_w as f64 / hm_w as f64, orig_h as f64 / hm_h as f64);

    // 3-8. Post-processing
    postprocess(&heatmap, config, model.meta.topline, &model.meta.bounding_regions)
}

/// Shared post-processing: vectorize heatmap → baselines/regions → boundaries → reading order.
fn postprocess(
    heatmap: &Heatmap,
    config: &SegmentationConfig,
    topline: bool,
    bounding_regions: &Option<Vec<String>>,
) -> Result<Segmentation> {
    // 3. Vectorize regions
    log::debug!("Vectorizing regions...");
    let mut regions: Vec<Region> = Vec::new();
    let mut suppl_obj_polys: Vec<Vec<(f64, f64)>> = Vec::new();

    for (reg_type, &idx) in &heatmap.cls_map.regions {
        let reg_heatmap = heatmap.probs.slice(ndarray::s![idx, .., ..]).to_owned();
        let region_polys = vectorize_regions(&reg_heatmap, 0.5);

        for poly in region_polys {
            let scaled: Vec<(f64, f64)> = poly.exterior.iter()
                .map(|p| (p.x * heatmap.scale.0, p.y * heatmap.scale.1))
                .collect();
            let reg = Region {
                id: format!("_{}", uuid_like()),
                boundary: scaled.clone(),
                region_type: reg_type.clone(),
            };
            regions.push(reg);

            let is_bounding = bounding_regions
                .as_ref()
                .map(|br| br.contains(reg_type))
                .unwrap_or(false);
            if is_bounding {
                suppl_obj_polys.push(scaled);
            }
        }
    }

    // 4. Vectorize baselines
    log::debug!("Vectorizing baselines...");
    let st_sep = *heatmap.cls_map.aux.get("_start_separator").unwrap_or(&0);
    let end_sep = *heatmap.cls_map.aux.get("_end_separator").unwrap_or(&1);

    let text_direction_base = if config.text_direction.starts_with("horizontal") {
        "horizontal"
    } else {
        "vertical"
    };

    let mut baselines: Vec<(String, Vec<Point>)> = Vec::new();
    for (bl_type, &idx) in &heatmap.cls_map.baselines {
        let stacked = ndarray::stack![
            ndarray::Axis(0),
            heatmap.probs.slice(ndarray::s![st_sep, .., ..]).to_owned(),
            heatmap.probs.slice(ndarray::s![end_sep, .., ..]).to_owned(),
            heatmap.probs.slice(ndarray::s![idx, .., ..]).to_owned(),
        ];

        let lines = vectorize_lines(&stacked, 0.17, 5, text_direction_base, 400);
        for line in lines {
            baselines.push((bl_type.clone(), line));
        }
    }

    // 4b. Filter fault baselines BEFORE boundary computation so they don't
    // pollute the neighbor-distance calculations in the polygonal environment.
    //
    // Two-step approach:
    //   1. Select the 70% longest baselines as "body" reference lines.
    //   2. Compute typical line spacing from consecutive body-line pairs (most
    //      similar cluster within 1 std-dev of median).
    //
    // A baseline is a fault if it is short (< 50% of typical body-line length)
    // AND too close to a body neighbor (gap < 50% of typical spacing) — i.e.
    // it sits inside another line's zone (diacritic, spot, strike, ink bleed).
    if baselines.len() > 3 {
        let avg_scale = (heatmap.scale.0 + heatmap.scale.1) / 2.0;
        let bl_lengths: Vec<f64> = baselines.iter()
            .map(|(_, bl)| bl_arc_length(bl))
            .collect();
        let centers: Vec<f64> = baselines.iter()
            .map(|(_, bl)| bl.iter().map(|p| p.y).sum::<f64>() / bl.len() as f64)
            .collect();

        // Step 1: select 70% longest as body lines.
        let mut indexed_lens: Vec<(usize, f64)> =
            bl_lengths.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed_lens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff = (baselines.len() as f64 * 0.7).ceil() as usize;
        let body_set: std::collections::HashSet<usize> =
            indexed_lens.iter().take(cutoff).map(|(i, _)| *i).collect();

        // Step 2: typical spacing from consecutive body lines.
        let mut body_ys: Vec<(usize, f64)> = centers
            .iter()
            .enumerate()
            .filter(|(i, _)| body_set.contains(i))
            .map(|(i, &y)| (i, y))
            .collect();
        body_ys.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let typical_spacing = if body_ys.len() >= 2 {
            let all_sp: Vec<f64> = body_ys.windows(2).map(|w| w[1].1 - w[0].1).collect();
            let mut sorted = all_sp.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = sorted[sorted.len() / 2];
            let mean = all_sp.iter().sum::<f64>() / all_sp.len() as f64;
            let var: f64 = all_sp.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / all_sp.len() as f64;
            let std = var.sqrt();
            let cluster: Vec<f64> = all_sp.iter().filter(|&&d| (d - med).abs() <= std).cloned().collect();
            if cluster.is_empty() { med } else { cluster.iter().sum::<f64>() / cluster.len() as f64 }
        } else {
            40.0 / avg_scale.min(1.0)
        };

        let typical_body_len = body_set.iter()
            .map(|&i| bl_lengths[i])
            .sum::<f64>() / body_set.len().max(1) as f64;
        // Short = less than 15px (original coords) or 20% of body length,
        // whichever is smaller. Using a high percentage would filter
        // legitimate subtitles and section headings that are shorter than
        // full-width body lines.
        let absolute_short = 15.0 / avg_scale.min(1.0);
        let short_threshold = absolute_short.min(typical_body_len * 0.2);
        let proximity_threshold = typical_spacing * 0.5;

        // Precompute x-ranges for overlap test.
        let x_ranges: Vec<(f64, f64)> = baselines.iter()
            .map(|(_, bl)| {
                let xs: Vec<f64> = bl.iter().map(|p| p.x).collect();
                (xs.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                 xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
            })
            .collect();

        let before = baselines.len();
        let keep: Vec<bool> = (0..baselines.len()).map(|i| {
            // Body lines are always kept.
            if bl_lengths[i] >= short_threshold {
                return true;
            }
            // Short line: check if it's too close to a body neighbor AND their
            // x-ranges overlap (side-by-side lines like title/page-number are kept).
            let too_close = (0..baselines.len()).any(|j| {
                if j == i || !body_set.contains(&j) {
                    return false;
                }
                let y_close = (centers[j] - centers[i]).abs() < proximity_threshold;
                let x_overlap = x_ranges[i].0 < x_ranges[j].1 && x_ranges[j].0 < x_ranges[i].1;
                y_close && x_overlap
            });
            !too_close
        }).collect();

        let filtered = before - keep.iter().filter(|&&k| k).count();
        if filtered > 0 {
            log::debug!("Filtered {} fault baselines (short<{short_threshold:.0}px & close<{proximity_threshold:.0}px to body)", filtered);
        }
        baselines = baselines.into_iter().zip(keep).filter(|(_, k)| *k).map(|(b, _)| b).collect();
    }

    // 5. Compute boundary polygons
    log::debug!("Computing boundary polygons...");
    let im_feats = gaussian_filter(&sobel(&heatmap.scal_im), 0.5);
    let (max_x, max_y) = (heatmap.scal_im.dim().1 as f64 - 1.0, heatmap.scal_im.dim().0 as f64 - 1.0);

    let bl_points: Vec<Vec<Point>> = baselines.iter().map(|(_, bl)| bl.clone()).collect();
    let suppl_as_points: Vec<Vec<Point>> = suppl_obj_polys.iter()
        .map(|poly| poly.iter().map(|&(x, y)| Point::new(x, y)).collect())
        .collect();

    let polygons = calculate_polygonal_environment(
        &bl_points, &im_feats, &suppl_as_points, topline, (max_x, max_y),
    );

    // 6. Build BaselineLine objects (scale to original coords)
    let mut lines: Vec<BaselineLine> = Vec::new();
    for (i, (bl_type, bl)) in baselines.iter().enumerate() {
        let scaled_bl: Vec<(f64, f64)> = bl.iter()
            .map(|p| (p.x * heatmap.scale.0, p.y * heatmap.scale.1))
            .collect();

        let scaled_boundary: Vec<(f64, f64)> = match &polygons[i] {
            Some(poly) => poly.exterior.iter()
                .map(|p| (p.x * heatmap.scale.0, p.y * heatmap.scale.1))
                .collect(),
            None => Vec::new(),
        };

        lines.push(BaselineLine {
            id: format!("_{}", uuid_like()),
            baseline: scaled_bl,
            boundary: scaled_boundary,
            script: bl_type.clone(),
            regions: Vec::new(),
        });
    }

    // 6b. Fix polygon height outliers BEFORE merging so over-height polygons
    // don't absorb adjacent lines.
    fix_polygon_outliers(&mut lines);

    // 6c. Merge short baselines that overlap with another line's polygon.
    if lines.len() > 2 {
        let bl_lens: Vec<f64> = lines.iter()
            .map(|l| l.baseline.windows(2)
                .map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt())
                .sum())
            .collect();
        // Bounding boxes of all line boundaries.
        let bboxes: Vec<(f64, f64, f64, f64)> = lines.iter()
            .map(|l| {
                if l.boundary.len() < 3 {
                    (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)
                } else {
                    let xs: Vec<f64> = l.boundary.iter().map(|p| p.0).collect();
                    let ys: Vec<f64> = l.boundary.iter().map(|p| p.1).collect();
                    (xs.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                     ys.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                     xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                     ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
                }
            })
            .collect();

        // Iteratively merge: a short line is merged into the LONGEST overlapping
        // line, but only if that parent is not itself being merged. Process
        // from shortest to longest so parents are resolved first.
        let mut order: Vec<usize> = (0..lines.len()).collect();
        order.sort_by(|&a, &b| bl_lens[a].partial_cmp(&bl_lens[b]).unwrap_or(std::cmp::Ordering::Equal));

        let mut keep = vec![true; lines.len()];
        for &i in &order {
            if !keep[i] || lines[i].boundary.len() < 3 {
                continue;
            }
            let bl_xs: Vec<f64> = lines[i].baseline.iter().map(|p| p.0).collect();
            let bl_ys: Vec<f64> = lines[i].baseline.iter().map(|p| p.1).collect();
            let (bl_xmin, bl_xmax) = (bl_xs.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                                       bl_xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
            let (bl_ymin, bl_ymax) = (bl_ys.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                                       bl_ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
            // Find the longest line that this short line overlaps.
            let mut best_parent: Option<usize> = None;
            for j in 0..lines.len() {
                if j == i || !keep[j] || lines[j].boundary.len() < 3 {
                    continue;
                }
                // Parent must be at least 2x longer.
                if bl_lens[j] < bl_lens[i] * 2.0 {
                    continue;
                }
                let (pxmin, pymin, pxmax, pymax) = bboxes[j];
                let x_overlap = bl_xmin < pxmax && pxmin < bl_xmax;
                let y_overlap = bl_ymin < pymax && pymin < bl_ymax;
                if x_overlap && y_overlap {
                    match best_parent {
                        None => best_parent = Some(j),
                        Some(prev) if bl_lens[j] > bl_lens[prev] => best_parent = Some(j),
                        _ => {}
                    }
                }
            }
            if let Some(p) = best_parent {
                log::debug!("MERGE: line {i} (len={:.0} y=[{bl_ymin:.0},{bl_ymax:.0}] x=[{bl_xmin:.0},{bl_xmax:.0}]) -> parent {p} (len={:.0} bnd_y=[{:.0},{:.0}] bnd_x=[{:.0},{:.0}])",
                    bl_lens[i], bl_lens[p],
                    bboxes[p].1, bboxes[p].3, bboxes[p].0, bboxes[p].2);
                keep[i] = false; // merge into parent
            }
        }

        let merged = lines.len() - keep.iter().filter(|&&k| k).count();
        if merged > 0 {
            log::debug!("Merged {merged} overlapping short baselines into parent lines");
        }
        lines = lines.into_iter().zip(keep).filter(|(_, k)| *k).map(|(l, _)| l).collect();
    }

    // 6d. Simplify polygons for straight baselines.
    //
    // The seam carver produces a wavy top/bottom edge that pinches text at the
    // valleys. For straight baselines (common in Burmese), prune the
    // non-extreme vertices on each side — keep only the most extreme points
    // (defined by KEEP_FRAC in simplify_straight_line_polygons) of the
    // above-baseline edge and the below-baseline edge. The surviving peaks,
    // joined by straight segments, cut across the valleys, giving a flatter,
    // taller band. Curved baselines keep the model's polygon.
    simplify_straight_line_polygons(&mut lines);

    // 7. Assign lines to regions
    let reg_polys: Vec<Polygon> = regions.iter()
        .map(|r| Polygon::from_tuples(&r.boundary))
        .collect();
    for line in &mut lines {
        if line.baseline.is_empty() { continue; }
        let mid_x = line.baseline.iter().map(|p| p.0).sum::<f64>() / line.baseline.len() as f64;
        let mid_y = line.baseline.iter().map(|p| p.1).sum::<f64>() / line.baseline.len() as f64;
        let mid = Point::new(mid_x, mid_y);
        for (reg_idx, reg_poly) in reg_polys.iter().enumerate() {
            if point_in_polygon(&mid, reg_poly) {
                line.regions.push(regions[reg_idx].id.clone());
            }
        }
    }

    // 8. Reading order
    log::debug!("Computing reading order...");
    let ro_text_direction = if config.text_direction.ends_with("lr") { "lr" } else { "rl" };
    let order = polygonal_reading_order(&lines, &regions, ro_text_direction);
    let ordered_lines: Vec<BaselineLine> = order.iter()
        .filter_map(|&i| lines.get(i).cloned())
        .collect();

    let script_detection = heatmap.cls_map.baselines.len() > 1;

    Ok(Segmentation {
        text_direction: config.text_direction.clone(),
        lines: ordered_lines,
        regions,
        script_detection,
    })
}

/// A monotonically increasing counter for generating unique IDs within a
/// single `detect` call. Combined with the timestamp prefix this avoids
/// collisions that a bare nanosecond clock would produce in tight loops.
static ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Generate a unique ID. Combines a coarse timestamp with a process-wide
/// atomic counter so that IDs generated in rapid succession never collide.
fn uuid_like() -> String {
    use std::sync::atomic::Ordering;
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("_{secs:x}{n:x}")
}

/// Total arc length of a baseline polyline (in heatmap coordinates).
fn bl_arc_length(bl: &[Point]) -> f64 {
    bl.windows(2)
        .map(|w| ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt())
        .sum()
}

/// Fix polygon height outliers by clamping points that extend beyond the
/// typical text-body extent.
///
/// Measures the actual above/below extents of all line polygons relative to
/// their baselines, computes median + MAD, and clamps any polygon point
/// exceeding median + 3*MAD. Only outliers are affected — normal polygons
/// are untouched.
fn fix_polygon_outliers(lines: &mut [BaselineLine]) {
    if lines.len() < 4 {
        return;
    }

    // Measure per-line extents (in original image coordinates).
    let mut extents: Vec<(f64, f64)> = Vec::new(); // (above, below)
    for line in lines.iter() {
        if line.boundary.len() < 3 || line.baseline.is_empty() {
            continue;
        }
        let bl_y: f64 = line.baseline.iter().map(|p| p.1).sum::<f64>() / line.baseline.len() as f64;
        let bnd_ys: Vec<f64> = line.boundary.iter().map(|p| p.1).collect();
        let above = bl_y - bnd_ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let below = bnd_ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - bl_y;
        if above > 0.0 && below > 0.0 {
            extents.push((above, below));
        }
    }

    if extents.len() < 4 {
        return;
    }

    let median_above = median(&extents.iter().map(|&(a, _)| a).collect::<Vec<_>>());
    let median_below = median(&extents.iter().map(|&(_, b)| b).collect::<Vec<_>>());

    let mad_above = median(&extents.iter().map(|&(a, _)| (a - median_above).abs()).collect::<Vec<_>>());
    let mad_below = median(&extents.iter().map(|&(_, b)| (b - median_below).abs()).collect::<Vec<_>>());

    let max_above = median_above + 3.0 * mad_above;
    let max_below = median_below + 3.0 * mad_below;

    let mut fixed_count = 0;
    for line in lines.iter_mut() {
        if line.boundary.len() < 3 || line.baseline.is_empty() {
            continue;
        }
        let bl_y: f64 = line.baseline.iter().map(|p| p.1).sum::<f64>() / line.baseline.len() as f64;
        let lo = bl_y - max_above;
        let hi = bl_y + max_below;
        let mut changed = false;
        for p in line.boundary.iter_mut() {
            if p.1 < lo {
                p.1 = lo;
                changed = true;
            } else if p.1 > hi {
                p.1 = hi;
                changed = true;
            }
        }
        if changed {
            fixed_count += 1;
        }
    }

    if fixed_count > 0 {
        log::debug!("Fixed {fixed_count} polygon height outliers (above<{max_above:.0}px, below<{max_below:.0}px)");
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

/// Simplify polygons for straight baselines by pruning low-extent vertices.
///
/// The seam carver produces a wavy top/bottom edge: generous where ink or
/// stacked diacritics push it out, pinched everywhere else. For straight
/// baselines (common in Burmese) that waviness clips text body. Rather than
/// moving points, this step *removes* the non-extreme vertices on each side:
/// it keeps only the top `KEEP_FRAC` (see constant below) highest-reaching
/// points of the above-baseline edge and the same fraction of the
/// lowest-reaching points of the below-baseline edge. The surviving peaks,
/// joined by straight segments, cut across the valleys — yielding a flatter,
/// taller, less curvy band. Curved baselines keep the model polygon untouched.
///
/// Vertices near the left/right edges (within `EDGE_MARGIN` of the polygon's
/// x-extent) are always retained so the band corners stay anchored to where
/// the top and bottom edges actually meet, instead of being clipped short.
fn simplify_straight_line_polygons(lines: &mut [BaselineLine]) {
    /// Fraction of the most extreme vertices to retain on each side.
    const KEEP_FRAC: f64 = 0.8;
    /// Fraction of the polygon's x-extent at each edge that is kept verbatim.
    const EDGE_MARGIN: f64 = 0.05;

    let mut simplified = 0;
    for line in lines.iter_mut() {
        if line.boundary.len() < 6 || line.baseline.len() < 2 {
            continue;
        }

        // Only touch straight / nearly-straight baselines; curved lines keep
        // the model's polygon (their top edge should follow the curve).
        let bl_ys_pts: Vec<f64> = line.baseline.iter().map(|p| p.1).collect();
        let bl_y_range = bl_ys_pts.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                          - bl_ys_pts.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let bl_len: f64 = line.baseline.windows(2)
            .map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt())
            .sum();
        if bl_len < 10.0 || bl_y_range / bl_len > 0.05 {
            continue;
        }

        let bl_y: f64 = bl_ys_pts.iter().sum::<f64>() / bl_ys_pts.len() as f64;

        // Left/right x-extent of the polygon and the edge bands to preserve.
        let xs: Vec<f64> = line.boundary.iter().map(|p| p.0).collect();
        let x_min = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let margin = (x_max - x_min) * EDGE_MARGIN;
        let left_edge = x_min + margin;
        let right_edge = x_max - margin;

        // Split boundary y-values into above (top edge) / below (bottom edge).
        let mut above_ys: Vec<f64> = Vec::new(); // y < bl_y
        let mut below_ys: Vec<f64> = Vec::new(); // y > bl_y
        for &(x, y) in &line.boundary {
            // Don't let the edge bands bias the extent cutoff — only the
            // interior points define what "extreme" means for the band.
            if x < left_edge || x > right_edge { continue; }
            if y < bl_y { above_ys.push(y); }
            else if y > bl_y { below_ys.push(y); }
        }

        // Cutoff y for the KEEP_FRAC most extreme interior points on each side.
        // Above: keep the smallest y (highest) → pass if y <= cutoff.
        // Below: keep the largest  y (lowest)  → pass if y >= cutoff.
        let cutoff_above = extent_cutoff(&above_ys, KEEP_FRAC, false);
        let cutoff_below = extent_cutoff(&below_ys, KEEP_FRAC, true);

        // Rebuild the ring: always keep the edge bands verbatim (so the
        // corners where top/bottom meet stay anchored), and within the
        // interior drop non-extreme vertices.
        let original = std::mem::take(&mut line.boundary);
        let mut kept: Vec<(f64, f64)> = Vec::new();
        for &(x, y) in &original {
            if x < left_edge || x > right_edge {
                // Edge band — keep verbatim to anchor the corners.
                kept.push((x, y));
            } else if y < bl_y {
                if y <= cutoff_above { kept.push((x, y)); }
            } else if y > bl_y {
                if y >= cutoff_below { kept.push((x, y)); }
            } else {
                kept.push((x, y)); // exactly on the baseline — keep
            }
        }

        if kept.len() >= 3 && kept.len() < original.len() {
            line.boundary = kept;
            simplified += 1;
        } else {
            line.boundary = original; // not enough pruning — restore
        }
    }

    if simplified > 0 {
        log::debug!(
            "Simplified {simplified} straight-line polygons (kept top {}% extent, {}% edge bands)",
            (KEEP_FRAC * 100.0) as u32,
            (EDGE_MARGIN * 100.0) as u32
        );
    }
}

/// Return the cutoff y-value such that the `frac` most extreme entries pass.
///
/// `largest == false` → keeps the *smallest* values (highest above baseline);
///   the cutoff is the largest y among the kept set (pass if `y <= cutoff`).
/// `largest == true`  → keeps the *largest* values (lowest below baseline);
///   the cutoff is the smallest y among the kept set (pass if `y >= cutoff`).
fn extent_cutoff(values: &[f64], frac: f64, largest: bool) -> f64 {
    if values.is_empty() {
        return if largest { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let take = ((sorted.len() as f64) * frac).ceil() as usize;
    let take = take.clamp(1, sorted.len());
    if largest {
        sorted[sorted.len() - take] // smallest of the kept large values
    } else {
        sorted[take - 1] // largest of the kept small values
    }
}
