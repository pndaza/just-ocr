//! Vectorization: heatmap channels → region polygons and baseline polylines.
//! Source: kraken/lib/segmentation.py (vectorize_regions, vectorize_lines)

use crate::kraken::ndimage::morphology::label;
use crate::kraken::ndimage::filters::{sato, maximum_filter, convolve2d_same};
use crate::kraken::ndimage::morphology::skeletonize;
use crate::kraken::ndimage::mcp::mcp_connect;
use crate::kraken::contours::boundary_trace;
use crate::kraken::polygon::{Point, Polygon, boolean::unary_union, simplify};
use ndarray::{Array2, Array3, s};
use std::collections::HashMap;

/// Vectorize regions from a single-class heatmap channel.
/// Source: kraken/lib/segmentation.py:422-449
///
/// Args:
/// - `heatmap`: (H, W) probability array for one region class
/// - `threshold`: binarization threshold (default 0.5)
///
/// Returns: list of region polygons (exterior boundary coordinates as Points).
pub fn vectorize_regions(heatmap: &Array2<f32>, threshold: f32) -> Vec<Polygon> {
    // Binarize
    let binary: Array2<f32> = heatmap.mapv(|v| if v > threshold { 1.0 } else { 0.0 });

    // Label connected components
    let (labels, n_components) = label(&binary);
    if n_components == 0 {
        return Vec::new();
    }

    // Trace boundary of each component
    let mut boundaries: Vec<Polygon> = Vec::new();
    for comp_id in 1..=n_components as u32 {
        let coords: Vec<(usize, usize)> = (0..labels.dim().0)
            .flat_map(|y| (0..labels.dim().1).map(move |x| (y, x)))
            .filter(|&(y, x)| labels[[y, x]] == comp_id)
            .collect();

        if coords.len() < 3 {
            continue;
        }

        let boundary = boundary_trace(&coords);
        if boundary.len() > 2 {
            boundaries.push(Polygon::new(boundary));
        }
    }

    // Merge overlapping regions (unary_union)
    let merged = unary_union(&boundaries);

    // Simplify each with tolerance 10 (matching shapely simplify(10))
    merged
        .into_iter()
        .map(|poly| {
            let simplified = simplify(&poly.exterior, 10.0);
            Polygon::new(simplified)
        })
        .collect()
}

/// Vectorize baselines from a 3-channel heatmap stack.
/// Source: kraken/lib/segmentation.py:316-419
///
/// Args:
/// - `im`: (3, H, W) heatmap stack: channel 0 = start separator,
///   channel 1 = end separator, channel 2 = baseline
/// - `threshold`: sato ridge binarization threshold
/// - `min_length`: minimum Euclidean length of a kept baseline
/// - `text_direction`: "horizontal" or "vertical" (orientation fallback heuristic)
/// - `max_endpoints`: cap on the number of endpoints before noise filtering kicks in
///
/// Returns: list of baselines, each an ordered polyline of `Point`s (x=col, y=row).
pub fn vectorize_lines(
    im: &Array3<f32>,
    threshold: f32,
    min_length: usize,
    text_direction: &str,
    max_endpoints: usize,
) -> Vec<Vec<Point>> {
    let st_map = im.slice(s![0, .., ..]).to_owned();
    let end_map = im.slice(s![1, .., ..]).to_owned();
    let bl_map = im.slice(s![2, .., ..]).to_owned();

    let bl_enhanced = sato(&bl_map, None, false);
    let bin_bl_map: Array2<f32> = bl_enhanced.mapv(|v| if v > threshold { 1.0 } else { 0.0 });
    let line_skel = skeletonize(&bin_bl_map);

    // Find endpoints: convolve skeleton with [[1,1,1],[1,10,1],[1,1,1]];
    // a skeleton pixel whose convolution == 11 has exactly one foreground
    // neighbor (10 for itself + 1 neighbor) -> a line endpoint.
    let kernel = ndarray::array![[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 1.0]];
    let conv = convolve2d_same(&line_skel, &kernel);
    let mut endpoints: Vec<(usize, usize)> = Vec::new();
    for y in 0..line_skel.dim().0 {
        for x in 0..line_skel.dim().1 {
            if line_skel[[y, x]] > 0.5 && (conv[[y, x]] - 11.0).abs() < 0.5 {
                endpoints.push((y, x));
            }
        }
    }

    if endpoints.len() > max_endpoints {
        endpoints = filter_noisy_endpoints(&line_skel, &endpoints, max_endpoints);
    }

    if endpoints.len() < 2 {
        return Vec::new();
    }

    // MCP_Connect: cost surface is 0.0 on the skeleton, 1.0 off it, so paths
    // are routed along the ridge centerline.
    let cost: Array2<f32> = line_skel.mapv(|v| if v > 0.5 { 0.0 } else { 1.0 });
    let connections = mcp_connect(&cost, &endpoints);

    let mut lines: Vec<Vec<Point>> = connections
        .into_iter()
        .map(|conn| {
            let points: Vec<Point> = conn
                .path
                .iter()
                .map(|&(y, x)| Point::new(x as f64, y as f64))
                .collect();
            simplify(&points, 3.0)
        })
        .collect();

    lines = extend_boundaries(&lines, &bin_bl_map);

    // Orient each line using the dilated start/end separator heatmaps.
    let f_st_map = maximum_filter(&st_map, 20);
    let f_end_map = maximum_filter(&end_map, 20);

    let mut oriented_lines: Vec<Vec<Point>> = Vec::new();
    for mut bl in lines {
        if bl.is_empty() {
            continue;
        }
        let l_end = bl[0];
        let r_end = bl[bl.len() - 1];

        let l_st = f_st_map[[l_end.y as usize, l_end.x as usize]];
        let l_end_val = f_end_map[[l_end.y as usize, l_end.x as usize]];
        let r_st = f_st_map[[r_end.y as usize, r_end.x as usize]];
        let r_end_val = f_end_map[[r_end.y as usize, r_end.x as usize]];

        if l_st - l_end_val > 0.2 && r_st - r_end_val < -0.2 {
            // correctly oriented: start sep dominant at left, end sep at right
        } else if l_st - l_end_val < -0.2 && r_st - r_end_val > 0.2 {
            bl.reverse();
        } else if text_direction == "horizontal" {
            // Insufficient separator confidence. Python checks if bl[0].y >
            // bl[-1].y (for vertical-ish orientation), but for nearly-horizontal
            // lines where y-ordering is ambiguous, the MCP-Connect path direction
            // is arbitrary. Default to LTR: ensure bl[0].x < bl[-1].x.
            // Source: segmentation.py:409-412
            if bl[0].y > bl[bl.len() - 1].y {
                bl.reverse();
            } else if (bl[0].y - bl[bl.len() - 1].y).abs() < 10.0 && bl[0].x > bl[bl.len() - 1].x {
                // Nearly horizontal and currently RTL — flip to LTR
                bl.reverse();
            }
        } else {
            if bl[0].x > bl[bl.len() - 1].x {
                bl.reverse();
            }
        }

        let length: f64 = bl
            .windows(2)
            .map(|w| {
                let dx = w[1].x - w[0].x;
                let dy = w[1].y - w[0].y;
                (dx * dx + dy * dy).sqrt()
            })
            .sum();
        if length >= min_length as f64 {
            oriented_lines.push(bl);
        }
    }
    oriented_lines
}

/// Filter noisy skeleton endpoints when their count exceeds `max_endpoints`.
/// Source: kraken/lib/segmentation.py:343-362 (compact label-bucket form).
///
/// Keeps endpoints belonging to connected components with at most 10 endpoints;
/// if that still exceeds the budget, trims the largest components first (the
/// endpoint-rich, typically-noise clusters) until within budget.
fn filter_noisy_endpoints(
    line_skel: &Array2<f32>,
    endpoints: &[(usize, usize)],
    max_endpoints: usize,
) -> Vec<(usize, usize)> {
    let (labels, n_cc) = label(line_skel);
    if n_cc == 0 {
        return endpoints.to_vec();
    }

    // Group endpoints by their connected component id.
    let mut cc_endpoint_counts: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    for &(y, x) in endpoints {
        let cc = labels[[y, x]];
        cc_endpoint_counts.entry(cc).or_default().push((y, x));
    }

    // CC sizes (pixel counts) for the budget-by-size fallback.
    let mut cc_sizes: HashMap<u32, usize> = HashMap::new();
    for y in 0..labels.dim().0 {
        for x in 0..labels.dim().1 {
            let cc = labels[[y, x]];
            if cc != 0 {
                *cc_sizes.entry(cc).or_insert(0) += 1;
            }
        }
    }

    // Keep CCs with <= 10 endpoints (low-endpoint components are real lines).
    let mut kept: Vec<(usize, usize)> = Vec::new();
    let mut kept_large: Vec<(u32, usize)> = Vec::new(); // (cc, size) for >10-endpoint CCs
    for (cc, eps) in &cc_endpoint_counts {
        if eps.len() <= 10 {
            kept.extend_from_slice(eps);
        } else {
            kept_large.push((*cc, *cc_sizes.get(cc).unwrap_or(&0)));
        }
    }

    // If still over budget, drop the largest high-endpoint CCs first.
    kept_large.sort_by(|a, b| b.1.cmp(&a.1));
    for (cc, _) in &kept_large {
        if kept.len() >= max_endpoints {
            break;
        }
        if let Some(eps) = cc_endpoint_counts.get(cc) {
            kept.extend_from_slice(eps);
        }
    }

    if kept.len() > max_endpoints {
        kept.truncate(max_endpoints);
    }
    kept
}

/// Extend each baseline polyline outward to the boundary of its enclosing blob.
/// Source: kraken/lib/segmentation.py:_extend_boundaries (simplified).
///
/// For each blob with area >= 6, trace its boundary. For each baseline, find the
/// containing blob and extrapolate its endpoints outward by up to 10 pixels along
/// the line direction, clamped to the blob's bounding extent.
fn extend_boundaries(lines: &[Vec<Point>], bin_bl_map: &Array2<f32>) -> Vec<Vec<Point>> {
    let (labels, n_cc) = label(bin_bl_map);
    if n_cc == 0 || lines.is_empty() {
        return lines.to_vec();
    }

    // Build per-component pixel sets and bounding boxes.
    let mut cc_pixels: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    let mut cc_bbox: HashMap<u32, (usize, usize, usize, usize)> = HashMap::new(); // min_y,min_x,max_y,max_x
    for y in 0..labels.dim().0 {
        for x in 0..labels.dim().1 {
            let cc = labels[[y, x]];
            if cc != 0 {
                cc_pixels.entry(cc).or_default().push((y, x));
                cc_bbox
                    .entry(cc)
                    .and_modify(|b| {
                        if y < b.0 { b.0 = y; }
                        if x < b.1 { b.1 = x; }
                        if y > b.2 { b.2 = y; }
                        if x > b.3 { b.3 = x; }
                    })
                    .or_insert((y, x, y, x));
            }
        }
    }

    // Map every pixel of a qualifying blob to its component id for containment lookup.
    let mut pixel_cc: HashMap<(usize, usize), u32> = HashMap::new();
    for (cc, pixels) in &cc_pixels {
        if pixels.len() >= 6 {
            for &p in pixels {
                pixel_cc.insert(p, *cc);
            }
        }
    }

    let mut result: Vec<Vec<Point>> = Vec::with_capacity(lines.len());
    for bl in lines {
        if bl.is_empty() {
            result.push(bl.clone());
            continue;
        }
        // Identify containing blob from the baseline's interior points.
        let mut found_cc: Option<u32> = None;
        for p in bl {
            let (y, x) = (p.y as usize, p.x as usize);
            if y < labels.dim().0 && x < labels.dim().1 {
                if let Some(&cc) = pixel_cc.get(&(y, x)) {
                    found_cc = Some(cc);
                    break;
                }
            }
        }

        let mut extended = bl.clone();
        if let Some(cc) = found_cc {
            if let Some(&(min_y, min_x, max_y, max_x)) = cc_bbox.get(&cc) {
                // Extrapolate the first endpoint backward along the initial segment.
                extrapolate(&mut extended, /*head=*/ true, min_y, min_x, max_y, max_x);
                // Extrapolate the last endpoint forward along the final segment.
                extrapolate(&mut extended, /*head=*/ false, min_y, min_x, max_y, max_x);
            }
        }
        result.push(extended);
    }
    result
}

/// Extrapolate a baseline endpoint outward by up to 10 pixels along the line's
/// end direction, clamped to the blob's bounding box.
fn extrapolate(
    bl: &mut Vec<Point>,
    head: bool,
    min_y: usize,
    min_x: usize,
    max_y: usize,
    max_x: usize,
) {
    let n = bl.len();
    if n < 2 {
        return;
    }
    let (anchor, support) = if head {
        (bl[0], bl[1])
    } else {
        (bl[n - 1], bl[n - 2])
    };
    let dx = anchor.x - support.x;
    let dy = anchor.y - support.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-9 {
        return;
    }
    let ux = dx / len;
    let uy = dy / len;

    let mut best = anchor;
    for step in 1..=10 {
        let cand = Point::new(anchor.x + ux * step as f64, anchor.y + uy * step as f64);
        let cx = cand.x.round() as isize;
        let cy = cand.y.round() as isize;
        if cy < min_y as isize || cy > max_y as isize || cx < min_x as isize || cx > max_x as isize {
            break;
        }
        best = cand;
    }
    if best != anchor {
        if head {
            bl.insert(0, best);
        } else {
            bl.push(best);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorize_regions_single_blob() {
        // Use a blob large enough that simplify(10.0) preserves corners.
        // A 5x5 block spans only ~4px — smaller than the tolerance — so use 30x30.
        let mut heatmap = Array2::<f32>::zeros((50, 50));
        for y in 10..40 {
            for x in 10..40 {
                heatmap[[y, x]] = 0.9;
            }
        }
        let regions = vectorize_regions(&heatmap, 0.5);
        assert_eq!(regions.len(), 1, "expected 1 region, got {}", regions.len());
        let region = &regions[0];
        assert!(region.exterior.len() >= 3, "region should have >=3 boundary points");
    }

    #[test]
    fn test_vectorize_regions_two_blobs() {
        let mut heatmap = Array2::<f32>::zeros((10, 20));
        for y in 2..7 {
            for x in 2..7 {
                heatmap[[y, x]] = 0.9;
            }
            for x in 12..17 {
                heatmap[[y, x]] = 0.9;
            }
        }
        let regions = vectorize_regions(&heatmap, 0.5);
        assert_eq!(regions.len(), 2, "expected 2 regions, got {}", regions.len());
    }

    #[test]
    fn test_vectorize_regions_empty() {
        let heatmap = Array2::<f32>::zeros((10, 10));
        let regions = vectorize_regions(&heatmap, 0.5);
        assert!(regions.is_empty(), "empty heatmap should produce no regions");
    }

    #[test]
    fn test_vectorize_lines_horizontal() {
        // A horizontal baseline: 3-channel stack (start_sep, end_sep, baseline)
        let mut im = ndarray::Array3::<f32>::zeros((3, 10, 20));
        // Baseline channel (index 2): bright horizontal line at y=5
        for x in 2..18 {
            im[[2, 5, x]] = 0.9;
        }
        // Start separator at left end
        im[[0, 5, 1]] = 0.9;
        // End separator at right end
        im[[1, 5, 18]] = 0.9;

        let lines = vectorize_lines(&im, 0.17, 5, "horizontal", 400);
        assert!(!lines.is_empty(), "should detect at least one line");
        let first = &lines[0];
        let xs: Vec<f64> = first.iter().map(|p| p.x).collect();
        let min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_x - min_x > 5.0, "line should span >5 pixels, got span {}", max_x - min_x);
    }

    #[test]
    fn test_vectorize_lines_empty() {
        let im = ndarray::Array3::<f32>::zeros((3, 10, 10));
        let lines = vectorize_lines(&im, 0.17, 5, "horizontal", 400);
        assert!(lines.is_empty(), "empty heatmap should produce no lines");
    }
}
