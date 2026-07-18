//! Reading order: heuristic bbox-overlap + topological sort.
//! Source: kraken/lib/segmentation.py:85-174 (_reading_order, topsort)
//!                              845-903 (polygonal_reading_order)

use crate::kraken::containers::{BaselineLine, Region};
use crate::kraken::polygon::{point_in_polygon, Point, Polygon};

/// A bounding box: (min_x, min_y, max_x, max_y).
type BBox = (f64, f64, f64, f64);

/// A slice: (y_range, x_range). Stored this way so the primary (vertical)
/// axis is first, matching kraken's `_reading_order` row/column convention.
type Slice = (std::ops::Range<f64>, std::ops::Range<f64>);

/// Compute the partial reading order from bounding boxes.
///
/// `order[i][j] == true` means line i comes before line j.
///
/// Source: kraken/lib/segmentation.py:85-130
pub fn reading_order_matrix(bounds: &[BBox], text_direction: &str) -> Vec<Vec<bool>> {
    let n = bounds.len();
    let mut order = vec![vec![false; n]; n];

    let slices: Vec<Slice> = bounds
        .iter()
        .map(|b| ((b.1)..(b.3), (b.0)..(b.2)))
        .collect();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let u = &slices[i];
            let v = &slices[j];
            if x_overlaps(u, v) {
                // Vertically stacked with x-overlap: the upper one first.
                if u.0.start < v.0.start {
                    order[i][j] = true;
                }
            } else if !separates_any(&slices, i, j) {
                // Side-by-side: left (or right for "rl") goes first.
                let is_left = left_of(u, v);
                if text_direction == "rl" {
                    if !is_left {
                        order[i][j] = true;
                    }
                } else if is_left {
                    order[i][j] = true;
                }
            }
        }
    }
    order
}

/// Whether two slices overlap horizontally (along the x axis, slice field 1).
fn x_overlaps(u: &Slice, v: &Slice) -> bool {
    u.1.start < v.1.end && u.1.end > v.1.start
}

/// Whether slice u lies entirely to the left of slice v.
fn left_of(u: &Slice, v: &Slice) -> bool {
    u.1.end < v.1.start
}

/// Whether any third slice separates i and j vertically (a "row" between them),
/// in which case they are not directly comparable side-by-side.
/// Source: kraken/lib/segmentation.py:115-130
fn separates_any(slices: &[Slice], i: usize, j: usize) -> bool {
    let u = &slices[i];
    let v = &slices[j];
    for (k, w) in slices.iter().enumerate() {
        if k == i || k == j {
            continue;
        }
        // Skip slices entirely above both.
        if w.0.end < u.0.start.min(v.0.start) {
            continue;
        }
        // Skip slices entirely below both.
        if w.0.start > u.0.end.max(v.0.end) {
            continue;
        }
        // w spans the vertical band between/over i and j and bridges their x
        // ranges -> it separates them.
        if w.1.start < u.1.end && w.1.end > v.1.start {
            return true;
        }
    }
    false
}

/// Topological sort of a partial order matrix.
///
/// `order[i][j] == true` means i must come before j. Returns a sequence of
/// indices respecting all constraints.
///
/// Source: kraken/lib/segmentation.py:154-174
pub fn topsort(order: &[Vec<bool>]) -> Vec<usize> {
    let n = order.len();
    let mut visited = vec![false; n];
    let mut result: Vec<usize> = Vec::new();

    fn visit(k: usize, order: &[Vec<bool>], visited: &mut [bool], result: &mut Vec<usize>) {
        if visited[k] {
            return;
        }
        visited[k] = true;
        // For every predecessor j (j before k), recurse first.
        for j in 0..order.len() {
            if order[j][k] {
                visit(j, order, visited, result);
            }
        }
        result.push(k);
    }

    for k in 0..n {
        visit(k, order, &mut visited, &mut result);
    }
    result
}

/// Convenience wrapper: compute the matrix then topologically sort it.
pub fn reading_order_from_bounds(bounds: &[BBox], text_direction: &str) -> Vec<usize> {
    let order = reading_order_matrix(bounds, text_direction);
    topsort(&order)
}

/// Compute reading order for [`BaselineLine`]s and [`Region`]s.
///
/// Lines are grouped by the region whose polygon contains their midpoint;
/// each region's lines are ordered internally, then regions and unassigned
/// lines are ordered at the top level. Returns indices into `lines`.
///
/// Source: kraken/lib/segmentation.py:845-903
pub fn polygonal_reading_order(
    lines: &[BaselineLine],
    regions: &[Region],
    text_direction: &str,
) -> Vec<usize> {
    let reg_polygons: Vec<Polygon> = regions
        .iter()
        .map(|r| Polygon::from_tuples(&r.boundary))
        .collect();

    let mut region_lines: Vec<Vec<(usize, BBox)>> = vec![Vec::new(); regions.len()];
    let mut unassigned: Vec<(usize, BBox)> = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        let bl_poly = Polygon::from_tuples(&line.baseline);
        let bbox = bl_poly.bounds();
        let mid = Point::new((bbox.0 + bbox.2) / 2.0, (bbox.1 + bbox.3) / 2.0);
        let mut in_region = false;
        for (reg_idx, reg_poly) in reg_polygons.iter().enumerate() {
            if point_in_polygon(&mid, reg_poly) {
                region_lines[reg_idx].push((line_idx, bbox));
                in_region = true;
                break;
            }
        }
        if !in_region {
            unassigned.push((line_idx, bbox));
        }
    }

    let mut intra_region_order: Vec<Vec<usize>> = vec![Vec::new(); regions.len()];
    let mut top_level_bounds: Vec<BBox> = Vec::new();
    let mut top_level_map: Vec<(String, usize)> = Vec::new();

    for (reg_idx, reg_lines) in region_lines.iter().enumerate() {
        if !reg_lines.is_empty() {
            let bounds: Vec<BBox> = reg_lines.iter().map(|(_, b)| *b).collect();
            let order = reading_order_from_bounds(&bounds, text_direction);
            intra_region_order[reg_idx] = order.iter().map(|&i| reg_lines[i].0).collect();
            let reg_bbox = reg_polygons[reg_idx].bounds();
            top_level_bounds.push(reg_bbox);
            top_level_map.push(("region".to_string(), reg_idx));
        }
    }

    for (line_idx, bbox) in &unassigned {
        top_level_bounds.push(*bbox);
        top_level_map.push(("line".to_string(), *line_idx));
    }

    let order = reading_order_from_bounds(&top_level_bounds, text_direction);

    let mut result: Vec<usize> = Vec::new();
    for i in order {
        let (kind, idx) = &top_level_map[i];
        match kind.as_str() {
            "line" => result.push(*idx),
            "region" => result.extend(&intra_region_order[*idx]),
            _ => {}
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reading_order_horizontal() {
        // Three lines stacked vertically, overlapping in x.
        let bounds = vec![
            (0.0, 0.0, 10.0, 2.0),   // top
            (0.0, 5.0, 10.0, 7.0),   // middle
            (0.0, 10.0, 10.0, 12.0), // bottom
        ];
        let order = reading_order_from_bounds(&bounds, "lr");
        assert_eq!(order, vec![0, 1, 2], "top-to-bottom order");
    }

    #[test]
    fn test_reading_order_two_columns() {
        let bounds = vec![
            (0.0, 0.0, 5.0, 2.0),   // left top
            (0.0, 5.0, 5.0, 7.0),   // left bottom
            (10.0, 0.0, 15.0, 2.0), // right top
            (10.0, 5.0, 15.0, 7.0), // right bottom
        ];
        let order = reading_order_from_bounds(&bounds, "lr");
        assert!(
            order.iter().position(|&i| i == 0) < order.iter().position(|&i| i == 2),
            "left before right"
        );
        assert!(
            order[0] == 0 || order[0] == 1,
            "first should be in left column"
        );
    }

    #[test]
    fn test_topsort_simple() {
        let order = vec![
            vec![false, true, false],
            vec![false, false, true],
            vec![false, false, false],
        ];
        let sorted = topsort(&order);
        assert_eq!(sorted, vec![0, 1, 2]);
    }
}
