//! Moore-neighborhood boundary tracing.
//! Source: kraken/lib/segmentation.py:189-247

use crate::polygon::Point;

const MOORE_OFFSETS: [(i64, i64); 8] = [
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
    (0, 1), (1, 1), (1, 0), (1, -1),
];

/// Moore-neighborhood boundary tracing of a binary region.
///
/// Given the foreground pixel coordinates of a region (as `(y, x)` tuples),
/// returns the ordered boundary as a polyline of `Point`s in the original
/// coordinate space.
pub fn boundary_trace(coords: &[(usize, usize)]) -> Vec<Point> {
    if coords.len() < 2 {
        return Vec::new();
    }

    let min_y = coords.iter().map(|&(y, _)| y).min().unwrap();
    let min_x = coords.iter().map(|&(_, x)| x).min().unwrap();
    let max_y = coords.iter().map(|&(y, _)| y).max().unwrap();
    let max_x = coords.iter().map(|&(_, x)| x).max().unwrap();

    // Build a padded binary image (1px zero border).
    let bh = (max_y - min_y + 3) as usize;
    let bw = (max_x - min_x + 3) as usize;
    let mut binary = vec![false; bh * bw];
    let idx = |y: usize, x: usize| y * bw + x;
    for &(y, x) in coords {
        let by = y - min_y + 1;
        let bx = x - min_x + 1;
        binary[idx(by, bx)] = true;
    }

    // Find starting point: first foreground pixel with at least one neighbor.
    let start = (0..bh)
        .flat_map(|y| (0..bw).map(move |x| (y, x)))
        .find(|&(y, x)| {
            binary[idx(y, x)] && {
                let mut count = 0;
                for &(dy, dx) in &MOORE_OFFSETS {
                    let ny = y as i64 + dy;
                    let nx = x as i64 + dx;
                    if ny >= 0 && ny < bh as i64 && nx >= 0 && nx < bw as i64
                        && binary[idx(ny as usize, nx as usize)]
                    {
                        count += 1;
                    }
                }
                count > 0
            }
        });

    let (sy, sx) = match start {
        Some(s) => s,
        None => return Vec::new(),
    };

    // Determine the initial backtrack pixel.
    let backtrack_start = if !binary[idx((sy as i64 + 1) as usize, sx)]
        && !binary[idx((sy as i64 + 1) as usize, (sx as i64 - 1) as usize)]
    {
        (sy as i64 + 1, sx as i64)
    } else {
        (sy as i64, sx as i64 - 1)
    };

    let mut boundary: Vec<Point> = Vec::new();
    let mut current = (sy as i64, sx as i64);
    let mut backtrack = backtrack_start;

    loop {
        let neighbors = moore_neighborhood_sorted(current, backtrack);
        let mut found_idx: Option<usize> = None;
        for (i, (ny, nx)) in neighbors.iter().enumerate() {
            if *ny >= 0 && *ny < bh as i64 && *nx >= 0 && *nx < bw as i64
                && binary[idx(*ny as usize, *nx as usize)]
            {
                found_idx = Some(i);
                break;
            }
        }

        boundary.push(Point::new(
            (current.1 - 1 + min_x as i64) as f64,
            (current.0 - 1 + min_y as i64) as f64,
        ));

        match found_idx {
            Some(i) => {
                let prev_idx = if i == 0 { neighbors.len() - 1 } else { i - 1 };
                backtrack = neighbors[prev_idx];
                current = neighbors[i];
            }
            None => break,
        }

        if current == (sy as i64, sx as i64) && backtrack == backtrack_start {
            break;
        }
    }

    boundary
}

/// Returns the Moore neighborhood of `current`, rotated to begin at `backtrack`
/// and walked clockwise.
fn moore_neighborhood_sorted(current: (i64, i64), backtrack: (i64, i64)) -> Vec<(i64, i64)> {
    let offsets = [
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0), (1, -1),
    ];
    let neighbors: Vec<(i64, i64)> = offsets
        .iter()
        .map(|&(dy, dx)| (current.0 + dy, current.1 + dx))
        .collect();
    let bt_offset = (backtrack.0 - current.0, backtrack.1 - current.1);
    if let Some(start_idx) = offsets.iter().position(|&o| o == bt_offset) {
        let mut sorted = Vec::with_capacity(8);
        for i in 0..8 {
            sorted.push(neighbors[(start_idx + i) % 8]);
        }
        sorted
    } else {
        neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_boundary_trace_square() {
        // A 3x3 filled square
        let region = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let coords: Vec<(usize, usize)> = (1..4)
            .flat_map(|y| (1..4).map(move |x| (y, x)))
            .collect();
        let boundary = boundary_trace(&coords);
        assert!(
            boundary.len() >= 8,
            "expected >= 8 boundary points, got {}",
            boundary.len()
        );
        assert!(boundary.iter().any(|p| p.x == 1.0 && p.y == 1.0));
        // Reference image is illustrative; consume to avoid unused warning.
        let _ = &region;
    }

    #[test]
    fn test_boundary_trace_single_pixel() {
        let coords = vec![(5, 5)];
        let boundary = boundary_trace(&coords);
        assert!(
            boundary.is_empty(),
            "single pixel should produce no boundary"
        );
    }

    #[test]
    fn test_boundary_trace_horizontal_line() {
        // A 1x3 horizontal line
        let coords: Vec<(usize, usize)> = vec![(2, 2), (2, 3), (2, 4)];
        let boundary = boundary_trace(&coords);
        // Every coord should be on the boundary; line has 3 pixels.
        assert!(
            boundary.len() >= 3,
            "expected >= 3 boundary points, got {}",
            boundary.len()
        );
        for &(y, x) in &coords {
            assert!(
                boundary.iter().any(|p| p.x == x as f64 && p.y == y as f64),
                "pixel ({}, {}) missing from boundary",
                y,
                x
            );
        }
    }
}
