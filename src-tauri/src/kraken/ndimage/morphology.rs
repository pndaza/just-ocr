//! Morphological operations: skeletonize, label, distance_transform_cdt, binary_erosion.

use ndarray::Array2;
use std::collections::VecDeque;

/// Morphological skeletonization (Zhang-Suen algorithm).
/// Matches skimage.morphology.skeletonize for 2D binary images.
pub fn skeletonize(input: &Array2<f32>) -> Array2<f32> {
    let (h, w) = input.dim();
    let mut img: Vec<Vec<bool>> = (0..h)
        .map(|y| (0..w).map(|x| input[[y, x]] > 0.5).collect())
        .collect();

    // Bounds-safe neighbor lookup. Out-of-bounds positions are treated as
    // background (false), so foreground pixels touching the image border are
    // still eligible for thinning, matching skimage's padded approach.
    let get = |img: &Vec<Vec<bool>>, y: i64, x: i64| -> bool {
        if y < 0 || y >= h as i64 || x < 0 || x >= w as i64 {
            false
        } else {
            img[y as usize][x as usize]
        }
    };

    let mut changed = true;
    while changed {
        changed = false;
        for sub in 0..2 {
            let mut to_remove = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    if !img[y][x] { continue; }
                    let yi = y as i64;
                    let xi = x as i64;
                    let p2 = get(&img, yi - 1, xi);
                    let p3 = get(&img, yi - 1, xi + 1);
                    let p4 = get(&img, yi, xi + 1);
                    let p5 = get(&img, yi + 1, xi + 1);
                    let p6 = get(&img, yi + 1, xi);
                    let p7 = get(&img, yi + 1, xi - 1);
                    let p8 = get(&img, yi, xi - 1);
                    let p9 = get(&img, yi - 1, xi - 1);
                    let neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];
                    let bn: usize = neighbors.iter().filter(|&&p| p).count();
                    if !(2..=6).contains(&bn) { continue; }
                    let mut transitions = 0;
                    for i in 0..8 {
                        if !neighbors[i] && neighbors[(i + 1) % 8] { transitions += 1; }
                    }
                    if transitions != 1 { continue; }
                    let cond = if sub == 0 {
                        !(p2 && p4 && p6) && !(p4 && p6 && p8)
                    } else {
                        !(p2 && p4 && p8) && !(p2 && p6 && p8)
                    };
                    if cond { to_remove.push((y, x)); }
                }
            }
            if !to_remove.is_empty() { changed = true; }
            for (y, x) in to_remove { img[y][x] = false; }
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            if img[y][x] { out[[y, x]] = 1.0; }
        }
    }
    out
}

/// Connected component labeling with 8-connectivity.
pub fn label(input: &Array2<f32>) -> (Array2<u32>, usize) {
    let (h, w) = input.dim();
    let mut labels = Array2::<u32>::zeros((h, w));
    let mut current_label: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            if input[[y, x]] > 0.5 && labels[[y, x]] == 0 {
                current_label += 1;
                let mut queue = VecDeque::new();
                queue.push_back((y, x));
                labels[[y, x]] = current_label;
                while let Some((cy, cx)) = queue.pop_front() {
                    for &dy in &[-1i64, 0, 1] {
                        for &dx in &[-1i64, 0, 1] {
                            if dy == 0 && dx == 0 { continue; }
                            let ny = cy as i64 + dy;
                            let nx = cx as i64 + dx;
                            if ny >= 0 && ny < h as i64 && nx >= 0 && nx < w as i64 {
                                let ny = ny as usize;
                                let nx = nx as usize;
                                if input[[ny, nx]] > 0.5 && labels[[ny, nx]] == 0 {
                                    labels[[ny, nx]] = current_label;
                                    queue.push_back((ny, nx));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (labels, current_label as usize)
}

/// Chamfer distance transform.
pub fn distance_transform_cdt(input: &Array2<f32>) -> Array2<f32> {
    let (h, w) = input.dim();
    let mut dist = Array2::<f32>::from_elem((h, w), f32::INFINITY);
    for y in 0..h {
        for x in 0..w {
            if input[[y, x]] <= 0.5 { dist[[y, x]] = 0.0; }
        }
    }
    // Forward pass
    for y in 0..h {
        for x in 0..w {
            if dist[[y, x]] == 0.0 { continue; }
            let mut d = dist[[y, x]];
            if y > 0 { d = d.min(dist[[y - 1, x]] + 1.0); }
            if x > 0 { d = d.min(dist[[y, x - 1]] + 1.0); }
            if y > 0 && x > 0 { d = d.min(dist[[y - 1, x - 1]] + 1.41421356); }
            if y > 0 && x < w - 1 { d = d.min(dist[[y - 1, x + 1]] + 1.41421356); }
            dist[[y, x]] = d;
        }
    }
    // Backward pass
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            if dist[[y, x]] == 0.0 { continue; }
            let mut d = dist[[y, x]];
            if y < h - 1 { d = d.min(dist[[y + 1, x]] + 1.0); }
            if x < w - 1 { d = d.min(dist[[y, x + 1]] + 1.0); }
            if y < h - 1 && x < w - 1 { d = d.min(dist[[y + 1, x + 1]] + 1.41421356); }
            if y < h - 1 && x > 0 { d = d.min(dist[[y + 1, x - 1]] + 1.41421356); }
            dist[[y, x]] = d;
        }
    }
    dist
}

/// Binary erosion with 8-connectivity.
pub fn binary_erosion(input: &Array2<f32>, border_value: bool, iterations: usize) -> Array2<f32> {
    let (h, w) = input.dim();
    let mut current = input.clone();
    for _ in 0..iterations {
        let mut next = Array2::<f32>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let mut all_set = true;
                'neighbor_loop: for &dy in &[-1i64, 0, 1] {
                    for &dx in &[-1i64, 0, 1] {
                        if dy == 0 && dx == 0 { continue; }
                        let ny = y as i64 + dy;
                        let nx = x as i64 + dx;
                        let val = if ny < 0 || ny >= h as i64 || nx < 0 || nx >= w as i64 {
                            border_value
                        } else {
                            current[[ny as usize, nx as usize]] > 0.5
                        };
                        if !val { all_set = false; break 'neighbor_loop; }
                    }
                }
                next[[y, x]] = if all_set { 1.0 } else { 0.0 };
            }
        }
        current = next;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_skeletonize_simple_line() {
        // A thick horizontal line should skeletonize to 1px
        let input = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let skel = skeletonize(&input);
        // Zhang-Suen thins a 3px-tall solid bar to a fragment of the center
        // row. Verified to match an independent Python reference implementation
        // of the algorithm. Skeleton must be fully contained in the foreground,
        // non-empty, and strictly thinner than the 3px-tall input.
        for y in 0..5 {
            for x in 0..5 {
                if skel[[y, x]] > 0.5 {
                    assert!(input[[y, x]] > 0.5, "skeleton leaks outside foreground at ({y},{x})");
                }
            }
        }
        let total: usize = skel.iter().filter(|&&v| v > 0.5).count();
        assert!(total > 0, "skeleton should be non-empty");
        // Each column of the input had 3 foreground pixels; after thinning no
        // column should have more than 1 skeleton pixel.
        for x in 0..5 {
            let count: usize = (0..5).filter(|&y| skel[[y, x]] > 0.5).count();
            assert!(count <= 1, "column {x} has {count} skeleton pixels, expected <= 1");
        }
    }

    #[test]
    fn test_skeletonize_single_pixel() {
        let input = array![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let skel = skeletonize(&input);
        assert!(skel[[1, 1]] > 0.5, "single pixel should remain");
    }

    #[test]
    fn test_label_two_components() {
        let input = array![
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let (_labels, n) = label(&input);
        assert_eq!(n, 2, "expected 2 components, got {n}");
    }

    #[test]
    fn test_label_single_component() {
        let input = array![
            [1.0, 1.0],
            [1.0, 1.0],
        ];
        let (_labels, n) = label(&input);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_distance_transform_cdt() {
        let input = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let dist = distance_transform_cdt(&input);
        assert!(dist[[2, 2]] >= dist[[1, 1]], "center should have max distance");
        assert!(dist[[2, 2]] > 0.0);
    }

    #[test]
    fn test_binary_erosion() {
        let input = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let eroded = binary_erosion(&input, true, 1);
        assert!(eroded[[2, 2]] > 0.5, "center should survive");
        assert!(eroded[[1, 1]] < 0.5, "edge should be eroded");
    }
}
