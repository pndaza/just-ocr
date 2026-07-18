//! Seam-carving boundary polygon computation.
//! Source: kraken/lib/segmentation.py:747-842 (calculate_polygonal_environment),
//!         560-635 (_calc_seam), 683-744 (_calc_roi)

use crate::ndimage::morphology::{distance_transform_cdt, binary_erosion};
use crate::polygon::{Polygon, Point, simplify};
use ndarray::Array2;

const MASK_VAL: f32 = 99999.0;

/// Dynamic-programming seam through a feature patch, biased by distance from
/// the baseline. Returns the minimum-cost vertical-path (left-to-right) seam
/// as a list of points in image coordinates.
///
/// Source: kraken/lib/segmentation.py:560-635 (_calc_seam)
pub fn calc_seam(
    baseline: &[Point],
    polygon: &[Point],
    angle: f64,
    im_feats: &Array2<f32>,
    bias: f64,
) -> Vec<Point> {
    // Bounding box of the polygon, clamped to the image and to non-negative.
    let c_min = polygon.iter().map(|p| p.x).fold(f64::INFINITY, f64::min).max(0.0) as usize;
    let c_max = polygon.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) as usize;
    let r_min = polygon.iter().map(|p| p.y).fold(f64::INFINITY, f64::min).max(0.0) as usize;
    let r_max = polygon.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) as usize;
    let c_max = c_max.min(im_feats.dim().1 - 1);
    let r_max = r_max.min(im_feats.dim().0 - 1);

    let patch_w = c_max - c_min + 2;
    let patch_h = r_max - r_min + 2;
    if patch_w < 2 || patch_h < 2 {
        return Vec::new();
    }

    // Extract patch from im_feats.
    let mut patch = Array2::<f32>::zeros((patch_h, patch_w));
    for y in 0..patch_h {
        for x in 0..patch_w {
            let fy = r_min + y;
            let fx = c_min + x;
            if fy < im_feats.dim().0 && fx < im_feats.dim().1 {
                patch[[y, x]] = im_feats[[fy, fx]];
            }
        }
    }

    // Distance bias from baseline: draw the baseline onto a mask (0 on the
    // line, 1 elsewhere) then take the distance transform so the cost grows
    // with distance from the baseline.
    let mut mask = Array2::<f32>::ones((patch_h, patch_w));
    for w in baseline.windows(2) {
        let x0 = (w[0].x as i64 - c_min as i64).max(0) as usize;
        let y0 = (w[0].y as i64 - r_min as i64).max(0) as usize;
        let x1 = (w[1].x as i64 - c_min as i64).max(0) as usize;
        let y1 = (w[1].y as i64 - r_min as i64).max(0) as usize;
        draw_line_on_mask(&mut mask, x0, y0, x1, y1);
    }
    let dist_bias = distance_transform_cdt(&mask);

    // Polygon mask: rasterize the polygon via scanline fill (much faster than
    // per-pixel point_in_polygon). inside_mask[y][x] = 1.0 for pixels INSIDE
    // the polygon, 0.0 outside. A subsequent erosion shrinks the valid region
    // so the seam cannot graze the polygon edge.
    let mut inside_mask = Array2::<f32>::zeros((patch_h, patch_w));
    rasterize_polygon_fill(
        &mut inside_mask,
        polygon,
        c_min as f64,
        r_min as f64,
        patch_w,
        patch_h,
    );
    let eroded = binary_erosion(&inside_mask, true, 2);
    let mask_outside = eroded.mapv(|v| if v < 0.5 { 1.0 } else { 0.0 });

    // Mean of the in-polygon patch features, used to scale the distance bias
    // into the same magnitude range as the features.
    let patch_mean: f64 = {
        let mut sum = 0.0f64;
        let mut count = 0u64;
        for y in 0..patch_h {
            for x in 0..patch_w {
                if mask_outside[[y, x]] < 0.5 && patch[[y, x]] != MASK_VAL {
                    sum += patch[[y, x]] as f64;
                    count += 1;
                }
            }
        }
        if count > 0 {
            sum / count as f64
        } else {
            1.0
        }
    };

    for y in 0..patch_h {
        for x in 0..patch_w {
            if mask_outside[[y, x]] > 0.5 {
                patch[[y, x]] = MASK_VAL;
            }
            patch[[y, x]] += dist_bias[[y, x]] * (patch_mean / bias) as f32;
        }
    }

    // Kraken implementation levels the baseline before seam carving. The DP
    // traverses columns, so doing this in image coordinates makes sloped lines
    // appear to wander or become artificially tight. Kraken uses a nearest
    // neighbour affine warp here (`order=0`); rotate the patch and map the
    // resulting seam back after the DP.
    //
    // Kraken also scales the x-axis of the patch so the rotated width never
    // exceeds ~600 px (`scale = min(1.0, 600 / width)`). This caps the DP cost
    // for wide lines and — critically — keeps the seam resolution comparable
    // across line widths, so the subsequent simplify(5) doesn't over-collapse
    // narrow features. Without it a 1100 px-wide line yields ~1100 seam points
    // that simplify(5) flattens to a handful of vertices; with scaling the
    // same line yields ~600 points that retain detail.
    let scale = if patch_w > 600 { 600.0 / patch_w as f64 } else { 1.0 };
    let (patch, rotation) = rotate_patch(&patch, angle, scale, MASK_VAL);

    // Pad the patch with an infinity border so the seam is constrained to the
    // interior rows during the DP traversal. Note: `patch` may have been
    // x-scaled by rotate_patch, so use its actual post-rotation dimensions.
    let (patch_h_rot, patch_w_rot) = patch.dim();
    let padded_h = patch_h_rot + 2;
    let padded_w = patch_w_rot;
    let mut padded = Array2::<f32>::from_elem((padded_h, padded_w), f32::INFINITY);
    for y in 0..patch_h_rot {
        for x in 0..patch_w_rot {
            padded[[y + 1, x]] = patch[[y, x]];
        }
    }

    let (r, c) = padded.dim();
    if c < 2 {
        return Vec::new();
    }
    let mut dp = padded.clone();
    let mut backtrack = Array2::<i64>::zeros((r, c));

    // Forward DP: for each column advance one step, choosing the cheapest
    // predecessor among the three vertical neighbours in the prior column.
    for i in 0..(c - 1) {
        for j in 1..(r - 1) {
            let mut min_val = dp[[j - 1, i]];
            let mut min_idx = j as i64 - 1;
            if dp[[j, i]] < min_val {
                min_val = dp[[j, i]];
                min_idx = j as i64;
            }
            if dp[[j + 1, i]] < min_val {
                min_val = dp[[j + 1, i]];
                min_idx = j as i64 + 1;
            }
            dp[[j, i + 1]] += min_val;
            backtrack[[j, i]] = min_idx - j as i64;
        }
    }

    // Backtrack from the cheapest row in the last column.
    let mut seam: Vec<Point> = Vec::new();
    let last_col = c - 1;
    let mut j = (1..(r - 1))
        .min_by(|&a, &b| {
            dp[[a, last_col]].partial_cmp(&dp[[b, last_col]]).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(1);

    for i in (0..c).rev() {
        seam.push(Point::new(i as f64, (j as i64 - 1) as f64));
        if i > 0 {
            j = ((j as i64 + backtrack[[j, i - 1]]) as usize).max(1).min(r - 2);
        }
    }
    seam.reverse();

    // Clip seam row coordinates to mean ± 1 std to suppress outlier excursions
    // where the DP path drifted toward a high-energy edge far from the line.
    // Source: kraken/lib/segmentation.py:624-626
    if !seam.is_empty() {
        let n = seam.len() as f64;
        let mean_y: f64 = seam.iter().map(|p| p.y).sum::<f64>() / n;
        let var: f64 = seam.iter().map(|p| (p.y - mean_y).powi(2)).sum::<f64>() / n;
        let std_y = var.sqrt();
        let lo = mean_y - std_y;
        let hi = mean_y + std_y;
        for p in seam.iter_mut() {
            if p.y < lo {
                p.y = lo;
            } else if p.y > hi {
                p.y = hi;
            }
        }
    }

    // Undo the patch rotation and translate back to image coordinates. Points
    // outside the original (eroded) polygon are discarded just as in Python.
    let seam: Vec<Point> = seam
        .into_iter()
        .map(|p| rotation.inverse(p))
        .filter(|p| {
            let x = p.x.round() as isize;
            let y = p.y.round() as isize;
            x >= 0
                && y >= 0
                && (x as usize) < mask_outside.dim().1
                && (y as usize) < mask_outside.dim().0
                && mask_outside[[y as usize, x as usize]] < 0.5
        })
        .map(|p| Point::new(p.x + c_min as f64, p.y + r_min as f64))
        .collect();
    seam
}

#[derive(Clone, Copy)]
struct PatchRotation {
    angle: f64,
    scale: f64,
    tx: f64,
    ty: f64,
}

impl PatchRotation {
    /// Map a point in the rotated/scaled patch back into the original patch.
    fn inverse(self, p: Point) -> Point {
        let (sin, cos) = self.angle.sin_cos();
        let x = p.x - self.tx;
        let y = p.y - self.ty;
        // Undo rotation, then undo the x-axis scale.
        let sx = cos * x - sin * y;
        let sy = sin * x + cos * y;
        Point::new(sx / self.scale, sy)
    }
}

/// Rotate an image patch into baseline-aligned coordinates using the same
/// nearest-neighbour convention as Kraken's scipy/skimage warp. Coordinates
/// are kept in the patch's local (x, y) space.
fn rotate_patch(image: &Array2<f32>, angle: f64, scale: f64, cval: f32) -> (Array2<f32>, PatchRotation) {
    let (h, w) = image.dim();
    let (sin, cos) = angle.sin_cos();
    // Forward map (matching skimage's AffineTransform with rotation=angle and
    // scale=(1/scale, 1)): first scale x, then rotate by -angle.
    let corners = [
        (0.0, 0.0),
        (0.0, (h - 1) as f64),
        ((w - 1) as f64, (h - 1) as f64),
        ((w - 1) as f64, 0.0),
    ];
    let transformed: Vec<(f64, f64)> = corners
        .iter()
        .map(|&(x, y)| {
            // Scale the x-axis (compress for wide lines), then rotate.
            let xs = x * scale;
            (cos * xs + sin * y, -sin * xs + cos * y)
        })
        .collect();
    let min_x = transformed.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let min_y = transformed.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_x = transformed.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let max_y = transformed.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let out_w = (max_x - min_x).round() as usize + 1;
    let out_h = (max_y - min_y).round() as usize + 1;
    let rotation = PatchRotation { angle, scale, tx: -min_x, ty: -min_y };
    let mut output = Array2::from_elem((out_h, out_w), cval);

    for y in 0..out_h {
        for x in 0..out_w {
            // Inverse map: undo translation, undo rotation, undo x-scale.
            let qx = x as f64 - rotation.tx;
            let qy = y as f64 - rotation.ty;
            let sx = cos * qx - sin * qy;
            let sy = sin * qx + cos * qy;
            // sx is in scaled space; divide by scale to return to source pixels.
            let src_x = sx / scale;
            let ix = src_x.round() as isize;
            let iy = sy.round() as isize;
            if ix >= 0 && iy >= 0 && (ix as usize) < w && (iy as usize) < h {
                output[[y, x]] = image[[iy as usize, ix as usize]];
            }
        }
    }
    (output, rotation)
}

/// Bresenham line drawing: sets every pixel on the line from (x0,y0) to
/// (x1,y1) to 0.0 on the mask. Coordinates outside the mask are skipped.
fn draw_line_on_mask(mask: &mut Array2<f32>, x0: usize, y0: usize, x1: usize, y1: usize) {
    let (h, w) = mask.dim();
    let dx = (x1 as i64 - x0 as i64).abs();
    let dy = (y1 as i64 - y0 as i64).abs();
    let sx = if x0 < x1 { 1i64 } else { -1 };
    let sy = if y0 < y1 { 1i64 } else { -1 };
    let mut err = dx - dy;
    let (mut x, mut y) = (x0 as i64, y0 as i64);
    loop {
        if x >= 0 && x < w as i64 && y >= 0 && y < h as i64 {
            mask[[y as usize, x as usize]] = 0.0;
        }
        if x == x1 as i64 && y == y1 as i64 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

/// Scanline polygon rasterization: fills `mask` with 1.0 for pixels inside
/// the polygon, 0.0 outside. Polygon vertices are in image coordinates;
/// `origin_x`/`origin_y` is the patch origin (top-left) to subtract.
///
/// Uses the even-odd rule via scanline-edge intersection, matching the
/// ray-casting point_in_polygon test it replaces. This is O(H·W + E·H) where
/// E is the edge count, vs O(H·W·E) for per-pixel testing.
fn rasterize_polygon_fill(
    mask: &mut Array2<f32>,
    polygon: &[Point],
    origin_x: f64,
    origin_y: f64,
    w: usize,
    h: usize,
) {
    if polygon.len() < 3 {
        return;
    }
    // Translate vertices into patch-local coordinates.
    let local: Vec<(f64, f64)> = polygon
        .iter()
        .map(|p| (p.x - origin_x, p.y - origin_y))
        .collect();
    let n = local.len();

    for py in 0..h {
        let yc = py as f64 + 0.5; // sample at pixel center
        // Collect x-intersections of polygon edges with scanline y=yc.
        let mut xs: Vec<f64> = Vec::with_capacity(n);
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = local[i];
            let (xj, yj) = local[j];
            if (yi > yc) != (yj > yc) {
                // Edge crosses this scanline; compute the x intersection.
                let t = (yc - yi) / (yj - yi);
                xs.push(xi + t * (xj - xi));
            }
            j = i;
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Fill spans between pairs (even-odd rule).
        let mut row = mask.slice_mut(ndarray::s![py, ..]);
        let row = row.as_slice_mut().unwrap();
        let mut k = 0;
        while k + 1 < xs.len() {
            let x_start = xs[k].max(0.0);
            let x_end = xs[k + 1].min(w as f64);
            if x_end > x_start {
                let px_start = x_start.floor() as usize;
                let px_end = (x_end.ceil() as usize).min(w);
                for px in px_start..px_end {
                    row[px] = 1.0;
                }
            }
            k += 2;
        }
    }
}

/// Compute seam-carved boundary polygons for a set of baselines.
///
/// For each baseline this computes an upper and a lower boundary by carving a
/// minimum-cost seam through the feature image inside a region of interest
/// derived from the baseline's principal direction. The returned polygon is
/// `start + upper_seam + end + bottom_seam_reversed`.
///
/// Source: kraken/lib/segmentation.py:747-842 (calculate_polygonal_environment)
pub fn calculate_polygonal_environment(
    baselines: &[Vec<Point>],
    im_feats: &Array2<f32>,
    suppl_obj: &[Vec<Point>],
    topline: bool,
    bounds: (f64, f64),
) -> Vec<Option<Polygon>> {
    let mut polygons: Vec<Option<Polygon>> = Vec::with_capacity(baselines.len());

    for (idx, line) in baselines.iter().enumerate() {
        if line.len() < 2 {
            polygons.push(None);
            continue;
        }

        let dir_vec = principal_direction(line);
        let angle = dir_vec[1].atan2(dir_vec[0]);

        // All other baselines plus supplied objects act as obstacles.
        let mut suppl: Vec<&Vec<Point>> = Vec::new();
        for (i, bl) in baselines.iter().enumerate() {
            if i != idx {
                suppl.push(bl);
            }
        }
        for obj in suppl_obj {
            suppl.push(obj);
        }

        let (env_up, env_bottom) = calc_roi(line, bounds, &suppl, &dir_vec);
        if env_up.is_empty() || env_bottom.is_empty() {
            polygons.push(None);
            continue;
        }

        // Polygon enclosing the upper / lower halves of the line's ROI.
        let mut upper_polygon: Vec<Point> = line.to_vec();
        upper_polygon.extend(env_up.iter().rev());
        let mut bottom_polygon: Vec<Point> = line.to_vec();
        bottom_polygon.extend(env_bottom.iter().rev());

        // Slightly shifted baseline so the seam is biased to stay near the
        // line side being carved. Source: segmentation.py:813-814.
        let offset = 8.0;
        let offset_baseline: Vec<Point> = if topline {
            line.iter().map(|p| Point::new(p.x, p.y + offset)).collect()
        } else {
            line.iter().map(|p| Point::new(p.x, p.y - offset)).collect()
        };

        let mut upper_offset_polygon: Vec<Point> = offset_baseline.clone();
        upper_offset_polygon.extend(env_up.iter().rev());
        let mut bottom_offset_polygon: Vec<Point> = offset_baseline.clone();
        bottom_offset_polygon.extend(env_bottom.iter().rev());

        let (upper_seam, bottom_seam) = if topline {
            (
                calc_seam(line, &upper_polygon, angle, im_feats, 150.0),
                calc_seam(&offset_baseline, &bottom_offset_polygon, angle, im_feats, 150.0),
            )
        } else {
            (
                calc_seam(&offset_baseline, &upper_offset_polygon, angle, im_feats, 150.0),
                calc_seam(line, &bottom_polygon, angle, im_feats, 150.0),
            )
        };

        let upper_simplified = simplify(&upper_seam, 5.0);
        let bottom_simplified = simplify(&bottom_seam, 5.0);

        // Push each seam `offset / 2` pixels perpendicular outward (away from
        // the baseline). The raw seam hugs the ink edge and can clip the
        // outermost strokes; this 4px shift leaves a safety margin so the OCR
        // crop captures a thin border of background around the glyphs.
        // Source: segmentation.py:660-669 (kraken uses shapely's
        // parallel_offset, which also densifies curves; we use a plain
        // per-vertex normal shift — same safety margin, sparser output).
        let margin = offset / 2.0;
        let upper_offsetted = parallel_shift(&upper_simplified, margin, ShiftSide::OutwardUpper);
        let bottom_offsetted = parallel_shift(&bottom_simplified, margin, ShiftSide::OutwardLower);

        let start = line[0];
        let end = line[line.len() - 1];
        let mut polygon_points: Vec<Point> = Vec::new();
        polygon_points.push(start);
        polygon_points.extend(upper_offsetted);
        polygon_points.push(end);
        polygon_points.extend(bottom_offsetted.iter().rev());

        // Clip the seam-derived polygon to the ROI envelope.
        // Source: segmentation.py:648,679
        let polygon_points = clip_to_roi(polygon_points, &upper_polygon, &bottom_polygon);

        if polygon_points.len() >= 3 {
            polygons.push(Some(Polygon::new(polygon_points)));
        } else {
            polygons.push(None);
        }
    }
    polygons
}

/// Which side of a seam to shift toward.
enum ShiftSide {
    /// Upper edge: shift perpendicular such that an LTR baseline's top edge
    /// moves further up (smaller y) — away from the baseline.
    OutwardUpper,
    /// Lower edge: shift further down (larger y) — away from the baseline.
    OutwardLower,
}

/// Shift every vertex of `seam` by `dist` pixels along the polyline's normal,
/// on the chosen side. For each interior vertex the normal is the average of
/// its two adjacent edge normals; endpoints use their single edge's normal.
///
/// This is a plain per-vertex parallel offset, matching the *intent* of
/// shapely's `LineString.parallel_offset(dist, side)` (a safety margin away
/// from the seam) without shapely's curve densification. The seam is nearly
/// straight for most lines, so per-vertex shifting is visually equivalent.
fn parallel_shift(seam: &[Point], dist: f64, side: ShiftSide) -> Vec<Point> {
    let n = seam.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return seam.to_vec();
    }
    // Sign of the perpendicular shift along the (dx, dy) → (dy, -dx) normal.
    // For an LTR baseline going +x, the upper edge wants the normal pointing
    // -y (up), the lower edge wants +y (down). Normal (dy, -dx) points up for
    // +x travel, so OutwardUpper keeps it as-is, OutwardLower negates it.
    let sign = match side {
        ShiftSide::OutwardUpper => 1.0,
        ShiftSide::OutwardLower => -1.0,
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Average direction of the two adjacent edges (one for endpoints).
        let (ax, ay) = if i > 0 {
            (seam[i].x - seam[i - 1].x, seam[i].y - seam[i - 1].y)
        } else {
            (seam[i + 1].x - seam[i].x, seam[i + 1].y - seam[i].y)
        };
        let (bx, by) = if i + 1 < n {
            (seam[i + 1].x - seam[i].x, seam[i + 1].y - seam[i].y)
        } else {
            (seam[i].x - seam[i - 1].x, seam[i].y - seam[i - 1].y)
        };
        let dx = ax + bx;
        let dy = ay + by;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            out.push(seam[i]);
            continue;
        }
        // Unit normal: rotate direction by 90°.
        let nx = dy / len;
        let ny = -dx / len;
        out.push(Point::new(
            seam[i].x + sign * dist * nx,
            seam[i].y + sign * dist * ny,
        ));
    }
    out
}

/// Clip a polygon to the union of two ROI polygons (upper and lower envelopes
/// joined at the baseline). Uses `geo`'s exact boolean intersection, matching
/// Shapely's `roi_polygon.intersection(polygon)`. Falls back to the unclipped
/// polygon if the intersection is empty or degenerate (e.g. due to numerical
/// edge cases in the envelope).
/// Source: kraken/lib/segmentation.py:648,679
fn clip_to_roi(
    polygon: Vec<Point>,
    upper_polygon: &[Point],
    bottom_polygon: &[Point],
) -> Vec<Point> {
    use geo::{BooleanOps, LineString, Polygon as GeoPolygon};

    if polygon.len() < 3 {
        return polygon;
    }

    let to_geo = |pts: &[Point]| {
        let coords: Vec<geo::Coord> = pts
            .iter()
            .map(|p| geo::Coord { x: p.x, y: p.y })
            .collect();
        GeoPolygon::new(LineString::from(coords), vec![])
    };

    let roi = to_geo(upper_polygon).union(&to_geo(bottom_polygon));
    let seam_poly = to_geo(&polygon);

    let clipped = roi.intersection(&seam_poly);

    // Take the largest exterior ring from the (possibly multi) result.
    let mut best: Vec<geo::Coord> = Vec::new();
    for poly in clipped.into_iter() {
        let exterior: Vec<geo::Coord> = poly.exterior().coords().copied().collect();
        if exterior.len() > best.len() {
            best = exterior;
        }
    }

    if best.len() < 3 {
        return polygon; // fallback: keep unclipped
    }

    let mut result: Vec<Point> = best.into_iter().map(|c| Point::new(c.x, c.y)).collect();
    // geo closes rings (first==last); drop the duplicate endpoint that
    // imageproc's polygon rasterizer rejects.
    if result.len() > 1 && result[0] == result[result.len() - 1] {
        result.pop();
    }
    result
}

/// Principal direction of a polyline: the magnitude-weighted average of its
/// consecutive segment vectors, normalized. This matches kraken's
/// `np.mean(np.diff(line.T) * lengths / lengths.sum(), axis=1)` and is more
/// robust than least-squares slope for curved or piecewise baselines.
/// Source: kraken/lib/segmentation.py:819-821
fn principal_direction(line: &[Point]) -> [f64; 2] {
    if line.len() < 2 {
        return [1.0, 0.0];
    }
    let mut sum_dx = 0.0f64;
    let mut sum_dy = 0.0f64;
    let mut total_len = 0.0f64;
    for w in line.windows(2) {
        let dx = w[1].x - w[0].x;
        let dy = w[1].y - w[0].y;
        let len = (dx * dx + dy * dy).sqrt();
        if len > 1e-10 {
            sum_dx += dx * len;
            sum_dy += dy * len;
            total_len += len;
        }
    }
    if total_len < 1e-10 {
        return [1.0, 0.0];
    }
    let px = sum_dx / total_len;
    let py = sum_dy / total_len;
    let norm = (px * px + py * py).sqrt();
    if norm < 1e-10 {
        return [1.0, 0.0];
    }
    [px / norm, py / norm]
}

/// Compute the upper and lower region-of-interest envelopes for a baseline.
///
/// The baseline is resampled at 10px intervals, then from each sample a ray
/// is cast in both orthogonal directions until it reaches the image bounds
/// or crosses a supplementary obstacle (neighboring baselines/regions).
/// The hit points form the two envelopes.
///
/// Source: kraken/lib/segmentation.py:683-744 (_calc_roi)
fn calc_roi(
    line: &[Point],
    bounds: (f64, f64),
    suppl: &[&Vec<Point>],
    p_dir: &[f64; 2],
) -> (Vec<Point>, Vec<Point>) {
    // Resample the baseline at 10px arclength intervals.
    let mut ip_line: Vec<Point> = vec![line[0]];
    let total_length: f64 = line
        .windows(2)
        .map(|w| {
            let dx = w[1].x - w[0].x;
            let dy = w[1].y - w[0].y;
            (dx * dx + dy * dy).sqrt()
        })
        .sum();

    let mut dist = 10.0f64;
    while dist < total_length {
        let mut remaining = dist;
        for w in line.windows(2) {
            let dx = w[1].x - w[0].x;
            let dy = w[1].y - w[0].y;
            let seg_len = (dx * dx + dy * dy).sqrt();
            if remaining <= seg_len {
                let t = remaining / seg_len.max(1e-6);
                ip_line.push(Point::new(w[0].x + t * dx, w[0].y + t * dy));
                break;
            }
            remaining -= seg_len;
        }
        dist += 10.0;
    }
    ip_line.push(line[line.len() - 1]);

    let (max_x, max_y) = bounds;
    // Orthogonal directions matching kraken's (p_dir * (-1, 1))[::-1] (upper)
    // and (p_dir * (1, -1))[::-1] (bottom). For a horizontal baseline
    // (p_dir=[1,0]) upper = [0,-1] (toward smaller y, i.e. visually up) and
    // bottom = [0,1] (toward larger y). Source: segmentation.py:696-697
    let ortho_up = [p_dir[1], -p_dir[0]];
    let ortho_down = [-p_dir[1], p_dir[0]];

    let mut env_up: Vec<Point> = Vec::new();
    let mut env_bottom: Vec<Point> = Vec::new();
    for pt in &ip_line {
        env_up.push(cast_ray(pt, &ortho_up, max_x, max_y, suppl));
        env_bottom.push(cast_ray(pt, &ortho_down, max_x, max_y, suppl));
    }
    (env_up, env_bottom)
}

/// Cast a ray from `pt` in direction `dir` until it hits the image bounds
/// or the nearest supplementary obstacle segment. Returns the hit point.
///
/// Obstacle segments are treated as 1px-buffered (pulling the hit back by 1px),
/// matching kraken's `unary_union(...).buffer(1)`. Source: segmentation.py:712-713
fn cast_ray(
    pt: &Point,
    dir: &[f64; 2],
    max_x: f64,
    max_y: f64,
    suppl: &[&Vec<Point>],
) -> Point {
    let mut t_max = f64::INFINITY;
    if dir[0] > 0.0 {
        t_max = t_max.min((max_x - pt.x) / dir[0]);
    } else if dir[0] < 0.0 {
        t_max = t_max.min(-pt.x / dir[0]);
    }
    if dir[1] > 0.0 {
        t_max = t_max.min((max_y - pt.y) / dir[1]);
    } else if dir[1] < 0.0 {
        t_max = t_max.min(-pt.y / dir[1]);
    }

    let far = t_max.max(1.0) * 2.0;
    let ray_end = Point::new(pt.x + dir[0] * far, pt.y + dir[1] * far);
    let ray_len_sq = (dir[0] * far).powi(2) + (dir[1] * far).powi(2);
    let ray_len = ray_len_sq.sqrt();
    for obj in suppl {
        for w in obj.windows(2) {
            if let Some(t_frac) = ray_segment_intersection(pt, &ray_end, &w[0], &w[1]) {
                let t_dist = (t_frac * ray_len - 1.0).max(0.0);
                if t_dist > 1e-6 && t_dist < t_max {
                    t_max = t_dist;
                }
            }
        }
    }

    Point::new(pt.x + dir[0] * t_max, pt.y + dir[1] * t_max)
}

/// Ray-segment intersection. Returns t in [0,1] — the fraction of the ray
/// (from ray_start to ray_end) at which it crosses the segment — or None.
fn ray_segment_intersection(
    ray_start: &Point,
    ray_end: &Point,
    seg_start: &Point,
    seg_end: &Point,
) -> Option<f64> {
    let r_dx = ray_end.x - ray_start.x;
    let r_dy = ray_end.y - ray_start.y;
    let s_dx = seg_end.x - seg_start.x;
    let s_dy = seg_end.y - seg_start.y;

    let denom = r_dx * s_dy - r_dy * s_dx;
    if denom.abs() < 1e-10 {
        return None; // parallel
    }

    let dx = seg_start.x - ray_start.x;
    let dy = seg_start.y - ray_start.y;
    let t = (dx * s_dy - dy * s_dx) / denom;
    let u = (dx * r_dy - dy * r_dx) / denom;

    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndimage::filters::{gaussian_filter, sobel};

    #[test]
    fn test_calc_seam_simple() {
        let mut im_feats = Array2::<f32>::zeros((10, 20));
        for y in 0..10 {
            for x in 0..20 {
                im_feats[[y, x]] = (x as f32).sin().abs();
            }
        }
        let baseline = vec![Point::new(2.0, 5.0), Point::new(17.0, 5.0)];
        let polygon = vec![
            Point::new(2.0, 2.0),
            Point::new(17.0, 2.0),
            Point::new(17.0, 8.0),
            Point::new(2.0, 8.0),
        ];
        let seam = calc_seam(&baseline, &polygon, 0.0, &im_feats, 150.0);
        assert!(!seam.is_empty(), "seam should not be empty");
    }

    #[test]
    fn test_calculate_polygonal_environment_basic() {
        let mut scal_im = Array2::<f32>::zeros((20, 30));
        for x in 5..25 {
            scal_im[[10, x]] = 200.0;
            scal_im[[11, x]] = 200.0;
        }
        let im_feats = gaussian_filter(&sobel(&scal_im), 0.5);
        let baseline = vec![Point::new(5.0, 10.0), Point::new(24.0, 10.0)];
        let baselines = vec![baseline.clone()];
        let suppl_obj: Vec<Vec<Point>> = vec![];
        let polygons = calculate_polygonal_environment(
            &baselines,
            &im_feats,
            &suppl_obj,
            false,
            (29.0, 19.0),
        );
        assert_eq!(polygons.len(), 1);
        assert!(
            polygons[0].as_ref().unwrap().exterior.len() > 3,
            "polygon should have >3 points"
        );
    }
}
