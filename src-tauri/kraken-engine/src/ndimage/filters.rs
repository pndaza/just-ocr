//! Native image filters: convolve2d, gaussian, sobel, maximum_filter.

use ndarray::{array, Array2};

/// 2D correlation with zero-padded 'same' boundary.
///
/// For each output pixel, sums `kernel * neighborhood` where out-of-bounds
/// input samples are treated as 0. The output has the same shape as `input`.
/// (The kernel is applied in correlation orientation, i.e. not flipped; for
/// the symmetric kernels used by this crate this matches scipy.convolve2d.)
pub fn convolve2d_same(input: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32> {
    let (h, w) = input.dim();
    let (kh, kw) = kernel.dim();
    let kc_y = kh as isize / 2;
    let kc_x = kw as isize / 2;
    let mut out = Array2::<f32>::zeros((h, w));
    for i in 0..h as isize {
        for j in 0..w as isize {
            let mut acc = 0.0f32;
            for ki in 0..kh as isize {
                for kj in 0..kw as isize {
                    let iy = i + ki - kc_y;
                    let ix = j + kj - kc_x;
                    if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                        acc += input[[iy as usize, ix as usize]]
                            * kernel[[ki as usize, kj as usize]];
                    }
                }
            }
            out[[i as usize, j as usize]] = acc;
        }
    }
    out
}

/// Build a normalized 1D Gaussian kernel of length `2 * radius + 1`.
pub fn gaussian_kernel_1d(sigma: f32, radius: usize) -> Vec<f32> {
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut kernel: Vec<f32> = (0..=2 * radius)
        .map(|k| {
            let x = (k as isize - radius as isize) as f32;
            (-(x * x) / two_sigma_sq).exp()
        })
        .collect();
    let sum: f32 = kernel.iter().copied().sum();
    for v in kernel.iter_mut() {
        *v /= sum;
    }
    kernel
}

/// Reflect an out-of-range index into `[0, len)` using half-sample symmetric
/// (scipy 'reflect') boundary handling: `(d c b a | a b c d | d c b a)`.
pub fn reflect_index(idx: isize, len: usize) -> usize {
    let len_i = len as isize;
    let period = 2 * len_i;
    let mut i = idx.rem_euclid(period); // in [0, period)
    if i >= len_i {
        i = period - 1 - i;
    }
    i as usize
}

/// Separable Gaussian filter with reflect boundary handling.
///
/// Applies the 1D Gaussian kernel along x (columns) then y (rows). Precomputes
/// reflected index tables to eliminate per-pixel reflect_index calls.
pub fn gaussian_filter(input: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let (h, w) = input.dim();
    if h == 0 || w == 0 {
        return input.clone();
    }
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let kernel = gaussian_kernel_1d(sigma, radius);
    let klen = kernel.len();
    let r = radius as isize;

    // Precompute reflected index tables once per axis.
    let col_idx: Vec<usize> = (0..w * klen)
        .map(|t| {
            let j = t / klen;
            let k = t % klen;
            reflect_index(j as isize + k as isize - r, w)
        })
        .collect();
    let row_idx: Vec<usize> = (0..h * klen)
        .map(|t| {
            let i = t / klen;
            let k = t % klen;
            reflect_index(i as isize + k as isize - r, h)
        })
        .collect();

    // Pass 1: blur along x (axis 1) — rows are contiguous in memory.
    let tmp = {
        let in_slice = input.as_slice_memory_order().unwrap();
        let mut buf = vec![0.0f32; h * w];
        for i in 0..h {
            let row = &in_slice[i * w..(i + 1) * w];
            let out_row = &mut buf[i * w..(i + 1) * w];
            for j in 0..w {
                let idx_base = j * klen;
                let mut acc = 0.0f32;
                for k in 0..klen {
                    acc += row[col_idx[idx_base + k]] * kernel[k];
                }
                out_row[j] = acc;
            }
        }
        buf
    };

    // Pass 2: blur along y (axis 0).
    let mut out = Array2::<f32>::zeros((h, w));
    let out_slice = out.as_slice_memory_order_mut().unwrap();
    for i in 0..h {
        let idx_base = i * klen;
        for j in 0..w {
            let mut acc = 0.0f32;
            for k in 0..klen {
                acc += tmp[row_idx[idx_base + k] * w + j] * kernel[k];
            }
            out_slice[i * w + j] = acc;
        }
    }
    out
}

/// Sobel gradient magnitude. Returns `sqrt(gx^2 + gy^2)` using the standard
/// 3x3 Sobel kernels with zero-padded 'same' boundary.
pub fn sobel(input: &Array2<f32>) -> Array2<f32> {
    let kx = array![[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let ky = array![[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]];
    let gx = convolve2d_same(input, &kx);
    let gy = convolve2d_same(input, &ky);
    let mut out = Array2::<f32>::zeros(input.dim());
    for ((&x, &y), o) in gx.iter().zip(gy.iter()).zip(out.iter_mut()) {
        *o = (x * x + y * y).sqrt();
    }
    out
}

/// Square-footprint maximum filter. For each pixel, returns the max value in
/// the `(size x size)` neighborhood centered on it; out-of-bounds positions
/// are skipped (the window is clipped to the valid region).
///
/// Implemented separably (horizontal pass, then vertical pass) so the cost is
/// O(H·W·size) instead of O(H·W·size²). Each 1-D pass is a sliding-window
/// max over contiguous memory (rows for the first pass, columns buffered into
/// contiguous slices for the second), making it cache-friendly.
pub fn maximum_filter(input: &Array2<f32>, size: usize) -> Array2<f32> {
    if size <= 1 {
        return input.clone();
    }
    let (h, w) = input.dim();
    let half = size / 2;

    // Pass 1: 1-D max along each row (axis 1).
    let horiz = {
        let mut tmp = vec![f32::NEG_INFINITY; h * w];
        for i in 0..h {
            let row = &input.as_slice_memory_order().unwrap()[i * w..(i + 1) * w];
            let out_row = &mut tmp[i * w..(i + 1) * w];
            for j in 0..w {
                let lo = j.saturating_sub(half);
                let hi = (j + half + 1).min(w);
                let mut m = f32::NEG_INFINITY;
                for &v in &row[lo..hi] {
                    if v > m {
                        m = v;
                    }
                }
                out_row[j] = m;
            }
        }
        tmp
    };

    // Pass 2: 1-D max along each column (axis 0) on the pass-1 result.
    // Process column-major: extract a contiguous column buffer, run the same
    // sliding-window max, write back.
    let mut col_buf = vec![f32::NEG_INFINITY; h];
    let mut out = Array2::<f32>::from_elem((h, w), f32::NEG_INFINITY);
    for j in 0..w {
        // Gather column j into contiguous buffer.
        for i in 0..h {
            col_buf[i] = horiz[i * w + j];
        }
        // Sliding-window max down the column.
        for i in 0..h {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(h);
            let mut m = f32::NEG_INFINITY;
            for &v in &col_buf[lo..hi] {
                if v > m {
                    m = v;
                }
            }
            out[[i, j]] = m;
        }
    }

    out
}

/// Sato tubular ridge filter. Matches skimage.filters.sato.
///
/// Computes Hessian-based ridge detection. For 2D, the Sato filter uses
/// eigenvalues of the Hessian matrix at multiple scales. For bright ridges
/// (black_ridges=False), a ridge pixel has one large positive eigenvalue.
///
/// The sigmas parameter controls the scale of ridges detected. If None,
/// uses skimage's default of range(1, 10, 2) = (1, 3, 5, 7, 9).
/// For kraken, sato is called as: filters.sato(bl_map, black_ridges=False, mode='constant')
pub fn sato(input: &Array2<f32>, sigmas: Option<Vec<f32>>, black_ridges: bool) -> Array2<f32> {
    let (h, w) = input.dim();
    let mut result = Array2::<f32>::zeros((h, w));

    // skimage default sigmas: range(1, 10, 2) = (1, 3, 5, 7, 9)
    let sigmas = match sigmas {
        Some(s) => s,
        None => vec![1.0, 3.0, 5.0, 7.0, 9.0],
    };

    // skimage convention: ridge filters operate on dark ridges, so negate
    // the image when detecting bright (white) ridges.
    let adjusted_input = if black_ridges {
        input.to_owned()
    } else {
        input.mapv(|v| -v)
    };

    for &sigma in &sigmas {
        // Compute Hessian components at this scale.
        // The σ² scaling is applied once here (in the Sato response), not in
        // hessian_components, so the multi-scale combination is effective.
        let smoothed = gaussian_filter(&adjusted_input, sigma);
        let (hxx, hxy, hyy) = hessian_components(&smoothed);

        let hxx_s = hxx.as_slice_memory_order().unwrap();
        let hxy_s = hxy.as_slice_memory_order().unwrap();
        let hyy_s = hyy.as_slice_memory_order().unwrap();
        let res_s = result.as_slice_memory_order_mut().unwrap();
        let sigma_sq = sigma * sigma;

        for idx in 0..h * w {
            let xx = hxx_s[idx];
            let xy = hxy_s[idx];
            let yy = hyy_s[idx];

            // Eigenvalues of 2x2 Hessian [[xx, xy], [xy, yy]]
            let tr = xx + yy;
            let det = xx * yy - xy * xy;
            let disc = ((tr * tr / 4.0) - det).max(0.0).sqrt();
            let lambda1 = tr / 2.0 + disc; // larger eigenvalue

            // Sato's tubeness formula for 2D (ref. skimage):
            //   sigma^2 * max(lambda_largest, 0)
            let val = sigma_sq * lambda1.max(0.0);
            if val > res_s[idx] {
                res_s[idx] = val;
            }
        }
    }

    result
}

/// Compute Hessian components (second derivatives) from a smoothed image.
/// Returns raw finite-difference second derivatives without σ normalization —
/// the σ² scaling is applied by the caller (sato) so it doesn't cancel out.
fn hessian_components(smoothed: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (h, w) = smoothed.dim();
    let mut hxx = Array2::<f32>::zeros((h, w));
    let mut hxy = Array2::<f32>::zeros((h, w));
    let mut hyy = Array2::<f32>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let xp = (x + 1).min(w - 1);
            let xm = x.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            let ym = y.saturating_sub(1);

            hxx[[y, x]] =
                smoothed[[y, xp]] - 2.0 * smoothed[[y, x]] + smoothed[[y, xm]];
            hyy[[y, x]] =
                smoothed[[yp, x]] - 2.0 * smoothed[[y, x]] + smoothed[[ym, x]];
            hxy[[y, x]] = (smoothed[[yp, xp]] - smoothed[[yp, xm]]
                - smoothed[[ym, xp]]
                + smoothed[[ym, xm]])
                / 4.0;
        }
    }

    (hxx, hxy, hyy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_convolve2d_identity() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let kernel = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let out = convolve2d_same(&input, &kernel);
        assert_eq!(out, input);
    }

    #[test]
    fn test_convolve2d_box_blur() {
        let input = array![[0.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 0.0]];
        // 3x3 uniform box-blur kernel, all elements 1.0/9.0
        let kernel = ndarray::Array2::from_elem((3, 3), 1.0 / 9.0);
        let out = convolve2d_same(&input, &kernel);
        assert!((out[[1, 1]] - 1.0).abs() < 1e-5, "center: {}", out[[1, 1]]);
    }

    #[test]
    fn test_convolve2d_endpoint_kernel() {
        let skel = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let kernel = array![[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 1.0]];
        let out = convolve2d_same(&skel, &kernel);
        assert_eq!(out[[1, 1]], 11.0, "left endpoint");
        assert_eq!(out[[1, 2]], 12.0, "middle");
        assert_eq!(out[[1, 3]], 11.0, "right endpoint");
    }

    #[test]
    fn test_sobel_basic() {
        let input = array![
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let gx = sobel(&input);
        assert!(gx[[1, 1]].abs() > 0.1, "expected gradient at edge");
        assert!(gx[[1, 2]].abs() > 0.1, "expected gradient at edge");
        assert!(gx[[1, 0]].abs() < 0.1, "expected ~0 in flat region, got {}", gx[[1, 0]]);
    }

    #[test]
    fn test_maximum_filter() {
        let input = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let out = maximum_filter(&input, 3);
        for y in 0..3 {
            for x in 1..4 {
                assert_eq!(out[[y, x]], 9.0, "({y},{x})");
            }
        }
    }

    #[test]
    fn test_gaussian_filter_smoothing() {
        let input = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let out = gaussian_filter(&input, 1.0);
        assert!(out[[2, 2]] < 1.0, "center not smoothed: {}", out[[2, 2]]);
        assert!(out[[2, 1]] > 0.0, "neighbor not smoothed");
        assert!(out[[1, 2]] > 0.0, "neighbor not smoothed");
        let sum_in: f32 = input.iter().sum();
        let sum_out: f32 = out.iter().sum();
        assert!((sum_out - sum_in).abs() < 0.2, "sum not preserved: {sum_out} vs {sum_in}");
    }

    #[test]
    fn test_sato_horizontal_line() {
        // A horizontal bright line on dark background
        let mut input = Array2::<f32>::zeros((9, 9));
        for x in 0..9 {
            input[[4, x]] = 1.0;
        }
        let out = sato(&input, None, false);
        // The ridge should be detected along the line
        assert!(out[[4, 4]] > 0.1, "sato should detect ridge: got {}", out[[4, 4]]);
        // Off-line should be low
        assert!(out[[0, 4]] < 0.1, "off-ridge should be low: got {}", out[[0, 4]]);
    }

    #[test]
    fn test_sato_vertical_line() {
        let mut input = Array2::<f32>::zeros((9, 9));
        for y in 0..9 {
            input[[y, 4]] = 1.0;
        }
        let out = sato(&input, None, false);
        assert!(out[[4, 4]] > 0.1, "sato should detect vertical ridge: got {}", out[[4, 4]]);
    }
}
