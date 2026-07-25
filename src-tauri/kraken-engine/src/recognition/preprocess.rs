//! Line image preprocessing for recognition.
//!
//! Port of kraken's `ImageInputTransforms` for the input spec `(1, 120, 0, 1)`:
//!   1. Convert to grayscale ('L')
//!   2. (Optional) Binarize — for models trained on 1-bit images
//!   3. Normalize to target height:
//!      - if `center_norm`: ocropy `CenterNormalizer` content dewarp + resize
//!        (kraken `_create_transforms` branch B; this is the path our model
//!        spec selects — see [`super::lineest`])
//!      - else: plain Lanczos resize keeping aspect ratio
//!   4. Pad 16px left + 16px right, fill=255 (white)
//!   5. Scale to [0,1] (uint8 / 255)
//!   6. Invert (1.0 - im) — ink becomes high values
//!
//! The input `image` is expected to be an already-dewarped flat strip from
//! [`super::dewarp::extract_polygon_line`] (Stage 1 geometric warp).
//! Output: `(1, 1, target_height, W)` f32 tensor (NCHW).

use anyhow::Result;
use candle_core::{Device, Tensor};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};

/// Optional binarization applied before the resize step.
///
/// Used to match the distribution of recognition models trained on 1-bit
/// (binarized) line images, where offline binarization was applied before
/// training and is therefore not recorded in the model's `one_channel_mode`
/// metadata (which stays `"L"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binarization {
    /// Global Otsu threshold — one value for the whole image. Fast and robust
    /// on evenly-lit crops.
    Otsu,
    /// Sauvola local adaptive threshold — per-pixel threshold from local mean
    /// and standard deviation. Better than Otsu for uneven lighting and for
    /// scripts (e.g. Burmese) where preserving thin strokes and diacritics
    /// matters. Faithful port of Leptonica's `pixSauvolaGetThreshold`.
    Sauvola,
}

/// Preprocess a line image for recognition.
///
/// When `binarize` is `Some(...)`, the grayscale image is binarized **before**
/// the height-normalize step. This is required for recognition models whose
/// training data was 1-bit (binarized) PNGs: the model learned anti-aliased-
/// binary edge profiles, so feeding continuous-tone grayscale at inference
/// causes a train/serve distribution mismatch. Binarizing before the resize
/// re-introduces the anti-aliased-binary edges the model expects.
///
/// When `center_norm` is true, the height normalization uses kraken's ocropy
/// `CenterNormalizer` (content dewarp + resize) instead of a plain Lanczos
/// resize — see [`super::lineest`].
///
/// Returns a `(1, 1, target_height, W)` f32 tensor on CPU.
pub fn preprocess_line(
    image: &DynamicImage,
    target_height: usize,
    padding: usize,
    binarize: Option<Binarization>,
    center_norm: bool,
) -> Result<Tensor> {
    // 1. Convert to grayscale.
    let gray = image.to_luma8();

    // 2. Optional binarization (before resize so Lanczos re-anti-aliases the
    //    binary edges, matching the 1-bit training distribution). Skip if the
    //    line is already binary (e.g. a 1-bit source PNG, or a page that came
    //    through PDF "B&W" extraction): re-running Otsu/Sauvola on {0,255}
    //    data is a no-op at best and degenerate (Sauvola's local variance
    //    collapses in uniform regions) at worst.
    let gray = match binarize {
        Some(_) if is_binary(&gray) => gray,
        Some(Binarization::Otsu) => binarize_otsu(&gray),
        Some(Binarization::Sauvola) => binarize_sauvola(&gray),
        None => gray,
    };

    // 3. Normalize to target_height.
    let resized = if center_norm {
        let mut lnorm = super::lineest::CenterNormalizer::new(target_height);
        super::lineest::dewarp_line(&mut lnorm, &gray)
    } else {
        resize_to_height(&gray, target_height)
    };

    // 4. Pad left and right with white (255).
    let (w, h) = (resized.width() as usize, resized.height() as usize);
    let padded_w = w + 2 * padding;
    let mut padded: GrayImage = ImageBuffer::from_pixel(
        padded_w as u32,
        h as u32,
        Luma([255]),
    );
    // Copy the resized image into the center.
    for y in 0..h {
        for x in 0..w {
            padded.put_pixel(
                (x + padding) as u32,
                y as u32,
                resized.get_pixel(x as u32, y as u32).clone(),
            );
        }
    }

    // 5 & 6. Scale to [0,1] and invert in one pass.
    let data: Vec<f32> = padded
        .iter()
        .map(|&px| 1.0 - (px as f32) / 255.0)
        .collect();

    // Build tensor: (1, 1, H, W) NCHW
    let tensor = Tensor::from_vec(data, (h, padded_w), &Device::Cpu)?
        .unsqueeze(0)? // (1, H, W)
        .unsqueeze(0)?; // (1, 1, H, W)

    Ok(tensor)
}

/// Resize a grayscale image to a target height, preserving aspect ratio.
///
/// Uses Lanczos resampling (matching kraken's default). The width is computed
/// proportionally: `new_w = round(orig_w * target_height / orig_h)`.
fn resize_to_height(image: &GrayImage, target_height: usize) -> GrayImage {
    let (orig_w, orig_h) = image.dimensions();
    if orig_h == 0 {
        return image.clone();
    }
    let new_h = target_height as u32;
    let new_w = ((orig_w as f64) * (target_height as f64) / (orig_h as f64)).round() as u32;
    let new_w = new_w.max(1);
    image::imageops::resize(image, new_w, new_h, image::imageops::FilterType::Lanczos3)
}

/// Binarize a grayscale image using Otsu's automatic threshold selection.
///
/// Computes a global threshold that maximizes inter-class variance, then maps
/// pixels to pure black (ink = 0) / white (background = 255). This preserves
/// the black-on-white document polarity that the subsequent invert step (in
/// [`preprocess_line`]) expects.
fn binarize_otsu(image: &GrayImage) -> GrayImage {
    use imageproc::contrast::{threshold, ThresholdType};
    let level = imageproc::contrast::otsu_level(image);
    threshold(image, level, ThresholdType::Binary)
}

/// Binarize a grayscale image using Sauvola local adaptive thresholding.
///
/// Faithful port of Leptonica's `pixSauvolaGetThreshold` + `pixApplyLocalThreshold`
/// (binarize.c). For each pixel the threshold is:
///
/// ```text
///     t = m * (1 + k * (s / R - 1))
/// ```
/// where `m` = local mean, `s` = local standard deviation, `R = 128` (half the
/// dynamic range of 8-bit data), `k = factor`. A pixel darker than `t` becomes
/// ink (0); otherwise background (255).
///
/// Local statistics are computed over a `(2*whsize + 1)` square window using
/// two integral images (sum and sum-of-squares), giving O(W*H) total work.
/// `whsize` is clamped so the window fits the image (Leptonica requires
/// `w,h >= 2*whsize + 3`).
fn binarize_sauvola(image: &GrayImage) -> GrayImage {
    // Defaults, matching Leptonica's typical usage (binarize.c notes: whsize
    // typically >= 7; factor typ. 0.34).
    binarize_sauvola_with(image, 15, 0.34)
}

/// Sauvola binarization with explicit window half-width and factor.
///
/// Exposed for testing; production callers should use [`binarize_sauvola`].
fn binarize_sauvola_with(image: &GrayImage, whsize: u32, factor: f32) -> GrayImage {
    let (w, h) = image.dimensions();
    assert!(w > 0 && h > 0, "binarize_sauvola: empty image");

    // Leptonica requires w,h >= 2*whsize + 3. Clamp whsize so the window fits.
    let max_half = (w.min(h).saturating_sub(3)) / 2;
    let whsize = whsize.min(max_half.max(1));
    let r = 128.0f32; // R: half the dynamic range of 8-bit data.

    // Integral images (flat buffers, (h+1)*(w+1)), storing running sum and
    // running sum-of-squares of pixel values. Row-major, row y at offset y*(w+1).
    let stride = w as usize + 1;
    let n = stride * (h as usize + 1);
    let mut integral = vec![0u64; n];
    let mut integral_sq = vec![0u64; n];
    for y in 0..h {
        let mut row_sum = 0u64;
        let mut row_sum_sq = 0u64;
        for x in 0..w {
            let v = image.get_pixel(x, y)[0] as u64;
            row_sum = row_sum.wrapping_add(v);
            row_sum_sq = row_sum_sq.wrapping_add(v * v);
            let cur = (y as usize + 1) * stride + (x as usize + 1);
            let above = y as usize * stride + (x as usize + 1);
            integral[cur] = row_sum.wrapping_add(integral[above]);
            integral_sq[cur] = row_sum_sq.wrapping_add(integral_sq[above]);
        }
    }

    // Apply the Sauvola threshold per pixel using the windowed stats.
    let half = whsize as i32;
    image::ImageBuffer::from_fn(w, h, |x, y| {
        // Window bounds, clamped to the image (Leptonica uses mirrored borders;
        // clamping is equivalent for windowed-mean on interior pixels and only
        // differs near the very edge).
        let x1 = (x as i32 - half).max(0) as usize;
        let y1 = (y as i32 - half).max(0) as usize;
        let x2 = ((x as i32 + half).min(w as i32 - 1)) as usize;
        let y2 = ((y as i32 + half).min(h as i32 - 1)) as usize;
        let area = ((x2 - x1 + 1) * (y2 - y1 + 1)) as f64;

        // Summed-area-table query over [x1..=x2] × [y1..=y2], half-open in the
        // padded integral: use (y2+1, x2+1) corner.
        let a = integral[(y2 + 1) * stride + (x2 + 1)] as f64;
        let b = integral[y1 * stride + (x2 + 1)] as f64;
        let c = integral[(y2 + 1) * stride + x1] as f64;
        let d = integral[y1 * stride + x1] as f64;
        let sum = a - b - c + d;

        let a = integral_sq[(y2 + 1) * stride + (x2 + 1)] as f64;
        let b = integral_sq[y1 * stride + (x2 + 1)] as f64;
        let c = integral_sq[(y2 + 1) * stride + x1] as f64;
        let d = integral_sq[y1 * stride + x1] as f64;
        let sum_sq = a - b - c + d;

        let mean = sum / area;
        // Var = E[X^2] - (E[X])^2. Guard against tiny negative from rounding.
        let var = (sum_sq / area - mean * mean).max(0.0);
        let sd = var.sqrt();

        // Sauvola threshold (Leptonica binarize.c:773):
        //   t = m * (1 + factor * (sd / R - 1))
        let t = mean * (1.0 + factor as f64 * (sd / r as f64 - 1.0));

        let pixel = image.get_pixel(x, y)[0] as f64;
        // pixApplyLocalThreshold: pixel < t → foreground (ink = 0).
        if pixel < t {
            Luma([0u8])
        } else {
            Luma([255u8])
        }
    })
}

/// True iff every pixel is a pure binary value (0 or 255) — i.e. the image
/// is already binarized and running Otsu/Sauvola again would be wasted work
/// (and, for Sauvola, degenerate: local variance collapses in uniform
/// regions). Used to short-circuit re-binarization of 1-bit source data.
///
/// The `image` crate has no 1-bit variant: a 1-bit source PNG is upsampled
/// to 8-bit luma (values 0 and 255) on decode, so the original bit-depth is
/// gone by the time we see it — only a content check like this can detect it.
fn is_binary(image: &GrayImage) -> bool {
    image.iter().all(|&p| p == 0 || p == 255)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_preprocess_line_shape() {
        // This test only runs if the sample file exists.
        let path = "/tmp/line_0.png";
        if !Path::new(path).exists() {
            eprintln!("Skipping preprocess test — {path} not found");
            return;
        }
        let img = image::open(path).unwrap();
        let tensor = preprocess_line(&img, 120, 16, None, false).unwrap();
        let dims = tensor.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[0], 1); // batch
        assert_eq!(dims[1], 1); // channels
        assert_eq!(dims[2], 120); // height
        // Width should be > 0
        assert!(dims[3] > 32); // at least 2*padding
    }

    #[test]
    fn test_binarize_otsu_produces_binary_output() {
        // Build a synthetic grayscale image: dark half (0..16) on the left,
        // bright half (200..255) on the right. Otsu should find a threshold
        // somewhere in the gap and split the two halves cleanly.
        let (w, h) = (32u32, 4u32);
        let mut img: GrayImage = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { (x * 4) as u8 } else { 200 + (x as u8 % 8) * 7 };
                img.put_pixel(x, y, Luma([v]));
            }
        }

        let out = binarize_otsu(&img);

        // Every output pixel must be exactly 0 (ink) or 255 (background).
        for p in out.iter() {
            assert!(
                *p == 0 || *p == 255,
                "binarize_otsu produced a non-binary pixel: {p}"
            );
        }

        // Polarity: dark input region → 0 (ink), bright input region → 255 (bg).
        // This is the black-on-white polarity the invert step expects.
        assert_eq!(out.get_pixel(0, 0)[0], 0, "dark input should map to ink (0)");
        assert_eq!(out.get_pixel(w - 1, 0)[0], 255, "bright input should map to bg (255)");
    }

    #[test]
    fn test_binarize_sauvola_produces_binary_output() {
        // Synthetic dark-text-on-light-background: dark stroke in a bright field.
        let (w, h) = (40u32, 40u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([240]));
        for x in 10..30 {
            img.put_pixel(x, 20, Luma([20])); // dark ink stroke
        }

        let out = binarize_sauvola(&img);

        // Output must be binary.
        for p in out.iter() {
            assert!(
                *p == 0 || *p == 255,
                "binarize_sauvola produced a non-binary pixel: {p}"
            );
        }

        // Polarity: dark input → 0 (ink), bright background → 255 (bg).
        assert_eq!(out.get_pixel(20, 20)[0], 0, "dark stroke should map to ink (0)");
        // Use a corner well away from the stroke's local window.
        assert_eq!(out.get_pixel(0, 0)[0], 255, "bright bg should map to 255");
    }

    #[test]
    fn test_binarize_sauvola_handles_tiny_image() {
        // Regression: must not panic when the image is smaller than the window.
        let img: GrayImage = ImageBuffer::from_pixel(8, 8, Luma([200]));
        let out = binarize_sauvola(&img);
        // All-bright input → all background.
        assert!(out.iter().all(|&p| p == 255));
    }

    #[test]
    fn test_is_binary_all_black() {
        let img: GrayImage = ImageBuffer::from_pixel(4, 4, Luma([0]));
        assert!(is_binary(&img));
    }

    #[test]
    fn test_is_binary_all_white() {
        let img: GrayImage = ImageBuffer::from_pixel(4, 4, Luma([255]));
        assert!(is_binary(&img));
    }

    #[test]
    fn test_is_binary_mixed_binary_values() {
        // Both 0 and 255 present — still binary. (A naive "all-same-value"
        // check would wrongly reject this; a 1-bit source typically has both.)
        let mut img: GrayImage = ImageBuffer::from_pixel(4, 4, Luma([255]));
        img.put_pixel(0, 0, Luma([0]));
        img.put_pixel(3, 3, Luma([0]));
        assert!(is_binary(&img));
    }

    #[test]
    fn test_is_binary_rejects_gray_pixel() {
        // Any gray value disqualifies the image — it's not in the 1-bit
        // training distribution, so binarization should run.
        let mut img: GrayImage = ImageBuffer::from_pixel(4, 4, Luma([255]));
        img.put_pixel(2, 2, Luma([128]));
        assert!(!is_binary(&img));
    }

    #[test]
    fn test_preprocess_line_skips_binarize_on_binary_input() {
        // Feature guard: when the input is already binary, the `binarize`
        // option must have NO effect on the output tensor — i.e. passing
        // `Some(Otsu)` produces the same tensor as `None`. This is the
        // regression test for the skip-binary-input behavior.
        //
        // Build a small binary image (mix of ink and background), wrap as a
        // DynamicImage, and compare the two preprocess_line outputs.
        let (w, h) = (16u32, 4u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for x in 2..6 {
            img.put_pixel(x, 1, Luma([0]));
            img.put_pixel(x, 2, Luma([0]));
        }
        let dyn_img = DynamicImage::ImageLuma8(img);

        let tensor_none =
            preprocess_line(&dyn_img, 4, 2, None, false).expect("None path failed");
        let tensor_otsu = preprocess_line(&dyn_img, 4, 2, Some(Binarization::Otsu), false)
            .expect("Otsu path failed");

        // Same shape (sanity) and identical values → binarize was skipped.
        assert_eq!(tensor_none.shape(), tensor_otsu.shape(), "shapes diverged");
        let none_vals = tensor_none.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let otsu_vals = tensor_otsu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            none_vals.len(),
            otsu_vals.len(),
            "value counts diverged (should be identical)"
        );
        for (i, (a, b)) in none_vals.iter().zip(otsu_vals.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "pixel {i} differs: None={a}, Otsu={b} — binarize was NOT skipped on binary input"
            );
        }
    }
}
