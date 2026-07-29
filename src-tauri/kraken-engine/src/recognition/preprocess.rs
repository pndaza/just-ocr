//! Line image preprocessing for recognition.
//!
//! Port of kraken's `ImageInputTransforms` for the input spec `(1, 48, 0, 1)`:
//!   1. Convert to grayscale ('L')
//!   2. Sauvola binarization on the full-resolution crop. Binarizing before
//!      resize gives Sauvola the maximum pixel population for its local-window
//!      threshold, and lets the bleed-trim (step 3) operate on a clean 1-bit
//!      strip at native resolution.
//!   3. Trim neighbor-line bleed + crop to the text body (rows between the
//!      first white gap above and below the ink center). See
//!      [`trim_neighbor_bleed`].
//!   4. Normalize to target height:
//!      - if `center_norm`: ocropy `CenterNormalizer` content dewarp + resize
//!        (kraken `_create_transforms` branch B; this is the path our model
//!        spec selects — see [`super::lineest`])
//!      - else: plain Lanczos resize keeping aspect ratio
//!   5. Pad 16px left + 16px right, fill=255 (white)
//!   6. Scale to [0,1] (uint8 / 255)
//!   7. Invert (1.0 - im) — ink becomes high values
//!
//! The input `image` is expected to be an already-dewarped flat strip from
//! [`super::dewarp::extract_polygon_line`] (Stage 1 geometric warp).
//! Output: `(1, 1, target_height, W)` f32 tensor (NCHW).

use anyhow::Result;
use candle_core::{Device, Tensor};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};

/// Preprocess a line image for recognition.
///
/// Binarization (Sauvola local adaptive threshold) is **always applied**, on
/// the full-resolution grayscale crop and *before* the resize. Binarizing at
/// native resolution gives Sauvola the maximum pixel population for its local-
/// window statistics (more accurate threshold), and lets the bleed-trim
/// operate on a clean 1-bit strip before any resampling. The subsequent resize
/// then scales only the cleaned text body.
///
/// Sauvola (not Otsu) is used because it adapts per-pixel to local lighting,
/// which preserves thin strokes and diacritics in scripts like Burmese where a
/// single global threshold can erode fine detail. It is skipped automatically
/// when the strip is already binary (1-bit source), where re-thresholding is a
/// no-op at best and degenerate at worst.
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
    center_norm: bool,
) -> Result<Tensor> {
    // Debug aid: when KRKN_DUMP_DIR is set, write the input crop and each
    // intermediate stage to that dir so the pipeline can be inspected.
    // One seq per line keeps a line's stages grouped.
    let dump_seq = crate::next_dump_seq();
    crate::dump_debug(image, "in", dump_seq);

    // 1. Convert to grayscale.
    let gray = image.to_luma8();

    // 2. Sauvola binarization on the full-resolution grayscale crop. Binarizing
    //    before resize gives Sauvola the maximum pixel population for its local
    //    window statistics (more accurate threshold), and lets the bleed-trim
    //    + crop (step 3) operate on a clean 1-bit strip at native resolution
    //    before any resampling. The subsequent resize then scales only the
    //    cleaned text body.
    let binary = if is_binary(&gray) {
        gray
    } else {
        binarize_sauvola(&gray)
    };

    // 3. Remove neighbor-line ink that bled into an over-tall quad, then CROP
    //    to the text body (rows [top..=bottom]). Cropping — not just clearing
    //    to white — means the resize scales only the real text, with no dead
    //    white rows diluting the height normalization.
    let binary = trim_neighbor_bleed(&binary);
    crate::dump_debug(
        &DynamicImage::ImageLuma8(binary.clone()),
        "trimmed",
        dump_seq,
    );

    // 4. Normalize to target_height.
    let resized = if center_norm {
        let mut lnorm = super::lineest::CenterNormalizer::new(target_height);
        super::lineest::dewarp_line(&mut lnorm, &binary)
    } else {
        resize_to_height(&binary, target_height)
    };
    crate::dump_debug(
        &DynamicImage::ImageLuma8(resized.clone()),
        "resized",
        dump_seq,
    );

    // 5. Pad left and right with white (255).
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

    // 6 & 7. Scale to [0,1] and invert in one pass.
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

/// Row ink-density threshold below which a row counts as a horizontal white
/// gap separating the main text body from neighbor-line bleed. Strict zero is
/// too fragile (a single stray pixel would block the trim); 0.5% is sparse
/// enough to be a real gap in a 1-bit strip yet tolerant of one or two specks.
const BLEED_GAP_DENSITY: f64 = 0.005;

/// Half-height of the protected band around the line center, as a fraction of
/// the core body height. Within `[center ± protect]` ink is assumed to belong
/// to THIS line and is never scanned for a separator gap.
///
/// This exists because some Myanmar font styles render vowel signs / stacked
/// consonants as ink **separated from the consonant by a real white gap**
/// (untouched diacritics). A naive first-gap scan outward from center would
/// mistake that internal gap for the bleed boundary and crop the diacritic.
/// The protect band makes the trim position-aware: only ink beyond this band
/// is treated as a candidate for neighbor bleed.
///
/// Measured on the dump corpus: legitimate gap-separated diacritics sit at up
/// to ~0.73 × body_h from the body midpoint; neighbor-line bleed clusters
/// tightly at 0.69–0.78× body_h. These overlap, so position alone can't
/// perfectly separate them. 0.6 is an aggressive trim that sits below the
/// observed bleed cluster (clears all 24 bleed bands on the test image) — at
/// the cost of potentially cropping a diacritic pushed out past 0.6× body_h on
/// some fonts. It relies on the longest-band center (which can't be biased by
/// asymmetric bleed) to place the band correctly. Tune up if diacritics get
/// cropped, down if bleed survives.
const BLEED_PROTECT_FRAC: f64 = 0.6;

/// Remove neighbor-line ink that bled into the top/bottom of an over-tall
/// PP-OCR quad, returning the CROPPED text body.
///
/// PP-OCR's quads are sometimes a few pixels taller than the text they bound,
/// capturing fragments of the lines above/below. After binarization this
/// shows up as ink separated from the main text body by a horizontal white
/// gap. This finds the first such gap in the OUTER region (beyond a protect
/// band around the center) and returns the sub-image `[top..=bottom]`.
///
/// **Protect band.** The gap search does NOT start at the body edge. It starts
/// only beyond `[center ± BLEED_PROTECT_FRAC × core_body_h]`. Anything inside
/// that band — including gap-separated diacritics that belong to this line —
/// is kept unconditionally. This is what distinguishes a diacritic (near
/// center) from bleed (the next line, far from center): the gap width alone
/// cannot, since tight line spacing can make the bleed gap as narrow as a
/// diacritic gap. Position relative to center is the only reliable signal.
///
/// Cropping rather than clearing-to-white means the subsequent resize scales
/// only the cleaned text body to `target_height`, so no dead rows dilute the
/// height normalization.
///
/// **Centering.** The line center and core body height are derived from the
/// **longest contiguous inked band**, not the median of all inked rows. The
/// median is biased toward asymmetric bleed: when a quad captures a fragment
/// of the line above but not below, the extra inked rows drag the median
/// toward the bleed side, which inflates the protect band there and hides the
/// separator gap from the scan. Taking the longest band as the body makes the
/// center a property of the body alone — bleed and separated diacritics are
/// shorter bands and cannot move it.
fn trim_neighbor_bleed(image: &GrayImage) -> GrayImage {
    let (w, h) = image.dimensions();

    // Per-row ink density. Ink = dark pixels (< 128) on the black-on-white
    // binarized strip.
    let mut density = vec![0.0f64; h as usize];
    let threshold = 128u8;
    for y in 0..h {
        let mut ink = 0u32;
        for x in 0..w {
            if image.get_pixel(x, y)[0] < threshold {
                ink += 1;
            }
        }
        density[y as usize] = ink as f64 / w as f64;
    }

    // Collect contiguous inked bands (runs of rows at/above the gap density).
    // If the strip is (near-) empty there's nothing to trim — return unchanged.
    let mut bands: Vec<(i32, i32)> = Vec::new();
    let mut in_band = false;
    let mut band_start = 0i32;
    for (y, &d) in density.iter().enumerate() {
        if d >= BLEED_GAP_DENSITY {
            if !in_band {
                band_start = y as i32;
                in_band = true;
            }
        } else if in_band {
            bands.push((band_start, y as i32 - 1));
            in_band = false;
        }
    }
    if in_band {
        bands.push((band_start, h as i32 - 1));
    }
    if bands.is_empty() {
        return image.clone();
    }

    // Body = longest contiguous inked band. The text body is definitionally
    // the largest ink mass; bleed fragments and separated diacritics are
    // shorter. Center and height both derive from THIS band, so asymmetric
    // bleed cannot bias the protect-band placement.
    let (body_top, body_bottom) = *bands
        .iter()
        .max_by_key(|&(s, e)| e - s + 1)
        .expect("bands is non-empty");
    let center = (body_top + body_bottom) / 2;
    let core_body_h = (body_bottom - body_top + 1).max(1);

    // Protect band: within [lo, hi] everything is kept unconditionally. The
    // separator-gap scan runs only in the outer region (rows < lo upward,
    // rows > hi downward). lo/hi are clamped to the strip bounds.
    let protect = (core_body_h as f64 * BLEED_PROTECT_FRAC).round() as i32;
    let lo = (center - protect).max(0);
    let hi = (center + protect).min((h - 1) as i32);

    // Scan upward through the outer region only; first gap marks the top
    // boundary. If no gap is found (diacritic extends to the edge, or the
    // whole upper region is inked), top stays 0 — keep everything (safe).
    let mut top = 0i32;
    for r in (0..lo).rev() {
        if density[r as usize] < BLEED_GAP_DENSITY {
            top = r + 1;
            break;
        }
    }
    // Scan downward through the outer region only; first gap marks the bottom.
    let mut bottom = (h - 1) as i32;
    for r in (hi + 1)..h as i32 {
        if density[r as usize] < BLEED_GAP_DENSITY {
            bottom = r - 1;
            break;
        }
    }

    // Keep a proportional white margin above/below the cropped text body. The
    // model expects the glyphs surrounded by some whitespace (training renders
    // weren't tight-cropped), so cropping flush would over-stretch it at the
    // resize. The margin extends outward through WHITE rows only — it stops at
    // the first inked row (or the strip edge) so it can never reach back into
    // bleed that sits just beyond a narrow gap (tight line spacing).
    let crop_h = (bottom - top + 1).max(1);
    let margin = (crop_h as f64 * BLEED_KEEP_MARGIN_FRAC).round() as i32;
    let mut mt = 0;
    for k in 1..=margin {
        let r = top - k;
        if r < 0 || density[r as usize] >= BLEED_GAP_DENSITY {
            break;
        }
        mt = k;
    }
    let mut mb = 0;
    for k in 1..=margin {
        let r = bottom + k;
        if r >= h as i32 || density[r as usize] >= BLEED_GAP_DENSITY {
            break;
        }
        mb = k;
    }
    let top = top - mt;
    let bottom = bottom + mb;

    // Return the cropped [top..=bottom] with the margin kept as white. The
    // margin rows beyond the original strip bounds are dropped (clamped above),
    // so they aren't synthesized — only rows that actually existed (and were
    // white, being outside the body) are kept.
    let new_h = (bottom - top + 1).max(1) as u32;
    let mut out = GrayImage::new(w, new_h);
    for (oy, sy) in (top..=bottom).enumerate() {
        for x in 0..w {
            out.put_pixel(x, oy as u32, *image.get_pixel(x, sy as u32));
        }
    }
    out
}

/// Fraction of the text-body height to keep as white margin above and below
/// when trimming neighbor-bleed. 0.15 ≈ 15%: enough breathing room that the
/// resize doesn't over-stretch the glyphs, small enough that a typical over-
/// tall quad still sheds its bleed. Tuned for the bundled Burmese model.
const BLEED_KEEP_MARGIN_FRAC: f64 = 0.15;

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
        // `preprocess_line` is generic over target height/padding — we pass
        // arbitrary values and assert the output shape tracks them. The
        // bundled bur_recog uses height=48 (read dynamically from its VGSL
        // spec by the loader) and padding=16 (currently hardcoded in
        // `recognition/model.rs::build`), but tying this test to those
        // would just make it brittle on a model swap. The invariant under
        // test is "output height == target_height", not "height is 48".
        let target_height = 48;
        let padding = 16;
        let tensor = preprocess_line(&img, target_height, padding, false).unwrap();
        let dims = tensor.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[0], 1); // batch
        assert_eq!(dims[1], 1); // channels
        assert_eq!(dims[2], target_height); // height tracks the argument
        // Width should be > 0
        assert!(dims[3] > 2 * padding); // at least 2*padding
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
    fn test_preprocess_line_skips_sauvola_on_binary_input() {
        // Feature guard: when the input is already binary, Sauvola must be
        // skipped (its local-variance estimate degenerates on uniform {0,255}
        // regions). We assert this indirectly — since resize legitimately
        // anti-aliases {0,255} edges into gray ramps when upscaling, we can't
        // check the full tensor for binary values. Instead we verify the
        // `is_binary` gate itself: a {0,255} image is detected as binary, so
        // the binarize stage is a no-op (the four `test_is_binary_*` cases
        // cover the detector; this test pins the pipeline-level consequence).
        let (w, h) = (16u32, 4u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for x in 2..6 {
            img.put_pixel(x, 1, Luma([0]));
            img.put_pixel(x, 2, Luma([0]));
        }
        // The is_binary gate must fire → Sauvola branch is skipped.
        assert!(is_binary(&img), "binary input not detected — Sauvola would run");

        // And the full pipeline must not error on binary input.
        let dyn_img = DynamicImage::ImageLuma8(img);
        let tensor = preprocess_line(&dyn_img, h as usize, 2, false).expect("preprocess failed");
        // Output shape is intact (1, 1, target_height, padded_w).
        let dims = tensor.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 1);
        assert_eq!(dims[2], h as usize);
    }

    #[test]
    fn test_trim_neighbor_bleed_crops_to_text_body() {
        // A 40-row strip with a contiguous text body in rows 10..19, plus
        // "bleed" ink far above (rows 0..1) and far below (rows 34..35) —
        // realistic spacing where bleed sits well outside the protect band
        // (0.6 × body_h from center). The trim should drop the bleed rows
        // and keep the body plus a small white margin around it.
        let (w, h) = (16u32, 40u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Main body rows 10..19.
        for y in 10..=19 {
            for x in 2..14 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Bleed above (rows 0..1).
        for x in 3..6 {
            img.put_pixel(x, 0, Luma([0]));
            img.put_pixel(x, 1, Luma([0]));
        }
        // Bleed below (rows 34..35).
        for x in 8..11 {
            img.put_pixel(x, 34, Luma([0]));
            img.put_pixel(x, 35, Luma([0]));
        }

        let out = trim_neighbor_bleed(&img);

        // Output is strictly shorter than the original (bleed + dead padding
        // dropped). Width unchanged.
        assert_eq!(out.width(), w, "width unchanged");
        assert!(
            out.height() < h,
            "should drop the bleed rows ({} < {})",
            out.height(),
            h
        );
        // All 10 body rows of ink must survive somewhere in the output.
        let ink_rows: Vec<usize> = (0..out.height())
            .map(|y| out.get_pixel(5, y)[0])
            .enumerate()
            .filter_map(|(y, v)| if v == 0 { Some(y) } else { None })
            .collect();
        assert_eq!(
            ink_rows.len(),
            10,
            "all 10 body ink rows must survive, got {ink_rows:?}"
        );
        // Bleed rows must be entirely absent. The body spans columns 2..14,
        // which overlaps the bleed columns, so we can't distinguish by column.
        // Instead count total dark pixels: body alone is 10 rows × 12 cols =
        // 120; if either bleed band (6 px each) leaked in, the count rises.
        let body_ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            body_ink, 120,
            "bleed ink leaked into output (expected exactly the 120 body px)"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_noop_on_no_gap() {
        // When ink fills the whole strip with no white gap, nothing should be
        // trimmed (the whole strip is the text body). Guards against trimming
        // legitimate full-height ink. Height is unchanged.
        let (w, h) = (8u32, 8u32);
        let img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([0]));
        let out = trim_neighbor_bleed(&img);
        assert_eq!(out.dimensions(), (w, h), "full-ink strip height should be unchanged");
        assert!(out.iter().all(|&p| p == 0), "full-ink strip was wrongly trimmed");
    }

    #[test]
    fn test_trim_neighbor_bleed_preserves_tall_diacritic() {
        // A tall vertical stroke (like a Burmese vowel sign) extending from
        // the body up to row 0, with NO white gap between it and the body,
        // must NOT be cropped away — it's connected to the centerline.
        let (w, h) = (16u32, 16u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Body rows 6..10.
        for y in 6..=10 {
            for x in 2..14 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Tall stroke from body up to row 0 (contiguous, no gap).
        for y in 0..=10 {
            img.put_pixel(7, y, Luma([0]));
        }

        let out = trim_neighbor_bleed(&img);

        // The tall stroke must survive at every output row — it's connected
        // to the body, so the upward scan finds no gap and top stays at row 0.
        for y in 0..=10 {
            assert_eq!(
                out.get_pixel(7, y)[0], 0,
                "tall diacritic at out-row {y} was wrongly cropped (not separated by a gap)"
            );
        }
    }

    #[test]
    fn test_trim_neighbor_bleed_preserves_gap_separated_diacritic() {
        // The case the protect band exists for: a diacritic (e.g. a Myanmar
        // vowel sign in a font style that doesn't touch the consonant) sitting
        // ABOVE the body, separated from it by a real white gap.
        //
        // Body rows 10..19 (10 rows, center ≈ row 14). A diacritic at rows 7..8
        // with a white gap at row 9. The diacritic sits at ~0.7×body_h from
        // center — just OUTSIDE the 0.6× protect band. It survives because the
        // upward scan runs from the band edge through its inked rows (7,8)
        // without finding a gap; the separator gap at row 9 is BELOW the
        // diacritic, inside the band, and never reached. The protect band's job
        // here is to block the scan from stopping at that internal gap.
        let (w, h) = (16u32, 40u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Main body rows 10..19.
        for y in 10..=19 {
            for x in 2..14 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Gap-separated diacritic at rows 7..8 (narrow, like a vowel mark).
        for y in 7..=8 {
            for x in 6..9 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // The diacritic must survive: at least one of its pixels must be dark
        // in the output. Under the old logic it would have been dropped at the
        // row-9 gap.
        let diacritic_survives = (0..out.height()).any(|y| {
            (6..9).any(|x| out.get_pixel(x, y)[0] == 0)
        });
        assert!(
            diacritic_survives,
            "gap-separated diacritic was cropped — protect band failed to keep it"
        );
        // And the body still survives too.
        let body_survives = (0..out.height()).any(|y| out.get_pixel(5, y)[0] == 0);
        assert!(body_survives, "body was lost");
    }

    #[test]
    fn test_trim_neighbor_bleed_crops_asymmetric_top_bleed() {
        // The case the longest-band center exists for: bleed ONLY at the top
        // (none at the bottom). With a median-of-inked-rows center, the extra
        // bleed rows drag the median up toward the bleed, inflating the protect
        // band on the bleed side and hiding the separator gap. The longest-band
        // center stays at the body's true midpoint, so the protect band doesn't
        // reach the gap and the bleed is cropped.
        //
        // Body rows 12..31 (20 rows, midpoint 21). Bleed rows 0..3 (4 rows),
        // separated by a white gap at rows 4..11. The gap sits at ~0.35 × body_h
        // below the body top — close to the body, exactly where a biased median
        // would hide it but the longest-band midpoint exposes it.
        let (w, h) = (16u32, 40u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Main body rows 12..31.
        for y in 12..=31 {
            for x in 2..14 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Asymmetric bleed above only (rows 0..3).
        for x in 3..6 {
            for y in 0..=3 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // The bleed must be dropped: no dark pixels in its column range should
        // survive. Use a total-ink-count check — the body has 20 rows × 12 cols
        // = 240 dark px; the bleed adds 4 rows × 3 cols = 12 px. If the bleed
        // leaked through, the count would be > 240.
        let body_ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            body_ink, 240,
            "top-only bleed leaked into output (expected exactly the 240 body px, got {body_ink})"
        );
    }
}
