//! Line image preprocessing for recognition.
//!
//! Port of kraken's `ImageInputTransforms` for the input spec `(1, 48, 0, 1)`:
//!   1. Convert to grayscale ('L')
//!   2. Sauvola binarization on the full-resolution crop. Binarizing before
//!      resize gives Sauvola the maximum pixel population for its local-window
//!      threshold, and lets the bleed-trim (step 3) operate on a clean 1-bit
//!      strip at native resolution.
//!   3. Trim neighbor-line bleed + crop to the text body. A two-tier white-gap
//!      scan — whole-width fast path, then a per-chunk fallback (equal
//!      ~100px windows from an even split of the strip width)
//!      for tight line spacing — finds the separator between body and bleed.
//!      See [`trim_neighbor_bleed`].
//!   3b. Pad [`BODY_PAD_PX`] white on all four sides — before the resize so it
//!       scales into proportional margin in the target-height output.
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
    crate::dump_debug(
        &DynamicImage::ImageLuma8(binary.clone()),
        "binary",
        dump_seq,
    );

    // 3. Remove neighbor-line ink that bled into an over-tall quad, then CROP
    //    to the text body (rows [top..=bottom]). Cropping — not just clearing
    //    to white — means the resize scales only the real text, with no dead
    //    white rows diluting the height normalization. This is the sole bleed
    //    defense (the source-level box shrink was removed — it moved accurate
    //    boxes as much as over-big ones). Verified essential: disabling it
    //    degrades recognition on ~half the lines of thawzin_02.
    let binary = trim_neighbor_bleed(&binary);
    crate::dump_debug(
        &DynamicImage::ImageLuma8(binary.clone()),
        "trimmed",
        dump_seq,
    );

    // 3b. Pad with white on all four sides — added BEFORE the height-normalizing
    //     resize so it scales with the body into proportional margin in the
    //     target-height output. A small border keeps glyphs off the frame edges
    //     (without it the recognizer occasionally drops a leading consonant at
    //     the left edge). The L/R padding from step 5 is separate — applied
    //     post-resize, fixed px.
    let binary = pad_white(&binary, BODY_PAD_PX);
    crate::dump_debug(
        &DynamicImage::ImageLuma8(binary.clone()),
        "padded",
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
///
/// When the whole-width scan finds no gap at this threshold on a side (tight
/// line spacing where every outer-region row carries some ink), the per-chunk
/// fallback fires for that side — narrow column windows are more likely to
/// contain a clean gap where the full width does not.
const BLEED_GAP_DENSITY: f64 = 0.005;

/// Fraction of the peak row density that still counts as "body" when walking
/// outward from the densest row to find the text-body extent. The body is the
/// densest ink region (the actual glyphs); bleed fragments are sparser. 0.30
/// means a row is body if its ink density exceeds 30% of the peak — wide
/// enough to capture ascenders/descenders at the body's edges, tight enough to
/// stop where ink thins into bleed. See [`trim_neighbor_bleed`].
const BLEED_BODY_DENSITY_FRAC: f64 = 0.30;

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
/// perfectly separate them. 0.7 is the sole bleed defense (the source-level
/// box shrink was removed — it moved accurate boxes as much as over-big ones):
/// wide enough to shield most ascenders/diacritics, narrow enough to act on
/// bleed beyond the band. Tune down if bleed survives, up if diacritics get
/// cropped.
const BLEED_PROTECT_FRAC: f64 = 0.70;

/// Target width for the per-chunk white-row gap scan. The strip width is
/// divided into `round(w / TARGET)` equal-width chunks (boundaries by floor
/// division, so every chunk — including the last — is within 1px of the
/// others; ~94–106px on typical ~750–950px strips).
///
/// The full strip width (~800px) often has no row-wide white gap between body
/// and bleed because bleed and body overlap in different columns. A ~100px
/// window is much more likely to contain a clean gap — bleed rarely spans
/// every column within such a window. The scan runs independently per chunk,
/// producing a piecewise-flat (staircase) boundary that adapts to where the
/// gap sits. Chunks with no gap keep everything (safe default).
///
/// Why equal division instead of fixed-width chunks: a fixed width leaves a
/// ragged LEFTOVER tail chunk (a 921px strip → 9 chunks of 100 + one of 21px).
/// In a tail that narrow, a single glyph's internal white band spans the whole
/// window, so the gap scan mistakes the glyph's own top for bleed and cuts it
/// (seen on line 0028: the last glyph's top loop sat in a 21px tail and was
/// severed from its strokes one chunk over). Equal chunks never get that
/// narrow.
const BLEED_CHUNK_TARGET: u32 = 100;

/// Rows of white required between a floating (band-disconnected) ink mark and
/// the strip's top/bottom edge for the mark to count as a gap-separated
/// diacritic of THIS line rather than bleed clipped from a neighbor line.
///
/// Bleed exists because the quad cut through the neighbor's glyphs, so it
/// abuts the strip edge: PP-OCR's unclip pad (~4px) plus Sauvola's erosion of
/// the outermost anti-aliased row put clipped bleed within ~5 rows of the
/// edge (measured on the dump corpus: bleed bands start at rows 0–4). A
/// genuine diacritic has real headroom above it (line 0056's leading vowel
/// curl started at row 10). Combined with the height and width limits
/// ([`BLEED_COMPONENT_FRAC`], [`BLEED_FLOAT_WIDTH_FRAC`]), only marks that are
/// too tall to be a fragment, one glyph wide, and clear of the edges are
/// protected from cuts.
const BLEED_EDGE_CLEARANCE: i32 = 6;

/// Width limit for a floating (band-disconnected) mark to count as a
/// gap-separated diacritic: at most this multiple of the core body height.
///
/// TALL + edge-clear alone is not enough: the historic 0009 bleed was an
/// 11-row, 50px-wide band of neighbor-line fragments floating clear of the
/// edge — taller than half that line's body and 9 rows off the edge. A
/// diacritic is one glyph wide (0056's vowel curl: 28px vs a 41px body; the
/// 0009 band: 50px vs a 16px body — three glyphs). Only marks narrower than
/// one body height earn the floating-diacritic protection.
const BLEED_FLOAT_WIDTH_FRAC: f64 = 1.0;

/// Minimum number of row-aligned floating fragments that constitutes a
/// "ghost line" — the previous line's glyph bottoms hovering above the band.
///
/// One floating mark above the band is ambiguous (0028's gap-separated glyph
/// top must keep its gap-path reprieve), but bleed comes as a LINE: an
/// over-tall quad clips many glyphs of the neighbor line at once, leaving a
/// horizontal band of fragments at the same rows. Line 0004's band had 8
/// fragments; line 0009's had 3, one of which earned individual diacritic
/// protection by a single pixel of height — so the chain pool includes
/// protected marks too (band membership is stronger evidence than shape
/// heuristics), and three aligned fragments already betray a line while
/// legitimate marks have only ever appeared alone. Unprotected floating TOP
/// components whose row spans overlap are chained; a chain this large is
/// killed outright, smaller groups keep the reprieve.
const BLEED_GHOST_MIN_FRAGS: usize = 3;

/// Remove neighbor-line ink that bled into the top/bottom of an over-tall
/// PP-OCR quad, returning the CROPPED text body.
///
/// PP-OCR's quads are sometimes a few pixels taller than the text they bound,
/// capturing fragments of the lines above/below. After binarization this shows
/// up as ink separated from the main text body by whitespace. The separator is
/// found by a **two-tier white-gap scan**:
///
/// 1. **Fast path (whole-width).** The per-row ink density is already computed
///    for body detection, so scanning it for the first row below the gap
///    threshold is O(H) and free. A gap here is white across EVERY column — a
///    true full-width separator — so a flat cut is exact. This handles the
///    common case (clean line spacing) and returns immediately with no
///    chunking overhead.
///
/// 2. **Fallback (per-chunk, tight spacing).** Only reached when one or both
///    sides have no full-width gap. The strip width is divided into equal
///    chunks of ~[`BLEED_CHUNK_TARGET`] px; each chunk is scanned
///    independently for its first white gap. A narrow chunk is far more
///    likely to contain a clean gap than the full width, because bleed and
///    body overlap in fewer columns within a small window. A side that DID
///    find a full-width gap is reused flat across all chunks. For chunks
///    that STILL find no gap (bleed covers every outer row in that window),
///    a connected-component pass drops blobs that float entirely outside the
///    protect band — see step "Contour fallback" below. This finds the
///    boundary LOCALLY, with no borrowing from neighbor chunks.
///
/// The per-chunk boundaries form a piecewise-flat (staircase) cut — the output
/// masks outside-boundary rows to white within a rectangular crop window.
///
/// **Protect band.** The gap scan does NOT start at the body edge. It starts
/// only beyond `[center ± BLEED_PROTECT_FRAC × core_body_h]`. Anything inside
/// that band — including gap-separated diacritics that belong to this line —
/// is kept unconditionally.
///
/// **Body detection.** The line center and core body height are derived from
/// the **densest ink region** (density-peak). Walking outward from the peak
/// while density stays above a fraction of it isolates the body even when bleed
/// and body share no row-wide white gap (tight line spacing).
fn trim_neighbor_bleed(image: &GrayImage) -> GrayImage {
    let (w, h) = image.dimensions();

    // Per-row ink density over the full width. Used for band detection and
    // density-peak body finding.
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

    // Collect contiguous inked bands. If the strip is (near-) empty there's
    // nothing to trim — return unchanged.
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

    // Body = contiguous run around the DENSEST row (density-peak). Lightly
    // smoothed so a single noisy row can't shift the peak.
    let smooth_window = 3usize;
    let mut smoothed = vec![0.0f64; h as usize];
    for y in 0..h as usize {
        let mut sum = 0.0;
        let mut n = 0.0;
        for dy in -(smooth_window as i32)..=(smooth_window as i32) {
            let yy = y as i32 + dy;
            if yy >= 0 && yy < h as i32 {
                sum += density[yy as usize];
                n += 1.0;
            }
        }
        smoothed[y] = sum / n;
    }
    let peak = (0..h as usize)
        .max_by(|&a, &b| smoothed[a].total_cmp(&smoothed[b]))
        .unwrap_or(0);
    let peak_density = smoothed[peak];
    let body_threshold = peak_density * BLEED_BODY_DENSITY_FRAC;
    let mut body_top = peak as i32;
    while body_top > 0 && smoothed[(body_top - 1) as usize] > body_threshold {
        body_top -= 1;
    }
    let mut body_bottom = peak as i32;
    while body_bottom < (h - 1) as i32
        && smoothed[(body_bottom + 1) as usize] > body_threshold
    {
        body_bottom += 1;
    }
    let center = (body_top + body_bottom) / 2;
    let core_body_h = (body_bottom - body_top + 1).max(1);

    // Protect band: within [lo, hi] everything is kept unconditionally.
    let protect = (core_body_h as f64 * BLEED_PROTECT_FRAC).round() as i32;
    let lo = (center - protect).max(0);
    let hi = (center + protect).min((h - 1) as i32);

    // --- Fast path: whole-width white-row gap scan (cheap + exact). ---
    // `density[]` already integrates ink over the full strip width, so a row
    // whose density is below the gap threshold is white across EVERY column —
    // a true row-wide separator. This finds the *global* gap floor/ceiling,
    // which every per-chunk cut below is clamped to (a chunk can only tighten
    // the cut toward the band, never loosen it past the global gap). Note the
    // fast path no longer returns early: it can't catch localized bleed that
    // sits below the global gap row in some chunks (see the per-chunk scan).
    let (flat_top, top_found) = {
        let mut t = lo;
        let mut found = false;
        for r in (0..lo as usize).rev() {
            if density[r] < BLEED_GAP_DENSITY {
                t = r as i32 + 1;
                found = true;
                break;
            }
        }
        // False-gap guard: a gap only at row 0 (the white image border) with
        // inked rows just inside is not a real separator — treat as not-found
        // so the chunked fallback handles it.
        if found && t == 1 && density[1] >= BLEED_GAP_DENSITY {
            found = false;
            t = 0;
        }
        (t, found)
    };
    let (flat_bot, bot_found) = {
        let mut b = hi;
        let mut found = false;
        for r in ((hi + 1) as usize)..h as usize {
            if density[r] < BLEED_GAP_DENSITY {
                b = r as i32 - 1;
                found = true;
                break;
            }
        }
        // False-gap guard (bottom): edge gap with ink just inside → not-found.
        if found && b == (h - 2) as i32 && density[(h - 2) as usize] >= BLEED_GAP_DENSITY {
            found = false;
            b = (h - 1) as i32;
        }
        (b, found)
    };
    // No fast-path early return: even when both global gaps were found, a
    // chunk may carry localized bleed below the global gap row (e.g. a
    // neighbor line's tail reaching only into its columns). Every chunk is
    // scanned independently below, clamped to the fast-path gap so it can
    // only tighten the cut, never loosen it. (Previously the fast path
    // returned a flat cut for all chunks and skipped this loop — localized
    // bleed survived. See `trim_trace.py` / 0009 c0.)
    //
    // Per-chunk white-row gap scan (handles tight line spacing). Divide the
    // width into equal chunks of ~BLEED_CHUNK_TARGET px (floor-division
    // boundaries, all within 1px of each other); for each chunk, find the
    // first white gap row above the protect band (top) and below it (bottom).
    // Chunks with no gap keep everything.
    let n_chunks = ((w as f64 / BLEED_CHUNK_TARGET as f64).round() as u32).max(1) as usize;
    let chunk_bounds: Vec<(u32, u32)> = (0..n_chunks)
        .map(|i| {
            (
                (i as u64 * w as u64 / n_chunks as u64) as u32,
                ((i + 1) as u64 * w as u64 / n_chunks as u64) as u32,
            )
        })
        .collect();
    let mut chunk_top: Vec<i32> = Vec::with_capacity(n_chunks);
    let mut chunk_top_found: Vec<bool> = Vec::with_capacity(n_chunks);
    let mut chunk_bot: Vec<i32> = Vec::with_capacity(n_chunks);
    let mut chunk_bot_found: Vec<bool> = Vec::with_capacity(n_chunks);
    for (ci, &(cx0, cx1)) in chunk_bounds.iter().enumerate() {
        let cw = (cx1 - cx0) as f64;

        // Per-row ink density within this chunk's columns.
        let chunk_dens = |r: usize| -> f64 {
            let mut ink = 0u32;
            for x in cx0..cx1 {
                if image.get_pixel(x, r as u32)[0] < threshold {
                    ink += 1;
                }
            }
            ink as f64 / cw
        };

        // Per-chunk independent scan. We ALWAYS scan each chunk for its own
        // gap, even when the fast path found a whole-width gap — a chunk can
        // carry *localized* bleed that sits below (inside) the global gap row
        // (e.g. a neighbor line's tail reaching only into this chunk's
        // columns). The fast path can't see that, so every chunk must be
        // allowed to tighten its own cut. See `trim_trace.py` for the case
        // (0009 c0): whole-width gap at row 8, but chunk 0 has 40%-density
        // bleed at rows 9-19 with its own gap at row 20.
        //
        // When the fast path DID find a global gap, we clamp the chunk cut so
        // it can only tighten (remove more bleed), never loosen: a chunk top
        // stays >= flat_top, a chunk bottom stays <= flat_bot. This preserves
        // the fast path's correctness for clean chunks while letting bleed-
        // heavy chunks cut deeper.
        let (top, top_found) = {
            let mut t = 0i32;
            let mut found = false;
            for r in (0..lo as usize).rev() {
                if chunk_dens(r) < BLEED_GAP_DENSITY {
                    t = r as i32 + 1;
                    found = true;
                    break;
                }
            }
            // False-gap guard: edge gap (row 0) with inked rows just inside is
            // the white image border, not a real separator → not-found, so the
            // chunk gets the contour fallback instead.
            if found && t == 1 && chunk_dens(1) >= BLEED_GAP_DENSITY {
                found = false;
                t = 0;
            }
            // Merge with fast path: clamp to flat_top so the chunk can only
            // cut deeper (toward the band), never above the global gap.
            if top_found {
                t = t.max(flat_top);
                found = true;
            }
            (t, found)
        };
        // Bottom: same independent scan + fast-path clamp.
        let (bottom, bottom_found) = {
            let mut b = (h - 1) as i32;
            let mut found = false;
            for r in ((hi + 1) as usize)..h as usize {
                if chunk_dens(r) < BLEED_GAP_DENSITY {
                    b = r as i32 - 1;
                    found = true;
                    break;
                }
            }
            // False-gap guard (bottom): edge gap with ink just inside → not-found.
            if found && b == (h - 2) as i32
                && chunk_dens((h - 2) as usize) >= BLEED_GAP_DENSITY
            {
                found = false;
                b = (h - 1) as i32;
            }
            // Merge with fast path: clamp to flat_bot so the chunk can only
            // cut deeper (toward the band), never below the global gap.
            if bot_found {
                b = b.min(flat_bot);
                found = true;
            }
            (b, found)
        };
        chunk_top.push(top);
        chunk_top_found.push(top_found);
        chunk_bot.push(bottom);
        chunk_bot_found.push(bottom_found);
    }

    // Contour fallback for no-gap chunks (tight line spacing). When a chunk
    // found no row-wide white gap, row density can't separate body from bleed
    // — but connectivity can. Run a connected-component pass over the chunk
    // and drop components that float entirely in the outer region (never reach
    // the protect band): those are bleed fragments disconnected from the body.
    // This finds the boundary LOCALLY per chunk — no borrowing from neighbors,
    // so it avoids the "blur" of interpolation when bleed heights differ.
    //
    // A component taller than BLEED_COMPONENT_FRAC × body_h is kept even if it
    // floats — a tall disconnected stroke is more likely a real ascender/
    // diacritic than bleed, so the safe call is to keep it.
    let max_bleed_h = (core_body_h as f64 * BLEED_COMPONENT_FRAC).round() as i32;
    for ci in 0..n_chunks {
        if chunk_top_found[ci] && chunk_bot_found[ci] {
            continue;
        }
        let (cx0, cx1) = chunk_bounds[ci];
        let spans = chunk_outer_components(image, cx0, cx1, lo, hi, threshold);
        // Top side: among components sitting entirely above `lo`, the lowest
        // bottom row defines where bleed ends. Set the boundary just below it.
        if !chunk_top_found[ci] {
            let mut cut_below = 0i32; // keep everything above by default
            let mut found_bleed = false;
            for &(ct, cb) in &spans {
                if cb < lo && (cb - ct + 1) <= max_bleed_h {
                    cut_below = cut_below.max(cb + 1);
                    found_bleed = true;
                }
            }
            if found_bleed {
                chunk_top[ci] = cut_below.max(0).min(lo);
            }
        }
        // Bottom side: mirror — components entirely below `hi`, set the
        // boundary just above the topmost one.
        if !chunk_bot_found[ci] {
            let mut cut_above = (h - 1) as i32; // keep everything below by default
            let mut found_bleed = false;
            for &(ct, cb) in &spans {
                if ct > hi && (cb - ct + 1) <= max_bleed_h {
                    cut_above = cut_above.min(ct - 1);
                    found_bleed = true;
                }
            }
            if found_bleed {
                chunk_bot[ci] = cut_above.max(hi).min((h - 1) as i32);
            }
        }
    }

    // --- Protected-ink clamp: no cut may sever ink that belongs to the line.---
    //
    // Every cut above — whole-width, chunk gap scan, or contour — is a FLAT
    // row boundary, but connectivity is the ground truth for "belongs to this
    // line". One full-strip 8-connectivity pass finds every component that
    // touches the protect band; for each chunk we union the row spans of such
    // components WITHIN the chunk's columns, and clamp its cuts to that
    // extent. This catches what chunk-local analysis cannot: a glyph
    // straddling a chunk boundary (loop in one chunk, stem in the next)
    // connects to the body through the neighbor's columns, so the chunk
    // holding the loop sees a chunk-wide white band beneath it — a false gap.
    //
    // The pass also protects tall, NARROW, edge-clear FLOATING marks
    // (disconnected from the body) — gap-separated diacritics such as line
    // 0056's leading vowel curl, which pokes 2 rows above the protect band
    // with real white above it. Bleed never earns that protection: bands are
    // wider than one body height or hug the edge where the quad clipped them
    // (see [`BLEED_EDGE_CLEARANCE`], [`BLEED_FLOAT_WIDTH_FRAC`]). For
    // genuinely separated bleed the clamp is a no-op, because ink beyond a
    // true separator touches neither.
    let (body_extents, kill) = strip_protected_extents(
        image,
        &chunk_bounds,
        lo,
        hi,
        threshold,
        max_bleed_h,
        core_body_h,
    );
    for ci in 0..n_chunks {
        if let Some((body_top, body_bot)) = body_extents[ci] {
            chunk_top[ci] = chunk_top[ci].min(body_top);
            chunk_bot[ci] = chunk_bot[ci].max(body_bot);
        }
    }

    // Per-chunk margin: keep a proportional white band so the resize doesn't
    // over-stretch the glyphs. Applied per-chunk, walking outward through white
    // rows only.
    let crop_h = {
        let mn = *chunk_top.iter().min().unwrap_or(&0);
        let mx = *chunk_bot.iter().max().unwrap_or(&((h - 1) as i32));
        (mx - mn + 1).max(1)
    };
    let margin = (crop_h as f64 * BLEED_KEEP_MARGIN_FRAC).round() as i32;
    for ci in 0..n_chunks {
        let (cx0, cx1) = chunk_bounds[ci];
        let cw = (cx1 - cx0) as f64;
        let chunk_dens = |r: usize| -> f64 {
            let mut ink = 0u32;
            for x in cx0..cx1 {
                if image.get_pixel(x, r as u32)[0] < threshold {
                    ink += 1;
                }
            }
            ink as f64 / cw
        };
        let mut mt = 0;
        for k in 1..=margin {
            let r = chunk_top[ci] - k;
            if r < 0 || chunk_dens(r as usize) >= BLEED_GAP_DENSITY {
                break;
            }
            mt = k;
        }
        chunk_top[ci] -= mt;
        let mut mb = 0;
        for k in 1..=margin {
            let r = chunk_bot[ci] + k;
            if r >= h as i32 || chunk_dens(r as usize) >= BLEED_GAP_DENSITY {
                break;
            }
            mb = k;
        }
        chunk_bot[ci] += mb;
    }

    // Build the output: mask everything outside the per-chunk boundaries to
    // white, crop to the overall bounding box.
    let crop_top = *chunk_top.iter().min().unwrap_or(&0);
    let crop_bottom = *chunk_bot.iter().max().unwrap_or(&((h - 1) as i32));
    let new_h = (crop_bottom - crop_top + 1).max(1) as u32;
    let mut out = GrayImage::new(w, new_h);
    for (ci, &(cx0, cx1)) in chunk_bounds.iter().enumerate() {
        let t = chunk_top[ci];
        let b = chunk_bot[ci];
        for x in cx0..cx1 {
            for oy in 0..new_h as i32 {
                let sy = crop_top + oy;
                // `sy ∈ [crop_top, crop_bottom] ⊆ [0, h)` by construction; the
                // get() fallback only covers the impossible out-of-range case.
                let killed = kill
                    .get(sy as usize * w as usize + x as usize)
                    .copied()
                    .unwrap_or(false);
                let px = if sy < t || sy > b || killed {
                    Luma([255])
                } else {
                    *image.get_pixel(x, sy as u32)
                };
                out.put_pixel(x, oy as u32, px);
            }
        }
    }
    out
}

/// Fraction of the text-body height to keep as white margin above and below
/// when trimming neighbor-bleed. 0.15 ≈ 15%: enough breathing room that the
/// resize doesn't over-stretch the glyphs, small enough that a typical over-
/// tall quad still sheds its bleed. Tuned for the bundled Burmese model.
const BLEED_KEEP_MARGIN_FRAC: f64 = 0.15;

/// White padding (px) added on all four sides of the trimmed line strip before
/// the height-normalizing resize (step 3b). A small border keeps glyphs off the
/// frame edges: without it the recognizer occasionally drops a leading consonant
/// at the left edge. Added before the resize so it scales with the body into
/// proportional margin in the target-height output.
const BODY_PAD_PX: u32 = 2;

/// Pad a grayscale image with white (255) on all four sides by `pad` pixels.
/// Returns a new, larger image with the original centered.
fn pad_white(image: &GrayImage, pad: u32) -> GrayImage {
    if pad == 0 {
        return image.clone();
    }
    let (w, h) = image.dimensions();
    let mut out = GrayImage::from_pixel(w + 2 * pad, h + 2 * pad, Luma([255]));
    for y in 0..h {
        for x in 0..w {
            out.put_pixel(x + pad, y + pad, *image.get_pixel(x, y));
        }
    }
    out
}

/// When the per-chunk white-row scan finds NO gap in a chunk (tight spacing),
/// fall back to a connected-component pass instead of borrowing a neighbor's
/// boundary. In such a chunk every outer row carries ink, so row density can't
/// separate body from bleed — but the bleed is almost always a distinct
/// connected component that floats near the edge and never reaches the protect
/// band. We drop any component whose height is at most this fraction of the
/// core body height AND that does not touch the protect band (i.e. it lives
/// entirely in the outer region). 0.5 = half the body height: bleed fragments
/// are short, genuine tall ascenders/descenders reach past it.
const BLEED_COMPONENT_FRAC: f64 = 0.5;

/// Per-chunk row extent of ink the trim must NOT remove, from ONE full-strip
/// 8-connectivity flood-fill pass, plus a pixel mask of floating bottom-side
/// ink to remove unconditionally. Two kinds of ink earn protection:
///
/// 1. **Body ink** — any component that touches the protect band `[lo, hi]`,
///    however far its strokes reach or which chunks they pass through.
/// 2. **Clear floating diacritics** — components disconnected from the body
///    that are taller than `max_bleed_h` (too tall to be a bleed fragment),
///    narrower than `core_body_h` (one glyph, not a multi-letter band), AND
///    at least [`BLEED_EDGE_CLEARANCE`] rows clear of the strip's top/bottom
///    edges (bleed is clipped by the quad and hugs the edge; a diacritic has
///    real headroom above it).
///
/// For each chunk in `chunks`, the extent is the union — over protected
/// components — of the component's row span RESTRICTED to the chunk's columns
/// `[cx0, cx1)`. The restriction matters: a component may span several chunks
/// while its tall excursions (ascenders, descenders) sit in only one of them,
/// and a tall stroke in chunk k must not protect unrelated bleed in chunk j
/// that the component merely spans. Chunks with no protected ink get `None`.
///
/// The returned kill mask covers two kinds of floating ink: components
/// entirely BELOW the band (tops of the next line's glyphs — the gap scan
/// alone cannot drop these, because a blob adjacent to the descender zone
/// inks the rows between itself and the band, so the "first white row" lands
/// below the blob and the flat cut keeps it, line 0056), and "ghost lines"
/// ABOVE the band — row-aligned chains of at least [`BLEED_GHOST_MIN_FRAGS`]
/// unprotected fragments, the previous line's glyph bottoms (line 0004).
/// Lone floating top marks keep the gap-path reprieve: Burmese stacks
/// gap-separated diacritics above the text (0028's glyph-top loop).
///
/// `trim_neighbor_bleed` clamps its flat cuts to the extents (so a cut can
/// never sever protected ink — including glyphs whose connection to the body
/// routes through a NEIGHBORING chunk's columns, which no chunk-local scan
/// can see) and whites out the kill mask in the composited output.
/// Ink = pixel < `threshold`.
fn strip_protected_extents(
    image: &GrayImage,
    chunks: &[(u32, u32)],
    lo: i32,
    hi: i32,
    threshold: u8,
    max_bleed_h: i32,
    core_body_h: i32,
) -> (Vec<Option<(i32, i32)>>, Vec<bool>) {
    let (w, h) = image.dimensions();
    let hi_edge = h as i32 - 1 - BLEED_EDGE_CLEARANCE;
    let n_chunks = chunks.len();
    let mut extents: Vec<Option<(i32, i32)>> = vec![None; n_chunks];
    let mut kill = vec![false; (w as usize) * (h as usize)];
    // Column → chunk lookup (the chunks tile [0, w) as contiguous intervals).
    let mut col_chunk = vec![usize::MAX; w as usize];
    for (ci, &(cx0, cx1)) in chunks.iter().enumerate() {
        for x in cx0..cx1 {
            col_chunk[x as usize] = ci;
        }
    }
    let mut visited = vec![false; (w as usize) * (h as usize)];
    let idx = |x: u32, y: u32| (y as usize * w as usize + x as usize) as usize;
    // Unprotected floating TOP components, deferred to the ghost-line pass.
    let mut top_floats: Vec<(i32, i32, Vec<(u32, u32)>)> = Vec::new();

    for sy in 0..h {
        for sx in 0..w {
            if visited[idx(sx, sy)] || image.get_pixel(sx, sy)[0] >= threshold {
                continue;
            }
            // Flood-fill this component, recording its pixel list, its
            // row/column span, its per-chunk row spans, and whether it
            // reaches the protect band.
            let mut stack = vec![(sx, sy)];
            let mut comp_px: Vec<(u32, u32)> = Vec::new();
            let mut comp_spans: Vec<Option<(i32, i32)>> = vec![None; n_chunks];
            let mut comp_top = sy as i32;
            let mut comp_bot = sy as i32;
            let mut comp_x0 = sx;
            let mut comp_x1 = sx;
            let mut touches_band = false;
            while let Some((x, y)) = stack.pop() {
                if x >= w || y >= h {
                    continue;
                }
                if visited[idx(x, y)] {
                    continue;
                }
                if image.get_pixel(x, y)[0] >= threshold {
                    continue;
                }
                visited[idx(x, y)] = true;
                comp_px.push((x, y));
                comp_top = comp_top.min(y as i32);
                comp_bot = comp_bot.max(y as i32);
                comp_x0 = comp_x0.min(x);
                comp_x1 = comp_x1.max(x);
                if let Some(&ci) = col_chunk.get(x as usize) {
                    if ci != usize::MAX {
                        comp_spans[ci] = Some(match comp_spans[ci] {
                            None => (y as i32, y as i32),
                            Some((t, b)) => (t.min(y as i32), b.max(y as i32)),
                        });
                    }
                }
                if y as i32 >= lo && y as i32 <= hi {
                    touches_band = true;
                }
                for dx in -1i32..=1 {
                    for dy in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
                            stack.push((nx as u32, ny as u32));
                        }
                    }
                }
            }
            // Protection test: body ink (band-touching), or a tall, narrow,
            // edge-clear floating mark (gap-separated diacritic).
            let tall = comp_bot - comp_top + 1 > max_bleed_h;
            let narrow = (comp_x1 - comp_x0 + 1) as f64
                <= core_body_h as f64 * BLEED_FLOAT_WIDTH_FRAC;
            let clear = comp_top >= BLEED_EDGE_CLEARANCE && comp_bot <= hi_edge;
            let protected_float = tall && narrow && clear;
            if !touches_band {
                // Floating BOTTOM-side ink that is not a protected mark is
                // next-line bleed — kill its pixels outright. A flat cut
                // cannot remove it: the blob inks the rows between itself
                // and the band, so the gap scan's "first white row" lands
                // below the blob (line 0056 kept three such blobs).
                if comp_top > hi && !protected_float {
                    for &(x, y) in &comp_px {
                        kill[idx(x, y)] = true;
                    }
                    continue;
                }
                if comp_bot < lo {
                    // Floating TOP-side ink goes into the ghost pool
                    // REGARDLESS of individual protection: a row-aligned
                    // band of >= BLEED_GHOST_MIN_FRAGS fragments is the
                    // previous line's bottoms even when one fragment is
                    // tall/narrow/clear by a hair (line 0009). Until such a
                    // band forms, each mark keeps its own protection (0028).
                    if !protected_float {
                        top_floats.push((comp_top, comp_bot, comp_px));
                        continue;
                    }
                    top_floats.push((comp_top, comp_bot, comp_px.clone()));
                }
            }
            for ci in 0..n_chunks {
                if let Some(span) = comp_spans[ci] {
                    extents[ci] = Some(match extents[ci] {
                        None => span,
                        Some((t, b)) => (t.min(span.0), b.max(span.1)),
                    });
                }
            }
        }
    }

    // Ghost-line pass: chain the deferred top floats whose row spans overlap
    // (union-find over interval intersection) and kill any chain holding at
    // least BLEED_GHOST_MIN_FRAGS fragments — a horizontal band of floating
    // marks is the previous line's glyph bottoms, not diacritics. Chains stay
    // small (few floats, O(n²) pairs) so the simple double loop is fine.
    let n_floats = top_floats.len();
    if n_floats >= BLEED_GHOST_MIN_FRAGS {
        let mut parent: Vec<usize> = (0..n_floats).collect();
        fn find(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        for i in 0..n_floats {
            for j in (i + 1)..n_floats {
                let (a0, a1, _) = top_floats[i];
                let (b0, b1, _) = top_floats[j];
                if a0 <= b1 && b0 <= a1 {
                    let ra = find(&mut parent, i);
                    let rb = find(&mut parent, j);
                    parent[ra] = rb;
                }
            }
        }
        let mut counts = vec![0usize; n_floats];
        for i in 0..n_floats {
            counts[find(&mut parent, i)] += 1;
        }
        for i in 0..n_floats {
            if counts[find(&mut parent, i)] >= BLEED_GHOST_MIN_FRAGS {
                for &(x, y) in &top_floats[i].2 {
                    kill[idx(x, y)] = true;
                }
            }
        }
    }
    (extents, kill)
}

/// Find ink components that live entirely OUTSIDE the protect band `[lo, hi]`
/// within a chunk's column range `[cx0, cx1)`, and return their row spans.
///
/// These are candidate bleed: connected blobs not touching the body (which
/// spans the protect band). In a no-gap chunk every outer row carries ink, so
/// row density cannot separate body from bleed — but connectivity can: the
/// bleed is almost always a distinct component floating near the edge.
///
/// Returns a `Vec<(top, bottom)>` of inclusive row spans for every component
/// whose rows are entirely above `lo` OR entirely below `hi`. The caller picks
/// the top-side / bottom-side components to set the chunk boundary locally — no
/// borrowing from neighbors, so it avoids the "blur" of interpolation when bleed
/// heights differ across columns. (The cut the caller derives is subsequently
/// clamped by [`strip_protected_extents`] — see its docs.)
///
/// The labeling is a simple 8-connectivity flood fill restricted to the chunk's
/// columns. Ink = pixel < `threshold`; white = otherwise.
fn chunk_outer_components(
    image: &GrayImage,
    cx0: u32,
    cx1: u32,
    lo: i32,
    hi: i32,
    threshold: u8,
) -> Vec<(i32, i32)> {
    let h = image.height() as i32;
    let cw = cx1 - cx0;
    // visited grid over the chunk's columns × full height.
    let mut visited = vec![false; (cw * h as u32) as usize];
    let mut spans: Vec<(i32, i32)> = Vec::new();

    let idx = |x: u32, y: i32| -> usize {
        (y as u32 * cw + (x - cx0)) as usize
    };

    for sx in cx0..cx1 {
        for sy in 0..h {
            if visited[idx(sx, sy)] {
                continue;
            }
            if image.get_pixel(sx, sy as u32)[0] >= threshold {
                continue;
            }
            // Flood-fill this component (8-connectivity), tracking its row span
            // and whether it touches the protect band.
            let mut stack = vec![(sx, sy)];
            let mut comp_top = sy;
            let mut comp_bot = sy;
            let mut touches_band = false;
            while let Some((x, y)) = stack.pop() {
                if x < cx0 || x >= cx1 || y < 0 || y >= h {
                    continue;
                }
                if visited[idx(x, y)] {
                    continue;
                }
                if image.get_pixel(x, y as u32)[0] >= threshold {
                    continue;
                }
                visited[idx(x, y)] = true;
                comp_top = comp_top.min(y);
                comp_bot = comp_bot.max(y);
                if y >= lo && y <= hi {
                    touches_band = true;
                }
                for dx in -1i32..=1 {
                    for dy in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y + dy;
                        if nx >= cx0 as i32 && nx < cx1 as i32 && ny >= 0 && ny < h {
                            stack.push((nx as u32, ny));
                        }
                    }
                }
            }
            // A component that never reaches the protect band is candidate
            // bleed — it floats in the outer region disconnected from the body.
            if !touches_band {
                spans.push((comp_top, comp_bot));
            }
        }
    }
    spans
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
        // A 200-wide × 40-tall strip with a contiguous text body in rows
        // 10..19 (12 cols of ink per row = 240 ink px > BLEED_GAP_DENSITY), plus
        // "bleed" ink far above (rows 0..1) and far below (rows 34..35) —
        // realistic spacing where bleed sits well outside the protect band.
        // The trim should drop the bleed rows and keep the body.
        let (w, h) = (200u32, 40u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Main body rows 10..19, columns 10..190 (180 cols of ink).
        for y in 10..=19 {
            for x in 10..190 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Bleed above (rows 0..1), sparse (20 cols — under the gap threshold).
        for x in 30..50 {
            img.put_pixel(x, 0, Luma([0]));
            img.put_pixel(x, 1, Luma([0]));
        }
        // Bleed below (rows 34..35), sparse.
        for x in 80..100 {
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
        // Body ink must survive: 10 rows × 180 cols = 1800 px. Bleed is
        // 2×20 + 2×20 = 80 px; if it leaked, count would exceed 1800.
        let body_ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            body_ink, 1800,
            "bleed ink leaked into output (expected exactly 1800 body px, got {body_ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_noop_on_no_gap() {
        // When ink fills the whole strip with no white gap, nothing should be
        // trimmed (the whole strip is the text body). Guards against trimming
        // legitimate full-height ink. Height is unchanged. Width must exceed
        // BLEED_GAP_DENSITY so rows register as inked.
        let (w, h) = (200u32, 8u32);
        let img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([0]));
        let out = trim_neighbor_bleed(&img);
        assert_eq!(out.dimensions(), (w, h), "full-ink strip height should be unchanged");
        assert!(out.iter().all(|&p| p == 0), "full-ink strip was wrongly trimmed");
    }

    #[test]
    fn test_trim_neighbor_bleed_preserves_tall_diacritic() {
        // A tall vertical stroke (like a Burmese vowel sign) extending above
        // the body but staying WITHIN the protect band must survive. The body
        // is dense (180 cols of ink) so density-peak detects it; the stroke
        // is a narrow column but connected, so the forbidden mask protects it.
        let (w, h) = (200u32, 20u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Body rows 8..12 (dense, columns 10..190).
        for y in 8..=12 {
            for x in 10..190 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Tall stroke from row 5 to 12 at col 100 (within the protect band).
        for y in 5..=12 {
            img.put_pixel(100, y, Luma([0]));
        }

        let out = trim_neighbor_bleed(&img);

        // The stroke must survive: its pixels at col 100 must be dark.
        let stroke_survives = (0..out.height()).any(|y| out.get_pixel(100, y)[0] == 0);
        assert!(stroke_survives, "tall diacritic at col 100 was entirely cropped");
    }

    #[test]
    fn test_trim_neighbor_bleed_preserves_gap_separated_diacritic() {
        // A diacritic (e.g. a Myanmar vowel sign in a font style that doesn't
        // touch the consonant) sitting ABOVE the body, separated from it by a
        // real white gap. The diacritic's ink count per row is under the gap
        // threshold (BLEED_GAP_DENSITY), so the horizontal scan sees it as part of
        // a gap — but the connected-component forbidden mask protects it because
        // it may be connected to the body. Here the diacritic is deliberately
        // NOT connected (gap-separated), so the scan will crop it — this is the
        // expected trade-off: gap-separated marks above the protect band are
        // treated as bleed. The test verifies the BODY survives regardless.
        let (w, h) = (200u32, 40u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Main body rows 10..19, columns 10..190 (180 ink px, above threshold).
        for y in 10..=19 {
            for x in 10..190 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Gap-separated diacritic at rows 7..8 (sparse, 20 cols — under threshold).
        for y in 7..=8 {
            for x in 60..80 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // The body must survive fully: 10 rows × 180 cols = 1800 px.
        let body_ink = out.iter().filter(|&&p| p == 0).count();
        assert!(
            body_ink >= 1800,
            "body must survive (expected ≥1800 px, got {body_ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_crops_asymmetric_top_bleed() {
        // Bleed ONLY at the top (none at the bottom). `trim_neighbor_bleed` is
        // the sole bleed defense (the source-level box shrink was removed — it
        // moved accurate boxes as much as over-big ones). The protect band
        // (0.70) shields ascenders/diacritics close to the body; this test
        // places the bleed far enough above the body that it falls outside the
        // band and gets cropped.
        // Body rows 16..35 (center 25, core 22, protect 15 → lo=10), bleed rows
        // 0..3 — well above lo=10, so the gap scan cuts it.
        let (w, h) = (200u32, 50u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Main body rows 16..35, columns 10..190 (180 cols, well above threshold).
        for y in 16..=35 {
            for x in 10..190 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Asymmetric bleed above only (rows 0..3), sparse (20 cols).
        for x in 30..50 {
            for y in 0..=3 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body ink = 20 rows × 180 cols = 3600 px. Bleed = 4×20 = 80 px, sitting
        // far above the protect band → cropped. If it leaked, count > 3600.
        let body_ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            body_ink, 3600,
            "far bleed should still be cropped (expected exactly 3600 body px, got {body_ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_chunked_scan() {
        // Fast path with a clean full-width gap: bleed leaves rows 11-15 white
        // across the ENTIRE width, so the whole-width scan finds the separator
        // directly (no chunking needed). Verifies the fast path crops the bleed
        // and preserves the body exactly.
        //
        // Layout (300 wide × 40 tall):
        //   Body:      rows 16-25, columns 10..290 (280 ink px/row).
        //   Top bleed: rows 0-10, columns 20..200 (180 px/row).
        let (w, h) = (300u32, 40u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Body: rows 16-25, columns 10..290.
        for y in 16..=25 {
            for x in 10..290 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Top bleed: rows 0-10, columns 20..200.
        for y in 0..=10 {
            for x in 20..200 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body must survive: 10 rows × 280 cols = 2800 px.
        let body_ink = out.iter().filter(|&&p| p == 0).count();
        assert!(
            body_ink >= 2800,
            "body must survive fully (expected ≥2800 px, got {body_ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_chunked_fallback_no_full_width_gap() {
        // Forces the per-chunk CONTOUR fallback. In chunk 0 (cols 0-99), three
        // small bleed blobs are staggered across rows so that EVERY top row
        // carries ink (no chunk-wide gap), but each blob is individually short
        // and near the edge. Row density can't separate them from the body, but
        // connectivity can: each blob is a distinct component floating in the
        // outer region, never reaching the protect band.
        //
        // Layout (300 wide × 50 tall):
        //   Body:      rows 20-29, columns 10..290 (280 ink px/row).
        //   Blob A:    cols 10-20,  rows 0-3   (in chunk 0)
        //   Blob B:    cols 30-40,  rows 4-7   (in chunk 0)
        //   Blob C:    cols 50-60,  rows 8-11  (in chunk 0)
        //   Chunks 1-2 are clean (no bleed).
        //
        // Body center=24, core=12, protect(0.70)=8 → lo=16. The blobs (rows 0-11)
        // sit well above lo=16, outside the protect band, so the contour fallback
        // can drop them.
        // Every row 0-11 has ≥11 ink cols → full-width density ≥ 3.7% > thresh,
        // so the whole-width fast path finds no gap on top. Chunk 1 (cols
        // 100-199) is clean → finds a gap. Chunk 0 has staggered blobs → no
        // chunk-wide gap → contour fallback fires, finds the 3 short disconnected
        // components, and drops them.
        let (w, h) = (300u32, 50u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Body: rows 20-29, columns 10..290.
        for y in 20..=29 {
            for x in 10..290 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Staggered bleed blobs in chunk 0 (cols 0-99). Each is 11 cols × 4 rows,
        // at different row bands so every row 0-11 is covered.
        let blobs = [(10u32..=20, 0i32..=3), (30..=40, 4..=7), (50..=60, 8..=11)];
        for (xs, ys) in blobs {
            for y in ys {
                for x in xs.clone() {
                    img.put_pixel(x, y as u32, Luma([0]));
                }
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body ink = 10 rows × 280 cols = 2800 px. Must survive fully. The three
        // blobs total 3 × (11 cols × 4 rows) = 132 px; the contour fallback must
        // drop them since each is a short disconnected component outside the band.
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 2800,
            "contour fallback must drop staggered blobs (expected exactly 2800 body px, got {ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_keeps_sparse_descenders_at_edge_gap() {
        // Regression for line 0005 (thawzin_02): legitimate tapering
        // descender legs end in a genuine white gap at the very bottom edge.
        // The false-gap guard must NOT reject that gap — sparse descenders
        // (density well below body threshold) are real text, not dense bleed.
        //
        // Layout (300 wide × 60 tall):
        //   Body:      rows 10-29, columns 10..290 (dense, 280 ink px/row).
        //   Descenders: rows 31-57, a few narrow strokes tapering off (3 px each
        //              per row → full-width density 3/300 = 1% < body 30%).
        //   Edge gap:   row 58 white (the only full-width gap, at h-2).
        //
        // The bottom fast-path scan finds the gap at row 58 → flat_bot=57 (=h-2).
        // The guard checks density[h-2]=density[57]. Descender strokes there are
        // ~1% << body_threshold, so the guard must KEEP the gap (not fire), and
        // the flat cut preserves all descender rows 31-57.
        let (w, h) = (300u32, 60u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Body: rows 10-29, columns 10..290.
        for y in 10..=29 {
            for x in 10..290 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Tapering descender strokes: narrow verticals at cols 40, 90, 140,
        // 190, 240, reaching rows 31..57 (sparse: 5 ink px per row → 1.7%).
        let stroke_cols = [40u32, 90, 140, 190, 240];
        for x in stroke_cols {
            for y in 31..=57 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body = 20 rows × 280 cols = 5600 px. Descenders = 5 cols × 27 rows =
        // 135 px. Total = 5735 px — all must survive. If the guard wrongly
        // fired, the chunked path would cut through the sparse descenders and
        // shed most of those 135 px.
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 5735,
            "sparse descenders at a real edge gap must be kept (expected 5735 px, got {ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_localized_below_global_gap() {
        // Regression for the fast-path short-circuit (0009 c0). A whole-width
        // gap exists at row 8 (rows 0-8 clean across the FULL width), so the
        // fast path finds flat_top=9. BUT chunk 0 carries localized bleed at
        // rows 9-19 that sits BELOW the global gap row — the fast path's flat
        // cut can't reach it, and previously the fast path's early return
        // skipped the per-chunk scan entirely, so the bleed survived.
        //
        // chunk 0 has its OWN gap at row 20 (white across chunk 0's columns),
        // which the per-chunk scan finds → chunk cut at 21, clamped to
        // max(21, flat_top=9) = 21. The bleed (rows 9-19) is dropped.
        //
        // The key wrinkle that makes this a real bug: chunk 1 has scattered
        // light ink at rows 9-29 (cols 100-104, 5 px/row), so the full-width
        // density at those rows is 27.5% (chunk 0 dense) to 2.5% (chunk 0
        // white) — all ABOVE the 0.5% gap threshold, so there is NO global
        // gap between the bleed and the body. The only global gap is row 8,
        // far above the bleed. chunk 0's gap at row 20 is not whole-width
        // (chunk 1 still has ink), so only the per-chunk scan can see it.
        //
        // Layout (200 wide × 50 tall, 2 chunks of 100):
        //   Global gap:   rows 0-8 (white across full width).
        //   chunk0 bleed: rows 9-19, cols 10-59  (50 px/row, dense).
        //   chunk0 gap:   rows 20-29 (white in chunk 0).
        //   chunk1 scatter: rows 9-29, cols 100-104 (5 px/row — blocks the
        //                   global gap so the fast path can't help).
        //   Body:         rows 30-39, cols 10-190 (181 px/row).
        let (w, h) = (200u32, 50u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        // Body: rows 30-39, cols 10-190.
        for y in 30..=39 {
            for x in 10..=190 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // chunk 0 localized bleed: rows 9-19, cols 10-59.
        for y in 9..=19 {
            for x in 10..=59 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // chunk 1 scattered ink: rows 9-29, cols 100-104 (blocks the global gap).
        for y in 9..=29 {
            for x in 100..=104 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // With the fix: chunk 0 bleed (550 px = 50 cols × 11 rows) is trimmed
        // by the per-chunk scan; body (1810) + chunk1 scatter (105) survive.
        // (Without the fix, the fast-path early return left flat_top=9 for
        // every chunk → 2465 px, and this assertion would fail.)
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 1915,
            "localized bleed below the global gap must be trimmed by the per-chunk scan \
             (expected 1915 px = body + chunk1 scatter, got {ink} — extra ink is \
             un-trimmed chunk-0 bleed)"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_no_ragged_tail_chunk_cuts_glyph_top() {
        // Regression for line 0028: fixed 100px chunks leave a ragged tail
        // chunk (330px strip -> tail of 30px). A single glyph sitting in that
        // tail has its own internal white band spanning the whole window, so
        // the chunk gap scan severed the glyph's top from its strokes one
        // chunk over. The equal split (3 chunks of 110px here) never produces
        // a tail that narrow: the ascender ink at cols 250-255 shares the
        // last chunk with the glyph top, breaking the false gap.
        //
        // Layout (330 wide × 50 tall):
        //   Body:      rows 25-34, cols 10-320 (311 ink px/row).
        //   Ascender:  cols 250-255, rows 8-24 (keeps rows 8-24 from forming a
        //              whole-width gap; lives in the same last chunk as the
        //              glyph top).
        //   Glyph top: rows 8-11, cols 302-312 — separated from the body by
        //              white rows 12-24 within cols 296-330 (the "internal
        //              band" that fooled the old 30px tail chunk).
        let (w, h) = (330u32, 50u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in 25..=34 {
            for x in 10..=320 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 8..=24 {
            for x in 250..=255 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 8..=11 {
            for x in 302..=312 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body 3110 + ascender 102 + glyph top 44 = 3256 — the glyph top must
        // survive. (Under fixed-100 chunking the 44px top was cut: 3212.)
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 3256,
            "glyph top in the former ragged tail must survive (expected 3256 px, got {ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_contour_cut_never_severs_connected_ink() {
        // Regression for line 0010 (chunk c1): in a no-gap chunk the contour
        // fallback sets a FLAT cut "just above the topmost floating bleed
        // component". A single stray pixel high in the outer region dragged
        // that cut 7 rows into a descender's tail, severing ink CONNECTED to
        // the body while the actual bleed sat far lower. The cut must be
        // clamped to the row extent of band-connected components: bleed below
        // the lowest connected ink still goes; connected ink never does.
        //
        // Layout (90 wide × 60 tall, single chunk; rows 34-59 all carry ink so
        // no white-row gap exists and the contour fallback fires):
        //   Body:      rows 10-29, cols 10-80.
        //   Descender: col 60, rows 30-52 — connected to the body, reaches
        //              18 rows past the band's bottom edge (hi=34).
        //   Speck:     1 px at (49, 30) — floating; the old cut placed itself
        //              just above this (row 48), severing the descender.
        //   Mini/blob/blob2: rows 53-55/56-57/58-59 — the real bleed.
        let (w, h) = (90u32, 60u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in 10..=29 {
            for x in 10..=80 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 30..=52 {
            img.put_pixel(60, y, Luma([0]));
        }
        img.put_pixel(30, 49, Luma([0])); // the speck
        for y in 53..=55 {
            img.put_pixel(70, y, Luma([0]));
        }
        for y in 56..=57 {
            for x in 20..=25 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 58..=59 {
            img.put_pixel(40, y, Luma([0]));
        }

        let out = trim_neighbor_bleed(&img);

        // Body 1420 + descender 23 = 1443 must survive: the cut clamps to
        // the descender's bottom (row 52) and the bottom-side floating speck
        // is killed outright by the strip pass. (Unclamped and un-killed:
        // 1444 with 5px of bleed surviving; a plain flat cut at 48 gave 1439.)
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 1443,
            "contour cut must not sever body-connected ink, bottom float must die \
             (expected 1443 px, got {ink})"
        );
        // The descender survives to its tip at row 52.
        assert_eq!(
            (0..out.height()).filter(|&y| out.get_pixel(60, y)[0] == 0).count(),
            43,
            "descender rows 10..=52 must all survive"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_chunk_gap_never_severs_straddling_glyph() {
        // Regression (review finding): a glyph straddling a chunk boundary
        // connects to the body through the NEIGHBOR chunk's columns. The chunk
        // holding its upper loop then sees a chunk-wide white band beneath the
        // loop — a false gap — and the gap scan cut the loop. 150px strip →
        // two 75px chunks; loop cols 68-74 in chunk 0, stem cols 75-80 in
        // chunk 1. The full-strip body clamp (strip_body_extents) sees the
        // loop+stem+body as ONE band-touching component overlapping both
        // chunks, so neither chunk's cut may enter it.
        //
        // Layout (150 wide × 40 tall):
        //   Body:  rows 20-29, cols 10-140.
        //   Loop:  rows 8-11, cols 68-74 (chunk 0 only) — diagonally adjacent
        //          to the stem at (74,11)↔(75,12).
        //   Stem:  rows 12-19, cols 75-80 (chunk 1), touching the body at
        //          rows 19/20.
        let (w, h) = (150u32, 40u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in 20..=29 {
            for x in 10..=140 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 8..=11 {
            for x in 68..=74 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 12..=19 {
            for x in 75..=80 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body 1310 + loop 28 + stem 48 = 1386 — all must survive. (Without
        // the clamp, chunk 0's gap scan cut at row 12 and dropped the loop:
        // 1358.)
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 1386,
            "straddling glyph's loop must survive the chunk gap scan (expected 1386 px, got {ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_keeps_clear_floating_diacritic_cuts_edge_bleed() {
        // Regression for line 0056: the first character's vowel curl (rows
        // 10-40, 31 tall, 28 wide) is gap-separated from the body and pokes
        // 2 rows above the protect band, so the whole-width gap scan's flat
        // cut removed it as bleed. A floating mark that is taller than a
        // bleed fragment can be, narrower than one body height, and clear of
        // the strip edges is a diacritic and must survive — while a band of
        // equal height that is wide (multi-letter neighbor ink, 0009-style)
        // and hugs the top edge (quad-clipped, 0049-style) must still go.
        //
        // Layout (300 wide × 100 tall, 3 chunks of 100; body sits low like
        // an over-tall quad, core=28, lo=42, max_bleed_h=14):
        //   Body:   rows 50-75, cols 10-290.
        //   Mark:   rows 10-40, cols 30-54 — 31 tall, 25 wide, 10 rows
        //           clear of the top edge, 1-row gap to the band → KEPT.
        //   Band:   rows 0-30, cols 150-250 — 31 tall but 101 wide and
        //           touching row 0 → CUT.
        let (w, h) = (300u32, 100u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in 50..=75 {
            for x in 10..=290 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 10..=40 {
            for x in 30..=54 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 0..=30 {
            for x in 150..=250 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body 26×281=7306 + mark 31×25=775 = 8081 survive; the band (3131px)
        // is dropped. (Without the floating-mark protection the mark died:
        // 7306.)
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 8081,
            "clear floating diacritic must survive, wide/edge bleed must not \
             (expected 8081 px, got {ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_kills_bottom_float_that_blocks_gap_scan() {
        // Regression for line 0056 (c0/c4): a floating bleed blob sitting
        // directly under the descender zone inks every row between the band
        // and itself, so the gap scan's "first white row" lands BELOW the
        // blob and the flat cut keeps it — the blob blocks its own separator.
        // The strip pass kills bottom-side floating ink outright, while the
        // connected descender keeps everything.
        //
        // Layout (150 wide × 60 tall, chunks [0,75) and [75,150); body rows
        // 19-41, band lo=14 hi=46, max_bleed_h=12):
        //   Body:     rows 20-40, cols 10-140.
        //   Descender: col 60, rows 41-50 — connected, ends 4 past the band.
        //   Blob:     rows 47-54, cols 100-115 — floating, first row hi+1:
        //             without the kill, chunk 1's scan found its first white
        //             row at 55 and kept rows ≤ 54 — blob included.
        let (w, h) = (150u32, 60u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in 20..=40 {
            for x in 10..=140 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 41..=50 {
            img.put_pixel(60, y, Luma([0]));
        }
        for y in 47..=54 {
            for x in 100..=115 {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body 21×131=2751 + descender 10 = 2761 survive; the blob (128px)
        // is killed. (Without the kill: 2889 — the blob survived whole.)
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 2761,
            "bottom floating blob must not block its own gap (expected 2761 px, got {ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_kills_ghost_line_of_top_fragments() {
        // Regression for line 0004 (c0): a row-aligned band of floating
        // fragments above the band — the previous line's glyph bottoms —
        // blocks the gap scan's white row (the fragments reach down to
        // lo-1), so the flat cut landed ABOVE them and kept them. One
        // floating mark is ambiguous (0028's glyph top keeps its reprieve),
        // but a horizontal band of >= BLEED_GHOST_MIN_FRAGS fragments is a
        // line's worth of neighbor ink and dies. The lone mark and the
        // band-connected ascender survive untouched.
        //
        // Layout (300 wide × 60 tall, band lo=21 hi=59):
        //   Body:     rows 30-55, cols 10-290.
        //   Ascender: rows 5-29, cols 50-55 — connected to the body.
        //   Lone:     rows 5-9, cols 70-80 — floats above the ascender;
        //             its chunk's scan finds white only at row 4, so the
        //             cut keeps it (the 0028 reprieve).
        //   Frags:    rows 10-20 at cols 130-142, 170-182, 210-222,
        //             250-262 — four fragments, rows 10-20 = lo-1 band;
        //             each chunk's scan finds white only at row 9.
        let (w, h) = (300u32, 60u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in 30..=55 {
            for x in 10..=290 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 5..=29 {
            for x in 50..=55 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 5..=9 {
            for x in 70..=80 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for &c0 in &[130u32, 170, 210, 250] {
            for y in 10..=20 {
                for x in c0..c0 + 12 {
                    img.put_pixel(x, y, Luma([0]));
                }
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body 7306 + ascender 150 + lone 55 = 7511 survive; the 572px ghost
        // band is killed. (Without the ghost pass the frags survived: 8083.)
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 7511,
            "ghost line of top fragments must die, lone float must live \
             (expected 7511 px, got {ink})"
        );
    }

    #[test]
    fn test_trim_neighbor_bleed_ghost_band_overrides_diacritic_protection() {
        // Regression for line 0009 (c0): three row-aligned fragments formed
        // the previous line's ghost band, but one was 11px tall vs
        // max_bleed_h=10 — individually "protected" by a hair — and the
        // chain of the remaining two fell short of the threshold, so all
        // three survived. Band membership is stronger evidence than shape
        // heuristics: the ghost pool includes protected marks, and a chain
        // of three already betrays a line (legitimate marks have only ever
        // appeared alone — 0028).
        //
        // Layout (300 wide × 60 tall, band lo=23 hi=59, max_bleed_h=14):
        //   Body:     rows 30-55, cols 10-290.
        //   Frag A/B: rows 7-16, cols 60-72 and 130-142.
        //   Tall frag: rows 6-20, cols 200-212 — 15 tall (> 14), narrow,
        //              edge-clear: individually protected, still banded.
        let (w, h) = (300u32, 60u32);
        let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255]));
        for y in 30..=55 {
            for x in 10..=290 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for &(y0, y1, x0) in &[(7u32, 16, 60u32), (7, 16, 130), (6, 20, 200)] {
            for y in y0..=y1 {
                for x in x0..x0 + 12 {
                    img.put_pixel(x, y, Luma([0]));
                }
            }
        }

        let out = trim_neighbor_bleed(&img);

        // Body 26×281 = 7306 survive; all three band members (455px) die,
        // the tall one included. (Pre-fix: the chain held only the two
        // unprotected frags — below the old threshold of 4 — all survived.)
        let ink = out.iter().filter(|&&p| p == 0).count();
        assert_eq!(
            ink, 7306,
            "ghost band must override individual diacritic protection \
             (expected 7306 px, got {ink})"
        );
    }

    #[test]
    fn test_pad_white_adds_border_and_centers_image() {
        // A 4×3 image with one ink pixel; pad by 2 on all sides → 8×7, original
        // centered, surrounding border all white.
        let mut img: GrayImage = GrayImage::from_pixel(4, 3, Luma([255]));
        img.put_pixel(1, 1, Luma([0]));
        let out = pad_white(&img, 2);
        assert_eq!(out.dimensions(), (8, 7), "padded dims = (w+2p, h+2p)");
        // Ink pixel moves from (1,1) to (1+2, 1+2) = (3,3).
        assert_eq!(out.get_pixel(3, 3)[0], 0, "ink pixel centered");
        // All border pixels white.
        assert_eq!(out.get_pixel(0, 0)[0], 255, "top-left border white");
        assert_eq!(out.get_pixel(7, 6)[0], 255, "bottom-right border white");
    }

    #[test]
    fn test_pad_white_zero_is_clone() {
        // pad=0 returns a clone of the input (no border added).
        let img: GrayImage = GrayImage::from_pixel(3, 3, Luma([128]));
        let out = pad_white(&img, 0);
        assert_eq!(out.dimensions(), (3, 3));
        assert!(out.iter().all(|&p| p == 128));
    }
}

