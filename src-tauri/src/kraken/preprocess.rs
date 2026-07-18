//! Image preprocessing: resize, pad, normalize, invert.

use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array2, Array3};

/// Preprocessed input: tensor [1, C, H, W] and grayscale scal_im.
pub struct Preprocessed {
    /// CHW tensor, shape [C, H, W], ready for ONNX (batch dim added at inference).
    pub tensor: Array3<f32>,
    /// Grayscale scaled image (H, W), pre-inversion.
    pub scal_im: Array2<f32>,
    /// Width of the actual image content (before width padding for fixed models).
    /// When 0, no width padding was applied. Used to crop the heatmap back.
    pub content_w: usize,
}

/// Preprocess an image for the segmentation model.
///
/// Args:
/// - `image`: Input image (any format DynamicImage supports).
/// - `height`: Fixed target height (e.g. 1800 for BLLA).
/// - `padding`: 4-tuple (left, right, top, bottom) padding in pixels.
/// - `target_width`: If > 0, right-pad the result to exactly this width
///   (needed for fixed-width CoreML-optimized models). 0 = no width padding.
pub fn preprocess(
    image: &DynamicImage,
    height: u32,
    padding: &[i64; 4],
    target_width: u32,
) -> Result<Preprocessed> {
    // 1. Convert to RGB
    let rgb = image.to_rgb8();
    let (orig_w, orig_h) = rgb.dimensions();

    // 2. Resize to fixed height preserving aspect ratio (pil_fixed_resize)
    // ow = int(w * height / h) when oh is fixed
    let new_w = (orig_w as f64 * height as f64 / orig_h as f64).round() as u32;
    let new_w = new_w.max(1);
    let resized = image::imageops::resize(&rgb, new_w, height, image::imageops::FilterType::Lanczos3);

    // 3. Pad with fill=255 (model padding + optional width padding for fixed models)
    let (pl, pr, pt, pb) = (padding[0] as u32, padding[1] as u32, padding[2] as u32, padding[3] as u32);
    // When using a fixed-width model, add extra right padding so the total
    // width matches model_width. The image content stays left-aligned.
    let extra_r = if target_width > 0 {
        let content_w = new_w + pl + pr;
        target_width.saturating_sub(content_w)
    } else {
        0
    };
    let total_pr = pr + extra_r;

    let padded = if pl > 0 || total_pr > 0 || pt > 0 || pb > 0 {
        let pad_w = new_w + pl + total_pr;
        let pad_h = height + pt + pb;
        let mut padded: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(pad_w, pad_h, Rgb([255, 255, 255]));
        image::imageops::overlay(&mut padded, &resized, pl as i64, pt as i64);
        padded
    } else {
        resized
    };

    let (padded_w, padded_h) = padded.dimensions();

    // 4. Build scal_im: grayscale of resized+padded image (before inversion)
    // PIL convert('L') uses: L = R * 299/1000 + G * 587/1000 + B * 114/1000
    let mut scal_im = Array2::<f32>::zeros((padded_h as usize, padded_w as usize));
    for y in 0..padded_h {
        for x in 0..padded_w {
            let p = padded.get_pixel(x, y);
            let gray = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
            scal_im[[y as usize, x as usize]] = gray;
        }
    }

    // 5. Build tensor: [0,1] scale then invert (1.0 - x)
    // Final shape [3, H, W] CHW
    let mut tensor = Array3::<f32>::zeros((3, padded_h as usize, padded_w as usize));
    for y in 0..padded_h {
        for x in 0..padded_w {
            let p = padded.get_pixel(x, y);
            for c in 0..3 {
                let normalized = p[c] as f32 / 255.0;
                tensor[[c, y as usize, x as usize]] = 1.0 - normalized; // invert
            }
        }
    }

    // Content width = image width + left/right model padding (no width padding).
    let content_w = (new_w + pl + pr) as usize;

    Ok(Preprocessed { tensor, scal_im, content_w })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    fn make_test_image(w: u32, h: u32) -> DynamicImage {
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_resize_to_fixed_height() {
        let img = make_test_image(800, 400);
        let pre = preprocess(&img, 1800, &[0, 0, 0, 0], 0).unwrap();
        // Height should be 1800, width scaled proportionally: 800 * 1800/400 = 3600
        assert_eq!(pre.tensor.dim(), (3, 1800, 3600));
        assert_eq!(pre.scal_im.dim(), (1800, 3600));
    }

    #[test]
    fn test_inversion() {
        let img = make_test_image(100, 50);
        let pre = preprocess(&img, 100, &[0, 0, 0, 0], 0).unwrap();
        // After [0,1] scaling and inversion: pixel value 128 -> 128/255 = 0.502
        // inverted: 1.0 - 0.502 = 0.498
        // Check the red channel at (0,0): original 0 -> 0.0 -> inverted 1.0
        let val = pre.tensor[[0, 0, 0]];
        assert!((val - 1.0).abs() < 1e-5, "Expected inverted value ~1.0, got {val}");
    }

    #[test]
    fn test_values_in_unit_range() {
        let img = make_test_image(200, 100);
        let pre = preprocess(&img, 100, &[0, 0, 0, 0], 0).unwrap();
        for elem in pre.tensor.iter() {
            assert!(*elem >= 0.0 && *elem <= 1.0, "Value out of [0,1]: {elem}");
        }
    }

    #[test]
    fn test_padding() {
        let img = make_test_image(100, 50);
        let pre = preprocess(&img, 50, &[10, 10, 5, 5], 0).unwrap();
        // padded: width 100+20=120, height 50+10=60
        assert_eq!(pre.tensor.dim(), (3, 60, 120));
        // padding fill is 255 -> scaled 1.0 -> inverted 0.0
        // Check top-left padded pixel
        let val = pre.tensor[[0, 0, 0]];
        assert!((val - 0.0).abs() < 1e-5, "Expected padded value 0.0, got {val}");
    }

    #[test]
    fn test_target_width_padding() {
        let img = make_test_image(100, 50);
        // Resize to height 50 -> width stays 100. Target width 128 -> pad right by 28.
        let pre = preprocess(&img, 50, &[0, 0, 0, 0], 128).unwrap();
        assert_eq!(pre.tensor.dim(), (3, 50, 128));
        // Padded pixels should be 0.0 (255 fill -> inverted)
        let val = pre.tensor[[0, 0, 120]];
        assert!((val - 0.0).abs() < 1e-5, "Expected padded value 0.0, got {val}");
    }
}
