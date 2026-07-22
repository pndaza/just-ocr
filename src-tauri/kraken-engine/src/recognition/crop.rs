//! Polygon line-crop masking: composite a line's boundary polygon onto a
//! white background so ink from neighboring lines (outside the polygon but
//! inside its bbox) is masked away before recognition.
//!
//! Faithful port of kraken-rust's `orchestrator.rs:128` `crop_polygon_white_bg`,
//! which was dropped when kraken-rust was vendored into this crate. Uses
//! `imageproc::drawing::draw_polygon_mut` to rasterize the polygon mask.

use image::{DynamicImage, GenericImageView};
use imageproc::drawing::draw_polygon_mut;
use imageproc::point::Point;

/// Crop a line's boundary polygon from the image, filling the area outside
/// the polygon (but inside its bounding box) with white.
///
/// This preserves the black-on-white text polarity the recognizers expect:
/// kraken's `preprocess` inverts to ink-high, and Tesseract operates on
/// dark-ink-on-light-bg. A plain bbox crop (`crop_imm`) would let neighboring
/// lines' ink bleed in at the rectangle corners; this masks it out.
///
/// Degenerate polygons (fewer than 3 distinct points after dedup) fall back
/// to a plain axis-aligned bbox crop.
pub fn crop_polygon_white_bg(image: &DynamicImage, boundary: &[(f64, f64)]) -> DynamicImage {
    let rgba = image.to_rgba8();
    let (min_x, min_y, w, h) = match polygon_bbox(image.dimensions(), boundary) {
        Some(b) => b,
        None => return image.clone(),
    };

    // Translate boundary points into crop-local coordinates and normalise
    // them for imageproc: drop a closing point equal to the first and dedup
    // consecutive duplicates (imageproc panics on first==last).
    let mut pts: Vec<Point<i32>> = boundary
        .iter()
        .map(|p| Point::new((p.0 - min_x as f64) as i32, (p.1 - min_y as f64) as i32))
        .collect();
    pts.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    if pts.len() >= 2 && pts.first() == pts.last() {
        pts.pop();
    }
    if pts.len() < 3 {
        // Degenerate polygon — fall back to a plain bbox crop.
        return image.crop_imm(min_x, min_y, w, h);
    }

    // Build a mask: opaque white inside the polygon, transparent outside.
    let mut mask = image::ImageBuffer::from_pixel(w, h, image::Rgba([0, 0, 0, 0]));
    draw_polygon_mut(&mut mask, &pts, image::Rgba([255, 255, 255, 255]));

    // Composite: start from white, copy source pixels where the mask is set.
    let mut out = image::ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] > 0 {
                out.put_pixel(x, y, *rgba.get_pixel(min_x + x, min_y + y));
            } else {
                out.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            }
        }
    }
    image::DynamicImage::ImageRgba8(out)
}

/// Axis-aligned bbox of a polygon, clamped to image bounds. Returns
/// `(min_x, min_y, width, height)` or `None` if the bbox is zero-area.
fn polygon_bbox(
    (img_w, img_h): (u32, u32),
    boundary: &[(f64, f64)],
) -> Option<(u32, u32, u32, u32)> {
    if boundary.is_empty() {
        return None;
    }
    let min_x = boundary
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min)
        .max(0.0) as u32;
    let min_y = boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min)
        .max(0.0) as u32;
    let max_x = boundary
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .min((img_w - 1) as f64) as u32;
    let max_y = boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .min((img_h - 1) as f64) as u32;
    let w = max_x.saturating_sub(min_x) + 1;
    let h = max_y.saturating_sub(min_y) + 1;
    if w == 0 || h == 0 {
        None
    } else {
        Some((min_x, min_y, w, h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    /// Build a 40x40 white image with a single dark pixel at each of the four
    /// bbox corners of a tight diamond polygon. The diamond's vertices touch
    /// the edge midpoints, so the corners are OUTSIDE the polygon and must be
    /// masked to white; the center is INSIDE and must stay dark.
    fn diamond_image() -> (ImageBuffer<Rgba<u8>, Vec<u8>>, Vec<(f64, f64)>) {
        let (w, h) = (40u32, 40u32);
        let mut img = ImageBuffer::from_pixel(w, h, Rgba([255, 255, 255, 255]));
        // Dark corners (outside the diamond).
        for &(x, y) in &[(0, 0), (39, 0), (0, 39), (39, 39)] {
            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
        // Dark center (inside the diamond).
        img.put_pixel(20, 20, Rgba([0, 0, 0, 255]));
        // Diamond touching edge midpoints: bbox = (0,0,40,40).
        let poly = vec![(20.0, 0.0), (40.0, 20.0), (20.0, 40.0), (0.0, 20.0)];
        (img, poly)
    }

    #[test]
    fn masks_ink_outside_polygon_to_white() {
        let (img, poly) = diamond_image();
        let dyn_img = DynamicImage::ImageRgba8(img);
        let crop = crop_polygon_white_bg(&dyn_img, &poly);
        // The crop bbox is (0,0,40,40); the four corners must be white now.
        assert_eq!(crop.get_pixel(0, 0), Rgba([255, 255, 255, 255]));
        assert_eq!(crop.get_pixel(39, 0), Rgba([255, 255, 255, 255]));
        assert_eq!(crop.get_pixel(0, 39), Rgba([255, 255, 255, 255]));
        assert_eq!(crop.get_pixel(39, 39), Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn preserves_ink_inside_polygon() {
        let (img, poly) = diamond_image();
        let dyn_img = DynamicImage::ImageRgba8(img);
        let crop = crop_polygon_white_bg(&dyn_img, &poly);
        // Center is inside the diamond — must stay dark.
        assert_eq!(crop.get_pixel(20, 20), Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn degenerate_polygon_falls_back_to_bbox_crop() {
        // Fewer than 3 points → degenerate → plain bbox crop, no panic.
        // Two points at opposite corners: a single dark pixel at (0,0) must
        // survive (it's inside the bbox, and there's no polygon to mask with).
        let mut img = ImageBuffer::from_pixel(10, 10, Rgba([255, 255, 255, 255]));
        img.put_pixel(0, 0, Rgba([0, 0, 0, 255]));
        let dyn_img = DynamicImage::ImageRgba8(img);
        let crop = crop_polygon_white_bg(&dyn_img, &[(0.0, 0.0), (9.0, 9.0)]);
        assert_eq!(crop.get_pixel(0, 0), Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn polygon_bbox_basic() {
        let b = vec![(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)];
        assert_eq!(polygon_bbox((100, 100), &b), Some((10, 20, 21, 21)));
    }

    #[test]
    fn polygon_bbox_empty_returns_none() {
        assert_eq!(polygon_bbox((100, 100), &[]), None);
    }
}
