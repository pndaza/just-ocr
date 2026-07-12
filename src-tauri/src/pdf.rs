//! Two ways to turn a PDF into per-page images for OCR:
//!
//! - **Extract** (default): pull the embedded raster image straight off each
//!   page via `lopdf`. No rendering, so it's fast and preserves the native scan
//!   resolution — correct for Tesseract. Best for scanned PDFs, which are just
//!   containers around full-page images.
//! - **Render**: rasterize each page with `hayro` at a fixed output height of
//!   1500 px. Slower, but handles PDFs with no extractable image (vector text,
//!   mixed content) by producing a faithful bitmap of the page.
//!
//! Both return one PNG per page so the result drops straight into the existing
//! image-based OCR pipeline.

use lopdf::{Dictionary, Document, Object};
use rayon::prelude::*;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Output color format for the per-page PNGs fed to Tesseract.
///
/// - `Color`: 24-bit RGB (verbatim from the source image).
/// - `Gray`: 8-bit grayscale — the recommended default. Tesseract 4/5 (LSTM)
///   binarizes internally from grayscale, so this keeps everything it needs
///   while cutting the PNG to ~1/3 the size of RGB.
/// - `Bw`: 8-bit black & white, Otsu-thresholded. Smallest and ideal for
///   pristine printed scans, but can hurt accuracy on noisy/photographic pages
///   because it discards the gray levels Tesseract relies on.
#[derive(Clone, Copy)]
pub(crate) enum ImageMode {
    Color,
    Gray,
    Bw,
}

impl Default for ImageMode {
    fn default() -> Self {
        ImageMode::Gray
    }
}

/// Convert a decoded page image (24-bit RGB, or RGB composited over white)
/// into the requested color format for Tesseract.
fn to_target(img: image::DynamicImage, mode: ImageMode) -> image::DynamicImage {
    match mode {
        ImageMode::Color => img,
        ImageMode::Gray => img.grayscale(),
        ImageMode::Bw => {
            let gray = img.grayscale().into_luma8();
            let thresh = otsu_threshold(&gray);
            let bin = image::GrayImage::from_fn(gray.width(), gray.height(), |x, y| {
                image::Luma([if gray.get_pixel(x, y).0[0] < thresh { 0 } else { 255 }])
            });
            image::DynamicImage::ImageLuma8(bin)
        }
    }
}

/// Otsu's method: pick the 0..=255 threshold maximizing between-class variance,
/// so foreground text and background are split automatically (no fixed cutoff).
fn otsu_threshold(img: &image::GrayImage) -> u8 {
    let mut hist = [0u32; 256];
    for p in img.pixels() {
        hist[p.0[0] as usize] += 1;
    }
    let total = img.pixels().count() as u64;
    let mut sum = 0u64;
    for (i, &c) in hist.iter().enumerate() {
        sum += i as u64 * c as u64;
    }
    let mut w_b = 0u64;
    let mut sum_b = 0u64;
    let mut max_var = 0.0_f64;
    let mut threshold = 127u8;
    for t in 0..256 {
        w_b += hist[t] as u64;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += t as u64 * hist[t] as u64;
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum - sum_b) as f64 / w_f as f64;
        let between = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if between > max_var {
            max_var = between;
            threshold = t as u8;
        }
    }
    threshold
}

/// Formats the display name for a page extracted from a PDF.
///
/// `pdf_name` is the original file name (e.g. `"scan_3.pdf"`); `page` is the
/// 1-based page index. Returns `"<stem> · p<n>"`, e.g. `"scan_3 · p4"`.
pub(crate) fn page_name(pdf_name: &str, page: usize) -> String {
    let stem = std::path::Path::new(pdf_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(pdf_name);
    format!("{stem} · p{page}")
}

/// Extract the largest embedded image from each page of `pdf_bytes`, decoded
/// to RGB8 and re-encoded as PNG (so it drops into the existing image pipeline).
/// Returns one entry per page that yielded an image, in page order. Pages with
/// no extractable image are skipped; the caller surfaces the count.
///
/// `on_progress(done, total)` is called once with the total page count, then
/// once per page after it is processed, so the UI can show progress.
///
/// Decoding/re-encoding each page's image is independent and CPU-bound (the
/// `image`/inflate/fax decoders are single-threaded), so pages run across a
/// rayon pool. The cheap "pick the largest image" step stays inline; a shared
/// `seen_ids` set preserves the cross-page dedup of repeated image XObjects.
pub(crate) fn extract_pages(
    pdf_bytes: &[u8],
    on_progress: impl Fn(usize, usize) + Send + Sync,
    image_mode: ImageMode,
) -> Result<Vec<Vec<u8>>, String> {
    // Load from memory via a temp file: lopdf's Document::load takes a path,
    // and load_from takes a Read. Bytes in memory satisfy the latter.
    let doc = Document::load_from(pdf_bytes).map_err(|e| format!("Failed to load PDF: {e}"))?;
    let pages = doc.get_pages();
    let total = pages.len();
    on_progress(0, total);

    // Dedup of image XObjects shared across pages (e.g. a logo on every page).
    // Guarded because pages decode on multiple threads.
    let seen_ids = Mutex::new(std::collections::HashSet::new());
    // Counts completed pages so progress events stay monotonic across threads.
    let completed = AtomicUsize::new(0);

    // BTreeMap iterates in page order; rayon preserves that order in `collect`,
    // so the returned PNGs line up with `page_name(..., i + 1)`.
    let pngs: Vec<Option<Vec<u8>>> = pages
        .par_iter()
        .map(|(&page_num, &page_id)| {
            // Locate + pick the largest image for this page (cheap, borrows doc).
            let pdf_images = match doc.get_page_images(page_id) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Warning: page {page_num} images: {e}");
                    let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                    on_progress(done, total);
                    return None;
                }
            };
            let best = pdf_images
                .iter()
                .filter(|i| seen_ids.lock().unwrap().insert(i.id))
                .max_by_key(|i| i.width * i.height);

            // Heavy, independent work: decode to RGB8, convert to the requested
            // color mode, then re-encode as PNG.
            let png = match best {
                Some(img) => match decode_image(img) {
                    Ok((rgb, w, h)) => {
                        let dyn_img = match image::RgbImage::from_raw(w, h, rgb) {
                            Some(b) => image::DynamicImage::ImageRgb8(b),
                            None => {
                                eprintln!("Warning: page {page_num} RGB buffer size mismatch");
                                return None;
                            }
                        };
                        match reencode_png(&to_target(dyn_img, image_mode)) {
                            Ok(png) => Some(png),
                            Err(e) => {
                                eprintln!("Warning: page {page_num} re-encode: {e}");
                                None
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: page {page_num} decode: {e}");
                        None
                    }
                },
                None => None,
            };

            let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
            on_progress(done, total);
            png
        })
        .collect();

    Ok(pngs.into_iter().flatten().collect())
}

/// Render every page of `pdf_bytes` to a PNG at a fixed output height of
/// `target_height` pixels (width scales to preserve aspect ratio). Used when a
/// PDF has no extractable image (vector text, mixed content) or when the user
/// explicitly wants a page bitmap rather than the embedded scan.
///
/// Scaling is derived from each page's own dimensions so every page ends up
/// the same pixel height regardless of its MediaBox — important because scanned
/// PDFs often declare oversized pages that would otherwise explode the output.
///
/// `on_progress(done, total)` is called once with the total page count, then
/// once per page after it is rasterized, so the UI can show progress.
pub(crate) fn render_pages(
    pdf_bytes: &[u8],
    target_height: u16,
    on_progress: impl Fn(usize, usize) + Send + Sync,
    image_mode: ImageMode,
) -> Result<Vec<Vec<u8>>, String> {
    use hayro::hayro_interpret::InterpreterSettings;
    use hayro::hayro_syntax::Pdf;
    use hayro::vello_cpu::color::palette::css::WHITE;
    use hayro::{RenderCache, RenderSettings, render};

    let pdf = Pdf::new(pdf_bytes.to_vec()).map_err(|e| format!("Failed to load PDF: {e:?}"))?;
    let cache = RenderCache::new();
    // Default font resolver (uses hayro's built-in standard-font fallbacks).
    // Scanned PDFs have no live text, so font resolution rarely matters here;
    // vector PDFs fall back to the standard 14 fonts.
    let interpreter_settings = InterpreterSettings::default();

    let pages = pdf.pages();
    let total = pages.len();
    on_progress(0, total);

    let mut out = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        // page.render_dimensions() is the unscaled (width, height) in points.
        // Scale so the rendered height is exactly target_height px.
        let (_w, h) = page.render_dimensions();
        let scale = if h > 0.0 {
            target_height as f32 / h
        } else {
            1.0
        };
        let settings = RenderSettings {
            x_scale: scale,
            y_scale: scale,
            bg_color: WHITE,
            ..Default::default()
        };
        let pixmap = render(page, &cache, &interpreter_settings, &settings);
        let (pw, ph) = (pixmap.width(), pixmap.height());
        // Composite the straight-alpha RGBA pixmap over white, then convert to
        // the requested color mode before encoding the PNG.
        let rgba = pixmap.take_unpremultiplied();
        let mut rgb_buf = Vec::with_capacity(rgba.len() * 3);
        for p in &rgba {
            let a = p.a as f32 / 255.0;
            let blend = |c: u8| (c as f32 * a + 255.0 * (1.0 - a)) as u8;
            rgb_buf.push(blend(p.r));
            rgb_buf.push(blend(p.g));
            rgb_buf.push(blend(p.b));
        }
        let rgb_img = match image::RgbImage::from_raw(pw as u32, ph as u32, rgb_buf) {
            Some(b) => b,
            None => {
                eprintln!("Warning: page {} RGB buffer size mismatch", i + 1);
                on_progress(i + 1, total);
                continue;
            }
        };
        let png = match reencode_png(&to_target(image::DynamicImage::ImageRgb8(rgb_img), image_mode)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: page {} PNG encode: {e}", i + 1);
                on_progress(i + 1, total);
                continue;
            }
        };
        out.push(png);
        on_progress(i + 1, total);
    }
    Ok(out)
}

/// Decode one PDF image XObject into RGB8 pixels (plus its dimensions),
/// dispatching on the stream's /Filter.
fn decode_image(img: &lopdf::xobject::PdfImage) -> Result<(Vec<u8>, u32, u32), String> {
    let width = img.width as u32;
    let height = img.height as u32;
    if width == 0 || height == 0 {
        return Err("zero dimension image".into());
    }
    let filters: Vec<String> = img.filters.clone().unwrap_or_default();
    let content = img.content;

    // JPEG and JPEG2000 are self-contained: hand the raw bytes to the image
    // crate. (JP2 is unsupported by default features; DCT = JPEG.)
    if filters.contains(&"DCTDecode".to_string()) {
        let rgb = image::load_from_memory(content)
            .map_err(|e| format!("JPEG decode: {e}"))?
            .to_rgb8()
            .into_raw();
        return Ok((rgb, width, height));
    }
    if filters.contains(&"JPXDecode".to_string()) {
        return Err("JPEG2000 not supported".into());
    }
    if filters.contains(&"JBIG2Decode".to_string()) {
        return Err("JBIG2 not supported".into());
    }

    // CCITT (fax) — black & white scans. Result is grayscale 8-bit.
    if filters.contains(&"CCITTFaxDecode".to_string()) {
        let dp = img.origin_dict.get(b"DecodeParms").ok();
        let gray = decode_ccitt(content, width, height, dp)?;
        let rgb = gray.iter().flat_map(|&v| [v, v, v]).collect();
        return Ok((rgb, width, height));
    }

    // Otherwise a chain of compression filters (FlateDecode, RunLengthDecode,
    // …) ending in raw pixel data we interpret by color space + bpc.
    let mut data = content.to_vec();
    for (i, filter) in filters.iter().enumerate() {
        data = apply_filter(&data, img.origin_dict, filter, i)?;
    }
    let bpc = img.bits_per_component.unwrap_or(8) as u32;
    let cs = img.color_space.as_deref().unwrap_or("DeviceRGB");
    let rgb = interpret_raw(&data, width, height, cs, bpc)?;
    Ok((rgb, width, height))
}

/// Apply one PDF decompression filter. `filter_index` selects the matching
/// /DecodeParms entry when /Filter is an array (one dict per filter).
fn apply_filter(
    data: &[u8],
    dict: &Dictionary,
    filter: &str,
    filter_index: usize,
) -> Result<Vec<u8>, String> {
    let mut out = match filter {
        "FlateDecode" => {
            let mut d = flate2::read::ZlibDecoder::new(data);
            let mut out = Vec::new();
            d.read_to_end(&mut out).map_err(|e| format!("FlateDecode: {e}"))?;
            out
        }
        "RunLengthDecode" => decode_runlength(data)?,
        other => return Err(format!("filter {other} not implemented")),
    };
    // A PNG predictor (DecodeParms /Predictor >= 10) reverses per-row filtering
    // applied before compression. Without this, FlateDecode-only images with
    // predictors render as skewed noise.
    if let Some((predictor, colors, bpc, columns)) = read_predictor(dict, filter_index) {
        if predictor >= 10 {
            out = depngify(&out, colors, bpc, columns)?;
        }
    }
    Ok(out)
}

/// RunLengthDecode: PDF run-length encoding. Length byte n in 0..=127 copies
/// the next n+1 bytes literally; n in 129..=255 repeats the next byte 257-n
/// times. 128 is end-of-data.
fn decode_runlength(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let n = data[i];
        i += 1;
        match n {
            0..=127 => {
                let count = n as usize + 1;
                if i + count > data.len() {
                    return Err("RunLengthDecode: short literal run".into());
                }
                out.extend_from_slice(&data[i..i + count]);
                i += count;
            }
            128 => break,
            129..=255 => {
                let count = 257 - n as usize;
                if i >= data.len() {
                    return Err("RunLengthDecode: short repeated run".into());
                }
                let b = data[i];
                out.extend(std::iter::repeat_n(b, count));
                i += 1;
            }
        }
    }
    Ok(out)
}

/// Read (predictor, colors, bpc, columns) from a stream's /DecodeParms.
/// `filter_index` selects the matching entry when DecodeParms is an array.
fn read_predictor(dict: &Dictionary, filter_index: usize) -> Option<(u32, u32, u32, u32)> {
    let dp = dict.get(b"DecodeParms").ok()?;
    let dp_dict = match dp {
        Object::Dictionary(d) => d,
        Object::Array(arr) => {
            let obj = arr.get(filter_index).or_else(|| arr.first())?;
            match obj {
                Object::Dictionary(d) => d,
                _ => return None,
            }
        }
        _ => return None,
    };
    let predictor = dp_dict.get(b"Predictor").and_then(|v| v.as_i64()).unwrap_or(1) as u32;
    let colors = dp_dict.get(b"Colors").and_then(|v| v.as_i64()).unwrap_or(1) as u32;
    let bpc = dp_dict.get(b"BitsPerComponent").and_then(|v| v.as_i64()).unwrap_or(8) as u32;
    let columns = dp_dict.get(b"Columns").and_then(|v| v.as_i64()).unwrap_or(1) as u32;
    Some((predictor, colors, bpc, columns))
}

/// Reverse PNG prediction on a decoded stream. Each row carries a 1-byte
/// filter type (None/Sub/Up/Average/Paeth) followed by the filtered pixels.
fn depngify(data: &[u8], colors: u32, bpc: u32, columns: u32) -> Result<Vec<u8>, String> {
    if bpc != 8 {
        return Err(format!("PNG predictor with BitsPerComponent {bpc} not supported"));
    }
    let bpp = colors as usize;
    let row_bytes = columns as usize * bpp;
    let stride = row_bytes + 1; // 1 filter byte per row
    if data.len() % stride != 0 {
        return Err(format!(
            "PNG predictor: data length {} not divisible by row stride {}",
            data.len(),
            stride
        ));
    }
    let nrows = data.len() / stride;
    let mut out = vec![0u8; nrows * row_bytes];
    let mut prev_row = vec![0u8; row_bytes];

    for r in 0..nrows {
        let row_start = r * stride;
        let filter = data[row_start];
        let enc = &data[row_start + 1..row_start + 1 + row_bytes];
        let cur = &mut out[r * row_bytes..(r + 1) * row_bytes];
        match filter {
            0 => cur.copy_from_slice(enc),
            1 => {
                for i in 0..row_bytes {
                    let left = if i >= bpp { cur[i - bpp] } else { 0 };
                    cur[i] = enc[i].wrapping_add(left);
                }
            }
            2 => {
                for i in 0..row_bytes {
                    cur[i] = enc[i].wrapping_add(prev_row[i]);
                }
            }
            3 => {
                for i in 0..row_bytes {
                    let left = if i >= bpp { cur[i - bpp] as u16 } else { 0 };
                    let up = prev_row[i] as u16;
                    cur[i] = enc[i].wrapping_add(((left + up) / 2) as u8);
                }
            }
            4 => {
                for i in 0..row_bytes {
                    let left = if i >= bpp { cur[i - bpp] } else { 0 };
                    let up = prev_row[i];
                    let upleft = if i >= bpp { prev_row[i - bpp] } else { 0 };
                    cur[i] = enc[i].wrapping_add(paeth(left, up, upleft));
                }
            }
            other => return Err(format!("PNG predictor: unknown row filter {other}")),
        }
        prev_row.copy_from_slice(cur);
    }
    Ok(out)
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (a, b, c) = (a as i32, b as i32, c as i32);
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

/// Interpret raw decoded pixel data as RGB8 by color space + bits-per-component.
/// CMYK is converted to RGB; grayscale is expanded to RGB triplets. Sub-byte
/// samples (bpc 1/2/4) are expanded to one byte per sample first.
fn interpret_raw(
    raw: &[u8],
    width: u32,
    height: u32,
    color_space: &str,
    bpc: u32,
) -> Result<Vec<u8>, String> {
    let raw = if matches!(bpc, 1 | 2 | 4) {
        let colors = match color_space {
            "DeviceRGB" => 3,
            "DeviceGray" => 1,
            "DeviceCMYK" => 4,
            _ => return Err(format!("color space {color_space} not supported for raw")),
        };
        expand_samples(raw, width, height, colors, bpc)?
    } else if bpc == 8 {
        raw.to_vec()
    } else {
        return Err(format!("BitsPerComponent {bpc} not supported for raw"));
    };

    match color_space {
        "DeviceRGB" => {
            let expected = (width * height * 3) as usize;
            if raw.len() < expected {
                return Err(format!("raw RGB: expected {expected} bytes, got {}", raw.len()));
            }
            Ok(raw[..expected].to_vec())
        }
        "DeviceGray" => {
            let expected = (width * height) as usize;
            if raw.len() < expected {
                return Err(format!("raw Gray: expected {expected} bytes, got {}", raw.len()));
            }
            let scaled = scale_gray(&raw[..expected], bpc);
            Ok(scaled.iter().flat_map(|&v| [v, v, v]).collect())
        }
        "DeviceCMYK" => {
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for chunk in raw.chunks_exact(4) {
                let c = chunk[0] as f32 / 255.0;
                let m = chunk[1] as f32 / 255.0;
                let y = chunk[2] as f32 / 255.0;
                let k = chunk[3] as f32 / 255.0;
                rgb.extend_from_slice(&[
                    ((1.0 - c) * (1.0 - k) * 255.0) as u8,
                    ((1.0 - m) * (1.0 - k) * 255.0) as u8,
                    ((1.0 - y) * (1.0 - k) * 255.0) as u8,
                ]);
            }
            Ok(rgb)
        }
        _ => Err(format!("color space {color_space} not supported for raw")),
    }
}

/// Expand sub-byte samples (bpc 1/2/4) into one byte per sample. Each row
/// begins on a byte boundary; trailing pad bits of the last byte per row are
/// skipped (treating the data as one continuous bit-stream would drift the
/// decode and produce diagonal distortion).
fn expand_samples(
    raw: &[u8],
    width: u32,
    height: u32,
    colors: u32,
    bpc: u32,
) -> Result<Vec<u8>, String> {
    let (w, h) = (width as usize, height as usize);
    let samples_per_row = w * colors as usize;
    let bits_per_row = samples_per_row * bpc as usize;
    let row_bytes = bits_per_row.div_ceil(8);
    let needed = row_bytes * h;
    if raw.len() < needed {
        return Err(format!(
            "expand_samples: need {needed} bytes for {h} rows of {row_bytes} bytes at {bpc} bpc, got {}",
            raw.len()
        ));
    }
    let mask = (1u8 << bpc) - 1;
    let mut out = Vec::with_capacity(samples_per_row * h);
    for row in 0..h {
        let row_start = row * row_bytes;
        for s in 0..samples_per_row {
            let bit_pos = s * bpc as usize;
            let byte_idx = row_start + (bit_pos >> 3);
            let bit_off = bit_pos & 7;
            let hi = raw[byte_idx] as u32;
            let lo = if byte_idx + 1 < raw.len() {
                raw[byte_idx + 1] as u32
            } else {
                0
            };
            let window = (hi << 8) | lo;
            let shift = 16 - bit_off - bpc as usize;
            out.push(((window >> shift) as u8) & mask);
        }
    }
    Ok(out)
}

/// Scale gray sample values to the full 0–255 range based on bpc.
fn scale_gray(data: &[u8], bpc: u32) -> Vec<u8> {
    if bpc == 8 {
        return data.to_vec();
    }
    let max_val = (1u32 << bpc) - 1;
    data.iter()
        .map(|&v| ((v as u32 * 255 + max_val / 2) / max_val).min(255) as u8)
        .collect()
}

/// Decode a CCITT (fax) encoded image to 8-bit grayscale. Tries Group 4 then
/// Group 3 based on /DecodeParms /K, falling back sensibly when absent.
fn decode_ccitt(
    content: &[u8],
    width: u32,
    height: u32,
    decode_parms: Option<&Object>,
) -> Result<Vec<u8>, String> {
    let columns: u16 = match decode_parms {
        Some(Object::Dictionary(d)) => d
            .get(b"Columns")
            .and_then(|v| v.as_i64())
            .unwrap_or(width as i64) as u16,
        Some(Object::Array(arr)) => arr
            .first()
            .and_then(|o| match o {
                Object::Dictionary(d) => d.get(b"Columns").and_then(|v| v.as_i64()).ok(),
                _ => None,
            })
            .unwrap_or(width as i64) as u16,
        _ => width as u16,
    };
    let k: i64 = match decode_parms {
        Some(Object::Dictionary(d)) => d.get(b"K").and_then(|v| v.as_i64()).unwrap_or(0),
        Some(Object::Array(arr)) => arr
            .first()
            .and_then(|o| match o {
                Object::Dictionary(d) => d.get(b"K").and_then(|v| v.as_i64()).ok(),
                _ => None,
            })
            .unwrap_or(0),
        _ => 0,
    };

    let mut pixels: Vec<u8> = Vec::with_capacity((width * height) as usize);
    let decode = |transitions: &[u16], px: &mut Vec<u8>| {
        for pel in fax::decoder::pels(transitions, columns) {
            px.push(if pel == fax::Color::Black { 0 } else { 255 });
        }
    };
    if k < 0 {
        fax::decoder::decode_g4(
            content.iter().copied(),
            columns,
            Some(height as u16),
            |t| decode(t, &mut pixels),
        );
    } else {
        fax::decoder::decode_g3(content.iter().copied(), |t| decode(t, &mut pixels));
    }
    // Pad if the decoder produced fewer pixels than expected (white).
    let expected = (width * height) as usize;
    while pixels.len() < expected {
        pixels.push(255);
    }
    Ok(pixels)
}

/// Re-encode RGB8 pixels as PNG bytes so the result drops into the existing
/// image-based OCR pipeline (which decodes via `image::load_from_memory`).
fn reencode_png(img: &image::DynamicImage) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| format!("PNG encode: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::page_name;

    #[test]
    fn strips_pdf_extension_and_appends_page() {
        assert_eq!(page_name("scan_3.pdf", 4), "scan_3 · p4");
    }

    #[test]
    fn handles_name_without_extension() {
        assert_eq!(page_name("plain", 1), "plain · p1");
    }

    #[test]
    fn handles_path_like_name() {
        assert_eq!(page_name("/tmp/foo/report.pdf", 12), "report · p12");
    }
}

#[cfg(test)]
mod extract_tests {
    use super::{extract_pages, ImageMode};
    use std::path::PathBuf;

    fn fixture() -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.pdf");
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    fn extracts_at_least_one_nonempty_png() {
        let Some(path) = fixture() else {
            eprintln!("skip: tests/fixtures/tiny.pdf not present");
            return;
        };
        let bytes = std::fs::read(&path).expect("read fixture");
        let pages = extract_pages(&bytes, |_, _| {}, ImageMode::Gray).expect("extract succeeds");
        assert!(!pages.is_empty(), "expected at least one extracted page");
        for png in &pages {
            assert!(!png.is_empty(), "extracted page PNG must be non-empty");
            assert_eq!(&png[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        }
    }

    #[test]
    fn rejects_garbage_bytes() {
        let err = extract_pages(b"not a pdf", |_, _| {}, ImageMode::Gray).unwrap_err();
        assert!(
            err.to_lowercase().contains("load") || err.to_lowercase().contains("pdf"),
            "expected a load error, got: {err}"
        );
    }
}
