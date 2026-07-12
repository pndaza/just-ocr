//! Extract embedded raster images from PDF pages.
//!
//! Scanned PDFs are containers around full-page raster scans. Rendering the
//! *page* (as a general-purpose PDF rasterizer would) is both slow and
//! inaccurate here: many scans declare an oversized MediaBox, so rendering at
//! a fixed DPI produces enormous, wrongly-scaled images that confuse OCR.
//! Extracting the embedded image directly yields the native scan resolution —
//! the correct input for Tesseract — with no rasterization cost.
//!
//! The heavy lifting (walking page resource trees, reporting image XObjects)
//! is done by `lopdf::Document::get_page_images`. This module decodes each
//! image's bytes into RGB8 according to its PDF filter and re-encodes to PNG so
//! the result feeds straight into the existing image-based OCR pipeline.

use lopdf::{Dictionary, Document, Object};
use std::io::Read;

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
pub(crate) fn extract_pages(pdf_bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    // Load from memory via a temp file: lopdf's Document::load takes a path,
    // and load_from takes a Read. Bytes in memory satisfy the latter.
    let doc = Document::load_from(pdf_bytes).map_err(|e| format!("Failed to load PDF: {e}"))?;
    let pages = doc.get_pages();

    let mut seen_ids = std::collections::HashSet::new();
    let mut out = Vec::new();
    // Pages is a BTreeMap<u32, ObjectId>, so iteration is already in page order.
    for (&page_num, &page_id) in &pages {
        let pdf_images = match doc.get_page_images(page_id) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Warning: page {page_num} images: {e}");
                continue;
            }
        };
        // Pick the largest image on the page — the full-page scan, typically.
        // Smaller images (logos, signatures) are ignored.
        let best = pdf_images
            .iter()
            .filter(|i| seen_ids.insert(i.id))
            .max_by_key(|i| i.width * i.height);
        let Some(img) = best else { continue };
        match decode_image(img) {
            Ok((rgb, w, h)) => match reencode_png(&rgb, w, h) {
                Ok(png) => out.push(png),
                Err(e) => eprintln!("Warning: page {page_num} re-encode: {e}"),
            },
            Err(e) => eprintln!("Warning: page {page_num} decode: {e}"),
        }
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
fn reencode_png(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let buf = image::RgbImage::from_raw(width, height, rgb.to_vec())
        .ok_or_else(|| "failed to create image buffer".to_string())?;
    let mut out = Vec::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
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
    use super::extract_pages;
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
        let pages = extract_pages(&bytes).expect("extract succeeds");
        assert!(!pages.is_empty(), "expected at least one extracted page");
        for png in &pages {
            assert!(!png.is_empty(), "extracted page PNG must be non-empty");
            assert_eq!(&png[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        }
    }

    #[test]
    fn rejects_garbage_bytes() {
        let err = extract_pages(b"not a pdf").unwrap_err();
        assert!(
            err.to_lowercase().contains("load") || err.to_lowercase().contains("pdf"),
            "expected a load error, got: {err}"
        );
    }
}
