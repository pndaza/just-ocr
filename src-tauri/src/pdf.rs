//! Two ways to turn a PDF into per-page images for OCR:
//!
//! - **Extract** (default): pull the embedded raster image straight off each
//!   page via `lopdf`. No rendering, so it's fast and preserves the native scan
//!   resolution — correct for Tesseract. Best for scanned PDFs, which are just
//!   containers around full-page images. An optional `max_height` bound can
//!   downscale oversized scans (see `maybe_downscale`) when native resolution
//!   is too high for good segmentation.
//! - **Render**: rasterize each page with `hayro` at a user-chosen output
//!   height (falling back to `PDF_RENDER_HEIGHT` in `lib.rs` when unset).
//!   Slower, but handles PDFs with no extractable image (vector text, mixed
//!   content) by producing a faithful bitmap of the page.
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
///
/// (A whole-page `Bw` mode existed previously — Otsu-thresholded at the page
/// level — but was removed: Tesseract binarizes internally, and the Myanmar/
/// Kraken path now binarizes per-line with Sauvola inside `preprocess_line`,
/// so whole-page binarization only threw away gray levels earlier with no
/// accuracy benefit.)
#[derive(Clone, Copy)]
pub(crate) enum ImageMode {
    Color,
    Gray,
}

impl Default for ImageMode {
    fn default() -> Self {
        ImageMode::Gray
    }
}

/// Convert a decoded page image (24-bit RGB, or RGB composited over white)
/// into the requested color format.
fn to_target(img: image::DynamicImage, mode: ImageMode) -> image::DynamicImage {
    match mode {
        ImageMode::Color => img,
        ImageMode::Gray => img.grayscale(),
    }
}

/// Downscale `img` so it is at most `max_height` px tall (width scaled to
/// preserve aspect ratio). Never upscales — pages already at or below the
/// limit pass through untouched. `None` disables resizing entirely (native
/// resolution, the historical default).
///
/// Exists because scans embedded at very high resolution (3000px+ page
/// height) produce line heights far above what the segmentation models were
/// trained on, degrading layout detection; a bounded downscale restores
/// workable proportions. Triangle (bilinear) filtering is chosen over sharper
/// kernels (Lanczos/CatmullRom) because large reductions with sharp kernels
/// alias high-frequency detail into noise, which hurts the recognizers more
/// than the slight softness of bilinear does.
fn maybe_downscale(
    img: image::DynamicImage,
    max_height: Option<u16>,
) -> image::DynamicImage {
    let Some(max) = max_height else {
        return img;
    };
    let h = img.height();
    if h == 0 || h <= max as u32 {
        return img;
    }
    let scale = max as f32 / h as f32;
    let w = ((img.width() as f32 * scale).round() as u32).max(1);
    img.resize_exact(w, max as u32, image::imageops::FilterType::Triangle)
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
/// `max_height` optionally bounds each page's pixel height (see
/// `maybe_downscale`) — pages taller than the limit are downscaled, aspect
/// preserved; shorter pages are untouched.
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
    max_height: Option<u16>,
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
                Some(img) => {
                    // For JBIG2 images, resolve the optional /JBIG2Globals
                    // shared-stream now — decode_image can't (no &Document).
                    // Cheap no-op for every non-JBIG2 image.
                    let globals = jbig2_globals(&doc, img);
                    match decode_image(img, globals.as_deref()) {
                        Ok((rgb, w, h)) => {
                            let dyn_img = match image::RgbImage::from_raw(w, h, rgb) {
                                Some(b) => image::DynamicImage::ImageRgb8(b),
                                None => {
                                    eprintln!("Warning: page {page_num} RGB buffer size mismatch");
                                    return None;
                                }
                            };
                            // Bound the page height before re-encoding —
                            // oversized scans are what this option exists for.
                            let dyn_img = maybe_downscale(dyn_img, max_height);
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
                    }
                }
                None => None,
            };

            let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
            on_progress(done, total);
            png
        })
        .collect();

    Ok(pngs.into_iter().flatten().collect())
}

/// Render every page of `pdf_bytes` to a PNG at an output height of
/// `target_height` pixels (width scales to preserve aspect ratio). Used when a
/// PDF has no extractable image (vector text, mixed content) or when the user
/// explicitly wants a page bitmap rather than the embedded scan. The height is
/// user-selectable from the frontend; `PDF_RENDER_HEIGHT` is the fallback when
/// none is supplied.
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
/// dispatching on the stream's /Filter chain.
///
/// `jbig2_globals` carries the optional `/JBIG2Globals` stream bytes for the
/// JBIG2Decode path. It's resolved by the caller (`extract_pages`), which has
/// the `&Document` needed to dereference the indirect object — `PdfImage`
/// exposes only the image dict + content, not the document. `None` for every
/// non-JBIG2 image (zero overhead) and for JBIG2 images with no globals.
fn decode_image(
    img: &lopdf::xobject::PdfImage,
    jbig2_globals: Option<&[u8]>,
) -> Result<(Vec<u8>, u32, u32), String> {
    let width = img.width as u32;
    let height = img.height as u32;
    if width == 0 || height == 0 {
        return Err("zero dimension image".into());
    }
    let filters: Vec<String> = img.filters.clone().unwrap_or_default();
    let content = img.content;

    // "Terminal" image formats — self-describing bitstreams (JPEG, JPEG2000,
    // JBIG2) or fax (CCITT) — can't go through the raw-pixel interpreter
    // below. But a compression filter (FlateDecode is the common one) may sit
    // earlier in the /Filter array and must be unwound first.
    //
    // Real-world example: some PDF producers wrap JPEG streams in FlateDecode
    // (/Filter [/FlateDecode /DCTDecode]). The old code did an early return
    // on `filters.contains("DCTDecode")` and handed the still-zlib bytes
    // straight to the JPEG decoder, which failed and silently dropped every
    // affected page. Apply every filter before the terminal one, then decode.
    let terminal_idx = filters.iter().position(|f| {
        matches!(
            f.as_str(),
            "DCTDecode" | "JPXDecode" | "JBIG2Decode" | "CCITTFaxDecode"
        )
    });
    if let Some(idx) = terminal_idx {
        // Walk the compression chain that precedes the terminal format.
        let mut data = content.to_vec();
        for (i, filter) in filters[..idx].iter().enumerate() {
            data = apply_filter(&data, img.origin_dict, filter, i)?;
        }
        return match filters[idx].as_str() {
            // DCT = JPEG; hand the (now-decompressed) bytes to the image crate.
            "DCTDecode" => {
                let rgb = image::load_from_memory(&data)
                    .map_err(|e| format!("JPEG decode: {e}"))?
                    .to_rgb8()
                    .into_raw();
                Ok((rgb, width, height))
            }
            "JPXDecode" => Err("JPEG2000 not supported".into()),
            // JBIG2 — 1-bit bilevel scans (the standard compression for
            // scanned documents). Optional shared symbol dictionary bytes
            // (`/JBIG2Globals`) precede the page stream per T.88 §7.5.
            // Dimensions come from the decoded image (region dims in the
            // codestream can differ from the image dict's W/H).
            "JBIG2Decode" => {
                let (gray, w, h) = decode_jbig2(&data, jbig2_globals, width, height)?;
                let rgb = gray.iter().flat_map(|&v| [v, v, v]).collect();
                Ok((rgb, w, h))
            }
            // CCITT (fax) — black & white scans. Result is grayscale 8-bit.
            "CCITTFaxDecode" => {
                let dp = img.origin_dict.get(b"DecodeParms").ok();
                let gray = decode_ccitt(&data, width, height, dp)?;
                let rgb = gray.iter().flat_map(|&v| [v, v, v]).collect();
                Ok((rgb, width, height))
            }
            _ => unreachable!("terminal_idx only matches the four filters above"),
        };
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
        "LZWDecode" => {
            // PDF defaults to EarlyChange=1 (Adobe PDF Ref §7.4.4.1): the code
            // width increases one code *earlier* than "original" LZW — i.e. the
            // "switch one symbol sooner" behavior. In weezl that is
            // `with_tiff_size_switch`; `Decoder::new` is EarlyChange=0 and
            // corrupts silently once the table grows past 9-bit codes. Streams
            // that explicitly declare `/EarlyChange 0` (DecodeParms) are not
            // honored — vanishingly rare in practice. TODO if one ever shows up.
            use weezl::{decode::Decoder, BitOrder};
            let mut dec = Decoder::with_tiff_size_switch(BitOrder::Msb, 8);
            dec.decode(data).map_err(|e| format!("LZWDecode: {e:?}"))?
        }
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
///
/// PNG filters operate on bytes, not samples. The "bytes per pixel" used as
/// the Sub/Average/Paeth left-neighbor distance is `ceil(colors * bpc / 8)`
/// (PNG spec §7, "filter byte 0"); for the common 8-bpc case that's just
/// `colors`. Each row is `ceil(columns * colors * bpc / 8)` bytes wide —
/// sub-byte samples (bpc 1/2/4) pack left-to-right within each row and every
/// row begins on a byte boundary, exactly matching what `expand_samples`
/// expects downstream. Without this packing the decoded bytes drift relative
/// to `expand_samples`' per-row walk and the image comes out skewed.
fn depngify(data: &[u8], colors: u32, bpc: u32, columns: u32) -> Result<Vec<u8>, String> {
    if !(1..=16).contains(&bpc) {
        return Err(format!("PNG predictor with BitsPerComponent {bpc} not supported"));
    }
    let bits_per_pixel = colors.checked_mul(bpc).ok_or("colors*bpc overflow")? as usize;
    // ceil — a 1-bpc grayscale pixel is 1 bit, bpp rounds up to 1 byte.
    let bpp = bits_per_pixel.div_ceil(8);
    let row_bytes = (columns as usize)
        .checked_mul(bits_per_pixel)
        .ok_or("columns*bits overflow")?
        .div_ceil(8);
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

/// Resolve the optional `/JBIG2Globals` shared-stream for a JBIG2 image.
///
/// JBIG2 images in scanned PDFs commonly reference a separate stream holding
/// a shared symbol dictionary (one dict reused across every page). The
/// reference lives in the image dict's `/DecodeParms << /JBIG2Globals N 0 R
/// >>`. Returns `None` for non-JBIG2 images (cheap filter-name check first)
/// and for JBIG2 images with no globals entry — both are normal.
///
/// `PdfImage` exposes the image dict but not the `Document`, so this runs in
/// `extract_pages` (where `&Document` is in scope) and the bytes are threaded
/// into `decode_image`. The globals stream may itself be Flate-compressed;
/// `all_content` applies its own `/Filter` chain, matching what poppler/mupdf
/// feed their decoders. A resolve failure logs and returns `None` rather than
/// failing the whole page — a missing dict usually still decodes the generic
/// regions, just without text-symbol reuse.
fn jbig2_globals(doc: &Document, img: &lopdf::xobject::PdfImage) -> Option<Vec<u8>> {
    // Cheap fast path: nothing to do unless this image uses JBIG2Decode.
    let is_jbig2 = img
        .filters
        .as_ref()
        .map(|fs| fs.iter().any(|f| f == "JBIG2Decode"))
        .unwrap_or(false);
    if !is_jbig2 {
        return None;
    }

    // /DecodeParms may be a single dict or an array (one per filter, aligned
    // to the /Filter array). Locate the JBIG2Globals entry in either shape.
    let dp = img.origin_dict.get(b"DecodeParms").ok()?;
    let dp_dict = match dp {
        Object::Dictionary(d) => d,
        Object::Array(arr) => {
            // Prefer the entry at the JBIG2 filter's index; fall back to the
            // first dict in the array that actually carries a JBIG2Globals key.
            let jbig2_idx = img
                .filters
                .as_ref()
                .and_then(|fs| fs.iter().position(|f| f == "JBIG2Decode"));
            jbig2_idx
                .and_then(|i| match arr.get(i)? {
                    Object::Dictionary(d) => Some(d),
                    _ => None,
                })
                .or_else(|| {
                    arr.iter().filter_map(|o| match o {
                        Object::Dictionary(d) => Some(d),
                        _ => None,
                    }).find(|d| d.get(b"JBIG2Globals").is_ok())
                })?
        }
        _ => return None,
    };

    let globals_ref = match dp_dict.get(b"JBIG2Globals").ok()? {
        Object::Reference(id) => *id,
        _ => return None,
    };

    // Dereference the stream and return its fully-decoded content. The globals
    // stream may itself be Flate-compressed; decompressed_content applies its
    // /Filter chain, matching what poppler/mupdf feed their decoders. A
    // resolve/decode failure logs and returns None — decode then proceeds
    // without the dict.
    let stream = match doc.get_object(globals_ref).and_then(|o| o.as_stream()) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("[pdf] JBIG2Globals {:?} resolve failed: {e}", globals_ref);
            return None;
        }
    };
    match stream.decompressed_content() {
        Ok(content) => Some(content),
        Err(e) => {
            log::warn!("[pdf] JBIG2Globals {:?} decode failed: {e}", globals_ref);
            None
        }
    }
}

/// Decode a JBIG2-encoded image to 8-bit grayscale (0 = black, 255 = white).
///
/// `globals` is the optional `/JBIG2Globals` stream bytes (resolved by the
/// caller). Backed by `hayro-jbig2` (pure Rust, T.88). JBIG2 is `black=1`,
/// the opposite of PDF/Tesseract, so the `Decoder` impl writes `0x00` for
/// black and leaves the default `0xFF` for white — matching the reference
/// implementation in hayro's own `jbig2` filter.
///
/// Returns the decoded pixels along with the image's *actual* dimensions
/// (`image.width()`/`image.height()`), NOT the caller-supplied dict W/H.
/// JBIG2 region dimensions live inside the codestream and can disagree with
/// the image dict (e.g. trailing padding rows); sizing the output buffer by
/// the dict would panic. The returned dimensions keep `RgbImage::from_raw`
/// consistent with the pixel count.
fn decode_jbig2(
    data: &[u8],
    globals: Option<&[u8]>,
    _dict_width: u32,
    _dict_height: u32,
) -> Result<(Vec<u8>, u32, u32), String> {
    let image = hayro_jbig2::Image::new_embedded(data, globals)
        .map_err(|e| format!("JBIG2: {e:?}"))?;
    let (width, height) = (image.width(), image.height());

    // 8-bit grayscale output buffer, defaulting to white (0xFF). The Decoder
    // impl only writes black pixels, so untouched areas (rare) read as white.
    let mut out = vec![0xFFu8; (width as usize) * (height as usize)];

    // Row-major writer, one byte per pixel. The pixel stream is contiguous
    // (width == stride for 8-bit gray), so next_line is a no-op — matching
    // the Luma8 path in hayro's own reference filter.
    struct Gray8Writer<'a> {
        buf: &'a mut [u8],
        pos: usize,
    }
    impl hayro_jbig2::Decoder for Gray8Writer<'_> {
        fn push_pixel(&mut self, black: bool) {
            if black && self.pos < self.buf.len() {
                self.buf[self.pos] = 0x00;
            }
            self.pos += 1;
        }
        fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
            let n = (chunk_count as usize) * 8;
            if black {
                let end = (self.pos + n).min(self.buf.len());
                if self.pos < end {
                    self.buf[self.pos..end].fill(0x00);
                }
            }
            self.pos += n;
        }
        fn next_line(&mut self) {}
    }

    let mut writer = Gray8Writer { buf: &mut out, pos: 0 };
    image
        .decode(&mut writer)
        .map_err(|e| format!("JBIG2 decode: {e:?}"))?;

    Ok((out, width, height))
}

/// Decode a CCITT (fax) encoded image to 8-bit grayscale. Tries Group 4 then
/// Group 3 based on /DecodeParms /K, falling back sensibly when absent.
///
/// `/BlackIs1` (PDF spec §7.4.6) controls decoded bit polarity: `false` (the
/// default) means a `1` bit is white; `true` means a `1` bit is black. The
/// `fax` crate decodes to semantic `Color::Black`/`Color::White` assuming a
/// fixed polarity (matching `BlackIs1 false`). So when `BlackIs1` is true we
/// invert the mapping — otherwise the page comes out white-on-black. Real-world
/// scans encoded with the TIFF/bitmap convention (`BlackIs1 true`) hit this.
fn decode_ccitt(
    content: &[u8],
    width: u32,
    height: u32,
    decode_parms: Option<&Object>,
) -> Result<Vec<u8>, String> {
    // Read a scalar DecodeParms field from either a dict or an array-of-dicts
    // (one entry per /Filter element). Avoids repeating the dict/array walk
    // for Columns, K, and BlackIs1.
    let parm = |key: &[u8]| -> Option<lopdf::Object> {
        let dp = decode_parms?;
        let d = match dp {
            Object::Dictionary(d) => d,
            Object::Array(arr) => arr.first().and_then(|o| match o {
                Object::Dictionary(d) => Some(d),
                _ => None,
            })?,
            _ => return None,
        };
        d.get(key).ok().cloned()
    };

    let columns: u16 = parm(b"Columns")
        .and_then(|v| v.as_i64().ok())
        .unwrap_or(width as i64) as u16;
    let k: i64 = parm(b"K").and_then(|v| v.as_i64().ok()).unwrap_or(0);
    // /BlackIs1 defaults to false (PDF §7.4.6). A bare boolean or an integer
    // (1 = true) both appear in the wild.
    let black_is_1 = match parm(b"BlackIs1") {
        Some(Object::Boolean(b)) => b,
        Some(o) => o.as_i64().map(|v| v != 0).unwrap_or(false),
        None => false,
    };

    let mut pixels: Vec<u8> = Vec::with_capacity((width * height) as usize);
    let decode = |transitions: &[u16], px: &mut Vec<u8>| {
        for pel in fax::decoder::pels(transitions, columns) {
            // fax's Color semantics match BlackIs1=false (1=white). When the
            // stream declares BlackIs1=true, flip black↔white.
            let is_black = pel == fax::Color::Black;
            let ink = if black_is_1 { !is_black } else { is_black };
            px.push(if ink { 0 } else { 255 });
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
    use super::{extract_pages, maybe_downscale, ImageMode};
    use std::path::PathBuf;

    #[test]
    fn downscale_bounds_height_without_upscaling() {
        // 3000×4000 capped at 1600 → 1200×1600 (aspect preserved, rounded).
        let big = image::DynamicImage::new_rgb8(3000, 4000);
        let out = maybe_downscale(big, Some(1600));
        assert_eq!((out.width(), out.height()), (1200, 1600));

        // Already at/below the cap → untouched, no upscale.
        let small = image::DynamicImage::new_rgb8(600, 800);
        let out = maybe_downscale(small, Some(1600));
        assert_eq!((out.width(), out.height()), (600, 800));

        // None → native resolution passthrough.
        let native = image::DynamicImage::new_rgb8(3000, 4000);
        let out = maybe_downscale(native, None);
        assert_eq!((out.width(), out.height()), (3000, 4000));
    }

    #[test]
    fn lzw_round_trips_through_apply_filter() {
        // weezl's encoder is a dev-only use here (weezl is already a direct
        // dep). Both sides use with_tiff_size_switch so the round trip
        // exercises the PDF EarlyChange=1 code-width transition that's the
        // whole point of the LZWDecode arm — a new()/new() pair would pass
        // this test without actually validating the PDF path.
        use weezl::{BitOrder, encode::Encoder};
        use super::apply_filter;
        use lopdf::Dictionary;

        // Enough repetition to grow the code table past 9 bits (where an
        // EarlyChange mismatch would corrupt the output).
        let original = b"ABCDABCDABCDABCD".repeat(64);
        let compressed = Encoder::with_tiff_size_switch(BitOrder::Msb, 8)
            .encode(&original)
            .expect("encode");
        // No DecodeParms → empty dict, predictor block is skipped.
        let out = apply_filter(&compressed, &Dictionary::new(), "LZWDecode", 0)
            .expect("decode");
        assert_eq!(out, original);
    }

    #[test]
    fn lzw_with_png_predictor_round_trips() {
        // LZWDecode + /Predictor 15: the post-decode depngify step must run
        // after LZW decompression. Build a 2-row RGB image, PNG-predict it,
        // LZW-compress, then hand apply_filter a dict carrying DecodeParms.
        use weezl::{BitOrder, encode::Encoder};
        use super::{apply_filter, depngify};
        use lopdf::{Dictionary, Object};

        let columns = 4u32; // 4 pixels
        let colors = 3u32; // RGB
        let bpc = 8u32;
        let row_bytes = (columns * colors) as usize;
        // Two distinct rows so the Up/Sub filters do real work.
        let row0: Vec<u8> = (0..row_bytes).map(|i| i as u8).collect();
        let row1: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_add(50)).collect();
        let raw: Vec<u8> = [row0.as_slice(), row1.as_slice()].concat();

        // PNG-filter the raw pixels the way a producer would (Paeth filter),
        // matching depngify's inverse. We reuse depngify on a None-filtered
        // image to avoid re-implementing the forward pass: forward None filter
        // = prepend a 0 filter byte per row, which depngify inverts trivially.
        // To actually exercise a non-trivial filter, encode row1 with Up(2).
        let stride = row_bytes + 1;
        let mut filtered = vec![0u8; 2 * stride];
        // Row 0: None filter (0), raw bytes.
        filtered[0] = 0;
        filtered[1..1 + row_bytes].copy_from_slice(&raw[..row_bytes]);
        // Row 1: Up filter (2), byte = raw - prev_row.
        filtered[stride] = 2;
        for i in 0..row_bytes {
            filtered[stride + 1 + i] =
                raw[row_bytes + i].wrapping_sub(raw[i]);
        }

        let compressed = Encoder::with_tiff_size_switch(BitOrder::Msb, 8)
            .encode(&filtered)
            .expect("encode");

        // Sanity: depngify alone inverts the filtering correctly.
        let direct = depngify(&filtered, colors, bpc, columns).expect("depngify");
        assert_eq!(direct, raw);

        // Build the image dict: /DecodeParms { /Predictor 15 /Colors 3
        // /BitsPerComponent 8 /Columns 4 }. apply_filter reads it via
        // read_predictor at filter_index 0.
        let mut dp = Dictionary::new();
        dp.set("Predictor", Object::Integer(15));
        dp.set("Colors", Object::Integer(colors as i64));
        dp.set("BitsPerComponent", Object::Integer(bpc as i64));
        dp.set("Columns", Object::Integer(columns as i64));
        let mut dict = Dictionary::new();
        dict.set("DecodeParms", Object::Dictionary(dp));

        let out = apply_filter(&compressed, &dict, "LZWDecode", 0).expect("decode");
        assert_eq!(out, raw);
    }

    #[test]
    fn lzw_rejects_garbage() {
        use super::apply_filter;
        use lopdf::Dictionary;
        let err = apply_filter(b"not lzw data at all", &Dictionary::new(), "LZWDecode", 0)
            .unwrap_err();
        assert!(
            err.contains("LZWDecode"),
            "expected an LZWDecode error, got: {err}"
        );
    }

    /// depngify must handle sub-byte sample depths (bpc 1/2/4): row width is
    /// the byte-packed width and bpp is ceil(colors*bpc/8). This is the
    /// `sample_png.pdf` (1-bpc grayscale + Predictor 15) failure, distilled
    /// to a tiny self-contained case so it runs in CI without the fixture.
    #[test]
    fn depngify_handles_sub_byte_bpc() {
        use super::depngify;
        // 1-bpc grayscale, 16 pixels/row → 2 bytes/row packed.
        // Two rows, Paeth(4) filter on row 1 to exercise the left/up/upleft
        // path at sub-byte bpp (=1 here).
        let columns = 16u32;
        let colors = 1u32;
        let bpc = 1u32;
        let row_bytes = (columns as usize * colors as usize * bpc as usize).div_ceil(8); // 2
        let row0 = [0b1010_1010, 0b1100_1100]; // raw row 0
        let row1 = [0b0000_0000, 0b0000_0000]; // raw row 1 (all black, say)
        let raw: Vec<u8> = [row0.as_slice(), row1.as_slice()].concat();

        // PNG-predict: row 0 = None(0), row 1 = Up(2) so enc = raw - prev.
        let stride = row_bytes + 1;
        let mut filtered = vec![0u8; 2 * stride];
        filtered[0] = 0;
        filtered[1..1 + row_bytes].copy_from_slice(&raw[..row_bytes]);
        filtered[stride] = 2; // Up
        for i in 0..row_bytes {
            filtered[stride + 1 + i] = raw[row_bytes + i].wrapping_sub(raw[i]);
        }

        let out = depngify(&filtered, colors, bpc, columns).expect("depngify 1bpc");
        assert_eq!(out, raw);

        // And the data-length check keys off the packed stride, not columns.
        // A 1-byte-short buffer must be rejected, not silently misdecode.
        let short = &filtered[..filtered.len() - 1];
        assert!(depngify(short, colors, bpc, columns).is_err());
    }

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
        let pages = extract_pages(&bytes, |_, _| {}, ImageMode::Gray, None).expect("extract succeeds");
        assert!(!pages.is_empty(), "expected at least one extracted page");
        for png in &pages {
            assert!(!png.is_empty(), "extracted page PNG must be non-empty");
            assert_eq!(&png[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        }
    }

    /// Regression for the FlateDecode-wrapped-JPEG case (`/Filter
    /// [/FlateDecode /DCTDecode]`): the old code early-returned on
    /// DCTDecode and handed still-zlib bytes to the JPEG decoder, dropping
    /// every page. Reproduced by `sample_pdf/tmp.pdf`. Skipped when the
    /// sample dir isn't present (not bundled in CI).
    #[test]
    fn extracts_flate_wrapped_jpeg_pages() {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sample_pdf/tmp.pdf");
        let Some(path) = (p.exists()).then_some(p) else {
            eprintln!("skip: sample_pdf/tmp.pdf not present");
            return;
        };
        let bytes = std::fs::read(&path).expect("read tmp.pdf");
        let pages = extract_pages(&bytes, |_, _| {}, ImageMode::Gray, None).expect("extract succeeds");
        // tmp.pdf has 5 pages, every one a FlateDecode→DCTDecode image.
        assert_eq!(pages.len(), 5, "all 5 pages should extract, got {}", pages.len());
        for (i, png) in pages.iter().enumerate() {
            assert!(!png.is_empty(), "page {} PNG empty", i + 1);
            assert_eq!(
                &png[0..8],
                &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
                "page {} is not a PNG",
                i + 1
            );
        }
    }

    /// Regression for 1-bpc grayscale images with a PNG predictor (`/Filter
    /// /FlateDecode /DecodeParms { /Predictor 15 /BitsPerComponent 1 ... }`):
    /// depngify used to hard-reject bpc != 8, dropping every page. Reproduced
    /// by `sample_pdf/sample_png.pdf` (5 pages, 3904×4976 @ 1bpc). Skipped
    /// when the sample dir isn't present (not bundled in CI).
    #[test]
    fn extracts_1bpc_png_predictor_pages() {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sample_pdf/sample_png.pdf");
        let Some(path) = (p.exists()).then_some(p) else {
            eprintln!("skip: sample_pdf/sample_png.pdf not present");
            return;
        };
        let bytes = std::fs::read(&path).expect("read sample_png.pdf");
        let pages = extract_pages(&bytes, |_, _| {}, ImageMode::Gray, None).expect("extract succeeds");
        assert_eq!(
            pages.len(),
            5,
            "all 5 pages should extract, got {}",
            pages.len()
        );
        for (i, png) in pages.iter().enumerate() {
            assert!(!png.is_empty(), "page {} PNG empty", i + 1);
            assert_eq!(
                &png[0..8],
                &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
                "page {} is not a PNG",
                i + 1
            );
        }
    }

    /// Regression for JBIG2-compressed scanned PDFs: previously every JBIG2
    /// page was dropped with "JBIG2 not supported". `tmp_2.pdf` is 135 pages
    /// of which 126 are JBIG2 (1-bpc bilevel, with a shared /JBIG2Globals
    /// stream) and 9 are JPEG — so a working extractor gets all 135. We
    /// assert a high bar (>= 130) rather than exactly 135 to tolerate a
    /// handful of edge-case decode failures from the young hayro-jbig2 crate.
    /// Skipped when the sample dir isn't present (not bundled in CI).
    #[test]
    fn extracts_jbig2_pages() {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sample_pdf/tmp_2.pdf");
        let Some(path) = (p.exists()).then_some(p) else {
            eprintln!("skip: sample_pdf/tmp_2.pdf not present");
            return;
        };
        let bytes = std::fs::read(&path).expect("read tmp_2.pdf");
        let pages = extract_pages(&bytes, |_, _| {}, ImageMode::Gray, None).expect("extract succeeds");
        assert!(
            pages.len() >= 130,
            "expected >=130 of 135 pages, got {}",
            pages.len()
        );
        for (i, png) in pages.iter().enumerate() {
            assert!(!png.is_empty(), "page {} PNG empty", i + 1);
            assert_eq!(
                &png[0..8],
                &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
                "page {} is not a PNG",
                i + 1
            );
        }
    }

    /// Regression for CCITT `/BlackIs1` handling: a 1-bit fax scan encoded
    /// with `/BlackIs1 true` (the TIFF/bitmap convention) used to come out
    /// inverted (white text on black). The page should be ~5-15% dark (text
    /// on white paper), not ~95%. Skipped when the sample isn't bundled.
    #[test]
    fn extracts_ccitt_black_is1_not_inverted() {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sample_pdf/sample_ccitt.pdf");
        let Some(path) = (p.exists()).then_some(p) else {
            eprintln!("skip: sample_pdf/sample_ccitt.pdf not present");
            return;
        };
        let bytes = std::fs::read(&path).expect("read sample_ccitt.pdf");
        let pages = extract_pages(&bytes, |_, _| {}, ImageMode::Gray, None).expect("extract succeeds");
        assert_eq!(pages.len(), 5, "expected 5 pages, got {}", pages.len());

        // Each page should be predominantly white (a text scan), not inverted.
        // Assert <50% dark — an inverted page is ~95%, a correct one ~5-15%.
        for (i, png) in pages.iter().enumerate() {
            assert_eq!(
                &png[0..8],
                &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
                "page {} is not a PNG",
                i + 1
            );
            let img = image::load_from_memory(png).expect("decode PNG");
            let g = img.to_luma8();
            let total = (g.width() as usize) * (g.height() as usize);
            let dark = g.pixels().filter(|p| p.0[0] < 128).count();
            let pct = dark as f32 * 100.0 / total as f32;
            assert!(
                pct < 50.0,
                "page {} looks inverted ({pct:.1}% dark) — BlackIs1 handling regressed",
                i + 1
            );
        }
    }

    /// Render-path regression for JBIG2: hayro's renderer applies its own
    /// internal JBIG2 filter (a separate copy of hayro-jbig2 pulled by
    /// hayro-syntax). Before we patched that copy to our vendored build, the
    /// unpatched MAX_INSTANCES = 10_000 silently failed and rendered every
    /// JBIG2 page blank. Renders the first page and asserts it's not empty —
    /// catches both a total render failure and a blank-output regression.
    /// Skipped when the sample dir isn't present (not bundled in CI).
    #[test]
    fn renders_jbig2_page_not_blank() {
        use super::{ImageMode, render_pages};
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sample_pdf/tmp_2.pdf");
        let Some(path) = (p.exists()).then_some(p) else {
            eprintln!("skip: sample_pdf/tmp_2.pdf not present");
            return;
        };
        let bytes = std::fs::read(&path).expect("read tmp_2.pdf");
        // Render just the first page at a small height to keep the test fast.
        // hayro doesn't expose per-page selection, so render and check page 1.
        let pages = render_pages(&bytes, 400, |_, _| {}, ImageMode::Gray).expect("render succeeds");
        assert!(!pages.is_empty(), "render produced no pages");
        let png = &pages[0];
        assert_eq!(
            &png[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            "rendered page 1 is not a PNG"
        );

        // Decode the PNG and assert it isn't blank/white. A text-bearing scan
        // is ~5-20% dark; a blank render (the failure mode) is ~0%. Use a low
        // bar (>1%) to stay robust across content.
        let img = image::load_from_memory(png).expect("decode rendered PNG");
        let rgb = img.to_rgb8();
        let total = (rgb.width() as usize) * (rgb.height() as usize);
        let dark = rgb
            .pixels()
            .filter(|p| p.0[0] < 128 || p.0[1] < 128 || p.0[2] < 128)
            .count();
        let pct = dark as f32 * 100.0 / total as f32;
        assert!(
            pct > 1.0,
            "rendered page 1 looks blank ({dark}/{total} = {pct:.1}% dark) — \
             JBIG2 render path likely regressed"
        );
    }

    #[test]
    fn rejects_garbage_bytes() {
        let err = extract_pages(b"not a pdf", |_, _| {}, ImageMode::Gray, None).unwrap_err();
        assert!(
            err.to_lowercase().contains("load") || err.to_lowercase().contains("pdf"),
            "expected a load error, got: {err}"
        );
    }
}
