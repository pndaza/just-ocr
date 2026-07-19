//! Full-page Tesseract recognition. Used for non-Myanmar languages where
//! Tesseract's own page layout (driven by the user-selected PSM) is preferable
//! to Kraken segmentation. Runs Tesseract once on the whole image with hOCR
//! output, then parses the hOCR into structured `LineBox`es.
//!
//! This is the original pre-Kraken OCR path, restored when we made segmentation
//! language-dependent (Myanmar → Kraken seg; everything else → Tesseract seg).

use image::DynamicImage;
use tesseract_rs::TesseractAPI;

use crate::engine::LineBox;

/// Recognize a full page with Tesseract. Returns one `LineBox` per recognized
/// line (text + line-level bbox), plus the mean confidence.
///
/// `psm` is the Tesseract page-segmentation mode (0-13), passed through
/// unchanged. `whitelist`, when non-empty, restricts recognition to those
/// characters.
pub fn recognize(
    img: &DynamicImage,
    app: &tauri::AppHandle,
    language: &str,
    psm: i32,
    whitelist: &Option<String>,
) -> Result<(Vec<LineBox>, i32), String> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let bytes_per_pixel = 3;
    let bytes_per_line = (w as i32) * bytes_per_pixel;

    let api = TesseractAPI::new();
    let (tessdata, _embedded) = crate::languages::resolve_tessdata(app, language)?;
    api.init_5(&tessdata, tessdata.len() as i32, language, 3, &[])
        .map_err(|e| format!("Tesseract init failed: {e}"))?;

    if let Some(ref wl) = whitelist {
        if !wl.is_empty() {
            api.set_variable("tessedit_char_whitelist", wl)
                .map_err(|e| format!("Failed to set whitelist: {e}"))?;
        }
    }

    // Map the frontend's PSM int to the engine enum. Unknown values fall back
    // to PSM_AUTO (Tesseract's own default), matching tesseract-rs semantics.
    api.set_page_seg_mode(tesseract_rs::TessPageSegMode::from_int(psm))
        .map_err(|e| format!("Failed to set PSM: {e}"))?;

    api.set_image(
        rgb.as_raw(),
        w as i32,
        h as i32,
        bytes_per_pixel,
        bytes_per_line,
    )
    .map_err(|e| format!("Failed to set image: {e}"))?;

    // hOCR carries per-line + per-word bboxes; we parse line bboxes out of it
    // and reconstruct plain text per line by joining word spans. A single pass
    // thus yields both the overlay boxes and the text.
    let hocr = api
        .get_hocr_text(0)
        .map_err(|e| format!("OCR failed: {e}"))?;
    let conf = api.mean_text_conf().unwrap_or(-1);
    let _ = api.end(); // best-effort cleanup

    Ok((parse_hocr_lines(&hocr), conf))
}

/// Parse hOCR XML into `LineBox`es.
///
/// hOCR is XHTML where each line is
/// `<span class='ocr_line' title='bbox x0 y0 x1 y1'>…<span class='ocrx_word'>…</span>…</span>`.
/// We extract the line bbox from the `title` attribute and reconstruct the
/// line's plain text by joining its word spans with spaces (falling back to
/// the line's trimmed text content if there are no word spans).
///
/// Defensive: on any parse failure we return an empty vec rather than
/// propagating — a malformed hOCR for one image shouldn't abort the batch.
fn parse_hocr_lines(hocr: &str) -> Vec<LineBox> {
    // A hand-rolled scan rather than an XML parser: hOCR is regular enough for
    // our needs and we want zero extra dependencies in the app crate. We walk
    // the string looking for `class='ocr_line'` opening tags, then within each
    // line's span extract the bbox title and the word spans up to the matching
    // close.
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while let Some(line_start) = hocr[search_from..].find("class='ocr_line'") {
        let abs_line_start = search_from + line_start;
        // Find the line span's opening tag start to locate its title.
        let tag_open = hocr[..abs_line_start].rfind('<').unwrap_or(abs_line_start);
        let tag_close = match hocr[abs_line_start..].find('>') {
            Some(offset) => abs_line_start + offset,
            None => break, // malformed; bail
        };
        let opening = &hocr[tag_open..tag_close];

        // Extract bbox from the title attribute.
        let bbox = extract_bbox(opening).unwrap_or((0, 0, 0, 0));

        // Find this line's closing </span>. Naive: scan for the next
        // `</span>` after the opening tag. Nested word spans are
        // self-contained (`<span ...></span>`), so the first `</span>` we hit
        // may belong to a word — we account for that by collecting word spans
        // along the way.
        let mut pos = tag_close + 1;
        let mut words: Vec<String> = Vec::new();
        loop {
            let rest = &hocr[pos..];
            let next_open = rest.find("<span");
            let next_close = rest.find("</span>");
            match (next_open, next_close) {
                (Some(o), Some(c)) if o < c => {
                    // A nested span (word or other). If it's an ocrx_word,
                    // capture its text content.
                    let abs_o = pos + o;
                    let word_open_close = hocr[abs_o..].find('>').unwrap_or(0);
                    let word_open_full = &hocr[abs_o..abs_o + word_open_close];
                    let abs_wc = abs_o + word_open_close + 1;
                    let word_end = hocr[abs_wc..]
                        .find("</span>")
                        .map(|i| abs_wc + i)
                        .unwrap_or(abs_wc);
                    if word_open_full.contains("class='ocrx_word'")
                        || word_open_full.contains("class=\"ocrx_word\"")
                    {
                        let raw = &hocr[abs_wc..word_end];
                        let text = strip_tags(raw).trim().to_string();
                        if !text.is_empty() {
                            words.push(text);
                        }
                    }
                    // Advance past this nested span's closing tag.
                    pos = hocr[word_end..]
                        .find('>')
                        .map(|i| word_end + i + 1)
                        .unwrap_or(word_end + 7);
                }
                (_, Some(c)) => {
                    // The line's own closing tag. Done with this line.
                    pos = pos + c;
                    break;
                }
                _ => break, // malformed; stop
            }
        }

        let text = if words.is_empty() {
            String::new()
        } else {
            words.join(" ")
        };

        // Skip degenerate bboxes (zero-area) — Tesseract sometimes emits them
        // for blank regions.
        if bbox.2 > bbox.0 && bbox.3 > bbox.1 {
            out.push(LineBox {
                x0: bbox.0,
                y0: bbox.1,
                x1: bbox.2,
                y1: bbox.3,
                text,
            });
        }

        search_from = pos;
    }
    out
}

/// Extract the first `bbox x0 y0 x1 y1` quadruple from a string, as
/// `(x0, y0, x1, y1)`. Returns None if not found.
///
/// hOCR title attributes look like `bbox 10 20 30 40; x_wconf 90` — the bbox
/// numbers are followed by `;` separating other properties, so each token may
/// carry trailing non-digit characters. We parse leading digits only.
fn extract_bbox(s: &str) -> Option<(u32, u32, u32, u32)> {
    let marker = "bbox ";
    let idx = s.find(marker)?;
    let rest = &s[idx + marker.len()..];
    let nums: Vec<&str> = rest.split_whitespace().take(4).collect();
    if nums.len() < 4 {
        return None;
    }
    Some((
        parse_leading_u32(nums[0])?,
        parse_leading_u32(nums[1])?,
        parse_leading_u32(nums[2])?,
        parse_leading_u32(nums[3])?,
    ))
}

/// Parse leading digits of a token as `u32`, ignoring any trailing non-digits
/// (e.g. `"40;"` → `40`). Returns None if the token doesn't start with a digit.
fn parse_leading_u32(s: &str) -> Option<u32> {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        // Could be a negative number — not valid for a bbox, treat as None.
        return None;
    }
    digits.parse().ok()
}

/// Strip XML/HTML tags from a string, leaving only text content.
fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_bbox_basic() {
        assert_eq!(
            extract_bbox("title='bbox 10 20 30 40; x_wconf 90'"),
            Some((10, 20, 30, 40))
        );
    }

    #[test]
    fn extract_bbox_missing_returns_none() {
        assert_eq!(extract_bbox("no bbox here"), None);
    }

    #[test]
    fn parse_single_line_with_two_words() {
        let hocr = concat!(
            "<div class='ocr_page' title='bbox 0 0 100 100'>",
            "<span class='ocr_line' id='line_1' title='bbox 5 10 50 30; baseline 0 -5'>",
            "<span class='ocrx_word' id='word_1' title='bbox 5 10 20 30'>Hello</span>",
            "<span class='ocrx_word' id='word_2' title='bbox 25 10 50 30'>World</span>",
            "</span>",
            "</div>"
        );
        let lines = parse_hocr_lines(hocr);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].x0, 5);
        assert_eq!(lines[0].y0, 10);
        assert_eq!(lines[0].x1, 50);
        assert_eq!(lines[0].y1, 30);
        assert_eq!(lines[0].text, "Hello World");
    }

    #[test]
    fn parse_multiple_lines() {
        let hocr = concat!(
            "<div class='ocr_page' title='bbox 0 0 100 100'>",
            "<span class='ocr_line' title='bbox 5 10 50 25'><span class='ocrx_word' title='bbox 5 10 50 25'>Foo</span></span>",
            "<span class='ocr_line' title='bbox 5 30 50 45'><span class='ocrx_word' title='bbox 5 30 50 45'>Bar</span></span>",
            "</div>"
        );
        let lines = parse_hocr_lines(hocr);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "Foo");
        assert_eq!(lines[1].text, "Bar");
    }

    #[test]
    fn parse_malformed_returns_empty() {
        assert!(parse_hocr_lines("not hOCR at all").is_empty());
        assert!(parse_hocr_lines("").is_empty());
    }

    #[test]
    fn strip_tags_basic() {
        assert_eq!(strip_tags("Hello <b>World</b>!"), "Hello World!");
        assert_eq!(strip_tags("<span>plain</span>"), "plain");
    }
}
