//! Tesseract per-line recognition. A single line crop is fed to Tesseract
//! with `PSM_RAW_LINE` (no further layout work — Kraken already segmented).

use image::DynamicImage;
use tesseract_rs::TesseractAPI;

/// Recognize one line crop. Returns `(text, mean_confidence)` where
/// `mean_confidence` is in [0,100], or -1 if Tesseract reports none.
pub fn recognize(
    crop: &DynamicImage,
    app: &tauri::AppHandle,
    language: &str,
    whitelist: &Option<String>,
) -> Result<(String, i32), String> {
    let rgb = crop.to_rgb8();
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

    // PSM_RAW_LINE (13): the image is a single text line; no page layout.
    api.set_page_seg_mode(tesseract_rs::TessPageSegMode::PSM_RAW_LINE)
        .map_err(|e| format!("Failed to set PSM: {e}"))?;

    api.set_image(rgb.as_raw(), w as i32, h as i32, bytes_per_pixel, bytes_per_line)
        .map_err(|e| format!("Failed to set image: {e}"))?;

    let text = api
        .get_utf8_text()
        .map_err(|e| format!("OCR failed: {e}"))?;
    let conf = api.mean_text_conf().unwrap_or(-1);
    let _ = api.end(); // best-effort cleanup

    Ok((text.trim_end().to_string(), conf))
}
