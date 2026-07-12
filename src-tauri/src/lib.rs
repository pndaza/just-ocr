use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::async_runtime;
use tesseract_rs::TesseractAPI;

mod languages;
mod pdf;

/// OCR options sent from the frontend.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrOpts {
    pub language: String,
    pub psm: i32,
    /// When non-null, restricts recognition to these characters.
    pub whitelist: Option<String>,
    /// "hocr" returns hOCR XML; anything else (or None) returns plain text.
    pub output_mode: Option<String>,
}

/// OCR result returned to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrResult {
    pub text: String,
    pub confidence: i32,
    pub elapsed_ms: u64,
}

/// Return the list of languages available for OCR: embedded + user-installed.
#[tauri::command]
fn available_languages(app: tauri::AppHandle) -> Vec<String> {
    languages::list_languages(app)
        .into_iter()
        .map(|l| l.code)
        .collect()
}

/// A file read from disk, returned to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadFile {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Read one or more files from disk by absolute path. Used to ingest files
/// dropped onto the window via the native drag-drop event (which only provides
/// paths, not bytes).
#[tauri::command]
fn read_files(paths: Vec<String>) -> Vec<ReadFile> {
    paths
        .into_iter()
        .filter_map(|p| {
            let path = std::path::Path::new(&p);
            let name = path.file_name()?.to_string_lossy().to_string();
            let bytes = std::fs::read(path).ok()?;
            Some(ReadFile { name, bytes })
        })
        .collect()
}

/// Run OCR on a raw image file (PNG/JPG/BMP/WebP/...). The heavy work is
/// performed on a blocking thread so the UI stays responsive.
#[tauri::command]
async fn ocr_from_bytes(
    app: tauri::AppHandle,
    bytes: Vec<u8>,
    opts: OcrOpts,
) -> Result<OcrResult, String> {
    async_runtime::spawn_blocking(move || run_ocr(&app, &bytes, opts))
        .await
        .map_err(|e| format!("OCR task failed: {e}"))?
}

fn run_ocr(
    app: &tauri::AppHandle,
    image_bytes: &[u8],
    opts: OcrOpts,
) -> Result<OcrResult, String> {
    let started = Instant::now();

    // Decode the image into raw RGB8 pixels for tesseract.
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| format!("Failed to decode image: {e}"))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let bytes_per_pixel = 3;
    let bytes_per_line = (w as i32) * bytes_per_pixel;

    // A fresh engine per call: init is cheap relative to recognition and
    // avoids cross-call state from a shared handle. Resolve the traineddata
    // from embedded bytes first, then user-installed files on disk.
    let api = TesseractAPI::new();
    let (tessdata, _embedded) = languages::resolve_tessdata(app, &opts.language)?;
    api.init_5(&tessdata, tessdata.len() as i32, &opts.language, 3, &[])
        .map_err(|e| format!("Tesseract init failed: {e}"))?;

    if let Some(ref whitelist) = opts.whitelist {
        if !whitelist.is_empty() {
            api.set_variable("tessedit_char_whitelist", whitelist)
                .map_err(|e| format!("Failed to set whitelist: {e}"))?;
        }
    }

    // Map our PSM int to the engine enum.
    api.set_page_seg_mode(tesseract_rs::TessPageSegMode::from_int(opts.psm))
        .map_err(|e| format!("Failed to set PSM: {e}"))?;

    api.set_image(
        img.as_raw(),
        w as i32,
        h as i32,
        bytes_per_pixel,
        bytes_per_line,
    )
    .map_err(|e| format!("Failed to set image: {e}"))?;

    // hOCR output returns structured XML with per-word bounding boxes;
    // otherwise we return plain UTF-8 text. Both getters run recognition on
    // the image set above, so only one is called.
    let is_hocr = opts
        .output_mode
        .as_deref()
        .map(|m| m.eq_ignore_ascii_case("hocr"))
        .unwrap_or(false);
    let text = if is_hocr {
        api.get_hocr_text(0)
            .map_err(|e| format!("OCR (hOCR) failed: {e}"))?
    } else {
        api.get_utf8_text()
            .map_err(|e| format!("OCR failed: {e}"))?
    };

    let confidence = api.mean_text_conf().unwrap_or(-1);
    let _ = api.end(); // best-effort cleanup

    Ok(OcrResult {
        text,
        confidence,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// DPI is fixed at 300 — the Tesseract-recommended sweet spot for scans.
const PDF_RENDER_DPI: f32 = 300.0;

/// Rasterize every page of a PDF to a PNG at 300 DPI. Runs on a blocking
/// thread so the UI stays responsive while large scans render. `pdf_name` is
/// the original file name, used to label each page `<stem> · p<n>`.
#[tauri::command]
async fn render_pdf(pdf_name: String, bytes: Vec<u8>) -> Result<Vec<ReadFile>, String> {
    async_runtime::spawn_blocking(move || {
        let pages = pdf::render_pages(&bytes, PDF_RENDER_DPI)?;
        Ok(pages
            .into_iter()
            .enumerate()
            .map(|(i, png)| ReadFile {
                name: pdf::page_name(&pdf_name, i + 1),
                bytes: png,
            })
            .collect())
    })
    .await
    .map_err(|e| format!("PDF render task failed: {e}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let version = TesseractAPI::version();
            let langs = languages::list_languages(app.handle().clone())
                .into_iter()
                .map(|l| l.code)
                .collect::<Vec<_>>()
                .join(", ");
            println!("just-ocr starting — tesseract {version}, languages: {langs}");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            available_languages,
            read_files,
            ocr_from_bytes,
            render_pdf,
            languages::list_languages,
            languages::downloadable_languages,
            languages::download_language,
            languages::install_local_language,
            languages::delete_language,
        ])
        .run(tauri::generate_context!())
        .expect("error while running just-ocr");
}

#[cfg(test)]
mod tests {
    use tesseract_rs::TessPageSegMode;

    /// Validates that PSM integers from the frontend map onto every enum
    /// variant without panicking. The real OCR pipeline is exercised manually
    /// via `cargo tauri dev` (see README).
    #[test]
    fn psm_mapping_covers_all_options() {
        for v in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] {
            let _ = TessPageSegMode::from_int(v);
        }
        // Unknown values fall back to PSM_AUTO.
        assert_eq!(TessPageSegMode::from_int(999), TessPageSegMode::PSM_AUTO);
    }
}
