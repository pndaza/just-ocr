use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::async_runtime;
use tauri::Emitter;
use tesseract_rs::TesseractAPI;

mod languages;
mod pdf;
#[allow(unused)]
mod kraken;

/// OCR options sent from the frontend.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrOpts {
    pub language: String,
    pub psm: i32,
    /// When non-null, restricts recognition to these characters.
    pub whitelist: Option<String>,
    /// Selects how the frontend *displays* the result: "hocr" shows the raw
    /// hOCR XML, anything else shows plain text. The engine always recognizes
    /// as hOCR (so bounding boxes are available for the preview in both modes);
    /// this flag only steers rendering/export on the frontend.
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

/// Progress for a PDF being turned into per-page images. Emitted by
/// `render_pdf` (and consumed by the frontend's in-app dialog) so the user
/// sees how far along extraction/rendering is.
#[derive(Clone, Serialize)]
struct PdfProgress {
    /// Source PDF file name, so the frontend can correlate events when several
    /// PDFs are processed in one batch.
    name: String,
    /// Total number of pages in the PDF.
    total: usize,
    /// Number of pages processed so far (0 right after load).
    done: usize,
}

/// A file read from disk, returned to the frontend.
///
/// For images read directly (e.g. native drag-drop) the bytes are returned
/// inline. For PDF pages produced by `render_pdf`, the PNG is written to a temp
/// file and only its `path` is returned — shipping many multi-megabyte page
/// images back over the IPC as JSON would be far too slow, so the frontend
/// loads them from disk on demand (thumbnail + OCR).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadFile {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
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
            Some(ReadFile {
                name,
                bytes: Some(bytes),
                path: None,
            })
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

    // Always run recognition as hOCR. The hOCR XML carries per-word bounding
    // boxes (overlaid on the preview for both output modes) and is parsed on
    // the frontend into either raw hOCR (hOCR mode) or plain text (text mode).
    // A single pass thus serves both outputs instead of running two getters.
    let text = api
        .get_hocr_text(0)
        .map_err(|e| format!("OCR failed: {e}"))?;

    let confidence = api.mean_text_conf().unwrap_or(-1);
    let _ = api.end(); // best-effort cleanup

    Ok(OcrResult {
        text,
        confidence,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// How to turn a PDF page into an image. "extract" pulls the embedded raster
/// scan (fast, native resolution); "render" rasterizes the page at 1500px
/// height (slower, handles vector/mixed content).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PdfMode {
    Extract,
    Render,
}

impl Default for PdfMode {
    fn default() -> Self {
        PdfMode::Extract
    }
}

/// Fixed output height for render mode. Chosen as a balance between OCR
/// accuracy and speed; wide enough to keep body text legible at any page size.
const PDF_RENDER_HEIGHT: u16 = 1500;

/// PDF page PNGs are written under the system temp dir to
/// `just-ocr-<pid>-<seq>`. We namespace by PID so a directory is unambiguously
/// "ours" only when it carries the current PID — letting us sweep leftovers from
/// crashes or previous launches without touching another running instance.
fn just_ocr_temp_pid(name: &str) -> Option<u32> {
    name.strip_prefix("just-ocr-")?
        .split('-')
        .next()?
        .parse()
        .ok()
}

/// Delete every `just-ocr-<pid>-<seq>` directory whose PID is not the current
/// process. Run at startup to reclaim temp dirs abandoned by crashes or prior
/// sessions (including a previous normal exit whose cleanup we never reached).
fn sweep_stale_temp_dirs() {
    let pid = std::process::id();
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(dir_pid) = just_ocr_temp_pid(&name) {
            if dir_pid != pid {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
}

/// Delete this session's `just-ocr-<pid>-<seq>` directories. Run after the
/// event loop ends so temp PNGs don't accumulate across launches.
fn remove_session_temp_dirs() {
    let pid = std::process::id();
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(dir_pid) = just_ocr_temp_pid(&name) {
            if dir_pid == pid {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
}

/// Turn each page of a PDF into a PNG, via extract or render mode. Runs on a
/// blocking thread so the UI stays responsive while large scans decode.
/// `pdf_name` labels each page `<stem> · p<n>`.
///
/// Emits a `pdf-progress` event as each page is processed so the frontend can
/// show a progress bar (extraction/rendering time scales with page count).
#[tauri::command]
async fn render_pdf(
    app: tauri::AppHandle,
    pdf_name: String,
    bytes: Vec<u8>,
    mode: Option<PdfMode>,
    image_mode: Option<String>,
) -> Result<Vec<ReadFile>, String> {
    let mode = mode.unwrap_or_default();
    let image_mode = match image_mode.as_deref() {
        Some("color") => pdf::ImageMode::Color,
        Some("bw") => pdf::ImageMode::Bw,
        _ => pdf::ImageMode::Gray,
    };
    async_runtime::spawn_blocking(move || {
        // Build an emitter the pure pdf functions can call per page. Cloning
        // the AppHandle/name into the closure keeps it 'static + Send.
        let app_handle = app.clone();
        let name = pdf_name.clone();
        let emit = move |done: usize, total: usize| {
            let _ = app_handle.emit("pdf-progress", PdfProgress { name: name.clone(), total, done });
        };
        let pages = match mode {
            PdfMode::Extract => pdf::extract_pages(&bytes, &emit, image_mode)?,
            PdfMode::Render => pdf::render_pages(&bytes, PDF_RENDER_HEIGHT, &emit, image_mode)?,
        };

        // Write each page PNG to a temp file and return only its path. Returning
        // the bytes inline would serialize many multi-MB images as JSON over the
        // IPC (Tauri encodes Vec<u8> as number arrays) — slow and memory-heavy.
        // The frontend loads them from disk on demand (thumbnail + OCR).
        static RENDER_SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = RENDER_SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("just-ocr-{}-{}", std::process::id(), seq));
        std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create temp dir: {e}"))?;

        let out = pages
            .into_iter()
            .enumerate()
            .map(|(i, png)| {
                let fname = format!("p{:03}.png", i + 1);
                let fpath = dir.join(&fname);
                std::fs::write(&fpath, &png).map_err(|e| format!("Failed to write {fname}: {e}"))?;
                Ok(ReadFile {
                    name: pdf::page_name(&pdf_name, i + 1),
                    bytes: None,
                    path: Some(fpath.to_string_lossy().into_owned()),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(out)
    })
    .await
    .map_err(|e| format!("PDF task failed: {e}"))?
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
            // Reclaim temp dirs left by crashes / previous runs.
            sweep_stale_temp_dirs();
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
    // The event loop has ended; remove this session's temp PDF-page PNGs.
    remove_session_temp_dirs();
}

#[cfg(test)]
mod tests {
    use crate::just_ocr_temp_pid;
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

    #[test]
    fn temp_dir_pid_parsing() {
        assert_eq!(just_ocr_temp_pid("just-ocr-1234-0"), Some(1234));
        assert_eq!(just_ocr_temp_pid("just-ocr-999-42"), Some(999));
        // Wrong prefix or no PID → None (never swept).
        assert_eq!(just_ocr_temp_pid("other-dir-1234-0"), None);
        assert_eq!(just_ocr_temp_pid("just-ocr-abc-0"), None);
        assert_eq!(just_ocr_temp_pid("just-ocr-"), None);
    }
}
