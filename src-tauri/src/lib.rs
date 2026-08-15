use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::async_runtime;
use tauri::Emitter;
use tesseract_rs::TesseractAPI;

mod engine;
mod languages;
mod llm_fix;
mod pdf;
mod segmentation;
mod segmenter_adapters;
mod spelling;
mod tesseract_line;
mod tesseract_page;

pub use engine::{LineBox, OcrResult};
pub use kraken_engine::Engine as KrakenEngine;
// Re-export the segmentation types + the poly segmenter for debug examples
// (smoke_ppocr_poly.rs) that exercise the poly path end-to-end. The modules
// themselves stay private; only these items are surfaced.
pub use segmentation::{DetectedLine, Segmenter};
pub use segmenter_adapters::PPOcrPolySegmenter;

/// OCR options sent from the frontend.
///
/// The pipeline shape is **language-driven**:
///   - `language == "mya"` → Kraken does layout (hidden from the user); the
///     recognizer is chosen by `engine` ("kraken" | "tesseract"). `psm` is
///     ignored — Kraken has already segmented.
///   - any other language → full-page Tesseract with `psm` (Tesseract does its
///     own layout). `engine` is ignored.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrOpts {
    /// Recognizer for the Myanmar path: "kraken" (default) or "tesseract".
    /// Ignored for non-Myanmar languages.
    pub engine: String,
    /// Tesseract language code. Also drives the pipeline dispatch: "mya"
    /// routes through Kraken segmentation; anything else routes through
    /// full-page Tesseract.
    pub language: String,
    /// Tesseract page-segmentation mode (0-13). Used by the non-Myanmar path;
    /// ignored by the Myanmar path (Kraken does layout there).
    pub psm: i32,
    /// Segmenter choice for the Myanmar path: `"ppocr"` (default) or `"kraken"`.
    /// `None`/unrecognized → PP-OCR. Ignored for non-Myanmar (full-page Tesseract).
    #[serde(default)]
    pub segmenter: Option<String>,
    /// PP-OCR detector variant for the Myanmar path: `"small"` (default,
    /// accuracy-oriented) or `"tiny"` (faster, smaller, less accurate on dense
    /// text). `None`/unrecognized → small. Ignored when `segmenter` is `"kraken"`
    /// and for non-Myanmar. Each variant lazy-loads its own bundled weights.
    #[serde(default)]
    pub det_variant: Option<String>,
    /// Whether the user wants Burmese spelling-fix applied. NOTE: the backend
    /// no longer applies this at OCR time — `ocr_from_bytes` always returns
    /// raw text, and the frontend drives the fix as a display-time projection
    /// via the separate `fix_burmese_spelling` command. Kept on `OcrOpts` so
    /// the toggle state round-trips for forward-compat and potential future
    /// "fix-at-OCR" modes; ignored by `run_ocr`.
    #[serde(default)]
    pub fix_burmese_spelling: bool,
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

/// Resolve a sensible default directory for the export save dialog, as an
/// absolute path string. Prefers the user's Documents dir; falls back to the
/// home dir, then the current working dir. The frontend joins this with the
/// default filename and passes it as `defaultPath`, so macOS `NSSavePanel`
/// opens in a predictable, user-visible location instead of wherever it last
/// remembered (which was sending exported files to surprising places).
///
/// We prefer Documents over Downloads here because the export is editable text
/// the user is likely to keep, not a throwaway download. Both are honoured if
/// the user navigates elsewhere in the dialog.
#[tauri::command]
fn default_save_dir() -> String {
    dirs::document_dir()
        .or_else(dirs::home_dir)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string())
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
    engine::run_ocr(app, image_bytes, &opts)
}

/// One line's spell-fix result: the corrected text plus how many
/// substitutions the two stages applied to that line. The frontend totals
/// `fixes` across the returned vec to populate the per-result fix count.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FixResult {
    pub text: String,
    pub fixes: u32,
}

/// Apply Burmese spelling normalization + dictionary correction to a list of
/// raw recognized line texts. This is a **display-time projection**, not part
/// of OCR: the backend always returns raw text from `ocr_from_bytes`, and the
/// frontend calls this lazily when the "Fix spelling" toggle is on, caching
/// the result per job so toggling back is instant. Keeping the fix logic in
/// Rust (rather than porting to JS) because the rule set includes 9
/// lookbehind/lookahead regexes that older macOS WKWebView doesn't support.
///
/// Runs on a blocking thread; the regex+dict passes are cheap but not free,
/// and big multi-page batches shouldn't tie up the async runtime.
#[tauri::command]
async fn fix_burmese_spelling(lines: Vec<String>) -> Result<Vec<FixResult>, String> {
    async_runtime::spawn_blocking(move || {
        let t = Instant::now();
        let n = lines.len();
        let out: Vec<FixResult> = lines
            .into_iter()
            .map(|text| {
                let (fixed, fixes) = spelling::fix(&text);
                FixResult { text: fixed, fixes }
            })
            .collect();
        let total: u32 = out.iter().map(|r| r.fixes).sum();
        log::info!(
            "[spelling] fix command: {:.0} ms ({} substitution{} across {} lines)",
            t.elapsed().as_secs_f64() * 1000.0,
            total,
            if total == 1 { "" } else { "s" },
            n
        );
        out
    })
    .await
    .map_err(|e| format!("Spell-fix task failed: {e}"))
}

/// How to turn a PDF page into an image. "extract" pulls the embedded raster
/// scan (fast, native resolution); "render" rasterizes the page at 1600px
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
/// accuracy and speed; ~53px per 30pt glyph (1600 / 30), tall enough to keep
/// body text legible at any page size.
const PDF_RENDER_HEIGHT: u16 = 1600;

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
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init());

    // Updater + process (relaunch after install) are desktop-only. This is a
    // desktop app, so the cfg gate is defensive/documentation rather than
    // strictly necessary — it matches the official Tauri pattern and keeps a
    // future mobile target clean. Registration is split out of the chain
    // because #[cfg(desktop)] cannot attach to a method call mid-chain.
    #[cfg(desktop)]
    {
        builder = builder
            .plugin(tauri_plugin_updater::Builder::new().build())
            .plugin(tauri_plugin_process::init());
    }

    builder
        .setup(|app| {
            // Initialize env_logger. Default to `info` for our crates so the
            // per-stage OCR timing logs print without setting RUST_LOG; set
            // RUST_LOG=debug for per-line recognition logs, RUST_LOG=trace for
            // kraken-rust's internal stage logs.
            let _ = env_logger::Builder::from_env(
                env_logger::Env::default().default_filter_or("info"),
            )
            .format_timestamp_millis()
            .try_init();

            let version = TesseractAPI::version();
            let langs = languages::list_languages(app.handle().clone())
                .into_iter()
                .map(|l| l.code)
                .collect::<Vec<_>>()
                .join(", ");
            log::info!("just-ocr starting — tesseract {version}, languages: {langs}");
            // Kraken models are loaded lazily on the first OCR call (see
            // engine::kraken_engine) so a missing-models error only surfaces
            // when OCR is actually attempted, not at startup.
            // Reclaim temp dirs left by crashes / previous runs.
            sweep_stale_temp_dirs();
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            available_languages,
            read_files,
            default_save_dir,
            ocr_from_bytes,
            render_pdf,
            fix_burmese_spelling,
            llm_fix::llm_spell_check,
            llm_fix::llm_rewrite_pages,
            llm_fix::llm_test_key,
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
