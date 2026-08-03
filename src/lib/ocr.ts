import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { save, open } from "@tauri-apps/plugin-dialog";
import { readFile, remove, writeFile } from "@tauri-apps/plugin-fs";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { plainText, type OcrResult } from "./result";

export type { OcrResult, LineBox } from "./result";

/** How a PDF is turned into per-page images before OCR.
 * - "extract": pull the embedded raster scan (fast, native resolution)
 * - "render":  rasterize the page at 1600px height (handles vector content) */
export type PdfMode = "extract" | "render";

/** Color format for the per-page PNGs a PDF is turned into before OCR.
 * Gray is the default (smaller, no accuracy loss — recognizers binarize
 * internally); Color keeps the source as-is. */
export type ImageMode = "color" | "gray";

/** Recognizer choice (Myanmar path only). Ignored for other languages, where
 * Tesseract handles both segmentation and recognition. */
export type Engine = "tesseract" | "kraken";

/** Segmentation stage for the Myanmar path (line-box detection). Ignored for
 * other languages. PP-OCR (tiny detector + quad postprocess) is the default;
 * PP-OCR (poly) opts into the wider small detector + multi-point polygon
 * postprocess + curvature-gated dewarp, which helps dense/curved Burmese;
 * Kraken is the baseline-aware alternative. */
export type Segmenter = "kraken" | "ppocr" | "ppocr-poly";

export interface OcrOpts {
  engine: Engine;
  language: string;
  /** Tesseract page-segmentation mode (0-13). Used by the non-Myanmar path
   * (full-page Tesseract); ignored for Myanmar, where Kraken segments. */
  psm: number;
  /** Myanmar path only. Which line-box detector runs before recognition:
   * "ppocr" (PP-OCRv6 tiny + quad, default), "ppocr-poly" (PP-OCRv6 small +
   * polygon), or "kraken". */
  segmenter: Segmenter;
}

/** A single file in the batch queue. */
export type JobStatus = "queued" | "running" | "done" | "error";

export interface Job {
  id: number;
  name: string;
  bytes: Uint8Array;
  /** For PDF pages, the temp PNG path. When set, `bytes` is empty and the
   * pixels are read from disk on demand (thumbnail + OCR) instead of held in
   * memory. `null` for regular image files. */
  path: string | null;
  url: string; // object URL for thumbnail (created lazily for path-based jobs)
  status: JobStatus;
  /** Structured OCR result from the backend; null until the job is `done`. */
  result: OcrResult | null;
  confidence: number;
  elapsedMs: number;
  error: string | null;
}

let nextId = 1;
export function makeJob(file: File): Promise<Job> {
  return file.arrayBuffer().then((buf) => ({
    id: nextId++,
    name: file.name,
    bytes: new Uint8Array(buf),
    path: null,
    url: URL.createObjectURL(file),
    status: "queued",
    result: null,
    confidence: -1,
    elapsedMs: 0,
    error: null,
  }));
}

/** Build jobs from pre-read files. A `path` (PDF page temp PNG) is used when
 * present; otherwise `bytes` (a regular image) is turned into a Blob URL. The
 * thumbnail for path-based jobs is loaded lazily via `ensureThumb`. */
export function makeJobsFromReadFiles(
  files: { name: string; bytes?: number[]; path?: string }[],
): Job[] {
  return files.map((f) => {
    if (f.path) {
      return {
        id: nextId++,
        name: f.name,
        bytes: new Uint8Array(),
        path: f.path,
        url: "", // filled in by ensureThumb() when the row becomes visible
        status: "queued" as const,
        result: null,
        confidence: -1,
        elapsedMs: 0,
        error: null,
      };
    }
    const bytes = new Uint8Array(f.bytes ?? []);
    // Create a Blob URL so the thumbnail/preview <img> can render it.
    const blob = new Blob([bytes]);
    return {
      id: nextId++,
      name: f.name,
      bytes,
      path: null,
      url: URL.createObjectURL(blob),
      status: "queued" as const,
      result: null,
      confidence: -1,
      elapsedMs: 0,
      error: null,
    };
  });
}

/**
 * Lazily load the thumbnail for a path-based job: read the temp PNG once and
 * cache it as a Blob URL on the job. Called for visible rows only (thumbnail
 * virtualization) and for the preview, so we never ship all page images at
 * once. No-op if the job has no path or its URL is already set.
 */
export async function ensureThumb(job: Job): Promise<void> {
  if (!job.path || job.url) return;
  try {
    const data = await readFile(job.path);
    job.url = URL.createObjectURL(new Blob([data]));
  } catch (e) {
    console.warn(`Could not load thumbnail for "${job.name}":`, e);
  }
}

/** Return the pixel bytes for a job, reading from its temp file if path-based. */
export async function readJobBytes(job: Job): Promise<Uint8Array> {
  if (job.path) return readFile(job.path);
  return job.bytes;
}

/** Best-effort removal of a path-based job's temp file (called on remove/clear). */
export async function disposeJobFile(job: Job): Promise<void> {
  if (!job.path) return;
  try {
    await remove(job.path);
  } catch {
    /* temp file may already be gone; ignore */
  }
}

export async function availableLanguages(): Promise<string[]> {
  return invoke<string[]>("available_languages");
}

export interface ReadFile {
  name: string;
  /** Inline bytes for regular images; absent for PDF pages (which use `path`). */
  bytes?: number[];
  /** Temp PNG path for PDF pages; absent for regular images. */
  path?: string;
}

/** Read files from disk by absolute path (for native drag-drop). */
export async function readFiles(paths: string[]): Promise<ReadFile[]> {
  return invoke<ReadFile[]>("read_files", { paths });
}

/** True if the file name has a .pdf extension (case-insensitive). */
export function isPdf(name: string): boolean {
  return /\.pdf$/i.test(name);
}

/** Progress payload emitted by the Rust `render_pdf` command per page. */
export interface PdfProgress {
  name: string;
  total: number;
  done: number;
}

/**
 * Extract or render each page of a PDF to a PNG via the Rust `render_pdf`
 * command. Returns one ReadFile per page, named `<stem> · p<n>`.
 *
 * `onProgress(done, total)` is called as each page is processed, driven by the
 * `pdf-progress` event the backend emits. Used to show a progress bar in the
 * PDF-mode dialog while a large PDF is read.
 */
export async function renderPdf(
  name: string,
  bytes: Uint8Array,
  mode: PdfMode,
  onProgress?: (done: number, total: number) => void,
  imageMode?: ImageMode,
): Promise<ReadFile[]> {
  let unlisten: UnlistenFn | null = null;
  if (onProgress) {
    // Listen before invoking so no per-page event is missed. The backend tags
    // each event with the PDF name; ignore events for other files in a batch.
    unlisten = await listen<PdfProgress>("pdf-progress", (e) => {
      if (e.payload.name === name) onProgress(e.payload.done, e.payload.total);
    });
  }
  try {
    return await invoke<ReadFile[]>("render_pdf", {
      pdfName: name,
      bytes: Array.from(bytes),
      mode,
      imageMode,
    });
  } finally {
    unlisten?.();
  }
}

export async function ocrFromBytes(
  bytes: Uint8Array,
  opts: OcrOpts,
): Promise<OcrResult> {
  // Tauri serdes: pass a plain number array for Vec<u8>.
  return invoke<OcrResult>("ocr_from_bytes", {
    bytes: Array.from(bytes),
    opts,
  });
}

/**
 * Write all completed jobs to a single .txt file via a native save dialog.
 *
 * Each completed job becomes a block:
 *
 *     === filename  (90% conf, 120 ms) ===
 *     <recognized text, with merge-paragraphs projection applied>
 *
 * Blocks are separated by a blank line. `mergeParagraphs` (default false)
 * is the same projection used by the Output panel, so the exported file
 * matches what the user sees on screen.
 */
export async function exportResults(
  jobs: Job[],
  opts?: { mergeParagraphs?: boolean },
): Promise<void> {
  const done = jobs.filter((j) => j.status === "done" && j.result);
  if (!done.length) return;

  // Resolve a concrete default directory from the backend (~/Documents, with
  // a home-dir fallback) and join it with the default filename. A bare
  // "ocr-results.txt" defaultPath left NSSavePanel to open wherever it last
  // remembered — sometimes an app-internal path — so exported files vanished
  // from the user's expected Downloads/Documents. Pinning the dir fixes that.
  let defaultPath = "ocr-results.txt";
  try {
    const dir = await invoke<string>("default_save_dir");
    if (dir) {
      // Join manually rather than pulling in a path lib; the dialog accepts a
      // full path string and treats the trailing component as the filename.
      defaultPath = `${dir.replace(/\/$/, "")}/ocr-results.txt`;
    }
  } catch {
    // backend call failed (older build, permission issue) — fall back to the
    // bare filename and let NSSavePanel pick the dir, as before.
  }

  const dest = await save({
    title: "Export OCR results",
    defaultPath,
    filters: [{ name: "Text", extensions: ["txt"] }],
  });
  if (!dest) return; // user cancelled

  const textOpts = opts?.mergeParagraphs ? { mergeParagraphs: true } : undefined;
  const blocks = done.map((j) => {
    const conf = j.confidence >= 0 ? `  (${j.confidence}% conf, ${j.elapsedMs} ms)` : `  (${j.elapsedMs} ms)`;
    return `=== ${j.name}${conf} ===\n` + plainText(j.result!, textOpts).replace(/\s+$/, "");
  });
  const content = blocks.join("\n\n") + "\n";

  const encoder = new TextEncoder();
  await writeFile(dest, encoder.encode(content));

  // Reveal the just-saved file in Finder/Explorer. Finder does not refresh its
  // directory listing when an external app writes a file, so without this the
  // user often couldn't see the export until they manually closed/reopened the
  // window or hit Cmd+Shift+G. Revealing is the macOS convention and forces the
  // file to appear, selected, in its parent folder. Best-effort: a failure here
  // (e.g. headless, no file manager) doesn't undo a successful write.
  try {
    await revealItemInDir(dest);
  } catch (e) {
    console.warn(`Could not reveal "${dest}" in file manager:`, e);
  }
}

// ── Language model management ────────────────────────────────────────────────

export interface LanguageInfo {
  code: string;
  name: string;
  source: "embedded" | "installed" | "available";
}

export interface DownloadProgress {
  language: string;
  downloaded: number;
  total: number;
}

export async function listLanguages(): Promise<LanguageInfo[]> {
  return invoke<LanguageInfo[]>("list_languages");
}

export async function downloadableLanguages(): Promise<LanguageInfo[]> {
  return invoke<LanguageInfo[]>("downloadable_languages");
}

export async function downloadLanguage(
  code: string,
  variant: string,
): Promise<void> {
  await invoke("download_language", { language: code, variant });
}

export async function installLocalLanguage(): Promise<LanguageInfo | null> {
  const picked = await open({
    title: "Select a .traineddata file",
    multiple: false,
    filters: [{ name: "Tesseract traineddata", extensions: ["traineddata"] }],
  });
  if (!picked || Array.isArray(picked)) return null;
  return invoke<LanguageInfo>("install_local_language", {
    sourcePath: picked,
  });
}

export async function deleteLanguage(code: string): Promise<void> {
  await invoke("delete_language", { code });
}

// ── Last-used OCR settings (persisted) ───────────────────────────────────────
// Mirrors the theme persistence in theme.ts: the chosen engine + language are
// stored in localStorage and pre-selected on the next launch. loadLanguages()
// still validates the language value against the available models and falls
// back if it was removed in the meantime.

const LAST_LANG_KEY = "just-ocr:language";
const LAST_ENGINE_KEY = "just-ocr:engine";
const LAST_SEGMENTER_KEY = "just-ocr:segmenter";
const MERGE_PARAGRAPHS_KEY = "just-ocr:merge-paragraphs";

/** Read the last-used OCR language from localStorage, or null if unset. */
export function lastLanguage(): string | null {
  try {
    return localStorage.getItem(LAST_LANG_KEY) ?? null;
  } catch {
    // storage may be unavailable (private mode) — behave as unset
    return null;
  }
}

/** Persist the chosen OCR language so it is pre-selected on next launch. */
export function saveLanguage(lang: string): void {
  try {
    localStorage.setItem(LAST_LANG_KEY, lang);
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

/** Read the last-used engine from localStorage; defaults to "tesseract". */
export function lastEngine(): Engine {
  try {
    return localStorage.getItem(LAST_ENGINE_KEY) === "kraken"
      ? "kraken"
      : "tesseract";
  } catch {
    return "tesseract";
  }
}

/** Persist the chosen engine so it is pre-selected on next launch. */
export function saveEngine(engine: Engine): void {
  try {
    localStorage.setItem(LAST_ENGINE_KEY, engine);
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

/** Read the last-used Myanmar segmenter from localStorage; defaults to "ppocr".
 * Validates the stored value against the known segmenter ids — anything stale
 * (e.g. a prior dev build's "ppocr-small") or missing falls back to "ppocr". */
export function lastSegmenter(): Segmenter {
  const KNOWN: Segmenter[] = ["kraken", "ppocr", "ppocr-poly"];
  try {
    const v = localStorage.getItem(LAST_SEGMENTER_KEY);
    return (v && KNOWN.includes(v as Segmenter) ? v : "ppocr") as Segmenter;
  } catch {
    // storage may be unavailable (private mode) — use the default
    return "ppocr";
  }
}

/** Persist the chosen segmenter so it is pre-selected on next launch. */
export function saveSegmenter(segmenter: Segmenter): void {
  try {
    localStorage.setItem(LAST_SEGMENTER_KEY, segmenter);
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

/** Read the merge-paragraphs view preference; defaults to false (line-by-line
 * output, the legacy behaviour). True → recognized lines are grouped into
 * paragraphs by the geometry heuristic in result.ts, for both the on-screen
 * text panel and TXT/CSV export. Does NOT affect what the OCR engine returns,
 * only how the result lines are projected for display. */
export function lastMergeParagraphs(): boolean {
  try {
    return localStorage.getItem(MERGE_PARAGRAPHS_KEY) === "true";
  } catch {
    // storage may be unavailable (private mode) — use the default
    return false;
  }
}

/** Persist the merge-paragraphs preference so it is sticky across launches. */
export function saveMergeParagraphs(on: boolean): void {
  try {
    localStorage.setItem(MERGE_PARAGRAPHS_KEY, on ? "true" : "false");
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}
