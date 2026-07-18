import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { save, open } from "@tauri-apps/plugin-dialog";
import { readFile, remove, writeFile } from "@tauri-apps/plugin-fs";
import { plainText, type OcrResult } from "./result";

export type { OcrResult, LineBox } from "./result";

/** How a PDF is turned into per-page images before OCR.
 * - "extract": pull the embedded raster scan (fast, native resolution)
 * - "render":  rasterize the page at 1500px height (handles vector content) */
export type PdfMode = "extract" | "render";

/** Color format for the per-page PNGs a PDF is turned into before OCR.
 * Gray is the Tesseract-friendly default (smaller, no accuracy loss);
 * B&W is Otsu-thresholded for pristine scans; Color keeps the source as-is. */
export type ImageMode = "color" | "gray" | "bw";

/** Recognizer choice. Layout segmentation is always Kraken; this selects the
 * recognizer applied to each line crop. */
export type Engine = "tesseract" | "kraken";

export interface OcrOpts {
  engine: Engine;
  language: string;
  /** Tesseract-only; ignored by the kraken recognizer. */
  whitelist: string | null;
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

/** CSV-escape a single field. */
function csvField(s: string): string {
  if (/[",\n\r]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

/**
 * Write all completed jobs to a combined file via a native save dialog.
 * Format is chosen by the dialog's default extension (CSV or TXT).
 */
export async function exportResults(jobs: Job[]): Promise<void> {
  const done = jobs.filter((j) => j.status === "done" && j.result);
  if (!done.length) return;

  const dest = await save({
    title: "Export OCR results",
    defaultPath: "ocr-results",
    filters: [
      { name: "CSV", extensions: ["csv"] },
      { name: "Text", extensions: ["txt"] },
    ],
  });
  if (!dest) return; // user cancelled

  const lower = dest.toLowerCase();
  const isCsv = lower.endsWith(".csv");
  let content: string;
  if (isCsv) {
    const rows = ["filename,confidence,elapsed_ms,text"];
    for (const j of done) {
      rows.push(
        [
          csvField(j.name),
          j.confidence,
          j.elapsedMs,
          csvField(plainText(j.result!).replace(/\s+$/, "")),
        ].join(","),
      );
    }
    content = rows.join("\n") + "\n";
  } else {
    const blocks = done.map(
      (j) =>
        `=== ${j.name}  (${j.confidence}% conf, ${j.elapsedMs} ms) ===\n` +
        plainText(j.result!).replace(/\s+$/, ""),
    );
    content = blocks.join("\n\n") + "\n";
  }

  const encoder = new TextEncoder();
  await writeFile(dest, encoder.encode(content));
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
