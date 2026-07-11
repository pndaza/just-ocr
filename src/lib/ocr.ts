import { invoke } from "@tauri-apps/api/core";
import { save, open } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";

/** How the OCR engine returns its result. hOCR is structured XML with
 * per-word bounding boxes; text is plain UTF-8. */
export type OutputMode = "text" | "hocr";

export interface OcrOpts {
  language: string;
  psm: number;
  whitelist: string | null;
  outputMode: OutputMode;
}

export interface OcrResult {
  text: string;
  confidence: number;
  elapsed_ms: number;
}

/** A single file in the batch queue. */
export type JobStatus = "queued" | "running" | "done" | "error";

export interface Job {
  id: number;
  name: string;
  bytes: Uint8Array;
  url: string; // object URL for thumbnail
  status: JobStatus;
  text: string;
  /** The mode this job's `text` was produced in (drives display/export). */
  outputMode: OutputMode;
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
    url: URL.createObjectURL(file),
    status: "queued",
    text: "",
    outputMode: "text",
    confidence: -1,
    elapsedMs: 0,
    error: null,
  }));
}

/** Build jobs from pre-read bytes (e.g. files dropped via the native event). */
export function makeJobsFromReadFiles(files: { name: string; bytes: number[] }[]): Job[] {
  return files.map((f) => {
    const bytes = new Uint8Array(f.bytes);
    // Create a Blob URL so the thumbnail/preview <img> can render it.
    const blob = new Blob([bytes]);
    return {
      id: nextId++,
      name: f.name,
      bytes,
      url: URL.createObjectURL(blob),
      status: "queued" as const,
      text: "",
      outputMode: "text" as const,
      confidence: -1,
      elapsedMs: 0,
      error: null,
    };
  });
}

export async function availableLanguages(): Promise<string[]> {
  return invoke<string[]>("available_languages");
}

export interface ReadFile {
  name: string;
  bytes: number[];
}

/** Read files from disk by absolute path (for native drag-drop). */
export async function readFiles(paths: string[]): Promise<ReadFile[]> {
  return invoke<ReadFile[]>("read_files", { paths });
}

export async function ocrFromBytes(
  bytes: Uint8Array,
  opts: OcrOpts
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
 * Format is chosen by the dialog's default extension.
 */
export async function exportResults(jobs: Job[]): Promise<void> {
  const done = jobs.filter((j) => j.status === "done");
  if (!done.length) return;

  const dest = await save({
    title: "Export OCR results",
    defaultPath: "ocr-results",
    filters: [
      { name: "CSV", extensions: ["csv"] },
      { name: "Text", extensions: ["txt"] },
      { name: "hOCR", extensions: ["hocr", "html"] },
    ],
  });
  if (!dest) return; // user cancelled

  const lower = dest.toLowerCase();
  const isCsv = lower.endsWith(".csv");
  const isHocr = lower.endsWith(".hocr") || lower.endsWith(".html");
  let content: string;
  if (isCsv) {
    const rows = ["filename,confidence,elapsed_ms,text"];
    for (const j of done) {
      rows.push(
        [
          csvField(j.name),
          j.confidence,
          j.elapsedMs,
          // hOCR XML is kept verbatim; plain text gets a trailing trim.
          csvField(j.outputMode === "hocr" ? j.text : j.text.replace(/\s+$/, "")),
        ].join(",")
      );
    }
    content = rows.join("\n") + "\n";
  } else if (isHocr) {
    // Each job's hOCR is a standalone document; separate them with XML
    // comments so the file remains parseable per-block.
    const blocks = done.map((j) => {
      if (j.outputMode === "hocr") {
        return `<!-- ${j.name}  (${j.confidence}% conf, ${j.elapsedMs} ms) -->\n${j.text}`;
      }
      // A text-mode job has no hOCR; record its result as a comment.
      const body = j.text.replace(/\s+$/, "");
      return `<!-- ${j.name}  (${j.confidence}% conf, ${j.elapsedMs} ms) — plain text, no hOCR -->\n<!-- ${body.replace(/-->/g, "-")} -->`;
    });
    content = blocks.join("\n\n") + "\n";
  } else {
    const blocks = done.map(
      (j) =>
        `=== ${j.name}  (${j.confidence}% conf, ${j.elapsedMs} ms) ===\n${j.text.replace(/\s+$/, "")}`
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
  variant: string
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

