import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { save, open } from "@tauri-apps/plugin-dialog";
import { readFile, remove, writeFile, mkdir } from "@tauri-apps/plugin-fs";
import { join } from "@tauri-apps/api/path";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { plainText, plainTextWithFix, type OcrResult } from "./result";

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
 * other languages. PP-OCR (quad postprocess) is the default;
 * PP-OCR (poly) opts into multi-point polygon postprocess + curvature-gated
 * dewarp, which helps dense/curved Burmese; Kraken is the baseline-aware
 * alternative. */
export type Segmenter = "kraken" | "ppocr" | "ppocr-poly";

/** PP-OCR detector variant (Myanmar path only). "small" (default) is
 * accuracy-oriented; "tiny" is faster/smaller but less accurate on dense text.
 * Ignored when `segmenter` is "kraken". */
export type DetVariant = "small" | "tiny";

export interface OcrOpts {
  engine: Engine;
  language: string;
  /** Tesseract page-segmentation mode (0-13). Used by the non-Myanmar path
   * (full-page Tesseract); ignored for Myanmar, where Kraken segments. */
  psm: number;
  /** Myanmar path only. Which line-box detector runs before recognition:
   * "ppocr" (PP-OCRv6 + quad, default), "ppocr-poly" (PP-OCRv6 + polygon),
   * or "kraken". */
  segmenter: Segmenter;
  /** Myanmar path only. PP-OCR detector backbone width: "small" (default,
   * accuracy-oriented) or "tiny" (faster). Ignored when segmenter is "kraken". */
  detVariant: DetVariant;
  /** Whether to apply the Burmese post-OCR spelling fix (curated wrong→right
   * word list, backend-side). Myanmar-only in effect — the list is Burmese.
   * Unlike `mergeParagraphs`, this changes what the OCR engine returns, so it
   * crosses the IPC boundary inside `opts`.
   *
   * TEMPORARILY UNUSED BY THE UI: the toolbar toggle is parked behind
   * `{#if false}` (rule-based fix is far behind AI spell check) and App
   * forces this to false so no persisted "on" sticks. The backend command
   * and rules (`burmese_spelling_rules.tsv`) stay live for when the toggle
   * returns. */
  fixBurmeseSpelling: boolean;
}

/** A single file in the batch queue. */
export type JobStatus = "queued" | "running" | "done" | "error";

export interface Job {
  id: number;
  name: string;
  /** Source grouping for PDF pages: the PDF's name stem (e.g. "report").
   *  Set by App when a PDF's pages enter the queue; used by image export to
   *  place the pages under a `<group>/` subfolder instead of a flat file.
   *  Undefined for regular image files. */
  group?: string;
  bytes: Uint8Array;
  /** On-disk path when the pixels live on disk instead of in `bytes`: the
   *  app-owned temp PNG for PDF pages, or the original location for
   *  drag-dropped files (which arrive as paths only — see `readFiles`).
   *  `null` for files added via the picker, which hold their bytes in memory.
   *  Ownership matters: only paths inside our temp namespace are ever
   *  deleted (see `disposeJobFile`). */
  path: string | null;
  url: string; // object URL for thumbnail (created lazily for path-based jobs)
  status: JobStatus;
  /** Structured OCR result from the backend; null until the job is `done`. */
  result: OcrResult | null;
  confidence: number;
  elapsedMs: number;
  /** Lazily-computed spell-fix projection of `result`, cached so toggling the
   *  "Fix spelling" switch back on is instant. Null until first computed;
   *  `fixedLines` parallels `result.lines`, `fixes` is the total substitution
   *  count across all lines. Owned by the job so it survives selection /
   *  toggle changes. */
  spellFix: { fixedLines: string[]; fixes: number } | null;
  /** Word-level fixes the user accepted from the AI spell check (Gemini),
   *  applied on top of the page's current basis (manual text, an earlier
   *  llmFix, the spell-fix projection, or raw lines — same precedence the
   *  Text panel shows). Fixes STACK: a later check/apply builds on the
   *  lines already here, so re-checking a page never discards earlier
   *  rounds' verified fixes; `fixes` counts lines differing from the raw
   *  OCR text. Manually-edited pages (see `manualText`) never carry an
   *  llmFix — their AI fixes are written straight into the manual text,
   *  which is authoritative and would shadow a projection here. Same
   *  non-destructive shape as `spellFix`: `job.result` is never mutated,
   *  and re-running OCR clears this. Null until the user applies fixes in
   *  the AI Check panel. */
  llmFix: { fixedLines: string[]; fixes: number } | null;
  /** Manual edits typed into the Text panel. When set, it REPLACES all
   *  projections (raw/spell-fix/AI-fix) for display, copy and export — the
   *  user's hand-edited text is authoritative. Kept as whole text (not
   *  lines) since edits are free-form; geometry overlays are unaffected.
   *  Null until the user types; Revert clears it; re-running OCR resets it. */
  manualText: string | null;
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
    spellFix: null,
    llmFix: null,
    manualText: null,
    error: null,
  }));
}

/** Build jobs from pre-read files. A `path` is used when present — the temp
 * PNG of a PDF page, or the original location of a drag-dropped image — with
 * `bytes` left empty and pixels read lazily (`readJobBytes`); a byte-backed
 * `ReadFile` (picker flow) is turned into a Blob URL instead. The thumbnail
 * for path-based jobs is loaded lazily via `ensureThumb`.
 * `group` (the source PDF's name stem) is stamped on every job when given —
 * see `Job.group`. */
export function makeJobsFromReadFiles(
  files: { name: string; bytes?: number[]; path?: string }[],
  group?: string,
): Job[] {
  return files.map((f) => {
    if (f.path) {
      return {
        id: nextId++,
        name: f.name,
        group,
        bytes: new Uint8Array(),
        path: f.path,
        url: "", // filled in by ensureThumb() when the row becomes visible
        status: "queued" as const,
        result: null,
        confidence: -1,
        elapsedMs: 0,
        spellFix: null,
        llmFix: null,
        manualText: null,
        error: null,
      };
    }
    const bytes = new Uint8Array(f.bytes ?? []);
    // Create a Blob URL so the thumbnail/preview <img> can render it.
    const blob = new Blob([bytes]);
    return {
      id: nextId++,
      name: f.name,
      group,
      bytes,
      path: null,
      url: URL.createObjectURL(blob),
      status: "queued" as const,
      result: null,
      confidence: -1,
      elapsedMs: 0,
      spellFix: null,
      llmFix: null,
      manualText: null,
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

/** True only for paths inside one of the app's own temp dirs —
 * `<temp>/just-ocr-<pid>-<seq>/pN.png`, written by the backend's `render_pdf`
 * (mirrors `just_ocr_temp_pid` in lib.rs, whose naming is load-bearing).
 * Drag-dropped files keep `job.path` at their original location, so this
 * check is what separates "ours, deletable" from "the user's file". */
export function isAppTempPath(path: string): boolean {
  return /[\\/]just-ocr-\d+-\d+[\\/]p\d+\.png$/.test(path);
}

/** Best-effort removal of a job's app-owned temp PNG (called on remove/clear).
 * Guarded by `isAppTempPath` so a drag-dropped image — whose `path` is the
 * user's original file — is never deleted; that was a data-destroying bug.
 * If the guard ever skips a real temp PNG, the backend's startup sweep and
 * shutdown cleanup reclaim it, so failing safe here costs nothing. */
export async function disposeJobFile(job: Job): Promise<void> {
  if (!job.path || !isAppTempPath(job.path)) return;
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
  /** Inline bytes; only used by the file-picker flow. Absent for drag-drop
   *  (paths only — inline `Vec<u8>` IPC is far too slow for multi-MB files)
   *  and for PDF pages (temp PNG `path`). */
  bytes?: number[];
  /** On-disk path; present for drag-drop files and PDF page PNGs. */
  path?: string;
}

/**
 * Read files from disk by absolute path (for native drag-drop). Returns name +
 * path only — the backend deliberately does not ship bytes, since Tauri's
 * `Vec<u8>`-as-JSON-array serialization makes multi-MB files crawl. Jobs load
 * their bytes on demand (`readJobBytes`), and PDFs are processed by path.
 */
export async function readFiles(paths: string[]): Promise<ReadFile[]> {
  return invoke<ReadFile[]>("read_files", { paths });
}

/** True if the file name has a .pdf extension (case-insensitive). */
export function isPdf(name: string): boolean {
  return /\.pdf$/i.test(name);
}

/** The PDF's file stem ("scan_3.pdf" → "scan_3"). Matches the stem the
 *  backend bakes into page names (`page_name`), so a job's group folder and
 *  its "<group> · pN" display name agree. Used as the image-export subfolder
 *  name for a PDF's pages. */
export function pdfStem(name: string): string {
  return name.replace(/\.pdf$/i, "");
}

/**
 * Page count of a PDF via the Rust `pdf_page_count` command (page-tree read
 * only, no decoding). The PDF dialog fetches this when it opens to show
 * "of N pages" and validate the range inputs before processing starts.
 *
 * Pass the file's absolute path when you have one (drag-drop flow): the
 * backend reads it itself, avoiding the `Vec<u8>`-as-JSON IPC tax that makes
 * multi-MB PDFs crawl. Inline bytes are only for the file-picker flow, which
 * already holds them in JS.
 */
export async function pdfPageCount(source: Uint8Array | string): Promise<number> {
  return invoke<number>("pdf_page_count", {
    ...(typeof source === "string" ? { pdfPath: source } : { bytes: Array.from(source) }),
  });
}

/** Progress payload emitted by the Rust `render_pdf` command per page. */
export interface PdfProgress {
  name: string;
  total: number;
  done: number;
}

/**
 * Inclusive 1-based page range restricting PDF processing. An omitted upper
 * bound is expressed as `4294967295` (`u32::MAX`, "from N to the end") — the
 * Rust side treats it as unbounded since documents can't have more pages.
 * Mirrors the Rust `PageRange` struct field-for-field.
 */
export interface PageRange {
  from: number;
  to: number;
}

/**
 * Extract or render each page of a PDF to a PNG via the Rust `render_pdf`
 * command. Returns one ReadFile per page, named `<stem> · p<n>` with the PDF's
 * original page numbers (a ranged selection keeps its true labels).
 *
 * `source` is the PDF as an absolute path (drag-drop flow — the backend reads
 * it itself; sending multi-MB bytes as a JSON number array would crawl) or as
 * inline bytes (file-picker flow, which already holds them in JS).
 *
 * `maxHeight` bounds the output page height in both modes. Extract downscales
 * pages taller than the limit (aspect preserved, never upscales); high-res
 * scans can confuse line segmentation, so the dialog offers bounded sizes —
 * `undefined` keeps native resolution. Render rasterizes at exactly this
 * height, so the dialog always sends a value there.
 *
 * `pageRange` optionally restricts processing to an inclusive 1-based page
 * range (both modes); `undefined` processes every page.
 *
 * `onProgress(done, total)` is called as each page is processed, driven by the
 * `pdf-progress` event the backend emits. Used to show a progress bar in the
 * PDF-mode dialog while a large PDF is read.
 */
export async function renderPdf(
  name: string,
  source: Uint8Array | string,
  mode: PdfMode,
  onProgress?: (done: number, total: number) => void,
  imageMode?: ImageMode,
  maxHeight?: number,
  pageRange?: PageRange,
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
      ...(typeof source === "string" ? { pdfPath: source } : { bytes: Array.from(source) }),
      mode,
      imageMode,
      maxHeight,
      pageRange,
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

/** One line's spell-fix result from the backend. */
export interface FixResult {
  text: string;
  fixes: number;
}

/**
 * Apply Burmese spelling normalization + dictionary correction to a list of
 * raw recognized line texts. This is a **display-time projection**: the OCR
 * result always holds raw text; the frontend calls this lazily when the
 * "Fix spelling" toggle is on and caches the result per job. Each returned
 * `{ text, fixes }` parallels the input line, with `fixes` counting the
 * substitutions applied to that line.
 */
export async function fixBurmeseSpelling(
  lines: string[],
): Promise<FixResult[]> {
  return invoke<FixResult[]>("fix_burmese_spelling", { lines });
}

// ── AI spell check (Gemini via Google AI Studio) ─────────────────────────────

/** One wrong→correct word pair proposed by the LLM for a page. Mirrors the
 *  Rust `LlmWordFix` (camelCase field names on both sides). `line` is the
 *  1-based line within the page the model flagged — the fix applies (and is
 *  validated) only on that line, so a short Burmese substring can't ripple
 *  into other paragraphs of the same page. Absent → page-wide matching. */
export interface LlmWordFix {
  wrong: string;
  correct: string;
  line?: number;
}

/** A flagged word the AI Check panel publishes for the Text panel to
 *  highlight: the word plus the 1-based line it was flagged on (null = the
 *  model didn't scope it to a line — any occurrence highlights). Carrying
 *  the line lets the Text panel mark only the flagged occurrence instead of
 *  every match across the page. */
export interface WordHighlight {
  wrong: string;
  line: number | null;
}

/** All proposed corrections for one page of the batch. `page` is 1-based,
 *  indexing into the `pages` array sent with the request. Pages the model
 *  found no errors on are simply absent from the response. */
export interface LlmPageFix {
  page: number;
  fixes: LlmWordFix[];
}

/**
 * Ask Gemini to proofread a batch of OCR'd page texts and return wrong→correct
 * word pairs per page. One call = one HTTP request (the backend never splits
 * or retries), so callers batch pages themselves — the AI Check panel sends
 * at most PAGES_PER_LLM_BATCH per call. Errors are thrown as strings from
 * the backend (invalid key, quota, model errors are user-actionable there).
 */
export async function llmSpellCheck(
  apiKey: string,
  model: string,
  pages: string[],
): Promise<LlmPageFix[]> {
  return invoke<LlmPageFix[]>("llm_spell_check", { apiKey, model, pages });
}

/** One page's fully rewritten text (direct-fix mode). `lines` parallels the
 *  page's input lines — the model is instructed to keep the exact line
 *  structure so the frontend can diff line-by-line. */
export interface LlmPageText {
  page: number;
  lines: string[];
}

/**
 * Direct-fix counterpart of {@link llmSpellCheck}: the model returns each
 * page's corrected text as an array of lines instead of word pairs. The
 * frontend diffs the lines against the originals and still reviews changes
 * per line. Same batching rules — one call per request, callers chunk.
 */
export async function llmRewritePages(
  apiKey: string,
  model: string,
  pages: string[],
): Promise<LlmPageText[]> {
  return invoke<LlmPageText[]>("llm_rewrite_pages", { apiKey, model, pages });
}

/**
 * Verify a Google AI Studio API key with a minimal request (backend uses the
 * cheap gemini-flash-lite-latest model). Resolves when the key authenticates;
 * rejects with the backend's user-facing message (invalid key, quota,
 * network…).
 */
export async function llmTestKey(apiKey: string): Promise<void> {
  return invoke<void>("llm_test_key", { apiKey });
}

/**
 * Write all completed jobs to a single .txt file via a native save dialog.
 *
 * Each completed job becomes a block of recognized text (with
 * merge-paragraphs + spell-fix projection applied — the same projections the
 * Output panel shows), separated by a blank line. A per-page
 * `=== filename (conf, ms) ===` header option existed previously but was
 * removed: body-only is what the export is for.
 *
 * `mergeParagraphs` (default false) and `fixSpelling` (default false) are the
 * same projections used by the Output panel, so the exported file matches
 * what the user sees on screen.
 */
export async function exportResults(
  jobs: Job[],
  opts?: {
    mergeParagraphs?: boolean;
    fixSpelling?: boolean;
  },
): Promise<void> {
  const done = jobs.filter((j) => j.status === "done" && j.result);
  if (!done.length) return;

  // If spell-fix is on, ensure every done job has its cached projection
  // BEFORE building text. The cache is normally populated lazily by App's
  // toggle-watching effect, but that's fire-and-forget — a user who flips the
  // toggle on and immediately hits Export could race the in-flight compute and
  // silently get RAW text in the file despite the toggle being on. Awaiting
  // here guarantees the exported content matches what's on screen. Jobs with a
  // populated cache skip the IPC call; the await is a no-op for them.
  if (opts?.fixSpelling) {
    await Promise.all(
      done.map(async (j) => {
        if (j.spellFix) return;
        if (!j.result || j.result.lines.length === 0) return;
        try {
          const fixed = await fixBurmeseSpelling(j.result.lines.map((l) => l.text));
          j.spellFix = {
            fixedLines: fixed.map((r) => r.text),
            fixes: fixed.reduce((sum, r) => sum + r.fixes, 0),
          };
        } catch {
          // Backend call failed — leave spellFix null; the body builder below
          // falls back to raw text for this job. Non-fatal.
        }
      }),
    );
  }

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
    // Projection precedence: manual edits are authoritative, then an applied
    // AI fix (built on top of the spell-fix basis lines), then the offline
    // spell-fix when toggled on, then raw. Honors mergeParagraphs where a
    // line-based projection is used.
    const body =
      j.manualText != null
        ? j.manualText
        : j.llmFix
          ? plainTextWithFix(j.result!, j.llmFix.fixedLines, textOpts)
          : opts?.fixSpelling && j.spellFix
            ? plainTextWithFix(j.result!, j.spellFix.fixedLines, textOpts)
            : plainText(j.result!, textOpts);
    return body.replace(/\s+$/, "");
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

// ── AI spell-fix suggestion export (review mode) ─────────────────────────────

/** One suggestion row for {@link exportSpellFixSuggestions} — a wrong→correct
 *  pair the model proposed in review mode, with its provenance and the user's
 *  accept/reject decision. */
export interface SpellFixExportRow {
  /** Page (job) display name — where the flagged word was found. */
  page: string;
  /** 1-based line on that page, when the model scoped the fix to one. */
  line?: number;
  wrong: string;
  correct: string;
  /** The row's checkbox state at export time: yes = applied (or selected,
   *  when exported mid-review), no = left out. The pairs a human accepted
   *  are the verified signal; the rejected ones are still worth keeping. */
  applied: boolean;
}

/** RFC-4180 quoting: wrap a field in double quotes when it contains a comma,
 *  quote, or newline, doubling embedded quotes. */
function csvField(v: string): string {
  return /[",\r\n]/.test(v) ? `"${v.replaceAll('"', '""')}"` : v;
}

/**
 * Write the review-mode spell-fix suggestions to a .csv file via a native
 * save dialog — same flow as {@link exportResults} (default dir, write,
 * best-effort reveal).
 *
 * The suggestions are otherwise ephemeral: they live only in the AI Check
 * panel and are dropped when it closes or a new check starts. The export is
 * the way to keep them — the wrong→correct pairs (with page/line and whether
 * the user applied them) are exactly what compiling a common-spelling-error
 * list needs. Rows are written verbatim, including repeats: how often a pair
 * occurs across pages is itself the frequency signal, so deduping here would
 * throw it away.
 */
export async function exportSpellFixSuggestions(
  rows: SpellFixExportRow[],
): Promise<void> {
  if (!rows.length) return;

  let defaultPath = "ai-spell-fixes.csv";
  try {
    const dir = await invoke<string>("default_save_dir");
    if (dir) defaultPath = `${dir.replace(/\/$/, "")}/ai-spell-fixes.csv`;
  } catch {
    // fall back to the bare filename, as exportResults does
  }

  const dest = await save({
    title: "Export spell-fix suggestions",
    defaultPath,
    filters: [{ name: "CSV", extensions: ["csv"] }],
  });
  if (!dest) return; // user cancelled

  const lines = [
    "page,line,wrong,correct,applied",
    ...rows.map((r) =>
      [
        csvField(r.page),
        r.line == null ? "" : String(r.line),
        csvField(r.wrong),
        csvField(r.correct),
        r.applied ? "yes" : "no",
      ].join(","),
    ),
  ];
  await writeFile(dest, new TextEncoder().encode(lines.join("\n") + "\n"));

  try {
    await revealItemInDir(dest);
  } catch (e) {
    console.warn(`Could not reveal "${dest}" in file manager:`, e);
  }
}

// ── Image export (thumbnail panel bottom bar) ────────────────────────────────

/** Output format for the "export images to folder" feature. */
export type ImageExportFormat = "png" | "jpg";

/** True when the byte stream is already the target format (magic-number sniff
 *  — PNG: 89 50 4E 47, JPG: FF D8). Matching sources pass through untouched,
 *  so a PNG→PNG export is a byte-for-byte copy with no re-encode loss. */
function isFormat(bytes: Uint8Array, format: ImageExportFormat): boolean {
  if (bytes.length < 4) return false;
  if (format === "png") {
    return bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
  }
  return bytes[0] === 0xff && bytes[1] === 0xd8;
}

/** Decode `bytes` and re-encode to the target format via a canvas. Runs in the
 *  webview (no backend round-trip); createImageBitmap honors EXIF orientation,
 *  and JPG gets a white backdrop because it has no alpha channel. */
async function reencode(bytes: Uint8Array, format: ImageExportFormat): Promise<Uint8Array> {
  const bmp = await createImageBitmap(new Blob([bytes]));
  const canvas = document.createElement("canvas");
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext("2d")!;
  if (format === "jpg") {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const type = format === "png" ? "image/png" : "image/jpeg";
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("canvas encode failed"))),
      type,
      0.92, // JPEG quality — high enough that OCR-grade scans stay clean
    );
  });
  return new Uint8Array(await blob.arrayBuffer());
}

/** File name stem without any extension ("scan.jpg" → "scan"; PDF-page names
 *  like "report · p1" carry no extension and pass through unchanged). */
function stemOf(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}

/**
 * Write every job's image into a folder the user picks (native directory
 * dialog), converting to `format` when the source isn't already that format
 * (matching sources are copied verbatim).
 *
 * PDF-page jobs (those carrying `Job.group`, set when the PDF's pages entered
 * the queue) are written into a `<pdf name>/` subfolder — pages keep their
 * `p001`, `p002`… labels (zero-padded so they sort) without the redundant
 * PDF-name prefix. Plain image jobs land directly in the chosen folder. Name
 * collisions within this run are de-duplicated with a "-2", "-3" suffix (two
 * PDFs can share a stem, so their pages can both have a "p001").
 *
 * Returns the number of images written, or 0 when the user cancelled the
 * folder dialog or the queue was empty. `onProgress(done, total)` fires after
 * each image so the bar can show a counter. Best-effort reveal of the folder
 * at the end (same Finder-refresh rationale as `exportResults`).
 */
export async function exportImages(
  jobs: Job[],
  format: ImageExportFormat,
  onProgress?: (done: number, total: number) => void,
): Promise<number> {
  if (!jobs.length) return 0;
  const dir = await open({
    title: "Choose a folder for the exported images",
    directory: true,
    multiple: false,
  });
  if (!dir) return 0; // user cancelled

  const ext = format === "png" ? "png" : "jpg";
  // Subfolders created so far (PDF groups) — mkdir is not free, so each is
  // created once per run.
  const madeDirs = new Set<string>();
  // Dedupe keys include the subfolder so "report/p1" and a flat "p1" never
  // shadow each other.
  const used = new Set<string>();
  let written = 0;
  let processed = 0;

  for (const job of jobs) {
    try {
      const src = await readJobBytes(job);
      const out = isFormat(src, format) ? src : await reencode(src, format);
      let targetDir = dir;
      let base = stemOf(job.name);
      if (job.group) {
        if (!madeDirs.has(job.group)) {
          await mkdir(await join(dir, job.group), { recursive: true });
          madeDirs.add(job.group);
        }
        targetDir = await join(dir, job.group);
        // Inside the group folder the name is just the page label — drop the
        // "<pdf name> · " prefix the queue display carries, and zero-pad the
        // page number (p3 → p003, matching the backend's temp-file `p{:03}`
        // convention) so the files sort in page order in Finder/Explorer.
        const prefix = `${job.group} · `;
        if (job.name.startsWith(prefix)) {
          const label = stemOf(job.name.slice(prefix.length));
          base = label.replace(/^p(\d+)$/, (_m, d: string) => `p${d.padStart(3, "0")}`);
        }
      }
      let fname = `${base}.${ext}`;
      const key = (f: string) => `${job.group ?? ""}/${f}`.toLowerCase();
      for (let n = 2; used.has(key(fname)); n++) {
        fname = `${base}-${n}.${ext}`;
      }
      used.add(key(fname));
      await writeFile(await join(targetDir, fname), out);
      written++;
    } catch (e) {
      // One bad image (unreadable source, failed encode) skips; the rest
      // still export.
      console.warn(`Could not export "${job.name}":`, e);
    }
    processed++;
    onProgress?.(processed, jobs.length);
  }

  try {
    await revealItemInDir(dir);
  } catch (e) {
    console.warn(`Could not reveal "${dir}" in file manager:`, e);
  }
  return written;
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
const LAST_DET_VARIANT_KEY = "just-ocr:det-variant";
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
 * Validates the stored value against the segmenter ids currently surfaced in
 * the UI — anything stale, missing, or a hidden option (e.g. "kraken", which
 * is retained in code but not accurate enough to expose yet) falls back to
 * "ppocr". The `"kraken"` variant stays in the `Segmenter` type + backend. */
export function lastSegmenter(): Segmenter {
  // Note: "kraken" is intentionally absent here — it's hidden from the UI
  // until accuracy improves, so a previously-persisted choice is migrated.
  const KNOWN: Segmenter[] = ["ppocr", "ppocr-poly"];
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

/** Read the last-used PP-OCR detector variant from localStorage; defaults to
 * "small" (the accuracy-oriented backbone). Validates the stored value —
 * anything stale or missing falls back to "small". Only affects the Myanmar
 * path, and is ignored when the segmenter is Kraken. */
export function lastDetVariant(): DetVariant {
  const KNOWN: DetVariant[] = ["small", "tiny"];
  try {
    const v = localStorage.getItem(LAST_DET_VARIANT_KEY);
    return (v && KNOWN.includes(v as DetVariant) ? v : "small") as DetVariant;
  } catch {
    // storage may be unavailable (private mode) — use the default
    return "small";
  }
}

/** Persist the chosen detector variant so it is pre-selected on next launch. */
export function saveDetVariant(variant: DetVariant): void {
  try {
    localStorage.setItem(LAST_DET_VARIANT_KEY, variant);
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

// ── Burmese spelling fix (persisted) ─────────────────────────────────────────
// Unlike mergeParagraphs (display-only), this changes what the OCR engine
// returns, so it crosses the IPC boundary inside `opts.fixBurmeseSpelling`.
// Persisted the same way so the toggle is sticky across launches. Default
// off: the user opts in, so they can compare raw vs corrected output.

const FIX_BURMESE_SPELLING_KEY = "just-ocr:fix-burmese-spelling";

/** Read the spelling-fix preference; defaults to false (off). */
export function lastFixBurmeseSpelling(): boolean {
  try {
    return localStorage.getItem(FIX_BURMESE_SPELLING_KEY) === "true";
  } catch {
    // storage may be unavailable (private mode) — use the default
    return false;
  }
}

/** Persist the spelling-fix preference so it is sticky across launches. */
export function saveFixBurmeseSpelling(on: boolean): void {
  try {
    localStorage.setItem(FIX_BURMESE_SPELLING_KEY, on ? "true" : "false");
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

// ── AI spell check preferences (persisted) ───────────────────────────────────
// Google AI Studio (Gemini) credentials for the AI Check tool. This is the
// app's only online feature — everything else works fully offline. The key
// lives in localStorage on the user's machine (like every other pref here;
// not encrypted, never sent anywhere but Google's API) and is passed to the
// backend per call.

const LLM_API_KEY_KEY = "just-ocr:llm-api-key";
const LLM_MODEL_KEY = "just-ocr:llm-model";
const LLM_BATCH_SIZE_KEY = "just-ocr:llm-batch-size";

/** Flash-family models offered in the AI Check dialog. The id is
 *  interpolated into the Gemini REST path by the backend; keep the ids
 *  exactly as Google names them. The first entry is the default — see the
 *  inline note for why that's flash-lite rather than full flash. */
export const LLM_MODELS = [
  // First entry = the default selection. Flash is the better proofreader,
  // but its free tier (~20 requests/day) runs dry partway through even a
  // modest book, while flash-lite allows ~500 — so the default is
  // flash-lite-latest, the alias tracking Google's current GA flash-lite
  // (the -latest aliases always exist, unlike pinned versions that may not
  // have shipped — a speculative "3.7" sat here once and 404'd).
  { value: "gemini-flash-lite-latest", label: "Flash Lite (latest)" },
  { value: "gemini-flash-latest", label: "Flash (latest)" },
  { value: "gemini-3.6-flash", label: "Gemini 3.6 Flash" },
  { value: "gemini-3.5-flash", label: "Gemini 3.5 Flash" },
  { value: "gemini-3.5-flash-lite", label: "Gemini 3.5 Flash Lite" },
  { value: "gemini-3.1-flash-lite", label: "Gemini 3.1 Flash Lite" },
] as const;

export type LlmModel = (typeof LLM_MODELS)[number]["value"];

/** Free-tier daily REQUEST limits on Google AI Studio (per project+model).
 *  Flash models are capped hard (~20/day); flash-lite far more loosely
 *  (~500/day). Used to warn before a check that would exhaust the quota
 *  partway through. Unknown ids assume the stricter flash cap. */
export function llmDailyLimit(model: string): number {
  if (model.includes("flash-lite")) return 500;
  return 20;
}

/** Batch sizes (pages per request) offered in the AI Check dialog. Bigger
 *  batches mean fewer requests but risk output-token limits and make a
 *  retry more expensive; the middle entry (30) is the default. */
export const LLM_BATCH_SIZES = [10, 20, 30, 40, 50] as const;

export type LlmBatchSize = (typeof LLM_BATCH_SIZES)[number];

/** How the AI Check talks to Gemini. "rewrite" (default, labeled "Auto
 *  apply") returns each page's corrected text, diffed per line and applied
 *  the moment each batch lands — broader fixes (punctuation, spacing,
 *  phrasing) at higher output-token cost, so smaller batches are wise.
 *  "review" (labeled "Manual apply") returns wrong→correct word pairs the
 *  user picks from before anything changes. */
export type AiCheckMode = "review" | "rewrite";

/** Read the chosen batch size; defaults to 30 (LLM_BATCH_SIZES[2]).
 * Validates the stored value so anything stale falls back to the default. */
export function lastLlmBatchSize(): LlmBatchSize {
  try {
    const v = Number(localStorage.getItem(LLM_BATCH_SIZE_KEY));
    return (
      LLM_BATCH_SIZES.find((s) => s === v) ?? LLM_BATCH_SIZES[2]
    );
  } catch {
    // storage may be unavailable (private mode) — use the default
    return LLM_BATCH_SIZES[2];
  }
}

/** Persist the chosen batch size so it is pre-selected on next launch. */
export function saveLlmBatchSize(size: LlmBatchSize): void {
  try {
    localStorage.setItem(LLM_BATCH_SIZE_KEY, String(size));
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

const LLM_CONCURRENCY_KEY = "just-ocr:llm-concurrency";

/** Parallel Gemini requests the AI Check keeps in flight. Two by default:
 *  roughly halves wall-clock time on big checks while staying comfortably
 *  inside the free tier's per-minute request limit; three is there for the
 *  patient with flash-lite's looser limits. */
export const LLM_CONCURRENCY = [1, 2, 3] as const;

export type LlmConcurrency = (typeof LLM_CONCURRENCY)[number];

/** Read the chosen concurrency; defaults to 2. Validates the stored value
 *  so anything stale falls back to the default. */
export function lastLlmConcurrency(): LlmConcurrency {
  try {
    const v = Number(localStorage.getItem(LLM_CONCURRENCY_KEY));
    return LLM_CONCURRENCY.find((c) => c === v) ?? 2;
  } catch {
    // storage may be unavailable (private mode) — use the default
    return 2;
  }
}

/** Persist the chosen concurrency so it is pre-selected on next launch. */
export function saveLlmConcurrency(concurrency: LlmConcurrency): void {
  try {
    localStorage.setItem(LLM_CONCURRENCY_KEY, String(concurrency));
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

const AI_CHECK_MODE_KEY = "just-ocr:ai-check-mode";

/** Read the AI Check mode preference; defaults to "rewrite" (Auto apply —
 *  instant corrected text with per-line revert is the smoother first run;
 *  anyone who wants to approve each fix switches to Manual apply). */
export function lastAiCheckMode(): AiCheckMode {
  try {
    return localStorage.getItem(AI_CHECK_MODE_KEY) === "review"
      ? "review"
      : "rewrite";
  } catch {
    // storage may be unavailable (private mode) — use the default
    return "rewrite";
  }
}

/** Persist the AI Check mode so it is pre-selected on next launch. */
export function saveAiCheckMode(mode: AiCheckMode): void {
  try {
    localStorage.setItem(AI_CHECK_MODE_KEY, mode);
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

/** Read the stored Google AI Studio API key, or "" if unset. */
export function lastLlmApiKey(): string {
  try {
    return localStorage.getItem(LLM_API_KEY_KEY) ?? "";
  } catch {
    // storage may be unavailable (private mode) — behave as unset
    return "";
  }
}

/** Persist the API key so the AI Check tool is usable on next launch. */
export function saveLlmApiKey(key: string): void {
  try {
    localStorage.setItem(LLM_API_KEY_KEY, key);
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

/** Read the chosen flash model; defaults to the first entry in LLM_MODELS.
 * Validates the stored value so a retired model id falls back gracefully. */
export function lastLlmModel(): LlmModel {
  try {
    const v = localStorage.getItem(LLM_MODEL_KEY);
    return (
      LLM_MODELS.find((m) => m.value === v)?.value ?? LLM_MODELS[0].value
    );
  } catch {
    // storage may be unavailable (private mode) — use the default
    return LLM_MODELS[0].value;
  }
}

/** Persist the chosen model so it is pre-selected on next launch. */
export function saveLlmModel(model: LlmModel): void {
  try {
    localStorage.setItem(LLM_MODEL_KEY, model);
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}

// ── Toolbar "Fix spelling" visibility (persisted) ────────────────────────────
// Whether the Myanmar "Fix spelling" checkbox appears in the toolbar. Some
// users never use the offline Burmese fix (or find the toolbar crowded), so
// it can be tucked away here. Default true — the toggle's behavior itself is
// unchanged and stays sticky independently.

const SHOW_FIX_SPELLING_KEY = "just-ocr:show-fix-spelling";

/** Read the toolbar "Fix spelling" visibility preference; defaults to true. */
export function lastShowFixSpelling(): boolean {
  try {
    const v = localStorage.getItem(SHOW_FIX_SPELLING_KEY);
    // Default true: null/absent → show. Explicit "false" → hide.
    return v !== "false";
  } catch {
    // storage may be unavailable (private mode) — use the default
    return true;
  }
}

/** Persist the toolbar "Fix spelling" visibility preference. */
export function saveShowFixSpelling(on: boolean): void {
  try {
    localStorage.setItem(SHOW_FIX_SPELLING_KEY, on ? "true" : "false");
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
}
