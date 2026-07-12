# PDF Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PDF support: dropping a multi-page PDF expands to one job per page, each OCR'd independently at a fixed 300 DPI.

**Architecture:** A new Rust command `render_pdf` uses the `pdq` crate (pure Rust, renders via `hayro`) to rasterize each PDF page to a PNG in a temp directory, then reads the PNGs back into memory. The frontend branches on `.pdf` extension: PDFs route through `renderPdf` first, then into the existing `makeJobsFromReadFiles` job-creation path. Preview, Output, OCR, and Export are unchanged — a rasterized page is byte-identical to an image job.

**Tech Stack:** Rust (`pdq` 0.3 crate, `hayro` render backend), Tauri 2, TypeScript/Svelte 5.

**Reference spec:** `docs/superpowers/specs/2026-07-12-pdf-support-design.md`

---

### Task 1: Add `pdq` dependency and bump MSRV

**Files:**
- Modify: `src-tauri/Cargo.toml`

- [ ] **Step 1: Add `pdq` and bump `rust-version`**

In `src-tauri/Cargo.toml`, change `rust-version = "1.88"` to `"1.92"`, and add `pdq` to `[dependencies]`. The final `[dependencies]` section should include this line alongside the existing crates:

```toml
pdq = { version = "0.3", default-features = false, features = ["render"] }
```

And the `rust-version` line in `[package]` should read:

```toml
rust-version = "1.92"
```

- [ ] **Step 2: Verify the dependency resolves**

Run:
```bash
cd src-tauri && cargo fetch
```
Expected: completes without error, downloading `pdq` 0.3.x and its `hayro` backend. If `features = ["render"]` is rejected, run `cargo tree -p pdq` and use the exact feature name shown (the crate may publish it as `render` or `hayro`); the spec's intent is "enable page rendering."

- [ ] **Step 3: Commit**

```bash
git add src-tauri/Cargo.toml src-tauri/Cargo.lock
git commit -m "Add pdq render dependency; bump MSRV to 1.92"
```

---

### Task 2: Write the page-name formatting helper with tests (TDD)

**Files:**
- Create: `src-tauri/src/pdf.rs`
- Modify: `src-tauri/src/lib.rs:7` (add `mod pdf;`)

This task establishes the `pdf` module with a pure, tested helper before any rendering code. Page names follow `"<pdf-stem> · p<n>"`.

- [ ] **Step 1: Declare the module**

In `src-tauri/src/lib.rs`, add at line 7 (next to `mod languages;`):

```rust
mod pdf;
```

- [ ] **Step 2: Write the failing test**

Create `src-tauri/src/pdf.rs` with only the test for now:

```rust
/// Formats the display name for a page rendered from a PDF.
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
```

- [ ] **Step 3: Run the test to verify it passes**

Run:
```bash
cd src-tauri && cargo test pdf::tests --lib
```
Expected: 3 tests pass. (They pass immediately because the helper is implemented alongside; this is the rare pure-function case where test + impl land together.)

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/pdf.rs src-tauri/src/lib.rs
git commit -m "Add pdf module with page-name helper and tests"
```

---

### Task 3: Implement `render_pages` with a real-PDF test (TDD)

**Files:**
- Modify: `src-tauri/src/pdf.rs`
- Create: `src-tauri/tests/fixtures/tiny.pdf` (committed binary fixture)

`render_pages(bytes, dpi)` writes the bytes to a temp file, calls `pdq::render::render_pages`, reads back the PNGs in page order, and cleans up the temp dir. The pdq API is path-based and writes PNG files to an output pattern containing `{:03}` (zero-padded page index, 1-based) — e.g. `"out_dir/page_{:03}.png"`.

- [ ] **Step 1: Add the test fixture**

Create a one-page PDF and verify it has exactly one page:
```bash
mkdir -p src-tauri/tests/fixtures
# Generate a minimal one-page PDF using Python (available on macOS).
python3 -c "
from fpdf import FPDF
" 2>/dev/null && pip3 install --quiet fpdf2 2>/dev/null
python3 -c "
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Helvetica', size=40)
pdf.cell(40, 50, 'Hello PDF')
pdf.output('src-tauri/tests/fixtures/tiny.pdf')
"
file src-tauri/tests/fixtures/tiny.pdf
```
Expected: `src-tauri/tests/fixtures/tiny.pdf: PDF document, version 1.x`. If `fpdf2` is unavailable, instead copy the smallest page of a sample PDF:
```bash
cp sample_pdf/scan_2.pdf src-tauri/tests/fixtures/tiny.pdf
```
(The test only asserts "at least one non-empty PNG is returned"; it does not assert page count, so a multi-page sample is acceptable as a fallback fixture.)

- [ ] **Step 2: Write the failing integration test**

Append to `src-tauri/src/pdf.rs` (keep the existing `page_name` + tests; add below them). Note the test is gated on the fixture existing so it doesn't fail in environments without it:

```rust
use std::fs;

/// Render every page of `pdf_bytes` to a PNG at `dpi`, returned in page order.
/// Uses a temp directory: pdq's API writes files, not in-memory buffers.
pub(crate) fn render_pages(pdf_bytes: &[u8], dpi: f32) -> Result<Vec<Vec<u8>>, String> {
    // Unique temp dir per call so concurrent renders can't collide.
    let tmp_dir = std::env::temp_dir().join(format!(
        "just-ocr-pdf-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    fs::create_dir_all(&tmp_dir).map_err(|e| format!("Failed to create temp dir: {e}"))?;

    let result = render_pages_into(pdf_bytes, dpi, &tmp_dir);

    // Best-effort cleanup regardless of success/failure.
    let _ = fs::remove_dir_all(&tmp_dir);
    result
}

fn render_pages_into(
    pdf_bytes: &[u8],
    dpi: f32,
    out_dir: &std::path::Path,
) -> Result<Vec<Vec<u8>>, String> {
    let pdf_path = out_dir.join("input.pdf");
    fs::write(&pdf_path, pdf_bytes).map_err(|e| format!("Failed to write temp PDF: {e}"))?;

    let opts = pdq::render::RenderOptions {
        dpi,
        pages: None, // render all pages
    };
    let pattern = out_dir.join("page_{:03}.png");
    let pattern_str = pattern.to_string_lossy().into_owned();
    pdq::render::render_pages(&pdf_path, &pattern_str, &opts)
        .map_err(|e| format!("PDF render failed: {e}"))?;

    // Collect rendered PNGs in ascending filename order (= page order).
    let mut files: Vec<_> = fs::read_dir(out_dir)
        .map_err(|e| format!("Failed to read temp dir: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("page_") && n.ends_with(".png"))
                .unwrap_or(false)
        })
        .collect();
    files.sort_by_key(|e| e.file_name());
    files
        .into_iter()
        .map(|e| fs::read(e.path()).map_err(|err| format!("Failed to read PNG: {err}")))
        .collect()
}

#[cfg(test)]
mod render_tests {
    use super::render_pages;
    use std::path::PathBuf;

    fn fixture() -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.pdf");
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    fn renders_at_least_one_nonempty_png() {
        let Some(path) = fixture() else {
            eprintln!("skip: tests/fixtures/tiny.pdf not present");
            return;
        };
        let bytes = std::fs::read(&path).expect("read fixture");
        let pages = render_pages(&bytes, 150.0).expect("render succeeds");
        assert!(!pages.is_empty(), "expected at least one rendered page");
        for png in &pages {
            assert!(!png.is_empty(), "rendered page PNG must be non-empty");
            // PNG magic bytes.
            assert_eq!(&png[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        }
    }

    #[test]
    fn rejects_garbage_bytes() {
        let err = render_pages(b"not a pdf", 150.0).unwrap_err();
        assert!(err.contains("render failed") || err.to_lowercase().contains("pdf"));
    }
}
```

- [ ] **Step 3: Run the tests to verify the render test passes**

Run:
```bash
cd src-tauri && cargo test pdf:: --lib
```
Expected: `renders_at_least_one_nonempty_png` passes, `rejects_garbage_bytes` passes, and the 3 `page_name` tests still pass.

If the render test errors with a signature/feature problem, inspect the actual API and adjust `render_pages_into` to match — the most likely variations are: (a) `render_pages` returns `pdq::PdfiumError` or `Box<dyn Error>` rather than a stringifiable error, so adapt the `.map_err` closure; (b) the output pattern uses `{}` instead of `{:03}` — try `page_{}.png`. Re-run until green.

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/pdf.rs src-tauri/tests/fixtures/tiny.pdf
git commit -m "Implement pdq-based render_pages with integration test"
```

---

### Task 4: Expose the `render_pdf` Tauri command

**Files:**
- Modify: `src-tauri/src/lib.rs` (add command + register handler)

The command wraps `render_pages` in `spawn_blocking` (rendering large scans is CPU-bound — mirrors `ocr_from_bytes` at `lib.rs:65-74`) and returns `Vec<ReadFile>` using the existing struct at `lib.rs:40-45`. It takes the PDF's file name so each rendered page can be named `<stem> · p<n>` (e.g. `scan_3 · p1`).

- [ ] **Step 1: Add the command**

In `src-tauri/src/lib.rs`, add this command (place it just before `run()`, after the `run_ocr` function ends at line 143):

```rust
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
```

- [ ] **Step 2: Register the handler**

In `src-tauri/src/lib.rs`, find the `invoke_handler!` macro call (lines 160-169) and add `render_pdf` to the list. The macro should now read:

```rust
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
```

- [ ] **Step 3: Verify the crate still builds**

Run:
```bash
cd src-tauri && cargo check
```
Expected: compiles without errors. (Warnings about unused fields in `RenderOptions` are fine.)

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/lib.rs
git commit -m "Expose render_pdf Tauri command at 300 DPI"
```

---

### Task 5: Add `isPdf` and `renderPdf` frontend helpers with tests (TDD)

**Files:**
- Modify: `src/lib/ocr.ts`

- [ ] **Step 1: Write the failing test**

Create `src/lib/ocr.test.ts`:

```ts
import { isPdf } from "./ocr";
import { describe, it, expect } from "vitest";

describe("isPdf", () => {
  it("matches .pdf extension case-insensitively", () => {
    expect(isPdf("scan.pdf")).toBe(true);
    expect(isPdf("SCAN.PDF")).toBe(true);
  });
  it("rejects non-PDF names", () => {
    expect(isPdf("photo.png")).toBe(false);
    expect(isPdf("scan.pdf.bak")).toBe(false);
    expect(isPdf("pdf")).toBe(false);
  });
});
```

Add `vitest` as a dev dependency:
```bash
npm install --save-dev vitest
```
Add a test script to `package.json`'s `"scripts"`:
```json
"test": "vitest run"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
npm test -- ocr.test
```
Expected: FAIL — `isPdf` is not exported from `./ocr` (or module resolution error).

- [ ] **Step 3: Add `isPdf` and `renderPdf` to `ocr.ts`**

In `src/lib/ocr.ts`, add near the other small helpers (e.g. just below `readFiles`):

```ts
/** True if the file name has a .pdf extension (case-insensitive). */
export function isPdf(name: string): boolean {
  return /\.pdf$/i.test(name);
}

/** Rasterize every page of a PDF to a PNG via the Rust `render_pdf` command.
 * Returns one ReadFile per page, named `<stem> · p<n>`. */
export async function renderPdf(name: string, bytes: Uint8Array): Promise<ReadFile[]> {
  return invoke<ReadFile[]>("render_pdf", {
    pdfName: name,
    bytes: Array.from(bytes),
  });
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
npm test -- ocr.test
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lib/ocr.ts src/lib/ocr.test.ts package.json package-lock.json
git commit -m "Add isPdf and renderPdf frontend helpers with tests"
```

---

### Task 6: Route PDFs through `renderPdf` in `App.svelte`

**Files:**
- Modify: `src/App.svelte` (`addPaths` at lines 119-127, imports at lines 8-17)

- [ ] **Step 1: Update imports**

In `src/App.svelte`, the import from `"./lib/ocr"` (lines 8-17) currently lists: `availableLanguages`, `makeJob`, `makeJobsFromReadFiles`, `ocrFromBytes`, `readFiles`, `exportResults`, `type OcrOpts`, `type Job`. Add `renderPdf` and `isPdf`:

```ts
import {
  availableLanguages,
  isPdf,
  makeJob,
  makeJobsFromReadFiles,
  ocrFromBytes,
  readFiles,
  renderPdf,
  exportResults,
  type OcrOpts,
  type Job,
} from "./lib/ocr";
```

- [ ] **Step 2: Rewrite `addPaths` to branch on file type**

Replace the existing `addPaths` function (lines 119-127) with:

```ts
  /** Ingest files dropped via the native drag-drop event (paths → bytes). */
  async function addPaths(paths: string[]) {
    if (!paths.length) return;
    const read = await readFiles(paths);
    if (!read.length) return;
    const added: Job[] = [];
    for (const f of read) {
      if (isPdf(f.name)) {
        const pages = await renderPdf(f.name, new Uint8Array(f.bytes));
        added.push(...makeJobsFromReadFiles(pages));
      } else {
        added.push(...makeJobsFromReadFiles([f]));
      }
    }
    jobs = [...jobs, ...added];
    if (selectedId === null && added.length) selectedId = added[0].id;
  }
```

- [ ] **Step 3: Rewrite `addFiles` to handle PDFs from the file picker too**

The file picker will be widened in Task 7 to accept PDFs, so `addFiles` (which receives browser `File` objects) must handle them the same way. Replace `addFiles` (lines 113-117) with:

```ts
  async function addFiles(files: FileList) {
    const added: Job[] = [];
    for (const file of Array.from(files)) {
      if (isPdf(file.name)) {
        const buf = new Uint8Array(await file.arrayBuffer());
        const pages = await renderPdf(file.name, buf);
        added.push(...makeJobsFromReadFiles(pages));
      } else {
        added.push(await makeJob(file));
      }
    }
    jobs = [...jobs, ...added];
    if (selectedId === null && added.length) selectedId = added[0].id;
  }
```

- [ ] **Step 4: Verify the frontend type-checks**

Run:
```bash
npm run build
```
Expected: Vite build completes without TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add src/App.svelte
git commit -m "Route PDF files through renderPdf in addPaths/addFiles"
```

---

### Task 7: Widen the file picker to accept PDFs

**Files:**
- Modify: `src/lib/Thumbnail.svelte:159`

- [ ] **Step 1: Update the `accept` attribute**

In `src/lib/Thumbnail.svelte`, the hidden input (around line 159) currently has `accept="image/*"`. Change it to:

```svelte
  <input
    bind:this={input}
    type="file"
    accept="image/*,.pdf,application/pdf"
    multiple
    onchange={(e) => e.currentTarget.files && onfiles(e.currentTarget.files)}
    hidden
  />
```

- [ ] **Step 2: Commit**

```bash
git add src/lib/Thumbnail.svelte
git commit -m "Accept PDFs in the file picker"
```

---

### Task 8: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Requirements and Usage**

In `README.md`:
- Change `- Rust 1.88+, Node 18+` to `- Rust 1.92+, Node 18+`.
- Change usage step 1 from:
  ```
  1. Drag an image (PNG/JPG/BMP/WebP) onto the dropzone, or click to browse.
  ```
  to:
  ```
  1. Drag an image (PNG/JPG/BMP/WebP) or a PDF onto the dropzone, or click to
     browse. Multi-page PDFs are rasterized at 300 DPI and added as one job per
     page.
  ```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Docs: note PDF support and MSRV bump in README"
```

---

### Task 9: Manual end-to-end validation

This task validates the whole feature against the real sample files. There is no code change and no commit unless a bug is found.

- [ ] **Step 1: Build and run the app**

```bash
cargo tauri dev
```
Expected: app window opens. The first build compiles pdq + hayro, so it will take longer than a normal incremental build.

- [ ] **Step 2: Drop `sample_pdf/scan_2.pdf` and verify expansion**

Drag `sample_pdf/scan_2.pdf` onto the window.
Expected: the thumbnail panel populates with one entry per page, named `scan_2 · p1`, `scan_2 · p2`, … Each thumbnail shows the rendered page image.

- [ ] **Step 3: Run OCR on one page**

Select any page, pick a language + PSM, click **Run OCR**.
Expected: recognized text, mean confidence, and elapsed time appear in the Output panel.

- [ ] **Step 4: Run All + Export**

Drop all three sample PDFs, click **Run All**, wait for completion, then **Export** to `.txt`.
Expected: export file contains a section per page with recognized text.

- [ ] **Step 5: Mixed drop sanity check**

Drag a PNG (or JPG) and a PDF together.
Expected: the image appears as a single job; the PDF expands to per-page jobs. Both run through **Run All**.

- [ ] **Step 6: Error case**

Drag a non-PDF file renamed to `.pdf` (e.g. `cp sample_pdf/scan_2.pdf /tmp/bad.pdf; head -c 100 /tmp/bad.pdf > /tmp/garbage.pdf` and drop `/tmp/garbage.pdf`).
Expected: `render_pdf` returns an error; no jobs are added. Check the devtools console (`View → Toggle Developer Tools`) for the error string. No crash.

- [ ] **Step 7: Final commit (only if fixes were needed)**

If any step surfaced a bug, fix it, add a regression test where feasible, and commit. Otherwise, no action — the feature is complete.
