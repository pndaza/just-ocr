# PDF Support — Design Spec

**Date:** 2026-07-12
**Status:** Draft (awaiting user review)

## Goal

Add PDF support: dropping a multi-page PDF expands to one job per page (one
thumbnail each), each OCR'd independently at a fixed 300 DPI. Scanned-document
assumption — always OCR, never extract an embedded text layer.

## Non-goals (out of scope)

- Configurable DPI (fixed at 300).
- Embedded-text extraction / hybrid mode.
- Page-range selection (all pages are rendered).
- Encrypted-PDF password prompt.
- Multi-platform prebuilt binaries beyond the dev host (macOS arm64).

## Decision: rasterizer

**pdq** (`pdq` crate on crates.io, v0.3.0) — pure-Rust PDF split/merge/render
library (renders via `hayro`, no C dependencies). Chosen over PDFium because it
preserves the project's "fully standalone, no system install, no prebuilt blob"
property and removes all `build.rs` binary-fetch wiring.

Tradeoffs accepted:
- **MSRV bump:** pdq requires Rust **1.92**. The project currently declares
  `rust-version = "1.88"` and the README says "Rust 1.88+". Both must be bumped
  to 1.92. Rust 1.92 is stable and current.
- **API is path-based, not bytes-in-memory:** `pdq::render::render_pages` takes
  a `&Path` input and writes PNG files to an output pattern (`%d`). Our flow is
  bytes-in-memory (`read_files` returns bytes), so we write the bytes to a temp
  file, render into a temp directory, then read the PNGs back. Contained in the
  Rust command; the frontend never sees this.

## Architecture

The app already operates on a clean per-image job model. PDF only introduces a
**rasterization step** that converts PDF bytes → N page images. Placing it in
Rust matches the existing pattern (Tesseract and Leptonica already do all heavy
work on the Rust side; the frontend stays thin).

```
PDF dropped/read ──▶ [new Rust command: render_pdf]
                         │  pdq::render_pages, 300 DPI
                         │  temp file in → PNGs out → bytes back
                         ▼
                   Vec<ReadFile{ name:"scan.pdf · p1", bytes:PNG }>
                         │
                         ▼
                   existing makeJobsFromReadFiles() ──▶ job queue
```

Image files keep the existing path unchanged. The frontend branches once, on
file extension.

## Components

### 1. Rust — PDF rendering (`src-tauri/src/pdf.rs`)

New module. Exposes one Tauri command and one internal helper:

- `pub fn render_pages(bytes: &[u8], dpi: f32) -> Result<Vec<Vec<u8>>, String>`
  — core rasterization.
  1. Create a fresh temp directory (`std::env::temp_dir()` + unique subdir).
  2. Write `bytes` to `<tmp>/input.pdf`.
  3. Call `pdq::render::render_pages(input_path, "<tmp>/page_%d.png",
     &RenderOptions { dpi, pages: None })`.
  4. List the resulting `page_*.png` files in numeric order, read each into
     bytes.
  5. Remove the temp dir (best-effort, even on error).
- `#[tauri::command] async fn render_pdf(bytes: Vec<u8>) ->
  Result<Vec<ReadFile>, String>` — wraps `render_pages` in `spawn_blocking`
  (rendering large scans is CPU-bound; mirrors `ocr_from_bytes`). Names each
  page `"<stem> · p<n>"`. Reuses the existing `ReadFile { name, bytes }` struct.

Register `render_pdf` in the `invoke_handler!` list in `lib.rs`.

### 2. Rust — build wiring (`Cargo.toml`)

- Add `pdq = { version = "0.3", default-features = false, features = ["render"] }`.
  The `render` feature pulls in `hayro`; disabling defaults drops split/merge
  code we don't use.
- Bump `rust-version` from `"1.88"` to `"1.92"`.

No `build.rs` changes. No system dependencies. No external binary fetch.

### 3. Frontend — branch on file type

**`src/lib/ocr.ts`:**

- New wrapper:
  ```ts
  export async function renderPdf(bytes: Uint8Array): Promise<ReadFile[]> {
    return invoke<ReadFile[]>("render_pdf", { bytes: Array.from(bytes) });
  }
  ```
- New pure helper for page naming / routing decisions:
  ```ts
  export function isPdf(name: string): boolean {
    return /\.pdf$/i.test(name);
  }
  ```
- Modify `makeJobsFromReadFiles` is unchanged — it already builds jobs from
  `{ name, bytes }` records and creates object URLs. The page PNGs from
  `render_pdf` slot straight in.

**`src/App.svelte`:**

- `addPaths` and `addFiles`: route each file by type. For a PDF, call
  `renderPdf` first, then feed the returned `ReadFile[]` into
  `makeJobsFromReadFiles`. Non-PDFs flow through unchanged. Mixed PDF+image
  drops work (each file routed independently).
  - Sketch (addPaths):
    ```ts
    const read = await readFiles(paths);
    const added: Job[] = [];
    for (const f of read) {
      if (isPdf(f.name)) {
        const pages = await renderPdf(new Uint8Array(f.bytes));
        added.push(...makeJobsFromReadFiles(pages));
      } else {
        added.push(...makeJobsFromReadFiles([f]));
      }
    }
    jobs = [...jobs, ...added];
    if (selectedId === null && added.length) selectedId = added[0].id;
    ```

**`src/lib/Thumbnail.svelte`:**

- Widen the hidden `<input>` `accept` from `"image/*"` to `"image/*,.pdf,application/pdf"`.

**No changes** to `Preview.svelte` or `Output.svelte` — a page job is
byte-identical to an image job once rasterized.

## Data flow example

Drop `scan_3.pdf` (a 50-page scan):

1. `read_files` returns `{ name: "scan_3.pdf", bytes }`.
2. Frontend sees `.pdf` → calls `renderPdf(bytes)` → 50 `ReadFile`s named
   `scan_3.pdf · p1` … `scan_3.pdf · p50`, each a PNG.
3. `makeJobsFromReadFiles` creates 50 jobs (object URLs, thumbnails).
4. **Run All** OCRs each page independently; confidence/status are per-page.
5. Export concatenates as today.

## Error handling

- **Corrupt/encrypted/undecodable PDF:** `render_pages` returns `Err(String)`.
  The frontend surfaces it as a non-blocking message (e.g. a brief toast or
  console error). No job is created for a failed PDF — fail the whole PDF
  atomically rather than inserting partial pages.
- **Zero-page PDF:** `render_pages` returns `Ok(vec![])`. Frontend treats empty
  result as a no-op (optionally a toast: "No pages in <name>").
- **Temp-dir write failure:** surfaced as the same `Err(String)` path; treated
  as a render failure.
- **pdq version/feature mismatch at build time:** standard cargo build error;
  resolved by pinning `features = ["render"]`.

## Testing

- **Rust unit test (`pdf.rs`):** `render_pages` on a tiny 1-page test PDF
  committed under `src-tauri/tests/fixtures/`. Assert exactly one non-empty PNG
  is returned. Marked `#[ignore]` if the fixture isn't present in CI, since the
  value is local correctness, not CI gating.
- **Rust unit test:** page-name formatting (`<stem> · p<n>`) — pure function,
  cheap to cover.
- **TS unit test:** `isPdf()` extension matching — pure function.
- **Manual (per README convention):** `cargo tauri dev`, drop the three
  `sample_pdf/*.pdf`, confirm page counts, thumbnails render, **Run All**
  completes, export contains expected text. This is how the existing OCR
  pipeline is validated.

## Documentation updates

- `README.md`:
  - Usage step 1: "Drag an image **or PDF** …".
  - Requirements: "Rust **1.92+**" (was 1.88+).
  - Add a short note under Usage explaining multi-page PDFs expand to one job
    per page at 300 DPI.

## Open questions for review

None anticipated beyond MSRV acceptance. If the MSRV bump is a blocker, the
fallback is pdfium-render (revisit the original Approach A), at the cost of the
prebuilt-binary fetch.

## Sources

- [meistrari/pdq (GitHub)](https://github.com/meistrari/pdq)
- [pdq::render::render_pages (docs.rs)](https://docs.rs/pdq/latest/pdq/render/fn.render_pages.html)
- [pdq::render::RenderOptions (docs.rs)](https://docs.rs/pdq/latest/pdq/render/struct.RenderOptions.html)
