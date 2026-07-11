## Plan: Add hOCR output mode (Text / hOCR toggle)

The tesseract-rs binding already exposes `api.get_hocr_text(page: i32) -> Result<String>` (`api.rs:309`) — no config flags or renderers needed. The existing `run_ocr` calls `get_utf8_text()`; we just branch on a new option. `OcrResult` already carries a plain `String`, so the result shape is unchanged — only how we interpret/display it changes.

### 1. Backend — `src-tauri/src/lib.rs`
- Add `pub output_mode: Option<String>` to `OcrOpts`. Optional so existing behavior (text) is the default when the field is absent. (serde `camelCase` makes the frontend's `outputMode` line up.)
- In `run_ocr`, after `api.set_image(...)`, branch:
  - `output_mode == Some("hocr")` → `api.get_hocr_text(0)?`
  - otherwise → `api.get_utf8_text()?` (unchanged)
- `mean_text_conf()` works the same after either getter — confidence semantics are unchanged.
- `OcrResult` unchanged.

### 2. Frontend types — `src/lib/ocr.ts`
- Add `outputMode: "text" | "hocr"` to `OcrOpts`.
- Add `outputMode: "text" | "hocr"` to `Job` so each job remembers the mode it was produced in (important: a user can run some images in text mode, switch to hOCR, run more — the display/export must follow the per-job mode, not the live setting).
- In `exportResults`: add a third filter `{ name: "hOCR", extensions: ["hocr"] }` and an `.html`/`.xml` alias. When the chosen extension is hOCR, write each job's raw text verbatim (no trim). For multi-job hOCR export, concatenate each document's `<body>` content or write each as a separate file — simplest correct approach: one `.hocr` file per job isn't possible with a single save dialog, so concatenate with a separator header per image (matching the TXT block style but writing raw XML). I'll write each job's hOCR wrapped so it remains valid standalone XML per block.
- `makeJob`/`makeJobsFromReadFiles`: stamp the current `opts.outputMode` onto each new job.

### 3. Output mode selector — `src/lib/Toolbar.svelte`
- Add an `outputModeOptions` array: `[{ value: "text", label: "Text" }, { value: "hocr", label: "hOCR" }]`.
- Add a `<label class="field">` with a `<select bind:value={opts.outputMode}>` next to the PSM field, reusing the existing `.field`/`.lbl`/`select` styles. Label: "Output".

### 4. Display — `src/lib/Output.svelte`
- Branch on `job.outputMode`:
  - `"text"` → unchanged `<pre class="text">` rendering.
  - `"hocr"` → render the XML in a `<pre class="text hocr">` (raw markup, monospace, scrollable, no trailing-trim). Change header label from "Text" to "hOCR".
- Copy button copies `job.text` verbatim either way (already does).
- Disable the `disabled={!job.text.trim()}` logic for hOCR (XML always has content if produced).

### 5. Wire mode through `src/App.svelte`
- Default `opts.outputMode = "text"` in the initial `opts` state.
- When running OCR, the existing `opts` is passed to `ocrFromBytes`; stamp the resulting `outputMode` onto the job's state in the run handlers (`runCurrent`/`runAll`).

### 6. Verify
- `cargo check` (Rust), `npm run build` (frontend type-check), then `cargo tauri dev` to confirm: Text mode unchanged, hOCR mode returns XML containing `ocr_page`/`ocr_line`/`ocr_word` elements, copy works, export-to-.hocr writes valid XML.

### Not doing
- Not rendering hOCR as formatted HTML — hOCR is structured layout data (bounding boxes per word), and showing the raw XML is the honest, useful representation for this tool. Rendered HTML would hide the very information hOCR provides.
- Not adding ALTO/TSV/box formats now — request was two modes (text or hOCR). The branch is trivially extensible later.