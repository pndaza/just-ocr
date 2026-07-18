# Kraken Engine — Design Spec

**Date:** 2026-07-18
**Status:** Approved (pending implementation)
**Related:** `2026-07-12-pdf-support-design.md`

## Goal

Add a Kraken OCR engine to just-ocr so Burmese and other scripts that Tesseract
handles poorly can be recognized. Kraken's neural segmentation runs the layout
stage for **both** engines; the user-facing "engine" selector chooses the
**recognizer** (Tesseract or Kraken).

## Decisions (locked)

| Decision | Choice |
|---|---|
| Output representation | **Structured `OcrResult`** (`{width, height, lines[]}`) — no hOCR XML |
| Segmentation | **Kraken always** (candle seg backend, `bur_segment.safetensors`) |
| Recognizer choice | User-facing "Engine" selector: **Tesseract** or **Kraken** |
| Kraken code incorporation | **Vendor** required modules into `src-tauri/src/kraken/` |
| Model location | `app_local_data_dir/kraken-models/`, dev fallback to `../kraken-models/` |
| Model caching | Load once into `tauri::State`, reuse across calls |
| PSM control | **Removed** — Kraken does layout, page-level PSM no longer applies |
| Output-mode toggle | **Removed** — result is always structured; plain text is a projection |
| Inference backend | **candle-core** only (no `ort`/ONNX) — matches `.safetensors` models |

## Pipeline

```
image bytes
   │
   ▼
Kraken segmentation (always)        bur_segment.safetensors
   │   produces: Vec<baseline + boundary polygon>
   ▼
for each line: crop boundary polygon from image
   │
   ├── engine == "tesseract" ──► Tesseract (PSM_RAW_LINE) per crop
   │                                → text + word confidences
   │
   └── engine == "kraken"     ──► Kraken recog (candle) per crop
                                    bur_recog.safetensors
                                    → text
   ▼
Vec<LineBox { x0, y0, x1, y1, text }>
   ▼
OcrResult { width, height, lines, confidence, elapsed_ms }
```

Tesseract is invoked once per line crop with `PSM_RAW_LINE` (13), which tells
it the input is a single text line with no further layout work. This keeps
Tesseract useful as a recognizer for scripts it handles well (eng, tur, mya)
while letting Kraken supply the line geometry.

## Backend structure

### New module: `src-tauri/src/engine.rs`

Owns the typed result contract and the dispatcher.

```rust
pub struct LineBox { pub x0: u32, pub y0: u32, pub x1: u32, pub y1: u32, pub text: String }

pub struct OcrResult {
    pub width: u32,
    pub height: u32,
    pub lines: Vec<LineBox>,
    pub confidence: i32,   // mean across lines; -1 if unknown
    pub elapsed_ms: u64,
}

pub fn run_ocr(app: &AppHandle, image_bytes: &[u8], opts: &OcrOpts) -> Result<OcrResult, String> {
    // 1. Decode image → DynamicImage (RGB8).
    // 2. kraken::segment(app, &img) → Vec<(boundary_polygon, baseline)>.
    // 3. For each line:
    //      crop = crop_boundary(&img, &boundary)
    //      match opts.engine {
    //          "tesseract" => tesseract_line::recognize(crop, &opts.language, &opts.whitelist),
    //          "kraken"    => kraken::recognize_line(app, &crop),
    //      }
    //    Build LineBox from axis-aligned bbox of boundary polygon.
    // 4. mean confidence (tesseract: per-line mean; kraken: -1).
}
```

`OcrOpts` changes: drop `psm`, add `engine: String` whose value is exactly
`"tesseract"` or `"kraken"` (default `"tesseract"` to preserve current
behavior on first launch). Keep `language` (tesseract) and `whitelist`
(tesseract-only). Any other `engine` value returns an error from
`run_ocr` rather than silently falling back.

### New module tree: `src-tauri/src/kraken/`

Vendored from `/Users/pndaza/Projects/playground/kraken-rust`. Only the
candle path is brought over — no `ort`, no ONNX, no CLI, no `inference.rs`
(the ONNX forward), no `model.rs` (the ONNX seg model), no `alto.rs`.

```
src-tauri/src/kraken/
├── mod.rs                  ← pub segment(), recognize_line(), resolve_models()
├── containers.rs           ← Segmentation, BaselineLine, Region (copied)
├── config.rs               ← SegmentationConfig (copied)
├── heatmap.rs              ← Heatmap (copied)
├── preprocess.rs           ← page preprocess (copied)
├── inference_candle.rs     ← candle forward (copied)
├── detect.rs               ← detect_candle() + postprocess() (copied, ONNX fns removed)
├── vectorize.rs            ← (copied)
├── boundaries.rs           ← (copied)
├── reading_order.rs        ← (copied)
├── contours.rs             ← (copied)
├── polygon/
│   ├── mod.rs              ← (copied)
│   └── boolean.rs          ← (copied)
├── ndimage/
│   ├── mod.rs
│   ├── filters.rs
│   ├── morphology.rs
│   └── mcp.rs
├── segmentation_candle/
│   └── model.rs            ← SegmentationModelCandle (copied)
└── recognition/
    ├── mod.rs
    ├── model.rs            ← RecognitionModel (copied)
    ├── preprocess.rs       ← preprocess_line (copied)
    ├── decode.rs           ← greedy_decode (copied)
    ├── codec.rs            ← Codec (copied)
    └── meta.rs             ← parse_recognition_meta (copied)
```

After copying, all `use crate::` paths in the copied files are rewritten to
`use crate::kraken::` so they resolve inside just-ocr. No `test_harness.rs`
(test-only). No `recognition/orchestrator.rs` — its `crop_boundary` helper
moves to `engine.rs` since we crop + dispatch per line there.

The architecture-specific constants in `recognition/model.rs` (4 conv blocks,
3 BiLSTM-200, 118 classes, hardcoded `C_0`/`L_12`/`O_18` names) are kept as-is
for now — they match `bur_recog.safetensors`. A `// TODO: generalize VGSL
parser` note is left for future models.

### `mod.rs` public surface

```rust
pub fn segment(app: &AppHandle, img: &DynamicImage)
    -> Result<Vec<BaselineLine>, String>;

pub fn recognize_line(app: &AppHandle, crop: &DynamicImage)
    -> Result<String, String>;

/// Resolve (seg_path, rec_path). App data dir first, dev path fallback.
pub fn resolve_models(app: &AppHandle) -> Result<(PathBuf, PathBuf), String>;
```

Models cached in `tauri::State<KrakenCache>`:

```rust
struct KrakenCache {
    seg: OnceLock<SegmentationModelCandle>,
    rec: OnceLock<RecognitionModel>,
}
```

`SegmentationModelCandle::load` and `RecognitionModel::load` are `Send + Sync`
(they wrap candle `Tensor`s, which are `Arc<RwLock<Storage>>`), so sharing by
shared reference from `State` is safe without a mutex.

### New module: `src-tauri/src/tesseract_line.rs`

```rust
pub fn recognize(crop: &DynamicImage, language: &str, whitelist: &Option<String>)
    -> Result<(String, i32), String>;   // (text, mean_conf)
```

Mirrors `lib.rs:127-161` today but operates on a single crop with
`PSM_RAW_LINE`, and returns plain text + confidence rather than hOCR.

### Changes to `lib.rs`

- `OcrOpts`: drop `psm`, add `engine: String`.
- `OcrResult`: replaced by `engine::OcrResult` (structured).
- `run_ocr` body removed; `ocr_from_bytes` calls `engine::run_ocr`.
- `setup()` initializer: print kraken model paths when resolved, alongside
  the existing tesseract version line.
- Remove the `tests::psm_mapping_covers_all_options` test (PSM is gone from
  the public surface); keep `temp_dir_pid_parsing`.

### Cargo.toml additions

```toml
candle-core = "0.11"
candle-nn   = "0.11"
ndarray     = "0.16"
rayon       = "1.10"
geo         = "0.29"
imageproc   = { version = "0.25", default-features = false }
anyhow      = "1"      # already transitively present via tesseract-rs; pin
```

Deliberately **not** added: `ort`, `clap`, `env_logger`. The vendored code's
ONNX modules (`inference.rs`, `model.rs`) and CLI (`main.rs`) are not copied,
so those deps are not needed.

## Frontend changes

### `src/lib/result.ts` (renamed from `hocr.ts`)

```ts
export interface LineBox { x0: number; y0: number; x1: number; y1: number; text: string }
export interface OcrResult { width: number; height: number; lines: LineBox[]; confidence: number; elapsedMs: number }

// Identity today — kept as a function so the call sites stay stable and any
// future post-processing has one home.
export function lineBoxes(result: OcrResult): LineBox[] { return result.lines; }
export function plainText(result: OcrResult): string {
  return result.lines.map(l => l.text).join("\n");
}
```

Drops `DOMParser`, the hOCR walk, and `.ocr_line` / `.ocrx_word` selectors.

### `src/lib/ocr.ts`

- `OcrResult` type matches the backend struct.
- `Job.text: string` → `Job.result: OcrResult`.
- `ocrFromBytes(bytes, opts)` returns `OcrResult`.
- `OcrOpts`: drop `psm`, add `engine: "tesseract" | "kraken"`.
- Persist `opts.engine` in `localStorage` alongside `opts.language`.

### `Preview.svelte`

Replace `parseHocrLines(job.text)` with `lineBoxes(job.result)`. The SVG
overlay loop is unchanged in shape — it iterates boxes and draws `<rect>`.

### `Output.svelte`

Render `plainText(job.result)` instead of conditionally parsing hOCR. The
Text/hOCR toggle and its state are removed.

### `Toolbar.svelte`

- Add **Engine** `<select>` (Tesseract / Kraken) before the Language select.
- **Remove** the PSM `<select>` and its label/state.
- Keep **Whitelist**; when engine is Kraken, disable it with a tooltip
  ("Whitelist applies to Tesseract only").
- **Remove** the Text/hOCR output-mode `<select>`.

### `App.svelte`

- `opts` default: `{ engine: "tesseract", language: ..., whitelist: null }`.
- Stop button, batch loop, `processJob` unchanged in shape — just consume
  `job.result` instead of `job.text`.

## Model resolution

`kraken::resolve_models(app)`:

1. Check `app_local_data_dir()/kraken-models/{bur_segment,bur_recog}.safetensors`.
2. If missing, check `../kraken-models/` relative to the current working dir
   (dev workflow — the repo has `kraken-models/` at its root).
3. If neither, return `Err` with a message naming both expected paths.

The existing `kraken-models/` directory stays where it is for dev. A future
LanguageManager-style UI could install kraken models into the app data dir;
out of scope for this iteration.

## Testing

**Unit (Rust):**
- `engine::run_ocr` dispatch on a synthetic image — both branches return a
  `LineBox` per detected line. Use a fixture where Kraken seg is known to
  produce N lines.
- Bbox math: boundary polygon → axis-aligned bbox clamped to image bounds.
- `result.ts` `plainText` joins with `\n`; empty result → empty string.

**Manual integration:**
- `cargo tauri dev`, load `sample_pdf/scan_4.pdf`, render to images, run both
  engines on the same page. Confirm:
  - Overlay boxes align with text rows for both engines.
  - Kraken recog output is coherent Burmese; Tesseract output matches today's
    behavior on the same line.
- Verify cached model load: second image is markedly faster than the first.

**Skipped:**
- kraken-rust's Python-fixture integration tests are already `#[ignore]`
  without fixtures and are not copied.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Candle build issues on the Tauri toolchain | candle is pure Rust CPU; the existing `[profile.dev.package."*"] opt-level = 3` keeps recog fast. First build will be slow. |
| Recognition model architecture is hardcoded to bur_recog (118 classes) | Documented as a TODO; out of scope for this iteration. Different models need the generalized VGSL parser from `segmentation_candle/model.rs`. |
| Tesseract per-line may be slower than full-page Tesseract | Acceptable: Kraken seg gives better line geometry; PSM_RAW_LINE skips layout. Benchmark during integration. |
| Tesseract word-level boxes are lost | Line bbox + joined text is visually equivalent for the overlay. If word boxes are wanted later, structured `LineBox` extends to `words: [...]` cleanly. |
| `kraken-models/` is untracked in git | Add to `.gitignore` formally (already effectively ignored); document that models must be supplied. |

## Out of scope

- ONNX/ort backend for segmentation.
- Installing kraken models from the LanguageManager UI.
- Generalizing the recognition VGSL parser for non-bur_recog models.
- CoreML acceleration (candle CPU is sufficient for the bur_recog model size).
- Word-level bounding boxes.
