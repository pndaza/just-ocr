# PP-OCR Segmentation Integration — Design

**Date:** 2026-07-27
**Status:** Draft (pending user review)
**Branch:** TBD (new branch off `main`)

## Goal

Add PaddleOCR's PP-OCRv6 **tiny** text detector as an alternative segmentation
stage for the Myanmar OCR path, selectable by the user via a new "Seg"
dropdown. The recognizer (Kraken or Tesseract) stays independently selectable,
so all four combinations work: `{Kraken, PP-OCR} × {Kraken recog, Tesseract recog}`.

## Non-goals

- No change to the Latin / non-Myanmar path — it stays full-page Tesseract.
- No PP-OCR **recognizer** — recognition continues to use Kraken or Tesseract.
  (PP-OCR's detector is general-purpose, but PP-OCR's recognizer is not vendored.)
- No GPU backend. CPU only (matches kraken-engine's deployment).
- No runtime model download — the tiny-det weights are bundled via
  `include_bytes!`, consistent with kraken's zero-setup promise.

## Background

Today, segmentation is Kraken-only and only runs when `opts.language == "mya"`:

- `kraken-engine::Engine::segment(img)` returns `Vec<BaselineLine>`.
- Each line carries a `baseline` polyline (used by Kraken recog for dewarp) and
  a closed `boundary` polygon (used for bbox, Tesseract crop, overlay, and
  Kraken dewarp fallback).
- The recognizer call sites in `engine.rs::run_myanmar` consume only
  `line.baseline` and `line.boundary` (verified: engine.rs:208–234).
- There is no trait abstraction over segmenters — `Engine` owns
  `Arc<SegmentationModelCandle>` concretely.

The PP-OCRv6 detector (vendored from `clones/ppocr-rs`) is a pure-Rust,
dependency-light (rayon + safetensors + image + anyhow) PP-LCNetV4 backbone +
RepLkFpn neck + DB (differentiable binarization) head. Its tiny tier uses
channels `[32, 48, 64, 160]` with a k=5 RepLkFpn. Inputs `[1,3,H,W]` (H, W
multiples of 32, default resize to longest side 736) → outputs `[1,1,H,W]`
sigmoid probability map. The free function `extract_detections` then produces
`Vec<Detection>` where `Detection { polygon: [Point; 4], score }` — a
4-corner rotated quad in source-image coordinates. The tiny-det safetensors is
**1.70 MB** — trivially bundleable.

## Decisions (settled in brainstorming)

| Question | Decision |
|---|---|
| Vendor strategy | **Vendor a slimmed crate** into `src-tauri/ppocr-engine/` — prune to detector + DB postprocess only. Mirror how `kraken-engine` is integrated. |
| Dispatch scope | **Myanmar-only** — PP-OCR is an alternative segmenter for `language == "mya"`. Non-Myanmar stays full-page Tesseract. |
| Recognizer pairing | **Either recognizer.** PP-OCR seg + Tesseract recog is the clean pairing; PP-OCR seg + Kraken recog is supported by **synthesizing a baseline** (quad midline) and relying on the existing dewarp fallback if it fails. |
| Model bundling | **`include_bytes!`** the tiny-det safetensors (1.70 MB), with a single-file user-override dir at `app_local_data_dir()/ppocr-models/tiny-det.safetensors`. |
| UX | **Separate "Seg" dropdown** in `Toolbar.svelte`, visible only when `language === "mya"`. Persisted under `just-ocr:segmenter`. |
| Integration shape | **Sibling `ppocr-engine` crate + `Segmenter` trait in the host.** Both engines wrapped behind `Arc<dyn Segmenter>`. |
| Lazy load shape | **`OnceCell<Arc<Engine>>`** for clean `'static` trait objects. When `seg=ppocr + recog=kraken`, both engines load (~21 MB resident). |
| Baseline for Kraken recog | **Synthesize a midline** from the quad's long edges; on dewarp failure, the existing `crop_polygon_white_bg` fallback runs. |

## Architecture

### Crate layout

```
src-tauri/
├── Cargo.toml                     # host crate; path-deps kraken-engine + ppocr-engine
├── src/
│   ├── lib.rs                     # OcrOpts gains `segmenter: Option<String>`; invoke_handler unchanged
│   ├── engine.rs                  # run_myanmar gains segmenter branch + refactor to Arc<dyn Segmenter>
│   ├── segmentation.rs            # NEW: Segmenter trait + DetectedLine
│   └── segmenter_adapters.rs      # NEW: KrakenSegmenter + PPOcrSegmenter wrappers
├── kraken-engine/                 # UNCHANGED (existing Kraken port)
├── ppocr-engine/                  # NEW vendored, slimmed crate (NOT a workspace member)
│   ├── Cargo.toml                 # rayon + safetensors + image + anyhow + serde
│   └── src/
│       ├── lib.rs                 # Detector wrapper + postprocess entry + helpers (synth_midline, close_polygon)
│       ├── model.rs               # ← ppocr-rs src/cpu/model.rs lines 1–1625 (Detector only)
│       ├── backend.rs             # ← ppocr-rs src/cpu/backend.rs (Conv2d/ConvTranspose2d/Linear/LayerNorm)
│       ├── ops.rs                 # ← ppocr-rs src/cpu/ops.rs
│       ├── kernels/{mod,neon,x86}.rs   # ← ppocr-rs src/cpu/kernels/
│       ├── tensor.rs, arena.rs   # ← ppocr-rs src/cpu/{tensor,arena}.rs
│       ├── weights.rs             # ← ppocr-rs src/cpu/weights.rs (SafeTensors + VarBuilder)
│       ├── postprocess.rs         # ← ppocr-rs src/ocr.rs (extract_detections + DB helpers + Detection/Point/Transform/Options)
│       └── preprocess.rs          # ← ppocr-rs src/preprocess.rs (+ kernels.rs)
└── ppocr-models/                  # NEW: tiny-det assets at repo root (like kraken-models/)
    └── tiny-det.safetensors       # 1.70 MB, LFS-tracked via .gitattributes
```

**Boundary rule:** `ppocr-engine` exposes one public type — `Detector` — with
`load_from_buffer(bytes) -> Result<Self>` and `detect(img: &DynamicImage) ->
Result<Vec<Detection>>`. Internally: preprocess → forward →
`extract_detections`, returning detections in source-image coordinates. The
crate has zero dependency on the host or on `kraken-engine`. The
`Segmenter` trait lives in the **host** crate so neither vendored crate knows
about the other or about `DetectedLine`.

### The `Segmenter` trait and `DetectedLine`

`DetectedLine` is a new host-side type carrying only the two fields the
recognizer path consumes. It is deliberately distinct from
`kraken_engine::BaselineLine` so the host doesn't depend on Kraken's container
type for the abstraction (and so we can eventually drop Kraken without
rewriting the trait).

```rust
// src-tauri/src/segmentation.rs
use image::DynamicImage;
use serde::Serialize;

/// A detected text line in source-image pixel coordinates.
/// Produced by every Segmenter; consumed by run_myanmar.
#[derive(Debug, Clone, Serialize)]
pub struct DetectedLine {
    /// Midline polyline, left → right. Used by Kraken recog for dewarp.
    /// For PP-OCR, synthesized as the quad's vertical midline.
    /// May be empty if only the boundary matters (e.g. tesseract recog).
    pub baseline: Vec<(f64, f64)>,
    /// Closed boundary polygon (≥ 3 points). Used for bbox, Tesseract crop,
    /// overlay, and Kraken dewarp fallback. For PP-OCR: 4 corners + repeat-first.
    pub boundary: Vec<(f64, f64)>,
}

pub trait Segmenter: Send + Sync {
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String>;
    /// Human-readable name for logs (e.g. "kraken", "ppocr-tiny").
    fn name(&self) -> &'static str;
}
```

### Data flow (Myanmar, PP-OCR seg, Kraken recog)

```
img (DynamicImage)
  │
  ▼ run_myanmar
  segmenter = resolve_segmenter(app, opts)        // Arc<dyn Segmenter>
  lines    = segmenter.segment(img)?              // Vec<DetectedLine>
  │
  │  for each line:
  │    (x0,y0,x1,y1) = polygon_bbox((w,h), &line.boundary)
  │    match opts.engine.as_str():
  │      "tesseract" → crop_polygon_white_bg(img, &line.boundary) → tesseract_line::recognize
  │      "kraken"    → recognize_line_dewarped(img, &line.baseline, &line.boundary, binarize)
  │                   ↑ falls back to crop_polygon_white_bg on dewarp failure (existing)
  ▼
Vec<LineBox> → OcrResult
```

### `resolve_segmenter(app, opts)` resolution table

| `opts.language` | `opts.segmenter` | segmenter used |
|---|---|---|
| `"mya"` | `"kraken"` (default) | `KrakenSegmenter` |
| `"mya"` | `"ppocr"` | `PPOcrSegmenter` |
| `"mya"` | `None` / unrecognized | `KrakenSegmenter` (safe default, log warning if a string was given) |
| anything else | * | n/a — full-page Tesseract path, no segmenter loaded |

### Quad → `DetectedLine` conversion

- `boundary` = the 4 quad corners as `(f64, f64)`, **closed** (repeat first
  point at end, matching Kraken's convention so `polygon_bbox` and
  point-in-polygon behave identically).
- `baseline` = synthesized midline: order corners into a proper quad,
  identify the two long (text-axis) edges, sample N=8 points along each long
  edge, average top and bottom at each sample → N midline points.

### Host dispatch + lazy loading

Two `OnceCell<Arc<...>>`, one per engine. Wrapping the constructed engine in
`Arc` keeps the trait object `'static` (no lifetime threading through
`run_myanmar`). `Engine` and `Detector` are already `Send + Sync`.

```rust
// src-tauri/src/engine.rs
static KRAKEN: OnceCell<Arc<kraken_engine::Engine>> = OnceCell::new();
static PPOCR:  OnceCell<Arc<ppocr_engine::Detector>> = OnceCell::new();

fn resolve_segmenter(app: &AppHandle, opts: &OcrOpts)
    -> Result<Arc<dyn Segmenter>, String>
{
    match opts.segmenter.as_deref() {
        Some("ppocr") => {
            let det = PPOCR.get_or_try_init(|| load_ppocr(app).map(Arc::new))?;
            Ok(Arc::new(PPOcrSegmenter::new(det.clone())) as Arc<dyn Segmenter>)
        }
        other => {
            if let Some(s) = other { log::warn!("[ocr] unknown segmenter {s:?}, falling back to kraken"); }
            let eng = KRAKEN.get_or_try_init(|| load_kraken(app).map(Arc::new))?;
            Ok(Arc::new(KrakenSegmenter::new(eng.clone())) as Arc<dyn Segmenter>)
        }
    }
}
```

When `seg=ppocr + recog=kraken`, both engines load on first call. The Kraken
engine for recog is fetched via `kraken_engine(app)?`, which shares the same
`KRAKEN: OnceCell<Arc<kraken_engine::Engine>>` as `KrakenSegmenter`. Its
signature stays `fn kraken_engine(app) -> Result<&kraken_engine::Engine, String>`
— it returns `arc.as_ref()` so the borrow ties to the OnceCell's lifetime and
existing recog call sites are unchanged. (The OnceCell now stores `Arc<Engine>`
instead of `Engine`; the only edit to `load_kraken` is wrapping the result in
`Arc::new`.)

### Model bundling

- `ppocr-models/tiny-det.safetensors` (1.70 MB) at repo root, LFS-tracked.
- Host `engine.rs`: `static BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/tiny-det.safetensors");`
- Override: `resolve_override_ppocr(app)` returns `Some(path)` iff
  `app_local_data_dir()/ppocr-models/tiny-det.safetensors` exists. If the file
  exists but fails to deserialize, the error surfaces (we don't silently fall
  back to bundled — a user who placed a broken file should hear about it).

### Frontend / IPC

`OcrOpts` gains one field. Rust `Option<String>` (defaults `None` → kraken) and
TS `Segmenter` (required, defaults `"kraken"`) stay in sync via the existing
`#[serde(rename_all = "camelCase")]` convention.

```rust
// src-tauri/src/lib.rs
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrOpts {
    pub engine: String,
    pub language: String,
    pub psm: i32,
    pub whitelist: Option<String>,
    pub binarize: Option<String>,
    #[serde(default)]
    pub segmenter: Option<String>,   // NEW
}
```

```ts
// src/lib/ocr.ts
export type Segmenter = "kraken" | "ppocr";

export interface OcrOpts {
  engine: Engine;
  language: string;
  psm: number;
  whitelist: string | null;
  binarize: Binarize;
  segmenter: Segmenter;             // NEW
}
```

UI — `Toolbar.svelte`, inside the existing `{#if isMyanmar}` branch (the
non-Myanmar `{:else}` PSM branch is untouched):

```svelte
{#if isMyanmar}
  <label class="field">
    <span class="lbl">Seg</span>
    <select bind:value={opts.segmenter}>
      <option value="kraken">Kraken</option>
      <option value="ppocr">PP-OCR</option>
    </select>
  </label>
  <label class="field"> ...existing Engine select... </label>
{:else}
  ...PSM...   <!-- unchanged -->
{/if}
```

Persistence mirrors `saveEngine`/`lastEngine`: new localStorage key
`just-ocr:segmenter`, `lastSegmenter()` defaults to `"kraken"`,
`saveSegmenter()` round-trips it. `App.svelte` initializes
`segmenter: lastSegmenter()` and adds a `$effect` to persist. IPC flow is
unchanged — `ocrFromBytes(bytes, opts)` already forwards the whole opts
struct; the new field rides along.

## Vendoring — what gets copied, what gets cut

**Copied (verbatim, then trimmed):**

| Source (clones/ppocr-rs) | Destination | Notes |
|---|---|---|
| `src/cpu/model.rs` lines 1–1625 | `src/model.rs` | Detector only (drop `Recognizer` + rec helpers from 1675+). Drop `gpu` cfg paths. |
| `src/cpu/backend.rs` | `src/backend.rs` | Conv2d / ConvTranspose2d / Linear / LayerNorm primitives. |
| `src/cpu/ops.rs` | `src/ops.rs` | Tensor ops used by the model. |
| `src/cpu/kernels/{mod,neon,x86}.rs` | `src/kernels/` | Both arm64 (neon) and x86_64 paths — release CI builds both macOS arches. |
| `src/cpu/tensor.rs`, `arena.rs` | `src/tensor.rs`, `src/arena.rs` | Public `Tensor` type + buffer pool. |
| `src/cpu/weights.rs` | `src/weights.rs` | SafeTensors loader + VarBuilder. |
| `src/preprocess.rs` + `src/preprocess/kernels.rs` | `src/preprocess.rs` | Resize + BGR + normalize + 32-align. |
| `src/pixels.rs` | `src/pixels.rs` | Minimal `RgbImage` (adapt to host's `DynamicImage` at the boundary). |
| `src/ocr.rs` (subset) | `src/postprocess.rs` | Only `extract_detections`, `DetectorTransform`, `Detection`/`Point`, `DetectorPostprocessOptions`, and the DB helpers (`collect_component`, `fit_rotated_box`, unclip). Drop CTC decode, `OcrEngine`, `OcrRuntime`, `load_dictionary`. |

**Cut entirely:**

- `src/gpu/` (WGPU backend).
- `src/models.rs` + `models.json` (ModelStore / download / SHA-256).
- `src/bin/` (CLI, benchmarks).
- `src/ocr.rs` recognition parts.
- Deps not needed: `ureq`, `fs2`, `sha2`, `clap`, `wgpu`.
- `cpu::Recognizer` (model.rs:1953+) and `gpu::Recognizer`.

**`ppocr-engine/Cargo.toml`:**

```toml
[package]
name = "ppocr-engine"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
rayon = "1.11"
safetensors = "=0.7.0"
image = "0.25"
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
```

`ppocr-engine` is **NOT** added to the host's `[workspace]` (AGENTS.md gotcha
#1). It's a plain path-dep, so `[profile.dev.package."*"] opt-level = 3`
applies to it and its deps, keeping dev-build detection fast — same reason
`kraken-engine` stays out of the workspace.

```toml
# src-tauri/Cargo.toml
[dependencies]
kraken-engine = { path = "kraken-engine" }
ppocr-engine  = { path = "ppocr-engine" }
```

## Error handling & edge cases

Match existing patterns: heavy work in `spawn_blocking`; errors bubble as
`String` via `.map_err(|e| e.to_string())?` (host's `run_ocr` already returns
`Result<OcrResult, String>`). No new error type.

| Case | Handling |
|---|---|
| PP-OCR returns zero detections | Empty `lines` in result. Log `[ocr] ppocr: 0 detections`. Same as Kraken's empty-segmentation path. |
| Detection quad has `< 4` corners | Skip (`filter_map`); shouldn't happen from DB but defensive. |
| Detection quad degenerate (zero-area / collinear) | Skip. Detected via `polygon_bbox` returning `lw == 0 \|\| lh == 0`; extend the existing `boundary.len() < 3` filter at engine.rs:208. |
| Synthesized baseline fails Kraken dewarp | Existing fallback `crop_polygon_white_bg` runs (lib.rs:143–144). No new code. |
| Huge source image | ppocr-rs preprocess resizes to longest side 736 by default — no unbounded allocation. |
| `opts.segmenter` is an unrecognized string | Falls through to Kraken default; log a warning. |
| PP-OCR override file missing | `resolve_override_ppocr` returns `None` → bundled bytes used. |
| PP-OCR override file present but corrupt | Error surfaces (does NOT silently fall back). |
| Switch language away from `mya` mid-flight | Segmenter never loads. No issue. |

## Logging additions

Per AGENTS.md's per-stage timing convention (`log::info!("[ocr] ...: {:.0} ms")`):

```
[ocr] segmentation (kraken): 45 ms
[ocr] segmentation (ppocr-tiny): 28 ms    ← uses segmenter.name()
[ocr] ppocr load: 12 ms                   ← first-call lazy load
[ocr] ppocr detections: 14                ← count for debugging
[ocr] unknown segmenter "xyz", falling back to kraken   ← warn level
```

## Testing

**1. Unit tests in `ppocr-engine`:**
- `postprocess.rs`: synthetic `[1,1,H,W]` prob map with a known white blob →
  assert expected quad shape and score from `extract_detections`.
- `weights.rs`: load bundled `tiny-det.safetensors` via `include_bytes!` →
  assert all tensors F32, prefix stripping works.
- `model.rs::Detector::forward`: smoke test on a tiny fixture → assert output
  shape `[1,1,H,W]`, value range `[0,1]` (post-sigmoid).
- `synth_midline` + `close_polygon` helpers: pure-function unit tests with
  hand-crafted quads.

**2. Host integration test** — mirror the existing
`bundled_models_load_from_buffers` test (engine.rs:434):

```rust
#[test]
fn bundled_ppocr_det_loads_from_buffer() {
    let det = ppocr_engine::Detector::load_from_buffer(BUNDLED_PPOCR_DET);
    assert!(det.is_ok(), "bundled tiny-det must deserialize");
}
```

**3. Smoke example** — add `src-tauri/examples/smoke_ppocr.rs` mirroring
`smoke_kraken.rs`: load bundled PP-OCR det, run on a Myanmar fixture, assert ≥1
detection, all quads have 4 corners, all coords in source-image bounds. Print
per-stage timing.

**Manual verification gate:** `cargo tauri dev` → load a Myanmar fixture →
toggle Seg dropdown Kraken↔PP-OCR → confirm both produce sensible line boxes
in the overlay.

**Explicitly NOT tested:** recognition accuracy on PP-OCR-segmented lines
(empirical — we log it, recommend Tesseract recog in a UI hint, and ship);
cross-platform SIMD kernel correctness (trusting ppocr-rs upstream; vendored
verbatim).

## Open questions deferred to implementation plan

- Exact Myanmar fixture image to use for the smoke test (check whether
  `kraken-engine/testdata/` has a reusable one, or add one).
- Whether to surface a one-line UI hint recommending Tesseract recog when
  Seg=PP-OCR (since Kraken recog on PP-OCR-shaped lines is unverified).
- Whether the `x86.rs` kernel needs the Accelerate conditional that
  `kraken-engine` uses on macOS arm64, or if ppocr-rs's standalone x86 path
  is sufficient.
