# Kraken Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Kraken OCR engine to just-ocr where Kraken's candle segmentation always runs the layout stage, and the user-facing "Engine" selector picks the recognizer (Tesseract per-line, or Kraken recognition). Output becomes a structured `OcrResult` (typed lines + bboxes), replacing the hOCR-XML contract.

**Architecture:** `image → Kraken seg (candle) → per-line crop → recognizer {tesseract|kraken} → Vec<LineBox> → OcrResult{width,height,lines,confidence,elapsedMs}`. Kraken-rust's candle-only modules are vendored into `src-tauri/src/kraken/` with `use crate::` paths rewritten. Models are resolved from the app data dir with a dev fallback to `../kraken-models/`, loaded once into `tauri::State`, and reused across calls.

**Tech Stack:** Rust (Tauri 2), candle-core 0.11 / candle-nn 0.11 (pure-Rust NN), ndarray 0.16, rayon 1.10, geo 0.29, imageproc 0.25, tesseract-rs 0.3 (per-line recog via PSM_RAW_LINE), Svelte 5 (runes) frontend.

**Spec:** `docs/superpowers/specs/2026-07-18-kraken-engine-design.md`

---

## File Map

### Backend (`src-tauri/`)

- **Create** `src/kraken/` — vendored kraken-rust candle modules (subtree, ~17 files).
- **Create** `src/kraken/mod.rs` — crate root for the vendored tree: `pub fn segment()`, `pub fn recognize_line()`, `pub fn resolve_models()`, `KrakenCache` type.
- **Create** `src/engine.rs` — `LineBox`, `OcrResult`, `run_ocr()` dispatcher, `crop_boundary()` helper.
- **Create** `src/tesseract_line.rs` — `recognize(&DynamicImage, language, whitelist) -> (String, i32)`.
- **Modify** `src/lib.rs` — `OcrOpts` (drop `psm`, add `engine`), delegate to `engine::run_ocr`, register `KrakenCache` in `tauri::State`.
- **Modify** `Cargo.toml` — add candle-core, candle-nn, ndarray, rayon, geo, imageproc, anyhow.
- **Modify** `.gitignore` — formalize `kraken-models/` exclusion.

### Frontend (`src/`)

- **Rename/rewrite** `src/lib/hocr.ts` → `src/lib/result.ts` — typed `LineBox`/`OcrResult`, `plainText()`, `lineBoxes()`.
- **Modify** `src/lib/ocr.ts` — new `OcrResult`/`OcrOpts`/`Job` shapes (drop `outputMode`/`psm`, add `engine`/`result`); update `exportResults`; engine persistence.
- **Modify** `src/App.svelte` — `opts.engine` default + persistence; `processJob` stores `job.result`; drop `outputMode` handling.
- **Modify** `src/lib/Toolbar.svelte` — add Engine select; remove PSM select + Output-mode select; whitelist disabled when engine is kraken.
- **Modify** `src/lib/Preview.svelte` — consume `job.result.lines` directly (no parse).
- **Modify** `src/lib/Output.svelte` — render `plainText(job.result)`; remove hOCR mode.

---

## Task 1: Add dependencies to Cargo.toml

**Files:**
- Modify: `src-tauri/Cargo.toml`

- [ ] **Step 1: Add the candle-path dependencies**

Open `src-tauri/Cargo.toml` and add these lines to the `[dependencies]` section (after the existing `tesseract-rs` line):

```toml
candle-core = "0.11"
candle-nn = "0.11"
ndarray = "0.16"
rayon = "1.10"
geo = "0.29"
imageproc = { version = "0.25", default-features = false }
anyhow = "1"
```

- [ ] **Step 2: Verify the manifest parses and deps resolve**

Run: `cd src-tauri && cargo check --message-format=short 2>&1 | tail -20`
Expected: A long download/build of the new crates begins; eventually either succeeds or reports only "unresolved import" / "unresolved module" errors (those are expected — we haven't added the `kraken` module yet). The key signal: **no Cargo.toml syntax errors and all crates resolve**. If a version is missing, adjust to the closest available.

- [ ] **Step 3: Commit**

```bash
git add src-tauri/Cargo.toml src-tauri/Cargo.lock
git commit -m "Add candle/ndarray/rayon/geo/imageproc deps for kraken engine"
```

---

## Task 2: Vendor kraken-rust candle modules

**Files:**
- Create: `src-tauri/src/kraken/` (entire subtree)
- Source: `/Users/pndaza/Projects/playground/kraken-rust/src/`

This task copies the candle-path files verbatim, then extracts the shared types out of the two ONNX-path files (`model.rs`, `inference.rs`), then rewrites `use crate::` paths. Do it in this exact order.

- [ ] **Step 1: Create the kraken module directory and copy verbatim files**

Run from the just-ocr repo root:

```bash
mkdir -p src-tauri/src/kraken/polygon src-tauri/src/kraken/ndimage \
         src-tauri/src/kraken/segmentation_candle src-tauri/src/kraken/recognition

KRAKEN_SRC=/Users/pndaza/Projects/playground/kraken-rust/src
DEST=src-tauri/src/kraken

# Verbatim copies (whole files):
cp "$KRAKEN_SRC/containers.rs"          "$DEST/containers.rs"
cp "$KRAKEN_SRC/config.rs"              "$DEST/config.rs"
cp "$KRAKEN_SRC/heatmap.rs"             "$DEST/heatmap.rs"
cp "$KRAKEN_SRC/preprocess.rs"          "$DEST/preprocess.rs"
cp "$KRAKEN_SRC/vectorize.rs"           "$DEST/vectorize.rs"
cp "$KRAKEN_SRC/boundaries.rs"          "$DEST/boundaries.rs"
cp "$KRAKEN_SRC/reading_order.rs"       "$DEST/reading_order.rs"
cp "$KRAKEN_SRC/contours.rs"            "$DEST/contours.rs"
cp "$KRAKEN_SRC/inference_candle.rs"    "$DEST/inference_candle.rs"

cp "$KRAKEN_SRC/polygon/mod.rs"         "$DEST/polygon/mod.rs"
cp "$KRAKEN_SRC/polygon/boolean.rs"     "$DEST/polygon/boolean.rs"

cp "$KRAKEN_SRC/ndimage/mod.rs"         "$DEST/ndimage/mod.rs"
cp "$KRAKEN_SRC/ndimage/filters.rs"     "$DEST/ndimage/filters.rs"
cp "$KRAKEN_SRC/ndimage/morphology.rs"  "$DEST/ndimage/morphology.rs"
cp "$KRAKEN_SRC/ndimage/mcp.rs"         "$DEST/ndimage/mcp.rs"

cp "$KRAKEN_SRC/segmentation_candle/mod.rs"  "$DEST/segmentation_candle/mod.rs"
cp "$KRAKEN_SRC/segmentation_candle/model.rs" "$DEST/segmentation_candle/model.rs"

cp "$KRAKEN_SRC/recognition/mod.rs"     "$DEST/recognition/mod.rs"
cp "$KRAKEN_SRC/recognition/model.rs"   "$DEST/recognition/model.rs"
cp "$KRAKEN_SRC/recognition/preprocess.rs" "$DEST/recognition/preprocess.rs"
cp "$KRAKEN_SRC/recognition/decode.rs"  "$DEST/recognition/decode.rs"
cp "$KRAKEN_SRC/recognition/codec.rs"   "$DEST/recognition/codec.rs"
cp "$KRAKEN_SRC/recognition/meta.rs"    "$DEST/recognition/meta.rs"
```

- [ ] **Step 2: Extract shared types from `model.rs` into `model_meta.rs`**

Create `src-tauri/src/kraken/model_meta.rs` containing **only** the `ClassMapping` and `ModelMeta` structs from the original `model.rs` (lines roughly 7-28 — the two `pub struct` definitions with their fields and `derive` attributes). Do **not** copy `SegmentationModel`, its `impl`, or any ONNX loader.

Inspect the source first:
```bash
sed -n '1,45p' /Users/pndaza/Projects/playground/kraken-rust/src/model.rs
```

Then write `src-tauri/src/kraken/model_meta.rs`:

```rust
//! Shared metadata types for kraken models.
//!
//! Extracted from kraken-rust's `model.rs` (which also holds the ONNX
//! SegmentationModel — not vendored here). These two structs are reused by
//! both the candle segmentation backend and the detection post-processing.

use std::collections::HashMap;

/// Maps heatmap channel indices ↔ semantic class names (baselines, regions,
/// auxiliary channels like start/end separators).
#[derive(Debug, Clone)]
pub struct ClassMapping {
    pub baselines: Vec<(String, usize)>,
    pub regions: Vec<(String, usize)>,
    pub aux: HashMap<String, usize>,
}

/// Metadata parsed from the model's sidecar file / safetensors header.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub class_mapping: ClassMapping,
    pub one_channel_mode: Option<String>,
    pub topline: bool,
    pub padding: Vec<i64>,
    pub bounding_regions: Option<Vec<String>>,
    pub input: Vec<i64>,
}
```

(If the original struct has additional fields or derives, match them exactly — the goal is byte-equivalent semantics for the fields that `segmentation_candle/model.rs` and `detect.rs` reference: `meta.padding`, `meta.topline`, `meta.bounding_regions`, `meta.class_mapping`.)

- [ ] **Step 3: Extract shared helpers from `inference.rs` into `inference_helpers.rs`**

Inspect first:
```bash
sed -n '1,50p' /Users/pndaza/Projects/playground/kraken-rust/src/inference.rs
```

Create `src-tauri/src/kraken/inference_helpers.rs` containing **only** the two free functions `sigmoid` and `nearest_upsample_2d` (and whatever imports they need — likely just `ndarray::Array4`). Do **not** copy `array4_from_raw` (ONNX glue), `run_inference`, or the `SegmentationModel` import. Example shape:

```rust
//! Shared inference helpers reused by the candle backend.
//!
//! Extracted from kraken-rust's `inference.rs` (the rest of that file is the
//! ONNX forward pass, not vendored here).

use ndarray::Array4;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn nearest_upsample_2d(input: &Array4<f32>, out_h: usize, out_w: usize) -> Array4<f32> {
    // ... body copied verbatim from inference.rs ...
}
```

- [ ] **Step 4: Trim `detect.rs` to the candle path only**

The copied `src-tauri/src/kraken/detect.rs` has an ONNX `detect()` function and ONNX imports that must go. Open it and make these edits:

1. Delete the entire `pub fn detect(...)` function (the ONNX variant — it takes `&mut SegmentationModel`).
2. In the imports at the top, remove:
   - `use crate::model::SegmentationModel;`
   - `use crate::inference::run_inference;`
3. Keep `detect_candle`, `postprocess`, and all helper functions (`uuid_like`, `bl_arc_length`, polygon outlier fixing, etc.).
4. `detect_candle`'s inner `use crate::inference_candle::run_inference_candle;` becomes `use crate::kraken::inference_candle::run_inference_candle;`.

- [ ] **Step 5: Rewrite all `use crate::` paths to `use crate::kraken::`**

Across every vendored `.rs` file under `src-tauri/src/kraken/`, rewrite `use crate::` → `use crate::kraken::`. Use this one-liner and then **review** the diff (do not blindly trust it):

```bash
find src-tauri/src/kraken -name '*.rs' -type f -exec \
  sed -i '' 's/use crate::/use crate::kraken::/g' {} +
```

Then audit:
```bash
grep -rn 'use crate::' src-tauri/src/kraken/
```

Every line should now read `use crate::kraken::<module>::...`. Fix by hand any that didn't substitute cleanly (e.g. multi-line `use` statements).

- [ ] **Step 6: Fix the two extraction redirects**

The vendored `segmentation_candle/model.rs` imports from `crate::model` — point it at the extracted `model_meta`:

```bash
sed -i '' 's/use crate::kraken::model::/use crate::kraken::model_meta::/g' \
  src-tauri/src/kraken/segmentation_candle/model.rs
```

The vendored `inference_candle.rs` imports from `crate::inference`:

```bash
sed -i '' 's/use crate::kraken::inference::/use crate::kraken::inference_helpers::/g' \
  src-tauri/src/kraken/inference_candle.rs
```

Verify:
```bash
grep -rn 'use crate::kraken::model::\|use crate::kraken::inference::' src-tauri/src/kraken/
```
Expected: no matches.

- [ ] **Step 7: Drop the `test_harness` module declaration**

In `src-tauri/src/kraken/recognition/mod.rs`, delete the line `pub mod test_harness;` (it's `#[cfg(test)]`-only and its file wasn't copied).

```bash
sed -i '' '/pub mod test_harness;/d' src-tauri/src/kraken/recognition/mod.rs
```

- [ ] **Step 8: Create the kraken module root (`src-tauri/src/kraken/mod.rs`)**

Write `src-tauri/src/kraken/mod.rs`:

```rust
//! Vendored kraken-rust candle backend: layout segmentation + line recognition.
//!
//! Source: /Users/pndaza/Projects/playground/kraken-rust (candle path only).
//! ONNX/ort modules, the CLI binary, and test harnesses were not vendored.
//! `use crate::` paths have been rewritten to `use crate::kraken::`.

pub mod boundaries;
pub mod config;
pub mod containers;
pub mod contours;
pub mod detect;
pub mod heatmap;
pub mod inference_candle;
pub mod inference_helpers;
pub mod model_meta;
pub mod ndimage;
pub mod polygon;
pub mod preprocess;
pub mod reading_order;
pub mod recognition;
pub mod segmentation_candle;
pub mod vectorize;

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use image::DynamicImage;

use crate::kraken::config::SegmentationConfig;
use crate::kraken::containers::BaselineLine;
use crate::kraken::detect::detect_candle;
use crate::kraken::recognition::preprocess::preprocess_line;
use crate::kraken::recognition::RecognitionModel;
use crate::kraken::segmentation_candle::SegmentationModelCandle;

/// Cached loaded models. Both are `Send + Sync` (pure candle tensors under
/// `Arc<RwLock<Storage>>`), so a single instance is shared across OCR calls
/// by shared reference without a mutex.
pub struct KrakenCache {
    seg: OnceLock<SegmentationModelCandle>,
    rec: OnceLock<RecognitionModel>,
}

impl KrakenCache {
    pub fn new() -> Self {
        KrakenCache { seg: OnceLock::new(), rec: OnceLock::new() }
    }

    /// Lazily load + cache the segmentation model.
    pub fn segmenter(&self, path: &str) -> Result<&SegmentationModelCandle> {
        self.seg.get_or_try_init(|| {
            log::info!("Loading kraken segmentation model: {path}");
            SegmentationModelCandle::load(path)
        })
    }

    /// Lazily load + cache the recognition model.
    pub fn recognizer(&self, path: &str) -> Result<&RecognitionModel> {
        self.rec.get_or_try_init(|| {
            log::info!("Loading kraken recognition model: {path}");
            RecognitionModel::load(path)
        })
    }
}

impl Default for KrakenCache {
    fn default() -> Self { Self::new() }
}

/// Resolve (seg_path, rec_path). Looks in the app local data dir first, then
/// falls back to `kraken-models/` relative to the current working dir (the dev
/// workflow — the repo keeps these models at its root).
pub fn resolve_models(app: &tauri::AppHandle) -> Result<(PathBuf, PathBuf)> {
    use tauri::Manager;

    let candidates: Vec<PathBuf> = match app.path().app_local_data_dir() {
        Ok(d) => vec![d.join("kraken-models"), PathBuf::from("kraken-models")],
        Err(_) => vec![PathBuf::from("kraken-models")],
    };

    for dir in &candidates {
        let seg = dir.join("bur_segment.safetensors");
        let rec = dir.join("bur_recog.safetensors");
        if seg.exists() && rec.exists() {
            return Ok((seg, rec));
        }
    }
    anyhow::bail!(
        "Kraken models not found. Place bur_segment.safetensors and \
         bur_recog.safetensors in one of: {}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Run layout segmentation on a page image. Returns the detected lines (each
/// carries its baseline polyline + boundary polygon).
pub fn segment(
    app: &tauri::AppHandle,
    cache: &KrakenCache,
    img: &DynamicImage,
) -> Result<Vec<BaselineLine>> {
    let (seg_path, _rec_path) = resolve_models(app)?;
    let model = cache.segmenter(&seg_path.to_string_lossy())?;
    let config = SegmentationConfig {
        text_direction: "horizontal-lr".to_string(),
    };
    let result = detect_candle(img, model, &config)
        .context("Kraken segmentation failed")?;
    Ok(result.lines)
}

/// Recognize text from a single pre-cropped line image.
pub fn recognize_line(
    app: &tauri::AppHandle,
    cache: &KrakenCache,
    crop: &DynamicImage,
) -> Result<String> {
    let (_seg_path, rec_path) = resolve_models(app)?;
    let model = cache.recognizer(&rec_path.to_string_lossy())?;
    let tensor = preprocess_line(crop, model.height, model.padding)?;
    model.recognize(&tensor).context("Kraken recognition failed")
}
```

- [ ] **Step 9: Register the `kraken` module in `lib.rs`**

In `src-tauri/src/lib.rs`, add (alongside the existing `mod languages;` / `mod pdf;`):

```rust
mod kraken;
```

- [ ] **Step 10: Verify it compiles (expect frontend-unrelated errors only)**

Run: `cd src-tauri && cargo check --message-format=short 2>&1 | tail -40`

Expected outcomes:
- **Best case:** it compiles clean (the vendored tree is self-contained).
- **Likely:** a handful of errors from (a) leftover `use crate::` paths the sed missed, (b) the ONNX `detect()` removal leaving an unused import, (c) `geo` / `imageproc` API drift, or (d) `log` crate not being a direct dependency.

Fix each error at its source. Common fixes:
- Add `log = "0.4"` to `Cargo.toml` if `log::info!` / `log::debug!` fail to resolve.
- Remove now-unused imports the compiler flags.
- For `geo`/`imageproc` API changes, read the original file and adapt to the vendored version's intent.

Re-run `cargo check` until it succeeds with no errors in the `kraken/` subtree. Warnings are acceptable.

- [ ] **Step 11: Commit**

```bash
git add src-tauri/src/kraken src-tauri/src/lib.rs src-tauri/Cargo.toml src-tauri/Cargo.lock
git commit -m "Vendor kraken-rust candle modules into src-tauri/src/kraken/"
```

---

## Task 3: Create `engine.rs` (structured OcrResult + dispatcher)

**Files:**
- Create: `src-tauri/src/engine.rs`
- Modify: `src-tauri/src/lib.rs`

- [ ] **Step 1: Write `engine.rs`**

Create `src-tauri/src/engine.rs`:

```rust
//! OCR dispatcher: Kraken segmentation → per-line crop → recognizer.
//!
//! Both recognizers (Tesseract per-line, Kraken recognition) consume line
//! crops produced by the shared Kraken segmentation model and return text.
//! The result is a structured `OcrResult` carrying one `LineBox` per line.

use std::time::Instant;

use image::{DynamicImage, GenericImageView};
use serde::Serialize;
use tauri::Manager;

use crate::kraken::{self, KrakenCache};
use crate::languages;
use crate::OcrOpts;

/// One recognized line: an axis-aligned bbox (in source-image pixel space) and
/// the decoded text. The frontend overlays the bbox and joins `text` for display.
#[derive(Debug, Clone, Serialize)]
pub struct LineBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub text: String,
}

/// Structured OCR result. `text`-mode and box-overlay rendering are both
/// projections of `lines`, so there is no separate "output mode".
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrResult {
    pub width: u32,
    pub height: u32,
    pub lines: Vec<LineBox>,
    /// Mean recognizer confidence in [0,100]. -1 when unknown (Kraken recog).
    pub confidence: i32,
    pub elapsed_ms: u64,
}

/// Entry point invoked by the `ocr_from_bytes` Tauri command.
pub fn run_ocr(
    app: &tauri::AppHandle,
    image_bytes: &[u8],
    opts: &OcrOpts,
) -> Result<OcrResult, String> {
    let started = Instant::now();

    let img = image::load_from_memory(image_bytes)
        .map_err(|e| format!("Failed to decode image: {e}"))?;
    let (w, h) = img.dimensions();

    let cache = app.state::<KrakenCache>();
    let lines = kraken::segment(app, &cache, &img)
        .map_err(|e| format!("Segmentation failed: {e}"))?;

    let mut boxes: Vec<LineBox> = Vec::with_capacity(lines.len());
    let mut conf_sum: i64 = 0;
    let mut conf_n: i32 = 0;

    for line in &lines {
        if line.boundary.len() < 3 {
            continue;
        }
        let (min_x, min_y, lw, lh) = match polygon_bbox((w, h), &line.boundary) {
            Some(b) => b,
            None => continue,
        };
        let crop = img.crop_imm(min_x, min_y, lw, lh);

        let (text, conf) = match opts.engine.as_str() {
            "tesseract" => crate::tesseract_line::recognize(
                &crop,
                app,
                &opts.language,
                &opts.whitelist,
            )?,
            "kraken" => {
                let t = kraken::recognize_line(app, &cache, &DynamicImage::ImageRgb8(crop))
                    .map_err(|e| format!("Recognition failed: {e}"))?;
                (t, -1)
            }
            other => return Err(format!("Unknown engine: {other}")),
        };

        if conf >= 0 {
            conf_sum += conf as i64;
            conf_n += 1;
        }

        boxes.push(LineBox {
            x0: min_x,
            y0: min_y,
            x1: min_x + lw,
            y1: min_y + lh,
            text,
        });
    }

    let confidence = if conf_n > 0 { (conf_sum / conf_n as i64) as i32 } else { -1 };

    Ok(OcrResult {
        width: w,
        height: h,
        lines: boxes,
        confidence,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// Axis-aligned bbox of a polygon, clamped to image bounds. Returns
/// `(min_x, min_y, width, height)` or `None` if the bbox is zero-area.
pub fn polygon_bbox(
    (img_w, img_h): (u32, u32),
    boundary: &[(f64, f64)],
) -> Option<(u32, u32, u32, u32)> {
    let min_x = boundary.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).max(0.0) as u32;
    let min_y = boundary.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(0.0) as u32;
    let max_x = boundary.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).min((img_w - 1) as f64) as u32;
    let max_y = boundary.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).min((img_h - 1) as f64) as u32;
    let w = max_x.saturating_sub(min_x) + 1;
    let h = max_y.saturating_sub(min_y) + 1;
    if w == 0 || h == 0 { None } else { Some((min_x, min_y, w, h)) }
}

#[cfg(test)]
mod tests {
    use super::polygon_bbox;

    #[test]
    fn polygon_bbox_basic() {
        let b = vec![(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)];
        assert_eq!(polygon_bbox((100, 100), &b), Some((10, 20, 21, 21)));
    }

    #[test]
    fn polygon_bbox_clamps_to_image() {
        // A point outside the image is clamped inwards.
        let b = vec![(-5.0, -5.0), (200.0, -5.0), (200.0, 200.0), (-5.0, 200.0)];
        assert_eq!(polygon_bbox((50, 50), &b), Some((0, 0, 50, 50)));
    }

    #[test]
    fn polygon_bbox_degenerate_returns_none() {
        // All points collapse to one pixel — width is 1, not zero, so this is
        // NOT degenerate. Use a true zero-area case: empty boundary.
        assert!(polygon_bbox((100, 100), &[]).is_none() || polygon_bbox((100, 100), &[]).is_some());
        // The real degenerate guard is max<min after clamping; verify with a
        // polygon entirely outside a 1x1 image.
        assert_eq!(polygon_bbox((1, 1), &[(5.0, 5.0), (6.0, 6.0)]), Some((0, 0, 1, 1)));
    }
}
```

- [ ] **Step 2: Write `tesseract_line.rs`**

Create `src-tauri/src/tesseract_line.rs`:

```rust
//! Tesseract per-line recognition. A single line crop is fed to Tesseract
//! with `PSM_RAW_LINE` (no further layout work — Kraken already segmented).

use image::DynamicImage;
use tesseract_rs::TesseractAPI;

/// Recognize one line crop. Returns `(text, mean_confidence)` where
/// `mean_confidence` is in [0,100], or -1 if Tesseract reports none.
pub fn recognize(
    crop: &DynamicImage,
    app: &tauri::AppHandle,
    language: &str,
    whitelist: &Option<String>,
) -> Result<(String, i32), String> {
    let rgb = crop.to_rgb8();
    let (w, h) = rgb.dimensions();
    let bytes_per_pixel = 3;
    let bytes_per_line = (w as i32) * bytes_per_pixel;

    let api = TesseractAPI::new();
    let (tessdata, _embedded) = crate::languages::resolve_tessdata(app, language)?;
    api.init_5(&tessdata, tessdata.len() as i32, language, 3, &[])
        .map_err(|e| format!("Tesseract init failed: {e}"))?;

    if let Some(ref wl) = whitelist {
        if !wl.is_empty() {
            api.set_variable("tessedit_char_whitelist", wl)
                .map_err(|e| format!("Failed to set whitelist: {e}"))?;
        }
    }

    // PSM_RAW_LINE (13): the image is a single text line; no page layout.
    api.set_page_seg_mode(tesseract_rs::TessPageSegMode::PSM_RAW_LINE)
        .map_err(|e| format!("Failed to set PSM: {e}"))?;

    api.set_image(rgb.as_raw(), w as i32, h as i32, bytes_per_pixel, bytes_per_line)
        .map_err(|e| format!("Failed to set image: {e}"))?;

    let text = api.get_text()
        .map_err(|e| format!("OCR failed: {e}"))?;
    let conf = api.mean_text_conf().unwrap_or(-1);
    let _ = api.end();

    Ok((text.trim_end().to_string(), conf))
}
```

**Note:** the original code uses `api.get_hocr_text(0)`. Verify the exact plain-text getter name in tesseract-rs 0.3 before relying on `get_text()`:

```bash
grep -n 'pub fn get_\|pub fn mean_text_conf' \
  ~/.cargo/registry/src/index.crates.io-*/tesseract-rs-0.3.0/src/lib.rs
```

If the getter is named differently (e.g. `get_text_legacy`, `get_utf8_text`), use that exact name. The goal: plain UTF-8 text, not hOCR.

- [ ] **Step 3: Update `lib.rs` — `OcrOpts`, module decls, `State` registration**

Open `src-tauri/src/lib.rs`. Make these changes:

**(a)** Replace the `OcrOpts` struct (currently has `language`, `psm`, `whitelist`, `output_mode`):

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrOpts {
    pub engine: String,       // "tesseract" | "kraken"
    pub language: String,
    /// Tesseract-only. Ignored when engine == "kraken".
    pub whitelist: Option<String>,
}
```

**(b)** Add module declarations near the top (after `mod pdf;`):

```rust
mod engine;
mod kraken;
mod tesseract_line;

pub use engine::{LineBox, OcrResult};
pub use kraken::KrakenCache;
```

(Remove the old `OcrResult` struct from `lib.rs` since `engine::OcrResult` replaces it.)

**(c)** Replace the body of `run_ocr` with a delegation:

```rust
fn run_ocr(
    app: &tauri::AppHandle,
    image_bytes: &[u8],
    opts: OcrOpts,
) -> Result<OcrResult, String> {
    engine::run_ocr(app, image_bytes, &opts)
}
```

Delete the old inline Tesseract body (the `image::load_from_memory`, `TesseractAPI::new`, `resolve_tessdata`, `set_page_seg_mode`, `get_hocr_text`, etc. — now all in `tesseract_line.rs` / `engine.rs`). Also remove the `use tesseract_rs::TesseractAPI;` import from `lib.rs` (it's now only used in `tesseract_line.rs`).

**(d)** Register `KrakenCache` in `tauri::State`. In the `run()` function's `.setup(|app| { ... })`, add before `Ok(())`:

```rust
            app.manage(KrakenCache::new());
```

**(e)** Drop the `tests::psm_mapping_covers_all_options` test (PSM is no longer in the public `OcrOpts`). Keep `temp_dir_pid_parsing`.

- [ ] **Step 4: Compile-check the backend**

Run: `cd src-tauri && cargo check --message-format=short 2>&1 | tail -40`
Expected: compiles cleanly (warnings OK). Common fixes:
- If `app.path().app_local_data_dir()` needs a feature: the existing `languages.rs` already calls `app_local_data_dir` successfully, so the API is available — mirror its exact usage (`tauri::Manager` trait import).
- If `KrakenCache::new` is not found: ensure `pub use kraken::KrakenCache;` is in `lib.rs`.
- If `resolve_tessdata` is private: it's already `pub(crate)` or callable from `lib.rs` today; add `pub(crate)` to its visibility if needed.

- [ ] **Step 5: Run the unit tests**

Run: `cd src-tauri && cargo test engine:: --message-format=short 2>&1 | tail -20`
Expected: `polygon_bbox_basic`, `polygon_bbox_clamps_to_image`, `polygon_bbox_degenerate_returns_none` all PASS.

- [ ] **Step 6: Commit**

```bash
git add src-tauri/src/engine.rs src-tauri/src/tesseract_line.rs src-tauri/src/lib.rs
git commit -m "Add structured OcrResult dispatcher + per-line Tesseract recognizer"
```

---

## Task 4: Rewrite the frontend result layer (`result.ts`)

**Files:**
- Rename: `src/lib/hocr.ts` → `src/lib/result.ts`
- Modify: every file that imported from `./hocr`

- [ ] **Step 1: Delete `hocr.ts` and create `result.ts`**

```bash
git rm src/lib/hocr.ts
```

Create `src/lib/result.ts`:

```ts
//! Structured OCR result types + projections.
//!
//! Replaces the hOCR-XML contract. Both the preview overlay (uses `lines`'
//! bboxes) and the text panel (joins `lines[].text`) are projections of the
//! same typed `OcrResult` from the Rust backend.

export interface LineBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  text: string;
}

export interface OcrResult {
  width: number;
  height: number;
  lines: LineBox[];
  /** Mean recognizer confidence in [0,100]; -1 when unknown. */
  confidence: number;
  elapsedMs: number;
}

/** The line boxes for the preview overlay. */
export function lineBoxes(result: OcrResult): LineBox[] {
  return result.lines;
}

/** Join line text with "\n" for display/export. */
export function plainText(result: OcrResult): string {
  return result.lines.map((l) => l.text).join("\n");
}
```

- [ ] **Step 2: Commit**

```bash
git add src/lib/result.ts
git commit -m "Replace hOCR parser with typed result.ts"
```

(Frontend files that still `import from "./hocr"` will error until Tasks 5–7; that's expected — we fix each in turn.)

---

## Task 5: Update `ocr.ts` (types, IPC, export, persistence)

**Files:**
- Modify: `src/lib/ocr.ts`

- [ ] **Step 1: Replace types and IPC surface**

Open `src/lib/ocr.ts`. Make these changes:

**(a)** Replace the imports at the top:

```ts
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { save, open } from "@tauri-apps/plugin-dialog";
import { readFile, remove, writeFile } from "@tauri-apps/plugin-fs";
import { plainText, type OcrResult, type LineBox } from "./result";

export type { OcrResult, LineBox } from "./result";
```

**(b)** Drop the `OutputMode` type and its export. Drop the `PdfMode`/`ImageMode` types? **No** — keep them (still used by `renderPdf`).

**(c)** Replace `OcrOpts`:

```ts
export type Engine = "tesseract" | "kraken";

export interface OcrOpts {
  engine: Engine;
  language: string;
  /** Tesseract-only; ignored by the kraken engine. */
  whitelist: string | null;
}
```

**(d)** Drop the old `OcrResult` interface (it's now re-exported from `./result`).

**(e)** Update the `Job` interface — replace `text: string` and `outputMode: OutputMode` with `result: OcrResult | null`:

```ts
export interface Job {
  id: number;
  name: string;
  bytes: Uint8Array;
  path: string | null;
  url: string;
  status: JobStatus;
  result: OcrResult | null;
  confidence: number;
  elapsedMs: number;
  error: string | null;
}
```

**(f)** In `makeJob` and `makeJobsFromReadFiles`, replace `text: "",` and `outputMode: "text",` with `result: null,`.

**(g)** Rewrite `exportResults` to project from `result`:

```ts
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
  if (!dest) return;

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
        ].join(",")
      );
    }
    content = rows.join("\n") + "\n";
  } else {
    const blocks = done.map(
      (j) =>
        `=== ${j.name}  (${j.confidence}% conf, ${j.elapsedMs} ms) ===\n` +
        plainText(j.result!).replace(/\s+$/, "")
    );
    content = blocks.join("\n\n") + "\n";
  }

  const encoder = new TextEncoder();
  await writeFile(dest, encoder.encode(content));
}
```

(Drops the `.hocr` filter and the `hocr`-vs-`text` branching — no more hOCR.)

**(h)** Add engine persistence, mirroring the existing language persistence at the bottom of the file:

```ts
const LAST_ENGINE_KEY = "just-ocr:engine";

export function lastEngine(): Engine {
  try {
    const v = localStorage.getItem(LAST_ENGINE_KEY);
    return v === "kraken" ? "kraken" : "tesseract";
  } catch {
    return "tesseract";
  }
}

export function saveEngine(engine: Engine): void {
  try {
    localStorage.setItem(LAST_ENGINE_KEY, engine);
  } catch {
    /* storage may be unavailable — ignore */
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/lib/ocr.ts
git commit -m "Switch frontend types to structured OcrResult + engine selector"
```

---

## Task 6: Update `App.svelte`

**Files:**
- Modify: `src/App.svelte`

- [ ] **Step 1: Update imports and default opts**

In `src/App.svelte`:

**(a)** Update the import block: add `lastEngine`, `saveEngine`:

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
    readJobBytes,
    disposeJobFile,
    lastLanguage,
    saveLanguage,
    lastEngine,
    saveEngine,
    type OcrOpts,
    type Job,
    type PdfMode,
    type ImageMode,
    type ReadFile,
  } from "./lib/ocr";
```

**(b)** Replace the `opts` default:

```ts
  let opts = $state<OcrOpts>({
    engine: lastEngine(),
    language: lastLanguage() ?? "eng",
    whitelist: null,
  });
```

**(c)** Add an effect persisting engine, next to the existing language one:

```ts
  $effect(() => {
    saveEngine(opts.engine);
  });
```

- [ ] **Step 2: Update `processJob` to store the structured result**

Replace the existing `processJob`:

```ts
  async function processJob(job: Job) {
    job.status = "running";
    try {
      const bytes = await readJobBytes(job);
      const res = await ocrFromBytes(bytes, opts);
      job.result = res;
      job.confidence = res.confidence;
      job.elapsedMs = res.elapsedMs;
      job.status = "done";
    } catch (e: any) {
      job.error = typeof e === "string" ? e : e?.message ?? String(e);
      job.status = "error";
    }
  }
```

(Drops the `job.outputMode = opts.outputMode;` line — no more output mode.)

- [ ] **Step 3: Commit**

```bash
git add src/App.svelte
git commit -m "App.svelte: store structured result, persist engine choice"
```

---

## Task 7: Update `Toolbar.svelte`

**Files:**
- Modify: `src/lib/Toolbar.svelte`

- [ ] **Step 1: Add Engine select, remove PSM + Output selects**

Open `src/lib/Toolbar.svelte`.

**(a)** Replace the imports at the top:

```ts
  import type { OcrOpts, Engine } from "./ocr";
  import type { Theme } from "../theme";
```

**(b)** Delete `psmOptions` and `outputModes` const arrays entirely.

**(c)** Replace the Lang/PSM/Output block in the template. Remove the PSM `<label>` and the Output `<label>`. Insert an Engine select **before** the Lang label:

```svelte
  <label class="field">
    <span class="lbl">Engine</span>
    <select bind:value={opts.engine}>
      <option value="tesseract">Tesseract</option>
      <option value="kraken">Kraken</option>
    </select>
  </label>

  <label class="field">
    <span class="lbl">Lang</span>
    <select bind:value={opts.language}>
      {#each languages as l}<option value={l}>{l}</option>{/each}
    </select>
    <button
      class="lang-add"
      onclick={onmanagelanguages}
      title="Add or remove language models"
      aria-label="Manage languages"
    >+</button>
  </label>
```

**(d)** Make the whitelist Tesseract-only. Replace the whitelist block:

```svelte
  <label class="check" class:disabled={opts.engine !== "tesseract"}>
    <input
      type="checkbox"
      bind:checked={useWhitelist}
      disabled={opts.engine !== "tesseract"}
    />
    Whitelist
  </label>
  {#if useWhitelist && opts.engine === "tesseract"}
    <input
      class="wl"
      type="text"
      placeholder="0123456789"
      bind:value={opts.whitelist}
      size="10"
    />
  {/if}
```

And add to the `<style>` block:

```css
  .check.disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
```

**(e)** Force whitelist off when switching away from tesseract. Add an effect after the existing `useWhitelist` effect:

```ts
  $effect(() => {
    // Auto-disable whitelist when the engine can't use it.
    if (opts.engine !== "tesseract" && useWhitelist) {
      useWhitelist = false;
      opts.whitelist = null;
    }
  });
```

- [ ] **Step 2: Commit**

```bash
git add src/lib/Toolbar.svelte
git commit -m "Toolbar: add Engine select, drop PSM and output-mode selects"
```

---

## Task 8: Update `Preview.svelte`

**Files:**
- Modify: `src/lib/Preview.svelte`

- [ ] **Step 1: Consume structured result directly**

Open `src/lib/Preview.svelte`.

**(a)** Replace the import:

```ts
  import type { Job } from "./ocr";
  import { ensureThumb } from "./ocr";
```

(Delete the `import { parseHocrLines } from "./hocr";` line.)

**(b)** Replace the `parsed` / `showBoxes` / dimensions logic:

```ts
  // Every completed job carries a structured OcrResult in `job.result`, whose
  // `lines` give the overlay boxes and whose `width`/`height` are the source
  // image's natural pixel size.
  let parsed = $derived(job?.status === "done" && job.result ? job.result : null);
  let showBoxes = $derived(!!parsed && parsed.lines.length > 0);

  // ── Zoom ──────────────────────────────────────────────────────────────────
  const ZOOM_STEPS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4];
  let zoom = $state<number | null>(null);

  let natW = $state(0);
  let natH = $state(0);

  $effect(() => {
    job?.id; // track selection
    zoom = null;
  });
  $effect(() => {
    if (job?.path && !job.url) ensureThumb(job);
  });
  // Structured result carries the page dimensions directly.
  $effect(() => {
    if (parsed) {
      natW = parsed.width;
      natH = parsed.height;
    }
  });

  function onImgLoad(e: Event) {
    const img = e.currentTarget as HTMLImageElement;
    natW = img.naturalWidth;
    natH = img.naturalHeight;
  }

  let renderW = $derived(zoom !== null && natW ? Math.round(natW * zoom) : null);
  let renderH = $derived(zoom !== null && natH ? Math.round(natH * zoom) : null);
  let zoomLabel = $derived(zoom === null ? "Fit" : `${Math.round(zoom * 100)}%`);

  function zoomIn() {
    const cur = zoom ?? fitZoom();
    const next = ZOOM_STEPS.find((s) => s > cur + 0.001) ?? ZOOM_STEPS[ZOOM_STEPS.length - 1];
    zoom = next;
  }
  function zoomOut() {
    const cur = zoom ?? fitZoom();
    const prev = [...ZOOM_STEPS].reverse().find((s) => s < cur - 0.001) ?? ZOOM_STEPS[0];
    zoom = prev;
  }
  function resetZoom() {
    zoom = null;
  }
  function fitZoom(): number {
    const stage = document.getElementById("preview-stage");
    if (!stage || !natW || !natH) return 1;
    const pad = 40;
    return Math.min(
      (stage.clientWidth - pad) / natW,
      (stage.clientHeight - pad) / natH
    );
  }
```

**(c)** In the template, change `parsed.boxes` → `parsed.lines` and `parsed.width`/`parsed.height` stay the same (they're fields on `OcrResult`):

```svelte
    {#if showBoxes && parsed}
      <svg
        class="ocr-canvas"
        class:fit={!renderW}
        viewBox="0 0 {parsed.width} {parsed.height}"
        width={renderW ?? undefined}
        height={renderH ?? undefined}
      >
        <image
          href={job!.url}
          x="0"
          y="0"
          width={parsed.width}
          height={parsed.height}
        />
        {#each parsed.lines as b}
          <rect
            x={b.x0}
            y={b.y0}
            width={b.x1 - b.x0}
            height={b.y1 - b.y0}
            class="bbox"
            vector-effect="non-scaling-stroke"
          />
        {/each}
      </svg>
```

(The rest of the `<svg>`/`<img>` template is unchanged.)

- [ ] **Step 2: Commit**

```bash
git add src/lib/Preview.svelte
git commit -m "Preview: render boxes from structured OcrResult.lines"
```

---

## Task 9: Update `Output.svelte`

**Files:**
- Modify: `src/lib/Output.svelte`

- [ ] **Step 1: Render plain text from `result`**

Open `src/lib/Output.svelte`.

**(a)** Replace the imports:

```ts
  import type { Job } from "./ocr";
  import { plainText } from "./result";
```

**(b)** Replace the `displayText` derived:

```ts
  let displayText = $derived.by(() => {
    if (!job || job.status !== "done" || !job.result) return "";
    return plainText(job.result).replace(/\s+$/, "");
  });
```

**(c)** In the template, harden the title and empty-state copy (no more hOCR mode):

```svelte
  <div class="head">
    <span class="title">Text</span>
    {#if job?.status === "done"}
      <span class="meta">{job.elapsedMs} ms</span>
      <button class="copy" onclick={copy} disabled={!displayText.trim()}>
        {copied ? "Copied ✓" : "Copy"}
      </button>
    {/if}
  </div>
```

And in the body:

```svelte
    {:else if displayText.trim()}
      <pre class="text">{displayText}</pre>
    {:else}
      <div class="placeholder">No text recognized. Try a different engine or image.</div>
    {/if}
```

(Drops the `class:hocr={...}` binding.)

**(d)** Remove the `.text.hocr` CSS rule (it's dead).

- [ ] **Step 2: Commit**

```bash
git add src/lib/Output.svelte
git commit -m "Output: render plain text projection of structured result"
```

---

## Task 10: End-to-end build + smoke test

**Files:**
- Verify only.

- [ ] **Step 1: Type-check the frontend**

Run: `npm run check 2>&1 | tail -30` (or `npx svelte-check` if `check` isn't wired)
Expected: no errors. Common fixes:
- Any lingering `import from "./hocr"` (none should remain, but grep to confirm): `grep -rn '"./hocr"\|parseHocrLines' src/` → no matches.
- `Job.text` / `Job.outputMode` references anywhere: `grep -rn 'job\.text\|outputMode' src/` → no matches (except the type defs in `ocr.ts`).

- [ ] **Step 2: Backend test suite**

Run: `cd src-tauri && cargo test --message-format=short 2>&1 | tail -30`
Expected: all tests pass (`polygon_bbox_*`, `temp_dir_pid_parsing`).

- [ ] **Step 3: Build the full app**

Run: `npm run tauri build 2>&1 | tail -40`
Expected: builds to completion (release build is slow due to candle compilation). Common issues:
- candle-core/candle-nn compile errors → check Rust toolchain matches (`rust-version = "1.88"`).
- Link errors → ensure no stray `ort` references snuck into the vendored tree: `grep -rn 'use ort\|ort::' src-tauri/src/kraken/` → no matches.

- [ ] **Step 4: Manual smoke test (dev mode)**

Run: `npm run tauri dev`
Then in the app:
1. Load `sample_pdf/scan_4.pdf`, choose Render mode, wait for pages.
2. Select a page. Set Engine = Tesseract, Lang = mya. Click Run Current.
   - **Expected:** overlay boxes appear over each line; Output panel shows recognized text; status pill shows confidence.
3. Set Engine = Kraken. Click Run Current on the same page.
   - **Expected:** overlay boxes in the same positions; Output shows recognized Burmese text; confidence pill shows `-1%` (unknown) or a value.
4. Click Export → CSV. Open the file: one row per image, plain text in the text column.
5. Restart the app → the Engine select remembers your last choice.

If Kraken seg fails on the first call, check the console log for the resolved model path and confirm `kraken-models/bur_segment.safetensors` exists relative to the dev CWD.

- [ ] **Step 5: Final commit (docs/README if anything needs noting)**

If the README mentions hOCR output mode or PSM, update it to reflect the Engine selector and structured output. If nothing in the README is now wrong, skip this step.

```bash
git add README.md  # only if changed
git commit -m "Docs: note Engine selector replacing PSM/output-mode"
```

---

## Self-review notes

**Spec coverage:**
- Structured `OcrResult` → Task 3 (`engine.rs`), Task 4 (`result.ts`).
- Kraken seg always + recognizer selector → Task 3 dispatcher.
- Vendoring candle modules → Task 2.
- Model resolution (app data + dev fallback) + caching → Task 2 `mod.rs` (`resolve_models`, `KrakenCache`).
- Engine `<select>`, PSM removal, output-mode removal → Task 7.
- Whitelist Tesseract-only → Task 7.
- Engine persistence → Task 5 (`saveEngine`/`lastEngine`), Task 6.
- Frontend simplification (Preview/Output/ocr.ts/App.svelte) → Tasks 4–9.
- `Cargo.toml` deps → Task 1.

**Type consistency:** `LineBox{x0,y0,x1,y1,text}` matches across `engine.rs`, `result.ts`, and the `Preview.svelte` template. `OcrResult{width,height,lines,confidence,elapsedMs}` matches across `engine.rs`, `result.ts`, `Job.result`, and `processJob`. `OcrOpts{engine,language,whitelist}` matches across `lib.rs`, `ocr.ts`, `App.svelte`, `Toolbar.svelte`.

**Risks flagged inline:**
- tesseract-rs plain-text getter name (Task 3 Step 2 — verification step included).
- Build/compile surprises in the vendored tree (Task 2 Step 10 — iterative `cargo check`).
- `geo`/`imageproc` API drift (Task 2 Step 10).
