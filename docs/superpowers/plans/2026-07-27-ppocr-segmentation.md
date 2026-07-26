# PP-OCR Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PaddleOCR's PP-OCRv6 tiny text detector as an alternative Myanmar-only segmentation stage, selectable via a new "Seg" dropdown in the UI.

**Architecture:** A new self-contained `src-tauri/ppocr-engine/` crate vendors ppocr-rs's detector + DB postprocess (slimmed, NOT a workspace member — same opt-level=3 trick as kraken-engine). A `Segmenter` trait in the host abstracts over Kraken and PP-OCR; both engines sit behind `Arc<dyn Segmenter>`. The tiny-det model (1.70 MB) is bundled via `include_bytes!` at repo-root `ppocr-models/`. Dispatch stays Myanmar-only: PP-OCR is an alternative segmenter for `language == "mya"`, paired with either recognizer.

**Tech Stack:** Rust (edition 2021), Tauri v2, Svelte 5 + TypeScript. Vendored deps: `rayon`, `safetensors = "=0.7.0"`, `image = "0.25"`, `anyhow`, `serde`. The new crate has no candle, no wgpu, no download infra.

**Spec:** `docs/superpowers/specs/2026-07-27-ppocr-segmentation-design.md`

---

## File Structure

**New files (in `src-tauri/ppocr-engine/`):**
- `Cargo.toml` — crate manifest, deps: rayon, safetensors, image, anyhow, serde, log
- `src/lib.rs` — crate root; declares modules, re-exports `Detector`, `Detection`, `Point`
- `src/model.rs` — vendored from `clones/ppocr-rs/src/cpu/model.rs` lines 1–1685 (detector + shared helpers, drop recognizer 1686+)
- `src/backend.rs` — vendored from `clones/ppocr-rs/src/cpu/backend.rs` (verbatim, all 1867 lines)
- `src/ops.rs` — vendored from `clones/ppocr-rs/src/cpu/ops.rs` (verbatim, all 2269 lines)
- `src/kernels/mod.rs`, `neon.rs`, `x86.rs` — vendored from `clones/ppocr-rs/src/cpu/kernels/` (verbatim)
- `src/tensor.rs` — vendored from `clones/ppocr-rs/src/cpu/tensor.rs` (verbatim, 235 lines)
- `src/arena.rs` — vendored from `clones/ppocr-rs/src/cpu/arena.rs` (verbatim, 463 lines)
- `src/weights.rs` — vendored from `clones/ppocr-rs/src/cpu/weights.rs` + new `from_bytes` method
- `src/postprocess.rs` — vendored subset of `clones/ppocr-rs/src/ocr.rs` (transform, plan, extract_detections, DB helpers)
- `src/preprocess.rs` — vendored from `clones/ppocr-rs/src/preprocess.rs` + `preprocess/kernels.rs` (detector path only)

**New files (in `src-tauri/src/`):**
- `segmentation.rs` — `Segmenter` trait + `DetectedLine` type
- `segmenter_adapters.rs` — `KrakenSegmenter` + `PPOcrSegmenter` adapters
- `examples/smoke_ppocr.rs` — standalone smoke example modeled on `smoke_kraken.rs`

**Modified files:**
- `src-tauri/Cargo.toml` — add `ppocr-engine` path-dep (line ~53)
- `src-tauri/src/lib.rs` — add `segmenter` field to `OcrOpts` (~line 25–46)
- `src-tauri/src/engine.rs` — bundle bytes, two `OnceCell<Arc<...>>`, refactor `run_myanmar` + `kraken_engine`
- `.gitattributes` — track `ppocr-models/*.safetensors` as LFS
- `src/lib/ocr.ts` — add `Segmenter` type + `segmenter` field + persistence helpers
- `src/lib/Toolbar.svelte` — add "Seg" dropdown inside `{#if isMyanmar}`
- `src/App.svelte` — init `segmenter` in opts + persist effect

**New binary asset (LFS):**
- `ppocr-models/tiny-det.safetensors` (1.70 MB, downloaded from `PaddlePaddle/PP-OCRv6_tiny_det_safetensors`)

---

## Task 0: Create the feature branch

**Files:** n/a (git only)

- [ ] **Step 1: Create and switch to a new branch off main**

```bash
git checkout main
git pull --ff-only origin main 2>/dev/null || true
git checkout -b feat/ppocr-segmentation
```

- [ ] **Step 2: Verify clean starting state**

```bash
git status
git log --oneline -1
```
Expected: on `feat/ppocr-segmentation`, HEAD is the spec commit `6af00d2 docs(spec): PP-OCR segmentation integration design` or newer.

- [ ] **Step 3: Commit nothing (no changes yet) — branch is ready**

---

## Task 1: Download the tiny-det model and configure Git LFS

**Files:**
- Create: `ppocr-models/tiny-det.safetensors`
- Modify: `.gitattributes`

- [ ] **Step 1: Ensure LFS is installed and the directory exists**

```bash
git lfs install
mkdir -p ppocr-models
```

- [ ] **Step 2: Track ppocr models via LFS**

Append to `.gitattributes` (create the patterns if the file lacks them — `kraken-models/*.safetensors` is the existing pattern to mirror):

```
ppocr-models/*.safetensors filter=lfs diff=lfs merge=lfs -text
```

Verify with `cat .gitattributes` that the line is present exactly once.

- [ ] **Step 3: Download the tiny-det safetensors from HuggingFace**

```bash
cd ppocr-models
curl -L -o tiny-det.safetensors \
  "https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_safetensors/resolve/eae2ee920a39fb3087637d3dbb58df1896ec1f24/model.safetensors"
cd ..
```

(Revision `eae2ee920a39fb3087637d3dbb58df1896ec1f24` matches the `tiny-det` entry in `clones/ppocr-rs/models.json`.)

- [ ] **Step 4: Verify the file size and SHA-256**

```bash
ls -l ppocr-models/tiny-det.safetensors
shasum -a 256 ppocr-models/tiny-det.safetensors
```
Expected: file size `1786412` bytes (~1.70 MB); SHA-256 `3ac018be6f97499a08faa3bbdeb33640968d9307f6736d152902747a9f259593` (matches `models.json` `tiny-det` → `model.safetensors`).

If either mismatches, STOP — wrong file downloaded.

- [ ] **Step 5: Stage and commit**

```bash
git add .gitattributes ppocr-models/tiny-det.safetensors
git commit -m "chore(ppocr-models): bundle PP-OCRv6 tiny-det safetensors (LFS)"
```

Verify LFS tracked it (should show an LFS pointer, not the raw bytes):
```bash
git diff HEAD~1 --stat
git cat-file -p HEAD:ppocr-models/tiny-det.safetensors | head -3
```
Expected output includes `version https://git-lfs.github.com/spec/v1`.

---

## Task 2: Scaffold the `ppocr-engine` crate

**Files:**
- Create: `src-tauri/ppocr-engine/Cargo.toml`
- Create: `src-tauri/ppocr-engine/src/lib.rs` (minimal — modules added in later tasks)

- [ ] **Step 1: Create the crate's `Cargo.toml`**

Write `src-tauri/ppocr-engine/Cargo.toml`:

```toml
[package]
name = "ppocr-engine"
version = "0.1.0"
edition = "2021"
publish = false

[lib]
name = "ppocr_engine"
path = "src/lib.rs"

# Vendored PP-OCRv6 detector. Kept out of the host's [workspace] so
# `[profile.dev.package."*"] opt-level = 3` applies to it (same trick as
# kraken-engine — keeps dev-build detection fast).
[dependencies]
rayon = "1.11"
safetensors = "=0.7.0"
image = "0.25"
anyhow = "1.0"
log = "0.4"
serde = { version = "1.0", features = ["derive"] }
```

- [ ] **Step 2: Create a minimal `src/lib.rs` that compiles**

Write `src-tauri/ppocr-engine/src/lib.rs`:

```rust
//! ppocr-engine: vendored PP-OCRv6 tiny text detector (DBNet).
//!
//! Slimmed subset of ppocr-rs (https://github.com/weidix/ppocr-rs): detector +
//! preprocess + DB postprocess only. The recognizer, GPU backend, model
//! download, and CLI were excluded. The tiny-det safetensors is bundled by the
//! host via `include_bytes!`; this crate exposes `Detector::load_from_buffer`.
//!
//! Public API:
//!   - [`Detector`] — loaded detector, reused across calls.
//!   - [`Detector::load_from_buffer`] — load the bundled tiny-det weights.
//!   - [`Detector::detect`] — image → quads in source-image pixel coords.
```

- [ ] **Step 3: Verify the crate compiles standalone**

```bash
cd src-tauri/ppocr-engine
cargo build
cd ../..
```
Expected: builds with no errors (warnings OK at this stage — empty crate).

- [ ] **Step 4: Commit**

```bash
git add src-tauri/ppocr-engine/Cargo.toml src-tauri/ppocr-engine/src/lib.rs
git commit -m "feat(ppocr-engine): scaffold slimmed ppocr-rs detector crate"
```

---

## Task 3: Vendor the tensor / arena / ops / backend primitives

These four files are verbatim copies. They have no upstream edits needed beyond the module path (the upstream uses `use crate::models::ModelSize` in `cpu/mod.rs:13` — but since `model.rs` lives in our crate root, internal `use super::` paths change).

**Files:**
- Create: `src-tauri/ppocr-engine/src/tensor.rs` ← `clones/ppocr-rs/src/cpu/tensor.rs`
- Create: `src-tauri/ppocr-engine/src/arena.rs` ← `clones/ppocr-rs/src/cpu/arena.rs`
- Create: `src-tauri/ppocr-engine/src/ops.rs` ← `clones/ppocr-rs/src/cpu/ops.rs`
- Create: `src-tauri/ppocr-engine/src/backend.rs` ← `clones/ppocr-rs/src/cpu/backend.rs`

- [ ] **Step 1: Copy the four primitive files verbatim**

```bash
cp ../clones/ppocr-rs/src/cpu/tensor.rs src-tauri/ppocr-engine/src/tensor.rs
cp ../clones/ppocr-rs/src/cpu/arena.rs  src-tauri/ppocr-engine/src/arena.rs
cp ../clones/ppocr-rs/src/cpu/ops.rs    src-tauri/ppocr-engine/src/ops.rs
cp ../clones/ppocr-rs/src/cpu/backend.rs src-tauri/ppocr-engine/src/backend.rs
```

(Paths relative to repo root. Adjust `../clones/ppocr-rs` if your clone lives elsewhere — it's at `/Users/pndaza/Projects/playground/clones/ppocr-rs`.)

- [ ] **Step 2: Fix cross-module imports inside the copies**

In each copied file, `use super::{...}` paths that referenced `crate::models::...` or `crate::ocr::...` must be redirected. Run these greps to find them:

```bash
cd src-tauri/ppocr-engine/src
grep -n "use crate::" tensor.rs arena.rs ops.rs backend.rs
grep -n "use super::" tensor.rs arena.rs ops.rs backend.rs
```

Expected: most `use super::` references resolve to items in sibling files within the same crate (which is what we want). `use crate::models::ModelSize` or `use crate::ocr::...` will be the ones to fix — but at this stage tensor/arena/ops/backend should NOT reference `models` or `ocr` (verify by grep). If they do, comment that import out for now and re-add the dependency in the task that needs it (Task 5 for `ModelSize`, Task 6 for `ocr::Point` etc.).

- [ ] **Step 3: Wire the modules in `lib.rs`**

Append to `src-tauri/ppocr-engine/src/lib.rs` (after the doc comment):

```rust
mod arena;
mod backend;
mod ops;
mod tensor;
```

- [ ] **Step 4: Verify it compiles**

```bash
cd src-tauri/ppocr-engine
cargo build 2>&1 | head -50
cd ../..
```
Expected: compiles, OR surfaces unresolved-import errors that tell you exactly which `use crate::...` to fix in Step 2. Fix those, rebuild, repeat until clean.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/ppocr-engine/src
git commit -m "feat(ppocr-engine): vendor tensor/arena/ops/backend primitives"
```

---

## Task 4: Vendor the SIMD kernels

Verbatim copy of the kernels module. Both neon (arm64 macOS) and x86 (x86_64 macOS, Linux, Windows) paths — release CI builds both.

**Files:**
- Create: `src-tauri/ppocr-engine/src/kernels/mod.rs` ← `clones/ppocr-rs/src/cpu/kernels/mod.rs`
- Create: `src-tauri/ppocr-engine/src/kernels/neon.rs` ← `clones/ppocr-rs/src/cpu/kernels/neon.rs`
- Create: `src-tauri/ppocr-engine/src/kernels/x86.rs` ← `clones/ppocr-rs/src/cpu/kernels/x86.rs`

- [ ] **Step 1: Create the kernels directory and copy the three files**

```bash
mkdir -p src-tauri/ppocr-engine/src/kernels
cp ../clones/ppocr-rs/src/cpu/kernels/mod.rs  src-tauri/ppocr-engine/src/kernels/mod.rs
cp ../clones/ppocr-rs/src/cpu/kernels/neon.rs src-tauri/ppocr-engine/src/kernels/neon.rs
cp ../clones/ppocr-rs/src/cpu/kernels/x86.rs  src-tauri/ppocr-engine/src/kernels/x86.rs
```

(Do NOT copy `accelerate.rs` — kraken-engine links Accelerate via candle; ppocr-rs's standalone kernels do not use it.)

- [ ] **Step 2: Verify cross-module imports resolve**

```bash
cd src-tauri/ppocr-engine/src
grep -n "use crate::" kernels/*.rs
grep -n "use super::" kernels/*.rs
```
Expected: only intra-kernel `use super::*` or `use crate::tensor::...`-style refs. If any reference `crate::models` or `crate::ocr`, defer fixing to Task 5/6 (these kernels shouldn't, but verify).

- [ ] **Step 3: Wire the kernels module in `lib.rs`**

Append to `src-tauri/ppocr-engine/src/lib.rs`:

```rust
mod kernels;
```

- [ ] **Step 4: Verify it compiles**

```bash
cd src-tauri/ppocr-engine
cargo build 2>&1 | head -50
cd ../..
```
Expected: compiles. On arm64 macOS the neon path compiles; x86_64 path is cfg-gated and skipped.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/ppocr-engine/src/kernels src-tauri/ppocr-engine/src/lib.rs
git commit -m "feat(ppocr-engine): vendor SIMD kernels (neon + x86)"
```

---

## Task 5: Vendor the weights loader + add `from_bytes`

**Files:**
- Create: `src-tauri/ppocr-engine/src/weights.rs` ← `clones/ppocr-rs/src/cpu/weights.rs` + new method

- [ ] **Step 1: Copy weights.rs verbatim**

```bash
cp ../clones/ppocr-rs/src/cpu/weights.rs src-tauri/ppocr-engine/src/weights.rs
```

- [ ] **Step 2: Add a `from_bytes` constructor**

In `src-tauri/ppocr-engine/src/weights.rs`, the existing `Weights::load(path)` reads bytes from disk then calls `SafeTensors::deserialize`. Factor the deserialization into a shared helper and add `from_bytes`. Replace the body of `impl Weights { pub(crate) fn load(path) ... }` so the file reads (showing the full `impl` block — replace lines 13–55 of the copy):

```rust
impl Weights {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("read model {}", path.display()))?;
        Self::from_bytes(&bytes)
            .with_context(|| format!("decode safetensors model {}", path.display()))
    }

    /// Deserialize weights directly from an in-memory safetensors buffer.
    /// Used by the host's `include_bytes!`-bundled loader.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let archive = SafeTensors::deserialize(bytes)?;
        let mut tensors = HashMap::with_capacity(archive.len());
        for (name, view) in archive.iter() {
            ensure!(
                view.dtype() == Dtype::F32,
                "tensor {name:?} uses {:?}; only F32 safetensors are supported",
                view.dtype()
            );
            let expected = element_count(view.shape()).context("safetensors shape overflow")?;
            let expected_bytes = expected
                .checked_mul(size_of::<f32>())
                .context("safetensors byte length overflow")?;
            ensure!(
                view.data().len() == expected_bytes,
                "tensor {name:?} has an invalid byte length"
            );
            let mut values = Vec::with_capacity(expected);
            for bytes in view.data().chunks_exact(size_of::<f32>()) {
                values.push(f32::from_le_bytes(
                    bytes.try_into().expect("f32 chunk has four bytes"),
                ));
            }
            tensors.insert(
                name.to_owned(),
                Tensor::from_f32(view.shape().to_vec(), values)
                    .with_context(|| format!("decode tensor {name:?}"))?,
            );
        }
        Ok(Self { tensors })
    }

    pub(crate) fn builder(&self) -> VarBuilder<'_> {
        VarBuilder {
            weights: self,
            prefix: String::new(),
        }
    }
}
```

(The `use anyhow::Context` import at the top of the file already covers `.with_context`.)

- [ ] **Step 3: Wire the module in `lib.rs`**

Append to `src-tauri/ppocr-engine/src/lib.rs`:

```rust
mod weights;
```

- [ ] **Step 4: Verify it compiles**

```bash
cd src-tauri/ppocr-engine
cargo build 2>&1 | head -40
cd ../..
```
Expected: compiles. (`Tensor` and `element_count` come from sibling `tensor.rs` via the existing `use super::{...}` at the top of the copied file.)

- [ ] **Step 5: Commit**

```bash
git add src-tauri/ppocr-engine/src/weights.rs src-tauri/ppocr-engine/src/lib.rs
git commit -m "feat(ppocr-engine): vendor weights loader + add from_bytes for bundled bytes"
```

---

## Task 6: Vendor the detector model (slimmed)

Copy `clones/ppocr-rs/src/cpu/model.rs` lines 1–1685 only (detector + shared helpers), drop the recognizer (lines 1686+) and the tests block (2125+).

**Files:**
- Create: `src-tauri/ppocr-engine/src/model.rs` ← `clones/ppocr-rs/src/cpu/model.rs` lines 1–1685

- [ ] **Step 1: Copy lines 1–1685 of the upstream model.rs**

```bash
sed -n '1,1685p' ../clones/ppocr-rs/src/cpu/model.rs > src-tauri/ppocr-engine/src/model.rs
```

- [ ] **Step 2: Cut the ModelSize enum dependency**

The vendored `Detector::load_from_buffer` pins `ModelSize::Tiny`, so we no longer need the `ModelSize` enum (it lives in `crate::models::ModelSize` upstream, which we don't vendor). Remove these items from the top of `src-tauri/ppocr-engine/src/model.rs`:

1. Any `use crate::models::ModelSize;` import.
2. The `match size { Medium => ..., Small => ..., Tiny => ... }` inside `Detector::load` — replace with just the `Tiny` arm's body (inlined).

After the edit, `Detector::load`'s body should construct only the Tiny topology:

```rust
pub fn load(path: impl AsRef<Path>, options: CpuOptions) -> Result<Self> {
    let pool = thread_pool(options)?;
    let weights = Weights::load(path)?;
    let vb = weights.builder();
    let encoder = vb.pp("model").pp("backbone").pp("encoder");
    let backbone = LcNetBackbone::load(
        encoder,
        &detector_stages_for_channels([32, 48, 64, 160]),
        StemSpec::Large {
            mid_channels: 16,
            out_channels: 32,
        },
        Activation::Relu,
    )?;
    let neck = DetectorNeckKind::RepLkFpn(RepLkFpn::load(
        vb.pp("model").pp("neck"),
        [32, 48, 64, 160],
        64,
        5,
    )?);
    let head = DetectorHead::load(vb.pp("head"), 64)?;
    Ok(Self {
        backbone,
        neck,
        head,
        pool,
        arena: InferenceArena::default(),
    })
}
```

- [ ] **Step 3: Add `load_from_buffer` to the impl block**

Add this method to `impl Detector { ... }` in `src-tauri/ppocr-engine/src/model.rs` (next to the existing `load`):

```rust
    /// Load the bundled tiny-det weights from an in-memory safetensors buffer.
    /// Used by the host's `include_bytes!` path — no model files on disk.
    /// Uses all available CPUs (rayon default thread count).
    pub fn load_from_buffer(bytes: &[u8]) -> Result<Self> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::load_from_buffer_with_threads(bytes, threads)
    }

    /// Same as [`load_from_buffer`](Self::load_from_buffer) but with an explicit
    /// worker count. `threads` must be ≥ 1 (the upstream `thread_pool` asserts
    /// this).
    pub fn load_from_buffer_with_threads(bytes: &[u8], threads: usize) -> Result<Self> {
        let pool = thread_pool(CpuOptions { threads })?;
        let weights = Weights::from_bytes(bytes)?;
        let vb = weights.builder();
        let encoder = vb.pp("model").pp("backbone").pp("encoder");
        let backbone = LcNetBackbone::load(
            encoder,
            &detector_stages_for_channels([32, 48, 64, 160]),
            StemSpec::Large {
                mid_channels: 16,
                out_channels: 32,
            },
            Activation::Relu,
        )?;
        let neck = DetectorNeckKind::RepLkFpn(RepLkFpn::load(
            vb.pp("model").pp("neck"),
            [32, 48, 64, 160],
            64,
            5,
        )?);
        let head = DetectorHead::load(vb.pp("head"), 64)?;
        Ok(Self {
            backbone,
            neck,
            head,
            pool,
            arena: InferenceArena::default(),
        })
    }
```

**Critical (verified by reading upstream `thread_pool` at model.rs:29):** `thread_pool` asserts `options.threads > 0` and panics otherwise. Do NOT pass `threads: 0` — use `available_parallelism()` (rayon default) instead.

- [ ] **Step 4: Fix imports**

```bash
cd src-tauri/ppocr-engine/src
grep -n "use crate::" model.rs
grep -n "use super::" model.rs
```
Expected: the file uses `use super::{...}` to pull `Tensor`, `Weights`, etc. from sibling modules. Remove any `use crate::models::...` line. If `thread_pool`, `CpuOptions`, `InferenceArena`, `LcNetBackbone`, `StemSpec`, `Activation`, `detector_stages_for_channels`, `DetectorNeckKind`, `RepLkFpn`, `DetectorHead` are not all defined in `model.rs` itself, the grep will tell you which are external — they should all be local to `model.rs` per upstream's structure (verified: lines 15–1685 contain all of these).

- [ ] **Step 5: Wire the module in `lib.rs` and re-export `Detector`**

Append to `src-tauri/ppocr-engine/src/lib.rs`:

```rust
mod model;

pub use model::{CpuOptions, Detector};
pub use tensor::Tensor;
```

- [ ] **Step 6: Verify it compiles**

```bash
cd src-tauri/ppocr-engine
cargo build 2>&1 | head -60
cd ../..
```
Expected: compiles. If errors appear, they'll be unresolved-import errors — fix the specific `use` lines flagged. The most likely: an `Activation::Relu` variant not in scope if upstream re-exported it from elsewhere; if so, add `use super::backend::Activation` (or wherever upstream defined it — check `grep -n "enum Activation" *.rs`).

- [ ] **Step 7: Commit**

```bash
git add src-tauri/ppocr-engine/src/model.rs src-tauri/ppocr-engine/src/lib.rs
git commit -m "feat(ppocr-engine): vendor detector model (tiny-only) + load_from_buffer"
```

---

## Task 7: Vendor the postprocess chain (DetectorTransform, plan, extract_detections)

Copy the detector-postprocess subset of `clones/ppocr-rs/src/ocr.rs`. This is the messiest task — many interlinked helpers — so work in small verbatim chunks.

**Files:**
- Create: `src-tauri/ppocr-engine/src/postprocess.rs`

- [ ] **Step 1: Create the postprocess.rs file with the constants + transform + plan**

Write `src-tauri/ppocr-engine/src/postprocess.rs` (this is the full content — assemble by reading the corresponding line ranges of `clones/ppocr-rs/src/ocr.rs`):

```rust
//! Detector postprocess: PP-OCRv6 DB (differentiable binarization) box
//! extraction. Vendored from ppocr-rs src/ocr.rs (detector subset only).
//!
//! Pipeline: probability map → threshold → connected components →
//! rotated-box fit (PCA) → PaddleOCR DB unclip → score/area gates.

use anyhow::{Context, Result, bail, ensure};

/// Detector input is resized so the longest side is at most this many pixels
/// (unless the image is already smaller). Matches PaddleOCR's defaults.
const DETECTOR_LIMIT_SIDE: f64 = 736.0;
const DEFAULT_DETECTOR_MAX_SIDE: u32 = 736;
/// Hard cap on the detector input's longest side (huge images get scaled down).
const DETECTOR_MAX_SIDE: f64 = 4_000.0;

/// Maps detector-input coordinates back to source-image coordinates. Built by
/// `DetectorInputPlan` and consumed by `extract_detections`.
#[derive(Clone, Copy, Debug)]
pub struct DetectorTransform {
    source_width: u32,
    source_height: u32,
    content_width: u32,
    content_height: u32,
}

impl DetectorTransform {
    pub fn new(
        source_width: u32,
        source_height: u32,
        content_width: u32,
        content_height: u32,
    ) -> Result<Self> {
        if source_width == 0 || source_height == 0 || content_width == 0 || content_height == 0 {
            bail!("detector transform dimensions must be non-zero");
        }
        Ok(Self {
            source_width,
            source_height,
            content_width,
            content_height,
        })
    }

    pub fn content_width(self) -> u32 { self.content_width }
    pub fn content_height(self) -> u32 { self.content_height }

    pub fn map_x_to_source(self, x: f32) -> f32 {
        (x * self.source_width as f32 / self.content_width as f32)
            .clamp(0.0, self.source_width as f32)
    }
    pub fn map_y_to_source(self, y: f32) -> f32 {
        (y * self.source_height as f32 / self.content_height as f32)
            .clamp(0.0, self.source_height as f32)
    }
}

/// Detector input geometry: resized input dims + the source↔input transform.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DetectorInputPlan {
    input_width: usize,
    input_height: usize,
    transform: DetectorTransform,
}

impl DetectorInputPlan {
    pub(crate) fn new(source_width: u32, source_height: u32, max_side: Option<u32>) -> Result<Self> {
        let ratio = match max_side {
            Some(limit) if limit > 0 => {
                (f64::from(limit) / f64::from(source_width.max(source_height))).min(1.0)
            }
            Some(_) => bail!("detector maximum side must be positive"),
            None => default_detector_ratio(source_width, source_height),
        };
        let input_width = aligned_dimension(f64::from(source_width) * ratio)?;
        let input_height = aligned_dimension(f64::from(source_height) * ratio)?;
        Ok(Self {
            input_width: input_width as usize,
            input_height: input_height as usize,
            transform: DetectorTransform::new(source_width, source_height, input_width, input_height)?,
        })
    }

    pub(crate) const fn input_width(self) -> usize { self.input_width }
    pub(crate) const fn input_height(self) -> usize { self.input_height }
    pub(crate) const fn transform(self) -> DetectorTransform { self.transform }

    pub(crate) fn corners(self) -> [Point; 4] {
        [
            Point(0.0, 0.0),
            Point(self.transform.source_width as f32, 0.0),
            Point(self.transform.source_width as f32, self.transform.source_height as f32),
            Point(0.0, self.transform.source_height as f32),
        ]
    }
}

fn default_detector_ratio(width: u32, height: u32) -> f64 {
    let min_side = f64::from(width.min(height));
    let mut ratio = if min_side < DETECTOR_LIMIT_SIDE {
        DETECTOR_LIMIT_SIDE / min_side
    } else {
        1.0
    };
    if f64::from(width.max(height)) * ratio > DETECTOR_MAX_SIDE {
        ratio = DETECTOR_MAX_SIDE / f64::from(width.max(height));
    }
    ratio
}

fn aligned_dimension(value: f64) -> Result<u32> {
    if !value.is_finite() || value <= 0.0 {
        bail!("invalid resized image dimension {value}");
    }
    let units = (value / 32.0).round().max(1.0);
    if units > f64::from(u32::MAX / 32) {
        bail!("resized image dimension {value} is too large");
    }
    Ok(units as u32 * 32)
}

// === vendored verbatim from clones/ppocr-rs/src/ocr.rs lines 222–268, 702–722,
//     723–791, 967–999, 1000–1055, 1056–1129, 1130–1154, 1168–1188, 1208–1229 ===

// >>> PASTE HERE: lines 222–268 (Point, Detection, DetectorPostprocessOptions + impl validate)
// >>> PASTE HERE: lines 723–791 (extract_detections)
// >>> PASTE HERE: lines 967–999 (detector_output_shape, validate_probability)
// >>> PASTE HERE: lines 1000–1055 (Component + collect_component)
// >>> PASTE HERE: lines 1056–1129 (fit_rotated_box)
// >>> PASTE HERE: lines 1130–1154 (sort_detections, polygon_center, polygon_aspect_ratio)
// >>> PASTE HERE: lines 1168–1188 (row_probability, argmax)
// >>> PASTE HERE: lines 1208–1229 (dot, add, scale, point_coordinates, distance Point helpers)
```

The `>>> PASTE HERE` markers indicate exactly which line ranges to copy verbatim from `clones/ppocr-rs/src/ocr.rs`. Read those ranges and paste the contents where each marker is, removing the marker comment.

- [ ] **Step 2: Copy each marked range from upstream**

For each `>>> PASTE HERE` marker, open `clones/ppocr-rs/src/ocr.rs`, copy the indicated line range verbatim, and paste it where the marker is (then delete the marker line). The ranges are non-overlapping and cover the complete detector-postprocess chain.

Specifically, use your editor to:
1. Read lines 222–268 of upstream `ocr.rs`; replace the first marker with those lines.
2. Read lines 723–791; replace the second marker.
3. Continue for each remaining marker (967–999, 1000–1055, 1056–1129, 1130–1154, 1168–1188, 1208–1229).

After all markers are replaced, the file should have no `>>> PASTE HERE` left:

```bash
grep -c ">>> PASTE HERE" src-tauri/ppocr-engine/src/postprocess.rs
```
Expected: `0`.

- [ ] **Step 3: Wire the module in `lib.rs` and re-export types**

Append to `src-tauri/ppocr-engine/src/lib.rs`:

```rust
mod postprocess;

pub use postprocess::{DetectorTransform, Detection, Point};
```

- [ ] **Step 4: Verify it compiles**

```bash
cd src-tauri/ppocr-engine
cargo build 2>&1 | head -60
cd ../..
```
Expected: compiles. Likely issues: some helpers reference `safeetensors` or `ndarray` — they don't (verified: these are pure-Rust f64/f32 helpers). If `ensure!` or `bail!` are flagged, the `use anyhow::{...}` import at the top covers them.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/ppocr-engine/src/postprocess.rs src-tauri/ppocr-engine/src/lib.rs
git commit -m "feat(ppocr-engine): vendor detector postprocess (DetectorTransform + extract_detections + DB helpers)"
```

---

## Task 8: Vendor the preprocess pipeline

The detector preprocess resizes the image, normalizes to BGR mean/std, and produces the `[1,3,H,W]` NCHW tensor the model wants. Copy `clones/ppocr-rs/src/preprocess.rs` + `preprocess/kernels.rs`, trimming recognizer paths.

**Files:**
- Create: `src-tauri/ppocr-engine/src/preprocess.rs` ← `clones/ppocr-rs/src/preprocess.rs` (detector path only)
- Create: `src-tauri/ppocr-engine/src/preprocess/kernels.rs` ← `clones/ppocr-rs/src/preprocess/kernels.rs` (verbatim)

- [ ] **Step 1: Copy the kernels module verbatim**

```bash
mkdir -p src-tauri/ppocr-engine/src/preprocess
cp ../clones/ppocr-rs/src/preprocess/kernels.rs src-tauri/ppocr-engine/src/preprocess/kernels.rs
```

- [ ] **Step 2: Create a detector-only preprocess.rs**

Copy `clones/ppocr-rs/src/preprocess.rs` to `src-tauri/ppocr-engine/src/preprocess.rs` but cut the recognizer code. The detector subset consists of:
- `mod kernels;` (line 3)
- `DETECTOR_MEAN_BGR` / `DETECTOR_STD_BGR` constants (lines 16–18)
- `PreparedInput` struct + impl (lines 22–34)
- `prepare_detector` fn (lines 36–47)
- `normalized_bgr` fn (lines 78–123)
- The detector test (lines 150–158) — keep as a sanity test.

Cut: `prepare_recognizer` (49–76), `RECOGNIZER_MEAN_BGR`/`RECOGNIZER_STD_BGR` (19–20), the recognizer test (160–179), the `RecognitionInputPlan` import (line 9), and the test helper `recognition_input_plan` import in the test mod (line 146).

The resulting file should look like this (paste verbatim, with recognizer parts deleted):

```rust
//! CPU execution of the detector preprocessing plan: resize + BGR +
//! mean/std normalize, producing an NCHW f32 tensor.

mod kernels;

use crate::postprocess::{DetectorInputPlan, Point};
use crate::tensor::Tensor;
use rayon::prelude::*;

use kernels::{Kernel, Normalization, RowPlan};

const DETECTOR_MEAN_BGR: [f32; 3] = [0.485, 0.456, 0.406];
const DETECTOR_STD_BGR: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Debug)]
pub(crate) struct PreparedInput {
    pub(crate) data: Vec<f32>,
    pub(crate) batch: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

impl PreparedInput {
    pub(crate) fn shape(&self) -> [usize; 4] {
        [self.batch, 3, self.height, self.width]
    }
}

pub(crate) fn prepare_detector(image: &crate::RgbImage, plan: DetectorInputPlan) -> PreparedInput {
    normalized_bgr(
        image,
        plan.corners(),
        plan.input_width(),
        plan.input_height(),
        plan.input_width(),
        &DETECTOR_MEAN_BGR,
        &DETECTOR_STD_BGR,
    )
}

fn normalized_bgr(
    image: &crate::RgbImage,
    corners: [Point; 4],
    canvas_width: usize,
    canvas_height: usize,
    content_width: usize,
    mean: &[f32; 3],
    standard_deviation: &[f32; 3],
) -> PreparedInput {
    let plane_len = canvas_height * canvas_width;
    let mut data = vec![0.0; 3 * plane_len];
    let (blue, green_red) = data.split_at_mut(plane_len);
    let (green, red) = green_red.split_at_mut(plane_len);
    let kernel = Kernel::detect();
    let normalization = Normalization::new(*mean, *standard_deviation);
    let corners = corners.map(|point| [point.0, point.1]);

    blue.par_chunks_mut(canvas_width)
        .zip(green.par_chunks_mut(canvas_width))
        .zip(red.par_chunks_mut(canvas_width))
        .enumerate()
        .for_each(|(y, ((blue, green), red))| {
            kernel.preprocess_row(
                image.pixels(),
                RowPlan {
                    source_width: image.width() as usize,
                    source_height: image.height() as usize,
                    corners,
                    destination_y: y,
                    destination_height: canvas_height,
                    content_width,
                    normalization,
                },
                blue,
                green,
                red,
            );
        });

    PreparedInput {
        data,
        batch: 1,
        height: canvas_height,
        width: canvas_width,
    }
}
```

NOTE: `DetectorInputPlan::new` upstream takes `&RgbImage` but we only need width/height for the ratio. The signature above takes `source_width, source_height` directly — if the vendored `DetectorInputPlan::new` signature in Task 7 differs, reconcile them. The body of `corners()` upstream references `self.transform.source_width`; we made `transform` private to `DetectorInputPlan`, so `corners()` is defined inside the impl (it is — Task 7 Step 1 includes `corners()` in the impl block).

- [ ] **Step 3: Make `DetectorInputPlan::new` take width/height, not `&RgbImage`**

If Task 7's `DetectorInputPlan::new` was written to take `(source_width: u32, source_height: u32, max_side: Option<u32>)`, it's already correct. If you wrote it to take `&RgbImage`, change it now to take width/height (the `RgbImage` reference is unnecessary — only dims are read).

- [ ] **Step 4: Wire the module in `lib.rs`**

Append to `src-tauri/ppocr-engine/src/lib.rs`:

```rust
mod preprocess;
```

- [ ] **Step 5: Add a minimal `RgbImage` shim**

The vendored crate needs an `RgbImage` type that exposes `.pixels() -> &[u8]`, `.width() -> u32`, `.height() -> u32`. The host passes an `image::DynamicImage`. Add this to `src-tauri/ppocr-engine/src/lib.rs` (a thin newtype wrapping the host's image):

```rust
use anyhow::{Context, Result, ensure};

/// Interleaved row-major RGB8 image view, built from the host's `DynamicImage`.
/// Confines the `image` crate to this type so the preprocess kernels operate
/// on a plain byte slice.
pub struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl RgbImage {
    /// Build from the host's `image::DynamicImage` (converts to RGB8).
    pub fn from_dynamic(img: &image::DynamicImage) -> Self {
        let rgb = img.to_rgb8();
        Self {
            width: rgb.width(),
            height: rgb.height(),
            pixels: rgb.into_raw(),
        }
    }

    pub const fn width(&self) -> u32 { self.width }
    pub const fn height(&self) -> u32 { self.height }
    pub fn pixels(&self) -> &[u8] { &self.pixels }
}
```

(If `ensure!` / `Context` turn out unused after this edit, drop the `use anyhow::...` line — keep it minimal.)

- [ ] **Step 6: Verify it compiles**

```bash
cd src-tauri/ppocr-engine
cargo build 2>&1 | head -60
cd ../..
```
Expected: compiles. Fix any path mismatches between `preprocess.rs`'s imports and the actual module layout.

- [ ] **Step 7: Commit**

```bash
git add src-tauri/ppocr-engine/src
git commit -m "feat(ppocr-engine): vendor detector preprocess + RgbImage shim"
```

---

## Task 9: Add the high-level `Detector::detect` API

The vendored `Detector` exposes `forward(Tensor)`; we add a higher-level `detect(&DynamicImage) -> Vec<Detection>` that does preprocess → forward → postprocess.

**Files:**
- Modify: `src-tauri/ppocr-engine/src/lib.rs`

- [ ] **Step 1: Add a `detect` method that takes a `DynamicImage`**

In `src-tauri/ppocr-engine/src/lib.rs`, add an `impl Detector` block below the `RgbImage` definition:

```rust
use crate::model::Detector;
use crate::postprocess::{DetectorInputPlan, DetectorPostprocessOptions, Detection};
use crate::preprocess::prepare_detector;
use crate::tensor::Tensor;

impl Detector {
    /// Run end-to-end detection: image → quads in source-image pixel coords.
    ///
    /// Resizes the input so its longest side is ≤ 736 (PaddleOCR default),
    /// aligned to 32-pixel multiples. Returns one `Detection` per text region
    /// (4-corner quad + score), with coords already mapped back to the source
    /// image via the transform baked into the input plan.
    pub fn detect(&self, img: &image::DynamicImage) -> Result<Vec<Detection>> {
        let rgb = RgbImage::from_dynamic(img);
        let plan = DetectorInputPlan::new(rgb.width(), rgb.height(), Some(736))?;
        let prepared = prepare_detector(&rgb, plan);
        let input = Tensor::from_f32(prepared.shape().to_vec(), prepared.data)?;
        let output = self.forward(input)?;
        // The output is [1, 1, H, W] — same H, W as the input (DB head preserves
        // spatial dims). extract_detections reads `values[y * width + x]`.
        let values: &[f32] = output.as_f32()?;
        let shape: &[usize] = output.shape();
        let opts = DetectorPostprocessOptions::default();
        crate::postprocess::extract_detections(values, shape, plan.transform(), opts)
    }
}
```

**Critical correctness note (verified by reading upstream `extract_detections`):** the `shape` passed must be the *output* tensor's shape (not the input's) — `extract_detections` computes `(height, width) = detector_output_shape(shape, values.len())` and reads `values[y * width + x]` against that width. The DB head preserves spatial dims so input H,W == output H,W, but always pass `output.shape()` to be safe.

**Type signature verification (run before finalizing this step):**
```bash
grep -n "pub fn from_f32\|pub fn as_f32\|pub fn shape\|pub fn dims4" src-tauri/ppocr-engine/src/tensor.rs
grep -n "pub fn extract_detections" -A 6 src-tauri/ppocr-engine/src/postprocess.rs
```
The vendored `Tensor::as_f32` returns `Result<&[f32], _>` (per upstream tensor.rs:117) and `Tensor::shape` returns `&[usize]` (or `Vec<usize>` — check and add `.as_slice()` if needed). If `as_f32` returns `Result<&Vec<f32>, _>` instead, change `let values: &[f32] = ...` to `let values: &[f32] = output.as_f32()?.as_slice();`. Reconcile the actual signatures before committing.

- [ ] **Step 2: Make the postprocess helpers `pub(crate)`-visible**

`DetectorInputPlan`, `prepare_detector`, `extract_detections` must be at least `pub(crate)` so `lib.rs` can reach them. Verify:
```bash
grep -n "pub(crate) fn prepare_detector\|pub(crate) struct DetectorInputPlan\|pub fn extract_detections" src-tauri/ppocr-engine/src/*.rs
```
Expected: `prepare_detector` is `pub(crate)`, `DetectorInputPlan` is `pub(crate)`, `extract_detections` is `pub`. If `DetectorInputPlan::new` isn't visible at `pub(crate)` level, bump its visibility.

- [ ] **Step 3: Verify it compiles**

```bash
cd src-tauri/ppocr-engine
cargo build 2>&1 | head -60
cd ../..
```
Expected: compiles.

- [ ] **Step 4: Commit**

```bash
git add src-tauri/ppocr-engine/src/lib.rs
git commit -m "feat(ppocr-engine): add Detector::detect (DynamicImage → Vec<Detection>)"
```

---

## Task 10: Add unit tests for the vendored crate

**Files:**
- Modify: `src-tauri/ppocr-engine/src/postprocess.rs` (append `#[cfg(test)] mod tests`)
- Modify: `src-tauri/ppocr-engine/src/lib.rs` (add bundled-bytes load test)

- [ ] **Step 1: Write a failing test for `DetectorInputPlan` + `DetectorTransform`**

Append to `src-tauri/ppocr-engine/src/postprocess.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_aligns_to_32_and_maps_back() {
        // 1920x1080 source → resized with max_side=736.
        let plan = DetectorInputPlan::new(1920, 1080, Some(736)).expect("plan");
        // 1920*736/1920 = 736 (longest), 1080*736/1920 = 414 → round to 416 (32-multiple).
        assert_eq!(plan.input_width(), 736);
        assert_eq!(plan.input_height(), 416);
        let t = plan.transform();
        assert!((t.map_x_to_source(736.0) - 1920.0).abs() < 1e-3);
        assert!((t.map_y_to_source(416.0) - 1080.0).abs() < 1e-3);
    }

    #[test]
    fn plan_rejects_zero_dimensions() {
        assert!(DetectorInputPlan::new(0, 100, Some(736)).is_err());
        assert!(DetectorInputPlan::new(100, 0, Some(736)).is_err());
    }
}
```

- [ ] **Step 2: Run the tests — expect them to pass (they exercise vendored code)**

```bash
cd src-tauri/ppocr-engine
cargo test postprocess -- --nocapture
cd ../..
```
Expected: 2 tests pass. If they fail, the vendored `aligned_dimension` rounding or `default_detector_ratio` math was altered in vendoring — diff against upstream.

- [ ] **Step 3: Write a test that loads the bundled bytes**

The vendored crate doesn't know about the host's `include_bytes!`. For unit testing, we load the tiny-det file from disk relative to the crate. Add to `src-tauri/ppocr-engine/src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Loads the bundled tiny-det from the repo-root ppocr-models/ dir.
    /// Verifies the safetensors deserializes and the detector builds.
    #[test]
    fn load_from_buffer_builds_detector() {
        let bytes = std::fs::read("../../../ppocr-models/tiny-det.safetensors")
            .expect("read bundled tiny-det (run from repo root)");
        assert!(bytes.len() > 1_000_000, "tiny-det too small: {}", bytes.len());
        let det = Detector::load_from_buffer(&bytes)
            .expect("tiny-det loads from buffer");
        let _ = det; // constructed successfully
    }
}
```

(Path `../../../ppocr-models/...` is relative to `src-tauri/ppocr-engine/` — three levels up reaches repo root. `cargo test` runs from the crate dir.)

- [ ] **Step 4: Run all crate tests**

```bash
cd src-tauri/ppocr-engine
cargo test -- --nocapture
cd ../..
```
Expected: 3 tests pass (2 postprocess + 1 load). If the load test fails on file-not-found, run from the right cwd or adjust the relative path.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/ppocr-engine/src/postprocess.rs src-tauri/ppocr-engine/src/lib.rs
git commit -m "test(ppocr-engine): add DetectorInputPlan + bundled-load unit tests"
```

---

## Task 11: Add `ppocr-engine` as a host path dependency

**Files:**
- Modify: `src-tauri/Cargo.toml` (line ~52, after `kraken-engine`)

- [ ] **Step 1: Add the path-dep**

Edit `src-tauri/Cargo.toml`. Find the `kraken-engine = { path = "kraken-engine" }` line (line 52) and add below it:

```toml
kraken-engine = { path = "kraken-engine" }
# PP-OCRv6 tiny detector (vendored from ppocr-rs, detector-only). Alternative
# Myanmar segmenter. Same out-of-workspace trick as kraken-engine.
ppocr-engine = { path = "ppocr-engine" }
```

- [ ] **Step 2: Verify the host crate still compiles**

```bash
cd src-tauri
cargo build 2>&1 | tail -20
cd ..
```
Expected: compiles (the new dep is unused so far, but it must resolve). First build will be slow — ppocr-engine + its deps compile.

- [ ] **Step 3: Commit**

```bash
git add src-tauri/Cargo.toml
git commit -m "build(src-tauri): add ppocr-engine path dependency"
```

---

## Task 12: Add the `Segmenter` trait + `DetectedLine` type

**Files:**
- Create: `src-tauri/src/segmentation.rs`
- Modify: `src-tauri/src/lib.rs` (declare the module)

- [ ] **Step 1: Write the failing test first**

Create `src-tauri/src/segmentation.rs` with the trait + type + a compile-only test:

```rust
//! Engine-agnostic segmentation abstraction. Both Kraken and PP-OCR segmenters
//! implement [`Segmenter`] so `run_myanmar` can hold either behind
//! `Arc<dyn Segmenter>` and call `segment()` uniformly.
//!
//! `DetectedLine` carries only the two fields the recognizer path consumes
//! (verified at engine.rs:208–234): a baseline polyline (Kraken recog dewarp)
//! and a closed boundary polygon (bbox, Tesseract crop, overlay, dewarp
//! fallback). It is deliberately distinct from `kraken_engine::BaselineLine`
//! so the host doesn't depend on Kraken's container type for the abstraction.

use image::DynamicImage;
use serde::Serialize;

/// One detected text line in source-image pixel coordinates.
#[derive(Debug, Clone, Serialize)]
pub struct DetectedLine {
    /// Midline polyline (left → right). Used by Kraken recog for dewarp.
    /// For PP-OCR, synthesized as the quad's vertical midline. May be empty
    /// if only the boundary matters (e.g. Tesseract recog only).
    pub baseline: Vec<(f64, f64)>,
    /// Closed boundary polygon (≥ 3 points). Used for bbox, Tesseract crop,
    /// overlay, and Kraken dewarp fallback. For PP-OCR: 4 corners + repeat-first.
    pub boundary: Vec<(f64, f64)>,
}

/// A text-line segmenter. Both vendored engines implement this so the host
/// dispatches uniformly.
pub trait Segmenter: Send + Sync {
    /// Segment the page image into detected text lines (source-image coords).
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String>;
    /// Human-readable name for logs (e.g. "kraken", "ppocr-tiny").
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detected_line_serializes_with_baseline_and_boundary() {
        let line = DetectedLine {
            baseline: vec![(1.0, 2.0), (3.0, 2.0)],
            boundary: vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)],
        };
        let json = serde_json::to_string(&line).expect("serialize");
        assert!(json.contains("\"baseline\""), "missing baseline in: {json}");
        assert!(json.contains("\"boundary\""), "missing boundary in: {json}");
    }
}
```

- [ ] **Step 2: Declare the module in `lib.rs`**

Find the `mod` declarations near the top of `src-tauri/src/lib.rs` and add:

```rust
mod segmentation;
```

(Place it alongside `mod engine;`, `mod tesseract_page;`, etc. Check the actual `mod` lines first via `grep -n "^mod " src-tauri/src/lib.rs`.)

- [ ] **Step 3: Run the test**

```bash
cd src-tauri
cargo test segmentation:: -- --nocapture
cd ..
```
Expected: 1 test passes.

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/segmentation.rs src-tauri/src/lib.rs
git commit -m "feat(host): add Segmenter trait + DetectedLine type"
```

---

## Task 13: Implement the `KrakenSegmenter` adapter

Wrap the existing kraken engine behind the `Segmenter` trait. This is a pure adapter — no behavior change.

**Files:**
- Create: `src-tauri/src/segmenter_adapters.rs`
- Modify: `src-tauri/src/lib.rs` (declare module)

- [ ] **Step 1: Write the adapter (Kraken only, for now)**

Create `src-tauri/src/segmenter_adapters.rs`:

```rust
//! Adapters that wrap each vendored engine behind the host's [`Segmenter`]
//! trait. Each adapter owns the type-shape conversion (engine-native line
//! type → [`DetectedLine`]) so the recognizer path stays uniform.

use crate::segmentation::{DetectedLine, Segmenter};
use image::DynamicImage;

/// Wraps a shared [`kraken_engine::Engine`] as a [`Segmenter`]. Kraken's
/// `BaselineLine` already carries both the baseline polyline and the boundary
/// polygon, so this is a 1:1 field copy.
pub struct KrakenSegmenter {
    engine: std::sync::Arc<kraken_engine::Engine>,
}

impl KrakenSegmenter {
    pub fn new(engine: std::sync::Arc<kraken_engine::Engine>) -> Self {
        Self { engine }
    }
    /// Borrow the underlying engine (for recog when seg=ppocr but recog=kraken).
    pub fn engine(&self) -> &kraken_engine::Engine { &self.engine }
}

impl Segmenter for KrakenSegmenter {
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String> {
        let lines = self.engine.segment(img).map_err(|e| e.to_string())?;
        Ok(lines
            .into_iter()
            .map(|l| DetectedLine {
                baseline: l.baseline,
                boundary: l.boundary,
            })
            .collect())
    }
    fn name(&self) -> &'static str { "kraken" }
}
```

- [ ] **Step 2: Declare the module**

In `src-tauri/src/lib.rs`, add (next to the `mod segmentation;` from Task 12):

```rust
mod segmenter_adapters;
```

- [ ] **Step 3: Verify it compiles**

```bash
cd src-tauri
cargo build 2>&1 | tail -20
cd ..
```
Expected: compiles.

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/segmenter_adapters.rs src-tauri/src/lib.rs
git commit -m "feat(host): add KrakenSegmenter adapter"
```

---

## Task 14: Implement the `PPOcrSegmenter` adapter + quad→line conversion

The substantive adapter: convert PP-OCR's 4-corner quads into `DetectedLine`s with a synthesized baseline + closed boundary.

**Files:**
- Modify: `src-tauri/src/segmenter_adapters.rs`

- [ ] **Step 1: Write the failing test for the quad→line helpers**

Append to `src-tauri/src/segmenter_adapters.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_polygon_repeats_first_point() {
        let quad = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let closed = close_polygon(&quad);
        assert_eq!(closed.len(), 5);
        assert_eq!(closed[0], closed[4]);
    }

    #[test]
    fn synth_midline_averages_top_and_bottom_edges() {
        // Axis-aligned rectangle: top edge y=0, bottom edge y=4.
        // Midline should be at y=2 along x=0..4.
        let quad = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let mid = synth_midline(&quad, 5);
        assert_eq!(mid.len(), 5);
        // First sample (u=0): midline at (0, 2).
        assert!((mid[0].0 - 0.0).abs() < 1e-6 && (mid[0].1 - 2.0).abs() < 1e-6);
        // Last sample (u=1): midline at (4, 2).
        assert!((mid[4].0 - 4.0).abs() < 1e-6 && (mid[4].1 - 2.0).abs() < 1e-6);
        // Middle sample (u=0.5): midline at (2, 2).
        assert!((mid[2].0 - 2.0).abs() < 1e-6 && (mid[2].1 - 2.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run the tests — expect compile failure (helpers undefined)**

```bash
cd src-tauri
cargo test segmenter_adapters:: -- --nocapture 2>&1 | tail -20
cd ..
```
Expected: FAIL — `close_polygon` and `synth_midline` not defined.

- [ ] **Step 3: Implement the helpers + the adapter**

Add to `src-tauri/src/segmenter_adapters.rs` (above the `#[cfg(test)] mod tests`):

```rust
use crate::segmentation::{DetectedLine, Segmenter};
use image::DynamicImage;
use ppocr_engine::{Detection, Point};

/// Close a polygon by repeating the first point at the end (if not already
/// closed). Matches Kraken's convention so `polygon_bbox` and point-in-polygon
/// behave identically across segmenters.
fn close_polygon(poly: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if poly.len() < 2 {
        return poly.to_vec();
    }
    let mut out = poly.to_vec();
    if out.first() != out.last() {
        out.push(out[0]);
    }
    out
}

/// Synthesize a baseline (midline) for a 4-corner quad by averaging the top
/// and bottom edges. Returns `n` samples along the text axis (left → right).
///
/// Assumes the quad is ordered counter-clockwise from the top-left corner:
///   `[top_left, top_right, bottom_right, bottom_left]` — the order PaddleOCR's
///   DB postprocess produces (verified in ppocr-rs `fit_rotated_box`).
/// If the quad is rotated, the midline tracks the rotation.
fn synth_midline(quad: &[(f64, f64); 4], n: usize) -> Vec<(f64, f64)> {
    let [tl, tr, br, bl] = [quad[0], quad[1], quad[2], quad[3]];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let u = if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
        // Top edge: tl → tr. Bottom edge: bl → br.
        let top_x = tl.0 + (tr.0 - tl.0) * u;
        let top_y = tl.1 + (tr.1 - tl.1) * u;
        let bot_x = bl.0 + (br.0 - bl.0) * u;
        let bot_y = bl.1 + (br.1 - bl.1) * u;
        out.push(((top_x + bot_x) / 2.0, (top_y + bot_y) / 2.0));
    }
    out
}

/// Wraps a shared [`ppocr_engine::Detector`] as a [`Segmenter`]. Converts each
/// PP-OCR detection quad into a [`DetectedLine`] (closed boundary + synthesized
/// baseline). The boundary feeds Tesseract recog + overlay; the baseline feeds
/// Kraken recog dewarp (with graceful fallback if dewarp rejects it).
pub struct PPOcrSegmenter {
    detector: std::sync::Arc<ppocr_engine::Detector>,
}

impl PPOcrSegmenter {
    pub fn new(detector: std::sync::Arc<ppocr_engine::Detector>) -> Self {
        Self { detector }
    }
}

impl Segmenter for PPOcrSegmenter {
    fn segment(&self, img: &DynamicImage) -> Result<Vec<DetectedLine>, String> {
        let detections = self.detector.detect(img).map_err(|e| e.to_string())?;
        log::info!("[ocr] ppocr detections: {}", detections.len());
        Ok(detections
            .into_iter()
            .filter_map(|d| detection_to_line(&d))
            .collect())
    }
    fn name(&self) -> &'static str { "ppocr-tiny" }
}

/// Convert a PP-OCR `Detection` (4-corner quad) to a `DetectedLine`. Returns
/// `None` if the quad is degenerate (wrong corner count).
fn detection_to_line(d: &Detection) -> Option<DetectedLine> {
    let quad: [(f64, f64); 4] = [
        (d.polygon[0].0 as f64, d.polygon[0].1 as f64),
        (d.polygon[1].0 as f64, d.polygon[1].1 as f64),
        (d.polygon[2].0 as f64, d.polygon[2].1 as f64),
        (d.polygon[3].0 as f64, d.polygon[3].1 as f64),
    ];
    let boundary = close_polygon(&quad);
    let baseline = synth_midline(&quad, 8);
    Some(DetectedLine { baseline, boundary })
}
```

NOTE on the `Detection` field shape: ppocr-rs's `Detection` has `polygon: [Point; 4]` where `Point(pub f32, pub f32)`. The conversion casts f32→f64 for the host's f64-based polygon code. Verify by `grep -n "pub struct Detection\|pub struct Point\|polygon" src-tauri/ppocr-engine/src/postprocess.rs` — if the vendored `Detection.polygon` isn't `[Point; 4]`, adjust `detection_to_line` to match.

- [ ] **Step 4: Run the tests**

```bash
cd src-tauri
cargo test segmenter_adapters:: -- --nocapture 2>&1 | tail -20
cd ..
```
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/segmenter_adapters.rs
git commit -m "feat(host): add PPOcrSegmenter adapter + quad→line conversion"
```

---

## Task 15: Add `segmenter` to `OcrOpts`

**Files:**
- Modify: `src-tauri/src/lib.rs` (`OcrOpts` struct, ~line 25–46)

- [ ] **Step 1: Read the current `OcrOpts` definition**

```bash
grep -n "pub struct OcrOpts" -A 12 src-tauri/src/lib.rs
```

- [ ] **Step 2: Add the `segmenter` field**

In `src-tauri/src/lib.rs`, in the `OcrOpts` struct (which has `#[serde(rename_all = "camelCase")]`), add as the last field:

```rust
    /// Segmenter choice for the Myanmar path: `"kraken"` (default) or `"ppocr"`.
    /// `None`/unrecognized → Kraken. Ignored for non-Myanmar (full-page Tesseract).
    #[serde(default)]
    pub segmenter: Option<String>,
```

- [ ] **Step 3: Verify it compiles**

```bash
cd src-tauri
cargo build 2>&1 | tail -10
cd ..
```
Expected: compiles. (`#[serde(default)]` makes the field optional for existing clients that don't send it.)

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/lib.rs
git commit -m "feat(host): add segmenter field to OcrOpts"
```

---

## Task 16: Refactor `engine.rs` — bundle bytes, two OnceCells, `resolve_segmenter`

The core refactor. Convert the existing `KRAKEN: OnceCell<Engine>` into `OnceCell<Arc<Engine>>`, add `PPOCR: OnceCell<Arc<Detector>>`, add `resolve_segmenter`, add PP-OCR bundled bytes + override resolver.

**Files:**
- Modify: `src-tauri/src/engine.rs` (multiple edits below)

- [ ] **Step 1: Add the PP-OCR bundled-bytes static**

In `src-tauri/src/engine.rs`, below the existing `BUNDLED_SEG`/`BUNDLED_REC` (line 67–68), add:

```rust
/// Bundled PP-OCRv6 tiny detector. Same `include_bytes!` pattern as the
/// Kraken models. Path is relative to `src-tauri/src/` (this file's dir).
static BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../../ppocr-models/tiny-det.safetensors");
```

- [ ] **Step 2: Convert `KRAKEN` to `OnceCell<Arc<...>>` and add `PPOCR`**

Replace the existing `static KRAKEN: OnceCell<kraken_engine::Engine>` (line 74) with:

```rust
/// Process-wide lazily-loaded kraken engine, wrapped in `Arc` so it can be
/// shared with `KrakenSegmenter` (which holds `Arc<Engine>` to satisfy the
/// `'static` requirement of `Arc<dyn Segmenter>`).
static KRAKEN: OnceCell<std::sync::Arc<kraken_engine::Engine>> = OnceCell::new();

/// Process-wide lazily-loaded PP-OCR detector, same Arc-wrapped shape.
static PPOCR: OnceCell<std::sync::Arc<ppocr_engine::Detector>> = OnceCell::new();
```

- [ ] **Step 3: Update `kraken_engine()` to wrap in `Arc`**

In `kraken_engine(app)` (line 84), change the `Ok(engine)` at line 103 to `Ok(std::sync::Arc::new(engine))`, and update the return type to `Result<&Arc<kraken_engine::Engine>, String>`. The body of `get_or_try_init` now returns `Arc<Engine>`; the function still returns a borrowed `&Arc<Engine>` tied to the OnceCell's lifetime — `&self.engine` in `KrakenSegmenter` and `arc.as_ref()` at recog call sites both work.

Edit the function signature and the final `Ok(...)`:

```rust
fn kraken_engine(app: &tauri::AppHandle) -> Result<&std::sync::Arc<kraken_engine::Engine>, String> {
    KRAKEN.get_or_try_init(|| {
        let t = Instant::now();
        let engine = match resolve_override_models(app) {
            Some((seg_path, rec_path)) => {
                log::info!(
                    "[kraken] using override models from {}",
                    seg_path.parent().unwrap_or(std::path::Path::new(".")).display()
                );
                kraken_engine::Engine::load(&seg_path, &rec_path)
                    .map_err(|e| format!("Kraken override load failed: {e}"))?
            }
            None => kraken_engine::Engine::load_from_buffers(BUNDLED_SEG, BUNDLED_REC)
                .map_err(|e| format!("Kraken bundled load failed: {e}"))?,
        };
        log::info!(
            "[kraken] models loaded in {:.0} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
        Ok(std::sync::Arc::new(engine))
    })
}
```

- [ ] **Step 4: Add the PP-OCR loader + override resolver**

Add these functions below `resolve_override_models` (line 119):

```rust
/// Load the PP-OCR detector (bundled or override). Returns `Arc<Detector>`.
fn load_ppocr(app: &tauri::AppHandle) -> Result<std::sync::Arc<ppocr_engine::Detector>, String> {
    let t = Instant::now();
    let det = match resolve_override_ppocr(app) {
        Some(path) => {
            log::info!("[ppocr] using override model from {}", path.display());
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("PP-OCR override read failed: {e}"))?;
            ppocr_engine::Detector::load_from_buffer(&bytes)
                .map_err(|e| format!("PP-OCR override load failed: {e}"))?
        }
        None => ppocr_engine::Detector::load_from_buffer(BUNDLED_PPOCR_DET)
            .map_err(|e| format!("PP-OCR bundled load failed: {e}"))?,
    };
    log::info!("[ppocr] det loaded in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    Ok(std::sync::Arc::new(det))
}

/// User-supplied PP-OCR override: a single `tiny-det.safetensors` in the
/// platform app-data dir's `ppocr-models/` subdir. Returns `Some(path)` only
/// if the file exists. (Unlike kraken's two-file rule, PP-OCR is one file.)
fn resolve_override_ppocr(app: &tauri::AppHandle) -> Option<PathBuf> {
    let dir = app.path().app_local_data_dir().ok()?.join("ppocr-models");
    let det = dir.join("tiny-det.safetensors");
    if det.exists() { Some(det) } else { None }
}
```

Note the method name: ppocr-rs's `Detector` exposes `load_from_buffer` (singular, set in Task 6 Step 3). Verify by `grep -n "pub fn load_from_buffer\|pub fn load_from_bytes" src-tauri/ppocr-engine/src/model.rs` — pick whichever name Task 6 actually used and use it here consistently.

- [ ] **Step 5: Add `resolve_segmenter`**

Add below the new PP-OCR loader:

```rust
/// Resolve the segmenter for this OCR call. Choices:
///   - `opts.segmenter == Some("ppocr")` → `PPOcrSegmenter` (lazy-loads PP-OCR det)
///   - anything else (including `None`) → `KrakenSegmenter` (lazy-loads Kraken)
///
/// Returns `Arc<dyn Segmenter>` so `run_myanmar` holds a uniform type.
fn resolve_segmenter(
    app: &tauri::AppHandle,
    opts: &OcrOpts,
) -> Result<std::sync::Arc<dyn crate::segmentation::Segmenter>, String> {
    use crate::segmenter_adapters::{KrakenSegmenter, PPOcrSegmenter};
    match opts.segmenter.as_deref() {
        Some("ppocr") => {
            let det = PPOCR.get_or_try_init(|| load_ppocr(app))?.clone();
            Ok(std::sync::Arc::new(PPOcrSegmenter::new(det)))
        }
        Some(other) => {
            log::warn!("[ocr] unknown segmenter {other:?}, falling back to kraken");
            let eng = KRAKEN.get_or_try_init(|| kraken_engine(app).cloned())?.clone();
            Ok(std::sync::Arc::new(KrakenSegmenter::new(eng)))
        }
        None => {
            let eng = KRAKEN.get_or_try_init(|| kraken_engine(app).cloned())?.clone();
            Ok(std::sync::Arc::new(KrakenSegmenter::new(eng)))
        }
    }
}
```

- [ ] **Step 6: Verify it compiles**

```bash
cd src-tauri
cargo build 2>&1 | tail -30
cd ..
```
Expected: compiles. The `run_myanmar` refactor in the next task will use these; for now the new functions are unused-warnings-OK.

- [ ] **Step 7: Commit**

```bash
git add src-tauri/src/engine.rs
git commit -m "refactor(engine): Arc-wrap KRAKEN, add PPOCR OnceCell + resolve_segmenter"
```

---

## Task 17: Refactor `run_myanmar` to dispatch through `Arc<dyn Segmenter>`

The final engine.rs change: replace the direct `engine.segment(img)` call with `segmenter.segment(img)`, and adapt the recog path to fetch Kraken separately when needed.

**Files:**
- Modify: `src-tauri/src/engine.rs` (`run_myanmar`, lines 165–324)

- [ ] **Step 1: Replace the segmenter acquisition + segment call**

In `src-tauri/src/engine.rs::run_myanmar`, replace lines 173–184 (the `let engine = kraken_engine(app)?;` through the segmentation log block) with:

```rust
    let segmenter = resolve_segmenter(app, opts)?;
    let seg_name = segmenter.name();

    let t = Instant::now();
    let lines = segmenter
        .segment(img)
        .map_err(|e| format!("Segmentation failed: {e}"))?;
    let segmentation_ms = t.elapsed().as_millis() as u64;
    log::info!(
        "[ocr] segmentation ({}): {:.0} ms ({} lines)",
        seg_name,
        segmentation_ms as f64,
        lines.len()
    );

    // If recog is Kraken, we need a Kraken engine handle regardless of which
    // segmenter produced the lines. Lazy-load it (shares the OnceCell with
    // KrakenSegmenter — no double-load).
    let kraken_rec_engine: Option<&kraken_engine::Engine> = if opts.engine == "kraken" {
        Some(kraken_engine(app)?.as_ref())
    } else {
        None
    };
```

- [ ] **Step 2: Update the `recognize` closure to consume `DetectedLine`**

The closure currently takes `&kraken_engine::BaselineLine`. Change its parameter to `&crate::segmentation::DetectedLine`, and the recog branches to use the `kraken_rec_engine` Option. Replace the existing `let recognize = |line: &kraken_engine::BaselineLine| ...` block (lines 207–252) with:

```rust
    let recognize = |line: &crate::segmentation::DetectedLine| -> Result<Option<(LineBox, i32)>, String> {
        if line.boundary.len() < 3 {
            return Ok(None);
        }
        let (min_x, min_y, lw, lh) = match polygon_bbox((w, h), &line.boundary) {
            Some(b) => b,
            None => return Ok(None),
        };

        let (text, conf) = match engine_kind {
            "tesseract" => {
                let crop_img = kraken_engine::crop_polygon_white_bg(img, &line.boundary);
                crate::tesseract_line::recognize(
                    &crop_img,
                    app,
                    &opts.language,
                    &opts.whitelist,
                )?
            }
            "kraken" => {
                // Safe unwrap: kraken_rec_engine is Some iff engine_kind == "kraken".
                let eng = kraken_rec_engine.expect("kraken engine loaded for kraken recog");
                let t = eng
                    .recognize_line_dewarped(img, &line.baseline, &line.boundary, binarize)
                    .map_err(|e| format!("Recognition failed: {e}"))?;
                (t, -1)
            }
            other => return Err(format!("Unknown engine: {other}")),
        };

        Ok(Some((
            LineBox {
                x0: min_x,
                y0: min_y,
                x1: min_x + lw,
                y1: min_y + lh,
                text,
                polygon: Some(line.boundary.iter().map(|p| [p.0, p.1]).collect()),
            },
            conf,
        )))
    };
```

- [ ] **Step 3: Verify it compiles + run the existing tests**

```bash
cd src-tauri
cargo build 2>&1 | tail -30
cargo test --lib engine:: -- --nocapture 2>&1 | tail -20
cd ..
```
Expected: compiles; existing engine tests still pass (the bundled-models test, polygon_bbox tests, LineBox serialization tests).

- [ ] **Step 4: Commit**

```bash
git add src-tauri/src/engine.rs
git commit -m "refactor(engine): dispatch run_myanmar through Arc<dyn Segmenter>"
```

---

## Task 18: Add the host bundled-load integration test

**Files:**
- Modify: `src-tauri/src/engine.rs` (append to the `tests` module)

- [ ] **Step 1: Add the test**

In `src-tauri/src/engine.rs`, in the `#[cfg(test)] mod tests` block (around line 408), add:

```rust
    /// Confirm the bundled PP-OCR tiny-det bytes are non-empty and load into
    /// a `Detector`. Mirrors `bundled_models_load_from_buffers` for kraken.
    #[test]
    fn bundled_ppocr_det_loads_from_buffer() {
        assert!(
            super::BUNDLED_PPOCR_DET.len() > 1_000_000,
            "ppocr det too small: {}",
            super::BUNDLED_PPOCR_DET.len()
        );
        let det = ppocr_engine::Detector::load_from_buffer(super::BUNDLED_PPOCR_DET)
            .expect("bundled ppocr det loads from buffer");
        let _ = det;
    }
```

Also update the test module's `use` line (line 410) to import `BUNDLED_PPOCR_DET`:

Change `use super::{polygon_bbox, LineBox, BUNDLED_REC, BUNDLED_SEG};` to:

```rust
use super::{polygon_bbox, LineBox, BUNDLED_PPOCR_DET, BUNDLED_REC, BUNDLED_SEG};
```

- [ ] **Step 2: Run the test**

```bash
cd src-tauri
cargo test bundled_ppocr_det_loads -- --nocapture 2>&1 | tail -15
cd ..
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src-tauri/src/engine.rs
git commit -m "test(engine): verify bundled PP-OCR det loads from buffer"
```

---

## Task 19: Add the `smoke_ppocr` example

**Files:**
- Create: `src-tauri/examples/smoke_ppocr.rs`

- [ ] **Step 1: Write the example**

Create `src-tauri/examples/smoke_ppocr.rs` (modeled on `smoke_kraken.rs`):

```rust
//! Smoke test for the PP-OCR detector: load the bundled tiny-det, detect text
//! regions on a page image, and print the resulting quads. Run with:
//!
//!   cargo run --release --example smoke_ppocr -- <image.png>
//!
//! Defaults to /tmp/scan2_p1.png if no arg is given. Loads the bundled
//! tiny-det bytes via `include_bytes!` (same path the host app uses).

use std::time::Instant;

use image::GenericImageView;
use ppocr_engine::Detector;

/// Bundled tiny-det, same path the host app uses (relative to src-tauri/src/).
const BUNDLED_PPOCR_DET: &[u8] = include_bytes!("../ppocr-models/tiny-det.safetensors");

fn main() -> anyhow::Result<()> {
    let img_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/scan2_p1.png".to_string());

    println!("Loading image: {img_path}");
    let img = image::open(&img_path)?;
    let (w, h) = img.dimensions();
    println!("Image dimensions: {w}x{h}");

    let t = Instant::now();
    println!("Loading PP-OCR tiny-det from bundled bytes...");
    let det = Detector::load_from_buffer(BUNDLED_PPOCR_DET)?;
    println!("  loaded in {:?}", t.elapsed());

    let t = Instant::now();
    let detections = det.detect(&img)?;
    println!("\nDetection in {:?}: {} regions", t.elapsed(), detections.len());

    for (i, d) in detections.iter().enumerate() {
        let poly = &d.polygon;
        println!(
            "  region {i:2} (score {:.2}): ({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0})",
            d.score, poly[0].0, poly[0].1, poly[1].0, poly[1].1,
            poly[2].0, poly[2].1, poly[3].0, poly[3].1,
        );
        // Sanity: all coords in source-image bounds.
        for p in poly {
            debug_assert!(
                p.0 >= 0.0 && p.0 <= w as f32 && p.1 >= 0.0 && p.1 <= h as f32,
                "detection {i} coord out of bounds: ({}, {})", p.0, p.1
            );
        }
    }

    println!("\nAll {} detections are within image bounds.", detections.len());
    Ok(())
}
```

NOTE: verify `Detection`'s field names match what was vendored. `grep -n "pub struct Detection\|pub polygon\|pub score" src-tauri/ppocr-engine/src/postprocess.rs` — if the field isn't `polygon: [Point; 4]` + `score: f32`, adjust the example's print loop to match.

- [ ] **Step 2: Verify it builds**

```bash
cd src-tauri
cargo build --release --example smoke_ppocr 2>&1 | tail -15
cd ..
```
Expected: builds (release to get optimized kernels — first release build is slow).

- [ ] **Step 3: Run the smoke test on a Myanmar fixture**

```bash
# If you have a Myanmar fixture image, point the example at it. Otherwise,
# use any text-bearing image.
ls /tmp/scan2_p1.png 2>/dev/null || echo "no default fixture; provide a path"
cd src-tauri
cargo run --release --example smoke_ppocr -- /path/to/a/text/image.png 2>&1 | tail -30
cd ..
```
Expected: prints image dimensions, load time, detection count, and per-region quads. If `0 regions`, the image has no detectable text or the model's thresholds need tuning (defer to manual tuning).

- [ ] **Step 4: Commit**

```bash
git add src-tauri/examples/smoke_ppocr.rs
git commit -m "feat(examples): add smoke_ppocr for the PP-OCR detector"
```

---

## Task 20: Add `Segmenter` type + field to the frontend

**Files:**
- Modify: `src/lib/ocr.ts` (type, OcrOpts interface, persistence)

- [ ] **Step 1: Read the relevant parts of `ocr.ts`**

```bash
grep -n "type Engine\|interface OcrOpts\|just-ocr:engine\|lastEngine\|saveEngine" src/lib/ocr.ts
```
Confirm the existing engine type (line ~21), the OcrOpts interface (~27), and the persistence helpers (~329–370).

- [ ] **Step 2: Add the `Segmenter` type + `OcrOpts.segmenter` field**

In `src/lib/ocr.ts`, near the existing `type Engine = ...` (line 21), add:

```ts
export type Segmenter = "kraken" | "ppocr";
```

In the `OcrOpts` interface (~line 27), add as the last field:

```ts
  segmenter: Segmenter;
```

- [ ] **Step 3: Add persistence helpers (mirror `lastEngine`/`saveEngine`)**

In `src/lib/ocr.ts`, near the existing engine persistence (~line 329–370), add:

```ts
const LAST_SEGMENTER_KEY = "just-ocr:segmenter";

export function lastSegmenter(): Segmenter {
  try {
    const v = localStorage.getItem(LAST_SEGMENTER_KEY);
    return v === "ppocr" ? "ppocr" : "kraken";
  } catch {
    return "kraken";
  }
}

export function saveSegmenter(s: Segmenter): void {
  try {
    localStorage.setItem(LAST_SEGMENTER_KEY, s);
  } catch {
    /* private mode */
  }
}
```

- [ ] **Step 4: Verify the type gate**

```bash
npm run build 2>&1 | tail -15
```
Expected: build succeeds. If it errors that `opts.segmenter` is missing somewhere `OcrOpts` is constructed, fix those call sites to add `segmenter: "kraken"` (the next task does this in App.svelte).

- [ ] **Step 5: Commit**

```bash
git add src/lib/ocr.ts
git commit -m "feat(frontend): add Segmenter type + persistence"
```

---

## Task 21: Wire `segmenter` into `App.svelte` opts + persistence

**Files:**
- Modify: `src/App.svelte` (opts init + persist effect)

- [ ] **Step 1: Add `segmenter` to the opts initialization**

Find the `let opts = $state<OcrOpts>({...})` block in `App.svelte` (~lines 60–66). Add `segmenter: lastSegmenter(),` to the object, and update the import at the top of the file to include `lastSegmenter` + `saveSegmenter`.

The opts block becomes:

```ts
let opts = $state<OcrOpts>({
  engine: lastEngine(),
  language: lastLanguage() ?? "eng",
  psm: 3,
  whitelist: null,
  binarize: lastBinarize(),
  segmenter: lastSegmenter(),
});
```

And the import (find the existing `import { lastEngine, saveEngine, ... } from "./lib/ocr"` line) gains `lastSegmenter` and `saveSegmenter`.

- [ ] **Step 2: Add the persist effect**

Find the existing `$effect(() => { saveEngine(opts.engine); });` (~line 71). Add a sibling:

```ts
$effect(() => { saveSegmenter(opts.segmenter); });
```

- [ ] **Step 3: Verify the build**

```bash
npm run build 2>&1 | tail -15
```
Expected: builds clean.

- [ ] **Step 4: Commit**

```bash
git add src/App.svelte
git commit -m "feat(frontend): init + persist opts.segmenter"
```

---

## Task 22: Add the "Seg" dropdown to `Toolbar.svelte`

**Files:**
- Modify: `src/lib/Toolbar.svelte` (inside the `{#if isMyanmar}` branch, ~line 118)

- [ ] **Step 1: Add the dropdown before the existing Engine dropdown**

In `src/lib/Toolbar.svelte`, find the `{#if isMyanmar}` block (~line 118). Insert this `<label>` before the existing Engine `<label class="field">`:

```svelte
{#if isMyanmar}
  <label class="field">
    <span class="lbl">Seg</span>
    <select bind:value={opts.segmenter}>
      <option value="kraken">Kraken</option>
      <option value="ppocr">PP-OCR</option>
    </select>
  </label>
  <label class="field">
    <span class="lbl">Engine</span>
    <select bind:value={opts.engine}>
      <option value="kraken">Kraken</option>
      <option value="tesseract">Tesseract</option>
    </select>
  </label>
{:else}
  ... (existing PSM block unchanged) ...
{/if}
```

(The existing Engine `<label>` is already there — only the new "Seg" `<label>` is inserted above it.)

- [ ] **Step 2: Verify the build**

```bash
npm run build 2>&1 | tail -15
```
Expected: builds clean.

- [ ] **Step 3: Run the frontend tests**

```bash
npm test 2>&1 | tail -30
```
Expected: all existing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib/Toolbar.svelte
git commit -m "feat(frontend): add Seg dropdown (Kraken / PP-OCR) for Myanmar"
```

---

## Task 23: Manual end-to-end verification

This is the human-eye gate before declaring done. No code changes — verification only.

- [ ] **Step 1: Start the dev app**

```bash
cargo tauri dev
```
(First build after the refactor will recompile tesseract + candle + ppocr-engine from source — expect several minutes.)

- [ ] **Step 2: Verify Kraken path still works (regression check)**

In the app:
1. Set Lang = `mya`, Seg = `Kraken`, Engine = `Kraken`.
2. Load a Myanmar fixture image.
3. Confirm: line boxes appear in the overlay, text is recognized, status bar shows segmentation + recognition timing.

Repeat with Engine = `Tesseract`. Confirm both recog paths still work.

- [ ] **Step 3: Verify PP-OCR seg + Tesseract recog**

1. Set Seg = `PP-OCR`, Engine = `Tesseract`.
2. Load the same image.
3. Confirm: line boxes appear (now from PP-OCR quads), text is recognized via Tesseract. Status bar shows `segmentation (ppocr-tiny)`.

- [ ] **Step 4: Verify PP-OCR seg + Kraken recog (the riskiest pairing)**

1. Set Seg = `PP-OCR`, Engine = `Kraken`.
2. Load the same image.
3. Confirm: lines are recognized (Kraken recog on a synthesized baseline). Quality may be lower than Kraken seg — that's expected and acceptable per spec. Verify no panics or hard errors.

- [ ] **Step 5: Verify the dropdown only appears for Myanmar**

1. Set Lang = `eng`. Confirm the "Seg" dropdown disappears (the PSM dropdown shows instead).
2. Set Lang back to `mya`. Confirm "Seg" reappears and remembers its previous value.

- [ ] **Step 6: Run the full Rust test suite one more time**

```bash
cd src-tauri && cargo test -- --nocapture 2>&1 | tail -20 && cd ..
```
Expected: all tests pass (kraken bundled-load, ppocr bundled-load, polygon_bbox, LineBox serialization, segmentation serialization, adapter tests, postprocess tests).

- [ ] **Step 7: Commit any fixes surfaced by manual testing**

If manual testing surfaced bugs, fix them in dedicated commits with clear messages. If clean, no commit needed.

---

## Task 24: Update docs

**Files:**
- Modify: `AGENTS.md` (note the new crate + segmenter field)

- [ ] **Step 1: Add `ppocr-engine` to the AGENTS.md key directories**

In `AGENTS.md`, in the "Key directories" section, add an entry for `ppocr-engine/` next to `kraken-engine/`:

```
  ppocr-engine/           vendored PP-OCRv6 tiny detector (separate crate,
                          NOT a workspace member — same opt-level trick as
                          kraken-engine). Detector + DB postprocess only.
```

- [ ] **Step 2: Note the segmenter dispatch in the architecture section**

In `AGENTS.md`, in the "OCR pipeline dispatch" section, add a bullet:

```
- `language == "mya"` + `segmenter == "ppocr"` → PP-OCR segmentation (tiny-det)
  → per-line recognition by `engine` ("kraken" | "tesseract"). PP-OCR's quads
  are converted to closed boundary polygons + a synthesized baseline; Kraken
  recog dewarp falls back to a masked bbox crop if the synth baseline fails.
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): note ppocr-engine crate + segmenter dispatch"
```

---

## Self-Review (run after the plan is written, not during execution)

After the plan is complete, run this checklist yourself:

1. **Spec coverage:** Does every spec section map to a task?
   - Architecture (sibling crate + trait): Tasks 2–11 (crate), 12 (trait), 13–14 (adapters)
   - `DetectedLine` + data flow: Task 12 (type), Task 17 (flow)
   - `resolve_segmenter` table + lazy load: Task 16
   - Model bundling (include_bytes! + override): Task 1 (asset), Task 16 Step 1+4 (bytes + override)
   - Frontend/IPC: Tasks 15 (Rust field), 20 (TS), 21 (App), 22 (Toolbar)
   - Vendoring strategy: Tasks 3–9
   - Error handling: Task 14 (filter degenerate), Task 16 (override rules)
   - Logging: Task 17 Step 1 (segmentation log), Task 16 Step 4 (load log)
   - Testing: Task 10 (crate unit), Task 18 (host integration), Task 19 (smoke), Task 23 (manual)

2. **Placeholder scan:** No "TBD", "TODO", "implement later". The `>>> PASTE HERE` markers in Task 7 are explicit copy instructions with line ranges — not placeholders.

3. **Type consistency:** `Detector::load_from_buffer` (singular) is used in Tasks 6, 9, 16, 18, 19 — consistent. `DetectedLine { baseline, boundary }` consistent across Tasks 12, 13, 14, 17. `resolve_segmenter` consistent across Tasks 16, 17.

4. **Scope check:** Single feature, ~24 tasks, each independently testable.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-27-ppocr-segmentation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
