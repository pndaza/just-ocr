# Polygon crop masking + frontend polygon overlays — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send polygon-masked (white-bg composite) line crops to recognition for the Myanmar path, and render the true boundary polygons as `<polygon>` overlays in the preview.

**Architecture:** Port kraken-rust's `crop_polygon_white_bg` into `kraken-engine` as a new `recognition/crop.rs` module. `engine.rs`'s Myanmar recognition closure uses it for both recognizers (Kraken + Tesseract). `LineBox` gains an optional `polygon: Option<Vec<[f64; 2]>>` (serde-skipped when `None`), populated from `BaselineLine::boundary` on the Kraken path and set to `None` on the Tesseract full-page path. The frontend renders `<polygon>` when present, falls back to `<rect>`.

**Tech Stack:** Rust (`kraken-engine` + `just_ocr_lib` app crate), `imageproc` 0.25 (`draw_polygon_mut`), `image` 0.25, serde, Svelte 5 + TypeScript.

**Spec:** `docs/superpowers/specs/2026-07-22-polygon-crop-masking-design.md`

**Reference (do not edit, read-only):** `/Users/pndaza/Projects/playground/kraken-rust/src/recognition/orchestrator.rs` — lines 110 (`crop_boundary`), 128 (`crop_polygon_white_bg`), 189 (`polygon_bbox`).

---

## File Structure

**Create:**
- `src-tauri/kraken-engine/src/recognition/crop.rs` — `crop_polygon_white_bg` port + its own `polygon_bbox` (crate-private).

**Modify:**
- `src-tauri/kraken-engine/src/recognition/mod.rs` — add `pub mod crop;` + re-export.
- `src-tauri/kraken-engine/src/lib.rs` — re-export `crop_polygon_white_bg` at crate root.
- `src-tauri/src/engine.rs` — `LineBox` gains `polygon` field; Myanmar closure uses masked crop + populates polygon.
- `src-tauri/src/tesseract_page.rs` — construct `LineBox`es with `polygon: None` (2 sites).
- `src/lib/result.ts` — `LineBox` gains optional `polygon`.
- `src/lib/Preview.svelte` — render `<polygon>` when present, else `<rect>`.

---

## Task 1: Add `crop_polygon_white_bg` to kraken-engine (TDD)

**Files:**
- Create: `src-tauri/kraken-engine/src/recognition/crop.rs`

- [ ] **Step 1: Write the failing tests**

Create `src-tauri/kraken-engine/src/recognition/crop.rs` with ONLY the test module (the function doesn't exist yet, so tests won't compile — that's the TDD red state):

```rust
//! Polygon line-crop masking: composite a line's boundary polygon onto a
//! white background so ink from neighboring lines (outside the polygon but
//! inside its bbox) is masked away before recognition.
//!
//! Faithful port of kraken-rust's `orchestrator.rs:128` `crop_polygon_white_bg`,
//! which was dropped when kraken-rust was vendored into this crate. Uses
//! `imageproc::drawing::draw_polygon_mut` to rasterize the polygon mask.

use image::{DynamicImage, GenericImageView};
use imageproc::drawing::draw_polygon_mut;
use imageproc::point::Point;

/// Crop a line's boundary polygon from the image, filling the area outside
/// the polygon (but inside its bounding box) with white.
///
/// This preserves the black-on-white text polarity the recognizers expect:
/// kraken's `preprocess` inverts to ink-high, and Tesseract operates on
/// dark-ink-on-light-bg. A plain bbox crop (`crop_imm`) would let neighboring
/// lines' ink bleed in at the rectangle corners; this masks it out.
///
/// Degenerate polygons (fewer than 3 distinct points after dedup) fall back
/// to a plain axis-aligned bbox crop.
pub fn crop_polygon_white_bg(image: &DynamicImage, boundary: &[(f64, f64)]) -> DynamicImage {
    todo!("implement in Step 3")
}

/// Axis-aligned bbox of a polygon, clamped to image bounds. Returns
/// `(min_x, min_y, width, height)` or `None` if the bbox is zero-area.
fn polygon_bbox(
    (img_w, img_h): (u32, u32),
    boundary: &[(f64, f64)],
) -> Option<(u32, u32, u32, u32)> {
    todo!("implement in Step 3")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    /// Build a 40x40 white image with a single dark pixel at each of the four
    /// bbox corners of a tight diamond polygon. The diamond's vertices touch
    /// the edge midpoints, so the corners are OUTSIDE the polygon and must be
    /// masked to white; the center is INSIDE and must stay dark.
    fn diamond_image() -> (ImageBuffer<Rgba<u8>, Vec<u8>>, Vec<(f64, f64)>) {
        let (w, h) = (40u32, 40u32);
        let mut img = ImageBuffer::from_pixel(w, h, Rgba([255, 255, 255, 255]));
        // Dark corners (outside the diamond).
        for &(x, y) in &[(0, 0), (39, 0), (0, 39), (39, 39)] {
            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
        // Dark center (inside the diamond).
        img.put_pixel(20, 20, Rgba([0, 0, 0, 255]));
        // Diamond touching edge midpoints: bbox = (0,0,40,40).
        let poly = vec![(20.0, 0.0), (40.0, 20.0), (20.0, 40.0), (0.0, 20.0)];
        (img, poly)
    }

    #[test]
    fn masks_ink_outside_polygon_to_white() {
        let (img, poly) = diamond_image();
        let dyn_img = DynamicImage::ImageRgba8(img);
        let crop = crop_polygon_white_bg(&dyn_img, &poly);
        // The crop bbox is (0,0,40,40); the four corners must be white now.
        assert_eq!(crop.get_pixel(0, 0), Rgba([255, 255, 255, 255]));
        assert_eq!(crop.get_pixel(39, 0), Rgba([255, 255, 255, 255]));
        assert_eq!(crop.get_pixel(0, 39), Rgba([255, 255, 255, 255]));
        assert_eq!(crop.get_pixel(39, 39), Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn preserves_ink_inside_polygon() {
        let (img, poly) = diamond_image();
        let dyn_img = DynamicImage::ImageRgba8(img);
        let crop = crop_polygon_white_bg(&dyn_img, &poly);
        // Center is inside the diamond — must stay dark.
        assert_eq!(crop.get_pixel(20, 20), Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn degenerate_polygon_falls_back_to_bbox_crop() {
        // Fewer than 3 points → degenerate → plain bbox crop, no panic.
        // Two points at opposite corners: a single dark pixel at (0,0) must
        // survive (it's inside the bbox, and there's no polygon to mask with).
        let mut img = ImageBuffer::from_pixel(10, 10, Rgba([255, 255, 255, 255]));
        img.put_pixel(0, 0, Rgba([0, 0, 0, 255]));
        let dyn_img = DynamicImage::ImageRgba8(img);
        let crop = crop_polygon_white_bg(&dyn_img, &[(0.0, 0.0), (9.0, 9.0)]);
        assert_eq!(crop.get_pixel(0, 0), Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn polygon_bbox_basic() {
        let b = vec![(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)];
        assert_eq!(polygon_bbox((100, 100), &b), Some((10, 20, 21, 21)));
    }

    #[test]
    fn polygon_bbox_empty_returns_none() {
        assert_eq!(polygon_bbox((100, 100), &[]), None);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile (red)**

Run from `src-tauri/kraken-engine/`:
```sh
cargo test crop
```
Expected: compile error / `todo!()` panic — the function bodies are stubs. This confirms the tests are wired up against the right symbols.

- [ ] **Step 3: Implement `crop_polygon_white_bg` and `polygon_bbox`**

Replace the two `todo!()` bodies in `src-tauri/kraken-engine/src/recognition/crop.rs` with:

```rust
pub fn crop_polygon_white_bg(image: &DynamicImage, boundary: &[(f64, f64)]) -> DynamicImage {
    let rgba = image.to_rgba8();
    let (min_x, min_y, w, h) = match polygon_bbox(image.dimensions(), boundary) {
        Some(b) => b,
        None => return image.clone(),
    };

    // Translate boundary points into crop-local coordinates and normalise
    // them for imageproc: drop a closing point equal to the first and dedup
    // consecutive duplicates (imageproc panics on first==last).
    let mut pts: Vec<Point<i32>> = boundary
        .iter()
        .map(|p| Point::new((p.0 - min_x as f64) as i32, (p.1 - min_y as f64) as i32))
        .collect();
    pts.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    if pts.len() >= 2 && pts.first() == pts.last() {
        pts.pop();
    }
    if pts.len() < 3 {
        // Degenerate polygon — fall back to a plain bbox crop.
        return image.crop_imm(min_x, min_y, w, h);
    }

    // Build a mask: opaque white inside the polygon, transparent outside.
    let mut mask = image::ImageBuffer::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    draw_polygon_mut(&mut mask, &pts, Rgba([255, 255, 255, 255]));

    // Composite: start from white, copy source pixels where the mask is set.
    let mut out = image::ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] > 0 {
                out.put_pixel(x, y, *rgba.get_pixel(min_x + x, min_y + y));
            } else {
                out.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
    }
    image::DynamicImage::ImageRgba8(out)
}

/// Axis-aligned bbox of a polygon, clamped to image bounds. Returns
/// `(min_x, min_y, width, height)` or `None` if the bbox is zero-area.
fn polygon_bbox(
    (img_w, img_h): (u32, u32),
    boundary: &[(f64, f64)],
) -> Option<(u32, u32, u32, u32)> {
    if boundary.is_empty() {
        return None;
    }
    let min_x = boundary
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min)
        .max(0.0) as u32;
    let min_y = boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min)
        .max(0.0) as u32;
    let max_x = boundary
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .min((img_w - 1) as f64) as u32;
    let max_y = boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .min((img_h - 1) as f64) as u32;
    let w = max_x.saturating_sub(min_x) + 1;
    let h = max_y.saturating_sub(min_y) + 1;
    if w == 0 || h == 0 {
        None
    } else {
        Some((min_x, min_y, w, h))
    }
}
```

- [ ] **Step 4: Run tests to verify they pass (green)**

Run from `src-tauri/kraken-engine/`:
```sh
cargo test crop
```
Expected: 5 passed; 0 failed.

- [ ] **Step 5: Commit**

```sh
git add src-tauri/kraken-engine/src/recognition/crop.rs
git commit -m "Add crop_polygon_white_bg to kraken-engine

Port of kraken-rust's orchestrator.rs:128. Composites a line's boundary
polygon onto white so neighboring-line ink is masked out before recognition."
```

---

## Task 2: Re-export `crop_polygon_white_bg` from kraken-engine

**Files:**
- Modify: `src-tauri/kraken-engine/src/recognition/mod.rs`
- Modify: `src-tauri/kraken-engine/src/lib.rs`

- [ ] **Step 1: Register the module + re-export in `recognition/mod.rs`**

In `src-tauri/kraken-engine/src/recognition/mod.rs`, add `pub mod crop;` to the module declarations (alongside `pub mod preprocess;` etc.) and add `crop::crop_polygon_white_bg` to the re-exports.

The current module declaration block is:
```rust
pub mod codec;
pub mod decode;
pub mod meta;
pub mod model;
pub mod preprocess;
```
Add `pub mod crop;` (keep alphabetical-ish / group with siblings). After the existing `pub use preprocess::{preprocess_line, Binarization};`, add:
```rust
pub use crop::crop_polygon_white_bg;
```

- [ ] **Step 2: Re-export at the crate root in `lib.rs`**

In `src-tauri/kraken-engine/src/lib.rs`, alongside the existing
`pub use recognition::{preprocess::preprocess_line, RecognitionModel};`
add `crop_polygon_white_bg` so callers can write `kraken_engine::crop_polygon_white_bg`.

- [ ] **Step 3: Verify it compiles**

Run from `src-tauri/kraken-engine/`:
```sh
cargo build
```
Expected: builds cleanly.

- [ ] **Step 4: Commit**

```sh
git add src-tauri/kraken-engine/src/recognition/mod.rs src-tauri/kraken-engine/src/lib.rs
git commit -m "Re-export crop_polygon_white_bg from kraken-engine"
```

---

## Task 3: Use masked crops + populate `LineBox.polygon` (TDD on serde)

**Files:**
- Modify: `src-tauri/src/engine.rs`

- [ ] **Step 1: Write the failing serde test**

In `src-tauri/src/engine.rs`, inside the existing `#[cfg(test)] mod tests` block, add a test verifying that a `LineBox` with `polygon: None` serializes WITHOUT a `polygon` key (byte-identical wire format to pre-change), and one with `Some(...)` includes it:

```rust
    #[test]
    fn linebox_without_polygon_omits_field() {
        let lb = LineBox {
            x0: 1,
            y0: 2,
            x1: 3,
            y1: 4,
            text: "hi".to_string(),
            polygon: None,
        };
        let json = serde_json::to_string(&lb).unwrap();
        assert!(
            !json.contains("polygon"),
            "polygon field must be absent when None, got: {json}"
        );
    }

    #[test]
    fn linebox_with_polygon_includes_field() {
        let lb = LineBox {
            x0: 1,
            y0: 2,
            x1: 3,
            y1: 4,
            text: "hi".to_string(),
            polygon: Some(vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        };
        let json = serde_json::to_string(&lb).unwrap();
        assert!(
            json.contains("\"polygon\":[[1.0,2.0],[3.0,4.0],[5.0,6.0]]"),
            "polygon field must serialize as array-of-pairs, got: {json}"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail to compile (red)**

Run from `src-tauri/`:
```sh
cargo test linebox_
```
Expected: compile error — `LineBox` has no `polygon` field.

- [ ] **Step 3: Add the `polygon` field to `LineBox`**

In `src-tauri/src/engine.rs`, replace the `LineBox` struct:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct LineBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub text: String,
}
```

with:

```rust
/// One recognized line: an axis-aligned bbox (in source-image pixel space),
/// the decoded text, and — for the Kraken-segmented path — the true boundary
/// polygon. The frontend overlays the polygon when present and falls back to
/// the bbox otherwise.
#[derive(Debug, Clone, Serialize)]
pub struct LineBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub text: String,
    /// True boundary polygon from Kraken segmentation (source-image pixel
    /// space). Present only for the Kraken-segmented (Myanmar) path; `None`
    /// for the Tesseract full-page path, which produces bboxes, not polygons.
    /// Skipped on serialization when `None` so the Tesseract-path wire format
    /// is byte-identical to pre-polygon builds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<[f64; 2]>>,
}
```

- [ ] **Step 4: Fix the two `LineBox` construction sites to add `polygon`**

**Site A — Myanmar closure in `engine.rs`** (the `recognize` closure, currently):

```rust
        Ok(Some((
            LineBox {
                x0: min_x,
                y0: min_y,
                x1: min_x + lw,
                y1: min_y + lh,
                text,
            },
            conf,
        )))
```

Replace with (populate polygon from `line.boundary`, converting `Vec<(f64,f64)>` → `Vec<[f64;2]>`):

```rust
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
```

**Site B — Myanmar closure crop line** (same closure, currently):

```rust
        let crop_img =
            image::DynamicImage::ImageRgb8(img.crop_imm(min_x, min_y, lw, lh).to_rgb8());
```

Replace with the masked crop:

```rust
        let crop_img = kraken_engine::crop_polygon_white_bg(img, &line.boundary);
```

**Site C — `tesseract_page.rs`** `parse_hocr_lines` (the `LineBox` literal near line 157):

```rust
            out.push(LineBox {
                x0: bbox.0,
                y0: bbox.1,
                x1: bbox.2,
                y1: bbox.3,
                text,
            });
```

Replace with:

```rust
            out.push(LineBox {
                x0: bbox.0,
                y0: bbox.1,
                x1: bbox.2,
                y1: bbox.3,
                text,
                polygon: None,
            });
```

- [ ] **Step 5: Run tests to verify they pass (green)**

Run from `src-tauri/`:
```sh
cargo test linebox_
```
Expected: 2 passed.

Then the full app-crate test suite to catch anything else:
```sh
cargo test
```
Expected: all pass.

- [ ] **Step 6: Commit**

```sh
git add src-tauri/src/engine.rs src-tauri/src/tesseract_page.rs
git commit -m "Mask line crops + populate LineBox.polygon

engine.rs Myanmar path now feeds polygon-masked (white-bg composite) crops
to both recognizers and populates LineBox.polygon from BaselineLine::boundary.
Tesseract full-page path sets polygon: None (serde-skipped, wire format
unchanged)."
```

---

## Task 4: Frontend type + polygon overlay rendering

**Files:**
- Modify: `src/lib/result.ts`
- Modify: `src/lib/Preview.svelte`

- [ ] **Step 1: Add `polygon` to the `LineBox` TS type**

In `src/lib/result.ts`, replace:

```ts
export interface LineBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  text: string;
}
```

with:

```ts
export interface LineBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  text: string;
  /** True boundary polygon (source-image pixel space). Present only for the
   *  Kraken-segmented (Myanmar) path; absent for Tesseract full-page. */
  polygon?: [number, number][];
}
```

- [ ] **Step 2: Render `<polygon>` when present in `Preview.svelte`**

In `src/lib/Preview.svelte`, replace the overlay loop:

```svelte
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
```

with:

```svelte
        {#each parsed.lines as b}
          {#if b.polygon}
            <polygon
              points={b.polygon.map(([x, y]) => `${x},${y}`).join(" ")}
              class="bbox"
              vector-effect="non-scaling-stroke"
            />
          {:else}
            <rect
              x={b.x0}
              y={b.y0}
              width={b.x1 - b.x0}
              height={b.y1 - b.y0}
              class="bbox"
              vector-effect="non-scaling-stroke"
            />
          {/if}
        {/each}
```

- [ ] **Step 3: Type-check with the vite build gate**

Run from repo root:
```sh
npm run build
```
Expected: build succeeds (this is the strict + checkJs type gate per `AGENTS.md`).

- [ ] **Step 4: Commit**

```sh
git add src/lib/result.ts src/lib/Preview.svelte
git commit -m "Render polygon line overlays in preview

LineBox gains optional polygon ([x,y][]). Preview renders <polygon> when
present (Kraken/Myanmar path), falls back to <rect> from bbox otherwise
(Tesseract full-page path)."
```

---

## Task 5: Full build + manual verification

- [ ] **Step 1: Run the full backend test suite**

From `src-tauri/`:
```sh
cargo test -- --nocapture
```
Expected: all pass; note the new crop + linebox tests by name.

- [ ] **Step 2: Build the frontend**

From repo root:
```sh
npm run build
```
Expected: success.

- [ ] **Step 3: Smoke-test via the kraken example (optional but recommended)**

From `src-tauri/`:
```sh
cargo run --example smoke_kraken
```
Expected: runs without panic; produces recognized text. (If no fixture is present, it skips — check the example's fixture path.)

- [ ] **Step 4: Manual UI check**

```sh
cargo tauri dev
```
Run OCR on a Myanmar image (dense/rotated page shows the effect best). Confirm:
- Lines overlay as tight polygons following the true line shape (not loose rectangles).
- Neighboring-line polygons no longer overlap where the page is dense.
- For a non-Myanmar (e.g. `eng`) image, overlays are still rectangles (Tesseract path, no polygon).

---

## Self-Review

**1. Spec coverage** — all three spec sections map to tasks:
- "Backend: masked crop" → Task 1 (port) + Task 2 (re-export) + Task 3 Step 4 Site B (use in engine).
- "Backend: `LineBox` gains optional polygon" → Task 3 Step 3 (struct) + Step 4 Sites A & C (populate both paths).
- "Frontend: polygon overlay" → Task 4.
- "Tests" → Task 1 (crop), Task 3 (serde), Task 4 Step 3 (type gate).
✅ No gaps.

**2. Placeholder scan** — no TBD/TODO/"add error handling"/"similar to Task N". The `todo!()` in Task 1 Step 1 is intentional TDD-red and is explicitly replaced in Step 3. The duplicate-`x1` typo in Task 4 Step 1 is called out inline with the corrected version. ✅

**3. Type consistency** —
- `crop_polygon_white_bg(image: &DynamicImage, boundary: &[(f64, f64)]) -> DynamicImage` — same signature in Task 1 (def) and Task 3 Step 4 Site B (call). ✅
- `LineBox.polygon: Option<Vec<[f64; 2]>>` (Rust) ↔ `polygon?: [number, number][]` (TS) — names + shape match across Task 3 and Task 4. ✅
- `boundary.iter().map(|p| [p.0, p.1])` produces `Vec<[f64;2]>` matching the field type. ✅
- `kraken_engine::crop_polygon_white_bg` re-export path (Task 2) matches the call in Task 3. ✅
