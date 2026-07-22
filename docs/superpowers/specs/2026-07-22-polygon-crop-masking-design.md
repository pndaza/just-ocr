# Polygon crop masking + frontend polygon overlays — design

**Date:** 2026-07-22
**Status:** Approved (2026-07-22)
**Resolves:** `docs/notes/2026-07-19-polygon-overlay.md` (full fix path: all
three parts — masking + `LineBox.polygon` + frontend `<polygon>` rendering).

## Problem

Kraken's segmentation returns, per text line, a **boundary polygon**
(`Vec<(f64, f64)>` in `BaselineLine::boundary`) following the true shape of
the line — curved, rotated, or skewed. Today `engine.rs` collapses each to
its axis-aligned bounding box and discards the shape. Two problems follow:

1. **Visual fidelity (cosmetic).** Rectangles can't represent rotated/curved
   lines; on dense or skewed pages neighboring bboxes overlap where the true
   polygons wouldn't.
2. **Recognition quality (the real bug).** `engine.rs` crops each line with
   `img.crop_imm(...)` — an axis-aligned rectangle whose corners can contain
   ink from neighboring lines, leaking into the recognizer input and degrading
   accuracy.

Kraken's own recognizer uses `crop_polygon_white_bg` (kraken-rust
`src/recognition/orchestrator.rs:128`), which composites the polygon onto a
white background. That helper was dropped when kraken-rust was vendored.

## Solution

Implement the full fix path from the note:

1. Port `crop_polygon_white_bg` into `kraken-engine` and use masked crops for
   recognition (both Kraken and Tesseract recognizers) in the Myanmar path.
2. Extend `LineBox` with an optional `polygon` field, populated from
   `BaselineLine::boundary` for the Kraken path, `None` for the Tesseract
   full-page path.
3. Render `<polygon>` overlays in `Preview.svelte` when `polygon` is present,
   fall back to `<rect>` from bbox otherwise.

## Backend

### `LineBox` gains an optional polygon

`src-tauri/src/engine.rs`:

```rust
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<[f64; 2]>>,
}
```

- **Kraken (Myanmar) path** — populate from `line.boundary`
  (`Vec<(f64, f64)>` → `Vec<[f64; 2]>`).
- **Tesseract full-page path** (`src-tauri/src/tesseract_page.rs`) —
  construct `LineBox`es with `polygon: None`. The serde `skip_serializing_if`
  keeps the field out of the payload entirely, so this path's wire format is
  byte-identical to today.

### Masked crop

**New file** `src-tauri/kraken-engine/src/recognition/crop.rs` — a faithful
port of kraken-rust's `orchestrator.rs:128` `crop_polygon_white_bg`. Public:

```rust
pub fn crop_polygon_white_bg(image: &DynamicImage, boundary: &[(f64, f64)]) -> DynamicImage
```

Steps (unchanged from the reference):

1. Compute the axis-aligned bbox of the polygon, clamped to image bounds
   (`polygon_bbox`).
2. Translate polygon points into crop-local coordinates; dedup consecutive
   duplicates and drop a closing point equal to the first (imageproc panics
   on `first == last`).
3. Rasterize the polygon into a mask via
   `imageproc::drawing::draw_polygon_mut` (white inside, transparent
   outside).
4. Composite: start white, copy source pixels where the mask is set.
   Outside-polygon → white, preserving the black-on-white polarity that
   `preprocess`'s invert step and Tesseract both expect.
5. Degenerate fallback: fewer than 3 valid points → plain `crop_imm`
   (matches the reference).

Dependencies already present in `kraken-engine/Cargo.toml`: `imageproc` (0.25)
and `image` (0.25). `imageproc::drawing::draw_polygon_mut` +
`imageproc::point::Point` are what the reference uses. No new deps.

`recognition/mod.rs` gets `pub mod crop;` and
`pub use crop::crop_polygon_white_bg;`. `lib.rs` re-exports it at the crate
root (`kraken_engine::crop_polygon_white_bg`).

### `engine.rs` recognition closure

Replace the axis-aligned crop with the masked crop:

```rust
// Before
let crop_img = image::DynamicImage::ImageRgb8(img.crop_imm(min_x, min_y, lw, lh).to_rgb8());
// After
let crop_img = kraken_engine::crop_polygon_white_bg(img, &line.boundary);
```

Both recognizers consume this same `crop_img`:

- `tesseract_line::recognize` calls `.to_rgb8()` on it.
- `engine.recognize_line` calls `.to_luma8()` internally.

Returning an RGBA `DynamicImage` (what the masking produces) is fine for both.

`polygon_bbox` is still called once per line to compute the `LineBox`
`{x0, y0, x1, y1}` overlay coords — that part is unchanged.

## Frontend

### `result.ts` — type addition

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

The optional field mirrors Rust's `skip_serializing_if` — the field is absent
from the payload for Tesseract-path lines, not `null`.

### `Preview.svelte` — polygon-aware overlay

In the `{#each parsed.lines}` block, render a `<polygon>` when `polygon` is
present, fall back to `<rect>`:

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

The existing `.bbox` style (`fill: none; stroke: var(--accent); ...`) applies
to both `<polygon>` and `<rect>`. No new CSS needed.

## Tests

- **`crop.rs`**: ink placed at bbox corners but outside the polygon → masked
  to white; ink inside the polygon → preserved from the source; degenerate
  boundary (`len < 3`) → falls back to plain bbox crop, no panic.
- **`engine.rs`**: the Myanmar closure now constructs `LineBox` with
  `polygon: Some(...)`; a serde test confirms the Tesseract-path `LineBox`
  (with `polygon: None`) serializes with no `polygon` field (byte-identical
  wire format to today).
- **Frontend** — the SVG branch is pure rendering with no extractable logic;
  the `vite build` type gate is the safety net. No new vitest test needed.

## Wire-format / compatibility

- The `polygon` field is **additive**. The Tesseract path omits it entirely
  via `skip_serializing_if`, so its payload is byte-identical to today.
- Existing frontend code is tolerant of unknown/absent fields; the
  `polygon?:` field is additive and defaults to "render rect."

## Out of scope

- No `plainText` / export changes (polygons are overlay-only).
- No persistence changes.
- No new frontend tests beyond the `vite build` type gate.

## Risks

- **Performance:** masking adds one composite pass per Myanmar line (cheap,
  O(bbox pixels)); negligible vs. the recognizer forward pass. The serial
  Tesseract-recognizer path gets a small constant per line — acceptable.
- **Thread-safety:** none — pure per-line work, no shared mutable state.
- **Wire format:** additive field with serde skip; Tesseract path unchanged.

## Reference

- kraken-rust `src/recognition/orchestrator.rs:128` — `crop_polygon_white_bg`
  (white-background polygon composite; the correct line-crop behavior).
- kraken-rust `src/recognition/orchestrator.rs:110` — `crop_boundary`
  (the axis-aligned version we currently use).
- `docs/notes/2026-07-19-polygon-overlay.md` — the deferred note this
  implements.
