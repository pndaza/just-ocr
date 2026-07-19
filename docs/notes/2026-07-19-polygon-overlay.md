# Polygon overlay & crop masking — followup note

**Status:** Deferred (2026-07-19). Captured for later discussion.

## Context

Kraken's segmentation returns, per text line, a **boundary polygon**
(`Vec<(f64, f64)>` in `BaselineLine::boundary`) that follows the actual
shape of the line — curved, rotated, or skewed as the page dictates.

Today `engine.rs` collapses each polygon to its **axis-aligned bounding box**
(`polygon_bbox` = `min/max` of polygon vertices) and discards the shape. The
frontend (`Preview.svelte`) then draws `<rect>` overlays from those bboxes.

## Two distinct problems this causes

### 1. Visual fidelity (cosmetic)

Rectangles can't represent rotated or curved lines. On dense or skewed pages,
neighboring line bboxes overlap where the underlying polygons would not. For
straight horizontal text the difference is small; for book photos, marginalia,
rotated scans, it's visibly loose.

### 2. Recognition quality (the real bug)

`engine.rs` crops each line for recognition with `img.crop_imm(min_x, min_y, lw, lh)`
— an axis-aligned rectangle. For a rotated or densely-packed line, that
rectangle's corners can contain ink from neighboring lines, which leaks into
the recognizer input and degrades accuracy.

Kraken's own recognizer uses **`crop_polygon_white_bg`** (in kraken-rust's
`src/recognition/orchestrator.rs:128`), which composites the polygon onto a
white background — masking out everything outside the line's true shape. We
vendored the kraken tree but **dropped `orchestrator.rs`** during the move,
so this helper is not currently available to `engine.rs`.

## Current location of the simplification

- `src-tauri/src/engine.rs` — `polygon_bbox()` (axis-aligned collapse) and
  `img.crop_imm(...)` inside the recognize closure (no polygon masking).
- `LineBox` struct — only `x0, y0, x1, y1`, no polygon field.
- `src/lib/Preview.svelte` — renders `<rect>` from `LineBox` coords.

## Fix path (when we pick this up)

1. **Backend:** extend `LineBox` with `polygon: Option<Vec<[u32;2]>>`.
   Populate for the Kraken path (from `BaselineLine::boundary`); leave `None`
   for the Tesseract full-page path (which has bboxes, not polygons).
2. **Crop masking:** port `crop_polygon_white_bg` from kraken-rust's
   `recognition/orchestrator.rs:128` into `kraken-engine` (it uses
   `imageproc::drawing::draw_polygon_mut`, already a dependency). Use it in
   `engine.rs` for the Kraken recognition path instead of `crop_imm`.
3. **Frontend:** in `Preview.svelte`, render `<polygon points={...}>` when
   `line.polygon` is present, fall back to `<rect>` from bbox otherwise.

## Reference

- kraken-rust `src/recognition/orchestrator.rs:128` — `crop_polygon_white_bg`
  (white-background polygon composite, the correct line-crop behavior).
- kraken-rust `src/recognition/orchestrator.rs:110` — `crop_boundary` (the
  axis-aligned version we currently use).
