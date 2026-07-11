## Plan: Draw hOCR bounding boxes on the preview image

When the selected job was produced in hOCR mode, overlay word-level bounding boxes on the preview image. Boxes come from the hOCR `title='bbox x0 y0 x1 y1; x_wconf N'` attributes on `span.ocrx_word` elements.

### Approach: SVG overlay in image-natural coordinates

The robust, resize-proof technique is to draw boxes in the image's natural pixel space and let an SVG `viewBox` scale them. The only layout math needed is positioning the SVG exactly over the *fitted* image area (since `object-fit: contain` letterboxes inside the `<img>` element).

### 1. `src/lib/hocr.ts` (new, ~30 lines)
A small pure helper:
- `export interface HocrBox { text: string; x0: number; y0: number; x1: number; y1: number; conf: number; }`
- `export function parseHocrWords(hocr: string): { width: number; height: number; boxes: HocrBox[] }`
  - Parse with `new DOMParser().parseFromString(hocr, "text/html")`.
  - Find the page element (`.ocr_page`); read its `title` bbox to get the page's natural width/height (`x1`/`y1` of the page box = image dimensions).
  - Query all `.ocrx_word` elements; from each `title` extract `bbox x0 y0 x1 y1` and `x_wconf N`; collect text from `textContent`.
  - Return the boxes + page dimensions. Defensive: returns empty boxes on any parse failure.

### 2. `src/lib/Preview.svelte` — overlay rendering
- Add an `imgEl` binding on the `<img>` and a `stageEl` binding on `.stage`.
- Add reactive state:
  - `parsed = $derived(job?.outputMode === "hocr" && job.status === "done" ? parseHocrWords(job.text) : null)`
  - `rendered = $state({ left: 0, top: 0, width: 0, height: 0 })` — the fitted image rect within `.stage`.
- A `measure()` function computes the fitted rect:
  - If no `imgEl` or no `parsed`, bail.
  - `const cw = imgEl.clientWidth, ch = imgEl.clientHeight` (the element box).
  - `const nw = parsed.width, nh = parsed.height` (natural image size from the page bbox).
  - Compute the contain fit: `scale = min(cw/nw, ch/nh)`; `drawW = nw*scale`, `drawH = nh*scale`; `left = (cw - drawW)/2`, `top = (ch - drawH)/2` (object-fit: contain centers).
  - Add the img's offset within `.stage` (img.getBoundingClientRect() minus stage.getBoundingClientRect()) to get the rect relative to the stage.
  - Set `rendered = { left, top, width: drawW, height: drawH }` (relative to stage, which is the positioning context).
- A `$effect` re-runs `measure()` whenever `parsed` changes or the img loads; a `ResizeObserver` on `.stage` re-runs `measure()` on panel resize. Also re-measure on `window` resize.
- Markup: wrap the existing `<img>` and (conditionally) an `<svg>` inside a positioned container. The SVG:
  - `style="position:absolute; left:{rendered.left}px; top:{rendered.top}px; width:{rendered.width}px; height:{rendered.height}px; pointer-events:none"`
  - `viewBox="0 0 {parsed.width} {parsed.height}"` — so coordinates are in natural image pixels and scale automatically.
  - One `<rect>` per box: `x={box.x0} y={box.y0} width={box.x1-box.x0} height={box.y1-box.y0}`, `class="bbox"`.
- The stage stays `position: relative` (add it) so the absolutely-positioned SVG anchors to it.
- Hide the overlay cleanly when `parsed` is null (text mode / not done) — just don't render the SVG.

### 3. CSS for boxes (in Preview.svelte `<style>`)
- `.bbox { fill: none; stroke: var(--accent); stroke-width: 2; opacity: 0.65; }`
- Since the viewBox is in natural pixels, `stroke-width` is in natural units too — scale it relative to image size so it stays ~1.5px on screen. Use `vector-effect="non-scaling-stroke"` on each `<rect>` to keep strokes a constant on-screen 1.5px regardless of zoom (cleaner than computing stroke-width).
- A container `.img-wrap` with `position: relative; display: inline-flex` so the SVG aligns to the img. Actually simpler: keep img in the centered flex `.stage`, and position the SVG absolutely relative to `.stage` using the computed `rendered` rect. This avoids wrapping.

### 4. Verify
- `npm run build` (type-check).
- `cargo tauri dev`: run an image in hOCR mode → word boxes overlay the preview, correctly aligned with the scaled image, and stay aligned when resizing the middle panel. Switch back to a text-mode result → no boxes. Switch to a queued/error image → no boxes.

### Not doing
- Not making boxes interactive (click-to-edit, hover tooltips showing the word/confidence) — request was "draw bounding box on preview image". Can extend later; the parsed data already carries text+confidence per box.
- Not adding a toggle to hide boxes — they only render for hOCR results, which is the explicit signal the user wants them. Easy to add a toggle later.
- Not redrawing on image natural-size changes (rotation, etc.) — the image source is fixed per job.