<script lang="ts">
  import type { Job } from "./ocr";
  import { ensureThumb } from "./ocr";
  import { parseHocrLines } from "./hocr";

  interface Props {
    job: Job | null;
  }
  let { job }: Props = $props();

  // Every completed job stores hOCR XML in `job.text`, so we can overlay
  // bounding boxes on the preview regardless of output mode. The image and the
  // boxes live in one SVG, sharing a coordinate system that scales as a single
  // unit — no JS measurement, ResizeObserver, or getBoundingClientRect, so
  // nothing can drift when the panel or window is resized.
  let parsed = $derived(
    job?.status === "done" && job.text ? parseHocrLines(job.text) : null
  );
  let showBoxes = $derived(!!parsed && parsed.boxes.length > 0);

  // ── Zoom ──────────────────────────────────────────────────────────────────
  // null = "fit" (CSS caps the image to the stage). A number is an explicit
  // zoom level where 1 = the image's natural pixel size.
  const ZOOM_STEPS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4];
  let zoom = $state<number | null>(null);

  // Natural pixel dimensions of the current image. For hOCR they come from the
  // parsed page bbox; for plain images we read them on load.
  let natW = $state(0);
  let natH = $state(0);

  // Reset zoom whenever the selected job changes.
  $effect(() => {
    job?.id; // track selection
    zoom = null;
  });
  // Path-based (PDF page) jobs hold their pixels in a temp file; load the
  // preview image on demand instead of keeping all pages in memory.
  $effect(() => {
    if (job?.path && !job.url) ensureThumb(job);
  });
  // hOCR mode: dimensions are known immediately from the page bbox.
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

  // Pixel size to render at, or null for "fit".
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
  // Rough estimate of the fit zoom so the in/out buttons step sensibly from
  // the current view. Good enough for picking the next step; exact fit is
  // handled by CSS when zoom is null.
  function fitZoom(): number {
    const stage = document.getElementById("preview-stage");
    if (!stage || !natW || !natH) return 1;
    const pad = 40; // stage padding * 2
    return Math.min(
      (stage.clientWidth - pad) / natW,
      (stage.clientHeight - pad) / natH
    );
  }
</script>

<div class="panel" role="region" aria-label="Image preview">
  <div class="head">
    <span class="title">{job ? job.name : "Preview"}</span>
    {#if job}
      <span class="status-pill {job.status}">
        {#if job.status === "queued"}Queued
        {:else if job.status === "running"}Recognizing…
        {:else if job.status === "done"}Done · {job.confidence}% conf
        {:else if job.status === "error"}Error{/if}
      </span>
    {/if}
    {#if job}
      <div class="zoom-controls">
        <button
          class="zoom-btn"
          onclick={zoomOut}
          title="Zoom out"
          aria-label="Zoom out"
          disabled={!natW}
        >−</button>
        <button class="zoom-label" onclick={resetZoom} title="Reset to fit">
          {zoomLabel}
        </button>
        <button
          class="zoom-btn"
          onclick={zoomIn}
          title="Zoom in"
          aria-label="Zoom in"
          disabled={!natW}
        >+</button>
      </div>
    {/if}
  </div>
  <div id="preview-stage" class="stage" class:zoomed={zoom !== null}>
    {#if showBoxes && parsed}
      <!-- Image + boxes in one SVG: they cannot drift apart. -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
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
        {#each parsed.boxes as b}
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
    {:else if job}
      <img
        src={job.url}
        alt={job.name}
        class:fit={!renderW}
        width={renderW ?? undefined}
        height={renderH ?? undefined}
        onload={onImgLoad}
      />
    {:else}
      <div class="placeholder">
        <div class="ph-icon">▢</div>
        <p>Select an image to preview</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    border-right: 1px solid var(--border);
  }
  .head {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
  }
  .title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin-right: auto;
    font-family: var(--mono);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 50%;
  }
  .status-pill {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--surface);
    color: var(--text-dim);
  }
  .status-pill.done { color: var(--ok); background: var(--ok-soft); }
  .status-pill.running { color: var(--accent); background: var(--accent-soft); }
  .status-pill.error { color: var(--danger); background: var(--danger-soft); }

  /* ── Zoom controls ─────────────────────────────────────────────────────── */
  .zoom-controls {
    display: flex;
    align-items: center;
    gap: 2px;
    margin-left: 4px;
  }
  .zoom-btn,
  .zoom-label {
    font-size: 12px;
    font-family: var(--mono);
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-dim);
    height: 24px;
    line-height: 1;
  }
  .zoom-btn {
    width: 24px;
    border-radius: 5px 0 0 5px;
    padding: 0;
  }
  .zoom-btn:last-child {
    border-radius: 0 5px 5px 0;
  }
  .zoom-label {
    min-width: 44px;
    border-left: none;
    border-right: none;
    padding: 0 6px;
    cursor: pointer;
  }
  .zoom-btn:hover:not(:disabled),
  .zoom-label:hover {
    color: var(--accent);
    border-color: var(--accent-dim);
    background: var(--accent-soft);
  }
  /* Keep the shared hover border-color on the middle button's neighbors. */
  .zoom-btn:hover:not(:disabled) + .zoom-label {
    border-left-color: var(--accent-dim);
  }
  .zoom-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* ── Stage ─────────────────────────────────────────────────────────────── */
  .stage {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    padding: 20px;
    background: var(--bg-inset);
  }
  /* When zoomed in, allow panning and align to top-left so scroll origin is
     the image corner. */
  .stage.zoomed {
    overflow: auto;
    align-items: flex-start;
    justify-content: flex-start;
  }
  .stage img,
  .ocr-canvas {
    border-radius: 4px;
    display: block;
    flex-shrink: 0;
  }
  /* "fit" mode: CSS caps the element to the stage, preserving aspect ratio. */
  .stage img.fit {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    image-orientation: none;
  }
  .stage .ocr-canvas.fit {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
  }
  /* Explicit zoom: fixed pixel size set via width/height attributes; CSS
     max-width/max-height must not interfere, so .fit is absent. */
  .stage img:not(.fit),
  .stage .ocr-canvas:not(.fit) {
    max-width: none;
    max-height: none;
  }
  .bbox {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.5;
    opacity: 0.6;
  }
  .placeholder {
    color: var(--text-faint);
    text-align: center;
  }
  .ph-icon {
    font-size: 40px;
    color: var(--border-strong);
    margin-bottom: 8px;
  }
</style>
