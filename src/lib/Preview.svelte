<script lang="ts">
  import type { Job } from "./ocr";
  import { parseHocrWords } from "./hocr";

  interface Props {
    job: Job | null;
  }
  let { job }: Props = $props();

  // ── hOCR bounding-box overlay ─────────────────────────────────────────────
  // When the selected job was produced in hOCR mode, parse its word boxes and
  // draw them as an SVG layer positioned exactly over the fitted image.
  let imgEl = $state<HTMLImageElement | null>(null);
  let stageEl = $state<HTMLDivElement | null>(null);

  let parsed = $derived(
    job?.outputMode === "hocr" && job.status === "done" && job.text
      ? parseHocrWords(job.text)
      : null
  );
  let showBoxes = $derived(!!parsed && parsed.boxes.length > 0);

  // Rendered image rect, relative to the stage. The img has no object-fit
  // (max-width/height already preserve aspect ratio for auto-sized replaced
  // elements), so its element box IS the drawn image — no letterbox math
  // needed, just getBoundingClientRect relative to the stage.
  let rect = $state({ left: 0, top: 0, width: 0, height: 0 });

  function measure() {
    const img = imgEl;
    const stage = stageEl;
    if (!img || !stage) return;
    const ir = img.getBoundingClientRect();
    const sr = stage.getBoundingClientRect();
    rect = {
      left: ir.left - sr.left,
      top: ir.top - sr.top,
      width: ir.width,
      height: ir.height,
    };
  }

  // Re-measure when the parsed result or image element changes.
  $effect(() => {
    parsed; // depend on parsed
    const img = imgEl;
    if (!img) return;
    if (img.complete) measure();
    const onLoad = () => measure();
    img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
  });

  // Re-measure on stage resize (panel dragged / window resized).
  $effect(() => {
    const stage = stageEl;
    if (!stage) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(stage);
    return () => ro.disconnect();
  });
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
  </div>
  <div class="stage" bind:this={stageEl}>
    {#if job}
      <img src={job.url} alt={job.name} bind:this={imgEl} />
    {:else}
      <div class="placeholder">
        <div class="ph-icon">▢</div>
        <p>Select an image to preview</p>
      </div>
    {/if}

    {#if showBoxes && parsed}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <svg
        class="bbox-layer"
        style="left:{rect.left}px; top:{rect.top}px; width:{rect.width}px; height:{rect.height}px;"
        viewBox="0 0 {parsed.width} {parsed.height}"
      >
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
    max-width: 60%;
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
  .stage {
    position: relative;
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    padding: 20px;
    background: var(--bg-inset);
  }
  .stage img {
    max-width: 100%;
    max-height: 100%;
    /* The OCR backend reads raw pixels without applying EXIF orientation, so
       the hOCR bbox coordinates are in the un-rotated pixel space. Force the
       browser to show the same orientation so boxes line up with the image. */
    image-orientation: none;
    border-radius: 4px;
  }
  .bbox-layer {
    position: absolute;
    pointer-events: none;
    overflow: visible;
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
