<script lang="ts">
  import type { Job } from "./ocr";
  import { parseHocrWords } from "./hocr";

  interface Props {
    job: Job | null;
  }
  let { job }: Props = $props();

  // When the job was produced in hOCR mode, embed the image inside the same SVG
  // as the bounding boxes. Both then share one coordinate system and scale as a
  // single unit — no JS measurement, ResizeObserver, or getBoundingClientRect,
  // so nothing can drift when the panel or window is resized.
  let parsed = $derived(
    job?.outputMode === "hocr" && job.status === "done" && job.text
      ? parseHocrWords(job.text)
      : null
  );
  let showBoxes = $derived(!!parsed && parsed.boxes.length > 0);
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
  <div class="stage">
    {#if showBoxes && parsed}
      <!-- Image + boxes in one SVG: they cannot drift apart. -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <svg
        class="ocr-canvas"
        viewBox="0 0 {parsed.width} {parsed.height}"
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
      <img src={job.url} alt={job.name} />
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
    /* OCR backend reads raw pixels (no EXIF rotation); match that here. */
    image-orientation: none;
    border-radius: 4px;
  }
  /* The SVG scales like an image: max-width/height cap it within the stage
     while the viewBox preserves aspect ratio, so image and boxes resize
     together with the panel. */
  .ocr-canvas {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    display: block;
    border-radius: 4px;
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
