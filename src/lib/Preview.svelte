<script lang="ts">
  import type { Job } from "./ocr";
  import { ensureThumb } from "./ocr";

  interface Props {
    job: Job | null;
  }
  let { job }: Props = $props();

  // Every completed job carries a structured OcrResult in `job.result`, whose
  // `lines` give the overlay boxes and whose `width`/`height` are the source
  // image's natural pixel size. The image and the boxes live in one SVG,
  // sharing a coordinate system that scales as a single unit — no JS
  // measurement, ResizeObserver, or getBoundingClientRect, so nothing can
  // drift when the panel or window is resized.
  let parsed = $derived(
    job?.status === "done" && job.result ? job.result : null,
  );
  let showBoxes = $derived(!!parsed && parsed.lines.length > 0);
  // Whether the line boxes/polys are drawn. Independent of `showBoxes` (which
  // just says boxes EXIST for this result) so hiding keeps zoom/pan state on
  // the SVG intact — only the strokes disappear.
  let overlayVisible = $state(true);

  // ── Zoom ──────────────────────────────────────────────────────────────────
  // null = "fit" (CSS caps the image to the stage). A number is an explicit
  // zoom level where 1 = the image's natural pixel size.
  const ZOOM_STEPS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4];
  let zoom = $state<number | null>(null);

  // Natural pixel dimensions of the current image. The structured result
  // carries the page bbox directly; for plain images we read them on load.
  let natW = $state(0);
  let natH = $state(0);

  // Reset zoom + dims whenever the selected job changes (avoids showing the
  // previous image's dimensions until the new one loads).
  $effect(() => {
    job?.id; // track selection
    zoom = null;
    natW = 0;
    natH = 0;
  });
  // Path-based (PDF page) jobs hold their pixels in a temp file; load the
  // preview image on demand instead of keeping all pages in memory.
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

  // ── Wheel/pinch zoom ───────────────────────────────────────────────────────
  // Ctrl/Cmd + wheel zooms; plain wheel still pans (the existing overflow:auto
  // behavior). Trackpad pinch works automatically: macOS synthesizes it as
  // ctrlKey + wheel, so the one ctrlKey branch catches both pinch and the
  // explicit Ctrl+scroll gesture from a mouse.
  //
  // Zoom is anchored at the cursor: the image-coordinate point under the
  // pointer stays put as the scale changes (standard image-viewer behavior).
  // Without anchoring, wheel-zoom always recenters to top-left and feels
  // broken.
  //
  // `preventDefault` is required so the webview doesn't also do its native
  // page-zoom (Cmd+plus/minus) on the gesture. Svelte's on:wheel may be
  // passive, so we attach a non-passive listener via the wheelZoom action
  // below.
  const WHEEL_K = 0.005; // deltaY → log-scale delta; tuned for trackpad feel
  const ZOOM_MIN = 0.1;
  const ZOOM_MAX = 10;

  function wheelZoom(node: HTMLElement) {
    const handler = (e: WheelEvent) => {
      if (!natW || !natH) return;
      if (!(e.ctrlKey || e.metaKey)) return; // plain wheel = pan (overflow:auto)
      e.preventDefault();
      const stage = node;
      // Hoist rect once — getBoundingClientRect is layout-flush and could
      // return different values if called across a style change.
      const rect = stage.getBoundingClientRect();
      const cursorLeft = e.clientX - rect.left;
      const cursorTop = e.clientY - rect.top;
      // Current scale: explicit `zoom`, or fitZoom() when in fit mode (so the
      // first wheel-zoom from fit is continuous, not a jump).
      const startScale = zoom ?? fitZoom();
      // Image-pixel coord under the cursor BEFORE the zoom.
      const imgX = (stage.scrollLeft + cursorLeft) / startScale;
      const imgY = (stage.scrollTop + cursorTop) / startScale;
      // Multiplicative scaling: exp(-deltaY * k) gives smooth trackpad pinch.
      // Pinch out (deltaY negative) → zoom in.
      const factor = Math.exp(-e.deltaY * WHEEL_K);
      const next = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, startScale * factor));
      zoom = next;
      // Defer the scroll restore until after the DOM has applied the new
      // width/height (Svelte updates the attributes asynchronously). Without
      // rAF, the browser clamps scrollLeft/Top to the OLD (smaller) scrollable
      // range and the cursor anchor drifts.
      requestAnimationFrame(() => {
        stage.scrollLeft = imgX * next - cursorLeft;
        stage.scrollTop = imgY * next - cursorTop;
      });
    };
    // passive:false so preventDefault works (blocks the webview's native zoom).
    node.addEventListener("wheel", handler, { passive: false });
    return {
      destroy() {
        node.removeEventListener("wheel", handler);
      },
    };
  }

  // ── Drag-to-pan (zoomed only) ──────────────────────────────────────────────
  // The zoomed stage already pans via scroll-wheel (overflow:auto). Drag-pan
  // reuses the SAME scroll offset — adjusting scrollTop/scrollLeft on pointer
  // move — so the two mechanisms share one coordinate system and can't fight
  // each other. The SVG (image + boxes) lives inside the stage, so panning
  // moves both together; they cannot drift apart.
  //
  // NOTE on Tauri native drag-drop: the app uses Tauri's onDragDropEvent
  // (App.svelte) for OS-level file drops, which captures pointer motion at the
  // webview boundary. We deliberately do NOT use setPointerCapture here — it
  // conflicts with that drop handling and can leave pointerup swallowed,
  // trapping the cursor in 'grabbing'. Instead we end the pan on pointerup,
  // pointercancel, AND pointerleave so a hijacked gesture can't strand state.
  let isPanning = $state(false);
  // Pan-origin refs are NOT reactive — they don't drive render, so plain lets.
  let panStart: {
    pointerId: number;
    startX: number;
    startY: number;
    scrollLeft: number;
    scrollTop: number;
  } | null = null;
  // Movement below this many pixels doesn't count as a pan — prevents tiny
  // accidental drags from engaging and lets future click targets work.
  const PAN_THRESHOLD = 3;

  function onPanStart(e: PointerEvent) {
    // Only pan when zoomed (overflow:auto is what makes scroll offset meaningful).
    if (zoom === null) return;
    // Only the primary mouse button initiates a pan. Without this check, a
    // right-click (context menu) or middle-click could leave panStart set
    // without a matching release.
    if (e.button !== 0) return;
    // If a pan is already in flight (e.g. a second finger touched mid-drag),
    // ignore the new pointer — the originating pointer owns the pan until it
    // lifts. Without this, panStart would be overwritten and the original
    // pointer's pointerup would end the pan prematurely.
    if (panStart !== null) return;
    const stage = e.currentTarget as HTMLElement;
    panStart = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      scrollLeft: stage.scrollLeft,
      scrollTop: stage.scrollTop,
    };
  }

  function onPanMove(e: PointerEvent) {
    if (!panStart) return;
    const stage = e.currentTarget as HTMLElement;
    const dx = e.clientX - panStart.startX;
    const dy = e.clientY - panStart.startY;
    // Don't flip to "panning" until the cursor moves past the threshold, so a
    // pure click (no drag) leaves isPanning false.
    if (!isPanning && Math.hypot(dx, dy) > PAN_THRESHOLD) isPanning = true;
    if (!isPanning) return;
    stage.scrollLeft = panStart.scrollLeft - dx;
    stage.scrollTop = panStart.scrollTop - dy;
  }

  function onPanEnd(e: PointerEvent) {
    if (!panStart) return;
    // Only the originating pointer ends the pan; a stray pointerup from a
    // second finger must not abort an in-flight drag.
    if (panStart.pointerId !== e.pointerId) return;
    panStart = null;
    isPanning = false;
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
    {#if natW && natH}
      <span class="dims">{natW}×{natH}</span>
    {/if}
    {#if job}
      <span class="status-pill {job.status}">
        {#if job.status === "queued"}Queued
        {:else if job.status === "running"}Recognizing…
        {:else if job.status === "done"}Done{#if job.confidence >= 0} · {job.confidence}% conf{/if}
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
  <!-- svelte-ignore a11y_no_static_element_interactions -- the stage is a
       canvas-like region; pointer handlers implement drag-to-pan when zoomed,
       not an interactive widget. Role=application would hurt screen-reader UX. -->
  <div
    id="preview-stage"
    class="stage"
    class:zoomed={zoom !== null}
    class:panning={isPanning}
    use:wheelZoom
    onpointerdown={onPanStart}
    onpointermove={onPanMove}
    onpointerup={onPanEnd}
    onpointercancel={onPanEnd}
    onpointerleave={onPanEnd}
  >
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
        {#if overlayVisible}
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
        {/if}
      </svg>
    {:else if job}
      <img
        src={job.url}
        alt={job.name}
        class:fit={!renderW}
        width={renderW ?? undefined}
        height={renderH ?? undefined}
        onload={onImgLoad}
        draggable="false"
      />
    {:else}
      <div class="placeholder">
        <div class="ph-icon">▢</div>
        <p>Select an image to preview</p>
      </div>
    {/if}
  </div>
  {#if job && (job.status === "done" || job.status === "running")}
    <div class="status-bar" role="status">
      {#if job.status === "running"}
        <span class="sb-pulse">Recognizing…</span>
      {:else if parsed && parsed.segmentationMs != null && parsed.recognitionMs != null}
        <span>Seg <span class="sb-num">{parsed.segmentationMs}</span> ms</span>
        <span class="sb-sep">·</span>
        <span>Recog <span class="sb-num">{parsed.recognitionMs}</span> ms</span>
        <span class="sb-sep">·</span>
        <span>Total <span class="sb-num">{job.elapsedMs}</span> ms</span>
      {:else if job.status === "done"}
        <span>Done in <span class="sb-num">{job.elapsedMs}</span> ms</span>
      {/if}
      {#if job.status === "done" && showBoxes}
        <span class="sb-sep">·</span>
        <button
          class="sb-toggle"
          class:off={!overlayVisible}
          onclick={() => (overlayVisible = !overlayVisible)}
          title="Show or hide the line boxes"
          aria-pressed={overlayVisible}
        >Boxes</button>
      {/if}
    </div>
  {/if}
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
  .dims {
    font-size: 11px;
    color: var(--text-faint);
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

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
  /* ── Status bar ────────────────────────────────────────────────────────── */
  .status-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 5px 12px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-faint);
    line-height: 1;
    white-space: nowrap;
  }
  .status-bar .sb-num {
    color: var(--text-dim);
    /* `font-variant-numeric` keeps digits the same width so the bar doesn't
       jitter as values change across images. */
    font-variant-numeric: tabular-nums;
  }
  .status-bar .sb-sep {
    color: var(--text-faint);
    opacity: 0.6;
  }
  .status-bar .sb-pulse {
    color: var(--accent);
  }
  /* Overlay visibility toggle — accent when boxes are shown, muted + struck
     styling via the .off class so the state reads at a glance. */
  .status-bar .sb-toggle {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 2px 8px;
    line-height: 1;
  }
  .status-bar .sb-toggle:hover {
    border-color: var(--accent-dim);
  }
  .status-bar .sb-toggle.off {
    color: var(--text-faint);
    background: var(--surface);
    border-color: var(--border);
  }
  /* When zoomed in, allow panning and align to top-left so scroll origin is
     the image corner. Cursor is 'grab' here (and 'grabbing' while a drag is
     active — see the .panning rule) to signal that the image can be dragged.
     touch-action: none stops the browser from hijacking touch drags as a
     page-scroll/zoom gesture that would fight our pointer handlers. */
  .stage.zoomed {
    overflow: auto;
    align-items: flex-start;
    justify-content: flex-start;
    cursor: grab;
    touch-action: none;
  }
  .stage.zoomed.panning {
    cursor: grabbing;
    /* Prevent text/box selection while dragging. */
    user-select: none;
  }
  .stage img,
  .ocr-canvas {
    border-radius: 4px;
    display: block;
    flex-shrink: 0;
    /* Images are draggable=true by default in HTML — a press-and-drag on the
       preview image starts a native ghost-image drag that preempts pointer
       events (so pan never engages) in BOTH fit and zoomed modes. Disable it
       via attribute (draggable=false) on the <img> and via CSS here for the
       SVG <image>. -webkit-user-drag covers WebKit/Chromium; the standard
       alias user-drag is included for forward-compat. */
    -webkit-user-drag: none;
    user-drag: none;
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
