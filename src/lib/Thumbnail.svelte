<script lang="ts">
  import { ensureThumb, exportImages, type ImageExportFormat } from "./ocr";
  import type { Job } from "./ocr";

  interface Props {
    jobs: Job[];
    selectedId: number | null;
    onselect: (id: number) => void;
    onfiles: (files: FileList) => void;
    onremove: (id: number) => void;
    onclear: () => void;
  }
  let {
    jobs,
    selectedId,
    onselect,
    onfiles,
    onremove,
    onclear,
  }: Props = $props();

  let dragging = $state(false);
  let input: HTMLInputElement;

  // ── Export images to folder ───────────────────────────────────────────────
  // The format popover ("PNG or JPG?") opens from the bottom-bar button; the
  // pick kicks off exportImages (folder dialog → write/convert per image).
  let fmtOpen = $state(false);
  // Progress while exporting: null when idle, else processed/total; `doneMsg`
  // holds a short-lived "✓ n exported" confirmation afterwards.
  let exporting = $state<{ done: number; total: number } | null>(null);
  let doneMsg = $state("");

  async function pickFormat(fmt: ImageExportFormat) {
    fmtOpen = false;
    if (exporting) return;
    exporting = { done: 0, total: jobs.length };
    doneMsg = "";
    try {
      const n = await exportImages(jobs, fmt, (done, total) => {
        exporting = { done, total };
      });
      if (n) {
        doneMsg = `✓ ${n} exported`;
        setTimeout(() => (doneMsg = ""), 2500);
      }
    } catch (e) {
      console.warn("Image export failed:", e);
    }
    exporting = null;
  }

  function drop(e: DragEvent) {
    e.preventDefault();
    dragging = false;
    if (e.dataTransfer?.files?.length) onfiles(e.dataTransfer.files);
  }

  // ── Virtual scrolling ────────────────────────────────────────────────────
  // Only render the rows visible in the scroll viewport + an overscan margin.
  // This keeps the DOM light even with thousands of images.
  const NAME_AREA = 24; // px — filename line + gap + row padding
  const OVERSCAN = 6;
  let contentW = $state(160); // inner content width (drives thumb height)
  let scrollTop = $state(0);
  let viewportH = $state(600);

  let container: HTMLDivElement;

  // 3:4 aspect ratio (portrait, common for book pages).
  // Row height = thumb height (width * 4/3) + filename/padding.
  let rowH = $derived(Math.round(contentW * (4 / 3)) + NAME_AREA);

  let total = $derived(jobs.length);
  let startIdx = $derived(Math.max(0, Math.floor(scrollTop / rowH) - OVERSCAN));
  let endIdx = $derived(
    Math.min(total, Math.ceil((scrollTop + viewportH) / rowH) + OVERSCAN)
  );
  let visible = $derived(jobs.slice(startIdx, endIdx));
  let spacerTop = $derived(startIdx * rowH);
  let spacerBottom = $derived((total - endIdx) * rowH);

  // Lazily load thumbnails for path-based jobs (PDF pages) as their rows become
  // visible. Only the on-screen rows are fetched, so we never ship all page
  // images at once. ensureThumb is idempotent (skips jobs that already have a URL).
  $effect(() => {
    for (const job of visible) {
      if (job.path && !job.url) ensureThumb(job);
    }
  });

  function handleScroll(e: Event) {
    scrollTop = (e.currentTarget as HTMLDivElement).scrollTop;
  }

  // Measure with offsetWidth (border-box), NOT clientWidth. clientWidth
  // loses the 10px styled scrollbar (::-webkit-scrollbar in styles.css)
  // whenever the rows overflow, and rowH feeds straight back into the total
  // scroll height — so for job counts whose content height sits near the
  // viewport height the scrollbar toggles every frame: scrollbar appears →
  // clientWidth drops → rowH shrinks → content fits → scrollbar hides → …
  // That endless toggle resized every thumbnail each frame (visible as
  // flicker until the panel was resized out of the unstable geometry).
  // offsetWidth includes the scrollbar strip, so it is identical whether or
  // not the scrollbar is showing — the loop can't start. The scroller also
  // reserves its gutter (scrollbar-gutter: stable) so thumbs don't jump 10px
  // when the queue starts/stops overflowing; where that property is
  // unsupported, this measurement alone still holds.
  function measureWidth(el: HTMLElement) {
    return el.offsetWidth - 16; // minus row horizontal padding (6+8+8)
  }

  $effect(() => {
    // Track viewport height + content width for virtualization and aspect ratio.
    const el = container;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      viewportH = el.clientHeight;
      contentW = measureWidth(el);
    });
    ro.observe(el);
    viewportH = el.clientHeight;
    contentW = measureWidth(el);
    return () => ro.disconnect();
  });

  // Keep the selected row in view when selection changes from outside.
  $effect(() => {
    const id = selectedId;
    if (id === null || !container) return;
    const idx = jobs.findIndex((j) => j.id === id);
    if (idx === -1) return;
    const top = idx * rowH;
    const bottom = top + rowH;
    if (top < container.scrollTop) {
      container.scrollTop = top;
    } else if (bottom > container.scrollTop + container.clientHeight) {
      container.scrollTop = bottom - container.clientHeight;
    }
  });
</script>

<div
  class="panel"
  class:dragging
  ondrop={drop}
  ondragover={(e) => {
    e.preventDefault();
    dragging = true;
  }}
  ondragleave={() => (dragging = false)}
  role="region"
  aria-label="Image queue"
>
  <div class="head">
    <button class="text-btn add" onclick={() => input.click()} title="Add images">+ Add</button>
    {#if jobs.length}
      <span class="count">{jobs.length}</span>
      <button class="text-btn clear" onclick={onclear} title="Remove all">Clear</button>
    {/if}
  </div>

  <div class="scroller" bind:this={container} onscroll={handleScroll}>
    {#if jobs.length === 0}
      <div class="empty">
        <div class="empty-icon">⬆</div>
        <div class="empty-title">Drop images</div>
        <div class="empty-sub">or click + Add</div>
      </div>
    {:else}
      <div class="inner" style="height:{total * rowH}px">
        <div style="height:{spacerTop}px"></div>
        {#each visible as job (job.id)}
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
          <div
            class="vrow"
            class:sel={job.id === selectedId}
            style="height:{rowH}px"
            role="button"
            tabindex="0"
            onclick={() => onselect(job.id)}
            title={job.name}
          >
            <div class="thumb-wrap">
              <img src={job.url} alt={job.name} loading="lazy" decoding="async" />
              {#if job.status === "done" && job.confidence >= 0}
                <span class="badge conf">{job.confidence}%</span>
              {:else if job.status === "error"}
                <span class="badge err">!</span>
              {/if}
              {#if job.id === selectedId && job.status === "running"}
                <span class="tile-spin" aria-hidden="true"></span>
              {/if}
              <span class="status-dot {job.status}"></span>
              <button
                class="remove"
                onclick={(e) => { e.stopPropagation(); onremove(job.id); }}
                title="Remove"
                aria-label="Remove image"
              >✕</button>
            </div>
            <span class="name">{job.name}</span>
          </div>
        {/each}
        <div style="height:{spacerBottom}px"></div>
      </div>
    {/if}
  </div>

  <input
    bind:this={input}
    type="file"
    accept="image/*,.pdf,application/pdf"
    multiple
    onchange={(e) => e.currentTarget.files && onfiles(e.currentTarget.files)}
    hidden
  />

  {#if jobs.length}
    <div class="foot">
      {#if exporting}
        <span class="exp-status">Exporting {exporting.done}/{exporting.total}…</span>
      {:else if doneMsg}
        <span class="exp-status ok">{doneMsg}</span>
      {:else}
        <span class="exp-hint">{jobs.length} image{jobs.length === 1 ? "" : "s"} in queue</span>
        <div class="exp-wrap">
          {#if fmtOpen}
            <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
            <div class="pop-backdrop" onclick={() => (fmtOpen = false)}></div>
            <div class="fmt-pop" role="menu" aria-label="Export image format">
              <button class="fmt-btn" role="menuitem" onclick={() => pickFormat("png")}>
                <span class="fmt-name">PNG</span>
                <span class="fmt-sub">lossless, larger</span>
              </button>
              <button class="fmt-btn" role="menuitem" onclick={() => pickFormat("jpg")}>
                <span class="fmt-name">JPG</span>
                <span class="fmt-sub">smaller files</span>
              </button>
            </div>
          {/if}
          <button
            class="text-btn export"
            onclick={() => (fmtOpen = !fmtOpen)}
            title="Export images to a folder"
          >⤓ Export images</button>
        </div>
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
    justify-content: space-between;
    gap: 6px;
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  /* With the "Images" label gone, the header holds only the count + buttons
     (or just "+ Add" when empty). Pin the whole group to the right. */
  .count {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text-dim);
    background: var(--surface);
    padding: 1px 6px;
    border-radius: 4px;
  }
  /* Outlined control buttons — readable against the header. The old
     borderless text buttons disappeared into the chrome. */
  .text-btn {
    display: inline-flex;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    color: var(--text-dim);
    font-size: 11px;
    font-weight: 500;
    line-height: 1;
    padding: 4px 8px;
    border-radius: 5px;
    transition: background 0.1s, border-color 0.1s, color 0.1s;
  }
  .text-btn:hover {
    color: var(--text);
    background: var(--bg-elev);
    border-color: var(--accent-dim);
  }
  /* "+ Add" is the primary action — the empty state points users here, so
     accent-tint it at rest and fill solid on hover. */
  .add {
    color: var(--accent);
    background: var(--accent-soft);
    border-color: var(--accent-dim);
  }
  .add:hover {
    color: var(--bg);
    background: var(--accent);
    border-color: var(--accent);
  }
  /* "Clear" is destructive: neutral outline at rest, warns danger on hover. */
  .clear:hover {
    color: var(--danger);
    background: var(--danger-soft);
    border-color: var(--danger);
  }

  .scroller {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    /* Always reserve the scrollbar strip so row/thumb width stays constant
       when the queue crosses the overflow threshold (and the ResizeObserver
       measurement above stays quiet). Ignored on engines without support —
       the offsetWidth measurement is what breaks the feedback loop there. */
    scrollbar-gutter: stable;
  }
  .panel.dragging .scroller {
    outline: 2px dashed var(--accent-dim);
    outline-offset: -8px;
    border-radius: 8px;
  }

  .vrow {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    padding: 6px 8px;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: background 0.08s;
  }
  .vrow:hover {
    background: var(--bg-inset);
  }
  /* The row is focusable (tabindex=0) for clicks, but the native focus outline
     is redundant with our selection ring and reads as a stray blue border.
     Suppress it; selection is shown by .vrow.sel .thumb-wrap. */
  .vrow:focus {
    outline: none;
  }
  .vrow.sel .thumb-wrap {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .thumb-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 3 / 4;
    border-radius: 7px;
    overflow: hidden;
    background: var(--bg-inset);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .thumb-wrap img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }
  .status-dot {
    position: absolute;
    top: 5px;
    left: 5px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-faint);
  }
  .status-dot.queued { background: var(--text-faint); }
  .status-dot.running { background: var(--accent); animation: pulse 1.2s infinite; }
  .status-dot.done { background: var(--ok); }
  .status-dot.error { background: var(--danger); }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .badge {
    position: absolute;
    bottom: 5px;
    left: 5px;
    font-size: 10px;
    font-family: var(--mono);
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--badge-bg);
    backdrop-filter: blur(3px);
  }
  .badge.conf { color: var(--ok); }
  .badge.err { color: var(--danger); font-weight: 700; }
  .tile-spin {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 20px;
    height: 20px;
    border: 2px solid var(--accent-soft);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  .name {
    font-size: 11px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 4px;
    padding: 0 1px;
  }
  .vrow.sel .name {
    color: var(--accent);
    font-weight: 600;
  }
  .remove {
    position: absolute;
    top: 4px;
    right: 4px;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--badge-bg);
    backdrop-filter: blur(3px);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    color: var(--text);
    font-size: 11px;
    opacity: 0;
    transition: opacity 0.1s;
  }
  .vrow:hover .remove,
  .vrow.sel .remove { opacity: 1; }
  .remove:hover {
    color: var(--danger);
    background: var(--badge-bg);
    border-color: var(--danger);
  }
  @keyframes spin { to { transform: translate(-50%, -50%) rotate(360deg); } }

  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-height: 200px;
    color: var(--text-faint);
    text-align: center;
    gap: 3px;
  }
  .empty-icon { font-size: 28px; color: var(--accent); }
  .empty-title { font-weight: 600; color: var(--text-dim); font-size: 13px; }
  .empty-sub { font-size: 11px; }

  /* Bottom bar: queue summary at left, image-export button at right. The
     format popover anchors to the button via .exp-wrap (position: relative). */
  .foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    padding: 8px 12px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
    min-height: 34px;
  }
  .exp-hint {
    font-size: 11px;
    color: var(--text-faint);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .exp-status {
    font-size: 11px;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .exp-status.ok { color: var(--ok); }
  .exp-wrap {
    position: relative;
    display: inline-flex;
    margin-left: auto;
  }
  .export {
    white-space: nowrap;
  }
  .pop-backdrop {
    position: fixed;
    inset: 0;
    z-index: 40;
  }
  .fmt-pop {
    position: absolute;
    bottom: calc(100% + 6px);
    right: 0;
    z-index: 41;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 148px;
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    padding: 4px;
    box-shadow: 0 12px 32px var(--overlay);
  }
  .fmt-btn {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1px;
    padding: 6px 9px;
    border-radius: 6px;
    border: none;
    background: none;
    text-align: left;
  }
  .fmt-btn:hover {
    background: var(--accent-soft);
  }
  .fmt-name {
    font-size: 12px;
    font-weight: 700;
    color: var(--text);
  }
  .fmt-sub {
    font-size: 10px;
    color: var(--text-faint);
  }
</style>
