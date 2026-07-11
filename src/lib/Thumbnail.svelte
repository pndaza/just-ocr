<script lang="ts">
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

  function handleScroll(e: Event) {
    scrollTop = (e.currentTarget as HTMLDivElement).scrollTop;
  }

  $effect(() => {
    // Track viewport height + content width for virtualization and aspect ratio.
    const el = container;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      viewportH = el.clientHeight;
      contentW = el.clientWidth - 16; // minus row horizontal padding (6+8+8)
    });
    ro.observe(el);
    viewportH = el.clientHeight;
    contentW = el.clientWidth - 16;
    return () => ro.disconnect();
  });

  // Keep the selected row in view when selection changes from outside.
  $effect(() => {
    const id = selectedId;
    if (id === null || !container) return;
    const idx = jobs.findIndex((j) => j.id === id);
    if (idx === -1) return;
    const top = idx * ROW_H;
    const bottom = top + ROW_H;
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
    <span class="title">Images</span>
    {#if jobs.length}
      <span class="count">{jobs.length}</span>
      <button class="text-btn clear" onclick={onclear} title="Remove all">Clear</button>
    {/if}
    <button class="text-btn add" onclick={() => input.click()} title="Add images">+ Add</button>
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
              {#if job.status === "done"}
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
    accept="image/*"
    multiple
    onchange={(e) => e.currentTarget.files && onfiles(e.currentTarget.files)}
    hidden
  />
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
    gap: 6px;
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin-right: auto;
  }
  .count {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text-dim);
    background: var(--surface);
    padding: 1px 6px;
    border-radius: 4px;
  }
  .text-btn {
    background: none;
    border: none;
    color: var(--text-faint);
    font-size: 11px;
    padding: 2px 4px;
    border-radius: 4px;
  }
  .text-btn:hover {
    color: var(--accent);
    background: var(--accent-soft);
  }
  .clear { margin-left: auto; }

  .scroller {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
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
  .vrow.sel {
    background: var(--accent-soft);
    border-left-color: var(--accent);
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
    box-shadow: 0 0 0 2px var(--overlay);
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
    background: var(--overlay);
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
  .vrow.sel .name { color: var(--accent); }
  .remove {
    position: absolute;
    top: 4px;
    right: 4px;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--overlay);
    backdrop-filter: blur(3px);
    border: none;
    border-radius: 5px;
    color: var(--text-dim);
    font-size: 11px;
    opacity: 0;
    transition: opacity 0.1s;
  }
  .vrow:hover .remove,
  .vrow.sel .remove { opacity: 1; }
  .remove:hover {
    color: var(--danger);
    background: var(--overlay);
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
</style>
