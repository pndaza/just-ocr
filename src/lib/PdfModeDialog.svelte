<script lang="ts">
  import type { ImageMode, PdfMode } from "./ocr";

  // Color format for the per-page PNGs. Grayscale is the default (smaller,
  // no accuracy loss — recognizers binarize internally); Color keeps the
  // source as-is.
  const imageModes: { value: ImageMode; label: string; hint: string }[] = [
    { value: "color", label: "Color", hint: "Keep the source as-is" },
    { value: "gray", label: "Gray", hint: "Best for OCR (default)" },
  ];

  // Page-height choices (px). "none" = native resolution, extract only —
  // very high-res scans produce line heights the segmentation models handle
  // poorly, so a bounded downscale often OCRs better. Render mode rasterizes
  // at a fixed height, so it has no "original" to keep and none isn't offered.
  const heights = [1000, 1200, 1400, 1600, 1800, 2000];

  // Long-form mode explanations, shown on hover of the ⓘ icon next to each
  // segment label (the segments themselves stay terse).
  const extractTip =
    "Pulls the embedded scan image straight out of the PDF — fast, and keeps " +
    "the scan's own pixels (capped by the height above). Best for scanned PDFs.";
  const renderTip =
    "Rasterizes each page from its text and vector content at the chosen " +
    "height — slower, but works for PDFs with no embedded image (vector " +
    "text, mixed content).";

  interface Props {
    /** The PDF file name being processed (shown in the dialog). */
    name: string;
    /** "choosing" shows the mode buttons; "working" shows the progress UI. */
    status: "choosing" | "working";
    /** The mode the user picked (set once they hit Process). */
    mode: PdfMode | null;
    /** Pages processed so far (from the backend pdf-progress event). */
    done: number;
    /** Total page count (0 until the backend reports it). */
    total: number;
    /** Called with the chosen mode + image format + page height when the user
     *  hits Process. `maxHeight` is null only for extract (native size). */
    onprocess: (mode: PdfMode, imageMode: ImageMode, maxHeight: number | null) => void;
    /** Called when the user cancels (backdrop click, Cancel button, or Esc). */
    oncancel: () => void;
  }
  let { name, status, mode, done, total, onprocess, oncancel }: Props = $props();

  // Per-PDF image format; defaults to grayscale, the OCR-friendly choice.
  let imageMode = $state<ImageMode>("gray");

  // Mode + height are picked first, then confirmed with Process (the height
  // dropdown depends on the mode, so clicking a mode can't start immediately).
  // Extract is the default — it's the right choice for most (scanned) PDFs.
  let selectedMode = $state<PdfMode | null>("extract");
  // 1600px default for both modes: high-res scans hurt segmentation in
  // extract, and render needs a fixed height anyway. "None" stays available
  // for extract users who want the native scan.
  let heightSel = $state("1600");
  let maxHeight = $derived(heightSel === "none" ? null : Number(heightSel));
  let isRender = $derived(selectedMode === "render");

  function selectMode(m: PdfMode) {
    selectedMode = m;
    // Render can't keep native size and its semantics differ from extract's
    // cap, so selecting it always resets to the 1600px default — the "None"
    // option isn't rendered in render mode, and a height picked for extract
    // doesn't silently carry over.
    if (m === "render") {
      heightSel = "1600";
    }
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      oncancel();
    }
  }

  // Progress fraction in [0, 1]; 0 until the backend reports a total.
  let pct = $derived(total > 0 ? Math.min(1, done / total) : 0);

  // Verb shown in the working state, matching the chosen mode.
  let verb = $derived(mode === "render" ? "Rendering" : "Extracting");
  let statusText = $derived(
    total > 0
      ? `${verb} page ${done} of ${total}…`
      : `Preparing “${name}”…`,
  );
</script>

<svelte:window onkeydown={onKey} />

{#snippet infoIcon(tip: string)}
  <span class="info" role="img" aria-label={tip} data-tip={tip}>
    <svg
      viewBox="0 0 16 16"
      width="13"
      height="13"
      aria-hidden="true"
      fill="none"
    >
      <circle cx="8" cy="8" r="6.7" stroke="currentColor" stroke-width="1.4" />
      <circle cx="8" cy="4.9" r="0.95" fill="currentColor" />
      <line
        x1="8"
        y1="7.1"
        x2="8"
        y2="11.6"
        stroke="currentColor"
        stroke-width="1.4"
        stroke-linecap="round"
      />
    </svg>
  </span>
{/snippet}

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="backdrop" onclick={oncancel} role="presentation">
  <!-- svelte-ignore a11y_no_static_element_interactions, a11y_interactive_supports_focus -->
  <div
    class="modal"
    onclick={(e) => e.stopPropagation()}
    role="dialog"
    aria-modal="true"
    aria-label="Choose how to process this PDF"
  >
    <h2>Process “{name}”</h2>

    {#if status === "choosing"}
      <p class="sub">How should this PDF be turned into images for OCR?</p>

      <div class="mode-row">
        <span class="lbl">Mode</span>
        <div class="seg mode-seg" role="radiogroup" aria-label="PDF processing mode">
          <button
            class="seg-btn"
            class:active={selectedMode === "extract"}
            onclick={() => selectMode("extract")}
            role="radio"
            aria-checked={selectedMode === "extract"}
          >
            Extract
            {@render infoIcon(extractTip)}
          </button>
          <button
            class="seg-btn"
            class:active={selectedMode === "render"}
            onclick={() => selectMode("render")}
            role="radio"
            aria-checked={selectedMode === "render"}
          >
            Render
            {@render infoIcon(renderTip)}
          </button>
        </div>
      </div>

      <div class="height-row">
        <span class="lbl">Page height</span>
        <select class="select" bind:value={heightSel} disabled={!selectedMode}>
          {#if !isRender}
            <option value="none">None (native)</option>
          {/if}
          {#each heights as h (h)}
            <option value={String(h)}>{h} px</option>
          {/each}
        </select>
      </div>

      <div class="image-mode">
        <span class="lbl">Image format</span>
        <div class="seg" role="radiogroup" aria-label="PDF page image format">
          {#each imageModes as m}
            <button
              class="seg-btn"
              class:active={imageMode === m.value}
              onclick={() => (imageMode = m.value)}
              role="radio"
              aria-checked={imageMode === m.value}
              title={m.hint}
            >{m.label}</button>
          {/each}
        </div>
        <span class="seg-hint">{imageModes.find((m) => m.value === imageMode)?.hint}</span>
      </div>

      <div class="actions">
        <button class="cancel" onclick={oncancel}>Cancel</button>
        <button
          class="primary"
          disabled={!selectedMode}
          onclick={() => selectedMode && onprocess(selectedMode, imageMode, maxHeight)}
        >Process</button>
      </div>
    {:else}
      <p class="sub">{statusText}</p>

      <div class="progress" aria-hidden="true">
        <div class="bar" style="width:{(pct * 100).toFixed(1)}%"></div>
      </div>
      <div class="meta">
        <span class="count">{total > 0 ? `${done} / ${total} pages` : "Working…"}</span>
        <span class="pct">{total > 0 ? `${Math.round(pct * 100)}%` : ""}</span>
      </div>

      <div class="actions">
        <button class="cancel" onclick={oncancel}>Cancel</button>
      </div>
    {/if}
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: var(--overlay);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }
  .modal {
    width: min(440px, 100%);
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 24px 70px var(--overlay);
  }
  h2 {
    margin: 0 0 4px;
    font-size: 16px;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .sub {
    margin: 0 0 18px;
    font-size: 13px;
    color: var(--text-dim);
  }
  .image-mode {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 16px 0 0;
  }
  .lbl {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-faint);
  }
  .seg {
    display: flex;
    background: var(--bg-inset);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 2px;
    gap: 1px;
  }
  .seg-btn {
    background: none;
    border: none;
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 5px;
    color: var(--text-faint);
    font-weight: 600;
  }
  .seg-btn:hover {
    color: var(--text-dim);
  }
  .seg-btn.active {
    background: var(--accent-dim);
    color: var(--bg);
  }
  .seg-hint {
    margin-left: auto;
    font-size: 11px;
    color: var(--text-faint);
  }
  .mode-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 16px;
  }
  .mode-seg {
    flex: 1;
  }
  .mode-seg .seg-btn {
    flex: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  .info {
    position: relative;
    display: inline-flex;
    color: var(--text-faint);
    cursor: help;
  }
  .seg-btn.active .info {
    color: var(--bg);
  }
  .info::after {
    content: attr(data-tip);
    position: absolute;
    top: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    width: max-content;
    max-width: 230px;
    background: var(--text);
    color: var(--bg-elev);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 400;
    line-height: 1.45;
    text-align: left;
    white-space: normal;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.12s ease 0.15s;
    pointer-events: none;
    z-index: 10;
    box-shadow: 0 8px 24px var(--overlay);
  }
  .info:hover::after,
  .info:focus-visible::after {
    opacity: 1;
    visibility: visible;
  }
  .height-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 16px 0 0;
  }
  .select {
    flex: 1;
    font-size: 12px;
    padding: 6px 8px;
    border-radius: 7px;
    border: 1px solid var(--border-strong);
    background: var(--bg-inset);
    color: var(--text);
  }
  .select:disabled {
    opacity: 0.5;
  }
  .primary {
    font-size: 12px;
    font-weight: 600;
    padding: 7px 16px;
    border-radius: 7px;
    border: none;
    background: var(--accent-dim);
    color: var(--bg);
  }
  .primary:disabled {
    opacity: 0.45;
  }
  .progress {
    height: 8px;
    border-radius: 999px;
    background: var(--bg-inset);
    border: 1px solid var(--border);
    overflow: hidden;
  }
  .bar {
    height: 100%;
    background: var(--accent-dim);
    transition: width 0.18s ease;
  }
  .meta {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    font-size: 12px;
    color: var(--text-faint);
    font-family: var(--mono);
  }
  .pct {
    color: var(--text-dim);
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 16px;
  }
  .cancel {
    font-size: 12px;
    padding: 7px 16px;
    border-radius: 7px;
    border: 1px solid var(--border-strong);
    background: transparent;
    color: var(--text-faint);
  }
  .cancel:hover {
    color: var(--text);
    border-color: var(--text-dim);
  }
</style>
