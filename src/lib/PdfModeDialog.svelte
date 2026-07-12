<script lang="ts">
  import type { ImageMode, PdfMode } from "./ocr";

  // Color format for the per-page PNGs. Grayscale is the Tesseract-friendly
  // default (smaller, no accuracy loss); B&W is Otsu-thresholded for pristine
  // scans; Color keeps the source as-is.
  const imageModes: { value: ImageMode; label: string; hint: string }[] = [
    { value: "color", label: "Color", hint: "Keep the source as-is" },
    { value: "gray", label: "Gray", hint: "Best for Tesseract (default)" },
    { value: "bw", label: "B&W", hint: "Otsu-thresholded, for clean scans" },
  ];

  interface Props {
    /** The PDF file name being processed (shown in the dialog). */
    name: string;
    /** "choosing" shows the mode buttons; "working" shows the progress UI. */
    status: "choosing" | "working";
    /** The mode the user picked (set once they click a button). */
    mode: PdfMode | null;
    /** Pages processed so far (from the backend pdf-progress event). */
    done: number;
    /** Total page count (0 until the backend reports it). */
    total: number;
    /** Called with the chosen mode + image format when the user picks a mode. */
    onprocess: (mode: PdfMode, imageMode: ImageMode) => void;
    /** Called when the user cancels (backdrop click, Cancel button, or Esc). */
    oncancel: () => void;
  }
  let { name, status, mode, done, total, onprocess, oncancel }: Props = $props();

  // Per-PDF image format; defaults to grayscale, the OCR-friendly choice.
  let imageMode = $state<ImageMode>("gray");

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

      <div class="options">
        <button class="opt" onclick={() => onprocess("extract", imageMode)}>
          <span class="opt-title">Extract</span>
          <span class="opt-desc">
            Pull the embedded scan at native resolution. Fast for already-rasterized
            pages.
          </span>
        </button>
        <button class="opt" onclick={() => onprocess("render", imageMode)}>
          <span class="opt-title">Render</span>
          <span class="opt-desc">
            Rasterize each page at 1500px height. Best for vector or mixed content.
          </span>
        </button>
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
  .options {
    display: grid;
    gap: 10px;
  }
  .opt {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    text-align: left;
    padding: 13px 15px;
    border-radius: 10px;
    border: 1px solid var(--border-strong);
    background: var(--bg-inset);
    color: var(--text);
    transition:
      border-color 0.12s,
      background 0.12s;
  }
  .opt:hover {
    border-color: var(--accent-dim);
    background: var(--accent-soft);
  }
  .opt-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--accent);
  }
  .opt-desc {
    font-size: 12px;
    line-height: 1.4;
    color: var(--text-dim);
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
