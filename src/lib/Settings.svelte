<script lang="ts">
  import type { OcrOpts } from "./ocr";
  import type { Theme } from "../theme";

  interface Props {
    /** Reactive OCR opts — `binarize` is bound here. */
    opts: OcrOpts;
    /** Current theme (kept in sync with the document root by App.svelte). */
    theme: Theme;
    /** Called when theme changes via the segmented control here. */
    onchangetheme: (t: Theme) => void;
    /** Called when the modal should close (backdrop click, ✕, or Esc). */
    onclose: () => void;
  }
  let { opts, theme, onchangetheme, onclose }: Props = $props();

  function onKey(e: KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      onclose();
    }
  }
</script>

<svelte:window onkeydown={onKey} />

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="backdrop" onclick={onclose} role="presentation">
  <!-- svelte-ignore a11y_no_static_element_interactions, a11y_interactive_supports_focus -->
  <div
    class="modal"
    onclick={(e) => e.stopPropagation()}
    role="dialog"
    aria-modal="true"
    aria-label="Settings"
  >
    <div class="head">
      <h2>Settings</h2>
      <button class="close" onclick={onclose} aria-label="Close settings">✕</button>
    </div>

    <div class="body">
      <section class="sec">
        <span class="lbl">Theme</span>
        <div class="seg" role="radiogroup" aria-label="Theme">
          <button
            class="seg-btn"
            class:active={theme === "light"}
            onclick={() => onchangetheme("light")}
            role="radio"
            aria-checked={theme === "light"}
          >Light</button>
          <button
            class="seg-btn"
            class:active={theme === "dark"}
            onclick={() => onchangetheme("dark")}
            role="radio"
            aria-checked={theme === "dark"}
          >Dark</button>
        </div>
      </section>

      <section class="sec">
        <span class="lbl">Binarization</span>
        <p class="desc">
          For recognition models trained on 1-bit images. Myanmar / Kraken path
          only; ignored by Tesseract.
        </p>
        <div class="seg" role="radiogroup" aria-label="Binarization mode">
          <button
            class="seg-btn"
            class:active={opts.binarize === null}
            onclick={() => (opts.binarize = null)}
            role="radio"
            aria-checked={opts.binarize === null}
          >Off</button>
          <button
            class="seg-btn"
            class:active={opts.binarize === "otsu"}
            onclick={() => (opts.binarize = "otsu")}
            role="radio"
            aria-checked={opts.binarize === "otsu"}
          >Otsu</button>
          <button
            class="seg-btn"
            class:active={opts.binarize === "sauvola"}
            onclick={() => (opts.binarize = "sauvola")}
            role="radio"
            aria-checked={opts.binarize === "sauvola"}
          >Sauvola</button>
        </div>
      </section>
    </div>
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
    width: min(420px, 100%);
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    box-shadow: 0 24px 70px var(--overlay);
    overflow: hidden;
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
  }
  h2 {
    margin: 0;
    font-size: 15px;
    color: var(--text);
  }
  .close {
    background: none;
    border: none;
    color: var(--text-faint);
    font-size: 14px;
    padding: 2px 6px;
    border-radius: 6px;
    line-height: 1;
  }
  .close:hover {
    color: var(--text);
  }
  .body {
    padding: 8px 20px 20px;
  }
  .sec {
    padding: 14px 0;
    border-bottom: 1px solid var(--border);
  }
  .sec:last-child {
    border-bottom: none;
  }
  .lbl {
    display: block;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-faint);
    margin-bottom: 8px;
  }
  .desc {
    margin: 0 0 10px;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-dim);
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
    flex: 1;
    background: none;
    border: none;
    font-size: 12px;
    padding: 6px 12px;
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
</style>
