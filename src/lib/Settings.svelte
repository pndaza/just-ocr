<script lang="ts">
  import type { OcrOpts } from "./ocr";
  import { llmTestKey } from "./ocr";
  import type { Theme } from "../theme";
  import {
    checkForUpdate,
    downloadAndInstall,
    type UpdateStatus,
  } from "./updater";
  import { getVersion } from "@tauri-apps/api/app";

  interface Props {
    /** Reactive OCR opts. (Theme lives outside opts and is bound separately.) */
    opts: OcrOpts;
    /** Current theme (kept in sync with the document root by App.svelte). */
    theme: Theme;
    /** Called when theme changes via the segmented control here. */
    onchangetheme: (t: Theme) => void;
    /** Called when the modal should close (backdrop click, ✕, or Esc). */
    onclose: () => void;
    /** Set by App's silent startup check. Pre-populates the Updates section so the
     *  user doesn't have to re-check after the badge drew them here. */
    updateAvailable: string | null;
    /** Stored Google AI Studio API key for the AI Check tool. */
    llmApiKey: string;
    /** Called when the user edits the API key. */
    onchangeapikey: (key: string) => void;
  }
  let {
    opts,
    theme,
    onchangetheme,
    onclose,
    updateAvailable,
    llmApiKey,
    onchangeapikey,
  }: Props = $props();
  // Current app version, fetched once on mount for the Updates section header.
  let appVersion = $state("");
  // Status of the MANUAL check button (separate from the silent startup
  // `updateAvailable` prop — opening Settings never re-fires the startup check).
  let status = $state<UpdateStatus>({ kind: "idle" });

  // getVersion() rejects only outside a Tauri runtime (e.g. vitest/jsdom), where
  // appVersion staying "" is the correct fallback — the header just omits the
  // version. Swallow so it never surfaces as an unhandled promise rejection.
  $effect(() => {
    getVersion().then((v) => (appVersion = v)).catch(() => {});
  });

  // If the startup check already found an update, pre-populate as "available".
  // Guards on status === "idle" so a manual check the user kicks off is never
  // clobbered by a stale startup result.
  $effect(() => {
    if (updateAvailable && status.kind === "idle") {
      status = { kind: "available", version: updateAvailable };
    }
  });

  async function onCheck() {
    status = { kind: "checking" };
    status = await checkForUpdate();
  }

  async function onInstall() {
    try {
      status = { kind: "downloading", percent: 0 };
      await downloadAndInstall((p) => {
        status = { kind: "downloading", percent: p };
      });
      status = { kind: "installing" };
      // relaunch() inside downloadAndInstall restarts the app; this line is
      // only reached on platforms where relaunch returns / the install defers.
    } catch (e: any) {
      const message = typeof e === "string" ? e : e?.message ?? String(e);
      status = { kind: "error", message };
    }
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      onclose();
    }
  }

  // ── Hover tooltips for section hints ──────────────────────────────────────
  // Hints used to be always-on paragraphs that dominated the dialog's vertical
  // space. They now live in a ⓘ tooltip per section. The tooltip is
  // position:fixed (viewport-relative) because every scrollable ancestor
  // (.body) and the modal's overflow:hidden would otherwise clip it.
  let tip = $state<{ x: number; y: number; text: string } | null>(null);

  const TIP_W = 270;

  function showTip(el: HTMLElement, text: string) {
    const r = el.getBoundingClientRect();
    // Center under the icon, clamped to the viewport so wide hints near a
    // window edge stay reachable.
    let x = r.left + r.width / 2 - TIP_W / 2;
    x = Math.min(Math.max(8, x), window.innerWidth - TIP_W - 8);
    tip = { x, y: r.bottom + 8, text };
  }

  function hideTip() {
    tip = null;
  }

  // ── API key test ──────────────────────────────────────────────────────────
  // The Test button makes a minimal backend call (gemini-flash-lite-latest)
  // to prove the key authenticates before the user relies on AI Check.
  // Editing the key resets the verdict so a stale ✓ can't mislead.
  let testState = $state<"idle" | "testing" | "ok" | "error">("idle");
  let testError = $state("");

  async function onTestKey() {
    if (testState === "testing") return;
    testState = "testing";
    testError = "";
    try {
      await llmTestKey(llmApiKey);
      testState = "ok";
    } catch (e: any) {
      testError = typeof e === "string" ? e : e?.message ?? String(e);
      testState = "error";
    }
  }

  function onApiKeyInput(v: string) {
    testState = "idle";
    testError = "";
    onchangeapikey(v);
  }
</script>

<svelte:window onkeydown={onKey} />

{#snippet info(tipText: string)}
  <button
    class="info"
    type="button"
    tabindex="0"
    aria-label={tipText}
    onmouseenter={(e) => showTip(e.currentTarget, tipText)}
    onmouseleave={hideTip}
    onfocus={(e) => showTip(e.currentTarget, tipText)}
    onblur={hideTip}
  >
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" fill="none">
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
  </button>
{/snippet}

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
          <button
            class="seg-btn"
            class:active={theme === "system"}
            onclick={() => onchangetheme("system")}
            role="radio"
            aria-checked={theme === "system"}
          >System</button>
        </div>
      </section>

      {#if opts.language === "mya" && opts.segmenter !== "kraken"}
        <!-- Myanmar + PP-OCR only: the PP-OCR line-box detector backbone width.
             Small = accuracy-oriented (default); Tiny = faster/smaller but less
             accurate on dense/curved Burmese. Hidden for Kraken (own segmenter
             model) and for non-Myanmar (full-page Tesseract, no PP-OCR). -->
        <section class="sec">
          <div class="lbl-row">
            <span class="lbl">PP-OCR detection model</span>
            {@render info(
              "Tiny   — good for most cases.\nSmall — good for curvy lines.",
            )}
          </div>
          <div class="seg" role="radiogroup" aria-label="PP-OCR detection model">
            <button
              class="seg-btn"
              class:active={opts.detVariant === "tiny"}
              onclick={() => (opts.detVariant = "tiny")}
              role="radio"
              aria-checked={opts.detVariant === "tiny"}
            >Tiny</button>
            <button
              class="seg-btn"
              class:active={opts.detVariant === "small"}
              onclick={() => (opts.detVariant = "small")}
              role="radio"
              aria-checked={opts.detVariant === "small"}
            >Small</button>
          </div>
        </section>
      {/if}

      <section class="sec">
        <div class="lbl-row">
          <span class="lbl">AI spell check</span>
          {@render info(
            "Powers the “AI Check” toolbar tool, which sends recognized text " +
            "to Gemini to find spelling errors — the model is chosen in the " +
            "AI Check dialog. This is the app's only online feature; get a " +
            "free key at aistudio.google.com. The key stays on this machine " +
            "and is sent only to Google's API.",
          )}
        </div>
        <label class="key-lbl" for="llm-api-key">Google AI Studio API key</label>
        <div class="key-row">
          <input
            id="llm-api-key"
            class="key-input"
            type="password"
            placeholder="Paste your key"
            spellcheck="false"
            autocomplete="off"
            value={llmApiKey}
            oninput={(e) => onApiKeyInput(e.currentTarget.value)}
          />
          <button
            class="upd-btn test-btn"
            onclick={onTestKey}
            disabled={testState === "testing" || !llmApiKey.trim()}
            title="Verify the key with a minimal Gemini request (gemini-flash-lite-latest)"
          >
            {testState === "testing" ? "Testing…" : "Test"}
          </button>
        </div>
        {#if testState === "ok"}
          <p class="test-result ok">✓ API key works.</p>
        {:else if testState === "error"}
          <p class="test-result err">{testError}</p>
        {/if}
      </section>

      <section class="sec">
        <span class="lbl">Updates {appVersion ? `· v${appVersion}` : ""}</span>

        {#if status.kind === "idle"}
          <button class="upd-btn" onclick={onCheck}>Check for updates</button>
        {:else if status.kind === "checking"}
          <button class="upd-btn" disabled>Checking…</button>
        {:else if status.kind === "up-to-date"}
          <span class="upd-ok">Just OCR is up to date ✓</span>
        {:else if status.kind === "available"}
          <div class="upd-available">
            <span>v{status.version} available</span>
            <button class="upd-btn primary" onclick={onInstall}>Download &amp; install</button>
            <span class="upd-note">The app will restart.</span>
          </div>
        {:else if status.kind === "downloading"}
          <div class="upd-progress">
            <div
              class="bar"
              role="progressbar"
              aria-valuenow={status.percent}
              aria-valuemin={0}
              aria-valuemax={100}
            ><div class="fill" style="width:{status.percent}%"></div></div>
            <span class="upd-note">Downloading… {status.percent}%</span>
          </div>
        {:else if status.kind === "installing"}
          <span class="upd-note">Installing…</span>
        {:else if status.kind === "error"}
          <span class="upd-err">{status.message}</span>
          <button class="upd-btn" onclick={onCheck}>Retry</button>
        {/if}
      </section>
    </div>

    {#if tip}
      <!-- Fixed/viewport-relative so the modal's overflow:hidden and the
           scrollable .body can't clip it; pointer-events keeps the hover
           stable while the cursor rests on the icon. -->
      <div class="tooltip" style="left:{tip.x}px; top:{tip.y}px" role="tooltip">
        {tip.text}
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
    width: min(420px, 100%);
    /* Cap the modal so tall stacks of sections never outgrow the window;
     * the header stays pinned and the body scrolls. */
    max-height: min(85vh, 640px);
    display: flex;
    flex-direction: column;
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
    flex-shrink: 0;
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
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }
  .sec {
    padding: 14px 0;
    border-bottom: 1px solid var(--border);
  }
  .sec:last-child {
    border-bottom: none;
  }
  .lbl-row {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 8px;
  }
  .lbl {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-faint);
  }
  .info {
    display: inline-flex;
    background: none;
    border: none;
    padding: 0;
    color: var(--text-faint);
    cursor: help;
  }
  .info:hover,
  .info:focus-visible {
    color: var(--text-dim);
  }
  .tooltip {
    position: fixed;
    width: 270px;
    background: var(--text);
    color: var(--bg-elev);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 11px;
    line-height: 1.45;
    text-align: left;
    /* Render \n in tip text as line breaks, and keep intentional space runs
     * (e.g. the aligned Tiny/Small dashes) from collapsing. */
    white-space: pre-wrap;
    pointer-events: none;
    z-index: 200;
    box-shadow: 0 8px 24px var(--overlay);
  }
  .key-lbl {
    display: block;
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 6px;
  }
  .key-row {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .key-input {
    flex: 1;
    min-width: 0;
    font-size: 12px;
    font-family: var(--mono);
    padding: 6px 9px;
    border-radius: 6px;
  }
  .test-btn {
    flex-shrink: 0;
  }
  .test-result {
    margin: 6px 0 0;
    font-size: 12px;
    line-height: 1.4;
    word-break: break-word;
  }
  .test-result.ok { color: var(--ok); }
  .test-result.err { color: var(--danger); }
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
  .upd-btn {
    font-size: 12px;
    font-weight: 600;
    padding: 6px 13px;
    border-radius: 6px;
    color: var(--text-dim);
    background: var(--surface);
    border: 1px solid var(--border);
  }
  .upd-btn:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--border-strong);
  }
  .upd-btn.primary {
    color: var(--bg);
    background: var(--accent);
    border-color: transparent;
  }
  .upd-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .upd-ok {
    font-size: 12px;
    color: var(--ok);
  }
  .upd-available {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-start;
  }
  .upd-available > span:first-child {
    font-size: 12px;
    color: var(--accent);
    font-weight: 600;
  }
  .upd-note {
    font-size: 11px;
    color: var(--text-faint);
  }
  .upd-progress {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .upd-progress .bar {
    width: 100%;
    height: 6px;
    background: var(--bg-inset);
    border-radius: 4px;
    overflow: hidden;
  }
  .upd-progress .fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.15s;
  }
  .upd-err {
    display: block;
    font-size: 12px;
    color: var(--danger);
    margin-bottom: 8px;
  }
</style>
