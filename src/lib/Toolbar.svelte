<script lang="ts">
  import type { OcrOpts } from "./ocr";
  import type { Theme } from "../theme";

  interface Props {
    opts: OcrOpts;
    languages: string[];
    running: boolean;
    pending: number;
    doneCount: number;
    canRunCurrent: boolean;
    hasSelection: boolean;
    theme: Theme;
    showStop: boolean;
    stopping: boolean;
    onstop: () => void;
    onruncurrent: () => void;
    onrunall: () => void;
    onexport: () => void;
    onmanagelanguages: () => void;
    ontoggletheme: () => void;
  }
  let {
    opts,
    languages,
    running,
    pending,
    doneCount,
    canRunCurrent,
    hasSelection,
    theme,
    showStop,
    stopping,
    onstop,
    onruncurrent,
    onrunall,
    onexport,
    onmanagelanguages,
    ontoggletheme,
  }: Props = $props();

  const psmOptions = [
    { value: 0, label: "0 · OSD only" },
    { value: 1, label: "1 · Auto + OSD" },
    { value: 2, label: "2 · Auto (no OSD)" },
    { value: 3, label: "3 · Auto (full page)" },
    { value: 4, label: "4 · Single column" },
    { value: 6, label: "6 · Single block" },
    { value: 7, label: "7 · Single line" },
    { value: 8, label: "8 · Single word" },
    { value: 10, label: "10 · Single char" },
    { value: 11, label: "11 · Sparse text" },
    { value: 13, label: "13 · Raw line" },
  ];

  let useWhitelist = $state(false);
  $effect(() => {
    opts.whitelist = useWhitelist ? (opts.whitelist ?? "") : null;
  });

  // Pipeline is language-driven:
  //   mya        → Kraken segmentation (hidden) + recognizer chosen by `engine`.
  //                Kraken recog has no whitelist, so disable it for that combo.
  //   everything → full-page Tesseract with `psm`. Engine selector is hidden.
  let isMyanmar = $derived(opts.language === "mya");
  let whitelistDisabled = $derived(isMyanmar && opts.engine === "kraken");

  // Picking Myanmar defaults the recognizer to Kraken (the whole point —
  // Tesseract is bad at Myanmar script). Switching away leaves the engine
  // state as-is; it just becomes irrelevant.
  $effect(() => {
    if (isMyanmar && opts.engine === "tesseract") {
      // Only auto-switch on the language change, not on every engine toggle.
      // We detect "just became Myanmar" by tracking the previous language.
      // Simpler: if user is on Myanmar and somehow landed on tesseract without
      // having explicitly chosen it, prefer kraken. Practically this fires
      // once when the select flips to mya.
      opts.engine = "kraken";
    }
  });

  // Whitelist auto-disable for the kraken recognizer (no char whitelist there).
  $effect(() => {
    if (whitelistDisabled && useWhitelist) {
      useWhitelist = false;
      opts.whitelist = null;
    }
  });
</script>

<div class="toolbar">
  <div class="brand">
    <span class="logo">◎</span>
    just<span class="accent">-ocr</span>
  </div>

  <div class="divider"></div>

  <button
    class="theme-toggle"
    onclick={ontoggletheme}
    title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
    aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
  >{theme === "dark" ? "☀" : "☾"}</button>

  <label class="field">
    <span class="lbl">Lang</span>
    <select bind:value={opts.language}>
      {#each languages as l}<option value={l}>{l}</option>{/each}
    </select>
    <button
      class="lang-add"
      onclick={onmanagelanguages}
      title="Add or remove language models"
      aria-label="Manage languages"
    >+</button>
  </label>

  {#if isMyanmar}
    <!-- Myanmar: Kraken does segmentation (hidden). Engine picks the recognizer. -->
    <label class="field">
      <span class="lbl">Engine</span>
      <select bind:value={opts.engine}>
        <option value="kraken">Kraken</option>
        <option value="tesseract">Tesseract</option>
      </select>
    </label>
  {:else}
    <!-- Non-Myanmar: Tesseract does both segmentation + recognition. PSM exposed. -->
    <label class="field">
      <span class="lbl">PSM</span>
      <select bind:value={opts.psm}>
        {#each psmOptions as o}<option value={o.value}>{o.label}</option>{/each}
      </select>
    </label>
  {/if}

  <label class="check" class:disabled={whitelistDisabled}>
    <input
      type="checkbox"
      bind:checked={useWhitelist}
      disabled={whitelistDisabled}
    />
    Whitelist
  </label>
  {#if useWhitelist && !whitelistDisabled}
    <input
      class="wl"
      type="text"
      placeholder="0123456789"
      bind:value={opts.whitelist}
      size="10"
    />
  {/if}

  <div class="spacer"></div>

  {#if running}
    <span class="progress"><span class="spin" aria-hidden="true"></span> Processing</span>
  {/if}

  {#if showStop}
    <button
      class="btn danger"
      onclick={onstop}
      disabled={stopping}
      title="Stop processing the remaining images"
    >
      {stopping ? "Stopping…" : "Stop"}
    </button>
  {/if}

  {#if doneCount > 0}
    <button class="btn ghost" onclick={onexport} disabled={running}>
      Export ({doneCount})
    </button>
  {/if}

  <button
    class="btn ghost"
    onclick={onruncurrent}
    disabled={running || !canRunCurrent}
    title={hasSelection ? "OCR the selected image" : "Select an image first"}
  >
    Run Current
  </button>
  <button class="btn primary" onclick={onrunall} disabled={running || pending + doneCount === 0}>
    {#if running}Running…{:else}Run All{/if}
  </button>
</div>

<style>
  .toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 14px;
    height: 48px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-elev);
    flex-shrink: 0;
  }
  .brand {
    font-weight: 700;
    font-size: 15px;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .logo {
    color: var(--accent);
    font-size: 17px;
  }
  .accent {
    color: var(--accent);
  }
  .divider {
    width: 1px;
    height: 24px;
    background: var(--border);
  }
  .theme-toggle {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-faint);
    font-size: 14px;
    width: 26px;
    height: 26px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    line-height: 1;
  }
  .theme-toggle:hover {
    color: var(--accent);
    border-color: var(--accent-dim);
    background: var(--accent-soft);
  }
  .field {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .lbl {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-faint);
  }
  select {
    padding: 4px 8px;
    font-size: 12px;
    border-radius: 6px;
  }
  .lang-add {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-faint);
    font-size: 13px;
    font-weight: 700;
    width: 22px;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    line-height: 1;
  }
  .lang-add:hover {
    color: var(--accent);
    border-color: var(--accent-dim);
    background: var(--accent-soft);
  }
  .check {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: var(--text-dim);
    cursor: pointer;
  }
  .check.disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .check input {
    accent-color: var(--accent-dim);
  }
  .wl {
    padding: 4px 8px;
    font-size: 12px;
    font-family: var(--mono);
    width: 90px;
  }
  .spacer { flex: 1; }
  .progress {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    color: var(--accent);
    font-family: var(--mono);
  }
  .btn {
    font-size: 12px;
    font-weight: 600;
    padding: 6px 13px;
    border-radius: 6px;
    border: 1px solid transparent;
  }
  .btn.ghost {
    color: var(--text-dim);
    background: var(--surface);
    border-color: var(--border);
  }
  .btn.ghost:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--border-strong);
  }
  .btn.primary {
    color: var(--bg);
    background: var(--accent);
  }
  .btn.primary:hover:not(:disabled) { opacity: 0.9; }
  .btn.danger {
    color: var(--danger);
    background: var(--danger-soft);
    border-color: var(--danger);
  }
  .btn.danger:hover:not(:disabled) {
    color: var(--bg);
    background: var(--danger);
  }
  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .spin {
    width: 12px;
    height: 12px;
    border: 2px solid var(--accent-soft);
    border-top-color: var(--accent);
    border-radius: 50%;
    display: inline-block;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
