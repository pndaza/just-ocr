<script lang="ts">
  import type { OcrOpts } from "./ocr";
  import logo from "../assets/logo.png";

  interface Props {
    opts: OcrOpts;
    languages: string[];
    running: boolean;
    pending: number;
    doneCount: number;
    /** Merge-paragraphs view toggle (display + export projection). */
    mergeParagraphs: boolean;
    /** Burmese post-OCR spelling-fix toggle. Myanmar-only in effect; shown
     *  only when Myanmar is selected so the toolbar doesn't carry a no-op
     *  toggle for other languages. Changes what the engine returns (not just
     *  the display projection). */
    fixBurmeseSpelling: boolean;
    /** Whether the "Fix spelling" checkbox appears here at all — controlled
     *  from Settings. Hiding it doesn't change the toggle's own state. */
    showFixSpelling: boolean;
    /** Current/total counter shown beside "Processing" during a batch run.
     *  Null for single runs (Run Current) so no counter appears. */
    batchProgress: { current: number; total: number } | null;
    canRunCurrent: boolean;
    hasSelection: boolean;
    showStop: boolean;
    stopping: boolean;
    onstop: () => void;
    onruncurrent: () => void;
    onrunall: () => void;
    onexport: () => void;
    /** Toggles the AI spell-check panel (Gemini) — a 4th main column. The
     *  panel itself handles the no-key case with a pointer to Settings. */
    onaicheck: () => void;
    /** True while the AI Check panel is open — styles the button active. */
    aiOpen: boolean;
    onmanagelanguages: () => void;
    onsettings: () => void;
    /** When non-null, a newer version exists — shows a badge on the gear. */
    updateAvailable: string | null;
    onchangemerge: (v: boolean) => void;
    onchangefix: (v: boolean) => void;
  }
  let {
    opts,
    languages,
    running,
    pending,
    doneCount,
    mergeParagraphs,
    fixBurmeseSpelling,
    showFixSpelling,
    batchProgress,
    canRunCurrent,
    hasSelection,
    showStop,
    stopping,
    onstop,
    onruncurrent,
    onrunall,
    onexport,
    onaicheck,
    aiOpen,
    onmanagelanguages,
    onsettings,
    updateAvailable,
    onchangemerge,
    onchangefix,
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

  // Pipeline is language-driven:
  //   mya        → Kraken segmentation (hidden) + recognizer chosen by `engine`.
  //   everything → full-page Tesseract with `psm`. Engine selector is hidden.
  let isMyanmar = $derived(opts.language === "mya");

  // Picking Myanmar defaults the recognizer to Kraken (the whole point —
  // Tesseract is bad at Myanmar script). This must fire ONLY on the language
  // transition (not-my → mya), not whenever engine happens to be tesseract —
  // otherwise the user can never select tesseract for myanmar after the
  // default fires. We track the previous language to detect the transition.
  let prevLang = $state(opts.language);
  $effect(() => {
    const becameMyanmar = opts.language === "mya" && prevLang !== "mya";
    prevLang = opts.language;
    if (becameMyanmar) {
      opts.engine = "kraken";
    }
  });
</script>

<div class="toolbar">
  <img class="brand-logo" src={logo} alt="Just OCR" />

  <div class="divider"></div>

  <button
    class="icon-btn"
    class:has-update={!!updateAvailable}
    onclick={onsettings}
    title={updateAvailable ? `Update available: v${updateAvailable}` : "Settings"}
    aria-label={updateAvailable ? `Settings — update available (v${updateAvailable})` : "Settings"}
  >⚙</button>

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
    <!-- Myanmar: Seg picks the line-box detector, Rec picks the recognizer.
         Kraken-as-segmenter is hidden from the UI (not accurate enough yet);
         its code path + type variant are retained for when it improves. -->
    <label class="field">
      <span class="lbl">Seg</span>
      <select bind:value={opts.segmenter}>
        <option value="ppocr">PP-OCR (quad)</option>
        <option value="ppocr-poly">PP-OCR (poly)</option>
      </select>
    </label>
    <label class="field">
      <span class="lbl">Rec</span>
      <select bind:value={opts.engine}>
        <option value="kraken">Kraken</option>
        <option value="tesseract">Tesseract</option>
      </select>
    </label>
    <!-- Binarize lives in Settings now (global, persisted preference). -->
  {:else}
    <!-- Non-Myanmar: Tesseract does both segmentation + recognition. PSM exposed. -->
    <label class="field">
      <span class="lbl">PSM</span>
      <select bind:value={opts.psm}>
        {#each psmOptions as o}<option value={o.value}>{o.label}</option>{/each}
      </select>
    </label>
  {/if}

  <label class="check" title="Join recognized lines into paragraphs (display + export)">
    <input
      type="checkbox"
      checked={mergeParagraphs}
      onchange={(e) => onchangemerge(e.currentTarget.checked)}
    />
    Merge lines
  </label>

  {#if false}
    <!-- TEMPORARILY HIDDEN: the rule-based Burmese spelling fix (curated
         wrong→right word list) is far behind the AI spell check, so the
         toggle is parked behind {#if false} until it improves. All props and
         the effect wiring are kept so re-enabling is a one-line change.
         opts.fixBurmeseSpelling is forced off in App while this is hidden —
         a stuck-on value would keep applying the fix with no way to turn it
         off. -->
    {#if isMyanmar && showFixSpelling}
      <label class="check" title="Correct common Burmese recognizer errors (word list)">
        <input
          type="checkbox"
          checked={fixBurmeseSpelling}
          onchange={(e) => onchangefix(e.currentTarget.checked)}
        />
        Fix spelling
      </label>
    {/if}
  {/if}

  <div class="spacer"></div>

  {#if running}
    <span class="progress">
      <span class="spin" aria-hidden="true"></span>
      Processing{#if batchProgress} <span class="prog-count">{batchProgress.current}/{batchProgress.total}</span>{/if}
    </span>
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

  {#if doneCount > 0}
    <button
      class="btn ghost"
      class:active={aiOpen}
      onclick={onaicheck}
      title="Toggle the AI spell-check panel (Gemini) — configure the API key in Settings"
    >
      ✦ AI Spell Fix
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
  .brand-logo {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    /* macOS-style squircle hint; harmless where unsupported. */
    object-fit: contain;
    flex-shrink: 0;
  }
  .divider {
    width: 1px;
    height: 24px;
    background: var(--border);
  }
  .icon-btn {
    /* relative so the ::after badge dot can absolute-position onto the gear */
    position: relative;
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
  .icon-btn:hover {
    color: var(--accent);
    border-color: var(--accent-dim);
    background: var(--accent-soft);
  }
  /* Accent dot on the gear when a newer version exists. Punch-through border
     uses the elevated bg so the dot reads cleanly on top of the gear glyph. */
  .icon-btn.has-update::after {
    content: "";
    position: absolute;
    top: 1px;
    right: 1px;
    width: 7px;
    height: 7px;
    background: var(--accent);
    border-radius: 50%;
    border: 1.5px solid var(--bg-elev);
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
  .check input {
    accent-color: var(--accent-dim);
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
  /* tabular-nums keeps the current/total digits the same width so the
     indicator doesn't shift as the counter ticks. */
  .progress .prog-count {
    font-variant-numeric: tabular-nums;
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
  /* Active (toggled-open) state for panel-toggle buttons like AI Check. */
  .btn.ghost.active {
    color: var(--accent);
    border-color: var(--accent-dim);
    background: var(--accent-soft);
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
