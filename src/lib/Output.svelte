<script lang="ts">
  import type { Job } from "./ocr";
  import { plainText, plainTextWithFix, formatDuration } from "./result";

  interface Props {
    job: Job | null;
    /** All jobs in the batch (not just the selected one) — aggregated in the
     *  bottom status bar for total processing time + spell-fixes across all
     *  completed pages. */
    jobs: Job[];
    /** When true, lines are grouped into paragraphs by the geometry heuristic
     *  in result.ts. Mirrors the toolbar toggle so the panel reflects the
     *  same projection used for export. */
    mergeParagraphs: boolean;
    /** When true AND the job has a cached spell-fix projection, the panel
     *  shows fixed text instead of raw. The toggle is a prop (not read from
     *  localStorage) so App remains the single source of truth for opts. */
    fixSpelling: boolean;
  }
  let { job, jobs, mergeParagraphs, fixSpelling }: Props = $props();

  // The recognized text is a projection of the structured `OcrResult`. With
  // mergeParagraphs off, lines join with "\n" (legacy behaviour); with it on,
  // close lines join with a space and paragraphs are separated by "\n\n".
  // When spell-fix is on and the job's cache is populated, fixed line text is
  // substituted in (geometry unchanged). Falls back to "" until done.
  let displayText = $derived.by(() => {
    if (!job || job.status !== "done" || !job.result) return "";
    const textOpts = mergeParagraphs ? { mergeParagraphs: true } : undefined;
    const body =
      fixSpelling && job.spellFix
        ? plainTextWithFix(job.result, job.spellFix.fixedLines, textOpts)
        : plainText(job.result, textOpts);
    return body.replace(/\s+$/, "");
  });

  // Fix count shows only when spell-fix is on and the cache is populated.
  // `null` otherwise (toggle off, or cache not yet computed) → no badge.
  let fixCount = $derived(
    fixSpelling && job?.spellFix ? job.spellFix.fixes : null,
  );

  // ── Batch aggregates for the bottom status bar ───────────────────────────
  // Totals across all COMPLETED jobs (the whole batch, not just the selected
  // page). Recomputed reactively as jobs finish or spell-fix caches populate.
  // Mirrors the Preview panel's status bar styling so the two read as a pair.
  let doneJobs = $derived(jobs.filter((j) => j.status === "done"));
  let totalPages = $derived(doneJobs.length);
  let totalMs = $derived(doneJobs.reduce((sum, j) => sum + j.elapsedMs, 0));
  // Spell-fix total counts only when the toggle is on; otherwise the bar
  // omits the fixes segment entirely (no point showing 0 for a disabled pass).
  let totalFixes = $derived(
    fixSpelling
      ? doneJobs.reduce((sum, j) => sum + (j.spellFix?.fixes ?? 0), 0)
      : null,
  );

  let copied = $state(false);

  async function copy() {
    if (!displayText) return;
    await navigator.clipboard.writeText(displayText);
    copied = true;
    setTimeout(() => (copied = false), 1300);
  }
</script>

<div class="panel" role="region" aria-label="Recognized text">
  <div class="head">
    <span class="title">Text</span>
    {#if job?.status === "done"}
      <span class="meta">{job.elapsedMs} ms</span>
      {#if fixCount !== null}
        <span
          class="meta fixes"
          title="Total Burmese spelling fixes applied (regex + dictionary)"
        >{fixCount === 0
          ? "no fixes"
          : `${fixCount} ${fixCount === 1 ? "fix" : "fixes"}`}</span>
      {/if}
      <button class="copy" onclick={copy} disabled={!displayText.trim()}>
        {copied ? "Copied ✓" : "Copy"}
      </button>
    {/if}
  </div>

  <div class="body">
    {#if !job}
      <div class="placeholder">Select an image to see its text.</div>
    {:else if job.status === "error"}
      <div class="error">
        <strong>OCR failed for this image.</strong>
        <pre>{job.error}</pre>
      </div>
    {:else if job.status === "running"}
      <div class="placeholder"><span class="spin" aria-hidden="true"></span> Recognizing…</div>
    {:else if job.status === "queued"}
      <div class="placeholder">Queued — run OCR to extract text.</div>
    {:else if displayText.trim()}
      <pre class="text">{displayText}</pre>
    {:else}
      <div class="placeholder">No text recognized. Try a different engine or image.</div>
    {/if}
  </div>

  {#if totalPages > 0}
    <!-- Batch totals across all completed pages. Mirrors the Preview status
         bar's styling/typography so the two panels read as a pair. The fixes
         segment appears only when spell-fix is on (otherwise omitted, not 0). -->
    <div class="status-bar" role="status">
      <span>{totalPages} {totalPages === 1 ? "page" : "pages"}</span>
      <span class="sb-sep">·</span>
      <span>Total <span class="sb-num">{formatDuration(totalMs)}</span></span>
      {#if totalFixes !== null}
        <span class="sb-sep">·</span>
        <span>{totalFixes} {totalFixes === 1 ? "fix" : "fixes"}</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    height: 100%;
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
  }
  .meta {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text-faint);
  }
  /* Spelling-fix count: accent-colored so a non-zero count reads as a
     "something happened" signal, distinct from the neutral timing meta. */
  .meta.fixes {
    color: var(--accent);
  }
  .copy {
    font-size: 11px;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 3px 9px;
  }
  .copy:hover:not(:disabled) { border-color: var(--accent-dim); }
  .copy:disabled { opacity: 0.4; cursor: not-allowed; }
  /* Bottom status bar — batch totals across all completed pages. Mirrors the
     Preview panel's status bar so the two read as a styled pair. */
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
    font-variant-numeric: tabular-nums;
  }
  .status-bar .sb-sep {
    color: var(--text-faint);
    opacity: 0.6;
  }
  .body {
    flex: 1;
    overflow-y: auto;
    padding: 14px;
  }
  .text {
    margin: 0;
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.6;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .placeholder {
    color: var(--text-faint);
    font-size: 13px;
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .error {
    color: var(--danger);
    background: var(--danger-soft);
    border: 1px solid var(--danger);
    border-radius: 8px;
    padding: 14px;
    font-size: 13px;
  }
  .error pre {
    margin: 8px 0 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    opacity: 0.85;
  }
  .spin {
    width: 12px;
    height: 12px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    display: inline-block;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
