<script lang="ts">
  import type { Job, WordHighlight } from "./ocr";
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
    /** Wrong words flagged by the AI spell check for this page, each with
     *  the 1-based line it was flagged on (null = not line-scoped). Marked
     *  in the displayed text only where flagged — a word that also occurs
     *  on other lines of the page stays unmarked there. Words already
     *  replaced by an applied fix simply no longer occur and don't match. */
    highlights: WordHighlight[];
  }
  let { job, jobs, mergeParagraphs, fixSpelling, highlights }: Props = $props();

  // The recognized text is a projection of the structured `OcrResult`. With
  // mergeParagraphs off, lines join with "\n" (legacy behaviour); with it on,
  // close lines join with a space and paragraphs are separated by "\n\n".
  // Projection precedence: manual edits (typed into the panel) are
  // authoritative, then an applied AI fix (built on top of the spell-fix
  // basis lines), then the offline spell-fix when toggled on, then raw.
  // Manual text is stored verbatim (no trailing trim). Falls back to "" until
  // done.
  let displayText = $derived.by(() => {
    if (!job || job.status !== "done" || !job.result) return "";
    if (job.manualText != null) return job.manualText;
    const textOpts = mergeParagraphs ? { mergeParagraphs: true } : undefined;
    const body =
      job.llmFix
        ? plainTextWithFix(job.result, job.llmFix.fixedLines, textOpts)
        : fixSpelling && job.spellFix
          ? plainTextWithFix(job.result, job.spellFix.fixedLines, textOpts)
          : plainText(job.result, textOpts);
    return body.replace(/\s+$/, "");
  });

  /** First keystroke in the panel promotes the projection to a manual edit. */
  function onEdit(e: Event & { currentTarget: HTMLTextAreaElement }) {
    if (!job) return;
    job.manualText = e.currentTarget.value;
  }

  /** Drop manual edits and fall back to the live projection. */
  function revert() {
    if (job) job.manualText = null;
  }

  // ── Edit mode ──────────────────────────────────────────────────────────────
  // Read-only (default) renders a <pre> that can carry the AI-highlight
  // <mark>s; a <textarea> can't (its content is raw text), so editing swaps
  // the element via an explicit Edit toggle. Resets when the page changes.
  let editing = $state(false);
  $effect(() => {
    job?.id; // track selection
    editing = false;
  });

  /** Focus the textarea the moment edit mode mounts it. */
  function focusNode(node: HTMLElement) {
    node.focus();
    return {};
  }

  // Fix count: applied AI fixes are always shown (the user explicitly
  // accepted them, independent of the spell-fix toggle); otherwise the
  // spell-fix count shows only when that toggle is on and the cache is
  // populated. `null` → no badge.
  let aiFixCount = $derived(job?.llmFix ? job.llmFix.fixes : null);
  let fixCount = $derived(
    aiFixCount !== null
      ? aiFixCount
      : fixSpelling && job?.spellFix
        ? job.spellFix.fixes
        : null,
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
  // AI fixes were explicitly applied by the user, so they're always reported
  // (separate from the spell-fix total — they were counted on top of it).
  let totalAiFixes = $derived(
    doneJobs.reduce((sum, j) => sum + (j.llmFix?.fixes ?? 0), 0),
  );

  // ── AI-flagged word highlighting ────────────────────────────────────────────
  // Only the FLAGGED occurrence gets a <mark>: each entry carries the line it
  // was flagged on (1-based, indexing the page's basis lines — the same
  // lines the AI Check panel's L chips count), so a word that recurs on
  // untouched lines of the same page doesn't light up there. In the
  // line-numbered view each row knows its number, so scoping is a filter. In
  // the merged-paragraph view rows are gone, but each line's text survives
  // as a contiguous run inside the merged text (the merge only inserts
  // " " / "\n\n" between trimmed line texts), so a forward cursor over the
  // merged text locates every line's span without duplicating the
  // paragraph-grouping heuristic that built it.

  /** One rendered stretch of text — marked or plain. */
  interface Seg {
    text: string;
    mark: boolean;
  }

  /** Split `text` into plain/marked segments, marking occurrences of `words`
   *  (longest-first so overlapping words match greedily; regex-escaped since
   *  OCR words can contain punctuation). `allow` gates each match — the
   *  merged view passes a flagged-line-span check. Null when nothing
   *  matches (keeps the plain render path untouched). */
  function markText(
    text: string,
    words: string[],
    allow?: (word: string, start: number) => boolean,
  ): Seg[] | null {
    const ws = [...new Set(words)].filter((w) => w && text.includes(w));
    if (!ws.length) return null;
    ws.sort((a, b) => b.length - a.length);
    const pattern = ws
      .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|");
    const out: Seg[] = [];
    let last = 0;
    for (const m of text.matchAll(new RegExp(pattern, "g"))) {
      const s = m.index!;
      const e = s + m[0].length;
      if (allow && !allow(m[0], s)) continue;
      if (s > last) out.push({ text: text.slice(last, s), mark: false });
      out.push({ text: m[0], mark: true });
      last = e;
    }
    if (!out.length) return null;
    if (last < text.length) out.push({ text: text.slice(last), mark: false });
    return out;
  }

  /** Words eligible on line `lineNo` (1-based): those flagged on that line
   *  plus the page-wide ones (line == null, from unscoped fixes). */
  function wordsForLine(lineNo: number): string[] {
    const words = new Set<string>();
    for (const h of highlights) {
      if (h.line === null || h.line === lineNo) words.add(h.wrong);
    }
    return [...words];
  }

  // The page's per-line texts in the same precedence as displayText (manual
  // edits, then an applied AI fix, then the spell-fix projection, then raw) —
  // the lines the highlight entries' line numbers refer to.
  let basisLines = $derived.by(() => {
    if (!job || job.status !== "done" || !job.result) return null;
    if (job.manualText != null) return job.manualText.split("\n");
    return (
      job.llmFix?.fixedLines ??
      (fixSpelling ? job.spellFix?.fixedLines : undefined) ??
      job.result.lines.map((l) => l.text)
    );
  });

  // Merged view only: each basis line's span (start, end exclusive) inside
  // the merged displayText, located by a forward cursor (see the section
  // comment). Null entries for lines without a locatable span (dropped
  // empty lines, defensive misses) — they just never match.
  let mergedSpans = $derived.by(() => {
    if (numberedLines || !basisLines) return null;
    const spans: ([number, number] | null)[] = [];
    let cursor = 0;
    for (const line of basisLines) {
      const t = line.trim();
      const at = t ? displayText.indexOf(t, cursor) : -1;
      if (at === -1) {
        spans.push(null);
        continue;
      }
      spans.push([at, at + t.length]);
      cursor = at + t.length;
    }
    return spans;
  });

  // Merged-view segments: all flagged words match text-wide, but a match
  // only becomes a <mark> when it starts inside a flagged line's span — or
  // the word was flagged page-wide (line == null).
  let mergedSegs = $derived.by(() => {
    if (!mergedSpans) return null;
    const pageWide = new Set(
      highlights.filter((h) => h.line === null).map((h) => h.wrong),
    );
    const linesByWord = new Map<string, Set<number>>();
    for (const h of highlights) {
      if (h.line == null) continue;
      let set = linesByWord.get(h.wrong);
      if (!set) linesByWord.set(h.wrong, (set = new Set()));
      set.add(h.line);
    }
    return markText(
      displayText,
      highlights.map((h) => h.wrong),
      (word, start) => {
        if (pageWide.has(word)) return true;
        for (const ln of linesByWord.get(word) ?? []) {
          const sp = mergedSpans[ln - 1];
          if (sp && start >= sp[0] && start < sp[1]) return true;
        }
        return false;
      },
    );
  });

  // Line-numbered view: only in line-by-line mode, where each displayed row
  // IS a recognized line — the numbers then match the AI Check panel's L
  // chips (both count result.lines). With mergeParagraphs on there is no
  // stable per-row mapping, so the gutter is omitted rather than lying.
  let numberedLines = $derived(
    mergeParagraphs ? null : displayText.split("\n"),
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
          title={aiFixCount !== null
            ? "AI spell-check fixes applied (Gemini, user-reviewed)"
            : "Total Burmese spelling fixes applied (regex + dictionary)"}
        >{fixCount === 0
          ? "no fixes"
          : `${fixCount} ${fixCount === 1 ? "fix" : "fixes"}`}{aiFixCount !== null ? " · AI" : ""}</span>
      {/if}
      <button
        class="copy"
        class:editing
        onclick={() => (editing = !editing)}
        title={editing ? "Return to read-only view" : "Edit the text by hand"}
      >{editing ? "Done" : "Edit"}</button>
      {#if job.manualText != null}
        <button
          class="copy revert"
          onclick={revert}
          title="Discard manual edits and return to the generated text"
        >Revert</button>
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
    {:else if editing}
      <!-- Edit mode: the first keystroke stores the text as a manual
           override (see onEdit); value stays in sync because displayText
           then reads the same manual text back. Escape leaves edit mode. -->
      <textarea
        class="text"
        spellcheck="false"
        aria-label="Recognized text (editable)"
        value={displayText}
        oninput={onEdit}
        onkeydown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            editing = false;
          }
        }}
        use:focusNode
      ></textarea>
    {:else if displayText.trim()}
      {#if numberedLines}
        <!-- Line-by-line mode: numbered rows, one per recognized line. The
             numbers match the AI Spell Fix panel's L chips. -->
        <div class="lines" aria-label="Recognized text">
          {#each numberedLines as line, i (i)}
            {@const segs = markText(line, wordsForLine(i + 1))}
            <div class="line-row">
              <span class="line-no" aria-hidden="true">{i + 1}</span>
              <span class="line-text">
                {#if segs}
                  {#each segs as seg, k (k)}{#if seg.mark}<mark class="ai-hl">{seg.text}</mark>{:else}{seg.text}{/if}{/each}
                {:else}{line}{/if}
              </span>
            </div>
          {/each}
        </div>
      {:else}
        <!-- Merged-paragraph view — no line numbers (no stable row mapping);
             AI-highlight <mark>s still render, scoped to flagged lines via
             the span map (see mergedSegs). -->
        <pre class="text readonly">{#if mergedSegs}{#each mergedSegs as seg, i}{#if seg.mark}<mark class="ai-hl">{seg.text}</mark>{:else}{seg.text}{/if}{/each}{:else}{displayText}{/if}</pre>
      {/if}
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
      {#if totalAiFixes > 0}
        <span class="sb-sep">·</span>
        <span>AI {totalAiFixes} {totalAiFixes === 1 ? "fix" : "fixes"}</span>
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
  .copy.editing { color: var(--bg); background: var(--accent); }
  .copy.revert { color: var(--text-dim); background: var(--surface); }
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
  /* Editable text — styled to read like the old read-only <pre>: borderless,
     transparent, filling the panel. resize:none (dragging the grip would
     fight the panel layout); height 100% so click-to-focus works anywhere. */
  .text {
    margin: 0;
    width: 100%;
    height: 100%;
    resize: none;
    border: none;
    background: transparent;
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.6;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    padding: 0;
    outline: none;
    display: block;
  }
  /* Read-only <pre> shares the textarea's typography; only the textarea
     needs the focus affordance of a caret (cursor: text). */
  .text.readonly { cursor: default; }
  /* Numbered line view — same typography as the <pre>, with a right-aligned
     gutter. The gutter is user-select:none so copying (or dragging to select
     text) doesn't pick up the numbers. */
  .lines {
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.6;
  }
  .line-row {
    display: flex;
    gap: 10px;
    align-items: baseline;
  }
  .line-no {
    flex-shrink: 0;
    min-width: 2.5ch;
    text-align: right;
    color: var(--text-faint);
    user-select: none;
    font-variant-numeric: tabular-nums;
  }
  .line-text {
    flex: 1;
    min-width: 0;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text);
  }
  /* AI-flagged wrong words — matches the danger styling of the AI panel's
     "wrong" column so the two read as the same entity. */
  .text .ai-hl {
    background: var(--danger-soft);
    color: var(--danger);
    border-radius: 3px;
    padding: 0 2px;
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
