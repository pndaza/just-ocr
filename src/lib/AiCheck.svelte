<script lang="ts">
  import {
    llmSpellCheck,
    llmRewritePages,
    llmDailyLimit,
    LLM_MODELS,
    LLM_BATCH_SIZES,
    LLM_CONCURRENCY,
    type AiCheckMode,
    type Job,
    type LlmBatchSize,
    type LlmConcurrency,
    type LlmModel,
    type LlmWordFix,
  } from "./ocr";
  import { applyWordFixes } from "./result";
  import { diffWords } from "./diff";

  interface Props {
    /** The full batch (only done jobs with recognized lines are checked). */
    jobs: Job[];
    /** Google AI Studio API key from Settings ("" = not configured). */
    apiKey: string;
    /** Chosen Gemini flash model id — picked here via two-way binding,
     *  persisted globally by App. Bindable (not value+callback) so the
     *  <select> never desyncs from the state. */
    model: LlmModel;
    /** Pages per Gemini request, picked here via two-way binding. */
    batchSize: LlmBatchSize;
    /** Gemini requests kept in flight at once (1–3). Picked here via
     *  two-way binding, persisted globally by App — parallel requests
     *  speed up big checks; see LLM_CONCURRENCY for the default rationale. */
    concurrency: LlmConcurrency;
    /** Fix mode: "review" (wrong→correct word pairs) or "rewrite" (the model
     *  returns each page's corrected text). Picked here, persisted by App. */
    mode: AiCheckMode;
    /** Hide the panel (✕ or Esc). */
    onclose: () => void;
    /** Open the Settings modal (from the no-key notice). */
    onopensettings: () => void;
    /** Select a page's job so the Preview/thumbnail panels track it —
     *  called as the review cursor moves, keeping the image, text and fix
     *  list pointed at the same page. */
    onselectpage: (jobId: number) => void;
    /** Reports the current suggestions (jobId → wrong words) so the Text
     *  panel can highlight flagged words in the page text. Called whenever
     *  the suggestion set changes (new results, re-check). */
    onsuggestions: (wrong: Record<number, string[]>) => void;
  }
  let {
    jobs,
    apiKey,
    model = $bindable(),
    batchSize = $bindable(),
    concurrency = $bindable(),
    mode = $bindable(),
    onclose,
    onopensettings,
    onselectpage,
    onsuggestions,
  }: Props = $props();

  /** A reviewable wrong→correct pair plus its checkbox state. `reverted`
   *  (rewrite mode) marks a line whose correction the user rolled back to
   *  keep the original spelling. */
  interface FixItem extends LlmWordFix {
    checked: boolean;
    reverted?: boolean;
  }

  /** All suggestions for one page, in batch order. Pages the model found no
   *  errors on get an empty `fixes` list so the user sees they were checked. */
  interface PageReview {
    jobId: number;
    name: string;
    fixes: FixItem[];
  }

  type Phase = "ready" | "checking" | "review" | "applied";

  let phase = $state<Phase>("ready");
  let suggestions = $state<PageReview[]>([]);
  let progress = $state<{ current: number; total: number } | null>(null);
  let error = $state<string | null>(null);
  let cancelRequested = $state(false);
  let applied = $state<{ pages: number; fixes: number } | null>(null);
  // The mode the CURRENT results were produced with — captured at check
  // start so validation/apply/display stay correct even if the user flips
  // the toggle mid-review.
  let checkMode = $state<AiCheckMode>("review");
  // Jobs whose llmFix was set by THIS rewrite-mode check — drives the
  // "Undo all" control (instant apply needs an escape hatch).
  let appliedJobIds = $state<number[]>([]);

  let hasKey = $derived(apiKey.trim().length > 0);
  // Quota/rate-limit errors (parsed + phrased by the backend from Gemini's
  // QuotaFailure details). A daily free-tier cap can't be retried today, so
  // the banner swaps to a neutral tone and hides Retry.
  let quotaError = $derived(!!error && /daily limit|quota/i.test(error));
  let modelLabel = $derived(
    LLM_MODELS.find((m) => m.value === model)?.label ?? model,
  );
  // Pages eligible for checking: completed with at least one recognized line.
  let checkable = $derived(
    jobs.filter(
      (j) => j.status === "done" && j.result && j.result.lines.length > 0,
    ),
  );

  // One pass over every fix for all three header counters — with a big check
  // (thousands of rows) three separate reduces per checkbox toggle add up;
  // a single loop is effectively free.
  let stats = $derived.by(() => {
    let selected = 0;
    let total = 0;
    let pages = 0;
    for (const s of suggestions) {
      if (s.fixes.length) pages += 1;
      for (const f of s.fixes) {
        if (f.checked) selected += 1;
        if (!f.reverted) total += 1;
      }
    }
    return { selected, total, pages };
  });
  let selectedCount = $derived(stats.selected);
  let totalFixCount = $derived(stats.total);
  let pagesWithFixes = $derived(stats.pages);

  // ── Page range ──────────────────────────────────────────────────────────────
  // The flash models' free tier allows only ~20 requests/day — far too few
  // for a 600-page book in one go. The range picks a contiguous slice of the
  // checkable pages (1-based, inclusive, queue order), with the same UI as
  // the PDF dialog's range. The inputs are PRE-FILLED with the full span
  // (see the fill effect below) rather than relying on "empty = all";
  // clearing ONE field stays open-ended ("from" only → that page to the
  // end, "to" only → page 1 through it), and an invalid range blocks Start.
  // NOTE: `bind:value` on a type="number" input assigns a *number* once the
  // text is parseable (and "" when empty/clearing) — handle both.
  let pageFrom = $state<number | "">("");
  let pageTo = $state<number | "">("");

  function parsePageField(v: number | ""): number | null {
    if (v === "") return null;
    return Number.isInteger(v) && v >= 1 ? v : Number.NaN;
  }

  // Non-null while the entered range is invalid; shown in red beside the
  // inputs and blocks Start so a bad range can never plan a check.
  let rangeError = $derived.by(() => {
    const from = parsePageField(pageFrom);
    const to = parsePageField(pageTo);
    if (Number.isNaN(from) || Number.isNaN(to)) {
      return "Pages must be whole numbers ≥ 1";
    }
    if (from !== null && to !== null && to < from) {
      return "“To” must be ≥ “From”";
    }
    if (from !== null && from > checkable.length) {
      const plural = checkable.length === 1 ? "page" : "pages";
      return `Only ${checkable.length} ${plural} with recognized text`;
    }
    return null;
  });

  // The pages the current (valid) range selects — the check plan, the
  // request count and the quota warning all derive from this slice.
  let checkSlice = $derived.by(() => {
    if (rangeError || !checkable.length) return [];
    const from = parsePageField(pageFrom);
    const to = parsePageField(pageTo);
    if (from === null && to === null) return checkable;
    return checkable.slice((from ?? 1) - 1, to ?? checkable.length);
  });

  // Pre-fill the range inputs with the full span (1..N) — no "empty = all"
  // convention to remember. Self-healing: re-fills whenever BOTH fields are
  // empty (panel opened, more pages finished, or the user cleared both), and
  // stops the moment the user types anything — their values are theirs.
  $effect(() => {
    if (checkable.length > 0 && pageFrom === "" && pageTo === "") {
      pageFrom = 1;
      pageTo = checkable.length;
    }
  });

  // ── Free-tier quota guidance ────────────────────────────────────────────────
  // Requests the planned check (over the selected range) needs vs. the
  // model's free daily cap. Warning only — the API is the source of truth
  // (other usage today counts too).
  let requestCount = $derived(
    Math.ceil(checkSlice.length / batchSize),
  );
  let dailyLimit = $derived(llmDailyLimit(model));
  let overLimit = $derived(requestCount > dailyLimit);

  /**
   * The basis lines for a page: whatever the user currently sees — the
   * spell-fix projection when that toggle is on, raw lines otherwise.
   * Reviewing and fixing the displayed text keeps the review honest and
   * lets AI fixes stack on top of the offline spell-fix.
   */
  // Basis lines are read by every validation pass (`invalidKeys` re-derives
  // on each keystroke in a correction editor) and by apply — cache the
  // mapped array per job so huge lists don't re-allocate it each time.
  // Both identities are tracked because either swap changes the basis:
  // `spellFix` when the spelling toggle recomputes the projection, and
  // `result` when the page is re-OCR'd in place (processJob assigns a fresh
  // result AND nulls spellFix — checking spellFix alone would return the
  // pre-re-OCR lines when the old and new spellFix are both null).
  const basisCache = new WeakMap<
    Job,
    { spell: Job["spellFix"]; result: Job["result"]; lines: string[] }
  >();

  function basisLines(job: Job): string[] {
    const hit = basisCache.get(job);
    if (hit && hit.spell === job.spellFix && hit.result === job.result) {
      return hit.lines;
    }
    const lines = job.spellFix?.fixedLines ?? job.result!.lines.map((l) => l.text);
    basisCache.set(job, { spell: job.spellFix, result: job.result, lines });
    return lines;
  }

  function pageText(job: Job): string {
    return basisLines(job).join("\n");
  }

  // Batch plan for the in-flight check plus resume state, all indexed by
  // plan position: `batchDone[b]` once batch b's results have landed,
  // `batchPages[b]` holding its review rows. Kept as plain variables
  // (nothing renders from them directly) so an interrupted check can resume
  // from exactly the batches that never landed — each request burns
  // free-tier quota, and a restart would also drop the checkbox edits the
  // user already made on the collected pages. Per-batch flags (rather than
  // the old single cursor) because batches complete out of order once
  // requests run in parallel.
  let batchPlan: Job[][] = [];
  let batchDone: boolean[] = [];
  let batchPages: PageReview[][] = [];

  /** Start a fresh check: rebuild the plan from the currently selected
   *  page range (all checkable pages when empty) and discard any previously
   *  collected results. */
  function startCheck() {
    if (!checkSlice.length || !hasKey) return;
    batchPlan = [];
    for (let i = 0; i < checkSlice.length; i += batchSize) {
      batchPlan.push(checkSlice.slice(i, i + batchSize));
    }
    batchDone = batchPlan.map(() => false);
    batchPages = batchPlan.map(() => []);
    suggestions = [];
    renderedSections = SECTION_CHUNK;
    appliedJobIds = [];
    checkMode = mode;
    applied = null;
    void runBatches();
  }

  /** Resume an interrupted check from the first unfinished batch, keeping
   *  the pages already collected (and any edits made to them). */
  function retryCheck() {
    if (!batchPlan.length || !hasKey) return;
    void runBatches();
  }

  /**
   * Run ONE batch: send the request and shape the response into review
   * rows for its pages. In "review" mode the model returns wrong→correct
   * word pairs; in "rewrite" mode it returns each page's corrected text,
   * diffed per line into change rows (old → new), applied to the job the
   * moment the response lands (instant-apply contract; "Undo all"
   * reverts). Only pages with actual diffs enter the list in rewrite mode
   * — clean pages aren't shown at all (the "No changes needed" notice
   * covers the all-clean case).
   */
  async function checkBatch(batch: Job[], texts: string[]): Promise<PageReview[]> {
    if (checkMode === "rewrite") {
      const result = await llmRewritePages(apiKey, model, texts);
      const pages: PageReview[] = [];
      for (let i = 0; i < batch.length; i++) {
        const job = batch[i];
        const newLines = result.find((p) => p.page === i + 1)?.lines ?? null;
        const basis = basisLines(job);
        const fixes: FixItem[] = [];
        if (newLines) {
          // Diff per line; a short/long response degrades gracefully
          // (extra lines ignored, missing lines stay original).
          const n = Math.min(newLines.length, basis.length);
          let changed = 0;
          const out = [...basis];
          for (let l = 0; l < n; l++) {
            if (newLines[l] !== basis[l]) {
              // Whitespace-only differences (trailing spaces, line-number
              // echo remnants) are noise — don't create ghost rows.
              if (newLines[l].trim() === basis[l].trim()) continue;
              fixes.push({
                wrong: basis[l],
                correct: newLines[l],
                line: l + 1,
                checked: true,
              });
              out[l] = newLines[l];
              changed += 1;
            }
          }
          if (changed > 0) {
            job.llmFix = { fixedLines: out, fixes: changed };
            appliedJobIds = [...appliedJobIds, job.id];
          }
        }
        if (fixes.length) {
          pages.push({ jobId: job.id, name: job.name, fixes });
        }
      }
      return pages;
    }
    const result = await llmSpellCheck(apiKey, model, texts);
    return batch.map((job, i) => ({
      jobId: job.id,
      name: job.name,
      fixes: (result.find((p) => p.page === i + 1)?.fixes ?? []).map(
        (f) => ({ ...f, checked: true }),
      ),
    }));
  }

  /**
   * Run the remaining batches with a worker pool of `concurrency` in-flight
   * requests (each still the user-chosen `batchSize` pages). Workers pull
   * the next un-started batch until the plan is exhausted, Stop lands, or a
   * request fails; in-flight requests always finish and their pages stay
   * collected — only NEW batches are held back, same contract as the old
   * sequential loop. A failure keeps the already-collected pages reviewable
   * — the error shows alongside the results rather than discarding them,
   * and Retry resumes from the batches that never landed instead of
   * re-sending them.
   */
  async function runBatches() {
    cancelRequested = false;
    error = null;
    phase = "checking";
    const total = batchPlan.length;
    // Progress counts batches STARTED (in flight or landed), not just
    // landed — with slow LLM responses, a landed-only counter sits at
    // "0 of 3" while the first requests are still in flight and reads as
    // stuck (same pre-tick convention as the batch Run All counter).
    // Resumes start from the already-collected count.
    let accounted = batchDone.filter(Boolean).length;
    progress = { current: Math.min(accounted, total), total };
    let cancelled = false;
    let firstError: string | null = null;
    let next = 0; // next batch index to START (skipping collected ones)
    // Pool size is clamped to the plan (and ≥1) so a 1-batch check with
    // concurrency 3 still sends exactly one request.
    const poolSize = Math.max(1, Math.min(concurrency, Math.max(total, 1)));

    const runWorker = async () => {
      // The check+pull below is synchronous (no await between), so workers
      // can't race past each other on `next`.
      while (!cancelRequested && firstError === null) {
        // Pull the next batch that hasn't been collected yet — with parallel
        // completion, later batches can be done while an earlier one failed,
        // and a resumed Retry must not re-send those (each request burns
        // free-tier quota).
        while (next < total && batchDone[next]) next += 1;
        if (next >= total) return;
        const b = next;
        next += 1;
        accounted += 1;
        progress = { current: Math.min(accounted, total), total };
        const batch = batchPlan[b];
        try {
          batchPages[b] = await checkBatch(batch, batch.map(pageText));
          batchDone[b] = true;
          // No progress tick on landing — the batch was counted when it
          // started; the bar reaches N/N as the last batches are pulled.
        } catch (e: any) {
          // Remember the first failure; the condition above stops every
          // worker from pulling further batches once it's set.
          if (firstError === null) {
            firstError = typeof e === "string" ? e : e?.message ?? String(e);
          }
          return;
        }
      }
      if (cancelRequested) cancelled = true;
    };
    await Promise.all(
      Array.from({ length: poolSize }, () => runWorker()),
    );

    // Assemble the review list in PLAN order — batches complete out of
    // order under parallel requests, but the list should read in page
    // order regardless of which response arrived first.
    suggestions = [];
    for (let b = 0; b < total; b++) {
      if (batchDone[b]) suggestions.push(...batchPages[b]);
    }
    error = firstError;
    progress = null;
    // A stop before anything came back just returns to the start screen.
    if (cancelled && !suggestions.length && !error) {
      phase = "ready";
      return;
    }
    phase = "review";
  }

  /** Stop after the in-flight batch; collected pages stay reviewable. */
  function stopCheck() {
    cancelRequested = true;
  }

  /** Revert every llmFix this rewrite-mode check applied. The diff rows stay
   *  as a reference of what the model proposed. */
  function undoAll() {
    for (const id of appliedJobIds) {
      const job = jobs.find((j) => j.id === id);
      if (job) job.llmFix = null;
    }
    appliedJobIds = [];
  }

  /** Rewrite mode: keep ONE line's original text — the model sometimes
   *  "corrects" old/traditional spelling to the modern form, which isn't
   *  always wanted. Restores the line in the applied projection and dims the
   *  row; the last revert on a page drops its (now no-op) llmFix. */
  function revertFix(s: PageReview, i: number) {
    const f = s.fixes[i];
    if (f.reverted) return;
    const job = jobs.find((j) => j.id === s.jobId);
    // Nothing applied left to revert (Undo all already ran, or the job went
    // away) — bail before marking the row so it doesn't dim for a no-op.
    if (!job?.result || !job.llmFix || f.line == null) return;
    const fixed = job.llmFix.fixedLines;
    if (f.line < 1 || f.line > fixed.length) return;
    f.reverted = true;
    fixed[f.line - 1] = f.wrong;
    const remaining = s.fixes.filter((x) => !x.reverted).length;
    job.llmFix = remaining === 0 ? null : { fixedLines: fixed, fixes: remaining };
  }

  /** Whether this page still has its rewrite corrections applied — drives
   *  the per-row revert buttons (after Undo all there's nothing to revert). */
  function pageApplied(s: PageReview): boolean {
    return jobs.find((j) => j.id === s.jobId)?.llmFix != null;
  }

  /**
   * Apply the checked items. For each page with ≥1 checked item, transform
   * that job's basis lines (spell-fixed when present, raw otherwise) and
   * cache the result on `job.llmFix` — a display-time projection exactly
   * like the offline spell-fix; `job.result` is never mutated. The Text
   * panel and exports pick the projection up reactively.
   *
   * Review mode replaces wrong→correct word pairs (line-scoped when the
   * model addressed a line). Rewrite mode replaces whole lines: `correct` is
   * the model's new line text, `line` picks the slot.
   */
  function applyFixes() {
    let pages = 0;
    let fixes = 0;
    for (const s of suggestions) {
      // Invalid rows (edited wrong-word no longer in the page) are excluded —
      // they're unchecked automatically, this is belt-and-braces.
      const checked = s.fixes.filter(
        (f, i) => f.checked && !invalidKeys.has(`${s.jobId}:${i}`),
      );
      if (!checked.length) continue;
      const job = jobs.find((j) => j.id === s.jobId);
      if (!job?.result) continue; // job was removed mid-review — skip
      const base = basisLines(job);
      let fixedLines: string[];
      let jobCount: number;
      if (checkMode === "rewrite") {
        // Direct line replacement; the count is changed lines.
        const out = [...base];
        let count = 0;
        for (const f of checked) {
          if (f.line == null || f.line < 1 || f.line > out.length) continue;
          out[f.line - 1] = f.correct;
          count += 1;
        }
        fixedLines = out;
        jobCount = count;
      } else {
        const { lines, count } = applyWordFixes(base, checked);
        fixedLines = lines;
        jobCount = count;
      }
      job.llmFix = { fixedLines, fixes: jobCount };
      pages += 1;
      fixes += jobCount;
    }
    applied = { pages, fixes };
    phase = "applied";
  }

  function pageCheckedCount(s: PageReview): number {
    return s.fixes.filter((f) => f.checked).length;
  }

  // ── Keyboard review ────────────────────────────────────────────────────────
  // Focus-independent cursor over the flattened list of checkbox rows (each
  // page's select-all row + its fix rows). Window-level keys drive it so
  // ↑/↓/Space/Enter work no matter where — or whether — DOM focus sits; the
  // cursor is rendered as a highlight rather than relying on native focus.
  // Clicking a row also moves the cursor (see the onclick handlers), so mouse
  // and keyboard share one selection instead of two competing highlights.
  let pagesEl = $state<HTMLElement | null>(null);

  interface CursorRow {
    /** Stable per-row key: "<jobId>:all" for a page select-all, "<jobId>:<i>" for fix i. */
    key: string;
    s: PageReview;
    /** Fix index within the page; -1 for the page select-all row. */
    i: number;
  }

  let flatRows = $derived.by(() => {
    const rows: CursorRow[] = [];
    for (const s of suggestions) {
      if (s.fixes.length) rows.push({ key: `${s.jobId}:all`, s, i: -1 });
      s.fixes.forEach((_, i) => rows.push({ key: `${s.jobId}:${i}`, s, i }));
    }
    return rows;
  });

  let cursorKey = $state("");

  // ── Inline correction editing ──────────────────────────────────────────────
  // Both words read as plain text; double-click turns either into an input.
  // Enter or blur commits, Escape reverts to the pre-edit value. editKey uses
  // the cursor's "<jobId>:<i>" scheme plus a field suffix (":w" / ":c").
  let editKey = $state<string | null>(null);
  let editOriginal = $state("");

  function startEdit(key: string, value: string) {
    editOriginal = value;
    editKey = key;
  }

  /** Focus + select the freshly-opened editor so typing replaces the word. */
  function autofocusSelect(node: HTMLInputElement) {
    node.focus();
    node.select();
    return {};
  }

  // ── Edited-wrong-word validation ───────────────────────────────────────────
  // A fix needs its `wrong` to occur where it's addressed: on its `line`
  // when the model gave one, anywhere on the page otherwise. The backend
  // pre-validates what the model returned, but a user EDIT can break the
  // match — so track invalid rows live, dim them, and exclude them from
  // Apply. Rewrite-mode rows replace the whole line by index, so only the
  // line index needs to be in range there (`wrong` is just the old-line
  // snapshot for display and may legitimately be an empty line).
  let invalidKeys = $derived.by(() => {
    const invalid = new Set<string>();
    for (const s of suggestions) {
      const job = jobs.find((j) => j.id === s.jobId);
      if (!job?.result) continue;
      const lines = basisLines(job);
      s.fixes.forEach((f, i) => {
        let ok: boolean;
        if (checkMode === "rewrite") {
          ok = f.line != null && f.line >= 1 && f.line <= lines.length;
        } else {
          ok =
            f.line != null
              ? f.line >= 1 &&
                f.line <= lines.length &&
                lines[f.line - 1].includes(f.wrong)
              : !!f.wrong && lines.some((l) => l.includes(f.wrong));
        }
        if (!ok) invalid.add(`${s.jobId}:${i}`);
      });
    }
    return invalid;
  });

  // Uncheck invalid rows as they become invalid — a checked-but-unappliable
  // fix would silently vanish at Apply and read as a bug.
  $effect(() => {
    for (const s of suggestions) {
      s.fixes.forEach((f, i) => {
        if (f.checked && invalidKeys.has(`${s.jobId}:${i}`)) f.checked = false;
      });
    }
  });

  // Keep the cursor valid: point it at the first row when the review list
  // appears (or when its row disappears because suggestions were replaced).
  $effect(() => {
    if (phase === "review" && flatRows.length && !flatRows.some((r) => r.key === cursorKey)) {
      cursorKey = flatRows[0].key;
    }
  });

  // ── Chunked list rendering ──────────────────────────────────────────────────
  // A big check can produce thousands of rows; mounting them all at once
  // freezes the UI on one long layout/paint and makes scrolling heavy. Page
  // sections mount in chunks: a sentinel at the end of the list extends the
  // mounted prefix as the user scrolls toward it, and keyboard moves extend
  // it themselves (moveCursor). Mounted sections stay mounted — DOM nodes
  // are only created once — while offscreen ones cost nothing to paint
  // thanks to `content-visibility` (see the .page rule).
  const SECTION_CHUNK = 12;
  let listSections = $derived(
    // Rewrite mode: only pages with actual changes — the "no changes" filler
    // sections are noise when the corrections are already applied. Review
    // mode keeps them as confirmation of coverage.
    checkMode === "rewrite"
      ? suggestions.filter((s) => s.fixes.length > 0)
      : suggestions,
  );
  let renderedSections = $state(SECTION_CHUNK);
  let visibleSections = $derived(listSections.slice(0, renderedSections));
  let sentinelEl = $state<HTMLElement | null>(null);

  // Extend while the sentinel is near the viewport (a screenful of lookahead
  // so fast scrolls don't hit pop-in). Re-arming is automatic in the other
  // direction: the sentinel remounts whenever the list grows past the
  // mounted prefix (e.g. a resumed Retry adds more pages).
  $effect(() => {
    const el = sentinelEl;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        renderedSections = Math.min(
          listSections.length,
          renderedSections + SECTION_CHUNK,
        );
        // The observer only fires on visibility TRANSITIONS — extending can
        // leave the sentinel still intersecting (short sections), which
        // would strand the list before the fold. Re-observing forces a
        // fresh initial callback, so the chain continues until the sentinel
        // really leaves the margin (or the list runs out).
        io.disconnect();
        io.observe(el);
      },
      // Null root = viewport (the panel body is the actual scroller, but
      // viewport intersection tracks its scroll just fine).
      { rootMargin: "600px" },
    );
    io.observe(el);
    return () => io.disconnect();
  });

  function moveCursor(delta: number) {
    if (!flatRows.length) return;
    let idx = flatRows.findIndex((r) => r.key === cursorKey);
    idx = Math.min(flatRows.length - 1, Math.max(0, (idx === -1 ? 0 : idx) + delta));
    const row = flatRows[idx];
    cursorKey = row.key;
    // The row's section may not be mounted yet — extend first, then scroll
    // after Svelte flushes the new DOM (rAF is past that flush).
    const secIdx = listSections.indexOf(row.s);
    if (secIdx >= 0 && secIdx >= renderedSections) {
      renderedSections = Math.min(listSections.length, secIdx + SECTION_CHUNK);
      requestAnimationFrame(() =>
        pagesEl
          ?.querySelector(`[data-key="${row.key}"]`)
          ?.scrollIntoView({ block: "nearest" }),
      );
    } else {
      pagesEl
        ?.querySelector(`[data-key="${row.key}"]`)
        ?.scrollIntoView({ block: "nearest" });
    }
  }

  // Sync the Preview/thumbnail panels with the review cursor: whatever row
  // the cursor sits on (after ↑/↓ or a click), that page gets selected so
  // its image shows beside the fixes being reviewed.
  $effect(() => {
    if (phase !== "review" || !cursorKey) return;
    const row = flatRows.find((r) => r.key === cursorKey);
    if (row) onselectpage(row.s.jobId);
  });

  // Publish the flagged wrong words per page so the Text panel can
  // highlight them while reviewing. Review mode only — rewrite mode applies
  // instantly, so the old text (and its highlights) is already replaced;
  // publishing {} then clears any highlights left over from a previous
  // review-mode check.
  $effect(() => {
    if (checkMode !== "review") {
      onsuggestions({});
      return;
    }
    const map: Record<number, string[]> = {};
    for (const s of suggestions) {
      if (s.fixes.length) {
        map[s.jobId] = [...new Set(s.fixes.map((f) => f.wrong))];
      }
    }
    onsuggestions(map);
  });

  /** Toggle the row under the cursor (a single fix, or a page's select-all). */
  function toggleCursor() {
    const row = flatRows.find((r) => r.key === cursorKey);
    if (!row) return;
    if (row.i === -1) togglePage(row.s, !pageAllChecked(row.s));
    else row.s.fixes[row.i].checked = !row.s.fixes[row.i].checked;
  }

  function pageAllChecked(s: PageReview): boolean {
    return s.fixes.length > 0 && s.fixes.every((f) => f.checked);
  }

  function togglePage(s: PageReview, on: boolean) {
    for (const f of s.fixes) f.checked = on;
  }

  function setAll(on: boolean) {
    for (const s of suggestions) togglePage(s, on);
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      if (phase !== "checking") onclose();
      return;
    }
    if (phase !== "review" || !flatRows.length) return;

    // When a real control has focus, let the browser handle its keys
    // natively (Space clicks a focused button, Enter opens a select) and
    // skip our global handling so actions never double-fire. Checkboxes are
    // exempt: native arrows do nothing there, so the cursor stays ours.
    // Text inputs (the editable corrections) and the Text panel's textarea
    // are also exempt — typing must reach the field, not toggle rows.
    const t = e.target as HTMLElement | null;
    const onNativeControl =
      !!t &&
      (t.tagName === "BUTTON" ||
        t.tagName === "SELECT" ||
        t.tagName === "TEXTAREA" ||
        (t.tagName === "INPUT" && (t as HTMLInputElement).type === "text"));

    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (onNativeControl) return;
      e.preventDefault();
      moveCursor(e.key === "ArrowDown" ? 1 : -1);
    } else if (e.key === " " || e.code === "Space") {
      if (onNativeControl || checkMode === "rewrite") return; // nothing to toggle
      e.preventDefault(); // keep the panel from scrolling
      toggleCursor();
    } else if (
      (e.key === "Enter" || e.key === "F2") &&
      phase === "review" &&
      checkMode === "review"
    ) {
      // Enter/F2 = Excel-style "edit cell": open the editor on the cursor
      // row's correction (Shift → the wrong word, which must keep matching
      // the page text). Page-header rows (-1) have no words to edit. While a
      // button/select has focus Enter keeps its native meaning (click/open).
      if (onNativeControl) return;
      const row = flatRows.find((r) => r.key === cursorKey);
      if (!row || row.i < 0) return;
      e.preventDefault();
      const fix = row.s.fixes[row.i];
      const field = e.shiftKey ? "w" : "c";
      startEdit(`${row.s.jobId}:${row.i}:${field}`, field === "w" ? fix.wrong : fix.correct);
    }
  }
</script>

<svelte:window onkeydown={onKey} />

<section class="panel" role="region" aria-label="AI spell fix">
  <div class="head">
    <span class="title">AI Spell Fix</span>
    <!-- Model chip only when the model selector isn't on screen (checking /
         review / applied). In the ready phase it would sit right above the
         selector and crowd it in a narrow panel. -->
    {#if phase !== "ready"}
      <span class="model-chip" title="Gemini model — change on the start screen">{modelLabel}</span>
    {/if}
    <button class="close" onclick={onclose} aria-label="Close AI spell fix panel">✕</button>
  </div>

  <div class="body">
    {#if phase === "ready"}
      {#if !hasKey}
        <div class="notice">
          <p><strong>No API key configured.</strong></p>
          <p class="hint">
            The spell fix sends recognized text to Gemini (Google AI Studio)
            to find spelling errors. Add your free API key in Settings —
            this is the app's only online feature.
          </p>
          <button class="btn primary" onclick={onopensettings}>
            Open Settings
          </button>
        </div>
      {:else if checkable.length === 0}
        <div class="notice">
          <p>No pages with recognized text yet.</p>
          <p class="hint">Run OCR on at least one image, then check it here.</p>
        </div>
      {:else}
          <div class="intro">
            <!-- Section label over the mode toggle — same tiny-uppercase
                 typography as the option-row labels below, and a bare noun
                 like them: the Auto/Manual buttons supply the "how". The old
                 "{N} pages will be proofread" intro lived here but said
                 nothing the range inputs and request count don't show. -->
            <span class="mode-lbl">Spell fixes</span>
            <div class="seg" role="radiogroup" aria-label="How to apply spell fixes">
              <button
                class="seg-btn"
                class:active={mode === "rewrite"}
                onclick={() => (mode = "rewrite")}
                role="radio"
                aria-checked={mode === "rewrite"}
                title="The model returns each page's corrected text, applied instantly — changed lines are listed so you can revert any"
              >Auto apply</button>
              <button
                class="seg-btn"
                class:active={mode === "review"}
                onclick={() => (mode = "review")}
                role="radio"
                aria-checked={mode === "review"}
                title="The model returns wrong→correct word pairs — you pick which to apply"
              >Manual apply</button>
            </div>
            <p class="hint">
              {#if mode === "review"}
                Gemini returns only the words it believes are wrong, with a
                suggested correction — you pick which to apply.
              {:else}
                Gemini rewrites each page's text — catches punctuation,
                spacing and phrasing too. Fixes apply the moment each batch
                returns; changed lines are listed so you can revert any (or
                Undo all). Uses more output tokens, so prefer a smaller
                batch size.
              {/if}
            </p>
            <!-- Four option rows (Pages / Model / Pages/req / Parallel) with
                 a fixed-width label column so every control starts at the
                 same left edge — same alignment idea as the PDF dialog. -->
            <div class="opts">
              <div class="opt-row">
                <span class="opt-lbl">Pages</span>
                <div class="range">
                  <input
                    class="num"
                    type="number"
                    min="1"
                    placeholder="from"
                    bind:value={pageFrom}
                    aria-label="First page to check"
                  />
                  <span class="dash" aria-hidden="true">–</span>
                  <input
                    class="num"
                    type="number"
                    min="1"
                    placeholder="to"
                    bind:value={pageTo}
                    aria-label="Last page to check"
                  />
                  {#if rangeError}
                    <span class="range-hint error">{rangeError}</span>
                  {/if}
                </div>
              </div>
              <label class="opt-row">
                <span class="opt-lbl">Model</span>
                <select bind:value={model} title="Free-tier daily request limits shown per model">
                  {#each LLM_MODELS as m}
                    <option value={m.value}>{m.label} · {llmDailyLimit(m.value)} req/day</option>
                  {/each}
                </select>
              </label>
              <label class="opt-row">
                <span class="opt-lbl">Pages/req</span>
                <select
                  class="sm"
                  bind:value={batchSize}
                  title="Fewer pages per request is steadier; more is faster but risks output limits"
                >
                  {#each LLM_BATCH_SIZES as s}<option value={s}>{s}</option>{/each}
                </select>
              </label>
              <label class="opt-row">
                <span class="opt-lbl">Parallel</span>
                <select
                  class="sm"
                  bind:value={concurrency}
                  title="Requests sent to Gemini at once — faster on big checks, but more can trip the per-minute rate limit"
                >
                  {#each LLM_CONCURRENCY as c}<option value={c}>{c}</option>{/each}
                </select>
              </label>
            </div>
            <span class="hint req-hint">
              → {requestCount}
              {requestCount === 1 ? "request" : "requests"}
              · free tier {dailyLimit}/day
            </span>
            {#if overLimit}
              <div class="limit-warning" role="alert">
                <strong>Over the free-tier daily limit.</strong>
                This check needs {requestCount} requests, but {modelLabel}
                allows {dailyLimit}/day — it will likely stop partway with a
                quota error. Increase pages per request, narrow the page
                range, or run the rest tomorrow.
              </div>
            {/if}
          <button class="btn primary" onclick={startCheck} disabled={!checkSlice.length}>
            Start Check
          </button>
        </div>
      {/if}
    {:else if phase === "checking"}
      <div class="checking">
        <span class="spin" aria-hidden="true"></span>
        <div class="check-progress">
          <p>
            <!-- Progress counts batches STARTED (in flight or landed) — see
                 runBatches for why a landed-only counter reads as stuck. -->
            Checking {progress?.current ?? 0} of {progress?.total ?? 0} batches…
          </p>
          <div
            class="bar"
            role="progressbar"
            aria-valuenow={progress?.current ?? 0}
            aria-valuemin={0}
            aria-valuemax={progress?.total ?? 1}
          >
            <div
              class="fill"
              style="width:{progress && progress.total ? (progress.current / progress.total) * 100 : 0}%"
            ></div>
          </div>
          <button class="btn ghost" onclick={stopCheck} disabled={cancelRequested}>
            {cancelRequested ? "Stopping…" : "Stop"}
          </button>
        </div>
      </div>
    {:else if phase === "review"}
      {#if error}
        <div class="error" class:quota={quotaError}>
          <strong>
            {quotaError ? "Gemini quota reached." : "The check didn't finish."}
          </strong>
          <p>{error}</p>
          {#if suggestions.length}
            <p class="hint">Pages already checked are listed below — you can still review and apply their fixes.</p>
          {/if}
          {#if !quotaError}
            <button
              class="btn ghost"
              onclick={retryCheck}
              title="Continue from where the check stopped — pages already collected aren't re-sent"
            >Retry</button>
          {/if}
        </div>
      {/if}

      {#if suggestions.length}
        {#if checkMode === "rewrite"}
          <div class="controls">
            <span class="summary">
              {pagesWithFixes}
              {pagesWithFixes === 1 ? "page" : "pages"} ·
              <strong>{totalFixCount}</strong>
              {totalFixCount === 1 ? "line" : "lines"} corrected — applied automatically
            </span>
            <span class="kbd-hint">↑↓ move</span>
            <span class="spacer"></span>
            <button class="btn ghost" onclick={startCheck}>Re-check</button>
            <button
              class="btn danger"
              onclick={undoAll}
              disabled={appliedJobIds.length === 0}
              title="Revert every correction this check applied"
            >Undo all</button>
          </div>
        {:else}
          <div class="controls">
            <span class="summary">
              {pagesWithFixes}
              {pagesWithFixes === 1 ? "page" : "pages"} ·
              <strong>{selectedCount}</strong>/{totalFixCount} selected
            </span>
            <span class="kbd-hint">↑↓ move · enter edit · space toggle</span>
            <span class="spacer"></span>
            <button class="btn ghost" onclick={() => setAll(true)}>All</button>
            <button class="btn ghost" onclick={() => setAll(false)}>None</button>
            <button
              class="btn primary"
              onclick={applyFixes}
              disabled={selectedCount === 0}
            >
              Apply {selectedCount} {selectedCount === 1 ? "Fix" : "Fixes"}
            </button>
          </div>
        {/if}

        <div class="bind-pages" bind:this={pagesEl}>
          <!-- Only the first `renderedSections` sections mount (chunked
               rendering — see script); the sentinel below extends that as
               the user scrolls. -->
          {#each visibleSections as s (s.jobId)}
            <section class="page">
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="page-head"
                class:muted={!s.fixes.length}
                class:cursor={cursorKey === `${s.jobId}:all` && checkMode === "review"}
                data-key={`${s.jobId}:all`}
                onclick={() =>
                  s.fixes.length && checkMode === "review" && (cursorKey = `${s.jobId}:all`)}
              >
                {#if s.fixes.length && checkMode === "review"}
                  <input
                    type="checkbox"
                    checked={pageAllChecked(s)}
                    onchange={(e) => togglePage(s, e.currentTarget.checked)}
                    title="Toggle all fixes on this page"
                  />
                {:else if !s.fixes.length}
                  <span class="ok-dot" aria-hidden="true">✓</span>
                {/if}
                <span class="page-name">{s.name}</span>
                {#if s.fixes.length}
                  <span class="page-count">
                    {checkMode === "rewrite"
                      ? `${s.fixes.length} ${s.fixes.length === 1 ? "line" : "lines"}`
                      : `${pageCheckedCount(s)}/${s.fixes.length}`}
                  </span>
                {:else}
                  <span class="page-ok">no changes</span>
                {/if}
              </div>
              {#if s.fixes.length}
                <ul class="fix-list">
                  {#each s.fixes as f, i (i)}
                    <li>
                      {#if checkMode === "rewrite"}
                        <!-- Audit view: corrections already applied; the row
                             shows a word-level inline diff of old → new. -->
                        <!-- svelte-ignore a11y_no_static_element_interactions -->
                        <div
                          class="fix diffline"
                          class:cursor={cursorKey === `${s.jobId}:${i}`}
                          class:reverted={f.reverted}
                          data-key={`${s.jobId}:${i}`}
                          onclick={() => (cursorKey = `${s.jobId}:${i}`)}
                        >
                          <button
                            class="revert-btn"
                            class:done={f.reverted}
                            onclick={(e) => {
                              e.stopPropagation();
                              revertFix(s, i);
                            }}
                            disabled={f.reverted || !pageApplied(s)}
                            title={f.reverted
                              ? "Original kept"
                              : pageApplied(s)
                                ? "Keep the original — revert this line's correction"
                                : "Corrections were undone"}
                            aria-label="Keep original line"
                          >↺</button>
                          <span class="line-chip" title="Line on the page">L{f.line}</span>
                          {#if f.reverted}
                            <span class="d-kept" title="Original spelling kept">{f.wrong}</span>
                          {:else}
                            <!-- Whole line with an inline word diff: unchanged
                                 words render plain, removed words on the danger
                                 tint, corrected words on the ok tint. -->
                            <div class="diff-body">
                              {#each diffWords(f.wrong, f.correct) as seg, k (k)}
                                {#if seg.type === "del"}
                                  <span class="d-del" title="removed">{seg.text}</span>
                                {:else if seg.type === "add"}
                                  <span class="d-add" title="corrected">{seg.text}</span>
                                {:else if seg.type !== "gap"}
                                  <span>{seg.text}</span>
                                {/if}
                              {/each}
                            </div>
                          {/if}
                        </div>
                      {:else}
                        <!-- Not a <label>: only the checkbox itself toggles
                             (clicking elsewhere in the row just moves the
                             cursor). -->
                        <!-- svelte-ignore a11y_no_static_element_interactions -->
                        <div
                          class="fix"
                          class:cursor={cursorKey === `${s.jobId}:${i}`}
                          class:invalid={invalidKeys.has(`${s.jobId}:${i}`)}
                          data-key={`${s.jobId}:${i}`}
                          onclick={() => (cursorKey = `${s.jobId}:${i}`)}
                        >
                          <input type="checkbox" bind:checked={f.checked} />
                          {#if f.line != null}
                            <span class="line-chip" title="Line on the page the word appears on">L{f.line}</span>
                          {/if}
                          {#if editKey === `${s.jobId}:${i}:w`}
                            <input
                              class="word-input wrong-edit"
                              type="text"
                              bind:value={f.wrong}
                              spellcheck="false"
                              aria-label="Wrong word"
                              use:autofocusSelect
                              onkeydown={(e) => {
                                if (e.key === "Enter") {
                                  e.preventDefault();
                                  editKey = null; // commit
                                } else if (e.key === "Escape") {
                                  e.stopPropagation();
                                  f.wrong = editOriginal;
                                  editKey = null;
                                }
                              }}
                              onblur={() => (editKey = null)}
                            />
                          {:else}
                            <span
                              class="wrong"
                              title="Double-click to edit — must match the page text exactly"
                              ondblclick={() => startEdit(`${s.jobId}:${i}:w`, f.wrong)}
                            >{f.wrong}</span>
                          {/if}
                          <span class="arrow" aria-hidden="true">→</span>
                          {#if editKey === `${s.jobId}:${i}:c`}
                            <input
                              class="word-input correct-edit"
                              type="text"
                              bind:value={f.correct}
                              spellcheck="false"
                              aria-label="Corrected word"
                              use:autofocusSelect
                              oninput={() => {
                                // Editing an unchecked row signals intent to
                                // apply the edited correction.
                                if (!f.checked) f.checked = true;
                              }}
                              onkeydown={(e) => {
                                if (e.key === "Enter") {
                                  e.preventDefault();
                                  editKey = null; // commit
                                } else if (e.key === "Escape") {
                                  // Cancel the edit — and stop the window-level
                                  // Escape from closing the whole panel.
                                  e.stopPropagation();
                                  f.correct = editOriginal;
                                  editKey = null;
                                }
                              }}
                              onblur={() => (editKey = null)}
                            />
                          {:else}
                            <span
                              class="correct"
                              title="Double-click to edit the correction"
                              ondblclick={() => startEdit(`${s.jobId}:${i}:c`, f.correct)}
                            >{f.correct}</span>
                          {/if}
                          {#if invalidKeys.has(`${s.jobId}:${i}`)}
                            <span class="invalid-note" title="This word doesn't occur in the page text, so the fix can't apply">not in page</span>
                          {/if}
                        </div>
                      {/if}
                    </li>
                  {/each}
                </ul>
              {/if}
            </section>
          {/each}
          {#if renderedSections < listSections.length}
            <!-- Invisible tripwire: intersecting the viewport (or its 600px
                 lookahead) mounts the next chunk of page sections. -->
            <div class="more-sentinel" bind:this={sentinelEl} aria-hidden="true"></div>
          {/if}
        </div>
      {:else if !error}
        <div class="notice">
          <p>
            {checkMode === "rewrite" ? "No changes needed. ✓" : "No spelling issues found. ✓"}
          </p>
        </div>
      {/if}
    {:else if phase === "applied"}
      <div class="notice">
        <p>
          <strong>
            {applied?.fixes ?? 0}
            {applied?.fixes === 1 ? "fix" : "fixes"}
            applied to {applied?.pages ?? 0}
            {applied?.pages === 1 ? "page" : "pages"}.
          </strong>
        </p>
        <p class="hint">
          Corrected text now shows in the Text panel and is included in
          exports. Re-run OCR on a page to discard its applied fixes.
        </p>
        <div class="row">
          <button class="btn ghost" onclick={startCheck}>Check Again</button>
        </div>
      </div>
    {/if}
  </div>
</section>

<style>
  /* Panel frame — mirrors Output.svelte so the two right-hand columns read
     as a pair under the toolbar. */
  .panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-width: 0;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 8px;
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
  .model-chip {
    font-size: 10px;
    font-family: var(--mono);
    color: var(--text-faint);
    background: var(--bg-inset);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 2px 8px;
    white-space: nowrap;
  }
  .close {
    background: none;
    border: none;
    color: var(--text-faint);
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 6px;
    line-height: 1;
  }
  .close:hover { color: var(--text); }
  .body {
    flex: 1;
    overflow-y: auto;
    padding: 14px;
    min-height: 0;
  }
  .notice,
  .intro {
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: flex-start;
    padding: 4px 0;
    font-size: 13px;
    color: var(--text-dim);
  }
  .notice p,
  .intro p {
    margin: 0;
  }
  .hint {
    font-size: 11px;
    color: var(--text-faint);
    line-height: 1.5;
  }
  .row {
    display: flex;
    gap: 8px;
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
  /* Run options on the ready screen: four rows (Pages / Model / Pages/req /
     Parallel) with a FIXED label column so every control starts at the same
     left edge — same mechanism as the PDF dialog's `.lbl` column. A shared
     grid can't do this (each row would size its own column), so the width
     is a constant sized to the longest label, "PAGES/REQ". */
  .opts {
    display: flex;
    flex-direction: column;
    gap: 10px;
    font-size: 12px;
    color: var(--text-dim);
  }
  .opt-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .opt-lbl {
    flex: 0 0 70px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-faint);
  }
  .opt-row select {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 6px;
    max-width: 300px;
  }
  /* The two numeric dropdowns (Pages/req, Parallel) get a shared fixed
     width so they match — their one- and two-digit options would otherwise
     size the selects differently. */
  .opt-row select.sm {
    width: 56px;
  }
  /* Page range — two number inputs with a right-aligned validation error
     (only rendered when the range is invalid; see the markup). */
  .range {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .num {
    width: 56px;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid var(--border-strong);
    background: var(--bg-inset);
    color: var(--text);
  }
  .num:focus {
    outline: none;
    border-color: var(--accent-dim);
  }
  .dash {
    color: var(--text-faint);
  }
  .range-hint {
    margin-left: auto;
    font-size: 11px;
    color: var(--danger);
    text-align: right;
  }
  .req-hint {
    font-variant-numeric: tabular-nums;
  }
  /* Pre-flight quota warning — shown before the check starts, unlike the
     post-failure quota banner in review phase. */
  .limit-warning {
    font-size: 12px;
    line-height: 1.5;
    color: var(--danger);
    background: var(--danger-soft);
    border: 1px solid var(--danger);
    border-radius: 8px;
    padding: 10px 12px;
  }
  /* Fix-mode segmented control (same visual language as Settings). The
     label above it shares typography with the option-row labels — the
     sizing lives with .opt-lbl so the two stay in sync. */
  .mode-lbl {
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
    align-self: stretch;
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
    white-space: nowrap;
  }
  .seg-btn:hover { color: var(--text-dim); }
  .seg-btn.active {
    background: var(--accent-dim);
    color: var(--bg);
  }
  /* Checking phase: spinner + batch progress bar + stop. */
  .checking {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 4px 0;
  }
  .check-progress {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: flex-start;
    font-size: 13px;
    color: var(--text-dim);
  }
  .check-progress p { margin: 0; }
  .bar {
    width: 100%;
    height: 6px;
    background: var(--bg-inset);
    border-radius: 4px;
    overflow: hidden;
  }
  .bar .fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.15s;
  }
  .spin {
    width: 14px;
    height: 14px;
    border: 2px solid var(--accent-soft);
    border-top-color: var(--accent);
    border-radius: 50%;
    display: inline-block;
    margin-top: 3px;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  /* Review phase. */
  .error {
    color: var(--danger);
    background: var(--danger-soft);
    border: 1px solid var(--danger);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 13px;
    margin-bottom: 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: flex-start;
  }
  .error p { margin: 0; font-size: 12px; }
  /* Quota errors aren't failures to fix — neutral accent tone, no Retry. */
  .error.quota {
    color: var(--accent);
    background: var(--accent-soft);
    border-color: var(--accent-dim);
  }
  .error.quota p { color: inherit; }
  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    position: sticky;
    top: -14px;
    /* cover scrolled content behind the sticky strip */
    margin: -14px -14px 12px;
    padding: 10px 14px;
    background: var(--bg-elev);
    border-bottom: 1px solid var(--border);
    z-index: 1;
  }
  .summary {
    font-size: 12px;
    color: var(--text-dim);
    margin-right: auto;
  }
  .kbd-hint {
    font-size: 10px;
    color: var(--text-faint);
    white-space: nowrap;
  }
  .pages {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  /* Bottom tripwire for chunked rendering — 1px tall, fully invisible. */
  .more-sentinel {
    height: 1px;
  }
  .page {
    border: 1px solid var(--border);
    border-radius: 9px;
    overflow: hidden;
    /* Offscreen sections skip layout/paint entirely — with a big check the
       list holds thousands of rows and only the visible window should cost
       anything. `contain-intrinsic-size` keeps the scrollbar steady by
       remembering each section's last real height once rendered (48px ≈ a
       bare header before that). No-op on engines without content-visibility. */
    content-visibility: auto;
    contain-intrinsic-size: auto 48px;
  }
  .page-head {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 12px;
    background: var(--surface);
    font-size: 12px;
    color: var(--text-dim);
  }
  .page-head.muted { cursor: default; }
  .page-head input { accent-color: var(--accent-dim); }
  .page-name {
    font-weight: 600;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-right: auto;
  }
  .page-count {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }
  .page-ok {
    font-size: 11px;
    color: var(--ok);
  }
  .ok-dot {
    color: var(--ok);
    font-size: 12px;
    width: 13px;
    text-align: center;
  }
  .fix-list {
    list-style: none;
    margin: 0;
    padding: 4px 0;
  }
  .fix {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 12px;
    font-size: 13px;
  }
  .fix input { accent-color: var(--accent-dim); }
  /* Line address of the flagged word — makes the scoping visible: the fix
     applies to this line only. */
  .line-chip {
    font-size: 10px;
    font-family: var(--mono);
    color: var(--text-faint);
    flex-shrink: 0;
  }
  .arrow { color: var(--text-faint); }
  /* Both words read as plain colored text; a dotted underline hints they're
     editable, and double-click opens the editor. */
  .wrong,
  .correct {
    font-family: var(--mono);
    word-break: break-word;
    flex: 1;
    min-width: 40px;
    border-bottom: 1px dotted var(--border-strong);
  }
  .wrong { color: var(--danger); }
  .correct { color: var(--ok); }
  /* Editor opened by double-click on either word. Field color echoes the
     word it replaces. Clearing the correction makes "apply" delete the wrong
     word (applyWordFixes allows an empty replacement). */
  .word-input {
    flex: 1;
    min-width: 40px;
    font-family: var(--mono);
    font-size: 13px;
    background: var(--bg-elev);
    border: 1px solid var(--accent-dim);
    border-radius: 4px;
    padding: 1px 4px;
    outline: none;
    color: var(--text);
  }
  .word-input.wrong-edit { border-color: var(--danger); }
  /* Row whose (edited) wrong word no longer occurs in the page — dimmed and
     excluded from Apply until the word matches the page text again. */
  .fix.invalid { opacity: 0.55; }
  .invalid-note {
    font-size: 10px;
    color: var(--danger);
    white-space: nowrap;
    flex-shrink: 0;
  }
  /* Rewrite-mode audit rows: word-level inline diff of the old line vs the
     corrected one (already applied). Removed words get the danger tint,
     corrected words the ok tint — backgrounds, not strike/underline, to keep
     complex-script glyphs readable. */
  .diffline { align-items: flex-start; cursor: default; }
  /* "Keep original" button at the start of each diff row. */
  .revert-btn {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    padding: 0;
    border-radius: 5px;
    font-size: 13px;
    line-height: 1;
    color: var(--text-faint);
    background: var(--surface);
    border: 1px solid var(--border);
  }
  .revert-btn:hover:not(:disabled) {
    color: var(--accent);
    border-color: var(--accent-dim);
    background: var(--accent-soft);
  }
  .revert-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .revert-btn.done { color: var(--ok); border-color: transparent; background: none; }
  /* Reverted row: correction rolled back — original shown plain, dimmed. */
  .fix.reverted { opacity: 0.55; }
  .d-kept {
    flex: 1;
    min-width: 0;
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-dim);
  }
  .diff-body {
    flex: 1;
    min-width: 0;
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text);
  }
  .d-del {
    background: var(--danger-soft);
    color: var(--danger);
    border-radius: 3px;
  }
  .d-add {
    background: var(--ok-soft);
    color: var(--ok);
    border-radius: 3px;
  }
  /* No hover highlight by design: the accent cursor (click or ↑/↓) is the
     only row highlight, so the pointer passing over rows doesn't flash. */
  .fix.cursor,
  .page-head.cursor {
    background: var(--accent-soft);
    outline: 1px solid var(--accent-dim);
    outline-offset: -1px;
  }
</style>
