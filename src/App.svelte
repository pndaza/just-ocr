<script lang="ts">
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import Toolbar from "./lib/Toolbar.svelte";
  import Thumbnail from "./lib/Thumbnail.svelte";
  import Preview from "./lib/Preview.svelte";
  import Output from "./lib/Output.svelte";
  import LanguageManager from "./lib/LanguageManager.svelte";
  import Settings from "./lib/Settings.svelte";
  import AiCheck from "./lib/AiCheck.svelte";
  import {
    availableLanguages,
    isPdf,
    makeJob,
    makeJobsFromReadFiles,
    ocrFromBytes,
    fixBurmeseSpelling,
    readFiles,
    renderPdf,
    exportResults,
    readJobBytes,
    disposeJobFile,
    lastLanguage,
    saveLanguage,
    lastEngine,
    saveEngine,
    lastSegmenter,
    saveSegmenter,
    lastDetVariant,
    saveDetVariant,
    lastMergeParagraphs,
    saveMergeParagraphs,
    lastFixBurmeseSpelling,
    saveFixBurmeseSpelling,
    lastExportIncludePageName,
    saveExportIncludePageName,
    lastLlmApiKey,
    saveLlmApiKey,
    lastLlmModel,
    saveLlmModel,
    lastLlmBatchSize,
    saveLlmBatchSize,
    lastAiCheckMode,
    saveAiCheckMode,
    lastShowFixSpelling,
    saveShowFixSpelling,
    type AiCheckMode,
    type LlmBatchSize,
    type LlmModel,
    type OcrOpts,
    type Job,
    type PdfMode,
    type ImageMode,
    type ReadFile,
  } from "./lib/ocr";
  import PdfModeDialog from "./lib/PdfModeDialog.svelte";
  import { currentTheme, setTheme, resolveTheme, type Theme } from "./theme";
  import { checkForUpdateSilent } from "./lib/updater";

  let languages = $state<string[]>(["mya"]);
  let theme = $state<Theme>(currentTheme());
  let showSettings = $state(false);

  // Set by the silent startup update check. Null = no update / not yet checked.
  // Non-null surfaces the gear badge (Toolbar) + pre-populates the Updates section.
  let updateAvailable = $state<string | null>(null);

  // Theme is now changed from the Settings modal (not a toolbar toggle).
  function changeTheme(t: Theme) {
    theme = setTheme(t);
  }

  // While the preference is "system", track OS theme changes live so flipping
  // the OS theme at runtime re-resolves the app without a reload. Registered
  // only in system mode; switching to explicit light/dark (or unmounting)
  // tears the listener down via the returned cleanup. The `theme === "system"`
  // read at the top is what makes this re-run when the preference changes.
  $effect(() => {
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const apply = () => {
      document.documentElement.dataset.theme = resolveTheme("system");
    };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  });

  let opts = $state<OcrOpts>({
    engine: lastEngine(),
    language: lastLanguage() ?? "mya",
    psm: 3,
    segmenter: lastSegmenter(),
    detVariant: lastDetVariant(),
    fixBurmeseSpelling: lastFixBurmeseSpelling(),
  });

  // Merge-paragraphs is a display-only preference (it does not change what the
  // OCR engine returns, only how the recognized lines are projected for the
  // text panel + export). Lives in the toolbar so it can be toggled before a
  // run; persisted globally like segmenter.
  let mergeParagraphs = $state(lastMergeParagraphs());

  // Whether exported files include the per-page `=== filename (conf, ms) ===`
  // header. Lives in Settings (not the toolbar) since it's an export-format
  // choice, not a per-run toggle. Default false = body-only export.
  let exportIncludePageName = $state(lastExportIncludePageName());

  // Google AI Studio (Gemini) credentials for the AI Check tool. The app's
  // only online feature — kept opt-in and configured in Settings; the key is
  // stored locally and passed to the backend per request.
  let llmApiKey = $state(lastLlmApiKey());
  let llmModel = $state<LlmModel>(lastLlmModel());
  // Pages per Gemini request — chosen in the AI Check panel (10-50) rather
  // than Settings, since it tunes the run the user is about to start.
  let llmBatchSize = $state<LlmBatchSize>(lastLlmBatchSize());
  // AI Check fix mode: "review" (word pairs) or "rewrite" (corrected page
  // text, diffed per line). Chosen in the panel, sticky across launches.
  let aiCheckMode = $state<AiCheckMode>(lastAiCheckMode());
  // The AI spell-check panel is a 4th main column (image · text · fixes side
  // by side), not a modal. Session-only: closed on each launch.
  let aiPanelOpen = $state(false);
  // Wrong words flagged by the AI check, per jobId — published by the panel
  // so the Text panel can highlight them while reviewing.
  let aiWrongWords = $state<Record<number, string[]>>({});
  // Whether the toolbar shows the "Fix spelling" checkbox (Myanmar only).
  // The toggle's own state is independent and stays sticky when hidden.
  let showFixSpelling = $state(lastShowFixSpelling());

  // Remember the chosen engine + language so they are pre-selected on the next
  // launch. loadLanguages() validates the language against available models,
  // so a value removed in the meantime is corrected automatically.
  $effect(() => {
    saveEngine(opts.engine);
  });
  $effect(() => {
    saveLanguage(opts.language);
  });
  // Segmenter (Myanmar line-box detector) is persisted so the chosen Seg
  // dropdown value is sticky across launches.
  $effect(() => {
    saveSegmenter(opts.segmenter);
  });
  // PP-OCR detector variant (small/tiny) is persisted so the chosen Det
  // dropdown value is sticky across launches. Only affects the PP-OCR
  // segmenters; ignored when segmenter is Kraken.
  $effect(() => {
    saveDetVariant(opts.detVariant);
  });
  $effect(() => {
    saveMergeParagraphs(mergeParagraphs);
  });
  $effect(() => {
    saveExportIncludePageName(exportIncludePageName);
  });
  $effect(() => {
    saveLlmApiKey(llmApiKey);
  });
  $effect(() => {
    saveLlmModel(llmModel);
  });
  $effect(() => {
    saveLlmBatchSize(llmBatchSize);
  });
  $effect(() => {
    saveAiCheckMode(aiCheckMode);
  });
  $effect(() => {
    saveShowFixSpelling(showFixSpelling);
  });
  // Spelling fix is sticky across launches, like the other OcrOpts fields.
  $effect(() => {
    saveFixBurmeseSpelling(opts.fixBurmeseSpelling);
  });
  // When the spell-fix toggle flips ON, lazily compute the projection for
  // every done Myanmar job (cache hit skips already-computed ones). Flipping
  // OFF needs no work: Output reads raw text again when job.spellFix isn't
  // surfaced (the toggle gates display in Output). The cache stays, so a
  // later ON-flip is instant. Reading opts.fixBurmeseSpelling + jobs here is
  // what makes this re-run on toggle / job-list changes.
  $effect(() => {
    if (!opts.fixBurmeseSpelling) return;
    for (const job of jobs) {
      if (opts.language === "mya" && job.status === "done") {
        void applySpellFixToJob(job);
      }
    }
  });

  let jobs = $state<Job[]>([]);
  let selectedId = $state<number | null>(null);
  let running = $state(false);
  let dropping = $state(false);
  let showLangManager = $state(false);

  // True while a batch ("Run All") is in flight. Single "Run Current" runs and
  // PDF dialogs don't set this, so the Stop button only appears for batches.
  let batchRun = $state(false);
  // Set when the user clicks Stop; checked between jobs so the queue halts
  // after the currently-running OCR finishes.
  let cancelRequested = $state(false);
  // Current/total counter shown beside "Processing" during a batch. Set only
  // while `batchRun` is true; null otherwise (single-run shows no counter).
  let batchProgress = $state<{ current: number; total: number } | null>(null);

  // ── PDF processing (in-app dialog with progress) ───────────────────────────
  // promptPdf() parks a promise that resolves to the per-page images (or null
  // if the user cancels). After a mode is chosen, runPdfRendering() drives the
  // backend and streams page progress into `pdfDialog` so the dialog can show
  // a progress bar.
  let pdfDialog = $state<{
    name: string;
    bytes: Uint8Array;
    status: "choosing" | "working";
    mode: PdfMode | null;
    done: number;
    total: number;
    resolve: (pages: ReadFile[] | null) => void;
  } | null>(null);

  function promptPdf(name: string, bytes: Uint8Array): Promise<ReadFile[] | null> {
    return new Promise((resolve) => {
      pdfDialog = { name, bytes, status: "choosing", mode: null, done: 0, total: 0, resolve };
    });
  }

  function onPdfModeChosen(
    mode: PdfMode,
    imageMode: ImageMode,
    maxHeight: number | null,
  ) {
    if (!pdfDialog) return;
    pdfDialog = { ...pdfDialog, status: "working", mode };
    runPdfRendering(mode, imageMode, maxHeight);
  }

  async function runPdfRendering(
    mode: PdfMode,
    imageMode: ImageMode,
    maxHeight: number | null,
  ) {
    const dlg = pdfDialog;
    if (!dlg) return;
    try {
      const pages = await renderPdf(
        dlg.name,
        dlg.bytes,
        mode,
        (done, total) => {
          if (pdfDialog) pdfDialog = { ...pdfDialog, done, total };
        },
        imageMode,
        maxHeight ?? undefined,
      );
      dlg.resolve(pages.length ? pages : null);
    } catch (e) {
      console.warn(`Could not process "${dlg.name}":`, e);
      dlg.resolve(null);
    }
    pdfDialog = null;
  }

  function cancelPdf() {
    pdfDialog?.resolve(null);
    pdfDialog = null;
  }

  // ── Resizable panels ──────────────────────────────────────────────────────
  // Widths are pixel values; the middle column splits the remaining space
  // proportionally. Drag handles sit between the panels. The AI Check panel
  // is an optional 4th column; when open, the Output|AI handle joins the
  // other two and the preview keeps its minimum width.
  let leftW = $state(200);
  let rightW = $state(460); // middle is flexible (flex: 1)
  let aiW = $state(380);

  let draggingHandle = $state<null | "left" | "right" | "ai">(null);

  function startDrag(which: "left" | "right" | "ai") {
    draggingHandle = which;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }

  function onDrag(e: MouseEvent) {
    if (!draggingHandle) return;
    const main = document.getElementById("main-area");
    if (!main) return;
    const rect = main.getBoundingClientRect();
    if (draggingHandle === "left") {
      // Left panel width = cursor x relative to main's left edge.
      leftW = Math.max(150, Math.min(250, e.clientX - rect.left));
    } else if (draggingHandle === "right") {
      // Cursor sits at the preview|output boundary. With the AI panel open,
      // the space right of the cursor covers BOTH right columns — subtract
      // the AI width so dragging only resizes Output.
      const aiSpace = aiPanelOpen ? aiW + 5 : 0;
      const avail = rect.right - e.clientX - aiSpace;
      rightW = Math.max(200, Math.min(rect.width - leftW - 200 - aiSpace, avail));
    } else {
      // AI panel width = distance from cursor to main's right edge, keeping
      // the preview at its 200px minimum.
      aiW = Math.max(240, Math.min(rect.right - e.clientX, rect.width - leftW - rightW - 205));
    }
  }

  function stopDrag() {
    draggingHandle = null;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }

  $effect(() => {
    if (draggingHandle === null) return;
    window.addEventListener("mousemove", onDrag);
    window.addEventListener("mouseup", stopDrag);
    return () => {
      window.removeEventListener("mousemove", onDrag);
      window.removeEventListener("mouseup", stopDrag);
    };
  });

  // ── Keyboard navigation of the thumbnail list ──────────────────────────────
  // Arrow keys move the selection (Up/Left = previous, Down/Right = next),
  // wrapping at the ends. Ignored while typing in a field (selects) or while
  // a modal dialog is open, so it never steals keystrokes there.
  // Reads happen inside the handler, so this effect registers exactly once.
  $effect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (pdfDialog || showLangManager || showSettings || aiPanelOpen) return;
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.tagName === "SELECT" ||
          t.isContentEditable)
      ) {
        return;
      }
      const up = e.key === "ArrowUp" || e.key === "ArrowLeft";
      const down = e.key === "ArrowDown" || e.key === "ArrowRight";
      if (!up && !down) return;
      if (!jobs.length) return;
      e.preventDefault();

      const ids = jobs.map((j) => j.id);
      const cur = selectedId === null ? -1 : ids.indexOf(selectedId);
      let next: number;
      if (cur === -1) {
        // No selection yet: start from the relevant end.
        next = up ? ids.length - 1 : 0;
      } else {
        // Clamp at the ends — don't wrap around to the opposite side.
        next = Math.min(ids.length - 1, Math.max(0, cur + (up ? -1 : 1)));
      }
      select(ids[next]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function openLangManager() {
    showLangManager = true;
  }

  async function onLanguagesChanged() {
    // Re-read the available language codes so the dropdown stays in sync.
    languages = await availableLanguages();
    if (languages.length && !languages.includes(opts.language)) {
      opts.language = languages[0];
    }
  }

  let pending = $derived(jobs.filter((j) => j.status === "queued").length);
  let doneCount = $derived(jobs.filter((j) => j.status === "done").length);
  let selected = $derived(
    selectedId !== null ? jobs.find((j) => j.id === selectedId) ?? null : null
  );
  let canRunCurrent = $derived(!!selected && selected.status !== "running");

  async function loadLanguages() {
    try {
      languages = await availableLanguages();
      if (languages.length && !languages.includes(opts.language)) {
        opts.language = languages[0];
      }
    } catch (e) {
      console.warn("available_languages failed", e);
    }
  }

  async function addFiles(files: FileList) {
    const added: Job[] = [];
    for (const file of Array.from(files)) {
      try {
        if (isPdf(file.name)) {
          // Ask how to process the PDF, then extract/render it (progress is
          // shown in the dialog). Cancel → skip the file.
          const buf = new Uint8Array(await file.arrayBuffer());
          const pages = await promptPdf(file.name, buf);
          if (!pages) continue;
          added.push(...makeJobsFromReadFiles(pages));
        } else {
          added.push(await makeJob(file));
        }
      } catch (e) {
        console.warn(`Could not add "${file.name}":`, e);
      }
    }
    jobs = [...jobs, ...added];
    if (selectedId === null && added.length) selectedId = added[0].id;
  }

  /** Ingest files dropped via the native drag-drop event (paths → bytes). */
  async function addPaths(paths: string[]) {
    if (!paths.length) return;
    const read = await readFiles(paths);
    if (!read.length) return;
    const added: Job[] = [];
    for (const f of read) {
      try {
        if (isPdf(f.name)) {
          const pages = await promptPdf(f.name, new Uint8Array(f.bytes));
          if (!pages) continue;
          added.push(...makeJobsFromReadFiles(pages));
        } else {
          added.push(...makeJobsFromReadFiles([f]));
        }
      } catch (e) {
        console.warn(`Could not add "${f.name}":`, e);
      }
    }
    jobs = [...jobs, ...added];
    if (selectedId === null && added.length) selectedId = added[0].id;
  }

  // Native drag-drop: Tauri emits file paths, not browser File objects, so the
  // HTML5 drop event never fires. Register once on mount.
  let unlisten: (() => void) | null = null;
  $effect(() => {
    let cancelled = false;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        if (event.payload.type === "enter" || event.payload.type === "over") {
          dropping = true;
        } else if (event.payload.type === "leave") {
          dropping = false;
        } else if (event.payload.type === "drop") {
          dropping = false;
          addPaths(event.payload.paths);
        }
      })
      .then((fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  });

  function select(id: number) {
    selectedId = id;
  }

  function remove(id: number) {
    const idx = jobs.findIndex((j) => j.id === id);
    if (idx === -1) return;
    const job = jobs[idx];
    URL.revokeObjectURL(job.url);
    disposeJobFile(job); // best-effort cleanup of the temp PNG
    jobs = jobs.filter((j) => j.id !== id);
    if (selectedId === id) {
      const next = jobs[idx] ?? jobs[idx - 1] ?? null;
      selectedId = next ? next.id : null;
    }
  }

  function clearAll() {
    for (const j of jobs) {
      URL.revokeObjectURL(j.url);
      disposeJobFile(j); // best-effort cleanup of temp PNGs
    }
    jobs = [];
    selectedId = null;
  }

  async function processJob(job: Job) {
    job.status = "running";
    try {
      // Path-based (PDF page) jobs read their pixels from the temp file; others
      // use the in-memory bytes.
      const bytes = await readJobBytes(job);
      const res = await ocrFromBytes(bytes, opts);
      job.result = res;
      job.confidence = res.confidence;
      job.elapsedMs = res.elapsedMs;
      job.spellFix = null; // reset any stale cache from a prior run
      job.llmFix = null; // AI fixes applied to the old text no longer apply
      job.manualText = null; // manual edits referred to the old text too
      job.status = "done";
      // If the spell-fix toggle is on for a Myanmar result, pre-compute the
      // projection now so the count + fixed text show immediately in Output,
      // without waiting for the toggle-watching effect below to pick it up.
      if (opts.fixBurmeseSpelling && opts.language === "mya") {
        void applySpellFixToJob(job);
      }
    } catch (e: any) {
      job.error = typeof e === "string" ? e : e?.message ?? String(e);
      job.status = "error";
    }
  }

  /**
   * Compute (or return cached) spell-fix projection for a job. The result is
   * stored on `job.spellFix` so the Output panel can read fixed text + count
   * reactively, and so toggling the switch off-then-on is instant after the
   * first compute. No-op for jobs that aren't done, aren't Myanmar, or have
   * no recognized lines. Safe to call concurrently — duplicate calls just
   * overwrite the cache with the same value.
   */
  async function applySpellFixToJob(job: Job): Promise<void> {
    if (job.spellFix) return; // cache hit
    if (job.status !== "done" || !job.result || job.result.lines.length === 0) return;
    try {
      const fixed = await fixBurmeseSpelling(job.result.lines.map((l) => l.text));
      job.spellFix = {
        fixedLines: fixed.map((r) => r.text),
        fixes: fixed.reduce((sum, r) => sum + r.fixes, 0),
      };
    } catch (e) {
      // Backend call failed (older build, model error) — leave spellFix null
      // so the Output panel falls back to raw text. Non-fatal: the user still
      // sees recognized text, just uncorrected.
      console.warn(`Spell-fix failed for "${job.name}":`, e);
    }
  }

  async function runCurrent() {
    if (running || !selected) return;
    running = true;
    batchRun = false;
    cancelRequested = false;
    await processJob(selected);
    running = false;
    cancelRequested = false;
  }

  async function runAll() {
    if (running) return;
    running = true;
    batchRun = true;
    cancelRequested = false;
    const total = jobs.length;
    let current = 0;
    for (const job of jobs) {
      if (cancelRequested) break;
      if (job.status === "running") continue;
      // Tick before processJob so the user sees 1/12 the moment the first page
      // starts — not 0/12 while work is already happening.
      current += 1;
      batchProgress = { current, total };
      await processJob(job);
    }
    batchRun = false;
    running = false;
    cancelRequested = false;
    batchProgress = null;
  }

  /** Request the batch stop. The in-flight OCR finishes; queued jobs are skipped. */
  function stopAll() {
    cancelRequested = true;
  }

  async function exportAll() {
    await exportResults(jobs, {
      mergeParagraphs,
      fixSpelling: opts.fixBurmeseSpelling,
      includePageName: exportIncludePageName,
    });
  }

  // Silent startup update check. Fire-and-forget, never blocks startup.
  // Errors are swallowed inside checkForUpdateSilent — an offline launch sees
  // nothing. Only a successful "update available" sets updateAvailable.
  // No reactive reads by design → the effect runs exactly once after mount.
  $effect(() => {
    checkForUpdateSilent((v) => (updateAvailable = v));
  });

  loadLanguages();
</script>

<div class="app" class:dropping>
  <Toolbar
    {opts}
    {languages}
    {running}
    {pending}
    {doneCount}
    {mergeParagraphs}
    batchProgress={batchProgress}
    canRunCurrent={canRunCurrent}
    hasSelection={!!selected}
    showStop={running && batchRun}
    stopping={cancelRequested}
    onstop={stopAll}
    onruncurrent={runCurrent}
    onrunall={runAll}
    onexport={exportAll}
    onaicheck={() => (aiPanelOpen = !aiPanelOpen)}
    aiOpen={aiPanelOpen}
    onmanagelanguages={openLangManager}
    onsettings={() => (showSettings = true)}
    updateAvailable={updateAvailable}
    onchangemerge={(v: boolean) => (mergeParagraphs = v)}
    fixBurmeseSpelling={opts.fixBurmeseSpelling}
    onchangefix={(v: boolean) => (opts.fixBurmeseSpelling = v)}
    {showFixSpelling}
  />
  <main id="main-area">
    <section class="col left" style="width:{leftW}px">
      <Thumbnail
        {jobs}
        {selectedId}
        onselect={select}
        onfiles={addFiles}
        onremove={remove}
        onclear={clearAll}
      />
    </section>
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="divider"
      class:active={draggingHandle === "left"}
      onmousedown={() => startDrag("left")}
      role="separator"
      aria-orientation="vertical"
    ></div>
    <section class="col mid">
      <Preview job={selected} />
    </section>
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="divider"
      class:active={draggingHandle === "right"}
      onmousedown={() => startDrag("right")}
      role="separator"
      aria-orientation="vertical"
    ></div>
    <section class="col right" style="width:{rightW}px">
      <Output
        job={selected}
        {jobs}
        {mergeParagraphs}
        fixSpelling={opts.fixBurmeseSpelling}
        highlights={selected ? aiWrongWords[selected.id] ?? [] : []}
      />
    </section>
    {#if aiPanelOpen}
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <div
        class="divider"
        class:active={draggingHandle === "ai"}
        onmousedown={() => startDrag("ai")}
        role="separator"
        aria-orientation="vertical"
      ></div>
      <section class="col ai" style="width:{aiW}px">
        <AiCheck
          {jobs}
          apiKey={llmApiKey}
          bind:model={llmModel}
          bind:batchSize={llmBatchSize}
          bind:mode={aiCheckMode}
          onclose={() => {
            aiPanelOpen = false;
            // Drop the highlights with the panel — they're meaningless
            // without the review list they belong to.
            aiWrongWords = {};
          }}
          onopensettings={() => {
            aiPanelOpen = false;
            showSettings = true;
          }}
          onselectpage={select}
          onsuggestions={(m) => (aiWrongWords = m)}
        />
      </section>
    {/if}
  </main>
  {#if dropping}
    <div class="drop-overlay" aria-hidden="true">
      <div class="drop-card">
        <span class="drop-icon">⬆</span>
        <span>Drop to add images</span>
      </div>
    </div>
  {/if}
</div>

{#if showLangManager}
  <LanguageManager
    onclose={() => (showLangManager = false)}
    onchanged={onLanguagesChanged}
  />
{/if}

{#if showSettings}
  <Settings
    {opts}
    {theme}
    onchangetheme={changeTheme}
    onclose={() => (showSettings = false)}
    {updateAvailable}
    {exportIncludePageName}
    onchangeexportpage={(v: boolean) => (exportIncludePageName = v)}
    {llmApiKey}
    onchangeapikey={(v: string) => (llmApiKey = v)}
    {showFixSpelling}
    onchangeShowfixspelling={(v: boolean) => (showFixSpelling = v)}
  />
{/if}

{#if pdfDialog}
  <PdfModeDialog
    name={pdfDialog.name}
    status={pdfDialog.status}
    mode={pdfDialog.mode}
    done={pdfDialog.done}
    total={pdfDialog.total}
    onprocess={onPdfModeChosen}
    oncancel={cancelPdf}
  />
{/if}

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: radial-gradient(
        1200px 600px at 100% -10%,
        var(--accent-soft),
        transparent 60%
      ),
      var(--bg);
  }
  main {
    flex: 1;
    display: flex;
    min-height: 0;
  }
  .col {
    min-height: 0;
    min-width: 0;
    overflow: hidden;
  }
  .col.left {
    flex-shrink: 0;
  }
  .col.mid {
    flex: 1;
    min-width: 200px;
  }
  .col.right {
    flex-shrink: 0;
  }
  .col.ai {
    flex-shrink: 0;
  }
  .divider {
    width: 5px;
    flex-shrink: 0;
    cursor: col-resize;
    background: var(--border);
    position: relative;
    transition: background 0.12s;
  }
  .divider::after {
    content: "";
    position: absolute;
    inset: 0 -3px;
  }
  .divider:hover,
  .divider.active {
    background: var(--accent-dim);
  }
  .drop-overlay {
    position: absolute;
    inset: 0;
    background: var(--overlay);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
    pointer-events: none;
  }
  .drop-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 32px 48px;
    border: 2px dashed var(--accent);
    border-radius: 14px;
    background: var(--bg-elev);
    color: var(--accent);
    font-weight: 600;
  }
  .drop-icon {
    font-size: 30px;
  }
</style>
