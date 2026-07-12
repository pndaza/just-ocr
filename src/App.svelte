<script lang="ts">
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import Toolbar from "./lib/Toolbar.svelte";
  import Thumbnail from "./lib/Thumbnail.svelte";
  import Preview from "./lib/Preview.svelte";
  import Output from "./lib/Output.svelte";
  import LanguageManager from "./lib/LanguageManager.svelte";
  import {
    availableLanguages,
    choosePdfMode,
    isPdf,
    makeJob,
    makeJobsFromReadFiles,
    ocrFromBytes,
    readFiles,
    renderPdf,
    exportResults,
    type OcrOpts,
    type Job,
  } from "./lib/ocr";
  import { currentTheme, toggleTheme, type Theme } from "./theme";

  let languages = $state<string[]>(["eng"]);
  let theme = $state<Theme>(currentTheme());

  function cycleTheme() {
    theme = toggleTheme();
  }
  let opts = $state<OcrOpts>({
    language: "eng",
    psm: 3,
    whitelist: null,
    outputMode: "text",
  });

  let jobs = $state<Job[]>([]);
  let selectedId = $state<number | null>(null);
  let running = $state(false);
  let dropping = $state(false);
  let showLangManager = $state(false);

  // ── Resizable panels ──────────────────────────────────────────────────────
  // Widths are pixel values; the middle/right columns split the remaining space
  // proportionally. Two drag handles sit between the three panels.
  let leftW = $state(200);
  let rightW = $state(460); // middle is flexible (flex: 1)

  let draggingHandle = $state<null | "left" | "right">(null);

  function startDrag(which: "left" | "right") {
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
    } else {
      // Right panel width = distance from cursor to main's right edge.
      rightW = Math.max(200, Math.min(rect.width - leftW - 200, rect.right - e.clientX));
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
          // Ask the user how to process each PDF. Cancel → skip the file.
          const mode = await choosePdfMode(file.name);
          if (!mode) continue;
          const buf = new Uint8Array(await file.arrayBuffer());
          const pages = await renderPdf(file.name, buf, mode);
          if (!pages.length) console.warn(`"${file.name}" has no pages`);
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
          const mode = await choosePdfMode(f.name);
          if (!mode) continue;
          const pages = await renderPdf(f.name, new Uint8Array(f.bytes), mode);
          if (!pages.length) console.warn(`"${f.name}" has no pages`);
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
    URL.revokeObjectURL(jobs[idx].url);
    jobs = jobs.filter((j) => j.id !== id);
    if (selectedId === id) {
      const next = jobs[idx] ?? jobs[idx - 1] ?? null;
      selectedId = next ? next.id : null;
    }
  }

  function clearAll() {
    for (const j of jobs) URL.revokeObjectURL(j.url);
    jobs = [];
    selectedId = null;
  }

  async function processJob(job: Job) {
    job.status = "running";
    // Record the output mode used so display/export follow the job, not the
    // live setting (which the user may change between runs).
    job.outputMode = opts.outputMode;
    try {
      const res = await ocrFromBytes(job.bytes, opts);
      job.text = res.text;
      job.confidence = res.confidence;
      job.elapsedMs = res.elapsed_ms;
      job.status = "done";
    } catch (e: any) {
      job.error = typeof e === "string" ? e : e?.message ?? String(e);
      job.status = "error";
    }
  }

  async function runCurrent() {
    if (running || !selected) return;
    running = true;
    await processJob(selected);
    running = false;
  }

  async function runAll() {
    if (running) return;
    running = true;
    for (const job of jobs) {
      if (job.status === "running") continue;
      await processJob(job);
    }
    running = false;
  }

  async function exportAll() {
    await exportResults(jobs);
  }

  loadLanguages();
</script>

<div class="app" class:dropping>
  <Toolbar
    {opts}
    {languages}
    {running}
    {pending}
    {doneCount}
    canRunCurrent={canRunCurrent}
    hasSelection={!!selected}
    onruncurrent={runCurrent}
    onrunall={runAll}
    onexport={exportAll}
    onmanagelanguages={openLangManager}
    {theme}
    ontoggletheme={cycleTheme}
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
      <Output job={selected} />
    </section>
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
