<script lang="ts">
  import { onMount } from "svelte";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import {
    listLanguages,
    downloadableLanguages,
    downloadLanguage,
    installLocalLanguage,
    deleteLanguage,
    type LanguageInfo,
    type DownloadProgress,
  } from "./ocr";

  interface Props {
    onclose: () => void;
    onchanged: () => void;
  }
  let { onclose, onchanged }: Props = $props();

  let installed = $state<LanguageInfo[]>([]);
  let catalog = $state<LanguageInfo[]>([]);
  let search = $state("");
  let variant = $state("fast");

  // Per-code download state: "downloading" with pct, or an error string.
  let dlState = $state<Record<string, { pct: number } | { error: string }>>({});

  let busy = $state(false);
  let banner = $state<string | null>(null);

  let filtered = $derived(
    catalog.filter(
      (l) =>
        !search ||
        l.code.includes(search.toLowerCase()) ||
        l.name.toLowerCase().includes(search.toLowerCase())
    )
  );

  async function refresh() {
    installed = await listLanguages();
    catalog = await downloadableLanguages();
  }

  onMount(() => {
    refresh();
  });

  async function doDownload(code: string) {
    banner = null;
    let unlisten: UnlistenFn | null = null;
    try {
      unlisten = await listen<DownloadProgress>(
        `lang-download://${code}`,
        (e) => {
          const { downloaded, total } = e.payload;
          const pct = total > 0 ? Math.round((downloaded / total) * 100) : 0;
          dlState = { ...dlState, [code]: { pct } };
        }
      );
      dlState = { ...dlState, [code]: { pct: 0 } };
      await downloadLanguage(code, variant);
      banner = `Installed ${code}.`;
      await refresh();
      onchanged();
    } catch (e: any) {
      dlState = {
        ...dlState,
        [code]: { error: typeof e === "string" ? e : e?.message ?? String(e) },
      };
    } finally {
      unlisten?.();
      const { [code]: _, ...rest } = dlState;
      dlState = rest;
    }
  }

  async function doLocal() {
    banner = null;
    try {
      const info = await installLocalLanguage();
      if (info) {
        banner = `Installed ${info.name} (${info.code}).`;
        await refresh();
        onchanged();
      }
    } catch (e: any) {
      banner = typeof e === "string" ? e : e?.message ?? String(e);
    }
  }

  async function doDelete(code: string) {
    banner = null;
    try {
      await deleteLanguage(code);
      banner = `Removed ${code}.`;
      await refresh();
      onchanged();
    } catch (e: any) {
      banner = typeof e === "string" ? e : e?.message ?? String(e);
    }
  }

  function fmtSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="backdrop" onclick={onclose}>
  <!-- svelte-ignore a11y_no_static_element_interactions, a11y_interactive_supports_focus -->
  <div class="modal" onclick={(e) => e.stopPropagation()} role="dialog" aria-label="Manage languages">
    <div class="modal-head">
      <h2>Language models</h2>
      <button class="x" onclick={onclose} aria-label="Close">✕</button>
    </div>

    {#if banner}
      <div class="banner">{banner}</div>
    {/if}

    <section class="block">
      <h3>Installed</h3>
      <div class="lang-list">
        {#each installed as l}
          <div class="lang-row">
            <div class="lang-info">
              <span class="lang-code">{l.code}</span>
              <span class="lang-name">{l.name}</span>
            </div>
            <span class="tag {l.source}">{l.source}</span>
            {#if l.source === "installed"}
              <button
                class="del"
                onclick={() => doDelete(l.code)}
                title="Remove"
              >Remove</button>
            {/if}
          </div>
        {/each}
      </div>
    </section>

    <section class="block">
      <h3>Add from file</h3>
      <p class="hint">Choose a <code>.traineddata</code> file from your computer.</p>
      <button class="add-local" onclick={doLocal}>Browse…</button>
    </section>

    <section class="block grow">
      <div class="block-head">
        <h3>Download</h3>
        <div class="dl-controls">
          <div class="variant" role="radiogroup" aria-label="Model quality">
            {#each ["fast", "standard", "best"] as v}
              <button
                class:active={variant === v}
                onclick={() => (variant = v)}
                role="radio"
                aria-checked={variant === v}
              >{v}</button>
            {/each}
          </div>
          <input
            type="search"
            placeholder="Search languages…"
            bind:value={search}
          />
        </div>
      </div>

      <div class="catalog">
        {#if filtered.length === 0}
          <div class="catalog-empty">
            {#if catalog.length === 0}
              All available languages are already installed.
            {:else}
              No languages match “{search}”.
            {/if}
          </div>
        {/if}
        {#each filtered as l (l.code)}
          {@const st = dlState[l.code]}
          <div class="cat-row">
            <div class="lang-info">
              <span class="lang-code">{l.code}</span>
              <span class="lang-name">{l.name}</span>
            </div>
            {#if st && "pct" in st}
              <div class="dl-bar">
                <div class="dl-fill" style="width:{st.pct}%"></div>
                <span class="dl-pct">{st.pct}%</span>
              </div>
            {:else if st && "error" in st}
              <span class="dl-err" title={st.error}>failed</span>
              <button class="get" onclick={() => doDownload(l.code)}>Retry</button>
            {:else}
              <button class="get" onclick={() => doDownload(l.code)}>Get</button>
            {/if}
          </div>
        {/each}
      </div>
    </section>
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }
  .modal {
    width: min(680px, 100%);
    max-height: min(82vh, 760px);
    display: flex;
    flex-direction: column;
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.5);
  }
  .modal-head {
    display: flex;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
  }
  .modal-head h2 {
    margin: 0;
    font-size: 16px;
    margin-right: auto;
  }
  .x {
    background: none;
    border: none;
    color: var(--text-faint);
    font-size: 16px;
    padding: 4px 8px;
    border-radius: 6px;
  }
  .x:hover {
    color: var(--text);
    background: var(--surface);
  }
  .banner {
    padding: 9px 20px;
    font-size: 12px;
    color: var(--ok);
    background: rgba(139, 216, 139, 0.08);
    border-bottom: 1px solid rgba(139, 216, 139, 0.2);
  }
  .block {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
  }
  .block:last-child {
    border-bottom: none;
  }
  .block.grow {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  h3 {
    margin: 0 0 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-faint);
  }
  .hint {
    margin: 0 0 10px;
    font-size: 12px;
    color: var(--text-dim);
  }
  .hint code {
    font-family: var(--mono);
    font-size: 11px;
    background: var(--bg-inset);
    padding: 1px 4px;
    border-radius: 3px;
  }
  .lang-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .lang-row,
  .cat-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 7px;
    background: var(--bg-inset);
  }
  .lang-info {
    display: flex;
    align-items: baseline;
    gap: 8px;
    min-width: 0;
    margin-right: auto;
  }
  .lang-code {
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
  }
  .lang-name {
    font-size: 12px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tag {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: 4px;
    background: var(--surface);
    color: var(--text-faint);
  }
  .tag.embedded,
  .tag.bundled {
    color: var(--accent);
    background: var(--accent-soft);
  }
  .del,
  .get {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 5px;
    border: 1px solid var(--border-strong);
    background: var(--surface);
    color: var(--text-dim);
  }
  .del:hover {
    color: var(--danger);
    border-color: var(--danger);
  }
  .get {
    color: var(--accent);
    border-color: var(--accent-dim);
  }
  .get:hover {
    background: var(--accent-soft);
  }
  .add-local {
    font-size: 12px;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 7px;
    border: 1px solid var(--accent-dim);
    background: var(--accent-soft);
    color: var(--accent);
  }
  .add-local:hover {
    opacity: 0.85;
  }
  .block-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
  }
  .block-head h3 {
    margin: 0;
  }
  .dl-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-left: auto;
  }
  .variant {
    display: flex;
    background: var(--bg-inset);
    border-radius: 6px;
    padding: 2px;
    gap: 1px;
  }
  .variant button {
    background: none;
    border: none;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 5px;
    color: var(--text-faint);
    text-transform: capitalize;
  }
  .variant button.active {
    background: var(--accent-dim);
    color: var(--bg);
    font-weight: 600;
  }
  input[type="search"] {
    width: 170px;
    font-size: 12px;
    padding: 5px 9px;
  }
  .catalog {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding-right: 4px;
  }
  .catalog-empty {
    padding: 24px;
    text-align: center;
    color: var(--text-faint);
    font-size: 13px;
  }
  .dl-bar {
    position: relative;
    width: 140px;
    height: 22px;
    background: var(--bg-inset);
    border-radius: 5px;
    overflow: hidden;
    border: 1px solid var(--border-strong);
  }
  .dl-fill {
    height: 100%;
    background: var(--accent-dim);
    transition: width 0.15s;
  }
  .dl-pct {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text);
  }
  .dl-err {
    font-size: 11px;
    color: var(--danger);
  }
</style>
