<script lang="ts">
  import type { OcrOpts } from "./ocr";
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
  }
  let { opts, theme, onchangetheme, onclose, updateAvailable }: Props = $props();

  // Current app version, fetched once on mount for the Updates section header.
  let appVersion = $state("");
  // Status of the MANUAL check button (separate from the silent startup
  // `updateAvailable` prop — opening Settings never re-fires the startup check).
  let status = $state<UpdateStatus>({ kind: "idle" });

  $effect(() => {
    getVersion().then((v) => (appVersion = v));
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
          <button
            class="seg-btn"
            class:active={theme === "system"}
            onclick={() => onchangetheme("system")}
            role="radio"
            aria-checked={theme === "system"}
          >System</button>
        </div>
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
            <div class="bar"><div class="fill" style="width:{status.percent}%"></div></div>
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
