# In-App Auto-Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Tauri v2 Updater-based auto-update: a silent startup check that surfaces an accent badge on the Settings gear when a newer version exists, plus a Settings "Updates" section with manual re-check and two-step download-and-install.

**Architecture:** Frontend drives the `@tauri-apps/plugin-updater` JS API directly (no new Rust commands). The Rust side only registers `tauri-plugin-updater` + `tauri-plugin-process`. A new `src/lib/updater.ts` module owns all update Tauri calls and the `UpdateStatus` state machine. `App.svelte` fires a silent startup check; `Toolbar` shows a badge; `Settings.svelte` hosts the manual-check UI. Releases flip to auto-publish and add signing secrets so `tauri-action` emits `latest.json`.

**Tech Stack:** Tauri v2 (`tauri-plugin-updater` `2`, `tauri-plugin-process` `2`), `@tauri-apps/plugin-updater` + `@tauri-apps/plugin-process` JS packages, Svelte 5 runes, vitest.

**Spec:** `docs/superpowers/specs/2026-07-29-app-updater-design.md`

---

## File Structure

**Create:**
- `src/lib/updater.ts` — all updater-related Tauri calls + `UpdateStatus` type + pure state-mapping helpers. Single home, mirrors the `ocr.ts` convention.
- `src/lib/updater.test.ts` — unit tests for the pure state-mapping helpers (no network).

**Modify:**
- `package.json` — add `@tauri-apps/plugin-updater`, `@tauri-apps/plugin-process` deps.
- `src-tauri/Cargo.toml` — add `tauri-plugin-updater`, `tauri-plugin-process` deps.
- `src-tauri/src/lib.rs` — register both plugins (desktop-gated) in `run()`.
- `src-tauri/capabilities/default.json` — add `updater:default`, `process:default`, `core:app:version`.
- `src-tauri/tauri.conf.json` — add `bundle.createUpdaterArtifacts`, `plugins.updater` (pubkey + endpoint).
- `.github/workflows/release.yml` — add signing secrets env, flip `releaseDraft` → `false`.
- `src/App.svelte` — startup check state + wiring into Toolbar/Settings.
- `src/lib/Toolbar.svelte` — `updateAvailable` prop + gear badge.
- `src/lib/Settings.svelte` — `updateAvailable` prop + Updates section.
- `AGENTS.md` — document the updater pipeline in the Release section.

---

## Task 1: Add frontend dependencies

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Install the updater + process JS plugins**

Run:
```sh
npm install @tauri-apps/plugin-updater @tauri-apps/plugin-process
```

Expected: both packages added to `package.json` `dependencies`, `node_modules/@tauri-apps/plugin-updater` and `node_modules/@tauri-apps/plugin-process` exist.

- [ ] **Step 2: Verify the type definitions resolve**

Run:
```sh
node -e "require('@tauri-apps/plugin-updater/package.json').name; require('@tauri-apps/plugin-process/package.json').name"
```

Expected: prints `@tauri-apps/plugin-updater` then `@tauri-apps/plugin-process`, exit 0.

- [ ] **Step 3: Commit**

```sh
git add package.json package-lock.json
git commit -m "deps: add @tauri-apps/plugin-updater + plugin-process"
```

---

## Task 2: Add backend dependencies

**Files:**
- Modify: `src-tauri/Cargo.toml`

- [ ] **Step 1: Add the two Rust plugin crates**

Append to the `[dependencies]` block in `src-tauri/Cargo.toml` (place near the other `tauri-plugin-*` entries, after the `tauri-plugin-opener` line, around line 31):

```toml
# In-app updater: serves the v2 updater plugin so the frontend can check for
# and install newer builds published to GitHub Releases (see updater.ts).
tauri-plugin-updater = "2"
# relaunch() after an update installs — the updater triggers a process restart.
tauri-plugin-process = "2"
```

- [ ] **Step 2: Verify it resolves**

Run (inside `src-tauri/`):
```sh
cargo fetch
```

Expected: completes without error (no network if already cached, otherwise fetches the two crates).

- [ ] **Step 3: Commit**

```sh
git add src-tauri/Cargo.toml
git commit -m "deps(rust): add tauri-plugin-updater + tauri-plugin-process"
```

---

## Task 3: Register the plugins in Rust

**Files:**
- Modify: `src-tauri/src/lib.rs` (the `run()` function, lines ~274–318)

- [ ] **Step 1: Break the builder chain to allow a desktop-gated block**

Replace the start of `run()` (lines 274–279) — from `pub fn run() {` through the `.plugin(tauri_plugin_opener::init())` line — with:

```rust
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init());

    // Updater + process (relaunch after install) are desktop-only. This is a
    // desktop app, so the cfg gate is defensive/documentation rather than
    // strictly necessary — it matches the official Tauri pattern and keeps a
    // future mobile target clean. Registration is split out of the chain
    // because #[cfg(desktop)] cannot attach to a method call mid-chain.
    #[cfg(desktop)]
    {
        builder = builder
            .plugin(tauri_plugin_updater::Builder::new().build())
            .plugin(tauri_plugin_process::init());
    }

    builder
```

Then de-indent the rest of the chain (`.setup(...)`, `.invoke_handler(...)`, `.run(...)`, the trailing cleanup comment) so it chains off `builder`. The final `.expect("error while running just-ocr");` and the post-loop temp-dir cleanup remain unchanged. Keep the existing `setup` closure, `invoke_handler`, and all comments intact — only the builder-head changes.

The resulting tail should read:

```rust
        .setup(|app| {
            // ... (unchanged env_logger + tesseract version + sweep_stale_temp_dirs) ...
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            available_languages,
            read_files,
            default_save_dir,
            ocr_from_bytes,
            render_pdf,
            languages::list_languages,
            languages::downloadable_languages,
            languages::download_language,
            languages::install_local_language,
            languages::delete_language,
        ])
        .run(tauri::generate_context!())
        .expect("error while running just-ocr");
    // The event loop has ended; remove this session's temp PDF-page PNGs.
    // ... (unchanged) ...
}
```

- [ ] **Step 2: Verify it compiles**

Run (inside `src-tauri/`):
```sh
cargo check
```

Expected: compiles with no errors. (Warnings about unused imports are fine; fix only if they relate to this change.)

- [ ] **Step 3: Commit**

```sh
git add src-tauri/src/lib.rs
git commit -m "feat(rust): register updater + process plugins (desktop-gated)"
```

---

## Task 4: Add capabilities (permissions)

**Files:**
- Modify: `src-tauri/capabilities/default.json`

- [ ] **Step 1: Add the three permission identifiers**

In `src-tauri/capabilities/default.json`, append to the `permissions` array (after `"opener:allow-reveal-item-in-dir"`):

```json
    "updater:default",
    "process:default",
    "core:app:version"
```

The full file becomes:

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Default capabilities for just-ocr main window",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "dialog:allow-save",
    "fs:allow-write-file",
    "fs:allow-read-file",
    "fs:allow-remove",
    "opener:allow-reveal-item-in-dir",
    {
      "identifier": "fs:scope",
      "allow": [
        { "path": "**" }
      ]
    },
    "updater:default",
    "process:default",
    "core:app:version"
  ]
}
```

- [ ] **Step 2: Verify the schema accepts these identifiers**

Run (inside `src-tauri/`):
```sh
cargo check
```

Expected: compiles with no errors. If `core:app:version` is rejected by the generated schema, the build will error at `generate_context!` — in that case run `grep -ri "version" src-tauri/gen/schemas/desktop-schema.json | head` to find the exact identifier the installed Tauri version expects (e.g. `core:app:allow-get-version`), substitute it, and re-check.

- [ ] **Step 3: Commit**

```sh
git add src-tauri/capabilities/default.json
git commit -m "feat(capabilities): grant updater + process + app-version perms"
```

---

## Task 5: Create `updater.ts` with the `UpdateStatus` type and pure helpers

This is the TDD task. The pure helpers (status mapping, percent computation) are testable without a network. The network-bound wrappers are added in Task 6 after the pure logic is green.

**Files:**
- Create: `src/lib/updater.ts`
- Test: `src/lib/updater.test.ts`

- [ ] **Step 1: Write the failing tests for the pure helpers**

Create `src/lib/updater.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { mapCheckResult, downloadPercent, type UpdateStatus } from "./updater";

describe("mapCheckResult", () => {
  // A null/undefined Update (no newer version) → up-to-date.
  it("returns up-to-date when no update object is given", () => {
    expect(mapCheckResult(null)).toEqual<UpdateStatus>({ kind: "up-to-date" });
    expect(mapCheckResult(undefined)).toEqual<UpdateStatus>({ kind: "up-to-date" });
  });

  // A present Update → available, carrying its version string.
  it("returns available with the update's version when an update is given", () => {
    const fake = { version: "0.4.0" } as any;
    expect(mapCheckResult(fake)).toEqual<UpdateStatus>({
      kind: "available",
      version: "0.4.0",
    });
  });
});

describe("downloadPercent", () => {
  // Progress percent = (downloaded / contentLength) * 100, floored, clamped 0..100.
  it("computes percent from downloaded bytes and total content length", () => {
    expect(downloadPercent(0, 1000)).toBe(0);
    expect(downloadPercent(250, 1000)).toBe(25);
    expect(downloadPercent(1000, 1000)).toBe(100);
  });

  it("clamps to 100 if downloaded exceeds contentLength", () => {
    expect(downloadPercent(1200, 1000)).toBe(100);
  });

  it("returns 0 when contentLength is 0 (avoids divide-by-zero)", () => {
    expect(downloadPercent(500, 0)).toBe(0);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```sh
npm test -- updater.test
```

Expected: FAIL — `Cannot find module './updater'` (file does not exist yet).

- [ ] **Step 3: Create `updater.ts` with the types and pure helpers**

Create `src/lib/updater.ts`:

```ts
import { check, type Update } from "@tauri-apps/plugin-updater";

/**
 * All states the update UI can be in.
 *
 * - `idle`         — nothing happening, user has not checked
 * - `checking`     — a check is in flight
 * - `up-to-date`   — last check found no newer version
 * - `available`    — a newer version exists, not yet installing
 * - `downloading`  — the update bundle is downloading (percent 0..100)
 * - `installing`   — download finished, installer is applying
 * - `error`        — the last action failed (message is user-facing)
 */
export type UpdateStatus =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "up-to-date" }
  | { kind: "available"; version: string }
  | { kind: "downloading"; percent: number }
  | { kind: "installing" }
  | { kind: "error"; message: string };

/**
 * Map the result of `check()` to a UI status. Pure: no side effects, no
 * network — the caller hands us whatever `check()` resolved to.
 *
 * `null`/`undefined` (the plugin returns null when already on the latest) maps
 * to `up-to-date`; a present `Update` maps to `available`.
 */
export function mapCheckResult(update: Update | null | undefined): UpdateStatus {
  if (!update) return { kind: "up-to-date" };
  return { kind: "available", version: update.version };
}

/**
 * Compute an integer percent (0..100) for a download progress event. Pure.
 * Clamps to 100 and guards divide-by-zero (contentLength can be 0 if the
 * server omits Content-Length).
 */
export function downloadPercent(downloaded: number, contentLength: number): number {
  if (contentLength <= 0) return 0;
  return Math.min(100, Math.floor((downloaded / contentLength) * 100));
}

// The most-recently-found update, held in module state so the manual "Download
// & install" flow can act on it without re-checking. Set by checkForUpdate().
let pendingUpdate: Update | null = null;

/**
 * Silent startup check. Errors are swallowed — an offline machine must see no
 * error UI. Calls `onAvailable(version)` only when a newer version exists.
 * Never throws.
 */
export async function checkForUpdateSilent(
  onAvailable: (version: string) => void,
): Promise<void> {
  try {
    const update = await check();
    if (update) {
      pendingUpdate = update;
      onAvailable(update.version);
    }
  } catch (e) {
    // Silent by design: the user did not ask for this check.
    console.warn("startup update check failed:", e);
  }
}

/**
 * Manual check. Errors propagate as an UpdateStatus so the Settings UI can
 * show them inline (the user clicked, so they're watching).
 */
export async function checkForUpdate(): Promise<UpdateStatus> {
  try {
    const update = await check();
    pendingUpdate = update;
    return mapCheckResult(update);
  } catch (e: any) {
    const message = typeof e === "string" ? e : e?.message ?? String(e);
    return { kind: "error", message };
  }
}

/**
 * Download + install the most-recently-found update, then relaunch.
 * `onProgress(percent)` fires with integer 0..100 as the download proceeds.
 * Throws if no check has found an update yet, or if the download/install fails
 * (the caller wraps it into an UpdateStatus).
 */
export async function downloadAndInstall(
  onProgress: (percent: number) => void,
): Promise<void> {
  if (!pendingUpdate) {
    throw new Error("No update available — check first.");
  }
  const update = pendingUpdate;
  let contentLength = 0;
  let downloaded = 0;
  await update.downloadAndInstall((event) => {
    switch (event.event) {
      case "Started":
        contentLength = event.data.contentLength;
        break;
      case "Progress":
        downloaded += event.data.chunkLength;
        onProgress(downloadPercent(downloaded, contentLength));
        break;
      case "Finished":
        onProgress(100);
        break;
    }
  });
  // downloadAndInstall has applied the update; relaunch on macOS/Linux.
  // On Windows the installer already exited the app, so this is a no-op there.
  const { relaunch } = await import("@tauri-apps/plugin-process");
  await relaunch();
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```sh
npm test -- updater.test
```

Expected: PASS — all four `mapCheckResult` / `downloadPercent` cases green.

- [ ] **Step 5: Verify the type gate**

Run:
```sh
npm run build
```

Expected: vite build succeeds, no TS errors. (This also catches a wrong `Update` import / type mismatch from the plugin.)

- [ ] **Step 6: Commit**

```sh
git add src/lib/updater.ts src/lib/updater.test.ts
git commit -m "feat(updater): add UpdateStatus state machine + check/install wrappers"
```

---

## Task 6: Surface the update badge in the Toolbar

**Files:**
- Modify: `src/lib/Toolbar.svelte`

- [ ] **Step 1: Add the `updateAvailable` prop**

In `src/lib/Toolbar.svelte`, add to the `Props` interface (after `onsettings: () => void;`, around line 25):

```ts
  /** When non-null, a newer version exists — shows a badge on the gear. */
  updateAvailable: string | null;
```

Destructure it in the `let { ... }: Props = $props();` block (add `updateAvailable,` alongside the other props, e.g. after `onsettings,`).

- [ ] **Step 2: Add the badge class + style to the Settings gear**

Find the Settings gear button (the `.icon-btn` with `onclick={onsettings}`, around line 88). Add the conditional class and `position: relative`:

```svelte
  <button
    class="icon-btn"
    class:has-update={!!updateAvailable}
    onclick={onsettings}
    title={updateAvailable ? `Update available: v${updateAvailable}` : "Settings"}
    aria-label={updateAvailable ? `Settings — update available (v${updateAvailable})` : "Settings"}
  >⚙</button>
```

Add to the `<style>` block (after the `.icon-btn:hover { ... }` rule):

```css
  .icon-btn {
    position: relative;
  }
  .icon-btn.has-update::after {
    content: "";
    position: absolute;
    top: 1px;
    right: 1px;
    width: 7px;
    height: 7px;
    background: var(--accent);
    border-radius: 50%;
    border: 1.5px solid var(--bg-elev); /* punch-through so the dot reads on the gear */
  }
```

> Note: `position: relative` is added as a separate rule rather than merged into the existing `.icon-btn` block so the diff stays surgical — but if the existing `.icon-btn` rule already sets it, omit the extra rule. (It does not today; it relies on flex centering.)

- [ ] **Step 3: Note on the build (do NOT run the full build yet)**

The full `npm run build` will **fail** here because `Toolbar` now requires the `updateAvailable` prop, but `App.svelte` doesn't pass it until Task 8. This is expected mid-implementation. Defer the build check to Task 8 Step 4, where the wiring is complete. To sanity-check this task in isolation without the type gate, run only:

```sh
npx svelte-check --threshold error --workspace src/lib/Toolbar.svelte
```

(if `svelte-check` isn't installed, skip — the Task 8 build is the real gate.)

- [ ] **Step 4: Commit**

```sh
git add src/lib/Toolbar.svelte
git commit -m "feat(toolbar): badge the Settings gear when an update is available"
```

---

## Task 7: Add the Updates section to the Settings modal

**Files:**
- Modify: `src/lib/Settings.svelte`

- [ ] **Step 1: Add imports, props, and state**

At the top of `src/lib/Settings.svelte`, extend the script block. Add the updater import after the existing imports:

```ts
  import {
    checkForUpdate,
    downloadAndInstall,
    type UpdateStatus,
  } from "./updater";
  import { getVersion } from "@tauri-apps/api/app";
```

Add to the `Props` interface (after `onclose: () => void;`):

```ts
  /** Set by App's silent startup check. Pre-populates the Updates section so the
   *  user doesn't have to re-check after the badge drew them here. */
  updateAvailable: string | null;
```

Destructure `updateAvailable` in the `let { ... }: Props = $props();` line.

Add state below the existing `let { ... }`:

```ts
  // Current app version, fetched once on mount for the Updates section header.
  let appVersion = $state("");
  // Status of the MANUAL check button (separate from the silent startup
  // `updateAvailable` prop — opening Settings never re-fires the startup check).
  let status = $state<UpdateStatus>({ kind: "idle" });

  $effect(() => {
    getVersion().then((v) => (appVersion = v));
  });

  // If the startup check already found an update, pre-populate as "available".
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
```

- [ ] **Step 2: Add the Updates section markup**

Inside the `<div class="body">`, after the closing `</section>` of the Theme section (around line 68), add:

```svelte
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
```

- [ ] **Step 3: Add the Updates section styles**

Append to the `<style>` block:

```css
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
```

- [ ] **Step 4: Note on the build (do NOT run the full build yet)**

The full `npm run build` will **fail** here: both `Toolbar` (Task 6) and `Settings` now require `updateAvailable`, but `App.svelte` doesn't pass it until Task 8. Defer the build to Task 8 Step 4. This task is verified by the build at Task 8.

- [ ] **Step 5: Commit**

```sh
git add src/lib/Settings.svelte
git commit -m "feat(settings): add Updates section (manual check + 2-step install)"
```

---

## Task 8: Wire the silent startup check into App.svelte

**Files:**
- Modify: `src/App.svelte`

- [ ] **Step 1: Import the silent checker and add state**

In `src/App.svelte`, add to the imports near the top (after the `Settings` import, line 8):

```ts
  import { checkForUpdateSilent } from "./lib/updater";
```

Add state near the other `$state` declarations (after `let showSettings = $state(false);`, line 39):

```ts
  // Set by the silent startup update check. Null = no update / not yet checked.
  // Non-null surfaces the gear badge (Toolbar) + pre-populates the Updates section.
  let updateAvailable = $state<string | null>(null);
```

- [ ] **Step 2: Fire the silent check on mount**

Add an `$effect` that runs once on mount (place it after the existing `loadLanguages();` call at the bottom of the script block — but `$effect` must be at top-level of the script, not inside a function; place it just above the `loadLanguages();` line):

```ts
  // Silent startup update check. Fire-and-forget, never blocks startup.
  // Errors are swallowed inside checkForUpdateSilent — an offline launch sees
  // nothing. Only a successful "update available" sets updateAvailable.
  $effect(() => {
    checkForUpdateSilent((v) => (updateAvailable = v));
  });
```

> Note: `$effect` runs after mount and re-runs on dependency changes. This one reads no reactive state, so it runs exactly once. If Svelte warns about missing dependencies, that's expected and harmless here.

- [ ] **Step 3: Pass `updateAvailable` to Toolbar and Settings**

In the `<Toolbar ... />` invocation (around line 435), add the prop:

```svelte
    updateAvailable={updateAvailable}
```

In the `<Settings ... />` invocation (around line 507), add the prop:

```svelte
    {updateAvailable}
```

- [ ] **Step 4: Verify the build + type gate**

Run:
```sh
npm run build
```

Expected: succeeds, no TS errors. The `updateAvailable` prop now flows from App → Toolbar (badge) and App → Settings (Updates section).

- [ ] **Step 5: Run the full frontend test suite**

Run:
```sh
npm test
```

Expected: all tests pass (existing tests unaffected; new updater tests from Task 5 green).

- [ ] **Step 6: Commit**

```sh
git add src/App.svelte
git commit -m "feat(app): silent startup update check + wire updateAvailable to UI"
```

---

## Task 9: Configure the updater in `tauri.conf.json`

This task requires the signing public key. **Generate it now if not already done** (Step 1), then paste it in (Step 2). The private key goes to CI in Task 10.

**Files:**
- Modify: `src-tauri/tauri.conf.json`

- [ ] **Step 1: Generate the signing keypair (one-time)**

Run:
```sh
cargo tauri signer generate -w ~/.tauri/just-ocr.updater.key
```

Expected: prompts for a password (enter one — do not leave empty), then writes:
- `~/.tauri/just-ocr.updater.key` (PRIVATE — never commit; back up securely)
- `~/.tauri/just-ocr.updater.key.pub` (PUBLIC — this gets embedded in config)

**Record the password** somewhere secure (password manager). **Back up the private key file** to offline storage. If either is lost, no future updates can ship to existing installs.

- [ ] **Step 2: Read the public key**

Run:
```sh
cat ~/.tauri/just-ocr.updater.key.pub
```

Copy the entire output (it looks like `dW50cnVzdGVkIGNvbW1l...` base64-ish blob, possibly multi-line).

- [ ] **Step 3: Add `createUpdaterArtifacts` + the updater plugin config**

In `src-tauri/tauri.conf.json`, modify the `bundle` object to add `createUpdaterArtifacts`, and add a top-level `plugins` object. Replace the existing `"bundle": { ... }` block with:

```json
  "bundle": {
    "active": true,
    "createUpdaterArtifacts": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ]
  },
  "plugins": {
    "updater": {
      "pubkey": "PASTE_PUBLIC_KEY_HERE",
      "endpoints": [
        "https://github.com/pndaza/just-ocr/releases/latest/download/latest.json"
      ]
    }
  },
```

Then replace `"PASTE_PUBLIC_KEY_HERE"` with the literal public-key string from Step 2 (keep it as a single JSON string — join multi-line output into one line, or escape newlines as `\n`). **Do not use a file path** — Tauri requires the literal key string.

- [ ] **Step 4: Verify the config parses + compiles**

Run (inside `src-tauri/`):
```sh
cargo check
```

Expected: compiles with no errors. `tauri-build` validates `tauri.conf.json` at this stage; a malformed config errors here.

- [ ] **Step 5: Commit**

```sh
git add src-tauri/tauri.conf.json
git commit -m "feat(config): enable updater — pubkey + endpoint, createUpdaterArtifacts"
```

---

## Task 10: Wire signing secrets + auto-publish into release CI

**Files:**
- Modify: `.github/workflows/release.yml`

- [ ] **Step 1: Add the GitHub Actions secrets (manual, outside the repo)**

In the GitHub repo UI (Settings → Secrets and variables → Actions → New repository secret), add two secrets:
- `TAURI_SIGNING_PRIVATE_KEY` — paste the **entire contents** of `~/.tauri/just-ocr.updater.key` (the private key file, including the `untrusted comment:` header line if present).
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` — the password you set in Task 9 Step 1.

There is no automation step for this; it is a manual prerequisite. Verify they exist before pushing the next release tag.

- [ ] **Step 2: Add the signing secrets to the `tauri-action` env**

In `.github/workflows/release.yml`, find the `uses: tauri-apps/tauri-action@v0` step's `env:` block (around line 93). Add the two signing secrets after `GITHUB_TOKEN`:

```yaml
      - uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          # Updater signing: the private key + its password. tauri-action signs
          # each updater bundle and emits a .sig alongside it, plus the
          # latest.json manifest (uploadUpdaterJson defaults to true). The
          # matching public key is embedded in tauri.conf.json (plugins.updater.pubkey).
          TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}
          CFLAGS: ${{ matrix.platform == 'macos-latest' && '-mmacosx-version-min=10.15' || '' }}
          CXXFLAGS: ${{ matrix.platform == 'macos-latest' && '-mmacosx-version-min=10.15' || '' }}
```

- [ ] **Step 3: Flip to auto-publish**

In the same `tauri-action` `with:` block, change `releaseDraft` from `true` to `false`:

```yaml
        with:
          tagName: app-v__VERSION__
          releaseName: 'just-ocr v__VERSION__'
          releaseBody: 'See the assets below to download and install this version.'
          # Auto-publish: the release (and its latest.json) goes live as soon as
          # all platforms pass CI, so existing installs discover the update via
          # the releases/latest/download endpoint. The manual review gate is
          # removed — fail-fast: false still surfaces every platform failure
          # before the release is created.
          releaseDraft: false
          prerelease: false
          args: ${{ matrix.args }}
```

- [ ] **Step 4: Lint the YAML**

Run:
```sh
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/release.yml')); print('ok')"
```

Expected: prints `ok`, exit 0. (If python/yaml unavailable, visually verify indentation matches the surrounding block.)

- [ ] **Step 5: Commit**

```sh
git add .github/workflows/release.yml
git commit -m "ci(release): sign updater bundles + auto-publish releases"
```

---

## Task 11: Document the updater in AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Extend the Release section**

In `AGENTS.md`, find the `## Release` section. Replace its single paragraph with:

```markdown
## Release

Push a tag matching `v*` (e.g. `v0.1.0`) to trigger `.github/workflows/release.yml`,
which builds macOS (aarch64 + x86_64), Linux, and Windows and **auto-publishes**
a GitHub Release on green CI (releases are no longer draft — the updater needs
the release published so the `latest.json` endpoint resolves). The app version
source of truth is `src-tauri/tauri.conf.json` `version` (mirrored in
`Cargo.toml` + `package.json`).

**In-app updater.** The app checks GitHub for a newer version on startup
(silent — errors swallowed for offline use) and surfaces an accent badge on the
Settings gear when one exists. A "Check for updates" action in Settings does a
manual re-check; install is two-step (Check → Download & install). The updater
is the Tauri v2 plugin: `tauri-plugin-updater` (Rust) +
`@tauri-apps/plugin-updater` (JS), driven entirely from the frontend (no custom
Rust commands). `tauri-plugin-process` provides the post-install `relaunch()`.

- **Endpoint:** `https://github.com/pndaza/just-ocr/releases/latest/download/latest.json`
  — auto-generated by `tauri-action` (`uploadUpdaterJson` is on by default).
- **Signing:** a keypair generated via `cargo tauri signer generate`. The
  **public** key is embedded as a literal string in `tauri.conf.json`
  (`plugins.updater.pubkey`). The **private** key + its password live only as
  the GitHub Actions secrets `TAURI_SIGNING_PRIVATE_KEY` and
  `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`. **If the private key is lost, no further
  updates can ship to existing installs** — back it up outside the repo.
- First release with the updater (the version that introduces it) cannot
  auto-update pre-existing installs — users on the prior version must download
  it manually once; from then on, auto-update works.
```

- [ ] **Step 2: Commit**

```sh
git add AGENTS.md
git commit -m "docs(agents): document the in-app updater pipeline + signing"
```

---

## Task 12: End-to-end manual verification

This task has no code changes — it verifies the integrated feature. Most steps require a published release and cannot run until after the first updater-enabled tag is pushed and CI passes. The local dry-run (Step 1) verifies the flow before any release.

- [ ] **Step 1: Local dry-run with a dev manifest**

Because the live endpoint only serves *published* releases, verify the install flow locally first:

1. Confirm `~/.tauri/just-ocr.updater.key` exists from Task 9.
2. Export the signing env for a local build:
   ```sh
   export TAURI_SIGNING_PRIVATE_KEY="$(cat ~/.tauri/just-ocr.updater.key)"
   export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="<password from Task 9>"
   ```
3. Build the current version (say `0.3.1` — bump in `tauri.conf.json` if needed for the test):
   ```sh
   cargo tauri build
   ```
   Confirm it emits a `.sig` next to the bundle (e.g. in `src-tauri/target/release/bundle/`).
4. Hand-write a `latest.json` pointing at the built bundle, with `version` higher than the installed build. Serve it over HTTP from a local dir:
   ```sh
   cd <dir-with-bundle-and-sig> && python3 -m http.server 8080
   ```
   Point the app at it via a **dev-only** config override (temporarily set `endpoints` to `http://localhost:8080/latest.json` and add `"dangerousInsecureTransportProtocol": true` under `plugins.updater`), rebuild at a lower version, run, and click "Check for updates" in Settings.
5. Expected: the Updates section shows the available version; clicking "Download & install" downloads, installs, and relaunches into the higher version.
6. **Revert** the dev-only endpoint/`dangerousInsecureTransportProtocol` overrides before committing — the shipped config must point at the HTTPS GitHub URL only.

- [ ] **Step 2: Offline launch (silent-failure check)**

With the app built and network disconnected (turn off Wi-Fi):
1. Launch the app.
2. Expected: no error UI, no badge, no console error surfaced to the user. `console.warn("startup update check failed:", ...)` appears in devtools. App is fully functional (can OCR an image).

- [ ] **Step 3: Manual check while offline (visible-failure check)**

With network still off:
1. Open Settings → Updates section.
2. Click "Check for updates".
3. Expected: the inline red error message appears (e.g. network/timeout error text), with a "Retry" button. No crash, no uncaught exception.

- [ ] **Step 4: Up-to-date check**

With network on, on the latest published version:
1. Open Settings → Updates → "Check for updates".
2. Expected: "Just OCR is up to date ✓".

- [ ] **Step 5: Full happy path per platform (post-release)**

After pushing the first updater-enabled tag and CI publishing the release:
1. Install an **older** build on each platform (macOS aarch64, macOS x86_64, Linux, Windows).
2. Launch, confirm the gear badge appears shortly after startup.
3. Open Settings, confirm the available version shows, click "Download & install".
4. Expected: progress bar fills, app relaunches (macOS/Linux) or exits-and-installs (Windows), and the relaunched app reports the new version in the Updates section header.
5. Repeat per platform.

- [ ] **Step 6: Final commit (verification notes, if any)**

If the dry-run surfaced any fix, commit it. Otherwise no commit — this task is verification only. Mark all steps complete.

---

## Notes for the implementer

- **Builds are slow.** First `cargo check` / `cargo tauri build` after adding the crates compiles Tesseract + candle from source (several minutes). Subsequent builds are fast.
- **The `Update` type** from `@tauri-apps/plugin-updater` is imported but only its `.version` property and `.downloadAndInstall()` method are used — both confirmed against the v2 API.
- **Svelte 5 runes:** `$state`, `$effect`, `$derived`, `$props` are already in use across the codebase; the new code follows the same patterns.
- **No new persisted state.** Unlike theme/engine/segmenter, the updater adds no localStorage key — the check is fresh each launch by design.
- **Windows auto-exit:** the installer exits the app during install; `relaunch()` is effectively a no-op there. The UI copy says "The app will restart" which holds for macOS/Linux; on Windows the app closes and the installer relaunches it.
