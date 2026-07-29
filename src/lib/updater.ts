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
