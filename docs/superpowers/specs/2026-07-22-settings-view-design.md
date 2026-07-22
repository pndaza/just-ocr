# Settings view

**Date:** 2026-07-22
**Status:** Approved (brainstormed)

## Goal

Add a **Settings modal** to Just OCR — a persistent home for user preferences
that today either live only in the toolbar or don't persist at all. This first
cut holds two settings (Theme, Binarization) and is structured to grow.

## Background / motivation

Today the only controls are crammed into the 48px Toolbar: theme toggle,
engine select, PSM, whitelist, and (for Myanmar+Kraken) a binarize dropdown.
Two problems:

1. **No persistence for binarization.** `opts.binarize` resets to `null` every
   launch. Users who depend on a 1-bit-trained recognition model must re-pick
   Sauvola/Otsu each session.
2. **Toolbar crowding.** Binarize is a niche, model-dependent control; it
   doesn't belong next to engine/language that most users touch every run.

The binarize dropdown was added in the preceding change (optional line-crop
binarization for the Kraken recognizer). Settings is the natural home for it.

## Design decisions (from brainstorm)

- **General settings home** — not binarize-only. Structured to grow (engine
  defaults, model paths, etc. in future), but ships with Theme + Binarization.
- **Binarize moves out of the Toolbar** into Settings only.
- **Theme toggle leaves the Toolbar** — theme is controlled only in Settings.
- **Segmented control** (`Off | Otsu | Sauvola`) for binarization — one control,
  three states, reuses the existing `PdfModeDialog` segmented pattern. Picking
  a mode turns it on; no separate on/off switch.
- **Modal dialog** (not a panel/sheet) — matches `PdfModeDialog` and
  `LanguageManager`, the two existing overlays.

## UX

### Entry point

A **gear glyph `⚙`** replaces the **theme toggle `☾`** in the Toolbar's left
cluster (after the brand + divider, before the language picker). Clicking it
opens the Settings modal. There is no longer a theme toggle in the toolbar.

### Settings modal

Modal shell reuses the existing pattern (`.backdrop` scrim + `.modal` card,
Esc + backdrop-click to close, `✕` close button). Width ~420px, centered.

Two sections, each a labeled block with a small uppercase `.lbl` heading:

1. **Theme** — segmented `Light | Dark`. Bound to the existing `data-theme`
   system (`src/theme.ts`). Selecting sets the `data-theme` attribute on
   `<html>` and persists via the existing `just-ocr:theme` key. The FOUC
   inline script in `index.html` is **unchanged** — it only governs initial
   load; Settings mutates the attribute reactively after mount.

2. **Binarization** — segmented `Off | Otsu | Sauvola`, with a one-line
   description: *"For recognition models trained on 1-bit images. Myanmar /
   Kraken path only; ignored by Tesseract."*

   - `Off` maps to `null` (no binarization).
   - `Otsu` → `"otsu"`, `Sauvola` → `"sauvola"` (the existing `Binarize` type).
   - Only affects the Kraken recognizer path; `engine.rs` already ignores
     `binarize` for Tesseract — no backend change.

   This control is always visible in Settings (not gated on language/engine),
   because it's a persisted global preference. Its value simply has no effect
   until the user runs Myanmar+Kraken OCR.

### State + data flow

All state stays owned in `App.svelte` (matching the existing one-way-flow
convention — no Svelte context API):

- `App.svelte` gains a `showSettings` `$state` boolean (mirrors `showLangManager`,
  `pdfDialog`).
- `<Settings>` receives `opts` (reactive ref, so `bind:` inside propagates) and
  an `onclose` callback prop. It binds `opts.binarize` directly.
- Theme is handled via the existing `theme.ts` helpers (`theme()`, `setTheme()`),
  not via `opts` — theme is a UI concern, not an OCR option.
- `Toolbar.svelte` loses its theme toggle and binarize dropdown; gains a gear
  button that fires an `onsettings` callback prop.

### Persistence

New persistence for binarize, mirroring the `just-ocr:` pattern in `ocr.ts`
(`lastEngine`/`saveEngine`, `lastLanguage`/`saveLanguage`):

- Key `just-ocr:binarize`
- `lastBinarize(): Binarize | null` — returns `null` on miss/error (wrapped in
  try/catch for private mode).
- `saveBinarize(b: Binarize): void` — writes the string (`"otsu"`/`"sauvola"`)
  or removes the key for `null` (so storage stays clean and `null` round-trips).

`App.svelte`:
- Initializes `opts.binarize` from `lastBinarize()` at startup.
- Adds a `$effect(() => saveBinarize(opts.binarize))` watcher.

Theme persistence is **unchanged** — it already works via `theme.ts`. Only the
control surface moves into the modal.

## Files

| File | Change |
|---|---|
| `src/lib/Settings.svelte` | **New.** Modal: Theme segmented + Binarization segmented. Props `{ opts, onclose }`. |
| `src/lib/Toolbar.svelte` | Remove theme toggle + binarize dropdown. Add gear `⚙` button → `onsettings` prop. |
| `src/App.svelte` | `showSettings` state; render `<Settings>`; wire `onsettings`; init + persist `opts.binarize`. |
| `src/lib/ocr.ts` | Add `lastBinarize()` / `saveBinarize()` + `BINARIZE_KEY`. |
| `src/theme.ts` | No change (helpers reused). |
| `src-tauri/` (backend) | **No change.** Binarize already flows through `OcrOpts.binarize` → `engine.rs`. |

## Out of scope

- No new Rust command or IPC change.
- No persistence for `psm` / `whitelist` (stay in toolbar, reset per session).
  The Settings home is structured to grow these later, but they're not in this cut.
- No settings for model paths, engine defaults, or output format — future work.
- No change to the FOUC-prevention inline script in `index.html`.
