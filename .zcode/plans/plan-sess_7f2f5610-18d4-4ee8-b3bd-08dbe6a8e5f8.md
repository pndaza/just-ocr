## Plan: Light/Dark theme support with a single toggle button

The app's entire color system already flows through 13 CSS variables in `src/styles.css :root`. There's no existing theme code (no `prefers-color-scheme`, `localStorage`, or `data-theme` anywhere). So the plan is: define a light palette as overrides on those same variables, add a `data-theme` attribute switch, persist the choice, and add one toggle button to the toolbar.

### 1. `index.html` — no-flash inline script
Add a tiny inline `<script>` in `<head>` that reads `localStorage["just-ocr:theme"]` (falling back to `matchMedia("(prefers-color-scheme: light)")`) and sets `document.documentElement.dataset.theme` **before** the app renders. This prevents a flash of the wrong theme on load. No styling in this file otherwise.

### 2. `src/styles.css` — light palette + shared translucent variables
- Keep the existing `:root` block as the **dark** (default) palette, but also set `data-theme="dark"` semantics by defining the dark values there.
- Introduce a few new shared variables for the translucent/overlay shades that are currently hardcoded as rgba literals in components, so both themes share them:
  - `--accent-soft` (already exists — keep)
  - `--ok-soft`, `--danger-soft` — replaces `rgba(139,216,139,0.1)` / `rgba(255,107,107,0.1/0.06/0.2)` in Preview, Output, LanguageManager
  - `--overlay` — replaces the `rgba(0,0,0,…)` scrim/backdrop/drop-overlay literals
- Add a `[data-theme="light"]` selector that re-defines all 13 color variables + the new soft/overlay ones to a warm light palette (warm off-white backgrounds, dark brown text, same orange accent). Dark remains the default in `:root`.

### 3. `src/theme.ts` (new, ~20 lines)
A small helper module so theme logic lives in one place:
- `type Theme = "light" | "dark"`
- `getTheme()` / `setTheme(t)` — reads/writes `localStorage["just-ocr:theme"]` and applies `document.documentElement.dataset.theme`
- `toggleTheme()` — flips between the two
- `initTheme()` — reads stored (or system) preference; the inline script already did this, but `initTheme` is the SSOT used on app boot so the Svelte state stays in sync.

### 4. `src/App.svelte` — theme state
- Add `let theme = $state<Theme>(currentTheme())` (from `theme.ts`).
- Add `function toggleTheme() { theme = toggleAndPersist(); }` and pass it down to the Toolbar as a callback prop.
- Replace the two hardcoded rgba literals identified by exploration so they theme-swap cleanly:
  - line ~276 app gradient `rgba(255,184,107,0.04)` → `var(--accent-soft)` (or a new `--bg-glow` var)
  - line ~322 drop overlay `rgba(22,19,15,0.7)` → `var(--overlay)`

### 5. `src/lib/Toolbar.svelte` — the toggle button
- Add `ontoggletheme: () => void` and `theme: "light" | "dark"` to the `Props` interface and destructure them.
- Insert a single compact icon button in the right cluster (after the spacer, near Export/Run buttons) styled like the existing `.btn.ghost`. It shows a sun glyph (`☀`) in dark mode (click → go light) and a moon glyph (`☾`) in light mode (click → go dark). One button, swaps both icon and `title`/`aria-label` based on current theme.

### 6. Replace hardcoded rgba literals in components (so light theme is fully correct)
- `src/lib/Preview.svelte` — `.status-pill.done`/`.error` backgrounds → `var(--ok-soft)` / `var(--danger-soft)`
- `src/lib/Output.svelte` — `.error` backgrounds → `var(--danger-soft)`
- `src/lib/LanguageManager.svelte` — `.backdrop` → `var(--overlay)`; `.banner` green tint → `var(--ok-soft)`
- `src/lib/Thumbnail.svelte` — `.thumb-wrap { background:#000 }` → `var(--bg-inset)`; the badge/remove black overlays → `var(--overlay)`; the spinner `rgba(255,255,255,0.3)` → `var(--accent-soft)` (already dark-on-dark friendly)

### 7. Verify
- `npm run build` (frontend type-check + build)
- `cargo tauri dev`, confirm: toggle flips all panels/modal/scrollbars, persists across reload (localStorage), and respects system preference on first launch.

### Not doing
- No new dependencies (no color-system libraries).
- No Rust/Tauri changes — theme is purely a frontend concern.
- Not adding an auto/system option as a third state — request was a single switch button that toggles between light and dark.