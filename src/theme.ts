//! Theme helpers. The user's *preference* is one of `"light" | "dark" |
//! "system"` — `"system"` defers to the OS via `prefers-color-scheme` and is
//! tracked live (the app re-resolves if the OS theme changes at runtime).
//!
//! The CSS, however, stays binary: it only knows `[data-theme="light"]` vs the
//! `:root` (dark) defaults. So we always resolve `"system"` to a concrete
//! `"light"` | `"dark"` before writing to the DOM. Storage holds the raw
//! preference so the picker can highlight "System" correctly on reload.
//!
//! An inline script in `index.html` mirrors this resolution before paint to
//! avoid a flash; these functions keep the app's reactive state in sync.

const KEY = "just-ocr:theme";

/** The user's stored preference. `"system"` means "follow the OS". */
export type Theme = "light" | "dark" | "system";
/** The concrete value actually applied to the DOM (`data-theme` attribute). */
export type ResolvedTheme = "light" | "dark";

/** Resolve the OS preference via `prefers-color-scheme`. Defaults to dark. */
function systemTheme(): ResolvedTheme {
  return window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

/** Resolve any preference to a concrete `"light"` | `"dark"` for the DOM. */
export function resolveTheme(t: Theme): ResolvedTheme {
  return t === "system" ? systemTheme() : t;
}

/**
 * The user's *stored preference*. Defaults to `"system"` when nothing (or an
 * unrecognized value) is persisted — matches the fresh-install behavior (no
 * stored value → follow OS) and is what most desktop apps do.
 *
 * Note: this returns the preference, not the resolved DOM value. Use
 * `resolveTheme(currentTheme())` if you need the concrete `"light"`|`"dark"`.
 */
export function currentTheme(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
  return "system";
}

/**
 * Persist a preference and apply its resolved value to the document root.
 * Returns the preference passed in (so callers can update reactive state).
 */
export function setTheme(t: Theme): Theme {
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
  document.documentElement.dataset.theme = resolveTheme(t);
  return t;
}
