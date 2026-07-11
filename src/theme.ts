//! Light/dark theme helpers. The chosen theme is stored in `localStorage`
//! under `just-ocr:theme` and applied as a `data-theme` attribute on the
//! document root. An inline script in `index.html` applies it before paint to
//! avoid a flash; these functions keep the app's reactive state in sync.

const KEY = "just-ocr:theme";

export type Theme = "light" | "dark";

function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

/** The theme currently applied to the document (falls back to system). */
export function currentTheme(): Theme {
  const v = document.documentElement.dataset.theme;
  return v === "light" || v === "dark" ? v : systemTheme();
}

/** Persist and apply a theme, returning it. */
export function setTheme(t: Theme): Theme {
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* storage may be unavailable (private mode) — ignore */
  }
  document.documentElement.dataset.theme = t;
  return t;
}

/** Flip between light and dark, persist, and return the new theme. */
export function toggleTheme(): Theme {
  return setTheme(currentTheme() === "light" ? "dark" : "light");
}
