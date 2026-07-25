import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { currentTheme, setTheme, resolveTheme } from "./theme";

// Theme helpers touch `localStorage`, `window.matchMedia`, and
// `document.documentElement.dataset` — none of which exist under the project's
// default node vitest environment. Stub all three. Mirrors the localStorage
// stub pattern in ocr.test.ts.

function stubLocalStorage() {
  let store: Record<string, string> = {};
  const impl = {
    getItem: (k: string) => (k in store ? store[k] : null),
    setItem: (k: string, v: string) => {
      store[k] = String(v);
    },
    removeItem: (k: string) => {
      delete store[k];
    },
    clear: () => {
      store = {};
    },
  };
  Object.defineProperty(globalThis, "localStorage", {
    value: impl,
    configurable: true,
    writable: true,
  });
  return impl;
}

/** Minimal matchMedia stub. `prefersLight` controls what `.matches` returns
 *  for the "(prefers-color-scheme: light)" query the helpers use. */
function stubMatchMedia(prefersLight: boolean) {
  const impl = (query: string) => ({
    matches: query.includes("light") ? prefersLight : !prefersLight,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  });
  // @ts-expect-error — installing onto a node global
  globalThis.window = globalThis.window || {};
  // @ts-expect-error — partial window install for the node env
  globalThis.window.matchMedia = impl;
  // The helpers read `window.matchMedia` via the global `window`.
  (globalThis as any).matchMedia = impl;
}

/** Minimal document.documentElement stub with a writable `dataset`. */
function stubDocumentRoot() {
  const dataset: Record<string, string> = {};
  const docEl = { dataset };
  // @ts-expect-error — partial document install for the node env
  globalThis.document = { documentElement: docEl };
}

describe("theme helpers", () => {
  let restoreLs: () => void;
  let restoreWindow: () => void;
  let restoreDocument: () => void;

  beforeEach(() => {
    const prevLs = (globalThis as any).localStorage;
    stubLocalStorage();
    restoreLs = () =>
      Object.defineProperty(globalThis, "localStorage", {
        value: prevLs,
        configurable: true,
        writable: true,
      });

    const prevWindow = (globalThis as any).window;
    const prevMatchMedia = (globalThis as any).matchMedia;
    stubMatchMedia(false); // default: OS prefers dark
    restoreWindow = () => {
      (globalThis as any).window = prevWindow;
      (globalThis as any).matchMedia = prevMatchMedia;
    };

    const prevDocument = (globalThis as any).document;
    stubDocumentRoot();
    restoreDocument = () => {
      (globalThis as any).document = prevDocument;
    };
  });
  afterEach(() => {
    restoreLs();
    restoreWindow();
    restoreDocument();
  });

  describe("currentTheme", () => {
    it("defaults to system when nothing is persisted", () => {
      expect(currentTheme()).toBe("system");
    });

    it("returns the stored preference for each of the three values", () => {
      localStorage.setItem("just-ocr:theme", "light");
      expect(currentTheme()).toBe("light");
      localStorage.setItem("just-ocr:theme", "dark");
      expect(currentTheme()).toBe("dark");
      localStorage.setItem("just-ocr:theme", "system");
      expect(currentTheme()).toBe("system");
    });

    it("falls back to system on an unrecognized stored value", () => {
      localStorage.setItem("just-ocr:theme", "hot-pink");
      expect(currentTheme()).toBe("system");
    });
  });

  describe("resolveTheme", () => {
    it("passes light and dark through unchanged", () => {
      expect(resolveTheme("light")).toBe("light");
      expect(resolveTheme("dark")).toBe("dark");
    });

    it("resolves system to OS preference (light)", () => {
      stubMatchMedia(true); // OS prefers light
      expect(resolveTheme("system")).toBe("light");
    });

    it("resolves system to OS preference (dark) when OS prefers dark", () => {
      stubMatchMedia(false); // OS prefers dark
      expect(resolveTheme("system")).toBe("dark");
    });
  });

  describe("setTheme", () => {
    it("persists the preference and returns it", () => {
      const got = setTheme("system");
      expect(got).toBe("system");
      expect(localStorage.getItem("just-ocr:theme")).toBe("system");
    });

    it("writes the RESOLVED value to data-theme, never 'system'", () => {
      // Explicit preferences write themselves.
      setTheme("light");
      expect(document.documentElement.dataset.theme).toBe("light");
      setTheme("dark");
      expect(document.documentElement.dataset.theme).toBe("dark");
      // System resolves to the OS preference — here stubbed to dark.
      stubMatchMedia(false);
      setTheme("system");
      expect(document.documentElement.dataset.theme).toBe("dark");
      // And to light when the OS prefers light.
      stubMatchMedia(true);
      setTheme("system");
      expect(document.documentElement.dataset.theme).toBe("light");
    });
  });
});
