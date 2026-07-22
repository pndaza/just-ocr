import { isPdf, lastBinarize, saveBinarize } from "./ocr";
import { describe, it, expect, beforeEach, afterEach } from "vitest";

describe("isPdf", () => {
  it("matches .pdf extension case-insensitively", () => {
    expect(isPdf("scan.pdf")).toBe(true);
    expect(isPdf("SCAN.PDF")).toBe(true);
  });
  it("rejects non-PDF names", () => {
    expect(isPdf("photo.png")).toBe(false);
    expect(isPdf("scan.pdf.bak")).toBe(false);
    expect(isPdf("pdf")).toBe(false);
  });
});

// Minimal localStorage stub so these persistence tests run under node (the
// project's default vitest environment has no DOM). Mirrors the subset of the
// Web Storage API the binarize helpers use.
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

describe("lastBinarize", () => {
  let restore: () => void;
  beforeEach(() => {
    const prev = (globalThis as any).localStorage;
    stubLocalStorage();
    restore = () => {
      Object.defineProperty(globalThis, "localStorage", {
        value: prev,
        configurable: true,
        writable: true,
      });
    };
  });
  afterEach(() => restore());

  it("defaults to sauvola when never set", () => {
    expect(lastBinarize()).toBe("sauvola");
  });

  it("returns an explicitly-disabled choice as null (sticky Off)", () => {
    saveBinarize(null);
    expect(lastBinarize()).toBeNull();
  });

  it("round-trips otsu and sauvola", () => {
    saveBinarize("otsu");
    expect(lastBinarize()).toBe("otsu");
    saveBinarize("sauvola");
    expect(lastBinarize()).toBe("sauvola");
  });
});
