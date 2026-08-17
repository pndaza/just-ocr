import {
  isPdf,
  isAppTempPath,
  lastLlmApiKey,
  saveLlmApiKey,
  lastLlmModel,
  saveLlmModel,
  lastLlmBatchSize,
  llmDailyLimit,
  saveLlmBatchSize,
  lastLlmConcurrency,
  saveLlmConcurrency,
  lastAiCheckMode,
  saveAiCheckMode,
  type LlmModel,
} from "./ocr";
import { describe, it, expect, beforeEach, afterEach } from "vitest";

// The LLM pref helpers read/write `localStorage`, which doesn't exist under
// the node vitest environment. Stub it with a Map-backed impl (same pattern
// as theme.test.ts). Restored in afterEach so leaking state can't affect
// other suites if this file grows.
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
  return {
    clear: () => (store = {}),
    restore: () => {
      Object.defineProperty(globalThis, "localStorage", {
        value: undefined,
        configurable: true,
        writable: true,
      });
    },
  };
}

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

// Guards disposeJobFile: only files the backend's render_pdf wrote into its
// `just-ocr-<pid>-<seq>` temp namespace may be deleted when a job is removed.
// Drag-dropped images carry their ORIGINAL path on the job — deleting those
// destroyed user data.
describe("isAppTempPath", () => {
  it("accepts render_pdf's temp page PNGs on both path styles", () => {
    expect(isAppTempPath("/var/folders/xx/T/just-ocr-1234-0/p001.png")).toBe(true);
    expect(isAppTempPath("C:\\Users\\me\\AppData\\Local\\Temp\\just-ocr-999-42\\p012.png")).toBe(true);
    // Page numbers beyond the {:03} zero-pad are min-width, not max.
    expect(isAppTempPath("/tmp/just-ocr-7-3/p1004.png")).toBe(true);
  });
  it("rejects user files — the drag-drop case the guard exists for", () => {
    expect(isAppTempPath("/Users/me/Pictures/photo.png")).toBe(false);
    expect(isAppTempPath("C:\\Users\\me\\Desktop\\scan.png")).toBe(false);
    expect(isAppTempPath("/Users/me/Documents/report.pdf")).toBe(false);
  });
  it("rejects lookalike paths outside the exact namespace shape", () => {
    // Prefix-sharing dir but no <pid>-<seq> components.
    expect(isAppTempPath("/tmp/just-ocr-backup/photo.png")).toBe(false);
    // Right dir shape but not a pN.png page inside it.
    expect(isAppTempPath("/Users/me/just-ocr-123-4/photo.png")).toBe(false);
    // Namespace dir as the final segment (path to the dir, not a page in it).
    expect(isAppTempPath("/tmp/just-ocr-1234-0")).toBe(false);
  });
});

describe("AI spell check prefs", () => {
  let ls: ReturnType<typeof stubLocalStorage>;

  beforeEach(() => {
    ls = stubLocalStorage();
  });
  afterEach(() => {
    ls.restore();
  });

  it("API key round-trips and defaults to empty string", () => {
    expect(lastLlmApiKey()).toBe("");
    saveLlmApiKey("AIzaSy-test-key");
    expect(lastLlmApiKey()).toBe("AIzaSy-test-key");
  });

  it("model defaults to flash-lite-latest (quota over quality) and validates stored values", () => {
    expect(lastLlmModel()).toBe("gemini-flash-lite-latest");
    saveLlmModel("gemini-3.6-flash");
    expect(lastLlmModel()).toBe("gemini-3.6-flash");
    // A retired/unknown model id falls back to the default rather than
    // surfacing a broken selection after an update — including the
    // speculative "3.7" entry that never shipped in the API.
    ls.clear();
    localStorage.setItem("just-ocr:llm-model", "gemini-3.7-flash");
    expect(lastLlmModel() satisfies LlmModel).toBe("gemini-flash-lite-latest");
    ls.clear();
    localStorage.setItem("just-ocr:llm-model", "gemini-1.0-nano");
    expect(lastLlmModel() satisfies LlmModel).toBe("gemini-flash-lite-latest");
  });

  it("batch size defaults to 30, round-trips, and rejects stale values", () => {
    expect(lastLlmBatchSize()).toBe(30);
    saveLlmBatchSize(50);
    expect(lastLlmBatchSize()).toBe(50);
    // Anything not in the offered sizes (10/20/30/40/50) falls back to 30.
    ls.clear();
    localStorage.setItem("just-ocr:llm-batch-size", "17");
    expect(lastLlmBatchSize()).toBe(30);
    ls.clear();
    localStorage.setItem("just-ocr:llm-batch-size", "not-a-number");
    expect(lastLlmBatchSize()).toBe(30);
  });

  it("concurrency defaults to 2, round-trips, and rejects stale values", () => {
    expect(lastLlmConcurrency()).toBe(2);
    saveLlmConcurrency(3);
    expect(lastLlmConcurrency()).toBe(3);
    // Anything not in the offered levels (1/2/3) falls back to 2.
    ls.clear();
    localStorage.setItem("just-ocr:llm-concurrency", "5");
    expect(lastLlmConcurrency()).toBe(2);
    ls.clear();
    localStorage.setItem("just-ocr:llm-concurrency", "");
    expect(lastLlmConcurrency()).toBe(2);
  });

  it("mode defaults to Auto apply (rewrite) and round-trips", () => {
    expect(lastAiCheckMode()).toBe("rewrite");
    saveAiCheckMode("review");
    expect(lastAiCheckMode()).toBe("review");
    // Unknown/stale stored values fall back to the rewrite default.
    ls.clear();
    localStorage.setItem("just-ocr:ai-check-mode", "nonsense");
    expect(lastAiCheckMode()).toBe("rewrite");
  });
});

describe("llmDailyLimit", () => {
  it("caps flash models at 20 requests/day", () => {
    expect(llmDailyLimit("gemini-3.7-flash")).toBe(20);
    expect(llmDailyLimit("gemini-3.6-flash")).toBe(20);
    expect(llmDailyLimit("gemini-flash-latest")).toBe(20);
  });
  it("allows 500/day for flash-lite models", () => {
    expect(llmDailyLimit("gemini-flash-lite-latest")).toBe(500);
  });
  it("assumes the stricter cap for unknown ids", () => {
    expect(llmDailyLimit("some-new-model")).toBe(20);
  });
});
