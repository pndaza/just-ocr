import {
  isPdf,
  lastLlmApiKey,
  saveLlmApiKey,
  lastLlmModel,
  saveLlmModel,
  lastLlmBatchSize,
  llmDailyLimit,
  saveLlmBatchSize,
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

  it("model defaults to the newest flash and validates stored values", () => {
    expect(lastLlmModel()).toBe("gemini-3.7-flash");
    saveLlmModel("gemini-3.6-flash");
    expect(lastLlmModel()).toBe("gemini-3.6-flash");
    // A retired/unknown model id falls back to the default rather than
    // surfacing a broken selection after an update.
    ls.clear();
    localStorage.setItem("just-ocr:llm-model", "gemini-1.0-nano");
    expect(lastLlmModel() satisfies LlmModel).toBe("gemini-3.7-flash");
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
