import { isPdf } from "./ocr";
import { describe, it, expect } from "vitest";

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
