import { describe, it, expect } from "vitest";
import { mapCheckResult, downloadPercent, type UpdateStatus } from "./updater";

describe("mapCheckResult", () => {
  // A null/undefined Update (no newer version) → up-to-date.
  it("returns up-to-date when no update object is given", () => {
    expect(mapCheckResult(null)).toEqual<UpdateStatus>({ kind: "up-to-date" });
    expect(mapCheckResult(undefined)).toEqual<UpdateStatus>({ kind: "up-to-date" });
  });

  // A present Update → available, carrying its version string.
  it("returns available with the update's version when an update is given", () => {
    const fake = { version: "0.4.0" } as any;
    expect(mapCheckResult(fake)).toEqual<UpdateStatus>({
      kind: "available",
      version: "0.4.0",
    });
  });
});

describe("downloadPercent", () => {
  // Progress percent = (downloaded / contentLength) * 100, floored, clamped 0..100.
  it("computes percent from downloaded bytes and total content length", () => {
    expect(downloadPercent(0, 1000)).toBe(0);
    expect(downloadPercent(250, 1000)).toBe(25);
    expect(downloadPercent(1000, 1000)).toBe(100);
  });

  it("clamps to 100 if downloaded exceeds contentLength", () => {
    expect(downloadPercent(1200, 1000)).toBe(100);
  });

  it("returns 0 when contentLength is 0 (avoids divide-by-zero)", () => {
    expect(downloadPercent(500, 0)).toBe(0);
  });
});
