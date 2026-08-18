// @vitest-environment jsdom
/**
 * Component-level regression tests for the AI Spell Fix panel, mounted for
 * real (Svelte 5 `mount`) with the Gemini backend mocked. These exist to
 * guard the TIER COHERENCE between what the panel checks/applies and what
 * the Text panel + exports project:
 *
 *   display/export precedence:  manualText > llmFix > spellFix > raw
 *
 * The original bug: the panel computed fixes on the raw/spell-fix basis and
 * wrote `job.llmFix` — silently shadowed by `manualText` on manually-edited
 * pages, so Apply reported success while neither the panel nor the exported
 * file changed. The basis now includes the manual tier, and apply writes
 * back to the tier it read from.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { mount, unmount } from "svelte";
import AiCheck from "./AiCheck.svelte";

// ── Mock the Gemini backend (what AiCheck imports from ./ocr) ────────────────
const llmSpellCheckMock = vi.fn();
const llmRewritePagesMock = vi.fn();
vi.mock("./ocr", async (importOriginal) => {
  const orig = await importOriginal<typeof import("./ocr")>();
  return {
    ...orig,
    llmSpellCheck: (...a: unknown[]) => llmSpellCheckMock(...a),
    llmRewritePages: (...a: unknown[]) => llmRewritePagesMock(...a),
  };
});

// ── Mock the Tauri APIs exportResults touches ─────────────────────────────────
const writeFileMock = vi.fn(async () => undefined);
vi.mock("@tauri-apps/plugin-dialog", () => ({ save: vi.fn(async () => "/tmp/out.txt") }));
vi.mock("@tauri-apps/plugin-fs", () => ({
  writeFile: (...a: unknown[]) => writeFileMock(...a),
  readFile: vi.fn(),
  remove: vi.fn(),
  mkdir: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-opener", () => ({
  revealItemInDir: vi.fn(async () => undefined),
}));
vi.mock("@tauri-apps/api/path", () => ({
  join: vi.fn(async (...s: string[]) => s.join("/")),
}));
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string) => {
    if (cmd === "default_save_dir") return "/tmp";
    throw new Error(`unexpected invoke: ${cmd}`);
  }),
}));

const { exportResults } = await import("./ocr");
import type { Job, WordHighlight } from "./ocr";

/** A done job whose raw OCR text still contains two typos. */
function doneJob(): Job {
  return {
    id: 1,
    name: "page1.png",
    bytes: new Uint8Array(),
    path: null,
    url: "blob:x",
    status: "done",
    result: {
      width: 100,
      height: 100,
      confidence: 90,
      elapsedMs: 5,
      lines: [
        { x0: 0, y0: 0, x1: 100, y1: 10, text: "hte cat sat" },
        { x0: 0, y0: 12, x1: 100, y1: 22, text: "on teh mat" },
      ],
    },
    confidence: 90,
    elapsedMs: 5,
    spellFix: null,
    llmFix: null,
    manualText: null,
    error: null,
  };
}

/** Word pairs the mocked backend returns for every page (1-based lines). */
const WORD_FIXES: { wrong: string; correct: string }[] = [
  { wrong: "hte", correct: "the" },
  { wrong: "teh", correct: "the" },
];

/** Default content-aware spell-check mock: flags each WORD_FIX that still
 *  occurs in the RECEIVED text, with 1-based line numbers computed from it —
 *  so a second round over already-fixed pages naturally finds nothing (the
 *  typos are gone from what the panel sends). */
function spellCheckFromText(pages: string[]) {
  return pages.map((text, i) => {
    const fixes: { wrong: string; correct: string; line: number }[] = [];
    text.split("\n").forEach((l, li) => {
      for (const w of WORD_FIXES) {
        if (l.includes(w.wrong)) fixes.push({ ...w, line: li + 1 });
      }
    });
    return { page: i + 1, fixes };
  });
}

async function click(el: Element) {
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  await new Promise((r) => setTimeout(r, 0));
}

/** Mount the panel and run a full check, resolving once the review list is
 * up. Returns the mount handle + host element for further interaction. */
async function runCheck(
  jobs: Job[],
  mode: "review" | "rewrite",
  onsuggestions: (m: Record<number, WordHighlight[]>) => void = () => {},
) {
  const target = document.createElement("div");
  document.body.appendChild(target);
  const instance = mount(AiCheck, {
    target,
    props: {
      jobs,
      apiKey: "test-key",
      model: "gemini-flash-lite-latest",
      batchSize: 10,
      concurrency: 1,
      mode,
      onclose: () => {},
      onopensettings: () => {},
      onselectpage: () => {},
      onsuggestions,
    },
  });
  const startBtn = [...target.querySelectorAll("button")].find((b) =>
    b.textContent?.includes("Start Check"),
  ) as HTMLButtonElement;
  expect(startBtn).toBeTruthy();
  await click(startBtn);
  // Wait (≤500ms) for the review/audit phase to render.
  for (let i = 0; i < 50; i++) {
    const done =
      mode === "review"
        ? [...target.querySelectorAll("button")].some((b) =>
            b.textContent?.startsWith("Apply"),
          )
        : target.textContent?.includes("applied automatically");
    if (done) break;
    await new Promise((r) => setTimeout(r, 10));
  }
  return { instance, target };
}

/** What exportResults would write to disk for these jobs. */
async function exportedText(job: Job): Promise<string> {
  await exportResults([job], { mergeParagraphs: false, fixSpelling: false });
  const bytes = writeFileMock.mock.calls.at(-1)![1] as Uint8Array;
  return new TextDecoder().decode(bytes);
}

beforeEach(() => {
  llmSpellCheckMock.mockImplementation(
    async (_k: string, _m: string, pages: string[]) => spellCheckFromText(pages),
  );
  llmRewritePagesMock.mockImplementation(async (_k: string, _m: string, pages: string[]) =>
    pages.map((text, i) => ({
      page: i + 1,
      // Same corrections the rewrite model would make, line-structured.
      lines: text.split("\n").map((l) => l.replaceAll("hte", "the").replaceAll("teh", "the")),
    })),
  );
  writeFileMock.mockClear();
});

describe("AI Spell Fix tier coherence (manual apply → export)", () => {
  it("writes llmFix and exports fixed text for an unedited page", async () => {
    const job = doneJob();
    const { instance, target } = await runCheck([job], "review");

    const applyBtn = [...target.querySelectorAll("button")].find((b) =>
      b.textContent?.startsWith("Apply"),
    ) as HTMLButtonElement;
    expect(applyBtn).toBeTruthy();
    expect(applyBtn.disabled).toBe(false);
    await click(applyBtn);

    expect(job.llmFix).toEqual({
      fixedLines: ["the cat sat", "on the mat"],
      fixes: 2,
    });
    const text = await exportedText(job);
    expect(text).toContain("the cat sat");
    expect(text).toContain("on the mat");

    unmount(instance);
    target.remove();
  });

  it("REGRESSION: applies onto manualText (not llmFix) for a hand-edited page, and exports the fixes", async () => {
    const job = doneJob();
    // The user hand-edited this page's text at some point before the check —
    // manual text is authoritative in display AND export, so the AI fixes
    // must land there. (The bug: llmFix was written and silently shadowed,
    // so the export kept the typos while Apply claimed success.)
    job.manualText = "hte cat sat\non teh mat";
    const { instance, target } = await runCheck([job], "review");

    const applyBtn = [...target.querySelectorAll("button")].find((b) =>
      b.textContent?.startsWith("Apply"),
    ) as HTMLButtonElement;
    await click(applyBtn);

    expect(job.manualText).toBe("the cat sat\non the mat");
    expect(job.llmFix).toBeNull(); // never set on a manual-basis page
    const text = await exportedText(job);
    expect(text).toContain("the cat sat");
    expect(text).toContain("on the mat");

    unmount(instance);
    target.remove();
  });

  it("rewrite mode applies instantly onto manualText, and Undo all restores the pre-check text", async () => {
    const job = doneJob();
    job.manualText = "hte cat sat\non teh mat";
    const { instance, target } = await runCheck([job], "rewrite");

    // Auto-apply contract: the correction is in the manual text immediately.
    expect(job.manualText).toBe("the cat sat\non the mat");
    expect(job.llmFix).toBeNull();
    expect(await exportedText(job)).toContain("on the mat");

    const undoBtn = [...target.querySelectorAll("button")].find((b) =>
      b.textContent?.includes("Undo all"),
    ) as HTMLButtonElement;
    expect(undoBtn).toBeTruthy();
    await click(undoBtn);
    expect(job.manualText).toBe("hte cat sat\non teh mat");

    unmount(instance);
    target.remove();
  });

  it("Undo all leaves the user's post-apply edits alone (edit is authoritative)", async () => {
    const job = doneJob();
    job.manualText = "hte cat sat\non teh mat";
    const { instance, target } = await runCheck([job], "rewrite");
    expect(job.manualText).toBe("the cat sat\non the mat");

    // The user edits after the auto-apply; undo must not clobber this.
    job.manualText = "my own text";
    const undoBtn = [...target.querySelectorAll("button")].find((b) =>
      b.textContent?.includes("Undo all"),
    ) as HTMLButtonElement;
    await click(undoBtn);
    expect(job.manualText).toBe("my own text");

    unmount(instance);
    target.remove();
  });

  // ── Stacking across re-checks (the 1-20, then 1-40 workflow) ────────────────
  // A second check that re-covers already-fixed pages must build ON TOP of
  // the applied fixes, never reset them — the user's verified round-one
  // corrections would otherwise be lost whenever the model's second pass
  // flags less (or the user unchecks rows).

  it("REGRESSION: re-checking an already-fixed page stacks round-2 fixes on round-1", async () => {
    const job = doneJob();
    // Round 1: check page, apply the flagged typos.
    const r1 = await runCheck([job], "review");
    const apply1 = [...r1.target.querySelectorAll("button")].find((b) =>
      b.textContent?.startsWith("Apply"),
    ) as HTMLButtonElement;
    await click(apply1);
    expect(job.llmFix).toEqual({
      fixedLines: ["the cat sat", "on the mat"],
      fixes: 2,
    });
    unmount(r1.instance);
    r1.target.remove();

    // Round 2 (the wider 1-40 range re-covering this page): the model now
    // sees the FIXED text and flags something new on it.
    llmSpellCheckMock.mockImplementation(async (_k: string, _m: string, pages: string[]) =>
      pages.map((text, i) => ({
        page: i + 1,
        fixes: text.includes("cat")
          ? [{ wrong: "cat", correct: "dog", line: 1 }]
          : [],
      })),
    );
    const r2 = await runCheck([job], "review");
    const apply2 = [...r2.target.querySelectorAll("button")].find((b) =>
      b.textContent?.startsWith("Apply"),
    ) as HTMLButtonElement;
    await click(apply2);

    // Round-1 corrections retained, round-2 added on top — not a reset.
    expect(job.llmFix).toEqual({
      fixedLines: ["the dog sat", "on the mat"],
      fixes: 2, // lines differing from raw OCR, cumulative
    });
    const text = await exportedText(job);
    expect(text).toContain("the dog sat");
    expect(text).toContain("on the mat");

    unmount(r2.instance);
    r2.target.remove();
  });

  it("REGRESSION: a re-check that finds nothing new leaves applied fixes intact", async () => {
    const job1 = doneJob(); // checked + fixed in round 1
    const job2 = doneJob(); // fresh page, only covered by round 2
    job2.id = 2;
    job2.name = "page2.png";

    const r1 = await runCheck([job1], "review");
    const apply1 = [...r1.target.querySelectorAll("button")].find((b) =>
      b.textContent?.startsWith("Apply"),
    ) as HTMLButtonElement;
    await click(apply1);
    const afterRound1 = job1.llmFix;
    expect(afterRound1).not.toBeNull();
    unmount(r1.instance);
    r1.target.remove();

    // Round 2 covers both pages; the content-aware mock finds nothing on
    // the fixed page 1 (typos gone) and the usual pair on page 2.
    const r2 = await runCheck([job1, job2], "review");
    const apply2 = [...r2.target.querySelectorAll("button")].find((b) =>
      b.textContent?.startsWith("Apply"),
    ) as HTMLButtonElement;
    await click(apply2);

    expect(job1.llmFix).toEqual(afterRound1); // untouched — no silent loss
    expect(job2.llmFix).toEqual({
      fixedLines: ["the cat sat", "on the mat"],
      fixes: 2,
    });

    unmount(r2.instance);
    r2.target.remove();
  });

  it("REGRESSION: rewrite-mode Undo all restores the pre-check llmFix (earlier rounds survive)", async () => {
    const job = doneJob();
    // Fixes verified and applied in an earlier round: only line 1 corrected.
    const earlierRound = {
      fixedLines: ["the cat sat", "on teh mat"],
      fixes: 1,
    };
    job.llmFix = earlierRound;

    const { instance, target } = await runCheck([job], "rewrite");
    // Auto-apply corrected line 2 on top of the earlier projection.
    expect(job.llmFix).toEqual({
      fixedLines: ["the cat sat", "on the mat"],
      fixes: 2,
    });

    const undoBtn = [...target.querySelectorAll("button")].find((b) =>
      b.textContent?.includes("Undo all"),
    ) as HTMLButtonElement;
    await click(undoBtn);
    // Back to exactly what the page had before this check — the earlier
    // round's verified fix is NOT thrown away (undo ≠ reset-to-raw).
    expect(job.llmFix).toEqual(earlierRound);

    unmount(instance);
    target.remove();
  });

  it("REGRESSION: Undo all still reverts the remaining lines after a per-line revert", async () => {
    const job = doneJob();
    job.manualText = "hte cat sat\non teh mat";
    const { instance, target } = await runCheck([job], "rewrite");
    expect(job.manualText).toBe("the cat sat\non the mat"); // both lines applied

    // Keep the original on line 1 only.
    await click(target.querySelector(".revert-btn")!);
    expect(job.manualText).toBe("hte cat sat\non the mat");

    // Undo all must revert line 2 as well — the panel's own revert must not
    // trip the edited-since guard (only real Text-panel edits should).
    const undoBtn = [...target.querySelectorAll("button")].find((b) =>
      b.textContent?.includes("Undo all"),
    ) as HTMLButtonElement;
    await click(undoBtn);
    expect(job.manualText).toBe("hte cat sat\non teh mat");

    unmount(instance);
    target.remove();
  });

  it("REGRESSION: after Undo all, revert is disabled and the restored llmFix is not aliased", async () => {
    const job = doneJob();
    const seeded = { fixedLines: ["the cat sat", "on teh mat"], fixes: 1 };
    job.llmFix = seeded;

    const { instance, target } = await runCheck([job], "rewrite");
    expect(job.llmFix?.fixedLines).toEqual(["the cat sat", "on the mat"]);

    const undoBtn = [...target.querySelectorAll("button")].find((b) =>
      b.textContent?.includes("Undo all"),
    ) as HTMLButtonElement;
    await click(undoBtn);

    // The earlier round's projection is restored by VALUE — the live job
    // state must not share the snapshot's object/array.
    expect(job.llmFix).toEqual(seeded);
    expect(job.llmFix).not.toBe(seeded);
    expect(job.llmFix!.fixedLines).not.toBe(seeded.fixedLines);

    // Nothing of this check is applied anymore → revert disabled (an
    // enabled button would mutate the earlier round's verified projection).
    const revertBtn = target.querySelector(".revert-btn") as HTMLButtonElement;
    expect(revertBtn).toBeTruthy();
    expect(revertBtn.disabled).toBe(true);

    unmount(instance);
    target.remove();
  });

  it("publishes line-scoped highlights so the Text panel marks only the flagged line", async () => {
    const job = doneJob();
    const got: Record<number, WordHighlight[]> = {};
    const { instance, target } = await runCheck(
      [job],
      "review",
      (m) => Object.assign(got, m),
    );

    // Both fixes carry their line; the Text panel scopes marks to it instead
    // of lighting up every occurrence of the word across the page.
    expect(got[job.id]).toEqual([
      { wrong: "hte", line: 1 },
      { wrong: "teh", line: 2 },
    ]);

    unmount(instance);
    target.remove();
  });
});

describe("AI Spell Fix suggestion export (review mode → CSV)", () => {
  /** The CSV text the last export wrote to disk. */
  function lastWrittenCsv(): string {
    const bytes = writeFileMock.mock.calls.at(-1)![1] as Uint8Array;
    return new TextDecoder().decode(bytes);
  }

  it("exports every suggestion with its checkbox state; commas in names are quoted", async () => {
    const job = doneJob();
    job.name = "scan, 01.png"; // comma forces RFC-4180 quoting of the page name
    const { instance, target } = await runCheck([job], "review");

    // Uncheck the second fix (teh → the) — the export must carry the
    // accept/reject decision, not just the model's proposals.
    const boxes = [
      ...target.querySelectorAll('.fix input[type="checkbox"]'),
    ] as HTMLInputElement[];
    expect(boxes.length).toBe(2);
    boxes[1].checked = false;
    boxes[1].dispatchEvent(new Event("change", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 0));

    const exportBtn = target.querySelector(
      'button[aria-label="Export suggestions as CSV"]',
    ) as HTMLButtonElement;
    expect(exportBtn).toBeTruthy();
    await click(exportBtn);

    const csv = lastWrittenCsv();
    expect(csv.split("\n")[0]).toBe("page,line,wrong,correct,applied");
    expect(csv).toContain('"scan, 01.png",1,hte,the,yes');
    expect(csv).toContain('"scan, 01.png",2,teh,the,no');

    unmount(instance);
    target.remove();
  });

  it("the applied-phase export marks the rows that were actually applied", async () => {
    const job = doneJob();
    const { instance, target } = await runCheck([job], "review");

    const applyBtn = [...target.querySelectorAll("button")].find((b) =>
      b.textContent?.startsWith("Apply"),
    ) as HTMLButtonElement;
    await click(applyBtn);

    // Suggestions survive the apply — this is the last chance to keep them.
    const exportBtn = target.querySelector(
      'button[aria-label="Export suggestions as CSV"]',
    ) as HTMLButtonElement;
    expect(exportBtn).toBeTruthy();
    await click(exportBtn);

    const csv = lastWrittenCsv();
    expect(csv).toContain("page1.png,1,hte,the,yes");
    expect(csv).toContain("page1.png,2,teh,the,yes");

    unmount(instance);
    target.remove();
  });
});
