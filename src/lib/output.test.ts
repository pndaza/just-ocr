// @vitest-environment jsdom
/**
 * Component-level tests for the Text panel's AI-flagged word highlighting —
 * the marks must be LINE-SCOPED: a word flagged on line 2 lights up only
 * there, not on every other line where it happens to occur. Covers both the
 * line-numbered view (a filter on the row number) and the merged-paragraph
 * view (a span map over the merged text — see Output.svelte).
 */
import { describe, it, expect } from "vitest";
import { mount, unmount } from "svelte";
import Output from "./Output.svelte";
import type { Job, WordHighlight } from "./ocr";

/** A done job whose two lines BOTH contain the flagged word "hte" — the
 * recurring occurrence is what line scoping must keep unmarked. The line
 * boxes are adjacent (zero vertical gap, full width) so mergeParagraphs
 * groups them into ONE paragraph — the interesting case for the span map. */
function job(): Job {
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
        { x0: 0, y0: 10, x1: 100, y1: 20, text: "an hte too" },
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

async function render(
  j: Job,
  mergeParagraphs: boolean,
  highlights: WordHighlight[],
) {
  const target = document.createElement("div");
  document.body.appendChild(target);
  const instance = mount(Output, {
    target,
    props: {
      job: j,
      jobs: [j],
      mergeParagraphs,
      fixSpelling: false,
      highlights,
    },
  });
  await new Promise((r) => setTimeout(r, 0));
  return { instance, target };
}

function marks(target: HTMLElement): HTMLElement[] {
  return [...target.querySelectorAll("mark.ai-hl")] as HTMLElement[];
}

/** Concatenated text of all nodes before/after `el` — Svelte renders empty
 * text anchors between segments, so a single previousSibling can be an
 * anchor rather than the adjacent plain segment. */
function textBefore(el: Element): string {
  let s = "";
  let n = el.previousSibling;
  while (n) {
    s = n.textContent + s;
    n = n.previousSibling;
  }
  return s;
}
function textAfter(el: Element): string {
  let s = "";
  let n = el.nextSibling;
  while (n) {
    s += n.textContent;
    n = n.nextSibling;
  }
  return s;
}

describe("Text panel AI highlighting (line-scoped)", () => {
  it("numbered view: marks the word only on the line it was flagged on", async () => {
    const { instance, target } = await render(job(), false, [
      { wrong: "hte", line: 1 },
    ]);

    const rows = [...target.querySelectorAll(".line-row")];
    expect(rows.length).toBe(2);
    const all = marks(target);
    expect(all.length).toBe(1); // NOT the second occurrence on line 2
    expect(rows[0].contains(all[0])).toBe(true);
    expect(rows[1].querySelector("mark")).toBeNull();

    unmount(instance);
    target.remove();
  });

  it("numbered view: an unscoped flag (line null) marks every occurrence", async () => {
    const { instance, target } = await render(job(), false, [
      { wrong: "hte", line: null },
    ]);

    expect(marks(target).length).toBe(2); // both lines light up

    unmount(instance);
    target.remove();
  });

  it("merged view: the span map keeps the mark on the flagged line's occurrence", async () => {
    const first = await render(job(), true, [{ wrong: "hte", line: 1 }]);
    // Merged text: "hte cat sat an hte too" — only the FIRST hte is marked.
    let all = marks(first.target);
    expect(all.length).toBe(1);
    expect(textBefore(all[0])).toBe("");
    expect(textAfter(all[0])).toBe(" cat sat an hte too");
    unmount(first.instance);
    first.target.remove();

    // Flagging line 2 instead marks only the SECOND occurrence.
    const second = await render(job(), true, [{ wrong: "hte", line: 2 }]);
    all = marks(second.target);
    expect(all.length).toBe(1);
    expect(textBefore(all[0])).toBe("hte cat sat an ");
    expect(textAfter(all[0])).toBe(" too");
    unmount(second.instance);
    second.target.remove();
  });

  it("merged view: applied fixes still holding the word don't light up (no match)", async () => {
    // Line 1 was AI-fixed ("the cat sat") so "hte" only occurs on line 2's
    // raw text — a flag for line 2 marks it; a stale flag for line 1 marks
    // nothing (the word is gone from that line's span).
    const j = job();
    j.llmFix = { fixedLines: ["the cat sat", "an hte too"], fixes: 1 };
    const stale = await render(j, true, [{ wrong: "hte", line: 1 }]);
    expect(marks(stale.target).length).toBe(0);
    unmount(stale.instance);
    stale.target.remove();

    const live = await render(j, true, [{ wrong: "hte", line: 2 }]);
    const all = marks(live.target);
    expect(all.length).toBe(1);
    expect(textBefore(all[0])).toBe("the cat sat an ");
    unmount(live.instance);
    live.target.remove();
  });
});
