import { plainText, plainTextWithFix, formatDuration, applyWordFixes, type LineBox, type OcrResult } from "./result";
import { describe, it, expect } from "vitest";

/** Build a LineBox with the geometry the paragraph heuristic reasons about. */
function line(text: string, x0: number, y0: number, x1: number, y1: number): LineBox {
  return { text, x0, y0, x1, y1 };
}

/** Minimal OcrResult wrapper — only `lines` matters for plainText projections. */
function result(lines: LineBox[]): OcrResult {
  return {
    width: 600,
    height: 800,
    lines,
    confidence: 90,
    elapsedMs: 100,
  };
}

// Geometry constants used across the merge-paragraph tests:
//   leftMargin=100 (10th pct of x0), rightMargin=500 (90th pct of x1),
//   blockWidth=400, line height=20 → medianHeight=20.
// Thresholds derived from those: indent 40 (10%), short-right 60 (15%),
// gap 10 (0.5× height).
const FULL = 500; // x1 of a full-width body line reaching the right margin.

describe("plainText — line-by-line (default)", () => {
  it("joins all lines with \\n when mergeParagraphs is omitted", () => {
    const r = result([line("a", 100, 0, FULL, 20), line("b", 100, 20, FULL, 40)]);
    expect(plainText(r)).toBe("a\nb");
  });

  it("joins all lines with \\n when mergeParagraphs is false", () => {
    const r = result([line("a", 100, 0, FULL, 20), line("b", 100, 20, FULL, 40)]);
    expect(plainText(r, { mergeParagraphs: false })).toBe("a\nb");
  });

  it("returns a single line's text unchanged", () => {
    const r = result([line("solo", 100, 0, FULL, 20)]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("solo");
  });
});

describe("plainText — mergeParagraphs", () => {
  it("joins tight full-width body lines into one paragraph (space-separated)", () => {
    const r = result([
      line("one", 100, 0, FULL, 20),
      line("two", 100, 20, FULL, 40),
      line("three", 100, 40, FULL, 60),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("one two three");
  });

  it("splits paragraphs on a large vertical gap", () => {
    // 3 tight lines, then a 60px gap (> 0.5×20), then 2 tight lines.
    const r = result([
      line("a1", 100, 0, FULL, 20),
      line("a2", 100, 20, FULL, 40),
      line("a3", 100, 40, FULL, 60),
      line("b1", 100, 120, FULL, 140),
      line("b2", 100, 140, FULL, 160),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("a1 a2 a3\n\nb1 b2");
  });

  it("splits tight paragraphs by a short last line (no vertical gap)", () => {
    // p1end ends well short of the right margin ⇒ paragraph end, even though
    // p2a follows with zero vertical gap. This is the "no gap between
    // paragraphs" case the gap-only heuristic misses.
    const r = result([
      line("p1a", 100, 0, FULL, 20),
      line("p1b", 100, 20, FULL, 40),
      line("p1c", 100, 40, FULL, 60),
      line("p1end", 100, 60, 300, 80),
      line("p2a", 100, 80, FULL, 100),
      line("p2b", 100, 100, FULL, 120),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("p1a p1b p1c p1end\n\np2a p2b");
  });

  it("treats a centered heading as its own block (no false break on its short right)", () => {
    // A centered heading's right edge is short, but so is its left — it must
    // NOT be classified as a paragraph end. It becomes its own block.
    const r = result([
      line("HEADING", 250, 0, 350, 20),
      line("body1", 100, 20, FULL, 40),
      line("body2", 100, 40, FULL, 60),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("HEADING\n\nbody1 body2");
  });

  it("breaks on body → centered → body transitions", () => {
    const r = result([
      line("body1", 100, 0, FULL, 20),
      line("TITLE", 250, 20, 350, 40),
      line("body2", 100, 40, FULL, 60),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("body1\n\nTITLE\n\nbody2");
  });

  it("keeps a multi-line centered run together", () => {
    // Two centered heading lines should stay in one block, separate from body.
    const r = result([
      line("TITLE A", 240, 0, 360, 20),
      line("TITLE B", 230, 20, 370, 40),
      line("body1", 100, 40, FULL, 60),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("TITLE A TITLE B\n\nbody1");
  });

  it("preserves input order (backend reading order is trusted, not re-sorted)", () => {
    // The old implementation re-sorted by y0, which re-interleaved columns
    // on multi-column pages and undid the backend's reading-order pass.
    // Grouping now trusts the incoming sequence; y-sorted single-column
    // input still merges, and shuffled input stays shuffled.
    const r = result([
      line("three", 100, 40, FULL, 60),
      line("one", 100, 0, FULL, 20),
      line("two", 100, 20, FULL, 40),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("three one two");
  });

  it("trims per-line whitespace and skips empty lines", () => {
    const r = result([
      line("  hello  ", 100, 0, FULL, 20),
      line("", 100, 20, FULL, 40),
      line("world", 100, 40, FULL, 60),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("hello world");
  });

  it("does not split when a paragraph's last line fills the width (known limitation)", () => {
    // No gap, no indent, and the would-be last line reaches the right margin:
    // geometry alone can't distinguish this from a mid-paragraph wrap. The
    // toggle is the escape hatch for documents that hit this.
    const r = result([
      line("a", 100, 0, FULL, 20),
      line("b", 100, 20, FULL, 40),
      line("c", 100, 40, FULL, 60),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("a b c");
  });

  it("degrades to one paragraph when geometry is degenerate (zero-width block)", () => {
    // All lines share the same x0/x1 → blockWidth 0. Must not throw or divide
    // by zero; falls back to joining everything with spaces.
    const r = result([
      line("x", 100, 0, 100, 20),
      line("y", 100, 20, 100, 40),
    ]);
    expect(plainText(r, { mergeParagraphs: true })).toBe("x y");
  });
});

describe("plainText — mergeParagraphs on multi-column pages", () => {
  // Geometry mirrors the two-column textbook sample: left column x∈[100,300],
  // right column x∈[350,550], staggered line y (right column sits a few px
  // lower), gutter 50px. Columns arrive in reading order (all left lines,
  // then all right) — the backend's column-aware ordering contract.
  const L_FULL = 300; // x1 of a full left-column line
  const R_X0 = 350;
  const R_FULL = 550; // x1 of a full right-column line
  // Enough lines per block that the 90th-percentile right margin isn't
  // swayed by a single outlier line (real pages have 30+ lines/column).
  const N = 12;

  function twoColumnLines(): LineBox[] {
    const lines: LineBox[] = [];
    for (let i = 0; i < N; i++) {
      const y = 60 + i * 25;
      // Last line of each column ends short ⇒ paragraphEnd inside the column.
      const x1 = i === N - 1 ? 250 : L_FULL;
      lines.push(line(`L${i + 1}`, 100, y, x1, y + 20));
    }
    for (let i = 0; i < N; i++) {
      const y = 65 + i * 25; // staggered vs the left column
      const x1 = i === N - 1 ? 440 : R_FULL;
      lines.push(line(`R${i + 1}`, R_X0, y, x1, y + 20));
    }
    return lines;
  }

  it("merges each column independently and keeps left-before-right order", () => {
    const r = result(twoColumnLines());
    expect(plainText(r, { mergeParagraphs: true })).toBe(
      `${Array.from({ length: N }, (_, i) => `L${i + 1}`).join(" ")}\n\n` +
        `${Array.from({ length: N }, (_, i) => `R${i + 1}`).join(" ")}`,
    );
  });

  it("keeps a full-width header its own paragraph above the columns", () => {
    const lines = [line("HEADER", 100, 0, R_FULL, 20), ...twoColumnLines()];
    const r = result(lines);
    expect(plainText(r, { mergeParagraphs: true })).toBe(
      `HEADER\n\n` +
        `${Array.from({ length: N }, (_, i) => `L${i + 1}`).join(" ")}\n\n` +
        `${Array.from({ length: N }, (_, i) => `R${i + 1}`).join(" ")}`,
    );
  });

  it("a badge overlapping the gutter by a few px does not chain the columns", () => {
    // Centered badge in the gutter: overlaps the left column by 20px (≥10%
    // of its own 67px width → joins the left block, where its centering
    // isolates it as its own paragraph) and the right column by 3px (< the
    // 4px floor → splits). Without the tolerance the badge chained both
    // columns into one block and the page shattered into one-line
    // paragraphs against page-wide margins.
    const lines = [
      ...twoColumnLines().slice(0, N), // left column
      line("BADGE", 280, 65, 347, 85),
      ...twoColumnLines().slice(N), // right column
    ];
    const r = result(lines);
    expect(plainText(r, { mergeParagraphs: true })).toBe(
      `${Array.from({ length: N }, (_, i) => `L${i + 1}`).join(" ")}\n\n` +
        `BADGE\n\n` +
        `${Array.from({ length: N }, (_, i) => `R${i + 1}`).join(" ")}`,
    );
  });
});

describe("plainTextWithFix — substitutes fixed line text", () => {
  it("uses fixed text in line-by-line mode", () => {
    // Two raw lines; the fix swaps each line's text. Geometry unchanged.
    const r = result([
      line("rawA", 100, 0, FULL, 20),
      line("rawB", 100, 20, FULL, 40),
    ]);
    expect(plainTextWithFix(r, ["fixA", "fixB"])).toBe("fixA\nfixB");
  });

  it("uses fixed text but keeps paragraph grouping from geometry", () => {
    // Three tight body lines (one paragraph) + one short last line that
    // triggers a paragraph break (the geometry heuristic, not the text).
    // The fix swaps text content; the paragraph break still fires because it
    // keys off the unchanged bboxes.
    const r = result([
      line("p1a", 100, 0, FULL, 20),
      line("p1b", 100, 20, FULL, 40),
      line("p1c", 100, 40, FULL, 60),
      line("p1end", 100, 60, 300, 80),
      line("p2a", 100, 80, FULL, 100),
    ]);
    const fixed = ["FA", "FB", "FC", "FEND", "F2"];
    expect(plainTextWithFix(r, fixed, { mergeParagraphs: true })).toBe(
      "FA FB FC FEND\n\nF2",
    );
  });

  it("falls back to raw text when fixedLines is shorter than lines", () => {
    // Defensive: a partial fixedLines array must not drop lines or throw —
    // missing entries fall through to the raw line text.
    const r = result([
      line("rawA", 100, 0, FULL, 20),
      line("rawB", 100, 20, FULL, 40),
    ]);
    expect(plainTextWithFix(r, ["fixA"])).toBe("fixA\nrawB");
  });

  it("matches plainText when fixedLines equals raw text", () => {
    // If the fix is a no-op (fixed text == raw text), the projection must be
    // byte-identical to plainText — confirms the swap is transparent.
    const r = result([
      line("a", 100, 0, FULL, 20),
      line("b", 100, 20, FULL, 40),
    ]);
    expect(plainTextWithFix(r, ["a", "b"], { mergeParagraphs: true })).toBe(
      plainText(r, { mergeParagraphs: true }),
    );
  });
});

describe("formatDuration — adaptive ms → human string", () => {
  it("renders sub-second durations as milliseconds", () => {
    expect(formatDuration(0)).toBe("0 ms");
    expect(formatDuration(823)).toBe("823 ms");
  });

  it("renders sub-minute durations as seconds with one decimal", () => {
    expect(formatDuration(1000)).toBe("1.0 s");
    expect(formatDuration(12345)).toBe("12.3 s");
    expect(formatDuration(59900)).toBe("59.9 s");
  });

  it("renders minute-plus durations as 'Mm SSs' with zero-padded seconds", () => {
    expect(formatDuration(60000)).toBe("1m 00s");
    expect(formatDuration(65000)).toBe("1m 05s");
    expect(formatDuration(125000)).toBe("2m 05s");
    expect(formatDuration(147000)).toBe("2m 27s");
  });

  it("rolls 59.6s up to the next minute rather than '0m 60s'", () => {
    // 59.95s rounds to 60s — must bump the minute, not print "0m 60s".
    expect(formatDuration(59950)).toBe("1m 00s");
  });
});

// ── applyWordFixes (AI spell-check projection) ─────────────────────────────

describe("applyWordFixes", () => {
  it("replaces every occurrence of each wrong word across all lines", () => {
    const { lines, count } = applyWordFixes(
      ["teh cat", "teh dog and teh bone"],
      [{ wrong: "teh", correct: "the" }],
    );
    expect(lines).toEqual(["the cat", "the dog and the bone"]);
    expect(count).toBe(3);
  });

  it("applies multiple fixes and counts each separately", () => {
    const { lines, count } = applyWordFixes(
      ["recogntion is haard"],
      [
        { wrong: "recogntion", correct: "recognition" },
        { wrong: "haard", correct: "hard" },
      ],
    );
    expect(lines).toEqual(["recognition is hard"]);
    expect(count).toBe(2);
  });

  it("replaces unspaced substrings (Burmese-style, no word boundaries)", () => {
    const { lines, count } = applyWordFixes(["အတာတ်ကို"], [
      { wrong: "တာတ်", correct: "ထာတ်" },
    ]);
    expect(lines).toEqual(["အထာတ်ကို"]);
    expect(count).toBe(1);
  });

  it("stacks on pre-fixed basis lines without mutating the input", () => {
    const basis = ["teh cat"];
    const { lines } = applyWordFixes(basis, [{ wrong: "cat", correct: "dog" }]);
    expect(lines).toEqual(["teh dog"]);
    expect(basis).toEqual(["teh cat"]); // pure — input untouched
  });

  it("ignores empty and no-op fixes", () => {
    const { lines, count } = applyWordFixes(
      ["same text"],
      [
        { wrong: "", correct: "x" }, // empty wrong — would replace everywhere
        { wrong: "same", correct: "same" }, // identical pair — no-op
        { wrong: "absent", correct: "missing" }, // not present
      ],
    );
    expect(lines).toEqual(["same text"]);
    expect(count).toBe(0);
  });

  it("leaves lines without any match unchanged", () => {
    const { lines, count } = applyWordFixes(["clean", "also clean"], [
      { wrong: "dirty", correct: "clean" },
    ]);
    expect(lines).toEqual(["clean", "also clean"]);
    expect(count).toBe(0);
  });
});

describe("applyWordFixes — line-addressed fixes", () => {
  it("replaces only on the addressed line, leaving other lines untouched", () => {
    // The exact Burmese-substring case: the word occurs on both lines, but
    // only line 2's occurrence is flagged.
    const { lines, count } = applyWordFixes(
      ["တို ပထမ", "တို ဒုတိယ"],
      [{ wrong: "တို", correct: "တို့", line: 2 }],
    );
    expect(lines).toEqual(["တို ပထမ", "တို့ ဒုတိယ"]);
    expect(count).toBe(1);
  });

  it("allows the same word on different lines to get different corrections", () => {
    const { lines, count } = applyWordFixes(
      ["teh first", "teh second"],
      [
        { wrong: "teh", correct: "the", line: 1 },
        { wrong: "teh", correct: "ten", line: 2 },
      ],
    );
    expect(lines).toEqual(["the first", "ten second"]);
    expect(count).toBe(2);
  });

  it("treats an out-of-range line as a no-op", () => {
    const { lines, count } = applyWordFixes(
      ["same"],
      [{ wrong: "same", correct: "other", line: 7 }],
    );
    expect(lines).toEqual(["same"]);
    expect(count).toBe(0);
  });

  it("falls back to page-wide replacement when line is absent", () => {
    const { lines, count } = applyWordFixes(
      ["teh one", "teh two"],
      [{ wrong: "teh", correct: "the" }],
    );
    expect(lines).toEqual(["the one", "the two"]);
    expect(count).toBe(2);
  });
});
