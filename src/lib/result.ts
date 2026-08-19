//! Structured OCR result types + projections.
//!
//! Replaces the hOCR-XML contract. Both the preview overlay (uses `lines`'
//! bboxes) and the text panel (joins `lines[].text`) are projections of the
//! same typed `OcrResult` returned by the Rust backend.

export interface LineBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  text: string;
  /** True boundary polygon (source-image pixel space). Present only for the
   *  Kraken-segmented (Myanmar) path; absent for Tesseract full-page. */
  polygon?: [number, number][];
}

export interface OcrResult {
  width: number;
  height: number;
  lines: LineBox[];
  /** Mean recognizer confidence in [0,100]; -1 when unknown. */
  confidence: number;
  elapsedMs: number;
  /** Per-stage timing (Myanmar/Kraken-segmented path only). Absent for
   *  full-page Tesseract, which doesn't measure the stages separately. */
  segmentationMs?: number;
  recognitionMs?: number;
}

/** Options for {@link plainText}. */
export interface PlainTextOpts {
  /** When true, group lines into paragraphs using a geometry heuristic on
 *  the per-line bboxes (see {@link groupParagraphs}): lines are split into
 *  column blocks first (reading order), then merged per block. Lines within
 *  a paragraph join with a single space; paragraphs join with "\n\n".
 *  Default false → every line joined with "\n" (legacy behaviour). */
  mergeParagraphs?: boolean;
}

/** The line boxes for the preview overlay. */
export function lineBoxes(result: OcrResult): LineBox[] {
  return result.lines;
}

/** Join line text with "\n" for display/export. */
export function plainText(result: OcrResult, opts?: PlainTextOpts): string {
  if (!opts?.mergeParagraphs) {
    return result.lines.map((l) => l.text).join("\n");
  }
  return groupParagraphs(result.lines)
    .map((para) => para.map((l) => l.text.trim()).filter((t) => t.length > 0).join(" "))
    .filter((p) => p.length > 0)
    .join("\n\n");
}

/**
 * Like {@link plainText}, but substitutes each line's text with its
 * spell-fixed counterpart from `fixedLines` before joining. Geometry (the
 * `LineBox` bboxes that drive paragraph grouping) is unchanged — only the
 * text content swaps, so the merge-paragraphs projection stays identical.
 *
 * `fixedLines` parallels `result.lines`; if it's shorter, missing entries
 * fall back to the raw line text (defensive — should not normally happen).
 */
export function plainTextWithFix(
  result: OcrResult,
  fixedLines: string[],
  opts?: PlainTextOpts,
): string {
  // Build a view of result.lines with text swapped in. We don't mutate the
  // underlying OcrResult; we map to a shallow copy carrying the fixed text.
  const swapped: LineBox[] = result.lines.map((l, i) => ({
    ...l,
    text: i < fixedLines.length ? fixedLines[i] : l.text,
  }));
  if (!opts?.mergeParagraphs) {
    return swapped.map((l) => l.text).join("\n");
  }
  return groupParagraphs(swapped)
    .map((para) => para.map((l) => l.text.trim()).filter((t) => t.length > 0).join(" "))
    .filter((p) => p.length > 0)
    .join("\n\n");
}

// ── Paragraph grouping heuristic ─────────────────────────────────────────────
//
// Pure geometry — no ML, no font/baseline analysis. Uses only the per-line
// bboxes every OCR path already produces. Lines are expected in READING
// ORDER — the backend's contract (the column-aware reading-order pass for
// the Myanmar/segmenter paths; Tesseract's own iterator for full-page,
// which is reading-ordered under the auto-segmentation PSMs but can emit
// raster order under sparse-text PSMs 11/12, where this grouping degrades
// to per-line blocks — no worse than the old global y-sort).
// Grouping runs in two stages:
//
// Stage 1 — column blocks. Walking the lines in reading order, consecutive
// lines stay in one block while their x-intervals overlap by a meaningful
// amount (≥ BLOCK_OVERLAP_FRACTION of the narrower line, ≥ a small px
// floor). A column switch — the left column's last line followed by the
// right column's first, which share no x-overlap — opens a new block. The
// fractional tolerance keeps a stray box sitting in the gutter (a centered
// badge overlapping both columns by a few px) from chaining the columns
// together. Zero-width boxes carry no horizontal signal and never split.
//
// Stage 2 — paragraphs within a block. Margins are computed PER BLOCK:
// against page-wide margins every left-column line reads as a paragraph end
// and every right-column line reads as centered, which is why multi-column
// pages used to shatter into one-line paragraphs. Within a block, a line is
// classified as one of:
//
//   body        — the dominant alignment: reaches the dominant right margin
//                 and sits at the majority left level. Mid-paragraph wrap.
//   start       — sits at a MINORITY level of the block's x0 distribution
//                 (see `paragraphStarts`) while still reaching the right
//                 margin. A paragraph's first line. This is style-agnostic:
//                 first-line-indent style puts starts RIGHT of the dominant
//                 edge (thawzin_02: starts ~160 vs body ~84), hanging-indent
//                 style puts them flush LEFT of the indented majority
//                 (two_colums-1: starts ~430 vs continuation ~560). Either
//                 way the starts are the minority level — only the direction
//                 differs, and the code doesn't care.
//   paragraphEnd— reaches the majority left level but ends well short of the
//                 right margin. Almost certainly the last line of a paragraph
//                 (the only full-width line that ends short is a paragraph's
//                 tail).
//   centered    — inset on BOTH sides. Headings/titles/captions, not body.
//                 Each centered run is its own block; this also keeps a
//                 heading's naturally-short right edge from being read as a
//                 false paragraph-end. A marked start that ALSO ends short
//                 isolates the same way (one-line paragraph / heading at the
//                 marked level).
//
// A paragraph break is inserted before line N+1 when ANY of:
//   1. The vertical gap after line N exceeds ~0.5× the median line height
//      (paragraphs separated by blank space).
//   2. Line N is a paragraphEnd (tight/justified text with no inter-paragraph
//      gap — the case pure gap detection misses).
//   3. Line N+1 is a start or centered (a paragraph start / heading);
//      consecutive centered lines stay together within their run, but every
//      start line begins its own paragraph (list-item style, one line each).
// Block boundaries always break (a column switch is always a new paragraph).
//
// Known limitations: a paragraph whose last line coincidentally fills the
// full width with no following gap is indistinguishable from a mid-paragraph
// wrap and won't be split (unless the NEXT paragraph starts at a marked
// level). Blocks with more than two x0 levels (hanging paragraphs plus
// deeper-indented verse) mark each outlier level as starts. A hanging
// indent deeper than ~12.5% of the block width (with starts at the 40%
// share cap) drags the mean-x0 reference toward the flush level far enough
// that short tails can isolate as headings. The merge toggle is the escape
// hatch for documents where that matters.

/** A line classified by its horizontal alignment within its column block. */
type LineKind = "body" | "start" | "paragraphEnd" | "centered";

const LEFT_MARGIN_PCT = 10; // low percentile of x0 → dominant left margin
const RIGHT_MARGIN_PCT = 90; // high percentile of x1 → dominant right margin
const INDENT_FRACTION = 0.05; // left inset > 5% of block width ⇒ not body-level
const SHORT_RIGHT_FRACTION = 0.15; // right inset > 15% of block width ⇒ paragraphEnd
const GAP_X_MEDIAN_HEIGHT = 0.5; // vertical gap > 0.5× median line height ⇒ break
// Paragraph-start x0 clustering (`paragraphStarts`): the split gap must
// clear an absolute floor — detector x0 jitter on body lines is ~±6px, so
// adjacent jittered lines can present a gap of up to ~12px; 16 leaves
// margin — and a fraction of the block width. Each extracted level must
// stay under 40% of its pool (above that — e.g. verse where most lines
// start paragraphs — the level structure carries no start/end
// information).
const X0_SPLIT_MIN_PX = 16;
const X0_SPLIT_MIN_FRAC = 0.025;
const START_LEVEL_MAX_SHARE = 0.4;
// A line wider than this multiple of the page's median line width is
// "spanning" (a full-width caption or rule BETWEEN columns, a section
// heading over a two-column body): it overlaps both sides, and pairwise
// overlap alone would chain the columns through it into one block.
// Spanning lines become their own blocks instead. The page-wide median
// can't be shifted by the handful of spanning lines a page carries.
const SPANNING_WIDTH_MULTIPLE = 2;
// Stage-1 column-block tolerance: consecutive lines belong to the same block
// only when their x-intervals overlap by this fraction of the narrower line
// (4px floor). Calibrated on the two-column sample: a centered badge in the
// gutter overlaps the left column by ~19% of its own width (joins — harmless,
// classification then isolates it) but the right column by ~1% (splits).
const BLOCK_OVERLAP_FRACTION = 0.1;
const BLOCK_OVERLAP_MIN_PX = 4;

/** Group reading-ordered lines into paragraphs. Exported for unit tests. */
export function groupParagraphs(lines: LineBox[]): LineBox[][] {
  if (lines.length === 0) return [];
  if (lines.length === 1) return [lines];
  return columnBlocks(lines).flatMap((block) => paragraphsInBlock(block));
}

/**
 * Split reading-ordered lines into column blocks: maximal runs of
 * consecutive lines whose x-intervals overlap meaningfully (pairwise, line
 * vs line, so an accumulated extent can't chain columns). Lines much wider
 * than the page median (["spanning"][SPANNING_WIDTH_MULTIPLE] captions,
 * rules, section headings) are emitted as their own blocks — they overlap
 * whatever follows and would otherwise glue the columns on either side
 * into one block with page-wide margins.
 */
function columnBlocks(lines: LineBox[]): LineBox[][] {
  const widths = lines
    .map((l) => Math.max(1, l.x1 - l.x0))
    .sort((a, b) => a - b);
  const medianWidth = widths[Math.floor(widths.length / 2)];
  const blocks: LineBox[][] = [];
  let run: LineBox[] = [];
  let prev: LineBox | null = null;
  for (const l of lines) {
    const spanning = l.x1 - l.x0 > SPANNING_WIDTH_MULTIPLE * medianWidth;
    // Zero-width boxes have no horizontal signal — keep them with their
    // neighbours instead of splitting on `overlap <= 0`.
    const overlapsEnough =
      !prev ||
      prev.x1 - prev.x0 <= 0 ||
      l.x1 - l.x0 <= 0 ||
      Math.min(prev.x1, l.x1) - Math.max(prev.x0, l.x0) >=
        Math.max(
          BLOCK_OVERLAP_MIN_PX,
          BLOCK_OVERLAP_FRACTION *
            Math.min(prev.x1 - prev.x0, l.x1 - l.x0),
        );
    if (run.length && (spanning || !overlapsEnough)) {
      blocks.push(run);
      run = [];
    }
    if (spanning) {
      blocks.push([l]);
    } else {
      run.push(l);
    }
    prev = l;
  }
  if (run.length) blocks.push(run);
  return blocks;
}

/**
 * Paragraphs within one column block: the alignment classification and
 * break rules above, against the block's own margins and median height.
 */
function paragraphsInBlock(block: LineBox[]): LineBox[][] {
  const leftMargin = percentile(block.map((l) => l.x0), LEFT_MARGIN_PCT);
  const rightMargin = percentile(block.map((l) => l.x1), RIGHT_MARGIN_PCT);
  const blockWidth = rightMargin - leftMargin;
  const heights = block
    .map((l) => Math.max(1, l.y1 - l.y0))
    .sort((a, b) => a - b);
  const medianHeight = heights[Math.floor(heights.length / 2)];

  const minSplitGap = Math.max(X0_SPLIT_MIN_PX, X0_SPLIT_MIN_FRAC * blockWidth);
  const starts = paragraphStarts(block, minSplitGap);
  // Mean x0 ≈ the majority "continuation" level: the reference for the
  // centered check. Against the 10th-percentile margin alone, a hanging
  // block's short continuation lines (tails at the indented majority
  // level) would read as inset-on-both-sides and isolate as fake headings.
  // Mean rather than median: with small blocks a lower-median can land on
  // the minority level itself.
  const majorityLeft = block.reduce((s, l) => s + l.x0, 0) / block.length;
  const kinds = block.map((l, i) =>
    classify(l, starts.has(i), leftMargin, majorityLeft, rightMargin, blockWidth, medianHeight),
  );

  // Break before line i when any of:
  //   — large vertical gap (paragraphs separated by blank space)
  //   — previous line was a paragraphEnd (short last line, no gap)
  //   — this line is a start (minority x0 level ⇒ paragraph first line)
  //   — entering or leaving a centered run (heading ↔ body transition);
  //     consecutive centered lines stay together within their run.
  const paragraphs: LineBox[][] = [];
  let current: LineBox[] = [block[0]];
  for (let i = 1; i < block.length; i++) {
    const prev = block[i - 1];
    const cur = block[i];
    const gap = cur.y0 - prev.y1;
    const enteringCentered = kinds[i] === "centered" && kinds[i - 1] !== "centered";
    const leavingCentered = kinds[i - 1] === "centered" && kinds[i] !== "centered";
    const breakBefore = gap > GAP_X_MEDIAN_HEIGHT * medianHeight
      || kinds[i - 1] === "paragraphEnd"
      || kinds[i] === "start"
      || enteringCentered
      || leavingCentered;
    if (breakBefore) {
      if (current.length) paragraphs.push(current);
      current = [cur];
    } else {
      current.push(cur);
    }
  }
  if (current.length) paragraphs.push(current);
  return paragraphs;
}

/**
 * Indices of paragraph-start lines: the minority level(s) of the block's x0
 * distribution. Sorts the pool's x0s, splits at the largest gap, and marks
 * the smaller side; then recurses on the larger side, so a block can carry
 * more than one marked level (e.g. a gutter badge at x0 ≈ 2135 hanging off
 * a two-column block whose paragraphs are flush ~430 vs continuation ~560).
 * Works for both indent conventions because only minority-ness matters:
 * first-line-indent starts sit right of the dominant edge, hanging-indent
 * starts sit flush left of the indented majority.
 *
 * Empty when no level structure clears the guards: gap below `minGap`
 * (jitter), a 40%+ "minority" (verse-like blocks where the level carries no
 * start information), or a degenerate pool.
 */
function paragraphStarts(block: LineBox[], minGap: number): Set<number> {
  const marked = new Set<number>();
  let pool = block.map((_, i) => i);
  while (pool.length >= 3) {
    pool.sort((a, b) => block[a].x0 - block[b].x0);
    let bestGap = 0;
    let splitAt = 0;
    for (let k = 1; k < pool.length; k++) {
      const gap = block[pool[k]].x0 - block[pool[k - 1]].x0;
      if (gap > bestGap) {
        bestGap = gap;
        splitAt = k;
      }
    }
    if (bestGap < minGap) break;
    const low = pool.slice(0, splitAt);
    const high = pool.slice(splitAt);
    const minority = low.length <= high.length ? low : high;
    const majority = minority === low ? high : low;
    if (minority.length === 0 || minority.length > START_LEVEL_MAX_SHARE * pool.length) {
      break;
    }
    for (const i of minority) marked.add(i);
    pool = majority;
  }
  return marked;
}

/** Classify a line by its horizontal position within its column block. */
function classify(
  l: LineBox,
  isStart: boolean,
  leftMargin: number,
  majorityLeft: number,
  rightMargin: number,
  blockWidth: number,
  _medianHeight: number,
): LineKind {
  // Degenerate geometry (zero-width block, e.g. all lines share x0/x1):
  // can't reason about alignment → treat as body so we join rather than split.
  if (blockWidth <= 0) return "body";
  const rightInset = rightMargin - l.x1;
  if (isStart) {
    // A marked start that also ends well short of the right margin is a
    // one-line paragraph or a heading at the marked level → isolate it
    // (centered), don't glue it to the next paragraph.
    return rightInset > SHORT_RIGHT_FRACTION * blockWidth ? "centered" : "start";
  }
  // Heading: short AND inset from the leftmost content AND from the
  // majority level (a hanging block's paragraph tails sit at the indented
  // majority level — short, but not headings).
  if (
    l.x0 - leftMargin > INDENT_FRACTION * blockWidth &&
    l.x0 - majorityLeft > INDENT_FRACTION * blockWidth &&
    rightInset > SHORT_RIGHT_FRACTION * blockWidth
  ) {
    return "centered";
  }
  if (rightInset > SHORT_RIGHT_FRACTION * blockWidth) return "paragraphEnd";
  return "body";
}

/** Nearest-rank percentile. `p` in [0, 100]. */
function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil((p / 100) * sorted.length) - 1),
  );
  return sorted[idx];
}

// ── AI word-fix projection ───────────────────────────────────────────────────

/** One wrong→correct word pair proposed by the LLM spell check. The `wrong`
 *  side is an exact substring of the OCR text (the backend filters pairs
 *  whose `wrong` doesn't occur), so replacement is a plain find/replace.
 *  `line`, when set, scopes the replacement to that 1-based line index —
 *  the AI check addresses fixes per line so short substrings (easy in
 *  unspaced Burmese) only touch the flagged line. */
export interface WordFix {
  wrong: string;
  correct: string;
  line?: number;
}

/**
 * Replace every occurrence of each fix's `wrong` with its `correct` and
 * report how many replacements were made. Pure — returns new strings, never
 * mutates the inputs — so the caller can cache the result as a display-time
 * projection (same shape as the offline spell-fix).
 *
 * A fix with a `line` replaces only within that line (out-of-range line →
 * no-op); without one, across all lines. Plain substring replacement rather
 * than word-boundary matching: Burmese and other complex scripts don't
 * separate words with spaces, so regex `\b` boundaries would silently skip
 * most fixes there. Overlapping pairs are applied in the given order (first
 * fix wins on a shared substring).
 */
export function applyWordFixes(
  lines: string[],
  fixes: WordFix[],
): { lines: string[]; count: number } {
  const usable = fixes.filter((f) => f.wrong.length > 0 && f.wrong !== f.correct);
  let count = 0;
  const out = [...lines];
  for (const f of usable) {
    // Line-addressed fix → that one line only; otherwise the whole page.
    const targets =
      f.line != null && f.line >= 1 && f.line <= out.length
        ? [f.line - 1]
        : f.line != null
          ? [] // invalid line index — apply nowhere (the review UI unchecks it)
          : out.map((_, i) => i);
    for (const i of targets) {
      if (!out[i].includes(f.wrong)) continue;
      count += out[i].split(f.wrong).length - 1;
      out[i] = out[i].split(f.wrong).join(f.correct);
    }
  }
  return { lines: out, count };
}

// ── Duration formatting ──────────────────────────────────────────────────────

/**
 * Format a millisecond duration for compact, human-friendly display. Adaptive
 * so it reads well across the range OCR times span — a single page (~hundreds
 * of ms) up through a large PDF batch (minutes):
 *
 *   < 1 s      → "823 ms"
 *   1–59.9 s   → "12.3 s"
 *   ≥ 60 s     → "2m 05s"  (minutes : zero-padded seconds)
 *
 * Used by the Output panel's batch status bar so a multi-minute total isn't
 * rendered as an opaque "147000 ms".
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  // Sub-minute: one decimal (12.3 s). But the one-decimal rounding can itself
  // roll over (59.95 → "60.0 s"), which would read as a nonsensical "sixty
  // point zero seconds" — so if the formatted seconds are ≥ 60, fall through
  // to the minute format instead.
  if (s < 60) {
    const dec = s.toFixed(1);
    if (parseFloat(dec) < 60) return `${dec} s`;
  }
  // Minute-plus (or a sub-minute value that rounded up to 60s): whole seconds,
  // "Mm SSs" with zero-padded seconds.
  const totalSec = Math.round(s);
  const m = Math.floor(totalSec / 60);
  const rem = totalSec - m * 60;
  return `${m}m ${String(rem).padStart(2, "0")}s`;
}
