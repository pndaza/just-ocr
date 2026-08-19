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
// the Myanmar/segmenter paths, Tesseract's own iterator for full-page).
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
//   body        — reaches the dominant left margin and the dominant right
//                 margin. Mid-paragraph wrap.
//   indent      — starts right of the dominant left margin but still reaches
//                 the right margin. A paragraph's first line (first-line
//                 indent — the standard paragraph convention in Burmese
//                 print, and often the ONLY signal: the previous paragraph's
//                 last line can end close enough to the right margin that
//                 the short-line rule below misses it).
//   paragraphEnd— reaches the left margin but ends well short of the right
//                 margin. Almost certainly the last line of a paragraph (the
//                 only full-width line that ends short is a paragraph's tail).
//   centered    — inset on BOTH sides. Headings/titles/captions, not body.
//                 Each centered run is its own block; this also keeps a
//                 heading's naturally-short right edge from being read as a
//                 false paragraph-end.
//
// A paragraph break is inserted before line N+1 when ANY of:
//   1. The vertical gap after line N exceeds ~0.5× the median line height
//      (paragraphs separated by blank space).
//   2. Line N is a paragraphEnd (tight/justified text with no inter-paragraph
//      gap — the case pure gap detection misses).
//   3. Line N+1 is an indent or centered (a paragraph start / heading);
//      consecutive centered lines stay together within their run, but every
//      indent line starts its own paragraph (list-item style, one line each).
// Block boundaries always break (a column switch is always a new paragraph).
//
// Known limitations: a paragraph whose last line coincidentally fills the
// full width with no following gap is indistinguishable from a mid-paragraph
// wrap and won't be split (unless the NEXT paragraph starts indented). A
// block quote where every line is indented breaks per line. The merge
// toggle is the escape hatch for documents where that matters.

/** A line classified by its horizontal alignment within its column block. */
type LineKind = "body" | "indent" | "paragraphEnd" | "centered";

const LEFT_MARGIN_PCT = 10; // low percentile of x0 → dominant left margin
const RIGHT_MARGIN_PCT = 90; // high percentile of x1 → dominant right margin
// Left-inset thresholds. Calibrated on sample pages: detector x0 jitter on
// body lines is ~±6px, real first-line indents run 9–11% of the block width
// (thawzin_02: ~80px on an ~820px block). 5% sits well above jitter and
// comfortably below every observed indent; the old 10% was one detector
// wobble away from misreading an indented line as a heading.
const INDENT_FRACTION = 0.05; // left inset > 5% of block width ⇒ indented
const SHORT_RIGHT_FRACTION = 0.15; // right inset > 15% of block width ⇒ paragraphEnd
const GAP_X_MEDIAN_HEIGHT = 0.5; // vertical gap > 0.5× median line height ⇒ break
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
 * consecutive lines whose x-intervals overlap meaningfully. Pairwise
 * (line vs line, not vs an accumulated extent) so a full-width header line
 * at the top of a run can't keep both columns chained into one block.
 */
function columnBlocks(lines: LineBox[]): LineBox[][] {
  const blocks: LineBox[][] = [];
  let run: LineBox[] = [lines[0]];
  let prev = lines[0];
  for (let i = 1; i < lines.length; i++) {
    const l = lines[i];
    const overlap = Math.min(prev.x1, l.x1) - Math.max(prev.x0, l.x0);
    // Zero-width boxes have no horizontal signal — keep them with their
    // neighbours instead of splitting on `overlap <= 0`.
    const overlapsEnough =
      prev.x1 - prev.x0 <= 0 ||
      l.x1 - l.x0 <= 0 ||
      overlap >=
        Math.max(
          BLOCK_OVERLAP_MIN_PX,
          BLOCK_OVERLAP_FRACTION *
            Math.min(prev.x1 - prev.x0, l.x1 - l.x0),
        );
    if (overlapsEnough) {
      run.push(l);
    } else {
      blocks.push(run);
      run = [l];
    }
    prev = l;
  }
  blocks.push(run);
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

  const kinds = block.map((l) =>
    classify(l, leftMargin, rightMargin, blockWidth, medianHeight),
  );

  // Break before line i when any of:
  //   — large vertical gap (paragraphs separated by blank space)
  //   — previous line was a paragraphEnd (short last line, no gap)
  //   — this line is an indent (first-line indent ⇒ paragraph start)
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
      || kinds[i] === "indent"
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

/** Classify a line by its horizontal position within its column block. */
function classify(
  l: LineBox,
  leftMargin: number,
  rightMargin: number,
  blockWidth: number,
  _medianHeight: number,
): LineKind {
  // Degenerate geometry (zero-width block, e.g. all lines share x0/x1):
  // can't reason about alignment → treat as body so we join rather than split.
  if (blockWidth <= 0) return "body";
  const leftInset = l.x0 - leftMargin;
  const rightInset = rightMargin - l.x1;
  // Both sides inset → heading/title. Checked before `indent` so a one-line
  // indented paragraph (short AND indented) still isolates correctly.
  if (leftInset > INDENT_FRACTION * blockWidth && rightInset > SHORT_RIGHT_FRACTION * blockWidth) {
    return "centered";
  }
  if (rightInset > SHORT_RIGHT_FRACTION * blockWidth) return "paragraphEnd";
  if (leftInset > INDENT_FRACTION * blockWidth) return "indent";
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
