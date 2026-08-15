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
   *  the per-line bboxes (see {@link groupParagraphs}). Lines within a
   *  paragraph join with a single space; paragraphs join with "\n\n".
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
// bboxes every OCR path already produces. A line is classified as one of:
//
//   body        — reaches the dominant left margin and the dominant right
//                 margin. Mid-paragraph wrap.
//   paragraphEnd— reaches the left margin but ends well short of the right
//                 margin. Almost certainly the last line of a paragraph (the
//                 only full-width line that ends short is a paragraph's tail).
//   centered    — inset on BOTH sides. Headings/titles/captions, not body.
//                 Each centered run is its own block; this also keeps a
//                 heading's naturally-short right edge from being read as a
//                 false paragraph-end.
//
// A paragraph break is inserted after line N when ANY of:
//   1. The vertical gap to line N+1 exceeds ~0.5× the median line height
//      (paragraphs separated by blank space).
//   2. Line N is a paragraphEnd (tight/justified text with no inter-paragraph
//      gap — the case pure gap detection misses).
//   3. Line N+1 is centered (transition body → heading, or heading → heading
//      with different centering) — symmetric with rule 2 for the start side.
//
// Known limitation: a paragraph whose last line coincidentally fills the
// full width with no following gap is indistinguishable from a mid-paragraph
// wrap and won't be split. The merge-paragraphs toggle is the escape hatch
// for documents where that matters.

/** A line classified by its horizontal alignment within the text block. */
type LineKind = "body" | "paragraphEnd" | "centered";

const LEFT_MARGIN_PCT = 10; // low percentile of x0 → dominant left margin
const RIGHT_MARGIN_PCT = 90; // high percentile of x1 → dominant right margin
const INDENT_FRACTION = 0.1; // left inset > 10% of block width ⇒ centered
const SHORT_RIGHT_FRACTION = 0.15; // right inset > 15% of block width ⇒ paragraphEnd
const GAP_X_MEDIAN_HEIGHT = 0.5; // vertical gap > 0.5× median line height ⇒ break

/** Group sorted-by-y0 lines into paragraphs. Exported for unit tests. */
export function groupParagraphs(lines: LineBox[]): LineBox[][] {
  const sorted = [...lines].sort((a, b) => a.y0 - b.y0);
  if (sorted.length === 0) return [];
  if (sorted.length === 1) return [sorted];

  const leftMargin = percentile(sorted.map((l) => l.x0), LEFT_MARGIN_PCT);
  const rightMargin = percentile(sorted.map((l) => l.x1), RIGHT_MARGIN_PCT);
  const blockWidth = rightMargin - leftMargin;
  const heights = sorted
    .map((l) => Math.max(1, l.y1 - l.y0))
    .sort((a, b) => a - b);
  const medianHeight = heights[Math.floor(heights.length / 2)];

  const kinds = sorted.map((l) =>
    classify(l, leftMargin, rightMargin, blockWidth, medianHeight),
  );

  // Break before line i when any of:
  //   — large vertical gap (paragraphs separated by blank space)
  //   — previous line was a paragraphEnd (short last line, no gap)
  //   — entering or leaving a centered run (heading ↔ body transition);
  //     consecutive centered lines stay together within their run.
  const paragraphs: LineBox[][] = [];
  let current: LineBox[] = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1];
    const cur = sorted[i];
    const gap = cur.y0 - prev.y1;
    const enteringCentered = kinds[i] === "centered" && kinds[i - 1] !== "centered";
    const leavingCentered = kinds[i - 1] === "centered" && kinds[i] !== "centered";
    const breakBefore = gap > GAP_X_MEDIAN_HEIGHT * medianHeight
      || kinds[i - 1] === "paragraphEnd"
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

/** Classify a line by its horizontal position within the text block. */
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
  if (leftInset > INDENT_FRACTION * blockWidth) return "centered";
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
