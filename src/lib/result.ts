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
}

export interface OcrResult {
  width: number;
  height: number;
  lines: LineBox[];
  /** Mean recognizer confidence in [0,100]; -1 when unknown. */
  confidence: number;
  elapsedMs: number;
}

/** The line boxes for the preview overlay. */
export function lineBoxes(result: OcrResult): LineBox[] {
  return result.lines;
}

/** Join line text with "\n" for display/export. */
export function plainText(result: OcrResult): string {
  return result.lines.map((l) => l.text).join("\n");
}
