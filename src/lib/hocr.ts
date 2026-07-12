//! Parse hOCR output into line-level bounding boxes for overlay rendering.
//!
//! hOCR is XHTML where each page is `<div class='ocr_page' title='bbox 0 0 W H'>`
//! and each line is `<span class='ocr_line' title='bbox x0 y0 x1 y1'>`.
//! Coordinates are in the source image's natural pixel space.

export interface HocrBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface HocrParseResult {
  width: number;
  height: number;
  boxes: HocrBox[];
  /// Plain UTF-8 text reconstructed from the word spans, one line per
  /// `.ocr_line` (joined with "\n"). Used to display `text`-mode jobs whose
  /// stored `text` is actually hOCR XML.
  text: string;
}

/** Extract the first `bbox x0 y0 x1 y1` from an hOCR element's `title`
 * attribute. */
function parseTitle(title: string): {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
} {
  const out = { x0: 0, y0: 0, x1: 0, y1: 0 };
  const bbox = title.match(/bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)/);
  if (bbox) {
    out.x0 = +bbox[1];
    out.y0 = +bbox[2];
    out.x1 = +bbox[3];
    out.y1 = +bbox[4];
  }
  return out;
}

/** Parse hOCR markup into page dimensions + line boxes + plain text.
 * Defensive: on any failure returns width/height 0, an empty box list, and
 * an empty text string. */
export function parseHocrLines(hocr: string): HocrParseResult {
  const empty: HocrParseResult = { width: 0, height: 0, boxes: [], text: "" };
  try {
    const doc = new DOMParser().parseFromString(hocr, "text/html");
    const page = doc.querySelector(".ocr_page");
    if (!page) return empty;
    const pg = parseTitle(page.getAttribute("title") || "");
    if (pg.x1 === 0 || pg.y1 === 0) return empty;

    const boxes: HocrBox[] = [];
    const lines: string[] = [];
    for (const el of doc.querySelectorAll(".ocr_line")) {
      const t = parseTitle(el.getAttribute("title") || "");
      if (t.x1 > t.x0 && t.y1 > t.y0) boxes.push(t);

      // Reconstruct the line's text from its word spans; fall back to the raw
      // line text if no word spans are present.
      const words = el.querySelectorAll(".ocrx_word");
      let line: string;
      if (words.length) {
        line = Array.from(words)
          .map((w) => w.textContent?.trim() ?? "")
          .filter((w) => w.length)
          .join(" ");
      } else {
        line = (el.textContent ?? "").trim();
      }
      lines.push(line);
    }
    return { width: pg.x1, height: pg.y1, boxes, text: lines.join("\n") };
  } catch {
    return empty;
  }
}

