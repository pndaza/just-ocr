//! Parse hOCR output into word-level bounding boxes for overlay rendering.
//!
//! hOCR is XHTML where each page is `<div class='ocr_page' title='bbox 0 0 W H'>`
//! and each word is `<span class='ocrx_word' title='bbox x0 y0 x1 y1; x_wconf N'>`.
//! Coordinates are in the source image's natural pixel space.

export interface HocrBox {
  text: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  conf: number;
}

export interface HocrParseResult {
  width: number;
  height: number;
  boxes: HocrBox[];
}

/** Extract the first `bbox x0 y0 x1 y1` (and optional `x_wconf N`) from an
 * hOCR element's `title` attribute. Returns nulls when absent. */
function parseTitle(title: string): {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  conf: number;
} {
  const out = { x0: 0, y0: 0, x1: 0, y1: 0, conf: -1 };
  const bbox = title.match(/bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)/);
  if (bbox) {
    out.x0 = +bbox[1];
    out.y0 = +bbox[2];
    out.x1 = +bbox[3];
    out.y1 = +bbox[4];
  }
  const conf = title.match(/x_wconf\s+(\d+)/);
  if (conf) out.conf = +conf[1];
  return out;
}

/** Parse hOCR markup into page dimensions + word boxes. Defensive: on any
 * failure returns width/height 0 and an empty box list. */
export function parseHocrWords(hocr: string): HocrParseResult {
  const empty: HocrParseResult = { width: 0, height: 0, boxes: [] };
  try {
    const doc = new DOMParser().parseFromString(hocr, "text/html");
    const page = doc.querySelector(".ocr_page");
    if (!page) return empty;
    const pageTitle = page.getAttribute("title") || "";
    const pg = parseTitle(pageTitle);
    if (pg.x1 === 0 || pg.y1 === 0) return empty;

    const boxes: HocrBox[] = [];
    for (const el of doc.querySelectorAll(".ocrx_word")) {
      const t = parseTitle(el.getAttribute("title") || "");
      if (t.x1 <= t.x0 || t.y1 <= t.y0) continue;
      boxes.push({
        text: (el.textContent || "").trim(),
        x0: t.x0,
        y0: t.y0,
        x1: t.x1,
        y1: t.y1,
        conf: t.conf,
      });
    }
    return { width: pg.x1, height: pg.y1, boxes };
  } catch {
    return empty;
  }
}
