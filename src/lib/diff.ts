//! Word-level inline diff for the AI Check's rewrite-mode review.
//!
//! Tokenizes by whitespace boundaries (whitespace kept as its own tokens so
//! spacing renders in place) and computes an LCS-based diff. For unspaced
//! scripts like Burmese a "word" degrades to a whole space-delimited chunk —
//! the diff then flags the chunk, which still localizes the change better
//! than showing whole lines.

/** One rendered piece of a diff: unchanged, removed (old), added (new), or
 *  a marker for elided unchanged text (compact edit view). */
export interface DiffSeg {
  type: "same" | "del" | "add" | "gap";
  text: string;
}

/** Split into word and whitespace tokens, alternating; whitespace tokens
 *  participate in matching so runs of spaces diff like words. */
function tokenize(text: string): string[] {
  return text.split(/(\s+)/).filter((t) => t.length > 0);
}

/**
 * Word-level diff of an old line against its corrected version. Adjacent
 * same-type segments are merged so the result renders compactly. Lines here
 * are single text lines, so the O(n·m) LCS table stays small.
 */
export function diffWords(a: string, b: string): DiffSeg[] {
  const at = tokenize(a);
  const bt = tokenize(b);
  const n = at.length;
  const m = bt.length;

  // dp[i][j] = LCS length of at[i..] vs bt[j..]
  const dp: number[][] = Array.from({ length: n + 1 }, () =>
    new Array<number>(m + 1).fill(0),
  );
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] =
        at[i] === bt[j]
          ? dp[i + 1][j + 1] + 1
          : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const segs: DiffSeg[] = [];
  const push = (type: DiffSeg["type"], text: string) => {
    const last = segs[segs.length - 1];
    if (last && last.type === type) last.text += text;
    else segs.push({ type, text });
  };
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (at[i] === bt[j]) {
      push("same", at[i]);
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      push("del", at[i]);
      i++;
    } else {
      push("add", bt[j]);
      j++;
    }
  }
  while (i < n) push("del", at[i++]);
  while (j < m) push("add", bt[j++]);
  return segs;
}

/**
 * Compact edit view: like {@link diffWords} but drops unchanged text,
 * keeping only the removed/added words. A single "gap" segment ("…") marks
 * each run of skipped non-whitespace content BETWEEN edits (never leading
 * or trailing), so multiple edits on one line read clearly without the
 * unchanged context. Adjacent same-type edits merge into one segment.
 */
export function diffEdits(a: string, b: string): DiffSeg[] {
  const words = diffWords(a, b);
  const out: DiffSeg[] = [];
  for (let i = 0; i < words.length; i++) {
    const seg = words[i];
    if (seg.type === "same") {
      if (!seg.text.trim() || out.length === 0) continue;
      // Mark the elision only when another edit actually follows.
      const hasNext = words.slice(i + 1).some((w) => w.type !== "same");
      if (hasNext && out[out.length - 1].type !== "gap") {
        out.push({ type: "gap", text: "…" });
      }
    } else {
      const last = out[out.length - 1];
      if (last && last.type === seg.type) last.text += seg.text;
      else out.push({ ...seg });
    }
  }
  return out;
}
