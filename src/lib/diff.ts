//! Word-level inline diff for the AI Check's rewrite-mode review.
//!
//! Tokenizes by whitespace boundaries (whitespace kept as its own tokens so
//! spacing renders in place) and computes an LCS-based diff. Tokens that
//! contain Myanmar script — an unspaced script where a whole phrase would
//! be a single token — are further split into base+combining-mark clusters,
//! so a one-syllable correction highlights just that syllable instead of
//! flagging the entire phrase.

/** One rendered piece of a diff: unchanged, removed (old), added (new), or
 *  a marker for elided unchanged text (compact edit view). */
export interface DiffSeg {
  type: "same" | "del" | "add" | "gap";
  text: string;
}

const MYANMAR = /\p{Script=Myanmar}/u;
const IS_MARK = /\p{M}/u;

/** Split a token into clusters — one base char plus its following combining
 *  marks (Burmese vowel/asat signs stack onto their consonant). Keeps every
 *  character: a leading mark (broken OCR) simply opens its own cluster, so
 *  `clusters(t).join("") === t` always holds. */
function clusters(tok: string): string[] {
  const out: string[] = [];
  let cur = "";
  for (const ch of tok) {
    if (cur && IS_MARK.test(ch)) cur += ch;
    else {
      if (cur) out.push(cur);
      cur = ch;
    }
  }
  if (cur) out.push(cur);
  return out;
}

/** Split into word and whitespace tokens, alternating; whitespace tokens
 *  participate in matching so runs of spaces diff like words. Myanmar-
 *  script tokens are cluster-split (see `clusters`) — Burmese has no spaces
 *  between words, so without that split any edit to a phrase would diff the
 *  whole phrase as one del+add pair and the inline diff view would be
 *  useless. Cluster granularity localizes the edit to the changed syllable
 *  while keeping syllable stacks (base + marks) intact. */
function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const tok of text.split(/(\s+)/)) {
    if (!tok) continue;
    if (MYANMAR.test(tok)) out.push(...clusters(tok));
    else out.push(tok);
  }
  return out;
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
