import { diffWords, diffEdits } from "./diff";
import { describe, it, expect } from "vitest";

describe("diffWords", () => {
  it("returns a single same segment for identical lines", () => {
    expect(diffWords("hello world", "hello world")).toEqual([
      { type: "same", text: "hello world" },
    ]);
  });

  it("marks a substituted word as del followed by add", () => {
    expect(diffWords("the cat sat", "the dog sat")).toEqual([
      { type: "same", text: "the " },
      { type: "del", text: "cat" },
      { type: "add", text: "dog" },
      { type: "same", text: " sat" },
    ]);
  });

  it("marks pure insertions and deletions", () => {
    // The whitespace token joins the changed run (" c"), not the same run.
    expect(diffWords("a b", "a b c")).toEqual([
      { type: "same", text: "a b" },
      { type: "add", text: " c" },
    ]);
    expect(diffWords("a b c", "a b")).toEqual([
      { type: "same", text: "a b" },
      { type: "del", text: " c" },
    ]);
  });

  it("handles a fully changed line", () => {
    expect(diffWords("foo", "bar")).toEqual([
      { type: "del", text: "foo" },
      { type: "add", text: "bar" },
    ]);
  });

  it("diffs Burmese chunks space-delimited", () => {
    // No intra-word splitting: the changed chunk is flagged whole.
    const segs = diffWords("အ တစ် နှစ်", "အ တစ် သုံး");
    expect(segs).toContainEqual({ type: "del", text: "နှစ်" });
    expect(segs).toContainEqual({ type: "add", text: "သုံး" });
  });

  it("merges adjacent same-type segments", () => {
    const segs = diffWords("a b c d", "a x c y");
    // "a " same, "b" del, "x" add, " c " same, "d" del, "y" add — the " c "
    // whitespace joins the same-run rather than emitting two segments.
    expect(segs.map((s) => s.type)).toEqual(["same", "del", "add", "same", "del", "add"]);
  });
});

describe("diffEdits — compact edit view", () => {
  it("drops unchanged context, keeping only the edited words", () => {
    expect(diffEdits("the cat sat quietly", "the dog sat quietly")).toEqual([
      { type: "del", text: "cat" },
      { type: "add", text: "dog" },
    ]);
  });

  it("marks skipped text between separate edits with a gap", () => {
    expect(diffEdits("a b c d e", "a x c d y")).toEqual([
      { type: "del", text: "b" },
      { type: "add", text: "x" },
      { type: "gap", text: "…" },
      { type: "del", text: "e" },
      { type: "add", text: "y" },
    ]);
  });

  it("never emits leading or trailing gaps", () => {
    expect(diffEdits("start middle", "start middle end")).toEqual([
      { type: "add", text: " end" },
    ]);
    expect(diffEdits("head tail", "tail")).toEqual([
      { type: "del", text: "head " },
    ]);
  });
});
