//! Column-aware reading order over detected text lines.
//!
//! The three segmenter paths order lines differently, and none guarantees
//! column-wise output on multi-column pages: the PP-OCR quad port's
//! `sort_detections` orders by polygon-center y-then-x across the whole
//! page (columns interleave line by line), the poly port emits raster
//! order, and Kraken's `polygonal_reading_order` (partial order +
//! topological sort) already groups well-separated columns but falls back
//! to index order when a full-width header satisfies `separates_any` for
//! every line pair. This pass normalizes all of them: on multi-column
//! pages (textbook spreads like `sample_images/two_colums-1.png`) it
//! recovers column-wise order; when no cut fires it re-sorts row-aware
//! y-then-x — matching the ppocr-quad path's previous behavior on text
//! (single columns are y-ordered under every segmenter) while reading
//! side-by-side elements (header page number beside the author name)
//! left-to-right instead of by a few px of quad skew.
//!
//! PaddleOCR solves this with a separate layout-detection model (PicoDet in
//! legacy PP-Structure, RT-DETR-based PP-DocLayout in 3.x) whose region boxes
//! reorder and group the OCR lines. We deliberately don't ship a layout
//! network; instead this module recovers column structure geometrically, the
//! way legacy PP-Structure's `sorted_layout_boxes` did — but driven by an
//! actual gutter search instead of fixed page-midline heuristics.
//!
//! The algorithm is an XY-cut variant operating on exact interval unions
//! (no pixel binning): at each recursion level over a set of line bboxes,
//!
//! 1. **Vertical cut first**: find the widest x-strip not intersected by any
//!    line bbox. Because line bboxes are rectangles, a gap in the union of
//!    their x-intervals is by construction an empty strip spanning the
//!    region's full height — i.e. a genuine column gutter no line crosses.
//!    Preferring this cut before any horizontal one is what keeps columns
//!    that run the full page height intact (a coincidental paragraph gap
//!    aligned across both columns cannot split them).
//! 2. **Else horizontal cut**: the tallest y-strip not intersected by any
//!    line — a full-width whitespace band. This isolates full-width elements
//!    (titles, section rules, figures) that would otherwise block the
//!    gutter, letting the recursive call below them find their own vertical
//!    cuts. A horizontal cut alone never changes the result versus plain
//!    y-sorting — bands are strictly y-disjoint — it only exists to expose
//!    vertical cuts; the largest gap is taken first so section separators
//!    win over ordinary paragraph spacing.
//! 3. **Else** the region is one block, ordered row-aware: lines whose
//!    vertical extents overlap the row anchor's by more than
//!    [`ROW_OVERLAP_SHARE`] of the shorter height form a row and read
//!    left-to-right; rows read top-to-bottom. Ordinary line spacing stays
//!    under the share, so text keeps plain y-then-x order.
//!
//! Gates keep single-column pages untouched: a column gutter must be at
//! least [`GUTTER_FRAC_OF_LINE_HEIGHT`] × and a band cut at least
//! [`BAND_FRAC_OF_LINE_HEIGHT`] × the region's median line height. The band
//! gate is deliberately taller: inter-line leading and word gaps stay well
//! under half a line height, and a band cut is only ever *needed* to isolate
//! a full-width element (header, rule, figure), which carries at least a
//! full line height of whitespace around it. The taller gate also lowers the
//! odds of a paragraph gap that happens to align across both columns being
//! mistaken for a document-level separator. A vertical cut additionally
//! needs ≥ [`MIN_LINES_PER_COLUMN`] lines on each side. Page margins are
//! ignored automatically: a margin gap has lines on only one side, so it is
//! not an interior cut.
//!
//! Not handled: genuinely nested/irregular layouts (marginalia tied to
//! paragraphs, side-by-side tables) — those need a real layout model. For
//! those pages this degrades to the old y-then-x order. Row-paired layouts
//! (TOC entries + a narrow page-number column) are deliberately left in
//! y-then-x order too — see [`COLUMN_MIN_WIDTH_SHARE`].
//!
//! All functions are pure over their inputs (no globals, no I/O), so the
//! reorder is safe anywhere, including inside rayon workers.

use crate::segmentation::DetectedLine;

/// Minimum column gutter width as a fraction of the region's median line
/// height. Word gaps and inter-line leading stay well under half a line
/// height even for tight Burmese stacks, while real column gutters are
/// typically ≥ 1 em.
const GUTTER_FRAC_OF_LINE_HEIGHT: f64 = 0.5;

/// Minimum full-width band height as a fraction of the median line height.
/// Taller than the gutter gate on purpose — see the module docs: band cuts
/// exist to isolate full-width elements, and ordinary paragraph spacing
/// must not trigger them. Calibrated against real pages (e.g. the
/// `two_colums-1.png` textbook): consecutive line bboxes overlap or nearly
/// touch (gap ≲ 0.2 × height, thanks to ascender/descender padding), while
/// the whitespace under a full-width header/figure block measures ≈ 0.85 ×
/// height — 0.75 sits between the two.
const BAND_FRAC_OF_LINE_HEIGHT: f64 = 0.75;

/// Absolute floor so sub-line-height boxes (thumbnail-scale scans) can't
/// open cuts on rounding-level gaps.
const GAP_MIN_ABS_PX: f64 = 4.0;

/// A vertical cut must leave at least this many lines on each side. A lone
/// short line beside a column (stray artifact, footnote number) should not
/// carve the page in two.
const MIN_LINES_PER_COLUMN: usize = 2;

/// A vertical cut is only taken when both sides look like real text
/// columns, not a narrow companion column (table-of-contents page numbers,
/// marginal numbering): the narrower side's median line width must be at
/// least this share of the wider side's. Row-paired layouts — entry, page
/// number, entry, page number — read correctly in plain y-then-x order; a
/// column cut there would hoist all the numbers to the end.
const COLUMN_MIN_WIDTH_SHARE: f64 = 0.3;

/// Skip the pass entirely below this many lines: no vertical cut is possible
/// (needs ≥ 2 + 2), and horizontal cuts never reorder versus y-sort.
const MIN_LINES_TO_TRY: usize = 4;

/// Axis-aligned bbox of one detected line, in source-image pixel
/// coordinates. `x1`/`y1` are exclusive, matching the width/height form of
/// [`crate::engine::polygon_bbox`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LineRect {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl LineRect {
    #[cfg(test)]
    pub(crate) fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self { x0, y0, x1, y1 }
    }

    fn height(&self) -> f64 {
        self.y1 - self.y0
    }

    fn center_x(&self) -> f64 {
        (self.x0 + self.x1) / 2.0
    }

    fn center_y(&self) -> f64 {
        (self.y0 + self.y1) / 2.0
    }
}

/// How many cuts the ordering applied, for the per-stage timing log.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct CutStats {
    /// Vertical (column) cuts taken.
    pub(crate) columns: usize,
    /// Horizontal (band) cuts taken.
    pub(crate) bands: usize,
}

/// Compute a reading-order permutation over `rects` (one per line). Returns
/// the permutation `out` such that `out[k]` is the index of the k-th line in
/// reading order, plus the cut counts. Pure; deterministic.
pub(crate) fn reading_order(rects: &[LineRect]) -> (Vec<usize>, CutStats) {
    let all: Vec<usize> = (0..rects.len()).collect();
    let mut stats = CutStats::default();
    let ordered = order_region(&all, rects, &mut stats);
    (ordered, stats)
}

/// Reorder detected lines into column-aware reading order and return them as
/// a new vector. Lines whose boundary has no usable bbox (< 2 points or
/// zero-area) are appended at the end in their original order — the
/// recognizer drops them anyway, so their position is cosmetic.
///
/// `dims` is the page size in pixels; it only bounds bboxes, it does not
/// affect the cut logic (which is driven purely by the line geometry).
pub(crate) fn sort_lines(
    lines: Vec<DetectedLine>,
    dims: (u32, u32),
) -> (Vec<DetectedLine>, CutStats) {
    if lines.len() < MIN_LINES_TO_TRY {
        return (lines, CutStats::default());
    }
    let (img_w, img_h) = (dims.0 as f64, dims.1 as f64);
    let rects: Vec<Option<LineRect>> = lines
        .iter()
        .map(|line| line_rect(img_w, img_h, &line.boundary))
        .collect();

    let valid: Vec<usize> = (0..rects.len()).filter(|&i| rects[i].is_some()).collect();
    let flat: Vec<LineRect> = rects.iter().flatten().copied().collect();
    // `flat` is indexed by position within `valid`, not by original line
    // index — keep the mapping around for the rebuild.
    let (perm, stats) = reading_order(&flat);

    let mut out: Vec<DetectedLine> = Vec::with_capacity(lines.len());
    let mut consumed = vec![false; lines.len()];
    for local in perm {
        let original = valid[local];
        out.push(lines[original].clone());
        consumed[original] = true;
    }
    // Degenerate lines keep their original relative order, at the end.
    for (i, line) in lines.iter().enumerate() {
        if !consumed[i] {
            out.push(line.clone());
        }
    }
    (out, stats)
}

/// Bbox of a boundary polygon, clamped to the image. `None` for degenerate
/// boundaries (< 2 points or zero area). Cheaper standalone twin of
/// [`crate::engine::polygon_bbox`] — this module stays independent of the
/// engine so it can be unit-tested without pulling in the OCR pipeline.
fn line_rect(img_w: f64, img_h: f64, boundary: &[(f64, f64)]) -> Option<LineRect> {
    if boundary.len() < 2 {
        return None;
    }
    let mut x0 = f64::INFINITY;
    let mut y0 = f64::INFINITY;
    let mut x1 = f64::NEG_INFINITY;
    let mut y1 = f64::NEG_INFINITY;
    for &(x, y) in boundary {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let x0 = x0.max(0.0).min(img_w);
    let y0 = y0.max(0.0).min(img_h);
    let x1 = x1.max(0.0).min(img_w);
    let y1 = y1.max(0.0).min(img_h);
    if x1 - x0 <= 0.0 || y1 - y0 <= 0.0 {
        return None;
    }
    Some(LineRect { x0, y0, x1, y1 })
}

/// Same text row: vertical interval overlap exceeds this share of the
/// shorter height, measured against the row's ANCHOR (first, topmost line)
/// — not the union extent of the row, which grows as members join and lets
/// a tall quad (rotated or merged detection) overlap two staggered text
/// lines and glue them into one x-sorted row. Side-by-side elements
/// (header page number beside the author name) clear the share against the
/// anchor; consecutive text lines in these pages overlap well under half
/// their height (tight pitch or negative gap), so they stay in y order.
const ROW_OVERLAP_SHARE: f64 = 0.5;

/// Order one region (subset of line indices) recursively.
fn order_region(indices: &[usize], rects: &[LineRect], stats: &mut CutStats) -> Vec<usize> {
    if indices.len() <= 1 {
        return indices.to_vec();
    }

    let median_h = median_height(indices, rects);
    let min_gutter = (GUTTER_FRAC_OF_LINE_HEIGHT * median_h).max(GAP_MIN_ABS_PX);
    let min_band = (BAND_FRAC_OF_LINE_HEIGHT * median_h).max(GAP_MIN_ABS_PX);

    // 1. Column cut: widest interior gap in the union of x-intervals, with
    //    enough lines on both sides.
    let x_intervals: Vec<(f64, f64)> = indices
        .iter()
        .map(|&i| (rects[i].x0, rects[i].x1))
        .collect();
    if let Some(gap) = widest_gap(&x_intervals, min_gutter) {
        let mid = (gap.0 + gap.1) / 2.0;
        let left: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| rects[i].center_x() < mid)
            .collect();
        let right: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| rects[i].center_x() >= mid)
            .collect();
        if left.len() >= MIN_LINES_PER_COLUMN
            && right.len() >= MIN_LINES_PER_COLUMN
            && columns_comparable(&left, &right, rects)
        {
            stats.columns += 1;
            let mut out = order_region(&left, rects, stats);
            out.extend(order_region(&right, rects, stats));
            return out;
        }
    }

    // 2. Band cut: tallest interior gap in the union of y-intervals. Split
    //    point is the gap itself; by construction no line straddles it.
    let y_intervals: Vec<(f64, f64)> = indices
        .iter()
        .map(|&i| (rects[i].y0, rects[i].y1))
        .collect();
    if let Some(gap) = widest_gap(&y_intervals, min_band) {
        let mid = (gap.0 + gap.1) / 2.0;
        let top: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| rects[i].center_y() < mid)
            .collect();
        let bottom: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| rects[i].center_y() >= mid)
            .collect();
        if !top.is_empty() && !bottom.is_empty() {
            stats.bands += 1;
            let mut out = order_region(&top, rects, stats);
            out.extend(order_region(&bottom, rects, stats));
            return out;
        }
    }

    // 3. Leaf region: one block of text, row-aware. Lines whose vertical
    //    extents overlap the row ANCHOR's by more than [`ROW_OVERLAP_SHARE`]
    //    of the shorter height form a row and read left-to-right
    //    (side-by-side header elements); everything else keeps
    //    top-to-bottom y order, matching `sort_detections` so
    //    single-column text pages are unchanged. Judging against the
    //    anchor (not a grown union extent) keeps a tall bridging quad from
    //    swallowing the text rows above and below it.
    let mut sorted = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        rects[a]
            .y0
            .total_cmp(&rects[b].y0)
            .then_with(|| rects[a].center_y().total_cmp(&rects[b].center_y()))
            .then_with(|| rects[a].center_x().total_cmp(&rects[b].center_x()))
    });
    let mut out: Vec<usize> = Vec::with_capacity(sorted.len());
    let mut row: Vec<usize> = Vec::with_capacity(sorted.len());
    let mut anchor_y0: f64 = 0.0;
    let mut anchor_y1: f64 = 0.0;
    for &i in &sorted {
        let r = rects[i];
        let joins = !row.is_empty() && {
            let overlap = anchor_y1.min(r.y1) - anchor_y0.max(r.y0);
            let shorter = (anchor_y1 - anchor_y0).min(r.y1 - r.y0);
            overlap > ROW_OVERLAP_SHARE * shorter
        };
        if joins {
            row.push(i);
        } else {
            emit_row(&mut out, &mut row, rects);
            row.push(i);
            anchor_y0 = r.y0;
            anchor_y1 = r.y1;
        }
    }
    emit_row(&mut out, &mut row, rects);
    out
}

/// Sort a completed row by center x and append it to the output. Empty rows
/// (region start) are a no-op.
fn emit_row(out: &mut Vec<usize>, row: &mut Vec<usize>, rects: &[LineRect]) {
    if row.is_empty() {
        return;
    }
    row.sort_by(|&a, &b| rects[a].center_x().total_cmp(&rects[b].center_x()));
    out.append(row);
}

/// The widest interior gap of at least `min_width` in the union of closed
/// intervals. Gaps before the first run / after the last run (page margins)
/// are not interior and never returned.
fn widest_gap(intervals: &[(f64, f64)], min_width: f64) -> Option<(f64, f64)> {
    let mut sorted = intervals.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    let mut best: Option<(f64, f64)> = None;
    let mut run_end = sorted[0].1;
    for &(start, end) in &sorted[1..] {
        if start > run_end {
            let width = start - run_end;
            if width >= min_width && width > best.map(|g| g.1 - g.0).unwrap_or(0.0) {
                best = Some((run_end, start));
            }
        }
        run_end = run_end.max(end);
    }
    best
}

/// Median bbox height over the region's lines, the scale all gap gates are
/// derived from. Median (not mean) so one giant figure bbox can't inflate
/// the gates and silence legitimate cuts.
fn median_height(indices: &[usize], rects: &[LineRect]) -> f64 {
    let mut heights: Vec<f64> = indices.iter().map(|&i| rects[i].height()).collect();
    heights.sort_by(|a, b| a.total_cmp(b));
    let mid = heights.len() / 2;
    if heights.len() % 2 == 1 {
        heights[mid]
    } else {
        (heights[mid - 1] + heights[mid]) / 2.0
    }
}

/// True when the two sides of a would-be column cut have comparable line
/// widths (see [`COLUMN_MIN_WIDTH_SHARE`]). MEDIAN width per side so a few
/// short paragraph tails can't flunk the gate for a genuine column.
fn columns_comparable(a: &[usize], b: &[usize], rects: &[LineRect]) -> bool {
    let widths = |idx: &[usize]| {
        let mut w: Vec<f64> = idx.iter().map(|&i| rects[i].x1 - rects[i].x0).collect();
        w.sort_by(|x, y| x.total_cmp(y));
        w[w.len() / 2]
    };
    let (wa, wb) = (widths(a), widths(b));
    let (narrow, wide) = if wa <= wb { (wa, wb) } else { (wb, wa) };
    narrow >= COLUMN_MIN_WIDTH_SHARE * wide
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmentation::Segmenter;
    use image::GenericImageView;

    /// A left-aligned text line of standard height 20 at (x, y).
    fn line(x: f64, y: f64, w: f64) -> LineRect {
        LineRect::new(x, y, x + w, y + 20.0)
    }

    /// `reading_order` mapped back to the x-position of each line, so tests
    /// read as the actual sequence a user would see.
    fn order_xs(rects: &[LineRect]) -> Vec<f64> {
        let (perm, _) = reading_order(rects);
        perm.into_iter().map(|i| rects[i].x0).collect()
    }

    #[test]
    fn two_columns_read_column_wise() {
        // Left column x=40, right column x=560, gutter 80px wide
        // (520..600), median line height 20 → gate 10px. Lines arrive
        // y-interleaved, exactly how the segmenters emit them.
        let mut rects = Vec::new();
        for row in 0..6 {
            let y = 100.0 + row as f64 * 30.0;
            rects.push(line(40.0, y, 460.0)); // left column
            rects.push(line(600.0, y, 460.0)); // right column
        }
        let xs = order_xs(&rects);
        let expected: Vec<f64> = std::iter::repeat_n(40.0, 6)
            .chain(std::iter::repeat_n(600.0, 6))
            .collect();
        assert_eq!(xs, expected);
        let (_, stats) = reading_order(&rects);
        assert_eq!(stats.columns, 1);
    }

    #[test]
    fn header_then_two_columns() {
        // Full-width header crossing the gutter blocks the top-level
        // vertical cut; the band cut below it exposes the columns.
        let mut rects = vec![line(40.0, 40.0, 1020.0)];
        for row in 0..4 {
            let y = 160.0 + row as f64 * 30.0;
            rects.push(line(40.0, y, 460.0));
            rects.push(line(600.0, y, 460.0));
        }
        let xs = order_xs(&rects);
        let expected: Vec<f64> = vec![40.0]
            .into_iter()
            .chain(std::iter::repeat_n(40.0, 4))
            .chain(std::iter::repeat_n(600.0, 4))
            .collect();
        assert_eq!(xs, expected);
        let (_, stats) = reading_order(&rects);
        assert_eq!(stats.columns, 1);
        assert!(stats.bands >= 1);
    }

    #[test]
    fn full_width_figure_splits_column_bands() {
        // Header, two columns, a full-width figure, then two more column
        // lines. Reading order must be: header, left-top, right-top,
        // figure, left-bottom, right-bottom.
        let mut rects = vec![line(40.0, 40.0, 1020.0)];
        for row in 0..3 {
            let y = 160.0 + row as f64 * 30.0;
            rects.push(line(40.0, y, 460.0));
            rects.push(line(600.0, y, 460.0));
        }
        rects.push(line(60.0, 300.0, 980.0)); // figure caption band
        for row in 0..3 {
            let y = 400.0 + row as f64 * 30.0;
            rects.push(line(40.0, y, 460.0));
            rects.push(line(600.0, y, 460.0));
        }
        let xs = order_xs(&rects);
        assert_eq!(
            xs,
            vec![
                40.0, // header
                40.0, 40.0, 40.0, // left column, above figure
                600.0, 600.0, 600.0, // right column, above figure
                60.0,  // figure band
                40.0, 40.0, 40.0, // left column, below figure
                600.0, 600.0, 600.0, // right column, below figure
            ]
        );
    }

    #[test]
    fn single_column_matches_y_then_x_sort() {
        // Staggered line starts (paragraph indents) must NOT open a false
        // gutter: every long line spans the would-be gap x-range.
        let mut rects = Vec::new();
        for row in 0..8 {
            let indent = if row % 3 == 0 { 80.0 } else { 40.0 };
            rects.push(line(indent, 100.0 + row as f64 * 30.0, 900.0));
        }
        let (perm, stats) = reading_order(&rects);
        assert_eq!(stats, CutStats::default(), "no cuts on a single column");
        // Already y-sorted input → identity permutation.
        assert_eq!(perm, (0..rects.len()).collect::<Vec<_>>());
    }

    #[test]
    fn margins_never_split() {
        // All lines in the left 60% of the page: the empty right margin is
        // not an interior gap, so nothing is cut.
        let rects: Vec<LineRect> = (0..6)
            .map(|row| line(40.0, 100.0 + row as f64 * 30.0, 500.0))
            .collect();
        let (_, stats) = reading_order(&rects);
        assert_eq!(stats, CutStats::default());
    }

    #[test]
    fn narrow_centered_title_does_not_split_body() {
        // A centered short title over full-width body text: the body's long
        // lines cover any candidate gutter, so the only cut is the band
        // under the title — order stays title, then body in y order.
        let mut rects = vec![line(400.0, 40.0, 240.0)];
        for row in 0..6 {
            rects.push(line(40.0, 160.0 + row as f64 * 30.0, 960.0));
        }
        let xs = order_xs(&rects);
        let expected: Vec<f64> = vec![400.0]
            .into_iter()
            .chain(std::iter::repeat_n(40.0, 6))
            .collect();
        assert_eq!(xs, expected);
        let (_, stats) = reading_order(&rects);
        assert_eq!(stats.columns, 0);
        assert_eq!(stats.bands, 1, "the band under the title is the only cut");
    }

    #[test]
    fn three_columns_left_to_right() {
        let mut rects = Vec::new();
        for row in 0..4 {
            let y = 100.0 + row as f64 * 30.0;
            rects.push(line(40.0, y, 240.0));
            rects.push(line(380.0, y, 240.0));
            rects.push(line(720.0, y, 240.0));
        }
        let xs = order_xs(&rects);
        let expected: Vec<f64> = std::iter::repeat_n(40.0, 4)
            .chain(std::iter::repeat_n(380.0, 4))
            .chain(std::iter::repeat_n(720.0, 4))
            .collect();
        assert_eq!(xs, expected);
    }

    #[test]
    fn thin_gutter_is_ignored() {
        // Two line groups separated by an x-gap below the gate (10px gate at
        // median height 20, gap here is 8px): must behave as one y-sorted
        // block.
        let mut rects = Vec::new();
        for row in 0..4 {
            let y = 100.0 + row as f64 * 30.0;
            rects.push(line(40.0, y, 400.0));
            rects.push(line(448.0, y + 5.0, 400.0)); // 8px gap < 10px gate
        }
        let (_, stats) = reading_order(&rects);
        assert_eq!(stats, CutStats::default(), "sub-gate gutter must not cut");
        // And the order is plain y-then-x on centers.
        let xs = order_xs(&rects);
        assert_eq!(xs[0], 40.0); // row 0 left (y=100) before right (y=105)
    }

    #[test]
    fn sort_lines_moves_degenerate_boundaries_last() {
        let mk = |x: f64, y: f64| DetectedLine {
            baseline: vec![(x, y + 10.0), (x + 100.0, y + 10.0)],
            boundary: vec![
                (x, y),
                (x + 100.0, y),
                (x + 100.0, y + 20.0),
                (x, y + 20.0),
                (x, y),
            ],
            quad: None,
        };
        let mut lines = Vec::new();
        for row in 0..3 {
            let y = 100.0 + row as f64 * 30.0;
            lines.push(mk(40.0, y));
            lines.push(mk(600.0, y));
        }
        lines.push(DetectedLine {
            baseline: vec![],
            boundary: vec![(5.0, 5.0)], // degenerate: 1 point
            quad: None,
        });

        let (ordered, stats) = sort_lines(lines, (1200, 800));
        assert_eq!(stats.columns, 1);
        assert_eq!(ordered.len(), 7);
        // First six: left column then right column; the degenerate line
        // (single-point boundary) keeps its trailing position.
        assert!(ordered[0].boundary[0].0 == 40.0);
        assert!(ordered[5].boundary[0].0 == 600.0);
        assert_eq!(ordered[6].boundary.len(), 1);
    }

    #[test]
    fn same_row_side_by_side_reads_left_to_right() {
        // thawzin header, real geometry: the author quad (right) starts
        // 11px higher than the page-number quad (left) — a plain y-then-x
        // sort reads "author, page number". Substantial y-overlap makes
        // them one row, so x wins: page number first.
        let page_number = LineRect::new(72.0, 49.0, 127.0, 105.0);
        let author = LineRect::new(762.0, 38.0, 913.0, 104.0);
        let mut rects = vec![author, page_number]; // author arrives first
        for row in 0..6 {
            let y = 200.0 + row as f64 * 30.0;
            rects.push(line(80.0, y, 800.0));
        }
        let xs = order_xs(&rects);
        assert_eq!(
            xs,
            vec![72.0, 762.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            "header row reads page number (left) then author (right)"
        );
    }

    #[test]
    fn consecutive_text_lines_stay_in_y_order() {
        // Line pitch 30 with height 20 → zero vertical overlap between
        // consecutive lines: every line is its own row, y order preserved
        // even when a later line starts further left (x would scramble it).
        let mut rects = Vec::new();
        for row in 0..5 {
            let x = 500.0 - row as f64 * 80.0; // staircase leftward
            rects.push(line(x, 100.0 + row as f64 * 30.0, 100.0));
        }
        let xs = order_xs(&rects);
        assert_eq!(xs, vec![500.0, 420.0, 340.0, 260.0, 180.0]);
    }

    #[test]
    fn tight_pitch_staircase_stays_in_y_order() {
        // Consecutive bboxes overlap ~35% of their height (pitch 13 vs
        // height 20) while drifting left — under the 50% row share, so each
        // line keeps its own row and the x-drift never reorders them. Pins
        // the margin the real pages run at (~25% overlap).
        let mut rects = Vec::new();
        for row in 0..5 {
            let x = 500.0 - row as f64 * 60.0;
            rects.push(line(x, 100.0 + row as f64 * 13.0, 100.0));
        }
        let xs = order_xs(&rects);
        assert_eq!(xs, vec![500.0, 440.0, 380.0, 320.0, 260.0]);
    }

    #[test]
    fn tall_bridge_does_not_glue_staggered_rows() {
        // A tall quad (rotated or merged detection, y 0..52) overlapping two
        // text lines (y 0..22 and 30..52) that barely overlap each other.
        // Judged against the row's grown union extent, the second text line
        // would join the first's row and x-sort ABOVE it; judged against
        // the row ANCHOR, it starts its own row and y order survives.
        let t1 = line(200.0, 0.0, 300.0); // x [200,500], y [0,22]
        let v = LineRect::new(600.0, 0.0, 650.0, 52.0); // tall box, y [0,52]
        let t2 = line(40.0, 30.0, 300.0); // x [40,340], y [30,52]
        let xs = order_xs(&[t1, v, t2]);
        assert_eq!(xs, vec![200.0, 600.0, 40.0], "T1, V, T2 — texts keep y order");
    }

    #[test]
    fn row_paired_toc_stays_row_wise() {
        // TOC layout: entry [40,500] + page number [900,930] per row. The
        // numbers form a "column" with a full-height gutter, but the sides'
        // median line widths (420 vs 30) are far apart — a column cut would
        // hoist every page number to the end. Row-wise y-then-x is correct.
        let mut rects = Vec::new();
        for row in 0..6 {
            let y = 100.0 + row as f64 * 30.0;
            rects.push(line(40.0, y, 460.0)); // entry
            rects.push(line(900.0, y, 30.0)); // page number
        }
        let (perm, stats) = reading_order(&rects);
        assert_eq!(stats, CutStats::default(), "narrow companion column must not cut");
        // Already y-sorted input → identity (entry, number, entry, number…).
        assert_eq!(perm, (0..rects.len()).collect::<Vec<_>>());
    }

    #[test]
    fn short_input_is_untouched() {
        let lines = vec![
            DetectedLine {
                baseline: vec![],
                boundary: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                quad: None,
            },
            DetectedLine {
                baseline: vec![],
                boundary: vec![(0.0, 20.0), (10.0, 20.0), (10.0, 30.0), (0.0, 30.0)],
                quad: None,
            },
        ];
        let (ordered, stats) = sort_lines(lines.clone(), (100, 100));
        assert_eq!(ordered.len(), 2);
        assert_eq!(stats, CutStats::default());
        assert_eq!(ordered[0].boundary[0].1, 0.0);
    }

    /// End-to-end check on the real two-column textbook page that motivated
    /// this module: run the bundled PP-OCR small-det over
    /// `sample_images/two_colums-1.png`, reorder, and verify the reading
    /// order is column-wise instead of row-interleaved. Skips silently when
    /// the sample dir isn't present — note `sample_images/` is gitignored
    /// and CI runs no `cargo test`, so in practice this exercises only on
    /// dev machines that have the sample checked out.
    #[test]
    fn two_column_sample_page_reads_column_wise() {
        // Same include pattern as `engine.rs`'s BUNDLED_PPOCR_DET_SMALL;
        // path relative to this file (`src-tauri/src/`).
        const SMALL_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");
        let img = match image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sample_images/two_colums-1.png"
        )) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("skipping: sample image not available ({e})");
                return;
            }
        };
        let (w, h) = img.dimensions();

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let det = ppocr_engine::Detector::load_from_buffer_with_config(
            SMALL_DET,
            threads,
            ppocr_engine::DetectorConfig::small(),
        )
        .expect("load bundled small-det");
        let segmenter = crate::segmenter_adapters::PPOcrSegmenter::new(std::sync::Arc::new(det));
        let lines = segmenter.segment(&img).expect("segment sample page");

        let rects: Vec<LineRect> = lines
            .iter()
            .filter_map(|l| line_rect(w as f64, h as f64, &l.boundary))
            .collect();
        let (perm, stats) = reading_order(&rects);
        eprintln!(
            "sample page: {} lines, {} column cuts, {} band cuts",
            lines.len(),
            stats.columns,
            stats.bands
        );
        assert!(stats.columns >= 1, "expected a column gutter on this page");

        // Column-wise: each column forms one large contiguous block, and the
        // left block ends before the right block starts. A small number of
        // band lines (the header row above the columns, whose elements sit
        // on either side of the midline) may precede the blocks.
        let mid = w as f64 / 2.0;
        let side: Vec<bool> = perm.iter().map(|&i| rects[i].center_x() >= mid).collect();
        let (mut left_run, mut right_run) = ((0usize, 0usize), (0usize, 0usize)); // (len, end_idx)
        let mut run_len = 0;
        for (i, &right) in side.iter().enumerate() {
            run_len = if right { 0 } else { run_len + 1 };
            if run_len > left_run.0 {
                left_run = (run_len, i);
            }
        }
        run_len = 0;
        for (i, &right) in side.iter().enumerate() {
            run_len = if right { run_len + 1 } else { 0 };
            if run_len > right_run.0 {
                right_run = (run_len, i);
            }
        }
        assert!(
            left_run.0 >= perm.len() / 3,
            "left column block too small: {} of {}",
            left_run.0,
            perm.len()
        );
        assert!(
            right_run.0 >= perm.len() / 3,
            "right column block too small: {} of {}",
            right_run.0,
            perm.len()
        );
        assert!(
            left_run.1 < right_run.1 - right_run.0 + 1,
            "left column block ends after right column block starts (interleaved)"
        );
    }

    /// The header of thawzin_02: page number (narrow, left) beside the
    /// author (wider, right). The author's quad starts ~11px higher, so the
    /// old y-then-x leaf sort emitted "author, page number". Same skip
    /// conditions as the two-column e2e above (dev machines only).
    #[test]
    fn thawzin_header_reads_page_number_then_author() {
        const SMALL_DET: &[u8] = include_bytes!("../../ppocr-models/small-det.safetensors");
        let img = match image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sample_images/thawzin_02.png"
        )) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("skipping: sample image not available ({e})");
                return;
            }
        };
        let (w, h) = img.dimensions();
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let det = ppocr_engine::Detector::load_from_buffer_with_config(
            SMALL_DET,
            threads,
            ppocr_engine::DetectorConfig::small(),
        )
        .expect("load bundled small-det");
        let segmenter = crate::segmenter_adapters::PPOcrSegmenter::new(std::sync::Arc::new(det));
        let lines = segmenter.segment(&img).expect("segment sample page");
        let (ordered, _) = sort_lines(lines, (w, h));
        assert!(ordered.len() > 10, "expected a full page of lines");

        let bb = |l: &crate::segmentation::DetectedLine| {
            line_rect(w as f64, h as f64, &l.boundary).expect("valid boundary")
        };
        let first = bb(&ordered[0]);
        let second = bb(&ordered[1]);
        assert!(
            first.y1 < 150.0 && second.y1 < 150.0,
            "first two lines should be the header row, got y1={} / {}",
            first.y1,
            second.y1
        );
        assert!(
            first.x0 < second.x0,
            "page number (left) must precede author (right): x0={} vs {}",
            first.x0,
            second.x0
        );
        assert!(
            first.x1 - first.x0 < second.x1 - second.x0,
            "the page number is the narrow quad"
        );
    }
}
