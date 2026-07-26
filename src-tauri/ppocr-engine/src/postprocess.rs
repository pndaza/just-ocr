//! Detector postprocess: PP-OCRv6 DB (differentiable binarization) box
//! extraction. Vendored from ppocr-rs src/ocr.rs (detector subset only).
//!
//! Pipeline: probability map → threshold → connected components →
//! rotated-box fit (PCA) → PaddleOCR DB unclip → score/area gates.

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::collections::VecDeque;

/// Detector input is resized so the longest side is at most this many pixels
/// (unless the image is already smaller). Matches PaddleOCR's defaults.
const DETECTOR_LIMIT_SIDE: f64 = 736.0;
const DEFAULT_DETECTOR_MAX_SIDE: u32 = 736;
/// Hard cap on the detector input's longest side (huge images get scaled down).
const DETECTOR_MAX_SIDE: f64 = 4_000.0;

/// Maps detector-input coordinates back to source-image coordinates. Built by
/// `DetectorInputPlan` and consumed by `extract_detections`.
#[derive(Clone, Copy, Debug)]
pub struct DetectorTransform {
    source_width: u32,
    source_height: u32,
    content_width: u32,
    content_height: u32,
}

impl DetectorTransform {
    pub fn new(
        source_width: u32,
        source_height: u32,
        content_width: u32,
        content_height: u32,
    ) -> Result<Self> {
        if source_width == 0 || source_height == 0 || content_width == 0 || content_height == 0 {
            bail!("detector transform dimensions must be non-zero");
        }
        Ok(Self {
            source_width,
            source_height,
            content_width,
            content_height,
        })
    }

    pub fn content_width(self) -> u32 { self.content_width }
    pub fn content_height(self) -> u32 { self.content_height }

    pub fn map_x_to_source(self, x: f32) -> f32 {
        (x * self.source_width as f32 / self.content_width as f32)
            .clamp(0.0, self.source_width as f32)
    }
    pub fn map_y_to_source(self, y: f32) -> f32 {
        (y * self.source_height as f32 / self.content_height as f32)
            .clamp(0.0, self.source_height as f32)
    }
}

/// Detector input geometry: resized input dims + the source↔input transform.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DetectorInputPlan {
    input_width: usize,
    input_height: usize,
    transform: DetectorTransform,
}

impl DetectorInputPlan {
    pub(crate) fn new(source_width: u32, source_height: u32, max_side: Option<u32>) -> Result<Self> {
        let ratio = match max_side {
            Some(limit) if limit > 0 => {
                (f64::from(limit) / f64::from(source_width.max(source_height))).min(1.0)
            }
            Some(_) => bail!("detector maximum side must be positive"),
            None => default_detector_ratio(source_width, source_height),
        };
        let input_width = aligned_dimension(f64::from(source_width) * ratio)?;
        let input_height = aligned_dimension(f64::from(source_height) * ratio)?;
        Ok(Self {
            input_width: input_width as usize,
            input_height: input_height as usize,
            transform: DetectorTransform::new(source_width, source_height, input_width, input_height)?,
        })
    }

    pub(crate) const fn input_width(self) -> usize { self.input_width }
    pub(crate) const fn input_height(self) -> usize { self.input_height }
    pub(crate) const fn transform(self) -> DetectorTransform { self.transform }

    pub(crate) fn corners(self) -> [Point; 4] {
        [
            Point(0.0, 0.0),
            Point(self.transform.source_width as f32, 0.0),
            Point(self.transform.source_width as f32, self.transform.source_height as f32),
            Point(0.0, self.transform.source_height as f32),
        ]
    }
}

fn default_detector_ratio(width: u32, height: u32) -> f64 {
    let min_side = f64::from(width.min(height));
    let mut ratio = if min_side < DETECTOR_LIMIT_SIDE {
        DETECTOR_LIMIT_SIDE / min_side
    } else {
        1.0
    };
    if f64::from(width.max(height)) * ratio > DETECTOR_MAX_SIDE {
        ratio = DETECTOR_MAX_SIDE / f64::from(width.max(height));
    }
    ratio
}

fn aligned_dimension(value: f64) -> Result<u32> {
    if !value.is_finite() || value <= 0.0 {
        bail!("invalid resized image dimension {value}");
    }
    let units = (value / 32.0).round().max(1.0);
    if units > f64::from(u32::MAX / 32) {
        bail!("resized image dimension {value} is too large");
    }
    Ok(units as u32 * 32)
}

// === vendored verbatim from clones/ppocr-rs/src/ocr.rs lines 222–268, 702–722,
//     723–791, 967–999, 1000–1055, 1056–1129, 1130–1154, 1168–1188, 1208–1229 ===

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Point(pub f32, pub f32);

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub polygon: [Point; 4],
    pub score: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct DetectorPostprocessOptions {
    pub binary_threshold: f32,
    pub box_threshold: f32,
    pub min_area: usize,
    pub unclip_ratio: f32,
    pub max_boxes: usize,
}

impl Default for DetectorPostprocessOptions {
    fn default() -> Self {
        Self {
            binary_threshold: 0.2,
            box_threshold: 0.4,
            min_area: 3,
            unclip_ratio: 1.4,
            max_boxes: 1_000,
        }
    }
}

impl DetectorPostprocessOptions {
    pub fn validate(self) -> Result<()> {
        validate_probability(self.binary_threshold, "detector binary threshold")?;
        validate_probability(self.box_threshold, "detector box threshold")?;
        if self.min_area == 0 {
            bail!("detector minimum area must be at least one pixel");
        }
        if self.max_boxes == 0 {
            bail!("detector maximum box count must be at least one");
        }
        if !self.unclip_ratio.is_finite() || self.unclip_ratio <= 0.0 {
            bail!("detector unclip ratio must be a finite value greater than zero");
        }
        Ok(())
    }
}

pub fn extract_detections(
    values: &[f32],
    shape: &[usize],
    transform: DetectorTransform,
    options: DetectorPostprocessOptions,
) -> Result<Vec<Detection>> {
    options.validate()?;
    let (height, width) = detector_output_shape(shape, values.len())?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("detector output contains non-finite values");
    }

    let content_width = usize::min(transform.content_width() as usize, width);
    let content_height = usize::min(transform.content_height() as usize, height);
    let visited_len = content_width
        .checked_mul(content_height)
        .context("detector content area overflow")?;
    let mut visited = vec![false; visited_len];
    let mut detections = Vec::new();
    for y in 0..content_height {
        for x in 0..content_width {
            let active_index = y * content_width + x;
            if visited[active_index] || values[y * width + x] < options.binary_threshold {
                continue;
            }
            let component = collect_component(
                values,
                width,
                content_width,
                content_height,
                x,
                y,
                options.binary_threshold,
                &mut visited,
            );
            if component.points.len() < options.min_area {
                continue;
            }
            let score = (component.score_sum / component.points.len() as f64) as f32;
            if score < options.box_threshold {
                continue;
            }
            let polygon = fit_rotated_box(&component.points, options.unclip_ratio).map(|point| {
                Point(
                    transform.map_x_to_source(point.0),
                    transform.map_y_to_source(point.1),
                )
            });
            detections.push(Detection { polygon, score });
        }
    }
    sort_detections(&mut detections);
    if detections.len() > options.max_boxes {
        detections.sort_by(|left, right| {
            right.score.total_cmp(&left.score).then_with(|| {
                let left_center = polygon_center(left.polygon);
                let right_center = polygon_center(right.polygon);
                left_center
                    .1
                    .total_cmp(&right_center.1)
                    .then_with(|| left_center.0.total_cmp(&right_center.0))
            })
        });
        detections.truncate(options.max_boxes);
        sort_detections(&mut detections);
    }
    Ok(detections)
}

fn detector_output_shape(shape: &[usize], value_len: usize) -> Result<(usize, usize)> {
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] == 0 || shape[3] == 0 {
        bail!("detector output shape {shape:?}, expected [1, 1, height, width]");
    }
    let expected = shape[2]
        .checked_mul(shape[3])
        .context("detector output shape overflow")?;
    if value_len != expected {
        bail!("detector output has {value_len} values, expected {expected} for shape {shape:?}");
    }
    Ok((shape[2], shape[3]))
}

fn validate_probability(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("{name} must be a finite value between zero and one");
    }
    Ok(())
}

struct Component {
    points: Vec<Point>,
    score_sum: f64,
}

#[allow(clippy::too_many_arguments)]
fn collect_component(
    values: &[f32],
    output_width: usize,
    content_width: usize,
    content_height: usize,
    start_x: usize,
    start_y: usize,
    threshold: f32,
    visited: &mut [bool],
) -> Component {
    const NEIGHBORS: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    let mut queue = VecDeque::new();
    queue.push_back((start_x, start_y));
    visited[start_y * content_width + start_x] = true;
    let mut points = Vec::new();
    let mut score_sum = 0.0;
    while let Some((x, y)) = queue.pop_front() {
        points.push(Point(x as f32 + 0.5, y as f32 + 0.5));
        score_sum += f64::from(values[y * output_width + x]);
        for (offset_x, offset_y) in NEIGHBORS {
            let next_x = x as isize + offset_x;
            let next_y = y as isize + offset_y;
            if next_x < 0
                || next_y < 0
                || next_x >= content_width as isize
                || next_y >= content_height as isize
            {
                continue;
            }
            let next_x = next_x as usize;
            let next_y = next_y as usize;
            let next_index = next_y * content_width + next_x;
            if !visited[next_index] && values[next_y * output_width + next_x] >= threshold {
                visited[next_index] = true;
                queue.push_back((next_x, next_y));
            }
        }
    }
    Component { points, score_sum }
}

fn fit_rotated_box(points: &[Point], unclip_ratio: f32) -> [Point; 4] {
    let count = points.len() as f32;
    let center = Point(
        points.iter().map(|point| point.0).sum::<f32>() / count,
        points.iter().map(|point| point.1).sum::<f32>() / count,
    );
    let (cov_xx, cov_xy, cov_yy) = points.iter().fold((0.0, 0.0, 0.0), |acc, point| {
        let dx = point.0 - center.0;
        let dy = point.1 - center.1;
        (acc.0 + dx * dx, acc.1 + dx * dy, acc.2 + dy * dy)
    });
    let angle = 0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy);
    let mut axis = Point(angle.cos(), angle.sin());
    if axis.0 < 0.0 || (axis.0.abs() < f32::EPSILON && axis.1 < 0.0) {
        axis = Point(-axis.0, -axis.1);
    }
    let normal = Point(-axis.1, axis.0);
    let (min_axis, max_axis, min_normal, max_normal) = points.iter().fold(
        (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ),
        |(min_axis, max_axis, min_normal, max_normal), point| {
            let delta = Point(point.0 - center.0, point.1 - center.1);
            let along_axis = dot(delta, axis);
            let along_normal = dot(delta, normal);
            (
                min_axis.min(along_axis),
                max_axis.max(along_axis),
                min_normal.min(along_normal),
                max_normal.max(along_normal),
            )
        },
    );
    let axis_center = (min_axis + max_axis) * 0.5;
    let normal_center = (min_normal + max_normal) * 0.5;
    // Base extents of the fitted rotated box. The +1.0 accounts for pixel
    // extent, since component points are sampled at pixel centers.
    let width = max_axis - min_axis + 1.0;
    let height = max_normal - min_normal + 1.0;
    // Match PaddleOCR DB postprocess: offset every edge by a fixed distance
    // proportional to text height, not text length. The previous formula scaled
    // each extent by `unclip_ratio`, which over-expanded the long axis
    // (left/right spilling past line ends) and under-expanded vertically.
    //   distance = area * ratio / perimeter = W*H*ratio / (2*(W+H))
    let distance = (width * height * unclip_ratio) / (2.0 * (width + height));
    let half_axis = width * 0.5 + distance;
    let half_normal = height * 0.5 + distance;
    let center = add(
        center,
        add(scale(axis, axis_center), scale(normal, normal_center)),
    );
    [
        add(
            center,
            add(scale(axis, -half_axis), scale(normal, -half_normal)),
        ),
        add(
            center,
            add(scale(axis, half_axis), scale(normal, -half_normal)),
        ),
        add(
            center,
            add(scale(axis, half_axis), scale(normal, half_normal)),
        ),
        add(
            center,
            add(scale(axis, -half_axis), scale(normal, half_normal)),
        ),
    ]
}

fn sort_detections(detections: &mut [Detection]) {
    detections.sort_by(|left, right| {
        let left_center = polygon_center(left.polygon);
        let right_center = polygon_center(right.polygon);
        left_center
            .1
            .total_cmp(&right_center.1)
            .then_with(|| left_center.0.total_cmp(&right_center.0))
            .then_with(|| left.score.total_cmp(&right.score))
    });
}

fn polygon_center(polygon: [Point; 4]) -> Point {
    Point(
        polygon.iter().map(|point| point.0).sum::<f32>() / polygon.len() as f32,
        polygon.iter().map(|point| point.1).sum::<f32>() / polygon.len() as f32,
    )
}

fn polygon_aspect_ratio(polygon: [Point; 4]) -> f32 {
    let width = distance(polygon[0], polygon[1]).max(distance(polygon[3], polygon[2]));
    let height = distance(polygon[0], polygon[3]).max(distance(polygon[1], polygon[2]));
    width / height.max(f32::EPSILON)
}

fn row_probability(row: &[f32], value: f32) -> f32 {
    let sum = row.iter().sum::<f32>();
    if row.iter().all(|candidate| *candidate >= 0.0) && (sum - 1.0).abs() <= 0.01 {
        return value.clamp(0.0, 1.0);
    }
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator = row
        .iter()
        .map(|candidate| (*candidate - maximum).exp())
        .sum::<f32>();
    ((value - maximum).exp() / denominator).clamp(0.0, 1.0)
}

fn argmax(row: &[f32]) -> (usize, f32) {
    row.iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("recognizer output rows are non-empty")
}

fn dot(left: Point, right: Point) -> f32 {
    left.0 * right.0 + left.1 * right.1
}

fn add(left: Point, right: Point) -> Point {
    Point(left.0 + right.0, left.1 + right.1)
}

fn scale(point: Point, factor: f32) -> Point {
    Point(point.0 * factor, point.1 * factor)
}

#[cfg(feature = "gpu")]
fn point_coordinates(point: Point) -> [f32; 2] {
    [point.0, point.1]
}

fn distance(left: Point, right: Point) -> f32 {
    ((left.0 - right.0).powi(2) + (left.1 - right.1).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_aligns_to_32_and_maps_back() {
        // 1920x1080 source → resized with max_side=736.
        let plan = DetectorInputPlan::new(1920, 1080, Some(736)).expect("plan");
        // 1920*736/1920 = 736 (longest), 1080*736/1920 = 414 → round to 416 (32-multiple).
        assert_eq!(plan.input_width(), 736);
        assert_eq!(plan.input_height(), 416);
        let t = plan.transform();
        assert!((t.map_x_to_source(736.0) - 1920.0).abs() < 1e-3);
        assert!((t.map_y_to_source(416.0) - 1080.0).abs() < 1e-3);
    }

    #[test]
    fn plan_rejects_zero_dimensions() {
        assert!(DetectorInputPlan::new(0, 100, Some(736)).is_err());
        assert!(DetectorInputPlan::new(100, 0, Some(736)).is_err());
    }
}
