//! Output container types matching kraken's containers.py.

/// A detected text line with baseline and boundary polygon.
#[derive(Debug, Clone)]
pub struct BaselineLine {
    pub id: String,
    /// Baseline polyline: Vec of (x, y) points.
    pub baseline: Vec<(f64, f64)>,
    /// Boundary polygon: Vec of (x, y) points.
    pub boundary: Vec<(f64, f64)>,
    /// Line type tag.
    pub script: String,
    /// Region IDs this line belongs to.
    pub regions: Vec<String>,
}

/// A detected region.
#[derive(Debug, Clone)]
pub struct Region {
    pub id: String,
    /// Region boundary polygon: Vec of (x, y) points.
    pub boundary: Vec<(f64, f64)>,
    /// Region type tag.
    pub region_type: String,
}

/// Full segmentation result.
#[derive(Debug, Clone)]
pub struct Segmentation {
    pub text_direction: String,
    pub lines: Vec<BaselineLine>,
    pub regions: Vec<Region>,
    pub script_detection: bool,
}
