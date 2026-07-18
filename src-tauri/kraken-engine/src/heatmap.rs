//! Heatmap data contract between neural and geometric halves.

use ndarray::Array3;

/// The probability heatmap output: (N, H, W) per-class probabilities in [0,1].
#[derive(Debug)]
pub struct Heatmap {
    /// Probability array, shape (N, H, W).
    pub probs: Array3<f32>,
    /// Class mapping (same as model meta).
    pub cls_map: crate::model_meta::ClassMapping,
    /// Scale factor (orig_w / heatmap_w, orig_h / heatmap_h) to map back to original coords.
    pub scale: (f64, f64),
    /// Grayscale scaled image (H, W) used by the seam-carver.
    pub scal_im: ndarray::Array2<f32>,
}

// Re-export ndarray Array2 for convenience.
pub use ndarray::Array2;
