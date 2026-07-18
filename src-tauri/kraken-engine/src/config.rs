//! Segmentation inference configuration.

#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    /// Text direction hint: "horizontal-lr", "horizontal-rl", "vertical-lr", "vertical-rl"
    pub text_direction: String,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            text_direction: "horizontal-lr".to_string(),
        }
    }
}
