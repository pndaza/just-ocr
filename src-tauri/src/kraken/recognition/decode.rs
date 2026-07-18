//! Greedy (best-path) CTC decoder.
//!
//! Port of kraken's `kraken/lib/ctc_decoder.py` (`greedy_decoder`).
//!
//! Algorithm:
//! 1. argmax over channels at each timestep → one label per timestep.
//! 2. Group consecutive identical labels.
//! 3. For each group, if label != 0 (not blank), emit (label, start, end, conf).
//!
//! Returns `Vec<Vec<(label, start_timestep, end_timestep, confidence)>>` —
//! one entry per sequence in the batch.

/// Greedy CTC decode of a softmax probability tensor.
///
/// `probs` has shape `(C, W)` for a single sequence (channels × timesteps),
/// or `(N, C, W)` for a batch. We handle the single-sequence case.
///
/// Returns a vector of `(label, start, end, confidence)` tuples where
/// `start`/`end` are timestep indices (inclusive) and `confidence` is the
/// max probability within the group.
pub fn greedy_decode(probs: &[f32], channels: usize, width: usize) -> Vec<(i64, usize, usize, f32)> {
    let mut result = Vec::new();
    if width == 0 {
        return result;
    }

    // Step 1: argmax over channels at each timestep + record confidence.
    let mut best_labels = Vec::with_capacity(width);
    let mut best_confs = Vec::with_capacity(width);
    for t in 0..width {
        let slice = &probs[t * channels..(t + 1) * channels];
        let (best_idx, &best_val) = slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        best_labels.push(best_idx as i64);
        best_confs.push(best_val);
    }

    // Step 2 & 3: collapse consecutive identical labels, drop blanks (0).
    let mut t = 0;
    while t < width {
        let label = best_labels[t];
        let start = t;
        let mut max_conf = best_confs[t];
        // Extend the group while the label repeats.
        while t + 1 < width && best_labels[t + 1] == label {
            t += 1;
            if best_confs[t] > max_conf {
                max_conf = best_confs[t];
            }
        }
        let end = t;
        if label != 0 {
            result.push((label, start, end, max_conf));
        }
        t += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_decode() {
        // 3 channels, 6 timesteps
        // Labels: [0, 1, 1, 0, 2, 2] → decoded: [1, 2]
        let probs: Vec<f32> = vec![
            // t0: blank wins
            0.9, 0.05, 0.05,
            // t1: label 1
            0.05, 0.9, 0.05,
            // t2: label 1 (repeat, collapsed)
            0.05, 0.8, 0.15,
            // t3: blank
            0.9, 0.05, 0.05,
            // t4: label 2
            0.05, 0.05, 0.9,
            // t5: label 2 (repeat)
            0.05, 0.05, 0.8,
        ];
        let decoded = greedy_decode(&probs, 3, 6);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].0, 1);
        assert_eq!(decoded[0].1, 1); // start
        assert_eq!(decoded[0].2, 2); // end
        assert_eq!(decoded[1].0, 2);
        assert_eq!(decoded[1].1, 4); // start
        assert_eq!(decoded[1].2, 5); // end
    }

    #[test]
    fn test_all_blank() {
        let probs: Vec<f32> = vec![0.9, 0.05, 0.05, 0.9, 0.05, 0.05];
        let decoded = greedy_decode(&probs, 3, 2);
        assert!(decoded.is_empty());
    }
}
