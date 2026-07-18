//! Character codec: bidirectional mapping between graphemes and label sequences.
//!
//! Port of kraken's `kraken/lib/codec.py` (`PytorchCodec`).
//!
//! The codec is a many-to-many mapping. A single grapheme (e.g. a Myanmar
//! consonant + vowel cluster) may map to multiple consecutive labels (`o2m`)
//! and vice versa. The model stores this as a `c2l` dict in the safetensors
//! metadata: `{"grapheme_str": [label, ...], ...}`.
//!
//! Label 0 is reserved for the CTC blank.

use std::collections::HashMap;

/// The recognition codec: maps label sequences back to Unicode graphemes.
#[derive(Debug, Clone)]
pub struct Codec {
    /// Single-label fast path: label -> grapheme.
    /// Most entries are single-label, so this covers the common case.
    pub l2c_single: HashMap<i64, String>,
    /// Multi-label mapping: label-tuple -> grapheme, sorted by length desc
    /// for greedy longest-match decoding.
    pub l2c_multi: Vec<(Vec<i64>, String)>,
}

impl Codec {
    /// Build a codec from the `c2l` dict (grapheme -> labels) stored in the
    /// safetensors metadata.
    pub fn from_c2l(c2l: &HashMap<String, Vec<i64>>) -> Self {
        let mut l2c_single = HashMap::new();
        let mut l2c_multi: Vec<(Vec<i64>, String)> = Vec::new();

        for (grapheme, labels) in c2l {
            if labels.len() == 1 {
                l2c_single.insert(labels[0], grapheme.clone());
            } else {
                l2c_multi.push((labels.clone(), grapheme.clone()));
            }
        }
        // Sort by label sequence length descending for greedy longest match.
        l2c_multi.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Codec {
            l2c_single,
            l2c_multi,
        }
    }

    /// Decode a sequence of collapsed CTC labels (blanks already removed,
    /// repeats already merged) into a Unicode string.
    ///
    /// Greedy longest-match: for each position, try multi-label mappings first
    /// (longest first), then fall back to the single-label map.
    pub fn decode(&self, labels: &[i64]) -> String {
        let mut result = String::new();
        let mut i = 0;
        while i < labels.len() {
            // Try multi-label mappings (already sorted by length desc).
            let mut matched = false;
            for (seq, grapheme) in &self.l2c_multi {
                if i + seq.len() <= labels.len() && &labels[i..i + seq.len()] == seq.as_slice() {
                    result.push_str(grapheme);
                    i += seq.len();
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
            // Single-label fallback.
            if let Some(grapheme) = self.l2c_single.get(&labels[i]) {
                result.push_str(grapheme);
            }
            // Non-decodable labels are silently skipped (kraken default).
            i += 1;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_label_codec() {
        let mut c2l = HashMap::new();
        c2l.insert("a".to_string(), vec![1]);
        c2l.insert("b".to_string(), vec![2]);
        c2l.insert("c".to_string(), vec![3]);
        let codec = Codec::from_c2l(&c2l);
        assert_eq!(codec.decode(&[1, 2, 3]), "abc");
        assert_eq!(codec.decode(&[1, 1, 1]), "aaa");
        assert_eq!(codec.decode(&[]), "");
    }

    #[test]
    fn test_multi_label_codec() {
        let mut c2l = HashMap::new();
        c2l.insert("x".to_string(), vec![10, 11, 12]);
        c2l.insert("y".to_string(), vec![5]);
        let codec = Codec::from_c2l(&c2l);
        assert_eq!(codec.decode(&[10, 11, 12]), "x");
        assert_eq!(codec.decode(&[5, 10, 11, 12, 5]), "yxy");
    }
}
