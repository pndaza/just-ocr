//! Parse kraken model metadata from the safetensors file header.
//!
//! The safetensors header contains a `kraken_meta` key whose value is a JSON
//! string. That JSON is a dict keyed by a UUID prefix (one entry per model in
//! the file). Each entry contains:
//!   - `vgsl`: the VGSL spec string (e.g. "[1,120,0,1 Cr3,13,32 ... O1c118]")
//!   - `codec`: the c2l dict (grapheme → labels)
//!   - `one_channel_mode`: "L", "1", or null
//!   - `seg_type`: "baselines", "bbox", or null
//!   - `_model`: model class name (e.g. "TorchVGSLModel")
//!   - `_tasks`: ["recognition"] or ["segmentation"]
//!
//! Every tensor name is prefixed with `<uuid>.nn.`.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;

/// Raw metadata entry for a single model within the safetensors file.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct KrakenMetaEntry {
    /// The VGSL spec, e.g. "[1,120,0,1 Cr3,13,32 Do Mp ... O1c118]".
    pub vgsl: Option<String>,
    /// The codec as a c2l dict: grapheme → label list.
    pub codec: Option<HashMap<String, Vec<i64>>>,
    /// One-channel mode: "L" (grayscale), "1" (binary), or null.
    #[serde(rename = "one_channel_mode")]
    pub one_channel_mode: Option<String>,
    /// Segmentation type: "baselines", "bbox", or null.
    #[serde(rename = "seg_type")]
    pub seg_type: Option<String>,
    /// Model tasks: ["recognition"] or ["segmentation"].
    #[serde(rename = "_tasks")]
    pub tasks: Option<Vec<String>>,
    /// Model class name, e.g. "TorchVGSLModel".
    #[serde(rename = "_model")]
    pub model: Option<String>,
}

/// Parsed recognition model metadata.
#[derive(Debug, Clone)]
pub struct RecogMeta {
    /// The VGSL spec string.
    pub vgsl: String,
    /// The codec (grapheme → labels).
    pub codec: HashMap<String, Vec<i64>>,
    /// Input shape parsed from the VGSL spec: (batch, height, width, channels).
    /// Width 0 means variable.
    pub input_nhwc: [i64; 4],
    /// One-channel mode ("L", "1", or "None").
    pub one_channel_mode: String,
    /// The UUID prefix used in tensor names (e.g. "937bdeb5-...").
    pub uuid: String,
}

/// Read the safetensors header (metadata only) without loading all tensors.
///
/// Safetensors format: 8-byte little-endian header length, then JSON header,
/// then tensor data. We only need the header's `__metadata__` field.
pub fn read_safetensors_metadata(path: &str) -> Result<HashMap<String, String>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open safetensors file: {path}"))?;
    let mut header_len_buf = [0u8; 8];
    file.read_exact(&mut header_len_buf)?;
    let header_len = u64::from_le_bytes(header_len_buf) as usize;
    // Sanity check: headers are typically < 1MB.
    if header_len > 100 * 1024 * 1024 {
        return Err(anyhow!("Safetensors header too large: {header_len} bytes"));
    }
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    file.seek(SeekFrom::Start(0)).ok();

    let header: serde_json::Value = serde_json::from_slice(&header_buf)
        .context("Failed to parse safetensors header JSON")?;

    let mut metadata = HashMap::new();
    if let Some(meta_obj) = header.get("__metadata__").and_then(|v| v.as_object()) {
        for (k, v) in meta_obj {
            if let Some(s) = v.as_str() {
                metadata.insert(k.clone(), s.to_string());
            }
        }
    }
    Ok(metadata)
}

/// Parse recognition metadata from a safetensors file.
///
/// Reads the `kraken_meta` header, finds the first recognition model entry,
/// and extracts VGSL spec, codec, input shape, and UUID prefix.
pub fn parse_recognition_meta(path: &str) -> Result<RecogMeta> {
    let metadata = read_safetensors_metadata(path)?;
    let kraken_meta_str = metadata
        .get("kraken_meta")
        .ok_or_else(|| anyhow!("No 'kraken_meta' in safetensors metadata"))?;

    // The kraken_meta is a JSON string mapping UUID -> metadata entry.
    let entries: HashMap<String, KrakenMetaEntry> =
        serde_json::from_str(kraken_meta_str).context("Failed to parse kraken_meta JSON")?;

    // Find the first recognition model.
    let (uuid, entry) = entries
        .iter()
        .find(|(_, e)| {
            e.tasks
                .as_ref()
                .map(|t| t.iter().any(|x| x == "recognition"))
                .unwrap_or(false)
        })
        .ok_or_else(|| anyhow!("No recognition model found in safetensors file"))?;

    let vgsl = entry
        .vgsl
        .clone()
        .ok_or_else(|| anyhow!("Missing VGSL spec in metadata"))?;

    let codec = entry.codec.clone().unwrap_or_default();
    let one_channel_mode = entry.one_channel_mode.clone().unwrap_or_else(|| "L".to_string());

    // Parse input shape from VGSL spec: "[batch,height,width,channels ...]"
    let input_nhwc = parse_vgsl_input(&vgsl)?;

    Ok(RecogMeta {
        vgsl,
        codec,
        input_nhwc,
        one_channel_mode,
        uuid: uuid.clone(),
    })
}

/// Parse the input block `[b,h,w,c]` from a VGSL spec string.
fn parse_vgsl_input(vgsl: &str) -> Result<[i64; 4]> {
    // Find the first '[' and the first ' ' (end of input block).
    let start = vgsl
        .find('[')
        .ok_or_else(|| anyhow!("VGSL spec missing '[': {vgsl}"))?;
    let end = vgsl
        .find(' ')
        .ok_or_else(|| anyhow!("VGSL spec has only an input block: {vgsl}"))?;
    let input_block = &vgsl[start + 1..end];
    let parts: Vec<&str> = input_block.split(',').collect();
    if parts.len() != 4 {
        return Err(anyhow!("VGSL input block must have 4 values: [{input_block}]"));
    }
    let vals: Vec<i64> = parts
        .iter()
        .map(|s| {
            s.trim()
                .parse::<i64>()
                .with_context(|| format!("Failed to parse VGSL input value: {s}"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok([vals[0], vals[1], vals[2], vals[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires kraken-rust's fixture model at models/myanmar_0.9973.safetensors"]
    fn test_parse_myanmar_model_meta() {
        let meta = parse_recognition_meta("models/myanmar_0.9973.safetensors").unwrap();
        assert_eq!(meta.input_nhwc, [1, 120, 0, 1]);
        // VGSL spec has name annotations like Cr{C_0}3,13,32
        assert!(meta.vgsl.contains("3,13,32"), "VGSL: {}", meta.vgsl);
        assert!(meta.vgsl.contains("1c118"), "VGSL: {}", meta.vgsl);
        assert_eq!(meta.one_channel_mode, "L");
        assert!(!meta.uuid.is_empty());
        // The codec should have 117 entries (labels 1..117, blank=0).
        assert_eq!(meta.codec.len(), 117);
        // Check a known entry: space → [1]
        assert_eq!(meta.codec.get(" "), Some(&vec![1]));
    }
}
