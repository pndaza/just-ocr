//! Segmentation model using candle-core (native Rust, no ONNX).
//!
//! Loads kraken segmentation models directly from `.safetensors` and runs the
//! forward pass via candle-core. This avoids ONNX Runtime entirely and is
//! especially useful for models where ONNX LSTM export is unreliable.
//!
//! Architecture (from VGSL spec):
//! ```text
//! [1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32
//!  Cr3,3,256 Gn32 Cr3,3,256 Gn32
//!  Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32 O2l4]
//! ```
//!
//! Input:  (1, 3, 1800, W) — RGB image, height=1800, variable width
//! Output: (1, 4, H/4, W/4) — heatmap logits for 4 channels

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, Module};
use candle_nn::rnn::{LSTM, RNN};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::model_meta::{ClassMapping, ModelMeta};

/// A segmentation model loaded from safetensors, running via candle-core.
pub struct SegmentationModelCandle {
    /// Conv layers with activations.
    convs: Vec<ActConv2d>,
    /// GroupNorm layers (indices into convs for interleaving).
    group_norms: Vec<GroupNorm>,
    /// LSTM layers with axis info.
    lstms: Vec<AxisBiLstm>,
    /// The layer execution order (parsed from VGSL spec).
    layer_order: Vec<LayerKind>,
    /// Model metadata.
    pub meta: ModelMeta,
    /// Fixed input height (1800).
    pub height: u32,
    /// Compute dtype the weights are stored as (F32 or BF16). The input is
    /// cast to this at the start of `forward` and the output cast back to F32.
    compute_dtype: DType,
}

#[derive(Debug, Clone, Copy)]
enum LayerKind {
    Conv(usize),       // index into convs
    GroupNorm(usize),  // index into group_norms
    Lstm(usize),       // index into lstms
}

/// Conv2d + activation (relu, sigmoid, or linear).
struct ActConv2d {
    inner: Conv2d,
    activation: Activation,
    pad_h: usize,
    pad_w: usize,
}

#[derive(Debug, Clone, Copy)]
enum Activation {
    Relu,
    Sigmoid,
    Linear,
}

impl ActConv2d {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = if self.pad_h > 0 || self.pad_w > 0 {
            let x = x.pad_with_zeros(2, self.pad_h, self.pad_h)?;
            x.pad_with_zeros(3, self.pad_w, self.pad_w)?
        } else {
            x.clone()
        };
        let x = self.inner.forward(&x)?;
        match self.activation {
            Activation::Relu => x.relu(),
            Activation::Sigmoid => candle_nn::ops::sigmoid(&x),
            Activation::Linear => Ok(x),
        }
    }
}

/// GroupNorm wrapper.
struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    num_groups: usize,
    num_channels: usize,
}

impl GroupNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // PyTorch GroupNorm: normalize within groups of channels.
        // x: (N, C, H, W). Reshape to (N, num_groups, C/num_groups * H*W),
        // compute mean/var over dim 2, normalize, scale+shift.
        let (n, c, h, w) = x.dims4()?;
        let x_flat = x.reshape((n, self.num_groups, c / self.num_groups * h * w))?;
        let mean = x_flat.mean(2)?.unsqueeze(2)?;
        let centered = x_flat.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean(2)?.unsqueeze(2)?;
        let eps = Tensor::new(1e-5f32, x.device())?;
        let x_norm = centered.broadcast_div(&var.broadcast_add(&eps)?.sqrt()?)?;
        let x_norm = x_norm.reshape((n, c, h, w))?;
        // Scale and shift: weight and bias are per-channel.
        let weight = self.weight.reshape((1, c, 1, 1))?;
        let bias = self.bias.reshape((1, c, 1, 1))?;
        x_norm.broadcast_mul(&weight)?.broadcast_add(&bias)
    }
}

/// Bidirectional LSTM that runs along a spatial axis (x or y) of a feature map.
struct AxisBiLstm {
    fwd: LSTM,
    bwd: LSTM,
    hidden_dim: usize,
    /// If true, LSTM runs along H (height). If false, runs along W (width).
    along_height: bool,
}

impl AxisBiLstm {
    /// Forward pass mirroring PyTorch's TransposedSummarizingRNN.
    ///
    /// PyTorch source (layers.py TransposedSummarizingRNN.forward):
    /// ```python
    /// inputs = inputs.permute(2, 0, 3, 1)       # NCHW -> HNWC
    /// if self.transpose:
    ///     inputs = inputs.transpose(0, 2)        # HNWC -> WNHC
    /// siz = inputs.size()                        # (H or W, N, W or H, C)
    /// inputs = inputs.view(-1, siz[2], siz[3])  # (H*N or W*N, seq, C)
    /// o, _ = self.layer(inputs)                  # (batch, seq, O)
    /// o = o.view(siz[0], siz[1], siz[2], self.output_size)  # (H/W, N, seq, O)
    /// if self.transpose:
    ///     o = o.transpose(0, 2)                  # undo earlier swap
    /// return o.permute(1, 3, 0, 2)              # -> NOHW = NCHW
    /// ```
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        // input: (N, C, H, W)
        let (n, _c, h, w) = input.dims4()?;
        let output_size = self.hidden_dim * 2; // bidirectional

        // Step 1: permute(2,0,3,1) — NCHW -> HNWC
        // Step 2: if transpose (along_height=x-axis), swap dim 0 and 2
        // After this: (seq_dim, N, spatial_dim, C)
        //   y-axis (transpose=False): (H, N, W, C), seq=W
        //   x-axis (transpose=True):  (W, N, H, C), seq=H
        // We defer the .contiguous() to just before the reshape (step 3),
        // since the intermediate permutes are themselves strided views —
        // materializing each one would copy the full tensor twice.
        let x = input.permute((2, 0, 3, 1))?;
        let x = if self.along_height {
            x.permute((2, 1, 0, 3))? // swap dim 0<->2
        } else {
            x
        };

        // Step 3: reshape to (batch, seq, C). Merging dims 0,1 requires them
        // to be contiguous in memory, so materialize once here.
        let (dim0, dim1, dim2, dim3) = x.dims4()?;
        let reshaped = x
            .contiguous()?
            .reshape((dim0 * dim1, dim2, dim3))?;

        // Step 4: run bidirectional LSTM
        let lstm_out = self.run_bilstm(&reshaped)?;
        // lstm_out: (dim0*dim1, dim2, output_size)

        // Step 5: reshape back to (dim0, dim1, dim2, output_size)
        let o = lstm_out.reshape((dim0, dim1, dim2, output_size))?;

        // Step 6: if transpose, undo the swap
        let o = if self.along_height {
            o.permute((2, 1, 0, 3))? // swap dim 0<->2 back
        } else {
            o
        };
        // Now: (H, N, W, O) for both axes

        // Step 7: permute(1,3,0,2) — HNWO -> NOHW = NCHW
        o.permute((1, 3, 0, 2))?.contiguous()
    }

    fn run_bilstm(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, _seq_len, _in_dim) = input.dims3()?;
        if batch == 0 {
            return Ok(input.clone());
        }

        // Parallelize across the batch dimension. Each sequence in the batch is
        // independent (no cross-batch interaction in an LSTM), so chunks can be
        // scanned concurrently. This is the key optimization for this model:
        // candle-nn's `seq_init` is a strictly sequential per-timestep loop
        // (each step depends on the previous state), but the batch dim has
        // hundreds of independent sequences (e.g. ~365 rows × 1 for Lbx).
        let num_threads = rayon::current_num_threads();
        if batch < 2 || num_threads < 2 {
            return self.run_bilstm_serial(input);
        }

        let num_chunks = batch.min(num_threads);
        let chunks: Vec<Tensor> = input.chunk(num_chunks, 0)?;

        let chunks_out: Vec<Tensor> = chunks
            .par_iter()
            .map(|chunk| self.run_bilstm_serial(chunk))
            .collect::<candle_core::Result<_>>()?;

        let refs: Vec<&Tensor> = chunks_out.iter().collect();
        Tensor::cat(&refs, 0)
    }

    fn run_bilstm_serial(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, _seq_len, _in_dim) = input.dims3()?;
        let init_fwd = self.fwd.zero_state(batch)?;
        let fwd_states = self.fwd.seq_init(input, &init_fwd)?;
        // Stack all forward hidden states into one (batch, seq_len, hidden) tensor.
        let fwd_outs: Vec<Tensor> = fwd_states.iter().map(|s| s.h.clone()).collect();
        let fwd_stacked = Tensor::stack(&fwd_outs, 1)?;

        let reversed = input.flip(&[1])?.contiguous()?;
        let init_bwd = self.bwd.zero_state(batch)?;
        let bwd_states = self.bwd.seq_init(&reversed, &init_bwd)?;
        let mut bwd_outs: Vec<Tensor> = bwd_states.iter().map(|s| s.h.clone()).collect();

        // bwd_outs[t] is the hidden state after consuming the reversed input up
        // to step t — i.e. after consuming the original sequence from the right
        // edge down to position (seq_len - 1 - t). To align it with fwd_outs[t]
        // (which covers positions 0..=t) we need bwd_outs[seq_len - 1 - t], so
        // reverse the bwd axis.
        bwd_outs.reverse();
        let bwd_stacked = Tensor::stack(&bwd_outs, 1)?;

        // Single concat along the hidden dimension: (batch, seq_len, 2*hidden).
        // Replaces the per-timestep loop that built seq_len small tensors then
        // stacked them — 2 cat/stack calls instead of seq_len+1.
        Tensor::cat(&[&fwd_stacked, &bwd_stacked], 2)
    }
}

impl SegmentationModelCandle {
    /// Load a segmentation model from a safetensors file (F32 compute).
    ///
    /// Reads the VGSL spec, codec, and class mapping from the safetensors
    /// metadata header and constructs the network layers.
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_dtype(path, DType::F32)
    }

    /// Load a segmentation model, quantizing weights to `dtype`.
    ///
    /// `DType::BF16` stores weights as brain-float16, which would double
    /// effective SIMD throughput on the conv/GEMM kernels (8 lanes per NEON
    /// register vs 4 for f32) at the cost of some numerical precision. The
    /// input is cast to `dtype` at the start of [`Self::forward`] and the
    /// output cast back to F32, so callers see the same F32 heatmap either way.
    ///
    /// NOTE: BF16/F16 are blocked by candle 0.11's CPU matmul dtype guard
    /// (`cpu_backend/mod.rs` accepts only F16|F32|F64, and the Accelerate
    /// backend bails on hgemm). This entry point is kept for forward
    /// compatibility if an upstream candle change lifts the restriction.
    pub fn load_with_dtype(path: &str, dtype: DType) -> Result<Self> {
        let device = Device::Cpu;

        // Parse metadata from the safetensors header.
        let seg_meta = parse_seg_meta(path)?;
        let raw_tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("Failed to load safetensors: {path}"))?;
        Self::build(raw_tensors, &seg_meta, dtype)
    }

    /// Load from an in-memory safetensors buffer (e.g. `include_bytes!`).
    ///
    /// Avoids all filesystem access — useful for bundling models into the
    /// binary so a fresh install works with zero setup.
    pub fn load_from_buffer(data: &[u8]) -> Result<Self> {
        Self::load_from_buffer_with_dtype(data, DType::F32)
    }

    /// Buffer-based loader with explicit compute dtype. See
    /// [`load_with_dtype`] for the dtype semantics.
    pub fn load_from_buffer_with_dtype(data: &[u8], dtype: DType) -> Result<Self> {
        let device = Device::Cpu;
        let seg_meta = parse_seg_meta_from_buffer(data)?;
        let raw_tensors = candle_core::safetensors::load_buffer(data, &device)
            .context("Failed to load safetensors from buffer")?;
        Self::build(raw_tensors, &seg_meta, dtype)
    }

    /// Construct the network layers from a tensor map + parsed seg metadata.
    /// Shared by the file- and buffer-based loaders.
    fn build(
        raw_tensors: HashMap<String, Tensor>,
        seg_meta: &SegMeta,
        dtype: DType,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let uuid = &seg_meta.uuid;
        let vgsl = &seg_meta.vgsl;

        // Strip the UUID prefix from tensor names and cast to the target dtype.
        let prefix = format!("{uuid}.nn.");
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, tensor) in raw_tensors {
            let stripped = name.strip_prefix(&prefix).unwrap_or(&name).to_string();
            let tensor = if dtype != DType::F32 {
                tensor.to_dtype(dtype)?
            } else {
                tensor
            };
            tensors.insert(stripped, tensor);
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);

        // Parse VGSL spec and build layers.
        let spec = VgslSpec::parse(vgsl)?;
        let (convs, group_norms, lstms, layer_order) = build_layers(&spec, &vb)?;

        // Build ModelMeta for compatibility with existing detect() pipeline.
        let meta = ModelMeta {
            class_mapping: seg_meta.class_mapping.clone(),
            one_channel_mode: seg_meta.one_channel_mode.clone(),
            topline: seg_meta.topline,
            padding: seg_meta.padding.clone(),
            bounding_regions: None,
            input: vec![1, 3, 1800, 0],
        };

        let height = 1800u32;

        Ok(SegmentationModelCandle {
            convs,
            group_norms,
            lstms,
            layer_order,
            meta,
            height,
            compute_dtype: dtype,
        })
    }

    /// Run the forward pass.
    ///
    /// Input: `(1, 3, 1800, W)` float tensor (NCHW, RGB, [0,1] range).
    /// Output: `(1, num_classes, H/4, W/4)` logits tensor (always F32).
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let profile = std::env::var("KRAKEN_PROFILE_LAYERS").is_ok();
        let mut t_conv = std::time::Duration::ZERO;
        let mut t_norm = std::time::Duration::ZERO;
        let mut t_lstm = std::time::Duration::ZERO;

        // Cast input to the compute dtype (no-op for F32; bf16 for the fast path).
        let mut x = if self.compute_dtype == DType::F32 {
            input.clone()
        } else {
            input.to_dtype(self.compute_dtype)?
        };
        for &layer in &self.layer_order {
            if profile {
                let t = std::time::Instant::now();
                x = match layer {
                    LayerKind::Conv(i) => {
                        let r = self.convs[i].forward(&x)?;
                        t_conv += t.elapsed();
                        r
                    }
                    LayerKind::GroupNorm(i) => {
                        let r = self.group_norms[i].forward(&x)?;
                        t_norm += t.elapsed();
                        r
                    }
                    LayerKind::Lstm(i) => {
                        let r = self.lstms[i].forward(&x)?;
                        t_lstm += t.elapsed();
                        r
                    }
                };
            } else {
                x = match layer {
                    LayerKind::Conv(i) => self.convs[i].forward(&x)?,
                    LayerKind::GroupNorm(i) => self.group_norms[i].forward(&x)?,
                    LayerKind::Lstm(i) => self.lstms[i].forward(&x)?,
                };
            }
        }
        // Cast output back to F32 for the downstream sigmoid/upsample (which
        // run in F32 in inference_candle.rs).
        let x = if self.compute_dtype == DType::F32 {
            x
        } else {
            x.to_dtype(DType::F32)?
        };
        if profile {
            eprintln!(
                "forward: conv {:.1} ms | groupnorm {:.1} ms | lstm {:.1} ms | dtype {:?}",
                t_conv.as_secs_f64() * 1000.0,
                t_norm.as_secs_f64() * 1000.0,
                t_lstm.as_secs_f64() * 1000.0,
                self.compute_dtype,
            );
        }
        Ok(x)
    }
}

// ── Metadata parsing ─────────────────────────────────────────────────

/// Parsed segmentation metadata from safetensors.
struct SegMeta {
    uuid: String,
    vgsl: String,
    class_mapping: ClassMapping,
    one_channel_mode: String,
    topline: bool,
    padding: Vec<i64>,
}

fn parse_seg_meta(path: &str) -> Result<SegMeta> {
    let metadata = crate::recognition::meta::read_safetensors_metadata(path)?;
    parse_seg_meta_impl(metadata)
}

fn parse_seg_meta_from_buffer(data: &[u8]) -> Result<SegMeta> {
    let metadata = crate::recognition::meta::read_safetensors_metadata_from_buffer(data)?;
    parse_seg_meta_impl(metadata)
}

fn parse_seg_meta_impl(
    metadata: HashMap<String, String>,
) -> Result<SegMeta> {
    let kraken_meta_str = metadata
        .get("kraken_meta")
        .ok_or_else(|| anyhow::anyhow!("No 'kraken_meta' in safetensors metadata"))?;

    let entries: HashMap<String, serde_json::Value> =
        serde_json::from_str(kraken_meta_str)?;

    let (uuid, entry) = entries
        .iter()
        .find(|(_, e)| {
            e.get("_tasks")
                .and_then(|t| t.as_array())
                .map(|t| t.iter().any(|x| x.as_str() == Some("segmentation")))
                .unwrap_or(false)
        })
        .ok_or_else(|| anyhow::anyhow!("No segmentation model in safetensors file"))?;

    let vgsl = entry
        .get("vgsl")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing vgsl spec"))?
        .to_string();

    let cm = entry.get("class_mapping").cloned().unwrap_or_default();
    let class_mapping: ClassMapping = serde_json::from_value(cm)?;

    let one_channel_mode = entry
        .get("one_channel_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("RGB")
        .to_string();

    let topline = entry
        .get("topline")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let padding = entry
        .get("hyper_params")
        .and_then(|h| h.get("padding"))
        .and_then(|p| p.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_i64())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(SegMeta {
        uuid: uuid.clone(),
        vgsl,
        class_mapping,
        one_channel_mode,
        topline,
        padding,
    })
}

// ── VGSL spec parsing ────────────────────────────────────────────────

/// Parsed VGSL specification for a segmentation model.
struct VgslSpec {
    /// Input shape [batch, height, width, channels] (NHWC order from spec).
    input: [usize; 4],
    /// Parsed layer blocks.
    blocks: Vec<VgslBlock>,
}

#[derive(Debug)]
enum VgslBlock {
    /// Conv: C[f] kernel_y,kernel_x,out_channels[,stride_y,stride_x]
    Conv {
        name: String,
        kernel: (usize, usize),
        out_channels: usize,
        stride: (usize, usize),
        activation: char, // 'r' = relu, 'l' = linear, 's' = sigmoid
    },
    /// GroupNorm: Gn{<name>}<groups>
    GroupNorm {
        name: String,
        groups: usize,
    },
    /// LSTM: L(b)(x|y)[s]<n>  (b=bidirectional, x/y=axis, s=summarize)
    Lstm {
        name: String,
        hidden: usize,
        along_height: bool, // x-axis = true, y-axis = false
    },
    /// Output: O{<name>}<dim><nl><n>
    Output {
        name: String,
        dim: usize,
        activation: char,
        num_classes: usize,
    },
}

impl VgslSpec {
    fn parse(spec: &str) -> Result<Self> {
        // Parse input block [b,h,w,c]
        let start = spec.find('[').ok_or_else(|| anyhow::anyhow!("No '[' in VGSL"))?;
        let end = spec.find(' ').ok_or_else(|| anyhow::anyhow!("VGSL has no layers"))?;
        let input_str = &spec[start + 1..end];
        let input_parts: Vec<usize> = input_str
            .split(',')
            .map(|s| s.trim().parse::<usize>().unwrap_or(0))
            .collect();
        let input = [input_parts[0], input_parts[1], input_parts[2], input_parts[3]];

        // Parse layer blocks
        let blocks_str = &spec[end + 1..].trim_end_matches(']');
        let mut blocks = Vec::new();
        for token in tokenize_vgsl(blocks_str) {
            if let Some(b) = parse_block(&token) {
                blocks.push(b);
            }
        }

        Ok(VgslSpec { input, blocks })
    }
}

/// Tokenize a VGSL layer string into individual blocks, handling {name} annotations.
/// Spaces inside braces are kept as part of the token; braces are preserved for
/// `parse_block` to extract the name.
fn tokenize_vgsl(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_braces = false;
    for ch in s.chars() {
        match ch {
            '{' => {
                in_braces = true;
                current.push(ch);
            }
            '}' => {
                in_braces = false;
                current.push(ch);
            }
            ' ' if !in_braces => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn parse_block(token: &str) -> Option<VgslBlock> {
    // Extract name from {name} if present.
    let (name, rest) = if let Some(start) = token.find('{') {
        let end = token.find('}')?;
        let n = token[start + 1..end].to_string();
        let r = format!("{}{}", &token[..start], &token[end + 1..]);
        (n, r)
    } else {
        (String::new(), token.to_string())
    };

    // Conv: C[t]<ky>,<kx>,<d>[,<sy>,<sx>]
    if rest.starts_with('C') {
        return parse_conv(&name, &rest);
    }
    // GroupNorm: Gn<groups>
    if rest.starts_with("Gn") {
        let groups: usize = rest[2..].parse().ok()?;
        return Some(VgslBlock::GroupNorm { name, groups });
    }
    // LSTM: L(b)(x|y)[s]<n>
    if rest.starts_with('L') {
        return parse_lstm(&name, &rest);
    }
    // Output: O<dim><nl><n>
    if rest.starts_with('O') {
        return parse_output(&name, &rest);
    }
    None
}

fn parse_conv(name: &str, rest: &str) -> Option<VgslBlock> {
    // Format: C<ky>,<kx>,<out>[,<sy>,<sx>]  (activation letter is before digits)
    // Or:    Cr<ky>,<kx>,<out>  (with activation 'r')
    let chars: Vec<char> = rest.chars().collect();
    if chars[0] != 'C' {
        return None;
    }
    let mut idx = 1;
    // Check for activation letter
    let activation = if idx < chars.len() && "strlm".contains(chars[idx]) {
        let a = chars[idx];
        idx += 1;
        a
    } else {
        'l' // default: linear
    };

    let params: String = chars[idx..].iter().collect();
    let parts: Vec<&str> = params.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let ky: usize = parts[0].parse().ok()?;
    let kx: usize = parts[1].parse().ok()?;
    let out_channels: usize = parts[2].parse().ok()?;
    let stride = if parts.len() >= 5 {
        (parts[3].parse().ok()?, parts[4].parse().ok()?)
    } else {
        (1, 1)
    };

    Some(VgslBlock::Conv {
        name: name.to_string(),
        kernel: (ky, kx),
        out_channels,
        stride,
        activation,
    })
}

fn parse_lstm(name: &str, rest: &str) -> Option<VgslBlock> {
    // Format: L<b?><x|y><s?><n>
    // Lbx32 = bidirectional, x-axis, hidden=32
    // Lby32 = bidirectional, y-axis, hidden=32
    let chars: Vec<char> = rest.chars().collect();
    let mut idx = 1;
    // Skip 'b' (bidirectional) — always present in these models
    if idx < chars.len() && chars[idx] == 'b' {
        idx += 1;
    }
    // Axis: x or y.
    // PyTorch's build_rnn: dim = (spec_dim == 'y'), passed as `transpose`.
    // So 'y' → transpose=True (along_height), 'x' → transpose=False.
    let along_height = if idx < chars.len() {
        match chars[idx] {
            'y' => true,
            'x' => false,
            _ => return None,
        }
    } else {
        return None;
    };
    idx += 1;
    // Skip 's' (summarize) if present
    if idx < chars.len() && chars[idx] == 's' {
        idx += 1;
    }
    // Hidden size
    let hidden_str: String = chars[idx..].iter().collect();
    let hidden: usize = hidden_str.parse().ok()?;

    Some(VgslBlock::Lstm {
        name: name.to_string(),
        hidden,
        along_height,
    })
}

fn parse_output(name: &str, rest: &str) -> Option<VgslBlock> {
    // Format: O<dim><nl><n>  e.g. O2l4
    // dim=2 (heatmap), nl=l (logistic/sigmoid), n=4 (num classes)
    let chars: Vec<char> = rest.chars().collect();
    if chars[0] != 'O' {
        return None;
    }
    let mut idx = 1;
    let dim: usize = chars.get(idx)?.to_digit(10)? as usize;
    idx += 1;
    let activation = *chars.get(idx)?;
    idx += 1;
    let n_str: String = chars[idx..].iter().collect();
    let num_classes: usize = n_str.parse().ok()?;
    Some(VgslBlock::Output {
        name: name.to_string(),
        dim,
        activation,
        num_classes,
    })
}

// ── Layer construction ───────────────────────────────────────────────

fn build_layers(
    spec: &VgslSpec,
    vb: &VarBuilder,
) -> Result<(Vec<ActConv2d>, Vec<GroupNorm>, Vec<AxisBiLstm>, Vec<LayerKind>)> {
    let mut convs = Vec::new();
    let mut group_norms = Vec::new();
    let mut lstms = Vec::new();
    let mut layer_order = Vec::new();

    let mut channels = spec.input[3]; // initial channel count

    for block in &spec.blocks {
        match block {
            VgslBlock::Conv {
                name,
                kernel,
                out_channels,
                stride,
                activation,
            } => {
                let conv = build_act_conv(vb, name, channels, *out_channels, *kernel, *stride, *activation)?;
                convs.push(conv);
                layer_order.push(LayerKind::Conv(convs.len() - 1));
                channels = *out_channels;
            }
            VgslBlock::GroupNorm { name, groups } => {
                let gn = build_group_norm(vb, name, channels, *groups)?;
                group_norms.push(gn);
                layer_order.push(LayerKind::GroupNorm(group_norms.len() - 1));
            }
            VgslBlock::Lstm {
                name,
                hidden,
                along_height,
            } => {
                let lstm = build_axis_bilstm(vb, name, channels, *hidden, *along_height)?;
                lstms.push(lstm);
                layer_order.push(LayerKind::Lstm(lstms.len() - 1));
                channels = hidden * 2; // bidirectional doubles the output
            }
            VgslBlock::Output {
                activation,
                num_classes,
                ..
            } => {
                // The output layer is a 1x1 conv with the specified activation.
                let conv = build_act_conv(
                    vb,
                    "l_16", // hardcoded name matching the safetensors keys
                    channels,
                    *num_classes,
                    (1, 1),
                    (1, 1),
                    *activation,
                )?;
                convs.push(conv);
                layer_order.push(LayerKind::Conv(convs.len() - 1));
                channels = *num_classes;
            }
        }
    }

    Ok((convs, group_norms, lstms, layer_order))
}

fn build_act_conv(
    vb: &VarBuilder,
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    activation: char,
) -> Result<ActConv2d> {
    let prefix = vb.pp(name);
    let weight = prefix.get(
        (out_channels, in_channels, kernel.0, kernel.1),
        "co.weight",
    )?;
    let bias = prefix.get(out_channels, "co.bias")?;
    // SAME padding: pad = dilation * (k - 1) / 2. For dilation=1, k=7 → pad=3.
    // candle's Conv2dConfig uses a single padding value. For non-square kernels
    // we'd need to pad manually, but all seg model kernels are square (7x7, 3x3, 1x1).
    let pad_h = kernel.0 / 2;
    let pad_w = kernel.1 / 2;
    let config = Conv2dConfig {
        padding: 0, // we pad manually for consistency
        stride: stride.0, // candle uses uniform stride; all our convs have equal sy,sx
        ..Default::default()
    };
    let act = match activation {
        'r' => Activation::Relu,
        's' => Activation::Sigmoid,
        _ => Activation::Linear,
    };
    Ok(ActConv2d {
        inner: Conv2d::new(weight, Some(bias), config),
        activation: act,
        pad_h,
        pad_w,
    })
}

// We need to store padding info with ActConv2d. Let me revise the struct.
// Actually, let me add a forward method that handles padding.

fn build_group_norm(
    vb: &VarBuilder,
    name: &str,
    num_channels: usize,
    num_groups: usize,
) -> Result<GroupNorm> {
    let prefix = vb.pp(name).pp("layer");
    let weight = prefix.get(num_channels, "weight")?;
    let bias = prefix.get(num_channels, "bias")?;
    Ok(GroupNorm {
        weight,
        bias,
        num_groups,
        num_channels,
    })
}

fn build_axis_bilstm(
    vb: &VarBuilder,
    name: &str,
    in_dim: usize,
    hidden_dim: usize,
    along_height: bool,
) -> Result<AxisBiLstm> {
    let prefix = vb.pp(name).pp("layer");
    let config_fwd = candle_nn::rnn::LSTMConfig {
        layer_idx: 0,
        direction: candle_nn::rnn::Direction::Forward,
        ..Default::default()
    };
    let config_bwd = candle_nn::rnn::LSTMConfig {
        layer_idx: 0,
        direction: candle_nn::rnn::Direction::Backward,
        ..Default::default()
    };
    let fwd = LSTM::new(in_dim, hidden_dim, config_fwd, prefix.clone())?;
    let bwd = LSTM::new(in_dim, hidden_dim, config_bwd, prefix.clone())?;
    Ok(AxisBiLstm {
        fwd,
        bwd,
        hidden_dim,
        along_height,
    })
}

