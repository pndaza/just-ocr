//! Recognition model: builds the VGSL network from safetensors weights and
//! runs the forward pass.
//!
//! The network graph is built **dynamically from the VGSL spec** (via
//! [`super::vgsl`]), so any kraken recognition model whose spec uses the
//! recognition dialect (`Cr`, `Do`, `Mp`, `S`, `Lbx`, `O...c<N>`) loads without
//! hardcoded layer names, counts, or dimensions. The LSTM input feature width
//! falls out of the spec's conv/maxpool/reshape shape tracking rather than
//! being a literal.
//!
//! Canonical architecture (e.g. `[1,48,0,1 Cr3,13,32 Do Mp ... S1(1x0)1,3
//! Lbx200 ... O1c119]`):
//!
//! Input: `(1, 1, H, W)` — NCHW, grayscale, variable width.
//! Output: `(1, 1, W', num_classes)` — timestep-major logits.

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor, DType};
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder, Module};
use candle_nn::rnn::{LSTM, RNN};
use std::collections::HashMap;

use super::codec::Codec;
use super::meta::RecogMeta;
use super::vgsl::{self, Axis, Direction, RecogNetwork, VgslBlock};

/// A conv layer with asymmetric padding (for non-square kernels).
struct PaddedConv2d {
    inner: Conv2d,
    pad_h: usize,
    pad_w: usize,
}

impl PaddedConv2d {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if self.pad_h > 0 || self.pad_w > 0 {
            let x = x.pad_with_zeros(2, self.pad_h, self.pad_h)?;
            let x = x.pad_with_zeros(3, self.pad_w, self.pad_w)?;
            self.inner.forward(&x)
        } else {
            self.inner.forward(x)
        }
    }
}

/// A bidirectional LSTM using candle's LSTM cells.
/// Input:  (batch, seq_len, in_dim)
/// Output: (batch, seq_len, 2 * hidden_dim)
struct BiLstm {
    fwd: LSTM,
    bwd: LSTM,
}

impl BiLstm {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        // input: (batch, seq_len, in_dim)
        let (batch, seq_len, _in_dim) = input.dims3()?;

        // Forward pass.
        let init_fwd = self.fwd.zero_state(batch)?;
        let fwd_states = self.fwd.seq_init(input, &init_fwd)?;
        let fwd_outs: Vec<Tensor> = fwd_states.iter().map(|s| s.h.clone()).collect();

        // Backward pass: reverse the sequence.
        let reversed = input.flip(&[1])?.contiguous()?;
        let init_bwd = self.bwd.zero_state(batch)?;
        let bwd_states = self.bwd.seq_init(&reversed, &init_bwd)?;
        let bwd_outs: Vec<Tensor> = bwd_states.iter().map(|s| s.h.clone()).collect();

        // Combine: concat forward and backward hidden states at each position.
        let mut combined = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let f = &fwd_outs[t];
            let b = &bwd_outs[seq_len - 1 - t];
            let cat = Tensor::cat(&[f, b], 1)?;
            combined.push(cat);
        }
        // Stack along seq dim: (batch, seq_len, 2*hidden)
        Tensor::stack(&combined, 1)
    }
}

/// A runtime layer built from a parsed VGSL block. Carries weights where the
/// block has them; MaxPool/Dropout/Reshape are weightless and either fold into
/// the forward control flow or are no-ops.
enum BlockLayer {
    /// Padded conv + its activation char ('r' ReLU, 'l' linear, ...).
    Conv(PaddedConv2d, char),
    /// MaxPool2d with explicit kernel/stride.
    MaxPool((usize, usize), (usize, usize)),
    /// Dropout — no-op at inference.
    Dropout,
    /// A bidirectional x-axis LSTM. Only Lbx blocks reach here; Ly/Ls are
    /// rejected during build.
    Lstm(BiLstm),
    /// Final linear layer (the `O...c<N>` output).
    Linear(Linear),
}

/// A recognition model loaded from safetensors, ready for inference.
pub struct RecognitionModel {
    /// Runtime layers in spec order (convs, maxpools, lstms, linear, ...).
    blocks: Vec<BlockLayer>,
    /// The codec for decoding labels → text.
    pub codec: Codec,
    /// Input height from the VGSL spec.
    pub height: usize,
    /// Padding (left, right) applied during preprocessing.
    pub padding: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Whether preprocessing should apply the ocropy `CenterNormalizer`
    /// content dewarp (kraken `_create_transforms` branch B: fixed height,
    /// variable width, single channel). Matches kraken's `valid_norm` default.
    pub center_norm: bool,
}

impl RecognitionModel {
    /// Load a recognition model from a safetensors file.
    pub fn load(path: &str) -> Result<Self> {
        let meta = super::meta::parse_recognition_meta(path)?;
        Self::load_with_meta(path, &meta)
    }

    /// Load a recognition model from an in-memory safetensors buffer.
    ///
    /// Used when the model bytes are embedded in the binary via
    /// `include_bytes!` — avoids any filesystem access.
    pub fn load_from_buffer(data: &[u8]) -> Result<Self> {
        let meta = super::meta::parse_recognition_meta_from_buffer(data)?;
        Self::load_with_meta_buffer(data, &meta)
    }

    /// Build the model from a safetensors file using pre-parsed metadata.
    pub fn load_with_meta(path: &str, meta: &RecogMeta) -> Result<Self> {
        let device = Device::Cpu;
        let raw_tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("Failed to load safetensors: {path}"))?;
        Self::build(raw_tensors, meta)
    }

    /// Build the model from an in-memory safetensors buffer + pre-parsed metadata.
    pub fn load_with_meta_buffer(data: &[u8], meta: &RecogMeta) -> Result<Self> {
        let device = Device::Cpu;
        let raw_tensors = candle_core::safetensors::load_buffer(data, &device)
            .context("Failed to load safetensors from buffer")?;
        Self::build(raw_tensors, meta)
    }

    /// Construct the network layers from a tensor map + metadata. Shared by
    /// the file- and buffer-based loaders.
    fn build(raw_tensors: HashMap<String, Tensor>, meta: &RecogMeta) -> Result<Self> {
        let device = Device::Cpu;

        // Parse the VGSL spec into a block list and resolve dynamic dims.
        let mut net = vgsl::parse(&meta.vgsl)
            .with_context(|| format!("failed to parse VGSL spec: {}", meta.vgsl))?;
        vgsl::resolve(&mut net)?;

        // Strip the `<uuid>.nn.` prefix from tensor names.
        let prefix = format!("{}.nn.", meta.uuid);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, tensor) in raw_tensors {
            let stripped = name.strip_prefix(&prefix).unwrap_or(&name).to_string();
            tensors.insert(stripped, tensor);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Build runtime layers from the parsed blocks, threading the running
        // channel count (for conv in_channels and the linear in_dim).
        let blocks = build_layers(&net, &vb)?;

        // input_nhwc = [batch, height, width, channels].
        let (_, h_spec, w_spec, c_spec) = (
            net.input_nhwc[0],
            net.input_nhwc[1],
            net.input_nhwc[2],
            net.input_nhwc[3],
        );
        let height = h_spec;
        // kraken `_create_transforms` branch B: CenterNormalizer is selected
        // when the input spec is fixed-height, variable-width, single-channel.
        let center_norm = h_spec > 1 && w_spec == 0 && c_spec == 1;

        let num_classes = net.num_classes();
        let codec = Codec::from_c2l(&meta.codec);

        Ok(RecognitionModel {
            blocks,
            codec,
            height,
            padding: 16,
            num_classes,
            center_norm,
        })
    }

    /// Run the forward pass.
    ///
    /// Input: `(1, 1, H, W)` float tensor (NCHW, grayscale, inverted).
    /// Output: `(1, 1, W', num_classes)` timestep-major logits tensor.
    ///
    /// Iterates the block list. The `S` reshape (collapse H into C) and the
    /// NCHW↔(batch,seq,features) transpose happen at the conv→LSTM transition;
    /// the linear runs on the `(W', features)` layout and is reshaped to
    /// `(1, 1, W', num_classes)` for decoding.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone(); // (1, 1, H, W)
        let mut in_lstm_phase = false; // have we crossed the S reshape yet?

        for block in &self.blocks {
            match block {
                BlockLayer::Conv(conv, act) => {
                    x = conv.forward(&x)?;
                    if *act == 'r' {
                        x = x.relu()?;
                    } else if *act == 's' {
                        x = candle_nn::ops::sigmoid(&x)?;
                    } else if *act == 't' {
                        x = x.tanh()?;
                    }
                    // 'l' (linear) → no activation.
                }
                BlockLayer::MaxPool((ky, kx), _stride) => {
                    x = x.max_pool2d((*ky, *kx))?;
                }
                BlockLayer::Dropout => { /* no-op at inference */ }
                BlockLayer::Lstm(lstm) => {
                    if !in_lstm_phase {
                        // First LSTM: apply the S1(1x0)1,3 reshape (collapse H
                        // into C) then transpose NCHW → (batch, seq, features).
                        x = collapse_h_into_c(&x)?;
                        // x is now (1, C', 1, W). Move W to the seq axis:
                        // (N,C,H,W) → permute(0,3,2,1) → (1, W', 1, C') → squeeze H.
                        x = x.permute((0, 3, 2, 1))?.contiguous()?;
                        x = x.squeeze(2)?; // (1, W', C')
                        in_lstm_phase = true;
                    }
                    x = lstm.forward(&x)?;
                }
                BlockLayer::Linear(lin) => {
                    // x: (1, W', features) → squeeze batch → (W', features)
                    x = x.squeeze(0)?;
                    x = lin.forward(&x)?; // (W', num_classes)
                    // Return as (1, 1, W', num_classes) — timestep-major.
                    x = x.unsqueeze(0)?.unsqueeze(0)?;
                }
            }
        }

        Ok(x)
    }

    /// Run recognition on a single preprocessed line tensor.
    ///
    /// Input: `(1, 1, H, W)` float tensor.
    /// Returns: the decoded text string.
    pub fn recognize(&self, input: &Tensor) -> Result<String> {
        let logits = self.forward(input)?.contiguous()?;
        // logits: (1, 1, W', num_classes) — timestep-major
        // softmax over the last dim (classes)
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        // Flatten to (W' * C) vec — already in the layout the decoder expects:
        // [t0_c0, t0_c1, ..., t0_c(C-1), t1_c0, ...]
        let probs = probs.squeeze(0)?.squeeze(0)?; // (W', C)
        let (w, c) = probs.dims2()?;
        let prob_slice = probs.flatten_all()?.to_vec1::<f32>()?;
        let decoded = super::decode::greedy_decode(&prob_slice, c, w);
        let labels: Vec<i64> = decoded.iter().map(|(l, _, _, _)| *l).collect();
        Ok(self.codec.decode(&labels))
    }
}

// ── Build (parsed blocks → runtime layers) ──────────────────────────

/// Walk a resolved [`RecogNetwork`] and build the runtime [`BlockLayer`] list,
/// threading the running channel count so each layer's in-channels/input-dim is
/// correct. Rejects blocks the forward pass doesn't implement (Ly/Ls LSTM,
/// summarize, non-CTC output).
fn build_layers(net: &RecogNetwork, vb: &VarBuilder) -> Result<Vec<BlockLayer>> {
    let mut layers = Vec::with_capacity(net.blocks.len());
    let mut channels = net.input_nhwc[3]; // running C
    for block in &net.blocks {
        match block {
            VgslBlock::Conv {
                name,
                kernel,
                out_channels,
                ..
            } => {
                let conv = build_padded_conv(vb, name, channels, *out_channels, *kernel)?;
                layers.push(BlockLayer::Conv(conv, block_activation(block)));
                channels = *out_channels;
            }
            VgslBlock::Dropout { .. } => {
                layers.push(BlockLayer::Dropout);
            }
            VgslBlock::MaxPool {
                kernel, stride, ..
            } => {
                layers.push(BlockLayer::MaxPool(*kernel, *stride));
            }
            VgslBlock::Reshape { .. } => {
                // No runtime layer — the collapse_h_into_c op runs at the first
                // LSTM. We only validate here (resolve() already checked the
                // supported shape).
            }
            VgslBlock::Lstm {
                name,
                hidden,
                direction,
                axis,
                summarize,
                input_dim,
            } => {
                if *axis == Axis::Y {
                    bail!("LSTM y-axis (Ly) is not supported by the recognition forward pass");
                }
                if *summarize {
                    bail!("summarizing LSTM (Ls) is not supported by the recognition forward pass");
                }
                // Only bidirectional LSTMs occur in recognition specs, but build
                // the BiLstm either way (forward/reverse use the same weight layout).
                let _ = direction;
                let lstm = build_bilstm(vb, name, *input_dim, *hidden)?;
                layers.push(BlockLayer::Lstm(lstm));
                channels = match direction {
                    Direction::Bidirectional => 2 * hidden,
                    _ => *hidden,
                };
            }
            VgslBlock::Output {
                name, num_classes, ..
            } => {
                let lin = build_linear(vb, name, channels, *num_classes)?;
                layers.push(BlockLayer::Linear(lin));
            }
        }
    }
    Ok(layers)
}

/// Extract the activation char from a Conv block (default 'l').
fn block_activation(b: &VgslBlock) -> char {
    match b {
        VgslBlock::Conv { activation, .. } => *activation,
        _ => 'l',
    }
}

/// The `S1(1x0)1,3` reshape: flatten height into channels.
///
/// `(N, C, H, W) → permute(0, 2, 1, 3) → reshape(N, H*C, 1, W)` — swaps C and
/// H axes, then flattens H*C into the channel dim. E.g.
/// `(1, 64, 6, W) → (1, 6, 64, W) → (1, 384, 1, W)`.
fn collapse_h_into_c(input: &Tensor) -> candle_core::Result<Tensor> {
    let (n, c, h, w) = input.dims4()?;
    input
        .permute((0, 2, 1, 3))?
        .contiguous()?
        .reshape((n, h * c, 1, w))
}

// ── Builder helpers ──────────────────────────────────────────────────

fn build_padded_conv(
    vb: &VarBuilder,
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
) -> Result<PaddedConv2d> {
    let prefix = vb.pp(name);
    let weight = prefix.get((out_channels, in_channels, kernel.0, kernel.1), "co.weight")?;
    let bias = prefix.get(out_channels, "co.bias")?;
    let config = Conv2dConfig { padding: 0, ..Default::default() };
    Ok(PaddedConv2d {
        inner: Conv2d::new(weight, Some(bias), config),
        pad_h: kernel.0 / 2,
        pad_w: kernel.1 / 2,
    })
}

fn build_bilstm(
    vb: &VarBuilder,
    name: &str,
    in_dim: usize,
    hidden_dim: usize,
) -> Result<BiLstm> {
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
    Ok(BiLstm { fwd, bwd })
}

fn build_linear(
    vb: &VarBuilder,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Linear> {
    let prefix = vb.pp(name);
    let weight = prefix.get((out_dim, in_dim), "lin.weight")?;
    let bias = prefix.get(out_dim, "lin.bias")?;
    Ok(Linear::new(weight, Some(bias)))
}
