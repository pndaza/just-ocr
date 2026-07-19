//! Recognition model: builds the VGSL network from safetensors weights and
//! runs the forward pass.
//!
//! The Myanmar model architecture (from VGSL spec):
//! ```text
//! [1,120,0,1 Cr3,13,32 Do Mp Cr3,13,32 Do Mp Cr3,9,64 Do Mp Cr3,9,64 Do
//!  S1(1x0)1,3 Lbx200 Do Lbx200 Do Lbx200 Do O1c118]
//! ```
//!
//! Input: (1, 1, 120, W) — NCHW, grayscale, height=120, variable width.
//! Output: (1, 118, 1, W/8) — logits over 118 classes (117 graphemes + blank).

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType};
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder, Module};
use candle_nn::rnn::{LSTM, RNN};
use std::collections::HashMap;

use super::codec::Codec;
use super::meta::RecogMeta;

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

/// A recognition model loaded from safetensors, ready for inference.
pub struct RecognitionModel {
    /// 4 conv blocks: (conv + relu). MaxPool(2,2) between the first 3.
    convs: [PaddedConv2d; 4],
    /// 3 bidirectional LSTM layers, each hidden=200, output=400.
    lstms: [BiLstm; 3],
    /// Final linear layer: 400 → num_classes.
    linear: Linear,
    /// The codec for decoding labels → text.
    pub codec: Codec,
    /// Input height from the VGSL spec (120 for this model).
    pub height: usize,
    /// Padding (left, right) applied during preprocessing.
    pub padding: usize,
    /// Number of output classes.
    pub num_classes: usize,
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

        // Strip the `<uuid>.nn.` prefix from tensor names.
        let prefix = format!("{}.nn.", meta.uuid);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, tensor) in raw_tensors {
            let stripped = name.strip_prefix(&prefix).unwrap_or(&name).to_string();
            tensors.insert(stripped, tensor);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Build layers matching the VGSL spec weight names.
        // C_0: Conv2d(1→32, k=3x13), C_3: Conv2d(32→32, k=3x13)
        // C_6: Conv2d(32→64, k=3x9),  C_9: Conv2d(64→64, k=3x9)
        let conv0 = build_padded_conv(&vb, "C_0", 1, 32, (3, 13))?;
        let conv1 = build_padded_conv(&vb, "C_3", 32, 32, (3, 13))?;
        let conv2 = build_padded_conv(&vb, "C_6", 32, 64, (3, 9))?;
        let conv3 = build_padded_conv(&vb, "C_9", 64, 64, (3, 9))?;

        // L_12: BiLSTM(960→200), L_14: BiLSTM(400→200), L_16: BiLSTM(400→200)
        let lstm0 = build_bilstm(&vb, "L_12", 960, 200)?;
        let lstm1 = build_bilstm(&vb, "L_14", 400, 200)?;
        let lstm2 = build_bilstm(&vb, "L_16", 400, 200)?;

        // O_18: Linear(400 → num_classes). The class count is model-specific
        // (bur_recog=119, myanmar=118) and is parsed from the VGSL `O1c<N>`
        // clause by the metadata loader rather than hardcoded.
        let num_classes = meta.num_classes;
        let linear = build_linear(&vb, "O_18", 400, num_classes)?;

        let codec = Codec::from_c2l(&meta.codec);
        let height = meta.input_nhwc[1] as usize;

        Ok(RecognitionModel {
            convs: [conv0, conv1, conv2, conv3],
            lstms: [lstm0, lstm1, lstm2],
            linear,
            codec,
            height,
            padding: 16,
            num_classes,
        })
    }

    /// Run the forward pass.
    ///
    /// Input: `(1, 1, H, W)` float tensor (NCHW, grayscale, inverted).
    /// Output: `(1, num_classes, 1, W/8)` logits tensor.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Conv block 0: conv + relu → maxpool(2,2)
        let x = self.conv_block(input, 0, true)?;
        // Conv block 1: conv + relu → maxpool(2,2)
        let x = self.conv_block(&x, 1, true)?;
        // Conv block 2: conv + relu → maxpool(2,2)
        let x = self.conv_block(&x, 2, true)?;
        // Conv block 3: conv + relu (no maxpool)
        let x = self.conv_block(&x, 3, false)?;

        // Reshape S1(1x0)1,3: (1, 64, 15, W') → (1, 960, 1, W')
        let x = self.reshape_s11x013(&x)?;

        // LSTM layers: transpose to (batch, seq, features) for candle's RNN.
        // (1, 960, 1, W') → permute → (1, W', 1, 960) → squeeze H → (1, W', 960)
        let x = x.permute((0, 3, 2, 1))?.contiguous()?;
        let x = x.squeeze(2)?;
        let x = self.lstms[0].forward(&x)?;
        let x = self.lstms[1].forward(&x)?;
        let x = self.lstms[2].forward(&x)?;
        // x: (1, W', 400)

        // Linear: (1, W', 400) → squeeze batch → (W', 400) → linear → (W', 118)
        let x = x.squeeze(0)?;
        let x = self.linear.forward(&x)?;
        // Return as (1, 1, W', num_classes) — timestep-major layout, which is
        // the natural memory order from the linear layer. The caller (recognize)
        // handles the layout for decoding.
        let x = x.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, W', num_classes)

        Ok(x)
    }

    /// Conv block: padded conv2d → ReLU → optional MaxPool(2,2).
    fn conv_block(&self, input: &Tensor, idx: usize, maxpool: bool) -> candle_core::Result<Tensor> {
        let x = self.convs[idx].forward(input)?;
        let x = x.relu()?;
        if maxpool {
            x.max_pool2d((2, 2))
        } else {
            Ok(x)
        }
    }

    /// Reshape S1(1x0)1,3: flatten the height and channel dims.
    ///
    /// PyTorch's Reshape layer with src_dim=H, high=H, low=C does:
    ///   (N, C, H, W) → permute(0, 2, 1, 3) → reshape(N, H*C, 1, W)
    ///
    /// I.e. it swaps C and H axes, then flattens them into the channel dim.
    /// E.g. (1, 64, 15, 107) → permute → (1, 15, 64, 107) → reshape → (1, 960, 1, 107)
    fn reshape_s11x013(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let (n, c, h, w) = input.dims4()?;
        // Permute (N, C, H, W) → (N, H, C, W), then flatten H*C → channels.
        input.permute((0, 2, 1, 3))?.contiguous()?.reshape((n, h * c, 1, w))
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

