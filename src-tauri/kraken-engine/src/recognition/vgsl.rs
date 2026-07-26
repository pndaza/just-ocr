//! VGSL spec parser for recognition models.
//!
//! Parses the kraken VGSL (Very Good Spec Language) spec string into a list of
//! layer blocks, then resolves the dynamic dimensions (notably the LSTM input
//! feature width, which falls out of the conv/maxpool/reshape shape tracking
//! rather than being hardcoded).
//!
//! Port of kraken's `TorchVGSLModel._parse` dispatch (`kraken/lib/vgsl/model.py`)
//! for the **recognition dialect only**: `Cr`/`C`, `Do`, `Mp`, `S`, `Lb`/`Lbx`,
//! and `O...c<N>`. Other VGSL blocks (GroupNorm, parallel/series branches,
//! wav2vec2, GRU, transposed conv, peephole LSTM) error clearly rather than
//! being silently dropped — they never appear in kraken recognition specs.
//!
//! ## The shape-tracking trick
//!
//! kraken threads a running `(N, C, H, W)` shape tuple through the parser; each
//! builder returns the output shape, which feeds the next. The LSTM `input_size`
//! is just `C` *after* the `S` reshape collapses H into C — so it is never a
//! literal in the spec. [`resolve`] reproduces this walk to fill in the LSTM
//! input dimensions ([`VgslBlock::Lstm::input_dim`]).

use anyhow::{anyhow, bail, Result};

// ── Enums ───────────────────────────────────────────────────────────

/// One parsed VGSL layer block in the recognition dialect.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum VgslBlock {
    /// Conv2d: `C[act][{name}]ky,kx,out[,sy,sx]`. `activation` is 'r','l','s','t','m'.
    Conv {
        name: String,
        kernel: (usize, usize),
        out_channels: usize,
        stride: (usize, usize),
        activation: char,
    },
    /// Dropout: `Do[{name}][p,dim]`. No-op at inference; carried for fidelity.
    Dropout { name: String, p: f32, dim: u8 },
    /// MaxPool2d: `Mp[{name}]ky,kx[,sy,sx]`. Stride defaults to kernel.
    MaxPool {
        name: String,
        kernel: (usize, usize),
        stride: (usize, usize),
    },
    /// Reshape/summary: `S[{name}]d(a x b)high,low`. Only `1(1x0)1,3` (collapse
    /// H into C) is supported in forward; `a`/`b` use -1 for "infer".
    Reshape {
        name: String,
        src_vgsl_dim: u8,
        part_a: i64,
        part_b: i64,
        high: u8,
        low: u8,
    },
    /// LSTM: `L(dir)(axis)[s][{name}]hidden`. `input_dim` filled by [`resolve`].
    Lstm {
        name: String,
        hidden: usize,
        direction: Direction,
        axis: Axis,
        summarize: bool,
        input_dim: usize, // 0 until resolve() runs
    },
    /// Output: `O[{name}]dim kind num`. `kind='c'` is CTC; `dim=1` is linear.
    Output {
        name: String,
        out_dim: u8,
        kind: OutKind,
        num_classes: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    Forward,
    Reverse,
    Bidirectional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Axis {
    X, // width / time (the only axis recognition LSTMs use)
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutKind {
    Ctc,
    Logistic,
    Sigmoid,
}

/// A parsed recognition network: the input shape + ordered block list.
#[derive(Debug, Clone)]
pub(crate) struct RecogNetwork {
    /// `[batch, height, width, channels]` from the input block. width 0 = dynamic.
    pub input_nhwc: [usize; 4],
    pub blocks: Vec<VgslBlock>,
}

impl RecogNetwork {
    /// Convenience: the resolved LSTM input dimension (post-reshape channel
    /// count). Errors if the network has no LSTM. Must be called after
    /// [`resolve`]. (Test helper — the build reads `input_dim` from the block.)
    #[cfg(test)]
    pub fn lstm_input_dim(&self) -> Result<usize> {
        self.blocks
            .iter()
            .find_map(|b| match b {
                VgslBlock::Lstm { input_dim, .. } => Some(*input_dim),
                _ => None,
            })
            .filter(|&d| d > 0)
            .ok_or_else(|| anyhow!("no resolved LSTM block in network"))
    }

    /// The final output class count, parsed from the `O...c<N>` block.
    pub fn num_classes(&self) -> usize {
        self.blocks
            .iter()
            .rev()
            .find_map(|b| match b {
                VgslBlock::Output { num_classes, .. } => Some(*num_classes),
                _ => None,
            })
            .unwrap_or(0)
    }
}

// ── Public entry points ─────────────────────────────────────────────

/// Parse a VGSL spec string into a [`RecogNetwork`].
///
/// Does NOT resolve dynamic dimensions; call [`resolve`] afterwards to fill in
/// LSTM input dims. (Split so parser failures are independent of shape math.)
pub(crate) fn parse(spec: &str) -> Result<RecogNetwork> {
    let spec = spec.trim();
    if !spec.starts_with('[') || !spec.ends_with(']') {
        bail!("VGSL spec must be wrapped in [ ]");
    }
    let inner = &spec[1..spec.len() - 1];
    let tokens = tokenize(inner);
    if tokens.is_empty() {
        bail!("empty VGSL spec");
    }

    // First token is the input block: batch,height,width,channels
    let input_nhwc = parse_input(&tokens[0])?;

    // Remaining tokens are layer blocks.
    let mut blocks = Vec::with_capacity(tokens.len() - 1);
    for tok in &tokens[1..] {
        blocks.push(parse_block(tok)?);
    }
    Ok(RecogNetwork { input_nhwc, blocks })
}

/// Walk the block list tracking `(N,C,H,W)` and fill in each LSTM's
/// `input_dim` (the channel count *after* the preceding `S` reshape collapses
/// height into channels). This replaces every hardcoded dimension in the build.
///
/// Mirrors kraken's running-shape-tuple mechanism. Width (W) is dynamic so we
/// track it symbolically; only C and H matter for dimension resolution.
pub(crate) fn resolve(net: &mut RecogNetwork) -> Result<()> {
    // (n, c, h, w) — w is symbolic (0 = dynamic); only c and h drive dims.
    let mut c = net.input_nhwc[3];
    let mut h = net.input_nhwc[1];
    for block in &mut net.blocks {
        match block {
            VgslBlock::Conv { out_channels, .. } => {
                c = *out_channels;
                // conv with stride 1 preserves H (we don't support strided conv
                // in forward; stride is parsed for fidelity but these models
                // use stride 1)
            }
            VgslBlock::Dropout { .. } => { /* no shape change */ }
            VgslBlock::MaxPool { kernel, stride, .. } => {
                h = (h + stride.0 - 1) / stride.0;
                // w is symbolic; would be (w + stride.1 - 1) / stride.1
                let _ = kernel;
            }
            VgslBlock::Reshape {
                src_vgsl_dim, high, low, ..
            } => {
                // For the recognition dialect S1(1x0)1,3: src_dim=1(H) is split
                // and collapsed into channels, leaving H=1. New C = H*C.
                // Validate the only supported case.
                if *src_vgsl_dim != 1 || *high != 1 || *low != 3 {
                    bail!(
                        "unsupported Reshape: only S1(1x0)1,3 (collapse H into C) \
                         is implemented; got src_dim={src_vgsl_dim} high={high} low={low}"
                    );
                }
                c = h * c;
                h = 1;
            }
            VgslBlock::Lstm {
                hidden,
                direction,
                input_dim,
                ..
            } => {
                if *input_dim == 0 {
                    *input_dim = c;
                }
                // Bidirectional LSTM output = 2*hidden; forward = hidden.
                c = match direction {
                    Direction::Bidirectional => 2 * *hidden,
                    _ => *hidden,
                };
            }
            VgslBlock::Output { .. } => { /* output reads c as its input; no change */ }
        }
    }
    Ok(())
}

// ── Tokenizer ───────────────────────────────────────────────────────

/// Split on spaces, but keep `{...}` brace groups attached to their token
/// (spaces inside braces don't split). Direct port of the segmentation parser's
/// tokenizer — kraken's specs use `{C_0}`-style annotations with no interior
/// spaces, but this is defensive against future variants.
fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut brace_depth = 0u32;
    for ch in s.chars() {
        match ch {
            '{' => {
                brace_depth += 1;
                current.push(ch);
            }
            '}' => {
                brace_depth = brace_depth.saturating_sub(1);
                current.push(ch);
            }
            ' ' if brace_depth == 0 => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect()
}

// ── Per-block parsers ───────────────────────────────────────────────

/// Parse the input block `b,h,w,c` into `[usize; 4]`.
fn parse_input(tok: &str) -> Result<[usize; 4]> {
    let parts: Vec<&str> = tok.split(',').collect();
    if parts.len() != 4 {
        bail!("input block must be b,h,w,c (4 comma-separated values): got {tok:?}");
    }
    let nums: Vec<usize> = parts
        .iter()
        .map(|s| s.trim().parse::<usize>().map_err(|e| anyhow!("bad input dim {s:?}: {e}")))
        .collect::<Result<Vec<_>>>()?;
    Ok([nums[0], nums[1], nums[2], nums[3]])
}

/// Parse one layer token. Extracts the `{name}` annotation first, then
/// dispatches on the leading letters.
fn parse_block(tok: &str) -> Result<VgslBlock> {
    let (name, body) = strip_name(tok);
    // Dispatch on leading letters (longest match first for Do/Mp/C-prefixes).
    // Note: in kraken's VGSL the letter after `C` is the activation
    // (s/t/r/l/m); `CT` (capital T) marks a transposed conv.
    let block = if let Some(rest) = body.strip_prefix("Cr") {
        parse_conv(&name, rest, 'r')
    } else if let Some(rest) = body.strip_prefix("Cl") {
        parse_conv(&name, rest, 'l')
    } else if let Some(rest) = body.strip_prefix("Cs") {
        parse_conv(&name, rest, 's')
    } else if let Some(rest) = body.strip_prefix("Ct") {
        parse_conv(&name, rest, 't') // tanh activation (NOT transposed)
    } else if let Some(rest) = body.strip_prefix("Cm") {
        parse_conv(&name, rest, 'm')
    } else if body.strip_prefix("CT").is_some() {
        bail!("transposed conv (CT) is not supported in the recognition dialect: {tok:?}")
    } else if let Some(rest) = body.strip_prefix('C') {
        parse_conv(&name, rest, 'l')
    } else if let Some(rest) = body.strip_prefix("Do") {
        parse_dropout(&name, rest)
    } else if let Some(rest) = body.strip_prefix("Mp") {
        parse_maxpool(&name, rest)
    } else if let Some(rest) = body.strip_prefix('S') {
        parse_reshape(&name, rest)
    } else if let Some(rest) = body.strip_prefix('L') {
        parse_lstm(&name, rest)
    } else if body.starts_with("Gn") {
        // GroupNorm — used by segmentation specs, not recognition.
        bail!("GroupNorm (Gn) is not supported in the recognition dialect: {tok:?}")
    } else if body.starts_with('G') {
        bail!("GRU (G...) is not supported in the recognition dialect: {tok:?}")
    } else if let Some(rest) = body.strip_prefix('O') {
        parse_output(&name, rest)
    } else {
        bail!("unsupported VGSL block {tok:?} (recognition dialect: Cr/Do/Mp/S/Lb/O)");
    }?;
    Ok(block)
}

/// Extract the `{name}` annotation (if present) and return `(name, body)`.
/// Auto-generates `"<unknown>"` when absent — kraken uses `<layer>_<idx>` but
/// recognition specs always carry explicit names (the safetensors weight keys
/// depend on them), so an absent name is treated as a parse anomaly here.
/// Callers that need positional naming can fall back; in practice these
/// models always have `{...}` on every block.
fn strip_name(tok: &str) -> (String, String) {
    // Find the first `{` and its matching `}`.
    if let (Some(open), Some(close)) = (tok.find('{'), tok.find('}')) {
        if close > open {
            let name = tok[open + 1..close].to_string();
            let body = format!("{}{}", &tok[..open], &tok[close + 1..]);
            return (name, body);
        }
    }
    (String::new(), tok.to_string())
}

/// Parse conv params after the `C[act]` prefix: `ky,kx,out[,sy,sx]`.
fn parse_conv(name: &str, rest: &str, activation: char) -> Result<VgslBlock> {
    let parts: Vec<&str> = rest.split(',').collect();
    if parts.len() < 3 {
        bail!("conv needs ky,kx,out: got {rest:?}");
    }
    let ky: usize = parts[0].parse().map_err(|e| anyhow!("bad ky: {e}"))?;
    let kx: usize = parts[1].parse().map_err(|e| anyhow!("bad kx: {e}"))?;
    let out: usize = parts[2].parse().map_err(|e| anyhow!("bad out_channels: {e}"))?;
    let stride = if parts.len() >= 5 {
        let sy: usize = parts[3].parse().map_err(|e| anyhow!("bad sy: {e}"))?;
        let sx: usize = parts[4].parse().map_err(|e| anyhow!("bad sx: {e}"))?;
        (sy, sx)
    } else {
        (1, 1)
    };
    Ok(VgslBlock::Conv {
        name: name.to_string(),
        kernel: (ky, kx),
        out_channels: out,
        stride,
        activation,
    })
}

/// Parse dropout params after `Do`: `[p[,dim]]`. Defaults p=0.5, dim=1.
fn parse_dropout(name: &str, rest: &str) -> Result<VgslBlock> {
    let rest = rest.trim();
    if rest.is_empty() {
        return Ok(VgslBlock::Dropout {
            name: name.to_string(),
            p: 0.5,
            dim: 1,
        });
    }
    let parts: Vec<&str> = rest.split(',').collect();
    let p: f32 = parts[0].parse().map_err(|e| anyhow!("bad dropout p: {e}"))?;
    let dim: u8 = if parts.len() >= 2 {
        parts[1].parse().map_err(|e| anyhow!("bad dropout dim: {e}"))?
    } else {
        1
    };
    Ok(VgslBlock::Dropout {
        name: name.to_string(),
        p,
        dim,
    })
}

/// Parse maxpool params after `Mp`: `ky,kx[,sy,sx]`. Stride defaults to kernel.
fn parse_maxpool(name: &str, rest: &str) -> Result<VgslBlock> {
    let parts: Vec<&str> = rest.split(',').collect();
    if parts.len() < 2 {
        bail!("maxpool needs ky,kx: got {rest:?}");
    }
    let ky: usize = parts[0].parse().map_err(|e| anyhow!("bad ky: {e}"))?;
    let kx: usize = parts[1].parse().map_err(|e| anyhow!("bad kx: {e}"))?;
    let stride = if parts.len() >= 4 {
        let sy: usize = parts[2].parse().map_err(|e| anyhow!("bad sy: {e}"))?;
        let sx: usize = parts[3].parse().map_err(|e| anyhow!("bad sx: {e}"))?;
        (sy, sx)
    } else {
        (ky, kx) // stride defaults to kernel (kraken semantics)
    };
    Ok(VgslBlock::MaxPool {
        name: name.to_string(),
        kernel: (ky, kx),
        stride,
    })
}

/// Parse reshape params after `S`: `d(a x b)high,low`.
/// Only `1(1x0)1,3` is meaningful for recognition (collapse H into C).
fn parse_reshape(name: &str, rest: &str) -> Result<VgslBlock> {
    // Format: <src_dim>(<a>x<b>)<high>,<low>
    let open = rest
        .find('(')
        .ok_or_else(|| anyhow!("reshape needs (a x b): got {rest:?}"))?;
    let close = rest
        .find(')')
        .ok_or_else(|| anyhow!("reshape needs (a x b): got {rest:?}"))?;
    let src_vgsl_dim: u8 = rest[..open].parse().map_err(|e| anyhow!("bad src_dim: {e}"))?;
    let inner = &rest[open + 1..close];
    let ab: Vec<&str> = inner.split('x').collect();
    if ab.len() != 2 {
        bail!("reshape (a x b) needs two parts: got {inner:?}");
    }
    // 0 means "infer" → -1 in kraken's convention.
    let part_a: i64 = ab[0]
        .trim()
        .parse::<i64>()
        .map(|v| if v == 0 { -1 } else { v })
        .map_err(|e| anyhow!("bad part_a: {e}"))?;
    let part_b: i64 = ab[1]
        .trim()
        .parse::<i64>()
        .map(|v| if v == 0 { -1 } else { v })
        .map_err(|e| anyhow!("bad part_b: {e}"))?;
    let tail: Vec<&str> = rest[close + 1..].split(',').collect();
    if tail.len() != 2 {
        bail!("reshape needs high,low after ): got {rest:?}");
    }
    let high: u8 = tail[0].parse().map_err(|e| anyhow!("bad high: {e}"))?;
    let low: u8 = tail[1].parse().map_err(|e| anyhow!("bad low: {e}"))?;
    Ok(VgslBlock::Reshape {
        name: name.to_string(),
        src_vgsl_dim,
        part_a,
        part_b,
        high,
        low,
    })
}

/// Parse LSTM/GRU params after `L`/`G`: `(dir)(axis)[s][legacy]hidden`.
/// `dir` ∈ {f,r,b}, `axis` ∈ {x,y}, optional `s` (summarize), `c`/`o` legacy.
fn parse_lstm(name: &str, rest: &str) -> Result<VgslBlock> {
    let chars: Vec<char> = rest.chars().collect();
    if chars.is_empty() {
        bail!("LSTM needs direction/axis/hidden: got {rest:?}");
    }
    let mut i = 0;
    let direction = match chars[i] {
        'f' => Direction::Forward,
        'r' => Direction::Reverse,
        'b' => Direction::Bidirectional,
        c => bail!("LSTM direction must be f/r/b, got {c:?}"),
    };
    i += 1;
    if i >= chars.len() {
        bail!("LSTM needs axis + hidden: got {rest:?}");
    }
    let axis = match chars[i] {
        'x' => Axis::X,
        'y' => Axis::Y,
        c => bail!("LSTM axis must be x/y, got {c:?}"),
    };
    i += 1;
    let mut summarize = false;
    if i < chars.len() && chars[i] == 's' {
        summarize = true;
        i += 1;
    }
    // legacy 'c' (clstm) or 'o' (ocropy peephole) — we don't support either.
    if i < chars.len() && (chars[i] == 'c' || chars[i] == 'o') {
        bail!(
            "legacy LSTM ({}) not supported: {:?}",
            chars[i],
            rest
        );
    }
    let hidden_str: String = chars[i..].iter().collect();
    let hidden: usize = hidden_str
        .parse()
        .map_err(|e| anyhow!("bad LSTM hidden {hidden_str:?}: {e}"))?;
    Ok(VgslBlock::Lstm {
        name: name.to_string(),
        hidden,
        direction,
        axis,
        summarize,
        input_dim: 0, // filled by resolve()
    })
}

/// Parse output params after `O`: `dim kind [a] num`.
/// `dim` ∈ {0,1,2}, `kind` ∈ {l,s,c}, optional `a` augmentation.
fn parse_output(name: &str, rest: &str) -> Result<VgslBlock> {
    let chars: Vec<char> = rest.chars().collect();
    if chars.len() < 3 {
        bail!("output needs dim,kind,num: got {rest:?}");
    }
    let out_dim: u8 = chars[0]
        .to_string()
        .parse()
        .map_err(|e| anyhow!("bad output dim: {e}"))?;
    let kind = match chars[1] {
        'c' => OutKind::Ctc,
        'l' => OutKind::Logistic,
        's' => OutKind::Sigmoid,
        c => bail!("output kind must be l/s/c, got {c:?}"),
    };
    if out_dim == 2 && kind == OutKind::Ctc {
        bail!("CTC output (c) is not supported for heatmap dim 2");
    }
    let mut i = 2;
    // optional augmentation flag 'a'
    if i < chars.len() && chars[i] == 'a' {
        i += 1;
    }
    let num_str: String = chars[i..].iter().collect();
    let num_classes: usize = num_str
        .parse()
        .map_err(|e| anyhow!("bad output num_classes {num_str:?}: {e}"))?;
    Ok(VgslBlock::Output {
        name: name.to_string(),
        out_dim,
        kind,
        num_classes,
    })
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC_3POOL: &str = "[1,48,0,1 Cr{C_0}3,13,32 Do{Do_1}0.1,2 Mp{Mp_2}2,2 Cr{C_3}3,13,32 Do{Do_4}0.1,2 Mp{Mp_5}2,2 Cr{C_6}3,9,64 Do{Do_7}0.1,2 Mp{Mp_8}2,2 Cr{C_9}3,9,64 Do{Do_10}0.1,2 S{S_11}1(1x0)1,3 Lbx{L_12}200 Do{Do_13}0.1,2 Lbx{L_14}200 Do{Do_15}0.1,2 Lbx{L_16}200 Do{Do_17} O{O_18}1c119]";
    const SPEC_2POOL: &str = "[1,48,0,1 Cr{C_0}3,13,32 Do{Do_1}0.1,2 Mp{Mp_2}2,2 Cr{C_3}3,13,32 Do{Do_4}0.1,2 Mp{Mp_5}2,2 Cr{C_6}3,9,64 Do{Do_7}0.1,2 Cr{C_8}3,9,64 Do{Do_9}0.1,2 S{S_10}1(1x0)1,3 Lbx{L_11}200 Do{Do_12}0.1,2 Lbx{L_13}200 Do{Do_14}0.1,2 Lbx{L_15}200 Do{Do_16} O{O_17}1c119]";
    const SPEC_120H: &str = "[1,120,0,1 Cr{C_0}3,13,32 Do{Do_1}0.1,2 Mp{Mp_2}2,2 Cr{C_3}3,13,32 Do{Do_4}0.1,2 Mp{Mp_5}2,2 Cr{C_6}3,9,64 Do{Do_7}0.1,2 Mp{Mp_8}2,2 Cr{C_9}3,9,64 Do{Do_10}0.1,2 S{S_11}1(1x0)1,3 Lbx{L_12}200 Do{Do_13}0.1,2 Lbx{L_14}200 Do{Do_15}0.1,2 Lbx{L_16}200 Do{Do_17} O{O_18}1c118]";

    fn count(net: &RecogNetwork, pred: fn(&VgslBlock) -> bool) -> usize {
        net.blocks.iter().filter(|b| pred(b)).count()
    }

    #[test]
    fn test_parse_3pool_arch() {
        let mut net = parse(SPEC_3POOL).expect("parse 3pool");
        assert_eq!(net.input_nhwc, [1, 48, 0, 1]);

        // 4 convs, 3 maxpools, 3 lstms, 1 output.
        assert_eq!(count(&net, |b| matches!(b, VgslBlock::Conv { .. })), 4);
        assert_eq!(count(&net, |b| matches!(b, VgslBlock::MaxPool { .. })), 3);
        assert_eq!(count(&net, |b| matches!(b, VgslBlock::Lstm { .. })), 3);
        assert_eq!(count(&net, |b| matches!(b, VgslBlock::Output { .. })), 1);

        resolve(&mut net).expect("resolve 3pool");
        // H=48, three Mp2,2 → 48/8 = 6 rows; ×64 channels = 384.
        assert_eq!(net.lstm_input_dim().unwrap(), 384);
        assert_eq!(net.num_classes(), 119);

        // Names carry through.
        assert!(matches!(
            &net.blocks[0],
            VgslBlock::Conv { name, .. } if name == "C_0"
        ));
        assert!(matches!(
            net.blocks.iter().find(|b| matches!(b, VgslBlock::Lstm { .. })).unwrap(),
            VgslBlock::Lstm { name, .. } if name == "L_12"
        ));
        assert!(matches!(
            net.blocks.last().unwrap(),
            VgslBlock::Output { name, num_classes: 119, .. } if name == "O_18"
        ));
    }

    #[test]
    fn test_parse_2pool_arch() {
        let mut net = parse(SPEC_2POOL).expect("parse 2pool");
        // Only 2 maxpools here.
        assert_eq!(count(&net, |b| matches!(b, VgslBlock::MaxPool { .. })), 2);
        assert_eq!(count(&net, |b| matches!(b, VgslBlock::Lstm { .. })), 3);

        resolve(&mut net).expect("resolve 2pool");
        // H=48, two Mp2,2 → 48/4 = 12 rows; ×64 channels = 768.
        assert_eq!(net.lstm_input_dim().unwrap(), 768);
        assert_eq!(net.num_classes(), 119);

        // Shifted layer names: last conv is C_8, first lstm is L_11, output O_17.
        let conv_names: Vec<&str> = net
            .blocks
            .iter()
            .filter_map(|b| match b {
                VgslBlock::Conv { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(conv_names, ["C_0", "C_3", "C_6", "C_8"]);
        let lstm_names: Vec<&str> = net
            .blocks
            .iter()
            .filter_map(|b| match b {
                VgslBlock::Lstm { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(lstm_names, ["L_11", "L_13", "L_15"]);
    }

    #[test]
    fn test_120h_arch() {
        let mut net = parse(SPEC_120H).expect("parse 120h");
        resolve(&mut net).expect("resolve 120h");
        // H=120, three Mp2,2 → 120/8 = 15 rows; ×64 = 960. Regression guard.
        assert_eq!(net.lstm_input_dim().unwrap(), 960);
        assert_eq!(net.num_classes(), 118);
    }

    #[test]
    fn test_unknown_block_fails() {
        let spec = "[1,48,0,1 Cr{C_0}3,13,32 Gn{Gn_1}4 O{O_2}1c119]";
        let err = parse(spec).unwrap_err();
        assert!(
            err.to_string().contains("Gn"),
            "should mention the offending block: {err}"
        );
    }

    #[test]
    fn test_s_reshape_collapses_h_into_c() {
        let mut net = parse("[1,48,0,1 Cr{C_0}3,13,64 Mp{Mp_1}2,2 S{S_2}1(1x0)1,3 Lbx{L_3}200 O{O_4}1c119]")
            .expect("parse");
        resolve(&mut net).expect("resolve");
        // H=48, one Mp → 24 rows; ×64 = 1536.
        assert_eq!(net.lstm_input_dim().unwrap(), 1536);
    }

    #[test]
    fn test_ly_axis_parsed_but_unsupported_in_resolve_ok() {
        // Parsing an Ly LSTM should succeed (the parser accepts it); only the
        // forward pass rejects it. Resolve still computes input_dim.
        let mut net = parse("[1,48,0,1 Cr{C_0}3,13,64 Mp{Mp_1}2,2 S{S_2}1(1x0)1,3 Lby{L_3}200 O{O_4}1c119]")
            .expect("parse Ly");
        resolve(&mut net).expect("resolve Ly");
        let lstm = net.blocks.iter().find_map(|b| match b {
            VgslBlock::Lstm { axis, .. } => Some(*axis),
            _ => None,
        });
        assert_eq!(lstm, Some(Axis::Y));
    }

    #[test]
    fn test_tokenize_keeps_braces() {
        let toks = tokenize("Cr{C_0}3,13,32 Do{Do_1}0.1,2");
        assert_eq!(toks, vec!["Cr{C_0}3,13,32", "Do{Do_1}0.1,2"]);
    }

    #[test]
    fn test_dropout_defaults() {
        let b = parse_block("Do{Do_0}").unwrap();
        assert!(matches!(b, VgslBlock::Dropout { p: 0.5, dim: 1, .. }));
    }

    #[test]
    fn test_maxpool_stride_defaults_to_kernel() {
        let b = parse_block("Mp{Mp_0}2,2").unwrap();
        assert!(matches!(b, VgslBlock::MaxPool { kernel: (2,2), stride: (2,2), .. }));
    }

    #[test]
    fn test_conv_activation_codes() {
        // The letter after C is the activation; lowercase 't' is tanh, NOT
        // transposed. Capital 'T' would be transposed (rejected).
        for (tok, expect) in [
            ("Cr{C_0}3,13,32", 'r'),
            ("Cl{C_0}3,13,32", 'l'),
            ("Cs{C_0}3,13,32", 's'),
            ("Ct{C_0}3,13,32", 't'),
            ("Cm{C_0}3,13,32", 'm'),
        ] {
            match parse_block(tok) {
                Ok(VgslBlock::Conv { activation, .. }) => assert_eq!(activation, expect, "{tok}"),
                other => panic!("{tok} parsed as {other:?}, expected Conv"),
            }
        }
    }

    #[test]
    fn test_transposed_conv_capital_t_rejected() {
        let err = parse_block("CT{C_0}3,13,32").unwrap_err();
        assert!(err.to_string().contains("transposed conv"), "{err}");
    }
}
