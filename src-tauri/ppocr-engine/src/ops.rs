use super::{
    arena::Buffer,
    kernels::{self, UnaryOperation},
    tensor::{Tensor, TensorData, element_count, strides},
};
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::sync::Arc;
#[cfg(feature = "cpu-profile")]
use std::time::Instant;

const SPATIAL_PANEL_COLUMNS: usize = 16;

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub name: String,
    pub operation: Operation,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) enum Operation {
    Add,
    AveragePool(PoolOptions),
    BatchNormalization { epsilon: f32 },
    BiasSoftmax { axis: i64 },
    Concat { axis: i64 },
    Conv(ConvOptions),
    ConvGelu(ConvOptions),
    ConvRelu(ConvOptions),
    ConvSilu(ConvOptions),
    ConvTranspose(ConvOptions),
    Div,
    Erf,
    Gelu,
    GlobalAveragePool,
    HardSigmoid { alpha: f32, beta: f32 },
    HardSwish,
    MatMul,
    MatMulBiasSoftmax { axis: i64 },
    MaxPool(PoolOptions),
    Mul,
    Pow,
    ReduceMean { axes: Vec<i64>, keep_dims: bool },
    Relu,
    Reshape,
    Resize,
    Shape,
    Sigmoid,
    Silu,
    Slice,
    Softmax { axis: i64 },
    Sqrt,
    Squeeze { axes: Vec<i64> },
    Sub,
    Transpose { permutation: Vec<usize> },
    Unsqueeze { axes: Vec<i64> },
}

#[derive(Clone, Debug)]
pub(crate) struct ConvOptions {
    pub strides: [usize; 2],
    pub pads: [usize; 4],
    pub groups: usize,
    pub direct_spatial: bool,
    pub packed_pointwise: bool,
    pub blocked_pointwise: bool,
    pub system_dense_pointwise: bool,
    pub system_dense_spatial: bool,
    pub exact_sparse_weights: Option<ExactSparseConvWeights>,
}

#[derive(Clone, Debug)]
pub(crate) struct ExactSparseConvWeights {
    row_offsets: Arc<Vec<usize>>,
    indices: Arc<Vec<u32>>,
    values: Arc<Vec<f32>>,
}

impl ExactSparseConvWeights {
    pub(crate) fn from_dense(weight: &[f32], rows: usize, inner: usize) -> Option<Self> {
        const BLOCK_ROWS: usize = 4;
        const MAXIMUM_ACTIVE_RATIO: f32 = 0.55;

        if !kernels::supports_exact_sparse_gemm()
            || !rows.is_multiple_of(BLOCK_ROWS)
            || weight.len() != rows * inner
            || inner > u32::MAX as usize
        {
            return None;
        }
        let mut row_offsets = Vec::with_capacity(rows / BLOCK_ROWS + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for row_start in (0..rows).step_by(BLOCK_ROWS) {
            for index in 0..inner {
                let block = std::array::from_fn::<_, BLOCK_ROWS, _>(|row| {
                    weight[(row_start + row) * inner + index]
                });
                if block.iter().any(|value| *value != 0.0) {
                    indices.push(index as u32);
                    values.extend_from_slice(&block);
                }
            }
            row_offsets.push(indices.len());
        }
        let possible = rows / BLOCK_ROWS * inner;
        if indices.len() as f32 > possible as f32 * MAXIMUM_ACTIVE_RATIO {
            return None;
        }
        Some(Self {
            row_offsets: Arc::new(row_offsets),
            indices: Arc::new(indices),
            values: Arc::new(values),
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PoolOptions {
    pub kernel: [usize; 2],
    pub strides: [usize; 2],
    pub pads: [usize; 4],
    pub ceil_mode: bool,
    pub count_include_pad: bool,
}

impl Node {
    pub(crate) fn run(&self, inputs: Vec<Tensor>) -> Result<Tensor> {
        #[cfg(feature = "cpu-profile")]
        let started = Instant::now();
        let result = match &self.operation {
            Operation::Add => binary(inputs, BinaryOperation::Add),
            Operation::AveragePool(options) => pool(inputs, options, false),
            Operation::BatchNormalization { epsilon } => batch_normalization(inputs, *epsilon),
            Operation::BiasSoftmax { axis } => bias_softmax(inputs, *axis),
            Operation::Concat { axis } => concat(inputs, *axis),
            Operation::Conv(options) => conv(inputs, options, None),
            Operation::ConvGelu(options) => conv(inputs, options, Some(UnaryOperation::Gelu)),
            Operation::ConvRelu(options) => conv(inputs, options, Some(UnaryOperation::Relu)),
            Operation::ConvSilu(options) => conv(inputs, options, Some(UnaryOperation::Silu)),
            Operation::ConvTranspose(options) => conv_transpose(inputs, options),
            Operation::Div => binary(inputs, BinaryOperation::Div),
            Operation::Erf => unary(inputs, UnaryOperation::Erf),
            Operation::Gelu => unary(inputs, UnaryOperation::Gelu),
            Operation::GlobalAveragePool => global_average_pool(inputs),
            Operation::HardSigmoid { alpha, beta } => unary(
                inputs,
                UnaryOperation::HardSigmoid {
                    alpha: *alpha,
                    beta: *beta,
                },
            ),
            Operation::HardSwish => unary(inputs, UnaryOperation::HardSwish),
            Operation::MatMul => matmul(inputs),
            Operation::MatMulBiasSoftmax { axis } => matmul_bias_softmax(inputs, *axis),
            Operation::MaxPool(options) => pool(inputs, options, true),
            Operation::Mul => binary(inputs, BinaryOperation::Mul),
            Operation::Pow => binary(inputs, BinaryOperation::Pow),
            Operation::ReduceMean { axes, keep_dims } => reduce_mean(inputs, axes, *keep_dims),
            Operation::Relu => unary(inputs, UnaryOperation::Relu),
            Operation::Reshape => reshape(inputs),
            Operation::Resize => resize(inputs),
            Operation::Shape => shape(inputs),
            Operation::Sigmoid => unary(inputs, UnaryOperation::Sigmoid),
            Operation::Silu => unary(inputs, UnaryOperation::Silu),
            Operation::Slice => slice(inputs),
            Operation::Softmax { axis } => softmax(inputs, *axis),
            Operation::Sqrt => unary(inputs, UnaryOperation::Sqrt),
            Operation::Squeeze { axes } => squeeze(inputs, axes),
            Operation::Sub => binary(inputs, BinaryOperation::Sub),
            Operation::Transpose { permutation } => transpose(inputs, permutation),
            Operation::Unsqueeze { axes } => unsqueeze(inputs, axes),
        };
        #[cfg(feature = "cpu-profile")]
        if let Ok(output) = &result {
            profile_operation(&self.operation, output, started.elapsed());
        }
        result.with_context(|| format!("execute {} ({:?})", self.name, self.operation))
    }
}

#[cfg(feature = "cpu-profile")]
fn profile_operation(operation: &Operation, output: &Tensor, elapsed: std::time::Duration) {
    eprintln!(
        "cpu-profile operation={} output={:?} elapsed_ms={:.6}",
        operation_name(operation),
        output.shape(),
        elapsed.as_secs_f64() * 1_000.0
    );
}

#[cfg(feature = "cpu-profile")]
const fn operation_name(operation: &Operation) -> &'static str {
    match operation {
        Operation::Add => "Add",
        Operation::AveragePool(_) => "AveragePool",
        Operation::BatchNormalization { .. } => "BatchNormalization",
        Operation::BiasSoftmax { .. } => "BiasSoftmax",
        Operation::Concat { .. } => "Concat",
        Operation::Conv(_) => "Conv",
        Operation::ConvGelu(_) => "ConvGelu",
        Operation::ConvRelu(_) => "ConvRelu",
        Operation::ConvSilu(_) => "ConvSilu",
        Operation::ConvTranspose(_) => "ConvTranspose",
        Operation::Div => "Div",
        Operation::Erf => "Erf",
        Operation::Gelu => "Gelu",
        Operation::GlobalAveragePool => "GlobalAveragePool",
        Operation::HardSigmoid { .. } => "HardSigmoid",
        Operation::HardSwish => "HardSwish",
        Operation::MatMul => "MatMul",
        Operation::MatMulBiasSoftmax { .. } => "MatMulBiasSoftmax",
        Operation::MaxPool(_) => "MaxPool",
        Operation::Mul => "Mul",
        Operation::Pow => "Pow",
        Operation::ReduceMean { .. } => "ReduceMean",
        Operation::Relu => "Relu",
        Operation::Reshape => "Reshape",
        Operation::Resize => "Resize",
        Operation::Shape => "Shape",
        Operation::Sigmoid => "Sigmoid",
        Operation::Silu => "Silu",
        Operation::Slice => "Slice",
        Operation::Softmax { .. } => "Softmax",
        Operation::Sqrt => "Sqrt",
        Operation::Squeeze { .. } => "Squeeze",
        Operation::Sub => "Sub",
        Operation::Transpose { .. } => "Transpose",
        Operation::Unsqueeze { .. } => "Unsqueeze",
    }
}

fn unary(mut inputs: Vec<Tensor>, operation: UnaryOperation) -> Result<Tensor> {
    ensure_input_count(&inputs, 1)?;
    let mut input = inputs.pop().expect("one input");
    kernels::unary_in_place(input.f32_mut()?, operation);
    Ok(input)
}

#[derive(Clone, Copy)]
enum BinaryOperation {
    Add,
    Div,
    Mul,
    Pow,
    Sub,
}

fn binary(mut inputs: Vec<Tensor>, operation: BinaryOperation) -> Result<Tensor> {
    ensure_input_count(&inputs, 2)?;
    let right = inputs.pop().expect("right input");
    let left = inputs.pop().expect("left input");
    let output_shape = broadcast_shape(&left.shape, &right.shape)?;
    let right_values = right.as_f32()?;

    if left.shape == output_shape && right.shape == output_shape {
        let mut output = left;
        let output_values = output.f32_mut()?;
        if apply_binary_vector(operation, output_values, right_values) {
            return Ok(output);
        } else {
            output_values
                .par_iter_mut()
                .zip(right_values.par_iter())
                .for_each(|(left, right)| *left = apply_binary(operation, *left, *right));
        }
        return Ok(output);
    }

    if right_values.len() == 1 && left.shape == output_shape {
        let right = right_values[0];
        let mut output = left;
        let output_values = output.f32_mut()?;
        if !apply_binary_scalar(operation, output_values, right) {
            output_values
                .par_iter_mut()
                .for_each(|left| *left = apply_binary(operation, *left, right));
        }
        return Ok(output);
    }

    if left.shape == output_shape && is_repeated_suffix(&right.shape, &output_shape) {
        let mut output = left;
        for chunk in output.f32_mut()?.chunks_mut(right_values.len()) {
            if !apply_binary_vector(operation, chunk, right_values) {
                for (left, right) in chunk.iter_mut().zip(right_values) {
                    *left = apply_binary(operation, *left, *right);
                }
            }
        }
        return Ok(output);
    }

    if left.shape == output_shape
        && output_shape.len() == 4
        && right.shape.len() == 4
        && right.shape[0] == 1
        && right.shape[1] == output_shape[1]
        && right.shape[2..] == [1, 1]
    {
        let mut output = left;
        let channel_size = output_shape[2] * output_shape[3];
        for (channel, values) in output.f32_mut()?.chunks_mut(channel_size).enumerate() {
            let right = right_values[channel % output_shape[1]];
            if !apply_binary_scalar(operation, values, right) {
                for value in values {
                    *value = apply_binary(operation, *value, right);
                }
            }
        }
        return Ok(output);
    }

    let left_values = left.as_f32()?;
    let output_len = element_count(&output_shape).context("broadcast output shape overflow")?;
    let left_strides = broadcast_strides(&left.shape, &output_shape)?;
    let right_strides = broadcast_strides(&right.shape, &output_shape)?;
    let output_strides = strides(&output_shape);
    let mut output = Buffer::for_overwrite(output_len);
    output.par_iter_mut().enumerate().for_each(|(flat, value)| {
        let mut remainder = flat;
        let mut left_index = 0;
        let mut right_index = 0;
        for dimension in 0..output_shape.len() {
            let coordinate = remainder / output_strides[dimension];
            remainder %= output_strides[dimension];
            left_index += coordinate * left_strides[dimension];
            right_index += coordinate * right_strides[dimension];
        }
        *value = apply_binary(
            operation,
            left_values[left_index],
            right_values[right_index],
        );
    });
    Ok(Tensor::new_f32(output_shape, output))
}

#[inline]
fn apply_binary(operation: BinaryOperation, left: f32, right: f32) -> f32 {
    match operation {
        BinaryOperation::Add => left + right,
        BinaryOperation::Div => left / right,
        BinaryOperation::Mul => left * right,
        BinaryOperation::Pow if right == 2.0 => left * left,
        BinaryOperation::Pow => left.powf(right),
        BinaryOperation::Sub => left - right,
    }
}

fn apply_binary_vector(operation: BinaryOperation, left: &mut [f32], right: &[f32]) -> bool {
    match operation {
        BinaryOperation::Add => kernels::add_in_place(left, right),
        BinaryOperation::Mul => kernels::mul_in_place(left, right),
        _ => return false,
    }
    true
}

fn apply_binary_scalar(operation: BinaryOperation, values: &mut [f32], right: f32) -> bool {
    match operation {
        BinaryOperation::Add => kernels::affine_in_place(values, 1.0, right),
        BinaryOperation::Div => kernels::affine_in_place(values, right.recip(), 0.0),
        BinaryOperation::Mul => kernels::affine_in_place(values, right, 0.0),
        BinaryOperation::Pow if right == 2.0 => kernels::square_in_place(values),
        BinaryOperation::Sub => kernels::affine_in_place(values, 1.0, -right),
        _ => return false,
    }
    true
}

fn is_repeated_suffix(input: &[usize], output: &[usize]) -> bool {
    let first_non_unit = input
        .iter()
        .position(|&dimension| dimension != 1)
        .unwrap_or(input.len());
    let suffix = &input[first_non_unit..];
    !suffix.is_empty()
        && suffix.len() <= output.len()
        && suffix == &output[output.len() - suffix.len()..]
}

fn conv(
    inputs: Vec<Tensor>,
    options: &ConvOptions,
    activation: Option<UnaryOperation>,
) -> Result<Tensor> {
    ensure!(
        (2..=3).contains(&inputs.len()),
        "Conv expects two or three inputs"
    );
    let input = inputs[0].as_f32()?;
    let weight = inputs[1].as_f32()?;
    let bias = inputs.get(2).map(Tensor::as_f32).transpose()?;
    let [batch, input_channels, input_height, input_width] = shape4(&inputs[0].shape)?;
    let [
        output_channels,
        channels_per_group,
        kernel_height,
        kernel_width,
    ] = shape4(&inputs[1].shape)?;
    ensure!(options.groups > 0, "Conv group count must be positive");
    ensure!(
        input_channels == channels_per_group * options.groups,
        "Conv input channels do not match weights and groups"
    );
    ensure!(
        output_channels % options.groups == 0,
        "Conv output channels are not divisible by groups"
    );
    if let Some(bias) = bias {
        ensure!(
            bias.len() == output_channels,
            "Conv bias has incorrect length"
        );
    }
    let output_height = conv_output_size(
        input_height,
        kernel_height,
        options.strides[0],
        options.pads[0],
        options.pads[2],
        false,
    )?;
    let output_width = conv_output_size(
        input_width,
        kernel_width,
        options.strides[1],
        options.pads[1],
        options.pads[3],
        false,
    )?;
    let input_plane = input_height * input_width;
    let output_plane = output_height * output_width;
    let channels_per_output_group = output_channels / options.groups;
    let mut output = Buffer::for_overwrite(batch * output_channels * output_plane);

    if kernel_height == 1
        && kernel_width == 1
        && options.strides == [1, 1]
        && options.pads == [0; 4]
        && options.groups == 1
    {
        for batch_index in 0..batch {
            let output = &mut output[batch_index * output_channels * output_plane
                ..(batch_index + 1) * output_channels * output_plane];
            let input = &input[batch_index * input_channels * input_plane
                ..(batch_index + 1) * input_channels * input_plane];
            if let Some(sparse) = &options.exact_sparse_weights {
                kernels::gemm_sparse_packed_left(
                    output,
                    input,
                    output_channels,
                    input_channels,
                    output_plane,
                    bias,
                    &sparse.row_offsets,
                    &sparse.indices,
                    &sparse.values,
                    activation,
                );
            } else if options.system_dense_pointwise {
                kernels::gemm_system_dense(
                    output,
                    weight,
                    input,
                    output_channels,
                    input_channels,
                    output_plane,
                    bias,
                    activation,
                );
            } else if options.blocked_pointwise {
                kernels::gemm_packed_left_blocked_6(
                    output,
                    weight,
                    input,
                    output_channels,
                    input_channels,
                    output_plane,
                    bias,
                    activation,
                );
            } else if options.packed_pointwise {
                kernels::gemm_packed_left_with_activation(
                    output,
                    weight,
                    input,
                    output_channels,
                    input_channels,
                    output_plane,
                    bias,
                    activation,
                );
            } else {
                kernels::gemm_with_activation(
                    output,
                    weight,
                    input,
                    output_channels,
                    input_channels,
                    output_plane,
                    bias,
                    activation,
                );
            }
        }
        return Ok(Tensor::new_f32(
            vec![batch, output_channels, output_height, output_width],
            output,
        ));
    }

    if options.direct_spatial {
        for batch_index in 0..batch {
            kernels::spatial_conv2d_direct(
                &mut output[batch_index * output_channels * output_plane
                    ..(batch_index + 1) * output_channels * output_plane],
                &input[batch_index * input_channels * input_plane
                    ..(batch_index + 1) * input_channels * input_plane],
                weight,
                bias,
                input_channels,
                input_height,
                input_width,
                output_channels,
                output_height,
                output_width,
                kernel_height,
                kernel_width,
                options.strides,
                options.pads,
                activation,
            );
        }
        return Ok(Tensor::new_f32(
            vec![batch, output_channels, output_height, output_width],
            output,
        ));
    }

    if options.system_dense_spatial {
        let patch_size = input_channels * kernel_height * kernel_width;
        for batch_index in 0..batch {
            conv_im2col_system_tiled(
                &mut output[batch_index * output_channels * output_plane
                    ..(batch_index + 1) * output_channels * output_plane],
                &input[batch_index * input_channels * input_plane
                    ..(batch_index + 1) * input_channels * input_plane],
                weight,
                bias,
                input_channels,
                input_height,
                input_width,
                output_channels,
                output_height,
                output_width,
                kernel_height,
                kernel_width,
                patch_size,
                options,
                activation,
            );
        }
        return Ok(Tensor::new_f32(
            vec![batch, output_channels, output_height, output_width],
            output,
        ));
    }

    if options.groups == 1
        && options.strides != [1, 1]
        && !((kernel_height != 1 || kernel_width != 1)
            && input_channels * kernel_height * kernel_width >= 128
            && output_channels >= 16)
    {
        let patch_size = input_channels * kernel_height * kernel_width;
        for batch_index in 0..batch {
            let columns = im2col(
                &input[batch_index * input_channels * input_plane
                    ..(batch_index + 1) * input_channels * input_plane],
                input_channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                output_height,
                output_width,
                options,
            );
            let output = &mut output[batch_index * output_channels * output_plane
                ..(batch_index + 1) * output_channels * output_plane];
            if options.packed_pointwise {
                kernels::gemm_packed_left_with_activation(
                    output,
                    weight,
                    &columns,
                    output_channels,
                    patch_size,
                    output_plane,
                    bias,
                    activation,
                );
            } else {
                kernels::gemm_with_activation(
                    output,
                    weight,
                    &columns,
                    output_channels,
                    patch_size,
                    output_plane,
                    bias,
                    activation,
                );
            }
        }
        return Ok(Tensor::new_f32(
            vec![batch, output_channels, output_height, output_width],
            output,
        ));
    }

    if options.groups == 1
        && (kernel_height != 1 || kernel_width != 1)
        && input_channels * kernel_height * kernel_width >= 128
        && output_channels >= 16
    {
        let patch_size = input_channels * kernel_height * kernel_width;
        for batch_index in 0..batch {
            conv_im2col_tiled(
                &mut output[batch_index * output_channels * output_plane
                    ..(batch_index + 1) * output_channels * output_plane],
                &input[batch_index * input_channels * input_plane
                    ..(batch_index + 1) * input_channels * input_plane],
                weight,
                bias,
                input_channels,
                input_height,
                input_width,
                output_channels,
                output_height,
                output_width,
                kernel_height,
                kernel_width,
                patch_size,
                options,
                activation,
            );
        }
        return Ok(Tensor::new_f32(
            vec![batch, output_channels, output_height, output_width],
            output,
        ));
    }

    let depthwise_padding = kernel_height / 2;
    if options.groups == input_channels
        && output_channels == input_channels
        && channels_per_group == 1
        && options.strides == [1, 1]
        && kernel_height == kernel_width
        && matches!(kernel_height, 3 | 5 | 7 | 9)
        && options.pads == [depthwise_padding; 4]
    {
        debug_assert_eq!(input_plane, output_plane);
        output.par_chunks_mut(output_plane).enumerate().for_each(
            |(plane_index, output_plane_values)| {
                let output_channel = plane_index % output_channels;
                let input_base = plane_index * input_plane;
                let weight_base = output_channel * kernel_height * kernel_width;
                kernels::depthwise_conv2d_same(
                    output_plane_values,
                    &input[input_base..input_base + input_plane],
                    &weight[weight_base..weight_base + kernel_height * kernel_width],
                    input_height,
                    input_width,
                    kernel_height,
                    bias.map_or(0.0, |bias| bias[output_channel]),
                );
                if let Some(activation) = activation {
                    kernels::unary_in_place(output_plane_values, activation);
                }
            },
        );
        return Ok(Tensor::new_f32(
            vec![batch, output_channels, output_height, output_width],
            output,
        ));
    }

    if options.groups == input_channels
        && output_channels == input_channels
        && channels_per_group == 1
        && options.strides == [2, 2]
        && kernel_height == 3
        && kernel_width == 3
        && options.pads == [1; 4]
    {
        output.par_chunks_mut(output_plane).enumerate().for_each(
            |(plane_index, output_plane_values)| {
                let output_channel = plane_index % output_channels;
                let input_base = plane_index * input_plane;
                let weight_base = output_channel * 9;
                kernels::depthwise_conv2d_same_3x3_stride2(
                    output_plane_values,
                    &input[input_base..input_base + input_plane],
                    &weight[weight_base..weight_base + 9],
                    input_height,
                    input_width,
                    bias.map_or(0.0, |bias| bias[output_channel]),
                );
                if let Some(activation) = activation {
                    kernels::unary_in_place(output_plane_values, activation);
                }
            },
        );
        return Ok(Tensor::new_f32(
            vec![batch, output_channels, output_height, output_width],
            output,
        ));
    }

    output.par_chunks_mut(output_plane).enumerate().for_each(
        |(plane_index, output_plane_values)| {
            let batch_index = plane_index / output_channels;
            let output_channel = plane_index % output_channels;
            kernels::fill(
                output_plane_values,
                bias.map_or(0.0, |bias| bias[output_channel]),
            );
            let group = output_channel / channels_per_output_group;
            let input_channel_start = group * channels_per_group;
            let weight_channel_start = output_channel * channels_per_group;

            if options.strides == [1, 1] {
                for channel_offset in 0..channels_per_group {
                    let input_channel = input_channel_start + channel_offset;
                    let input_base = (batch_index * input_channels + input_channel) * input_plane;
                    let weight_base =
                        (weight_channel_start + channel_offset) * kernel_height * kernel_width;
                    for kernel_y in 0..kernel_height {
                        let output_y_start = options.pads[0].saturating_sub(kernel_y);
                        let output_y_end = output_height.min(
                            input_height
                                .saturating_add(options.pads[0])
                                .saturating_sub(kernel_y),
                        );
                        for output_y in output_y_start..output_y_end {
                            let input_y = output_y + kernel_y - options.pads[0];
                            for kernel_x in 0..kernel_width {
                                let output_x_start = options.pads[1].saturating_sub(kernel_x);
                                let output_x_end = output_width.min(
                                    input_width
                                        .saturating_add(options.pads[1])
                                        .saturating_sub(kernel_x),
                                );
                                if output_x_start >= output_x_end {
                                    continue;
                                }
                                let input_x = output_x_start + kernel_x - options.pads[1];
                                let source_start = input_base + input_y * input_width + input_x;
                                let destination_start = output_y * output_width + output_x_start;
                                let len = output_x_end - output_x_start;
                                kernels::axpy(
                                    &mut output_plane_values
                                        [destination_start..destination_start + len],
                                    &input[source_start..source_start + len],
                                    weight[weight_base + kernel_y * kernel_width + kernel_x],
                                );
                            }
                        }
                    }
                }
            } else {
                for output_y in 0..output_height {
                    for output_x in 0..output_width {
                        let mut sum = output_plane_values[output_y * output_width + output_x];
                        for channel_offset in 0..channels_per_group {
                            let input_channel = input_channel_start + channel_offset;
                            let input_base =
                                (batch_index * input_channels + input_channel) * input_plane;
                            let weight_base = (weight_channel_start + channel_offset)
                                * kernel_height
                                * kernel_width;
                            for kernel_y in 0..kernel_height {
                                let input_y = output_y * options.strides[0] + kernel_y;
                                if input_y < options.pads[0]
                                    || input_y - options.pads[0] >= input_height
                                {
                                    continue;
                                }
                                let input_y = input_y - options.pads[0];
                                for kernel_x in 0..kernel_width {
                                    let input_x = output_x * options.strides[1] + kernel_x;
                                    if input_x < options.pads[1]
                                        || input_x - options.pads[1] >= input_width
                                    {
                                        continue;
                                    }
                                    let input_x = input_x - options.pads[1];
                                    sum = input[input_base + input_y * input_width + input_x]
                                        .mul_add(
                                            weight
                                                [weight_base + kernel_y * kernel_width + kernel_x],
                                            sum,
                                        );
                                }
                            }
                        }
                        output_plane_values[output_y * output_width + output_x] = sum;
                    }
                }
            }
            if let Some(activation) = activation {
                kernels::unary_in_place(output_plane_values, activation);
            }
        },
    );

    Ok(Tensor::new_f32(
        vec![batch, output_channels, output_height, output_width],
        output,
    ))
}

#[allow(clippy::too_many_arguments)]
fn conv_im2col_system_tiled(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    output_channels: usize,
    output_height: usize,
    output_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    patch_size: usize,
    options: &ConvOptions,
    activation: Option<UnaryOperation>,
) {
    const TARGET_COLUMN_ELEMENTS: usize = 4 * 1024 * 1024;

    let output_plane = output_height * output_width;
    let rows_per_tile =
        (TARGET_COLUMN_ELEMENTS / (patch_size * output_width)).clamp(1, output_height);
    let maximum_tile_plane = rows_per_tile * output_width;
    let mut columns = Buffer::for_overwrite(patch_size * maximum_tile_plane);
    let mut tile_output = Buffer::for_overwrite(output_channels * maximum_tile_plane);
    for output_y_start in (0..output_height).step_by(rows_per_tile) {
        let tile_rows = (output_height - output_y_start).min(rows_per_tile);
        let tile_plane = tile_rows * output_width;
        let columns = &mut columns[..patch_size * tile_plane];
        im2col_tile(
            columns,
            input,
            input_channels,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            output_y_start,
            tile_rows,
            output_width,
            options,
        );
        let tile_output = &mut tile_output[..output_channels * tile_plane];
        kernels::gemm_system_dense(
            tile_output,
            weight,
            columns,
            output_channels,
            patch_size,
            tile_plane,
            bias,
            activation,
        );
        output
            .par_chunks_mut(output_plane)
            .zip(tile_output.par_chunks(tile_plane))
            .for_each(|(output, tile)| {
                let start = output_y_start * output_width;
                output[start..start + tile_plane].copy_from_slice(tile);
            });
    }
}

#[allow(clippy::too_many_arguments)]
fn im2col_tile(
    columns: &mut [f32],
    input: &[f32],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    output_y_start: usize,
    output_rows: usize,
    output_width: usize,
    options: &ConvOptions,
) {
    let input_plane = input_height * input_width;
    let tile_plane = output_rows * output_width;
    debug_assert_eq!(
        columns.len(),
        input_channels * kernel_height * kernel_width * tile_plane
    );
    columns
        .par_chunks_mut(tile_plane)
        .enumerate()
        .for_each(|(patch_index, output)| {
            output.fill(0.0);
            let kernel_x = patch_index % kernel_width;
            let patch = patch_index / kernel_width;
            let kernel_y = patch % kernel_height;
            let channel = patch / kernel_height;
            let input = &input[channel * input_plane..(channel + 1) * input_plane];
            for local_y in 0..output_rows {
                let input_y = (output_y_start + local_y) * options.strides[0] + kernel_y;
                if input_y < options.pads[0] || input_y - options.pads[0] >= input_height {
                    continue;
                }
                let input_y = input_y - options.pads[0];
                let output_row = &mut output[local_y * output_width..(local_y + 1) * output_width];
                if options.strides[1] == 1 {
                    let output_x_start = options.pads[1].saturating_sub(kernel_x);
                    let output_x_end = output_width.min(
                        input_width
                            .saturating_add(options.pads[1])
                            .saturating_sub(kernel_x),
                    );
                    if output_x_start < output_x_end {
                        let input_x = output_x_start + kernel_x - options.pads[1];
                        let len = output_x_end - output_x_start;
                        let source = input_y * input_width + input_x;
                        output_row[output_x_start..output_x_end]
                            .copy_from_slice(&input[source..source + len]);
                    }
                } else {
                    for (output_x, destination) in output_row.iter_mut().enumerate() {
                        let input_x = output_x * options.strides[1] + kernel_x;
                        if input_x >= options.pads[1] && input_x - options.pads[1] < input_width {
                            *destination = input[input_y * input_width + input_x - options.pads[1]];
                        }
                    }
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn conv_im2col_tiled(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    output_channels: usize,
    output_height: usize,
    output_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    patch_size: usize,
    options: &ConvOptions,
    activation: Option<UnaryOperation>,
) {
    let output_plane = output_height * output_width;
    const TARGET_COLUMN_ELEMENTS: usize = 4 * 1024 * 1024;

    let rows_per_tile =
        (TARGET_COLUMN_ELEMENTS / (patch_size * output_width)).clamp(1, output_height);
    let panels_per_row = output_width.div_ceil(SPATIAL_PANEL_COLUMNS);
    let maximum_panels = rows_per_tile * panels_per_row;
    let mut columns = Buffer::for_overwrite(maximum_panels * patch_size * SPATIAL_PANEL_COLUMNS);
    let mut tile_output =
        Buffer::for_overwrite(maximum_panels * output_channels * SPATIAL_PANEL_COLUMNS);
    for output_y_start in (0..output_height).step_by(rows_per_tile) {
        let tile_rows = (output_height - output_y_start).min(rows_per_tile);
        let tile_plane = tile_rows * output_width;
        let tile_panels = tile_rows * panels_per_row;
        let columns = &mut columns[..tile_panels * patch_size * SPATIAL_PANEL_COLUMNS];
        im2col_panels(
            columns,
            input,
            input_channels,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            output_y_start,
            tile_rows,
            output_width,
            options,
        );
        let tile_output = &mut tile_output[..tile_panels * output_channels * SPATIAL_PANEL_COLUMNS];
        if let Some(sparse) = &options.exact_sparse_weights {
            kernels::gemm_sparse_packed_panels(
                tile_output,
                columns,
                output_channels,
                patch_size,
                tile_panels,
                bias,
                &sparse.row_offsets,
                &sparse.indices,
                &sparse.values,
                activation,
            );
        } else {
            kernels::gemm_packed_panels(
                tile_output,
                weight,
                columns,
                output_channels,
                patch_size,
                tile_panels,
                bias,
                activation,
            );
        }
        output
            .par_chunks_mut(output_plane)
            .enumerate()
            .for_each(|(channel, output)| {
                let start = output_y_start * output_width;
                let output = &mut output[start..start + tile_plane];
                for local_y in 0..tile_rows {
                    for panel_x in 0..panels_per_row {
                        let panel = local_y * panels_per_row + panel_x;
                        let source = panel * output_channels * SPATIAL_PANEL_COLUMNS
                            + channel * SPATIAL_PANEL_COLUMNS;
                        let destination = local_y * output_width + panel_x * SPATIAL_PANEL_COLUMNS;
                        let len = (output_width - panel_x * SPATIAL_PANEL_COLUMNS)
                            .min(SPATIAL_PANEL_COLUMNS);
                        output[destination..destination + len]
                            .copy_from_slice(&tile_output[source..source + len]);
                    }
                }
            });
    }
}

#[allow(clippy::too_many_arguments)]
fn im2col_panels(
    columns: &mut [f32],
    input: &[f32],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    output_y_start: usize,
    output_rows: usize,
    output_width: usize,
    options: &ConvOptions,
) {
    let input_plane = input_height * input_width;
    let panels_per_row = output_width.div_ceil(SPATIAL_PANEL_COLUMNS);
    let stride2_simd = options.strides[1] == 2 && kernels::supports_stride2_simd_copy();
    debug_assert_eq!(
        columns.len(),
        output_rows
            * panels_per_row
            * input_channels
            * kernel_height
            * kernel_width
            * SPATIAL_PANEL_COLUMNS
    );

    columns
        .par_chunks_mut(input_channels * kernel_height * kernel_width * SPATIAL_PANEL_COLUMNS)
        .enumerate()
        .for_each(|(panel, output)| {
            output.fill(0.0);
            let local_y = panel / panels_per_row;
            let output_x = panel % panels_per_row * SPATIAL_PANEL_COLUMNS;
            let available = (output_width - output_x).min(SPATIAL_PANEL_COLUMNS);
            for channel in 0..input_channels {
                let channel_input = &input[channel * input_plane..(channel + 1) * input_plane];
                for kernel_y in 0..kernel_height {
                    let input_y = (output_y_start + local_y) * options.strides[0] + kernel_y;
                    if input_y < options.pads[0] || input_y - options.pads[0] >= input_height {
                        continue;
                    }
                    let input_y = input_y - options.pads[0];
                    for kernel_x in 0..kernel_width {
                        let patch_index =
                            (channel * kernel_height + kernel_y) * kernel_width + kernel_x;
                        let destination =
                            unsafe { output.as_mut_ptr().add(patch_index * SPATIAL_PANEL_COLUMNS) };
                        let padded_input_x = output_x * options.strides[1] + kernel_x;
                        if available == SPATIAL_PANEL_COLUMNS
                            && options.strides[1] == 1
                            && padded_input_x >= options.pads[1]
                            && padded_input_x - options.pads[1] + SPATIAL_PANEL_COLUMNS
                                <= input_width
                        {
                            let input_x = padded_input_x - options.pads[1];
                            let source = input_y * input_width + input_x;
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    channel_input.as_ptr().add(source),
                                    destination,
                                    SPATIAL_PANEL_COLUMNS,
                                )
                            };
                            continue;
                        }
                        if available == SPATIAL_PANEL_COLUMNS
                            && stride2_simd
                            && padded_input_x >= options.pads[1]
                            && padded_input_x - options.pads[1] + 31 <= input_width
                        {
                            let input_x = padded_input_x - options.pads[1];
                            let source = input_y * input_width + input_x;
                            unsafe {
                                kernels::copy_stride2_16(
                                    destination,
                                    channel_input.as_ptr().add(source),
                                )
                            };
                            continue;
                        }
                        for lane in 0..available {
                            let input_x = (output_x + lane) * options.strides[1] + kernel_x;
                            if input_x >= options.pads[1] && input_x - options.pads[1] < input_width
                            {
                                unsafe {
                                    *destination.add(lane) = channel_input
                                        [input_y * input_width + input_x - options.pads[1]]
                                };
                            }
                        }
                    }
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    output_height: usize,
    output_width: usize,
    options: &ConvOptions,
) -> Buffer {
    let input_plane = input_height * input_width;
    let output_plane = output_height * output_width;
    let mut columns = Buffer::for_overwrite(channels * kernel_height * kernel_width * output_plane);
    columns
        .par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(patch_index, output)| {
            let kernel_x = patch_index % kernel_width;
            let patch_index = patch_index / kernel_width;
            let kernel_y = patch_index % kernel_height;
            let channel = patch_index / kernel_height;
            let input = &input[channel * input_plane..(channel + 1) * input_plane];
            for output_y in 0..output_height {
                let input_y = output_y * options.strides[0] + kernel_y;
                if input_y < options.pads[0] || input_y - options.pads[0] >= input_height {
                    continue;
                }
                let input_y = input_y - options.pads[0];
                for output_x in 0..output_width {
                    let input_x = output_x * options.strides[1] + kernel_x;
                    if input_x >= options.pads[1] && input_x - options.pads[1] < input_width {
                        output[output_y * output_width + output_x] =
                            input[input_y * input_width + input_x - options.pads[1]];
                    }
                }
            }
        });
    columns
}

fn conv_transpose(inputs: Vec<Tensor>, options: &ConvOptions) -> Result<Tensor> {
    ensure!(
        (2..=3).contains(&inputs.len()),
        "ConvTranspose expects two or three inputs"
    );
    let input = inputs[0].as_f32()?;
    let weight = inputs[1].as_f32()?;
    let bias = inputs.get(2).map(Tensor::as_f32).transpose()?;
    let [batch, input_channels, input_height, input_width] = shape4(&inputs[0].shape)?;
    let [
        weight_input_channels,
        output_channels_per_group,
        kernel_height,
        kernel_width,
    ] = shape4(&inputs[1].shape)?;
    ensure!(
        weight_input_channels == input_channels,
        "ConvTranspose input channel mismatch"
    );
    ensure!(
        input_channels % options.groups == 0,
        "ConvTranspose group mismatch"
    );
    let output_channels = output_channels_per_group * options.groups;
    let output_height =
        (input_height - 1) * options.strides[0] + kernel_height - options.pads[0] - options.pads[2];
    let output_width =
        (input_width - 1) * options.strides[1] + kernel_width - options.pads[1] - options.pads[3];
    let input_plane = input_height * input_width;
    let output_plane = output_height * output_width;
    let input_channels_per_group = input_channels / options.groups;
    let mut output = Buffer::zeroed(batch * output_channels * output_plane);

    if kernel_height == 2
        && kernel_width == 2
        && options.strides == [2, 2]
        && options.pads == [0; 4]
        && options.groups == 1
    {
        for batch_index in 0..batch {
            let input = &input[batch_index * input_channels * input_plane
                ..(batch_index + 1) * input_channels * input_plane];
            let batch_output = &mut output[batch_index * output_channels * output_plane
                ..(batch_index + 1) * output_channels * output_plane];
            for kernel_y in 0..2 {
                for kernel_x in 0..2 {
                    let mut matrix = Buffer::for_overwrite(output_channels * input_channels);
                    for output_channel in 0..output_channels {
                        for input_channel in 0..input_channels {
                            matrix[output_channel * input_channels + input_channel] = weight
                                [((input_channel * output_channels + output_channel) * 2
                                    + kernel_y)
                                    * 2
                                    + kernel_x];
                        }
                    }
                    let mut tile = Buffer::for_overwrite(output_channels * input_plane);
                    kernels::gemm(
                        &mut tile,
                        &matrix,
                        input,
                        output_channels,
                        input_channels,
                        input_plane,
                        None,
                    );
                    batch_output
                        .par_chunks_mut(output_plane)
                        .zip(tile.par_chunks(input_plane))
                        .enumerate()
                        .for_each(|(output_channel, (output, tile))| {
                            let bias = bias.map_or(0.0, |bias| bias[output_channel]);
                            for input_y in 0..input_height {
                                for input_x in 0..input_width {
                                    output[(input_y * 2 + kernel_y) * output_width
                                        + input_x * 2
                                        + kernel_x] = tile[input_y * input_width + input_x] + bias;
                                }
                            }
                        });
                }
            }
        }
        return Ok(Tensor::new_f32(
            vec![batch, output_channels, output_height, output_width],
            output,
        ));
    }
    output
        .par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(plane_index, output_values)| {
            let batch_index = plane_index / output_channels;
            let output_channel = plane_index % output_channels;
            kernels::fill(output_values, bias.map_or(0.0, |bias| bias[output_channel]));
            let group = output_channel / output_channels_per_group;
            let output_channel_in_group = output_channel % output_channels_per_group;
            for input_channel in
                group * input_channels_per_group..(group + 1) * input_channels_per_group
            {
                let input_base = (batch_index * input_channels + input_channel) * input_plane;
                let weight_base = (input_channel * output_channels_per_group
                    + output_channel_in_group)
                    * kernel_height
                    * kernel_width;
                for input_y in 0..input_height {
                    for input_x in 0..input_width {
                        let value = input[input_base + input_y * input_width + input_x];
                        for kernel_y in 0..kernel_height {
                            let output_y = input_y * options.strides[0] + kernel_y;
                            if output_y < options.pads[0]
                                || output_y - options.pads[0] >= output_height
                            {
                                continue;
                            }
                            let output_y = output_y - options.pads[0];
                            for kernel_x in 0..kernel_width {
                                let output_x = input_x * options.strides[1] + kernel_x;
                                if output_x < options.pads[1]
                                    || output_x - options.pads[1] >= output_width
                                {
                                    continue;
                                }
                                let output_x = output_x - options.pads[1];
                                output_values[output_y * output_width + output_x] = value.mul_add(
                                    weight[weight_base + kernel_y * kernel_width + kernel_x],
                                    output_values[output_y * output_width + output_x],
                                );
                            }
                        }
                    }
                }
            }
        });
    Ok(Tensor::new_f32(
        vec![batch, output_channels, output_height, output_width],
        output,
    ))
}

fn matmul(inputs: Vec<Tensor>) -> Result<Tensor> {
    matmul_impl(inputs, None)
}

fn matmul_bias_softmax(inputs: Vec<Tensor>, axis: i64) -> Result<Tensor> {
    matmul_impl(inputs, Some(axis))
}

fn matmul_impl(inputs: Vec<Tensor>, softmax_axis: Option<i64>) -> Result<Tensor> {
    ensure_input_count(&inputs, if softmax_axis.is_some() { 3 } else { 2 })?;
    let left = inputs[0].as_f32()?;
    let right = inputs[1].as_f32()?;
    ensure!(
        inputs[0].shape.len() >= 2 && inputs[1].shape.len() >= 2,
        "MatMul rank must be at least two"
    );
    let left_rank = inputs[0].shape.len();
    let right_rank = inputs[1].shape.len();
    let m = inputs[0].shape[left_rank - 2];
    let k = inputs[0].shape[left_rank - 1];
    ensure!(
        inputs[1].shape[right_rank - 2] == k,
        "MatMul inner dimension mismatch"
    );
    let n = inputs[1].shape[right_rank - 1];
    let batch_shape = broadcast_shape(
        &inputs[0].shape[..left_rank - 2],
        &inputs[1].shape[..right_rank - 2],
    )?;
    let column_bias = softmax_axis
        .map(|axis| {
            ensure!(
                normalize_axis(axis, batch_shape.len() + 2)? == batch_shape.len() + 1,
                "only last-dimension Softmax is supported"
            );
            let bias = inputs[2].as_f32()?;
            ensure!(bias.len() == n, "MatMul bias length does not match columns");
            Ok::<_, anyhow::Error>(bias)
        })
        .transpose()?;
    let batches = element_count(&batch_shape).context("MatMul batch shape overflow")?;
    let left_batch_strides =
        broadcast_batch_offsets(&inputs[0].shape[..left_rank - 2], &batch_shape, m * k)?;
    let right_batch_strides =
        broadcast_batch_offsets(&inputs[1].shape[..right_rank - 2], &batch_shape, k * n)?;
    let batch_strides = strides(&batch_shape);
    let mut output = Buffer::for_overwrite(batches * m * n);
    for batch in 0..batches {
        let (left_batch, right_batch) = batch_offsets(
            batch,
            &batch_shape,
            &batch_strides,
            &left_batch_strides,
            &right_batch_strides,
        );
        let output = &mut output[batch * m * n..(batch + 1) * m * n];
        if let Some(column_bias) = column_bias {
            kernels::gemm_column_bias_softmax(
                output,
                &left[left_batch..left_batch + m * k],
                &right[right_batch..right_batch + k * n],
                m,
                k,
                n,
                column_bias,
            );
        } else {
            kernels::gemm(
                output,
                &left[left_batch..left_batch + m * k],
                &right[right_batch..right_batch + k * n],
                m,
                k,
                n,
                None,
            );
        }
    }
    let mut output_shape = batch_shape;
    output_shape.extend([m, n]);
    Ok(Tensor::new_f32(output_shape, output))
}

fn batch_normalization(inputs: Vec<Tensor>, epsilon: f32) -> Result<Tensor> {
    ensure_input_count(&inputs, 5)?;
    let shape = inputs[0].shape.clone();
    ensure!(
        shape.len() >= 2,
        "BatchNormalization input rank must be at least two"
    );
    let channels = shape[1];
    let channel_size = element_count(&shape[2..]).context("BatchNormalization shape overflow")?;
    let batch = shape[0];
    let scale = inputs[1].as_f32()?;
    let bias = inputs[2].as_f32()?;
    let mean = inputs[3].as_f32()?;
    let variance = inputs[4].as_f32()?;
    ensure!(
        [scale.len(), bias.len(), mean.len(), variance.len()]
            .into_iter()
            .all(|len| len == channels),
        "BatchNormalization parameter length mismatch"
    );
    let mut output = inputs[0].clone();
    output
        .f32_mut()?
        .par_chunks_mut(channel_size)
        .enumerate()
        .for_each(|(index, values)| {
            let channel = index % channels;
            let multiplier = scale[channel] / (variance[channel] + epsilon).sqrt();
            let offset = bias[channel] - mean[channel] * multiplier;
            for value in values {
                *value = value.mul_add(multiplier, offset);
            }
        });
    debug_assert_eq!(output.len(), batch * channels * channel_size);
    Ok(output)
}

fn reduce_mean(mut inputs: Vec<Tensor>, axes: &[i64], keep_dims: bool) -> Result<Tensor> {
    ensure_input_count(&inputs, 1)?;
    let input = inputs.pop().expect("one input");
    let rank = input.shape.len();
    let mut axes = normalize_axes(axes, rank)?;
    axes.sort_unstable();
    axes.dedup();
    ensure!(!axes.is_empty(), "ReduceMean axes must not be empty");
    let suffix_start = rank - axes.len();
    ensure!(
        axes.iter().copied().eq(suffix_start..rank),
        "only contiguous suffix ReduceMean axes are supported"
    );
    let reduction_len =
        element_count(&input.shape[suffix_start..]).context("ReduceMean shape overflow")?;
    let output_len = input.len() / reduction_len;
    let input_values = input.as_f32()?;
    let mut output = Buffer::for_overwrite(output_len);
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            let values = &input_values[index * reduction_len..(index + 1) * reduction_len];
            *value = kernels::mean(values);
        });
    let mut output_shape = input.shape[..suffix_start].to_vec();
    if keep_dims {
        output_shape.resize(rank, 1);
    }
    Ok(Tensor::new_f32(output_shape, output))
}

fn global_average_pool(mut inputs: Vec<Tensor>) -> Result<Tensor> {
    ensure_input_count(&inputs, 1)?;
    let input = inputs.pop().expect("one input");
    let [batch, channels, height, width] = shape4(&input.shape)?;
    let spatial = height * width;
    let input_values = input.as_f32()?;
    let mut output = Buffer::for_overwrite(batch * channels);
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            let values = &input_values[index * spatial..(index + 1) * spatial];
            *value = kernels::mean(values);
        });
    Ok(Tensor::new_f32(vec![batch, channels, 1, 1], output))
}

fn pool(inputs: Vec<Tensor>, options: &PoolOptions, maximum: bool) -> Result<Tensor> {
    ensure_input_count(&inputs, 1)?;
    let input = inputs[0].as_f32()?;
    let [batch, channels, height, width] = shape4(&inputs[0].shape)?;
    let output_height = conv_output_size(
        height,
        options.kernel[0],
        options.strides[0],
        options.pads[0],
        options.pads[2],
        options.ceil_mode,
    )?;
    let output_width = conv_output_size(
        width,
        options.kernel[1],
        options.strides[1],
        options.pads[1],
        options.pads[3],
        options.ceil_mode,
    )?;
    let input_plane = height * width;
    let output_plane = output_height * output_width;
    let mut output = Buffer::for_overwrite(batch * channels * output_plane);
    if maximum
        && options.kernel == [2, 2]
        && options.strides == [1, 1]
        && options.pads == [0, 0, 1, 1]
        && output_height == height
        && output_width == width
    {
        output
            .par_chunks_mut(output_plane)
            .zip(input.par_chunks(input_plane))
            .for_each(|(output, input)| kernels::max_pool_2x2_same_upper(output, input, width));
        return Ok(Tensor::new_f32(
            vec![batch, channels, output_height, output_width],
            output,
        ));
    }
    if !maximum
        && options.kernel == [3, 2]
        && options.strides == [3, 2]
        && options.pads == [0; 4]
        && height == output_height * 3
        && width == output_width * 2
    {
        output
            .par_chunks_mut(output_plane)
            .zip(input.par_chunks(input_plane))
            .for_each(|(output, input)| {
                for output_y in 0..output_height {
                    let row0 = &input[(output_y * 3) * width..(output_y * 3 + 1) * width];
                    let row1 = &input[(output_y * 3 + 1) * width..(output_y * 3 + 2) * width];
                    let row2 = &input[(output_y * 3 + 2) * width..(output_y * 3 + 3) * width];
                    for output_x in 0..output_width {
                        let x = output_x * 2;
                        output[output_y * output_width + output_x] =
                            (row0[x] + row0[x + 1] + row1[x] + row1[x + 1] + row2[x] + row2[x + 1])
                                / 6.0;
                    }
                }
            });
        return Ok(Tensor::new_f32(
            vec![batch, channels, output_height, output_width],
            output,
        ));
    }
    output
        .par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(plane, output_values)| {
            let input_values = &input[plane * input_plane..(plane + 1) * input_plane];
            for output_y in 0..output_height {
                for output_x in 0..output_width {
                    let mut value = if maximum { f32::NEG_INFINITY } else { 0.0 };
                    let mut count = 0;
                    for kernel_y in 0..options.kernel[0] {
                        let input_y = output_y * options.strides[0] + kernel_y;
                        let in_y = input_y >= options.pads[0] && input_y - options.pads[0] < height;
                        for kernel_x in 0..options.kernel[1] {
                            let input_x = output_x * options.strides[1] + kernel_x;
                            let valid = in_y
                                && input_x >= options.pads[1]
                                && input_x - options.pads[1] < width;
                            if valid {
                                let sample = input_values[(input_y - options.pads[0]) * width
                                    + input_x
                                    - options.pads[1]];
                                value = if maximum {
                                    value.max(sample)
                                } else {
                                    value + sample
                                };
                                count += 1;
                            } else if !maximum && options.count_include_pad {
                                count += 1;
                            }
                        }
                    }
                    if !maximum {
                        value /= count as f32;
                    }
                    output_values[output_y * output_width + output_x] = value;
                }
            }
        });
    Ok(Tensor::new_f32(
        vec![batch, channels, output_height, output_width],
        output,
    ))
}

fn resize(inputs: Vec<Tensor>) -> Result<Tensor> {
    ensure!(
        inputs.len() >= 3,
        "Resize expects at least three present inputs"
    );
    let input = inputs[0].as_f32()?;
    let [batch, channels, height, width] = shape4(&inputs[0].shape)?;
    let target = inputs.last().context("Resize has no size or scale input")?;
    let (output_height, output_width) = match &target.data {
        TensorData::I64(values) => {
            ensure!(values.len() == 4, "Resize sizes must have four dimensions");
            (usize::try_from(values[2])?, usize::try_from(values[3])?)
        }
        TensorData::F32(values) => {
            ensure!(values.len() == 4, "Resize scales must have four dimensions");
            (
                (height as f32 * values[2]) as usize,
                (width as f32 * values[3]) as usize,
            )
        }
    };
    ensure!(
        output_height > 0 && output_width > 0,
        "Resize output dimensions must be positive"
    );
    let input_plane = height * width;
    let output_plane = output_height * output_width;
    let mut output = Buffer::for_overwrite(batch * channels * output_plane);
    output
        .par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(plane, output_values)| {
            let input_values = &input[plane * input_plane..(plane + 1) * input_plane];
            for output_y in 0..output_height {
                let input_y = output_y * height / output_height;
                for output_x in 0..output_width {
                    let input_x = output_x * width / output_width;
                    output_values[output_y * output_width + output_x] =
                        input_values[input_y * width + input_x];
                }
            }
        });
    Ok(Tensor::new_f32(
        vec![batch, channels, output_height, output_width],
        output,
    ))
}

fn concat(inputs: Vec<Tensor>, axis: i64) -> Result<Tensor> {
    ensure!(!inputs.is_empty(), "Concat expects at least one input");
    let rank = inputs[0].shape.len();
    let axis = normalize_axis(axis, rank)?;
    let mut output_shape = inputs[0].shape.clone();
    output_shape[axis] = 0;
    for input in &inputs {
        ensure!(input.shape.len() == rank, "Concat rank mismatch");
        for (dimension, (&input_dimension, &output_dimension)) in
            input.shape.iter().zip(&output_shape).enumerate()
        {
            if dimension != axis {
                ensure!(input_dimension == output_dimension, "Concat shape mismatch");
            }
        }
        output_shape[axis] += input.shape[axis];
    }
    let outer = element_count(&output_shape[..axis]).context("Concat shape overflow")?;
    let inner = element_count(&output_shape[axis + 1..]).context("Concat shape overflow")?;
    match &inputs[0].data {
        TensorData::F32(_) => {
            let mut output = Buffer::with_capacity(
                element_count(&output_shape).context("Concat shape overflow")?,
            );
            for outer_index in 0..outer {
                for input in &inputs {
                    let values = input.as_f32()?;
                    let chunk = input.shape[axis] * inner;
                    output
                        .extend_from_slice(&values[outer_index * chunk..(outer_index + 1) * chunk]);
                }
            }
            Ok(Tensor::new_f32(output_shape, output))
        }
        TensorData::I64(_) => {
            let mut output =
                Vec::with_capacity(element_count(&output_shape).context("Concat shape overflow")?);
            for outer_index in 0..outer {
                for input in &inputs {
                    let values = input.as_i64()?;
                    let chunk = input.shape[axis] * inner;
                    output
                        .extend_from_slice(&values[outer_index * chunk..(outer_index + 1) * chunk]);
                }
            }
            Ok(Tensor::new_i64(output_shape, output))
        }
    }
}

fn transpose(mut inputs: Vec<Tensor>, permutation: &[usize]) -> Result<Tensor> {
    ensure_input_count(&inputs, 1)?;
    let input = inputs.pop().expect("one input");
    let rank = input.shape.len();
    ensure!(
        permutation.len() == rank,
        "Transpose permutation rank mismatch"
    );
    let mut seen = vec![false; rank];
    for &dimension in permutation {
        ensure!(
            dimension < rank && !seen[dimension],
            "invalid Transpose permutation"
        );
        seen[dimension] = true;
    }
    let output_shape = permutation
        .iter()
        .map(|&index| input.shape[index])
        .collect::<Vec<_>>();
    if rank == 3 && permutation == [0, 2, 1] {
        let batches = input.shape[0];
        let rows = input.shape[1];
        let columns = input.shape[2];
        return match &input.data {
            TensorData::F32(values) => Ok(Tensor::new_f32(
                output_shape,
                transpose_3d_last_two_f32(values, batches, rows, columns),
            )),
            TensorData::I64(values) => Ok(Tensor::new_i64(
                output_shape,
                transpose_3d_last_two(values, batches, rows, columns),
            )),
        };
    }
    let input_strides = strides(&input.shape);
    let output_strides = strides(&output_shape);
    match &input.data {
        TensorData::F32(values) => {
            let mut output = Buffer::for_overwrite(values.len());
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(flat, output)| {
                    let mut remainder = flat;
                    let mut input_index = 0;
                    for output_dimension in 0..rank {
                        let coordinate = remainder / output_strides[output_dimension];
                        remainder %= output_strides[output_dimension];
                        input_index += coordinate * input_strides[permutation[output_dimension]];
                    }
                    *output = values[input_index];
                });
            Ok(Tensor::new_f32(output_shape, output))
        }
        TensorData::I64(values) => {
            let mut output = vec![0; values.len()];
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(flat, output)| {
                    let mut remainder = flat;
                    let mut input_index = 0;
                    for output_dimension in 0..rank {
                        let coordinate = remainder / output_strides[output_dimension];
                        remainder %= output_strides[output_dimension];
                        input_index += coordinate * input_strides[permutation[output_dimension]];
                    }
                    *output = values[input_index];
                });
            Ok(Tensor::new_i64(output_shape, output))
        }
    }
}

fn transpose_3d_last_two<T: Copy + Default>(
    input: &[T],
    batches: usize,
    rows: usize,
    columns: usize,
) -> Vec<T> {
    let matrix_len = rows * columns;
    let mut output = vec![T::default(); batches * matrix_len];
    for batch in 0..batches {
        let input = &input[batch * matrix_len..(batch + 1) * matrix_len];
        let output = &mut output[batch * matrix_len..(batch + 1) * matrix_len];
        for column in 0..columns {
            let output = &mut output[column * rows..(column + 1) * rows];
            for row in 0..rows {
                output[row] = input[row * columns + column];
            }
        }
    }
    output
}

fn transpose_3d_last_two_f32(input: &[f32], batches: usize, rows: usize, columns: usize) -> Buffer {
    let matrix_len = rows * columns;
    let mut output = Buffer::for_overwrite(batches * matrix_len);
    for batch in 0..batches {
        let input = &input[batch * matrix_len..(batch + 1) * matrix_len];
        let output = &mut output[batch * matrix_len..(batch + 1) * matrix_len];
        for column in 0..columns {
            let output = &mut output[column * rows..(column + 1) * rows];
            for row in 0..rows {
                output[row] = input[row * columns + column];
            }
        }
    }
    output
}

fn reshape(mut inputs: Vec<Tensor>) -> Result<Tensor> {
    ensure_input_count(&inputs, 2)?;
    let requested = inputs.pop().expect("shape input").into_i64()?;
    let input = inputs.pop().expect("data input");
    let mut output_shape = Vec::with_capacity(requested.len());
    let mut inferred = None;
    let mut known = 1usize;
    for (index, dimension) in requested.into_iter().enumerate() {
        match dimension {
            -1 => {
                ensure!(
                    inferred.replace(index).is_none(),
                    "Reshape has multiple inferred dimensions"
                );
                output_shape.push(1);
            }
            0 => {
                let dimension = *input
                    .shape
                    .get(index)
                    .context("Reshape zero dimension is out of range")?;
                known = known
                    .checked_mul(dimension)
                    .context("Reshape shape overflow")?;
                output_shape.push(dimension);
            }
            value if value > 0 => {
                let dimension = usize::try_from(value)?;
                known = known
                    .checked_mul(dimension)
                    .context("Reshape shape overflow")?;
                output_shape.push(dimension);
            }
            _ => bail!("invalid Reshape dimension {dimension}"),
        }
    }
    if let Some(index) = inferred {
        ensure!(
            known != 0 && input.len().is_multiple_of(known),
            "cannot infer Reshape dimension"
        );
        output_shape[index] = input.len() / known;
    }
    ensure!(
        element_count(&output_shape) == Some(input.len()),
        "Reshape element count mismatch"
    );
    Ok(Tensor {
        shape: output_shape,
        data: input.data,
    })
}

fn shape(mut inputs: Vec<Tensor>) -> Result<Tensor> {
    ensure_input_count(&inputs, 1)?;
    let input = inputs.pop().expect("one input");
    let values = input
        .shape
        .iter()
        .map(|&dimension| i64::try_from(dimension))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(Tensor::new_i64(vec![values.len()], values))
}

fn slice(inputs: Vec<Tensor>) -> Result<Tensor> {
    ensure!(
        (3..=5).contains(&inputs.len()),
        "Slice expects three to five inputs"
    );
    let starts = inputs[1].as_i64()?;
    let ends = inputs[2].as_i64()?;
    let default_axes;
    let axes = if let Some(axes) = inputs.get(3) {
        axes.as_i64()?
    } else {
        default_axes = (0..starts.len())
            .map(|axis| axis as i64)
            .collect::<Vec<_>>();
        &default_axes
    };
    let default_steps;
    let steps = if let Some(steps) = inputs.get(4) {
        steps.as_i64()?
    } else {
        default_steps = vec![1; starts.len()];
        &default_steps
    };
    ensure!(
        starts.len() == ends.len() && starts.len() == axes.len() && starts.len() == steps.len(),
        "Slice parameter length mismatch"
    );
    let rank = inputs[0].shape.len();
    let mut ranges = inputs[0]
        .shape
        .iter()
        .map(|&dimension| (0usize, dimension, 1usize))
        .collect::<Vec<_>>();
    for index in 0..starts.len() {
        ensure!(
            steps[index] > 0,
            "negative or zero Slice steps are unsupported"
        );
        let axis = normalize_axis(axes[index], rank)?;
        let dimension = inputs[0].shape[axis] as i64;
        let start = if starts[index] < 0 {
            dimension + starts[index]
        } else {
            starts[index]
        }
        .clamp(0, dimension);
        let end = if ends[index] < 0 {
            dimension + ends[index]
        } else {
            ends[index]
        }
        .clamp(0, dimension);
        ranges[axis] = (
            usize::try_from(start)?,
            usize::try_from(end.max(start))?,
            usize::try_from(steps[index])?,
        );
    }
    let output_shape = ranges
        .iter()
        .map(|&(start, end, step)| (end - start).div_ceil(step))
        .collect::<Vec<_>>();
    let output_len = element_count(&output_shape).context("Slice output shape overflow")?;
    let input_strides = strides(&inputs[0].shape);
    let output_strides = strides(&output_shape);
    match &inputs[0].data {
        TensorData::F32(values) => {
            let mut output = Buffer::for_overwrite(output_len);
            copy_slice(
                &mut output,
                values,
                &ranges,
                &input_strides,
                &output_strides,
            );
            Ok(Tensor::new_f32(output_shape, output))
        }
        TensorData::I64(values) => {
            let mut output = vec![0; output_len];
            copy_slice(
                &mut output,
                values,
                &ranges,
                &input_strides,
                &output_strides,
            );
            Ok(Tensor::new_i64(output_shape, output))
        }
    }
}

fn copy_slice<T: Copy + Send + Sync>(
    output: &mut [T],
    input: &[T],
    ranges: &[(usize, usize, usize)],
    input_strides: &[usize],
    output_strides: &[usize],
) {
    output.par_iter_mut().enumerate().for_each(|(flat, value)| {
        let mut remainder = flat;
        let mut input_index = 0;
        for dimension in 0..ranges.len() {
            let coordinate = remainder / output_strides[dimension];
            remainder %= output_strides[dimension];
            input_index +=
                (ranges[dimension].0 + coordinate * ranges[dimension].2) * input_strides[dimension];
        }
        *value = input[input_index];
    });
}

fn squeeze(mut inputs: Vec<Tensor>, attribute_axes: &[i64]) -> Result<Tensor> {
    ensure!(
        !inputs.is_empty() && inputs.len() <= 2,
        "Squeeze expects one or two inputs"
    );
    let axes_tensor = if inputs.len() == 2 {
        Some(inputs.pop().expect("axes"))
    } else {
        None
    };
    let input = inputs.pop().expect("data");
    let axes = axes_tensor
        .as_ref()
        .map(Tensor::as_i64)
        .transpose()?
        .unwrap_or(attribute_axes);
    let normalized = normalize_axes(axes, input.shape.len())?;
    let output_shape = if normalized.is_empty() {
        input
            .shape
            .iter()
            .copied()
            .filter(|&dimension| dimension != 1)
            .collect()
    } else {
        input
            .shape
            .iter()
            .enumerate()
            .filter_map(|(index, &dimension)| {
                if normalized.contains(&index) {
                    debug_assert_eq!(dimension, 1);
                    None
                } else {
                    Some(dimension)
                }
            })
            .collect()
    };
    for axis in normalized {
        ensure!(
            input.shape[axis] == 1,
            "cannot squeeze a non-unit dimension"
        );
    }
    Ok(Tensor {
        shape: output_shape,
        data: input.data,
    })
}

fn unsqueeze(mut inputs: Vec<Tensor>, attribute_axes: &[i64]) -> Result<Tensor> {
    ensure!(
        !inputs.is_empty() && inputs.len() <= 2,
        "Unsqueeze expects one or two inputs"
    );
    let axes_tensor = if inputs.len() == 2 {
        Some(inputs.pop().expect("axes"))
    } else {
        None
    };
    let input = inputs.pop().expect("data");
    let axes = axes_tensor
        .as_ref()
        .map(Tensor::as_i64)
        .transpose()?
        .unwrap_or(attribute_axes);
    let output_rank = input.shape.len() + axes.len();
    let mut axes = normalize_axes(axes, output_rank)?;
    axes.sort_unstable();
    ensure!(
        axes.windows(2).all(|pair| pair[0] != pair[1]),
        "duplicate Unsqueeze axis"
    );
    let mut output_shape = Vec::with_capacity(output_rank);
    let mut input_index = 0;
    for output_index in 0..output_rank {
        if axes.binary_search(&output_index).is_ok() {
            output_shape.push(1);
        } else {
            output_shape.push(input.shape[input_index]);
            input_index += 1;
        }
    }
    Ok(Tensor {
        shape: output_shape,
        data: input.data,
    })
}

fn softmax(mut inputs: Vec<Tensor>, axis: i64) -> Result<Tensor> {
    ensure_input_count(&inputs, 1)?;
    let mut input = inputs.pop().expect("one input");
    let axis = normalize_axis(axis, input.shape.len())?;
    let axis_len = input.shape[axis];
    let inner = element_count(&input.shape[axis + 1..]).context("Softmax shape overflow")?;
    ensure!(inner == 1, "only last-dimension Softmax is supported");
    input
        .f32_mut()?
        .par_chunks_mut(axis_len)
        .for_each(|values| {
            kernels::softmax_in_place(values);
        });
    Ok(input)
}

fn bias_softmax(mut inputs: Vec<Tensor>, axis: i64) -> Result<Tensor> {
    ensure_input_count(&inputs, 2)?;
    let bias = inputs.pop().expect("bias input");
    let mut input = inputs.pop().expect("data input");
    let axis = normalize_axis(axis, input.shape.len())?;
    let axis_len = input.shape[axis];
    let inner = element_count(&input.shape[axis + 1..]).context("Softmax shape overflow")?;
    ensure!(inner == 1, "only last-dimension Softmax is supported");
    let bias = bias.as_f32()?;
    ensure!(
        bias.len() == axis_len,
        "Softmax bias length does not match its axis"
    );
    input
        .f32_mut()?
        .par_chunks_mut(axis_len)
        .for_each(|values| {
            kernels::bias_softmax_in_place(values, bias);
        });
    Ok(input)
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut output = vec![1; rank];
    for (index, dimension) in output.iter_mut().enumerate() {
        let left = left
            .get(left.len().wrapping_sub(rank - index))
            .copied()
            .unwrap_or(1);
        let right = right
            .get(right.len().wrapping_sub(rank - index))
            .copied()
            .unwrap_or(1);
        ensure!(
            left == right || left == 1 || right == 1,
            "incompatible broadcast dimensions {left} and {right}"
        );
        *dimension = left.max(right);
    }
    Ok(output)
}

fn broadcast_strides(input: &[usize], output: &[usize]) -> Result<Vec<usize>> {
    ensure!(
        input.len() <= output.len(),
        "broadcast input rank exceeds output rank"
    );
    let input_strides = strides(input);
    let offset = output.len() - input.len();
    let mut result = vec![0; output.len()];
    for index in 0..input.len() {
        ensure!(
            input[index] == 1 || input[index] == output[offset + index],
            "invalid broadcast shape"
        );
        result[offset + index] = if input[index] == 1 {
            0
        } else {
            input_strides[index]
        };
    }
    Ok(result)
}

fn broadcast_batch_offsets(
    input: &[usize],
    output: &[usize],
    matrix_size: usize,
) -> Result<Vec<usize>> {
    let mut result = broadcast_strides(input, output)?;
    for stride in &mut result {
        *stride *= matrix_size;
    }
    Ok(result)
}

fn batch_offsets(
    batch: usize,
    shape: &[usize],
    shape_strides: &[usize],
    left_strides: &[usize],
    right_strides: &[usize],
) -> (usize, usize) {
    let mut remainder = batch;
    let mut left = 0;
    let mut right = 0;
    for dimension in 0..shape.len() {
        let coordinate = remainder / shape_strides[dimension];
        remainder %= shape_strides[dimension];
        left += coordinate * left_strides[dimension];
        right += coordinate * right_strides[dimension];
    }
    (left, right)
}

fn conv_output_size(
    input: usize,
    kernel: usize,
    stride: usize,
    pad_start: usize,
    pad_end: usize,
    ceil: bool,
) -> Result<usize> {
    ensure!(
        stride > 0 && kernel > 0,
        "kernel and stride must be positive"
    );
    let padded = input
        .checked_add(pad_start)
        .and_then(|value| value.checked_add(pad_end))
        .context("padded shape overflow")?;
    ensure!(padded >= kernel, "kernel is larger than padded input");
    let numerator = padded - kernel;
    Ok(if ceil {
        numerator.div_ceil(stride) + 1
    } else {
        numerator / stride + 1
    })
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank_i64 = i64::try_from(rank)?;
    let axis = if axis < 0 { rank_i64 + axis } else { axis };
    ensure!(
        (0..rank_i64).contains(&axis),
        "axis {axis} is out of range for rank {rank}"
    );
    Ok(usize::try_from(axis)?)
}

fn normalize_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>> {
    axes.iter()
        .map(|&axis| normalize_axis(axis, rank))
        .collect()
}

fn shape4(shape: &[usize]) -> Result<[usize; 4]> {
    shape
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected rank four, found shape {shape:?}"))
}

fn ensure_input_count(inputs: &[Tensor], expected: usize) -> Result<()> {
    ensure!(
        inputs.len() == expected,
        "expected {expected} inputs, found {}",
        inputs.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sparse_weights_retain_every_nonzero_value() {
        if !kernels::supports_exact_sparse_gemm() {
            return;
        }
        let smallest = f32::from_bits(1);
        let weights = [smallest, 0.0, -smallest, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sparse = ExactSparseConvWeights::from_dense(&weights, 4, 2)
            .expect("one of two exact blocks is active");
        assert_eq!(sparse.row_offsets.as_slice(), [0, 1]);
        assert_eq!(sparse.indices.as_slice(), [0]);
        assert_eq!(sparse.values.as_slice(), [smallest, -smallest, 0.0, 0.0]);
    }
}
