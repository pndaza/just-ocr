//! AArch64 NEON kernels.
use core::arch::aarch64::*;
#[target_feature(enable = "neon")]
pub(super) unsafe fn axpy(output: &mut [f32], input: &[f32], scale: f32) {
    let vector_len = output.len() / 16 * 16;
    let mut index = 0;
    while index < vector_len {
        // Four independent accumulators hide FMA latency on Apple cores.
        let x0 = unsafe { vld1q_f32(input.as_ptr().add(index)) };
        let x1 = unsafe { vld1q_f32(input.as_ptr().add(index + 4)) };
        let x2 = unsafe { vld1q_f32(input.as_ptr().add(index + 8)) };
        let x3 = unsafe { vld1q_f32(input.as_ptr().add(index + 12)) };
        let y0 = unsafe { vld1q_f32(output.as_ptr().add(index)) };
        let y1 = unsafe { vld1q_f32(output.as_ptr().add(index + 4)) };
        let y2 = unsafe { vld1q_f32(output.as_ptr().add(index + 8)) };
        let y3 = unsafe { vld1q_f32(output.as_ptr().add(index + 12)) };
        unsafe {
            vst1q_f32(output.as_mut_ptr().add(index), vfmaq_n_f32(y0, x0, scale));
            vst1q_f32(
                output.as_mut_ptr().add(index + 4),
                vfmaq_n_f32(y1, x1, scale),
            );
            vst1q_f32(
                output.as_mut_ptr().add(index + 8),
                vfmaq_n_f32(y2, x2, scale),
            );
            vst1q_f32(
                output.as_mut_ptr().add(index + 12),
                vfmaq_n_f32(y3, x3, scale),
            );
        }
        index += 16;
    }
    for index in vector_len..output.len() {
        output[index] = input[index].mul_add(scale, output[index]);
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn depthwise_conv2d_same<const K: usize>(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    bias: f32,
) {
    debug_assert!(matches!(K, 3 | 5 | 7 | 9));
    debug_assert_eq!(output.len(), height * width);
    debug_assert_eq!(input.len(), height * width);
    debug_assert_eq!(weights.len(), K * K);
    let padding = K / 2;

    for y in 0..height {
        let kernel_y_start = padding.saturating_sub(y);
        let kernel_y_end = K.min(height + padding - y);
        let vector_start = if width >= K { padding } else { 0 };
        let vector_columns = if width >= K {
            (width - 2 * padding) / 16 * 16
        } else {
            0
        };
        let vector_end = vector_start + vector_columns;

        for x in 0..vector_start {
            output[y * width + x] =
                unsafe { depthwise_conv2d_pixel::<K>(input, weights, height, width, y, x, bias) };
        }

        for x in (vector_start..vector_end).step_by(16) {
            let mut sums = [vdupq_n_f32(bias); 4];
            for kernel_y in kernel_y_start..kernel_y_end {
                let input_y = y + kernel_y - padding;
                for kernel_x in 0..K {
                    let input_x = x + kernel_x - padding;
                    let input_base = unsafe { input.as_ptr().add(input_y * width + input_x) };
                    let weight = unsafe { *weights.get_unchecked(kernel_y * K + kernel_x) };
                    for (vector, sum) in sums.iter_mut().enumerate() {
                        let values = unsafe { vld1q_f32(input_base.add(vector * 4)) };
                        *sum = vfmaq_n_f32(*sum, values, weight);
                    }
                }
            }
            let output_base = unsafe { output.as_mut_ptr().add(y * width + x) };
            for (vector, sum) in sums.into_iter().enumerate() {
                unsafe { vst1q_f32(output_base.add(vector * 4), sum) };
            }
        }

        for x in vector_end..width {
            output[y * width + x] =
                unsafe { depthwise_conv2d_pixel::<K>(input, weights, height, width, y, x, bias) };
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn depthwise_conv2d_same_3x3_stride2(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    bias: f32,
) {
    let output_height = height.div_ceil(2);
    let output_width = width.div_ceil(2);
    debug_assert_eq!(output.len(), output_height * output_width);
    debug_assert_eq!(input.len(), height * width);
    debug_assert_eq!(weights.len(), 9);

    for output_y in 0..output_height {
        let center_y = output_y * 2;
        let kernel_y_start = usize::from(center_y == 0);
        let kernel_y_end = if center_y + 1 < height { 3 } else { 2 };
        output[output_y * output_width] = unsafe {
            depthwise_conv2d_stride2_pixel(input, weights, height, width, output_y, 0, bias)
        };

        let mut output_x = 1;
        while output_x + 8 <= output_width && 2 * output_x + 16 < width {
            let mut sums0 = vdupq_n_f32(bias);
            let mut sums1 = vdupq_n_f32(bias);
            for kernel_y in kernel_y_start..kernel_y_end {
                let input_y = center_y + kernel_y - 1;
                for kernel_x in 0..3 {
                    let input_x = output_x * 2 + kernel_x - 1;
                    let input_base = unsafe { input.as_ptr().add(input_y * width + input_x) };
                    let values0 = unsafe { vld2q_f32(input_base) }.0;
                    let values1 = unsafe { vld2q_f32(input_base.add(8)) }.0;
                    let weight = unsafe { *weights.get_unchecked(kernel_y * 3 + kernel_x) };
                    sums0 = vfmaq_n_f32(sums0, values0, weight);
                    sums1 = vfmaq_n_f32(sums1, values1, weight);
                }
            }
            let output_base =
                unsafe { output.as_mut_ptr().add(output_y * output_width + output_x) };
            unsafe {
                vst1q_f32(output_base, sums0);
                vst1q_f32(output_base.add(4), sums1);
            }
            output_x += 8;
        }
        for output_x in output_x..output_width {
            output[output_y * output_width + output_x] = unsafe {
                depthwise_conv2d_stride2_pixel(
                    input, weights, height, width, output_y, output_x, bias,
                )
            };
        }
    }
}

#[inline(always)]
unsafe fn depthwise_conv2d_stride2_pixel(
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    output_y: usize,
    output_x: usize,
    bias: f32,
) -> f32 {
    debug_assert!(output_y < height.div_ceil(2));
    debug_assert!(output_x < width.div_ceil(2));
    let center_y = output_y * 2;
    let center_x = output_x * 2;
    let kernel_y_start = usize::from(center_y == 0);
    let kernel_y_end = if center_y + 1 < height { 3 } else { 2 };
    let kernel_x_start = usize::from(center_x == 0);
    let kernel_x_end = if center_x + 1 < width { 3 } else { 2 };
    let mut sum = bias;
    for kernel_y in kernel_y_start..kernel_y_end {
        let input_y = center_y + kernel_y - 1;
        for kernel_x in kernel_x_start..kernel_x_end {
            let input_x = center_x + kernel_x - 1;
            sum = unsafe { *input.get_unchecked(input_y * width + input_x) }.mul_add(
                unsafe { *weights.get_unchecked(kernel_y * 3 + kernel_x) },
                sum,
            );
        }
    }
    sum
}

#[inline(always)]
unsafe fn depthwise_conv2d_pixel<const K: usize>(
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    y: usize,
    x: usize,
    bias: f32,
) -> f32 {
    let padding = K / 2;
    let kernel_y_start = padding.saturating_sub(y);
    let kernel_y_end = K.min(height + padding - y);
    let kernel_x_start = padding.saturating_sub(x);
    let kernel_x_end = K.min(width + padding - x);
    let mut sum = bias;
    for kernel_y in kernel_y_start..kernel_y_end {
        let input_y = y + kernel_y - padding;
        for kernel_x in kernel_x_start..kernel_x_end {
            let input_x = x + kernel_x - padding;
            sum = unsafe { *input.get_unchecked(input_y * width + input_x) }.mul_add(
                unsafe { *weights.get_unchecked(kernel_y * K + kernel_x) },
                sum,
            );
        }
    }
    sum
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn relu(values: &mut [f32]) {
    let zero = vdupq_n_f32(0.0);
    let vector_len = values.len() / 16 * 16;
    let mut index = 0;
    while index < vector_len {
        let v0 = unsafe { vld1q_f32(values.as_ptr().add(index)) };
        let v1 = unsafe { vld1q_f32(values.as_ptr().add(index + 4)) };
        let v2 = unsafe { vld1q_f32(values.as_ptr().add(index + 8)) };
        let v3 = unsafe { vld1q_f32(values.as_ptr().add(index + 12)) };
        unsafe {
            vst1q_f32(values.as_mut_ptr().add(index), vmaxnmq_f32(v0, zero));
            vst1q_f32(values.as_mut_ptr().add(index + 4), vmaxnmq_f32(v1, zero));
            vst1q_f32(values.as_mut_ptr().add(index + 8), vmaxnmq_f32(v2, zero));
            vst1q_f32(values.as_mut_ptr().add(index + 12), vmaxnmq_f32(v3, zero));
        }
        index += 16;
    }
    for value in &mut values[vector_len..] {
        *value = value.max(0.0);
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn mul_in_place(output: &mut [f32], input: &[f32]) {
    let vector_len = output.len() / 16 * 16;
    for index in (0..vector_len).step_by(16) {
        for vector in 0..4 {
            let offset = index + vector * 4;
            let left = unsafe { vld1q_f32(output.as_ptr().add(offset)) };
            let right = unsafe { vld1q_f32(input.as_ptr().add(offset)) };
            unsafe { vst1q_f32(output.as_mut_ptr().add(offset), vmulq_f32(left, right)) };
        }
    }
    for index in vector_len..output.len() {
        output[index] *= input[index];
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn affine(values: &mut [f32], scale: f32, bias: f32) {
    let bias = vdupq_n_f32(bias);
    let vector_len = values.len() / 16 * 16;
    for index in (0..vector_len).step_by(16) {
        for vector in 0..4 {
            let offset = index + vector * 4;
            let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            unsafe {
                vst1q_f32(
                    values.as_mut_ptr().add(offset),
                    vfmaq_n_f32(bias, value, scale),
                )
            };
        }
    }
    for value in &mut values[vector_len..] {
        *value = value.mul_add(scale, vgetq_lane_f32::<0>(bias));
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn residual_mul(values: &mut [f32], gate: f32) {
    let zero = vdupq_n_f32(0.0);
    let vector_len = values.len() / 16 * 16;
    for index in (0..vector_len).step_by(16) {
        for vector in 0..4 {
            let offset = index + vector * 4;
            let original = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            let scaled = vfmaq_n_f32(zero, original, gate);
            let output = vfmaq_n_f32(original, scaled, 1.0);
            unsafe { vst1q_f32(values.as_mut_ptr().add(offset), output) };
        }
    }
    for value in &mut values[vector_len..] {
        let original = *value;
        let scaled = original.mul_add(gate, 0.0);
        *value = scaled.mul_add(1.0, original);
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn square(values: &mut [f32]) {
    let vector_len = values.len() / 16 * 16;
    for index in (0..vector_len).step_by(16) {
        for vector in 0..4 {
            let offset = index + vector * 4;
            let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            unsafe { vst1q_f32(values.as_mut_ptr().add(offset), vmulq_f32(value, value)) };
        }
    }
    for value in &mut values[vector_len..] {
        *value *= *value;
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_4x16(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
) {
    debug_assert_eq!(output.len(), 4 * columns);
    debug_assert_eq!(left.len(), 4 * inner);
    debug_assert!(right.len() >= (inner - 1) * right_stride + columns);
    debug_assert!(column_bias.is_none_or(|bias| bias.len() == columns));
    let vector_columns = columns / 16 * 16;
    for column in (0..vector_columns).step_by(16) {
        let mut accumulators = [vdupq_n_f32(0.0); 16];
        let column_initials = column_bias.map(|bias| unsafe {
            let base = bias.as_ptr().add(column);
            [
                vld1q_f32(base),
                vld1q_f32(base.add(4)),
                vld1q_f32(base.add(8)),
                vld1q_f32(base.add(12)),
            ]
        });
        for row in 0..4 {
            for vector in 0..4 {
                accumulators[row * 4 + vector] = column_initials.map_or_else(
                    || vdupq_n_f32(bias.map_or(0.0, |bias| bias[row])),
                    |initials| initials[vector],
                );
            }
        }
        for index in 0..inner {
            let right_base = unsafe { right.as_ptr().add(index * right_stride + column) };
            let right0 = unsafe { vld1q_f32(right_base) };
            let right1 = unsafe { vld1q_f32(right_base.add(4)) };
            let right2 = unsafe { vld1q_f32(right_base.add(8)) };
            let right3 = unsafe { vld1q_f32(right_base.add(12)) };
            for row in 0..4 {
                // SAFETY: `left` was validated as four complete rows above.
                let scale = unsafe { *left.get_unchecked(row * inner + index) };
                let offset = row * 4;
                accumulators[offset] = vfmaq_n_f32(accumulators[offset], right0, scale);
                accumulators[offset + 1] = vfmaq_n_f32(accumulators[offset + 1], right1, scale);
                accumulators[offset + 2] = vfmaq_n_f32(accumulators[offset + 2], right2, scale);
                accumulators[offset + 3] = vfmaq_n_f32(accumulators[offset + 3], right3, scale);
            }
        }
        for row in 0..4 {
            let output_base = unsafe { output.as_mut_ptr().add(row * columns + column) };
            let offset = row * 4;
            unsafe {
                vst1q_f32(output_base, accumulators[offset]);
                vst1q_f32(output_base.add(4), accumulators[offset + 1]);
                vst1q_f32(output_base.add(8), accumulators[offset + 2]);
                vst1q_f32(output_base.add(12), accumulators[offset + 3]);
            }
        }
    }
    for row in 0..4 {
        for column in vector_columns..columns {
            let mut sum =
                column_bias.map_or_else(|| bias.map_or(0.0, |bias| bias[row]), |bias| bias[column]);
            for index in 0..inner {
                // SAFETY: Matrix dimensions were validated at entry.
                sum = unsafe { *left.get_unchecked(row * inner + index) }.mul_add(
                    unsafe { *right.get_unchecked(index * right_stride + column) },
                    sum,
                );
            }
            output[row * columns + column] = sum;
        }
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_8x12(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
) {
    debug_assert_eq!(output.len(), 8 * columns);
    debug_assert_eq!(left.len(), 8 * inner);
    debug_assert!(right.len() >= (inner - 1) * right_stride + columns);
    debug_assert!(column_bias.is_none_or(|bias| bias.len() == columns));
    let vector_columns = columns / 12 * 12;
    for column in (0..vector_columns).step_by(12) {
        let mut accumulators = [vdupq_n_f32(0.0); 24];
        let column_initials = column_bias.map(|bias| unsafe {
            let base = bias.as_ptr().add(column);
            [
                vld1q_f32(base),
                vld1q_f32(base.add(4)),
                vld1q_f32(base.add(8)),
            ]
        });
        for row in 0..8 {
            for vector in 0..3 {
                accumulators[row * 3 + vector] = column_initials.map_or_else(
                    || vdupq_n_f32(bias.map_or(0.0, |bias| bias[row])),
                    |initials| initials[vector],
                );
            }
        }
        for index in 0..inner {
            let right_base = unsafe { right.as_ptr().add(index * right_stride + column) };
            let right0 = unsafe { vld1q_f32(right_base) };
            let right1 = unsafe { vld1q_f32(right_base.add(4)) };
            let right2 = unsafe { vld1q_f32(right_base.add(8)) };
            for row in 0..8 {
                // SAFETY: `left` was validated as eight complete rows above.
                let scale = unsafe { *left.get_unchecked(row * inner + index) };
                let offset = row * 3;
                accumulators[offset] = vfmaq_n_f32(accumulators[offset], right0, scale);
                accumulators[offset + 1] = vfmaq_n_f32(accumulators[offset + 1], right1, scale);
                accumulators[offset + 2] = vfmaq_n_f32(accumulators[offset + 2], right2, scale);
            }
        }
        for row in 0..8 {
            let output_base = unsafe { output.as_mut_ptr().add(row * columns + column) };
            let offset = row * 3;
            unsafe {
                vst1q_f32(output_base, accumulators[offset]);
                vst1q_f32(output_base.add(4), accumulators[offset + 1]);
                vst1q_f32(output_base.add(8), accumulators[offset + 2]);
            }
        }
    }
    for row in 0..8 {
        for column in vector_columns..columns {
            let mut sum =
                column_bias.map_or_else(|| bias.map_or(0.0, |bias| bias[row]), |bias| bias[column]);
            for index in 0..inner {
                // SAFETY: Matrix dimensions were validated at entry.
                sum = unsafe { *left.get_unchecked(row * inner + index) }.mul_add(
                    unsafe { *right.get_unchecked(index * right_stride + column) },
                    sum,
                );
            }
            output[row * columns + column] = sum;
        }
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_4x16_packed(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    accumulate: bool,
) {
    debug_assert_eq!(output.len(), 4 * columns);
    debug_assert_eq!(left.len(), 4 * inner);
    debug_assert!(right.len() >= inner.saturating_sub(1) * right_stride + columns);
    debug_assert!(bias.is_none_or(|bias| bias.len() == 4));
    let vector_columns = columns / 16 * 16;
    for column in (0..vector_columns).step_by(16) {
        let mut accumulators = [vdupq_n_f32(0.0); 16];
        for row in 0..4 {
            if accumulate {
                let output_base = unsafe { output.as_ptr().add(row * columns + column) };
                for vector in 0..4 {
                    accumulators[row * 4 + vector] =
                        unsafe { vld1q_f32(output_base.add(vector * 4)) };
                }
            } else {
                let initial = vdupq_n_f32(bias.map_or(0.0, |bias| bias[row]));
                for vector in 0..4 {
                    accumulators[row * 4 + vector] = initial;
                }
            }
        }
        macro_rules! k_step {
            ($index:expr) => {{
                let index = $index;
                let right_base = unsafe { right.as_ptr().add(index * right_stride + column) };
                let right0 = unsafe { vld1q_f32(right_base) };
                let right1 = unsafe { vld1q_f32(right_base.add(4)) };
                let right2 = unsafe { vld1q_f32(right_base.add(8)) };
                let right3 = unsafe { vld1q_f32(right_base.add(12)) };
                // SAFETY: `left` contains `inner` complete groups of four packed rows.
                let weights = unsafe { vld1q_f32(left.as_ptr().add(index * 4)) };
                accumulators[0] = vfmaq_laneq_f32::<0>(accumulators[0], right0, weights);
                accumulators[1] = vfmaq_laneq_f32::<0>(accumulators[1], right1, weights);
                accumulators[2] = vfmaq_laneq_f32::<0>(accumulators[2], right2, weights);
                accumulators[3] = vfmaq_laneq_f32::<0>(accumulators[3], right3, weights);
                accumulators[4] = vfmaq_laneq_f32::<1>(accumulators[4], right0, weights);
                accumulators[5] = vfmaq_laneq_f32::<1>(accumulators[5], right1, weights);
                accumulators[6] = vfmaq_laneq_f32::<1>(accumulators[6], right2, weights);
                accumulators[7] = vfmaq_laneq_f32::<1>(accumulators[7], right3, weights);
                accumulators[8] = vfmaq_laneq_f32::<2>(accumulators[8], right0, weights);
                accumulators[9] = vfmaq_laneq_f32::<2>(accumulators[9], right1, weights);
                accumulators[10] = vfmaq_laneq_f32::<2>(accumulators[10], right2, weights);
                accumulators[11] = vfmaq_laneq_f32::<2>(accumulators[11], right3, weights);
                accumulators[12] = vfmaq_laneq_f32::<3>(accumulators[12], right0, weights);
                accumulators[13] = vfmaq_laneq_f32::<3>(accumulators[13], right1, weights);
                accumulators[14] = vfmaq_laneq_f32::<3>(accumulators[14], right2, weights);
                accumulators[15] = vfmaq_laneq_f32::<3>(accumulators[15], right3, weights);
            }};
        }
        let mut index = 0;
        while index + 4 <= inner {
            k_step!(index);
            k_step!(index + 1);
            k_step!(index + 2);
            k_step!(index + 3);
            index += 4;
        }
        while index < inner {
            k_step!(index);
            index += 1;
        }
        for row in 0..4 {
            let output_base = unsafe { output.as_mut_ptr().add(row * columns + column) };
            let offset = row * 4;
            unsafe {
                vst1q_f32(output_base, accumulators[offset]);
                vst1q_f32(output_base.add(4), accumulators[offset + 1]);
                vst1q_f32(output_base.add(8), accumulators[offset + 2]);
                vst1q_f32(output_base.add(12), accumulators[offset + 3]);
            }
        }
    }
    for row in 0..4 {
        for column in vector_columns..columns {
            let mut sum = if accumulate {
                output[row * columns + column]
            } else {
                bias.map_or(0.0, |bias| bias[row])
            };
            for index in 0..inner {
                // SAFETY: Matrix dimensions were validated at entry.
                sum = unsafe { *left.get_unchecked(index * 4 + row) }.mul_add(
                    unsafe { *right.get_unchecked(index * right_stride + column) },
                    sum,
                );
            }
            output[row * columns + column] = sum;
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn gemm_4x16_sparse(
    output: &mut [f32],
    right: &[f32],
    indices: &[u32],
    weights: &[f32],
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
) {
    const ROWS: usize = 4;

    debug_assert_eq!(output.len(), ROWS * columns);
    debug_assert_eq!(weights.len(), indices.len() * ROWS);
    debug_assert!(bias.is_none_or(|bias| bias.len() == ROWS));
    let vector_columns = columns / 16 * 16;
    for column in (0..vector_columns).step_by(16) {
        let mut accumulators = [vdupq_n_f32(0.0); 16];
        for row in 0..ROWS {
            let initial = vdupq_n_f32(bias.map_or(0.0, |bias| bias[row]));
            for vector in 0..4 {
                accumulators[row * 4 + vector] = initial;
            }
        }
        for (entry, &index) in indices.iter().enumerate() {
            let right_base = unsafe { right.as_ptr().add(index as usize * right_stride + column) };
            let right0 = unsafe { vld1q_f32(right_base) };
            let right1 = unsafe { vld1q_f32(right_base.add(4)) };
            let right2 = unsafe { vld1q_f32(right_base.add(8)) };
            let right3 = unsafe { vld1q_f32(right_base.add(12)) };
            let weights = unsafe { vld1q_f32(weights.as_ptr().add(entry * ROWS)) };
            accumulators[0] = vfmaq_laneq_f32::<0>(accumulators[0], right0, weights);
            accumulators[1] = vfmaq_laneq_f32::<0>(accumulators[1], right1, weights);
            accumulators[2] = vfmaq_laneq_f32::<0>(accumulators[2], right2, weights);
            accumulators[3] = vfmaq_laneq_f32::<0>(accumulators[3], right3, weights);
            accumulators[4] = vfmaq_laneq_f32::<1>(accumulators[4], right0, weights);
            accumulators[5] = vfmaq_laneq_f32::<1>(accumulators[5], right1, weights);
            accumulators[6] = vfmaq_laneq_f32::<1>(accumulators[6], right2, weights);
            accumulators[7] = vfmaq_laneq_f32::<1>(accumulators[7], right3, weights);
            accumulators[8] = vfmaq_laneq_f32::<2>(accumulators[8], right0, weights);
            accumulators[9] = vfmaq_laneq_f32::<2>(accumulators[9], right1, weights);
            accumulators[10] = vfmaq_laneq_f32::<2>(accumulators[10], right2, weights);
            accumulators[11] = vfmaq_laneq_f32::<2>(accumulators[11], right3, weights);
            accumulators[12] = vfmaq_laneq_f32::<3>(accumulators[12], right0, weights);
            accumulators[13] = vfmaq_laneq_f32::<3>(accumulators[13], right1, weights);
            accumulators[14] = vfmaq_laneq_f32::<3>(accumulators[14], right2, weights);
            accumulators[15] = vfmaq_laneq_f32::<3>(accumulators[15], right3, weights);
        }
        for row in 0..ROWS {
            let output_base = unsafe { output.as_mut_ptr().add(row * columns + column) };
            let offset = row * 4;
            unsafe {
                vst1q_f32(output_base, accumulators[offset]);
                vst1q_f32(output_base.add(4), accumulators[offset + 1]);
                vst1q_f32(output_base.add(8), accumulators[offset + 2]);
                vst1q_f32(output_base.add(12), accumulators[offset + 3]);
            }
        }
    }
    for row in 0..ROWS {
        for column in vector_columns..columns {
            let mut sum = bias.map_or(0.0, |bias| bias[row]);
            for (entry, &index) in indices.iter().enumerate() {
                sum = unsafe { *weights.get_unchecked(entry * ROWS + row) }.mul_add(
                    unsafe { *right.get_unchecked(index as usize * right_stride + column) },
                    sum,
                );
            }
            output[row * columns + column] = sum;
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn gemm_8x12_packed(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
) {
    debug_assert_eq!(output.len(), 8 * columns);
    debug_assert_eq!(left.len(), 8 * inner);
    debug_assert!(right.len() >= inner.saturating_sub(1) * right_stride + columns);
    debug_assert!(bias.is_none_or(|bias| bias.len() == 8));
    let vector_columns = columns / 12 * 12;
    for column in (0..vector_columns).step_by(12) {
        let mut accumulators = [vdupq_n_f32(0.0); 24];
        for row in 0..8 {
            let initial = vdupq_n_f32(bias.map_or(0.0, |bias| bias[row]));
            accumulators[row * 3] = initial;
            accumulators[row * 3 + 1] = initial;
            accumulators[row * 3 + 2] = initial;
        }
        for index in 0..inner {
            let right_base = unsafe { right.as_ptr().add(index * right_stride + column) };
            let right0 = unsafe { vld1q_f32(right_base) };
            let right1 = unsafe { vld1q_f32(right_base.add(4)) };
            let right2 = unsafe { vld1q_f32(right_base.add(8)) };
            // SAFETY: `left` contains `inner` complete groups of eight packed rows.
            let weights0 = unsafe { vld1q_f32(left.as_ptr().add(index * 8)) };
            let weights1 = unsafe { vld1q_f32(left.as_ptr().add(index * 8 + 4)) };
            accumulators[0] = vfmaq_laneq_f32::<0>(accumulators[0], right0, weights0);
            accumulators[1] = vfmaq_laneq_f32::<0>(accumulators[1], right1, weights0);
            accumulators[2] = vfmaq_laneq_f32::<0>(accumulators[2], right2, weights0);
            accumulators[3] = vfmaq_laneq_f32::<1>(accumulators[3], right0, weights0);
            accumulators[4] = vfmaq_laneq_f32::<1>(accumulators[4], right1, weights0);
            accumulators[5] = vfmaq_laneq_f32::<1>(accumulators[5], right2, weights0);
            accumulators[6] = vfmaq_laneq_f32::<2>(accumulators[6], right0, weights0);
            accumulators[7] = vfmaq_laneq_f32::<2>(accumulators[7], right1, weights0);
            accumulators[8] = vfmaq_laneq_f32::<2>(accumulators[8], right2, weights0);
            accumulators[9] = vfmaq_laneq_f32::<3>(accumulators[9], right0, weights0);
            accumulators[10] = vfmaq_laneq_f32::<3>(accumulators[10], right1, weights0);
            accumulators[11] = vfmaq_laneq_f32::<3>(accumulators[11], right2, weights0);
            accumulators[12] = vfmaq_laneq_f32::<0>(accumulators[12], right0, weights1);
            accumulators[13] = vfmaq_laneq_f32::<0>(accumulators[13], right1, weights1);
            accumulators[14] = vfmaq_laneq_f32::<0>(accumulators[14], right2, weights1);
            accumulators[15] = vfmaq_laneq_f32::<1>(accumulators[15], right0, weights1);
            accumulators[16] = vfmaq_laneq_f32::<1>(accumulators[16], right1, weights1);
            accumulators[17] = vfmaq_laneq_f32::<1>(accumulators[17], right2, weights1);
            accumulators[18] = vfmaq_laneq_f32::<2>(accumulators[18], right0, weights1);
            accumulators[19] = vfmaq_laneq_f32::<2>(accumulators[19], right1, weights1);
            accumulators[20] = vfmaq_laneq_f32::<2>(accumulators[20], right2, weights1);
            accumulators[21] = vfmaq_laneq_f32::<3>(accumulators[21], right0, weights1);
            accumulators[22] = vfmaq_laneq_f32::<3>(accumulators[22], right1, weights1);
            accumulators[23] = vfmaq_laneq_f32::<3>(accumulators[23], right2, weights1);
        }
        for row in 0..8 {
            let output_base = unsafe { output.as_mut_ptr().add(row * columns + column) };
            let offset = row * 3;
            unsafe {
                vst1q_f32(output_base, accumulators[offset]);
                vst1q_f32(output_base.add(4), accumulators[offset + 1]);
                vst1q_f32(output_base.add(8), accumulators[offset + 2]);
            }
        }
    }
    for row in 0..8 {
        for column in vector_columns..columns {
            let mut sum = bias.map_or(0.0, |bias| bias[row]);
            for index in 0..inner {
                // SAFETY: Matrix dimensions were validated at entry.
                sum = unsafe { *left.get_unchecked(index * 8 + row) }.mul_add(
                    unsafe { *right.get_unchecked(index * right_stride + column) },
                    sum,
                );
            }
            output[row * columns + column] = sum;
        }
    }
}
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_12x8_packed(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    accumulate: bool,
) {
    debug_assert_eq!(output.len(), 12 * columns);
    debug_assert_eq!(left.len(), 12 * inner);
    debug_assert!(right.len() >= inner.saturating_sub(1) * right_stride + columns);
    debug_assert!(bias.is_none_or(|bias| bias.len() == 12));
    let vector_columns = columns / 8 * 8;
    for column in (0..vector_columns).step_by(8) {
        let mut accumulators = [vdupq_n_f32(0.0); 24];
        for row in 0..12 {
            if accumulate {
                let output_base = unsafe { output.as_ptr().add(row * columns + column) };
                accumulators[row * 2] = unsafe { vld1q_f32(output_base) };
                accumulators[row * 2 + 1] = unsafe { vld1q_f32(output_base.add(4)) };
            } else {
                let initial = vdupq_n_f32(bias.map_or(0.0, |bias| bias[row]));
                accumulators[row * 2] = initial;
                accumulators[row * 2 + 1] = initial;
            }
        }
        macro_rules! depth_step {
            ($index:expr) => {{
                let index = $index;
                let right_base = unsafe { right.as_ptr().add(index * right_stride + column) };
                let right0 = unsafe { vld1q_f32(right_base) };
                let right1 = unsafe { vld1q_f32(right_base.add(4)) };
                // SAFETY: `left` contains `inner` complete groups of twelve packed rows.
                let weights0 = unsafe { vld1q_f32(left.as_ptr().add(index * 12)) };
                let weights1 = unsafe { vld1q_f32(left.as_ptr().add(index * 12 + 4)) };
                let weights2 = unsafe { vld1q_f32(left.as_ptr().add(index * 12 + 8)) };
                accumulators[0] = vfmaq_laneq_f32::<0>(accumulators[0], right0, weights0);
                accumulators[1] = vfmaq_laneq_f32::<0>(accumulators[1], right1, weights0);
                accumulators[2] = vfmaq_laneq_f32::<1>(accumulators[2], right0, weights0);
                accumulators[3] = vfmaq_laneq_f32::<1>(accumulators[3], right1, weights0);
                accumulators[4] = vfmaq_laneq_f32::<2>(accumulators[4], right0, weights0);
                accumulators[5] = vfmaq_laneq_f32::<2>(accumulators[5], right1, weights0);
                accumulators[6] = vfmaq_laneq_f32::<3>(accumulators[6], right0, weights0);
                accumulators[7] = vfmaq_laneq_f32::<3>(accumulators[7], right1, weights0);
                accumulators[8] = vfmaq_laneq_f32::<0>(accumulators[8], right0, weights1);
                accumulators[9] = vfmaq_laneq_f32::<0>(accumulators[9], right1, weights1);
                accumulators[10] = vfmaq_laneq_f32::<1>(accumulators[10], right0, weights1);
                accumulators[11] = vfmaq_laneq_f32::<1>(accumulators[11], right1, weights1);
                accumulators[12] = vfmaq_laneq_f32::<2>(accumulators[12], right0, weights1);
                accumulators[13] = vfmaq_laneq_f32::<2>(accumulators[13], right1, weights1);
                accumulators[14] = vfmaq_laneq_f32::<3>(accumulators[14], right0, weights1);
                accumulators[15] = vfmaq_laneq_f32::<3>(accumulators[15], right1, weights1);
                accumulators[16] = vfmaq_laneq_f32::<0>(accumulators[16], right0, weights2);
                accumulators[17] = vfmaq_laneq_f32::<0>(accumulators[17], right1, weights2);
                accumulators[18] = vfmaq_laneq_f32::<1>(accumulators[18], right0, weights2);
                accumulators[19] = vfmaq_laneq_f32::<1>(accumulators[19], right1, weights2);
                accumulators[20] = vfmaq_laneq_f32::<2>(accumulators[20], right0, weights2);
                accumulators[21] = vfmaq_laneq_f32::<2>(accumulators[21], right1, weights2);
                accumulators[22] = vfmaq_laneq_f32::<3>(accumulators[22], right0, weights2);
                accumulators[23] = vfmaq_laneq_f32::<3>(accumulators[23], right1, weights2);
            }};
        }
        let mut index = 0;
        while index + 4 <= inner {
            depth_step!(index);
            depth_step!(index + 1);
            depth_step!(index + 2);
            depth_step!(index + 3);
            index += 4;
        }
        while index < inner {
            depth_step!(index);
            index += 1;
        }
        for row in 0..12 {
            let output_base = unsafe { output.as_mut_ptr().add(row * columns + column) };
            let offset = row * 2;
            unsafe {
                vst1q_f32(output_base, accumulators[offset]);
                vst1q_f32(output_base.add(4), accumulators[offset + 1]);
            }
        }
    }
    for row in 0..12 {
        for column in vector_columns..columns {
            let mut sum = if accumulate {
                output[row * columns + column]
            } else {
                bias.map_or(0.0, |bias| bias[row])
            };
            for index in 0..inner {
                // SAFETY: Matrix dimensions were validated at entry.
                sum = unsafe { *left.get_unchecked(index * 12 + row) }.mul_add(
                    unsafe { *right.get_unchecked(index * right_stride + column) },
                    sum,
                );
            }
            output[row * columns + column] = sum;
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn max_pool_2x2_row(output: &mut [f32], current: &[f32], next: Option<&[f32]>) {
    debug_assert_eq!(output.len(), current.len());
    debug_assert!(next.is_none_or(|next| next.len() == current.len()));
    let width = current.len();
    let vector_len = width.saturating_sub(1) / 16 * 16;
    for x in (0..vector_len).step_by(16) {
        for vector in 0..4 {
            let offset = x + vector * 4;
            let mut maximum =
                vmaxnmq_f32(unsafe { vld1q_f32(current.as_ptr().add(offset)) }, unsafe {
                    vld1q_f32(current.as_ptr().add(offset + 1))
                });
            if let Some(next) = next {
                maximum = vmaxnmq_f32(
                    maximum,
                    vmaxnmq_f32(unsafe { vld1q_f32(next.as_ptr().add(offset)) }, unsafe {
                        vld1q_f32(next.as_ptr().add(offset + 1))
                    }),
                );
            }
            unsafe { vst1q_f32(output.as_mut_ptr().add(offset), maximum) };
        }
    }
    for x in vector_len..width {
        let mut maximum = current[x];
        if x + 1 < width {
            maximum = maximum.max(current[x + 1]);
        }
        if let Some(next) = next {
            maximum = maximum.max(next[x]);
            if x + 1 < width {
                maximum = maximum.max(next[x + 1]);
            }
        }
        output[x] = maximum;
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn gelu(values: &mut [f32]) {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let half = vdupq_n_f32(0.5);
    let inv_sqrt_two = vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2);
    let vector_len = values.len() / 16 * 16;
    for index in (0..vector_len).step_by(16) {
        for vector in 0..4 {
            let offset = index + vector * 4;
            let input = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            let scaled = vmulq_f32(input, inv_sqrt_two);
            let absolute = vabsq_f32(scaled);
            let t = reciprocalq(vfmaq_n_f32(one, absolute, 0.327_591_1));
            let mut polynomial = vfmaq_n_f32(vdupq_n_f32(-1.453_152_1), t, 1.061_405_4);
            polynomial = vfmaq_f32(vdupq_n_f32(1.421_413_8), polynomial, t);
            polynomial = vfmaq_f32(vdupq_n_f32(-0.284_496_72), polynomial, t);
            polynomial = vfmaq_f32(vdupq_n_f32(0.254_829_6), polynomial, t);
            polynomial = vmulq_f32(polynomial, t);
            let exponential = expq(vnegq_f32(vmulq_f32(absolute, absolute)));
            let positive_erf = vsubq_f32(one, vmulq_f32(polynomial, exponential));
            let erf = vbslq_f32(
                vcltq_f32(scaled, zero),
                vnegq_f32(positive_erf),
                positive_erf,
            );
            let output = vmulq_f32(vmulq_f32(half, input), vaddq_f32(one, erf));
            unsafe { vst1q_f32(values.as_mut_ptr().add(offset), output) };
        }
    }
    for value in &mut values[vector_len..] {
        let input = *value;
        let x = input * std::f32::consts::FRAC_1_SQRT_2;
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / x.mul_add(0.327_591_1, 1.0);
        let polynomial = t
            * (0.254_829_6
                + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
        let erf = sign * (1.0 - polynomial * (-x * x).exp());
        *value = 0.5 * input * (1.0 + erf);
    }
}

#[target_feature(enable = "neon")]
fn reciprocalq(value: float32x4_t) -> float32x4_t {
    vdivq_f32(vdupq_n_f32(1.0), value)
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax(values: &mut [f32]) {
    let vector_len = values.len() / 16 * 16;
    let mut maxima = [vdupq_n_f32(f32::NEG_INFINITY); 4];
    for index in (0..vector_len).step_by(16) {
        for (vector, maximum) in maxima.iter_mut().enumerate() {
            let offset = index + vector * 4;
            let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            *maximum = vmaxq_f32(*maximum, value);
        }
    }
    let mut maximum = vmaxvq_f32(maxima[0])
        .max(vmaxvq_f32(maxima[1]))
        .max(vmaxvq_f32(maxima[2]))
        .max(vmaxvq_f32(maxima[3]));
    for &value in &values[vector_len..] {
        maximum = maximum.max(value);
    }

    let maximum_vector = vdupq_n_f32(maximum);
    let mut sums = [vdupq_n_f32(0.0); 4];
    for index in (0..vector_len).step_by(16) {
        for (vector, sum) in sums.iter_mut().enumerate() {
            let offset = index + vector * 4;
            let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            let exponential = expq(vsubq_f32(value, maximum_vector));
            *sum = vaddq_f32(*sum, exponential);
            unsafe { vst1q_f32(values.as_mut_ptr().add(offset), exponential) };
        }
    }
    let mut sum =
        vaddvq_f32(sums[0]) + vaddvq_f32(sums[1]) + vaddvq_f32(sums[2]) + vaddvq_f32(sums[3]);
    for value in &mut values[vector_len..] {
        *value = (*value - maximum).exp();
        sum += *value;
    }

    let reciprocal = vdupq_n_f32(sum.recip());
    for index in (0..vector_len).step_by(16) {
        for vector in 0..4 {
            let offset = index + vector * 4;
            let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            unsafe {
                vst1q_f32(
                    values.as_mut_ptr().add(offset),
                    vmulq_f32(value, reciprocal),
                )
            };
        }
    }
    let reciprocal = vgetq_lane_f32::<0>(reciprocal);
    for value in &mut values[vector_len..] {
        *value *= reciprocal;
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn bias_softmax(values: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(values.len(), bias.len());
    let vector_len = values.len() / 16 * 16;
    let mut maxima = [vdupq_n_f32(f32::NEG_INFINITY); 4];
    for index in (0..vector_len).step_by(16) {
        for (vector, maximum) in maxima.iter_mut().enumerate() {
            let offset = index + vector * 4;
            let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            let bias = unsafe { vld1q_f32(bias.as_ptr().add(offset)) };
            *maximum = vmaxq_f32(*maximum, vfmaq_n_f32(value, bias, 1.0));
        }
    }
    let mut maximum = vmaxvq_f32(maxima[0])
        .max(vmaxvq_f32(maxima[1]))
        .max(vmaxvq_f32(maxima[2]))
        .max(vmaxvq_f32(maxima[3]));
    for index in vector_len..values.len() {
        maximum = maximum.max(bias[index].mul_add(1.0, values[index]));
    }

    let maximum_vector = vdupq_n_f32(maximum);
    let mut sums = [vdupq_n_f32(0.0); 4];
    for index in (0..vector_len).step_by(16) {
        for (vector, sum) in sums.iter_mut().enumerate() {
            let offset = index + vector * 4;
            let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            let bias = unsafe { vld1q_f32(bias.as_ptr().add(offset)) };
            let biased = vfmaq_n_f32(value, bias, 1.0);
            let exponential = expq(vsubq_f32(biased, maximum_vector));
            *sum = vaddq_f32(*sum, exponential);
            unsafe { vst1q_f32(values.as_mut_ptr().add(offset), exponential) };
        }
    }
    let mut sum =
        vaddvq_f32(sums[0]) + vaddvq_f32(sums[1]) + vaddvq_f32(sums[2]) + vaddvq_f32(sums[3]);
    for index in vector_len..values.len() {
        values[index] = (bias[index].mul_add(1.0, values[index]) - maximum).exp();
        sum += values[index];
    }

    let reciprocal = vdupq_n_f32(sum.recip());
    for index in (0..vector_len).step_by(16) {
        for vector in 0..4 {
            let offset = index + vector * 4;
            let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
            unsafe {
                vst1q_f32(
                    values.as_mut_ptr().add(offset),
                    vmulq_f32(value, reciprocal),
                )
            };
        }
    }
    let reciprocal = vgetq_lane_f32::<0>(reciprocal);
    for value in &mut values[vector_len..] {
        *value *= reciprocal;
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn sum(values: &[f32]) -> f32 {
    let vector_len = values.len() / 16 * 16;
    let mut sums = [vdupq_n_f32(0.0); 4];
    for index in (0..vector_len).step_by(16) {
        for (vector, sum) in sums.iter_mut().enumerate() {
            let value = unsafe { vld1q_f32(values.as_ptr().add(index + vector * 4)) };
            *sum = vaddq_f32(*sum, value);
        }
    }
    let mut sum =
        vaddvq_f32(sums[0]) + vaddvq_f32(sums[1]) + vaddvq_f32(sums[2]) + vaddvq_f32(sums[3]);
    for &value in &values[vector_len..] {
        sum += value;
    }
    sum
}

#[target_feature(enable = "neon")]
fn expq(value: float32x4_t) -> float32x4_t {
    let value = vmaxq_f32(vdupq_n_f32(-87.0), vminq_f32(vdupq_n_f32(87.0), value));
    let exponent = vcvtnq_s32_f32(vmulq_n_f32(value, std::f32::consts::LOG2_E));
    let remainder = vfmsq_n_f32(value, vcvtq_f32_s32(exponent), std::f32::consts::LN_2);
    // Range reduction bounds the fourth-order remainder tightly enough for
    // both GELU and normalized softmax outputs.
    let mut polynomial = vdupq_n_f32(1.0 / 24.0);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(0.5), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0), polynomial, remainder);
    let power_of_two =
        vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(exponent, vdupq_n_s32(127)), 23));
    vmulq_f32(polynomial, power_of_two)
}
