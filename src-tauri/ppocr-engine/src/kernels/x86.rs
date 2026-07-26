//! x86-64 AVX2 and FMA kernels.

use core::arch::x86_64::*;

// Every entry point in this module requires the caller to have checked both
// AVX2 and FMA at runtime. Loads and stores are unaligned and only cover full
// vectors; each kernel handles its remaining elements with safe slice access.

#[target_feature(enable = "avx2,fma")]
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
    unsafe {
        depthwise_conv2d_same_rows::<K>(output, input, weights, height, width, 0, height, bias)
    };
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn depthwise_conv2d_same_rows<const K: usize>(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    y_start: usize,
    rows: usize,
    bias: f32,
) {
    debug_assert!(matches!(K, 3 | 5 | 7 | 9));
    debug_assert!(y_start <= height && rows <= height - y_start);
    debug_assert_eq!(output.len(), rows * width);
    debug_assert_eq!(input.len(), height * width);
    debug_assert_eq!(weights.len(), K * K);
    let padding = K / 2;

    for local_y in 0..rows {
        let y = y_start + local_y;
        let kernel_y_start = padding.saturating_sub(y);
        let kernel_y_end = K.min(height + padding - y);
        let vector_start = if width >= K { padding } else { 0 };
        let vector_end = if width >= K { width - padding } else { 0 };

        for x in 0..vector_start {
            output[local_y * width + x] =
                unsafe { depthwise_conv2d_pixel::<K>(input, weights, height, width, y, x, bias) };
        }

        let mut x = vector_start;
        while x + 32 <= vector_end {
            let mut sums = [_mm256_set1_ps(bias); 4];
            for kernel_y in kernel_y_start..kernel_y_end {
                let input_y = y + kernel_y - padding;
                for kernel_x in 0..K {
                    let input_x = x + kernel_x - padding;
                    let input_base = unsafe { input.as_ptr().add(input_y * width + input_x) };
                    let weight =
                        _mm256_set1_ps(unsafe { *weights.get_unchecked(kernel_y * K + kernel_x) });
                    for (vector, sum) in sums.iter_mut().enumerate() {
                        let values = unsafe { _mm256_loadu_ps(input_base.add(vector * 8)) };
                        *sum = _mm256_fmadd_ps(values, weight, *sum);
                    }
                }
            }
            let output_base = unsafe { output.as_mut_ptr().add(local_y * width + x) };
            for (vector, sum) in sums.into_iter().enumerate() {
                unsafe { _mm256_storeu_ps(output_base.add(vector * 8), sum) };
            }
            x += 32;
        }

        while x + 8 <= vector_end {
            let mut sum = _mm256_set1_ps(bias);
            for kernel_y in kernel_y_start..kernel_y_end {
                let input_y = y + kernel_y - padding;
                for kernel_x in 0..K {
                    let input_x = x + kernel_x - padding;
                    let input_base = unsafe { input.as_ptr().add(input_y * width + input_x) };
                    let values = unsafe { _mm256_loadu_ps(input_base) };
                    let weight =
                        _mm256_set1_ps(unsafe { *weights.get_unchecked(kernel_y * K + kernel_x) });
                    sum = _mm256_fmadd_ps(values, weight, sum);
                }
            }
            unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(local_y * width + x), sum) };
            x += 8;
        }

        for x in x..width {
            output[local_y * width + x] =
                unsafe { depthwise_conv2d_pixel::<K>(input, weights, height, width, y, x, bias) };
        }
    }
}

#[target_feature(enable = "avx2,fma")]
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
    let even_indices = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);

    for output_y in 0..output_height {
        let center_y = output_y * 2;
        let kernel_y_start = usize::from(center_y == 0);
        let kernel_y_end = if center_y + 1 < height { 3 } else { 2 };
        output[output_y * output_width] = unsafe {
            depthwise_conv2d_stride2_pixel(input, weights, height, width, output_y, 0, bias)
        };

        let mut output_x = 1;
        while output_x + 32 <= output_width && 2 * output_x + 64 < width {
            let mut sums = [_mm256_set1_ps(bias); 4];
            for kernel_y in kernel_y_start..kernel_y_end {
                let input_y = center_y + kernel_y - 1;
                for kernel_x in 0..3 {
                    let input_x = output_x * 2 + kernel_x - 1;
                    let input_base = unsafe { input.as_ptr().add(input_y * width + input_x) };
                    let weight =
                        _mm256_set1_ps(unsafe { *weights.get_unchecked(kernel_y * 3 + kernel_x) });
                    for (vector, sum) in sums.iter_mut().enumerate() {
                        let input_base = unsafe { input_base.add(vector * 16) };
                        let low = unsafe { _mm256_loadu_ps(input_base) };
                        let high = unsafe { _mm256_loadu_ps(input_base.add(8)) };
                        let low = _mm256_permutevar8x32_ps(low, even_indices);
                        let high = _mm256_permutevar8x32_ps(high, even_indices);
                        let values = _mm256_permute2f128_ps::<0x20>(low, high);
                        *sum = _mm256_fmadd_ps(values, weight, *sum);
                    }
                }
            }
            let output_base =
                unsafe { output.as_mut_ptr().add(output_y * output_width + output_x) };
            for (vector, sum) in sums.into_iter().enumerate() {
                unsafe { _mm256_storeu_ps(output_base.add(vector * 8), sum) };
            }
            output_x += 32;
        }
        while output_x + 8 <= output_width && 2 * output_x + 16 < width {
            let mut sum = _mm256_set1_ps(bias);
            for kernel_y in kernel_y_start..kernel_y_end {
                let input_y = center_y + kernel_y - 1;
                for kernel_x in 0..3 {
                    let input_x = output_x * 2 + kernel_x - 1;
                    let input_base = unsafe { input.as_ptr().add(input_y * width + input_x) };
                    let low = unsafe { _mm256_loadu_ps(input_base) };
                    let high = unsafe { _mm256_loadu_ps(input_base.add(8)) };
                    let low = _mm256_permutevar8x32_ps(low, even_indices);
                    let high = _mm256_permutevar8x32_ps(high, even_indices);
                    let values = _mm256_permute2f128_ps::<0x20>(low, high);
                    let weight =
                        _mm256_set1_ps(unsafe { *weights.get_unchecked(kernel_y * 3 + kernel_x) });
                    sum = _mm256_fmadd_ps(values, weight, sum);
                }
            }
            unsafe {
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(output_y * output_width + output_x),
                    sum,
                )
            };
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

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(super) unsafe fn spatial_conv2d_packed<const OUTPUT_CHANNELS: usize>(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    bias: Option<&[f32]>,
    activation: Option<super::UnaryOperation>,
) {
    debug_assert!(matches!(OUTPUT_CHANNELS, 2 | 4 | 6));
    debug_assert!(input_channels > 0);
    debug_assert!(input_height > 0 && input_width > 0);
    debug_assert!(output_height > 0 && output_width > 0);
    debug_assert!(kernel_height > 0 && kernel_width > 0);
    debug_assert!(strides.into_iter().all(|stride| matches!(stride, 1 | 2)));
    debug_assert_eq!(input.len(), input_channels * input_height * input_width);
    debug_assert_eq!(
        weight.len(),
        input_channels * kernel_height * kernel_width * OUTPUT_CHANNELS
    );
    debug_assert_eq!(output.len(), OUTPUT_CHANNELS * output_height * output_width);
    debug_assert!(bias.is_none_or(|bias| bias.len() == OUTPUT_CHANNELS));
    debug_assert!(input_height + pads[0] + pads[2] >= kernel_height);
    debug_assert!(input_width + pads[1] + pads[3] >= kernel_width);
    debug_assert_eq!(
        output_height,
        (input_height + pads[0] + pads[2] - kernel_height) / strides[0] + 1
    );
    debug_assert_eq!(
        output_width,
        (input_width + pads[1] + pads[3] - kernel_width) / strides[1] + 1
    );

    let input_plane = input_height * input_width;
    let output_plane = output_height * output_width;
    let vector_start = pads[1].div_ceil(strides[1]).min(output_width);
    let horizontal_extent = input_width + pads[1];
    let vector_end = if horizontal_extent >= kernel_width {
        ((horizontal_extent - kernel_width) / strides[1] + 1).min(output_width)
    } else {
        0
    };
    let gather_low = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let gather_high = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);

    for output_y in 0..output_height {
        for output_x in 0..vector_start {
            let values = spatial_conv2d_packed_pixel::<OUTPUT_CHANNELS>(
                input,
                weight,
                input_channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                output_y,
                output_x,
                strides,
                pads,
                bias,
                activation,
            );
            for output_channel in 0..OUTPUT_CHANNELS {
                output[output_channel * output_plane + output_y * output_width + output_x] =
                    values[output_channel];
            }
        }

        let mut output_x = vector_start;
        while output_x + 16 <= vector_end {
            let mut sums = [[_mm256_setzero_ps(); 2]; OUTPUT_CHANNELS];
            for output_channel in 0..OUTPUT_CHANNELS {
                let initial = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[output_channel]));
                sums[output_channel] = [initial; 2];
            }

            for input_channel in 0..input_channels {
                let channel_base = input_channel * input_plane;
                for kernel_y in 0..kernel_height {
                    let padded_input_y = output_y * strides[0] + kernel_y;
                    if padded_input_y < pads[0] || padded_input_y - pads[0] >= input_height {
                        continue;
                    }
                    let input_y = padded_input_y - pads[0];
                    for kernel_x in 0..kernel_width {
                        let padded_input_x = output_x * strides[1] + kernel_x;
                        debug_assert!(padded_input_x >= pads[1]);
                        let input_x = padded_input_x - pads[1];
                        let input_base = unsafe {
                            input
                                .as_ptr()
                                .add(channel_base + input_y * input_width + input_x)
                        };
                        let (input0, input1) = if strides[1] == 1 {
                            unsafe {
                                (
                                    _mm256_loadu_ps(input_base),
                                    _mm256_loadu_ps(input_base.add(8)),
                                )
                            }
                        } else {
                            unsafe {
                                (
                                    _mm256_i32gather_ps::<4>(input_base, gather_low),
                                    _mm256_i32gather_ps::<4>(input_base, gather_high),
                                )
                            }
                        };
                        let weight_base =
                            ((input_channel * kernel_height + kernel_y) * kernel_width + kernel_x)
                                * OUTPUT_CHANNELS;
                        for output_channel in 0..OUTPUT_CHANNELS {
                            let scale = _mm256_set1_ps(unsafe {
                                *weight.get_unchecked(weight_base + output_channel)
                            });
                            sums[output_channel][0] =
                                _mm256_fmadd_ps(input0, scale, sums[output_channel][0]);
                            sums[output_channel][1] =
                                _mm256_fmadd_ps(input1, scale, sums[output_channel][1]);
                        }
                    }
                }
            }

            for output_channel in 0..OUTPUT_CHANNELS {
                let output_base = unsafe {
                    output
                        .as_mut_ptr()
                        .add(output_channel * output_plane + output_y * output_width + output_x)
                };
                unsafe {
                    _mm256_storeu_ps(
                        output_base,
                        apply_vector_post_op(sums[output_channel][0], activation),
                    );
                    _mm256_storeu_ps(
                        output_base.add(8),
                        apply_vector_post_op(sums[output_channel][1], activation),
                    );
                }
            }
            output_x += 16;
        }

        for output_x in output_x..output_width {
            let values = spatial_conv2d_packed_pixel::<OUTPUT_CHANNELS>(
                input,
                weight,
                input_channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                output_y,
                output_x,
                strides,
                pads,
                bias,
                activation,
            );
            for output_channel in 0..OUTPUT_CHANNELS {
                output[output_channel * output_plane + output_y * output_width + output_x] =
                    values[output_channel];
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn spatial_conv2d_packed_pixel<const OUTPUT_CHANNELS: usize>(
    input: &[f32],
    weight: &[f32],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    output_y: usize,
    output_x: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    bias: Option<&[f32]>,
    activation: Option<super::UnaryOperation>,
) -> [f32; OUTPUT_CHANNELS] {
    let input_plane = input_height * input_width;
    let mut sums =
        std::array::from_fn(|output_channel| bias.map_or(0.0, |bias| bias[output_channel]));
    for input_channel in 0..input_channels {
        let channel_base = input_channel * input_plane;
        for kernel_y in 0..kernel_height {
            let padded_input_y = output_y * strides[0] + kernel_y;
            if padded_input_y < pads[0] || padded_input_y - pads[0] >= input_height {
                continue;
            }
            let input_y = padded_input_y - pads[0];
            for kernel_x in 0..kernel_width {
                let padded_input_x = output_x * strides[1] + kernel_x;
                if padded_input_x < pads[1] || padded_input_x - pads[1] >= input_width {
                    continue;
                }
                let input_x = padded_input_x - pads[1];
                let input_value = input[channel_base + input_y * input_width + input_x];
                let weight_base = ((input_channel * kernel_height + kernel_y) * kernel_width
                    + kernel_x)
                    * OUTPUT_CHANNELS;
                for output_channel in 0..OUTPUT_CHANNELS {
                    sums[output_channel] = input_value
                        .mul_add(weight[weight_base + output_channel], sums[output_channel]);
                }
            }
        }
    }
    for sum in &mut sums {
        *sum = apply_scalar_post_op(*sum, activation);
    }
    sums
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn copy_stride2_16(output: *mut f32, input: *const f32) {
    let low_indices = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let high_indices = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    let low = unsafe { _mm256_i32gather_ps::<4>(input, low_indices) };
    let high = unsafe { _mm256_i32gather_ps::<4>(input, high_indices) };
    unsafe {
        _mm256_storeu_ps(output, low);
        _mm256_storeu_ps(output.add(8), high);
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn max_pool_2x2_row(output: &mut [f32], current: &[f32], next: Option<&[f32]>) {
    debug_assert_eq!(output.len(), current.len());
    debug_assert!(next.is_none_or(|next| next.len() == current.len()));

    // The shifted load needs one additional source value, so leave the final
    // one-to-eight outputs to the scalar tail.
    let vector_len = current.len().saturating_sub(1) / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        let current0 = unsafe { _mm256_loadu_ps(current.as_ptr().add(offset)) };
        let current1 = unsafe { _mm256_loadu_ps(current.as_ptr().add(offset + 1)) };
        let mut maximum = max_number(current0, current1);
        if let Some(next) = next {
            let next0 = unsafe { _mm256_loadu_ps(next.as_ptr().add(offset)) };
            let next1 = unsafe { _mm256_loadu_ps(next.as_ptr().add(offset + 1)) };
            maximum = max_number(maximum, next0);
            maximum = max_number(maximum, next1);
        }
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(offset), maximum) };
    }

    for x in vector_len..current.len() {
        let mut maximum = current[x];
        if x + 1 < current.len() {
            maximum = maximum.max(current[x + 1]);
        }
        if let Some(next) = next {
            maximum = maximum.max(next[x]);
            if x + 1 < next.len() {
                maximum = maximum.max(next[x + 1]);
            }
        }
        output[x] = maximum;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn max_number(left: __m256, right: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let left_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(left, left);
    let right_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(right, right);
    let right_only_nan = _mm256_andnot_ps(left_nan, right_nan);
    let mut maximum = _mm256_max_ps(left, right);
    maximum = _mm256_blendv_ps(maximum, left, right_only_nan);

    // `f32::max` selects +0.0 when the operands are opposite signed zeros.
    let both_zero = _mm256_and_ps(
        _mm256_cmp_ps::<_CMP_EQ_OQ>(left, zero),
        _mm256_cmp_ps::<_CMP_EQ_OQ>(right, zero),
    );
    let maximum_zero = _mm256_and_ps(left, right);
    _mm256_blendv_ps(maximum, maximum_zero, both_zero)
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

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn axpy(output: &mut [f32], input: &[f32], scale: f32) {
    debug_assert_eq!(output.len(), input.len());
    let scale = _mm256_set1_ps(scale);
    let vector_len = output.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is inside both equal-length slices.
        let (input, output_value) = unsafe {
            (
                _mm256_loadu_ps(input.as_ptr().add(offset)),
                _mm256_loadu_ps(output.as_ptr().add(offset)),
            )
        };
        let result = _mm256_fmadd_ps(input, scale, output_value);
        // SAFETY: The same full-vector bound applies to the output store.
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(offset), result) };
    }
    let scale = _mm256_cvtss_f32(scale);
    for index in vector_len..output.len() {
        output[index] = input[index].mul_add(scale, output[index]);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn mul_in_place(output: &mut [f32], input: &[f32]) {
    debug_assert_eq!(output.len(), input.len());
    let vector_len = output.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is inside both equal-length slices.
        let (left, right) = unsafe {
            (
                _mm256_loadu_ps(output.as_ptr().add(offset)),
                _mm256_loadu_ps(input.as_ptr().add(offset)),
            )
        };
        // SAFETY: The store covers the same in-bounds output vector.
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(offset), _mm256_mul_ps(left, right)) };
    }
    for index in vector_len..output.len() {
        output[index] *= input[index];
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn affine(values: &mut [f32], scale: f32, bias: f32) {
    let scale_vector = _mm256_set1_ps(scale);
    let bias_vector = _mm256_set1_ps(bias);
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let result = _mm256_fmadd_ps(value, scale_vector, bias_vector);
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), result) };
    }
    for value in &mut values[vector_len..] {
        *value = value.mul_add(scale, bias);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn residual_mul(values: &mut [f32], gate: f32) {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let gate_vector = _mm256_set1_ps(gate);
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let original = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let scaled = _mm256_fmadd_ps(original, gate_vector, zero);
        let output = _mm256_fmadd_ps(scaled, one, original);
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), output) };
    }
    for value in &mut values[vector_len..] {
        let original = *value;
        let scaled = original.mul_add(gate, 0.0);
        *value = scaled.mul_add(1.0, original);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn square(values: &mut [f32]) {
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), _mm256_mul_ps(value, value)) };
    }
    for value in &mut values[vector_len..] {
        *value *= *value;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn relu(values: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // Passing zero as the second operand also matches `f32::max` for NaN.
        let result = _mm256_max_ps(value, zero);
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), result) };
    }
    for value in &mut values[vector_len..] {
        *value = value.max(0.0);
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_rows_8<const ROWS: usize, const PACKED_LEFT: bool>(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
    activation: Option<super::UnaryOperation>,
) {
    debug_assert!(ROWS > 0 && ROWS <= 12);
    debug_assert_eq!(output.len(), ROWS * columns);
    debug_assert_eq!(left.len(), ROWS * inner);
    debug_assert!(inner == 0 || right.len() >= inner.saturating_sub(1) * right_stride + columns);
    debug_assert!(bias.is_none_or(|bias| bias.len() == ROWS));
    debug_assert!(column_bias.is_none_or(|bias| bias.len() == columns));
    debug_assert!(bias.is_none() || column_bias.is_none());
    debug_assert!(!PACKED_LEFT || column_bias.is_none());

    let vector_columns = columns / 8 * 8;
    for column in (0..vector_columns).step_by(8) {
        let mut accumulators = [_mm256_setzero_ps(); ROWS];
        if let Some(column_bias) = column_bias {
            // SAFETY: A full vector is available because `column < vector_columns`.
            let initial = unsafe { _mm256_loadu_ps(column_bias.as_ptr().add(column)) };
            accumulators.fill(initial);
        } else {
            for row in 0..ROWS {
                accumulators[row] = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[row]));
            }
        }

        macro_rules! k_step {
            ($index:expr) => {{
                let index = $index;
                // SAFETY: Matrix dimensions and `right_stride` were validated above.
                let right_vector =
                    unsafe { _mm256_loadu_ps(right.as_ptr().add(index * right_stride + column)) };
                for (row, accumulator) in accumulators.iter_mut().enumerate() {
                    let left_index = if PACKED_LEFT {
                        index * ROWS + row
                    } else {
                        row * inner + index
                    };
                    // SAFETY: Both supported layouts contain exactly `ROWS * inner` values.
                    let scale = _mm256_set1_ps(unsafe { *left.get_unchecked(left_index) });
                    *accumulator = _mm256_fmadd_ps(scale, right_vector, *accumulator);
                }
            }};
        }
        const PREFETCH_DISTANCE: usize = 16;
        let mut index = 0;
        while index + 4 <= inner {
            if inner - index > PREFETCH_DISTANCE {
                let prefetch = index + PREFETCH_DISTANCE;
                unsafe {
                    _mm_prefetch::<_MM_HINT_T0>(
                        right.as_ptr().add(prefetch * right_stride + column).cast(),
                    );
                    if PACKED_LEFT {
                        _mm_prefetch::<_MM_HINT_T0>(left.as_ptr().add(prefetch * ROWS).cast());
                    }
                }
            }
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

        for (row, accumulator) in accumulators.iter().copied().enumerate() {
            // SAFETY: Each output row has `columns` values and this is a full vector.
            unsafe {
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(row * columns + column),
                    apply_vector_post_op(accumulator, activation),
                )
            };
        }
    }

    for row in 0..ROWS {
        for column in vector_columns..columns {
            let mut sum =
                column_bias.map_or_else(|| bias.map_or(0.0, |bias| bias[row]), |bias| bias[column]);
            for index in 0..inner {
                let left_index = if PACKED_LEFT {
                    index * ROWS + row
                } else {
                    row * inner + index
                };
                sum = left[left_index].mul_add(right[index * right_stride + column], sum);
            }
            output[row * columns + column] = apply_scalar_post_op(sum, activation);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(super) unsafe fn linear_rows_8<const ROWS: usize>(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<super::UnaryOperation>,
) {
    debug_assert!(ROWS > 0 && ROWS <= 8);
    debug_assert_eq!(output.len(), ROWS * columns);
    debug_assert_eq!(input.len(), ROWS * inner);
    debug_assert_eq!(weight.len(), columns * inner);
    debug_assert!(bias.is_none_or(|bias| bias.len() == columns));

    let vector_columns = columns / 8 * 8;
    for column in (0..vector_columns).step_by(8) {
        let initial = bias.map_or_else(
            || _mm256_setzero_ps(),
            |bias| unsafe { _mm256_loadu_ps(bias.as_ptr().add(column)) },
        );
        let mut accumulators = [initial; ROWS];
        let weight_base = column * inner;
        for index in 0..inner {
            // SAFETY: Each complete output block stores eight weights per K index.
            let weights = unsafe { _mm256_loadu_ps(weight.as_ptr().add(weight_base + index * 8)) };
            for (row, accumulator) in accumulators.iter_mut().enumerate() {
                let value = _mm256_set1_ps(unsafe { *input.get_unchecked(row * inner + index) });
                *accumulator = _mm256_fmadd_ps(value, weights, *accumulator);
            }
        }
        for (row, accumulator) in accumulators.into_iter().enumerate() {
            unsafe {
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(row * columns + column),
                    apply_vector_post_op(accumulator, activation),
                )
            };
        }
    }

    let tail_columns = columns - vector_columns;
    if tail_columns == 0 {
        return;
    }
    let weight_base = vector_columns * inner;
    for row in 0..ROWS {
        for lane in 0..tail_columns {
            let column = vector_columns + lane;
            let mut sum = bias.map_or(0.0, |bias| bias[column]);
            for index in 0..inner {
                sum = input[row * inner + index]
                    .mul_add(weight[weight_base + index * tail_columns + lane], sum);
            }
            output[row * columns + column] = apply_scalar_post_op(sum, activation);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(super) unsafe fn linear_6x16_packed<const ROWS: usize>(
    output: &mut [f32],
    output_stride: usize,
    packed_input: &[f32],
    weight: &[f32],
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<super::UnaryOperation>,
) {
    debug_assert!(ROWS > 0 && ROWS <= 6);
    debug_assert!(columns > 0 && columns <= 16);
    debug_assert!(output_stride >= columns);
    debug_assert!(output.len() >= (ROWS - 1) * output_stride + columns);
    debug_assert_eq!(packed_input.len(), ROWS * inner);
    debug_assert_eq!(weight.len(), columns * inner);
    debug_assert!(bias.is_none_or(|bias| bias.len() == columns));

    let lane_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let low_columns = columns.min(8);
    let high_columns = columns.saturating_sub(8);
    let low_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(low_columns as i32), lane_indices);
    let high_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(high_columns as i32), lane_indices);
    let zero = _mm256_setzero_ps();
    let (initial0, initial1) = match bias {
        Some(bias) => {
            let initial0 = if low_columns == 8 {
                unsafe { _mm256_loadu_ps(bias.as_ptr()) }
            } else {
                unsafe { _mm256_maskload_ps(bias.as_ptr(), low_mask) }
            };
            let initial1 = if high_columns == 8 {
                unsafe { _mm256_loadu_ps(bias.as_ptr().add(8)) }
            } else if high_columns > 0 {
                unsafe { _mm256_maskload_ps(bias.as_ptr().add(8), high_mask) }
            } else {
                zero
            };
            (initial0, initial1)
        }
        None => (zero, zero),
    };

    let mut sum00 = initial0;
    let mut sum01 = initial1;
    let mut sum10 = initial0;
    let mut sum11 = initial1;
    let mut sum20 = initial0;
    let mut sum21 = initial1;
    let mut sum30 = initial0;
    let mut sum31 = initial1;
    let mut sum40 = initial0;
    let mut sum41 = initial1;
    let mut sum50 = initial0;
    let mut sum51 = initial1;

    macro_rules! accumulate {
        ($index:expr, $weight0:expr, $weight1:expr) => {{
            let input = unsafe { packed_input.as_ptr().add($index * ROWS) };
            let scale = _mm256_set1_ps(unsafe { *input });
            sum00 = _mm256_fmadd_ps(scale, $weight0, sum00);
            sum01 = _mm256_fmadd_ps(scale, $weight1, sum01);
            if ROWS > 1 {
                let scale = _mm256_set1_ps(unsafe { *input.add(1) });
                sum10 = _mm256_fmadd_ps(scale, $weight0, sum10);
                sum11 = _mm256_fmadd_ps(scale, $weight1, sum11);
            }
            if ROWS > 2 {
                let scale = _mm256_set1_ps(unsafe { *input.add(2) });
                sum20 = _mm256_fmadd_ps(scale, $weight0, sum20);
                sum21 = _mm256_fmadd_ps(scale, $weight1, sum21);
            }
            if ROWS > 3 {
                let scale = _mm256_set1_ps(unsafe { *input.add(3) });
                sum30 = _mm256_fmadd_ps(scale, $weight0, sum30);
                sum31 = _mm256_fmadd_ps(scale, $weight1, sum31);
            }
            if ROWS > 4 {
                let scale = _mm256_set1_ps(unsafe { *input.add(4) });
                sum40 = _mm256_fmadd_ps(scale, $weight0, sum40);
                sum41 = _mm256_fmadd_ps(scale, $weight1, sum41);
            }
            if ROWS > 5 {
                let scale = _mm256_set1_ps(unsafe { *input.add(5) });
                sum50 = _mm256_fmadd_ps(scale, $weight0, sum50);
                sum51 = _mm256_fmadd_ps(scale, $weight1, sum51);
            }
        }};
    }

    macro_rules! full_step {
        ($index:expr) => {{
            let weight = unsafe { weight.as_ptr().add($index * 16) };
            let weight0 = unsafe { _mm256_loadu_ps(weight) };
            let weight1 = unsafe { _mm256_loadu_ps(weight.add(8)) };
            accumulate!($index, weight0, weight1);
        }};
    }

    macro_rules! tail_step {
        ($index:expr) => {{
            let weight = unsafe { weight.as_ptr().add($index * columns) };
            let weight0 = if low_columns == 8 {
                unsafe { _mm256_loadu_ps(weight) }
            } else {
                unsafe { _mm256_maskload_ps(weight, low_mask) }
            };
            let weight1 = if high_columns > 0 {
                unsafe { _mm256_maskload_ps(weight.add(8), high_mask) }
            } else {
                zero
            };
            accumulate!($index, weight0, weight1);
        }};
    }

    let mut index = 0;
    if columns == 16 {
        while index + 4 <= inner {
            full_step!(index);
            full_step!(index + 1);
            full_step!(index + 2);
            full_step!(index + 3);
            index += 4;
        }
        while index < inner {
            full_step!(index);
            index += 1;
        }
    } else {
        while index + 4 <= inner {
            tail_step!(index);
            tail_step!(index + 1);
            tail_step!(index + 2);
            tail_step!(index + 3);
            index += 4;
        }
        while index < inner {
            tail_step!(index);
            index += 1;
        }
    }

    macro_rules! store_row {
        ($row:expr, $sum0:expr, $sum1:expr) => {{
            let output = unsafe { output.as_mut_ptr().add($row * output_stride) };
            let value0 = apply_vector_post_op($sum0, activation);
            if low_columns == 8 {
                unsafe { _mm256_storeu_ps(output, value0) };
            } else {
                unsafe { _mm256_maskstore_ps(output, low_mask, value0) };
            }
            if high_columns > 0 {
                let value1 = apply_vector_post_op($sum1, activation);
                if high_columns == 8 {
                    unsafe { _mm256_storeu_ps(output.add(8), value1) };
                } else {
                    unsafe { _mm256_maskstore_ps(output.add(8), high_mask, value1) };
                }
            }
        }};
    }

    store_row!(0, sum00, sum01);
    if ROWS > 1 {
        store_row!(1, sum10, sum11);
    }
    if ROWS > 2 {
        store_row!(2, sum20, sum21);
    }
    if ROWS > 3 {
        store_row!(3, sum30, sum31);
    }
    if ROWS > 4 {
        store_row!(4, sum40, sum41);
    }
    if ROWS > 5 {
        store_row!(5, sum50, sum51);
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(super) unsafe fn gemm_4x16_sparse(
    output: &mut [f32],
    right: &[f32],
    indices: &[u32],
    weights: &[f32],
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    activation: Option<super::UnaryOperation>,
) {
    const ROWS: usize = 4;

    debug_assert_eq!(output.len(), ROWS * columns);
    debug_assert_eq!(weights.len(), indices.len() * ROWS);
    debug_assert!(bias.is_none_or(|bias| bias.len() == ROWS));
    debug_assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(indices.iter().all(|&index| {
        (index as usize)
            .checked_mul(right_stride)
            .and_then(|start| start.checked_add(columns))
            .is_some_and(|end| end <= right.len())
    }));

    let vector_columns = columns / 16 * 16;
    for column in (0..vector_columns).step_by(16) {
        let mut accumulators = [_mm256_setzero_ps(); ROWS * 2];
        for row in 0..ROWS {
            let initial = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[row]));
            accumulators[row * 2] = initial;
            accumulators[row * 2 + 1] = initial;
        }
        for (entry, &index) in indices.iter().enumerate() {
            let right_base = unsafe { right.as_ptr().add(index as usize * right_stride + column) };
            let right0 = unsafe { _mm256_loadu_ps(right_base) };
            let right1 = unsafe { _mm256_loadu_ps(right_base.add(8)) };
            for row in 0..ROWS {
                let scale = _mm256_set1_ps(unsafe { *weights.get_unchecked(entry * ROWS + row) });
                accumulators[row * 2] = _mm256_fmadd_ps(scale, right0, accumulators[row * 2]);
                accumulators[row * 2 + 1] =
                    _mm256_fmadd_ps(scale, right1, accumulators[row * 2 + 1]);
            }
        }
        for row in 0..ROWS {
            let output_base = unsafe { output.as_mut_ptr().add(row * columns + column) };
            unsafe {
                _mm256_storeu_ps(
                    output_base,
                    apply_vector_post_op(accumulators[row * 2], activation),
                );
                _mm256_storeu_ps(
                    output_base.add(8),
                    apply_vector_post_op(accumulators[row * 2 + 1], activation),
                );
            }
        }
    }

    let mut column = vector_columns;
    if column + 8 <= columns {
        let mut accumulators = [_mm256_setzero_ps(); ROWS];
        for row in 0..ROWS {
            accumulators[row] = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[row]));
        }
        for (entry, &index) in indices.iter().enumerate() {
            let right_value = unsafe {
                _mm256_loadu_ps(right.as_ptr().add(index as usize * right_stride + column))
            };
            for row in 0..ROWS {
                let scale = _mm256_set1_ps(unsafe { *weights.get_unchecked(entry * ROWS + row) });
                accumulators[row] = _mm256_fmadd_ps(scale, right_value, accumulators[row]);
            }
        }
        for (row, accumulator) in accumulators.into_iter().enumerate() {
            unsafe {
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(row * columns + column),
                    apply_vector_post_op(accumulator, activation),
                )
            };
        }
        column += 8;
    }

    for row in 0..ROWS {
        for column in column..columns {
            let mut sum = bias.map_or(0.0, |bias| bias[row]);
            for (entry, &index) in indices.iter().enumerate() {
                sum = weights[entry * ROWS + row]
                    .mul_add(right[index as usize * right_stride + column], sum);
            }
            output[row * columns + column] = apply_scalar_post_op(sum, activation);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_4x16_packed(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    output_stride: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    accumulate: bool,
    activation: Option<super::UnaryOperation>,
) {
    const ROWS: usize = 4;
    const PREFETCH_DISTANCE: usize = 16;

    debug_assert!(output_stride >= columns);
    debug_assert!(columns == 0 || output.len() >= (ROWS - 1) * output_stride + columns);
    debug_assert_eq!(left.len(), ROWS * inner);
    debug_assert!(right_stride >= columns);
    debug_assert!(inner == 0 || right.len() >= inner.saturating_sub(1) * right_stride + columns);
    debug_assert!(bias.is_none_or(|bias| bias.len() == ROWS));

    let vector_columns = columns / 16 * 16;
    for column in (0..vector_columns).step_by(16) {
        let initial0 = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[0]));
        let initial1 = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[1]));
        let initial2 = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[2]));
        let initial3 = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[3]));
        let output0 = unsafe { output.as_ptr().add(column) };
        let output1 = unsafe { output.as_ptr().add(output_stride + column) };
        let output2 = unsafe { output.as_ptr().add(2 * output_stride + column) };
        let output3 = unsafe { output.as_ptr().add(3 * output_stride + column) };
        let mut sum00 = if accumulate {
            unsafe { _mm256_loadu_ps(output0) }
        } else {
            initial0
        };
        let mut sum01 = if accumulate {
            unsafe { _mm256_loadu_ps(output0.add(8)) }
        } else {
            initial0
        };
        let mut sum10 = if accumulate {
            unsafe { _mm256_loadu_ps(output1) }
        } else {
            initial1
        };
        let mut sum11 = if accumulate {
            unsafe { _mm256_loadu_ps(output1.add(8)) }
        } else {
            initial1
        };
        let mut sum20 = if accumulate {
            unsafe { _mm256_loadu_ps(output2) }
        } else {
            initial2
        };
        let mut sum21 = if accumulate {
            unsafe { _mm256_loadu_ps(output2.add(8)) }
        } else {
            initial2
        };
        let mut sum30 = if accumulate {
            unsafe { _mm256_loadu_ps(output3) }
        } else {
            initial3
        };
        let mut sum31 = if accumulate {
            unsafe { _mm256_loadu_ps(output3.add(8)) }
        } else {
            initial3
        };
        macro_rules! k_step {
            ($index:expr) => {{
                let index = $index;
                let right_base = unsafe { right.as_ptr().add(index * right_stride + column) };
                let right0 = unsafe { _mm256_loadu_ps(right_base) };
                let right1 = unsafe { _mm256_loadu_ps(right_base.add(8)) };
                let left = unsafe { left.as_ptr().add(index * ROWS) };
                let scale0 = _mm256_set1_ps(unsafe { *left });
                let scale1 = _mm256_set1_ps(unsafe { *left.add(1) });
                let scale2 = _mm256_set1_ps(unsafe { *left.add(2) });
                let scale3 = _mm256_set1_ps(unsafe { *left.add(3) });
                sum00 = _mm256_fmadd_ps(scale0, right0, sum00);
                sum01 = _mm256_fmadd_ps(scale0, right1, sum01);
                sum10 = _mm256_fmadd_ps(scale1, right0, sum10);
                sum11 = _mm256_fmadd_ps(scale1, right1, sum11);
                sum20 = _mm256_fmadd_ps(scale2, right0, sum20);
                sum21 = _mm256_fmadd_ps(scale2, right1, sum21);
                sum30 = _mm256_fmadd_ps(scale3, right0, sum30);
                sum31 = _mm256_fmadd_ps(scale3, right1, sum31);
            }};
        }
        let mut index = 0;
        while index + 4 <= inner {
            if inner - index > PREFETCH_DISTANCE {
                let prefetch = index + PREFETCH_DISTANCE;
                unsafe {
                    _mm_prefetch::<_MM_HINT_T0>(left.as_ptr().add(prefetch * ROWS).cast());
                    _mm_prefetch::<_MM_HINT_T0>(
                        right.as_ptr().add(prefetch * right_stride + column).cast(),
                    );
                }
            }
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
        unsafe {
            _mm256_storeu_ps(
                output.as_mut_ptr().add(column),
                apply_vector_post_op(sum00, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(column + 8),
                apply_vector_post_op(sum01, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(output_stride + column),
                apply_vector_post_op(sum10, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(output_stride + column + 8),
                apply_vector_post_op(sum11, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(2 * output_stride + column),
                apply_vector_post_op(sum20, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(2 * output_stride + column + 8),
                apply_vector_post_op(sum21, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(3 * output_stride + column),
                apply_vector_post_op(sum30, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(3 * output_stride + column + 8),
                apply_vector_post_op(sum31, activation),
            );
        }
    }

    let mut column = vector_columns;
    if column + 8 <= columns {
        let mut sum0 = if accumulate {
            unsafe { _mm256_loadu_ps(output.as_ptr().add(column)) }
        } else {
            _mm256_set1_ps(bias.map_or(0.0, |bias| bias[0]))
        };
        let mut sum1 = if accumulate {
            unsafe { _mm256_loadu_ps(output.as_ptr().add(output_stride + column)) }
        } else {
            _mm256_set1_ps(bias.map_or(0.0, |bias| bias[1]))
        };
        let mut sum2 = if accumulate {
            unsafe { _mm256_loadu_ps(output.as_ptr().add(2 * output_stride + column)) }
        } else {
            _mm256_set1_ps(bias.map_or(0.0, |bias| bias[2]))
        };
        let mut sum3 = if accumulate {
            unsafe { _mm256_loadu_ps(output.as_ptr().add(3 * output_stride + column)) }
        } else {
            _mm256_set1_ps(bias.map_or(0.0, |bias| bias[3]))
        };
        macro_rules! k_step {
            ($index:expr) => {{
                let index = $index;
                let right_value =
                    unsafe { _mm256_loadu_ps(right.as_ptr().add(index * right_stride + column)) };
                let left = unsafe { left.as_ptr().add(index * ROWS) };
                let scale0 = _mm256_set1_ps(unsafe { *left });
                let scale1 = _mm256_set1_ps(unsafe { *left.add(1) });
                let scale2 = _mm256_set1_ps(unsafe { *left.add(2) });
                let scale3 = _mm256_set1_ps(unsafe { *left.add(3) });
                sum0 = _mm256_fmadd_ps(scale0, right_value, sum0);
                sum1 = _mm256_fmadd_ps(scale1, right_value, sum1);
                sum2 = _mm256_fmadd_ps(scale2, right_value, sum2);
                sum3 = _mm256_fmadd_ps(scale3, right_value, sum3);
            }};
        }
        let mut index = 0;
        while index + 4 <= inner {
            if inner - index > PREFETCH_DISTANCE {
                let prefetch = index + PREFETCH_DISTANCE;
                unsafe {
                    _mm_prefetch::<_MM_HINT_T0>(left.as_ptr().add(prefetch * ROWS).cast());
                    _mm_prefetch::<_MM_HINT_T0>(
                        right.as_ptr().add(prefetch * right_stride + column).cast(),
                    );
                }
            }
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
        unsafe {
            _mm256_storeu_ps(
                output.as_mut_ptr().add(column),
                apply_vector_post_op(sum0, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(output_stride + column),
                apply_vector_post_op(sum1, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(2 * output_stride + column),
                apply_vector_post_op(sum2, activation),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(3 * output_stride + column),
                apply_vector_post_op(sum3, activation),
            );
        }
        column += 8;
    }

    for row in 0..ROWS {
        for column in column..columns {
            let mut sum = if accumulate {
                output[row * output_stride + column]
            } else {
                bias.map_or(0.0, |bias| bias[row])
            };
            for index in 0..inner {
                sum = left[index * ROWS + row].mul_add(right[index * right_stride + column], sum);
            }
            output[row * output_stride + column] = apply_scalar_post_op(sum, activation);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_6x16_packed<const SOFTWARE_PREFETCH: bool>(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    output_stride: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    accumulate: bool,
    activation: Option<super::UnaryOperation>,
) {
    const ROWS: usize = 6;
    const PREFETCH_DISTANCE: usize = 16;

    debug_assert!(output_stride >= columns);
    debug_assert!(columns == 0 || output.len() >= (ROWS - 1) * output_stride + columns);
    debug_assert_eq!(left.len(), ROWS * inner);
    debug_assert!(right_stride >= columns);
    debug_assert!(inner == 0 || right.len() >= inner.saturating_sub(1) * right_stride + columns);
    debug_assert!(bias.is_none_or(|bias| bias.len() == ROWS));

    let vector_columns = columns / 16 * 16;
    for column in (0..vector_columns).step_by(16) {
        macro_rules! initial {
            ($row:expr, $lane:expr) => {{
                if accumulate {
                    unsafe {
                        _mm256_loadu_ps(output.as_ptr().add($row * output_stride + column + $lane))
                    }
                } else {
                    _mm256_set1_ps(bias.map_or(0.0, |bias| bias[$row]))
                }
            }};
        }
        let mut sum00 = initial!(0, 0);
        let mut sum01 = initial!(0, 8);
        let mut sum10 = initial!(1, 0);
        let mut sum11 = initial!(1, 8);
        let mut sum20 = initial!(2, 0);
        let mut sum21 = initial!(2, 8);
        let mut sum30 = initial!(3, 0);
        let mut sum31 = initial!(3, 8);
        let mut sum40 = initial!(4, 0);
        let mut sum41 = initial!(4, 8);
        let mut sum50 = initial!(5, 0);
        let mut sum51 = initial!(5, 8);

        macro_rules! k_step {
            ($index:expr) => {{
                let index = $index;
                let right_base = unsafe { right.as_ptr().add(index * right_stride + column) };
                let right0 = unsafe { _mm256_loadu_ps(right_base) };
                let right1 = unsafe { _mm256_loadu_ps(right_base.add(8)) };
                let left = unsafe { left.as_ptr().add(index * ROWS) };
                let scale = _mm256_set1_ps(unsafe { *left });
                sum00 = _mm256_fmadd_ps(scale, right0, sum00);
                sum01 = _mm256_fmadd_ps(scale, right1, sum01);
                let scale = _mm256_set1_ps(unsafe { *left.add(1) });
                sum10 = _mm256_fmadd_ps(scale, right0, sum10);
                sum11 = _mm256_fmadd_ps(scale, right1, sum11);
                let scale = _mm256_set1_ps(unsafe { *left.add(2) });
                sum20 = _mm256_fmadd_ps(scale, right0, sum20);
                sum21 = _mm256_fmadd_ps(scale, right1, sum21);
                let scale = _mm256_set1_ps(unsafe { *left.add(3) });
                sum30 = _mm256_fmadd_ps(scale, right0, sum30);
                sum31 = _mm256_fmadd_ps(scale, right1, sum31);
                let scale = _mm256_set1_ps(unsafe { *left.add(4) });
                sum40 = _mm256_fmadd_ps(scale, right0, sum40);
                sum41 = _mm256_fmadd_ps(scale, right1, sum41);
                let scale = _mm256_set1_ps(unsafe { *left.add(5) });
                sum50 = _mm256_fmadd_ps(scale, right0, sum50);
                sum51 = _mm256_fmadd_ps(scale, right1, sum51);
            }};
        }

        let mut index = 0;
        while index + 4 <= inner {
            if SOFTWARE_PREFETCH && inner - index > PREFETCH_DISTANCE {
                let prefetch = index + PREFETCH_DISTANCE;
                unsafe {
                    _mm_prefetch::<_MM_HINT_T0>(left.as_ptr().add(prefetch * ROWS).cast());
                    _mm_prefetch::<_MM_HINT_T0>(
                        right.as_ptr().add(prefetch * right_stride + column).cast(),
                    );
                }
            }
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

        macro_rules! store {
            ($row:expr, $lane:expr, $sum:expr) => {{
                let value = activation.map_or($sum, |activation| {
                    apply_vector_post_op($sum, Some(activation))
                });
                unsafe {
                    _mm256_storeu_ps(
                        output
                            .as_mut_ptr()
                            .add($row * output_stride + column + $lane),
                        value,
                    )
                };
            }};
        }
        store!(0, 0, sum00);
        store!(0, 8, sum01);
        store!(1, 0, sum10);
        store!(1, 8, sum11);
        store!(2, 0, sum20);
        store!(2, 8, sum21);
        store!(3, 0, sum30);
        store!(3, 8, sum31);
        store!(4, 0, sum40);
        store!(4, 8, sum41);
        store!(5, 0, sum50);
        store!(5, 8, sum51);
    }

    let mut column = vector_columns;
    if column + 8 <= columns {
        macro_rules! initial {
            ($row:expr) => {{
                if accumulate {
                    unsafe { _mm256_loadu_ps(output.as_ptr().add($row * output_stride + column)) }
                } else {
                    _mm256_set1_ps(bias.map_or(0.0, |bias| bias[$row]))
                }
            }};
        }
        let mut sum0 = initial!(0);
        let mut sum1 = initial!(1);
        let mut sum2 = initial!(2);
        let mut sum3 = initial!(3);
        let mut sum4 = initial!(4);
        let mut sum5 = initial!(5);
        macro_rules! k_step {
            ($index:expr) => {{
                let index = $index;
                let right =
                    unsafe { _mm256_loadu_ps(right.as_ptr().add(index * right_stride + column)) };
                let left = unsafe { left.as_ptr().add(index * ROWS) };
                let scale = _mm256_set1_ps(unsafe { *left });
                sum0 = _mm256_fmadd_ps(scale, right, sum0);
                let scale = _mm256_set1_ps(unsafe { *left.add(1) });
                sum1 = _mm256_fmadd_ps(scale, right, sum1);
                let scale = _mm256_set1_ps(unsafe { *left.add(2) });
                sum2 = _mm256_fmadd_ps(scale, right, sum2);
                let scale = _mm256_set1_ps(unsafe { *left.add(3) });
                sum3 = _mm256_fmadd_ps(scale, right, sum3);
                let scale = _mm256_set1_ps(unsafe { *left.add(4) });
                sum4 = _mm256_fmadd_ps(scale, right, sum4);
                let scale = _mm256_set1_ps(unsafe { *left.add(5) });
                sum5 = _mm256_fmadd_ps(scale, right, sum5);
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
        macro_rules! store {
            ($row:expr, $sum:expr) => {{
                let value = activation.map_or($sum, |activation| {
                    apply_vector_post_op($sum, Some(activation))
                });
                unsafe {
                    _mm256_storeu_ps(
                        output.as_mut_ptr().add($row * output_stride + column),
                        value,
                    )
                };
            }};
        }
        store!(0, sum0);
        store!(1, sum1);
        store!(2, sum2);
        store!(3, sum3);
        store!(4, sum4);
        store!(5, sum5);
        column += 8;
    }

    for row in 0..ROWS {
        for column in column..columns {
            let mut sum = if accumulate {
                output[row * output_stride + column]
            } else {
                bias.map_or(0.0, |bias| bias[row])
            };
            for index in 0..inner {
                sum = left[index * ROWS + row].mul_add(right[index * right_stride + column], sum);
            }
            output[row * output_stride + column] = apply_scalar_post_op(sum, activation);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_16x6_packed<
    const COLUMNS: usize,
    const HAS_BIAS: bool,
    const HAS_ACTIVATION: bool,
>(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    output_stride: usize,
    bias: Option<&[f32]>,
    activation: Option<super::UnaryOperation>,
) {
    const ROWS: usize = 16;
    const RIGHT_STRIDE: usize = 6;

    debug_assert!((1..=RIGHT_STRIDE).contains(&COLUMNS));
    debug_assert!(output.len() >= (ROWS - 1) * output_stride + COLUMNS);
    debug_assert_eq!(left.len(), ROWS * inner);
    debug_assert!(right.len() >= inner * RIGHT_STRIDE);
    debug_assert!(bias.is_none_or(|bias| bias.len() == ROWS));
    debug_assert_eq!(HAS_BIAS, bias.is_some());
    debug_assert_eq!(HAS_ACTIVATION, activation.is_some());

    let (initial0, initial1) = if HAS_BIAS {
        let bias = unsafe { bias.unwrap_unchecked() };
        unsafe {
            (
                _mm256_loadu_ps(bias.as_ptr()),
                _mm256_loadu_ps(bias.as_ptr().add(8)),
            )
        }
    } else {
        (_mm256_setzero_ps(), _mm256_setzero_ps())
    };
    let mut sums0 = [initial0; RIGHT_STRIDE];
    let mut sums1 = [initial1; RIGHT_STRIDE];

    macro_rules! k_step {
        ($index:expr) => {{
            let index = $index;
            let left = unsafe { left.as_ptr().add(index * ROWS) };
            let left0 = unsafe { _mm256_loadu_ps(left) };
            let left1 = unsafe { _mm256_loadu_ps(left.add(8)) };
            let right = unsafe { right.as_ptr().add(index * RIGHT_STRIDE) };
            macro_rules! column {
                ($column:literal) => {
                    if COLUMNS > $column {
                        let scale = _mm256_set1_ps(unsafe { *right.add($column) });
                        sums0[$column] = _mm256_fmadd_ps(left0, scale, sums0[$column]);
                        sums1[$column] = _mm256_fmadd_ps(left1, scale, sums1[$column]);
                    }
                };
            }
            column!(0);
            column!(1);
            column!(2);
            column!(3);
            column!(4);
            column!(5);
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

    let mut lower = [[0.0f32; 8]; RIGHT_STRIDE];
    let mut upper = [[0.0f32; 8]; RIGHT_STRIDE];
    for column in 0..COLUMNS {
        let (lower_value, upper_value) = if HAS_ACTIVATION {
            (
                apply_vector_post_op(sums0[column], activation),
                apply_vector_post_op(sums1[column], activation),
            )
        } else {
            (sums0[column], sums1[column])
        };
        unsafe {
            _mm256_storeu_ps(lower[column].as_mut_ptr(), lower_value);
            _mm256_storeu_ps(upper[column].as_mut_ptr(), upper_value);
        }
    }
    for row in 0..8 {
        for column in 0..COLUMNS {
            unsafe {
                *output.get_unchecked_mut(row * output_stride + column) = lower[column][row];
                *output.get_unchecked_mut((row + 8) * output_stride + column) = upper[column][row];
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
fn apply_vector_post_op(value: __m256, activation: Option<super::UnaryOperation>) -> __m256 {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    match activation {
        None => value,
        Some(super::UnaryOperation::Relu) => _mm256_max_ps(value, zero),
        Some(super::UnaryOperation::Erf) => erf_vector(value),
        Some(super::UnaryOperation::Gelu) => gelu_vector(value),
        Some(super::UnaryOperation::HardSwish) => {
            let gate = _mm256_fmadd_ps(value, _mm256_set1_ps(1.0 / 6.0), _mm256_set1_ps(0.5));
            _mm256_mul_ps(value, _mm256_min_ps(_mm256_max_ps(gate, zero), one))
        }
        Some(super::UnaryOperation::Sigmoid) => sigmoid_vector(value),
        Some(super::UnaryOperation::Silu) => silu_vector(value),
        Some(super::UnaryOperation::Sqrt) => _mm256_sqrt_ps(value),
        Some(super::UnaryOperation::HardSigmoid { alpha, beta }) => {
            let result = _mm256_fmadd_ps(value, _mm256_set1_ps(alpha), _mm256_set1_ps(beta));
            _mm256_min_ps(_mm256_max_ps(result, zero), one)
        }
    }
}

#[inline]
fn apply_scalar_post_op(value: f32, activation: Option<super::UnaryOperation>) -> f32 {
    activation.map_or(value, |activation| activation.apply(value))
}

#[target_feature(enable = "avx2,fma")]
fn erf_vector(input: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let sign_bit = _mm256_set1_ps(-0.0);
    let absolute = _mm256_andnot_ps(sign_bit, input);
    let denominator = _mm256_fmadd_ps(absolute, _mm256_set1_ps(0.327_591_1), one);
    let t = reciprocal(denominator);
    let mut polynomial =
        _mm256_fmadd_ps(t, _mm256_set1_ps(1.061_405_4), _mm256_set1_ps(-1.453_152_1));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(1.421_413_8));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(-0.284_496_72));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(0.254_829_6));
    polynomial = _mm256_mul_ps(polynomial, t);
    let exponential = exp256(_mm256_sub_ps(zero, _mm256_mul_ps(absolute, absolute)));
    let positive = _mm256_fnmadd_ps(polynomial, exponential, one);
    let negative_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(input, zero);
    _mm256_blendv_ps(positive, _mm256_sub_ps(zero, positive), negative_mask)
}

#[target_feature(enable = "avx2,fma")]
fn gelu_vector(input: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let scaled = _mm256_mul_ps(input, _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2));
    _mm256_mul_ps(
        _mm256_mul_ps(_mm256_set1_ps(0.5), input),
        _mm256_add_ps(one, erf_vector(scaled)),
    )
}

#[target_feature(enable = "avx2,fma")]
fn sigmoid_vector(input: __m256) -> __m256 {
    reciprocal(_mm256_add_ps(
        _mm256_set1_ps(1.0),
        exp256(_mm256_sub_ps(_mm256_setzero_ps(), input)),
    ))
}

#[target_feature(enable = "avx2,fma")]
fn silu_vector(input: __m256) -> __m256 {
    _mm256_mul_ps(input, sigmoid_vector(input))
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn gelu(values: &mut [f32]) {
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let input = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), gelu_vector(input)) };
    }
    for value in &mut values[vector_len..] {
        *value = scalar_gelu(*value);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn silu(values: &mut [f32]) {
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let input = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), silu_vector(input)) };
    }
    for value in &mut values[vector_len..] {
        *value = *value / (1.0 + (-*value).exp());
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn sigmoid(values: &mut [f32]) {
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let input = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), sigmoid_vector(input)) };
    }
    for value in &mut values[vector_len..] {
        *value = 1.0 / (1.0 + (-*value).exp());
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn softmax(values: &mut [f32]) {
    let vector_len = values.len() / 8 * 8;
    let mut maxima = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let maximum = &mut maxima[vector & 3];
        // Value is first so a NaN lane leaves the previous maximum intact.
        *maximum = _mm256_max_ps(value, *maximum);
    }
    let mut maximum_lanes = [f32::NEG_INFINITY; 32];
    for (vector, maximum) in maxima.into_iter().enumerate() {
        // SAFETY: Each store writes eight values into its own array segment.
        unsafe { _mm256_storeu_ps(maximum_lanes.as_mut_ptr().add(vector * 8), maximum) };
    }
    let mut maximum = maximum_lanes.into_iter().fold(f32::NEG_INFINITY, f32::max);
    for &value in &values[vector_len..] {
        maximum = maximum.max(value);
    }

    let maximum_vector = _mm256_set1_ps(maximum);
    let mut sums = [_mm256_setzero_ps(); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let exponential = exp256(_mm256_sub_ps(value, maximum_vector));
        let sum = &mut sums[vector & 3];
        *sum = _mm256_add_ps(*sum, exponential);
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), exponential) };
    }
    let mut sum_lanes = [0.0; 32];
    for (vector, sum) in sums.into_iter().enumerate() {
        // SAFETY: Each store writes eight values into its own array segment.
        unsafe { _mm256_storeu_ps(sum_lanes.as_mut_ptr().add(vector * 8), sum) };
    }
    let mut sum = sum_lanes.into_iter().sum::<f32>();
    for value in &mut values[vector_len..] {
        *value = (*value - maximum).exp();
        sum += *value;
    }

    let reciprocal = _mm256_set1_ps(sum.recip());
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // SAFETY: The store covers the same in-bounds vector.
        unsafe {
            _mm256_storeu_ps(
                values.as_mut_ptr().add(offset),
                _mm256_mul_ps(value, reciprocal),
            )
        };
    }
    let reciprocal = _mm256_cvtss_f32(reciprocal);
    for value in &mut values[vector_len..] {
        *value *= reciprocal;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn bias_softmax(values: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(values.len(), bias.len());
    let vector_len = values.len() / 8 * 8;
    let mut maxima = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        // SAFETY: Both slices contain a complete vector at this offset.
        let biased = unsafe {
            _mm256_add_ps(
                _mm256_loadu_ps(values.as_ptr().add(offset)),
                _mm256_loadu_ps(bias.as_ptr().add(offset)),
            )
        };
        let maximum = &mut maxima[vector & 3];
        *maximum = _mm256_max_ps(biased, *maximum);
    }
    let mut maximum_lanes = [f32::NEG_INFINITY; 32];
    for (vector, maximum) in maxima.into_iter().enumerate() {
        unsafe { _mm256_storeu_ps(maximum_lanes.as_mut_ptr().add(vector * 8), maximum) };
    }
    let mut maximum = maximum_lanes.into_iter().fold(f32::NEG_INFINITY, f32::max);
    for index in vector_len..values.len() {
        maximum = maximum.max(values[index] + bias[index]);
    }

    let maximum_vector = _mm256_set1_ps(maximum);
    let mut sums = [_mm256_setzero_ps(); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        // SAFETY: Both input loads and the output store are in bounds.
        let biased = unsafe {
            _mm256_add_ps(
                _mm256_loadu_ps(values.as_ptr().add(offset)),
                _mm256_loadu_ps(bias.as_ptr().add(offset)),
            )
        };
        let exponential = exp256(_mm256_sub_ps(biased, maximum_vector));
        let sum = &mut sums[vector & 3];
        *sum = _mm256_add_ps(*sum, exponential);
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), exponential) };
    }
    let mut sum_lanes = [0.0; 32];
    for (vector, sum) in sums.into_iter().enumerate() {
        unsafe { _mm256_storeu_ps(sum_lanes.as_mut_ptr().add(vector * 8), sum) };
    }
    let mut sum = sum_lanes.into_iter().sum::<f32>();
    for index in vector_len..values.len() {
        values[index] = (values[index] + bias[index] - maximum).exp();
        sum += values[index];
    }

    let reciprocal = _mm256_set1_ps(sum.recip());
    for offset in (0..vector_len).step_by(8) {
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        unsafe {
            _mm256_storeu_ps(
                values.as_mut_ptr().add(offset),
                _mm256_mul_ps(value, reciprocal),
            )
        };
    }
    let reciprocal = _mm256_cvtss_f32(reciprocal);
    for value in &mut values[vector_len..] {
        *value *= reciprocal;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn layer_norm(values: &mut [f32], weight: &[f32], bias: &[f32], epsilon: f32) {
    debug_assert!(!values.is_empty());
    debug_assert_eq!(values.len(), weight.len());
    debug_assert_eq!(values.len(), bias.len());
    debug_assert!(epsilon >= 0.0);

    let mean = unsafe { sum(values) } / values.len() as f32;
    let mean_vector = _mm256_set1_ps(mean);
    let vector_len = values.len() / 8 * 8;
    let mut squared_sums = [_mm256_setzero_ps(); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let centered = _mm256_sub_ps(value, mean_vector);
        let squared = _mm256_mul_ps(centered, centered);
        let sum = &mut squared_sums[vector & 3];
        *sum = _mm256_add_ps(*sum, squared);
    }
    let mut squared_lanes = [0.0; 32];
    for (vector, sum) in squared_sums.into_iter().enumerate() {
        unsafe { _mm256_storeu_ps(squared_lanes.as_mut_ptr().add(vector * 8), sum) };
    }
    let mut squared_sum = squared_lanes.into_iter().sum::<f32>();
    for &value in &values[vector_len..] {
        let centered = value - mean;
        squared_sum += centered * centered;
    }
    let variance = squared_sum / values.len() as f32;
    let inverse_std = (variance + epsilon).sqrt().recip();
    let inverse_std_vector = _mm256_set1_ps(inverse_std);

    for offset in (0..vector_len).step_by(8) {
        // SAFETY: Values, weight, and bias have equal lengths and a full vector remains.
        let (value, weight, bias_value) = unsafe {
            (
                _mm256_loadu_ps(values.as_ptr().add(offset)),
                _mm256_loadu_ps(weight.as_ptr().add(offset)),
                _mm256_loadu_ps(bias.as_ptr().add(offset)),
            )
        };
        let centered = _mm256_sub_ps(value, mean_vector);
        let scale = _mm256_mul_ps(weight, inverse_std_vector);
        let normalized = _mm256_fmadd_ps(centered, scale, bias_value);
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), normalized) };
    }
    for index in vector_len..values.len() {
        values[index] = (values[index] - mean).mul_add(inverse_std * weight[index], bias[index]);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn sum(values: &[f32]) -> f32 {
    let vector_len = values.len() / 8 * 8;
    let mut sums = [_mm256_setzero_ps(); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        // SAFETY: Each offset points at a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let sum = &mut sums[vector & 3];
        *sum = _mm256_add_ps(*sum, value);
    }
    let mut lanes = [0.0; 32];
    for (vector, sum) in sums.into_iter().enumerate() {
        // SAFETY: Each store writes eight values into its own array segment.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr().add(vector * 8), sum) };
    }
    lanes.into_iter().sum::<f32>() + values[vector_len..].iter().sum::<f32>()
}

#[target_feature(enable = "avx2,fma")]
fn reciprocal(value: __m256) -> __m256 {
    let two = _mm256_set1_ps(2.0);
    let mut estimate = _mm256_rcp_ps(value);
    estimate = _mm256_mul_ps(estimate, _mm256_fnmadd_ps(value, estimate, two));
    _mm256_mul_ps(estimate, _mm256_fnmadd_ps(value, estimate, two))
}

#[target_feature(enable = "avx2,fma")]
fn exp256(value: __m256) -> __m256 {
    let value = _mm256_max_ps(
        _mm256_set1_ps(-87.0),
        _mm256_min_ps(_mm256_set1_ps(87.0), value),
    );
    let exponent = _mm256_cvtps_epi32(_mm256_mul_ps(
        value,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));
    let remainder = _mm256_fnmadd_ps(
        _mm256_cvtepi32_ps(exponent),
        _mm256_set1_ps(std::f32::consts::LN_2),
        value,
    );
    let mut polynomial = _mm256_set1_ps(1.0 / 120.0);
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(1.0 / 24.0));
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(1.0 / 6.0));
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(0.5));
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(1.0));
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(1.0));
    let exponent_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)));
    _mm256_mul_ps(polynomial, _mm256_castsi256_ps(exponent_bits))
}

#[inline]
fn scalar_gelu(input: f32) -> f32 {
    let x = input * std::f32::consts::FRAC_1_SQRT_2;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / x.mul_add(0.327_591_1, 1.0);
    let polynomial = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    let erf = sign * (1.0 - polynomial * (-x * x).exp());
    0.5 * input * (1.0 + erf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simd_available() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), actual.len());
        let maximum_error = expected
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            maximum_error <= tolerance,
            "maximum error {maximum_error} exceeded {tolerance}"
        );
    }

    #[test]
    fn spatial_conv2d_packed_6_matches_scalar_with_padding_and_stride() {
        if !simd_available() {
            return;
        }

        let cases = [
            (3, 7, 37, [1, 1], [1, 1, 1, 1], true),
            (2, 8, 65, [2, 2], [1, 1, 1, 1], false),
            (2, 9, 66, [2, 2], [1, 1, 1, 1], true),
            (3, 6, 34, [1, 1], [1, 2, 0, 1], false),
        ];
        for (input_channels, input_height, input_width, strides, pads, relu) in cases {
            let kernel_height = 3;
            let kernel_width = 3;
            let output_height = (input_height + pads[0] + pads[2] - kernel_height) / strides[0] + 1;
            let output_width = (input_width + pads[1] + pads[3] - kernel_width) / strides[1] + 1;
            let input = (0..input_channels * input_height * input_width)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 13.0)
                .collect::<Vec<_>>();
            let weight = (0..input_channels * kernel_height * kernel_width * 6)
                .map(|index| ((index * 11 % 31) as f32 - 15.0) / 19.0)
                .collect::<Vec<_>>();
            let bias = [-0.375, 0.25, -0.125, 0.5, -0.75, 0.625];
            let activation = relu.then_some(super::super::UnaryOperation::Relu);
            let output_plane = output_height * output_width;
            let mut expected = vec![0.0; 6 * output_plane];
            for output_channel in 0..6 {
                for output_y in 0..output_height {
                    for output_x in 0..output_width {
                        let mut sum = bias[output_channel];
                        for input_channel in 0..input_channels {
                            for kernel_y in 0..kernel_height {
                                let padded_input_y = output_y * strides[0] + kernel_y;
                                if padded_input_y < pads[0]
                                    || padded_input_y - pads[0] >= input_height
                                {
                                    continue;
                                }
                                let input_y = padded_input_y - pads[0];
                                for kernel_x in 0..kernel_width {
                                    let padded_input_x = output_x * strides[1] + kernel_x;
                                    if padded_input_x < pads[1]
                                        || padded_input_x - pads[1] >= input_width
                                    {
                                        continue;
                                    }
                                    let input_x = padded_input_x - pads[1];
                                    let input_value =
                                        input[input_channel * input_height * input_width
                                            + input_y * input_width
                                            + input_x];
                                    let weight_index = ((input_channel * kernel_height + kernel_y)
                                        * kernel_width
                                        + kernel_x)
                                        * 6
                                        + output_channel;
                                    sum = input_value.mul_add(weight[weight_index], sum);
                                }
                            }
                        }
                        expected
                            [output_channel * output_plane + output_y * output_width + output_x] =
                            activation.map_or(sum, |activation| activation.apply(sum));
                    }
                }
            }

            let mut actual = vec![0.0; expected.len()];
            // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
            unsafe {
                spatial_conv2d_packed::<6>(
                    &mut actual,
                    &input,
                    &weight,
                    input_channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    kernel_height,
                    kernel_width,
                    strides,
                    pads,
                    Some(&bias),
                    activation,
                )
            };
            assert_close(&expected, &actual, 2e-5);
        }
    }

    #[test]
    fn stride2_copy_uses_exact_31_value_source_extent() {
        if !simd_available() {
            return;
        }
        let input = (0..31).map(|value| value as f32).collect::<Vec<_>>();
        let mut actual = [0.0; 16];
        // SAFETY: The source contains offsets 0 through 30 and the destination
        // contains all sixteen gathered values.
        unsafe { copy_stride2_16(actual.as_mut_ptr(), input.as_ptr()) };
        let expected = std::array::from_fn(|lane| (lane * 2) as f32);
        assert_eq!(actual, expected);
    }

    #[test]
    fn elementwise_kernels_match_scalar_with_tail() {
        if !simd_available() {
            return;
        }
        let input = (0..39)
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 7.0)
            .collect::<Vec<_>>();
        let base = (0..39)
            .map(|index| ((index * 17 % 37) as f32 - 18.0) / 11.0)
            .collect::<Vec<_>>();

        let mut expected = base.clone();
        expected
            .iter_mut()
            .zip(&input)
            .for_each(|(output, input)| *output = input.mul_add(-0.37, *output));
        let mut actual = base.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { axpy(&mut actual, &input, -0.37) };
        assert_close(&expected, &actual, 2e-7);

        let mut expected = base.clone();
        expected
            .iter_mut()
            .zip(&input)
            .for_each(|(output, input)| *output *= input);
        let mut actual = base.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { mul_in_place(&mut actual, &input) };
        assert_close(&expected, &actual, 0.0);

        let mut expected = base.clone();
        expected
            .iter_mut()
            .for_each(|value| *value = value.mul_add(1.27, -0.13));
        let mut actual = base.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { affine(&mut actual, 1.27, -0.13) };
        assert_close(&expected, &actual, 2e-7);

        let mut expected = base.clone();
        expected.iter_mut().for_each(|value| *value *= *value);
        let mut actual = base.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { square(&mut actual) };
        assert_close(&expected, &actual, 0.0);
    }

    #[test]
    fn activation_kernels_match_scalar_with_tail() {
        if !simd_available() {
            return;
        }
        let input = (0..43)
            .map(|index| ((index * 19 % 47) as f32 - 23.0) / 8.0)
            .collect::<Vec<_>>();

        let expected = input.iter().map(|value| value.max(0.0)).collect::<Vec<_>>();
        let mut actual = input.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { relu(&mut actual) };
        assert_close(&expected, &actual, 0.0);

        let expected = input
            .iter()
            .map(|&value| scalar_gelu(value))
            .collect::<Vec<_>>();
        let mut actual = input.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { gelu(&mut actual) };
        assert_close(&expected, &actual, 3e-6);

        let expected = input
            .iter()
            .map(|&value| value / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let mut actual = input.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { silu(&mut actual) };
        assert_close(&expected, &actual, 4e-6);

        let expected = input
            .iter()
            .map(|&value| 1.0 / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let mut actual = input;
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { sigmoid(&mut actual) };
        assert_close(&expected, &actual, 4e-6);
    }

    #[test]
    fn max_pool_rows_match_scalar_for_odd_even_widths_and_nans() {
        if !simd_available() {
            return;
        }
        for width in [17, 18] {
            let mut current = (0..width)
                .map(|index| (index as f32 - 9.0) * 0.25)
                .collect::<Vec<_>>();
            let mut next = (0..width)
                .map(|index| (7.0 - index as f32) * 0.375)
                .collect::<Vec<_>>();
            current[2] = f32::NAN;
            current[7] = -0.0;
            current[8] = 0.0;
            next[10] = f32::NAN;
            next[width - 1] = f32::NAN;

            for next_row in [None, Some(next.as_slice())] {
                let mut expected = vec![0.0; width];
                for x in 0..width {
                    let mut maximum = current[x];
                    if x + 1 < width {
                        maximum = maximum.max(current[x + 1]);
                    }
                    if let Some(next) = next_row {
                        maximum = maximum.max(next[x]);
                        if x + 1 < width {
                            maximum = maximum.max(next[x + 1]);
                        }
                    }
                    expected[x] = maximum;
                }
                let mut actual = vec![0.0; width];
                // SAFETY: The runtime feature check above covers this AVX2 kernel.
                unsafe { max_pool_2x2_row(&mut actual, &current, next_row) };
                for (expected, actual) in expected.iter().zip(&actual) {
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(expected.to_bits(), actual.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn sigmoid_preserves_nan_and_extreme_values_with_tail() {
        if !simd_available() {
            return;
        }
        let input = [
            f32::NEG_INFINITY,
            -100.0,
            -10.0,
            -0.0,
            0.0,
            10.0,
            100.0,
            f32::INFINITY,
            f32::NAN,
            -1.25,
            2.5,
        ];
        let expected = input.map(|value| 1.0 / (1.0 + (-value).exp()));
        let mut actual = input;
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { sigmoid(&mut actual) };
        for (expected, actual) in expected.iter().zip(actual) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!((expected - actual).abs() <= 4e-6);
            }
        }
    }

    #[test]
    fn softmax_matches_scalar_with_tail() {
        if !simd_available() {
            return;
        }
        let mut actual = (0..37)
            .map(|index| ((index * 23 % 41) as f32 - 20.0) / 6.0)
            .collect::<Vec<_>>();
        let mut expected = actual.clone();
        let maximum = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = expected
            .iter_mut()
            .map(|value| {
                *value = (*value - maximum).exp();
                *value
            })
            .sum::<f32>();
        expected.iter_mut().for_each(|value| *value /= sum);

        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { softmax(&mut actual) };
        assert_close(&expected, &actual, 3e-6);
        assert!((actual.iter().sum::<f32>() - 1.0).abs() < 2e-6);
    }

    #[test]
    fn bias_softmax_matches_scalar_with_tail() {
        if !simd_available() {
            return;
        }
        let mut actual = (0..37)
            .map(|index| ((index * 23 % 41) as f32 - 20.0) / 6.0)
            .collect::<Vec<_>>();
        let bias = (0..37)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 9.0)
            .collect::<Vec<_>>();
        let mut expected = actual
            .iter()
            .zip(&bias)
            .map(|(value, bias)| value + bias)
            .collect::<Vec<_>>();
        let maximum = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = expected
            .iter_mut()
            .map(|value| {
                *value = (*value - maximum).exp();
                *value
            })
            .sum::<f32>();
        expected.iter_mut().for_each(|value| *value /= sum);

        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { bias_softmax(&mut actual, &bias) };
        assert_close(&expected, &actual, 3e-6);
        assert!((actual.iter().sum::<f32>() - 1.0).abs() < 2e-6);
    }

    #[test]
    fn layer_norm_matches_centered_scalar_with_tail() {
        if !simd_available() {
            return;
        }
        let mut actual = (0..37)
            .map(|index| ((index * 17 % 43) as f32 - 21.0) / 7.0)
            .collect::<Vec<_>>();
        let weight = (0..37)
            .map(|index| 0.5 + (index * 7 % 19) as f32 / 23.0)
            .collect::<Vec<_>>();
        let bias = (0..37)
            .map(|index| ((index * 13 % 29) as f32 - 14.0) / 17.0)
            .collect::<Vec<_>>();
        let mut expected = actual.clone();
        let mean = expected.iter().sum::<f32>() / expected.len() as f32;
        let variance = expected
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / expected.len() as f32;
        let inverse_std = (variance + 1e-5).sqrt().recip();
        for ((value, weight), bias) in expected.iter_mut().zip(&weight).zip(&bias) {
            *value = (*value - mean).mul_add(inverse_std * *weight, *bias);
        }

        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { layer_norm(&mut actual, &weight, &bias, 1e-5) };
        assert_close(&expected, &actual, 3e-5);
    }

    #[test]
    fn gemm_kernels_match_scalar_with_column_tail() {
        if !simd_available() {
            return;
        }
        const ROWS: usize = 8;
        let inner = 7;
        let columns = 19;
        let left = (0..ROWS * inner)
            .map(|index| ((index * 17 % 29) as f32 - 14.0) / 11.0)
            .collect::<Vec<_>>();
        let right = (0..inner * columns)
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 9.0)
            .collect::<Vec<_>>();
        let column_bias = (0..columns)
            .map(|column| (column as f32 - 8.0) / 7.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; ROWS * columns];
        for row in 0..ROWS {
            for column in 0..columns {
                let mut sum = column_bias[column];
                for index in 0..inner {
                    sum = left[row * inner + index].mul_add(right[index * columns + column], sum);
                }
                expected[row * columns + column] = sum;
            }
        }
        let mut actual = vec![0.0; expected.len()];
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe {
            gemm_rows_8::<ROWS, false>(
                &mut actual,
                &left,
                &right,
                inner,
                columns,
                columns,
                None,
                Some(&column_bias),
                None,
            )
        };
        assert_close(&expected, &actual, 2e-6);
    }

    #[test]
    fn packed_gemm_kernel_matches_scalar_with_column_tail() {
        if !simd_available() {
            return;
        }
        const ROWS: usize = 12;
        let inner = 5;
        let columns = 17;
        let row_major = (0..ROWS * inner)
            .map(|index| ((index * 11 % 37) as f32 - 18.0) / 12.0)
            .collect::<Vec<_>>();
        let mut packed = Vec::with_capacity(row_major.len());
        for index in 0..inner {
            for row in 0..ROWS {
                packed.push(row_major[row * inner + index]);
            }
        }
        let right = (0..inner * columns)
            .map(|index| ((index * 7 % 23) as f32 - 11.0) / 8.0)
            .collect::<Vec<_>>();
        let bias = (0..ROWS)
            .map(|row| (row as f32 - 5.0) / 9.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; ROWS * columns];
        for row in 0..ROWS {
            for column in 0..columns {
                let mut sum: f32 = bias[row];
                for index in 0..inner {
                    sum = row_major[row * inner + index]
                        .mul_add(right[index * columns + column], sum);
                }
                expected[row * columns + column] = sum;
            }
        }
        let mut actual = vec![0.0; expected.len()];
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe {
            gemm_rows_8::<ROWS, true>(
                &mut actual,
                &packed,
                &right,
                inner,
                columns,
                columns,
                Some(&bias),
                None,
                None,
            )
        };
        assert_close(&expected, &actual, 2e-6);
    }

    fn check_direct_linear_6x16<const ROWS: usize>(columns: usize) {
        const CANARY: f32 = 12_345.0;
        let inner = 13;
        let output_stride = columns + 3;
        let row_major_input = (0..ROWS * inner)
            .map(|index| ((index * 17 % 31) as f32 - 15.0) / 19.0)
            .collect::<Vec<_>>();
        let mut packed_input = Vec::with_capacity(row_major_input.len());
        for index in 0..inner {
            for row in 0..ROWS {
                packed_input.push(row_major_input[row * inner + index]);
            }
        }
        let row_major_weight = (0..columns * inner)
            .map(|index| ((index * 13 % 37) as f32 - 18.0) / 23.0)
            .collect::<Vec<_>>();
        let mut packed_weight = Vec::with_capacity(row_major_weight.len());
        for index in 0..inner {
            for column in 0..columns {
                packed_weight.push(row_major_weight[column * inner + index]);
            }
        }
        let bias = (0..columns)
            .map(|column| (column as f32 - 9.0) / 29.0)
            .collect::<Vec<_>>();

        for (bias, activation, tolerance) in [
            (None, None, 2e-6),
            (
                Some(bias.as_slice()),
                Some(super::super::UnaryOperation::Silu),
                3e-5,
            ),
        ] {
            let mut expected = vec![0.0; ROWS * columns];
            for row in 0..ROWS {
                for column in 0..columns {
                    let mut sum = bias.map_or(0.0, |bias| bias[column]);
                    for index in 0..inner {
                        sum = row_major_input[row * inner + index]
                            .mul_add(row_major_weight[column * inner + index], sum);
                    }
                    expected[row * columns + column] =
                        activation.map_or(sum, |activation| activation.apply(sum));
                }
            }

            let mut output = vec![CANARY; ROWS * output_stride + 8];
            // SAFETY: The caller checked AVX2/FMA. Buffers contain the exact
            // packed dimensions passed to the kernel.
            unsafe {
                linear_6x16_packed::<ROWS>(
                    &mut output,
                    output_stride,
                    &packed_input,
                    &packed_weight,
                    inner,
                    columns,
                    bias,
                    activation,
                )
            };
            let actual = (0..ROWS)
                .flat_map(|row| output[row * output_stride..row * output_stride + columns].iter())
                .copied()
                .collect::<Vec<_>>();
            assert_close(&expected, &actual, tolerance);
            for row in 0..ROWS {
                assert!(
                    output[row * output_stride + columns..(row + 1) * output_stride]
                        .iter()
                        .all(|&value| value == CANARY),
                    "row {row}, columns {columns}: output tail was overwritten"
                );
            }
            assert!(
                output[ROWS * output_stride..]
                    .iter()
                    .all(|&value| value == CANARY)
            );
        }
    }

    #[test]
    fn direct_linear_6x16_matches_all_row_and_column_tails() {
        if !simd_available() {
            return;
        }
        macro_rules! check_rows {
            ($rows:literal) => {
                for columns in [1, 6, 7, 8, 9, 10, 15, 16] {
                    check_direct_linear_6x16::<$rows>(columns);
                }
            };
        }
        check_rows!(1);
        check_rows!(2);
        check_rows!(3);
        check_rows!(4);
        check_rows!(5);
        check_rows!(6);
    }

    #[test]
    fn sparse_gemm_kernel_handles_16_8_and_scalar_column_blocks() {
        if !simd_available() {
            return;
        }
        const ROWS: usize = 4;
        let inner = 7;
        let columns = 25;
        let indices = [0, 2, 6];
        let weights: [f32; 12] = [
            1.0, -0.5, 0.25, 2.0, 0.75, 1.5, -1.0, 0.5, -2.0, 0.125, 0.375, 1.25,
        ];
        let right = (0..inner * columns)
            .map(|index| ((index * 13 % 37) as f32 - 18.0) / 11.0)
            .collect::<Vec<_>>();
        let bias = [0.5, -0.25, 1.0, -1.0];
        let mut expected = vec![0.0; ROWS * columns];
        for row in 0..ROWS {
            for column in 0..columns {
                let mut sum: f32 = bias[row];
                for (entry, &index) in indices.iter().enumerate() {
                    sum = weights[entry * ROWS + row]
                        .mul_add(right[index as usize * columns + column], sum);
                }
                expected[row * columns + column] = sum.max(0.0);
            }
        }
        let mut actual = vec![0.0; expected.len()];
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe {
            gemm_4x16_sparse(
                &mut actual,
                &right,
                &indices,
                &weights,
                columns,
                columns,
                Some(&bias),
                Some(super::super::UnaryOperation::Relu),
            )
        };
        assert_close(&expected, &actual, 2e-6);
    }

    #[test]
    fn packed_4x16_gemm_accumulates_depth_blocks_before_activation() {
        if !simd_available() {
            return;
        }
        const ROWS: usize = 4;
        let inner = 37;
        let split = 19;
        let columns = 25;
        let left = (0..inner * ROWS)
            .map(|index| ((index * 17 % 41) as f32 - 20.0) / 13.0)
            .collect::<Vec<_>>();
        let right = (0..inner * columns)
            .map(|index| ((index * 11 % 43) as f32 - 21.0) / 15.0)
            .collect::<Vec<_>>();
        let bias = [0.5, -0.25, 1.0, -1.0];
        let mut expected = vec![0.0; ROWS * columns];
        for row in 0..ROWS {
            for column in 0..columns {
                let mut sum = bias[row];
                for index in 0..inner {
                    sum = left[index * ROWS + row].mul_add(right[index * columns + column], sum);
                }
                expected[row * columns + column] = sum / (1.0 + (-sum).exp());
            }
        }
        let mut actual = vec![0.0; expected.len()];
        // SAFETY: The runtime feature check above covers both AVX2+FMA calls.
        unsafe {
            gemm_4x16_packed(
                &mut actual,
                &left[..split * ROWS],
                &right,
                split,
                columns,
                columns,
                columns,
                Some(&bias),
                false,
                None,
            );
            gemm_4x16_packed(
                &mut actual,
                &left[split * ROWS..],
                &right[split * columns..],
                inner - split,
                columns,
                columns,
                columns,
                Some(&bias),
                true,
                Some(super::super::UnaryOperation::Silu),
            );
        }
        assert_close(&expected, &actual, 1e-5);
    }
}
