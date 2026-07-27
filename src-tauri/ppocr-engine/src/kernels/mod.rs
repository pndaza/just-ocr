//! Architecture-specific CPU kernels.

#[cfg(target_arch = "x86_64")]
use super::arena::Buffer;
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
use super::arena::Handle as ArenaHandle;
use rayon::prelude::*;

#[cfg(target_os = "macos")]
mod accelerate;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[inline]
pub(crate) fn supports_pointwise_pair_fusion() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2_fma()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub(crate) fn supports_exact_sparse_gemm() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2_fma()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
pub(crate) fn supports_stride2_simd_copy() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2_fma()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub(crate) fn supports_direct_spatial_conv() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2_fma()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
pub(crate) unsafe fn copy_stride2_16(output: *mut f32, input: *const f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        x86::copy_stride2_16(output, input);
    }
    #[cfg(not(target_arch = "x86_64"))]
    for lane in 0..16 {
        unsafe { *output.add(lane) = *input.add(lane * 2) };
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn spatial_conv2d_direct(
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
    strides: [usize; 2],
    pads: [usize; 4],
    activation: Option<UnaryOperation>,
) {
    const BLOCK_ROWS: usize = 6;
    assert!(supports_direct_spatial_conv());
    assert!(output_channels.is_multiple_of(4));
    assert_eq!(input.len(), input_channels * input_height * input_width);
    assert_eq!(output.len(), output_channels * output_height * output_width);
    let patch_size = input_channels * kernel_height * kernel_width;
    assert_eq!(weight.len(), output_channels * patch_size);
    assert!(bias.is_none_or(|bias| bias.len() == output_channels));
    let output_plane = output_height * output_width;
    #[cfg(all(feature = "cpu-profile", target_arch = "x86_64"))]
    eprintln!(
        "cpu-profile spatial-conv input_channels={input_channels} output_channels={output_channels} input={input_height}x{input_width} output={output_height}x{output_width} kernel={kernel_height}x{kernel_width} stride={} micro_panels={}",
        strides[0],
        rayon::current_num_threads() == 1
            && output_channels >= 16
            && spatial_panel_working_set_fits(weight.len(), patch_size, output_channels)
    );
    #[cfg(target_arch = "x86_64")]
    if rayon::current_num_threads() == 1
        && output_channels >= 16
        && spatial_panel_working_set_fits(weight.len(), patch_size, output_channels)
    {
        // SAFETY: The runtime AVX2/FMA check is covered by
        // supports_direct_spatial_conv(), and all tensor dimensions were
        // validated above.
        unsafe {
            spatial_conv2d_micro_panels(
                output,
                input,
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
                strides,
                pads,
                activation,
            )
        };
        return;
    }

    output
        .par_chunks_mut(BLOCK_ROWS * output_plane)
        .enumerate()
        .for_each(|(block, output)| {
            let row_start = block * BLOCK_ROWS;
            let block_rows = output.len() / output_plane;
            let weight_start = row_start * patch_size;
            let weight = &weight[weight_start..weight_start + block_rows * patch_size];
            let bias = bias.map(|bias| &bias[row_start..row_start + block_rows]);
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Runtime AVX2/FMA support and all matrix/image dimensions
            // are validated above. The kernel handles borders without OOB loads.
            unsafe {
                macro_rules! run {
                    ($rows:literal) => {
                        x86::spatial_conv2d_packed::<$rows>(
                            output,
                            input,
                            weight,
                            input_channels,
                            input_height,
                            input_width,
                            output_height,
                            output_width,
                            kernel_height,
                            kernel_width,
                            strides,
                            pads,
                            bias,
                            activation,
                        )
                    };
                }
                match block_rows {
                    2 => run!(2),
                    4 => run!(4),
                    6 => run!(6),
                    _ => unreachable!("direct spatial output-channel tail"),
                }
            };
        });
}

#[cfg(target_arch = "x86_64")]
fn spatial_panel_working_set_fits(
    weight_elements: usize,
    patch_size: usize,
    output_channels: usize,
) -> bool {
    const PANEL_COLUMNS: usize = 16;
    const MAX_WORKING_SET_BYTES: usize = 1024 * 1024;

    patch_size
        .checked_mul(PANEL_COLUMNS)
        .and_then(|scratch| scratch.checked_add(weight_elements))
        .and_then(|elements| {
            output_channels
                .checked_mul(PANEL_COLUMNS)
                .and_then(|output| elements.checked_add(output))
        })
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .is_some_and(|bytes| bytes <= MAX_WORKING_SET_BYTES)
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn spatial_conv2d_micro_panels(
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
    strides: [usize; 2],
    pads: [usize; 4],
    activation: Option<UnaryOperation>,
) {
    const PANEL_COLUMNS: usize = 16;
    const BLOCK_ROWS: usize = 6;

    let patch_size = input_channels * kernel_height * kernel_width;
    let output_plane = output_height * output_width;
    let mut panel = Buffer::for_overwrite(patch_size * PANEL_COLUMNS);
    let tail_rows = output_channels % BLOCK_ROWS;
    let panel_rows = if tail_rows == 2 {
        output_channels - tail_rows
    } else {
        output_channels
    };

    for output_y in 0..output_height {
        for output_x in (0..output_width).step_by(PANEL_COLUMNS) {
            let columns = (output_width - output_x).min(PANEL_COLUMNS);
            pack_spatial_panel(
                &mut panel,
                input,
                input_channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                output_y,
                output_x,
                columns,
                strides,
                pads,
            );

            let mut output_channel = 0usize;
            while output_channel < panel_rows {
                let block_rows = (panel_rows - output_channel).min(BLOCK_ROWS);
                let weight_start = output_channel * patch_size;
                let output_start =
                    output_channel * output_plane + output_y * output_width + output_x;
                let block_bias =
                    bias.map(|bias| &bias[output_channel..output_channel + block_rows]);
                match block_rows {
                    6 => {
                        // SAFETY: The packed weight block is [K][6], the RHS
                        // panel is [K][16], and every output row has
                        // output_plane elements.
                        unsafe {
                            x86::gemm_6x16_packed::<false>(
                                &mut output[output_start..],
                                &weight[weight_start..weight_start + patch_size * 6],
                                &panel,
                                patch_size,
                                columns,
                                output_plane,
                                PANEL_COLUMNS,
                                block_bias,
                                false,
                                activation,
                            )
                        };
                    }
                    4 => {
                        // SAFETY: Same layout and bounds argument as the
                        // six-row block, with the final packed [K][4] tail.
                        unsafe {
                            x86::gemm_4x16_packed(
                                &mut output[output_start..],
                                &weight[weight_start..weight_start + patch_size * 4],
                                &panel,
                                patch_size,
                                columns,
                                output_plane,
                                PANEL_COLUMNS,
                                block_bias,
                                false,
                                activation,
                            )
                        };
                    }
                    _ => unreachable!("micro-panel output-channel block"),
                }
                output_channel += block_rows;
            }
        }
    }

    if panel_rows < output_channels {
        let output_channel = panel_rows;
        let weight_start = output_channel * patch_size;
        // SAFETY: A two-row tail remains in the original direct-spatial layout
        // and is evaluated by the existing kernel.
        unsafe {
            x86::spatial_conv2d_packed::<2>(
                &mut output[output_channel * output_plane..],
                input,
                &weight[weight_start..],
                input_channels,
                input_height,
                input_width,
                output_height,
                output_width,
                kernel_height,
                kernel_width,
                strides,
                pads,
                bias.map(|bias| &bias[output_channel..]),
                activation,
            )
        };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn spatial_conv2d_direct(
    _output: &mut [f32],
    _input: &[f32],
    _weight: &[f32],
    _bias: Option<&[f32]>,
    _input_channels: usize,
    _input_height: usize,
    _input_width: usize,
    _output_channels: usize,
    _output_height: usize,
    _output_width: usize,
    _kernel_height: usize,
    _kernel_width: usize,
    _strides: [usize; 2],
    _pads: [usize; 4],
    _activation: Option<UnaryOperation>,
) {
    unreachable!("direct spatial convolution is only available on x86-64");
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn pack_spatial_panel(
    panel: &mut [f32],
    input: &[f32],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    output_y: usize,
    output_x: usize,
    columns: usize,
    strides: [usize; 2],
    pads: [usize; 4],
) {
    const PANEL_COLUMNS: usize = 16;

    let input_plane = input_height * input_width;
    if columns == PANEL_COLUMNS {
        let padded_y = output_y * strides[0];
        let padded_x = output_x * strides[1];
        if padded_y >= pads[0] && padded_x >= pads[1] {
            let input_y = padded_y - pads[0];
            let input_x = padded_x - pads[1];
            let input_span = (PANEL_COLUMNS - 1) * strides[1] + kernel_width;
            if input_y + kernel_height <= input_height && input_x + input_span <= input_width {
                for input_channel in 0..input_channels {
                    let channel =
                        &input[input_channel * input_plane..(input_channel + 1) * input_plane];
                    for kernel_y in 0..kernel_height {
                        let source_row = (input_y + kernel_y) * input_width + input_x;
                        for kernel_x in 0..kernel_width {
                            let patch_index = (input_channel * kernel_height + kernel_y)
                                * kernel_width
                                + kernel_x;
                            let destination = &mut panel
                                [patch_index * PANEL_COLUMNS..(patch_index + 1) * PANEL_COLUMNS];
                            let source = source_row + kernel_x;
                            if strides[1] == 1 {
                                destination
                                    .copy_from_slice(&channel[source..source + PANEL_COLUMNS]);
                            } else {
                                // SAFETY: The full-window check above includes
                                // every stride-two source offset through lane 15.
                                unsafe {
                                    copy_stride2_16(
                                        destination.as_mut_ptr(),
                                        channel.as_ptr().add(source),
                                    )
                                };
                            }
                        }
                    }
                }
                return;
            }
        }
    }

    for input_channel in 0..input_channels {
        let channel = &input[input_channel * input_plane..(input_channel + 1) * input_plane];
        for kernel_y in 0..kernel_height {
            let padded_input_y = output_y * strides[0] + kernel_y;
            let valid_y = padded_input_y >= pads[0] && padded_input_y - pads[0] < input_height;
            let input_y = padded_input_y.saturating_sub(pads[0]);
            for kernel_x in 0..kernel_width {
                let patch_index =
                    (input_channel * kernel_height + kernel_y) * kernel_width + kernel_x;
                let destination =
                    &mut panel[patch_index * PANEL_COLUMNS..(patch_index + 1) * PANEL_COLUMNS];
                if !valid_y {
                    destination.fill(0.0);
                    continue;
                }

                let padded_input_x = output_x * strides[1] + kernel_x;
                if columns == PANEL_COLUMNS
                    && strides[1] == 1
                    && padded_input_x >= pads[1]
                    && padded_input_x - pads[1] + PANEL_COLUMNS <= input_width
                {
                    let input_x = padded_input_x - pads[1];
                    let source = input_y * input_width + input_x;
                    destination.copy_from_slice(&channel[source..source + PANEL_COLUMNS]);
                    continue;
                }
                if columns == PANEL_COLUMNS
                    && strides[1] == 2
                    && padded_input_x >= pads[1]
                    && padded_input_x - pads[1] + 31 <= input_width
                {
                    let input_x = padded_input_x - pads[1];
                    let source = input_y * input_width + input_x;
                    // SAFETY: The source contains offsets 0 through 30 and the
                    // destination is one complete sixteen-value panel row.
                    unsafe {
                        copy_stride2_16(destination.as_mut_ptr(), channel.as_ptr().add(source))
                    };
                    continue;
                }

                destination.fill(0.0);
                for (lane, destination) in destination[..columns].iter_mut().enumerate() {
                    let input_x = (output_x + lane) * strides[1] + kernel_x;
                    if input_x >= pads[1] && input_x - pads[1] < input_width {
                        *destination = channel[input_y * input_width + input_x - pads[1]];
                    }
                }
            }
        }
    }
}

#[inline]
pub(crate) fn fill(values: &mut [f32], value: f32) {
    values.fill(value);
}

#[inline]
pub(crate) fn axpy(output: &mut [f32], input: &[f32], scale: f32) {
    assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The implementation only performs unaligned loads/stores within
        // the bounds of equally sized slices. NEON is mandatory on AArch64.
        unsafe { neon::axpy(output, input, scale) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime, and equal-length
        // slices bound every unaligned vector load and store.
        unsafe { x86::axpy(output, input, scale) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (output, input) in output.iter_mut().zip(input) {
        *output = input.mul_add(scale, *output);
    }
}

#[inline]
pub(crate) fn add_in_place(output: &mut [f32], input: &[f32]) {
    axpy(output, input, 1.0);
}

pub(crate) fn depthwise_conv2d_same(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    kernel: usize,
    bias: f32,
) {
    assert!(height > 0 && width > 0);
    assert!(matches!(kernel, 3 | 5 | 7 | 9));
    assert_eq!(output.len(), height * width);
    assert_eq!(input.len(), height * width);
    assert_eq!(weights.len(), kernel * kernel);

    macro_rules! dispatch {
        ($kernel:literal) => {{
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: Slice dimensions are checked above. The NEON kernel
                // vectorizes only columns whose complete KxK window is in bounds.
                unsafe {
                    neon::depthwise_conv2d_same::<$kernel>(
                        output, input, weights, height, width, bias,
                    )
                };
            }
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma() {
                // SAFETY: AVX2 and FMA were detected at runtime. Slice dimensions
                // are checked above, and the kernel vectorizes only complete
                // interior windows.
                unsafe {
                    x86::depthwise_conv2d_same::<$kernel>(
                        output, input, weights, height, width, bias,
                    )
                };
                return;
            }
            #[cfg(not(target_arch = "aarch64"))]
            depthwise_conv2d_same_scalar::<$kernel>(output, input, weights, height, width, bias);
        }};
    }

    match kernel {
        3 => dispatch!(3),
        5 => dispatch!(5),
        7 => dispatch!(7),
        9 => dispatch!(9),
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_conv2d_same_strip(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    channels: usize,
    height: usize,
    width: usize,
    kernel: usize,
    y_start: usize,
    rows: usize,
) {
    assert!(channels > 0 && height > 0 && width > 0 && rows > 0);
    assert!(matches!(kernel, 3 | 5 | 7 | 9));
    assert!(y_start <= height && rows <= height - y_start);
    assert_eq!(output.len(), channels * rows * width);
    assert_eq!(input.len(), channels * height * width);
    assert_eq!(weights.len(), channels * kernel * kernel);
    assert!(bias.is_none_or(|bias| bias.len() == channels));

    macro_rules! dispatch {
        ($kernel:literal) => {{
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma() {
                for channel in 0..channels {
                    let output_start = channel * rows * width;
                    let input_start = channel * height * width;
                    let weight_start = channel * $kernel * $kernel;
                    // SAFETY: Runtime AVX2/FMA support and every per-channel
                    // strip, input plane, and weight extent are checked above.
                    unsafe {
                        x86::depthwise_conv2d_same_rows::<$kernel>(
                            &mut output[output_start..output_start + rows * width],
                            &input[input_start..input_start + height * width],
                            &weights[weight_start..weight_start + $kernel * $kernel],
                            height,
                            width,
                            y_start,
                            rows,
                            bias.map_or(0.0, |bias| bias[channel]),
                        )
                    };
                }
                return;
            }

            for channel in 0..channels {
                let output_start = channel * rows * width;
                let input_start = channel * height * width;
                let weight_start = channel * $kernel * $kernel;
                depthwise_conv2d_same_rows_scalar::<$kernel>(
                    &mut output[output_start..output_start + rows * width],
                    &input[input_start..input_start + height * width],
                    &weights[weight_start..weight_start + $kernel * $kernel],
                    height,
                    width,
                    y_start,
                    rows,
                    bias.map_or(0.0, |bias| bias[channel]),
                );
            }
        }};
    }

    match kernel {
        3 => dispatch!(3),
        5 => dispatch!(5),
        7 => dispatch!(7),
        9 => dispatch!(9),
        _ => unreachable!(),
    }
}

pub(crate) fn depthwise_conv2d_same_3x3_stride2(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    bias: f32,
) {
    assert!(height > 0 && width > 0);
    assert_eq!(output.len(), height.div_ceil(2) * width.div_ceil(2));
    assert_eq!(input.len(), height * width);
    assert_eq!(weights.len(), 9);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: Slice dimensions are checked above. The NEON kernel only
    // vectorizes complete interior windows and handles borders separately.
    unsafe {
        neon::depthwise_conv2d_same_3x3_stride2(output, input, weights, height, width, bias)
    };
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime. Slice dimensions are
        // checked above, and complete interior vectors are bounded by the input.
        unsafe {
            x86::depthwise_conv2d_same_3x3_stride2(output, input, weights, height, width, bias)
        };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let output_width = width.div_ceil(2);
        for output_y in 0..height.div_ceil(2) {
            for output_x in 0..output_width {
                let mut sum = bias;
                for kernel_y in 0..3 {
                    let input_y = output_y * 2 + kernel_y;
                    if input_y == 0 || input_y > height {
                        continue;
                    }
                    for kernel_x in 0..3 {
                        let input_x = output_x * 2 + kernel_x;
                        if input_x == 0 || input_x > width {
                            continue;
                        }
                        sum = input[(input_y - 1) * width + input_x - 1]
                            .mul_add(weights[kernel_y * 3 + kernel_x], sum);
                    }
                }
                output[output_y * output_width + output_x] = sum;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn depthwise_conv2d_same_scalar<const K: usize>(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    bias: f32,
) {
    depthwise_conv2d_same_rows_scalar::<K>(output, input, weights, height, width, 0, height, bias);
}

#[allow(clippy::too_many_arguments)]
fn depthwise_conv2d_same_rows_scalar<const K: usize>(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    y_start: usize,
    rows: usize,
    bias: f32,
) {
    let padding = K / 2;
    for local_y in 0..rows {
        let y = y_start + local_y;
        let kernel_y_start = padding.saturating_sub(y);
        let kernel_y_end = K.min(height + padding - y);
        for x in 0..width {
            let kernel_x_start = padding.saturating_sub(x);
            let kernel_x_end = K.min(width + padding - x);
            let mut sum = bias;
            for kernel_y in kernel_y_start..kernel_y_end {
                let input_y = y + kernel_y - padding;
                for kernel_x in kernel_x_start..kernel_x_end {
                    let input_x = x + kernel_x - padding;
                    sum = input[input_y * width + input_x]
                        .mul_add(weights[kernel_y * K + kernel_x], sum);
                }
            }
            output[local_y * width + x] = sum;
        }
    }
}

pub(crate) fn mul_in_place(output: &mut [f32], input: &[f32]) {
    assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Equal-length slices bound every vector load and store.
        unsafe { neon::mul_in_place(output, input) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime, and both slices have
        // the same length.
        unsafe { x86::mul_in_place(output, input) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (output, input) in output.iter_mut().zip(input) {
        *output *= *input;
    }
}

pub(crate) fn affine_in_place(values: &mut [f32], scale: f32, bias: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The in-place kernel only accesses the supplied slice.
        unsafe { neon::affine(values, scale, bias) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the kernel stays
        // within the supplied slice.
        unsafe { x86::affine(values, scale, bias) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for value in values {
        *value = value.mul_add(scale, bias);
    }
}

pub(crate) fn residual_mul_in_place(values: &mut [f32], gate: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The in-place kernel only accesses the supplied slice.
        unsafe { neon::residual_mul(values, gate) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the kernel stays
        // within the supplied slice.
        unsafe { x86::residual_mul(values, gate) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for value in values {
        let original = *value;
        let scaled = original.mul_add(gate, 0.0);
        *value = scaled.mul_add(1.0, original);
    }
}

pub(crate) fn square_in_place(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The in-place kernel only accesses the supplied slice.
        unsafe { neon::square(values) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the kernel stays
        // within the supplied slice.
        unsafe { x86::square(values) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for value in values {
        *value *= *value;
    }
}

pub(crate) fn gemm(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
) {
    gemm_with_activation(output, left, right, rows, inner, columns, bias, None);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_with_activation(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    gemm_impl(
        output, left, right, rows, inner, columns, bias, None, false, activation, false,
    );
}

#[cfg(test)]
pub(crate) fn gemm_packed_left(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
) {
    gemm_packed_left_with_activation(output, left, right, rows, inner, columns, bias, None);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_packed_left_with_activation(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    gemm_impl(
        output, left, right, rows, inner, columns, bias, None, true, activation, false,
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_packed_left_cached_blocked_16(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    macro_rules! dispatch {
        ($column_block:literal) => {
            match (bias.is_some(), activation.is_some()) {
                (false, false) => {
                    gemm_packed_left_cached_blocked_16_impl::<$column_block, false, false>(
                        output, left, right, rows, inner, columns, bias, activation,
                    )
                }
                (false, true) => {
                    gemm_packed_left_cached_blocked_16_impl::<$column_block, false, true>(
                        output, left, right, rows, inner, columns, bias, activation,
                    )
                }
                (true, false) => {
                    gemm_packed_left_cached_blocked_16_impl::<$column_block, true, false>(
                        output, left, right, rows, inner, columns, bias, activation,
                    )
                }
                (true, true) => {
                    gemm_packed_left_cached_blocked_16_impl::<$column_block, true, true>(
                        output, left, right, rows, inner, columns, bias, activation,
                    )
                }
            }
        };
    }
    if inner >= 768 || (inner <= 128 && columns >= 10_000) {
        dispatch!(96);
    } else {
        dispatch!(24);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_left_cached_blocked_16_impl<
    const COLUMN_BLOCK: usize,
    const HAS_BIAS: bool,
    const HAS_ACTIVATION: bool,
>(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    const BLOCK_ROWS: usize = 16;
    const MICRO_COLUMNS: usize = 6;

    assert!(rows > 0 && inner > 0 && columns > 0);
    assert!(COLUMN_BLOCK > 0 && COLUMN_BLOCK.is_multiple_of(MICRO_COLUMNS));
    assert!(rows.is_multiple_of(BLOCK_ROWS));
    assert_eq!(output.len(), rows * columns);
    assert_eq!(left.len(), rows * inner);
    assert_eq!(right.len(), inner * columns);
    assert!(bias.is_none_or(|bias| bias.len() == rows));
    assert_eq!(HAS_BIAS, bias.is_some());
    assert_eq!(HAS_ACTIVATION, activation.is_some());

    if has_avx2_fma() {
        let mut packed_right = Buffer::zeroed(inner * COLUMN_BLOCK);
        for column_block_start in (0..columns).step_by(COLUMN_BLOCK) {
            let column_block = (columns - column_block_start).min(COLUMN_BLOCK);
            let panels = column_block.div_ceil(MICRO_COLUMNS);
            for index in 0..inner {
                let source = &right[index * columns + column_block_start..];
                for panel in 0..panels {
                    let panel_column = panel * MICRO_COLUMNS;
                    let panel_columns = (column_block - panel_column).min(MICRO_COLUMNS);
                    let packed_start = (panel * inner + index) * MICRO_COLUMNS;
                    packed_right[packed_start..packed_start + panel_columns]
                        .copy_from_slice(&source[panel_column..panel_column + panel_columns]);
                }
            }
            for panel in 0..panels {
                let panel_column = panel * MICRO_COLUMNS;
                let panel_columns = (column_block - panel_column).min(MICRO_COLUMNS);
                let column_start = column_block_start + panel_column;
                let right_start = panel * inner * MICRO_COLUMNS;
                let right = &packed_right[right_start..right_start + inner * MICRO_COLUMNS];
                for row_start in (0..rows).step_by(BLOCK_ROWS) {
                    let output = &mut output[row_start * columns + column_start..];
                    let left = &left[row_start * inner..(row_start + BLOCK_ROWS) * inner];
                    let bias = bias.map(|bias| &bias[row_start..row_start + BLOCK_ROWS]);
                    macro_rules! kernel {
                        ($columns:literal) => {
                            // SAFETY: AVX2/FMA were detected above. The operands
                            // contain complete 16-row and 6-column packed panels.
                            unsafe {
                                x86::gemm_16x6_packed::<$columns, HAS_BIAS, HAS_ACTIVATION>(
                                    output, left, right, inner, columns, bias, activation,
                                )
                            }
                        };
                    }
                    match panel_columns {
                        1 => kernel!(1),
                        2 => kernel!(2),
                        3 => kernel!(3),
                        4 => kernel!(4),
                        5 => kernel!(5),
                        6 => kernel!(6),
                        _ => unreachable!(),
                    }
                }
            }
        }
        return;
    }

    for row_start in (0..rows).step_by(BLOCK_ROWS) {
        for row in 0..BLOCK_ROWS {
            for column in 0..columns {
                let mut sum = bias.map_or(0.0, |bias| bias[row_start + row]);
                for index in 0..inner {
                    sum = left[row_start * inner + index * BLOCK_ROWS + row]
                        .mul_add(right[index * columns + column], sum);
                }
                output[(row_start + row) * columns + column] =
                    activation.map_or(sum, |activation| activation.apply(sum));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_packed_left_blocked_6(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    const BLOCK_ROWS: usize = 6;

    assert!(rows > 0 && inner > 0 && columns > 0);
    assert_eq!(output.len(), rows * columns);
    assert_eq!(left.len(), rows * inner);
    assert_eq!(right.len(), inner * columns);
    assert!(bias.is_none_or(|bias| bias.len() == rows));

    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        const COLUMN_BLOCK: usize = 16;
        const TARGET_LEFT_ELEMENTS: usize = 224 * 1024;
        let depth_block = (TARGET_LEFT_ELEMENTS / rows).clamp(64, 256) / 64 * 64;
        let full_rows = rows / BLOCK_ROWS * BLOCK_ROWS;

        if full_rows > 0 {
            for depth_start in (0..inner).step_by(depth_block) {
                let depth = (inner - depth_start).min(depth_block);
                let final_activation = (depth_start + depth == inner)
                    .then_some(activation)
                    .flatten();
                for column_start in (0..columns).step_by(COLUMN_BLOCK) {
                    let block_columns = (columns - column_start).min(COLUMN_BLOCK);
                    let right_start = depth_start * columns + column_start;
                    let right = &right[right_start..];
                    for row_start in (0..full_rows).step_by(BLOCK_ROWS) {
                        let output_start = row_start * columns + column_start;
                        let output = &mut output[output_start..];
                        let left_start = row_start * inner + depth_start * BLOCK_ROWS;
                        let left = &left[left_start..left_start + depth * BLOCK_ROWS];
                        let bias = bias.map(|bias| &bias[row_start..row_start + BLOCK_ROWS]);
                        // SAFETY: AVX2/FMA were detected above. The slices
                        // describe one packed six-row by at-most-sixteen-column
                        // tile with independent source and destination strides.
                        unsafe {
                            x86::gemm_6x16_packed::<true>(
                                output,
                                left,
                                right,
                                depth,
                                block_columns,
                                columns,
                                columns,
                                bias,
                                depth_start != 0,
                                final_activation,
                            )
                        };
                    }
                }
            }
        }

        let tail_rows = rows - full_rows;
        if tail_rows > 0 {
            let output = &mut output[full_rows * columns..];
            let left = &left[full_rows * inner..];
            let bias = bias.map(|bias| &bias[full_rows..]);
            macro_rules! tail {
                ($rows:literal) => {
                    // SAFETY: The final packed block stores exactly this many
                    // interleaved rows for every K index.
                    unsafe {
                        x86::gemm_rows_8::<$rows, true>(
                            output, left, right, inner, columns, columns, bias, None, activation,
                        )
                    }
                };
            }
            match tail_rows {
                1 => tail!(1),
                2 => tail!(2),
                3 => tail!(3),
                4 => tail!(4),
                5 => tail!(5),
                _ => unreachable!(),
            }
        }
        return;
    }

    for row_start in (0..rows).step_by(BLOCK_ROWS) {
        let block_rows = (rows - row_start).min(BLOCK_ROWS);
        let left = &left[row_start * inner..(row_start + block_rows) * inner];
        for row in 0..block_rows {
            for column in 0..columns {
                let mut sum = bias.map_or(0.0, |bias| bias[row_start + row]);
                for index in 0..inner {
                    sum = left[index * block_rows + row]
                        .mul_add(right[index * columns + column], sum);
                }
                output[(row_start + row) * columns + column] =
                    activation.map_or(sum, |activation| activation.apply(sum));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_packed_left_tile(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
    block_rows: usize,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert!(columns <= 32);
    assert!(matches!(block_rows, 6 | 12));
    assert_eq!(output.len(), rows * columns);
    assert_eq!(left.len(), rows * inner);
    assert!(right_stride >= columns);
    assert!(right.len() >= (inner - 1) * right_stride + columns);
    assert!(bias.is_none_or(|bias| bias.len() == rows));

    for row_start in (0..rows).step_by(block_rows) {
        let tile_rows = (rows - row_start).min(block_rows);
        let output = &mut output[row_start * columns..(row_start + tile_rows) * columns];
        let left = &left[row_start * inner..(row_start + tile_rows) * inner];
        let bias = bias.map(|bias| &bias[row_start..row_start + tile_rows]);

        #[cfg(target_arch = "x86_64")]
        if block_rows == 6 && tile_rows == 6 && has_avx2_fma() {
            // SAFETY: AVX2/FMA support and all matrix extents are checked above.
            // The packed block contains six weights per K position, while the
            // right-hand tile uses the owning NCHW plane as its row stride.
            unsafe {
                if right_stride == columns {
                    x86::gemm_6x16_packed::<false>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        columns,
                        right_stride,
                        bias,
                        false,
                        activation,
                    )
                } else {
                    x86::gemm_6x16_packed::<true>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        columns,
                        right_stride,
                        bias,
                        false,
                        activation,
                    )
                }
            };
            continue;
        }

        gemm_rows(
            output,
            left,
            right,
            tile_rows,
            inner,
            columns,
            right_stride,
            bias,
            None,
            true,
            activation,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_system_dense(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert!(bias.is_none_or(|bias| bias.len() == rows));
    #[cfg(target_os = "macos")]
    {
        accelerate::sgemm(output, left, right, rows, inner, columns);
        output
            .par_chunks_mut(columns)
            .enumerate()
            .for_each(|(row, output)| {
                if let Some(bias) = bias {
                    affine_in_place(output, 1.0, bias[row]);
                }
                if let Some(activation) = activation {
                    unary_chunk(output, activation);
                }
            });
    }
    #[cfg(not(target_os = "macos"))]
    gemm_with_activation(output, left, right, rows, inner, columns, bias, activation);
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
pub(crate) fn linear_system_dense(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    softmax: bool,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert!(bias.is_none_or(|bias| bias.len() == columns));
    accelerate::sgemm_right_transposed(output, input, weight, rows, inner, columns);
    output.par_chunks_mut(columns).for_each(|row| {
        if softmax {
            if let Some(bias) = bias {
                bias_softmax_in_place(row, bias);
            } else {
                softmax_in_place(row);
            }
        } else if let Some(bias) = bias {
            add_in_place(row, bias);
        }
    });
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_os = "macos"))]
pub(crate) fn linear_right_transposed(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    #[cfg(target_arch = "x86_64")] weight_block_columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
    softmax: bool,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert_eq!(output.len(), rows * columns);
    assert_eq!(input.len(), rows * inner);
    assert_eq!(weight.len(), columns * inner);
    assert!(bias.is_none_or(|bias| bias.len() == columns));
    assert!(!softmax || activation.is_none());

    #[cfg(target_arch = "x86_64")]
    {
        assert!(matches!(weight_block_columns, 8 | 16));
        if weight_block_columns == 16 {
            linear_right_transposed_6x16(
                output, input, weight, rows, inner, columns, bias, activation, softmax,
            );
            return;
        }
    }

    const MICRO_ROWS: usize = 8;
    let row_blocks = rows.div_ceil(MICRO_ROWS);
    let blocks_per_task = row_blocks.div_ceil(rayon::current_num_threads()).max(1);
    let task_rows = blocks_per_task * MICRO_ROWS;
    output
        .par_chunks_mut(task_rows * columns)
        .enumerate()
        .for_each(|(task, output)| {
            let row_start = task * task_rows;
            let task_row_count = (rows - row_start).min(task_rows);
            for local_row in (0..task_row_count).step_by(MICRO_ROWS) {
                let block_rows = (task_row_count - local_row).min(MICRO_ROWS);
                let input_start = (row_start + local_row) * inner;
                let input = &input[input_start..input_start + block_rows * inner];
                let output = &mut output[local_row * columns..(local_row + block_rows) * columns];

                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma() {
                    macro_rules! dispatch_rows {
                        ($rows:literal) => {
                            // SAFETY: AVX2/FMA were checked above. All slices
                            // contain the complete row block described here.
                            unsafe {
                                x86::linear_rows_8::<$rows>(
                                    output, input, weight, inner, columns, bias, activation,
                                )
                            }
                        };
                    }
                    match block_rows {
                        1 => dispatch_rows!(1),
                        2 => dispatch_rows!(2),
                        3 => dispatch_rows!(3),
                        4 => dispatch_rows!(4),
                        5 => dispatch_rows!(5),
                        6 => dispatch_rows!(6),
                        7 => dispatch_rows!(7),
                        8 => dispatch_rows!(8),
                        _ => unreachable!(),
                    }
                } else {
                    linear_rows_scalar(
                        output, input, weight, block_rows, inner, columns, bias, activation,
                    );
                }

                #[cfg(not(target_arch = "x86_64"))]
                linear_rows_scalar(
                    output, input, weight, block_rows, inner, columns, bias, activation,
                );

                if softmax {
                    output.chunks_mut(columns).for_each(softmax_in_place);
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
fn linear_right_transposed_6x16(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
    softmax: bool,
) {
    const MICRO_ROWS: usize = 6;
    let row_blocks = rows.div_ceil(MICRO_ROWS);
    let blocks_per_task = row_blocks.div_ceil(rayon::current_num_threads()).max(1);
    let task_rows = blocks_per_task * MICRO_ROWS;
    let arena = ArenaHandle::current();
    output
        .par_chunks_mut(task_rows * columns)
        .enumerate()
        .for_each(|(task, output)| {
            let row_start = task * task_rows;
            let task_row_count = (rows - row_start).min(task_rows);

            if has_avx2_fma() {
                let task_input_start = row_start * inner;
                let task_input =
                    &input[task_input_start..task_input_start + task_row_count * inner];
                let mut packed_input = arena.zeroed(task_row_count * inner);
                for local_row in (0..task_row_count).step_by(MICRO_ROWS) {
                    let block_rows = (task_row_count - local_row).min(MICRO_ROWS);
                    let packed_start = local_row * inner;
                    for index in 0..inner {
                        for row in 0..block_rows {
                            packed_input[packed_start + index * block_rows + row] =
                                task_input[(local_row + row) * inner + index];
                        }
                    }
                }

                for column_start in (0..columns).step_by(16) {
                    let block_columns = (columns - column_start).min(16);
                    let weight_start = column_start * inner;
                    let weight = &weight[weight_start..weight_start + block_columns * inner];
                    let bias = bias.map(|bias| &bias[column_start..column_start + block_columns]);

                    for local_row in (0..task_row_count).step_by(MICRO_ROWS) {
                        let block_rows = (task_row_count - local_row).min(MICRO_ROWS);
                        let packed_start = local_row * inner;
                        let packed_input =
                            &packed_input[packed_start..packed_start + block_rows * inner];
                        let output_start = local_row * columns + column_start;
                        let output = &mut output[output_start..];
                        macro_rules! dispatch_rows {
                            ($rows:literal) => {
                                // SAFETY: AVX2/FMA were detected above. Input
                                // and weights use the exact packed block widths.
                                unsafe {
                                    x86::linear_6x16_packed::<$rows>(
                                        output,
                                        columns,
                                        packed_input,
                                        weight,
                                        inner,
                                        block_columns,
                                        bias,
                                        activation,
                                    )
                                }
                            };
                        }
                        match block_rows {
                            1 => dispatch_rows!(1),
                            2 => dispatch_rows!(2),
                            3 => dispatch_rows!(3),
                            4 => dispatch_rows!(4),
                            5 => dispatch_rows!(5),
                            6 => dispatch_rows!(6),
                            _ => unreachable!(),
                        }
                    }
                }

                if softmax {
                    output.chunks_mut(columns).for_each(softmax_in_place);
                }
                return;
            }

            for local_row in (0..task_row_count).step_by(MICRO_ROWS) {
                let block_rows = (task_row_count - local_row).min(MICRO_ROWS);
                let input_start = (row_start + local_row) * inner;
                let input = &input[input_start..input_start + block_rows * inner];
                let output = &mut output[local_row * columns..(local_row + block_rows) * columns];
                linear_rows_scalar_x86_packed(
                    output, input, weight, block_rows, inner, columns, 16, bias, activation,
                );
                if softmax {
                    output.chunks_mut(columns).for_each(softmax_in_place);
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
fn linear_rows_scalar_x86_packed(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    weight_block_columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    assert!(weight_block_columns > 0);
    for column_start in (0..columns).step_by(weight_block_columns) {
        let block_columns = (columns - column_start).min(weight_block_columns);
        let weight = &weight[column_start * inner..(column_start + block_columns) * inner];
        for row in 0..rows {
            for column in 0..block_columns {
                let mut sum = bias.map_or(0.0, |bias| bias[column_start + column]);
                for index in 0..inner {
                    sum = input[row * inner + index]
                        .mul_add(weight[index * block_columns + column], sum);
                }
                output[row * columns + column_start + column] =
                    activation.map_or(sum, |activation| activation.apply(sum));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_os = "macos"))]
fn linear_rows_scalar(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    #[cfg(target_arch = "x86_64")]
    for column_start in (0..columns).step_by(8) {
        let block_columns = (columns - column_start).min(8);
        let weight = &weight[column_start * inner..(column_start + block_columns) * inner];
        for row in 0..rows {
            for column in 0..block_columns {
                let mut sum = bias.map_or(0.0, |bias| bias[column_start + column]);
                for index in 0..inner {
                    sum = input[row * inner + index]
                        .mul_add(weight[index * block_columns + column], sum);
                }
                output[row * columns + column_start + column] =
                    activation.map_or(sum, |activation| activation.apply(sum));
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = bias.map_or(0.0, |bias| bias[column]);
            for index in 0..inner {
                sum = input[row * inner + index].mul_add(weight[column * inner + index], sum);
            }
            output[row * columns + column] =
                activation.map_or(sum, |activation| activation.apply(sum));
        }
    }
}

pub(crate) fn gemm_column_bias_softmax(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    column_bias: &[f32],
) {
    gemm_impl(
        output,
        left,
        right,
        rows,
        inner,
        columns,
        None,
        Some(column_bias),
        false,
        None,
        true,
    );
}

#[allow(clippy::too_many_arguments)]
fn gemm_impl(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
    packed_left: bool,
    activation: Option<UnaryOperation>,
    row_softmax: bool,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert_eq!(rows.checked_mul(columns), Some(output.len()));
    assert_eq!(rows.checked_mul(inner), Some(left.len()));
    assert_eq!(inner.checked_mul(columns), Some(right.len()));
    assert!(bias.is_none_or(|bias| bias.len() == rows));
    assert!(column_bias.is_none_or(|bias| bias.len() == columns));
    assert!(bias.is_none() || column_bias.is_none());
    assert!(!packed_left || column_bias.is_none());
    let micro_rows = if packed_left { 12 } else { 8 };
    let row_blocks = rows.div_ceil(micro_rows);
    let blocks_per_task = row_blocks.div_ceil(rayon::current_num_threads()).max(1);
    let task_rows = blocks_per_task * micro_rows;
    output
        .par_chunks_mut(task_rows * columns)
        .enumerate()
        .for_each(|(task, output)| {
            let task_row_start = task * task_rows;
            let task_row_count = (rows - task_row_start).min(task_rows);
            for local_row in (0..task_row_count).step_by(micro_rows) {
                let block_rows = (task_row_count - local_row).min(micro_rows);
                let row_start = task_row_start + local_row;
                let output = &mut output[local_row * columns..(local_row + block_rows) * columns];
                let left = &left[row_start * inner..(row_start + block_rows) * inner];
                let bias = bias.map(|bias| &bias[row_start..row_start + block_rows]);
                gemm_rows(
                    output,
                    left,
                    right,
                    block_rows,
                    inner,
                    columns,
                    columns,
                    bias,
                    column_bias,
                    packed_left,
                    activation,
                );
                if row_softmax {
                    for row in output.chunks_mut(columns) {
                        softmax_in_place(row);
                    }
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_packed_panels(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    panels: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    const PANEL_COLUMNS: usize = 16;

    assert_eq!(output.len(), panels * rows * PANEL_COLUMNS);
    assert_eq!(left.len(), rows * inner);
    assert_eq!(right.len(), panels * inner * PANEL_COLUMNS);
    #[cfg(target_arch = "aarch64")]
    if rows.is_multiple_of(4) {
        gemm_packed_panels_blocked(output, left, right, rows, inner, panels, bias, activation);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if rows.is_multiple_of(4) && has_avx2_fma() {
        gemm_packed_panels_blocked(output, left, right, rows, inner, panels, bias, activation);
        return;
    }
    output
        .par_chunks_mut(rows * PANEL_COLUMNS)
        .zip(right.par_chunks(inner * PANEL_COLUMNS))
        .for_each(|(output, right)| {
            for row_start in (0..rows).step_by(4) {
                let block_rows = (rows - row_start).min(4);
                let output = &mut output
                    [row_start * PANEL_COLUMNS..(row_start + block_rows) * PANEL_COLUMNS];
                let left = &left[row_start * inner..(row_start + block_rows) * inner];
                let bias = bias.map(|bias| &bias[row_start..row_start + block_rows]);
                gemm_rows(
                    output,
                    left,
                    right,
                    block_rows,
                    inner,
                    PANEL_COLUMNS,
                    PANEL_COLUMNS,
                    bias,
                    None,
                    true,
                    activation,
                );
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_sparse_packed_panels(
    output: &mut [f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    panels: usize,
    bias: Option<&[f32]>,
    row_offsets: &[usize],
    indices: &[u32],
    values: &[f32],
    activation: Option<UnaryOperation>,
) {
    const PANEL_COLUMNS: usize = 16;
    const BLOCK_ROWS: usize = 4;

    assert_eq!(output.len(), panels * rows * PANEL_COLUMNS);
    assert_eq!(right.len(), panels * inner * PANEL_COLUMNS);
    assert!(rows.is_multiple_of(BLOCK_ROWS));
    assert_eq!(row_offsets.len(), rows / BLOCK_ROWS + 1);
    assert_eq!(values.len(), indices.len() * BLOCK_ROWS);
    output
        .par_chunks_mut(rows * PANEL_COLUMNS)
        .zip(right.par_chunks(inner * PANEL_COLUMNS))
        .for_each(|(output, right)| {
            for block in 0..rows / BLOCK_ROWS {
                let row_start = block * BLOCK_ROWS;
                let entry_start = row_offsets[block];
                let entry_end = row_offsets[block + 1];
                let output = &mut output
                    [row_start * PANEL_COLUMNS..(row_start + BLOCK_ROWS) * PANEL_COLUMNS];
                let bias = bias.map(|bias| &bias[row_start..row_start + BLOCK_ROWS]);
                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma() {
                    // SAFETY: AVX2/FMA were checked above. Every sparse index
                    // selects a complete packed RHS row.
                    unsafe {
                        x86::gemm_4x16_sparse(
                            output,
                            right,
                            &indices[entry_start..entry_end],
                            &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                            PANEL_COLUMNS,
                            PANEL_COLUMNS,
                            bias,
                            activation,
                        )
                    };
                    continue;
                }
                #[cfg(target_arch = "aarch64")]
                // SAFETY: Indices reference complete 16-column rows in the
                // packed RHS, and weights contain four values per entry.
                unsafe {
                    neon::gemm_4x16_sparse(
                        output,
                        right,
                        &indices[entry_start..entry_end],
                        &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                        PANEL_COLUMNS,
                        PANEL_COLUMNS,
                        bias,
                    );
                    if let Some(activation) = activation {
                        unary_chunk(output, activation);
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                gemm_4_sparse_scalar(
                    output,
                    right,
                    &indices[entry_start..entry_end],
                    &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                    PANEL_COLUMNS,
                    PANEL_COLUMNS,
                    bias,
                    activation,
                );
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_sparse_packed_left(
    output: &mut [f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    row_offsets: &[usize],
    indices: &[u32],
    values: &[f32],
    activation: Option<UnaryOperation>,
) {
    const BLOCK_ROWS: usize = 4;

    assert_eq!(output.len(), rows * columns);
    assert_eq!(right.len(), inner * columns);
    assert!(rows.is_multiple_of(BLOCK_ROWS));
    assert_eq!(row_offsets.len(), rows / BLOCK_ROWS + 1);
    assert_eq!(values.len(), indices.len() * BLOCK_ROWS);
    output
        .par_chunks_mut(BLOCK_ROWS * columns)
        .enumerate()
        .for_each(|(block, output)| {
            let entry_start = row_offsets[block];
            let entry_end = row_offsets[block + 1];
            let row_start = block * BLOCK_ROWS;
            let bias = bias.map(|bias| &bias[row_start..row_start + BLOCK_ROWS]);
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma() {
                // SAFETY: AVX2/FMA were checked above. Every sparse index
                // selects a complete RHS row of `columns` values.
                unsafe {
                    x86::gemm_4x16_sparse(
                        output,
                        right,
                        &indices[entry_start..entry_end],
                        &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                        columns,
                        columns,
                        bias,
                        activation,
                    )
                };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: Every sparse index identifies a complete RHS row and
            // weights contain four values per entry.
            unsafe {
                neon::gemm_4x16_sparse(
                    output,
                    right,
                    &indices[entry_start..entry_end],
                    &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                    columns,
                    columns,
                    bias,
                );
                if let Some(activation) = activation {
                    unary_chunk(output, activation);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            gemm_4_sparse_scalar(
                output,
                right,
                &indices[entry_start..entry_end],
                &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                columns,
                columns,
                bias,
                activation,
            );
        });
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
fn gemm_4_sparse_scalar(
    output: &mut [f32],
    right: &[f32],
    indices: &[u32],
    values: &[f32],
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    for row in 0..4 {
        for column in 0..columns {
            let mut sum = bias.map_or(0.0, |bias| bias[row]);
            for (entry, &index) in indices.iter().enumerate() {
                sum = values[entry * 4 + row]
                    .mul_add(right[index as usize * right_stride + column], sum);
            }
            output[row * columns + column] =
                activation.map_or(sum, |activation| activation.apply(sum));
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_panels_blocked(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    _panels: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    const PANEL_COLUMNS: usize = 16;
    const DEPTH_BLOCK: usize = 256;

    for depth_start in (0..inner).step_by(DEPTH_BLOCK) {
        let depth = (inner - depth_start).min(DEPTH_BLOCK);
        output
            .par_chunks_mut(rows * PANEL_COLUMNS)
            .zip(right.par_chunks(inner * PANEL_COLUMNS))
            .for_each(|(output, right)| {
                let right_start = depth_start * PANEL_COLUMNS;
                let right = &right[right_start..right_start + depth * PANEL_COLUMNS];
                for row_start in (0..rows).step_by(4) {
                    let output =
                        &mut output[row_start * PANEL_COLUMNS..(row_start + 4) * PANEL_COLUMNS];
                    let left_start = row_start * inner + depth_start * 4;
                    let left = &left[left_start..left_start + depth * 4];
                    let bias = bias.map(|bias| &bias[row_start..row_start + 4]);
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2/FMA availability is checked by the caller;
                    // slices describe one packed 4x16 tile.
                    unsafe {
                        x86::gemm_4x16_packed(
                            output,
                            left,
                            right,
                            depth,
                            PANEL_COLUMNS,
                            PANEL_COLUMNS,
                            PANEL_COLUMNS,
                            bias,
                            depth_start != 0,
                            (depth_start + depth == inner)
                                .then_some(activation)
                                .flatten(),
                        )
                    };
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: The slices describe a complete 4x16 tile and
                    // NEON is mandatory on AArch64.
                    unsafe {
                        neon::gemm_4x16_packed(
                            output,
                            left,
                            right,
                            depth,
                            PANEL_COLUMNS,
                            PANEL_COLUMNS,
                            bias,
                            depth_start != 0,
                        )
                    };
                }
            });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(activation) = activation {
        output
            .par_chunks_mut(rows * PANEL_COLUMNS)
            .for_each(|output| unary_chunk(output, activation));
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_rows(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
    packed_left: bool,
    activation: Option<UnaryOperation>,
) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        if rows == 12 && packed_left {
            debug_assert!(column_bias.is_none());
            // SAFETY: AVX2 and FMA were detected at runtime. The caller
            // supplies twelve packed left rows and complete output rows.
            unsafe {
                x86::gemm_rows_8::<12, true>(
                    output,
                    left,
                    right,
                    inner,
                    columns,
                    right_stride,
                    bias,
                    None,
                    activation,
                )
            };
            return;
        }
        if rows == 8 {
            // SAFETY: AVX2 and FMA were detected at runtime. Slice dimensions
            // describe eight complete rows in the selected left layout.
            unsafe {
                if packed_left {
                    debug_assert!(column_bias.is_none());
                    x86::gemm_rows_8::<8, true>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        right_stride,
                        bias,
                        None,
                        activation,
                    )
                } else {
                    x86::gemm_rows_8::<8, false>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        right_stride,
                        bias,
                        column_bias,
                        activation,
                    )
                }
            };
            return;
        }
        if rows == 4 {
            // SAFETY: Same runtime feature and matrix-bounds argument as the
            // eight-row kernel above.
            unsafe {
                if packed_left {
                    debug_assert!(column_bias.is_none());
                    x86::gemm_rows_8::<4, true>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        right_stride,
                        bias,
                        None,
                        activation,
                    )
                } else {
                    x86::gemm_rows_8::<4, false>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        right_stride,
                        bias,
                        column_bias,
                        activation,
                    )
                }
            };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if rows == 12 && packed_left {
        // SAFETY: The caller supplies exactly twelve complete output rows and
        // twelve interleaved weights for every inner-dimension position.
        debug_assert!(column_bias.is_none());
        const DEPTH_BLOCK: usize = 256;
        for depth_start in (0..inner).step_by(DEPTH_BLOCK) {
            let depth = (inner - depth_start).min(DEPTH_BLOCK);
            let packed_left = &left[depth_start * 12..(depth_start + depth) * 12];
            let right = &right[depth_start * right_stride..];
            unsafe {
                neon::gemm_12x8_packed(
                    output,
                    packed_left,
                    right,
                    depth,
                    columns,
                    right_stride,
                    bias,
                    depth_start != 0,
                )
            };
        }
        if let Some(activation) = activation {
            unary_chunk(output, activation);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if rows == 8 {
        // SAFETY: The caller supplies exactly eight complete output/left rows;
        // right_stride is validated by the owning full matrix in gemm.
        unsafe {
            if packed_left {
                debug_assert!(column_bias.is_none());
                neon::gemm_8x12_packed(output, left, right, inner, columns, right_stride, bias)
            } else {
                neon::gemm_8x12(
                    output,
                    left,
                    right,
                    inner,
                    columns,
                    right_stride,
                    bias,
                    column_bias,
                )
            }
        };
        if let Some(activation) = activation {
            unary_chunk(output, activation);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if rows == 4 {
        // SAFETY: Same bounds argument as the eight-row kernel.
        unsafe {
            if packed_left {
                debug_assert!(column_bias.is_none());
                neon::gemm_4x16_packed(
                    output,
                    left,
                    right,
                    inner,
                    columns,
                    right_stride,
                    bias,
                    false,
                )
            } else {
                neon::gemm_4x16(
                    output,
                    left,
                    right,
                    inner,
                    columns,
                    right_stride,
                    bias,
                    column_bias,
                )
            }
        };
        if let Some(activation) = activation {
            unary_chunk(output, activation);
        }
        return;
    }
    gemm_scalar_strided(
        output,
        left,
        right,
        rows,
        inner,
        columns,
        right_stride,
        bias,
        column_bias,
        packed_left,
    );
    if let Some(activation) = activation {
        unary_chunk(output, activation);
    }
}

pub(crate) fn max_pool_2x2_same_upper(output: &mut [f32], input: &[f32], width: usize) {
    assert_eq!(output.len(), input.len());
    assert!(width > 0 && input.len().is_multiple_of(width));
    let height = input.len() / width;
    for y in 0..height {
        let current = &input[y * width..(y + 1) * width];
        let next = (y + 1 < height).then(|| &input[(y + 1) * width..(y + 2) * width]);
        let output = &mut output[y * width..(y + 1) * width];
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: Rows have identical lengths and the kernel handles the
            // right edge without reading past either row.
            unsafe { neon::max_pool_2x2_row(output, current, next) };
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: AVX2/FMA were detected at runtime. Both source rows and
            // the destination have the same width, and the kernel handles the
            // right edge without an out-of-bounds shifted load.
            unsafe { x86::max_pool_2x2_row(output, current, next) };
            continue;
        }
        #[cfg(not(target_arch = "aarch64"))]
        for x in 0..width {
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
}

#[allow(clippy::too_many_arguments)]
fn gemm_scalar_strided(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
    packed_left: bool,
) {
    for row in 0..rows {
        let output = &mut output[row * columns..(row + 1) * columns];
        if let Some(column_bias) = column_bias {
            output.copy_from_slice(column_bias);
        } else {
            output.fill(bias.map_or(0.0, |bias| bias[row]));
        }
        for index in 0..inner {
            axpy(
                output,
                &right[index * right_stride..index * right_stride + columns],
                if packed_left {
                    left[index * rows + row]
                } else {
                    left[row * inner + index]
                },
            );
        }
    }
}

pub(crate) fn unary_in_place(values: &mut [f32], operation: UnaryOperation) {
    const PARALLEL_CHUNK: usize = 32 * 1024;
    if values.len() <= PARALLEL_CHUNK {
        unary_chunk(values, operation);
        return;
    }
    values.par_chunks_mut(PARALLEL_CHUNK).for_each(|values| {
        unary_chunk(values, operation);
    });
}

pub(crate) fn softmax_in_place(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The kernel reads and writes only complete vectors inside the
        // supplied slice and handles the remaining values with safe indexing.
        unsafe { neon::softmax(values) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the kernel only
        // accesses full vectors and a scalar tail inside the slice.
        unsafe { x86::softmax(values) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for value in values.iter_mut() {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        let reciprocal = sum.recip();
        for value in values {
            *value *= reciprocal;
        }
    }
}

pub(crate) fn bias_softmax_in_place(values: &mut [f32], bias: &[f32]) {
    assert_eq!(values.len(), bias.len());
    #[cfg(target_arch = "aarch64")]
    // SAFETY: Equal-length slices bound all vector loads and stores.
    unsafe {
        neon::bias_softmax(values, bias)
    };
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2/FMA were detected and equal-length slices bound every
        // load and store, including the scalar tail.
        unsafe { x86::bias_softmax(values, bias) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        add_in_place(values, bias);
        softmax_in_place(values);
    }
}

pub(crate) fn mean(values: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    let sum = {
        // SAFETY: The kernel only loads complete vectors within the slice and
        // handles its tail through safe indexing.
        unsafe { neon::sum(values) }
    };
    #[cfg(target_arch = "x86_64")]
    let sum = if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the reduction only
        // reads full vectors and a scalar tail inside the slice.
        unsafe { x86::sum(values) }
    } else {
        values.iter().copied().sum::<f32>()
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let sum = values.iter().copied().sum::<f32>();
    sum / values.len() as f32
}

pub(crate) fn layer_norm_in_place(values: &mut [f32], weight: &[f32], bias: &[f32], epsilon: f32) {
    assert!(!values.is_empty());
    assert_eq!(values.len(), weight.len());
    assert_eq!(values.len(), bias.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: Runtime feature detection covers the AVX2/FMA kernel and all
        // three slices have the same validated length.
        unsafe { x86::layer_norm(values, weight, bias, epsilon) };
        return;
    }

    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    let inverse_std = (variance + epsilon).sqrt().recip();
    for ((value, weight), bias) in values.iter_mut().zip(weight).zip(bias) {
        *value = (*value - mean).mul_add(inverse_std * *weight, *bias);
    }
}

fn unary_chunk(values: &mut [f32], operation: UnaryOperation) {
    #[cfg(target_arch = "aarch64")]
    {
        match operation {
            UnaryOperation::Relu => {
                // SAFETY: The operation is in-place and stays in slice bounds.
                unsafe { neon::relu(values) };
                return;
            }
            UnaryOperation::Gelu => {
                // SAFETY: The operation is in-place and stays in slice bounds.
                unsafe { neon::gelu(values) };
                return;
            }
            _ => {}
        }
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match operation {
            UnaryOperation::Relu => {
                // SAFETY: AVX2 and FMA were detected at runtime, and the
                // operation stays within the supplied slice.
                unsafe { x86::relu(values) };
                return;
            }
            UnaryOperation::Gelu => {
                // SAFETY: Same feature and slice-bounds argument as ReLU.
                unsafe { x86::gelu(values) };
                return;
            }
            UnaryOperation::Silu => {
                // SAFETY: Same feature and slice-bounds argument as ReLU.
                unsafe { x86::silu(values) };
                return;
            }
            UnaryOperation::Sigmoid => {
                // SAFETY: Same feature and slice-bounds argument as ReLU.
                unsafe { x86::sigmoid(values) };
                return;
            }
            _ => {}
        }
    }
    for value in values {
        *value = operation.apply(*value);
    }
}

#[derive(Clone, Copy)]
pub(crate) enum UnaryOperation {
    Relu,
    Erf,
    Gelu,
    HardSwish,
    Sigmoid,
    Silu,
    Sqrt,
    HardSigmoid { alpha: f32, beta: f32 },
}

impl UnaryOperation {
    #[inline]
    pub(super) fn apply(self, value: f32) -> f32 {
        match self {
            Self::Relu => value.max(0.0),
            Self::Erf => erf(value),
            Self::Gelu => 0.5 * value * (1.0 + erf(value * std::f32::consts::FRAC_1_SQRT_2)),
            Self::HardSwish => value * (value / 6.0 + 0.5).clamp(0.0, 1.0),
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Silu => value / (1.0 + (-value).exp()),
            Self::Sqrt => value.sqrt(),
            Self::HardSigmoid { alpha, beta } => (value.mul_add(alpha, beta)).clamp(0.0, 1.0),
        }
    }
}

// Abramowitz-Stegun 7.1.26. The maximum absolute error is about 1.5e-7,
// which is below the accumulated F32 error of the surrounding GELU graph.
#[inline]
fn erf(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / x.mul_add(0.327_591_1, 1.0);
    let polynomial = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - polynomial * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        assert!(erf(0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 2e-7);
        assert!((erf(-2.0) + 0.995_322_3).abs() < 2e-7);
    }

    #[test]
    fn vector_gelu_stays_close_to_scalar_formula() {
        let input = (0..1025)
            .map(|index| index as f32 * (16.0 / 1024.0) - 8.0)
            .collect::<Vec<_>>();
        let expected = input
            .iter()
            .map(|value| UnaryOperation::Gelu.apply(*value))
            .collect::<Vec<_>>();
        let mut actual = input;
        unary_in_place(&mut actual, UnaryOperation::Gelu);
        let maximum_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 2.0e-5, "maximum error: {maximum_error}");
    }

    #[test]
    fn packed_gemm_matches_row_major_gemm() {
        let rows = 28;
        let inner = 7;
        let columns = 19;
        let left = (0..rows * inner)
            .map(|index| ((index * 17 % 29) as f32 - 14.0) / 11.0)
            .collect::<Vec<_>>();
        let right = (0..inner * columns)
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 9.0)
            .collect::<Vec<_>>();
        let bias = (0..rows)
            .map(|row| (row as f32 - 8.0) / 7.0)
            .collect::<Vec<_>>();
        let mut packed = Vec::with_capacity(left.len());
        for row_start in (0..rows).step_by(12) {
            let block_rows = (rows - row_start).min(12);
            for index in 0..inner {
                for row in 0..block_rows {
                    packed.push(left[(row_start + row) * inner + index]);
                }
            }
        }
        let mut expected = vec![0.0; rows * columns];
        let mut actual = vec![0.0; rows * columns];
        gemm(
            &mut expected,
            &left,
            &right,
            rows,
            inner,
            columns,
            Some(&bias),
        );
        gemm_packed_left(
            &mut actual,
            &packed,
            &right,
            rows,
            inner,
            columns,
            Some(&bias),
        );
        let maximum_error = expected
            .iter()
            .zip(&actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 2e-6, "maximum error: {maximum_error}");
    }

    #[test]
    fn blocked_6x16_gemm_matches_scalar_across_row_depth_and_column_tails() {
        let rows = 10;
        let inner = 319;
        let columns = 25;
        let left = (0..rows * inner)
            .map(|index| ((index * 19 % 59) as f32 - 29.0) / 31.0)
            .collect::<Vec<_>>();
        let right = (0..inner * columns)
            .map(|index| ((index * 11 % 43) as f32 - 21.0) / 29.0)
            .collect::<Vec<_>>();
        let bias = (0..rows)
            .map(|row| (row as f32 - 4.0) / 13.0)
            .collect::<Vec<_>>();
        let mut packed = Vec::with_capacity(left.len());
        for row_start in (0..rows).step_by(6) {
            let block_rows = (rows - row_start).min(6);
            for index in 0..inner {
                for row in 0..block_rows {
                    packed.push(left[(row_start + row) * inner + index]);
                }
            }
        }

        let mut expected = vec![0.0; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = bias[row];
                for index in 0..inner {
                    sum = left[row * inner + index].mul_add(right[index * columns + column], sum);
                }
                expected[row * columns + column] = UnaryOperation::Silu.apply(sum);
            }
        }
        let mut actual = vec![0.0; rows * columns];
        gemm_packed_left_blocked_6(
            &mut actual,
            &packed,
            &right,
            rows,
            inner,
            columns,
            Some(&bias),
            Some(UnaryOperation::Silu),
        );
        let maximum_error = expected
            .iter()
            .zip(&actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 3e-5, "maximum error: {maximum_error}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cached_16x6_gemm_matches_scalar_across_depth_and_column_tails() {
        let rows = 32;
        let inner = 319;
        let columns = 25;
        let left = (0..rows * inner)
            .map(|index| ((index * 19 % 59) as f32 - 29.0) / 31.0)
            .collect::<Vec<_>>();
        let right = (0..inner * columns)
            .map(|index| ((index * 11 % 43) as f32 - 21.0) / 29.0)
            .collect::<Vec<_>>();
        let bias = (0..rows)
            .map(|row| (row as f32 - 4.0) / 13.0)
            .collect::<Vec<_>>();
        let mut packed = Vec::with_capacity(left.len());
        for row_start in (0..rows).step_by(16) {
            for index in 0..inner {
                for row in 0..16 {
                    packed.push(left[(row_start + row) * inner + index]);
                }
            }
        }

        let mut expected = vec![0.0; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = bias[row];
                for index in 0..inner {
                    sum = left[row * inner + index].mul_add(right[index * columns + column], sum);
                }
                expected[row * columns + column] = UnaryOperation::Silu.apply(sum);
            }
        }
        let mut actual = vec![0.0; expected.len()];
        gemm_packed_left_cached_blocked_16(
            &mut actual,
            &packed,
            &right,
            rows,
            inner,
            columns,
            Some(&bias),
            Some(UnaryOperation::Silu),
        );
        let maximum_error = expected
            .iter()
            .zip(&actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 3e-5, "maximum error: {maximum_error}");
    }

    #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
    #[test]
    fn linear_scalar_fallback_decodes_packed_sixteen_column_weights() {
        let rows = 13;
        let inner = 9;
        let columns = 22;
        let input = (0..rows * inner)
            .map(|index| ((index * 17 % 41) as f32 - 20.0) / 23.0)
            .collect::<Vec<_>>();
        let row_major_weight = (0..columns * inner)
            .map(|index| ((index * 11 % 43) as f32 - 21.0) / 29.0)
            .collect::<Vec<_>>();
        let bias = (0..columns)
            .map(|column| (column as f32 - 10.0) / 31.0)
            .collect::<Vec<_>>();
        let mut packed_weight = Vec::with_capacity(row_major_weight.len());
        for column_start in (0..columns).step_by(16) {
            let block_columns = (columns - column_start).min(16);
            for index in 0..inner {
                for column in 0..block_columns {
                    packed_weight.push(row_major_weight[(column_start + column) * inner + index]);
                }
            }
        }

        let mut expected = vec![0.0; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = bias[column];
                for index in 0..inner {
                    sum = input[row * inner + index]
                        .mul_add(row_major_weight[column * inner + index], sum);
                }
                expected[row * columns + column] = UnaryOperation::Silu.apply(sum);
            }
        }
        let mut actual = vec![0.0; expected.len()];
        linear_rows_scalar_x86_packed(
            &mut actual,
            &input,
            &packed_weight,
            rows,
            inner,
            columns,
            16,
            Some(&bias),
            Some(UnaryOperation::Silu),
        );
        assert_eq!(actual, expected);
    }

    #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
    #[test]
    fn conditional_linear_6x16_matches_large_layout_with_full_row_softmax() {
        let rows = 40;
        let inner = 192;
        let columns = 6_906;
        let input = (0..rows * inner)
            .map(|index| ((index * 17 % 47) as f32 - 23.0) / 41.0)
            .collect::<Vec<_>>();
        let row_major_weight = (0..columns * inner)
            .map(|index| ((index * 11 % 53) as f32 - 26.0) / 97.0)
            .collect::<Vec<_>>();
        let bias = (0..columns)
            .map(|column| ((column * 7 % 37) as f32 - 18.0) / 89.0)
            .collect::<Vec<_>>();
        let mut packed_weight = Vec::with_capacity(row_major_weight.len());
        for column_start in (0..columns).step_by(16) {
            let block_columns = (columns - column_start).min(16);
            for index in 0..inner {
                for column in 0..block_columns {
                    packed_weight.push(row_major_weight[(column_start + column) * inner + index]);
                }
            }
        }

        let mut expected = vec![0.0; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = bias[column];
                for index in 0..inner {
                    sum = input[row * inner + index]
                        .mul_add(row_major_weight[column * inner + index], sum);
                }
                expected[row * columns + column] = sum;
            }
            softmax_in_place(&mut expected[row * columns..(row + 1) * columns]);
        }

        let mut actual = vec![0.0; expected.len()];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread test pool");
        pool.install(|| {
            linear_right_transposed(
                &mut actual,
                &input,
                &packed_weight,
                rows,
                inner,
                columns,
                16,
                Some(&bias),
                None,
                true,
            );
        });
        let maximum_error = expected
            .iter()
            .zip(&actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 2e-7, "maximum error: {maximum_error}");
        for row in actual.chunks_exact(columns) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 2e-5);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn direct_spatial_micro_panels_match_s1_s2_padding_and_column_tails() {
        if !has_avx2_fma() {
            return;
        }

        let cases = [
            (5usize, 16usize, [1, 1], [1, 1, 1, 1], 16usize),
            (7, 15, [2, 2], [1, 1, 1, 1], 8),
            (6, 14, [1, 1], [1, 1, 1, 1], 14),
        ];
        let input_channels = 3;
        let output_channels = 16;
        let kernel_height = 3;
        let kernel_width = 3;
        let patch_size = input_channels * kernel_height * kernel_width;
        let dense_weight = (0..output_channels * patch_size)
            .map(|index| ((index * 17 % 47) as f32 - 23.0) / 29.0)
            .collect::<Vec<_>>();
        let bias = (0..output_channels)
            .map(|channel| (channel as f32 - 7.0) / 19.0)
            .collect::<Vec<_>>();
        let mut packed_weight = Vec::with_capacity(dense_weight.len());
        for row_start in (0..output_channels).step_by(6) {
            let block_rows = (output_channels - row_start).min(6);
            for index in 0..patch_size {
                for row in 0..block_rows {
                    packed_weight.push(dense_weight[(row_start + row) * patch_size + index]);
                }
            }
        }
        assert!(spatial_panel_working_set_fits(
            packed_weight.len(),
            patch_size,
            output_channels
        ));

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread test pool");
        for (input_height, input_width, strides, pads, expected_width) in cases {
            let output_height = (input_height + pads[0] + pads[2] - kernel_height) / strides[0] + 1;
            let output_width = (input_width + pads[1] + pads[3] - kernel_width) / strides[1] + 1;
            assert_eq!(output_width, expected_width);
            let input = (0..input_channels * input_height * input_width)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 23.0)
                .collect::<Vec<_>>();
            let output_plane = output_height * output_width;
            let mut expected = vec![0.0; output_channels * output_plane];
            for output_channel in 0..output_channels {
                for output_y in 0..output_height {
                    for output_x in 0..output_width {
                        let mut sum = bias[output_channel];
                        for input_channel in 0..input_channels {
                            for kernel_y in 0..kernel_height {
                                let padded_y = output_y * strides[0] + kernel_y;
                                if padded_y < pads[0] || padded_y - pads[0] >= input_height {
                                    continue;
                                }
                                let input_y = padded_y - pads[0];
                                for kernel_x in 0..kernel_width {
                                    let padded_x = output_x * strides[1] + kernel_x;
                                    if padded_x < pads[1] || padded_x - pads[1] >= input_width {
                                        continue;
                                    }
                                    let input_x = padded_x - pads[1];
                                    let input_value =
                                        input[input_channel * input_height * input_width
                                            + input_y * input_width
                                            + input_x];
                                    let patch_index = (input_channel * kernel_height + kernel_y)
                                        * kernel_width
                                        + kernel_x;
                                    sum = input_value.mul_add(
                                        dense_weight[output_channel * patch_size + patch_index],
                                        sum,
                                    );
                                }
                            }
                        }
                        expected
                            [output_channel * output_plane + output_y * output_width + output_x] =
                            UnaryOperation::Relu.apply(sum);
                    }
                }
            }

            let mut actual = vec![0.0; expected.len()];
            pool.install(|| {
                spatial_conv2d_direct(
                    &mut actual,
                    &input,
                    &packed_weight,
                    Some(&bias),
                    input_channels,
                    input_height,
                    input_width,
                    output_channels,
                    output_height,
                    output_width,
                    kernel_height,
                    kernel_width,
                    strides,
                    pads,
                    Some(UnaryOperation::Relu),
                );
            });
            let maximum_error = expected
                .iter()
                .zip(&actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0f32, f32::max);
            assert!(
                maximum_error < 3e-5,
                "shape {input_height}x{input_width}, stride {strides:?}, output width {output_width}, maximum error {maximum_error}"
            );
        }
    }

    #[test]
    fn exact_sparse_gemm_handles_dynamic_column_tails() {
        let rows = 8;
        let inner = 7;
        let row_offsets = [0, 3, 5];
        let indices = [0, 2, 6, 1, 5];
        let weights = [
            1.0, -0.5, 0.25, 2.0, 0.75, 1.5, -1.0, 0.5, -2.0, 0.125, 0.375, 1.25, 0.5, -0.75, 2.0,
            1.0, -1.5, 0.25, 0.625, -0.125,
        ];
        let bias = [0.5, -0.25, 1.0, -1.0, 0.75, 0.0, -0.5, 0.25];

        for columns in [1, 15, 16, 17, 31, 32, 33] {
            let right = (0..inner * columns)
                .map(|index| ((index * 13 % 37) as f32 - 18.0) / 11.0)
                .collect::<Vec<_>>();
            let mut actual = vec![0.0; rows * columns];
            gemm_sparse_packed_left(
                &mut actual,
                &right,
                rows,
                inner,
                columns,
                Some(&bias),
                &row_offsets,
                &indices,
                &weights,
                None,
            );

            let mut expected = vec![0.0; rows * columns];
            for block in 0..2 {
                for row in 0..4 {
                    for column in 0..columns {
                        let output_row = block * 4 + row;
                        let mut sum = bias[output_row];
                        for entry in row_offsets[block]..row_offsets[block + 1] {
                            sum = weights[entry * 4 + row]
                                .mul_add(right[indices[entry] as usize * columns + column], sum);
                        }
                        expected[output_row * columns + column] = sum;
                    }
                }
            }
            assert_eq!(actual, expected, "column count {columns}");
        }
    }

    #[test]
    fn depthwise_same_matches_scalar_reference() {
        for kernel in [3, 5, 7, 9] {
            for (height, width) in [
                (2, 3),
                (9, 7),
                (9, 8),
                (9, 9),
                (9, 15),
                (9, 16),
                (9, 17),
                (9, 31),
                (9, 32),
                (9, 33),
                (9, 37),
                (11, 41),
            ] {
                let input = (0..height * width)
                    .map(|index| ((index * 17 % 43) as f32 - 21.0) / 13.0)
                    .collect::<Vec<_>>();
                let weights = (0..kernel * kernel)
                    .map(|index| ((index * 11 % 31) as f32 - 15.0) / 19.0)
                    .collect::<Vec<_>>();
                let bias = -0.375;
                let padding = kernel / 2;
                let mut expected = vec![0.0; input.len()];
                for y in 0..height {
                    for x in 0..width {
                        let mut sum = bias;
                        for kernel_y in 0..kernel {
                            let padded_y = y + kernel_y;
                            if padded_y < padding || padded_y - padding >= height {
                                continue;
                            }
                            for kernel_x in 0..kernel {
                                let padded_x = x + kernel_x;
                                if padded_x < padding || padded_x - padding >= width {
                                    continue;
                                }
                                sum = input[(padded_y - padding) * width + padded_x - padding]
                                    .mul_add(weights[kernel_y * kernel + kernel_x], sum);
                            }
                        }
                        expected[y * width + x] = sum;
                    }
                }

                let mut actual = vec![0.0; input.len()];
                depthwise_conv2d_same(&mut actual, &input, &weights, height, width, kernel, bias);
                let maximum_error = expected
                    .iter()
                    .zip(&actual)
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    maximum_error < 2e-5,
                    "kernel={kernel}, shape={height}x{width}, maximum error={maximum_error}"
                );
            }
        }
    }

    #[test]
    fn depthwise_stride2_matches_scalar_reference() {
        for (height, width) in [
            (2usize, 3usize),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 37),
            (10, 38),
            (47, 92),
        ] {
            let input = (0..height * width)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 13.0)
                .collect::<Vec<_>>();
            let weights = (0..9)
                .map(|index| ((index * 11 % 31) as f32 - 15.0) / 19.0)
                .collect::<Vec<_>>();
            let bias = -0.375;
            let output_height = height.div_ceil(2);
            let output_width = width.div_ceil(2);
            let mut expected = vec![0.0; output_height * output_width];
            for output_y in 0..output_height {
                for output_x in 0..output_width {
                    let mut sum = bias;
                    for kernel_y in 0..3 {
                        let input_y = output_y * 2 + kernel_y;
                        if input_y == 0 || input_y > height {
                            continue;
                        }
                        for kernel_x in 0..3 {
                            let input_x = output_x * 2 + kernel_x;
                            if input_x == 0 || input_x > width {
                                continue;
                            }
                            sum = input[(input_y - 1) * width + input_x - 1]
                                .mul_add(weights[kernel_y * 3 + kernel_x], sum);
                        }
                    }
                    expected[output_y * output_width + output_x] = sum;
                }
            }
            let mut actual = vec![0.0; expected.len()];
            depthwise_conv2d_same_3x3_stride2(&mut actual, &input, &weights, height, width, bias);
            assert_eq!(actual, expected, "shape={height}x{width}");
        }
    }

    #[test]
    fn softmax_is_normalized_and_ordered() {
        let input = [-3.0, 0.5, 2.0, -0.25, 1.0, 0.0, -1.0];
        let maximum = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut expected = input.map(|value| (value - maximum).exp());
        let sum = expected.iter().sum::<f32>();
        expected.iter_mut().for_each(|value| *value /= sum);
        let mut values = input;
        softmax_in_place(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 2e-6);
        assert!(
            values
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).abs() < 2e-6)
        );
        assert_eq!(
            values
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index),
            Some(2)
        );
    }

    #[test]
    fn fused_bias_softmax_matches_separate_operations() {
        let mut expected = (0..37)
            .map(|index| ((index * 17 % 43) as f32 - 21.0) / 13.0)
            .collect::<Vec<_>>();
        let bias = (0..37)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 19.0)
            .collect::<Vec<_>>();
        let mut actual = expected.clone();
        add_in_place(&mut expected, &bias);
        softmax_in_place(&mut expected);
        bias_softmax_in_place(&mut actual, &bias);
        assert_eq!(actual, expected);
    }

    #[test]
    fn fused_residual_mul_preserves_two_step_rounding() {
        let mut values = (0..37)
            .map(|index| (index as f32 - 19.0) / 7.0)
            .collect::<Vec<_>>();
        let mut expected = values.clone();
        for value in &mut expected {
            let original = *value;
            let scaled = original.mul_add(0.375, 0.0);
            *value = scaled.mul_add(1.0, original);
        }
        residual_mul_in_place(&mut values, 0.375);
        assert_eq!(values, expected);
    }

    #[test]
    fn relu_and_max_pool_ignore_a_single_nan() {
        let mut values = vec![-1.0; 17];
        values[0] = f32::NAN;
        unary_in_place(&mut values, UnaryOperation::Relu);
        assert_eq!(values[0], 0.0);

        let mut input = vec![1.0; 17];
        input[0] = f32::NAN;
        input[1] = 2.0;
        let mut output = vec![0.0; input.len()];
        max_pool_2x2_same_upper(&mut output, &input, 17);
        assert_eq!(output[0], 2.0);
    }
}
