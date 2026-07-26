//! Fused architecture-specific image preprocessing kernels.

#[derive(Clone, Copy)]
pub(super) struct Normalization {
    scale: [f32; 3],
    bias: [f32; 3],
}

impl Normalization {
    pub(super) fn new(mean: [f32; 3], standard_deviation: [f32; 3]) -> Self {
        let mut scale = [0.0; 3];
        let mut bias = [0.0; 3];
        for channel in 0..3 {
            scale[channel] = 1.0 / (255.0 * standard_deviation[channel]);
            bias[channel] = -mean[channel] / standard_deviation[channel];
        }
        Self { scale, bias }
    }
}

#[derive(Clone, Copy)]
pub(super) struct RowPlan {
    pub(super) source_width: usize,
    pub(super) source_height: usize,
    pub(super) corners: [[f32; 2]; 4],
    pub(super) destination_y: usize,
    pub(super) destination_height: usize,
    pub(super) content_width: usize,
    pub(super) normalization: Normalization,
}

#[derive(Clone, Copy)]
pub(super) enum Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse2,
}

impl Kernel {
    pub(super) fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Neon;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            return Self::Sse2;
        }
        #[allow(unreachable_code)]
        Self::Scalar
    }

    pub(super) fn preprocess_row(
        self,
        pixels: &[u8],
        plan: RowPlan,
        blue: &mut [f32],
        green: &mut [f32],
        red: &mut [f32],
    ) {
        assert!(plan.source_width > 0 && plan.source_height > 0);
        assert!(blue.len() >= plan.content_width);
        assert!(green.len() >= plan.content_width);
        assert!(red.len() >= plan.content_width);
        match self {
            Self::Scalar => scalar(pixels, plan, blue, green, red, 0),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => {
                // AArch64 requires Advanced SIMD, so this target feature is
                // available on every supported AArch64 CPU.
                unsafe { neon(pixels, plan, blue, green, red) };
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                // `detect` checks AVX2 before constructing this variant.
                unsafe { avx2(pixels, plan, blue, green, red) };
            }
            #[cfg(target_arch = "x86_64")]
            Self::Sse2 => {
                // SSE2 is part of the x86_64 architecture baseline.
                unsafe { sse2(pixels, plan, blue, green, red) };
            }
        }
    }
}

fn scalar(
    pixels: &[u8],
    plan: RowPlan,
    blue: &mut [f32],
    green: &mut [f32],
    red: &mut [f32],
    start_x: usize,
) {
    let v = (plan.destination_y as f32 + 0.5) / plan.destination_height as f32;
    let one_minus_v = 1.0 - v;
    let max_x = (plan.source_width - 1) as f32;
    let max_y = (plan.source_height - 1) as f32;
    for x in start_x..plan.content_width {
        let u = (x as f32 + 0.5) / plan.content_width as f32;
        let one_minus_u = 1.0 - u;
        let source_x = ((plan.corners[0][0] * one_minus_u + plan.corners[1][0] * u) * one_minus_v
            + (plan.corners[3][0] * one_minus_u + plan.corners[2][0] * u) * v
            - 0.5)
            .clamp(0.0, max_x);
        let source_y = ((plan.corners[0][1] * one_minus_u + plan.corners[1][1] * u) * one_minus_v
            + (plan.corners[3][1] * one_minus_u + plan.corners[2][1] * u) * v
            - 0.5)
            .clamp(0.0, max_y);
        let x0 = source_x.floor() as usize;
        let y0 = source_y.floor() as usize;
        let x1 = (x0 + 1).min(plan.source_width - 1);
        let y1 = (y0 + 1).min(plan.source_height - 1);
        let tx = source_x - x0 as f32;
        let ty = source_y - y0 as f32;
        let offsets = sample_offsets(plan.source_width, x0, y0, x1, y1);
        let rgb = [
            interpolate_channel(pixels, offsets, tx, ty, 0),
            interpolate_channel(pixels, offsets, tx, ty, 1),
            interpolate_channel(pixels, offsets, tx, ty, 2),
        ];
        blue[x] = normalize(rgb[2], plan.normalization, 0);
        green[x] = normalize(rgb[1], plan.normalization, 1);
        red[x] = normalize(rgb[0], plan.normalization, 2);
    }
}

#[inline(always)]
fn sample_offsets(source_width: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> [usize; 4] {
    [
        (y0 * source_width + x0) * 3,
        (y0 * source_width + x1) * 3,
        (y1 * source_width + x0) * 3,
        (y1 * source_width + x1) * 3,
    ]
}

#[inline(always)]
fn interpolate_channel(
    pixels: &[u8],
    offsets: [usize; 4],
    tx: f32,
    ty: f32,
    channel: usize,
) -> f32 {
    let top_left = f32::from(pixels[offsets[0] + channel]);
    let top_right = f32::from(pixels[offsets[1] + channel]);
    let bottom_left = f32::from(pixels[offsets[2] + channel]);
    let bottom_right = f32::from(pixels[offsets[3] + channel]);
    let top = top_left * (1.0 - tx) + top_right * tx;
    let bottom = bottom_left * (1.0 - tx) + bottom_right * tx;
    (top * (1.0 - ty) + bottom * ty).round().clamp(0.0, 255.0)
}

#[inline(always)]
fn normalize(value: f32, normalization: Normalization, channel: usize) -> f32 {
    value * normalization.scale[channel] + normalization.bias[channel]
}

struct Samples<const LANES: usize> {
    top_left: [[f32; LANES]; 3],
    top_right: [[f32; LANES]; 3],
    bottom_left: [[f32; LANES]; 3],
    bottom_right: [[f32; LANES]; 3],
}

fn gather<const LANES: usize>(
    pixels: &[u8],
    source_width: usize,
    source_height: usize,
    x0: [i32; LANES],
    y0: [i32; LANES],
) -> Samples<LANES> {
    let mut samples = Samples {
        top_left: [[0.0; LANES]; 3],
        top_right: [[0.0; LANES]; 3],
        bottom_left: [[0.0; LANES]; 3],
        bottom_right: [[0.0; LANES]; 3],
    };
    for lane in 0..LANES {
        let x0 = x0[lane] as usize;
        let y0 = y0[lane] as usize;
        debug_assert!(x0 < source_width && y0 < source_height);
        let x1 = (x0 + 1).min(source_width - 1);
        let y1 = (y0 + 1).min(source_height - 1);
        let offsets = sample_offsets(source_width, x0, y0, x1, y1);
        for channel in 0..3 {
            samples.top_left[channel][lane] = f32::from(pixels[offsets[0] + channel]);
            samples.top_right[channel][lane] = f32::from(pixels[offsets[1] + channel]);
            samples.bottom_left[channel][lane] = f32::from(pixels[offsets[2] + channel]);
            samples.bottom_right[channel][lane] = f32::from(pixels[offsets[3] + channel]);
        }
    }
    samples
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2(pixels: &[u8], plan: RowPlan, blue: &mut [f32], green: &mut [f32], red: &mut [f32]) {
    use core::arch::x86_64::*;

    const LANES: usize = 4;
    let v = (plan.destination_y as f32 + 0.5) / plan.destination_height as f32;
    let mut x = 0;
    while x + LANES <= plan.content_width {
        let mut x0 = [0i32; LANES];
        let mut y0 = [0i32; LANES];
        let mut tx_values = [0.0f32; LANES];
        let mut ty_values = [0.0f32; LANES];
        // Vector memory operations only address fixed-size stack arrays.
        unsafe {
            let offsets = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
            let positions = _mm_add_ps(_mm_set1_ps(x as f32 + 0.5), offsets);
            let u = _mm_mul_ps(positions, _mm_set1_ps(1.0 / plan.content_width as f32));
            let one = _mm_set1_ps(1.0);
            let one_minus_u = _mm_sub_ps(one, u);
            let one_minus_v = 1.0 - v;
            let source_x = coordinates_sse2(plan.corners, 0, u, one_minus_u, v, one_minus_v);
            let source_y = coordinates_sse2(plan.corners, 1, u, one_minus_u, v, one_minus_v);
            let source_x = _mm_min_ps(
                _mm_max_ps(source_x, _mm_setzero_ps()),
                _mm_set1_ps((plan.source_width - 1) as f32),
            );
            let source_y = _mm_min_ps(
                _mm_max_ps(source_y, _mm_setzero_ps()),
                _mm_set1_ps((plan.source_height - 1) as f32),
            );
            // Coordinates are non-negative after clamping, so truncation is floor.
            let x0_vector = _mm_cvttps_epi32(source_x);
            let y0_vector = _mm_cvttps_epi32(source_y);
            _mm_storeu_si128(x0.as_mut_ptr().cast(), x0_vector);
            _mm_storeu_si128(y0.as_mut_ptr().cast(), y0_vector);
            _mm_storeu_ps(
                tx_values.as_mut_ptr(),
                _mm_sub_ps(source_x, _mm_cvtepi32_ps(x0_vector)),
            );
            _mm_storeu_ps(
                ty_values.as_mut_ptr(),
                _mm_sub_ps(source_y, _mm_cvtepi32_ps(y0_vector)),
            );
        }

        let samples = gather(pixels, plan.source_width, plan.source_height, x0, y0);
        unsafe {
            interpolate_sse2(
                &samples,
                2,
                &tx_values,
                &ty_values,
                plan.normalization.scale[0],
                plan.normalization.bias[0],
                (&mut blue[x..x + LANES]).try_into().expect("four outputs"),
            );
            interpolate_sse2(
                &samples,
                1,
                &tx_values,
                &ty_values,
                plan.normalization.scale[1],
                plan.normalization.bias[1],
                (&mut green[x..x + LANES]).try_into().expect("four outputs"),
            );
            interpolate_sse2(
                &samples,
                0,
                &tx_values,
                &ty_values,
                plan.normalization.scale[2],
                plan.normalization.bias[2],
                (&mut red[x..x + LANES]).try_into().expect("four outputs"),
            );
        }
        x += LANES;
    }
    scalar(pixels, plan, blue, green, red, x);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn coordinates_sse2(
    corners: [[f32; 2]; 4],
    coordinate: usize,
    u: core::arch::x86_64::__m128,
    one_minus_u: core::arch::x86_64::__m128,
    v: f32,
    one_minus_v: f32,
) -> core::arch::x86_64::__m128 {
    use core::arch::x86_64::*;
    let top = _mm_add_ps(
        _mm_mul_ps(one_minus_u, _mm_set1_ps(corners[0][coordinate])),
        _mm_mul_ps(u, _mm_set1_ps(corners[1][coordinate])),
    );
    let bottom = _mm_add_ps(
        _mm_mul_ps(one_minus_u, _mm_set1_ps(corners[3][coordinate])),
        _mm_mul_ps(u, _mm_set1_ps(corners[2][coordinate])),
    );
    _mm_sub_ps(
        _mm_add_ps(
            _mm_mul_ps(top, _mm_set1_ps(one_minus_v)),
            _mm_mul_ps(bottom, _mm_set1_ps(v)),
        ),
        _mm_set1_ps(0.5),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn interpolate_sse2(
    samples: &Samples<4>,
    channel: usize,
    tx: &[f32; 4],
    ty: &[f32; 4],
    scale: f32,
    bias: f32,
    output: &mut [f32; 4],
) {
    use core::arch::x86_64::*;
    unsafe {
        let one = _mm_set1_ps(1.0);
        let tx = _mm_loadu_ps(tx.as_ptr());
        let ty = _mm_loadu_ps(ty.as_ptr());
        let top = _mm_add_ps(
            _mm_mul_ps(
                _mm_loadu_ps(samples.top_left[channel].as_ptr()),
                _mm_sub_ps(one, tx),
            ),
            _mm_mul_ps(_mm_loadu_ps(samples.top_right[channel].as_ptr()), tx),
        );
        let bottom = _mm_add_ps(
            _mm_mul_ps(
                _mm_loadu_ps(samples.bottom_left[channel].as_ptr()),
                _mm_sub_ps(one, tx),
            ),
            _mm_mul_ps(_mm_loadu_ps(samples.bottom_right[channel].as_ptr()), tx),
        );
        let value = _mm_add_ps(_mm_mul_ps(top, _mm_sub_ps(one, ty)), _mm_mul_ps(bottom, ty));
        // Interpolated RGB values are non-negative, so +0.5 then truncation
        // matches scalar round-to-nearest for this domain without SSE4.1.
        let rounded = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_add_ps(value, _mm_set1_ps(0.5))));
        let rounded = _mm_min_ps(_mm_max_ps(rounded, _mm_setzero_ps()), _mm_set1_ps(255.0));
        _mm_storeu_ps(
            output.as_mut_ptr(),
            _mm_add_ps(_mm_mul_ps(rounded, _mm_set1_ps(scale)), _mm_set1_ps(bias)),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(pixels: &[u8], plan: RowPlan, blue: &mut [f32], green: &mut [f32], red: &mut [f32]) {
    use core::arch::aarch64::*;

    const LANES: usize = 4;
    let v = (plan.destination_y as f32 + 0.5) / plan.destination_height as f32;
    let mut x = 0;
    while x + LANES <= plan.content_width {
        let mut x0 = [0i32; LANES];
        let mut y0 = [0i32; LANES];
        let mut tx_values = [0.0f32; LANES];
        let mut ty_values = [0.0f32; LANES];
        // All vector loads and stores below address fixed-size stack arrays.
        unsafe {
            let offsets = vld1q_f32([0.0, 1.0, 2.0, 3.0].as_ptr());
            let positions = vaddq_f32(vdupq_n_f32(x as f32 + 0.5), offsets);
            let u = vmulq_n_f32(positions, 1.0 / plan.content_width as f32);
            let one = vdupq_n_f32(1.0);
            let one_minus_u = vsubq_f32(one, u);
            let one_minus_v = 1.0 - v;
            let source_x = coordinates_neon(plan.corners, 0, u, one_minus_u, v, one_minus_v);
            let source_y = coordinates_neon(plan.corners, 1, u, one_minus_u, v, one_minus_v);
            let source_x = vminq_f32(
                vmaxq_f32(source_x, vdupq_n_f32(0.0)),
                vdupq_n_f32((plan.source_width - 1) as f32),
            );
            let source_y = vminq_f32(
                vmaxq_f32(source_y, vdupq_n_f32(0.0)),
                vdupq_n_f32((plan.source_height - 1) as f32),
            );
            let x0_vector = vcvtq_s32_f32(vrndmq_f32(source_x));
            let y0_vector = vcvtq_s32_f32(vrndmq_f32(source_y));
            vst1q_s32(x0.as_mut_ptr(), x0_vector);
            vst1q_s32(y0.as_mut_ptr(), y0_vector);
            vst1q_f32(
                tx_values.as_mut_ptr(),
                vsubq_f32(source_x, vcvtq_f32_s32(x0_vector)),
            );
            vst1q_f32(
                ty_values.as_mut_ptr(),
                vsubq_f32(source_y, vcvtq_f32_s32(y0_vector)),
            );
        }

        let samples = gather(pixels, plan.source_width, plan.source_height, x0, y0);
        unsafe {
            interpolate_neon(
                &samples,
                2,
                &tx_values,
                &ty_values,
                plan.normalization.scale[0],
                plan.normalization.bias[0],
                (&mut blue[x..x + LANES]).try_into().expect("four outputs"),
            );
            interpolate_neon(
                &samples,
                1,
                &tx_values,
                &ty_values,
                plan.normalization.scale[1],
                plan.normalization.bias[1],
                (&mut green[x..x + LANES]).try_into().expect("four outputs"),
            );
            interpolate_neon(
                &samples,
                0,
                &tx_values,
                &ty_values,
                plan.normalization.scale[2],
                plan.normalization.bias[2],
                (&mut red[x..x + LANES]).try_into().expect("four outputs"),
            );
        }
        x += LANES;
    }
    scalar(pixels, plan, blue, green, red, x);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn coordinates_neon(
    corners: [[f32; 2]; 4],
    coordinate: usize,
    u: core::arch::aarch64::float32x4_t,
    one_minus_u: core::arch::aarch64::float32x4_t,
    v: f32,
    one_minus_v: f32,
) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    let top = vaddq_f32(
        vmulq_n_f32(one_minus_u, corners[0][coordinate]),
        vmulq_n_f32(u, corners[1][coordinate]),
    );
    let bottom = vaddq_f32(
        vmulq_n_f32(one_minus_u, corners[3][coordinate]),
        vmulq_n_f32(u, corners[2][coordinate]),
    );
    vsubq_f32(
        vaddq_f32(vmulq_n_f32(top, one_minus_v), vmulq_n_f32(bottom, v)),
        vdupq_n_f32(0.5),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn interpolate_neon(
    samples: &Samples<4>,
    channel: usize,
    tx: &[f32; 4],
    ty: &[f32; 4],
    scale: f32,
    bias: f32,
    output: &mut [f32; 4],
) {
    use core::arch::aarch64::*;
    unsafe {
        let one = vdupq_n_f32(1.0);
        let tx = vld1q_f32(tx.as_ptr());
        let ty = vld1q_f32(ty.as_ptr());
        let top = vaddq_f32(
            vmulq_f32(
                vld1q_f32(samples.top_left[channel].as_ptr()),
                vsubq_f32(one, tx),
            ),
            vmulq_f32(vld1q_f32(samples.top_right[channel].as_ptr()), tx),
        );
        let bottom = vaddq_f32(
            vmulq_f32(
                vld1q_f32(samples.bottom_left[channel].as_ptr()),
                vsubq_f32(one, tx),
            ),
            vmulq_f32(vld1q_f32(samples.bottom_right[channel].as_ptr()), tx),
        );
        let value = vaddq_f32(vmulq_f32(top, vsubq_f32(one, ty)), vmulq_f32(bottom, ty));
        let rounded = vrndmq_f32(vaddq_f32(value, vdupq_n_f32(0.5)));
        let rounded = vminq_f32(vmaxq_f32(rounded, vdupq_n_f32(0.0)), vdupq_n_f32(255.0));
        vst1q_f32(
            output.as_mut_ptr(),
            vaddq_f32(vmulq_n_f32(rounded, scale), vdupq_n_f32(bias)),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(pixels: &[u8], plan: RowPlan, blue: &mut [f32], green: &mut [f32], red: &mut [f32]) {
    use core::arch::x86_64::*;

    const LANES: usize = 8;
    let v = (plan.destination_y as f32 + 0.5) / plan.destination_height as f32;
    let mut x = 0;
    while x + LANES <= plan.content_width {
        let mut x0 = [0i32; LANES];
        let mut y0 = [0i32; LANES];
        let mut tx_values = [0.0f32; LANES];
        let mut ty_values = [0.0f32; LANES];
        // All vector loads and stores below address fixed-size stack arrays.
        unsafe {
            let offsets = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
            let positions = _mm256_add_ps(_mm256_set1_ps(x as f32 + 0.5), offsets);
            let u = _mm256_mul_ps(positions, _mm256_set1_ps(1.0 / plan.content_width as f32));
            let one = _mm256_set1_ps(1.0);
            let one_minus_u = _mm256_sub_ps(one, u);
            let one_minus_v = 1.0 - v;
            let source_x = coordinates_avx2(plan.corners, 0, u, one_minus_u, v, one_minus_v);
            let source_y = coordinates_avx2(plan.corners, 1, u, one_minus_u, v, one_minus_v);
            let source_x = _mm256_min_ps(
                _mm256_max_ps(source_x, _mm256_setzero_ps()),
                _mm256_set1_ps((plan.source_width - 1) as f32),
            );
            let source_y = _mm256_min_ps(
                _mm256_max_ps(source_y, _mm256_setzero_ps()),
                _mm256_set1_ps((plan.source_height - 1) as f32),
            );
            let x0_vector = _mm256_cvttps_epi32(_mm256_floor_ps(source_x));
            let y0_vector = _mm256_cvttps_epi32(_mm256_floor_ps(source_y));
            _mm256_storeu_si256(x0.as_mut_ptr().cast(), x0_vector);
            _mm256_storeu_si256(y0.as_mut_ptr().cast(), y0_vector);
            _mm256_storeu_ps(
                tx_values.as_mut_ptr(),
                _mm256_sub_ps(source_x, _mm256_cvtepi32_ps(x0_vector)),
            );
            _mm256_storeu_ps(
                ty_values.as_mut_ptr(),
                _mm256_sub_ps(source_y, _mm256_cvtepi32_ps(y0_vector)),
            );
        }

        let samples = gather(pixels, plan.source_width, plan.source_height, x0, y0);
        unsafe {
            interpolate_avx2(
                &samples,
                2,
                &tx_values,
                &ty_values,
                plan.normalization.scale[0],
                plan.normalization.bias[0],
                (&mut blue[x..x + LANES]).try_into().expect("eight outputs"),
            );
            interpolate_avx2(
                &samples,
                1,
                &tx_values,
                &ty_values,
                plan.normalization.scale[1],
                plan.normalization.bias[1],
                (&mut green[x..x + LANES])
                    .try_into()
                    .expect("eight outputs"),
            );
            interpolate_avx2(
                &samples,
                0,
                &tx_values,
                &ty_values,
                plan.normalization.scale[2],
                plan.normalization.bias[2],
                (&mut red[x..x + LANES]).try_into().expect("eight outputs"),
            );
        }
        x += LANES;
    }
    scalar(pixels, plan, blue, green, red, x);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn coordinates_avx2(
    corners: [[f32; 2]; 4],
    coordinate: usize,
    u: core::arch::x86_64::__m256,
    one_minus_u: core::arch::x86_64::__m256,
    v: f32,
    one_minus_v: f32,
) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let top = _mm256_add_ps(
        _mm256_mul_ps(one_minus_u, _mm256_set1_ps(corners[0][coordinate])),
        _mm256_mul_ps(u, _mm256_set1_ps(corners[1][coordinate])),
    );
    let bottom = _mm256_add_ps(
        _mm256_mul_ps(one_minus_u, _mm256_set1_ps(corners[3][coordinate])),
        _mm256_mul_ps(u, _mm256_set1_ps(corners[2][coordinate])),
    );
    _mm256_sub_ps(
        _mm256_add_ps(
            _mm256_mul_ps(top, _mm256_set1_ps(one_minus_v)),
            _mm256_mul_ps(bottom, _mm256_set1_ps(v)),
        ),
        _mm256_set1_ps(0.5),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interpolate_avx2(
    samples: &Samples<8>,
    channel: usize,
    tx: &[f32; 8],
    ty: &[f32; 8],
    scale: f32,
    bias: f32,
    output: &mut [f32; 8],
) {
    use core::arch::x86_64::*;
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let tx = _mm256_loadu_ps(tx.as_ptr());
        let ty = _mm256_loadu_ps(ty.as_ptr());
        let top = _mm256_add_ps(
            _mm256_mul_ps(
                _mm256_loadu_ps(samples.top_left[channel].as_ptr()),
                _mm256_sub_ps(one, tx),
            ),
            _mm256_mul_ps(_mm256_loadu_ps(samples.top_right[channel].as_ptr()), tx),
        );
        let bottom = _mm256_add_ps(
            _mm256_mul_ps(
                _mm256_loadu_ps(samples.bottom_left[channel].as_ptr()),
                _mm256_sub_ps(one, tx),
            ),
            _mm256_mul_ps(_mm256_loadu_ps(samples.bottom_right[channel].as_ptr()), tx),
        );
        let value = _mm256_add_ps(
            _mm256_mul_ps(top, _mm256_sub_ps(one, ty)),
            _mm256_mul_ps(bottom, ty),
        );
        let rounded = _mm256_floor_ps(_mm256_add_ps(value, _mm256_set1_ps(0.5)));
        let rounded = _mm256_min_ps(
            _mm256_max_ps(rounded, _mm256_setzero_ps()),
            _mm256_set1_ps(255.0),
        );
        _mm256_storeu_ps(
            output.as_mut_ptr(),
            _mm256_add_ps(
                _mm256_mul_ps(rounded, _mm256_set1_ps(scale)),
                _mm256_set1_ps(bias),
            ),
        );
    }
}
