use std::ffi::c_int;

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transpose_a: c_int,
        transpose_b: c_int,
        rows: c_int,
        columns: c_int,
        inner: c_int,
        alpha: f32,
        left: *const f32,
        left_stride: c_int,
        right: *const f32,
        right_stride: c_int,
        beta: f32,
        output: *mut f32,
        output_stride: c_int,
    );
}

pub(super) fn sgemm(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    assert_eq!(rows.checked_mul(columns), Some(output.len()));
    assert_eq!(rows.checked_mul(inner), Some(left.len()));
    assert_eq!(inner.checked_mul(columns), Some(right.len()));
    let rows = c_int::try_from(rows).expect("SGEMM row count exceeds c_int");
    let inner = c_int::try_from(inner).expect("SGEMM inner dimension exceeds c_int");
    let columns = c_int::try_from(columns).expect("SGEMM column count exceeds c_int");

    // SAFETY: Slice lengths are validated against the row-major matrix
    // dimensions, all dimensions and strides fit c_int, and output is uniquely
    // borrowed for the duration of the call.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows,
            columns,
            inner,
            1.0,
            left.as_ptr(),
            inner,
            right.as_ptr(),
            columns,
            0.0,
            output.as_mut_ptr(),
            columns,
        );
    }
}

pub(super) fn sgemm_right_transposed(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    assert_eq!(rows.checked_mul(columns), Some(output.len()));
    assert_eq!(rows.checked_mul(inner), Some(left.len()));
    assert_eq!(columns.checked_mul(inner), Some(right.len()));
    let rows = c_int::try_from(rows).expect("SGEMM row count exceeds c_int");
    let inner = c_int::try_from(inner).expect("SGEMM inner dimension exceeds c_int");
    let columns = c_int::try_from(columns).expect("SGEMM column count exceeds c_int");

    // SAFETY: The right matrix is stored as [columns, inner]. CBLAS interprets
    // it as its transpose without materializing a second copy.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            rows,
            columns,
            inner,
            1.0,
            left.as_ptr(),
            inner,
            right.as_ptr(),
            inner,
            0.0,
            output.as_mut_ptr(),
            columns,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_matches_row_major_matrix_product() {
        let left = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
        let right = [2.0, -1.0, 0.0, 3.0, 1.0, 2.0];
        let mut output = [0.0; 4];
        sgemm(&mut output, &left, &right, 2, 3, 2);
        assert_eq!(output, [5.0, 11.0, 2.0, 10.5]);
    }

    #[test]
    fn sgemm_accepts_a_transposed_right_matrix() {
        let left = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
        let right = [2.0, 0.0, 1.0, -1.0, 3.0, 2.0];
        let mut output = [0.0; 4];
        sgemm_right_transposed(&mut output, &left, &right, 2, 3, 2);
        assert_eq!(output, [5.0, 11.0, 2.0, 10.5]);
    }
}
