//! CPU execution of the detector preprocessing plan: resize + BGR +
//! mean/std normalize, producing an NCHW f32 tensor.

mod kernels;

use crate::postprocess::{DetectorInputPlan, Point};
use rayon::prelude::*;

use kernels::{Kernel, Normalization, RowPlan};

const DETECTOR_MEAN_BGR: [f32; 3] = [0.485, 0.456, 0.406];
const DETECTOR_STD_BGR: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Debug)]
pub(crate) struct PreparedInput {
    pub(crate) data: Vec<f32>,
    pub(crate) batch: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

impl PreparedInput {
    pub(crate) fn shape(&self) -> [usize; 4] {
        [self.batch, 3, self.height, self.width]
    }
}

pub(crate) fn prepare_detector(image: &crate::RgbImage, plan: DetectorInputPlan) -> PreparedInput {
    normalized_bgr(
        image,
        plan.corners(),
        plan.input_width(),
        plan.input_height(),
        plan.input_width(),
        &DETECTOR_MEAN_BGR,
        &DETECTOR_STD_BGR,
    )
}

fn normalized_bgr(
    image: &crate::RgbImage,
    corners: [Point; 4],
    canvas_width: usize,
    canvas_height: usize,
    content_width: usize,
    mean: &[f32; 3],
    standard_deviation: &[f32; 3],
) -> PreparedInput {
    let plane_len = canvas_height * canvas_width;
    let mut data = vec![0.0; 3 * plane_len];
    let (blue, green_red) = data.split_at_mut(plane_len);
    let (green, red) = green_red.split_at_mut(plane_len);
    let kernel = Kernel::detect();
    let normalization = Normalization::new(*mean, *standard_deviation);
    let corners = corners.map(|point| [point.0, point.1]);

    blue.par_chunks_mut(canvas_width)
        .zip(green.par_chunks_mut(canvas_width))
        .zip(red.par_chunks_mut(canvas_width))
        .enumerate()
        .for_each(|(y, ((blue, green), red))| {
            kernel.preprocess_row(
                image.pixels(),
                RowPlan {
                    source_width: image.width() as usize,
                    source_height: image.height() as usize,
                    corners,
                    destination_y: y,
                    destination_height: canvas_height,
                    content_width,
                    normalization,
                },
                blue,
                green,
                red,
            );
        });

    PreparedInput {
        data,
        batch: 1,
        height: canvas_height,
        width: canvas_width,
    }
}
