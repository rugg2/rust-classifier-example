
use plotlib::{
    grid::Grid,
    page::Page,
    repr::Plot,
    style::{PointMarker, PointStyle},
    view::{ContinuousView, View},
};

use linfa::DatasetBase;
use ndarray::ArrayBase;
use ndarray::OwnedRepr;
use ndarray::Dim;


pub fn plot_data(
    train: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >,
) {
    let mut positive = vec![];
    let mut negative = vec![];

    let records = train.records().clone().into_raw_vec();
    let features: Vec<&[f64]> = records.chunks(2).collect();
    let targets = train.targets().clone().into_raw_vec();
    for i in 0..features.len() {
        let feature = features.get(i).expect("feature exists");
        if let Some(&"accepted") = targets.get(i) {
            positive.push((feature[0], feature[1]));
        } else {
            negative.push((feature[0], feature[1]));
        }
    }

    let plot_positive = Plot::new(positive)
        .point_style(
            PointStyle::new()
                .size(2.0)
                .marker(PointMarker::Square)
                .colour("#00ff00"),
        )
        .legend("Exam Results".to_string());

    let plot_negative = Plot::new(negative).point_style(
        PointStyle::new()
            .size(2.0)
            .marker(PointMarker::Circle)
            .colour("#ff0000"),
    );

    let grid = Grid::new(0, 0);

    let mut image = ContinuousView::new()
        .add(plot_positive)
        .add(plot_negative)
        .x_range(0.0, 120.0)
        .y_range(0.0, 120.0)
        .x_label("Test 1")
        .y_label("Test 2");

    image.add_grid(grid);

    Page::single(&image)
        .save("plot.svg")
        .expect("can generate svg for plot");
}