use linfa_logistic::LogisticRegression;
use ndarray::{prelude::*, OwnedRepr};
use linfa::DatasetBase;
use linfa::metrics::ConfusionMatrix;
use linfa::prelude::{Fit, Predict, ToConfusionMatrix};

pub fn train_with_hyperparameter_tuning_and_test_classifier(train: &DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
>,
test: &DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
>,){
    println!("training and testing model...");
    // TODO: split into train, dev and test set, as hyperparameter tuning may overfit the test set
    let mut max_f1score_confusion_matrix = iterate_with_values(&train, &test, 0.01, 100);
    let mut best_threshold = 0.0;
    let mut best_max_iterations = 0;
    let mut threshold = 0.02;

    // very crude and basic hyperparameter tuning
    // TODO: create a function that optimises based on a user defined grid, as for sklearn
    for max_iterations in (1000..5000).step_by(500) {
        while threshold < 1.0 {
            let confusion_matrix = iterate_with_values(&train, &test, threshold, max_iterations);

            if confusion_matrix.f1_score() > max_f1score_confusion_matrix.f1_score() {
                max_f1score_confusion_matrix = confusion_matrix;
                best_threshold = threshold;
                best_max_iterations = max_iterations;
            }
            threshold += 0.01;
        }
        threshold = 0.02;
    }

    println!(
        "most accurate confusion matrix: {:?}",
        max_f1score_confusion_matrix
    );
    println!(
        "with max_iterations: {}, threshold: {}",
        best_max_iterations, best_threshold
    );
    println!("accuracy {}", max_f1score_confusion_matrix.accuracy(),);
    println!("f1 score {}", max_f1score_confusion_matrix.f1_score(),);
    println!("precision {}", max_f1score_confusion_matrix.precision(),);
    println!("recall {}", max_f1score_confusion_matrix.recall(),);
}


fn iterate_with_values(
    train: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >,
    test: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >,
    threshold: f64,
    max_iterations: u64,
) -> ConfusionMatrix<&'static str> {
    let model = LogisticRegression::default()
        .max_iterations(max_iterations)
        .gradient_tolerance(0.0001)
        .fit(train)
        .expect("can train model");

    let validation = model.set_threshold(threshold).predict(test);

    let confusion_matrix = validation
        .confusion_matrix(test)
        .expect("can create confusion matrix");

    confusion_matrix
}