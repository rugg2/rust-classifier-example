use linfa::prelude::*;
mod load;
use load::load_data;
mod plotting;
use plotting::plot_data;
mod classifier;
use classifier::train_and_test_classifier;

// getting large inspirations from the following article, as a starting point
// https://blog.logrocket.com/machine-learning-in-rust-using-linfa/
// no efficient or rigorous in any way: the purpose here is only to play with rust CSV IO and LINFA crate

fn main() {
    println!("Hello, world! Trying out linfa::Dataset");

    let train = load_data("data/train.csv");
    let test = load_data("data/test.csv");

    let features = train.nfeatures();
    let targets = train.ntargets();

    println!(
        "training with {} samples, testing with {} samples, {} features and {} target",
        train.nsamples(),
        test.nsamples(),
        features,
        targets
    );

    println!("plotting data...");
    plot_data(&train);

    // train and test model model
    train_and_test_classifier(&train, &test);

    println!("end of main, our main end");
}
