use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use ndarray::s;
use ndarray::Array2;
use linfa::Dataset;

pub fn load_data(path: &str) -> Dataset<f64, &'static str>{
    let array = csv_to_array2(path);
    
    // from ndarray to linfa::Dataset
    let (data, targets) = (
        array.slice(s![..,0..2]).to_owned(),
        array.column(2).to_owned(),
    );

    let feature_names = vec!["test 1", "test 2"];

    Dataset::new(data, targets)
        .map_targets(|x| {
            if *x as usize==1 {
                "accepted"
            } else {
                "denied"
            }
        })
        .with_feature_names(feature_names)
}

fn csv_to_array2(path: &str) -> Array2<f64>{
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_path(path)
        .expect("can create reader");
    
    // use ndarray-csv library to load csv into Array2
    let array: Array2<f64> = reader
        .deserialize_array2_dynamic()
        .expect("can deserialise array");

    array
}