extern crate serde_json;
use serde_json::Value;
use std::collections::HashMap;

pub trait DocumentEncoder {
    // instantiating a new DocumentEncoder instance
    fn new(model_name: &str, tokenizer_name: Option<&str>) -> Self;

    // Encode a document or a set of documents into a vector of floats
    fn encode(&self, texts: &Vec<String>, titles: &Vec<String>, pooler_type: &str) -> Vec<f32>;
}

pub trait QueryEncoder {
    // Encode a query into a vector of floats
    fn encode(&self, query: &str) -> Vec<f64>;
}

pub trait RepresentationWriter {
    // Write a representation to a file
    fn write(&mut self, batch_info: &HashMap<&str, Value>);

    // Create a new instance of a RepresentationWriter
    fn new(path: &str) -> Self;

    // Open File
    fn open_file(&mut self);

    fn save_index(&mut self);

    fn init_index(&mut self, dim: u32, index_type: &str);
}
