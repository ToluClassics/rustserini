extern crate serde_json;
use anyhow::{Error, Result};
use candle_core::Tensor;
use std::collections::HashMap;

/// A base trait for document encoders
pub trait DocumentEncoder {
    // instantiating a new DocumentEncoder instance
    fn new(
        model_name: &str,
        lowercase: bool,
        strip_accents: bool,
    ) -> Self;

    // Encode a document or a set of documents into a vector of floats
    fn encode(
        &self,
        texts: &Vec<String>,
        titles: &Vec<String>,
        pooler_type: &str,
    ) -> Result<Tensor, Error>;
}

pub trait RepresentationWriter {
    // Write a representation to a file
    fn write(
        &mut self,
        batch_info: &HashMap<&str, Vec<String>>,
        embedding: &mut Vec<f32>,
    ) -> Result<(), anyhow::Error>;

    // Create a new instance of a RepresentationWriter
    fn new(path: &str, dimension: u32) -> Self;

    // Open File
    fn open_file(&mut self) -> Result<(), anyhow::Error>;

    // Save Index to file
    fn save_index(&mut self) -> Result<(), anyhow::Error>;

    // Initialize Index
    fn init_index(&mut self, dim: u32, index_type: &str);

    // Save Docids to file
    fn save_docids(&mut self) -> Result<(), anyhow::Error>;
}
