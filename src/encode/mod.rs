pub mod auto;
pub mod base;
pub mod vector_writer;

// Path: src/encode/auto.rs

pub use auto::AutoDocumentEncoder;
pub use base::DocumentEncoder;
pub use vector_writer::{JsonlCollectionIterator, JsonlRepresentationWriter};
