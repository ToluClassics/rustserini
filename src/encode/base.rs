use std::collections::HashMap;

pub trait DocumentEncoder {
    // instantiating a new DocumentEncoder instance
    fn new(model_name: &str, tokenizer_name: Option<&str>, pooling: &str, l2_norm: bool) -> Self;

    // Encode a document or a set of documents into a vector of floats
    fn encode(
        &self,
        texts: &Vec<String>,
        titles: &Vec<String>,
        kwargs: HashMap<&str, &str>,
    ) -> Vec<f32>;
}

pub trait QueryEncoder {
    // Encode a query into a vector of floats
    fn encode(&self, query: &str, kwargs: HashMap<&str, &str>) -> Vec<f64>;
}
