use std::collections::HashMap;
use tch::{Tensor, Kind};

trait DocumentEncoder {

    // Encode a document or a set of documents into a vector of floats
    fn encode(&self, texts: &Vec<String>, kwargs: HashMap<&str, &str>) -> Vec<f64>;

    // Mean pooling of the last hidden states of the model
    fn mean_pooling(last_hidden_state: &Tensor, attention_mask: &Tensor) -> Tensor {
        let token_embeddings = last_hidden_state.copy();
        let input_mask_expanded = attention_mask.unsqueeze(-1).f_expand_as(&token_embeddings).to_kind(Kind::Float);
        let sum_embeddings = (token_embeddings * &input_mask_expanded).sum_dim_intlist(&[1], false, Kind::Float);
        let sum_mask = input_mask_expanded.sum_dim_intlist(&[1], false, Kind::Float).clamp_min(1e-9);
        return &sum_embeddings / &sum_mask
    }
}

trait QueryEncoder {
    // Encode a query into a vector of floats
    fn encode(&self, query: &str, kwargs: HashMap<&str, &str>) -> Vec<f64>;
}

struct AutoDocumentEncoder {
    model_name: String,
    tokenizer_name: String,
    device: String,
    pooling: String,
    l2_norm: bool,
}

impl DocumentEncoder for AutoDocumentEncoder {

}