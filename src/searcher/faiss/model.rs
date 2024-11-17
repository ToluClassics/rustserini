use crate::encode::auto::{
    build_roberta_model_and_tokenizer, mean_pooling, Model
};

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use anyhow::{Error as E, Result};


pub enum QueryType {
    Query { query: String },
    Queries { query: Vec<String> },
}

/// A base trait for query encoders/// A base trait for document encoders
pub trait QueryEncoder {
    // instantiating a new DocumentEncoder instance
    fn new(
        model_name: &str,
        revision: &str,
    ) -> Self;

    // Encode a document or a set of documents into a vector of floats
    fn encode(&self, query: QueryType, pooler_type: &str) -> Result<Tensor, E>;
}

pub struct AutoQueryEncoder {
    model: Model,
    tokenizer: Tokenizer,
    device: Device
}

impl QueryEncoder for AutoQueryEncoder {
    fn new(
        model_name: &str,
        revision: &str,
    ) -> Self {
        let device = Device::Cpu;
        let (model, tokenizer) = build_roberta_model_and_tokenizer(model_name, false, "BertModel", revision).unwrap();
        Self { model, tokenizer, device }
    }

    fn encode(&self, queries: QueryType, pooler_type: &str) -> Result<Tensor, E> {
        let texts = match queries {
            QueryType::Query { query } => vec![query],
            QueryType::Queries { query } => query,
        };


        let tokens = self.tokenizer
            .encode_batch(texts, true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask  = Tensor::stack(&attention_mask, 0)?;

        let hidden_state: Tensor = match &self.model {
            Model::BertModel {model} => {
                let hidden_state: Tensor = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
                hidden_state
            },
            Model::BertForMaskedLM {model} => {
                let hidden_state: Tensor = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
                hidden_state
            },            
        };

        let embeddings: Tensor = if pooler_type == "mean" {
            mean_pooling(hidden_state, false)?
            
        } else if pooler_type == "cls" {
            
            let (_n_sentence, _n_tokens, _hidden_size) = hidden_state.dims3()?;

            let mut out_embeding = vec![];
            for i in 0.._n_sentence{

                let embeding = hidden_state.get(i)?.get(0)?;
                out_embeding.push(embeding);
            }
            Tensor::stack(&out_embeding, 0)?
        } else {
            panic!("pooler_type must be either mean or cls");
        };

        Ok(embeddings)
    }
}
