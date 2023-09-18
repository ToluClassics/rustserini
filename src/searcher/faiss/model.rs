use crate::encode::auto::{
    fetch_bert_style_config, fetch_bert_style_model, fetch_bert_style_vocab, mean_pooling,
};
use rust_bert::bert::BertForSentenceEmbeddings;
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};

use tch::{no_grad, Device, Tensor};

/// A base trait for query encoders/// A base trait for document encoders
pub trait QueryEncoder {
    // instantiating a new DocumentEncoder instance
    fn new(
        model_name: &str,
        tokenizer_name: Option<&str>,
        lowercase: bool,
        strip_accents: bool,
    ) -> Self;

    // Encode a document or a set of documents into a vector of floats
    fn encode(&self, queries: &Vec<String>, pooler_type: &str) -> Vec<f32>;

    fn encode_single(&self, queries: String, pooler_type: &str) -> Vec<f32>;
}

pub struct AutoQueryEncoder {
    model: BertForSentenceEmbeddings,
    tokenizer: BertTokenizer,
}

impl QueryEncoder for AutoQueryEncoder {
    fn new(
        model_name: &str,
        _tokenizer_name: Option<&str>,
        lowercase: bool,
        strip_accents: bool,
    ) -> Self {
        let _device = Device::cuda_if_available();
        let config = fetch_bert_style_config(&model_name);
        let model = fetch_bert_style_model(&model_name, config);

        let tokenizer = fetch_bert_style_vocab(&model_name, lowercase, strip_accents);
        Self { model, tokenizer }
    }

    fn encode(&self, queries: &Vec<String>, pooler_type: &str) -> Vec<f32> {
        let texts = queries;

        let tokenized_input =
            self.tokenizer
                .encode_list(&texts, 128, &TruncationStrategy::LongestFirst, 0);

        let max_len = 128;

        let pad_token_id = 0;
        let tokens_ids = tokenized_input
            .into_iter()
            .map(|input| {
                let mut token_ids = input.token_ids;
                token_ids.extend(vec![pad_token_id; max_len - token_ids.len()]);
                token_ids
            })
            .collect::<Vec<_>>();

        let tokens_masks = tokens_ids
            .iter()
            .map(|input| {
                Tensor::of_slice(
                    &input
                        .iter()
                        .map(|&e| i64::from(e != pad_token_id))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let tokens_ids = tokens_ids
            .into_iter()
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();

        let tokens_ids = Tensor::stack(&tokens_ids, 0);
        let tokens_masks = Tensor::stack(&tokens_masks, 0);
        let output: Tensor;

        if pooler_type == "mean" {
            let hidden_state: Tensor = no_grad(|| {
                self.model
                    .forward_t(
                        Some(&tokens_ids),
                        Some(&tokens_masks),
                        None,
                        None,
                        None,
                        None,
                        None,
                        false,
                    )
                    .unwrap()
            })
            .hidden_state;

            output = mean_pooling(hidden_state, tokens_masks);
        } else if pooler_type == "cls" {
            output = no_grad(|| {
                self.model
                    .forward_t(
                        Some(&tokens_ids),
                        Some(&tokens_masks),
                        None,
                        None,
                        None,
                        None,
                        None,
                        false,
                    )
                    .unwrap()
            })
            .hidden_state
            .select(1, 0);
        } else {
            panic!("pooler_type must be either mean or cls");
        }

        let embeddings: Vec<f32> = Vec::from(output);
        return embeddings;
    }

    fn encode_single(&self, queries: String, pooler_type: &str) -> Vec<f32> {
        let texts = &vec![queries];
        let tokenized_input =
            self.tokenizer
                .encode_list(&texts, 128, &TruncationStrategy::LongestFirst, 0);

        let max_len = 128;

        let pad_token_id = 0;
        let tokens_ids = tokenized_input
            .into_iter()
            .map(|input| {
                let mut token_ids = input.token_ids;
                token_ids.extend(vec![pad_token_id; max_len - token_ids.len()]);
                token_ids
            })
            .collect::<Vec<_>>();

        let tokens_masks = tokens_ids
            .iter()
            .map(|input| {
                Tensor::of_slice(
                    &input
                        .iter()
                        .map(|&e| i64::from(e != pad_token_id))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let tokens_ids = tokens_ids
            .into_iter()
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();

        let tokens_ids = Tensor::stack(&tokens_ids, 0);
        let tokens_masks = Tensor::stack(&tokens_masks, 0);
        let output: Tensor;

        if pooler_type == "mean" {
            let hidden_state: Tensor = no_grad(|| {
                self.model
                    .forward_t(
                        Some(&tokens_ids),
                        Some(&tokens_masks),
                        None,
                        None,
                        None,
                        None,
                        None,
                        false,
                    )
                    .unwrap()
            })
            .hidden_state;

            output = mean_pooling(hidden_state, tokens_masks);
        } else if pooler_type == "cls" {
            output = no_grad(|| {
                self.model
                    .forward_t(
                        Some(&tokens_ids),
                        Some(&tokens_masks),
                        None,
                        None,
                        None,
                        None,
                        None,
                        false,
                    )
                    .unwrap()
            })
            .hidden_state
            .select(1, 0);
        } else {
            panic!("pooler_type must be either mean or cls");
        }

        let embeddings: Vec<f32> = Vec::from(output);
        return embeddings;
    }
}
