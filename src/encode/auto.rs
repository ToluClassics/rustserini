use crate::encode::base::DocumentEncoder;
use rust_bert::bert::{BertConfig, BertForSentenceEmbeddings};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, no_grad, Device, Kind, Tensor};

const BERT_MODELS: [&str; 3] = ["bert-base-uncased", "bert-base-cased", "bert-large-uncased"];

/// An AutoDocumentEncoder for encoding documents with BERT-style  encoding models
pub struct AutoDocumentEncoder {
    model: BertForSentenceEmbeddings,
    tokenizer: BertTokenizer,
}

pub fn fetch_bert_style_config(model_name: &str) -> BertConfig {
    let remote_config_path = format!(
        "https://huggingface.co/{model_name}/resolve/main/config.json",
        model_name = model_name
    );
    let alias = format!("{model_name}/model", model_name = model_name);
    let config_tuple = (alias.as_str(), remote_config_path.as_str());
    let config_resource = RemoteResource::from_pretrained(config_tuple);
    let config_path = config_resource.get_local_path().unwrap();
    let config = BertConfig::from_file(config_path);
    config
}

pub fn fetch_bert_style_vocab(
    model_name: &str,
    lowercase: bool,
    strip_accents: bool,
) -> BertTokenizer {
    let remote_vocab_path = format!(
        "https://huggingface.co/{model_name}/resolve/main/vocab.txt",
        model_name = model_name
    );
    let alias = format!("{model_name}/model", model_name = model_name);
    let vocab_tuple = (alias.as_str(), remote_vocab_path.as_str());
    let vocab_resource = RemoteResource::from_pretrained(vocab_tuple);
    let vocab_path = vocab_resource.get_local_path().unwrap();

    let vocab = BertTokenizer::from_file(vocab_path.to_str().unwrap(), lowercase, strip_accents)
        .expect("Couldn't build tokenizer");
    vocab
}

pub fn fetch_bert_style_model(model_name: &str, config: BertConfig) -> BertForSentenceEmbeddings {
    let remote_model_path = format!(
        "https://huggingface.co/{model_name}/resolve/main/rust_model.ot",
        model_name = &model_name
    );
    let alias = format!("{model_name}/model", model_name = model_name);
    let model_tuple = (alias.as_str(), remote_model_path.as_str());
    let model_resource = RemoteResource::from_pretrained(model_tuple);
    let model_path = model_resource.get_local_path().unwrap();
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let model: BertForSentenceEmbeddings;

    if BERT_MODELS.contains(&model_name) {
        model = BertForSentenceEmbeddings::new(&vs.root() / "bert", &config);
    } else {
        model = BertForSentenceEmbeddings::new(&vs.root(), &config);
    }
    vs.load(model_path).expect("Couldn't load model weights");

    model
}

pub fn mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor {
    let mut output_vectors = Vec::new();
    let input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(&last_hidden_state);
    let sum_embeddings = (last_hidden_state * &input_mask_expanded).sum_dim_intlist(
        [1].as_slice(),
        false,
        Kind::Float,
    );
    let sum_mask = input_mask_expanded.sum_dim_intlist([1].as_slice(), false, Kind::Float);
    let sum_mask = sum_mask.clamp_min(1e-9);

    output_vectors.push(&sum_embeddings / &sum_mask);

    Tensor::cat(&output_vectors, 1)
}

impl DocumentEncoder for AutoDocumentEncoder {
    // instantiating a new AutoDocumentEncoder instance
    fn new(
        model_name: &str,
        _tokenizer_name: Option<&str>,
        lowercase: bool,
        strip_accents: bool,
    ) -> AutoDocumentEncoder {
        let _device = Device::cuda_if_available();
        let config = fetch_bert_style_config(&model_name);
        let model = fetch_bert_style_model(&model_name, config);

        let tokenizer = fetch_bert_style_vocab(&model_name, lowercase, strip_accents);
        Self { model, tokenizer }
    }

    fn encode(&self, texts: &Vec<String>, titles: &Vec<String>, pooler_type: &str) -> Vec<f32> {
        let texts = if !titles.is_empty() {
            texts
                .iter()
                .zip(titles.iter())
                .map(|(text, title)| format!("{} {}", title, text))
                .collect::<Vec<_>>()
        } else {
            texts.clone()
        };

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
