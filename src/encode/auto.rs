use crate::encode::base::DocumentEncoder;
use rust_bert::bert::{
    BertConfig, BertConfigResources, BertForSentenceEmbeddings, BertModelResources,
    BertVocabResources,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, no_grad, Device, Kind, Tensor};

pub struct AutoDocumentEncoder {
    model: BertForSentenceEmbeddings,
    tokenizer: BertTokenizer,
}

fn mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor {
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
    fn new(_model_name: &str, _tokenizer_name: Option<&str>) -> AutoDocumentEncoder {
        let device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(device);
        let model_config_resource = RemoteResource::from_pretrained(BertConfigResources::BERT);
        let config = BertConfig::from_file(model_config_resource.get_local_path().unwrap());
        let model_resource = RemoteResource::from_pretrained(BertModelResources::BERT);
        let model_path = model_resource.get_local_path().unwrap();
        let model = BertForSentenceEmbeddings::new(&vs.root() / "bert", &config);

        vs.load(model_path).expect("Couldn't load model weights");

        let tokenizer_resource = RemoteResource::from_pretrained(BertVocabResources::BERT);
        let tokenizer_path = tokenizer_resource.get_local_path().unwrap();
        let tokenizer = BertTokenizer::from_file(tokenizer_path.to_str().unwrap(), true, true)
            .expect("Couldn't build tokenizer");

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

        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap_or(0);

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
