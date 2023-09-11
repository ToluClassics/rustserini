use rust_bert::bert::{
    BertConfig, BertConfigResources, BertForSentenceEmbeddings, BertModelResources,
    BertVocabResources,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, no_grad, Device, Tensor};

// A Rust BERT example file to load bert-base-uncased model and encode a simple sentence

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_resource = RemoteResource::from_pretrained(BertConfigResources::BERT);
    let vocab_resource = RemoteResource::from_pretrained(BertVocabResources::BERT);
    let weights_resource = RemoteResource::from_pretrained(BertModelResources::BERT);

    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = BertConfig::from_file(config_path);
    let model = BertForSentenceEmbeddings::new(&vs.root() / "bert", &config);

    vs.load(weights_path)?;

    let tokenized_input = tokenizer.encode_list(
        &["Hello, I am a sentence!"],
        128,
        &TruncationStrategy::LongestFirst,
        0,
    );

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

    let output = no_grad(|| {
        model
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
    .pooled_output
    .unwrap()
    .get(0);

    let embeddings: Vec<f32> = Vec::from(output);
    dbg!(embeddings);
    Ok(())
}
