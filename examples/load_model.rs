extern crate anyhow;

use std::path::PathBuf;

use rust_bert::bert::{BertConfig, BertForSentenceEmbeddings};
use rust_bert::resources::{LocalResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{BertVocab, Vocab};
use tch::{nn, no_grad, Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_resource = LocalResource {
        local_path: PathBuf::from(
            "/Users/mac/Desktop/rust-bert/resources/mdpr-tied-pft-msmarco/config.json",
        ),
    };
    let vocab_resource = LocalResource {
        local_path: PathBuf::from(
            "/Users/mac/Desktop/rust-bert/resources/mdpr-tied-pft-msmarco/vocab.txt",
        ),
    };
    let weights_resource = LocalResource {
        local_path: PathBuf::from(
            "/Users/mac/Desktop/rust-bert/resources/mdpr-tied-pft-msmarco/rust_model.ot",
        ),
    };
    let strip_accents = false;
    let lower_case = false;

    let config_path = config_resource.get_local_path().unwrap();
    let vocab_path = vocab_resource.get_local_path().unwrap();
    let weights_path = weights_resource.get_local_path().unwrap();

    let vocab = BertVocab::from_file(&vocab_path).unwrap();
    let bert_tokenizer: BertTokenizer =
        BertTokenizer::from_existing_vocab(vocab, lower_case, strip_accents);

    let config = BertConfig::from_file(config_path.to_str().unwrap());

    println!("{:?}", config);

    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    let model = BertForSentenceEmbeddings::new(&vs.root(), &config);
    vs.load(weights_path).expect("Couldn't load model weights");

    let tokenized_input = bert_tokenizer.encode_list(
        &["Title 1 Hello, I am a sentence!"],
        128,
        &TruncationStrategy::LongestFirst,
        0,
    );

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

    println!("{:?}", tokens_ids[0]);
    println!("{:?}", tokens_masks[0]);

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
    .hidden_state
    .select(1, 0);

    let embeddings: Vec<f32> = Vec::from(output);
    println!("{:?}", &embeddings[0..10]);
    Ok(())
}
