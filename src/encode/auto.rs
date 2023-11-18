use crate::encode::base::DocumentEncoder;

use anyhow::{anyhow, Error as E, Result};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use candle_core::{DType, Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

pub const FLOATING_DTYPE: DType = DType::F32;
pub const LONG_DTYPE: DType = DType::I64;

pub enum Model {
    BertModel {model: BertModel},
}

/// An AutoDocumentEncoder for encoding documents with BERT-style  encoding models
pub struct AutoDocumentEncoder {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}


pub fn build_roberta_model_and_tokenizer(model_name_or_path: impl Into<String>, offline: bool, model_type: &str) -> Result<(Model, Tokenizer)> {
    let device = Device::Cpu;
    let (model_id, revision) = (model_name_or_path.into(), "main".to_string());
    let repo = Repo::with_revision(model_id, RepoType::Model, revision);

    let (config_filename, tokenizer_filename, weights_filename) = if offline {
        let cache = Cache::default().repo(repo);
        (
            cache
                .get("config.json")
                .ok_or(anyhow!("Missing config file in cache"))?,
            cache
                .get("tokenizer.json")
                .ok_or(anyhow!("Missing tokenizer file in cache"))?,
            cache
                .get("model.safetensors")
                .ok_or(anyhow!("Missing weights file in cache"))?,
        )
    } else {
        let api = Api::new()?;
        let api = api.repo(repo);
        (
            api.get("config.json")?,
            api.get("tokenizer.json")?,
            api.get("model.safetensors")?,
        )
    };

    println!("config_filename: {}", config_filename.display());
    println!("tokenizer_filename: {}", tokenizer_filename.display());
    println!("weights_filename: {}", weights_filename.display());


    let config = std::fs::read_to_string(config_filename)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], FLOATING_DTYPE, &device)? };

    let model = match model_type {
        "BertModel" => {
            let config: Config = serde_json::from_str(&config)?;
            let model = BertModel::load(vb, &config)?;
            Model::BertModel {model}
        }
        _ => panic!("Invalid model_type")
    };

    Ok((model, tokenizer))
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub fn mean_pooling(last_hidden_state: Tensor, normalize_embeddings: bool) -> Result<Tensor, E> {
    /*
    Compute mean pooling of BERT hidden states
    */
    let (_n_sentence, n_tokens, _hidden_size) = last_hidden_state.dims3()?;
    let embeddings = (last_hidden_state.sum(1)? / (n_tokens as f64))?;
    let embeddings = if normalize_embeddings {
        normalize_l2(&embeddings)?
    } else {
        embeddings
    };

    Ok(embeddings)
}

impl DocumentEncoder for AutoDocumentEncoder {
    // instantiating a new AutoDocumentEncoder instance
    fn new(
        model_name: &str,
        lowercase: bool,
        strip_accents: bool,
    ) -> AutoDocumentEncoder {
        let device = Device::Cpu;
        let (model, tokenizer) = build_roberta_model_and_tokenizer(model_name, false, "BertModel").unwrap();
        Self { model, tokenizer, device }
    }

    fn encode(
        &self,
        texts: &Vec<String>,
        titles: &Vec<String>,
        pooler_type: &str,
    ) -> Result<Tensor, E> {
        /*
        Encode a list of texts and/or titles into a list of vectors
        */
        let texts = if !titles.is_empty() {
            texts
                .iter()
                .zip(titles.iter())
                .map(|(text, title)| format!("{} {}", title, text))
                .collect::<Vec<_>>()
        } else {
            texts.to_owned()
        };

        let max_len = 128;
        let pad_token_id = 0;

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

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let mut embeddings: Tensor;
        let model = match &self.model {
            Model::BertModel {model} => model,
        };

        if pooler_type == "mean" {
    
            let hidden_state: Tensor = model.forward(&token_ids, &token_type_ids)?;
            embeddings = mean_pooling(hidden_state, false)?;
        } else if pooler_type == "cls" {
            embeddings = model.forward(&token_ids, &token_type_ids)?;
            embeddings = embeddings.i((.., 0))?
        } else {
            panic!("pooler_type must be either mean or cls");
        }

        Ok(embeddings)
    }
}
