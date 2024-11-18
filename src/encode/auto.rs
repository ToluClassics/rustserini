use std::vec;

use crate::encode::base::DocumentEncoder;

use anyhow::{anyhow, Error as E, Result};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel,BertForMaskedLM, Config};
use tokenizers::{PaddingParams, Tokenizer};
use serde_json::Value;

pub const FLOATING_DTYPE: DType = DType::F32;
pub const LONG_DTYPE: DType = DType::I64;

pub enum Model {
    BertModel {model: BertModel},
    BertForMaskedLM {model: BertForMaskedLM},
}

pub enum OutputModelType{
    BertModel,
    BertForMaskedLM,
}

/// An AutoDocumentEncoder for encoding documents with BERT-style  encoding models
pub struct AutoDocumentEncoder {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}


pub fn build_model_and_tokenizer(model_name_or_path: impl Into<String>, offline: bool, revision: &str) -> Result<(Model, Tokenizer)> {
    let device = Device::Cpu;
    let (model_id, revision) = (model_name_or_path.into(), revision.into());
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
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], FLOATING_DTYPE, &device)? };
    
    let model_configuration: Value = serde_json::from_str(&config)?;
    let model_architecture = &model_configuration["architectures"][0].as_str();

    println!("model_architecture: {:?}", model_architecture);

    let model = match model_architecture {
        Some("BertModel") => {
            let config: Config = serde_json::from_str(&config)?;
            let model = BertModel::load(vb, &config)?;
            Model::BertModel {model}
        }
        Some("BertForMaskedLM") => {
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
        revision: &str,
    ) -> AutoDocumentEncoder {
        let device = Device::Cpu;
        let (model, tokenizer) = build_model_and_tokenizer(model_name, false, revision).unwrap();
        Self { model, tokenizer, device }
    }

    fn encode(
        &self,
        texts: &Vec<String>,
        titles: Option<&Vec<String>>,
        pooler_type: &str,
    ) -> Result<Tensor, E> {
        /*
        Encode a list of texts and/or titles into a list of vectors
        */
        let texts = if let Some(titles) = titles  {
            texts
                .iter()
                .zip(titles.iter())
                .map(|(text, title)| format!("{} {}", title, text))
                .collect::<Vec<_>>()
        } else {
            texts.to_owned()
        };

        let tokens = self.tokenizer
            .encode_batch(texts, true, )
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
        let attention_mask = Tensor::stack(&attention_mask, 0)?;

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
