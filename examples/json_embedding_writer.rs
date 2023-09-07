use clap::{ArgAction, Parser};
use rustserini::encode::auto::AutoDocumentEncoder;
use rustserini::encode::base::{DocumentEncoder, RepresentationWriter};
use rustserini::encode::vector_writer::{JsonlCollectionIterator, JsonlRepresentationWriter};
use serde_json::{Number, Value};
use std::collections::HashMap;

/// Simple program to encode a corpus and store the embeddings in a jsonl file
/// Download the msmarco passage dataset using the below command:
/// mkdir corpus/msmarco-passage
/// wget  https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus/resolve/main/corpus.jsonl.gz -P corpus/msmarco-passage
/// cargo run --example json_embedding_writer --  --corpus corpus/msmarco-passage/corpus.jsonl.gz  --embeddings-dir corpus/msmarco-passage --encoder bert-base-uncased --tokenizer bert-base-uncased
///
///
///
///
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory that contains corpus files to be encoded, in jsonl format.
    #[arg(short, long)]
    corpus: String,

    /// Fields that contents in jsonl has (in order) separated by comma.
    #[arg(short, long, default_value = "text")]
    fields: String,

    /// delimiter for the fields
    #[arg(short, long, default_value = "\n")]
    delimiter: String,

    /// shard-id 0-based
    #[arg(short, long, default_value_t = 0)]
    shard_id: u8,

    /// number of shards
    #[arg(long, default_value_t = 1)]
    shard_num: u8,

    /// directory to store encoded corpus
    #[arg(short, long, required = true)]
    embeddings_dir: String,

    /// Whether to store the embeddings in a faiss index or in a jsonl file
    #[arg(long, action=ArgAction::SetFalse)]
    to_faiss: bool,

    /// Encoder name or path
    #[arg(long)]
    encoder: String,

    /// Tokenizer name or path
    #[arg(long)]
    tokenizer: String,

    /// Batch size for encoding
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// GPU Device ==> cpu or cuda:0
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Whether to use fp16
    #[arg(long, action=ArgAction::SetTrue)]
    fp16: bool,

    /// max length of the input
    #[arg(short, long, default_value_t = 512)]
    max_length: u16,

    /// Embedding dimension
    #[arg(long, default_value_t = 768)]
    embedding_dim: u16,
}

fn main() {
    let args = Args::parse();

    let fields = args.fields.split(",").collect::<Vec<&str>>();

    let mut iterator = JsonlCollectionIterator::new(
        &args.corpus,
        Some(fields),
        &args.delimiter,
        &args.batch_size,
    );
    iterator.load();

    let mut writer = JsonlRepresentationWriter::new(&args.embeddings_dir);
    writer.open_file();

    let encoder = AutoDocumentEncoder::new(&args.encoder, Some(&args.tokenizer));

    for batch in iterator.iter() {
        let mut batch_info = HashMap::new();

        let batch_text: Vec<String> = batch["text"]
            .iter()
            .map(|x| x.to_string().replace("\"", "").replace("\\", ""))
            .collect();

        let batch_title: Vec<String> = batch["title"]
            .iter()
            .map(|x| x.to_string().replace("\"", "").replace("\\", ""))
            .collect();

        let batch_id: Vec<String> = batch["id"]
            .iter()
            .map(|x| x.to_string().replace("\"", "").replace("\\", ""))
            .collect();

        let embeddings = &encoder.encode(&batch_text, &batch_title, "cls");
        let embeddings: Vec<Value> = embeddings
            .iter()
            .map(|x| Value::Number(Number::from_f64(*x as f64).unwrap()))
            .collect();

        let embeddings_value: Vec<_> = embeddings.chunks(args.embedding_dim as usize).collect();

        let embeddings: Vec<Value> = embeddings_value
            .iter()
            .map(|x| Value::Array(x.to_vec()))
            .collect();

        batch_info.insert(
            "text",
            Value::Array(
                batch_text
                    .iter()
                    .map(|x| Value::String(x.to_string()))
                    .collect(),
            ),
        );

        batch_info.insert(
            "title",
            Value::Array(
                batch_title
                    .iter()
                    .map(|x| Value::String(x.to_string()))
                    .collect(),
            ),
        );

        batch_info.insert(
            "id",
            Value::Array(
                batch_id
                    .iter()
                    .map(|x| Value::String(x.to_string()))
                    .collect(),
            ),
        );

        batch_info.insert("vector", Value::Array(embeddings));

        let _ = &writer.write(&batch_info);
    }
}
