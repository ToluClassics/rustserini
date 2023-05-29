use clap::{ArgAction, Parser};
use rustserini::encode::vector_writer::{JsonlCollectionIterator, JsonlRepresentationWriter};
use std::path::{Path, PathBuf};

/// Simple program to greet a person
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
}
