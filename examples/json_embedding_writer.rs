use clap::{ArgAction, Parser};
use rustserini::encode::auto::AutoDocumentEncoder;
use rustserini::encode::base::{DocumentEncoder, RepresentationWriter};
use rustserini::encode::vector_writer::{JsonlCollectionIterator, JsonlRepresentationWriter};
use std::collections::HashMap;
use std::time::Instant;

/// Simple program to encode a corpus and store the embeddings in a jsonl file
/// Download the msmarco passage dataset using the below command:
/// mkdir corpus/msmarco-passage
/// wget  https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus/resolve/main/corpus.jsonl.gz -P corpus/msmarco-passage
/// cargo run --example json_embedding_writer --  --corpus corpus/msmarco-passage/corpus.jsonl.gz  --embeddings-dir corpus/msmarco-passage --encoder bert-base-uncased --tokenizer bert-base-uncased
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

    /// Use lowercase in tokenizer
    #[arg(long, action=ArgAction::SetTrue)]
    lowercase: bool,

    /// Strip accents in tokenizer
    #[arg(long, action=ArgAction::SetTrue)]
    strip_accents: bool,

    /// Encoder name or path
    #[arg(long)]
    encoder: String,

    /// Encoder Revision
    #[arg(long, default_value = "main")]
    revision: String,

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
    embedding_dim: u32,
}

fn sanitize_string(s: &str) -> String {
    s.replace("\"", "").replace("\\", "")
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args = Args::parse();

    let fields: Vec<String> = args.fields.split(',').map(|s| s.to_string()).collect();
    let mut iterator: JsonlCollectionIterator =
        JsonlCollectionIterator::new(fields, "docid".to_string(), args.delimiter, args.batch_size);
    let _ = iterator.load(args.corpus);

    println!("Initialize a representation writer and open a file to store the embeddings");
    let mut writer = JsonlRepresentationWriter::new(&args.embeddings_dir, args.embedding_dim);
    let _ = writer.open_file();

    let lowercase = args.lowercase;
    let strip_accents = args.strip_accents;

    println!("Tokenizer lowercase: {:?}", lowercase);

    let encoder = AutoDocumentEncoder::new(
        &args.encoder,
        lowercase,
        strip_accents,
        &args.revision,
    );

    let mut counter: usize = 0;
    for batch in iterator.iter() {
        let mut batch_info = HashMap::new();

        let batch_text: Vec<String> = batch["text"].iter().map(|x| sanitize_string(x)).collect();
        let batch_title: Vec<String> = batch["title"].iter().map(|x| sanitize_string(x)).collect();
        let batch_id: Vec<String> = batch["id"].iter().map(|x| sanitize_string(x)).collect();

        let embeddings = &encoder.encode(&batch_text, &batch_title, "cls")?;

        let mut embeddings: Vec<f32> = embeddings.squeeze(0)?.to_vec1::<f32>()?;

        batch_info.insert("text", batch_text);
        batch_info.insert("title", batch_title);
        batch_info.insert("id", batch_id);

        let _ = &writer.write(&batch_info, &mut embeddings);

        counter += 1;
        println!("Batch {} encoded", counter);
    }

    let duration = start.elapsed();
    println!("Time elapsed in expensive_function() is: {:?}", duration);

    Ok(())
}
