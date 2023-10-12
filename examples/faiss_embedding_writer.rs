use clap::{ArgAction, Parser};
use rustserini::encode::auto::AutoDocumentEncoder;
use rustserini::encode::base::{DocumentEncoder, RepresentationWriter};
use rustserini::encode::vector_writer::{FaissRepresentationWriter, JsonlCollectionIterator};
use std::collections::HashMap;
use std::time::Instant;

/// A Rust example of encoding a corpus and store the embeddings in a FAISS Index
/// Download the msmarco passage dataset using the below command:
/// mkdir corpus/msmarco-passage
/// wget  https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus/resolve/main/corpus.jsonl.gz -P corpus/msmarco-passage
/// cargo run --example faiss_embedding_writer --  --corpus corpus/msmarco-passage/corpus.jsonl.gz  --embeddings-dir corpus/msmarco-passage --encoder bert-base-uncased --tokenizer bert-base-uncased
///

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory that contains corpus files to be encoded, in jsonl format.
    #[arg(short, long)]
    corpus: String,

    /// Fields that contents in jsonl has (in order) separated by comma.
    #[arg(short, long, default_value = "text,title")]
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

    /// Tokenizer name or path
    #[arg(long)]
    tokenizer: String,

    /// Batch size for encoding
    #[arg(short, long, default_value_t = 8)]
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
    #[arg(long, default_value_t = 2)]
    embedding_dim: u32,
}

fn main() {
    let start = Instant::now();
    let args = Args::parse();

    let fields: Vec<String> = args.fields.split(',').map(|s| s.to_string()).collect();
    let mut iterator: JsonlCollectionIterator =
        JsonlCollectionIterator::new(fields, "docid".to_string(), args.delimiter, args.batch_size);
    let _ = iterator.load(args.corpus);

    let mut writer: FaissRepresentationWriter =
        FaissRepresentationWriter::new(&args.embeddings_dir, args.embedding_dim);
    writer.init_index(768, "Flat");
    let _ = writer.open_file();

    let lowercase = args.lowercase;
    let strip_accents = args.strip_accents;

    let encoder: AutoDocumentEncoder = AutoDocumentEncoder::new(
        &args.encoder,
        Some(&args.tokenizer),
        lowercase,
        strip_accents,
    );

    let mut counter: usize = 0;
    // let pb = ProgressBarIter(iterator.iter());

    for batch in iterator.iter() {
        let mut batch_info = HashMap::new();

        let batch_text: Vec<String> = batch["text"].to_vec();
        let batch_title: Vec<String> = batch["title"].to_vec();
        let batch_id: Vec<String> = batch["id"].to_vec();

        let embeddings = &encoder.encode(&batch_text, &batch_title, "cls");

        let mut embeddings: Vec<f32> = match embeddings {
            Ok(embeddings) => embeddings.to_vec(),
            Err(_) => vec![],
        };

        batch_info.insert("text", batch_text);
        batch_info.insert("title", batch_title);
        batch_info.insert("id", batch_id);

        let _ = &writer.write(&batch_info, &mut embeddings);

        counter += 1;
        if counter % 100 == 0 {
            // Reduce console output
            println!("Batch {} encoded", counter);
        }
    }
    let _ = writer.save_index();
    let _ = writer.save_docids();

    let duration = start.elapsed();
    println!("Time elapsed in expensive_function() is: {:?}", duration);
}
