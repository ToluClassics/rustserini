# Rusterini 🦀🦆
![](https://img.shields.io/badge/Rust-1.32+-orange.svg)
[![Maven Central](https://img.shields.io/maven-central/v/io.anserini/anserini?color=brightgreen)](https://search.maven.org/search?q=a:anserini)
[![Generic badge](https://img.shields.io/badge/Lucene-v9.5.0-brightgreen.svg)](https://archive.apache.org/dist/lucene/java/9.5.0/)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

Rusterini is a direct port of [Pyserini](https://github.com/castorini/pyserini) written mostly in RUST with bindings to [Anserini 🦆](https://github.com/castorini/anserini) in JAVA using JNI (Java Native Interface) to enable Lucene capabilities. 

This was mostly developed as a learning project to explore the speed and memory safety that Rust offers in a Library I am familiar with the inner workings ❗️❗️.

The plan is to expose as much of Pyserini as I can in this repository without directly binding to the python code.


## Installation

To install Rusterini, you need to have Rust and Cargo installed on your system. If you don't have Rust installed, you can install it from the official website: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

Once Rust is installed, you can install Rusterini by running the following command:

- Development Install
    - This repo depends on the Rust bindings of FAISS in C++. Thus we have to install Faiss using CMAKE by following the [instructions here](https://github.com/Enet4/faiss/blob/c_api_head/INSTALL.md#step-1-invoking-cmake) or [Here](https://github.com/Enet4/faiss-rs#installing-with-dynamic-linking)
    - To Interface with huggingface models for generating sentence embeddings, this project depends on [rust_bert](https://github.com/guillaume-be/rust-bert) which inturn depends on the C++ Libtorch API. To Install, [follow the instruction here](https://github.com/Enet4/faiss-rs#installing-with-dynamic-linking)
    - Clone the repo and experiment away!


- Install From Cargo
    ```bash
    cargo install rusterini
    ```

## Examples

#### (1.) Simple Lucene Index Searcher
Below example shows how to search through a Lucene Index of the MS Marco Passage Corpus

```rust
use rustserini::searcher::lucene::searcher::{LuceneQuery, LuceneSearcher};

let lucene_searcher = LuceneSearcher::new(String::from("indexes/msmarco-passage/lucene-index-msmarco"), None)?;

let search_query = LuceneQuery::String(
"did scientific minds lead to the success of the manhattan project".to_string(),
);

let hits = lucene_searcher.search(search_query, 10, None, None, false, false)?;

assert_eq!(lucene_searcher.num_docs, 8841823);
assert_eq!(result[0].docid, "0")
```

#### (2.) Simple Search example over a FAISS Flat Index

```rust
use rustserini::searcher::faiss::model::{AutoQueryEncoder, QueryEncoder};
use rustserini::searcher::faiss::searcher::{FaissSearchReturn, FaissSearcher};

let model_name = "castorini/mdpr-tied-pft-msmarco";
let tokenizer_name =  "castorini/mdpr-tied-pft-msmarco";
let query_encoder: AutoQueryEncoder =
    AutoQueryEncoder::new(model_name, tokenizer_name, true, true);

let mut searcher = FaissSearcher::new(
    "corpus/msmarco-passage-mini/pyserini".to_string(),
    query_encoder,
    768 as usize,
);

let result = searcher.search(
    "did scientific minds lead to the success of the manhattan project".to_string(),
    10,
    false,
);
let result = result?;
match result {
    FaissSearchReturn::Dense(search_results) => {
        println!("Result {:?}", search_results[0].docid);
        assert_eq!(search_results[0].docid, "0")
    }
    _ => panic!("Unexpected result type"),
}
```

#### (3.) Embedding Index (Faiss and JSON)
When Encoding a corpus, Pyserini provides capabilities to either write the embeddings to FAISS or in a JSON file. This repo [contains examples](examples) that can be run as CLI functions with different parameters. Some example below for encoding the msmarco passage corpus using multilingual Dense Passage Retriever(mDPR) on huggingface:

- Create a directory and download the jsonlines corpus
    ```bash
    $ mkdir corpus/msmarco-passage
    $ wget  https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus/resolve/main/corpus.jsonl.gz -P corpus/msmarco-passage
    ```

- Index the corpus. Run the command below to run the example shown [here](examples/faiss_embedding_writer.rs):: 
    ```bash
    $ cargo run --example faiss_embedding_writer --  --corpus corpus/msmarco-passage/corpus.jsonl.gz  --embeddings-dir indexes/msmarco-passage --encoder castorini/mdpr-tied-pft-msmarco --tokenizer castorini/mdpr-tied-pft-msmarco
    ```


## Benchmark

Coming soon......
