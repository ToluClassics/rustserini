# Rusterini
![Rust](https://github.com/toml-lang/toml/workflows/Rust/badge.svg)

Rusterini is a high-performance toolkit for reproducible information retrieval research, ported from Pyserini and implemented in Rust. Like Pyserini, Rusterini supports both sparse and dense representations for information retrieval.

## Introduction

Rusterini leverages the speed and memory efficiency of Rust to provide high-performance information retrieval. Retrieval using sparse representations is powered by integration with Anserini IR toolkit, a project built on Lucene that our team also maintains. Retrieval using dense representations, on the other hand, is powered by integration with Facebook's Faiss library.

Primarily designed for first-stage retrieval in multi-stage ranking architectures, Rusterini is a self-contained standard Rust package that aims to make information retrieval research both effective and reproducible. It comes bundled with queries, relevance judgments, pre-built indexes, and evaluation scripts for commonly used IR test collections, making it simple to reproduce runs on a number of standard IR test collections.

## Features

- High-performance information retrieval with Rust
- Sparse representation retrieval with Anserini
- Dense representation retrieval with Faiss
- Pre-packaged with queries, relevance judgments, and pre-built indexes
- Easy reproduction of runs on standard IR test collections
- Comprehensive evaluation scripts

## Installation

To install Rusterini, you need to have Rust and Cargo installed on your system. If you don't have Rust installed, you can install it from the official website: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

Once Rust is installed, you can install Rusterini by running the following command:

```bash
cargo install rusterini
```