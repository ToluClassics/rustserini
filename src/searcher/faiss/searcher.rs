use crate::searcher::faiss::model::{AutoQueryEncoder, QueryEncoder};

use faiss::index::io::read_index;
use faiss::index::IndexImpl;
use faiss::{index_factory, Index, MetricType};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub struct FaissSearcher {
    index_dir: String,
    query_encoder: AutoQueryEncoder,
    dimension: usize,
    index: IndexImpl,
    docids: Vec<String>,
}

impl FaissSearcher {
    pub fn new(index_dir: String, query_encoder: AutoQueryEncoder, dimension: usize) -> Self {
        let index: IndexImpl = Self::load_index(&index_dir);
        let docids: Vec<String> = Self::load_docids(&index_dir);
        Self {
            index_dir,
            query_encoder,
            dimension,
            index,
            docids,
        }
    }

    fn load_index(index_dir: &String) -> IndexImpl {
        let index_dir: PathBuf = PathBuf::from(index_dir);
        let index_path: PathBuf = index_dir.join("index");
        let index: IndexImpl = read_index(index_path.as_path().display().to_string()).unwrap();

        index
    }

    fn load_docids(index_dir: &String) -> Vec<String> {
        let index_dir: PathBuf = PathBuf::from(index_dir);
        let docid_path: PathBuf = index_dir.join("docid");
        let file = File::open(docid_path).map_err(|err| err.to_string());
        let reader = BufReader::new(file.unwrap());

        reader
            .lines()
            .map(|l| l.expect("Could not parse line"))
            .collect()
    }

    pub fn search_vector(&mut self, query: String, k: usize, threads: usize, return_vector: bool) {
        let emb_q = self.query_encoder.encode_single(query, "cls");

        assert_eq!(&emb_q.len(), &self.dimension);
        let result = self.index.search(&emb_q, k).unwrap();

        println!("Result {:?}", result);
    }
}
