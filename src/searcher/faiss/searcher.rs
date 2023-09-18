use crate::searcher::faiss::model::{AutoQueryEncoder, QueryEncoder};

use faiss::index::io::read_index;
use faiss::index::IndexImpl;
use faiss::Index;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Debug)]
pub enum FaissSearchReturn {
    Dense(Vec<DenseSearchResult>),
    PRFDense(Vec<PRFDenseSearchResult>),
}

pub struct FaissSearcher {
    query_encoder: AutoQueryEncoder,
    dimension: usize,
    index: IndexImpl,
    docids: Vec<String>,
}

#[derive(Debug)]
#[records::record]
pub struct DenseSearchResult {
    docid: String,
    score: f32,
}

#[derive(Debug)]
#[records::record]
pub struct PRFDenseSearchResult {
    docid: String,
    score: f32,
    prf_score: Vec<f32>,
}

impl FaissSearcher {
    pub fn new(index_dir: String, query_encoder: AutoQueryEncoder, dimension: usize) -> Self {
        let index: IndexImpl = Self::load_index(&index_dir);
        let docids: Vec<String> = Self::load_docids(&index_dir);
        Self {
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

    pub fn search_vector(
        &mut self,
        query: String,
        k: usize,
        return_vector: bool,
    ) -> FaissSearchReturn {
        let emb_q = self.query_encoder.encode_single(query, "cls");

        assert_eq!(&emb_q.len(), &self.dimension);
        let result = self.index.search(&emb_q, k).unwrap();

        println!("Result {:?}", result);
        let scores = result.distances.iter();
        let indices = result.labels.iter();

        if return_vector {
            // Needs more work
            let result_iter = indices
                .zip(scores)
                .map(|(x, y)| PRFDenseSearchResult::new(x.to_string(), *y, emb_q.clone()));

            FaissSearchReturn::PRFDense(result_iter.collect())
        } else {
            let result_iter = indices.zip(scores).map(|(x, y)| {
                DenseSearchResult::new(
                    self.docids[usize::try_from(x.get().unwrap()).unwrap()].clone(),
                    *y,
                )
            });

            FaissSearchReturn::Dense(result_iter.collect())
        }
    }
}
