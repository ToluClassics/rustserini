use crate::searcher::faiss::model::{AutoQueryEncoder, QueryEncoder, QueryType};

use anyhow::Ok;
use faiss::index::io::read_index;
use faiss::index::IndexImpl;
use faiss::Index;
use std::collections::HashMap;
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
        /*
        Create a new instance of FaissSearcher
         */
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
        /*
        Load a Faiss index from a directory
         */
        let index_dir: PathBuf = PathBuf::from(index_dir);
        let index_path: PathBuf = index_dir.join("index");
        let index: IndexImpl = read_index(index_path.as_path().display().to_string()).unwrap();

        index
    }

    fn load_docids(index_dir: &String) -> Vec<String> {
        /*
        Load a list of docids from a file
         */
        let index_dir: PathBuf = PathBuf::from(index_dir);
        let docid_path: PathBuf = index_dir.join("docid");
        let file = File::open(docid_path).map_err(|err| err.to_string());
        let reader = BufReader::new(file.unwrap());

        reader
            .lines()
            .map(|l| l.expect("Could not parse line"))
            .collect()
    }

    pub fn search(
        &mut self,
        query: String,
        k: usize,
        return_vector: bool,
    ) -> Result<FaissSearchReturn, anyhow::Error> {
        /*
        Search a query and return the top k results
         */
        let query = QueryType::Query { query };
        let emb_q = self.query_encoder.encode(query, "cls")?;
        let emb_q = emb_q.squeeze(0)?.to_vec1::<f32>()?;


        assert_eq!(&emb_q.len(), &self.dimension);
        let result = self.index.search(&emb_q, k).unwrap();

        let scores = result.distances.iter();
        let indices = result.labels.iter();

        if return_vector {
            // Needs more work
            let result_iter = indices
                .zip(scores)
                .map(|(x, y)| PRFDenseSearchResult::new(x.to_string(), *y, emb_q.clone()));

            Ok(FaissSearchReturn::PRFDense(result_iter.collect()))
        } else {
            let result_iter = indices.zip(scores).map(|(x, y)| {
                DenseSearchResult::new(
                    self.docids[usize::try_from(x.get().unwrap()).unwrap()].clone(),
                    *y,
                )
            });

            Ok(FaissSearchReturn::Dense(result_iter.collect()))
        }
    }

    pub fn batch_search(
        &mut self,
        queries: Vec<String>,
        q_ids: Vec<String>,
        k: usize,
        _return_vector: bool,
    ) -> Result<HashMap<String, FaissSearchReturn>, anyhow::Error> {
        /*
        Search a batch of queries and return the top k results
         */
        let queries = QueryType::Queries { query: queries };
        let emb_q = self.query_encoder.encode(queries, "cls")?;
        let emb_q = emb_q.flatten_all()?.to_vec1::<f32>()?;

        let embedding_length = self.dimension * &q_ids.len();
        assert_eq!(&emb_q.len(), &embedding_length);

        let result = self.index.search(&emb_q, k).unwrap();

        let scores_indices = result.distances.into_iter().zip(result.labels.into_iter());
        let scores_indices: Vec<(f32, faiss::Idx)> = scores_indices.collect();

        let mut results: HashMap<String, FaissSearchReturn> = HashMap::new();

        for (i, doc_result) in scores_indices.chunks(k).enumerate() {
            let index_result = doc_result.iter().map(|(score, idx)| {
                let docid = self.docids[usize::try_from(idx.get().unwrap()).unwrap()].clone();
                DenseSearchResult::new(docid, *score)
            });

            let index_result = FaissSearchReturn::Dense(index_result.collect());
            let query_index = q_ids[i].clone();
            results.insert(query_index, index_result);
        }

        Ok(results)
    }
}
