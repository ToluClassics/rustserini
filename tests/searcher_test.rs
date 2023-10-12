#[cfg(test)]
mod tests {
    use rustserini::searcher::faiss::model::{AutoQueryEncoder, QueryEncoder};
    use rustserini::searcher::faiss::searcher::{FaissSearchReturn, FaissSearcher};
    use rustserini::searcher::lucene::searcher::{LuceneQuery, LuceneSearcher};
    use std::time::Instant;

    #[test]
    fn test_faiss_searcher() {
        let start = Instant::now();
        let model_name = "castorini/mdpr-tied-pft-msmarco";
        let tokenizer_name = None;
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
        let result = result.unwrap();
        match result {
            FaissSearchReturn::Dense(search_results) => {
                println!("Result {:?}", search_results[0].docid);
                assert_eq!(search_results[0].docid, "0")
            }
            _ => panic!("Unexpected result type"),
        }
        let duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);
    }

    #[test]
    fn test_faiss_batch_searcher() {
        let start = Instant::now();
        let model_name = "castorini/mdpr-tied-pft-msmarco";
        let tokenizer_name = None;
        let query_encoder: AutoQueryEncoder =
            AutoQueryEncoder::new(model_name, tokenizer_name, true, true);

        let mut searcher = FaissSearcher::new(
            "corpus/msmarco-passage-mini/pyserini".to_string(),
            query_encoder,
            768 as usize,
        );

        let result = searcher.batch_search(
            vec![
                "did scientific minds lead to the success of the manhattan project".to_string(),
                "did scientific minds lead to the success of the manhattan project".to_string(),
            ],
            vec!["0".to_string(), "1".to_string()],
            10,
            false,
        );
        let result = result.unwrap();
        let search_results = result.get("0").unwrap();
        match search_results {
            FaissSearchReturn::Dense(search_results) => {
                println!("Result {:?}", search_results[0].docid);
                assert_eq!(search_results[0].docid, "0")
            }
            _ => panic!("Unexpected result type"),
        }

        let duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);
    }

    #[test]
    fn test_lucene_searcher() {
        let search_instance = LuceneSearcher::new(
            "/Users/mac/Documents/castorini/anserini/indexes/msmarco-passage/lucene-index-msmarco"
                .to_string(),
            None,
        )
        .unwrap();

        assert_eq!(search_instance.num_docs, 8841823);

        let search_query = LuceneQuery::String(
            "did scientific minds lead to the success of the manhattan project".to_string(),
        );

        let result = search_instance
            .search(search_query, 10, None, None, false, false)
            .unwrap();

        assert_eq!(result[0].docid, "0")
    }
}
