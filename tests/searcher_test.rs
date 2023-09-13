#[cfg(test)]
mod tests {
    use rustserini::searcher::faiss::model::{AutoQueryEncoder, QueryEncoder};
    use rustserini::searcher::faiss::searcher::FaissSearcher;
    use std::time::Instant;

    // fn round_to_decimal_places(n: f32, places: u32) -> f32 {
    //     let multiplier: f32 = 10f32.powi(places as i32);
    //     (n * multiplier).round() / multiplier
    // }

    #[test]
    fn test_auto_query_encoder_cls_pooling() {
        let start = Instant::now();
        let model_name = "castorini/mdpr-tied-pft-msmarco";
        let tokenizer_name = None;
        let query_encoder: AutoQueryEncoder =
            AutoQueryEncoder::new(model_name, tokenizer_name, true, true);

        let queries =
            vec!["did scientific minds lead to the success of the manhattan project".to_string()];
        let embeddings = query_encoder.encode(&queries, "cls");

        assert_eq!(embeddings.len(), 768);
        let duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);

        let mut searcher = FaissSearcher::new(
            "corpus/msmarco-passage-mini".to_string(),
            query_encoder,
            768 as usize,
        );

        searcher.search_vector(
            "did scientific minds lead to the success of the manhattan project".to_string(),
            10,
            1,
            true,
        );
    }
}
