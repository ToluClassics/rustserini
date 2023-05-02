#[cfg(test)]
mod tests {
    use rustserini::encode::auto::AutoDocumentEncoder;
    use rustserini::encode::base::DocumentEncoder;
    use std::collections::HashMap;

    #[test]
    fn test_auto_document_encoder() {
        let model_name = "bert-base-uncased";
        let tokenizer_name = None;
        let pooling = "mean";
        let l2_norm = false;
        let document_encoder: AutoDocumentEncoder =
            AutoDocumentEncoder::new(model_name, tokenizer_name, pooling, l2_norm);

        let texts = vec![
            "Hello, I am a sentence!".to_string(),
            "And another sentence.".to_string(),
        ];
        let titles = vec!["Title 1".to_string(), "Title 2".to_string()];
        let kwargs = HashMap::new();
        let embeddings = document_encoder.encode(&texts, &titles, kwargs);
        assert_eq!(embeddings.len(), 768);
    }
}
