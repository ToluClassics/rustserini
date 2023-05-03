#[cfg(test)]
mod tests {
    use rustserini::encode::auto::AutoDocumentEncoder;
    use rustserini::encode::base::{DocumentEncoder, RepresentationWriter};
    use rustserini::encode::vector_writer::JsonlRepresentationWriter;
    use serde_json::{Map, Number, Value};
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

    #[test]
    fn test_json_representation_writer() {
        let path = "test";
        let mut writer = JsonlRepresentationWriter::new(path);
        writer.open_file();
        let mut batch_info = HashMap::new();
        batch_info.insert(
            "id",
            Value::Array(vec![
                Value::String("0".to_string()),
                Value::String("1".to_string()),
            ]),
        );
        batch_info.insert(
            "text",
            Value::Array(vec![
                Value::String("Hello, I am a sentence!".to_string()),
                Value::String("Hello, I am a sentences!".to_string()),
            ]),
        );
        batch_info.insert(
            "title",
            Value::Array(vec![
                Value::String("Hello, I am a sentence!".to_string()),
                Value::String("Hello, I am a sentences!".to_string()),
            ]),
        );
        batch_info.insert(
            "vector",
            Value::Array(vec![
                Value::Array(vec![
                    Value::Number(Number::from_f64(0.1).unwrap()),
                    Value::Number(Number::from_f64(0.2).unwrap()),
                    Value::Number(Number::from_f64(0.3).unwrap()),
                ]),
                Value::Array(vec![
                    Value::Number(Number::from_f64(0.1).unwrap()),
                    Value::Number(Number::from_f64(0.2).unwrap()),
                    Value::Number(Number::from_f64(0.3).unwrap()),
                ]),
            ]),
        );
        let fields = vec!["text".to_string(), "title".to_string()];
        writer.write(&batch_info, &fields);
    }
}
