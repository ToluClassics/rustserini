#[cfg(test)]
mod tests {
    use rustserini::encode::auto::AutoDocumentEncoder;
    use rustserini::encode::base::{DocumentEncoder, RepresentationWriter};
    use rustserini::encode::vector_writer::{JsonlCollectionIterator, JsonlRepresentationWriter};
    use serde_json::{Number, Value};
    use std::collections::HashMap;

    #[test]
    fn test_auto_document_encoder() {
        let model_name = "bert-base-uncased";
        let tokenizer_name = None;
        let document_encoder: AutoDocumentEncoder =
            AutoDocumentEncoder::new(model_name, tokenizer_name);

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
        let fields = vec![&"text".to_string(), &"title".to_string()];
        writer.write(&batch_info);
    }

    #[test]
    fn test_jsonl_collection_iterator() {
        let path = "tests/test_files";
        let fields: Vec<&str> = vec!["docid", "text", "title"];
        let delimiter = "\t";
        let batch_size = 2;
        let mut iterator =
            JsonlCollectionIterator::new(&path, Some(fields), delimiter, &batch_size);

        iterator.load();
        assert_eq!(iterator.size, 10);
        assert_eq!(iterator.all_info["docid"].as_array().unwrap().len(), 10);

        assert_eq!(
            iterator.iter().next().unwrap()["title"][0],
            String::from("\"Introduction\"")
        );
    }
}
