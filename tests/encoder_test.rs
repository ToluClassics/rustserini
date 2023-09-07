#[cfg(test)]
mod tests {
    use faiss::Index;
    use rustserini::encode::auto::AutoDocumentEncoder;
    use rustserini::encode::base::{DocumentEncoder, RepresentationWriter};
    use rustserini::encode::vector_writer::FaissRepresentationWriter;
    use rustserini::encode::vector_writer::{JsonlCollectionIterator, JsonlRepresentationWriter};
    use serde_json::{Number, Value};
    use std::collections::HashMap;

    fn round_to_decimal_places(n: f32, places: u32) -> f32 {
        let multiplier: f32 = 10f32.powi(places as i32);
        (n * multiplier).round() / multiplier
    }

    #[test]
    fn test_auto_document_encoder_cls_pooling() {
        let model_name = "bert-base-uncased";
        let tokenizer_name = None;
        let document_encoder: AutoDocumentEncoder =
            AutoDocumentEncoder::new(model_name, tokenizer_name);

        let texts = vec![
            "Hello, I am a sentence!".to_string(),
            "And another sentence.".to_string(),
        ];
        let titles = vec!["Title 1".to_string(), "Title 2".to_string()];
        let embeddings = document_encoder.encode(&texts, &titles, "cls");

        let bert_output_text1: Vec<f32> = vec![
            0.12826118,
            -0.0657022,
            0.15965648,
            -0.12741008,
            -0.15683924,
            -0.43648475,
            0.365709,
            0.21293718,
            0.15961173,
            -0.45108163,
        ];

        let bert_output_text1: Vec<f32> = bert_output_text1
            .iter()
            .map(|&x| round_to_decimal_places(x, 2))
            .collect();

        let bert_ground_truth_1: Vec<f32> = embeddings[0..10]
            .iter()
            .map(|&x| round_to_decimal_places(x, 2))
            .collect();

        let bert_output_text2: Vec<f32> = vec![
            -0.26028907,
            -0.00148716,
            -0.24410412,
            -0.12738985,
            -0.28535867,
            -0.33303016,
            -0.02690398,
            0.23215225,
            -0.03950802,
            -0.09100585,
        ];

        let bert_output_text2: Vec<f32> = bert_output_text2
            .iter()
            .map(|&x| round_to_decimal_places(x, 2))
            .collect();

        let bert_ground_truth_2: Vec<f32> = embeddings[768..778]
            .iter()
            .map(|&x| round_to_decimal_places(x, 2))
            .collect();

        assert_eq!(bert_ground_truth_1, bert_output_text1);
        assert_eq!(bert_ground_truth_2, bert_output_text2);
        assert_eq!(embeddings.len(), 1536);
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
                Value::String(0.to_string()),
                Value::String(1.to_string()),
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
        // let fields = vec![&"text".to_string(), &"title".to_string()];
        writer.write(&batch_info);
    }

    #[test]
    fn test_faiss_representation_writer() {
        let path = "test";
        let mut writer = FaissRepresentationWriter::new(path);
        writer.init_index(3, "Flat");
        writer.open_file();

        let mut batch_info = HashMap::new();
        batch_info.insert(
            "id",
            Value::Array(vec![
                Value::String(0.to_string()),
                Value::String(1.to_string()),
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
        writer.write(&batch_info);

        writer.save_index();

        assert_eq!(writer.index.is_trained(), true);
        assert_eq!(writer.index.ntotal(), 2);
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
