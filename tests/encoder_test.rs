#[cfg(test)]
mod tests {
    use faiss::Index;
    use rustserini::encode::auto::AutoDocumentEncoder;
    use rustserini::encode::base::{DocumentEncoder, RepresentationWriter};
    use rustserini::encode::vector_writer::FaissRepresentationWriter;
    use rustserini::encode::vector_writer::{JsonlCollectionIterator, JsonlRepresentationWriter};
    use std::collections::HashMap;
    use std::time::Instant;

    fn round_to_decimal_places(n: f32, places: u32) -> f32 {
        let multiplier: f32 = 10f32.powi(places as i32);
        (n * multiplier).round() / multiplier
    }

    #[test]
    fn test_auto_document_encoder_cls_pooling() -> anyhow::Result<()> {
        let model_name = "bert-base-uncased";
        let revision = "refs/pr/70";
        let document_encoder: AutoDocumentEncoder =
            AutoDocumentEncoder::new(model_name, revision);
        let start = Instant::now();

        let texts = vec![
            "Hello, I am a sentence!".to_string(),
            "And another sentence.".to_string(),
        ];
        let titles = vec!["Title 1".to_string(), "Title 2".to_string()];
        let embeddings = document_encoder.encode(&texts, Some(&titles), "cls")?;

        let embeddings = embeddings.flatten_all()?;
        let embeddings = embeddings.to_vec1()?;

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
        let duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);

        Ok(())
    }

    #[test]
    fn test_mdpr_document_encoder_cls_pooling() -> anyhow::Result<()> {
        let model_name = "castorini/mdpr-tied-pft-msmarco-ft-miracl-zh";
        let revision = "refs/pr/1";
        let document_encoder: AutoDocumentEncoder =
            AutoDocumentEncoder::new(model_name, revision);

        let texts = vec![
            "Hello, I am a sentence!".to_string(),
            "And another sentence.".to_string(),
        ];
        let titles = vec![
            "Title 1".to_string(),
            "Title 2".to_string()
            ];
        let embeddings = document_encoder.encode(&texts, Some(&titles), "cls")?;

        let embeddings = embeddings.flatten_all()?;
        let embeddings = embeddings.to_vec1()?;

        let bert_output_text1: Vec<f32> = vec![
            0.0935,
            0.1650,
            -0.2026,
            0.0945,
            0.0554,
            0.1779,
            -0.1653,
            -0.1215,
            0.0335,
            0.5885];

        let bert_output_text1: Vec<f32> = bert_output_text1
            .iter()
            .map(|&x| round_to_decimal_places(x, 4))
            .collect();

        let bert_ground_truth_1: Vec<f32> = embeddings[0..10]
            .iter()
            .map(|&x| round_to_decimal_places(x, 4))
            .collect();

        let bert_output_text2: Vec<f32> = vec![-0.0892,
            -0.0782,
            -0.1189,
            0.3578,
            0.2832,
            0.1035,
            -0.2080,
            -0.1795,
            -0.0405,
            0.2710];

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

        Ok(())
    }

    #[test]
    fn test_json_representation_writer() -> anyhow::Result<()> {
        let path = "test";
        let mut writer = JsonlRepresentationWriter::new(path, 3);
        let _ = writer.open_file();
        let mut batch_info = HashMap::new();
        batch_info.insert("id", vec!["0".to_string(), "1".to_string()]);
        batch_info.insert(
            "text",
            vec![
                "Hello, I am a sentence!".to_string(),
                "Hello, I am a sentences!".to_string(),
            ],
        );
        batch_info.insert(
            "title",
            vec![
                "Hello, I am a sentence!".to_string(),
                "Hello, I am a sentences!".to_string(),
            ],
        );

        let mut embeddings: Vec<f32> = vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3];
        let _ = writer.write(&batch_info, &mut embeddings);

        Ok(())
    }

    #[test]
    fn test_faiss_representation_writer() -> anyhow::Result<()> {
        let path = "test";
        let mut writer = FaissRepresentationWriter::new(path, 3);
        let _ = writer.init_index(3, "Flat");
        let _ = writer.open_file();
    
        let mut batch_info = HashMap::new();
        batch_info.insert("id", vec!["0".to_string(), "1".to_string()]);
        batch_info.insert(
            "text",
            vec![
                "Hello, I am a sentence!".to_string(),
                "Hello, I am a sentences!".to_string(),
            ],
        );
        batch_info.insert(
            "title",
            vec![
                "Hello, I am a sentence!".to_string(),
                "Hello, I am a sentences!".to_string(),
            ],
        );
    
        let mut embeddings: Vec<f32> = vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3];
        let _ = writer.write(&batch_info, &mut embeddings);
    
        let _ = writer.save_index();
    
        assert_eq!(writer.index.is_trained(), true);
        assert_eq!(writer.index.ntotal(), 2);
    
        Ok(())
    }

    #[test]
    fn test_jsonl_collection_iterator() -> anyhow::Result<()> {
        let path = "tests/test_files".to_string();
        let fields: Vec<String> =
            vec!["docid".to_string(), "text".to_string(), "title".to_string()];
        let delimiter = "\t".to_string();
        let batch_size = 8;
        let mut iterator =
            JsonlCollectionIterator::new(fields, "docid".to_string(), delimiter, batch_size);

        let _ = iterator.load_compressed(path);
        assert_eq!(iterator.size, 10);
        assert_eq!(iterator.all_info.docid.len(), 10);

        assert_eq!(
            iterator.iter().next().unwrap()["title"][0],
            String::from("Introduction")
        );

        Ok(())
    }
}
