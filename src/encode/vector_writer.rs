use crate::encode::base::RepresentationWriter;
use serde_json::{json, Map, Number, Value};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

pub struct JsonlRepresentationWriter {
    dir_path: PathBuf,
    filename: String,
    file: Option<std::fs::File>,
}

impl RepresentationWriter for JsonlRepresentationWriter {
    // Write a representation to a file
    fn write(
        self: &JsonlRepresentationWriter,
        batch_info: &HashMap<&str, Value>,
        fields: &Vec<std::string::String>,
    ) {
        let mut file = match &self.file {
            Some(file) => file,
            None => {
                panic!("File is not open for writing!");
            }
        };

        let batch_id: Vec<Value> = match batch_info["id"] {
            Value::Array(ref batch_id) => batch_id.clone(),
            _ => panic!("Invalid batch id type!"),
        };
        for i in 0..batch_id.len() {
            let contents = batch_info["text"][i].clone();
            let vector = &batch_info["vector"][i];
            let record = json!({
                "id": batch_info["id"][i],
                "contents": contents,
                "vector": vector,
            });
            writeln!(file, "{}", record).unwrap();
        }
    }

    // Create a new instance of a RepresentationWriter
    fn new(path: &str) -> JsonlRepresentationWriter {
        let dir_path = PathBuf::from(path);
        let filename = "embeddings.jsonl".to_string();
        let file = None;

        JsonlRepresentationWriter {
            dir_path,
            filename,
            file,
        }
    }

    // Open File
    fn open_file(&mut self) {
        if !self.dir_path.exists() {
            std::fs::create_dir_all(&self.dir_path).unwrap();
        }

        let file_path = self.dir_path.join(&self.filename);
        let file = std::fs::File::create(file_path).unwrap();
        self.file = Some(file);
    }
}
