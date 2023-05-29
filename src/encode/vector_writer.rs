use crate::encode::base::RepresentationWriter;
use flate2::read::GzDecoder;
use kdam::tqdm;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub struct JsonlRepresentationWriter {
    dir_path: PathBuf,
    filename: String,
    file: Option<std::fs::File>,
}

pub struct JsonlCollectionIterator<'a> {
    fields: Vec<&'a str>,
    delimiter: &'a str,
    pub all_info: HashMap<&'a str, Value>,
    pub size: usize,
    batch_size: &'a usize,
    shard_id: usize,
    shard_num: usize,
    collection_path: &'a str,
}

impl RepresentationWriter for JsonlRepresentationWriter {
    // Write a representation to a file
    fn write(
        self: &JsonlRepresentationWriter,
        batch_info: &HashMap<&str, Value>,
        _fields: &Vec<std::string::String>,
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

impl<'a> JsonlCollectionIterator<'a> {
    pub fn new(
        _collection_path: &'a str,
        fields: Option<Vec<&'a str>>,
        delimiter: &'a str,
        batch_size: &'a usize,
    ) -> Self {
        let fields = match fields {
            Some(f) => f,
            None => vec!["text"],
        };
        let all_info: HashMap<&'a str, Value> = HashMap::new();
        let size = 0;

        Self {
            fields,
            delimiter,
            all_info,
            size,
            batch_size,
            shard_id: 0,
            shard_num: 1,
            collection_path: _collection_path,
        }
    }

    pub fn load(&mut self) {
        let mut filenames = Vec::new();
        let collection_path = Path::new(&self.collection_path);

        if collection_path.is_file() {
            filenames.push(collection_path.to_path_buf());
        } else {
            for filename in std::fs::read_dir(collection_path).unwrap() {
                let filename = filename.unwrap().path();

                if filename.is_file() {
                    filenames.push(filename);
                }
            }
        }
        let mut all_info: HashMap<&str, Value> = HashMap::new();

        all_info.insert("id", Value::Array(Vec::new()));
        for key in self.fields.iter() {
            all_info.insert(key, Value::Array(Vec::new()));
        }
        let mut counter: usize = 0;
        for filename in filenames {
            let file = File::open(filename.clone()).map_err(|err| err.to_string());
            let gz = GzDecoder::new(file.unwrap());
            let reader = BufReader::new(gz);
            let lines = reader
                .lines()
                .map(|line| line.map_err(|err| err.to_string()));

            for (line_i, line) in tqdm!(lines.enumerate()) {
                let line = line.map_err(|err| err.to_string());
                let info: serde_json::Value =
                    serde_json::from_str(&line.unwrap().to_string()).unwrap();
                let id: String = match info.get("id") {
                    Some(id_val) => id_val.to_string(),
                    None => {
                        match info.get("docid") {
                            Some(docid_val) => docid_val.to_string(),
                            None => {
                                println!("No id or docid found at Line#{:?} in file {}. Line content: {}", line_i, filename.to_str().unwrap(), info.get("contents").unwrap().to_string());
                                continue;
                            }
                        }
                    }
                };
                let fields_info = self.parse_fields_from_info(&info);
                all_info
                    .get_mut("id")
                    .unwrap()
                    .as_array_mut()
                    .unwrap()
                    .push(Value::String(id.to_string()));

                for (i, key) in self.fields.iter().enumerate() {
                    all_info
                        .get_mut(key)
                        .unwrap()
                        .as_array_mut()
                        .unwrap()
                        .push(Value::String(fields_info.clone().unwrap()[i].to_string()));
                }
                counter += 1;
            }
        }
        self.size = counter;
        self.all_info = all_info;
    }

    pub fn iter(&mut self) -> impl Iterator<Item = HashMap<&'a str, Vec<Value>>> + '_ {
        let total_len = self.size;
        let shard_size = total_len / self.shard_num;
        let start_idx = self.shard_id * shard_size;
        let end_idx = if self.shard_id == self.shard_num - 1 {
            total_len
        } else {
            start_idx + shard_size
        };
        let mut to_yield = HashMap::new();

        (start_idx..end_idx)
            .step_by(*self.batch_size)
            .map(move |idx| {
                for (key, value) in self.all_info.clone() {
                    to_yield.insert(
                        key,
                        value.as_array().unwrap()
                            [idx..std::cmp::min(idx + self.batch_size, end_idx)]
                            .to_vec(),
                    );
                }
                to_yield.clone()
            })
    }

    fn parse_fields_from_info(&self, info: &serde_json::Value) -> Result<Vec<String>, String> {
        /// Parse fields from info
        ///
        let n_fields = self.fields.len();

        if self.fields.iter().all(|&field| info.get(field).is_some()) {
            let fields_info: Result<Vec<String>, String> = self
                .fields
                .iter()
                .map(|&field| Ok(info.get(field).unwrap().to_string().trim().to_string()))
                .collect();
            return fields_info;
        }

        let contents = info
            .get("text")
            .ok_or(format!("'contents' not found in info: {}", info))?;
        let contents_str = contents.to_string();
        let mut contents_vec: Vec<&str> = contents_str.split(self.delimiter).collect();
        if contents_vec.len() == n_fields + 1 && contents_vec[n_fields].is_empty() {
            contents_vec.truncate(n_fields);
        }
        if contents_vec.len() != n_fields {
            return Err(format!(
                "{} fields are found in the input contents, but {} fields are expected.",
                contents_vec.len(),
                n_fields
            ));
        }

        let fields_info: Vec<String> = contents_vec
            .iter()
            .map(|&field| field.trim().to_string())
            .collect();

        Ok(fields_info)
    }
}
