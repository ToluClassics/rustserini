use crate::encode::base::RepresentationWriter;
use anyhow::Ok;
use faiss::index::io::write_index;
use faiss::index::IndexImpl;
use faiss::{index_factory, Index, MetricType};
use flate2::read::GzDecoder;
use kdam::tqdm;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// JsonlRepresentationWriter is a struct that writes for writing embeddings to a jsonl file
/// It is designed to be a parallel of this Python Class
/// https://github.com/castorini/pyserini/blob/45edec7e618db621339958c89fdff1d4a7a8cb90/pyserini/encode/_base.py#L162
pub struct JsonlRepresentationWriter {
    dir_path: PathBuf,
    filename: String,
    file: Option<std::fs::File>,
    pub dimension: u32,
}

/// JsonlCollectionIterator is a struct created for iterating over the items in a jsonl file
/// It is designed to be a parallel of this Python Class
/// https://github.com/castorini/pyserini/blob/45edec7e618db621339958c89fdff1d4a7a8cb90/pyserini/encode/_base.py#L59
pub struct JsonlCollectionIterator {
    fields: Vec<String>,
    docid_field: String,
    delimiter: String,
    batch_size: usize,
    pub size: usize,
    shard_id: usize,
    shard_num: usize,
    pub all_info: AllInfo,
}

#[derive(Serialize, Deserialize, Debug)]
struct DataFields {
    docid: String,
    text: String,
    title: String,
}

#[records::record]
pub struct AllInfo {
    docid: Vec<String>,
    texts: Vec<String>,
    titles: Vec<String>,
}

/// FaissRepresentationWriter is a struct that writes for writing embeddings to a faiss index
pub struct FaissRepresentationWriter {
    pub dir_path: PathBuf,
    index_name: String,
    file_name: String,
    pub dimension: u32,
    pub index: IndexImpl,
    file: Option<std::fs::File>,
    pub docids: Vec<String>,
}

///jsonl_collection_iterator is a struct created for iterating over the items in a jsonl file
impl JsonlCollectionIterator {
    pub fn new(
        fields: Vec<String>,
        docid_field: String,
        delimiter: String,
        batch_size: usize,
    ) -> JsonlCollectionIterator {
        let fields = fields;
        let docid_field = docid_field;
        let delimiter = delimiter;
        let batch_size = batch_size;
        let all_info = AllInfo::new(Vec::new(), Vec::new(), Vec::new());
        let size = 0;
        let shard_id = 0;
        let shard_num = 1;

        JsonlCollectionIterator {
            fields,
            docid_field,
            delimiter,
            batch_size,
            size,
            shard_id,
            shard_num,
            all_info,
        }
    }

    pub fn load(&mut self, collection_path: String) -> Result<(), anyhow::Error> {
        let mut filenames = Vec::new();
        let collection_path = Path::new(&collection_path);

        if collection_path.is_file() {
            filenames.push(collection_path.to_path_buf());
        } else {
            for filename in std::fs::read_dir(collection_path)? {
                let filename = filename?.path();

                if filename.is_file() {
                    filenames.push(filename);
                }
            }
        }

        let mut all_doc_ids: Vec<String> = Vec::new();
        let mut all_texts: Vec<String> = Vec::new();
        let mut all_titles: Vec<String> = Vec::new();

        for filename in filenames {
            println!("Loading file: {:?}", &filename);

            let file = File::open(filename)?;
            let reader = BufReader::new(file);
            let lines = reader.lines();

            for line in tqdm!(lines) {
                let line = line?;
                let json: DataFields = serde_json::from_str(&line)?;

                let docid = &json.docid;
                all_doc_ids.push(docid.to_string());

                for field in self.fields.to_vec() {
                    match field.as_str() {
                        "text" => {
                            let value = &json.text;
                            all_texts.push(value.to_string());
                        }
                        "title" => {
                            let value = &json.title;
                            all_titles.push(value.to_string());
                        }
                        _ => {}
                    }
                }
            }
        }

        println!("Loaded {} documents", all_doc_ids.len());
        self.size = all_doc_ids.len();
        self.all_info = AllInfo::new(all_doc_ids, all_texts, all_titles);

        Ok(())
    }

    pub fn load_compressed(&mut self, collection_path: String) -> Result<(), anyhow::Error> {
        let mut filenames = Vec::new();
        let collection_path = Path::new(&collection_path);

        if collection_path.is_file() {
            filenames.push(collection_path.to_path_buf());
        } else {
            for filename in std::fs::read_dir(collection_path)? {
                let filename = filename?.path();

                if filename.is_file() {
                    filenames.push(filename);
                }
            }
        }

        let mut all_doc_ids: Vec<String> = Vec::new();
        let mut all_texts: Vec<String> = Vec::new();
        let mut all_titles: Vec<String> = Vec::new();

        for filename in filenames {
            println!("Loading file: {:?}", &filename);

            let file = File::open(filename)?;

            let gz = GzDecoder::new(file);
            let reader = BufReader::new(gz);
            let lines = reader.lines();

            for line in tqdm!(lines) {
                let line = line?;
                let json: DataFields = serde_json::from_str(&line)?;

                let docid = &json.docid;
                all_doc_ids.push(docid.to_string());

                for field in self.fields.to_vec() {
                    match field.as_str() {
                        "text" => {
                            let value = &json.text;
                            all_texts.push(value.to_string());
                        }
                        "title" => {
                            let value = &json.title;
                            all_titles.push(value.to_string());
                        }
                        _ => {}
                    }
                }
            }
        }

        println!("Loaded {} documents", all_doc_ids.len());
        self.size = all_doc_ids.len();
        self.all_info = AllInfo::new(all_doc_ids, all_texts, all_titles);
        Ok(())
    }

    pub fn iter(&mut self) -> impl Iterator<Item = HashMap<&str, Vec<String>>> {
        let total_len = self.size;
        let shard_size = total_len / self.shard_num;
        let start_idx = self.shard_id * shard_size;
        let end_idx = if self.shard_id == self.shard_num - 1 {
            total_len
        } else {
            start_idx + shard_size
        };

        (start_idx..end_idx)
            .step_by(self.batch_size)
            .map(move |idx| {
                let mut batch_info = HashMap::new();
                let batch_docid: Vec<String> = self.all_info.docid[idx..idx + self.batch_size]
                    .iter()
                    .map(|x| x.to_string())
                    .collect();
                let batch_text: Vec<String> = self.all_info.texts[idx..idx + self.batch_size]
                    .iter()
                    .map(|x| x.to_string())
                    .collect();
                let batch_title: Vec<String> = self.all_info.titles[idx..idx + self.batch_size]
                    .iter()
                    .map(|x| x.to_string())
                    .collect();

                batch_info.insert("id", batch_docid);
                batch_info.insert("text", batch_text);
                batch_info.insert("title", batch_title);

                batch_info
            })
    }
}

impl RepresentationWriter for JsonlRepresentationWriter {
    // Write a representation to a file
    fn write(
        &mut self,
        batch_info: &HashMap<&str, Vec<String>>,
        embeddings: &mut Vec<f32>,
    ) -> Result<(), anyhow::Error> {
        let mut file = match &self.file {
            Some(file) => file,
            None => {
                panic!("File is not open for writing!");
            }
        };

        let batch_id: &Vec<String> = &batch_info["id"];
        let embeddings_value: Vec<_> = embeddings.chunks(self.dimension as usize).collect();
        let embeddings: Vec<Vec<f32>> = embeddings_value.iter().map(|x| x.to_vec()).collect();

        for i in 0..batch_id.len() {
            let contents = batch_info["text"][i].clone();
            let vector = &embeddings[i];
            let record = json!({
                "id": batch_info["id"][i],
                "contents": contents,
                "vector": vector,
            });
            writeln!(file, "{}", record);
        }

        Ok(())
    }

    // Create a new instance of a RepresentationWriter
    fn new(path: &str, dimension: u32) -> JsonlRepresentationWriter {
        let dir_path = PathBuf::from(path);
        let filename = "embeddings.jsonl".to_string();
        let file = None;

        JsonlRepresentationWriter {
            dir_path,
            filename,
            file,
            dimension,
        }
    }

    // Open File
    fn open_file(&mut self) -> Result<(), anyhow::Error> {
        if !self.dir_path.exists() {
            std::fs::create_dir_all(&self.dir_path)?;
        }

        let file_path = self.dir_path.join(&self.filename);
        let file = std::fs::File::create(file_path)?;
        self.file = Some(file);

        Ok(())
    }

    fn save_index(&mut self) -> Result<(), anyhow::Error> {
        panic!("Not implemented!");
    }

    fn init_index(&mut self, _dim: u32, _index_type: &str) {
        panic!("Not implemented!");
    }

    fn save_docids(&mut self) -> Result<(), anyhow::Error> {
        panic!("Not implemented!");
    }
}

impl Default for FaissRepresentationWriter {
    fn default() -> Self {
        Self {
            dir_path: PathBuf::from("corpus"),
            index_name: String::from("index"),
            file_name: String::from("docid"),
            dimension: 768,
            index: index_factory(768, "Flat", MetricType::InnerProduct).unwrap(),
            file: None,
            docids: Vec::new(),
        }
    }
}

impl RepresentationWriter for FaissRepresentationWriter {
    // Create a new instance of a RepresentationWriter
    fn new(path: &str, dimension: u32) -> Self {
        let dir_path = PathBuf::from(path);
        if !dir_path.exists() {
            std::fs::create_dir_all(&dir_path).unwrap();
        }

        Self {
            dir_path: dir_path,
            dimension: dimension,
            ..Default::default()
        }
    }

    fn init_index(&mut self, dim: u32, index_type: &str) {
        self.dimension = dim;
        self.index = index_factory(dim, index_type, MetricType::InnerProduct).unwrap();
    }

    fn write(
        &mut self,
        _batch_info: &HashMap<&str, Vec<String>>,
        embeddings: &mut Vec<f32>,
    ) -> Result<(), anyhow::Error> {
        let embeddings = embeddings.as_mut_slice();
        self.index.add(embeddings).unwrap();

        Ok(())
    }

    // Open File
    fn open_file(&mut self) -> Result<(), anyhow::Error> {
        if !self.dir_path.exists() {
            std::fs::create_dir_all(&self.dir_path)?;
        }

        let file_path = self.dir_path.join(&self.file_name);
        let file = std::fs::File::create(file_path)?;
        self.file = Some(file);

        Ok(())
    }

    fn save_index(&mut self) -> Result<(), anyhow::Error> {
        let index_file_path: PathBuf = self.dir_path.join(&self.index_name);
        write_index(&self.index, index_file_path.as_path().display().to_string()).unwrap();

        Ok(())
    }

    fn save_docids(&mut self) -> Result<(), anyhow::Error> {
        let mut file = match &self.file {
            Some(file) => file,
            None => {
                panic!("File is not open for writing!");
            }
        };

        for docid in &self.docids {
            writeln!(file, "{}", docid)?;
        }

        Ok(())
    }
}
