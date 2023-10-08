use j4rs::{ClasspathEntry, Instance, InvocationArg, JavaClass, Jvm, JvmBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct LuceneSearcher {
    // index_dir: InvocationArg,
    num_docs: Option<usize>,
    jvm_object: Jvm,
    searcher: Instance,
    prebuilt_index_name: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct SearcherResult {
    docid: String,
    lucene_docid: i32,
    score: f32,
    contents: String,
    raw: String,
    lucene_document: String,
}

pub enum Query {
    String(String),
    Instance(Instance),
}

impl LuceneSearcher {
    pub fn new(
        index_dir: String,
        prebuilt_index_name: Option<String>,
    ) -> Result<Self, anyhow::Error> {
        let index_dir = InvocationArg::try_from(index_dir).unwrap();
        let entry = ClasspathEntry::new("resources/anserini-0.20.1-SNAPSHOT-fatjar.jar");
        let jvm_object: Jvm = JvmBuilder::new().classpath_entry(entry).build()?;

        let searcher =
            jvm_object.create_instance("io.anserini.search.SimpleSearcher", &[index_dir])?;
        let num_docs = None;

        Ok(Self {
            num_docs,
            jvm_object,
            searcher,
            prebuilt_index_name,
        })
    }

    pub fn search(
        &self,
        q: Query,
        k: i8,
        query_generator: Option<Instance>,
        fields: Option<HashMap<String, f32>>,
        strip_segment_id: bool,
        remove_dups: bool,
    ) -> Result<Vec<SearcherResult>, anyhow::Error> {
        let jfields: Option<Instance>;
        let hits: Vec<SearcherResult>;
        match fields {
            Some(fields) => {
                jfields = Some(self.jvm_object.java_map(
                    JavaClass::String,
                    JavaClass::Float,
                    fields,
                )?);
            }
            None => {
                jfields = None;
            }
        }

        match q {
            Query::String(q) => {
                let query_str = InvocationArg::try_from(q)?;
                let results = self
                    .jvm_object
                    .invoke(&self.searcher, "search", &vec![query_str])?;
                hits = self.jvm_object.to_rust(results)?;
            }
            Query::Instance(q) => {
                let results = self.jvm_object.invoke(&self.searcher, "search", &[])?;
                hits = self.jvm_object.to_rust(results)?;
            }
        }

        Ok(hits)
    }

    pub fn check_num_docs(&self) -> Result<usize, anyhow::Error> {
        let num_docs = self
            .jvm_object
            .invoke(&self.searcher, "get_total_num_docs", &Vec::new())
            .unwrap();

        let num_docs: usize = self.jvm_object.to_rust(num_docs)?;

        Ok(num_docs)
    }
}
