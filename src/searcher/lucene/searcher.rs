use j4rs::{ClasspathEntry, Instance, InvocationArg, JavaClass, Jvm, JvmBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct LuceneSearcher {
    pub num_docs: usize,
    jvm: Jvm,
    searcher: Instance,
    prebuilt_index_name: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct LuceneSearcherResult {
    pub docid: String,
    pub lucene_docid: i32,
    pub score: f32,
}

pub enum LuceneQuery {
    String(String),
    Instance(Instance),
}

impl LuceneSearcher {
    pub fn new(
        index_dir: impl Into<String>,
        prebuilt_index_name: Option<String>,
    ) -> Result<Self, anyhow::Error> {
        let entry = ClasspathEntry::new("resources/anserini-0.35.1-SNAPSHOT-fatjar.jar");
        let jvm: Jvm = JvmBuilder::new().classpath_entry(entry).build()?;

        let index_dir = InvocationArg::try_from(index_dir.into())?;

        let searcher =
        jvm.create_instance("io.anserini.search.SimpleSearcher", &[index_dir])?;

        let num_docs = jvm.invoke(&searcher, "get_total_num_docs", InvocationArg::empty())?;
        let num_docs: usize = jvm.to_rust(num_docs)?;

        Ok(Self {
            num_docs,
            jvm,
            searcher,
            prebuilt_index_name,
        })
    }

    pub fn search(
        &self,
        q: LuceneQuery,
        k: i32,
        _query_generator: Option<Instance>,
        fields: Option<HashMap<String, f32>>,
        _strip_segment_id: bool,
        _remove_dups: bool,
    ) -> Result<Vec<LuceneSearcherResult>, anyhow::Error> {
        let jfields: Option<Instance>;
        let hits: Vec<LuceneSearcherResult>;
        match fields {
            Some(fields) => {
                jfields = Some(self.jvm.java_map(
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
            LuceneQuery::String(q) => {
                println!("Query: {:?}", &q);
                let query_str = InvocationArg::try_from(q)?;
                let k = InvocationArg::try_from(k)?.into_primitive()?;

                if Option::is_some(&jfields) {
                    let results = self.jvm.invoke(
                        &self.searcher,
                        "search_fields",
                        &vec![query_str, jfields.unwrap().into(), k],
                    )?;
                    hits = self.jvm.to_rust(results)?;
                } else {
                    let results =
                        self.jvm
                            .invoke(&self.searcher, "search", &vec![query_str, k])?;
                    hits = self.jvm.to_rust(results)?;
                }
            }
            LuceneQuery::Instance(_q) => {
                // to be implemented
                hits = Vec::new();
            }
        }

        Ok(hits)
    }

    pub fn batch_search(
        &self,
        queries: Vec<String>,
        qids: Vec<String>,
        k: i32,
        threads: i32,
        _query_generator: Option<Instance>,
        fields: Option<HashMap<String, f32>>,
    ) -> Result<HashMap<String, Vec<LuceneSearcherResult>>, anyhow::Error> {
        let jfields: Option<Instance>;
        let hits: HashMap<String, Vec<LuceneSearcherResult>>;

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

        let query_strings = self.jvm_object.java_list(JavaClass::String, queries)?;
        let qid_strings = self.jvm_object.java_list(JavaClass::String, qids)?;
        let k = InvocationArg::try_from(k)?.into_primitive()?;
        let threads = InvocationArg::try_from(threads)?.into_primitive()?;

        if Option::is_some(&jfields) {
            let results = self.jvm_object.invoke(
                &self.searcher,
                "batch_search_fields",
                &vec![
                    query_strings.into(),
                    qid_strings.into(),
                    k,
                    threads,
                    jfields.unwrap().into(),
                ],
            )?;
            hits = self.jvm_object.to_rust(results)?;
        } else {
            let results = self.jvm_object.invoke(
                &self.searcher,
                "batch_search",
                &vec![query_strings.into(), qid_strings.into(), k, threads],
            )?;
            hits = self.jvm_object.to_rust(results)?;
        }

        Ok(hits)
    }
}
