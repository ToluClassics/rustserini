use j4rs::{Instance, JavaClass, Jvm};
use std::collections::HashMap;

struct LuceneSearcher {
    index_dir: InvocationArg,
    num_docs: usize,
    object: Jvm,
    searcher: Instance,
    prebuilt_index_name: Option<String>,
}

pub enum Query {
    String,
    Instance,
}

impl LuceneSearcher {
    pub fn search(
        &self,
        q: Query,
        k: i8,
        query_generator: Option<Instance>,
        fields: Option<HashMap>,
        strip_segment_id: bool,
        remove_dups: bool,
    ) -> Result<Vec<Result>, anyhow::Error> {
        let jfields: Instance;
        let hits: Vec<Result>;
        match fields {
            Some(fields) => {
                jfields = jvm.java_map(JavaClass::String, JavaClass::Float, fields)?;
            }
            None => {
                jfields = None;
            }
        }

        match q {
            Query::String => {
                let query_str = InvocationArg::from(q)?;
                let results = self.jvm.invoke(&searcher, "search", &vec![query_str])?;
                hits = self.jvm.to_rust(results)?;
            }
        }
    }
}
