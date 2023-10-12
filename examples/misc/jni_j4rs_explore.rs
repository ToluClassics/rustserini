use j4rs::{ClasspathEntry, InvocationArg, Jvm, JvmBuilder};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
struct Result {
    docid: String,
    // lucene_docid: i32,
    score: f32,
    // contents: String,
    raw: String,
    // lucene_document: String,
}

// public static class Result {
//     public String docid;
//     public int lucene_docid;
//     public float score;
//     public String contents;
//     public String raw;
//     public Document lucene_document;

//     public Result(String docid, int lucene_docid, float score, String contents, String raw, Document lucene_document) {
//       this.docid = docid;
//       this.lucene_docid = lucene_docid;
//       this.score = score;
//       this.contents = contents;
//       this.raw = raw;
//       this.lucene_document = lucene_document;
//     }
//   }

fn main() {
    let entry = ClasspathEntry::new(
        "/Users/mac/Documents/castorini/anserini/target/anserini-0.20.1-SNAPSHOT-fatjar.jar",
    );
    let jvm: Jvm = JvmBuilder::new().classpath_entry(entry).build().unwrap();

    let index_str = InvocationArg::try_from(
        "/Users/mac/Documents/castorini/anserini/indexes/msmarco-passage/lucene-index-msmarco",
    )
    .unwrap();

    let searcher = jvm
        .create_instance(
            "io.anserini.search.SimpleSearcher", // The Java class to create an instance for
            &[index_str], // The `InvocationArg`s to use for the constructor call - empty for this example
        )
        .unwrap();

    let num_docs = jvm
        .invoke(
            &searcher,            // The String instance created above
            "get_total_num_docs", // The method of the String instance to invoke
            &Vec::new(), // The `InvocationArg`s to use for the invocation - empty for this example
        )
        .unwrap();

    let num_docs: usize = jvm.to_rust(num_docs).unwrap();
    println!("Total number of docs: {:?}", num_docs);

    let query_str =
        InvocationArg::try_from("the presence of communication amid scientific minds").unwrap();

    // let k = InvocationArg::try_from(10).unwrap();

    let results = jvm.invoke(&searcher, "search", &vec![query_str]).unwrap();

    let results: Vec<Result> = jvm.to_rust(results).unwrap();

    println!("Results docid: {:?}", results[0].docid);
    println!("Results raw: {:?}", results[0].raw);
    println!("Results score: {:?}", results[0].score);

    // let docid = InvocationArg::try_from("0").unwrap();

    // let result = jvm.invoke(&searcher, "doc_raw", &[docid]).unwrap();
    // let result: String = jvm.to_rust(result).unwrap();

    // println!("Result: {:?}", result);
}

// use jni::objects::{JClass, JObject, JObjectArray, JString, JValue};
// use jni::sys::{jobject, jobjectArray};
// use jni::{InitArgsBuilder, JNIEnv, JNIVersion, JavaVM};

// fn main() {
//     let jvm_args = InitArgsBuilder::new()
//         .version(JNIVersion::V8)
//         .option("-Djava.class.path=/Users/mac/Documents/castorini/anserini/target/anserini-0.20.1-SNAPSHOT-fatjar.jar")
//         .build()
//         .unwrap();
//     let jvm = JavaVM::new(jvm_args).unwrap();

//     let mut env = jvm.attach_current_thread().unwrap();

//     let index_dir = env
//         .new_string(
//             "/Users/mac/Documents/castorini/anserini/indexes/msmarco-passage/lucene-index-msmarco",
//         )
//         .unwrap();
//     let index_dir_obj = JObject::from(index_dir);

//     let result_class = env
//         .find_class("io/anserini/search/SimpleSearcher$Result")
//         .unwrap();

//     let searcher_class = env.find_class("io/anserini/search/SimpleSearcher").unwrap();

//     let searcher_instance = env
//         .new_object(
//             &searcher_class,
//             "(Ljava/lang/String;)V",
//             &[JValue::Object(&index_dir_obj)],
//         )
//         .unwrap();

//     let hits = env
//         .call_method(&searcher_instance, "get_total_num_docs", "()I", &[])
//         .unwrap();

//     println!("Total number of docs: {}", hits.i().unwrap());

//     let query = env
//         .new_string("treating tension headaches without medication?")
//         .unwrap();

//     let results = env
//         .call_method(
//             &searcher_instance,
//             "search",
//             "(Ljava/lang/String;)[Lio/anserini/search/SimpleSearcher$Result;",
//             &[JValue::Object(&query)],
//         )
//         .unwrap()
//         .l()
//         .unwrap();

//     let res = env
//         .get_object_array_element(JObjectArray::from(results), 0)
//         .unwrap();

//     let contents = env
//         .get_field(res, "contents", "Ljava/lang/String;")
//         .unwrap()
//         .l()
//         .unwrap();

//     let contents_str: String = env.get_string(&JString::from(contents)).unwrap().into();
//     // let contents_string = *contents_str.to_string();

//     println!("Results list: {:?}", contents_str);

//     // let result = env.get_object_array_element(res, 0).unwrap();
//     // let contents = env.get_field(result, "contents", "Ljava/lang/String;");
// }
