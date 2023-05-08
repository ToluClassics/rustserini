use rustserini::encode::vector_writer::JsonlCollectionIterator;
use std::path::PathBuf;

fn main() {
    let path = "example_directory";
    let collection_path = PathBuf::from(path);
    let fields: Vec<&str> = vec!["text", "title"];
    let delimiter = "\t";
    let mut iterator = JsonlCollectionIterator::new(path, Some(fields), delimiter);
    iterator.load(&collection_path);
    println!("{:?}", iterator.size);
    println!("{:?}", iterator.iter().next());
    println!("{:?}", iterator.iter().next());
    println!("{:?}", iterator.iter().next());
}
