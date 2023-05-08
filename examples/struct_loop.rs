use std::collections::HashMap;

struct BatchIterator<'a> {
    data: &'a HashMap<String, Vec<u8>>,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = HashMap<String, Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let start_idx = self.current_idx;
        let end_idx = std::cmp::min(
            start_idx + self.batch_size,
            self.data.values().map(|v| v.len()).min().unwrap_or(0),
        );
        if start_idx >= end_idx {
            return None;
        }
        self.current_idx = end_idx;
        Some(
            self.data
                .iter()
                .map(|(k, v)| (k.clone(), v[start_idx..end_idx].to_vec()))
                .collect(),
        )
    }
}

fn main() {
    let mut data = HashMap::new();
    data.insert(String::from("a"), vec![1, 2, 3, 4, 5]);
    data.insert(String::from("b"), vec![6, 7, 8, 9]);
    data.insert(String::from("c"), vec![10, 11, 12]);

    let batch_size = 2;
    let mut iter = BatchIterator {
        data: &data,
        batch_size: batch_size,
        current_idx: 0,
    };

    while let Some(batch) = iter.next() {
        println!("{:?}", batch);
    }
}
