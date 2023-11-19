struct DefaultQueryIterator {
    topics: HashMap<String, HashMap<String, String>>,
}

impl DefaultQueryIterator {
    pub fn new() {}

    pub fn get_query(&self, id: &str) -> Option<&String> {
        self.topics.get(id).get("title".to_string())
    }

    pub fn from_topics(&self) {}
}
