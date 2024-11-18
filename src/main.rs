use serde_json::{Result, Value};

pub fn untyped_example() -> Result<()> {
    // Some JSON input data as a &str. Maybe this comes from the user.
    let data = r#"
        {
            "name": "John Doe",
            "age": 43,
            "phones": [
                "+44 1234567",
                "+44 2345678"
            ]
        }"#;

    // Parse the string of data into serde_json::Value.
    let v: Value = serde_json::from_str(data)?;

    // Access parts of the data by indexing with square brackets.
    println!("Please call {} at the number {}", v["names"], v["phones"][0]);

    Ok(())
}

fn main() {
    if let Err(e) = untyped_example() {
        eprintln!("error: {}", e);
    }
}