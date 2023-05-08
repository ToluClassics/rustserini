use std::path::PathBuf;
use tch::{jit, Device, Tensor};
use tokenizers::tokenizer::{Result, Tokenizer};

// fn main() {
//     let path = "example_directory";
//     let collection_path = PathBuf::from(path);
//     let fields: Vec<&str> = vec!["text", "title"];
//     let delimiter = "\t";
//     let mut iterator = JsonlCollectionIterator::new(path, Some(fields), delimiter);
//     iterator.load(&collection_path);
//     println!("{:?}", iterator.size);
//     println!("{:?}", iterator.iter().next());
//     println!("{:?}", iterator.iter().next());
//     println!("{:?}", iterator.iter().next());
// }

fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    let model_file = PathBuf::from("traced_bert.pt");
    let tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None)?;

    let encoding = tokenizer.encode(
        "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]",
        false,
    )?;
    // println!("{:?}", encoding.get_ids());

    let token_ids = encoding.get_ids();
    let i32_vec: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();

    let tokens_masks = encoding.get_attention_mask();
    let i32_mask: Vec<i32> = tokens_masks.iter().map(|&x| x as i32).collect();
    let input_mask = Tensor::of_slice(&i32_mask).unsqueeze(0).to(device);

    let input = Tensor::of_slice(&i32_vec).unsqueeze(0).to(device);

    // // Load the Python saved module.
    let model = jit::CModule::load(model_file).unwrap();

    let output = model.forward_ts(&[input, input_mask]).unwrap();
    let result_vector = Vec::<f32>::from(output);

    dbg!(result_vector);
    Ok(())
}
