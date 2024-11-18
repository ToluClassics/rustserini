use anyhow::Ok;
use j4rs::{ClasspathEntry, Instance, InvocationArg, JavaClass, Jvm, JvmBuilder};
use clap::{ArgAction, Parser};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args{

    /// Collection type to use in loading the document
    #[arg(short, long)]
    collection: String,

    /// Path to the input directory containing the files to be indexed
    #[arg(short, long)]
    input: String,

    /// Path to the output directory where the index will be stored
    #[arg(long)]
    index: String,

    /// Lucene Generator to use
    #[arg(short, long)]
    generator: String,

    /// Number of threads to use
    #[arg(short, long, default_value_t = 8)]
    threads: u8,

    /// Whether to store the term vectors
    #[arg(long)]
    store_positions: bool,

    /// Whether to store the term vectors
    #[arg(long)]
    store_docvectors: bool,

    /// Whether to store the raw documents
    #[arg(long)]
    store_raw: bool,
}

fn main() -> anyhow::Result<()>{

    let args = Args::parse();

    let entry = ClasspathEntry::new("resources/anserini-0.35.1-SNAPSHOT-fatjar.jar");
    let jvm: Jvm = JvmBuilder::new().classpath_entry(entry).build()?;

    let mut java_args = vec![];

    java_args.push(InvocationArg::try_from("-collection".to_owned())?);
    java_args.push(InvocationArg::try_from(&args.collection)?);

    java_args.push(InvocationArg::try_from("-input".to_owned())?);
    java_args.push(InvocationArg::try_from(&args.input)?);

    java_args.push(InvocationArg::try_from("-index".to_owned())?);
    java_args.push(InvocationArg::try_from(&args.index)?);

    java_args.push(InvocationArg::try_from("-generator".to_owned())?);
    java_args.push(InvocationArg::try_from(&args.generator)?);

    java_args.push(InvocationArg::try_from("-threads".to_owned())?);
    java_args.push(InvocationArg::try_from( &args.threads.to_string())?);

    if args.store_positions{
        java_args.push(InvocationArg::try_from("-storePositions")?);
    }

    if args.store_docvectors{
        java_args.push(InvocationArg::try_from("-storeDocvectors")?);
    }

    if args.store_raw{
        java_args.push(InvocationArg::try_from("-storeRaw")?);
    }

    let arr_instance = jvm.create_java_array("java.lang.String", &java_args)?;

    let indexer =
        jvm.create_instance("io.anserini.index.IndexCollection", InvocationArg::empty())?;

    jvm.invoke(&indexer, "main", &[InvocationArg::from(arr_instance)])?;


    println!("Starting JVM...");

    Ok(())

}