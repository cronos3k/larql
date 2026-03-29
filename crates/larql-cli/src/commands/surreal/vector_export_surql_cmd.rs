use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_surreal::loader::{self, discover_vector_files, VectorReader};

#[derive(Args)]
pub struct VectorExportSurqlArgs {
    /// Directory containing .vectors.jsonl files from vector-extract.
    input: PathBuf,

    /// Output directory for .surql files (one per component).
    #[arg(short, long)]
    output: PathBuf,

    /// Tables to export (comma-separated). Default: all found in input dir.
    #[arg(long, value_delimiter = ',')]
    tables: Option<Vec<String>>,

    /// Layers to export (comma-separated). Default: all.
    #[arg(long, value_delimiter = ',')]
    layers: Option<Vec<usize>>,

    /// SurrealDB namespace (for USE statement in output).
    #[arg(long, default_value = "larql")]
    ns: String,

    /// SurrealDB database (for USE statement in output).
    #[arg(long, default_value = "gemma3_4b")]
    db: String,
}

pub fn run(args: VectorExportSurqlArgs) -> Result<(), Box<dyn std::error::Error>> {
    let all_files = discover_vector_files(&args.input)?;
    if all_files.is_empty() {
        return Err(format!("no .vectors.jsonl files in {}", args.input.display()).into());
    }

    let files: Vec<_> = match &args.tables {
        Some(tables) => all_files
            .into_iter()
            .filter(|(name, _)| tables.contains(name))
            .collect(),
        None => all_files,
    };

    let layer_filter: Option<std::collections::HashSet<usize>> =
        args.layers.map(|ls| ls.into_iter().collect());

    std::fs::create_dir_all(&args.output)?;

    eprintln!("Exporting to SurQL files:");
    eprintln!(
        "  tables: {}",
        files
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let overall_start = Instant::now();

    for (component, path) in &files {
        let start = Instant::now();
        let mut reader = VectorReader::open(path)?;
        let dimension = reader.header().dimension;

        let surql_path = args.output.join(format!("{component}.surql"));
        let file = std::fs::File::create(&surql_path)?;
        let mut writer = BufWriter::new(file);

        // Write header: OPTION IMPORT (required by surreal import) + USE + schema
        writeln!(writer, "OPTION IMPORT;")?;
        writeln!(writer, "USE NS {}; USE DB {};", args.ns, args.db)?;
        writeln!(writer)?;

        let schema = loader::schema_sql(component, dimension)?;
        writer.write_all(schema.as_bytes())?;
        writeln!(writer)?;

        let progress_schema = loader::progress_table_sql();
        writer.write_all(progress_schema.as_bytes())?;
        writeln!(writer)?;

        // Stream records
        let mut count = 0;
        let mut current_layer: Option<usize> = None;
        let mut layer_count = 0;

        while let Some(record) = reader.next_record()? {
            if let Some(ref filter) = layer_filter {
                if !filter.contains(&record.layer) {
                    continue;
                }
            }

            // Layer transition — write progress marker
            if current_layer.is_some() && current_layer != Some(record.layer) {
                if let Some(prev) = current_layer {
                    let sql = loader::mark_layer_done_sql(component, prev, layer_count);
                    writeln!(writer, "{sql}")?;
                    layer_count = 0;
                }
            }
            current_layer = Some(record.layer);

            let sql = loader::single_insert_sql(component, &record);
            writeln!(writer, "{sql}")?;
            count += 1;
            layer_count += 1;

            if count % 10000 == 0 {
                eprint!("\r  {component}: {count} records...");
            }
        }

        // Final layer progress
        if let Some(last_layer) = current_layer {
            if layer_count > 0 {
                let sql = loader::mark_layer_done_sql(component, last_layer, layer_count);
                writeln!(writer, "{sql}")?;
            }
        }

        writer.flush()?;
        let elapsed = start.elapsed();
        let size = std::fs::metadata(&surql_path)?.len();
        eprintln!(
            "\r  {component}: {count} records → {} ({:.1} GB, {:.0}s)",
            surql_path.display(),
            size as f64 / 1024.0 / 1024.0 / 1024.0,
            elapsed.as_secs_f64(),
        );
    }

    let elapsed = overall_start.elapsed();
    eprintln!("\nDone in {:.1}s", elapsed.as_secs_f64());
    eprintln!("\nImport into SurrealDB with:");
    eprintln!("  surreal import --conn http://localhost:8000 --ns {} --db {} --user root --pass root <file.surql>", args.ns, args.db);

    Ok(())
}
