use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use crate::utils::base64_encode;
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_surreal::loader::{self, discover_vector_files, VectorReader};

#[derive(Args)]
pub struct VectorImportArgs {
    /// Directory containing .vectors.jsonl files from vector-extract.
    input: PathBuf,

    /// Tables to import (comma-separated). Default: all found in input dir.
    #[arg(long, value_delimiter = ',')]
    tables: Option<Vec<String>>,

    /// Layers to import (comma-separated). Default: all.
    #[arg(long, value_delimiter = ',')]
    layers: Option<Vec<usize>>,

    /// SurrealDB endpoint.
    #[arg(long, default_value = "http://localhost:8000")]
    endpoint: String,

    /// SurrealDB namespace.
    #[arg(long)]
    ns: String,

    /// SurrealDB database.
    #[arg(long)]
    db: String,

    /// SurrealDB username.
    #[arg(long, default_value = "root")]
    user: String,

    /// SurrealDB password.
    #[arg(long, default_value = "root")]
    pass: String,

    /// Records per batch file. Each batch is imported separately.
    #[arg(long, default_value = "5000")]
    batch_size: usize,

    /// Resume from previous import (skips completed layers).
    #[arg(long)]
    resume: bool,
}

pub fn run(args: VectorImportArgs) -> Result<(), Box<dyn std::error::Error>> {
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
        args.layers.as_ref().map(|ls| ls.iter().copied().collect());

    // Check surreal CLI exists
    let surreal_check = Command::new("surreal").arg("version").output();
    if surreal_check.is_err() {
        return Err("surreal CLI not found. Install from https://surrealdb.com/install".into());
    }

    let tmp_dir = std::env::temp_dir().join("larql_import");
    std::fs::create_dir_all(&tmp_dir)?;

    eprintln!("Importing to SurrealDB: {}", args.endpoint);
    eprintln!("  ns={}, db={}", args.ns, args.db);
    eprintln!(
        "  tables: {}",
        files
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("  batch_size: {}", args.batch_size);

    // First: create schema via surreal import
    let schema_path = tmp_dir.join("_schema.surql");
    {
        let mut f = BufWriter::new(std::fs::File::create(&schema_path)?);
        writeln!(f, "OPTION IMPORT;")?;
        writeln!(f, "USE NS {}; USE DB {};", args.ns, args.db)?;
        writeln!(f)?;
        for (component, path) in &files {
            let reader = VectorReader::open(path)?;
            let dim = reader.header().dimension;
            let schema = loader::schema_sql(component, dim)?;
            f.write_all(schema.as_bytes())?;
            writeln!(f)?;
        }
        let progress = loader::progress_table_sql();
        f.write_all(progress.as_bytes())?;
        f.flush()?;
    }
    eprintln!("\n  Creating schema...");
    run_surreal_import(&args, &schema_path)?;
    std::fs::remove_file(&schema_path)?;
    eprintln!("  Schema ready.");

    let overall_start = Instant::now();
    let mut total_imported = 0usize;

    for (component, path) in &files {
        let comp_start = Instant::now();

        // Check completed layers for resume
        let completed_layers: std::collections::HashSet<usize> = if args.resume {
            query_completed_layers(
                &args.endpoint,
                &args.ns,
                &args.db,
                &args.user,
                &args.pass,
                component,
            )
            .unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        };

        if !completed_layers.is_empty() {
            eprintln!(
                "  {component}: resuming ({} layers already loaded)",
                completed_layers.len()
            );
        }

        // Count records for progress bar
        let total_records = {
            let f = std::fs::File::open(path)?;
            let r = std::io::BufReader::new(f);
            r.lines().count().saturating_sub(1)
        };

        let bar = ProgressBar::new(total_records as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "  {component}: {{bar:40}} {{pos}}/{{len}} ({{eta}} remaining) {{msg}}"
                ))
                .unwrap(),
        );

        let mut reader = VectorReader::open(path)?;
        let mut batch: Vec<String> = Vec::new();
        let mut batch_num = 0usize;
        let mut comp_count = 0usize;
        let mut current_layer: Option<usize> = None;
        let mut layer_count = 0usize;
        let mut skipped = 0usize;

        while let Some(record) = reader.next_record()? {
            // Layer filter
            if let Some(ref filter) = layer_filter {
                if !filter.contains(&record.layer) {
                    continue;
                }
            }
            // Resume: skip completed layers
            if completed_layers.contains(&record.layer) {
                skipped += 1;
                bar.set_position((comp_count + skipped) as u64);
                continue;
            }

            // Layer transition — mark previous layer done
            if current_layer.is_some() && current_layer != Some(record.layer) {
                // Flush remaining batch for previous layer
                if !batch.is_empty() {
                    batch_num += 1;
                    let count = batch.len();
                    write_and_import_batch(
                        &args, &tmp_dir, component, batch_num, &batch, &args.ns, &args.db,
                    )?;
                    comp_count += count;
                    total_imported += count;
                    bar.set_position((comp_count + skipped) as u64);
                    batch.clear();
                }
                // Mark layer done
                if let Some(prev) = current_layer {
                    let mark = vec![loader::mark_layer_done_sql(component, prev, layer_count)];
                    write_and_import_batch(
                        &args, &tmp_dir, component, 0, &mark, &args.ns, &args.db,
                    )?;
                    layer_count = 0;
                }
            }
            current_layer = Some(record.layer);
            layer_count += 1;

            let sql = loader::single_insert_sql(component, &record);
            batch.push(sql);

            if batch.len() >= args.batch_size {
                batch_num += 1;
                let count = batch.len();
                write_and_import_batch(
                    &args, &tmp_dir, component, batch_num, &batch, &args.ns, &args.db,
                )?;
                comp_count += count;
                total_imported += count;
                bar.set_position((comp_count + skipped) as u64);
                bar.set_message(format!("batch {batch_num}"));
                batch.clear();
            }
        }

        // Final partial batch
        if !batch.is_empty() {
            batch_num += 1;
            let count = batch.len();
            write_and_import_batch(
                &args, &tmp_dir, component, batch_num, &batch, &args.ns, &args.db,
            )?;
            comp_count += count;
            total_imported += count;
            bar.set_position((comp_count + skipped) as u64);
        }

        // Mark final layer done
        if let Some(last_layer) = current_layer {
            if layer_count > 0 {
                let mark = vec![loader::mark_layer_done_sql(
                    component,
                    last_layer,
                    layer_count,
                )];
                write_and_import_batch(&args, &tmp_dir, component, 0, &mark, &args.ns, &args.db)?;
            }
        }

        bar.finish_with_message(format!(
            "{comp_count} records, {batch_num} batches, {:.0}s",
            comp_start.elapsed().as_secs_f64()
        ));
    }

    // Cleanup temp dir
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let elapsed = overall_start.elapsed();
    eprintln!("\nCompleted in {:.1}min", elapsed.as_secs_f64() / 60.0);
    eprintln!("  Total records imported: {total_imported}");

    Ok(())
}

/// Query completed layers for a component from load_progress via HTTP.
fn query_completed_layers(
    endpoint: &str,
    ns: &str,
    db: &str,
    user: &str,
    pass: &str,
    component: &str,
) -> Result<std::collections::HashSet<usize>, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();
    let sql = loader::completed_layers_sql(component);
    let resp = client
        .post(format!("{}/sql", endpoint.trim_end_matches('/')))
        .header("surreal-ns", ns)
        .header("surreal-db", db)
        .header("Accept", "application/json")
        .header(
            "Authorization",
            format!("Basic {}", base64_encode(&format!("{user}:{pass}"))),
        )
        .body(sql)
        .send()?;

    let json: serde_json::Value = resp.json()?;
    let mut layers = std::collections::HashSet::new();
    if let Some(arr) = json.as_array() {
        for result in arr {
            if let Some(rows) = result.get("result").and_then(|r| r.as_array()) {
                for row in rows {
                    if let Some(layer) = row.get("layer").and_then(|l| l.as_u64()) {
                        layers.insert(layer as usize);
                    }
                }
            }
        }
    }
    Ok(layers)
}

fn write_and_import_batch(
    args: &VectorImportArgs,
    tmp_dir: &std::path::Path,
    component: &str,
    batch_num: usize,
    statements: &[String],
    ns: &str,
    db: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let batch_path = tmp_dir.join(format!("{component}_batch_{batch_num}.surql"));

    {
        let mut f = BufWriter::new(std::fs::File::create(&batch_path)?);
        writeln!(f, "OPTION IMPORT;")?;
        writeln!(f, "USE NS {ns}; USE DB {db};")?;
        for stmt in statements {
            writeln!(f, "{stmt}")?;
        }
        f.flush()?;
    }

    run_surreal_import(args, &batch_path)?;
    std::fs::remove_file(&batch_path)?;
    Ok(())
}

fn run_surreal_import(
    args: &VectorImportArgs,
    file: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("surreal")
        .arg("import")
        .arg("--endpoint")
        .arg(&args.endpoint)
        .arg("--namespace")
        .arg(&args.ns)
        .arg("--database")
        .arg(&args.db)
        .arg("--username")
        .arg(&args.user)
        .arg("--password")
        .arg(&args.pass)
        .arg("-l")
        .arg("warn")
        .arg(file)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Check if it's just a "already exists" error — skip those
        if stderr.contains("already exists") {
            return Ok(());
        }
        return Err(format!(
            "surreal import failed: {}",
            stderr.lines().take(5).collect::<Vec<_>>().join("\n")
        )
        .into());
    }

    Ok(())
}
