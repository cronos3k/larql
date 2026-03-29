use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_surreal::loader::{
    self, discover_vector_files, LoadCallbacks, TableSummary, VectorReader,
};
use larql_surreal::{SurrealClient, SurrealError};

#[derive(Args)]
pub struct VectorLoadArgs {
    /// Directory containing .vectors.jsonl files from vector-extract.
    input: PathBuf,

    /// SurrealDB endpoint (e.g. http://localhost:8000).
    #[arg(long, default_value = "http://localhost:8000")]
    surreal: String,

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

    /// Tables to load (comma-separated). Default: all found in input dir.
    #[arg(long, value_delimiter = ',')]
    tables: Option<Vec<String>>,

    /// Layers to load (comma-separated). Default: all.
    #[arg(long, value_delimiter = ',')]
    layers: Option<Vec<usize>>,

    /// Batch size for INSERT transactions. Lower for large vectors.
    #[arg(long, default_value = "50")]
    batch_size: usize,

    /// Resume interrupted load (skips completed layers).
    #[arg(long)]
    resume: bool,

    /// Create schema only (no data load).
    #[arg(long)]
    schema_only: bool,
}

/// Query completed layers from load_progress table.
fn completed_layers(client: &SurrealClient, table: &str) -> Result<HashSet<usize>, SurrealError> {
    let sql = loader::completed_layers_sql(table);
    let resp = client.query(&sql)?;

    let mut layers = HashSet::new();
    if let Some(arr) = resp.as_array() {
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

struct ProgressCallbacks {
    bar: ProgressBar,
}

impl LoadCallbacks for ProgressCallbacks {
    fn on_table_start(&mut self, table: &str, total_records: usize) {
        self.bar.set_length(total_records as u64);
        self.bar.set_position(0);
        self.bar
            .set_message(format!("{table}: {total_records} records"));
    }

    fn on_batch_done(&mut self, _table: &str, _batch_num: usize, records_loaded: usize) {
        self.bar.set_position(records_loaded as u64);
    }

    fn on_table_done(&mut self, table: &str, total_loaded: usize, elapsed_ms: f64) {
        self.bar.set_position(total_loaded as u64);
        eprintln!(
            "  {table}: {total_loaded} records loaded ({:.0}s)",
            elapsed_ms / 1000.0,
        );
    }
}

pub fn run(args: VectorLoadArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Discover vector files
    let all_files = discover_vector_files(&args.input)?;
    if all_files.is_empty() {
        return Err(format!("no .vectors.jsonl files found in {}", args.input.display()).into());
    }

    // Filter by requested tables
    let files: Vec<_> = match &args.tables {
        Some(tables) => all_files
            .into_iter()
            .filter(|(name, _)| tables.contains(name))
            .collect(),
        None => all_files,
    };

    let layer_filter: Option<HashSet<usize>> = args.layers.map(|ls| ls.into_iter().collect());

    eprintln!("Connecting to SurrealDB: {}", args.surreal);
    eprintln!("  ns={}, db={}", args.ns, args.db);
    eprintln!(
        "  tables: {}",
        files
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let client = SurrealClient::new(&args.surreal, &args.ns, &args.db, &args.user, &args.pass);

    // Setup namespace and database
    let setup_sql = loader::setup_sql(&args.ns, &args.db);
    client.exec(&setup_sql)?;
    eprintln!("  namespace/database ready");

    // Create progress tracking table
    client.exec(&loader::progress_table_sql())?;

    // Create schemas for each table
    for (component, path) in &files {
        let reader = VectorReader::open(path)?;
        let dimension = reader.header().dimension;
        let schema = loader::schema_sql(component, dimension)?;
        client.exec(&schema)?;
        eprintln!("  schema: {component} (dim={dimension})");
    }

    if args.schema_only {
        eprintln!("\nSchema created. No data loaded (--schema-only).");
        return Ok(());
    }

    // Load data
    let overall_start = Instant::now();
    let mut summaries = Vec::new();

    let bar = ProgressBar::new(0);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap(),
    );
    let mut callbacks = ProgressCallbacks { bar };

    for (component, path) in &files {
        let table_start = Instant::now();

        // Check completed layers for resume
        let completed = if args.resume {
            match completed_layers(&client, component) {
                Ok(layers) => {
                    if !layers.is_empty() {
                        eprintln!(
                            "\n  {component}: resuming ({} layers already loaded)",
                            layers.len()
                        );
                    }
                    layers
                }
                Err(_) => HashSet::new(), // table might not exist yet
            }
        } else {
            HashSet::new()
        };

        // Stream records one at a time from NDJSON.
        // Each record is a single HTTP request (~30KB for 2560-dim vectors).
        // No batching needed — avoids HTTP body size limits entirely.
        let mut reader = VectorReader::open(path)?;

        // Quick line count for progress bar
        let total_estimate = {
            let f = std::fs::File::open(path)?;
            let r = std::io::BufReader::new(f);
            use std::io::BufRead;
            r.lines().count().saturating_sub(1)
        };
        callbacks.on_table_start(component, total_estimate);

        let mut total_loaded = 0;
        let mut current_layer: Option<usize> = None;
        let mut layer_count = 0;
        let progress_interval = (total_estimate / 100).max(1);

        while let Some(record) = reader.next_record()? {
            // Skip completed layers
            if completed.contains(&record.layer) {
                continue;
            }
            // Skip filtered layers
            if let Some(ref filter) = layer_filter {
                if !filter.contains(&record.layer) {
                    continue;
                }
            }

            // Layer transition — mark previous layer done
            if current_layer.is_some() && current_layer != Some(record.layer) {
                if let Some(prev) = current_layer {
                    let sql = loader::mark_layer_done_sql(component, prev, layer_count);
                    client.exec(&sql)?;
                    layer_count = 0;
                }
            }
            current_layer = Some(record.layer);

            // Insert single record
            let sql = loader::single_insert_sql(component, &record);
            client.exec(&sql)?;
            total_loaded += 1;
            layer_count += 1;

            if total_loaded % progress_interval == 0 {
                callbacks.on_batch_done(component, 0, total_loaded);
            }
        }

        // Mark final layer done
        if let Some(last_layer) = current_layer {
            if layer_count > 0 {
                let sql = loader::mark_layer_done_sql(component, last_layer, layer_count);
                client.exec(&sql)?;
            }
        }

        let elapsed_ms = table_start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_table_done(component, total_loaded, elapsed_ms);

        summaries.push(TableSummary {
            table: component.clone(),
            records_loaded: total_loaded,
            elapsed_secs: table_start.elapsed().as_secs_f64(),
        });
    }

    callbacks.bar.finish_and_clear();

    let elapsed = overall_start.elapsed();
    let total: usize = summaries.iter().map(|s| s.records_loaded).sum();

    eprintln!("\nCompleted in {:.1}min", elapsed.as_secs_f64() / 60.0);
    eprintln!("  Total records loaded: {total}");
    for s in &summaries {
        if s.records_loaded > 0 {
            eprintln!(
                "  {}: {} records ({:.0}s)",
                s.table, s.records_loaded, s.elapsed_secs,
            );
        }
    }

    Ok(())
}
