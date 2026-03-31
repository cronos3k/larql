# larql-knowledge

Knowledge pipeline for [LARQL](https://github.com/chrishayuk/chuk-larql-rs). Produces reference databases and probe labels that the LARQL engine reads.

The LARQL engine reads JSON files. This project produces them.

```
larql-knowledge (this project)        larql (the engine)
  ┌──────────────────────┐              ┌──────────────────────┐
  │ Ingest               │    JSON      │  extract-index       │
  │   Wikidata/DBpedia   │──────────>   │  label               │
  │   WordNet            │   files      │  describe             │
  │   AST corpora        │              │  walk                │
  │   Morphological      │              │                      │
  │   FrameNet           │              │                      │
  │                      │              │                      │
  │ Probe                │              │                      │
  │   MLX inference      │──────────>   │  feature_labels.json │
  │   Template probing   │   labels     │                      │
  └──────────────────────┘              └──────────────────────┘
```

## Quick Start

```bash
# Install in dev mode
pip install -e ".[wordnet]"

# Assemble combined triples from individual files
python3 scripts/assemble_triples.py

# Extract WordNet relations
python3 scripts/fetch_wordnet_relations.py

# Generate morphological pairs
python3 scripts/fetch_morphological.py

# Extract AST pairs from local stdlibs (Rust, JS, TS, C, etc.)
python3 scripts/extract_all_ast_pairs.py --stdlib

# Run the full probe (requires MLX on Apple Silicon)
pip install mlx mlx-lm
python3 scripts/probe_mlx.py

# Run tests
python3 -m pytest tests/ -v
```

## Structure

```
data/
  triples/                    # 144 relation files (model-agnostic)
    capital.json              #   France->Paris, Germany->Berlin
    language.json             #   France->French, Germany->German
    occupation.json           #   Einstein->physicist, Mozart->composer
    ...                       #   (141 more)
  ast/                        # AST pairs per programming language
    python_ast.json           #   def->main, class->Model, import->torch
    rust_ast.json             #   fn->main, struct->Point, impl->Display
    javascript_ast.json       #   function->render, const->app, class->Router
    typescript_ast.json       #   interface->Props, type->Config
    c_ast.json                #   int->main, struct->node, #include->stdio
  wikidata_triples.json       # Combined: all 144 relations in one file
  wordnet_relations.json      # Synonyms, hypernyms, antonyms, meronyms
  morphological_relations.json # Plural, gerund, past_tense, comparative, etc.
  english_grammar.json        # Determiner->noun, preposition->object
  probe_templates.json        # 142 relations x 2-3 templates = 426 templates

probes/
  gemma-3-4b-it/              # Model-specific probe results
    feature_labels.json       #   157 probe-confirmed feature labels
    probe_meta.json           #   Probe run metadata

scripts/
  # Data ingestion
  assemble_triples.py         # Combine triple files into wikidata_triples.json
  ingest_dbpedia.py           # Pull triples from DBpedia SPARQL endpoint
  ingest_wikidata_dump.py     # Parse Wikidata NT dump (for 500K+ scale)
  fetch_wordnet_relations.py  # Extract WordNet relations via NLTK
  fetch_morphological.py      # Generate morphological pairs via lemminflect
  fetch_framenet.py           # Extract FrameNet frame-element pairs
  extract_ast_pairs.py        # Extract AST pairs from Python source
  extract_all_ast_pairs.py    # Extract AST pairs for 19 languages

  # Probing
  probe_mlx.py                # Run MLX inference probes (Apple Silicon)
  build_feature_labels.py     # Gate KNN probes (no model needed)

  # Analysis
  compare_probes.py           # Compare probe results across models
  coverage_report.py          # Report coverage across all data sources
  quality_check.py            # Validate triple quality

  # Utilities
  filter_entities.py          # Filter entities to single/few-token forms
  normalize_triples.py        # Case normalize, deduplicate

src/larql_knowledge/          # Python package (pip installable)
  ingest/                     # Ingestion modules
    ast_extract.py            #   Python AST extractor (built-in ast module)
    treesitter_extract.py     #   19-language AST extractor (tree-sitter + regex)
    dbpedia.py                #   DBpedia SPARQL client
    wordnet.py                #   WordNet extraction via NLTK
    grammar.py                #   English grammar pair extraction
  probe/                      # Probe modules
    labels.py                 #   Rich label format (layer/feature/relation/confidence)
    vindex.py                 #   VindexReader for Python-side gate queries
  analysis/                   # Analysis modules
    coverage.py               #   Coverage report across all sources
  triples.py                  # Triple loading, assembly, merging
  cli.py                      # CLI entry point

tests/                        # 666 tests
  test_triples_format.py      #   Validates all 144 triple files
  test_templates.py           #   Validates 142 template relations
  test_treesitter_extract.py  #   23 tests for 19-language AST extraction
  test_morphological.py       #   Validates morphological output
  test_wikidata_combined.py   #   Validates combined triples file
  test_ast_extract.py         #   Python AST extraction
  test_grammar.py             #   Grammar pair extraction
  test_labels.py              #   Probe label format
  test_probe_output.py        #   Probe output validation
  test_triples.py             #   Triple loading/merging
```

## Data Pipeline

### Reference Triples

Structured (subject, object) pairs grouped by relation. Model-agnostic.

**144 relations across 18 domains:**

| Domain | Relations | Example |
|--------|-----------|---------|
| Geography | capital, language, continent, borders, currency, ... | France->Paris |
| Cities | located_in, city_country, landmark, river, ... | London->Thames |
| People | occupation, birthplace, nationality, spouse, ... | Einstein->physicist |
| Politics | party, position, country_leader, ... | Obama->Democrat |
| Music | genre, instrument, composer, band_member, ... | Beatles->Lennon |
| Film & TV | director, starring, film_genre, tv_network, ... | Jaws->Spielberg |
| Literature | author, poet, playwright, literary_genre, ... | Hamlet->Shakespeare |
| Sports | team, league, sport, championship, ... | Messi->Barcelona |
| Companies | founder, ceo, headquarters, industry, ... | Apple->Jobs |
| Science | inventor, chemical_symbol, programming_language, ... | gold->Au |
| Food | ingredient, cuisine_origin, dish_country, ... | pizza->Italy |
| Art | painter, art_movement, art_museum, ... | Mona Lisa->Da Vinci |
| History | event_year, dynasty, historical_figure, ... | WW2->1939 |
| Animals | animal_class, animal_habitat, endangered, ... | dog->mammal |
| Education | university_city, academic_field, ... | Harvard->Cambridge |
| Religion | religion_founder, religion_text, ... | Christianity->Bible |
| Transport | manufacturer, airline_country, airport_city, ... | Boeing 747->Boeing |
| Language | language_family, language_script, ... | French->Romance |

### Linguistic Databases

| Source | Relations | Pairs | Layer Range |
|--------|-----------|-------|-------------|
| WordNet | synonym, hypernym, antonym, meronym, derivation | 17,800 | L0-13 |
| Morphological | plural, gerund, past_tense, comparative, ... | 3,952 | L0-7 |
| English Grammar | determiner->noun, preposition->object, ... | 1,040 | L4-13 |
| AST (5 langs) | def->identifier, fn->identifier, ... | 13,012 | L0-13 |

### Probe Labels

Ground truth from actual model inference. Two matching strategies:

```
Strategy 1 (gate matching):
  Entity + Template -> Forward Pass -> Gate Scores -> Down Meta Token -> Match Triple

Strategy 2 (prediction matching):
  Entity + Template -> Forward Pass -> LM Head Logits -> Top Predictions -> Match Triple
```

Strategy 2 captures what the model actually predicts (e.g. "Macron" for "The president of France is"), then matches against normalized triple objects (e.g. "Emmanuel Macron" -> ["emmanuel macron", "macron", "emmanuel"]).

**157 probe-confirmed features** for gemma-3-4b-it across 17 relations.

## Examples

### Adding a New Relation

```bash
# 1. Create the triple file
cat > data/triples/habitat.json << 'EOF'
{
  "relation": "habitat",
  "description": "Natural habitat of an animal or plant",
  "source": "hand-curated",
  "pairs": [
    ["polar bear", "Arctic"],
    ["penguin", "Antarctica"],
    ["camel", "desert"],
    ["dolphin", "ocean"],
    ["eagle", "mountains"]
  ]
}
EOF

# 2. Rebuild combined triples
python3 scripts/assemble_triples.py

# 3. Add probe templates (edit data/probe_templates.json)
# "habitat": ["The natural habitat of a {X} is", "{X} lives in"]

# 4. Re-run probe to discover features for this relation
python3 scripts/probe_mlx.py
```

### Extracting AST Pairs for a New Language

```bash
# From a corpus directory
python3 scripts/extract_all_ast_pairs.py \
  --corpus-dir /path/to/code/ \
  --languages rust,go,java \
  --max-files 200

# From detected stdlibs on your machine
python3 scripts/extract_all_ast_pairs.py --stdlib --languages rust,javascript,c

# Check what languages are available
python3 scripts/extract_all_ast_pairs.py --language-info
```

### Ingesting from DBpedia

```bash
# Pull up to 500 pairs per relation from DBpedia SPARQL
python3 scripts/ingest_dbpedia.py

# Rebuild combined triples after ingestion
python3 scripts/assemble_triples.py
```

### Scaling with Wikidata Dump

```bash
# Download the dump (~100GB)
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz

# Parse and extract (streaming, handles any size)
python3 scripts/ingest_wikidata_dump.py \
  --dump latest-truthy.nt.gz \
  --max-per-relation 5000 \
  --output data/triples/

# Rebuild
python3 scripts/assemble_triples.py
```

### Comparing Probes Across Models

```bash
# After probing multiple models
python3 scripts/compare_probes.py \
  probes/gemma-3-4b-it/ \
  probes/llama-3-8b/ \
  probes/mistral-7b/
```

### Running the Full Pipeline

```bash
# 1. Install everything
pip install -e ".[all]"

# 2. Build reference data
python3 scripts/assemble_triples.py
python3 scripts/fetch_wordnet_relations.py
python3 scripts/fetch_morphological.py
python3 scripts/extract_all_ast_pairs.py --stdlib

# 3. Probe (requires MLX + model)
python3 scripts/probe_mlx.py

# 4. Validate
python3 -m pytest tests/ -v

# 5. Check coverage
python3 scripts/coverage_report.py
```

### Using the Python API

```python
from pathlib import Path
from larql_knowledge.triples import load_all_triples, stats
from larql_knowledge.ingest.treesitter_extract import (
    extract_pairs_from_source,
    SUPPORTED_LANGUAGES,
)
from larql_knowledge.probe.labels import load_feature_labels

# Load all triples from the triples directory
triples = load_all_triples(Path("data/triples"))
s = stats(triples)
print(f"{s['num_relations']} relations, {s['total_pairs']} pairs")

# Extract AST pairs from code
pairs = extract_pairs_from_source("fn main() { println!(\"hello\"); }", "rust")
print(pairs)  # {"fn": [["fn", "main"]], ...}

# List supported languages
print(SUPPORTED_LANGUAGES)  # ['rust', 'javascript', 'typescript', ...]

# Load probe labels
labels = load_feature_labels(Path("probes/gemma-3-4b-it/feature_labels.json"))
print(f"{len(labels)} labeled features")
```

### Using the CLI

```bash
# Assemble triples
larql-knowledge assemble

# Ingest from DBpedia
larql-knowledge ingest-dbpedia --limit 500

# Extract WordNet
larql-knowledge ingest-wordnet

# Coverage report
larql-knowledge coverage
```

## Contributing

### Adding Triples

Create a JSON file in `data/triples/`:

```json
{
  "relation": "habitat",
  "pid": "P2974",
  "description": "Natural habitat of an animal or plant species",
  "source": "hand-curated",
  "pairs": [
    ["polar bear", "Arctic"],
    ["penguin", "Antarctica"]
  ]
}
```

Run `python3 scripts/assemble_triples.py` to rebuild, then `python3 -m pytest tests/test_triples_format.py` to validate.

### Adding AST Languages

The tree-sitter extractor supports 19 languages with regex fallback. To add a new language:

1. Add keyword patterns to `src/larql_knowledge/ingest/treesitter_extract.py`
2. Add file extensions to `LANGUAGE_EXTENSIONS`
3. Add tests to `tests/test_treesitter_extract.py`

### Adding Probe Templates

Edit `data/probe_templates.json`:

```json
{
  "habitat": [
    "The natural habitat of a {X} is",
    "{X} lives in",
    "The {X} is found in"
  ]
}
```

Each relation should have 2-3 template variants to maximize probe coverage.

### Running Probes for a New Model

```bash
# 1. Build the vindex (in the larql repo)
larql extract-index google/gemma-3-4b-it -o output/gemma3-4b.vindex

# 2. Run the probe
python3 scripts/probe_mlx.py

# 3. Labels are saved to both:
#    - output/gemma3-4b-full.vindex/feature_labels.json (for the engine)
#    - probes/gemma-3-4b-it/feature_labels.json (for comparison)
```

## Testing

```bash
# Run all 666 tests
python3 -m pytest tests/ -v

# Run specific test groups
python3 -m pytest tests/test_triples_format.py -v   # Validate all 144 triple files
python3 -m pytest tests/test_templates.py -v         # Validate 426 templates
python3 -m pytest tests/test_treesitter_extract.py -v # AST extraction (19 languages)
python3 -m pytest tests/test_morphological.py -v     # Morphological pairs
python3 -m pytest tests/test_wikidata_combined.py -v # Combined triples
```

## Current Stats

- **144 relation types**, 18,502 pairs across 18 domains
- **142 probe template groups**, 426 templates (2-3 variants each)
- **9 WordNet relations**, 17,800 pairs
- **10 morphological relations**, 3,952 pairs
- **5 AST languages** (Python, Rust, JavaScript, TypeScript, C), 13,012 pairs
- **157 probe-confirmed features** for gemma-3-4b-it
- **666 tests**, all passing

## Label Priority

Labels come from multiple sources. Higher priority overrides lower:

1. **Probe-confirmed** -- model inference confirmed this feature encodes this relation
2. **Wikidata output matching** -- cluster outputs match Wikidata objects
3. **WordNet output matching** -- cluster outputs match WordNet pairs (L0-13)
4. **AST output matching** -- cluster outputs match AST pairs (L0-13)
5. **Entity pattern detection** -- cluster members match known entity lists
6. **Morphological detection** -- cluster members are short suffixes/prefixes
7. **TF-IDF top tokens** -- fallback: most distinctive tokens in the cluster

## Roadmap

| Phase | Triples | Relations | AST Languages | Probe Features | Status |
|-------|---------|-----------|---------------|----------------|--------|
| 1 | 16K | 32 | 1 | 112 | Done |
| 2 | 18.5K | 144 | 5 | 157 | Current |
| 3 | 500K | 150+ | 15 | 5,000+ | Next |
| 4 | 2M+ | 200+ | 30+ | 20,000+ | Future |

See [docs/knowledge-pipeline-spec.md](docs/knowledge-pipeline-spec.md) for the full specification.
