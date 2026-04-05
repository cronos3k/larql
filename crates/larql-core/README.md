# larql-core

Knowledge graph engine for LARQL. Triple-based edges, graph algorithms, LLM extraction, multi-format serialization.

## What it does

Provides the core `Graph` data structure that the rest of LARQL builds on. A graph is a set of `(subject, relation, object)` edges with confidence scores, source attribution, and metadata. The engine supports querying, algorithms, LLM-based extraction, and efficient serialization.

```rust
use larql_core::*;

let mut graph = Graph::new();
graph.add_edge(
    Edge::new("France", "capital", "Paris")
        .with_confidence(0.95)
        .with_source(SourceType::Parametric)
);
graph.add_edge(
    Edge::new("Paris", "river", "Seine")
        .with_confidence(0.88)
);

// Query
let capitals = graph.select("France", Some("capital"));
let (dest, path) = graph.walk("France", &["capital", "river"]).unwrap();
assert_eq!(dest, "Seine");

// Algorithms
let result = shortest_path(&graph, "France", "Seine");
let ranks = pagerank(&graph, 0.85, 100, 1e-6);
let comps = connected_components(&graph);

// Serialize
save_json(&graph, "knowledge.larql.json").unwrap();
```

## Core Types

| Type | Purpose |
|------|---------|
| `Graph` | Indexed edge collection with adjacency, reverse, keyword indexes |
| `Edge` | Directed fact: subject --relation--> object, with confidence and metadata |
| `Schema` | Optional relation type registry and node type inference rules |
| `Node` | Computed entity with degree info and inferred type |
| `SourceType` | Edge origin: Parametric, Document, Installed, Wikidata, Manual, Unknown |

## Algorithms

| Algorithm | Function | Complexity |
|-----------|----------|------------|
| Shortest path | `shortest_path()`, `astar()` | O((E+V) log V) |
| PageRank | `pagerank()` | O(iterations x (E+V)) |
| BFS/DFS | `bfs_traversal()`, `dfs()` | O(E+V) |
| Components | `connected_components()`, `are_connected()` | O(E+V) |
| Walk | `graph.walk()` (highest confidence), `walk_all_paths()` | O(branching^depth) |
| Filter | `filter_graph()` | O(E) |
| Merge | `merge_graphs()` with Union/MaxConfidence/SourcePriority | O(E) |
| Diff | `diff()` | O(E) |
| Subgraph | `graph.subgraph()` | O(E within depth) |

## LLM Integration

| Component | Purpose |
|-----------|---------|
| `ModelProvider` trait | Single interface to any LLM backend |
| `HttpProvider` | OpenAI-compatible API (ollama, vLLM, llama.cpp) |
| `MockProvider` | Testing with fixture-based knowledge injection |
| `chain_tokens()` | Multi-token answer extraction |
| `extract_bfs()` | BFS knowledge graph extraction from LLM |
| `TemplateRegistry` | Relation-specific prompt templates |

## Serialization

| Format | Extension | Size (100K edges) | Serialize | Deserialize |
|--------|-----------|-------------------|-----------|-------------|
| JSON | `.larql.json` | 9.9 MB | 135ms | 325ms |
| MessagePack | `.larql.bin` | 4.7 MB (53% smaller) | 126ms | 303ms |
| Packed binary | `.larql.pak` | 4.3 MB (56% smaller) | 22ms | 244ms |
| CSV | `.csv` | (simple interchange) | - | - |
| Checkpoint | (append-only) | (crash-safe log) | - | - |

Packed binary uses string interning — repeated relation names stored once.

## Crate Structure

```
larql-core/src/
├── lib.rs                  Re-exports
├── core/
│   ├── graph.rs            Graph struct, indexes, queries, walk, search, subgraph, stats
│   ├── edge.rs             Edge struct, builder pattern, compact serialization
│   ├── schema.rs           Schema, RelationMeta, TypeRule, type inference
│   ├── node.rs             Node (computed entity with degree)
│   └── enums.rs            SourceType, MergeStrategy
├── algo/
│   ├── shortest_path.rs    Dijkstra and A* with custom weight functions
│   ├── pagerank.rs         Iterative PageRank
│   ├── traversal.rs        BFS and DFS with depth tracking
│   ├── components.rs       Connected components enumeration, connectivity check
│   ├── walk.rs             Multi-path walk with confidence ranking
│   ├── filter.rs           Predicate-based edge filtering
│   ├── merge.rs            Graph merging with conflict resolution strategies
│   └── diff.rs             Graph diffing (added, removed, changed)
├── engine/
│   ├── provider.rs         ModelProvider trait, PredictionResult
│   ├─�� http_provider.rs    OpenAI-compatible HTTP provider (feature-gated)
│   ├── mock_provider.rs    Mock provider for testing
│   ├── bfs.rs              BFS knowledge extraction from LLM
│   ├── chain.rs            Multi-token chaining
│   └── templates.rs        Prompt template registry
└── io/
    ├── format.rs           Format enum, auto-detection from extension
    ├── json.rs             JSON serialization (Python-compatible)
    ├── msgpack.rs          MessagePack (feature-gated)
    ├── packed.rs           String-interned binary format
    ├── csv.rs              Simple CSV import/export
    └── checkpoint.rs       Append-only crash-safe log
```

## Testing

```bash
cargo test -p larql-core                                  # 167 tests
cargo run --release -p larql-core --example bench_graph   # Benchmark
cargo run -p larql-core --example graph_demo              # Feature showcase
cargo run -p larql-core --example algorithm_demo          # Algorithm examples
```

### Benchmarks (100K edges, M3 Max)

| Operation | Latency |
|-----------|---------|
| Insert (100K edges) | 152ms (1.5us/edge) |
| select(entity, relation) | 0.1us |
| exists(s, r, o) | 0.1us |
| search(keyword, 10) | 0.5us |
| shortest_path (1K nodes) | 14us |
| connected_components (1K nodes) | 478us |
| are_connected (1K nodes) | 14us |
| walk_all_paths (3 hops) | 1.1us |
| bfs_traversal (depth=5) | 11us |
| pagerank (1K nodes) | 12ms |
| filter (100K, confidence) | 56ms |
| Packed binary serialize (100K) | 22ms |

### Test Coverage (167 tests)

- Graph: construction, queries, walk, search, subgraph, stats, dedupe
- Edge: builder pattern, equality, hashing, compact serialization
- Schema: type rules, inference, JSON roundtrip
- Algorithms: shortest path, PageRank, BFS/DFS, merge, diff, filter
- Components: enumeration, connectivity, disconnected graphs, edge cases
- Walk: highest-confidence selection, multi-path, all-paths, limits
- Remove edge: index rebuild correctness
- Search: empty query, no match, case insensitive
- Serialization: JSON/MsgPack/Packed roundtrips, metadata preservation
- BFS extraction: mock provider, depth, multi-seed, max_entities
- Token chaining: multi-token, stop tokens, probability threshold
- Templates: registry, JSON load/save
- Checkpoint: append, replay, persistence
- Python compatibility: format interop

## Design Principles

1. **Triple-based** — every fact is (subject, relation, object) with confidence
2. **Indexed** — adjacency, reverse, keyword indexes kept in sync
3. **Schema-optional** — type inference rules loaded externally, not hardcoded
4. **Provider-agnostic** — any LLM backend via ModelProvider trait
5. **Multi-format** — JSON (human), MsgPack (compact), Packed (fast), CSV (interchange)
6. **Crash-safe** — CheckpointLog for long-running extractions
7. **Zero unsafe** — all safe Rust, minimal dependencies

## Features

- `http` (default) — HTTP model provider for ollama/vLLM/llama.cpp
- `msgpack` (default) — MessagePack serialization

## License

Apache-2.0
