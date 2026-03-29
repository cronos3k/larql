.PHONY: build release test check clean fmt lint demos

# Build
build:
	cargo build --workspace

release:
	cargo build --release -p larql-cli

# Test
test:
	cargo test --workspace

# Check (compile without building)
check:
	cargo check --workspace

# Code quality
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --tests -- -D warnings

# All quality checks
ci: fmt-check lint test

# Clean
clean:
	cargo clean

# Demos
demos:
	cargo run --release -p larql-models --example architecture_demo
	cargo run --release -p larql-core --example graph_demo
	cargo run --release -p larql-core --example edge_demo
	cargo run --release -p larql-core --example serialization_demo
	cargo run --release -p larql-core --example algorithm_demo
	cargo run --release -p larql-surreal --example sql_demo

demos-inference:
	cargo run --release -p larql-inference --example inference_demo

# Benchmarks
bench: bench-core bench-surreal

bench-core:
	cargo run --release -p larql-core --example bench_graph

bench-inference:
	cargo run --release -p larql-inference --example bench_inference

bench-surreal:
	cargo run --release -p larql-surreal --example bench_sql

bench-all: bench-core bench-surreal bench-inference

# Python extension (requires virtualenv)
python-build:
	cd crates/larql-python && maturin develop --release

python-check:
	cargo check -p larql-python

# Extraction
extract-test:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		--layer 26 -o output/test-L26.larql.json \
		--stats output/test-L26-stats.json

extract-full:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		-o output/gemma-3-4b-knowledge.larql.json \
		--stats output/gemma-3-4b-stats.json

# Inference
predict:
	cargo run --release -p larql-cli -- predict google/gemma-3-4b-it \
		--prompt "The capital of France is" -k 10
