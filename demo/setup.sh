#!/usr/bin/env bash
# HuggingFace Spaces setup script
# Builds the larql binary from source (Linux x86_64, CPU-only build).
# Place this file at the root of your Space alongside app.py.
set -e

echo "=== LARQL Spaces setup ==="

# Install Rust if not present
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi

echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"

# Build the CLI (CPU only, skip larql-server to avoid protobuf issues)
echo "Building larql CLI..."
cargo build --release -p larql-cli

# Copy binary to demo directory for easy discovery
cp target/release/larql demo/larql
echo "Binary: $(ls -lh demo/larql)"

echo "=== Setup complete ==="
