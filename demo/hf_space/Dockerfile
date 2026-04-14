# Stage 1: Build the larql binary from source
FROM rust:1.82-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    git pkg-config libssl-dev ca-certificates \
    libopenblas-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

# Clone the fork and build only the CLI crate
RUN git clone --depth 1 https://github.com/cronos3k/larql /build
WORKDIR /build
RUN cargo build --release -p larql-cli

# Stage 2: Lean runtime image
FROM python:3.11-slim-bookworm

# OpenBLAS runtime lib needed by the larql binary
RUN apt-get update && apt-get install -y libopenblas0 && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary
COPY --from=builder /build/target/release/larql /usr/local/bin/larql

# Copy the Gradio demo app
COPY app.py utils.py requirements.txt /app/

WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace Spaces runs as non-root UID 1000
RUN useradd -m -u 1000 hfuser && mkdir -p /app/models && chown -R hfuser /app
USER hfuser

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# app.py downloads the demo vindex from HF Hub on first startup
CMD ["python", "app.py"]
