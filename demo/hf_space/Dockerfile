# Stage 1: Build the larql binary from source
FROM rust:1.82-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    git pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone the fork (CPU-only build; no CUDA toolkit needed in the Space)
RUN git clone --depth 1 https://github.com/cronos3k/larql /build
WORKDIR /build

# Build only the CLI crate (avoids protobuf issues in larql-server)
RUN cargo build --release -p larql-cli

# Stage 2: Runtime image
FROM python:3.11-slim-bookworm

# Copy the compiled binary
COPY --from=builder /build/target/release/larql /usr/local/bin/larql

# Copy the Gradio demo
COPY app.py utils.py requirements.txt /app/

WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace Spaces expects a non-root user with UID 1000
RUN useradd -m -u 1000 hfuser
USER hfuser

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
