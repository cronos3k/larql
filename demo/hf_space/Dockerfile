# LARQL Explorer — HuggingFace Space
# Downloads the pre-built Linux binary from the cronos3k/larql GitHub Release.
# No Rust build step: fast image build, no compiler toolchain needed.

FROM python:3.11-slim-bookworm

# Runtime deps only: libopenblas0 for the larql binary, curl for the download
RUN apt-get update && apt-get install -y \
    libopenblas0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download the latest pre-built Linux binary from GitHub Releases
RUN curl -fsSL \
    https://github.com/cronos3k/larql/releases/download/latest-linux/larql-linux-x86_64 \
    -o /usr/local/bin/larql \
    && chmod +x /usr/local/bin/larql \
    && larql --version

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

# app.py downloads the demo vindex from HF dataset on first startup (~480 MB, fast on HF infra)
CMD ["python", "app.py"]
