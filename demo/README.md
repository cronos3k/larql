---
title: LARQL Explorer
emoji: 🧠
colorFrom: violet
colorTo: blue
sdk: gradio
sdk_version: "6.2.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Query neural network weights like a graph database
---

# LARQL Explorer

**The model IS the database.** Browse and query transformer weights as a knowledge
graph — no SQL, no GPU needed for basic queries.

Original LARQL system by **Chris Hayuk** — [chrishayuk/larql](https://github.com/chrishayuk/larql)  
This Gradio UI + Windows/Linux/CUDA port by **Gregor Koch** — [cronos3k/larql](https://github.com/cronos3k/larql)

> The original was a command-line tool for macOS only. This fork opens it up:
> any platform, any hardware, any browser.

## What you can do

| Tab | What it does |
|---|---|
| 🔍 Walk Explorer | See which FFN features fire for any prompt, layer by layer |
| 🧪 Knowledge Probe | Compare three prompts side-by-side at the same layer |
| 💻 LQL Console | Full LQL query interface — WALK, DESCRIBE, INFER, INSERT |
| 📊 Vindex Info | Inspect vindex metadata and verify file checksums |
| ⬇️ Extract | Download a model from HuggingFace and extract a vindex |
| ℹ️ Setup | Build instructions and environment info |

## Running locally

```bash
git clone https://github.com/cronos3k/larql
cd larql

# Build the binary
cargo build --release          # CPU
# or
cargo build --release --features cuda   # NVIDIA GPU

# Install Python dependencies and run the demo
pip install -r demo/requirements.txt
python demo/app.py
```

Then open http://localhost:7860

## Quick example

1. Extract a small model (takes ~5 min):
   ```bash
   HF_TOKEN=hf_... ./target/release/larql extract-index Qwen/Qwen2.5-0.5B-Instruct -o models/qwen.vindex
   ```
2. Open the **Walk Explorer** tab  
3. Type `The capital of France is` and click **Walk →**  
4. Watch the model's internal "beliefs" light up layer by layer

## Credits

- **Original LARQL / LQL design:** Chris Hayuk ([@chrishayuk](https://github.com/chrishayuk))
- **Windows/Linux/CUDA port and this demo:** Gregor Koch ([@cronos3k](https://github.com/cronos3k))
- License: Apache-2.0
