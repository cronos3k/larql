---
title: LARQL Explorer
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
short_description: Query neural network weights like a knowledge graph
---

# LARQL Explorer

**The model IS the database.** Browse transformer weight space as a knowledge graph —
no terminal, no LQL syntax needed.

Original LARQL system by **Chris Hayuk** — [chrishayuk/larql](https://github.com/chrishayuk/larql)  
This UI + Windows/Linux/CUDA port by **Gregor Koch** — [cronos3k/larql](https://github.com/cronos3k/larql)

> The original was a command-line tool for macOS only.
> This fork opens it up: any platform, any hardware, any browser.

## Features

| Tab | What it does |
|---|---|
| 🔍 Walk Explorer | See which FFN features fire for any prompt, layer by layer |
| 🧪 Knowledge Probe | Compare three prompts side-by-side at the same layer |
| 💻 LQL Console | Full LQL query interface with example buttons |
| 📊 Vindex Info | Model metadata + SHA256 checksum verification |
| ⬇️ Extract | Download + extract a model from HuggingFace Hub |
| ℹ️ Setup | Build instructions and environment info |

## Running locally

```bash
git clone https://github.com/cronos3k/larql
cd larql
cargo build --release          # or --features cuda for NVIDIA GPU
pip install -r demo/requirements.txt
python demo/app.py
```

## Credits

- **LARQL / LQL / vindex format:** Chris Hayuk ([@chrishayuk](https://github.com/chrishayuk))
- **Gradio UI + cross-platform port:** Gregor Koch ([@cronos3k](https://github.com/cronos3k))
- License: Apache-2.0
