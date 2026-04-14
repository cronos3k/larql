"""
LARQL Explorer — Gradio 6 demo
Browse neural network weights as a knowledge graph using LQL (Lazarus Query Language).

Built on top of: https://github.com/chrishayuk/larql (Chris Hayuk)
Fork / Windows + CUDA port: https://github.com/cronos3k/larql
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import gradio as gr
import pandas as pd

# Add demo dir to path so utils is importable both locally and on HF Spaces
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    LARQL, larql_available, run_larql,
    parse_walk_output, load_vindex_info, format_vindex_summary,
    list_local_vindexes,
)

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
# On HF Spaces (Docker) __file__ is /app/app.py, so REPO_ROOT is /
# Store the demo vindex alongside the app instead
_RUNNING_IN_SPACE = os.environ.get("SPACE_ID") is not None or Path("/app").exists()
MODELS_DIR = Path("/app/models") if _RUNNING_IN_SPACE else REPO_ROOT / "models"

# ---------------------------------------------------------------------------
# Demo vindex: auto-download from HF if no local vindexes are found
# ---------------------------------------------------------------------------
DEMO_DATASET = "cronos3k/qwen2.5-0.5b-instruct-vindex"
DEMO_VINDEX_DIR = MODELS_DIR / "qwen2.5-0.5b-instruct.vindex"

def maybe_download_demo_vindex(progress_fn=None):
    """
    Download the demo vindex from HF Hub if no local vindexes are available.
    Called once at startup. Safe to call multiple times (no-op if already present).
    """
    # Already have it?
    if (DEMO_VINDEX_DIR / "index.json").exists():
        return str(DEMO_VINDEX_DIR)
    # Any other local vindex?
    if list_local_vindexes(str(MODELS_DIR)):
        return None

    try:
        import huggingface_hub as hfh
    except ImportError:
        print("[demo] huggingface_hub not installed — skipping demo vindex download.")
        return None

    print(f"[demo] No local vindex found. Downloading demo from {DEMO_DATASET}...")
    DEMO_VINDEX_DIR.mkdir(parents=True, exist_ok=True)
    try:
        hfh.snapshot_download(
            repo_id=DEMO_DATASET,
            repo_type="dataset",
            local_dir=str(DEMO_VINDEX_DIR),
            ignore_patterns=["*.md"],  # skip dataset card
        )
        print(f"[demo] Demo vindex ready at {DEMO_VINDEX_DIR}")
        return str(DEMO_VINDEX_DIR)
    except Exception as e:
        print(f"[demo] Could not download demo vindex: {e}")
        return None

# Download at startup (blocking — fast on HF Spaces internal network, ~5-10s)
maybe_download_demo_vindex()

def get_vindex_choices():
    paths = list_local_vindexes(str(MODELS_DIR)) if MODELS_DIR.exists() else []
    paths += list_local_vindexes(".")
    # deduplicate
    seen = set()
    unique = []
    for p in paths:
        key = str(Path(p).resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique if unique else ["(no vindexes found — enter path manually)"]

DEFAULT_VINDEX = get_vindex_choices()[0]

# ---------------------------------------------------------------------------
# Backend check banner
# ---------------------------------------------------------------------------
def binary_status_md() -> str:
    if larql_available():
        rc, out, err = run_larql("--version")
        ver = (out or err).strip().split("\n")[0]
        return f"✅ **larql binary found:** `{LARQL}`  \n_Version: {ver}_"
    return (
        "⚠️ **larql binary not found.**  \n"
        "Build it with `cargo build --release` from the repo root, "
        "or see the **Setup** tab for instructions."
    )


# ---------------------------------------------------------------------------
# Tab 1 — Walk Explorer
# ---------------------------------------------------------------------------
def do_walk(vindex_path, prompt, layer_from, layer_to, top_k):
    if not prompt.strip():
        return pd.DataFrame(), "Enter a prompt above."
    if not vindex_path.strip():
        return pd.DataFrame(), "Enter a vindex path."

    layers_arg = f"{int(layer_from)}-{int(layer_to)}"
    rc, out, err = run_larql(
        "walk",
        "--prompt", prompt,
        "--index", vindex_path.strip(),
        "--layers", layers_arg,
        "--top-k", str(int(top_k)),
        timeout=60,
    )
    combined = (out + "\n" + err).strip()
    if rc != 0:
        return pd.DataFrame(), f"**Error:**\n```\n{combined}\n```"

    rows = parse_walk_output(combined)
    if not rows:
        return pd.DataFrame(), f"No features returned.\n\nRaw output:\n```\n{combined}\n```"

    df = pd.DataFrame(rows)
    # Summary footer from last line of output
    summary = [l for l in combined.splitlines() if l.startswith("Walk:")]
    status = summary[-1] if summary else ""
    return df, f"✓ {status}"


def update_layer_max(vindex_path):
    """Read num_layers from index.json to set sensible layer slider bounds."""
    try:
        info = load_vindex_info(vindex_path.strip())
        n = info.get("num_layers", 24)
        return gr.Slider(maximum=n - 1, value=n - 1), gr.Slider(maximum=n - 1, value=max(0, n - 4))
    except Exception:
        return gr.Slider(maximum=47), gr.Slider(maximum=47)


# ---------------------------------------------------------------------------
# Tab 2 — Knowledge Probe (side-by-side comparison)
# ---------------------------------------------------------------------------
def do_probe(vindex_path, prompt1, prompt2, prompt3, layer, top_k):
    results = []
    for prompt in [prompt1, prompt2, prompt3]:
        if not prompt.strip():
            results.append("_(empty)_")
            continue
        rc, out, err = run_larql(
            "walk",
            "--prompt", prompt,
            "--index", vindex_path.strip(),
            "--layers", str(int(layer)),
            "--top-k", str(int(top_k)),
            timeout=60,
        )
        combined = (out + "\n" + err).strip()
        rows = parse_walk_output(combined)
        if not rows:
            results.append(f"```\n{combined[:400]}\n```")
            continue
        lines = [f"**Prompt:** _{prompt}_\n"]
        for r in rows:
            bar = "█" * int(abs(r["Gate"]) * 100) or "·"
            arrow = "▲" if r["Gate"] > 0 else "▼"
            lines.append(
                f"`{r['Feature']}` {arrow} gate={r['Gate']:+.3f}  "
                f"hears=**\"{r['Hears']}\"**  → {r['Top tokens (down)']}"
            )
        results.append("\n".join(lines))

    return results[0], results[1], results[2]


# ---------------------------------------------------------------------------
# Tab 3 — LQL Console
# ---------------------------------------------------------------------------
LQL_EXAMPLES = [
    'USE "{vindex}"; WALK "The capital of France is" TOP 10;',
    'USE "{vindex}"; WALK "Python is a programming" TOP 5;',
    'USE "{vindex}"; WALK "Shakespeare wrote" TOP 8;',
    'USE "{vindex}"; WALK "Water boils at 100 degrees" TOP 5;',
]

def do_lql(vindex_path, statement):
    if not statement.strip():
        return "Enter an LQL statement."
    # Auto-inject USE if the user forgot it
    stmt = statement.strip()
    if not stmt.upper().startswith("USE") and vindex_path.strip():
        stmt = f'USE "{vindex_path.strip()}"; {stmt}'
    rc, out, err = run_larql("lql", stmt, timeout=90)
    combined = (out + "\n" + err).strip()
    return combined if combined else "(no output)"


def fill_lql_example(vindex_path, example_template):
    return example_template.replace("{vindex}", vindex_path.strip() or "path/to/your.vindex")


# ---------------------------------------------------------------------------
# Tab 4 — Vindex Info & Verify
# ---------------------------------------------------------------------------
def do_vindex_info(vindex_path):
    path = vindex_path.strip()
    if not path:
        return "_Enter a vindex path._", "_—_"
    info = load_vindex_info(path)
    summary = format_vindex_summary(info, path)

    # Run verify
    rc, out, err = run_larql("verify", path, timeout=120)
    verify_out = (out + "\n" + err).strip()
    verify_md = f"```\n{verify_out}\n```"
    return summary, verify_md


# ---------------------------------------------------------------------------
# Tab 5 — Extract / Download
# ---------------------------------------------------------------------------
def do_extract(model_id, output_name, level, hf_token):
    if not model_id.strip():
        return "Enter a HuggingFace model ID."
    out_dir = str(MODELS_DIR / (output_name.strip() or model_id.split("/")[-1] + ".vindex"))
    level_flag = {"Browse (smallest, ~0.5 GB)": "browse",
                  "Inference (~1 GB)": "inference",
                  "All (~2 GB)": "all"}[level]
    env_extra = {}
    if hf_token.strip():
        env_extra["HF_TOKEN"] = hf_token.strip()
    yield f"⏳ Extracting `{model_id}` → `{out_dir}` (level={level_flag})…\n\nThis can take 5–20 minutes."
    rc, out, err = run_larql(
        "extract-index", model_id.strip(),
        "-o", out_dir,
        "--level", level_flag,
        timeout=1800,
        env_extra=env_extra,
    )
    combined = (out + "\n" + err).strip()
    if rc == 0:
        yield f"✅ Done!\n\nVindex saved to: `{out_dir}`\n\n```\n{combined[-1000:]}\n```"
    else:
        yield f"❌ Failed (exit {rc})\n\n```\n{combined[-2000:]}\n```"


# ---------------------------------------------------------------------------
# Tab 6 — Setup / About
# ---------------------------------------------------------------------------
SETUP_MD = """
## About LARQL

**LARQL** decompiles transformer models into a queryable format called a **vindex**,
then provides **LQL** (Lazarus Query Language) to browse and edit the model's knowledge —
without running a forward pass.

> _"The model IS the database."_

| Original work | [chrishayuk/larql](https://github.com/chrishayuk/larql) — **Chris Hayuk** |
|---|---|
| This fork | [cronos3k/larql](https://github.com/cronos3k/larql) — Windows/Linux + CUDA port |

---

## Build the binary (first time)

```bash
# CPU only (works everywhere)
cargo build --release

# With NVIDIA CUDA GPU acceleration
cargo build --release --features cuda

# The binary ends up at:
#   target/release/larql        (Linux/macOS)
#   target/release/larql.exe    (Windows)
```

**Requirements:** Rust stable (`rustup`), a C compiler (gcc/clang/MSVC).
For CUDA: CUDA 12.x toolkit installed and `nvcc` in PATH.

---

## Quick LQL reference

```sql
-- Load a vindex
USE "path/to/model.vindex";

-- Walk: what features fire for this prompt?
WALK "The capital of France is" TOP 10;

-- Predict next token (needs --level inference or higher)
INFER "The capital of France is" TOP 5;

-- Edit knowledge
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon");
```

---

## Running on HuggingFace Spaces

1. Fork [cronos3k/larql](https://github.com/cronos3k/larql)
2. Create a new Space (Gradio SDK)
3. Add this `demo/` folder as your Space root
4. Add a `setup.sh` that builds the binary (see the repo for the template)
5. Upload a pre-extracted vindex as a Space dataset
"""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
)
_CSS = """
    .feature-row-up   { background: #f0fff4 !important; }
    .feature-row-down { background: #fff5f5 !important; }
    .larql-header { font-size: 1.6em; font-weight: bold; margin-bottom: 0.2em; }
    footer { display: none !important; }
"""

with gr.Blocks(title="LARQL Explorer") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding: 1.2em 0 0.6em 0;">
      <div style="font-size:2.2em; font-weight:800; letter-spacing:-1px;">
        🧠 LARQL Explorer
      </div>
      <div style="color:#666; font-size:1.05em; margin-top:0.3em;">
        Query neural network weights like a graph database &mdash; no SQL needed
      </div>
      <div style="font-size:0.85em; margin-top:0.5em; color:#888;">
        Based on <a href="https://github.com/chrishayuk/larql" target="_blank">chrishayuk/larql</a>
        by Chris Hayuk &nbsp;·&nbsp;
        Windows/CUDA fork: <a href="https://github.com/cronos3k/larql" target="_blank">cronos3k/larql</a>
      </div>
    </div>
    """)

    binary_status = gr.Markdown(binary_status_md())

    # ── Shared vindex selector (visible at top) ─────────────────────────────
    with gr.Row():
        vindex_choices = get_vindex_choices()
        vindex_dd = gr.Dropdown(
            choices=vindex_choices,
            value=vindex_choices[0],
            label="Active vindex",
            allow_custom_value=True,
            scale=4,
            info="Select a pre-extracted vindex or type a custom path",
        )
        refresh_btn = gr.Button("🔄 Refresh list", scale=1, variant="secondary")

    def refresh_vindex_list():
        choices = get_vindex_choices()
        return gr.Dropdown(choices=choices, value=choices[0])

    refresh_btn.click(refresh_vindex_list, outputs=vindex_dd)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Walk Explorer ─────────────────────────────────────────────
        with gr.Tab("🔍 Walk Explorer"):
            gr.Markdown("""
            **Walk the model:** for each layer in the range, find the FFN features
            that fire most strongly for your prompt. Positive gate = the feature
            *pushes* the residual stream toward its output tokens. Negative = it
            *pulls* away.
            """)
            with gr.Row():
                walk_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="The capital of France is",
                    scale=4,
                )
                walk_btn = gr.Button("Walk →", variant="primary", scale=1)

            with gr.Row():
                layer_from = gr.Slider(
                    minimum=0, maximum=23, value=20, step=1,
                    label="Layer from", scale=2,
                )
                layer_to = gr.Slider(
                    minimum=0, maximum=23, value=23, step=1,
                    label="Layer to", scale=2,
                )
                walk_topk = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Top-K features per layer", scale=2,
                )

            walk_status = gr.Markdown("")
            walk_table = gr.DataFrame(
                label="Active features",
                wrap=True,
                column_widths=["80px", "90px", "80px", "110px", "140px", "80px", "auto"],
            )
            gr.Markdown(
                "_💡 **Tip:** If clicking other tabs stops working after running Walk, "
                "refresh the page (F5) and navigate to the desired tab first. "
                "This is a Gradio 6.12 interaction bug that only appears after "
                "the feature table is populated._",
                visible=True,
            )

            walk_btn.click(
                do_walk,
                inputs=[vindex_dd, walk_prompt, layer_from, layer_to, walk_topk],
                outputs=[walk_table, walk_status],
            )
            walk_prompt.submit(
                do_walk,
                inputs=[vindex_dd, walk_prompt, layer_from, layer_to, walk_topk],
                outputs=[walk_table, walk_status],
            )
            vindex_dd.change(
                update_layer_max,
                inputs=[vindex_dd],
                outputs=[layer_to, layer_from],
            )

            gr.Examples(
                examples=[
                    ["The capital of France is"],
                    ["Python is a programming"],
                    ["Shakespeare wrote"],
                    ["Water boils at 100 degrees"],
                    ["The speed of light is"],
                    ["Einstein discovered"],
                    ["The largest planet in the solar system is"],
                ],
                inputs=walk_prompt,
                label="Try these prompts",
            )

        # ── Tab 2: Knowledge Probe ───────────────────────────────────────────
        with gr.Tab("🧪 Knowledge Probe"):
            gr.Markdown("""
            Compare how the model encodes **three different prompts** at the same layer.
            Use this to see which features are concept-specific vs. shared.
            """)
            with gr.Row():
                probe_layer = gr.Slider(
                    minimum=0, maximum=23, value=23, step=1,
                    label="Layer to inspect", scale=3,
                )
                probe_topk = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Top-K features", scale=2,
                )
                probe_btn = gr.Button("Compare →", variant="primary", scale=1)

            with gr.Row():
                probe_p1 = gr.Textbox(label="Prompt A", value="The capital of France is", scale=1)
                probe_p2 = gr.Textbox(label="Prompt B", value="Python is a programming", scale=1)
                probe_p3 = gr.Textbox(label="Prompt C", value="Shakespeare wrote", scale=1)

            with gr.Row():
                probe_out1 = gr.Markdown(label="Result A")
                probe_out2 = gr.Markdown(label="Result B")
                probe_out3 = gr.Markdown(label="Result C")

            probe_btn.click(
                do_probe,
                inputs=[vindex_dd, probe_p1, probe_p2, probe_p3, probe_layer, probe_topk],
                outputs=[probe_out1, probe_out2, probe_out3],
            )

        # ── Tab 3: LQL Console ───────────────────────────────────────────────
        with gr.Tab("💻 LQL Console"):
            gr.Markdown("""
            **LQL** (Lazarus Query Language) — the full query interface.
            The active vindex above is injected automatically as `USE "…";` if not already present.
            """)
            with gr.Row():
                lql_input = gr.Textbox(
                    label="LQL statement",
                    placeholder='WALK "The capital of France is" TOP 10;',
                    lines=4,
                    scale=5,
                )
                lql_btn = gr.Button("Run ▶", variant="primary", scale=1)

            lql_output = gr.Code(label="Output", language=None, lines=20)

            lql_btn.click(do_lql, inputs=[vindex_dd, lql_input], outputs=lql_output)
            lql_input.submit(do_lql, inputs=[vindex_dd, lql_input], outputs=lql_output)

            gr.Markdown("**Quick examples** (click to load):")
            with gr.Row():
                for tpl in LQL_EXAMPLES:
                    short = tpl.split(";")[1].strip()[:50] if ";" in tpl else tpl[:50]
                    btn = gr.Button(short, size="sm", variant="secondary")
                    btn.click(
                        lambda vp, t=tpl: fill_lql_example(vp, t),
                        inputs=[vindex_dd],
                        outputs=lql_input,
                    )

        # ── Tab 4: Vindex Info ───────────────────────────────────────────────
        with gr.Tab("📊 Vindex Info"):
            gr.Markdown("Inspect the active vindex's metadata and verify file integrity.")
            info_btn = gr.Button("Load info + verify checksums", variant="primary")
            with gr.Row():
                info_summary = gr.Markdown("_Click the button above._")
                verify_out = gr.Markdown("_—_")

            info_btn.click(
                do_vindex_info,
                inputs=[vindex_dd],
                outputs=[info_summary, verify_out],
            )

        # ── Tab 5: Extract New Vindex ────────────────────────────────────────
        with gr.Tab("⬇️ Extract"):
            gr.Markdown("""
            Download a model from HuggingFace and extract it into a vindex.
            **Browse** level is enough for `WALK` and `DESCRIBE` queries.
            You need **Inference** level for `INFER` (next-token prediction).
            """)
            with gr.Row():
                extract_model = gr.Textbox(
                    label="HuggingFace model ID",
                    placeholder="Qwen/Qwen2.5-0.5B-Instruct",
                    scale=3,
                )
                extract_name = gr.Textbox(
                    label="Output vindex name",
                    placeholder="qwen2.5-0.5b.vindex (auto if empty)",
                    scale=2,
                )
            with gr.Row():
                extract_level = gr.Radio(
                    choices=["Browse (smallest, ~0.5 GB)", "Inference (~1 GB)", "All (~2 GB)"],
                    value="Browse (smallest, ~0.5 GB)",
                    label="Extraction level",
                )
            with gr.Row():
                hf_token = gr.Textbox(
                    label="HuggingFace token (required for gated models)",
                    placeholder="hf_…",
                    type="password",
                    scale=2,
                )
                extract_btn = gr.Button("Extract →", variant="primary", scale=1)

            extract_out = gr.Markdown("_Enter a model ID and click Extract._")

            extract_btn.click(
                do_extract,
                inputs=[extract_model, extract_name, extract_level, hf_token],
                outputs=extract_out,
            )

        # ── Tab 6: Setup / About ─────────────────────────────────────────────
        with gr.Tab("ℹ️ Setup & About"):
            gr.Markdown(SETUP_MD)
            gr.Markdown("### Current environment")
            gr.Markdown(binary_status_md())
            gr.Markdown(
                f"- Python: `{sys.version.split()[0]}`\n"
                f"- Gradio: `{gr.__version__}`\n"
                f"- Repo root: `{REPO_ROOT}`\n"
                f"- Models dir: `{MODELS_DIR}`\n"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Check for an available port
    import socket

    def is_port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    port = 7860
    for candidate in range(7860, 7880):
        if is_port_free(candidate):
            port = candidate
            break

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,          # set True to get a public Gradio link
        show_error=True,
        theme=_THEME,
        css=_CSS,
    )
