"""
Utility helpers for the LARQL Gradio demo.
Handles binary discovery, subprocess calls, and output parsing.
"""
import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def find_larql_binary() -> Optional[str]:
    """Locate the larql executable. Returns the path or None."""
    candidates = [
        # Sibling target directory (running from demo/)
        Path(__file__).parent.parent / "target" / "release" / "larql",
        Path(__file__).parent.parent / "target" / "release" / "larql.exe",
        # Current working directory
        Path("larql"),
        Path("larql.exe"),
        # HuggingFace Spaces: binary shipped alongside app.py
        Path(__file__).parent / "larql",
        Path(__file__).parent / "larql.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fall back to PATH
    found = shutil.which("larql")
    return found


LARQL = find_larql_binary()


def larql_available() -> bool:
    return LARQL is not None


def run_larql(*args, timeout: int = 120, env_extra: Optional[dict] = None) -> tuple[int, str, str]:
    """
    Run larql with the given arguments.
    Returns (returncode, stdout, stderr).
    """
    if LARQL is None:
        return 1, "", "larql binary not found. See the Setup tab."
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        result = subprocess.run(
            [LARQL] + list(args),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return 1, "", f"Timed out after {timeout}s"
    except Exception as e:
        return 1, "", str(e)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

_WALK_FEATURE_RE = re.compile(
    r"^\s+\d+\.\s+(F\d+)\s+gate=([+-]?\d+\.\d+)\s+hears=\"([^\"]*)\"\s+c=(\d+\.\d+)\s+down=\[(.+)\]\s*$"
)
_WALK_LAYER_RE = re.compile(r"^Layer (\d+):")


def parse_walk_output(text: str) -> list[dict]:
    """
    Parse the text output of `larql walk` into a list of row dicts.
    Each row: {layer, rank, feature_id, gate, hears, cosine, down_tokens}
    """
    rows = []
    current_layer = None
    for line in text.splitlines():
        m = _WALK_LAYER_RE.match(line)
        if m:
            current_layer = int(m.group(1))
            continue
        m = _WALK_FEATURE_RE.match(line)
        if m and current_layer is not None:
            feature_id, gate_raw, hears, cosine_raw, down_raw = m.groups()
            gate = float(gate_raw)
            cosine = float(cosine_raw)
            # Parse down tokens: "token (score), token2 (score2), ..."
            down_tokens = re.findall(r"(.+?)\s+\([\d.]+\)", down_raw)
            down_str = " · ".join(down_tokens[:5])
            rows.append({
                "Layer": current_layer,
                "Feature": feature_id,
                "Gate": round(gate, 4),
                "Direction": "▲ excites" if gate > 0 else "▼ inhibits",
                "Hears": hears,
                "Cosine": round(cosine, 3),
                "Top tokens (down)": down_str,
            })
    return rows


def load_vindex_info(vindex_path: str) -> dict:
    """Load index.json from a vindex directory."""
    idx = Path(vindex_path) / "index.json"
    if not idx.exists():
        return {}
    try:
        with open(idx) as f:
            return json.load(f)
    except Exception:
        return {}


def format_vindex_summary(info: dict, vindex_path: str) -> str:
    """Render a Markdown summary of a vindex."""
    if not info:
        return "_No index.json found._"
    model = info.get("model", "unknown")
    family = info.get("family", "?")
    layers = info.get("num_layers", "?")
    hidden = info.get("hidden_size", "?")
    intermediate = info.get("intermediate_size", "?")
    vocab = info.get("vocab_size", "?")
    level = info.get("extract_level", "?")
    dtype = info.get("dtype", "?")
    extracted_at = info.get("source", {}).get("extracted_at", "?")
    bands = info.get("layer_bands", {})
    band_str = "  ".join(f"**{k}**: L{v[0]}–L{v[1]}" for k, v in bands.items())

    # File sizes
    size_str = ""
    for fname in ["gate_vectors.bin", "embeddings.bin", "down_meta.bin"]:
        fp = Path(vindex_path) / fname
        if fp.exists():
            mb = fp.stat().st_size / 1e6
            size_str += f"  - `{fname}`: {mb:.1f} MB\n"

    return f"""### {model}

| Property | Value |
|---|---|
| Family | `{family}` |
| Layers | {layers} |
| Hidden size | {hidden} |
| Intermediate size | {intermediate} |
| Vocab size | {vocab:,} |
| Extract level | `{level}` |
| Storage dtype | `{dtype}` |
| Extracted | {extracted_at} |

**Layer bands:** {band_str}

**Files:**
{size_str}"""


def list_local_vindexes(root: str = ".") -> list[str]:
    """Find all .vindex directories in the given root."""
    results = []
    for entry in Path(root).rglob("index.json"):
        vindex_dir = str(entry.parent)
        if not any(part.startswith(".") for part in entry.parts):
            results.append(vindex_dir)
    return sorted(results)
