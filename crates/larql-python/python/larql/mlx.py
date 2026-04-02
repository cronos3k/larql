"""
Load MLX models directly from a vindex.

No safetensors. No intermediate files. The vindex binary files are mmap'd
and wrapped as mx.array views — zero-copy on Apple Silicon unified memory.

Usage:
    import larql
    model, tokenizer = larql.mlx.load("gemma3-4b.vindex")
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _weight_prefix(config: dict) -> Tuple[str, str]:
    """Determine the weight name prefix MLX expects for this architecture."""
    family = config.get("family", "")
    model_type = config.get("model_config", {}).get("model_type", family)
    if "gemma3" in model_type:
        return "language_model.model.", "language_model."
    return "model.", ""


def _load_weights(vindex_path: str) -> dict:
    """Mmap vindex binaries and produce {tensor_name: mx.array} dict.

    Mmap'd numpy views are converted to mx.array in bfloat16 (what MLX uses
    natively). For f32 vindexes this halves memory and speeds up load.
    """
    import mlx.core as mx

    vpath = Path(vindex_path)

    with open(vpath / "index.json") as f:
        config = json.load(f)
    with open(vpath / "weight_manifest.json") as f:
        manifest = json.load(f)

    dtype_str = config.get("dtype", "f32")
    np_dtype = np.float16 if dtype_str == "f16" else np.float32
    hidden = config["hidden_size"]
    prefix, lm_prefix = _weight_prefix(config)

    def to_mx(view):
        return mx.array(view)

    # Mmap each binary file once
    mmaps = {}
    for entry in manifest:
        fname = entry["file"]
        if fname not in mmaps:
            fpath = vpath / fname
            if fpath.exists():
                mmaps[fname] = np.memmap(str(fpath), dtype=np.uint8, mode="r")

    weights = {}

    # Manifest weights (attention, FFN up/down, norms, lm_head)
    for entry in manifest:
        raw_name = entry["key"]
        if "vision_tower" in raw_name or "multi_modal" in raw_name:
            continue

        fname = entry["file"]
        if fname not in mmaps:
            continue

        shape = tuple(entry["shape"])
        offset = entry["offset"]
        length = entry["length"]

        view = mmaps[fname][offset:offset + length].view(np_dtype).reshape(shape)

        if raw_name == "lm_head.weight":
            name = f"{lm_prefix}lm_head.weight"
        elif raw_name == "norm.weight":
            name = f"{prefix}norm.weight"
        else:
            name = f"{prefix}{raw_name}"

        weights[name] = to_mx(view)

    # Embeddings
    embed_path = vpath / "embeddings.bin"
    if embed_path.exists():
        vocab = config.get("vocab_size", 0)
        if vocab > 0:
            em = np.memmap(str(embed_path), dtype=np_dtype, mode="r",
                           shape=(vocab, hidden))
            weights[f"{prefix}embed_tokens.weight"] = to_mx(em)

    # FFN gate (gate_vectors.bin = mlp.gate_proj.weight)
    gate_path = vpath / "gate_vectors.bin"
    if gate_path.exists():
        gate_mm = np.memmap(str(gate_path), dtype=np.uint8, mode="r")
        for info in config.get("layers", []):
            layer = info["layer"]
            nf = info["num_features"]
            off = info["offset"]
            length = info["length"]
            view = gate_mm[off:off + length].view(np_dtype).reshape(nf, hidden)
            weights[f"{prefix}layers.{layer}.mlp.gate_proj.weight"] = to_mx(view)

    return weights


def _build_config(vindex_path: str) -> dict:
    """Build MLX config. Uses HF cache if available, else vindex metadata."""
    vpath = Path(vindex_path)

    with open(vpath / "index.json") as f:
        vc = json.load(f)

    # Try HF cache first (has all fields MLX needs)
    try:
        from mlx_lm.utils import load_config, hf_repo_to_path
        hf_path = hf_repo_to_path(vc.get("model", ""))
        if hf_path:
            return load_config(hf_path)
    except Exception:
        pass

    # Build from vindex
    mc = vc.get("model_config", {})
    tc = {
        "model_type": mc.get("model_type", vc.get("family", "")),
        "hidden_size": vc["hidden_size"],
        "intermediate_size": vc["intermediate_size"],
        "num_hidden_layers": vc["num_layers"],
        "vocab_size": vc["vocab_size"],
        "head_dim": mc.get("head_dim", 256),
        "num_attention_heads": mc.get("num_q_heads", 8),
        "num_key_value_heads": mc.get("num_kv_heads", 4),
        "rope_theta": mc.get("rope_base", 1000000.0),
        "rms_norm_eps": 1e-6,
    }

    if "gemma3" in vc.get("family", ""):
        return {"model_type": "gemma3", "text_config": tc,
                "sliding_window": mc.get("sliding_window", 1024)}
    return tc


def load(vindex_path: str, lazy: bool = False) -> Tuple:
    """Load an MLX model directly from a vindex.

    No safetensors. Weights are mmap'd from vindex binaries and
    wrapped as mx.array — zero-copy on unified memory.

    Args:
        vindex_path: Path to .vindex directory
        lazy: If True, don't eval weights immediately

    Returns:
        (model, tokenizer) — ready for mlx_lm.generate()
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm.utils as mlx_utils

    vpath = Path(vindex_path)

    config = _build_config(vindex_path)
    model_class, model_args_class = mlx_utils._get_classes(config=config)
    model = model_class(model_args_class.from_dict(config))

    weights = _load_weights(vindex_path)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model.eval()
    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    tokenizer = mlx_utils.load_tokenizer(vpath)

    return model, tokenizer
