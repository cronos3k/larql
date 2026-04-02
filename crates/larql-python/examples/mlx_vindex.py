"""
MLX generation powered by vindex.

Weights load from the vindex, not safetensors.
The vindex IS the model.

Usage:
    python examples/demos/mlx_vindex.py
"""

import larql
import mlx_lm

# Load model from vindex — weights come from vindex binary files
model, tokenizer = larql.mlx.load("output/gemma3-4b-v2.vindex")

# Generate
response = mlx_lm.generate(model, tokenizer, prompt="The capital of France is", max_tokens=20)
print(response)
