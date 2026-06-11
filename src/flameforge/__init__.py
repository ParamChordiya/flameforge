"""FlameForge — fine-tune any open-source LLM from a beautiful terminal UI.

FlameForge is a Textual-based TUI that takes a user from raw data to a working
fine-tuned model with zero configuration. It auto-detects the compute backend
(NVIDIA CUDA via PyTorch/PEFT, or Apple Silicon via MLX), budgets memory safely,
and guides the user through model selection, data loading, training, and export.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
