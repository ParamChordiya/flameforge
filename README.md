<div align="center">

# 🔥 FlameForge

**Fine-tune any LLM with a beautiful TUI. Zero config, maximum power.**

Supports NVIDIA CUDA (PyTorch/PEFT) and Apple Silicon (MLX).

[![CI](https://github.com/parampatil/flameforge/actions/workflows/ci.yml/badge.svg)](https://github.com/parampatil/flameforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

FlameForge is a terminal user interface that takes you from raw data to a working
fine-tuned model in under ten minutes of active effort — no YAML wrangling, no
CUDA incantations, no guessing whether a model will fit in memory. Pick a model,
pick a method, load your data, and watch the loss curve fall in real time.

## Features

- **Zero configuration.** Device, data format, chat template, and batch size are
  all auto-detected. You only answer questions the tool genuinely can't.
- **Runs everywhere.** NVIDIA GPUs via PyTorch + PEFT + TRL, Apple Silicon via
  MLX. The right backend is chosen for you.
- **Memory-safe by design.** Every model is size-checked against a conservative
  budget *before* loading, so your Mac never freezes.
- **LoRA, QLoRA, DoRA, and full fine-tuning** with sensible, auto-tuned defaults.
- **A live training dashboard** with a real-time loss chart, throughput, memory
  usage, and ETA.
- **Helpful errors, never tracebacks.** Every failure explains what happened and
  what to do next.

## Quick start

```bash
# Apple Silicon
pip install "flameforge[mlx]"

# NVIDIA CUDA
pip install "flameforge[cuda]"

# Then just run:
flameforge
```

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA, **or** an Apple Silicon Mac (M1/M2/M3/M4)
- (CPU-only works for the smallest models, but is very slow)

## Documentation

A full guide to supported models, data formats, configuration, and
troubleshooting lives further down this README and is expanded with each release.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) to hack on FlameForge itself.

## License

[MIT](LICENSE)
