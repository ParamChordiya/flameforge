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

```
  ┌─ Loss ──────────────────────────────────────────────┐
  │ 2.80 │●                                              │
  │      │ ●●                                            │
  │      │   ●●●                                         │
  │      │      ●●●●●●                                   │
  │      │            ●●●●●●●●●●●●●                      │
  │ 0.74 │                         ●●●●●●●●●●●●●●●●●●●●●● │
  │      └───────────────────────────────────────────── │
  └──────────────────────────────────────────────────────┘
```

## ✨ Features

- **Zero configuration.** Device, data format, chat template, and batch size are
  all auto-detected. You only answer questions the tool genuinely can't.
- **Runs everywhere.** NVIDIA GPUs via PyTorch + PEFT + TRL, Apple Silicon via
  MLX. The right backend is chosen for you automatically.
- **Memory-safe by design.** Every model is size-checked against a conservative
  budget *before* loading, so your Mac never freezes mid-train.
- **LoRA, QLoRA, DoRA, and full fine-tuning** with sensible, auto-tuned defaults.
- **A live training dashboard** with a real-time loss chart, throughput, memory
  usage, ETA, and pause / checkpoint / stop controls.
- **Frictionless auth.** Gated models walk you through getting a token instead of
  throwing a cryptic 401.
- **Helpful errors, never tracebacks.** Every failure explains what happened and
  exactly what to do next.
- **Flexible export.** Save the adapter, a merged standalone model, or a GGUF for
  llama.cpp / Ollama.

## 🚀 Quick start

```bash
# Apple Silicon (M1/M2/M3/M4)
pip install "flameforge[mlx]"

# NVIDIA CUDA
pip install "flameforge[cuda]"

# Then just run:
flameforge
```

That's it. The welcome screen confirms your device and memory budget, and from
there it's six guided steps: **model → method → data → config → train → export**.

> **No GPU handy?** FlameForge still launches and runs the whole flow in a
> **simulation mode** with a synthetic loss curve, so you can explore the UI
> before installing a backend.

## 📋 Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA, **or** an Apple Silicon Mac
- CPU-only works for the smallest models, but is very slow

## 🤖 Supported models

FlameForge ships a curated registry (browse it in the TUI), but you can fine-tune
**any** HuggingFace causal-LM by typing its id or pointing at a local path. A few
highlights:

| Model | Size | License | Auth | Cheapest method |
|-------|------|---------|------|-----------------|
| Llama 3.2 1B Instruct | 1.0B | Llama 3.2 | 🔒 | QLoRA ~2 GB |
| Qwen 2.5 0.5B / 1.5B / 3B / 7B | 0.5–7B | Apache 2.0 | — | QLoRA |
| Mistral 7B Instruct v0.3 | 7.0B | Apache 2.0 | — | QLoRA ~5 GB |
| Llama 3.1 8B Instruct | 8.0B | Llama 3.1 | 🔒 | QLoRA ~6 GB |
| Gemma 2 2B / 9B Instruct | 2–9B | Gemma | 🔒 | QLoRA |
| Phi-3 / Phi-3.5 Mini | 3.8B | MIT | — | QLoRA ~3 GB |
| Llama 3.1 70B Instruct | 70B | Llama 3.1 | 🔒 | QLoRA ~40 GB |

🔒 = requires a free HuggingFace account + license acceptance (FlameForge guides
you through it).

## 📂 Data format guide

FlameForge auto-detects your format from the file. All of these work out of the box:

**Alpaca / instruction** (`.jsonl` or `.json`)
```json
{"instruction": "Summarize this.", "input": "Long text…", "output": "Short summary."}
```
Alternate keys are auto-mapped: `prompt`/`completion`, `question`/`answer`.

**Conversational / ShareGPT / OpenAI** (`.jsonl` or `.json`)
```json
{"messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```
`conversations` with `from`/`value` turns is also recognised. Multi-turn supported.

**CSV / TSV** — columns are auto-mapped (e.g. `question` → instruction,
`answer` → output); ambiguous files get a column picker.

**Raw text** (`.txt`) — documents separated by blank lines, for continued
pre-training.

The correct **chat template** for your model family (Llama 3, Mistral, ChatML,
Gemma, Phi-3, Qwen) is applied automatically — preferring the tokenizer's own
template when available. A preview shows you exactly what the model will see
before you commit.

There are ready-to-run samples in [`examples/`](examples/).

## ⚙️ Configuration

Sensible defaults are auto-tuned from your model, hardware, and dataset size
(fewer epochs for tiny datasets, bf16→fp16 on older GPUs, a memory-aware batch
size, and so on). You can accept them or adjust anything on the config screen.

To start from a custom defaults file:
```bash
flameforge --config my_config.yaml      # see configs/default.yaml for the schema
```

Other flags:
```bash
flameforge --model meta-llama/Llama-3.2-3B-Instruct   # pre-select a model
flameforge --max-memory-gb 14                          # cap the memory budget
flameforge --version
```

Authentication is read from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` or
`~/.cache/huggingface/token` (e.g. after `huggingface-cli login`).

## 📦 Export

When training finishes you can export:

- **Adapter** — just the LoRA weights (smallest; load on top of the base model).
- **Merged model** — a standalone model with the adapter baked in.
- **GGUF** — for llama.cpp / Ollama, at your choice of quantization (Q4_K_M,
  Q5_K_M, Q8_0, F16). Requires a local llama.cpp checkout pointed to by
  `FLAMEFORGE_LLAMACPP`.

## 🛟 Troubleshooting

**Out of memory.** FlameForge size-checks before loading, but if a run still OOMs,
switch LoRA → QLoRA, lower the max sequence length, or pass `--max-memory-gb` with
a smaller cap. Your last checkpoint is always saved.

**`ImportError: bitsandbytes` (CUDA QLoRA).** `pip install bitsandbytes`.
bitsandbytes only supports Linux/Windows + NVIDIA; on Mac, FlameForge uses MLX.

**HuggingFace 401 / gated model.** Accept the license on the model's HF page,
then `huggingface-cli login` (or set `HF_TOKEN`). The auth screen walks you
through it.

**Can't reach the Hub.** Check your connection and
[status.huggingface.co](https://status.huggingface.co). You can always train from
a local model directory.

Full operational logs are written to `flameforge.log` (never your data or tokens).

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md). The quality bar is
`ruff check . && ruff format --check . && mypy src/ && pytest` — all green.

## 📄 License

[MIT](LICENSE)
