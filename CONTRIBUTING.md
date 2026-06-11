# Contributing to FlameForge

Thanks for your interest in improving FlameForge! This project aims to make LLM
fine-tuning approachable for everyone, and contributions of all sizes are
welcome.

## Development setup

```bash
git clone https://github.com/parampatil/flameforge.git
cd flameforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # base + dev tools
# add a backend for end-to-end work:
pip install -e ".[dev,mlx]"      # Apple Silicon
pip install -e ".[dev,cuda]"     # NVIDIA
```

## Quality gate

Every change must pass the same checks CI runs. Please run them locally before
opening a pull request:

```bash
ruff check .             # lint
ruff format --check .    # formatting
mypy src/                # type checking
pytest -v                # tests
```

`ruff check . --fix` and `ruff format .` will fix most issues automatically.

## Guidelines

- **Type hints everywhere.** All functions and methods are fully annotated.
- **Docstrings on every public function, class, and module.**
- **Errors must be actionable.** Raise a `FlameForgeError` subclass with a clear
  message and concrete suggestions — never let a raw traceback reach the TUI.
- **Tests for non-GPU logic.** Data parsing, config validation, memory
  estimation, and device detection are all unit-tested without a GPU.
- **Keep the TUI responsive.** Long-running work belongs in a worker thread.

## Adding a model to the registry

Edit `src/flameforge/models/registry.py` and append a `ModelInfo` entry with
accurate `param_count`, `min_memory_gb`, `chat_template`, and `requires_auth`
values. Add a test in `tests/test_model_registry.py` if it exercises new logic.

## Reporting bugs

Open an issue with your platform, the FlameForge version (`flameforge
--version`), and the relevant section of `flameforge.log`. The log never contains
your data or tokens — only operational events.
