"""Curated registry of popular fine-tunable models with rich metadata.

This list powers the "Browse Popular Models" tab in the TUI. It is purely a
convenience layer: a user can always type any HuggingFace model id or point at a
local path. Each entry carries enough metadata to estimate memory, recommend a
method, and drive the auth flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flameforge.constants import FineTuningMethod
from flameforge.device.memory import estimate_model_memory_gb


@dataclass(frozen=True)
class ModelInfo:
    """Metadata describing a single registered model.

    Attributes:
        hf_id: The HuggingFace repository id.
        display_name: Friendly name shown in the UI.
        family: Model family key (drives the chat template).
        param_count: Approximate parameter count.
        context_length: Maximum context window in tokens.
        license: License name/identifier.
        description: One-line description for the browser.
        recommended_methods: Methods recommended for this model, best first.
        min_memory_gb: Rough minimum memory (GB) per method to train.
        chat_template: Chat-template key (see ``data.templates``).
        requires_auth: Whether a HuggingFace token + license acceptance is needed.
    """

    hf_id: str
    display_name: str
    family: str
    param_count: int
    context_length: int
    license: str
    description: str
    recommended_methods: list[str]
    min_memory_gb: dict[str, int]
    chat_template: str
    requires_auth: bool = False
    tags: list[str] = field(default_factory=list)

    @property
    def param_count_str(self) -> str:
        """Human-readable parameter count (e.g. "8.0B", "3.8B", "500M")."""
        if self.param_count >= 1_000_000_000:
            return f"{self.param_count / 1e9:.1f}B"
        return f"{self.param_count / 1e6:.0f}M"

    def cheapest_method(self) -> str:
        """Return the lowest-memory recommended method for this model."""
        return min(self.recommended_methods, key=lambda m: self.min_memory_gb.get(m, 10**9))

    def fits_in_budget(self, budget_gb: float) -> bool:
        """Whether the model fits in ``budget_gb`` using its cheapest method."""
        method = self.cheapest_method()
        return estimate_model_memory_gb(self.param_count, method) <= budget_gb

    def estimated_memory_gb(self, method: FineTuningMethod | str) -> float:
        """Estimated training memory (GB) for this model under ``method``."""
        return estimate_model_memory_gb(self.param_count, method)


POPULAR_MODELS: list[ModelInfo] = [
    ModelInfo(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama 3.1 8B Instruct",
        family="llama",
        param_count=8_000_000_000,
        context_length=131072,
        license="Llama 3.1 Community",
        description="Meta's flagship 8B instruction-tuned model. Great all-rounder.",
        recommended_methods=["qlora", "lora"],
        min_memory_gb={"qlora": 6, "lora": 18, "full": 64},
        chat_template="llama3",
        requires_auth=True,
    ),
    ModelInfo(
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama 3.2 3B Instruct",
        family="llama",
        param_count=3_000_000_000,
        context_length=131072,
        license="Llama 3.2 Community",
        description="Compact Llama model. Fits on most hardware with LoRA.",
        recommended_methods=["lora", "qlora", "full"],
        min_memory_gb={"qlora": 3, "lora": 8, "full": 24},
        chat_template="llama3",
        requires_auth=True,
    ),
    ModelInfo(
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        display_name="Llama 3.2 1B Instruct",
        family="llama",
        param_count=1_000_000_000,
        context_length=131072,
        license="Llama 3.2 Community",
        description="Tiny but capable. Can even full fine-tune on consumer hardware.",
        recommended_methods=["lora", "full", "qlora"],
        min_memory_gb={"qlora": 2, "lora": 4, "full": 8},
        chat_template="llama3",
        requires_auth=True,
    ),
    ModelInfo(
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        display_name="Mistral 7B Instruct v0.3",
        family="mistral",
        param_count=7_000_000_000,
        context_length=32768,
        license="Apache 2.0",
        description="Strong 7B model with Apache license. No auth required.",
        recommended_methods=["qlora", "lora"],
        min_memory_gb={"qlora": 5, "lora": 16, "full": 56},
        chat_template="mistral",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="mistralai/Mistral-Nemo-Instruct-2407",
        display_name="Mistral Nemo 12B Instruct",
        family="mistral",
        param_count=12_000_000_000,
        context_length=131072,
        license="Apache 2.0",
        description="12B multilingual model with a huge context window. Apache 2.0.",
        recommended_methods=["qlora", "lora"],
        min_memory_gb={"qlora": 8, "lora": 28, "full": 96},
        chat_template="mistral",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="google/gemma-2-9b-it",
        display_name="Gemma 2 9B Instruct",
        family="gemma",
        param_count=9_000_000_000,
        context_length=8192,
        license="Gemma Terms",
        description="Google's instruction-tuned 9B model. Strong reasoning.",
        recommended_methods=["qlora", "lora"],
        min_memory_gb={"qlora": 6, "lora": 20, "full": 72},
        chat_template="gemma",
        requires_auth=True,
    ),
    ModelInfo(
        hf_id="google/gemma-2-2b-it",
        display_name="Gemma 2 2B Instruct",
        family="gemma",
        param_count=2_000_000_000,
        context_length=8192,
        license="Gemma Terms",
        description="Compact Gemma model. Great for experimentation.",
        recommended_methods=["lora", "qlora", "full"],
        min_memory_gb={"qlora": 2, "lora": 6, "full": 16},
        chat_template="gemma",
        requires_auth=True,
    ),
    ModelInfo(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen 2.5 7B Instruct",
        family="qwen",
        param_count=7_000_000_000,
        context_length=131072,
        license="Apache 2.0",
        description="Alibaba's multilingual 7B model. Excellent for non-English.",
        recommended_methods=["qlora", "lora"],
        min_memory_gb={"qlora": 5, "lora": 16, "full": 56},
        chat_template="qwen",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        display_name="Qwen 2.5 3B Instruct",
        family="qwen",
        param_count=3_000_000_000,
        context_length=131072,
        license="Qwen Research",
        description="Compact Qwen. Good multilingual support.",
        recommended_methods=["lora", "qlora", "full"],
        min_memory_gb={"qlora": 3, "lora": 8, "full": 24},
        chat_template="qwen",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        display_name="Qwen 2.5 1.5B Instruct",
        family="qwen",
        param_count=1_500_000_000,
        context_length=131072,
        license="Apache 2.0",
        description="Tiny Qwen. Fast iteration, full fine-tune on modest hardware.",
        recommended_methods=["lora", "full", "qlora"],
        min_memory_gb={"qlora": 2, "lora": 5, "full": 12},
        chat_template="qwen",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        display_name="Qwen 2.5 0.5B Instruct",
        family="qwen",
        param_count=500_000_000,
        context_length=131072,
        license="Apache 2.0",
        description="Smallest Qwen. Ideal for testing the full pipeline quickly.",
        recommended_methods=["full", "lora", "qlora"],
        min_memory_gb={"qlora": 1, "lora": 2, "full": 4},
        chat_template="qwen",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="microsoft/Phi-3-mini-4k-instruct",
        display_name="Phi-3 Mini (3.8B)",
        family="phi",
        param_count=3_800_000_000,
        context_length=4096,
        license="MIT",
        description="Microsoft's small but mighty model. MIT licensed.",
        recommended_methods=["lora", "qlora", "full"],
        min_memory_gb={"qlora": 3, "lora": 9, "full": 30},
        chat_template="phi3",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="microsoft/Phi-3.5-mini-instruct",
        display_name="Phi-3.5 Mini (3.8B)",
        family="phi",
        param_count=3_800_000_000,
        context_length=131072,
        license="MIT",
        description="Updated Phi-3.5 with a 128k context window. MIT licensed.",
        recommended_methods=["lora", "qlora", "full"],
        min_memory_gb={"qlora": 3, "lora": 9, "full": 30},
        chat_template="phi3",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        display_name="SmolLM2 1.7B Instruct",
        family="llama",
        param_count=1_700_000_000,
        context_length=8192,
        license="Apache 2.0",
        description="Efficient small model from HuggingFace. Llama-style template.",
        recommended_methods=["lora", "full", "qlora"],
        min_memory_gb={"qlora": 2, "lora": 5, "full": 14},
        chat_template="chatml",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        display_name="TinyLlama 1.1B Chat",
        family="llama",
        param_count=1_100_000_000,
        context_length=2048,
        license="Apache 2.0",
        description="Famously tiny Llama chat model. Perfect for smoke tests.",
        recommended_methods=["lora", "full", "qlora"],
        min_memory_gb={"qlora": 2, "lora": 4, "full": 9},
        chat_template="chatml",
        requires_auth=False,
    ),
    ModelInfo(
        hf_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        display_name="Llama 3.1 70B Instruct",
        family="llama",
        param_count=70_000_000_000,
        context_length=131072,
        license="Llama 3.1 Community",
        description="Massive 70B model. QLoRA only, needs a high-memory GPU.",
        recommended_methods=["qlora"],
        min_memory_gb={"qlora": 40, "lora": 160, "full": 560},
        chat_template="llama3",
        requires_auth=True,
    ),
]

# Index for O(1) lookup by HuggingFace id.
_MODELS_BY_ID: dict[str, ModelInfo] = {m.hf_id: m for m in POPULAR_MODELS}


def get_model(hf_id: str) -> ModelInfo | None:
    """Look up a registered model by its exact HuggingFace id.

    Args:
        hf_id: The HuggingFace repository id.

    Returns:
        The matching :class:`ModelInfo`, or None if not registered.
    """
    return _MODELS_BY_ID.get(hf_id)


def search_models(query: str) -> list[ModelInfo]:
    """Case-insensitively search the registry by id, name, or family.

    Args:
        query: A free-text query; empty returns the full list.

    Returns:
        Matching models in registry order.
    """
    q = query.strip().lower()
    if not q:
        return list(POPULAR_MODELS)
    return [m for m in POPULAR_MODELS if q in m.hf_id.lower() or q in m.display_name.lower() or q in m.family.lower()]


def models_fitting_budget(budget_gb: float) -> list[ModelInfo]:
    """Return registered models that fit within ``budget_gb`` using any method.

    Args:
        budget_gb: The available memory budget in GB.

    Returns:
        Models whose cheapest method fits the budget.
    """
    return [m for m in POPULAR_MODELS if m.fits_in_budget(budget_gb)]
