"""Tests for chat-template formatting, token counting, and validation."""

from __future__ import annotations

from pathlib import Path

from flameforge.constants import DataFormat
from flameforge.data.formatter import Formatter
from flameforge.data.loader import Dataset, Example, Message, load_dataset
from flameforge.data.templates import CHAT_TEMPLATES, resolve_template_key
from flameforge.data.validator import IssueLevel, validate_dataset


def _chat(*turns: tuple[str, str]) -> Example:
    return Example(messages=[Message(r, c) for r, c in turns])


def test_resolve_template_key_by_family() -> None:
    assert resolve_template_key(family="llama") == "llama3"
    assert resolve_template_key(family="qwen") == "qwen"
    assert resolve_template_key(template_key="phi3") == "phi3"
    # Unknown family falls back to the default.
    assert resolve_template_key(family="unknown") == "chatml"


def test_all_families_render() -> None:
    example = _chat(("system", "Be nice."), ("user", "Hi"), ("assistant", "Hello!"))
    for key in CHAT_TEMPLATES:
        text = Formatter(template_key=key).format_example(example)
        assert "Hi" in text and "Hello!" in text


def test_llama3_template_markers() -> None:
    text = Formatter(template_key="llama3").format_example(_chat(("user", "Q"), ("assistant", "A")))
    assert "<|begin_of_text|>" in text
    assert "<|start_header_id|>user<|end_header_id|>" in text
    assert "<|eot_id|>" in text


def test_system_merged_for_mistral() -> None:
    # Mistral has no system role; the system text must fold into the user turn.
    text = Formatter(template_key="mistral").format_example(_chat(("system", "SECRET"), ("user", "hello")))
    assert "SECRET" in text
    assert "[INST]" in text


def test_multi_turn_renders_all_turns() -> None:
    text = Formatter(template_key="qwen").format_example(
        _chat(("user", "one"), ("assistant", "two"), ("user", "three"), ("assistant", "four"))
    )
    for token in ("one", "two", "three", "four"):
        assert token in text


def test_text_example_passthrough() -> None:
    text = Formatter(template_key="llama3").format_example(Example(text="raw document"))
    assert text == "raw document"


def test_token_count_heuristic() -> None:
    fmt = Formatter(template_key="llama3")
    assert fmt.count_tokens("") == 1  # never zero
    assert fmt.count_tokens("a" * 40) == 10


def test_uses_tokenizer_when_available() -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [0] * len(text.split())

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            return "TEMPLATED:" + " ".join(m["content"] for m in messages)

    fmt = Formatter(template_key="llama3", tokenizer=FakeTokenizer())
    text = fmt.format_example(_chat(("user", "hello world"), ("assistant", "hi")))
    assert text.startswith("TEMPLATED:")
    assert fmt.count_tokens("one two three") == 3


def test_to_mlx_record() -> None:
    rec = Formatter(template_key="qwen").to_mlx_record(_chat(("user", "x"), ("assistant", "y")))
    assert set(rec) == {"text"}
    assert "x" in rec["text"]


def test_validate_clean_dataset(alpaca_file: Path) -> None:
    ds = load_dataset(alpaca_file)
    report = validate_dataset(ds, max_seq_length=2048)
    assert report.is_trainable
    assert not report.has_errors
    assert report.valid_examples == 3
    assert report.train_count + report.eval_count == report.valid_examples


def test_validate_flags_empty_output() -> None:
    ds = Dataset(
        data_format=DataFormat.ALPACA,
        examples=[_chat(("user", "q"), ("assistant", "")), _chat(("user", "q2"), ("assistant", "real"))],
        source_path=Path("mem"),
    )
    report = validate_dataset(ds)
    assert report.has_errors
    assert report.valid_examples == 1
    assert any(i.level == IssueLevel.ERROR for i in report.issues)


def test_validate_flags_truncation() -> None:
    long_text = "word " * 5000
    ds = Dataset(
        data_format=DataFormat.ALPACA,
        examples=[_chat(("user", "q"), ("assistant", long_text))],
        source_path=Path("mem"),
    )
    report = validate_dataset(ds, max_seq_length=128)
    assert report.truncated == 1
    assert any(i.level == IssueLevel.WARNING and "truncated" in i.message for i in report.issues)


def test_validate_detects_duplicates() -> None:
    dup = _chat(("user", "same"), ("assistant", "same"))
    ds = Dataset(
        data_format=DataFormat.ALPACA,
        examples=[
            _chat(("user", "same"), ("assistant", "same")),
            _chat(("user", "same"), ("assistant", "same")),
        ],
        source_path=Path("mem"),
    )
    report = validate_dataset(ds)
    assert any("duplicate" in i.message.lower() for i in report.issues)
    assert dup.messages is not None  # sanity: helper builds a chat example
