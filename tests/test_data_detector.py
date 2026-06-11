"""Tests for data-format auto-detection and loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameforge.constants import DataFormat
from flameforge.data.detector import detect_format, load_first_records, suggest_csv_mapping
from flameforge.data.loader import count_records, load_dataset
from flameforge.utils.errors import (
    DataFormatError,
    FormatDetectionError,
    UnsupportedFormatError,
)


def test_detect_alpaca(alpaca_file: Path) -> None:
    assert detect_format(alpaca_file) == DataFormat.ALPACA


def test_detect_conversational(chat_file: Path) -> None:
    assert detect_format(chat_file) == DataFormat.CONVERSATIONAL


def test_detect_text(text_file: Path) -> None:
    assert detect_format(text_file) == DataFormat.TEXT


def test_detect_csv(csv_file: Path) -> None:
    assert detect_format(csv_file) == DataFormat.CSV


def test_detect_conversations_key(tmp_path: Path) -> None:
    path = tmp_path / "sharegpt.jsonl"
    path.write_text(
        json.dumps({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}) + "\n",
        encoding="utf-8",
    )
    assert detect_format(path) == DataFormat.CONVERSATIONAL


@pytest.mark.parametrize("a,b", [("question", "answer"), ("prompt", "completion")])
def test_detect_qa_pairs(tmp_path: Path, a: str, b: str) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps({a: "q", b: "r"}) + "\n", encoding="utf-8")
    assert detect_format(path) == DataFormat.ALPACA


def test_detect_unknown_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "weird.jsonl"
    path.write_text(json.dumps({"foo": 1, "bar": 2}) + "\n", encoding="utf-8")
    with pytest.raises(FormatDetectionError) as exc:
        detect_format(path)
    # The error should surface the keys we actually saw.
    assert "foo" in str(exc.value)


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        detect_format(path)


def test_invalid_jsonl_raises_with_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"instruction": "ok", "output": "fine"}\n{not valid}\n', encoding="utf-8")
    with pytest.raises(DataFormatError):
        load_dataset(path)


def test_load_first_records_json_array(tmp_path: Path) -> None:
    path = tmp_path / "arr.json"
    path.write_text(json.dumps([{"instruction": "a", "output": "b"}, {"instruction": "c", "output": "d"}]))
    recs = load_first_records(path, n=1)
    assert len(recs) == 1 and recs[0]["instruction"] == "a"


def test_load_alpaca_normalizes_messages(alpaca_file: Path) -> None:
    ds = load_dataset(alpaca_file)
    assert len(ds) == 3
    ex = ds.examples[0]
    assert ex.is_chat
    assert ex.messages is not None
    roles = [m.role for m in ex.messages]
    assert "user" in roles and "assistant" in roles


def test_load_alpaca_merges_input(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps({"instruction": "Add", "input": "2 and 3", "output": "5"}) + "\n")
    ds = load_dataset(path)
    user_msg = next(m for m in ds.examples[0].messages or [] if m.role == "user")
    assert "Add" in user_msg.content and "2 and 3" in user_msg.content


def test_load_conversation_normalizes_roles(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}) + "\n"
    )
    ds = load_dataset(path)
    roles = [m.role for m in ds.examples[0].messages or []]
    assert roles == ["user", "assistant"]


def test_load_text_splits_on_blank_lines(text_file: Path) -> None:
    ds = load_dataset(text_file)
    assert len(ds) == 2
    assert all(ex.text for ex in ds.examples)


def test_load_csv_auto_maps(csv_file: Path) -> None:
    ds = load_dataset(csv_file)
    assert len(ds) == 2
    ex = ds.examples[0]
    assert ex.messages is not None
    assert any("2+2" in m.content for m in ex.messages)


def test_suggest_csv_mapping(csv_file: Path) -> None:
    headers, mapping = suggest_csv_mapping(csv_file)
    assert headers == ["question", "answer"]
    assert mapping["instruction"] == "question"
    assert mapping["output"] == "answer"


def test_count_records(alpaca_file: Path, csv_file: Path, text_file: Path) -> None:
    assert count_records(alpaca_file) == 3
    assert count_records(csv_file) == 2
    assert count_records(text_file) == 2
