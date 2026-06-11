"""Auto-detect a dataset's on-disk format from its extension and contents.

Detection is best-effort and conservative: when a JSON/JSONL file's schema is
ambiguous we raise :class:`FormatDetectionError` with the keys we saw rather than
guessing wrong. CSV/TSV files always need column mapping, so they resolve to
:data:`DataFormat.CSV` and the UI offers a column picker.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from flameforge.constants import DataFormat
from flameforge.utils.errors import (
    DataFormatError,
    FormatDetectionError,
    UnsupportedFormatError,
)

_CONVERSATION_KEYS = ("messages", "conversations")
_INSTRUCTION_KEYS = ("instruction", "prompt", "human")
_QA_PAIRS = (("question", "answer"), ("prompt", "completion"), ("human", "assistant"))

_JSON_SUFFIXES = (".jsonl", ".json")
_CSV_SUFFIXES = (".csv", ".tsv")


def load_first_records(path: Path, n: int = 5) -> list[dict[str, object]]:
    """Load up to ``n`` JSON records from a .json or .jsonl file.

    Args:
        path: The file to sample.
        n: Maximum number of records to return.

    Returns:
        Up to ``n`` record dicts (fewer if the file is shorter).

    Raises:
        DataFormatError: If the file cannot be parsed as JSON/JSONL.
    """
    suffix = path.suffix.lower()
    records: list[dict[str, object]] = []

    if suffix == ".json":
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataFormatError(
                message=f"'{path.name}' is not valid JSON.",
                suggestions=["Check the file for syntax errors."],
                details=str(exc),
            ) from exc
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items[:n]:
            if isinstance(item, dict):
                records.append(item)
        return records

    # JSONL: read line by line so we never load a huge file just to peek.
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataFormatError(
                    message=f"'{path.name}' is not valid JSONL.",
                    suggestions=["Each line must be a complete JSON object."],
                    details=str(exc),
                ) from exc
            if isinstance(obj, dict):
                records.append(obj)
            if len(records) >= n:
                break
    return records


def _classify_record(record: dict[str, object]) -> DataFormat | None:
    """Classify a single JSON record, or None if it matches no known schema."""
    keys = set(record.keys())
    if keys & set(_CONVERSATION_KEYS):
        return DataFormat.CONVERSATIONAL
    if keys & set(_INSTRUCTION_KEYS):
        return DataFormat.ALPACA
    for a, b in _QA_PAIRS:
        if a in keys and b in keys:
            return DataFormat.ALPACA
    return None


def detect_format(path: str | Path) -> DataFormat:
    """Detect the :class:`DataFormat` of a data file.

    Args:
        path: Path to the data file.

    Returns:
        The detected format.

    Raises:
        UnsupportedFormatError: If the extension is not supported.
        FormatDetectionError: If a JSON/JSONL schema matches no known format.
        DataFormatError: If a supported file cannot be read.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".txt":
        return DataFormat.TEXT
    if suffix in _CSV_SUFFIXES:
        return DataFormat.CSV
    if suffix not in _JSON_SUFFIXES:
        raise UnsupportedFormatError(suffix)

    sample = load_first_records(p, n=5)
    if not sample:
        raise DataFormatError(
            message=f"'{p.name}' contains no JSON records to inspect.",
            suggestions=["Ensure the file is non-empty and contains JSON objects."],
        )
    for record in sample:
        result = _classify_record(record)
        if result is not None:
            return result
    raise FormatDetectionError(keys=sample[0].keys())


def suggest_csv_mapping(path: str | Path) -> tuple[list[str], dict[str, str]]:
    """Read a CSV/TSV header and suggest a logical-field → column mapping.

    Args:
        path: Path to the CSV/TSV file.

    Returns:
        A tuple of (header columns, suggested mapping). The mapping keys are
        logical fields ("instruction"/"input"/"output").

    Raises:
        DataFormatError: If the file has no header row.
    """
    p = Path(path)
    delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader, None)
    if not headers:
        raise DataFormatError(
            message=f"'{p.name}' has no header row.",
            suggestions=["Add a header row naming each column."],
        )

    lower = {h.lower(): h for h in headers}
    mapping: dict[str, str] = {}
    aliases = {
        "instruction": ("instruction", "prompt", "question", "input_text"),
        "input": ("input", "context"),
        "output": ("output", "completion", "answer", "response", "target"),
    }
    for logical, candidates in aliases.items():
        for alias in candidates:
            if alias in lower:
                mapping[logical] = lower[alias]
                break
    if "instruction" not in mapping and headers:
        mapping["instruction"] = headers[0]
    if "output" not in mapping and len(headers) > 1:
        mapping["output"] = headers[1]
    return headers, mapping
