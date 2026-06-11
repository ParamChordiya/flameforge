"""Load datasets from disk and normalize them to a common representation.

Every supported on-disk format (Alpaca JSON/JSONL, conversational JSON/JSONL,
CSV/TSV, and raw text) is parsed into a list of :class:`Example` objects. An
example is either a *chat* (an ordered list of :class:`Message`) or a *text*
document. Downstream code (formatter, validator) only ever deals with this
normalized form, never the raw files.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from flameforge.constants import DataFormat
from flameforge.data.detector import detect_format, load_first_records, suggest_csv_mapping
from flameforge.utils.errors import DataFormatError
from flameforge.utils.logging import get_logger

_log = get_logger("data.loader")

# Accepted key aliases, mapped to the canonical Alpaca field they represent.
_INSTRUCTION_KEYS = ("instruction", "prompt", "question", "human")
_INPUT_KEYS = ("input", "context")
_OUTPUT_KEYS = ("output", "completion", "answer", "response", "assistant")
_MESSAGES_KEYS = ("messages", "conversations")
# Role/content key aliases inside a single conversation turn.
_ROLE_KEYS = ("role", "from")
_CONTENT_KEYS = ("content", "value", "text")
_ROLE_NORMALIZE = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
}


@dataclass(frozen=True)
class Message:
    """A single conversation turn.

    Attributes:
        role: One of "system", "user", or "assistant".
        content: The text of the turn.
    """

    role: str
    content: str


@dataclass
class Example:
    """A single normalized training example.

    Exactly one of ``messages`` or ``text`` is populated.

    Attributes:
        messages: The conversation turns, for chat/instruction data.
        text: The raw document, for language-modeling data.
        raw: The original record, retained for debugging and validation.
    """

    messages: list[Message] | None = None
    text: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def is_chat(self) -> bool:
        """Whether this example is a conversation (vs. raw text)."""
        return self.messages is not None


@dataclass
class Dataset:
    """A loaded, normalized dataset.

    Attributes:
        data_format: The detected/decided on-disk format.
        examples: The normalized examples.
        source_path: Where the data was loaded from.
    """

    data_format: DataFormat
    examples: list[Example]
    source_path: Path

    def __len__(self) -> int:
        return len(self.examples)


def _first_present(record: dict[str, object], keys: tuple[str, ...]) -> str | None:
    """Return the first present, non-None value among ``keys`` as a string."""
    for key in keys:
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def _normalize_alpaca(record: dict[str, object]) -> Example:
    """Convert an Alpaca-style record into a chat :class:`Example`.

    Args:
        record: A raw dict with instruction/input/output-style keys.

    Returns:
        An :class:`Example` with system/user/assistant messages.

    Raises:
        DataFormatError: If neither an instruction nor an output can be found.
    """
    instruction = _first_present(record, _INSTRUCTION_KEYS)
    output = _first_present(record, _OUTPUT_KEYS)
    extra_input = _first_present(record, _INPUT_KEYS)
    system = _first_present(record, ("system", "system_prompt"))

    if instruction is None and output is None:
        raise DataFormatError(
            message="An Alpaca record is missing both an instruction and an output.",
            suggestions=["Ensure each row has an 'instruction'/'prompt' and an 'output'/'answer'."],
            details=f"Record keys: {list(record.keys())}",
        )

    user_text = instruction or ""
    if extra_input:
        user_text = f"{user_text}\n\n{extra_input}" if user_text else extra_input

    messages: list[Message] = []
    if system:
        messages.append(Message("system", system))
    messages.append(Message("user", user_text))
    messages.append(Message("assistant", output or ""))
    return Example(messages=messages, raw=record)


def _normalize_conversation(record: dict[str, object]) -> Example:
    """Convert a conversational record (messages/conversations) to an Example.

    Args:
        record: A raw dict containing a list of role/content turns.

    Returns:
        An :class:`Example` with normalized messages.

    Raises:
        DataFormatError: If the turns list is missing or malformed.
    """
    turns_obj: object | None = None
    for key in _MESSAGES_KEYS:
        if key in record:
            turns_obj = record[key]
            break
    if not isinstance(turns_obj, list):
        raise DataFormatError(
            message="A conversational record has no list of messages.",
            suggestions=["Each row needs a 'messages' or 'conversations' list of turns."],
            details=f"Record keys: {list(record.keys())}",
        )

    messages: list[Message] = []
    for turn in turns_obj:
        if not isinstance(turn, dict):
            continue
        raw_role = _first_present(turn, _ROLE_KEYS) or "user"
        content = _first_present(turn, _CONTENT_KEYS) or ""
        role = _ROLE_NORMALIZE.get(raw_role.lower(), raw_role.lower())
        messages.append(Message(role, content))

    if not messages:
        raise DataFormatError(
            message="A conversational record contained no usable turns.",
            suggestions=["Each turn needs a role ('user'/'assistant') and content."],
        )
    return Example(messages=messages, raw=record)


def _load_jsonl_or_json(path: Path) -> list[dict[str, object]]:
    """Load a .jsonl (one object per line) or .json (array/object) file.

    Args:
        path: The file to read.

    Returns:
        A list of record dicts.

    Raises:
        DataFormatError: If the file is not valid JSON or contains non-objects.
    """
    text = path.read_text(encoding="utf-8")
    records: list[dict[str, object]] = []

    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataFormatError(
                message=f"'{path.name}' is not valid JSON.",
                suggestions=[
                    "Check for trailing commas or unquoted keys.",
                    "JSONL (one object per line) is also supported.",
                ],
                details=str(exc),
            ) from exc
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if isinstance(item, dict):
                records.append(item)
        return records

    # JSONL: tolerate blank lines, report the offending line on error.
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DataFormatError(
                message=f"Invalid JSON on line {lineno} of '{path.name}'.",
                suggestions=["Each line must be a complete JSON object.", f"Offending line: {stripped[:80]}"],
                details=str(exc),
            ) from exc
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _load_text(path: Path) -> list[Example]:
    """Load a raw-text file into text examples split on blank lines."""
    content = path.read_text(encoding="utf-8")
    # Split documents on blank lines; fall back to the whole file as one doc.
    chunks = [chunk.strip() for chunk in content.split("\n\n")]
    docs = [c for c in chunks if c]
    if not docs and content.strip():
        docs = [content.strip()]
    return [Example(text=doc, raw={"text": doc}) for doc in docs]


def _load_csv(path: Path, column_mapping: dict[str, str] | None) -> list[Example]:
    """Load a CSV/TSV file, mapping columns to Alpaca fields.

    Args:
        path: The CSV/TSV file.
        column_mapping: Optional explicit mapping of logical field
            ("instruction"/"input"/"output") to column name. If omitted, columns
            are auto-mapped by header name.

    Returns:
        Normalized chat examples.

    Raises:
        DataFormatError: If the file has no header row.
    """
    mapping = column_mapping or suggest_csv_mapping(path)[1]
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        examples: list[Example] = []
        for row in reader:
            record: dict[str, object] = {logical: row.get(column, "") for logical, column in mapping.items()}
            examples.append(_normalize_alpaca(record))
    return examples


def load_dataset(
    path: str | Path,
    data_format: DataFormat | None = None,
    column_mapping: dict[str, str] | None = None,
) -> Dataset:
    """Load and normalize a dataset from a file.

    Args:
        path: Path to the data file.
        data_format: The format to use; auto-detected if None.
        column_mapping: For CSV/TSV, an optional logical-field → column mapping.

    Returns:
        A normalized :class:`Dataset`.

    Raises:
        DataFormatError: If the file cannot be parsed or yields no examples.
        UnsupportedFormatError: If the file extension is unsupported.
    """
    p = Path(path)
    fmt = data_format or detect_format(p)
    _log.info("Loading dataset %s as %s", p, fmt.value)

    if fmt == DataFormat.TEXT:
        examples = _load_text(p)
    elif fmt == DataFormat.CSV:
        examples = _load_csv(p, column_mapping)
    else:
        records = _load_jsonl_or_json(p)
        if not records:
            raise DataFormatError(
                message=f"No records found in '{p.name}'.",
                suggestions=["The file is empty or contains no JSON objects."],
            )
        if fmt == DataFormat.CONVERSATIONAL:
            examples = [_normalize_conversation(r) for r in records]
        else:
            examples = [_normalize_alpaca(r) for r in records]

    if not examples:
        raise DataFormatError(
            message=f"'{p.name}' produced no usable training examples.",
            suggestions=["Check the file contents match the detected format."],
        )
    return Dataset(data_format=fmt, examples=examples, source_path=p)


def count_records(path: str | Path) -> int:
    """Cheaply count records without fully normalizing the dataset.

    Args:
        path: Path to the data file.

    Returns:
        The number of records (lines for JSONL/CSV, documents for text).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return len(_load_text(p))
    if suffix in (".csv", ".tsv"):
        with p.open("r", encoding="utf-8", newline="") as f:
            return max(0, sum(1 for _ in f) - 1)  # minus header
    if suffix == ".json":
        return len(load_first_records(p, n=10**9))
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
