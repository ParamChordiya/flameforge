"""Validate a normalized dataset and produce an actionable report.

The validator never raises on bad *data* — instead it collects issues into a
:class:`ValidationReport` that the TUI renders, classifying each as informational,
a warning (training can proceed), or an error (those examples are skipped). It
also computes the stats users care about: token distribution, train/eval split,
and total training tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from flameforge.data.formatter import Formatter
from flameforge.data.loader import Dataset, Example

_SHORT_TOKEN_THRESHOLD = 10


class IssueLevel(str, Enum):
    """Severity of a validation finding."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation finding.

    Attributes:
        level: Severity of the finding.
        message: Human-readable description, ready to render.
        count: How many examples are affected.
    """

    level: IssueLevel
    message: str
    count: int = 0


@dataclass
class ValidationReport:
    """The result of validating a dataset.

    Attributes:
        total_examples: Number of examples inspected.
        valid_examples: Number that will actually be trained on.
        issues: Ordered findings to display.
        avg_tokens: Mean token count across valid examples.
        max_tokens: Largest token count seen.
        total_tokens: Sum of tokens across valid examples.
        train_count: Examples assigned to the training split.
        eval_count: Examples assigned to the eval split.
        truncated: How many examples exceed ``max_seq_length``.
    """

    total_examples: int
    valid_examples: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    avg_tokens: float = 0.0
    max_tokens: int = 0
    total_tokens: int = 0
    train_count: int = 0
    eval_count: int = 0
    truncated: int = 0

    @property
    def has_errors(self) -> bool:
        """Whether any error-level issues were found."""
        return any(i.level == IssueLevel.ERROR for i in self.issues)

    @property
    def is_trainable(self) -> bool:
        """Whether at least one valid example remains after validation."""
        return self.valid_examples > 0


def _example_is_empty(example: Example) -> bool:
    """Return True if an example has no usable assistant/output content."""
    if example.is_chat:
        assert example.messages is not None
        assistant_turns = [m for m in example.messages if m.role == "assistant"]
        if not assistant_turns:
            return True
        return all(not m.content.strip() for m in assistant_turns)
    return not (example.text or "").strip()


def _has_unanswered_user(example: Example) -> bool:
    """Return True if a chat ends without an assistant reply to a user turn."""
    if not example.is_chat:
        return False
    assert example.messages is not None
    roles = [m.role for m in example.messages]
    if "user" not in roles:
        return False
    return roles[-1] == "user"


def validate_dataset(
    dataset: Dataset,
    max_seq_length: int = 2048,
    train_eval_split: float = 0.95,
    formatter: Formatter | None = None,
) -> ValidationReport:
    """Validate a dataset and compute statistics for the UI.

    Args:
        dataset: The normalized dataset to inspect.
        max_seq_length: Sequences longer than this will be truncated at train time.
        train_eval_split: Fraction of valid examples used for training.
        formatter: Formatter used for token counting; a heuristic one is created
            if omitted.

    Returns:
        A populated :class:`ValidationReport`.
    """
    fmt = formatter or Formatter()
    report = ValidationReport(total_examples=len(dataset))

    empty_count = 0
    short_count = 0
    unanswered_count = 0
    truncated_count = 0
    duplicates = 0
    token_counts: list[int] = []
    seen: set[str] = set()

    for example in dataset.examples:
        text, n_tokens = fmt.format_with_token_count(example)

        if _example_is_empty(example):
            empty_count += 1
            continue  # skipped: cannot train on an empty target

        if _has_unanswered_user(example):
            unanswered_count += 1

        fingerprint = text.strip()
        if fingerprint in seen:
            duplicates += 1
        else:
            seen.add(fingerprint)

        if n_tokens < _SHORT_TOKEN_THRESHOLD:
            short_count += 1
        if n_tokens > max_seq_length:
            truncated_count += 1

        token_counts.append(n_tokens)

    report.valid_examples = len(token_counts)
    report.truncated = truncated_count
    if token_counts:
        report.total_tokens = sum(token_counts)
        report.avg_tokens = report.total_tokens / len(token_counts)
        report.max_tokens = max(token_counts)
        report.eval_count = max(1, round(report.valid_examples * (1 - train_eval_split)))
        report.eval_count = min(report.eval_count, report.valid_examples - 1) if report.valid_examples > 1 else 0
        report.train_count = report.valid_examples - report.eval_count

    report.issues = _build_issues(
        dataset=dataset,
        valid=report.valid_examples,
        empty=empty_count,
        short=short_count,
        unanswered=unanswered_count,
        truncated=truncated_count,
        duplicates=duplicates,
        max_seq_length=max_seq_length,
    )
    return report


def _build_issues(
    *,
    dataset: Dataset,
    valid: int,
    empty: int,
    short: int,
    unanswered: int,
    truncated: int,
    duplicates: int,
    max_seq_length: int,
) -> list[ValidationIssue]:
    """Assemble the ordered list of findings shown in the validation report."""
    issues: list[ValidationIssue] = [
        ValidationIssue(IssueLevel.OK, f"{valid:,} examples ready to train", valid),
        ValidationIssue(IssueLevel.OK, f"Format: {dataset.data_format.value}"),
    ]
    if empty:
        issues.append(
            ValidationIssue(IssueLevel.ERROR, f"{empty:,} examples have an empty response (will be skipped)", empty)
        )
    if truncated:
        issues.append(
            ValidationIssue(
                IssueLevel.WARNING,
                f"{truncated:,} examples exceed {max_seq_length} tokens (will be truncated)",
                truncated,
            )
        )
    if short:
        issues.append(ValidationIssue(IssueLevel.WARNING, f"{short:,} examples are very short (< 10 tokens)", short))
    if unanswered:
        issues.append(
            ValidationIssue(
                IssueLevel.WARNING, f"{unanswered:,} conversations end on a user turn (no reply)", unanswered
            )
        )
    if duplicates:
        issues.append(ValidationIssue(IssueLevel.WARNING, f"{duplicates:,} duplicate examples found", duplicates))
    else:
        issues.append(ValidationIssue(IssueLevel.OK, "No duplicates found"))
    return issues
