"""Answer parsing for model generations."""

from __future__ import annotations

import re

from carry_trace.arithmetic import DIGIT_ALPHABET
from carry_trace.enums import AnswerFormat

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*([0-9A-Z,|_]+)", re.IGNORECASE)


def normalize_answer(text: str | None, base: int = 10) -> str | None:
    """Normalize a conventional most-significant-first answer string."""
    if text is None:
        return None
    allowed = DIGIT_ALPHABET[:base]
    cleaned = text.strip().upper().replace(",", "").replace("|", "")
    if "_" in cleaned:
        cleaned = cleaned.split("_", 1)[0]
    cleaned = "".join(char for char in cleaned if char in allowed)
    if not cleaned:
        return None
    stripped = cleaned.lstrip("0")
    return stripped or "0"


def normalize_output_digits(text: str | None, base: int = 10) -> str | None:
    """Extract answer digits in emitted order while preserving leading zeros."""
    if text is None:
        return None
    allowed = DIGIT_ALPHABET[:base]
    cleaned = text.strip().upper().replace(",", "").replace("|", "")
    if "_" in cleaned:
        cleaned = cleaned.split("_", 1)[0]
    cleaned = "".join(char for char in cleaned if char in allowed)
    return cleaned or None


def output_digits_to_canonical(
    output_digits: str | None,
    answer_format: AnswerFormat | str = AnswerFormat.STANDARD,
    base: int = 10,
) -> str | None:
    """Convert emitted answer digits to canonical most-significant-first form."""
    if output_digits is None:
        return None
    answer_format = AnswerFormat(answer_format)
    if answer_format == AnswerFormat.STANDARD:
        return normalize_answer(output_digits, base=base)
    if answer_format == AnswerFormat.LSD:
        return normalize_answer("".join(reversed(output_digits)), base=base)
    raise ValueError(f"unknown answer format {answer_format!r}")


def parse_final_output_digits(text: str, base: int = 10) -> str | None:
    """Parse the final answer-like digit sequence in emitted order."""
    for regex in (BOXED_RE, ANSWER_RE):
        matches = regex.findall(text)
        if matches:
            parsed = normalize_output_digits(matches[-1], base=base)
            if parsed is not None:
                return parsed

    allowed = DIGIT_ALPHABET[:base]
    token_re = re.compile(rf"[{re.escape(allowed)},|]+(?:_{base})?", re.IGNORECASE)
    candidates = token_re.findall(text.upper())
    for candidate in reversed(candidates):
        parsed = normalize_output_digits(candidate, base=base)
        if parsed is not None:
            return parsed
    return None


def parse_final_answer(
    text: str,
    base: int = 10,
    answer_format: AnswerFormat | str = AnswerFormat.STANDARD,
) -> str | None:
    """Parse the final answer and convert it to canonical MSD-first form."""
    output_digits = parse_final_output_digits(text, base=base)
    return output_digits_to_canonical(output_digits, answer_format, base)
