"""Answer parsing for model generations."""

from __future__ import annotations

import re

from carry_trace.arithmetic import DIGIT_ALPHABET

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*([0-9A-Z,|_]+)", re.IGNORECASE)


def normalize_answer(text: str | None, base: int = 10) -> str | None:
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


def parse_final_answer(text: str, base: int = 10) -> str | None:
    for regex in (BOXED_RE, ANSWER_RE):
        matches = regex.findall(text)
        if matches:
            parsed = normalize_answer(matches[-1], base=base)
            if parsed is not None:
                return parsed

    allowed = DIGIT_ALPHABET[:base]
    token_re = re.compile(rf"[{re.escape(allowed)},|]+(?:_{base})?", re.IGNORECASE)
    candidates = token_re.findall(text.upper())
    for candidate in reversed(candidates):
        parsed = normalize_answer(candidate, base=base)
        if parsed is not None:
            return parsed
    return None
