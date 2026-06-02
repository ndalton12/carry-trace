"""Inspection utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DIGITS = tuple("0123456789")
DEFAULT_DIGIT_COMBINATION_MAX_LENGTH = 4
EXACT_DIGIT_STRINGS_MAX_LENGTH = 3
SAMPLE_LIMIT = 40
ADDITION_SPLIT_STYLES = {
    "space_between_digits": " ",
    "pipe_between_digits": "|",
    "space_pipe_space_between_digits": " | ",
}


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode text without adding tokenizer-specific special tokens."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _tokens_for_ids(tokenizer: Any, input_ids: Iterable[int]) -> list[str]:
    """Convert token IDs to tokenizer vocabulary token strings."""
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
    if isinstance(tokens, str):
        return [tokens]
    return list(tokens)


def _tokenization_record(tokenizer: Any, text: str) -> dict[str, Any]:
    """Return token IDs and token strings for a specific text sample."""
    input_ids = _encode(tokenizer, text)
    tokens = _tokens_for_ids(tokenizer, input_ids)
    return {
        "text": text,
        "n_tokens": len(input_ids),
        "input_ids": input_ids,
        "tokens": tokens,
        "is_single_token": len(input_ids) == 1,
        "token_id": input_ids[0] if len(input_ids) == 1 else None,
        "token": tokens[0] if len(tokens) == 1 else None,
    }


def _decode_token_id(tokenizer: Any, token_id: int) -> str:
    """Decode one token ID to the text surface it contributes."""
    try:
        return tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def _is_ascii_digit_string(text: str) -> bool:
    """Return whether text is a non-empty string of ASCII base-10 digits."""
    return bool(text) and all(char in DIGITS for char in text)


def _ascii_digit_count(text: str) -> int:
    """Count ASCII base-10 digits in text."""
    return sum(char in DIGITS for char in text)


def _digit_token_variant(decoded_text: str) -> tuple[str, str] | None:
    """Classify decoded token text as a bare or delimiter-prefixed digit string."""
    if _is_ascii_digit_string(decoded_text):
        return "bare", decoded_text
    if decoded_text.startswith(" ") and _is_ascii_digit_string(decoded_text[1:]):
        return "leading_space", decoded_text[1:]
    if decoded_text.startswith("|") and _is_ascii_digit_string(decoded_text[1:]):
        return "leading_pipe", decoded_text[1:]
    return None


def _vocab_digit_token_records(tokenizer: Any) -> dict[str, list[dict[str, Any]]]:
    """Scan the vocabulary for token IDs whose decoded text is a digit string."""
    vocab = tokenizer.get_vocab()
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))
    records: dict[str, list[dict[str, Any]]] = {
        "bare": [],
        "leading_space": [],
        "leading_pipe": [],
    }

    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        decoded_text = _decode_token_id(tokenizer, token_id)
        variant = _digit_token_variant(decoded_text)
        if variant is None:
            continue

        variant_name, digits = variant
        records[variant_name].append(
            {
                "digits": digits,
                "text": decoded_text,
                "token_id": token_id,
                "token": token,
                "length": len(digits),
                "is_special_token": token_id in special_token_ids,
                "reencodes_to_same_single_token": _encode(tokenizer, decoded_text) == [token_id],
            }
        )

    return records


def _summarize_digit_token_records(
    records: list[dict[str, Any]],
    *,
    max_length: int,
    exact_strings_max_length: int,
    sample_limit: int,
) -> dict[str, Any]:
    """Summarize discovered digit-string token records by digit-string length."""
    by_length: dict[str, Any] = {}
    max_discovered_length = max((record["length"] for record in records), default=0)
    max_summary_length = max(max_length, max_discovered_length)

    for length in range(1, max_summary_length + 1):
        length_records = [record for record in records if record["length"] == length]
        unique_digit_strings = sorted({record["digits"] for record in length_records})
        by_length[str(length)] = {
            "total_possible_digit_strings": len(DIGITS) ** length if length <= max_length else None,
            "unique_digit_strings_count": len(unique_digit_strings),
            "token_count": len(length_records),
            "unique_digit_string_fraction": (
                len(unique_digit_strings) / (len(DIGITS) ** length)
                if length <= max_length
                else None
            ),
            "exact_records": length_records if length <= exact_strings_max_length else None,
            "record_samples": length_records[:sample_limit],
        }

    return by_length


def _single_token_digit_coverage(
    tokenizer: Any,
    *,
    max_length: int = DEFAULT_DIGIT_COMBINATION_MAX_LENGTH,
    exact_strings_max_length: int = EXACT_DIGIT_STRINGS_MAX_LENGTH,
    sample_limit: int = SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Find digit-string tokens by scanning decoded vocabulary entries."""
    digit_token_records = _vocab_digit_token_records(tokenizer)
    coverage = {
        variant_name: _summarize_digit_token_records(
            records,
            max_length=max_length,
            exact_strings_max_length=exact_strings_max_length,
            sample_limit=sample_limit,
        )
        for variant_name, records in digit_token_records.items()
    }

    return {
        "method": "scan vocabulary, decode each token ID, then classify digit-string surfaces",
        "max_fraction_denominator_length": max_length,
        "exact_strings_max_length": exact_strings_max_length,
        "variants": {
            "bare": "decoded token text is only digits, e.g. '487'",
            "leading_space": "decoded token text is one leading space plus digits, e.g. ' 487'",
            "leading_pipe": "decoded token text is one leading pipe plus digits, e.g. '|487'",
        },
        "coverage": coverage,
    }


def _prefixes(text: str) -> list[str]:
    """Return all prefixes of text, including the empty prefix."""
    return [text[:length] for length in range(len(text) + 1)]


def _suffixes(text: str) -> list[str]:
    """Return all suffixes of text, including the empty suffix."""
    return [text[index:] for index in range(len(text) + 1)]


def _can_cover_multiple_split_digits(decoded_text: str, separator: str) -> bool:
    """Check whether decoded text can span multiple digits in a split digit sequence."""
    if _ascii_digit_count(decoded_text) < 2:
        return False

    digits = [char for char in decoded_text if char in DIGITS]
    digit_core = separator.join(digits)
    return any(
        decoded_text == leading + digit_core + trailing
        for leading in _suffixes(separator)
        for trailing in _prefixes(separator)
    )


def _addition_split_guarantees(
    tokenizer: Any,
    *,
    sample_limit: int = SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Scan vocabulary for tokens that could span multiple split digits."""
    vocab = tokenizer.get_vocab()
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))
    style_records: dict[str, list[dict[str, Any]]] = {
        style_name: [] for style_name in ADDITION_SPLIT_STYLES
    }

    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        decoded_text = _decode_token_id(tokenizer, token_id)
        for style_name, separator in ADDITION_SPLIT_STYLES.items():
            if not _can_cover_multiple_split_digits(decoded_text, separator):
                continue
            style_records[style_name].append(
                {
                    "text": decoded_text,
                    "token_id": token_id,
                    "token": token,
                    "digit_count": _ascii_digit_count(decoded_text),
                    "is_special_token": token_id in special_token_ids,
                    "reencodes_to_same_single_token": (
                        _encode(tokenizer, decoded_text) == [token_id]
                    ),
                }
            )

    return {
        "method": (
            "scan decoded vocabulary for any token text that could be a substring of "
            "a split digit sequence while covering two or more digits"
        ),
        "styles": {
            style_name: {
                "separator": separator,
                "guarantees_one_token_per_digit_for_arbitrary_length_numbers": not records,
                "multi_digit_token_count": len(records),
                "multi_digit_token_samples": records[:sample_limit],
            }
            for style_name, separator in ADDITION_SPLIT_STYLES.items()
            for records in [style_records[style_name]]
        },
    }


def inspect_tokenizer(
    model_id: str,
    revision: str | None = None,
    *,
    digit_combination_max_length: int = DEFAULT_DIGIT_COMBINATION_MAX_LENGTH,
    exact_digit_strings_max_length: int = EXACT_DIGIT_STRINGS_MAX_LENGTH,
    sample_limit: int = SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Inspect tokenizer metadata and arithmetic-relevant digit tokenization behavior."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    sample = "4879 + 2568 = ?"
    sample_record = _tokenization_record(tokenizer, sample)
    addition_split_guarantees = _addition_split_guarantees(tokenizer, sample_limit=sample_limit)
    return {
        "model_id": model_id,
        "revision": revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "configured_tokenizer_class": getattr(tokenizer, "init_kwargs", {}).get("tokenizer_class"),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "sample": sample,
        "sample_input_ids": sample_record["input_ids"],
        "sample_tokens": sample_record["tokens"],
        "single_digit_tokens": {
            "bare": [_tokenization_record(tokenizer, digit) for digit in DIGITS],
            "leading_space": [_tokenization_record(tokenizer, f" {digit}") for digit in DIGITS],
            "leading_pipe": [_tokenization_record(tokenizer, f"|{digit}") for digit in DIGITS],
        },
        "digit_string_single_token_coverage": _single_token_digit_coverage(
            tokenizer,
            max_length=digit_combination_max_length,
            exact_strings_max_length=exact_digit_strings_max_length,
            sample_limit=sample_limit,
        ),
        "addition_split_guarantees": addition_split_guarantees,
        "recommended_addition_split_styles": [
            style_name
            for style_name, style_info in addition_split_guarantees["styles"].items()
            if style_info["guarantees_one_token_per_digit_for_arbitrary_length_numbers"]
        ],
        "note": (
            "OLMo 3 Hugging Face configs identify GPT2Tokenizer, "
            "i.e. GPT-2-style byte-level BPE. A digit string is reported as an "
            "actual token when encoding that exact text yields exactly one token ID."
        ),
    }
