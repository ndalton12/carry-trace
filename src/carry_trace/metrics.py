"""Behavioral metrics for Goal 1 runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from carry_trace.arithmetic import str_to_digits_lsd
from carry_trace.enums import AnswerFormat
from carry_trace.io import ensure_dir, read_jsonl, write_jsonl
from carry_trace.parsing import (
    normalize_answer,
    normalize_output_digits,
    output_digits_to_canonical,
    parse_final_output_digits,
)


def score_records(
    examples: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score model-call records against their dataset examples."""
    examples_by_id = {row["id"]: row for row in examples}
    scored = []
    for record in records:
        example = examples_by_id[record["example_id"]]
        base = int(example["base"])
        answer_format = AnswerFormat(example.get("answer_format", AnswerFormat.STANDARD))
        answer = normalize_answer(example["answer"], base=base)
        expected_output = normalize_output_digits(
            example.get("expected_output") or example["answer"],
            base=base,
        )
        parsed_output = parse_final_output_digits(record.get("decoded_output") or "", base=base)
        parsed = output_digits_to_canonical(parsed_output, answer_format, base)
        decoded = normalize_answer(record.get("decoded_output"), base=base)
        first_lsd, first_msd = first_wrong_digit_positions(parsed, answer, base)
        scored_record = {
            **record,
            "exact_match": decoded == answer,
            "parsed_output": parsed_output,
            "expected_output": expected_output,
            "parsed_output_format_correct": parsed_output == expected_output,
            "parsed_answer_correct": parsed == answer,
            "first_wrong_digit_lsd": first_lsd,
            "first_wrong_digit_msd": first_msd,
        }
        scored.append(scored_record)
    return scored


def first_wrong_digit_positions(
    parsed: str | None,
    answer: str | None,
    base: int = 10,
) -> tuple[int | None, int | None]:
    """Return first wrong digit positions from LSD and MSD directions."""
    if parsed is None or answer is None:
        return 0, 0
    if parsed == answer:
        return None, None
    parsed_digits = str_to_digits_lsd(parsed, base=base)
    answer_digits = str_to_digits_lsd(answer, base=base)
    max_len = max(len(parsed_digits), len(answer_digits))
    for idx in range(max_len):
        parsed_digit = parsed_digits[idx] if idx < len(parsed_digits) else None
        answer_digit = answer_digits[idx] if idx < len(answer_digits) else None
        if parsed_digit != answer_digit:
            first_lsd = idx
            break
    else:
        first_lsd = None

    parsed_msd = list(reversed(parsed_digits))
    answer_msd = list(reversed(answer_digits))
    for idx in range(max_len):
        parsed_digit = parsed_msd[idx] if idx < len(parsed_msd) else None
        answer_digit = answer_msd[idx] if idx < len(answer_msd) else None
        if parsed_digit != answer_digit:
            first_msd = idx
            break
    else:
        first_msd = None
    return first_lsd, first_msd


def summarize_goal1(
    scored_records: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate scored Goal 1 records into a metrics summary table."""
    examples_df = pd.DataFrame(examples)
    records_df = pd.DataFrame(scored_records)
    merged = records_df.merge(
        examples_df[
            [
                "id",
                "prompt_mode",
                "digit_format",
                "answer_format",
                "slice_name",
                "n_digits",
                "carry_count",
                "max_carry_chain",
                "first_carry_position",
            ]
        ],
        left_on="example_id",
        right_on="id",
        how="left",
    )
    merged["has_carry"] = merged["carry_count"] > 0
    grouped = merged.groupby(
        [
            "model_name",
            "prompt_mode",
            "digit_format",
            "answer_format",
            "n_digits",
            "max_carry_chain",
            "has_carry",
        ],
        dropna=False,
    )
    return grouped.agg(
        examples=("example_id", "count"),
        parsed_accuracy=("parsed_answer_correct", "mean"),
        output_format_accuracy=("parsed_output_format_correct", "mean"),
        exact_match_accuracy=("exact_match", "mean"),
        avg_output_tokens=("token_count_output", "mean"),
        avg_latency_seconds=("latency_seconds", "mean"),
    ).reset_index()


def score_run(run_dir: Path) -> tuple[Path, Path]:
    """Score a saved run directory and write scored-call and summary artifacts."""
    examples = read_jsonl(run_dir / "dataset.jsonl")
    records = read_jsonl(run_dir / "calls.jsonl")
    scored = score_records(examples, records)
    scored_path = run_dir / "scored_calls.jsonl"
    summary_path = run_dir / "metrics_summary.csv"
    ensure_dir(run_dir)
    write_jsonl(scored_path, scored)
    summarize_goal1(scored, examples).to_csv(summary_path, index=False)
    return scored_path, summary_path
