"""Export full prompt and output text from a Goal 1 run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def merged_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Merge scored call records with their source examples."""
    examples = {row["id"]: row for row in read_jsonl(run_dir / "dataset.jsonl")}
    calls_path = run_dir / "scored_calls.jsonl"
    if not calls_path.exists():
        calls_path = run_dir / "calls.jsonl"
    rows = []
    for record in read_jsonl(calls_path):
        example = examples[record["example_id"]]
        rows.append(
            {
                "model_name": record.get("model_name"),
                "example_id": record.get("example_id"),
                "problem_id": example.get("problem_id"),
                "n_digits": example.get("n_digits"),
                "slice_name": example.get("slice_name"),
                "prompt_mode": example.get("prompt_mode"),
                "digit_format": example.get("digit_format"),
                "answer_format": example.get("answer_format"),
                "a": example.get("a"),
                "b": example.get("b"),
                "answer": example.get("answer"),
                "expected_output": example.get("expected_output"),
                "parsed_answer": record.get("parsed_answer"),
                "parsed_output": record.get("parsed_output"),
                "parsed_answer_correct": record.get("parsed_answer_correct"),
                "parsed_output_format_correct": record.get("parsed_output_format_correct"),
                "token_count_input": record.get("token_count_input"),
                "token_count_output": record.get("token_count_output"),
                "latency_seconds": record.get("latency_seconds"),
                "prompt": record.get("prompt"),
                "rendered_prompt": record.get("rendered_prompt"),
                "decoded_output": record.get("decoded_output"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write output rows as a CSV file."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a human-readable full prompt/output dump."""
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write("=" * 100 + "\n")
            handle.write(f"Example {index}/{len(rows)} example_id={row['example_id']}\n")
            for key in [
                "model_name",
                "problem_id",
                "n_digits",
                "slice_name",
                "prompt_mode",
                "digit_format",
                "answer_format",
                "a",
                "b",
                "answer",
                "expected_output",
                "parsed_answer",
                "parsed_output",
                "parsed_answer_correct",
                "parsed_output_format_correct",
                "token_count_input",
                "token_count_output",
                "latency_seconds",
            ]:
                handle.write(f"{key}: {row.get(key)}\n")
            handle.write("prompt:\n")
            handle.write(f"{row.get('prompt')}\n")
            handle.write("rendered_prompt:\n")
            handle.write(f"{row.get('rendered_prompt')}\n")
            handle.write("decoded_output:\n")
            handle.write(f"{row.get('decoded_output')}\n")


def export_outputs(run_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Export full Goal 1 outputs to JSONL, CSV, and text files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = merged_rows(run_dir)
    jsonl_path = output_dir / "full_outputs.jsonl"
    csv_path = output_dir / "full_outputs.csv"
    text_path = output_dir / "full_outputs.txt"
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    write_text(text_path, rows)
    return jsonl_path, csv_path, text_path


def main() -> None:
    """Parse CLI arguments and export a Goal 1 run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir
    for path in export_outputs(args.run_dir, output_dir):
        print(path)


if __name__ == "__main__":
    main()
