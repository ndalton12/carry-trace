"""Goal 1 dataset generation."""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

from carry_trace.arithmetic import generate_problem
from carry_trace.config import DatasetConfig
from carry_trace.io import ensure_dir, stable_hash, utc_now_iso, write_json, write_jsonl
from carry_trace.prompts import render_prompt
from carry_trace.schemas import AdditionExample


def generate_dataset(config: DatasetConfig) -> tuple[Path, Path, list[AdditionExample]]:
    config_hash = stable_hash(config.model_dump(mode="json"))
    dataset_dir = ensure_dir(config.output_dir / config.name)
    jsonl_path = dataset_dir / "examples.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    rows: list[AdditionExample] = []

    for split, split_seed_offset in config.splits.items():
        rng = Random(config.seed + split_seed_offset)
        for n_digits in config.digit_lengths:
            for slice_name in config.slices:
                for replica in range(config.examples_per_slice_per_length):
                    problem = generate_problem(
                        n_digits=n_digits,
                        rng=rng,
                        base=config.base,
                        slice_name=slice_name,
                    )
                    for prompt_mode in config.prompt_modes:
                        for digit_format in config.digit_formats:
                            prompt, template_id, messages, prompt_a, prompt_b = render_prompt(
                                problem,
                                prompt_mode,
                                digit_format=digit_format,
                                digit_delimiter=config.digit_delimiter,
                            )
                            row_payload: dict[str, Any] = {
                                **problem,
                                "schema_version": config.schema_version,
                                "split": split,
                                "seed": config.seed,
                                "generator_config_hash": config_hash,
                                "slice_name": slice_name,
                                "prompt_mode": prompt_mode,
                                "digit_format": digit_format,
                                "digit_delimiter": config.digit_delimiter,
                                "prompt_a": prompt_a,
                                "prompt_b": prompt_b,
                                "template_id": template_id,
                                "prompt": prompt,
                                "messages": messages,
                                "first_carry_position": _first_or_none(problem["carry_positions"]),
                                "metadata": {"replica": replica},
                            }
                            row_payload["id"] = stable_hash(
                                {
                                    "split": split,
                                    "n_digits": n_digits,
                                    "slice": slice_name,
                                    "replica": replica,
                                    "prompt_mode": prompt_mode,
                                    "digit_format": digit_format,
                                    "a": problem["a"],
                                    "b": problem["b"],
                                    "seed": config.seed,
                                },
                                length=16,
                            )
                            rows.append(AdditionExample.model_validate(row_payload))

    write_jsonl(jsonl_path, [row.model_dump(mode="json") for row in rows])
    parquet_path = None
    if config.write_parquet:
        parquet_path = dataset_dir / "examples.parquet"
        pd.DataFrame([row.model_dump(mode="json") for row in rows]).to_parquet(parquet_path)

    manifest = {
        "name": config.name,
        "created_at": utc_now_iso(),
        "schema_version": config.schema_version,
        "config_hash": config_hash,
        "config": config.model_dump(mode="json"),
        "row_count": len(rows),
        "jsonl_path": str(jsonl_path),
        "parquet_path": str(parquet_path) if parquet_path else None,
    }
    write_json(manifest_path, manifest)
    return jsonl_path, manifest_path, rows


def _first_or_none(values: object) -> int | None:
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, int):
            return first
    return None
