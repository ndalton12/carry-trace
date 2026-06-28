"""Goal 1 dataset generation."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from random import Random
from typing import Any

import pandas as pd

from carry_trace.arithmetic import generate_problem, generate_random_problem_with_carry_count
from carry_trace.config import DatasetConfig, SplitConfig
from carry_trace.enums import AnswerFormat, DigitFormat, SliceName
from carry_trace.io import ensure_dir, stable_hash, utc_now_iso, write_json, write_jsonl
from carry_trace.prompts import render_prompt
from carry_trace.schemas import AdditionExample

OPTIONAL_DATASET_FIELDS = {
    "match_group_id",
    "match_role",
    "target_column_lsd",
    "intervention_variable",
    "match_family",
    "match_constraints",
    "partner_problem_ids",
}


def generate_dataset(config: DatasetConfig) -> tuple[Path, Path, list[AdditionExample]]:
    """Generate and save a labeled synthetic addition dataset."""
    config_hash = stable_hash(config.model_dump(mode="json"))
    dataset_dir = ensure_dir(config.output_dir / config.name)
    jsonl_path = dataset_dir / "examples.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    rows: list[AdditionExample] = []

    for split, split_config in config.splits.items():
        split_seed = _split_seed(config.seed, split)
        rng = Random(split_seed)
        for n_digits in config.digit_lengths:
            for slice_name, digit_format, answer_format in _format_conditions(config):
                examples_per_slice_per_length = _examples_per_slice_per_length(
                    config,
                    split_config,
                    slice_name,
                )
                carry_count_targets = _carry_count_targets(
                    n_digits,
                    examples_per_slice_per_length,
                    rng,
                    balance=(
                        slice_name == SliceName.RANDOM
                        and config.random_sampling.balance_carry_count
                    ),
                )
                for replica, target_carry_count in enumerate(carry_count_targets):
                    metadata: dict[str, Any] = {"replica": replica}
                    if target_carry_count is None:
                        problem = generate_problem(
                            n_digits=n_digits,
                            rng=rng,
                            base=config.base,
                            slice_name=slice_name,
                        )
                    else:
                        problem = generate_random_problem_with_carry_count(
                            n_digits=n_digits,
                            rng=rng,
                            base=config.base,
                            carry_count=target_carry_count,
                        )
                        metadata["target_carry_count"] = target_carry_count

                    for prompt_mode in config.prompt_modes:
                        (
                            prompt,
                            template_id,
                            messages,
                            prompt_a,
                            prompt_b,
                            expected_output,
                        ) = render_prompt(
                            problem,
                            prompt_mode,
                            digit_format=digit_format,
                            digit_delimiter=config.digit_delimiter,
                            answer_format=answer_format,
                        )
                        row_payload: dict[str, Any] = {
                            **problem,
                            "schema_version": config.schema_version,
                            "split": split,
                            "seed": config.seed,
                            "split_seed": split_seed,
                            "generator_config_hash": config_hash,
                            "slice_name": slice_name,
                            "prompt_mode": prompt_mode,
                            "digit_format": digit_format,
                            "digit_delimiter": config.digit_delimiter,
                            "answer_format": answer_format,
                            "expected_output": expected_output,
                            "prompt_a": prompt_a,
                            "prompt_b": prompt_b,
                            "template_id": template_id,
                            "prompt": prompt,
                            "messages": messages,
                            "first_carry_position": _first_or_none(problem["carry_positions"]),
                            "metadata": metadata,
                        }
                        row_payload["problem_id"] = stable_hash(
                            {
                                "split": split,
                                "n_digits": n_digits,
                                "slice": slice_name,
                                "replica": replica,
                                "a": problem["a"],
                                "b": problem["b"],
                                "seed": config.seed,
                            },
                            length=16,
                        )
                        row_payload["id"] = stable_hash(
                            {
                                "problem_id": row_payload["problem_id"],
                                "split": split,
                                "n_digits": n_digits,
                                "slice": slice_name,
                                "replica": replica,
                                "prompt_mode": prompt_mode,
                                "digit_format": digit_format,
                                "answer_format": answer_format,
                                "a": problem["a"],
                                "b": problem["b"],
                                "seed": config.seed,
                            },
                            length=16,
                        )
                        rows.append(AdditionExample.model_validate(row_payload))

    write_jsonl(jsonl_path, [dump_dataset_row(row) for row in rows])
    parquet_path = None
    if config.write_parquet:
        parquet_path = dataset_dir / "examples.parquet"
        pd.DataFrame([dump_dataset_row(row) for row in rows]).to_parquet(parquet_path)

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


def upload_dataset_to_hub(
    dataset_dir: Path,
    repo_id: str,
    *,
    path_in_repo: str | None = None,
    private: bool = False,
    revision: str | None = None,
    create_pr: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    create_repo: bool = True,
) -> dict[str, str | None]:
    """Upload one generated dataset directory into a subdirectory of a HF dataset repo."""
    from huggingface_hub import HfApi

    dataset_dir = dataset_dir.expanduser().resolve()
    _validate_dataset_dir(dataset_dir)
    upload_path = _validate_hub_dataset_path(path_in_repo or dataset_dir.name)
    api = HfApi()

    if create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )

    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
        path_in_repo=upload_path,
        revision=revision,
        create_pr=create_pr,
        token=token,
        commit_message=commit_message or f"Upload carry-trace dataset {dataset_dir.name}",
        ignore_patterns=[".DS_Store"],
    )
    return {
        "repo_id": repo_id,
        "dataset_dir": str(dataset_dir),
        "path_in_repo": upload_path,
        "commit_url": str(getattr(commit_info, "commit_url", "")) or None,
    }


def _validate_dataset_dir(dataset_dir: Path) -> None:
    """Validate that a path looks like a generated carry-trace dataset directory."""
    if not dataset_dir.is_dir():
        raise ValueError(f"dataset_dir must be a directory: {dataset_dir}")
    missing = [
        filename
        for filename in ("examples.jsonl", "manifest.json")
        if not (dataset_dir / filename).is_file()
    ]
    if missing:
        raise ValueError(f"dataset_dir is missing required files: {', '.join(missing)}")


def _validate_hub_dataset_path(path_in_repo: str) -> str:
    """Validate the non-root HF repo path used for a local dataset upload."""
    path = path_in_repo.strip("/")
    parts = PurePosixPath(path).parts
    if not path or path == "." or "\\" in path or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path_in_repo must be a non-root relative path")
    return path


def _format_conditions(config: DatasetConfig) -> list[tuple[SliceName, DigitFormat, AnswerFormat]]:
    """Return valid slice and format conditions for dataset generation."""
    conditions: list[tuple[SliceName, DigitFormat, AnswerFormat]] = []
    digit_formats = set(config.digit_formats)
    answer_formats = set(config.answer_formats)
    if DigitFormat.STANDARD in digit_formats and AnswerFormat.STANDARD in answer_formats:
        conditions.extend(
            (slice_name, DigitFormat.STANDARD, AnswerFormat.STANDARD)
            for slice_name in config.slices
        )
    if DigitFormat.DELIMITED in digit_formats and AnswerFormat.STANDARD in answer_formats:
        conditions.append((SliceName.RANDOM, DigitFormat.DELIMITED, AnswerFormat.STANDARD))
    if DigitFormat.STANDARD in digit_formats and AnswerFormat.LSD in answer_formats:
        conditions.append((SliceName.RANDOM, DigitFormat.STANDARD, AnswerFormat.LSD))
    return conditions


def _examples_per_slice_per_length(
    config: DatasetConfig,
    split_config: SplitConfig,
    slice_name: SliceName,
) -> int:
    """Return the replicate count for one split and slice condition."""
    if slice_name in split_config.slice_examples_per_length:
        return split_config.slice_examples_per_length[slice_name]
    if split_config.examples_per_slice_per_length is not None:
        return split_config.examples_per_slice_per_length
    if slice_name in config.slice_examples_per_length:
        return config.slice_examples_per_length[slice_name]
    return config.examples_per_slice_per_length


def _carry_count_targets(
    n_digits: int,
    count: int,
    rng: Random,
    balance: bool,
) -> list[int | None]:
    """Return target carry counts for balanced random generation."""
    if not balance:
        return [None] * count

    targets: list[int] = []
    carry_counts = list(range(n_digits + 1))
    while len(targets) < count:
        cycle = carry_counts.copy()
        rng.shuffle(cycle)
        targets.extend(cycle)
    return targets[:count]


def _first_or_none(values: object) -> int | None:
    """Return the first integer from a list-like object, if present."""
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, int):
            return first
    return None


def dump_dataset_row(row: AdditionExample) -> dict[str, Any]:
    """Serialize a dataset row while omitting empty optional Goal 3 metadata."""
    payload = row.model_dump(mode="json")
    for field in OPTIONAL_DATASET_FIELDS:
        value = payload.get(field)
        if value is None or value == [] or value == {}:
            payload.pop(field, None)
    return payload


def _split_seed(base_seed: int, split: str) -> int:
    """Derive a deterministic per-split RNG seed from the base seed and split name."""
    return int(stable_hash({"seed": base_seed, "split": split}, length=16), 16)
