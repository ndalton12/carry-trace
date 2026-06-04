"""Experiment orchestration."""

from __future__ import annotations

from pathlib import Path

from carry_trace.config import ExperimentConfig
from carry_trace.datasets import dump_dataset_row
from carry_trace.io import ensure_dir, read_jsonl, stable_hash, utc_now_iso, write_json, write_jsonl
from carry_trace.metrics import score_run
from carry_trace.models import make_runner
from carry_trace.schemas import AdditionExample, ModelCallRecord


def run_goal1(config: ExperimentConfig) -> Path:
    """Run a Goal 1 experiment and write all run artifacts."""
    run_id = f"{config.name}-{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    run_dir = ensure_dir(config.output_dir / run_id)
    raw_examples = read_jsonl(config.dataset_path)
    examples = [AdditionExample.model_validate(row) for row in raw_examples]
    if config.splits is not None:
        allowed = set(config.splits)
        examples = [example for example in examples if example.split in allowed]
    if config.prompt_modes is not None:
        allowed = set(config.prompt_modes)
        examples = [example for example in examples if example.prompt_mode in allowed]
    if config.digit_formats is not None:
        allowed = set(config.digit_formats)
        examples = [example for example in examples if example.digit_format in allowed]
    if config.answer_formats is not None:
        allowed = set(config.answer_formats)
        examples = [example for example in examples if example.answer_format in allowed]
    if config.max_examples is not None:
        examples = examples[: config.max_examples]

    write_jsonl(
        run_dir / "dataset.jsonl",
        [dump_dataset_row(example) for example in examples],
    )
    manifest = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "config_hash": stable_hash(config.model_dump(mode="json")),
        "config": config.model_dump(mode="json"),
        "dataset_path": str(config.dataset_path),
        "example_count": len(examples),
    }
    write_json(run_dir / "manifest.json", manifest)

    records: list[ModelCallRecord] = []
    for model in config.models:
        runner = make_runner(model, config.runner, config.generation)
        records.extend(runner.generate(examples, run_id=run_id, seed=config.seed))

    write_jsonl(run_dir / "calls.jsonl", [record.model_dump(mode="json") for record in records])
    score_run(run_dir)
    return run_dir
