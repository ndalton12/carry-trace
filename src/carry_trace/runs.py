"""Experiment orchestration."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

from carry_trace.config import ExperimentConfig
from carry_trace.datasets import dump_dataset_row
from carry_trace.io import (
    append_jsonl,
    ensure_dir,
    read_json,
    read_jsonl,
    stable_hash,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from carry_trace.metrics import score_run
from carry_trace.models import make_runner
from carry_trace.schemas import AdditionExample


def run_goal1(config: ExperimentConfig) -> Path:
    """Run a Goal 1 experiment and write all run artifacts."""
    config_hash = stable_hash(config.model_dump(mode="json"))
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

    expected_record_count = len(examples) * len(config.models)
    run_dir = _find_resumable_run_dir(
        output_dir=config.output_dir,
        run_name=config.name,
        config_hash=config_hash,
        expected_record_count=expected_record_count,
    )
    if run_dir is None:
        run_id = f"{config.name}-{utc_now_iso().replace(':', '').replace('+', 'Z')}"
        run_dir = ensure_dir(config.output_dir / run_id)
        created_at = utc_now_iso()
    else:
        run_id = _run_id_from_manifest(run_dir) or run_dir.name
        created_at = _created_at_from_manifest(run_dir) or utc_now_iso()

    write_jsonl(
        run_dir / "dataset.jsonl",
        [dump_dataset_row(example) for example in examples],
    )
    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": utc_now_iso(),
        "status": "running",
        "config_hash": config_hash,
        "config": config.model_dump(mode="json"),
        "dataset_path": str(config.dataset_path),
        "example_count": len(examples),
        "expected_record_count": expected_record_count,
    }
    write_json(run_dir / "manifest.json", manifest)

    calls_path = run_dir / "calls.jsonl"
    existing_records = _load_repairable_jsonl(calls_path)
    completed = {_call_key(record) for record in existing_records}
    for model in config.models:
        model_examples = [
            example for example in examples if _model_example_key(model, example) not in completed
        ]
        completed_for_model = len(examples) - len(model_examples)
        print(
            f"[goal1] model={model.name} pending={len(model_examples)} "
            f"completed={completed_for_model}/{len(examples)}",
            flush=True,
        )
        if not model_examples:
            continue
        runner = make_runner(model, config.runner, config.generation)
        started = time.perf_counter()
        try:
            for record in runner.generate(model_examples, run_id=run_id, seed=config.seed):
                payload = record.model_dump(mode="json")
                append_jsonl(calls_path, [payload])
                completed.add(_call_key(payload))
                completed_for_model += 1
                elapsed = time.perf_counter() - started
                print(
                    f"[goal1] model={model.name} example={completed_for_model}/{len(examples)} "
                    f"example_id={record.example_id} output_tokens={record.token_count_output} "
                    f"latency={record.latency_seconds:.2f}s elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
        finally:
            _cleanup_runner(runner)

    score_run(run_dir)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now_iso()
    manifest["updated_at"] = manifest["completed_at"]
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _find_resumable_run_dir(
    output_dir: Path,
    run_name: str,
    config_hash: str,
    expected_record_count: int,
) -> Path | None:
    """Return the newest incomplete run directory for the same config hash."""
    candidates = sorted(
        output_dir.glob(f"{run_name}-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("config_hash") != config_hash or manifest.get("completed_at"):
            continue
        calls_count = len(_load_repairable_jsonl(run_dir / "calls.jsonl"))
        if calls_count < expected_record_count or not (run_dir / "scored_calls.jsonl").exists():
            return run_dir
    return None


def _load_repairable_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load valid JSONL rows and truncate a malformed tail after interrupted writes."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    had_malformed_tail = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                had_malformed_tail = True
                break
    if had_malformed_tail:
        write_jsonl(path, rows)
    return rows


def _call_key(record: dict[str, Any]) -> tuple[object, object, object, object]:
    """Return the resume key for a saved model-call record."""
    return (
        record.get("model_name"),
        record.get("model_id"),
        record.get("model_revision"),
        record.get("example_id"),
    )


def _model_example_key(
    model: object,
    example: AdditionExample,
) -> tuple[object, object, object, str]:
    """Return the resume key expected for one model and example pair."""
    return (
        getattr(model, "name", None),
        getattr(model, "model_id", None),
        getattr(model, "revision", None),
        example.id,
    )


def _run_id_from_manifest(run_dir: Path) -> str | None:
    """Return a run ID from a manifest if one is available."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    value = read_json(manifest_path).get("run_id")
    return value if isinstance(value, str) else None


def _created_at_from_manifest(run_dir: Path) -> str | None:
    """Return a created timestamp from a manifest if one is available."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    value = read_json(manifest_path).get("created_at")
    return value if isinstance(value, str) else None


def _cleanup_runner(runner: object) -> None:
    """Release model runner references and clear CUDA caches when available."""
    for attr in ("model", "llm"):
        if hasattr(runner, attr):
            delattr(runner, attr)
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
