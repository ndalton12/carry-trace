"""Run the OLMo 3.1 32B subset-planning benchmark in a standalone process."""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from carry_trace.config import DatasetConfig, GenerationParams, ModelSpec, RunnerConfig
from carry_trace.datasets import dump_dataset_row, generate_dataset
from carry_trace.io import ensure_dir, read_jsonl, utc_now_iso, write_json, write_jsonl
from carry_trace.metrics import score_records, summarize_goal1
from carry_trace.models import make_runner

DEFAULT_NAME = "colab_olmo31_32b_think_instruct_subset_planner"
DEFAULT_SEED = 20260611
DEFAULT_INSTRUCT_MODEL_ID = "allenai/Olmo-3.1-32B-Instruct"
DEFAULT_THINK_MODEL_ID = "allenai/Olmo-3.1-32B-Think"


def main() -> None:
    """Parse CLI arguments and run the requested subset-planning command."""
    args = parse_args()
    run_root = ensure_dir(
        args.run_root
        or Path("runs")
        / "subset_planner"
        / f"{args.name}-{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )
    if args.checkpoint == "summary":
        summarize_outputs(args, run_root)
        return
    example_rows, examples = prepare_dataset(args, run_root)
    run_checkpoint(args, run_root, example_rows, examples)


def parse_args() -> argparse.Namespace:
    """Return parsed command-line arguments for the benchmark script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["instruct", "think", "summary"], required=True)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--examples-per-slice-per-length", type=int, default=1)
    parser.add_argument("--digit-lengths", default="4,8,12")
    parser.add_argument("--slices", default="no_carry,isolated_carry,long_carry_chain")
    parser.add_argument("--prompt-modes", default="answer_only,free_cot")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--compile-vllm", action="store_false", dest="enforce_eager")
    parser.add_argument("--target-hours-per-checkpoint", type=float, default=24.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--instruct-model-id", default=DEFAULT_INSTRUCT_MODEL_ID)
    parser.add_argument("--think-model-id", default=DEFAULT_THINK_MODEL_ID)
    parser.add_argument("--instruct-max-new-tokens", type=int, default=2048)
    parser.add_argument("--think-thinking-tokens", type=int, default=4096)
    parser.add_argument("--thinking-final-answer-tokens", type=int, default=100)
    return parser.parse_args()


def prepare_dataset(
    args: argparse.Namespace,
    run_root: Path,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Generate the deterministic candidate pool and save it under the run root."""
    dataset_config = DatasetConfig(
        name=args.name,
        seed=args.seed,
        output_dir=args.output_dir,
        write_parquet=False,
        splits={
            "subset_probe": {
                "examples_per_slice_per_length": args.examples_per_slice_per_length
            }
        },
        digit_lengths=parse_int_list(args.digit_lengths),
        slices=parse_str_list(args.slices),
        prompt_modes=parse_str_list(args.prompt_modes),
        digit_formats=["standard"],
        answer_formats=["standard"],
    )
    jsonl_path, manifest_path, examples = generate_dataset(dataset_config)
    example_rows = [dump_dataset_row(example) for example in examples]
    write_jsonl(run_root / "dataset.jsonl", example_rows)
    write_json(
        run_root / "subset_planner_config.json",
        {
            "dataset_path": str(jsonl_path),
            "dataset_manifest_path": str(manifest_path),
            "script_args": vars(args),
            "example_count": len(examples),
        },
    )
    print(f"Dataset: {jsonl_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Examples: {len(examples)}", flush=True)
    return example_rows, examples


def run_checkpoint(
    args: argparse.Namespace,
    run_root: Path,
    example_rows: list[dict[str, Any]],
    examples: list[Any],
) -> None:
    """Run one checkpoint, score outputs, and write benchmark artifacts."""
    model_name, model_id, max_new_tokens = checkpoint_settings(args)
    print(f"\n=== Loading {model_name}: {model_id} ===", flush=True)
    runner = make_runner(
        ModelSpec(name=model_name, model_id=model_id),
        runner_config(args),
        generation_params(args, max_new_tokens),
    )
    records = []
    started = time.perf_counter()
    try:
        for index, record in enumerate(
            runner.generate(examples, run_id=args.name, seed=args.seed),
            start=1,
        ):
            payload = record.model_dump(mode="json")
            records.append(payload)
            elapsed = time.perf_counter() - started
            avg = elapsed / index
            remaining = max(len(examples) - index, 0) * avg
            forced = sum(
                1 for row in records if row.get("metadata", {}).get("thinking_force_closed")
            )
            print(
                f"[{model_name}] {index}/{len(examples)} complete | "
                f"elapsed {elapsed / 60:.1f} min | avg {avg:.1f} sec/ex | "
                f"ETA {remaining / 60:.1f} min | forced_close {forced}/{index}",
                flush=True,
            )
    finally:
        cleanup_runner(runner)

    total_seconds = time.perf_counter() - started
    scored = score_records(example_rows, records)
    detail_rows = record_detail_rows(model_name, scored, example_rows)
    forced_rows = [row for row in detail_rows if row["forced_close"]]
    write_jsonl(run_root / f"{model_name}_calls.jsonl", records)
    write_jsonl(run_root / f"{model_name}_scored_calls.jsonl", scored)
    pd.DataFrame(scored).to_csv(run_root / f"{model_name}_scored_calls.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(run_root / f"{model_name}_example_details.csv", index=False)
    pd.DataFrame(forced_rows).to_csv(
        run_root / f"{model_name}_forced_close_examples.csv",
        index=False,
    )
    summarize_goal1(scored, example_rows).to_csv(
        run_root / f"{model_name}_metrics_summary.csv",
        index=False,
    )
    runtime = {
        "model_name": model_name,
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "total_seconds": total_seconds,
        "seconds_per_example": total_seconds / len(examples),
        "examples_per_hour": len(examples) * 3600 / total_seconds,
        "examples_per_24h": len(examples) * args.target_hours_per_checkpoint * 3600
        / total_seconds,
    }
    write_json(run_root / f"{model_name}_runtime.json", runtime)
    print(f"Wrote {model_name} artifacts to {run_root}", flush=True)
    if forced_rows:
        forced_preview = pd.DataFrame(forced_rows)[
            [
                "n_digits",
                "slice_name",
                "prompt_mode",
                "a",
                "b",
                "answer",
                "token_count_output",
                "parsed_answer_correct",
            ]
        ]
        print("\nForced-close examples:", flush=True)
        print(forced_preview.to_string(index=False), flush=True)


def summarize_outputs(args: argparse.Namespace, run_root: Path) -> None:
    """Summarize completed checkpoint artifacts and estimate viable subset sizes."""
    example_rows = read_jsonl(run_root / "dataset.jsonl")
    num_cells = (
        len(parse_int_list(args.digit_lengths))
        * len(parse_str_list(args.slices))
        * len(parse_str_list(args.prompt_modes))
    )
    summary_rows = []
    cell_summaries = []
    detail_tables = []
    for model_name in ("olmo31_32b_instruct", "olmo31_32b_think"):
        scored_path = run_root / f"{model_name}_scored_calls.jsonl"
        runtime_path = run_root / f"{model_name}_runtime.json"
        if not scored_path.exists() or not runtime_path.exists():
            continue
        scored = read_jsonl(scored_path)
        runtime = read_json(runtime_path)
        summary_rows.append(summarize_model(model_name, scored, runtime, num_cells))
        cell_summaries.append(summarize_cells(model_name, scored, example_rows))
        detail_tables.append(pd.DataFrame(record_detail_rows(model_name, scored, example_rows)))

    if not summary_rows:
        raise FileNotFoundError(f"no completed scored-call artifacts found in {run_root}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_root / "runtime_accuracy_termination_summary.csv", index=False)
    print("\nRuntime / accuracy / termination summary:", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    cell_summary_df = pd.concat(cell_summaries, ignore_index=True)
    cell_summary_df.to_csv(run_root / "cell_level_summary.csv", index=False)
    print("\nCell-level summary:", flush=True)
    print(cell_summary_df.to_string(index=False), flush=True)

    detail_df = pd.concat(detail_tables, ignore_index=True)
    detail_df.to_csv(run_root / "example_details.csv", index=False)
    forced_df = detail_df[detail_df["forced_close"]].copy()
    forced_df.to_csv(run_root / "forced_close_examples.csv", index=False)
    print("\nForced-close examples:", flush=True)
    if forced_df.empty:
        print("None", flush=True)
    else:
        forced_cols = [
            "model_name",
            "n_digits",
            "slice_name",
            "prompt_mode",
            "a",
            "b",
            "answer",
            "token_count_output",
            "parsed_answer_correct",
        ]
        print(forced_df[forced_cols].to_string(index=False), flush=True)

    paired_examples_per_cell = int(summary_df["examples_per_cell_24h"].min())
    paired_total = paired_examples_per_cell * num_cells
    recommendation = {
        "cells": num_cells,
        "paired_examples_per_cell": paired_examples_per_cell,
        "paired_total_examples_per_checkpoint": paired_total,
        "basis": "minimum measured 24h capacity across completed checkpoints",
    }
    write_json(run_root / "subset_recommendation.json", recommendation)
    print("\nRecommendation:", flush=True)
    print(recommendation, flush=True)


def summarize_model(
    model_name: str,
    scored: list[dict[str, Any]],
    runtime: dict[str, Any],
    num_cells: int,
) -> dict[str, Any]:
    """Return aggregate runtime, accuracy, and termination metrics for one model."""
    df = pd.DataFrame(scored)
    forced = df["metadata"].apply(lambda value: bool((value or {}).get("thinking_force_closed")))
    return {
        "model_name": model_name,
        "examples": len(df),
        "total_minutes": runtime["total_seconds"] / 60,
        "seconds_per_example": runtime["seconds_per_example"],
        "examples_per_hour": runtime["examples_per_hour"],
        "examples_per_24h": runtime["examples_per_24h"],
        "examples_per_cell_24h": math.floor(runtime["examples_per_24h"] / num_cells),
        "parsed_accuracy": df["parsed_answer_correct"].mean(),
        "output_format_accuracy": df["parsed_output_format_correct"].mean(),
        "forced_close_rate": forced.mean(),
        "avg_output_tokens": df["token_count_output"].mean(),
        "p50_output_tokens": df["token_count_output"].quantile(0.5),
        "p90_output_tokens": df["token_count_output"].quantile(0.9),
    }


def record_detail_rows(
    model_name: str,
    scored: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-example rows with metadata useful for diagnosing Think termination."""
    examples_by_id = {row["id"]: row for row in example_rows}
    details = []
    for row in scored:
        example = examples_by_id[row["example_id"]]
        metadata = row.get("metadata") or {}
        decoded = row.get("decoded_output") or ""
        details.append(
            {
                "model_name": model_name,
                "example_id": row["example_id"],
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
                "carry_count": example.get("carry_count"),
                "max_carry_chain": example.get("max_carry_chain"),
                "first_carry_position": example.get("first_carry_position"),
                "parsed_answer": row.get("parsed_answer"),
                "parsed_output": row.get("parsed_output"),
                "parsed_answer_correct": row.get("parsed_answer_correct"),
                "parsed_output_format_correct": row.get("parsed_output_format_correct"),
                "token_count_input": row.get("token_count_input"),
                "token_count_output": row.get("token_count_output"),
                "latency_seconds": row.get("latency_seconds"),
                "forced_close": bool(metadata.get("thinking_force_closed")),
                "thinking_first_pass_token_count": metadata.get(
                    "thinking_first_pass_token_count"
                ),
                "thinking_final_answer_token_budget": metadata.get(
                    "thinking_final_answer_token_budget"
                ),
                "thinking_stop_expected_output_digits": metadata.get(
                    "thinking_stop_expected_output_digits"
                ),
                "prompt": row.get("prompt"),
                "decoded_output_prefix": decoded[:800],
                "decoded_output_suffix": decoded[-800:],
            }
        )
    return details


def summarize_cells(
    model_name: str,
    scored: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return cell-level accuracy and runtime metrics for one model."""
    scored_df = pd.DataFrame(scored)
    example_df = pd.DataFrame(example_rows)
    scored_df["forced_close"] = scored_df["metadata"].apply(
        lambda value: bool((value or {}).get("thinking_force_closed"))
    )
    merged = scored_df.merge(
        example_df[["id", "n_digits", "slice_name", "prompt_mode", "a", "b", "answer"]],
        left_on="example_id",
        right_on="id",
        how="left",
    )
    grouped = (
        merged.groupby(["n_digits", "slice_name", "prompt_mode"], dropna=False)
        .agg(
            examples=("example_id", "count"),
            parsed_accuracy=("parsed_answer_correct", "mean"),
            output_format_accuracy=("parsed_output_format_correct", "mean"),
            forced_close_rate=("forced_close", "mean"),
            avg_output_tokens=("token_count_output", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "model_name", model_name)
    return grouped


def checkpoint_settings(args: argparse.Namespace) -> tuple[str, str, int]:
    """Return model name, model ID, and max-new-token budget for a checkpoint."""
    if args.checkpoint == "instruct":
        return "olmo31_32b_instruct", args.instruct_model_id, args.instruct_max_new_tokens
    if args.checkpoint == "think":
        return (
            "olmo31_32b_think",
            args.think_model_id,
            args.think_thinking_tokens + args.thinking_final_answer_tokens,
        )
    raise ValueError(f"unknown checkpoint {args.checkpoint!r}")


def generation_params(args: argparse.Namespace, max_new_tokens: int) -> GenerationParams:
    """Return generation parameters shared by the benchmark checkpoints."""
    return GenerationParams(
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        force_close_thinking=True,
        thinking_final_answer_tokens=args.thinking_final_answer_tokens,
    )


def runner_config(args: argparse.Namespace) -> RunnerConfig:
    """Return vLLM runner configuration for the benchmark process."""
    payload: dict[str, Any] = {
        "kind": "vllm",
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "trust_remote_code": False,
        "quantization": "none",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len is not None:
        payload["max_model_len"] = args.max_model_len
    print(f"Runner config: {payload}", flush=True)
    return RunnerConfig(**payload)


def cleanup_runner(runner: Any) -> None:
    """Best-effort cleanup for a vLLM runner before process exit."""
    try:
        del runner.llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception as exc:
        print(f"vLLM cleanup note: {type(exc).__name__}: {exc}", flush=True)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    """Parse a comma-separated string list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def read_json(path: Path) -> Any:
    """Read a JSON file from disk."""
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
