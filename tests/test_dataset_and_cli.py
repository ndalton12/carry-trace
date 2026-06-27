from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from carry_trace.cli import app
from carry_trace.config import (
    DatasetConfig,
    ExperimentConfig,
    GenerationParams,
    ModelSpec,
    RunnerConfig,
)
from carry_trace.datasets import generate_dataset
from carry_trace.enums import (
    AnswerFormat,
    DigitFormat,
    PromptMode,
    QuantizationKind,
    RunnerKind,
    SliceName,
)
from carry_trace.io import read_json, read_jsonl, stable_hash, write_json, write_jsonl
from carry_trace.metrics import score_records, summarize_goal1
from carry_trace.models import FakeModelRunner
from carry_trace.runs import run_goal1


def test_generate_dataset_writes_reusable_schema(tmp_path: Path) -> None:
    config = DatasetConfig(
        name="tiny",
        seed=1,
        output_dir=tmp_path,
        write_parquet=False,
        splits={"smoke": {"examples_per_slice_per_length": 1}},
        digit_lengths=[2],
        slices=["no_carry", "long_carry_chain"],
        prompt_modes=["answer_only", "structured_column_cot"],
        digit_formats=["standard", "delimited"],
        answer_formats=["standard", "lsd"],
    )
    jsonl_path, manifest_path, rows = generate_dataset(config)
    assert jsonl_path.exists()
    assert manifest_path.exists()
    assert len(rows) == 8
    saved = read_jsonl(jsonl_path)
    assert saved[0]["schema_version"] == "goal1.v1"
    assert "incoming_carry" in saved[0]
    assert "messages" in saved[0]
    assert "problem_id" in saved[0]
    assert "split_seed" in saved[0]
    assert "match_group_id" not in saved[0]
    assert "partner_problem_ids" not in saved[0]
    delimited = [row for row in saved if row["digit_format"] == "delimited"]
    assert delimited
    assert "|" in delimited[0]["prompt"]
    assert "|" not in delimited[0]["a"]
    assert "|" not in delimited[0]["answer"]
    lsd = [row for row in saved if row["answer_format"] == "lsd"]
    assert lsd
    assert "|" not in lsd[0]["expected_output"]
    assert not [
        row
        for row in saved
        if row["digit_format"] == "delimited" and row["answer_format"] == "lsd"
    ]
    assert {row["slice_name"] for row in delimited} == {"random"}
    assert {row["slice_name"] for row in lsd} == {"random"}


def test_generate_dataset_uses_slice_count_overrides(tmp_path: Path) -> None:
    config = DatasetConfig(
        name="slice_counts",
        seed=1,
        output_dir=tmp_path,
        write_parquet=False,
        splits={
            "smoke": {
                "examples_per_slice_per_length": 2,
                "slice_examples_per_length": {"random": 5},
            }
        },
        digit_lengths=[2],
        slices=["no_carry", "random"],
        prompt_modes=["answer_only"],
        digit_formats=["standard"],
        answer_formats=["standard"],
    )
    _, _, rows = generate_dataset(config)
    counts = Counter(row.slice_name for row in rows)
    assert counts[SliceName.NO_CARRY] == 2
    assert counts[SliceName.RANDOM] == 5


def test_generate_dataset_balances_random_carry_counts(tmp_path: Path) -> None:
    config = DatasetConfig(
        name="balanced_random",
        seed=1,
        output_dir=tmp_path,
        write_parquet=False,
        splits={"smoke": {"slice_examples_per_length": {"random": 8}}},
        digit_lengths=[3],
        slices=["random"],
        prompt_modes=["answer_only"],
        digit_formats=["standard"],
        answer_formats=["standard"],
        random_sampling={"balance_carry_count": True},
    )
    _, _, rows = generate_dataset(config)
    counts = Counter(row.carry_count for row in rows)
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2}
    assert all(row.metadata["target_carry_count"] == row.carry_count for row in rows)


def test_goal1_fake_run_scores_outputs(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="tiny",
        seed=1,
        output_dir=tmp_path / "data",
        write_parquet=False,
        splits={"smoke": {"examples_per_slice_per_length": 1}},
        digit_lengths=[2],
        slices=["no_carry"],
        prompt_modes=["answer_only"],
        digit_formats=["standard"],
        answer_formats=["lsd"],
    )
    dataset_path, _, _ = generate_dataset(dataset_config)
    run_dir = run_goal1(
        ExperimentConfig(
            name="tiny",
            dataset_path=dataset_path,
            output_dir=tmp_path / "runs",
            digit_formats=["standard"],
            answer_formats=["lsd"],
            models=[ModelSpec(name="fake", model_id="allenai/Olmo-3-7B-Think")],
            runner=RunnerConfig(kind="fake"),
        )
    )
    assert (run_dir / "calls.jsonl").exists()
    assert (run_dir / "scored_calls.jsonl").exists()
    examples = read_jsonl(run_dir / "dataset.jsonl")
    assert {example["digit_format"] for example in examples} == {"standard"}
    assert {example["answer_format"] for example in examples} == {"lsd"}
    scored = read_jsonl(run_dir / "scored_calls.jsonl")
    assert scored[0]["parsed_answer_correct"] is True
    assert scored[0]["parsed_output_format_correct"] is True
    calls = read_jsonl(run_dir / "calls.jsonl")
    assert "git_commit" in calls[0]
    assert "package_versions" not in calls[0]


def test_goal1_run_filters_splits(tmp_path: Path) -> None:
    dataset_path, _, _ = generate_dataset(
        DatasetConfig(
            name="two_splits",
            seed=1,
            output_dir=tmp_path / "data",
            write_parquet=False,
            splits={
                "test_behavior": {"examples_per_slice_per_length": 1},
                "test_patch": {"examples_per_slice_per_length": 1},
            },
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
        )
    )
    run_dir = run_goal1(
        ExperimentConfig(
            name="split_filter",
            dataset_path=dataset_path,
            output_dir=tmp_path / "runs",
            splits=["test_behavior"],
            models=[ModelSpec(name="fake", model_id="allenai/Olmo-3-7B-Think")],
            runner=RunnerConfig(kind="fake"),
        )
    )
    examples = read_jsonl(run_dir / "dataset.jsonl")
    assert {example["split"] for example in examples} == {"test_behavior"}


def test_goal1_run_resumes_incomplete_calls(tmp_path: Path) -> None:
    dataset_path, _, examples = generate_dataset(
        DatasetConfig(
            name="resume_dataset",
            seed=1,
            output_dir=tmp_path / "data",
            write_parquet=False,
            splits={"smoke": {"examples_per_slice_per_length": 2}},
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
        )
    )
    model = ModelSpec(name="fake", model_id="dummy")
    config = ExperimentConfig(
        name="resume_run",
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        models=[model],
        runner=RunnerConfig(kind="fake"),
    )
    run_dir = tmp_path / "runs" / "resume_run-existing"
    run_dir.mkdir(parents=True)
    config_hash = stable_hash(config.model_dump(mode="json"))
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_dir.name,
            "created_at": "2026-01-01T00:00:00Z",
            "status": "running",
            "config_hash": config_hash,
            "config": config.model_dump(mode="json"),
            "dataset_path": str(dataset_path),
            "example_count": len(examples),
            "expected_record_count": len(examples),
        },
    )
    first_record = next(
        FakeModelRunner(model, GenerationParams()).generate(
            [examples[0]],
            run_id=run_dir.name,
            seed=config.seed,
        )
    )
    write_jsonl(run_dir / "calls.jsonl", [first_record.model_dump(mode="json")])

    resumed_run_dir = run_goal1(config)

    assert resumed_run_dir == run_dir
    calls = read_jsonl(run_dir / "calls.jsonl")
    assert len(calls) == 2
    assert {call["example_id"] for call in calls} == {example.id for example in examples}
    assert read_json(run_dir / "manifest.json")["status"] == "complete"
    assert (run_dir / "scored_calls.jsonl").exists()


def test_goal1_scoring_marks_token_limit_hits_invalid(tmp_path: Path) -> None:
    dataset_path, _, _ = generate_dataset(
        DatasetConfig(
            name="token_limit_dataset",
            seed=1,
            output_dir=tmp_path / "data",
            write_parquet=False,
            splits={"smoke": {"examples_per_slice_per_length": 2}},
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
        )
    )
    examples = read_jsonl(dataset_path)
    records = [
        {
            "example_id": examples[0]["id"],
            "model_name": "dummy",
            "decoded_output": "999",
            "token_count_output": 3,
            "latency_seconds": 1.0,
            "metadata": {"hit_token_limit": True},
        },
        {
            "example_id": examples[1]["id"],
            "model_name": "dummy",
            "decoded_output": examples[1]["answer"],
            "token_count_output": 1,
            "latency_seconds": 1.0,
            "metadata": {"hit_token_limit": False},
        },
    ]

    scored = score_records(examples, records)
    summary = summarize_goal1(scored, examples).iloc[0]

    assert [record["generation_valid"] for record in scored] == [False, True]
    assert [record["hit_token_limit"] for record in scored] == [True, False]
    assert summary["examples"] == 2
    assert summary["valid_examples"] == 1
    assert summary["token_limit_hits"] == 1
    assert summary["avg_output_tokens"] == 2
    assert summary["avg_output_tokens_valid"] == 1
    assert summary["parsed_accuracy_valid"] == 1


def test_cli_dataset_generate(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_tiny",
                "seed: 1",
                f"output_dir: {tmp_path / 'data'}",
                "write_parquet: false",
                "splits:",
                "  smoke:",
                "    examples_per_slice_per_length: 1",
                "digit_lengths: [2]",
                "slices: [no_carry]",
                "prompt_modes: [answer_only]",
                "digit_formats: [standard, delimited]",
                "answer_formats: [standard, lsd]",
            ]
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["dataset", "generate", "--config", str(config_path)])
    assert result.exit_code == 0
    rows = read_jsonl(tmp_path / "data" / "cli_tiny" / "examples.jsonl")
    assert len(rows) == 3
    assert {row["digit_format"] for row in rows} == {"standard", "delimited"}
    assert {row["answer_format"] for row in rows} == {"standard", "lsd"}
    assert not [
        row
        for row in rows
        if row["digit_format"] == "delimited" and row["answer_format"] == "lsd"
    ]


def test_config_closed_fields_are_enums() -> None:
    dataset_config = DatasetConfig(
        name="typed",
        slices=["no_carry"],
        prompt_modes=["answer_only"],
        digit_formats=["standard", "delimited"],
        answer_formats=["lsd"],
    )
    assert dataset_config.slices == [SliceName.NO_CARRY]
    assert dataset_config.prompt_modes == [PromptMode.ANSWER_ONLY]
    assert dataset_config.digit_formats == [DigitFormat.STANDARD, DigitFormat.DELIMITED]
    assert dataset_config.answer_formats == [AnswerFormat.LSD]

    legacy_config = DatasetConfig(name="legacy", digit_formats=["plain"])
    assert legacy_config.digit_formats == [DigitFormat.STANDARD]

    runner_config = RunnerConfig(kind="hf")
    assert runner_config.kind == RunnerKind.HF
    vllm_runner_config = RunnerConfig(kind="vllm", tensor_parallel_size=2)
    assert vllm_runner_config.kind == RunnerKind.VLLM
    quantized_runner_config = RunnerConfig(kind="hf", quantization="bitsandbytes_8bit")
    assert quantized_runner_config.quantization == QuantizationKind.BITSANDBYTES_8BIT


def test_config_rejects_unknown_enum_values() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(name="bad", prompt_modes=["scratchpad"])
    with pytest.raises(ValidationError):
        DatasetConfig(name="bad", slices=["all_the_carries"])
    with pytest.raises(ValidationError):
        DatasetConfig(name="bad", digit_formats=["spaces"])
    with pytest.raises(ValidationError):
        DatasetConfig(name="bad", answer_formats=["backwards"])
    with pytest.raises(ValidationError):
        RunnerConfig(kind="local")
    with pytest.raises(ValidationError):
        RunnerConfig(dtype="int8")
    with pytest.raises(ValidationError):
        RunnerConfig(quantization="fp8")
    with pytest.raises(ValidationError):
        RunnerConfig(tensor_parallel_size=0)
