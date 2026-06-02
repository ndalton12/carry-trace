from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from carry_trace.cli import app
from carry_trace.config import DatasetConfig, ExperimentConfig, ModelSpec, RunnerConfig
from carry_trace.datasets import generate_dataset
from carry_trace.enums import DigitFormat, PromptMode, RunnerKind, SliceName
from carry_trace.io import read_jsonl
from carry_trace.runs import run_goal1


def test_generate_dataset_writes_reusable_schema(tmp_path: Path) -> None:
    config = DatasetConfig(
        name="tiny",
        seed=1,
        output_dir=tmp_path,
        write_parquet=False,
        splits={"smoke": 0},
        digit_lengths=[2],
        slices=["no_carry", "long_carry_chain"],
        prompt_modes=["answer_only", "structured_column_cot"],
        digit_formats=["plain", "delimited"],
        examples_per_slice_per_length=1,
    )
    jsonl_path, manifest_path, rows = generate_dataset(config)
    assert jsonl_path.exists()
    assert manifest_path.exists()
    assert len(rows) == 8
    saved = read_jsonl(jsonl_path)
    assert saved[0]["schema_version"] == "goal1.v1"
    assert "incoming_carry" in saved[0]
    assert "messages" in saved[0]
    delimited = [row for row in saved if row["digit_format"] == "delimited"]
    assert delimited
    assert "|" in delimited[0]["prompt"]
    assert "|" not in delimited[0]["a"]
    assert "|" not in delimited[0]["answer"]


def test_goal1_fake_run_scores_outputs(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="tiny",
        seed=1,
        output_dir=tmp_path / "data",
        write_parquet=False,
        splits={"smoke": 0},
        digit_lengths=[2],
        slices=["no_carry"],
        prompt_modes=["answer_only"],
        digit_formats=["plain", "delimited"],
    )
    dataset_path, _, _ = generate_dataset(dataset_config)
    run_dir = run_goal1(
        ExperimentConfig(
            name="tiny",
            dataset_path=dataset_path,
            output_dir=tmp_path / "runs",
            digit_formats=["delimited"],
            models=[ModelSpec(name="fake", model_id="allenai/Olmo-3-7B-Think")],
            runner=RunnerConfig(kind="fake"),
        )
    )
    assert (run_dir / "calls.jsonl").exists()
    assert (run_dir / "scored_calls.jsonl").exists()
    examples = read_jsonl(run_dir / "dataset.jsonl")
    assert {example["digit_format"] for example in examples} == {"delimited"}
    scored = read_jsonl(run_dir / "scored_calls.jsonl")
    assert scored[0]["parsed_answer_correct"] is True


def test_cli_dataset_generate(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_tiny",
                "seed: 1",
                f"output_dir: {tmp_path / 'data'}",
                "write_parquet: false",
                "splits: {smoke: 0}",
                "digit_lengths: [2]",
                "slices: [no_carry]",
                "prompt_modes: [answer_only]",
                "digit_formats: [plain, delimited]",
                "examples_per_slice_per_length: 1",
            ]
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["dataset", "generate", "--config", str(config_path)])
    assert result.exit_code == 0
    rows = read_jsonl(tmp_path / "data" / "cli_tiny" / "examples.jsonl")
    assert len(rows) == 2
    assert {row["digit_format"] for row in rows} == {"plain", "delimited"}


def test_config_closed_fields_are_enums() -> None:
    dataset_config = DatasetConfig(
        name="typed",
        slices=["no_carry"],
        prompt_modes=["answer_only"],
        digit_formats=["delimited"],
    )
    assert dataset_config.slices == [SliceName.NO_CARRY]
    assert dataset_config.prompt_modes == [PromptMode.ANSWER_ONLY]
    assert dataset_config.digit_formats == [DigitFormat.DELIMITED]

    runner_config = RunnerConfig(kind="hf")
    assert runner_config.kind == RunnerKind.HF


def test_config_rejects_unknown_enum_values() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(name="bad", prompt_modes=["scratchpad"])
    with pytest.raises(ValidationError):
        DatasetConfig(name="bad", slices=["all_the_carries"])
    with pytest.raises(ValidationError):
        DatasetConfig(name="bad", digit_formats=["spaces"])
    with pytest.raises(ValidationError):
        RunnerConfig(kind="local")
    with pytest.raises(ValidationError):
        RunnerConfig(dtype="int8")
