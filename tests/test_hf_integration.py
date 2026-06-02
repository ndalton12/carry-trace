import os
from pathlib import Path

import pytest

from carry_trace.config import DatasetConfig, ExperimentConfig, ModelSpec, RunnerConfig
from carry_trace.datasets import generate_dataset
from carry_trace.io import read_jsonl
from carry_trace.runs import run_goal1


@pytest.mark.skipif(
    os.environ.get("CARRY_TRACE_RUN_MODEL_TESTS") != "1",
    reason="set CARRY_TRACE_RUN_MODEL_TESTS=1 to run Hugging Face generation tests",
)
def test_optional_hugging_face_generation(tmp_path: Path) -> None:
    dataset_path, _, _ = generate_dataset(
        DatasetConfig(
            name="hf_tiny",
            seed=1,
            output_dir=tmp_path / "data",
            write_parquet=False,
            splits={"smoke": 0},
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
        )
    )
    run_dir = run_goal1(
        ExperimentConfig(
            name="hf_tiny",
            dataset_path=dataset_path,
            output_dir=tmp_path / "runs",
            max_examples=1,
            models=[ModelSpec(name="tiny-gpt2", model_id="sshleifer/tiny-gpt2")],
            runner=RunnerConfig(kind="hf", device="cpu"),
        )
    )
    calls = read_jsonl(run_dir / "calls.jsonl")
    assert calls
    assert calls[0]["runner_kind"] == "hf"
