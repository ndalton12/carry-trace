import os
from pathlib import Path

import pytest
import torch

from carry_trace.config import (
    DatasetConfig,
    ExperimentConfig,
    GenerationParams,
    ModelSpec,
    RunnerConfig,
)
from carry_trace.datasets import generate_dataset
from carry_trace.io import read_jsonl
from carry_trace.models import HuggingFaceModelRunner
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
            splits={"smoke": {"examples_per_slice_per_length": 1}},
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
            answer_formats=["standard"],
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


@pytest.mark.skipif(
    os.environ.get("CARRY_TRACE_RUN_OLMO3_7B_INSTRUCT_TEST") != "1",
    reason=(
        "set CARRY_TRACE_RUN_OLMO3_7B_INSTRUCT_TEST=1 to run local "
        "allenai/Olmo-3-7B-Instruct generation"
    ),
)
def test_optional_olmo3_7b_instruct_hf_generation(tmp_path: Path) -> None:
    """Run one local OLMo 3 7B Instruct HF generation for chat-template debugging."""
    _, _, examples = generate_dataset(
        DatasetConfig(
            name="olmo3_7b_instruct_hf_debug",
            seed=20260627,
            output_dir=tmp_path / "data",
            write_parquet=False,
            splits={"debug": {"examples_per_slice_per_length": 1}},
            digit_lengths=[2],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
            digit_formats=["standard"],
            answer_formats=["standard"],
        )
    )
    runner = HuggingFaceModelRunner(
        ModelSpec(name="olmo3_7b_instruct", model_id="allenai/Olmo-3-7B-Instruct"),
        RunnerConfig(
            kind="hf",
            device=os.environ.get("CARRY_TRACE_OLMO3_DEVICE", "auto"),
            dtype=os.environ.get("CARRY_TRACE_OLMO3_DTYPE", "float16"),
            batch_size=1,
        ),
        GenerationParams(
            max_new_tokens=int(os.environ.get("CARRY_TRACE_OLMO3_MAX_NEW_TOKENS", "16")),
            temperature=float(os.environ.get("CARRY_TRACE_OLMO3_TEMPERATURE", "0.0")),
            top_p=float(os.environ.get("CARRY_TRACE_OLMO3_TOP_P", "1.0")),
            do_sample=os.environ.get("CARRY_TRACE_OLMO3_DO_SAMPLE", "0") == "1",
        ),
    )

    example = examples[0]
    rendered_prompt, inputs, input_ids_by_row = runner._tokenize_batch([example])
    generate_kwargs = runner._hf_generate_kwargs()
    print(f"rendered_prompt:\n{rendered_prompt[0]}", flush=True)
    print(f"input_ids_by_row_length: {len(input_ids_by_row[0])}", flush=True)
    print(f"hf_generate_kwargs: {generate_kwargs}", flush=True)

    print("raw_generate_start", flush=True)
    with torch.no_grad():
        raw_outputs = runner.model.generate(**inputs, **generate_kwargs)
    raw_generated_ids = runner._generated_rows(raw_outputs, int(inputs["input_ids"].shape[-1]))[0]
    raw_decoded = runner.tokenizer.decode(raw_generated_ids, skip_special_tokens=True)
    print("raw_generate_done", flush=True)
    print(f"raw_decoded_output:\n{raw_decoded}", flush=True)

    print("runner_generate_start", flush=True)
    record = next(runner.generate([example], run_id="olmo3_7b_instruct_hf_debug", seed=20260627))
    print("runner_generate_done", flush=True)

    print(f"generation_config: {record.generation_config}", flush=True)
    print(f"token_count_input: {record.token_count_input}", flush=True)
    print(f"token_count_output: {record.token_count_output}", flush=True)
    print(f"decoded_output:\n{record.decoded_output}", flush=True)
    assert record.decoded_output.strip()
    assert "<|im_start|>" not in record.decoded_output
    assert "<|im_end|>" not in record.decoded_output
    assert "\nuser" not in record.decoded_output.lower()
    assert "\nassistant" not in record.decoded_output.lower()
