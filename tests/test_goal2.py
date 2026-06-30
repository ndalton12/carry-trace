from pathlib import Path
from typing import Any

import pytest
import yaml

from carry_trace.config import load_goal2_config
from carry_trace.enums import ActivationLocation, AnswerFormat, DigitFormat, PromptMode, RunnerKind
from carry_trace.goal2 import (
    _completed_activation_keys,
    _find_resumable_goal2_run_dir,
    resolve_activation_locations,
)
from carry_trace.io import write_json, write_jsonl
from carry_trace.schemas import AdditionExample, ModelCallRecord


class CharTokenizer:
    """Character-level tokenizer for deterministic location-resolution tests."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        """Tokenize text as one character per token."""
        payload: dict[str, Any] = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode character token IDs back into text."""
        return "".join(chr(token_id) for token_id in token_ids)


def test_goal2_config_loads_activation_locations(tmp_path: Path) -> None:
    """Verify Goal 2 YAML configs load activation and upload settings."""
    config_path = tmp_path / "goal2.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "goal2_test",
                "dataset_path": "data/generated/test/examples.jsonl",
                "models": [{"name": "model", "model_id": "model-id"}],
                "runner": {"kind": "hf"},
                "activations": {
                    "locations": ["prompt_final", "answer_digits"],
                    "storage_dtype": "float16",
                },
                "upload": {"enabled": True, "repo_id": "user/repo"},
            }
        ),
        encoding="utf-8",
    )

    config = load_goal2_config(config_path)

    assert config.runner.kind == RunnerKind.HF
    assert config.activations.locations == [
        ActivationLocation.PROMPT_FINAL,
        ActivationLocation.ANSWER_DIGITS,
    ]
    assert config.upload.repo_id == "user/repo"


def test_goal2_config_requires_repo_id_when_upload_enabled(tmp_path: Path) -> None:
    """Verify upload.enabled requires upload.repo_id."""
    config_path = tmp_path / "goal2.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "goal2_test",
                "dataset_path": "data/generated/test/examples.jsonl",
                "upload": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="upload.repo_id"):
        load_goal2_config(config_path)


def test_resolve_activation_locations_uses_token_count_cot_thirds() -> None:
    """Verify symbolic Goal 2 locations resolve to prompt, CoT, and answer tokens."""
    tokenizer = CharTokenizer()
    example = _example()
    output = "Reasoning step. Final answer: 46"
    call = _call(example, output)

    locations = resolve_activation_locations(
        example=example,
        call=call,
        tokenizer=tokenizer,
        requested_locations=[
            ActivationLocation.OPERAND_DIGITS,
            ActivationLocation.QUESTION_TOKEN,
            ActivationLocation.PROMPT_FINAL,
            ActivationLocation.COT_START,
            ActivationLocation.COT_1_3,
            ActivationLocation.COT_2_3,
            ActivationLocation.COT_END,
            ActivationLocation.ANSWER_DIGITS,
        ],
    )

    by_name = {location["name"]: location for location in locations}
    assert "operand_a_digit_lsd_1" in by_name
    assert "operand_a_digit_lsd_0" in by_name
    assert "operand_b_digit_lsd_1" in by_name
    assert "operand_b_digit_lsd_0" in by_name
    assert by_name["question_token"]["token_text"] == "?"
    assert by_name["prompt_final"]["absolute_token_index"] == len(call.input_ids) - 1
    assert by_name["cot_start"]["token_text"] == "R"
    assert by_name["cot_start"]["absolute_token_index"] < by_name["cot_1_3"][
        "absolute_token_index"
    ]
    assert by_name["cot_1_3"]["absolute_token_index"] <= by_name["cot_2_3"][
        "absolute_token_index"
    ]
    assert by_name["cot_2_3"]["absolute_token_index"] <= by_name["cot_end"][
        "absolute_token_index"
    ]
    assert by_name["answer_digit_lsd_1"]["token_text"] == "4"
    assert by_name["answer_digit_lsd_0"]["token_text"] == "6"


def test_completed_activation_keys_require_tensor_file(tmp_path: Path) -> None:
    """Verify resume completion only counts activation rows with tensor files."""
    run_dir = tmp_path / "run"
    tensor_path = run_dir / "activations" / "model" / "ex-1.pt"
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"tensor")
    records = [
        {
            "model_name": "model",
            "model_id": "model-id",
            "model_revision": None,
            "example_id": "ex-1",
            "activation_path": "activations/model/ex-1.pt",
        },
        {
            "model_name": "model",
            "model_id": "model-id",
            "model_revision": None,
            "example_id": "ex-2",
            "activation_path": "activations/model/ex-2.pt",
        },
    ]

    completed = _completed_activation_keys(run_dir, records)

    assert completed == {("model", "model-id", None, "ex-1")}


def test_find_resumable_goal2_run_dir_finalizes_complete_unmarked_run(tmp_path: Path) -> None:
    """Verify a fully written but unmarked Goal 2 run is still resumable."""
    run_dir = tmp_path / "goal2_test-20260629"
    tensor_path = run_dir / "activations" / "model" / "ex-1.pt"
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"tensor")
    write_json(
        run_dir / "manifest.json",
        {
            "config_hash": "abc",
            "run_id": run_dir.name,
            "status": "running",
        },
    )
    write_jsonl(
        run_dir / "activations.jsonl",
        [
            {
                "model_name": "model",
                "model_id": "model-id",
                "model_revision": None,
                "example_id": "ex-1",
                "activation_path": "activations/model/ex-1.pt",
            }
        ],
    )

    resumable = _find_resumable_goal2_run_dir(
        output_dir=tmp_path,
        run_name="goal2_test",
        config_hash="abc",
        expected_record_count=1,
    )

    assert resumable == run_dir


def _example() -> AdditionExample:
    """Return a minimal valid addition example for Goal 2 tests."""
    return AdditionExample.model_validate(
        {
            "id": "ex-1",
            "problem_id": "problem-1",
            "schema_version": "goal1.v1",
            "split": "test",
            "seed": 1,
            "generator_config_hash": "hash",
            "slice_name": "no_carry",
            "base": 10,
            "n_digits": 2,
            "a": "12",
            "b": "34",
            "answer": "46",
            "prompt_mode": PromptMode.FREE_COT,
            "digit_format": DigitFormat.STANDARD,
            "digit_delimiter": "|",
            "answer_format": AnswerFormat.STANDARD,
            "expected_output": "46",
            "prompt_a": "12",
            "prompt_b": "34",
            "template_id": "test",
            "prompt": "What is 12 + 34? Solve the problem step by step.",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 12 + 34? Solve the problem step by step.",
                }
            ],
            "digits_a_lsd": [2, 1],
            "digits_b_lsd": [4, 3],
            "raw_sum": [6, 4],
            "incoming_carry": [0, 0],
            "outgoing_carry": [0, 0],
            "output_digits_lsd": [6, 4],
            "carry_count": 0,
            "max_carry_chain": 0,
            "carry_positions": [],
            "answer_length_change": False,
        }
    )


def _call(example: AdditionExample, output: str) -> ModelCallRecord:
    """Return a model call record with character token IDs."""
    return ModelCallRecord(
        run_id="run",
        example_id=example.id,
        model_name="model",
        model_id="model-id",
        runner_kind=RunnerKind.HF,
        seed=1,
        timestamp="2026-06-29T00:00:00+00:00",
        prompt=example.prompt,
        messages=example.messages,
        rendered_prompt=example.prompt,
        input_ids=[ord(char) for char in example.prompt],
        output_ids=[ord(char) for char in output],
        decoded_output=output,
        parsed_answer="46",
        generation_config={},
        token_count_input=len(example.prompt),
        token_count_output=len(output),
        latency_seconds=0.1,
    )
