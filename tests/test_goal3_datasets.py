from pathlib import Path

import pytest
from typer.testing import CliRunner

from carry_trace.cli import app
from carry_trace.config import Goal3DatasetConfig
from carry_trace.goal3_datasets import _align_output_token_index, generate_goal3_dataset_bundle
from carry_trace.io import read_json, read_jsonl, write_json, write_jsonl


class CharacterTokenizer:
    """Provide deterministic character-level tokenization for dataset tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode each character as its code point."""
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        """Decode character code points back into text."""
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


def test_generate_goal3_dataset_bundle_uses_goal2_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Goal 3 bundles materialize replay and residual cases from Goal 2."""
    source_run, source_dataset = _write_source_goal2_run(tmp_path)
    monkeypatch.setattr(
        "carry_trace.goal3_datasets._load_tokenizer",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    config = _goal3_config(tmp_path, source_run, source_dataset)

    manifest_path, counts = generate_goal3_dataset_bundle(config)

    assert counts == {
        "source_activation_records": 2,
        "shared_examples": 1,
        "replay_prefixes": 5,
        "replay_cases": 10,
        "residual_intervention_cases": 12,
    }
    output_dir = manifest_path.parent
    prefixes = read_jsonl(output_dir / "replay_prefixes.jsonl")
    cases = read_jsonl(output_dir / "replay_cases.jsonl")
    residual = read_jsonl(output_dir / "residual_intervention_cases.jsonl")

    cot_two_thirds = [row for row in prefixes if row["location_kind"] == "cot_2_3"]
    assert {row["assistant_prefix"] for row in cot_two_thirds} == {"ABCD"}
    assert {tuple(row["assistant_prefix_token_ids"]) for row in cot_two_thirds} == {
        tuple(map(ord, "ABCD"))
    }
    assert {row["prefix_token_source"] for row in cot_two_thirds} == {"reconstructed"}
    assert {row["prefix_alignment_delta"] for row in cot_two_thirds} == {0}
    assert {row["answer_only_example_id"] for row in prefixes} == {"answer-only-example"}
    assert {row["replay_kind"] for row in cases} == {"no_reasoning", "self", "crossed"}
    assert {
        (row["source_model_name"], row["receiver_model_name"])
        for row in cases
        if row["source_model_name"] is not None
    } == {
        ("olmo3-instruct-sft", "olmo3-instruct-sft"),
        ("olmo3-instruct-sft", "olmo3-instruct-full"),
        ("olmo3-instruct-full", "olmo3-instruct-sft"),
        ("olmo3-instruct-full", "olmo3-instruct-full"),
    }
    assert {row["target"] for row in residual} == {"incoming_carry", "outgoing_carry"}
    assert {
        (row["target"], row["target_column_lsd"], row["affected_output_column_lsd"])
        for row in residual
    } == {
        ("incoming_carry", 1, 1),
        ("outgoing_carry", 0, 1),
    }
    assert {row["factual_carry"] for row in residual} == {1}
    assert {row["counterfactual_carry"] for row in residual} == {0}
    assert {row["factual_output_digit"] for row in residual} == {8}
    assert {row["counterfactual_output_digit"] for row in residual} == {7}
    assert {row["counterfactual_answer"] for row in residual} == {"73"}
    assert {
        row["metadata"]["equivalent_incoming_carry_column_lsd"] for row in residual
    } == {1}
    assert {row["activation_location_index"] for row in residual} == {0, 2, 3}
    assert read_json(manifest_path)["artifact_kind"] == "goal3_natural_cot_dataset_bundle"


def test_goal3_dataset_cli_writes_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the Goal 3 dataset command loads config and writes all artifacts."""
    source_run, source_dataset = _write_source_goal2_run(tmp_path)
    monkeypatch.setattr(
        "carry_trace.goal3_datasets._load_tokenizer",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    config_path = tmp_path / "goal3.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_goal3",
                f"source_goal2_run_dir: {source_run}",
                f"source_dataset_path: {source_dataset}",
                f"output_dir: {tmp_path / 'generated'}",
                "splits: [test_probe]",
                "models: [olmo3-instruct-sft, olmo3-instruct-full]",
                "digit_lengths: [2]",
                "replay:",
                "  locations: [prompt_final, cot_2_3, cot_end]",
                "  crossed_models: true",
                "residual:",
                "  locations: [prompt_final, cot_2_3, cot_end]",
                "  layers: [8]",
                "  target_columns_by_digit_length:",
                "    2: [1]",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["dataset", "goal3", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "5 replay prefixes" in result.output
    output_dir = tmp_path / "generated" / "cli_goal3"
    assert (output_dir / "replay_prefixes.jsonl").exists()
    assert (output_dir / "replay_cases.jsonl").exists()
    assert (output_dir / "residual_intervention_cases.jsonl").exists()
    assert (output_dir / "manifest.json").exists()


def test_goal3_config_rejects_non_carry_residual_targets(tmp_path: Path) -> None:
    """Verify Goal 3 residual dataset configs stay scoped to carry targets."""
    with pytest.raises(ValueError, match="incoming_carry or outgoing_carry"):
        Goal3DatasetConfig(
            name="bad",
            source_goal2_run_dir=tmp_path,
            residual={"targets": ["output_digit"]},
        )


def test_align_output_token_index_allows_legacy_retokenization_shift() -> None:
    """Verify a legacy prefix aligns to the nearest saved Goal 2 token anchor."""
    output_ids = list(map(ord, "ABCDE"))

    aligned = _align_output_token_index(
        output_ids=output_ids,
        recorded_output_index=4,
        expected_token_id=ord("D"),
        max_shift=2,
        context="example/model/cot_2_3",
    )

    assert aligned == 3


def _goal3_config(
    tmp_path: Path,
    source_run: Path,
    source_dataset: Path,
) -> Goal3DatasetConfig:
    """Return a compact Goal 3 dataset config for tests."""
    return Goal3DatasetConfig(
        name="goal3_test",
        source_goal2_run_dir=source_run,
        source_dataset_path=source_dataset,
        output_dir=tmp_path / "generated",
        splits=["test_probe"],
        models=["olmo3-instruct-sft", "olmo3-instruct-full"],
        digit_lengths=[2],
        replay={
            "locations": ["prompt_final", "cot_2_3", "cot_end"],
            "crossed_models": True,
        },
        residual={
            "targets": ["incoming_carry", "outgoing_carry"],
            "locations": ["prompt_final", "cot_2_3", "cot_end"],
            "layers": [8],
            "target_columns_by_target": {
                "incoming_carry": {2: [1]},
                "outgoing_carry": {2: [0]},
            },
        },
    )


def _write_source_goal2_run(tmp_path: Path) -> tuple[Path, Path]:
    """Write a tiny paired-model Goal 2 activation run and source dataset."""
    source_dataset = tmp_path / "source" / "examples.jsonl"
    write_jsonl(
        source_dataset,
        [
            _source_example("answer-only-example", "answer_only"),
            _source_example("free-cot-example", "free_cot"),
        ],
    )
    source_run = tmp_path / "goal2-run"
    write_json(
        source_run / "manifest.json",
        {
            "run_id": "goal2-test-run",
            "dataset_path": str(source_dataset),
        },
    )
    records = []
    for model_name in ["olmo3-instruct-sft", "olmo3-instruct-full"]:
        activation_path = source_run / "activations" / model_name / "free-cot-example.pt"
        activation_path.parent.mkdir(parents=True, exist_ok=True)
        activation_path.touch()
        records.append(_activation_record(model_name, activation_path.relative_to(source_run)))
    write_jsonl(source_run / "activations.jsonl", records)
    return source_run, source_dataset


def _source_example(example_id: str, prompt_mode: str) -> dict[str, object]:
    """Return one source example with a local incoming-carry counterfactual."""
    instruction = "Give only the answer." if prompt_mode == "answer_only" else "Solve step by step."
    return {
        "id": example_id,
        "problem_id": "problem-19-plus-64",
        "split": "test_probe",
        "n_digits": 2,
        "prompt_mode": prompt_mode,
        "digit_format": "standard",
        "answer_format": "standard",
        "expected_output": "83",
        "prompt": f"What is 19 + 64? {instruction}",
        "messages": [{"role": "user", "content": f"What is 19 + 64? {instruction}"}],
        "base": 10,
        "answer": "83",
        "raw_sum": [13, 7],
        "incoming_carry": [0, 1],
        "outgoing_carry": [1, 0],
        "output_digits_lsd": [3, 8],
    }


def _activation_record(model_name: str, activation_path: Path) -> dict[str, object]:
    """Return one source activation record with exact natural-CoT locations."""
    decoded_output = "ABCDE83"
    locations = [
        _location("prompt_final", "prompt_final", None, "\n", 10),
        _location("cot_1_3", "cot_1_3", 1, "B", 11),
        _location("cot_2_3", "cot_2_3", 3, "D", 13),
        _location("cot_end", "cot_end", 4, "E", 14),
    ]
    return {
        "example_id": "free-cot-example",
        "model_name": model_name,
        "model_id": f"allenai/{model_name}",
        "tokenizer_id": "test-tokenizer",
        "tokenizer_revision": None,
        "decoded_output": decoded_output,
        "parsed_answer": "83",
        "call_metadata": {"hit_token_limit": False},
        "activation_path": activation_path.as_posix(),
        "layer_indices": [8, 16],
        "locations": locations,
    }


def _location(
    kind: str,
    name: str,
    output_index: int | None,
    token_text: str,
    absolute_index: int,
) -> dict[str, object]:
    """Return one saved Goal 2 activation location."""
    metadata = {} if output_index is None else {"output_token_index": output_index}
    return {
        "absolute_token_index": absolute_index,
        "kind": kind,
        "metadata": metadata,
        "name": name,
        "source": "prompt" if output_index is None else "generated",
        "token_id": ord(token_text) if len(token_text) == 1 else 0,
        "token_text": token_text,
    }
