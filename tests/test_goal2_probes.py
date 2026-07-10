from pathlib import Path

import pandas as pd
import torch
import yaml

from carry_trace.config import load_goal2_probe_config
from carry_trace.enums import ProbeTarget
from carry_trace.goal2_probes import _ambiguous_digit_location_indexes, run_goal2_probes
from carry_trace.io import read_jsonl, write_jsonl


def test_goal2_probe_config_loads_targets(tmp_path: Path) -> None:
    """Verify Goal 2 probe configs load target and split settings."""
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "probe",
                "goal2_run_dir": "runs/goal2",
                "train_split": "train_probe",
                "test_split": "test_probe",
                "targets": ["any_carry", "outgoing_carry"],
            }
        ),
        encoding="utf-8",
    )

    config = load_goal2_probe_config(config_path)

    assert config.targets == [ProbeTarget.ANY_CARRY, ProbeTarget.OUTGOING_CARRY]
    assert config.require_unambiguous_digit_tokens is False
    assert config.n_jobs == 1


def test_run_goal2_probes_filters_token_limits_and_ambiguous_digit_tokens(
    tmp_path: Path,
) -> None:
    """Verify probe training excludes capped generations and keeps column metadata."""
    goal2_run_dir = _write_probe_activation_run(tmp_path)
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "probe",
                "goal2_run_dir": str(goal2_run_dir),
                "output_dir": str(tmp_path / "probes"),
                "train_split": "train_probe",
                "test_split": "test_probe",
                "targets": ["any_carry", "incoming_carry", "outgoing_carry"],
                "min_train_examples": 2,
                "min_test_examples": 2,
                "max_iter": 200,
            }
        ),
        encoding="utf-8",
    )

    probe_dir = run_goal2_probes(load_goal2_probe_config(config_path))
    metrics = pd.DataFrame(read_jsonl(probe_dir / "probe_metrics.jsonl"))
    predictions = pd.DataFrame(read_jsonl(probe_dir / "probe_predictions.jsonl"))

    assert "ex-capped" not in set(predictions["example_id"])
    assert "target_column_lsd" in predictions.columns
    assert (
        metrics[
            (metrics["status"] == "fitted")
            & (metrics["target"] == "any_carry")
            & (metrics["location_kind"] == "prompt_final")
            & (metrics["target_column_lsd"].isna())
        ]["test_accuracy"].max()
        == 1.0
    )
    assert (
        metrics[
            (metrics["status"] == "fitted")
            & (metrics["target"] == "outgoing_carry")
            & (metrics["location_kind"] == "operand_digits")
            & (metrics["target_column_lsd"] == 0)
        ]["test_accuracy"].max()
        == 1.0
    )


def test_run_goal2_probes_uses_cot_locations_only_for_free_cot(tmp_path: Path) -> None:
    """Verify CoT locations are ignored for non-free-CoT examples."""
    goal2_run_dir = _write_probe_activation_run(tmp_path, prompt_mode="answer_only")
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "probe",
                "goal2_run_dir": str(goal2_run_dir),
                "output_dir": str(tmp_path / "probes"),
                "targets": ["any_carry"],
                "min_train_examples": 2,
                "min_test_examples": 2,
            }
        ),
        encoding="utf-8",
    )

    probe_dir = run_goal2_probes(load_goal2_probe_config(config_path))
    metrics = pd.DataFrame(read_jsonl(probe_dir / "probe_metrics.jsonl"))

    assert "cot_start" not in set(metrics["location_kind"])
    assert "prompt_final" in set(metrics["location_kind"])


def test_run_goal2_probes_can_exclude_ambiguous_digit_tokens(tmp_path: Path) -> None:
    """Verify ambiguous digit tokens can still be excluded by config."""
    goal2_run_dir = _write_probe_activation_run(tmp_path)
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "probe",
                "goal2_run_dir": str(goal2_run_dir),
                "output_dir": str(tmp_path / "probes"),
                "targets": ["outgoing_carry"],
                "min_train_examples": 2,
                "min_test_examples": 2,
                "require_unambiguous_digit_tokens": True,
            }
        ),
        encoding="utf-8",
    )

    probe_dir = run_goal2_probes(load_goal2_probe_config(config_path))
    metrics = pd.DataFrame(read_jsonl(probe_dir / "probe_metrics.jsonl"))

    assert not (
        (metrics["status"] == "fitted") & (metrics["location_kind"] == "operand_digits")
    ).any()
    assert (
        metrics[
            (metrics["status"] == "fitted")
            & (metrics["location_kind"] == "answer_digits")
            & (metrics["target_column_lsd"] == 0)
        ]["test_accuracy"].max()
        == 1.0
    )


def test_run_goal2_probes_adds_per_column_targets_for_all_locations(tmp_path: Path) -> None:
    """Verify new per-column probe targets train from non-digit locations."""
    goal2_run_dir = _write_probe_activation_run(tmp_path)
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "probe",
                "goal2_run_dir": str(goal2_run_dir),
                "output_dir": str(tmp_path / "probes"),
                "targets": [
                    "incoming_carry",
                    "outgoing_carry",
                    "output_digit",
                    "raw_sum",
                    "carry_chain_membership",
                    "column_pointer",
                ],
                "min_train_examples": 2,
                "min_test_examples": 2,
                "n_jobs": 2,
            }
        ),
        encoding="utf-8",
    )

    probe_dir = run_goal2_probes(load_goal2_probe_config(config_path))
    metrics = pd.DataFrame(read_jsonl(probe_dir / "probe_metrics.jsonl"))

    fitted = metrics[
        (metrics["status"] == "fitted")
        & (metrics["location_kind"] == "cot_start")
        & (metrics["target_column_lsd"] == 0)
    ]
    assert {
        "incoming_carry",
        "outgoing_carry",
        "output_digit",
        "raw_sum",
        "carry_chain_membership",
        "column_pointer",
    }.issubset(set(fitted["target"]))


def test_ambiguous_digit_location_indexes_detect_shared_tokens() -> None:
    """Verify duplicated digit locations on one token are marked ambiguous."""
    locations = [
        _location("operand_a_digit_lsd_1", "operand_digits", 10, 1),
        _location("operand_a_digit_lsd_0", "operand_digits", 10, 0),
        _location("answer_digit_lsd_0", "answer_digits", 20, 0),
    ]

    assert _ambiguous_digit_location_indexes(locations) == {0, 1}


def _write_probe_activation_run(tmp_path: Path, prompt_mode: str = "free_cot") -> Path:
    """Write a tiny Goal 2 activation run for probe tests."""
    run_dir = tmp_path / "goal2"
    labels = {
        "train-0a": ("train_probe", 0),
        "train-0b": ("train_probe", 0),
        "train-1a": ("train_probe", 1),
        "train-1b": ("train_probe", 1),
        "test-0": ("test_probe", 0),
        "test-1": ("test_probe", 1),
        "ex-capped": ("test_probe", 1),
    }
    write_jsonl(
        run_dir / "dataset.jsonl",
        [
            _example_row(example_id, split, label, prompt_mode)
            for example_id, (split, label) in labels.items()
        ],
    )
    records = []
    for example_id, (_, label) in labels.items():
        path = run_dir / "activations" / "model" / f"{example_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        locations = [
            _location("prompt_final", "prompt_final", 0, None),
            _location("operand_a_digit_lsd_1", "operand_digits", 1, 1),
            _location("operand_a_digit_lsd_0", "operand_digits", 1, 0),
            _location("answer_digit_lsd_0", "answer_digits", 2, 0),
            _location("cot_start", "cot_start", 3, None),
        ]
        torch.save(
            {
                "activations": _activation_tensor(label),
                "layer_indices": [0, 1],
                "locations": locations,
            },
            path,
        )
        records.append(
            {
                "example_id": example_id,
                "model_name": "model",
                "activation_path": path.relative_to(run_dir).as_posix(),
                "layer_indices": [0, 1],
                "locations": locations,
                "call_metadata": {"hit_token_limit": example_id == "ex-capped"},
                "parsed_answer": str(label),
            }
        )
    write_jsonl(run_dir / "activations.jsonl", records)
    return run_dir


def _activation_tensor(label: int) -> torch.Tensor:
    """Return a small linearly separable activation tensor."""
    positive = float(label)
    negative = 1.0 - positive
    rows = []
    for _location_index in range(5):
        rows.append(
            torch.tensor(
                [
                    [positive, negative, 0.0],
                    [positive, negative, 1.0],
                ],
                dtype=torch.float32,
            )
        )
    return torch.stack(rows)


def _example_row(
    example_id: str,
    split: str,
    label: int,
    prompt_mode: str,
) -> dict[str, object]:
    """Return a minimal dataset row for probe tests."""
    return {
        "id": example_id,
        "split": split,
        "expected_output": str(label),
        "answer": str(label),
        "prompt_mode": prompt_mode,
        "digit_format": "standard",
        "answer_format": "standard",
        "slice_name": "random",
        "n_digits": 2,
        "carry_count": label,
        "max_carry_chain": label,
        "first_carry_position": 0 if label else None,
        "incoming_carry": [label, 0],
        "outgoing_carry": [label, 0],
        "output_digits_lsd": [label, 1],
        "raw_sum": [label, 1],
    }


def _location(
    name: str,
    kind: str,
    absolute_token_index: int,
    lsd_index: int | None,
) -> dict[str, object]:
    """Return a saved activation-location row."""
    metadata = {} if lsd_index is None else {"lsd_index": lsd_index}
    return {
        "name": name,
        "kind": kind,
        "absolute_token_index": absolute_token_index,
        "token_id": absolute_token_index,
        "token_text": str(absolute_token_index),
        "source": "prompt",
        "metadata": metadata,
    }
