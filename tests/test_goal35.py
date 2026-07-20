from pathlib import Path

from typer.testing import CliRunner

from carry_trace.cli import app
from carry_trace.config import Goal35Config
from carry_trace.datasets import generate_dataset
from carry_trace.enums import ActivationLocation, SliceName
from carry_trace.goal35 import (
    REPLAY_GENERATIONS_FILENAME,
    REPLAY_SCORES_FILENAME,
    _completion_coverage_rows,
    _dataset_config,
    _goal35_run_dir,
    _historical_error_prefixes,
    _primary_metric_rows,
    _replay_cases,
    _shared_example_ids,
    _shared_replay_prefixes,
    _source_replay_prefixes,
)
from carry_trace.io import stable_hash, write_json, write_jsonl


class CharacterTokenizer:
    """Provide character-level IDs and offsets for replay-prefix tests."""

    all_special_ids = [999]

    def __call__(self, text, **kwargs):
        """Return IDs and offsets matching the supplied text exactly."""
        del kwargs
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode character-level token IDs."""
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        """Encode text as character-level token IDs."""
        del add_special_tokens
        return [ord(character) for character in text]


def _config(tmp_path: Path, examples_per_length: int = 2) -> Goal35Config:
    """Return a compact Goal 3.5 config for tests."""
    return Goal35Config(
        name="goal35-test",
        seed=7,
        output_dir=tmp_path / "runs",
        dataset={
            "name": "goal35-test-data",
            "output_dir": tmp_path / "data",
            "examples_per_digit_length": examples_per_length,
            "digit_lengths": [4, 6],
        },
        models=[
            {"name": "sft", "model_id": "model-sft"},
            {"name": "full", "model_id": "model-full"},
        ],
        tokenizer_id="shared-tokenizer",
        runner={"kind": "hf", "batch_size": 2},
        analysis={"bootstrap_samples": 100},
    )


def test_goal35_dataset_generates_requested_problem_count(tmp_path: Path) -> None:
    """Verify Goal 3.5 creates the configured count at each digit length."""
    config = _config(tmp_path, examples_per_length=2)

    _, _, examples = generate_dataset(_dataset_config(config))

    assert len(examples) == 4
    assert sum(example.n_digits == 4 for example in examples) == 2
    assert sum(example.n_digits == 6 for example in examples) == 2
    assert {example.prompt_mode.value for example in examples} == {"free_cot"}


def test_goal35_derives_shared_prefixes_and_full_cross(tmp_path: Path) -> None:
    """Verify valid source completions produce three prefixes and a full 2x2 replay."""
    config = _config(tmp_path, examples_per_length=1)
    _, _, examples = generate_dataset(_dataset_config(config))
    example = examples[0]
    decoded = f"Work through the columns. Final answer: {example.expected_output}"
    records = [
        {
            "example_id": example.id,
            "model_name": model.name,
            "model_id": model.model_id,
            "output_ids": [ord(character) for character in decoded],
            "decoded_output": decoded,
            "token_count_output": len(decoded),
            "hit_token_limit": False,
            "parsed_answer": example.expected_output,
            "parsed_answer_correct": True,
            "metadata": {},
            "generation_config": {},
        }
        for model in config.models
    ]

    source_prefixes, statuses = _source_replay_prefixes(
        config,
        "run",
        [example],
        records,
        CharacterTokenizer(),
    )
    shared_ids = _shared_example_ids(config, statuses)
    replay_prefixes = _shared_replay_prefixes(
        config,
        "run",
        [example],
        source_prefixes,
        shared_ids,
    )
    cases = _replay_cases(config, replay_prefixes)

    assert len(source_prefixes) == 6
    assert shared_ids == {example.id}
    assert len(replay_prefixes) == 7
    assert len(cases) == 14
    assert {case.replay_kind for case in cases} == {"no_reasoning", "self", "crossed"}
    cot_end = next(
        prefix
        for prefix in source_prefixes
        if prefix.source_model_name == "sft" and prefix.location_kind == ActivationLocation.COT_END
    )
    assert cot_end.assistant_prefix.endswith("Final answer: ")


def test_goal35_ignores_terminal_special_token_during_alignment(tmp_path: Path) -> None:
    """Verify an invisible generated EOS token does not invalidate replay boundaries."""
    config = _config(tmp_path, examples_per_length=1)
    _, _, examples = generate_dataset(_dataset_config(config))
    example = examples[0]
    decoded = f"Reasoning. Final answer: {example.expected_output}"
    record = {
        "example_id": example.id,
        "model_name": "sft",
        "model_id": "model-sft",
        "output_ids": [ord(character) for character in decoded] + [999],
        "decoded_output": decoded,
        "token_count_output": len(decoded) + 1,
        "hit_token_limit": False,
        "parsed_answer": example.expected_output,
        "parsed_answer_correct": True,
        "metadata": {},
        "generation_config": {},
    }

    prefixes, statuses = _source_replay_prefixes(
        config,
        "run",
        [example],
        [record],
        CharacterTokenizer(),
    )

    assert len(prefixes) == 3
    assert statuses[0]["answer_boundary_available"] is True
    assert statuses[0]["eligible_for_shared_replay"] is True


def test_goal35_reports_token_limit_hits_before_shared_filtering(tmp_path: Path) -> None:
    """Verify token-limit failures remain in coverage but not crossed replay."""
    config = _config(tmp_path, examples_per_length=1)
    _, _, examples = generate_dataset(_dataset_config(config))
    example = examples[0]
    decoded = f"Reasoning. Final answer: {example.expected_output}"
    records = []
    for model in config.models:
        records.append(
            {
                "example_id": example.id,
                "model_name": model.name,
                "model_id": model.model_id,
                "output_ids": [ord(character) for character in decoded],
                "decoded_output": decoded,
                "token_count_output": len(decoded),
                "hit_token_limit": model.name == "full",
                "parsed_answer": example.expected_output,
                "parsed_answer_correct": True,
                "metadata": {},
                "generation_config": {},
            }
        )

    _, statuses = _source_replay_prefixes(
        config,
        "run",
        [example],
        records,
        CharacterTokenizer(),
    )
    shared_ids = _shared_example_ids(config, statuses)
    for row in statuses:
        row["selected_for_shared_replay"] = row["example_id"] in shared_ids
        row["clean_terminal_answer"] = True
    coverage = _completion_coverage_rows(config, statuses)
    full_four_digit = next(
        row
        for row in coverage
        if row["model_name"] == "full" and row["n_digits"] == example.n_digits
    )

    assert shared_ids == set()
    assert full_four_digit["token_limit_hits"] == 1
    assert full_four_digit["eligible_for_shared_replay"] == 0
    assert full_four_digit["selected_for_shared_replay"] == 0


def test_goal35_imports_only_clean_incorrect_historical_completions(tmp_path: Path) -> None:
    """Verify historical replay imports enforce natural-format and quality filters."""
    config = _config(tmp_path, examples_per_length=2)
    _, _, examples = generate_dataset(_dataset_config(config))
    selected, unclean, nonrandom = examples[:3]
    nonrandom = nonrandom.model_copy(update={"slice_name": SliceName.NO_CARRY})
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    write_jsonl(
        history_dir / "dataset.jsonl",
        [
            selected.model_dump(mode="json"),
            unclean.model_dump(mode="json"),
            nonrandom.model_dump(mode="json"),
        ],
    )
    write_jsonl(
        history_dir / "activations.jsonl",
        [
            {
                "example_id": selected.id,
                "model_name": "sft",
                "model_id": "model-sft",
                "decoded_output": "Work through it. Final answer: 0",
                "parsed_answer": "0",
                "call_metadata": {"hit_token_limit": False},
            },
            {
                "example_id": unclean.id,
                "model_name": "sft",
                "model_id": "model-sft",
                "decoded_output": "Work through it. Final answer: 0 then continue",
                "parsed_answer": "0",
                "call_metadata": {"hit_token_limit": False},
            },
            {
                "example_id": nonrandom.id,
                "model_name": "sft",
                "model_id": "model-sft",
                "decoded_output": "Work through it. Final answer: 0",
                "parsed_answer": "0",
                "call_metadata": {"hit_token_limit": False},
            },
        ],
    )
    config = config.model_copy(update={"historical_source_run_dirs": [history_dir]})

    prefixes, imported_examples, statuses = _historical_error_prefixes(
        config,
        CharacterTokenizer(),
        excluded_problem_ids=set(),
    )

    assert len(prefixes) == 1
    assert prefixes[0].example_id == selected.id
    assert prefixes[0].prefix_token_source == "reconstructed"
    assert prefixes[0].metadata["source_cohort"] == "historical"
    assert [example.id for example in imported_examples] == [selected.id]
    assert {row["status"] for row in statuses} == {
        "selected",
        "unclean_terminal_answer",
    }


def test_goal35_primary_metrics_store_paired_interaction(tmp_path: Path) -> None:
    """Verify primary artifacts include paired source and receiver interaction effects."""
    config = _config(tmp_path, examples_per_length=1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    problem = "problem"
    locations = ["cot_1_3", "cot_2_3", "cot_end"]
    score_rows = []
    generation_rows = []
    for receiver in ["sft", "full"]:
        score_rows.append(
            {
                "id": f"baseline-{receiver}",
                "problem_id": problem,
                "n_digits": 4,
                "receiver_model_name": receiver,
                "source_model_name": None,
                "location_kind": "prompt_final",
                "expected_answer_logprob": 0.0,
            }
        )
        generation_rows.append(
            {
                **score_rows[-1],
                "exact_match": True,
                "parsed_answer": "10",
            }
        )
        for location in locations:
            for source in ["sft", "full"]:
                value = 1.0 if source == "sft" else (2.0 if receiver == "sft" else 3.0)
                row = {
                    "id": f"{receiver}-{source}-{location}",
                    "problem_id": problem,
                    "n_digits": 4,
                    "receiver_model_name": receiver,
                    "source_model_name": source,
                    "location_kind": location,
                    "expected_answer_logprob": value,
                }
                score_rows.append(row)
                if location in {"cot_2_3", "cot_end"}:
                    generation_rows.append(
                        {
                            **row,
                            "exact_match": source == "full",
                            "parsed_answer": "10" if source == "full" else "11",
                        }
                    )
    write_jsonl(run_dir / REPLAY_SCORES_FILENAME, score_rows)
    write_jsonl(run_dir / REPLAY_GENERATIONS_FILENAME, generation_rows)
    audits = [
        {
            "example_id": "example",
            "problem_id": problem,
            "source_model_name": model,
            "source_answer_correct": True,
            "clean_terminal_answer": True,
            "parsed_answer": "10",
        }
        for model in ["sft", "full"]
    ]

    rows = _primary_metric_rows(config, run_dir, audits)
    interaction = next(
        row
        for row in rows
        if row["analysis"] == "receiver_by_source_interaction"
        and row["subset"] == "all_shared"
        and row["n_digits"] is None
        and row["location_kind"] == "cot_end"
    )

    assert interaction["mean_logprob_interaction"] == 1.0
    assert interaction["mean_generation_accuracy_interaction"] == 0.0
    assert interaction["logprob_interaction_ci_lower"] == 1.0
    assert interaction["logprob_interaction_ci_upper"] == 1.0


def test_goal35_historical_errors_do_not_enter_paired_interaction(tmp_path: Path) -> None:
    """Verify historical one-source donors only contribute error-entrainment metrics."""
    config = _config(tmp_path, examples_per_length=1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    problem = "historical-problem"
    score_rows = []
    generation_rows = []
    for receiver in ["sft", "full"]:
        baseline = {
            "id": f"baseline-{receiver}",
            "problem_id": problem,
            "n_digits": 4,
            "receiver_model_name": receiver,
            "source_model_name": None,
            "location_kind": "prompt_final",
            "expected_answer_logprob": -1.0,
            "exact_match": True,
            "parsed_answer": "10",
        }
        replay = {
            "id": f"replay-{receiver}",
            "problem_id": problem,
            "n_digits": 4,
            "receiver_model_name": receiver,
            "source_model_name": "sft",
            "location_kind": "cot_end",
            "expected_answer_logprob": -2.0,
            "exact_match": False,
            "parsed_answer": "11",
        }
        score_rows.extend([baseline, replay])
        generation_rows.extend([baseline, replay])
    write_jsonl(run_dir / REPLAY_SCORES_FILENAME, score_rows)
    write_jsonl(run_dir / REPLAY_GENERATIONS_FILENAME, generation_rows)
    audits = [
        {
            "example_id": "example",
            "problem_id": problem,
            "source_model_name": "sft",
            "source_answer_correct": False,
            "clean_terminal_answer": True,
            "parsed_answer": "11",
            "source_cohort": "historical",
            "source_run_id": "old-run",
        }
    ]

    rows = _primary_metric_rows(config, run_dir, audits)

    assert not any(row["analysis"] == "receiver_by_source_interaction" for row in rows)
    pooled = [
        row
        for row in rows
        if row["analysis"] == "incorrect_source_entrainment"
        and row["source_cohort"] == "historical"
        and row["n_digits"] is None
    ]
    assert len(pooled) == 2
    assert {row["n"] for row in pooled} == {1}
    assert {row["same_source_answer_rate"] for row in pooled} == {1.0}


def test_goal35_run_directory_resumes_matching_incomplete_run(tmp_path: Path) -> None:
    """Verify Goal 3.5 resumes the newest incomplete matching config hash."""
    config = _config(tmp_path)
    config_hash = stable_hash(config.model_dump(mode="json"))
    run_dir = config.output_dir / "goal35-test-existing"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "manifest.json",
        {"config_hash": config_hash, "status": "running"},
    )

    assert _goal35_run_dir(config, config_hash) == run_dir


def test_goal35_explicitly_resumes_a_completed_run_after_config_change(tmp_path: Path) -> None:
    """Verify an explicit run directory bypasses automatic config-hash selection."""
    config = _config(tmp_path)
    run_dir = config.output_dir / "goal35-test-complete"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "manifest.json",
        {
            "artifact_kind": "goal35_generation_only_cot_replay",
            "config_hash": "old-config",
            "status": "complete",
        },
    )

    assert _goal35_run_dir(config, "new-config", resume_run_dir=run_dir) == run_dir


def test_goal35_cli_loads_config(tmp_path: Path, monkeypatch) -> None:
    """Verify the Goal 3.5 CLI loads its config and reports the run directory."""
    config_path = tmp_path / "goal35.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: goal35-test",
                "dataset:",
                "  name: goal35-data",
                "models:",
                "  - name: sft",
                "    model_id: model-sft",
                "  - name: full",
                "    model_id: model-full",
                "tokenizer_id: shared-tokenizer",
                "runner:",
                "  kind: hf",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "goal35-run"
    monkeypatch.setattr(
        "carry_trace.cli.run_goal35",
        lambda config, resume_run_dir=None: run_dir,
    )

    result = CliRunner().invoke(app, ["run", "goal35", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Wrote Goal 3.5 artifacts" in result.output
