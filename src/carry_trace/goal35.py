"""Goal 3.5 generation-only natural-CoT replay experiment."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from carry_trace.config import (
    DatasetConfig,
    Goal3RunConfig,
    Goal35Config,
    RandomSamplingConfig,
    SplitConfig,
)
from carry_trace.datasets import dump_dataset_row, generate_dataset
from carry_trace.enums import (
    ActivationLocation,
    AnswerFormat,
    DigitFormat,
    PromptMode,
    SliceName,
)
from carry_trace.goal2 import _answer_start_token, _final_output_digit_chars, _token_offsets
from carry_trace.goal3 import (
    COMPLETION_AUDITS_FILENAME,
    REPLAY_GENERATIONS_FILENAME,
    REPLAY_SCORES_FILENAME,
    _cleanup_runner,
    _completion_audit_rows,
    _run_replay_model,
    _write_analysis_artifacts,
)
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
from carry_trace.metrics import score_records
from carry_trace.models import HuggingFaceModelRunner
from carry_trace.runs import _load_repairable_jsonl
from carry_trace.schemas import AdditionExample, Goal3ReplayCase, Goal3ReplayPrefix

SOURCE_CALLS_FILENAME = "source_calls.jsonl"
SOURCE_SCORED_CALLS_FILENAME = "source_scored_calls.jsonl"
SOURCE_PREFIXES_FILENAME = "source_prefixes.jsonl"
SOURCE_AUDITS_FILENAME = "source_completion_audits.jsonl"
SOURCE_STATUS_FILENAME = "source_completion_status.jsonl"
COMPLETION_COVERAGE_FILENAME = "completion_coverage.jsonl"
REPLAY_PREFIXES_FILENAME = "replay_prefixes.jsonl"
REPLAY_CASES_FILENAME = "replay_cases.jsonl"
PRIMARY_METRICS_FILENAME = "replay_primary_metrics.jsonl"
HISTORICAL_IMPORT_STATUS_FILENAME = "historical_import_status.jsonl"


def run_goal35(config: Goal35Config, resume_run_dir: Path | None = None) -> Path:
    """Generate problems, gather natural CoTs, and run crossed replay."""
    dataset_path, dataset_manifest_path, examples = generate_dataset(_dataset_config(config))
    config_hash = stable_hash(config.model_dump(mode="json"))
    run_dir = _goal35_run_dir(config, config_hash, resume_run_dir=resume_run_dir)
    manifest = _initial_manifest(
        config,
        config_hash,
        run_dir,
        dataset_path,
        dataset_manifest_path,
        examples,
    )
    write_json(run_dir / "manifest.json", manifest)
    write_jsonl(run_dir / "dataset.jsonl", [dump_dataset_row(example) for example in examples])

    source_calls = _gather_source_completions(config, run_dir, examples, manifest)
    source_scored = score_records(
        [dump_dataset_row(example) for example in examples],
        source_calls,
    )
    write_jsonl(run_dir / SOURCE_SCORED_CALLS_FILENAME, source_scored)

    tokenizer = _load_shared_tokenizer(config)
    source_prefixes, source_status = _source_replay_prefixes(
        config,
        run_dir.name,
        examples,
        source_scored,
        tokenizer,
    )
    historical_prefixes, historical_examples, historical_status = _historical_error_prefixes(
        config,
        tokenizer,
        excluded_problem_ids={example.problem_id for example in examples},
    )
    all_source_prefixes = [*source_prefixes, *historical_prefixes]
    source_audits = _annotate_source_audits(
        _completion_audit_rows(all_source_prefixes),
        all_source_prefixes,
    )
    source_audit_index = {
        (row["example_id"], row["source_model_name"]): row for row in source_audits
    }
    shared_example_ids = _shared_example_ids(config, source_status)
    replay_prefixes = _shared_replay_prefixes(
        config,
        run_dir.name,
        examples,
        source_prefixes,
        shared_example_ids,
    )
    replay_prefixes.extend(
        _historical_replay_prefixes(
            config,
            historical_examples,
            historical_prefixes,
        )
    )
    replay_prefixes.sort(key=lambda row: row.id)
    replay_cases = _replay_cases(config, replay_prefixes)
    replay_audits = _annotate_source_audits(
        _completion_audit_rows(replay_prefixes),
        replay_prefixes,
    )
    source_status = _annotate_source_status(
        source_status,
        source_audit_index,
        shared_example_ids,
    )

    write_jsonl(run_dir / SOURCE_PREFIXES_FILENAME, _prefix_payloads(all_source_prefixes))
    write_jsonl(run_dir / SOURCE_AUDITS_FILENAME, source_audits)
    write_jsonl(run_dir / SOURCE_STATUS_FILENAME, source_status)
    write_jsonl(run_dir / HISTORICAL_IMPORT_STATUS_FILENAME, historical_status)
    write_jsonl(run_dir / REPLAY_PREFIXES_FILENAME, _prefix_payloads(replay_prefixes))
    write_jsonl(run_dir / REPLAY_CASES_FILENAME, _case_payloads(replay_cases))
    write_jsonl(run_dir / COMPLETION_AUDITS_FILENAME, replay_audits)
    coverage = _completion_coverage_rows(config, source_status)
    write_jsonl(run_dir / COMPLETION_COVERAGE_FILENAME, coverage)

    manifest["shared_problem_count"] = len(shared_example_ids)
    manifest["shared_problem_count_by_digit_length"] = _shared_counts(
        examples,
        shared_example_ids,
    )
    manifest["historical_error_problem_count"] = len(historical_examples)
    manifest["historical_error_source_count"] = len(historical_prefixes)
    manifest["expected_replay_scores"] = len(replay_cases)
    decode_locations = set(config.replay.decode_locations)
    manifest["expected_replay_generations"] = sum(
        case.location_kind in decode_locations for case in replay_cases
    )
    manifest["updated_at"] = utc_now_iso()
    write_json(run_dir / "manifest.json", manifest)
    print(
        f"[goal35-prefixes] shared={len(shared_example_ids)}/{len(examples)} "
        f"historical_errors={len(historical_prefixes)} replay_cases={len(replay_cases)}",
        flush=True,
    )

    _repair_replay_artifacts(run_dir)
    replay_config = _replay_run_config(config, run_dir)
    if replay_cases:
        for model_spec in config.models:
            runner = HuggingFaceModelRunner(model_spec, config.runner, config.replay.generation)
            try:
                model_cases = [
                    case for case in replay_cases if case.receiver_model_name == model_spec.name
                ]
                _run_replay_model(replay_config, run_dir, runner, model_cases)
            finally:
                _cleanup_runner(runner)

    _write_analysis_artifacts(run_dir)
    primary_metrics = _primary_metric_rows(config, run_dir, replay_audits)
    write_jsonl(run_dir / PRIMARY_METRICS_FILENAME, primary_metrics)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now_iso()
    manifest["updated_at"] = manifest["completed_at"]
    manifest["artifact_counts"] = _artifact_counts(run_dir)
    write_json(run_dir / "manifest.json", manifest)
    print(
        f"[goal35-analysis] metrics={len(primary_metrics)} "
        f"artifact={run_dir / PRIMARY_METRICS_FILENAME}",
        flush=True,
    )
    return run_dir


def _dataset_config(config: Goal35Config) -> DatasetConfig:
    """Translate Goal 3.5 settings into the shared dataset generator config."""
    return DatasetConfig(
        name=config.dataset.name,
        seed=config.seed,
        output_dir=config.dataset.output_dir,
        write_parquet=False,
        schema_version=config.schema_version,
        splits={
            config.dataset.split: SplitConfig(
                examples_per_slice_per_length=config.dataset.examples_per_digit_length
            )
        },
        digit_lengths=config.dataset.digit_lengths,
        slices=[SliceName.RANDOM],
        prompt_modes=[PromptMode.FREE_COT],
        digit_formats=[DigitFormat.STANDARD],
        answer_formats=[AnswerFormat.STANDARD],
        random_sampling=RandomSamplingConfig(
            balance_carry_count=config.dataset.balance_carry_count
        ),
    )


def _goal35_run_dir(
    config: Goal35Config,
    config_hash: str,
    resume_run_dir: Path | None = None,
) -> Path:
    """Return an incomplete matching run or create a new timestamped directory."""
    if resume_run_dir is not None:
        manifest_path = resume_run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"resume directory has no manifest: {resume_run_dir}")
        manifest = read_json(manifest_path)
        if manifest.get("artifact_kind") != "goal35_generation_only_cot_replay":
            raise ValueError(f"resume directory is not a Goal 3.5 run: {resume_run_dir}")
        return resume_run_dir
    candidates = sorted(
        config.output_dir.glob(f"{config.name}-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("config_hash") == config_hash and manifest.get("status") != "complete":
            return candidate
    run_id = f"{config.name}-{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    return ensure_dir(config.output_dir / run_id)


def _initial_manifest(
    config: Goal35Config,
    config_hash: str,
    run_dir: Path,
    dataset_path: Path,
    dataset_manifest_path: Path,
    examples: list[AdditionExample],
) -> dict[str, Any]:
    """Build or refresh the running Goal 3.5 manifest."""
    existing = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    return {
        "run_id": existing.get("run_id", run_dir.name),
        "created_at": existing.get("created_at", utc_now_iso()),
        "updated_at": utc_now_iso(),
        "status": "running",
        "artifact_kind": "goal35_generation_only_cot_replay",
        "config_hash": config_hash,
        "config": config.model_dump(mode="json"),
        "dataset_path": str(dataset_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "problem_count": len(examples),
        "expected_source_calls": len(examples) * len(config.models),
    }


def _gather_source_completions(
    config: Goal35Config,
    run_dir: Path,
    examples: list[AdditionExample],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate missing natural CoT source completions with append-only progress."""
    calls_path = run_dir / SOURCE_CALLS_FILENAME
    existing = _load_repairable_jsonl(calls_path)
    completed = {
        (
            row.get("model_name"),
            row.get("model_id"),
            row.get("model_revision"),
            row.get("example_id"),
        )
        for row in existing
    }
    for model_spec in config.models:
        pending = [
            example
            for example in examples
            if (model_spec.name, model_spec.model_id, model_spec.revision, example.id)
            not in completed
        ]
        completed_for_model = len(examples) - len(pending)
        print(
            f"[goal35-source] model={model_spec.name} pending={len(pending)} "
            f"completed={completed_for_model}/{len(examples)}",
            flush=True,
        )
        if not pending:
            continue
        runner = HuggingFaceModelRunner(model_spec, config.runner, config.source_generation)
        started = time.perf_counter()
        try:
            for record in runner.generate(pending, run_id=run_dir.name, seed=config.seed):
                payload = record.model_dump(mode="json")
                append_jsonl(calls_path, [payload])
                completed_for_model += 1
                elapsed = (time.perf_counter() - started) / 60.0
                print(
                    f"[goal35-source] model={model_spec.name} "
                    f"completed={completed_for_model}/{len(examples)} "
                    f"example_id={record.example_id} output_tokens={record.token_count_output} "
                    f"elapsed={elapsed:.1f}m",
                    flush=True,
                )
        finally:
            _cleanup_runner(runner)
        manifest["updated_at"] = utc_now_iso()
        write_json(run_dir / "manifest.json", manifest)
    return _load_repairable_jsonl(calls_path)


def _load_shared_tokenizer(config: Goal35Config) -> Any:
    """Load the tokenizer shared by source and receiver checkpoints."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.tokenizer_id,
        revision=config.tokenizer_revision,
        trust_remote_code=config.runner.trust_remote_code,
        local_files_only=config.tokenizer_local_files_only,
    )


def _historical_error_prefixes(
    config: Goal35Config,
    tokenizer: Any,
    excluded_problem_ids: set[str],
) -> tuple[list[Goal3ReplayPrefix], list[AdditionExample], list[dict[str, Any]]]:
    """Import clean incorrect endpoint CoTs from configured historical runs."""
    prefixes: list[Goal3ReplayPrefix] = []
    imported_examples: dict[str, AdditionExample] = {}
    status_rows: list[dict[str, Any]] = []
    seen_problem_ids = set(excluded_problem_ids)
    allowed_models = {model.name for model in config.models}
    allowed_lengths = set(config.dataset.digit_lengths)
    for source_run_dir in config.historical_source_run_dirs:
        dataset_path = source_run_dir / "dataset.jsonl"
        records_path = _historical_records_path(source_run_dir)
        examples_by_id = {str(row["id"]): row for row in read_jsonl(dataset_path)}
        records = []
        for raw_record in read_jsonl(records_path):
            example_row = examples_by_id.get(str(raw_record.get("example_id")))
            if example_row is None or not _historical_example_allowed(
                example_row,
                allowed_lengths,
            ):
                continue
            if raw_record.get("model_name") not in allowed_models:
                continue
            record = _normalize_historical_record(raw_record, tokenizer)
            scored = score_records([example_row], [record])[0]
            if scored["hit_token_limit"] or scored["parsed_answer_correct"]:
                continue
            records.append((AdditionExample.model_validate(example_row), scored))

        candidates_by_problem: dict[str, list[tuple[AdditionExample, dict[str, Any]]]] = (
            defaultdict(list)
        )
        for example, record in records:
            candidates_by_problem[example.problem_id].append((example, record))
        for problem_id, candidates in sorted(candidates_by_problem.items()):
            if problem_id in seen_problem_ids:
                status_rows.extend(
                    _historical_status_rows(source_run_dir, candidates, "duplicate_problem")
                )
                continue
            candidate_prefixes = []
            candidate_examples = []
            for example, record in candidates:
                record_prefixes = _record_prefixes(
                    config,
                    source_run_dir.name,
                    example,
                    record,
                    tokenizer,
                )
                endpoint = next(
                    (
                        prefix
                        for prefix in record_prefixes
                        if prefix.location_kind == ActivationLocation.COT_END
                    ),
                    None,
                )
                if endpoint is None:
                    status_rows.extend(
                        _historical_status_rows(
                            source_run_dir,
                            [(example, record)],
                            "missing_answer_boundary",
                        )
                    )
                    continue
                audit = _completion_audit_rows([endpoint])[0]
                if not audit["clean_terminal_answer"]:
                    status_rows.extend(
                        _historical_status_rows(
                            source_run_dir,
                            [(example, record)],
                            "unclean_terminal_answer",
                        )
                    )
                    continue
                endpoint.metadata["source_cohort"] = "historical"
                endpoint.metadata["source_run_id"] = source_run_dir.name
                candidate_prefixes.append(endpoint)
                candidate_examples.append(example)
                status_rows.extend(
                    _historical_status_rows(
                        source_run_dir,
                        [(example, record)],
                        "selected",
                    )
                )
            if candidate_prefixes:
                prefixes.extend(candidate_prefixes)
                imported_examples[problem_id] = candidate_examples[0]
                seen_problem_ids.add(problem_id)
    return prefixes, list(imported_examples.values()), status_rows


def _historical_records_path(source_run_dir: Path) -> Path:
    """Return the supported completion artifact in one historical run."""
    for filename in [SOURCE_CALLS_FILENAME, "calls.jsonl", "activations.jsonl"]:
        path = source_run_dir / filename
        if path.exists():
            return path
    raise ValueError(f"historical run has no completion records: {source_run_dir}")


def _historical_example_allowed(row: dict[str, Any], allowed_lengths: set[int]) -> bool:
    """Return whether a historical example matches the natural replay design."""
    return (
        row.get("prompt_mode") == PromptMode.FREE_COT.value
        and row.get("slice_name") == SliceName.RANDOM.value
        and row.get("digit_format") == DigitFormat.STANDARD.value
        and row.get("answer_format") == AnswerFormat.STANDARD.value
        and int(row.get("n_digits", -1)) in allowed_lengths
    )


def _normalize_historical_record(record: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Normalize legacy Goal 1/2 completion fields for Goal 3.5 replay."""
    normalized = dict(record)
    metadata = dict(record.get("metadata") or record.get("call_metadata") or {})
    normalized["metadata"] = metadata
    if record.get("output_ids") is None:
        normalized["output_ids"] = tokenizer.encode(
            str(record.get("decoded_output", "")),
            add_special_tokens=False,
        )
        normalized["prefix_token_source"] = "reconstructed"
    else:
        normalized["prefix_token_source"] = "recorded"
    return normalized


def _historical_status_rows(
    source_run_dir: Path,
    candidates: list[tuple[AdditionExample, dict[str, Any]]],
    status: str,
) -> list[dict[str, Any]]:
    """Describe selection outcomes for historical incorrect completions."""
    return [
        {
            "source_run_id": source_run_dir.name,
            "example_id": example.id,
            "problem_id": example.problem_id,
            "model_name": record["model_name"],
            "n_digits": example.n_digits,
            "parsed_answer": record.get("parsed_answer"),
            "status": status,
        }
        for example, record in candidates
    ]


def _source_replay_prefixes(
    config: Goal35Config,
    run_id: str,
    examples: list[AdditionExample],
    records: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[Goal3ReplayPrefix], list[dict[str, Any]]]:
    """Derive CoT-relative replay prefixes and per-completion eligibility rows."""
    example_by_id = {example.id: example for example in examples}
    prefixes: list[Goal3ReplayPrefix] = []
    statuses: list[dict[str, Any]] = []
    required_locations = set(config.source_locations)
    for record in records:
        example = example_by_id[str(record["example_id"])]
        record_prefixes = _record_prefixes(config, run_id, example, record, tokenizer)
        available_locations = {prefix.location_kind for prefix in record_prefixes}
        hit_token_limit = bool(record.get("hit_token_limit"))
        generation_valid = not hit_token_limit or not config.require_no_token_limit_hit
        eligible = generation_valid and required_locations.issubset(available_locations)
        prefixes.extend(record_prefixes)
        statuses.append(
            {
                "example_id": example.id,
                "problem_id": example.problem_id,
                "model_name": record["model_name"],
                "n_digits": example.n_digits,
                "token_count_output": record.get("token_count_output"),
                "hit_token_limit": hit_token_limit,
                "generation_valid": generation_valid,
                "answer_boundary_available": ActivationLocation.COT_END in available_locations,
                "all_replay_locations_available": required_locations.issubset(available_locations),
                "eligible_for_shared_replay": eligible,
                "parsed_answer": record.get("parsed_answer"),
                "parsed_answer_correct": record.get("parsed_answer_correct"),
            }
        )
    return prefixes, statuses


def _record_prefixes(
    config: Goal35Config,
    run_id: str,
    example: AdditionExample,
    record: dict[str, Any],
    tokenizer: Any,
) -> list[Goal3ReplayPrefix]:
    """Build requested replay prefixes from one source completion."""
    output_ids = _trim_trailing_special_tokens(
        tokenizer,
        [int(token_id) for token_id in record.get("output_ids", [])],
    )
    decoded_output = str(record.get("decoded_output", ""))
    offsets = _token_offsets(tokenizer, decoded_output, output_ids)
    answer = _final_output_digit_chars(decoded_output, base=example.base)
    reasoning_count = _answer_start_token(answer, offsets)
    if reasoning_count is None or reasoning_count <= 0:
        return []
    output_indices = {
        ActivationLocation.COT_1_3: (reasoning_count - 1) // 3,
        ActivationLocation.COT_2_3: ((reasoning_count - 1) * 2) // 3,
        ActivationLocation.COT_END: reasoning_count - 1,
    }
    rows = []
    for location_kind in config.source_locations:
        output_index = output_indices[location_kind]
        prefix_ids = output_ids[: output_index + 1]
        row_id = stable_hash(
            {
                "run_id": run_id,
                "example_id": example.id,
                "source_model_name": record["model_name"],
                "location_kind": location_kind,
            },
            length=16,
        )
        rows.append(
            Goal3ReplayPrefix(
                id=row_id,
                schema_version=config.schema_version,
                source_goal2_run_id=run_id,
                example_id=example.id,
                problem_id=example.problem_id,
                split=example.split,
                n_digits=example.n_digits,
                source_model_name=str(record["model_name"]),
                source_model_id=str(record["model_id"]),
                location_kind=location_kind,
                recorded_output_token_end_index=output_index,
                replay_output_token_end_index=output_index,
                prefix_token_source=str(record.get("prefix_token_source", "recorded")),
                prefix_alignment_delta=0,
                assistant_prefix_token_ids=prefix_ids,
                assistant_prefix=tokenizer.decode(prefix_ids, skip_special_tokens=True),
                decoded_output=decoded_output,
                parsed_answer=record.get("parsed_answer"),
                expected_output=str(example.expected_output or example.answer),
                prompt=example.prompt,
                messages=example.messages,
                metadata={
                    "source_answer_correct": record.get("parsed_answer_correct"),
                    "source_call_metadata": record.get("metadata") or {},
                    "source_generation_config": record.get("generation_config") or {},
                    "source_cohort": record.get("source_cohort", "goal35_native"),
                    "source_run_id": record.get("source_run_id", run_id),
                },
            )
        )
    return rows


def _trim_trailing_special_tokens(tokenizer: Any, token_ids: list[int]) -> list[int]:
    """Remove invisible terminal control tokens before text-offset alignment."""
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] in special_ids:
        end -= 1
    return token_ids[:end]


def _shared_example_ids(config: Goal35Config, statuses: list[dict[str, Any]]) -> set[str]:
    """Return examples eligible for replay from both configured source models."""
    eligible_models: dict[str, set[str]] = defaultdict(set)
    for row in statuses:
        if row["eligible_for_shared_replay"]:
            eligible_models[str(row["example_id"])].add(str(row["model_name"]))
    required = {model.name for model in config.models}
    return {
        example_id
        for example_id, model_names in eligible_models.items()
        if required.issubset(model_names)
    }


def _shared_replay_prefixes(
    config: Goal35Config,
    run_id: str,
    examples: list[AdditionExample],
    source_prefixes: list[Goal3ReplayPrefix],
    shared_example_ids: set[str],
) -> list[Goal3ReplayPrefix]:
    """Combine shared source prefixes with one no-reasoning prefix per problem."""
    rows = [prefix for prefix in source_prefixes if prefix.example_id in shared_example_ids]
    for example in examples:
        if example.id not in shared_example_ids:
            continue
        rows.append(_no_reasoning_prefix(config, run_id, example, "goal35_native"))
    return sorted(rows, key=lambda row: row.id)


def _historical_replay_prefixes(
    config: Goal35Config,
    examples: list[AdditionExample],
    source_prefixes: list[Goal3ReplayPrefix],
) -> list[Goal3ReplayPrefix]:
    """Combine historical incorrect endpoints with one direct baseline per problem."""
    rows = list(source_prefixes)
    source_run_by_problem = {
        prefix.problem_id: prefix.source_goal2_run_id for prefix in source_prefixes
    }
    for example in examples:
        rows.append(
            _no_reasoning_prefix(
                config,
                source_run_by_problem[example.problem_id],
                example,
                "historical",
            )
        )
    return rows


def _no_reasoning_prefix(
    config: Goal35Config,
    source_run_id: str,
    example: AdditionExample,
    source_cohort: str,
) -> Goal3ReplayPrefix:
    """Build an empty assistant prefix for a direct-generation baseline."""
    row_id = stable_hash(
        {
            "run_id": source_run_id,
            "example_id": example.id,
            "source_model_name": None,
            "location_kind": ActivationLocation.PROMPT_FINAL,
        },
        length=16,
    )
    return Goal3ReplayPrefix(
        id=row_id,
        schema_version=config.schema_version,
        source_goal2_run_id=source_run_id,
        example_id=example.id,
        problem_id=example.problem_id,
        split=example.split,
        n_digits=example.n_digits,
        location_kind=ActivationLocation.PROMPT_FINAL,
        prefix_token_source="empty",
        assistant_prefix_token_ids=[],
        assistant_prefix="",
        decoded_output="",
        expected_output=str(example.expected_output or example.answer),
        prompt=example.prompt,
        messages=example.messages,
        metadata={
            "source_cohort": source_cohort,
            "source_run_id": source_run_id,
        },
    )


def _replay_cases(config: Goal35Config, prefixes: list[Goal3ReplayPrefix]) -> list[Goal3ReplayCase]:
    """Cross every shared source prefix with both receiving checkpoints."""
    rows = []
    for prefix in prefixes:
        for receiver in [model.name for model in config.models]:
            replay_kind = (
                "no_reasoning"
                if prefix.source_model_name is None
                else "self"
                if receiver == prefix.source_model_name
                else "crossed"
            )
            row_id = stable_hash(
                {"replay_prefix_id": prefix.id, "receiver_model_name": receiver},
                length=16,
            )
            rows.append(
                Goal3ReplayCase(
                    id=row_id,
                    schema_version=config.schema_version,
                    replay_prefix_id=prefix.id,
                    source_goal2_run_id=prefix.source_goal2_run_id,
                    example_id=prefix.example_id,
                    problem_id=prefix.problem_id,
                    split=prefix.split,
                    n_digits=prefix.n_digits,
                    replay_kind=replay_kind,
                    source_model_name=prefix.source_model_name,
                    receiver_model_name=receiver,
                    location_kind=prefix.location_kind,
                    assistant_prefix_token_ids=prefix.assistant_prefix_token_ids,
                    assistant_prefix=prefix.assistant_prefix,
                    expected_output=prefix.expected_output,
                    prompt=prefix.prompt,
                    messages=prefix.messages,
                    metadata={
                        "source_answer_correct": prefix.metadata.get("source_answer_correct"),
                        "source_cohort": prefix.metadata.get("source_cohort"),
                        "source_run_id": prefix.metadata.get("source_run_id"),
                    },
                )
            )
    return sorted(rows, key=lambda row: row.id)


def _annotate_source_status(
    statuses: list[dict[str, Any]],
    audit_index: dict[tuple[str, str], dict[str, Any]],
    shared_example_ids: set[str],
) -> list[dict[str, Any]]:
    """Attach completion cleanliness and shared-selection fields."""
    rows = []
    for row in statuses:
        audit = audit_index.get((str(row["example_id"]), str(row["model_name"])), {})
        rows.append(
            {
                **row,
                "clean_terminal_answer": audit.get("clean_terminal_answer"),
                "selected_for_shared_replay": row["example_id"] in shared_example_ids,
            }
        )
    return rows


def _annotate_source_audits(
    audits: list[dict[str, Any]],
    prefixes: list[Goal3ReplayPrefix],
) -> list[dict[str, Any]]:
    """Attach source-run and cohort provenance to completion audits."""
    provenance = {
        (prefix.example_id, prefix.source_model_name): {
            "source_run_id": prefix.metadata.get(
                "source_run_id",
                prefix.source_goal2_run_id,
            ),
            "source_cohort": prefix.metadata.get("source_cohort", "goal35_native"),
        }
        for prefix in prefixes
        if prefix.source_model_name is not None
    }
    return [
        {
            **row,
            **provenance.get(
                (row["example_id"], row["source_model_name"]),
                {},
            ),
        }
        for row in audits
    ]


def _completion_coverage_rows(
    config: Goal35Config,
    statuses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate source completion and shared-selection coverage."""
    rows = []
    for model in config.models:
        model_rows = [row for row in statuses if row["model_name"] == model.name]
        for n_digits in [None, *config.dataset.digit_lengths]:
            group = (
                model_rows
                if n_digits is None
                else [row for row in model_rows if row["n_digits"] == n_digits]
            )
            if not group:
                continue
            rows.append(
                {
                    "model_name": model.name,
                    "n_digits": n_digits,
                    "requested": len(group),
                    "token_limit_hits": sum(row["hit_token_limit"] for row in group),
                    "token_limit_hit_rate": _mean(group, "hit_token_limit"),
                    "answer_boundary_available": sum(
                        row["answer_boundary_available"] for row in group
                    ),
                    "eligible_for_shared_replay": sum(
                        row["eligible_for_shared_replay"] for row in group
                    ),
                    "selected_for_shared_replay": sum(
                        row["selected_for_shared_replay"] for row in group
                    ),
                    "eligible_source_accuracy": _conditional_mean(
                        group,
                        "parsed_answer_correct",
                        "eligible_for_shared_replay",
                    ),
                    "selected_source_accuracy": _conditional_mean(
                        group,
                        "parsed_answer_correct",
                        "selected_for_shared_replay",
                    ),
                    "selected_clean_rate": _conditional_mean(
                        group,
                        "clean_terminal_answer",
                        "selected_for_shared_replay",
                    ),
                }
            )
    return rows


def _replay_run_config(config: Goal35Config, run_dir: Path) -> Goal3RunConfig:
    """Build the Goal 3 replay config consumed by shared execution helpers."""
    return Goal3RunConfig(
        name=config.name,
        seed=config.seed,
        dataset_bundle_dir=run_dir,
        output_dir=config.output_dir,
        models=config.models,
        runner=config.runner,
        replay=config.replay,
        residual={"enabled": False},
    )


def _repair_replay_artifacts(run_dir: Path) -> None:
    """Repair malformed JSONL tails before replay resumes."""
    for filename in [REPLAY_SCORES_FILENAME, REPLAY_GENERATIONS_FILENAME]:
        _load_repairable_jsonl(run_dir / filename)


def _primary_metric_rows(
    config: Goal35Config,
    run_dir: Path,
    audits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute paired self, crossed-source, interaction, and error-entrainment metrics."""
    if not (run_dir / REPLAY_SCORES_FILENAME).exists():
        return []
    scores = read_jsonl(run_dir / REPLAY_SCORES_FILENAME)
    generations = (
        read_jsonl(run_dir / REPLAY_GENERATIONS_FILENAME)
        if (run_dir / REPLAY_GENERATIONS_FILENAME).exists()
        else []
    )
    score_index = {
        (
            row["problem_id"],
            row["receiver_model_name"],
            row["source_model_name"],
            row["location_kind"],
        ): row
        for row in scores
    }
    generation_index = {
        (
            row["problem_id"],
            row["receiver_model_name"],
            row["source_model_name"],
            row["location_kind"],
        ): row
        for row in generations
    }
    audit_index = {(row["problem_id"], row["source_model_name"]): row for row in audits}
    problems = sorted(
        {
            row["problem_id"]
            for row in scores
            if row["location_kind"] == ActivationLocation.PROMPT_FINAL.value
        }
    )
    n_digits_by_problem = {
        row["problem_id"]: int(row["n_digits"]) for row in scores if row["problem_id"] in problems
    }
    model_names = [model.name for model in config.models]
    paired_problems = [
        problem
        for problem in problems
        if all(
            (problem, model) in audit_index
            and audit_index[(problem, model)].get("source_cohort", "goal35_native")
            == "goal35_native"
            for model in model_names
        )
    ]
    subsets = _analysis_subsets(paired_problems, model_names, audit_index)
    rng = np.random.default_rng(config.seed)
    rows = []
    for subset_name, subset_problems in subsets.items():
        for n_digits in [None, *config.dataset.digit_lengths]:
            selected = [
                problem
                for problem in subset_problems
                if n_digits is None or n_digits_by_problem[problem] == n_digits
            ]
            if not selected:
                continue
            rows.extend(
                _self_replay_metric_rows(
                    config,
                    selected,
                    subset_name,
                    n_digits,
                    score_index,
                    generation_index,
                    rng,
                )
            )
            rows.extend(
                _crossed_replay_metric_rows(
                    config,
                    selected,
                    subset_name,
                    n_digits,
                    score_index,
                    generation_index,
                    rng,
                )
            )
    rows.extend(
        _incorrect_source_metric_rows(
            config,
            problems,
            n_digits_by_problem,
            audit_index,
            generation_index,
            rng,
        )
    )
    return rows


def _analysis_subsets(
    problems: list[str],
    model_names: list[str],
    audits: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[str]]:
    """Return all-shared and source-quality sensitivity subsets."""
    return {
        "all_shared": problems,
        "both_source_correct": [
            problem
            for problem in problems
            if all(audits[(problem, model)]["source_answer_correct"] for model in model_names)
        ],
        "both_source_correct_clean": [
            problem
            for problem in problems
            if all(
                audits[(problem, model)]["source_answer_correct"]
                and audits[(problem, model)]["clean_terminal_answer"]
                for model in model_names
            )
        ],
    }


def _self_replay_metric_rows(
    config: Goal35Config,
    problems: list[str],
    subset: str,
    n_digits: int | None,
    scores: dict[tuple[Any, ...], dict[str, Any]],
    generations: dict[tuple[Any, ...], dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Compute paired self-replay changes from each receiver's direct baseline."""
    rows = []
    for model in [spec.name for spec in config.models]:
        for location in config.source_locations:
            pairs = [
                (
                    scores[(problem, model, None, ActivationLocation.PROMPT_FINAL.value)],
                    scores[(problem, model, model, location.value)],
                )
                for problem in problems
            ]
            deltas = [
                replay["expected_answer_logprob"] - baseline["expected_answer_logprob"]
                for baseline, replay in pairs
            ]
            row = {
                "analysis": "self_replay",
                "subset": subset,
                "n_digits": n_digits,
                "receiver_model_name": model,
                "source_model_name": model,
                "location_kind": location.value,
                **_estimate_fields("logprob_delta", deltas, config, rng),
            }
            generation_pairs = _generation_pairs(
                problems,
                model,
                model,
                location,
                generations,
            )
            row.update(_generation_effect_fields(generation_pairs, config, rng))
            rows.append(row)
    return rows


def _crossed_replay_metric_rows(
    config: Goal35Config,
    problems: list[str],
    subset: str,
    n_digits: int | None,
    scores: dict[tuple[Any, ...], dict[str, Any]],
    generations: dict[tuple[Any, ...], dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Compute source effects within receivers and their interaction."""
    baseline_source, comparison_source = [model.name for model in config.models]
    rows = []
    for location in config.source_locations:
        receiver_effects: dict[str, list[float]] = {}
        receiver_generation_effects: dict[str, list[float]] = {}
        for receiver in [baseline_source, comparison_source]:
            effects = [
                scores[(problem, receiver, comparison_source, location.value)][
                    "expected_answer_logprob"
                ]
                - scores[(problem, receiver, baseline_source, location.value)][
                    "expected_answer_logprob"
                ]
                for problem in problems
            ]
            receiver_effects[receiver] = effects
            generation_pairs = _source_generation_pairs(
                problems,
                receiver,
                baseline_source,
                comparison_source,
                location,
                generations,
            )
            receiver_generation_effects[receiver] = [
                comparison - baseline for baseline, comparison in generation_pairs
            ]
            row = {
                "analysis": "source_effect",
                "subset": subset,
                "n_digits": n_digits,
                "receiver_model_name": receiver,
                "baseline_source_model_name": baseline_source,
                "comparison_source_model_name": comparison_source,
                "location_kind": location.value,
                **_estimate_fields("logprob_delta", effects, config, rng),
            }
            row.update(_source_generation_effect_fields(generation_pairs, config, rng))
            rows.append(row)
        interactions = [
            comparison - baseline
            for baseline, comparison in zip(
                receiver_effects[baseline_source],
                receiver_effects[comparison_source],
                strict=True,
            )
        ]
        generation_interactions = [
            comparison - baseline
            for baseline, comparison in zip(
                receiver_generation_effects[baseline_source],
                receiver_generation_effects[comparison_source],
                strict=True,
            )
        ]
        rows.append(
            {
                "analysis": "receiver_by_source_interaction",
                "subset": subset,
                "n_digits": n_digits,
                "baseline_receiver_model_name": baseline_source,
                "comparison_receiver_model_name": comparison_source,
                "baseline_source_model_name": baseline_source,
                "comparison_source_model_name": comparison_source,
                "location_kind": location.value,
                **_estimate_fields("logprob_interaction", interactions, config, rng),
                **_estimate_fields(
                    "generation_accuracy_interaction",
                    generation_interactions,
                    config,
                    rng,
                ),
            }
        )
    return rows


def _incorrect_source_metric_rows(
    config: Goal35Config,
    problems: list[str],
    n_digits_by_problem: dict[str, int],
    audits: dict[tuple[str, str], dict[str, Any]],
    generations: dict[tuple[Any, ...], dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Measure copying and correction of naturally incorrect clean CoTs."""
    rows = []
    available_endpoints = {
        (key[0], key[2]) for key in generations if key[3] == ActivationLocation.COT_END.value
    }
    for source in [model.name for model in config.models]:
        incorrect_audits = [
            audits[(problem, source)]
            for problem in problems
            if (problem, source) in audits
            and not audits[(problem, source)]["source_answer_correct"]
            and audits[(problem, source)]["clean_terminal_answer"]
            and (problem, source) in available_endpoints
        ]
        source_runs = sorted(
            {
                str(audit.get("source_run_id"))
                for audit in incorrect_audits
                if audit.get("source_cohort") == "historical"
            }
        )
        groups = [
            ("all", None),
            ("goal35_native", None),
            ("historical", None),
            *[("historical_run", source_run) for source_run in source_runs],
        ]
        for source_cohort, source_run_id in groups:
            grouped = [
                audit
                for audit in incorrect_audits
                if _audit_in_cohort(audit, source_cohort, source_run_id)
            ]
            for n_digits in [None, *config.dataset.digit_lengths]:
                selected = [
                    str(audit["problem_id"])
                    for audit in grouped
                    if n_digits is None or n_digits_by_problem[str(audit["problem_id"])] == n_digits
                ]
                if not selected:
                    continue
                for receiver in [model.name for model in config.models]:
                    pairs = _generation_pairs(
                        selected,
                        receiver,
                        source,
                        ActivationLocation.COT_END,
                        generations,
                    )
                    replay_rows = [
                        generations[
                            (
                                problem,
                                receiver,
                                source,
                                ActivationLocation.COT_END.value,
                            )
                        ]
                        for problem in selected
                    ]
                    same_source = [
                        str(row.get("parsed_answer"))
                        == str(audits[(row["problem_id"], source)].get("parsed_answer"))
                        for row in replay_rows
                    ]
                    rows.append(
                        {
                            "analysis": "incorrect_source_entrainment",
                            "subset": "source_incorrect_clean",
                            "source_cohort": source_cohort,
                            "source_run_id": source_run_id,
                            "n_digits": n_digits,
                            "source_model_name": source,
                            "receiver_model_name": receiver,
                            "location_kind": ActivationLocation.COT_END.value,
                            "n": len(selected),
                            "direct_generation_accuracy": float(
                                np.mean([baseline for baseline, _ in pairs])
                            ),
                            "replay_generation_accuracy": float(
                                np.mean([replay for _, replay in pairs])
                            ),
                            **_estimate_fields(
                                "generation_accuracy_delta",
                                [replay - baseline for baseline, replay in pairs],
                                config,
                                rng,
                            ),
                            "same_source_answer_rate": float(np.mean(same_source)),
                        }
                    )
    return rows


def _audit_in_cohort(
    audit: dict[str, Any],
    source_cohort: str,
    source_run_id: str | None,
) -> bool:
    """Return whether an incorrect-source audit belongs to one metric cohort."""
    audit_cohort = str(audit.get("source_cohort", "goal35_native"))
    if source_cohort == "all":
        return True
    if source_cohort == "historical_run":
        return audit_cohort == "historical" and audit.get("source_run_id") == source_run_id
    return audit_cohort == source_cohort


def _generation_pairs(
    problems: list[str],
    receiver: str,
    source: str,
    location: ActivationLocation,
    generations: dict[tuple[Any, ...], dict[str, Any]],
) -> list[tuple[float, float]]:
    """Return baseline and replay exact-match pairs when both were decoded."""
    pairs = []
    for problem in problems:
        baseline = generations.get((problem, receiver, None, ActivationLocation.PROMPT_FINAL.value))
        replay = generations.get((problem, receiver, source, location.value))
        if baseline is not None and replay is not None:
            pairs.append((float(baseline["exact_match"]), float(replay["exact_match"])))
    return pairs


def _source_generation_pairs(
    problems: list[str],
    receiver: str,
    baseline_source: str,
    comparison_source: str,
    location: ActivationLocation,
    generations: dict[tuple[Any, ...], dict[str, Any]],
) -> list[tuple[float, float]]:
    """Return paired source-specific exact-match outcomes within one receiver."""
    pairs = []
    for problem in problems:
        baseline = generations.get((problem, receiver, baseline_source, location.value))
        comparison = generations.get((problem, receiver, comparison_source, location.value))
        if baseline is not None and comparison is not None:
            pairs.append((float(baseline["exact_match"]), float(comparison["exact_match"])))
    return pairs


def _generation_effect_fields(
    pairs: list[tuple[float, float]],
    config: Goal35Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Summarize baseline-to-replay exact-match changes."""
    if not pairs:
        return {"generation_n": 0}
    return {
        "generation_n": len(pairs),
        "baseline_generation_accuracy": float(np.mean([pair[0] for pair in pairs])),
        "replay_generation_accuracy": float(np.mean([pair[1] for pair in pairs])),
        **_estimate_fields(
            "generation_accuracy_delta",
            [replay - baseline for baseline, replay in pairs],
            config,
            rng,
        ),
    }


def _source_generation_effect_fields(
    pairs: list[tuple[float, float]],
    config: Goal35Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Summarize exact-match changes between two reasoning sources."""
    if not pairs:
        return {"generation_n": 0}
    return {
        "generation_n": len(pairs),
        "baseline_source_generation_accuracy": float(np.mean([pair[0] for pair in pairs])),
        "comparison_source_generation_accuracy": float(np.mean([pair[1] for pair in pairs])),
        **_estimate_fields(
            "generation_accuracy_delta",
            [comparison - baseline for baseline, comparison in pairs],
            config,
            rng,
        ),
    }


def _estimate_fields(
    name: str,
    values: list[float],
    config: Goal35Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Return a mean and paired problem-bootstrap confidence interval."""
    if not values:
        return {
            f"{name}_n": 0,
            f"mean_{name}": None,
            f"{name}_ci_lower": None,
            f"{name}_ci_upper": None,
        }
    array = np.asarray(values, dtype=float)
    indices = rng.integers(
        0,
        len(array),
        size=(config.analysis.bootstrap_samples, len(array)),
    )
    bootstrap_means = array[indices].mean(axis=1)
    alpha = 1.0 - config.analysis.confidence_level
    lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        f"{name}_n": len(values),
        f"mean_{name}": float(array.mean()),
        f"{name}_ci_lower": float(lower),
        f"{name}_ci_upper": float(upper),
    }


def _prefix_payloads(rows: list[Goal3ReplayPrefix]) -> list[dict[str, Any]]:
    """Serialize replay prefixes for JSONL output."""
    return [row.model_dump(mode="json") for row in rows]


def _case_payloads(rows: list[Goal3ReplayCase]) -> list[dict[str, Any]]:
    """Serialize replay cases for JSONL output."""
    return [row.model_dump(mode="json") for row in rows]


def _shared_counts(
    examples: list[AdditionExample],
    shared_example_ids: set[str],
) -> dict[str, int]:
    """Count shared replay problems by digit length."""
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        if example.id in shared_example_ids:
            counts[str(example.n_digits)] += 1
    return dict(counts)


def _conditional_mean(
    rows: list[dict[str, Any]],
    value_field: str,
    condition_field: str,
) -> float | None:
    """Return a numeric mean among rows satisfying one Boolean condition."""
    values = [
        float(row[value_field])
        for row in rows
        if row.get(condition_field) and row.get(value_field) is not None
    ]
    return float(np.mean(values)) if values else None


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    """Return the mean of available numeric values."""
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _artifact_counts(run_dir: Path) -> dict[str, int]:
    """Count rows in Goal 3.5 JSONL artifacts."""
    return {
        path.name: len(_load_repairable_jsonl(path)) for path in sorted(run_dir.glob("*.jsonl"))
    }
