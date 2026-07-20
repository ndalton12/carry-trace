"""Goal 3 natural-CoT dataset derivation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from carry_trace.arithmetic import digits_lsd_to_str
from carry_trace.config import Goal3DatasetConfig
from carry_trace.enums import ActivationLocation, ProbeTarget
from carry_trace.io import (
    ensure_dir,
    read_json,
    read_jsonl,
    stable_hash,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from carry_trace.schemas import (
    Goal3ReplayCase,
    Goal3ReplayPrefix,
    Goal3ResidualInterventionCase,
)

GOAL3_REPLAY_PREFIXES_FILENAME = "replay_prefixes.jsonl"
GOAL3_REPLAY_CASES_FILENAME = "replay_cases.jsonl"
GOAL3_RESIDUAL_CASES_FILENAME = "residual_intervention_cases.jsonl"


def generate_goal3_dataset_bundle(config: Goal3DatasetConfig) -> tuple[Path, dict[str, int]]:
    """Derive Goal 3 replay and intervention datasets from a Goal 2 run."""
    source_manifest = read_json(config.source_goal2_run_dir / "manifest.json")
    source_dataset_path = _source_dataset_path(config, source_manifest)
    examples = read_jsonl(source_dataset_path)
    example_by_id = {str(row["id"]): row for row in examples}
    records = _filtered_activation_records(config, example_by_id)
    model_names = _selected_model_names(config, records)
    records_by_example = _records_by_shared_example(config, records, model_names)
    answer_only_ids = _answer_only_example_ids(examples)
    source_run_id = str(source_manifest.get("run_id", config.source_goal2_run_dir.name))
    tokenizer_cache: dict[tuple[str, str | None], Any] = {}

    replay_prefixes = (
        _replay_prefixes(
            config,
            source_run_id,
            source_dataset_path,
            example_by_id,
            records_by_example,
            answer_only_ids,
            tokenizer_cache,
        )
        if config.replay.enabled
        else []
    )
    replay_cases = (
        _replay_cases(config, replay_prefixes, model_names)
        if config.replay.enabled
        else []
    )
    residual_cases = (
        _residual_intervention_cases(
            config,
            source_run_id,
            example_by_id,
            records_by_example,
        )
        if config.residual.enabled
        else []
    )

    output_dir = ensure_dir(config.output_dir / config.name)
    write_jsonl(
        output_dir / GOAL3_REPLAY_PREFIXES_FILENAME,
        [row.model_dump(mode="json") for row in replay_prefixes],
    )
    write_jsonl(
        output_dir / GOAL3_REPLAY_CASES_FILENAME,
        [row.model_dump(mode="json") for row in replay_cases],
    )
    write_jsonl(
        output_dir / GOAL3_RESIDUAL_CASES_FILENAME,
        [row.model_dump(mode="json") for row in residual_cases],
    )

    counts = {
        "source_activation_records": len(records),
        "shared_examples": len(records_by_example),
        "replay_prefixes": len(replay_prefixes),
        "replay_cases": len(replay_cases),
        "residual_intervention_cases": len(residual_cases),
    }
    config_hash = stable_hash(config.model_dump(mode="json"))
    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "name": config.name,
            "created_at": utc_now_iso(),
            "schema_version": config.schema_version,
            "artifact_kind": "goal3_natural_cot_dataset_bundle",
            "config_hash": config_hash,
            "config": config.model_dump(mode="json"),
            "source_goal2_run_id": source_run_id,
            "source_goal2_run_dir": str(config.source_goal2_run_dir),
            "source_dataset_path": str(source_dataset_path),
            "models": model_names,
            "counts": counts,
            "files": {
                "replay_prefixes": GOAL3_REPLAY_PREFIXES_FILENAME,
                "replay_cases": GOAL3_REPLAY_CASES_FILENAME,
                "residual_intervention_cases": GOAL3_RESIDUAL_CASES_FILENAME,
            },
        },
    )
    return manifest_path, counts


def _source_dataset_path(
    config: Goal3DatasetConfig,
    source_manifest: dict[str, Any],
) -> Path:
    """Resolve the source dataset path for a Goal 3 derivation."""
    configured = config.source_dataset_path
    path = configured or Path(str(source_manifest.get("dataset_path", "")))
    if not str(path):
        raise ValueError("source_dataset_path is required when the Goal 2 manifest omits it")
    if not path.exists():
        raise FileNotFoundError(f"source Goal 2 dataset does not exist: {path}")
    return path


def _filtered_activation_records(
    config: Goal3DatasetConfig,
    examples: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return Goal 2 activation records matching Goal 3 dataset filters."""
    allowed_splits = set(config.splits)
    allowed_lengths = set(config.digit_lengths)
    allowed_prompt_modes = {value.value for value in config.prompt_modes}
    allowed_digit_formats = {value.value for value in config.digit_formats}
    allowed_answer_formats = {value.value for value in config.answer_formats}
    allowed_models = set(config.models) if config.models is not None else None
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(config.source_goal2_run_dir / "activations.jsonl"):
        example = examples.get(str(record.get("example_id")))
        if example is None:
            continue
        if example.get("split") not in allowed_splits:
            continue
        if int(example.get("n_digits", -1)) not in allowed_lengths:
            continue
        if example.get("prompt_mode") not in allowed_prompt_modes:
            continue
        if example.get("digit_format") not in allowed_digit_formats:
            continue
        if example.get("answer_format") not in allowed_answer_formats:
            continue
        if allowed_models is not None and record.get("model_name") not in allowed_models:
            continue
        if not config.include_token_limit_hits and _hit_token_limit(record):
            continue
        activation_path = config.source_goal2_run_dir / str(record.get("activation_path", ""))
        if not activation_path.exists():
            continue
        rows.append(record)
    return rows


def _selected_model_names(
    config: Goal3DatasetConfig,
    records: list[dict[str, Any]],
) -> list[str]:
    """Return the configured or inferred source model names."""
    model_names = config.models or sorted({str(row["model_name"]) for row in records})
    if not model_names:
        raise ValueError("no models remain after Goal 3 dataset filtering")
    return list(model_names)


def _records_by_shared_example(
    config: Goal3DatasetConfig,
    records: list[dict[str, Any]],
    model_names: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Group activation records by examples available for the requested models."""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[str(record["example_id"])][str(record["model_name"])] = record
    if not config.require_shared_models:
        return dict(grouped)
    required = set(model_names)
    return {
        example_id: model_records
        for example_id, model_records in grouped.items()
        if required.issubset(model_records)
    }


def _answer_only_example_ids(examples: list[dict[str, Any]]) -> dict[tuple[object, ...], str]:
    """Index native answer-only examples by problem and rendered condition."""
    result: dict[tuple[object, ...], str] = {}
    for row in examples:
        if row.get("prompt_mode") != "answer_only":
            continue
        result[_problem_condition_key(row)] = str(row["id"])
    return result


def _problem_condition_key(example: dict[str, Any]) -> tuple[object, ...]:
    """Return the join key shared by prompt variants of one arithmetic problem."""
    return (
        example.get("problem_id"),
        example.get("split"),
        example.get("digit_format"),
        example.get("answer_format"),
    )


def _replay_prefixes(
    config: Goal3DatasetConfig,
    source_run_id: str,
    source_dataset_path: Path,
    examples: dict[str, dict[str, Any]],
    records_by_example: dict[str, dict[str, dict[str, Any]]],
    answer_only_ids: dict[tuple[object, ...], str],
    tokenizer_cache: dict[tuple[str, str | None], Any],
) -> list[Goal3ReplayPrefix]:
    """Build unique natural-CoT prefixes at recorded Goal 2 locations."""
    rows: list[Goal3ReplayPrefix] = []
    locations = list(dict.fromkeys(config.replay.locations))
    for example_id, model_records in sorted(records_by_example.items()):
        example = examples[example_id]
        answer_only_example_id = answer_only_ids.get(_problem_condition_key(example))
        if ActivationLocation.PROMPT_FINAL in locations:
            rows.append(
                _no_reasoning_prefix(
                    config,
                    source_run_id,
                    source_dataset_path,
                    example,
                    answer_only_example_id,
                )
            )
        for model_name, record in sorted(model_records.items()):
            tokenizer = _record_tokenizer(config, record, tokenizer_cache)
            recorded_output_ids = record.get("output_ids")
            if recorded_output_ids is not None:
                output_ids = [int(token_id) for token_id in recorded_output_ids]
                prefix_token_source = "recorded"
            else:
                output_ids = tokenizer.encode(
                    str(record["decoded_output"]),
                    add_special_tokens=False,
                )
                prefix_token_source = "reconstructed"
            for location_kind in locations:
                if location_kind == ActivationLocation.PROMPT_FINAL:
                    continue
                location = _single_location(record, location_kind)
                if location is None:
                    continue
                output_index = int(location.get("metadata", {}).get("output_token_index", -1))
                if output_index < 0:
                    raise ValueError(
                        f"invalid {location_kind.value} output index for {example_id}/{model_name}"
                    )
                expected_token_id = int(location["token_id"])
                if location_kind == ActivationLocation.COT_END and _is_special_token(
                    tokenizer,
                    location,
                ):
                    replay_output_index = min(output_index - 1, len(output_ids) - 1)
                    if replay_output_index < 0:
                        continue
                else:
                    replay_output_index = _align_output_token_index(
                        output_ids=output_ids,
                        recorded_output_index=output_index,
                        expected_token_id=expected_token_id,
                        max_shift=(
                            0
                            if prefix_token_source == "recorded"
                            else config.replay.max_token_alignment_shift
                        ),
                        context=f"{example_id}/{model_name}/{location_kind.value}",
                    )
                prefix_ids = output_ids[: replay_output_index + 1]
                prefix_text = tokenizer.decode(
                    prefix_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                row_id = stable_hash(
                    {
                        "source_run_id": source_run_id,
                        "example_id": example_id,
                        "source_model_name": model_name,
                        "location_kind": location_kind,
                    },
                    length=16,
                )
                rows.append(
                    Goal3ReplayPrefix(
                        id=row_id,
                        schema_version=config.schema_version,
                        source_goal2_run_id=source_run_id,
                        example_id=example_id,
                        problem_id=str(example["problem_id"]),
                        split=str(example["split"]),
                        n_digits=int(example["n_digits"]),
                        source_model_name=model_name,
                        source_model_id=str(record["model_id"]),
                        location_kind=location_kind,
                        recorded_output_token_end_index=output_index,
                        replay_output_token_end_index=replay_output_index,
                        prefix_token_source=prefix_token_source,
                        prefix_alignment_delta=replay_output_index - output_index,
                        assistant_prefix_token_ids=prefix_ids,
                        assistant_prefix=prefix_text,
                        decoded_output=str(record["decoded_output"]),
                        parsed_answer=record.get("parsed_answer"),
                        expected_output=str(example["expected_output"]),
                        prompt=str(example["prompt"]),
                        messages=list(example["messages"]),
                        answer_only_example_id=answer_only_example_id,
                        metadata={
                            "source_dataset_path": str(source_dataset_path),
                            "source_call_metadata": record.get("call_metadata") or {},
                            "source_location": location,
                            "source_answer_correct": (
                                str(record.get("parsed_answer")) == str(example["expected_output"])
                            ),
                        },
                    )
                )
    return rows


def _is_special_token(tokenizer: Any, location: dict[str, Any]) -> bool:
    """Return whether a saved location points to a tokenizer control token."""
    token_id = int(location["token_id"])
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    token_text = str(location.get("token_text", ""))
    return token_id in special_ids or (token_text.startswith("<|") and token_text.endswith("|>"))


def _align_output_token_index(
    output_ids: list[int],
    recorded_output_index: int,
    expected_token_id: int,
    max_shift: int,
    context: str,
) -> int:
    """Align a legacy retokenized output to a recorded Goal 2 token anchor."""
    candidates = [
        index
        for index, token_id in enumerate(output_ids)
        if token_id == expected_token_id
        and abs(index - recorded_output_index) <= max_shift
    ]
    if not candidates:
        raise ValueError(
            "could not align output to the recorded Goal 2 token at "
            f"{context} within {max_shift} tokens"
        )
    return min(candidates, key=lambda index: abs(index - recorded_output_index))


def _no_reasoning_prefix(
    config: Goal3DatasetConfig,
    source_run_id: str,
    source_dataset_path: Path,
    example: dict[str, Any],
    answer_only_example_id: str | None,
) -> Goal3ReplayPrefix:
    """Build the no-reasoning baseline prefix for one free-CoT example."""
    row_id = stable_hash(
        {
            "source_run_id": source_run_id,
            "example_id": example["id"],
            "source_model_name": None,
            "location_kind": ActivationLocation.PROMPT_FINAL,
        },
        length=16,
    )
    return Goal3ReplayPrefix(
        id=row_id,
        schema_version=config.schema_version,
        source_goal2_run_id=source_run_id,
        example_id=str(example["id"]),
        problem_id=str(example["problem_id"]),
        split=str(example["split"]),
        n_digits=int(example["n_digits"]),
        location_kind=ActivationLocation.PROMPT_FINAL,
        prefix_token_source="empty",
        assistant_prefix_token_ids=[],
        assistant_prefix="",
        decoded_output="",
        expected_output=str(example["expected_output"]),
        prompt=str(example["prompt"]),
        messages=list(example["messages"]),
        answer_only_example_id=answer_only_example_id,
        metadata={"source_dataset_path": str(source_dataset_path)},
    )


def _replay_cases(
    config: Goal3DatasetConfig,
    prefixes: list[Goal3ReplayPrefix],
    model_names: list[str],
) -> list[Goal3ReplayCase]:
    """Cross natural reasoning prefixes with configured receiver models."""
    rows: list[Goal3ReplayCase] = []
    for prefix in prefixes:
        if prefix.source_model_name is None or config.replay.crossed_models:
            receivers = model_names
        else:
            receivers = [prefix.source_model_name]
        for receiver in receivers:
            replay_kind = (
                "no_reasoning"
                if prefix.source_model_name is None
                else "self"
                if receiver == prefix.source_model_name
                else "crossed"
            )
            row_id = stable_hash(
                {
                    "replay_prefix_id": prefix.id,
                    "receiver_model_name": receiver,
                },
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
                    answer_only_example_id=prefix.answer_only_example_id,
                    metadata={
                        "source_answer_correct": prefix.metadata.get("source_answer_correct")
                    },
                )
            )
    return rows


def _residual_intervention_cases(
    config: Goal3DatasetConfig,
    source_run_id: str,
    examples: dict[str, dict[str, Any]],
    records_by_example: dict[str, dict[str, dict[str, Any]]],
) -> list[Goal3ResidualInterventionCase]:
    """Build column-specific residual intervention specifications."""
    rows: list[Goal3ResidualInterventionCase] = []
    for example_id, model_records in sorted(records_by_example.items()):
        example = examples[example_id]
        for model_name, record in sorted(model_records.items()):
            for target in config.residual.targets:
                target_columns = config.residual.target_columns_by_target.get(target, {}).get(
                    int(example["n_digits"]), []
                )
                for target_column in target_columns:
                    counterfactual = _carry_counterfactual(
                        config,
                        example,
                        target,
                        target_column,
                    )
                    if counterfactual is None:
                        continue
                    for location_kind in config.residual.locations:
                        location_entry = _single_location_with_index(record, location_kind)
                        if location_entry is None:
                            continue
                        location_index, location = location_entry
                        layer_indices = [int(value) for value in record.get("layer_indices", [])]
                        for layer_index in config.residual.layers:
                            if layer_index not in layer_indices:
                                continue
                            layer_offset = layer_indices.index(layer_index)
                            direction_id = stable_hash(
                                {
                                    "model_name": model_name,
                                    "target": target,
                                    "target_column_lsd": target_column,
                                    "location_kind": location_kind,
                                    "layer_index": layer_index,
                                    "train_split": config.residual.direction_train_split,
                                },
                                length=16,
                            )
                            row_id = stable_hash(
                                {
                                    "source_run_id": source_run_id,
                                    "example_id": example_id,
                                    "model_name": model_name,
                                    "target": target,
                                    "target_column_lsd": target_column,
                                    "location_kind": location_kind,
                                    "layer_index": layer_index,
                                },
                                length=16,
                            )
                            rows.append(
                                Goal3ResidualInterventionCase(
                                    id=row_id,
                                    schema_version=config.schema_version,
                                    source_goal2_run_id=source_run_id,
                                    example_id=example_id,
                                    problem_id=str(example["problem_id"]),
                                    split=str(example["split"]),
                                    n_digits=int(example["n_digits"]),
                                    model_name=model_name,
                                    model_id=str(record["model_id"]),
                                    target=target,
                                    target_column_lsd=target_column,
                                    affected_output_column_lsd=counterfactual[
                                        "affected_output_column_lsd"
                                    ],
                                    factual_carry=counterfactual["factual_carry"],
                                    counterfactual_carry=counterfactual[
                                        "counterfactual_carry"
                                    ],
                                    factual_output_digit=counterfactual[
                                        "factual_output_digit"
                                    ],
                                    counterfactual_output_digit=counterfactual[
                                        "counterfactual_output_digit"
                                    ],
                                    factual_answer=str(example["answer"]),
                                    counterfactual_answer=counterfactual[
                                        "counterfactual_answer"
                                    ],
                                    unchanged_output_columns_lsd=counterfactual[
                                        "unchanged_output_columns_lsd"
                                    ],
                                    location_kind=location_kind,
                                    activation_location_name=str(location["name"]),
                                    activation_location_index=location_index,
                                    layer_index=layer_index,
                                    activation_layer_index=layer_offset,
                                    activation_path=str(record["activation_path"]),
                                    direction_id=direction_id,
                                    direction_train_split=(
                                        config.residual.direction_train_split
                                    ),
                                    metadata={
                                        "source_location": location,
                                        "source_call_metadata": (
                                            record.get("call_metadata") or {}
                                        ),
                                        "raw_sum": counterfactual["raw_sum"],
                                        "factual_affected_outgoing_carry": counterfactual[
                                            "factual_affected_outgoing_carry"
                                        ],
                                        "equivalent_incoming_carry_column_lsd": (
                                            target_column + 1
                                            if target == ProbeTarget.OUTGOING_CARRY
                                            else target_column
                                        ),
                                    },
                                )
                            )
    return rows


def _carry_counterfactual(
    config: Goal3DatasetConfig,
    example: dict[str, Any],
    target: ProbeTarget,
    target_column: int,
) -> dict[str, Any] | None:
    """Return a local carry counterfactual and its affected output digit."""
    n_digits = int(example["n_digits"])
    if target == ProbeTarget.INCOMING_CARRY:
        affected_column = target_column
        carry_values = example["incoming_carry"]
        if target_column <= 0 or target_column >= n_digits:
            return None
    elif target == ProbeTarget.OUTGOING_CARRY:
        affected_column = target_column + 1
        carry_values = example["outgoing_carry"]
        if target_column < 0 or affected_column >= n_digits:
            return None
    else:
        return None
    carries = [int(value) for value in carry_values]
    raw_sum = [int(value) for value in example["raw_sum"]]
    output_digits = [int(value) for value in example["output_digits_lsd"]]
    if target_column >= len(carries) or affected_column >= len(output_digits):
        return None
    base = int(example.get("base", 10))
    factual_carry = carries[target_column]
    counterfactual_carry = 1 - factual_carry
    factual_total = raw_sum[affected_column] + factual_carry
    counterfactual_total = raw_sum[affected_column] + counterfactual_carry
    if (
        config.residual.require_stable_outgoing_carry
        and factual_total // base != counterfactual_total // base
    ):
        return None
    counterfactual_digits = output_digits.copy()
    counterfactual_digits[affected_column] = counterfactual_total % base
    return {
        "affected_output_column_lsd": affected_column,
        "factual_carry": factual_carry,
        "counterfactual_carry": counterfactual_carry,
        "factual_output_digit": output_digits[affected_column],
        "counterfactual_output_digit": counterfactual_digits[affected_column],
        "counterfactual_answer": digits_lsd_to_str(counterfactual_digits, base),
        "unchanged_output_columns_lsd": [
            index for index in range(len(output_digits)) if index != affected_column
        ],
        "raw_sum": raw_sum[affected_column],
        "factual_affected_outgoing_carry": factual_total // base,
    }


def _record_tokenizer(
    config: Goal3DatasetConfig,
    record: dict[str, Any],
    cache: dict[tuple[str, str | None], Any],
) -> Any:
    """Load and cache the tokenizer needed to recover exact CoT prefixes."""
    tokenizer_id = (
        config.replay.tokenizer_id
        or record.get("tokenizer_id")
        or record.get("model_id")
    )
    if not tokenizer_id:
        raise ValueError("Goal 2 activation record has no tokenizer or model ID")
    revision = config.replay.tokenizer_revision or record.get("tokenizer_revision")
    key = (str(tokenizer_id), str(revision) if revision is not None else None)
    if key not in cache:
        cache[key] = _load_tokenizer(
            key[0],
            key[1],
            local_files_only=config.replay.tokenizer_local_files_only,
        )
    return cache[key]


def _load_tokenizer(
    tokenizer_id: str,
    revision: str | None,
    *,
    local_files_only: bool,
) -> Any:
    """Load one Hugging Face tokenizer for Goal 3 prefix recovery."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=revision,
        local_files_only=local_files_only,
    )


def _single_location(
    record: dict[str, Any],
    location_kind: ActivationLocation,
) -> dict[str, Any] | None:
    """Return the first saved activation location of the requested kind."""
    entry = _single_location_with_index(record, location_kind)
    return entry[1] if entry is not None else None


def _single_location_with_index(
    record: dict[str, Any],
    location_kind: ActivationLocation,
) -> tuple[int, dict[str, Any]] | None:
    """Return the index and first saved activation location of the requested kind."""
    for index, location in enumerate(record.get("locations", [])):
        if location.get("kind") == location_kind.value:
            return index, location
    return None


def _hit_token_limit(record: dict[str, Any]) -> bool:
    """Return whether a Goal 2 activation record exhausted its token budget."""
    return bool((record.get("call_metadata") or {}).get("hit_token_limit"))
