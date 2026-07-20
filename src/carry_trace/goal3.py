"""Goal 3 natural-CoT replay and residual-intervention execution."""

from __future__ import annotations

import gc
import re
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from carry_trace.config import Goal2ProbeConfig, Goal3RunConfig
from carry_trace.enums import ActivationLocation, PromptMode
from carry_trace.goal2_probes import _build_probe_groups, _valid_activation_records
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
from carry_trace.models import (
    AnswerDigitStoppingCriteria,
    HuggingFaceModelRunner,
    _generation_stop_token_ids,
)
from carry_trace.parsing import normalize_output_digits, parse_final_answer
from carry_trace.schemas import Goal3ReplayCase, Goal3ReplayPrefix, Goal3ResidualInterventionCase

REPLAY_SCORES_FILENAME = "replay_scores.jsonl"
REPLAY_GENERATIONS_FILENAME = "replay_generations.jsonl"
REPLAY_EFFECTS_FILENAME = "replay_effects.jsonl"
RESIDUAL_SCORES_FILENAME = "residual_scores.jsonl"
RESIDUAL_GENERATIONS_FILENAME = "residual_generations.jsonl"
DIRECTION_METADATA_FILENAME = "direction_metadata.jsonl"
DIRECTION_TENSORS_FILENAME = "directions.pt"
SUMMARY_METRICS_FILENAME = "summary_metrics.jsonl"
COMPLETION_AUDITS_FILENAME = "completion_audits.jsonl"
RESIDUAL_CONTROL_CONTRASTS_FILENAME = "residual_control_contrasts.jsonl"


@dataclass(frozen=True)
class ResidualExecution:
    """One residual intervention case, scale, and control-direction assignment."""

    case: Goal3ResidualInterventionCase
    intervention_scale: float
    control_direction: str
    intervention_mode: str = "fixed_gap"
    intervention_site: str = "prefix_boundary"

    @property
    def id(self) -> str:
        """Return a stable ID for this intervention execution variant."""
        return stable_hash(
            {
                "case_id": self.case.id,
                "intervention_scale": self.intervention_scale,
                "control_direction": self.control_direction,
                "intervention_mode": self.intervention_mode,
                "intervention_site": self.intervention_site,
            },
            length=16,
        )

    @property
    def layer_index(self) -> int:
        """Expose the case layer for grouped batching."""
        return self.case.layer_index


@dataclass(frozen=True)
class ResidualIntervention:
    """Describe one fixed or projection-clamped residual intervention."""

    carry_direction: Any
    applied_direction: Any
    mode: str
    scale: float
    fixed_projection_delta: float | None = None
    target_projection: float | None = None


def run_goal3(config: Goal3RunConfig) -> Path:
    """Run Goals 3A-3C and write resumable raw and aggregate artifacts."""
    bundle_manifest = read_json(config.dataset_bundle_dir / "manifest.json")
    replay_cases = _load_replay_cases(config)
    replay_prefixes = _load_replay_prefixes(config)
    residual_cases = _load_residual_cases(config)
    residual_cases = _filter_residual_alignment(config, residual_cases, replay_prefixes)
    residual_cases = _limit_residual_problems(config, residual_cases)
    residual_cases = _filter_residual_source_quality(
        config,
        residual_cases,
        replay_prefixes,
    )
    residual_cases = _filter_residual_score_grid(config, residual_cases)
    _validate_model_names(config, replay_cases, residual_cases)
    config_hash = stable_hash(config.model_dump(mode="json"))
    run_dir = _goal3_run_dir(config, config_hash)
    manifest = _initial_manifest(
        config,
        config_hash,
        bundle_manifest,
        replay_cases,
        residual_cases,
        run_dir,
    )
    write_json(run_dir / "manifest.json", manifest)
    write_jsonl(
        run_dir / COMPLETION_AUDITS_FILENAME,
        _completion_audit_rows(replay_prefixes),
    )

    directions, direction_metadata = _fit_residual_directions(
        config,
        bundle_manifest,
        residual_cases,
    )
    _save_directions(run_dir, directions, direction_metadata)

    for model_spec in config.models:
        runner = HuggingFaceModelRunner(model_spec, config.runner, config.replay.generation)
        try:
            model_replay_cases = [
                case for case in replay_cases if case.receiver_model_name == model_spec.name
            ]
            model_residual_cases = [
                case for case in residual_cases if case.model_name == model_spec.name
            ]
            if config.replay.enabled:
                _run_replay_model(config, run_dir, runner, model_replay_cases)
            if config.residual.enabled:
                _run_residual_model(
                    config,
                    run_dir,
                    runner,
                    model_residual_cases,
                    replay_prefixes,
                    directions,
                )
        finally:
            _cleanup_runner(runner)

    _write_analysis_artifacts(run_dir)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now_iso()
    manifest["updated_at"] = manifest["completed_at"]
    manifest["artifact_counts"] = _artifact_counts(run_dir)
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _load_replay_cases(config: Goal3RunConfig) -> list[Goal3ReplayCase]:
    """Load replay cases when replay execution is enabled."""
    if not config.replay.enabled:
        return []
    path = config.dataset_bundle_dir / "replay_cases.jsonl"
    return [Goal3ReplayCase.model_validate(row) for row in read_jsonl(path)]


def _load_replay_prefixes(config: Goal3RunConfig) -> list[Goal3ReplayPrefix]:
    """Load replay prefixes needed to reconstruct residual contexts."""
    if not config.replay.enabled and not config.residual.enabled:
        return []
    path = config.dataset_bundle_dir / "replay_prefixes.jsonl"
    return [Goal3ReplayPrefix.model_validate(row) for row in read_jsonl(path)]


def _load_residual_cases(config: Goal3RunConfig) -> list[Goal3ResidualInterventionCase]:
    """Load residual cases when intervention execution is enabled."""
    if not config.residual.enabled:
        return []
    path = config.dataset_bundle_dir / "residual_intervention_cases.jsonl"
    return [Goal3ResidualInterventionCase.model_validate(row) for row in read_jsonl(path)]


def _filter_residual_alignment(
    config: Goal3RunConfig,
    cases: list[Goal3ResidualInterventionCase],
    prefixes: list[Goal3ReplayPrefix],
) -> list[Goal3ResidualInterventionCase]:
    """Exclude interventions whose replay token differs from the saved activation token."""
    if not config.residual.require_exact_prefix_alignment:
        return cases
    prefix_index = _residual_prefix_index(prefixes)
    filtered = []
    for case in cases:
        source_model = (
            None if case.location_kind == ActivationLocation.PROMPT_FINAL else case.model_name
        )
        prefix = prefix_index.get((case.example_id, source_model, case.location_kind))
        if prefix is None:
            continue
        if prefix.prefix_alignment_delta in (None, 0):
            filtered.append(case)
    return filtered


def _limit_residual_problems(
    config: Goal3RunConfig,
    cases: list[Goal3ResidualInterventionCase],
) -> list[Goal3ResidualInterventionCase]:
    """Select a deterministic problem subset within each digit length for pilot runs."""
    limit = config.residual.max_problems_per_digit_length
    if limit is None:
        return cases
    selected: set[tuple[int, str]] = set()
    problem_ids_by_length: dict[int, set[str]] = defaultdict(set)
    for case in cases:
        problem_ids_by_length[case.n_digits].add(case.problem_id)
    for n_digits, problem_ids in problem_ids_by_length.items():
        ranked = sorted(
            problem_ids,
            key=lambda problem_id: stable_hash(
                {
                    "seed": config.seed,
                    "n_digits": n_digits,
                    "problem_id": problem_id,
                },
                length=16,
            ),
        )
        selected.update((n_digits, problem_id) for problem_id in ranked[:limit])
    return [case for case in cases if (case.n_digits, case.problem_id) in selected]


def _filter_residual_source_quality(
    config: Goal3RunConfig,
    cases: list[Goal3ResidualInterventionCase],
    prefixes: list[Goal3ReplayPrefix],
) -> list[Goal3ResidualInterventionCase]:
    """Keep problems with correct clean source completions from every intervention model."""
    if not config.residual.require_shared_correct_source:
        return cases
    required_models = {case.model_name for case in cases}
    quality_by_problem: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in _completion_audit_rows(prefixes):
        model_name = row["source_model_name"]
        if model_name in required_models:
            quality_by_problem[str(row["problem_id"])][str(model_name)] = bool(
                row["source_answer_correct"] and row["clean_terminal_answer"]
            )
    allowed = {
        problem_id
        for problem_id, quality in quality_by_problem.items()
        if all(quality.get(model_name, False) for model_name in required_models)
    }
    return [case for case in cases if case.problem_id in allowed]


def _filter_residual_score_grid(
    config: Goal3RunConfig,
    cases: list[Goal3ResidualInterventionCase],
) -> list[Goal3ResidualInterventionCase]:
    """Restrict residual scoring to configured locations and layers."""
    locations = (
        set(config.residual.score_locations)
        if config.residual.score_locations is not None
        else None
    )
    layers = set(config.residual.score_layers) if config.residual.score_layers is not None else None
    return [
        case
        for case in cases
        if (locations is None or case.location_kind in locations)
        and (layers is None or case.layer_index in layers)
    ]


def _residual_executions(
    config: Goal3RunConfig,
    cases: list[Goal3ResidualInterventionCase],
) -> list[ResidualExecution]:
    """Cross residual cases with configured scales and control directions."""
    controls = _expanded_control_directions(config)
    return [
        ResidualExecution(
            case,
            scale,
            control,
            config.residual.intervention_mode,
            site,
        )
        for case in cases
        for scale in config.residual.intervention_scales
        for site in config.residual.intervention_sites
        for control in controls
    ]


def _expanded_control_directions(config: Goal3RunConfig) -> list[str]:
    """Expand the configured orthogonal control into deterministic replicates."""
    controls = []
    for control in config.residual.control_directions:
        if control != "orthogonal" or config.residual.orthogonal_control_count == 1:
            controls.append(control)
            continue
        controls.extend(
            f"orthogonal_{index + 1}" for index in range(config.residual.orthogonal_control_count)
        )
    return controls


def _validate_model_names(
    config: Goal3RunConfig,
    replay_cases: list[Goal3ReplayCase],
    residual_cases: list[Goal3ResidualInterventionCase],
) -> None:
    """Require every receiver and intervention model to have a model specification."""
    configured = {model.name for model in config.models}
    required = {case.receiver_model_name for case in replay_cases} | {
        case.model_name for case in residual_cases
    }
    missing = sorted(required - configured)
    if missing:
        raise ValueError(f"Goal 3 model specs are missing: {', '.join(missing)}")


def _goal3_run_dir(config: Goal3RunConfig, config_hash: str) -> Path:
    """Return a resumable run directory or create a timestamped one."""
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
    config: Goal3RunConfig,
    config_hash: str,
    bundle_manifest: dict[str, Any],
    replay_cases: list[Goal3ReplayCase],
    residual_cases: list[Goal3ResidualInterventionCase],
    run_dir: Path,
) -> dict[str, Any]:
    """Build the running Goal 3 manifest."""
    existing = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    residual_executions = _residual_executions(config, residual_cases)
    residual_decode_executions = [
        execution
        for execution in residual_executions
        if _decode_residual_execution(config, execution)
    ]
    return {
        "run_id": existing.get("run_id", run_dir.name),
        "created_at": existing.get("created_at", utc_now_iso()),
        "updated_at": utc_now_iso(),
        "status": "running",
        "artifact_kind": "goal3_natural_cot_causal_coupling",
        "config_hash": config_hash,
        "config": config.model_dump(mode="json"),
        "dataset_bundle_dir": str(config.dataset_bundle_dir),
        "dataset_bundle_config_hash": bundle_manifest.get("config_hash"),
        "expected_replay_scores": len(replay_cases),
        "expected_replay_generations": sum(
            case.location_kind in set(config.replay.decode_locations) for case in replay_cases
        ),
        "selected_residual_cases": len(residual_cases),
        "selected_problem_ids_by_digit_length": {
            str(n_digits): sorted(
                {case.problem_id for case in residual_cases if case.n_digits == n_digits}
            )
            for n_digits in sorted({case.n_digits for case in residual_cases})
        },
        "expected_residual_scores": len(residual_executions),
        "expected_residual_generations": len(residual_decode_executions),
        "expected_residual_baseline_decodes": len(
            {_residual_baseline_key(execution.case) for execution in residual_decode_executions}
        ),
        "expected_residual_intervention_decodes": len(residual_decode_executions),
    }


def _decode_residual_execution(
    config: Goal3RunConfig,
    execution: ResidualExecution,
) -> bool:
    """Return whether an execution belongs to the focused decoding subset."""
    return (
        execution.control_direction == "carry"
        and execution.intervention_scale == config.residual.decode_intervention_scale
        and execution.case.location_kind in set(config.residual.decode_locations)
        and execution.case.layer_index in set(config.residual.decode_layers)
    )


def _fit_residual_directions(
    config: Goal3RunConfig,
    bundle_manifest: dict[str, Any],
    cases: list[Goal3ResidualInterventionCase],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit signed residual directions from the configured Goal 2 training split."""
    if not config.residual.enabled or not cases:
        return {}, []
    goal2_run_dir = Path(str(bundle_manifest["source_goal2_run_dir"]))
    dataset_config = bundle_manifest.get("config") or {}
    examples = {
        str(row["id"]): row
        for row in read_jsonl(goal2_run_dir / "dataset.jsonl")
        if _matches_direction_population(row, dataset_config)
    }
    direction_train_split = cases[0].direction_train_split
    targets = sorted({case.target for case in cases}, key=lambda value: value.value)
    probe_config = Goal2ProbeConfig(
        name="goal3_directions",
        goal2_run_dir=goal2_run_dir,
        train_split=direction_train_split,
        test_split="__goal3_unused__",
        prompt_modes=[PromptMode.FREE_COT],
        targets=targets,
        min_train_examples=config.residual.min_train_examples,
        max_iter=config.residual.max_iter,
        c=config.residual.c,
        random_state=config.seed,
    )
    records = _valid_activation_records(probe_config, examples)
    groups = _build_probe_groups(probe_config, examples, records)
    required = {
        (
            case.model_name,
            case.target.value,
            case.location_kind.value,
            str(case.layer_index),
            case.target_column_lsd,
        )
        for case in cases
    }
    directions: dict[str, Any] = {}
    metadata: list[dict[str, Any]] = []
    for key in sorted(required):
        rows = groups.get(key, {}).get("train", [])
        direction_id = _direction_id_from_key(key, direction_train_split)
        direction, row = _fit_one_direction(config, key, direction_id, rows)
        direction_data = {
            "carry": direction,
            "class_zero_projection_mean": row["class_zero_projection_mean"],
            "class_one_projection_mean": row["class_one_projection_mean"],
            "class_mean_gap": row["class_mean_gap"],
        }
        orthogonal_dot_products = {}
        for control_name in _expanded_control_directions(config):
            if not control_name.startswith("orthogonal"):
                continue
            orthogonal = _orthogonal_direction(
                direction,
                f"{direction_id}:{control_name}",
                config.seed,
            )
            direction_data[control_name] = orthogonal
            orthogonal_dot_products[control_name] = float(direction @ orthogonal)
        directions[direction_id] = direction_data
        row["orthogonal_dot_products"] = orthogonal_dot_products
        if "orthogonal" in orthogonal_dot_products:
            row["orthogonal_dot_product"] = orthogonal_dot_products["orthogonal"]
        metadata.append(row)
    return directions, metadata


def _matches_direction_population(row: dict[str, Any], dataset_config: dict[str, Any]) -> bool:
    """Return whether a Goal 2 example belongs to the Goal 3 direction population."""
    return (
        row.get("prompt_mode") in set(dataset_config.get("prompt_modes", ["free_cot"]))
        and int(row.get("n_digits", -1)) in set(dataset_config.get("digit_lengths", []))
        and row.get("digit_format") in set(dataset_config.get("digit_formats", ["standard"]))
        and row.get("answer_format") in set(dataset_config.get("answer_formats", ["standard"]))
    )


def _direction_id_from_key(
    key: tuple[str, str, str, str, int | None],
    train_split: str,
) -> str:
    """Return the dataset-compatible direction ID for a probe group key."""
    model_name, target, location_kind, layer_index, target_column = key
    return stable_hash(
        {
            "model_name": model_name,
            "target": target,
            "target_column_lsd": target_column,
            "location_kind": location_kind,
            "layer_index": int(layer_index),
            "train_split": train_split,
        },
        length=16,
    )


def _fit_one_direction(
    config: Goal3RunConfig,
    key: tuple[str, str, str, str, int | None],
    direction_id: str,
    rows: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    """Fit one standardized logistic direction and projected class-mean gap."""
    if len(rows) < config.residual.min_train_examples:
        raise ValueError(f"direction {direction_id} has only {len(rows)} training examples")
    x = np.stack([row["x"] for row in rows]).astype(np.float32)
    y = np.array([row["y"] for row in rows], dtype=np.int64)
    if set(y.tolist()) != {0, 1}:
        raise ValueError(f"direction {direction_id} requires both carry classes")
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(
        C=config.residual.c,
        class_weight="balanced",
        max_iter=config.residual.max_iter,
        random_state=config.seed,
    ).fit(scaler.transform(x), y)
    raw_direction = classifier.coef_[0].astype(np.float32) / scaler.scale_.astype(np.float32)
    norm = float(np.linalg.norm(raw_direction))
    if norm == 0.0:
        raise ValueError(f"direction {direction_id} has zero norm")
    unit_direction = raw_direction / norm
    projections = x @ unit_direction
    class_zero_mean = float(projections[y == 0].mean())
    class_one_mean = float(projections[y == 1].mean())
    class_mean_gap = class_one_mean - class_zero_mean
    if class_mean_gap <= 0.0:
        raise ValueError(f"direction {direction_id} has non-positive signed class gap")
    import torch

    tensor = torch.from_numpy(unit_direction.copy())
    model_name, target, location_kind, layer_index, target_column = key
    return tensor, {
        "direction_id": direction_id,
        "model_name": model_name,
        "target": target,
        "target_column_lsd": target_column,
        "location_kind": location_kind,
        "layer_index": int(layer_index),
        "train_split": rows[0]["metadata"]["split"],
        "train_examples": len(rows),
        "train_positive_rate": float(y.mean()),
        "direction_norm_before_normalization": norm,
        "class_zero_projection_mean": class_zero_mean,
        "class_one_projection_mean": class_one_mean,
        "class_mean_gap": class_mean_gap,
    }


def _orthogonal_direction(direction: Any, direction_id: str, seed: int) -> Any:
    """Return a deterministic unit vector orthogonal to a fitted carry direction."""
    import torch

    random_seed = int(
        stable_hash({"direction_id": direction_id, "seed": seed}, length=16),
        16,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    candidate = torch.randn(direction.shape, generator=generator, dtype=direction.dtype)
    candidate = candidate - torch.dot(candidate, direction) * direction
    norm = torch.linalg.vector_norm(candidate)
    if float(norm) == 0.0:
        raise ValueError(f"could not construct orthogonal control for {direction_id}")
    return candidate / norm


def _save_directions(
    run_dir: Path,
    directions: dict[str, Any],
    metadata: list[dict[str, Any]],
) -> None:
    """Write fitted direction vectors and metadata."""
    import torch

    torch.save(directions, run_dir / DIRECTION_TENSORS_FILENAME)
    write_jsonl(run_dir / DIRECTION_METADATA_FILENAME, metadata)


def _run_replay_model(
    config: Goal3RunConfig,
    run_dir: Path,
    runner: HuggingFaceModelRunner,
    cases: list[Goal3ReplayCase],
) -> None:
    """Score all replay cases and decode the configured endpoint subset."""
    scores_path = run_dir / REPLAY_SCORES_FILENAME
    completed_scores = _completed_ids(scores_path)
    pending_scores = [case for case in cases if case.id not in completed_scores]
    score_completed = len(cases) - len(pending_scores)
    score_started = time.perf_counter()
    for batch in _batches(pending_scores, config.runner.batch_size):
        rows = _score_replay_batch(config, runner, batch)
        append_jsonl(scores_path, rows)
        score_completed += len(rows)
        _print_progress(
            "goal3-replay-score",
            runner.model_spec.name,
            score_completed,
            len(cases),
            score_started,
        )

    generations_path = run_dir / REPLAY_GENERATIONS_FILENAME
    completed_generations = _completed_ids(generations_path)
    decode_locations = set(config.replay.decode_locations)
    pending_generations = [
        case
        for case in cases
        if case.location_kind in decode_locations and case.id not in completed_generations
    ]
    generation_total = sum(case.location_kind in decode_locations for case in cases)
    generation_completed = generation_total - len(pending_generations)
    generation_started = time.perf_counter()
    for batch in _batches(pending_generations, config.runner.batch_size):
        rows = _generate_replay_batch(config, runner, batch)
        append_jsonl(generations_path, rows)
        generation_completed += len(rows)
        _print_progress(
            "goal3-replay-decode",
            runner.model_spec.name,
            generation_completed,
            generation_total,
            generation_started,
        )


def _score_replay_batch(
    config: Goal3RunConfig,
    runner: HuggingFaceModelRunner,
    cases: list[Goal3ReplayCase],
) -> list[dict[str, Any]]:
    """Teacher-force the expected answer for a replay batch."""
    sequences = []
    for case in cases:
        context = _replay_context_ids(
            runner.tokenizer,
            case.messages,
            case.assistant_prefix_token_ids,
        )
        at_answer_boundary = case.location_kind == ActivationLocation.COT_END
        sequences.append(
            _scoring_sequence(
                runner.tokenizer,
                context,
                "" if at_answer_boundary else config.replay.answer_cue,
                case.expected_output,
                answer_separator="" if at_answer_boundary else " ",
            )
        )
    scores = _score_sequences(runner.model, sequences)
    return [
        {
            **_replay_identity(case),
            "expected_output": case.expected_output,
            "expected_answer_logprob": score,
            "expected_answer_token_count": len(sequence[1]),
            "expected_answer_mean_logprob": score / len(sequence[1]),
        }
        for case, sequence, score in zip(cases, sequences, scores, strict=True)
    ]


def _generate_replay_batch(
    config: Goal3RunConfig,
    runner: HuggingFaceModelRunner,
    cases: list[Goal3ReplayCase],
) -> list[dict[str, Any]]:
    """Generate deterministic answers from a reduced replay subset."""
    contexts = [
        _generation_context(
            runner.tokenizer,
            _replay_context_ids(runner.tokenizer, case.messages, case.assistant_prefix_token_ids),
            ("" if case.location_kind == ActivationLocation.COT_END else config.replay.answer_cue),
        )
        for case in cases
    ]
    outputs = _generate_contexts(
        runner,
        contexts,
        config.replay.generation,
        expected_output_lengths=[len(case.expected_output) for case in cases],
    )
    rows = []
    for case, output in zip(cases, outputs, strict=True):
        parsed = parse_final_answer(output, base=10, answer_format="standard")
        rows.append(
            {
                **_replay_identity(case),
                "expected_output": case.expected_output,
                "decoded_output": output,
                "parsed_answer": parsed,
                "exact_match": parsed == case.expected_output,
            }
        )
    return rows


def _run_residual_model(
    config: Goal3RunConfig,
    run_dir: Path,
    runner: HuggingFaceModelRunner,
    cases: list[Goal3ResidualInterventionCase],
    prefixes: list[Goal3ReplayPrefix],
    directions: dict[str, Any],
) -> None:
    """Score all residual cases and decode the configured focused subset."""
    executions = _residual_executions(config, cases)
    prefix_index = _residual_prefix_index(prefixes)
    scores_path = run_dir / RESIDUAL_SCORES_FILENAME
    completed_scores = _completed_ids(scores_path)
    score_baseline_cache = _residual_score_baseline_cache(scores_path)
    pending_scores = [execution for execution in executions if execution.id not in completed_scores]
    score_completed = len(executions) - len(pending_scores)
    score_started = time.perf_counter()
    for batch in _grouped_batches(
        pending_scores,
        config.runner.batch_size,
        key=lambda case: case.layer_index,
    ):
        rows = _score_residual_batch(
            config,
            runner,
            batch,
            prefix_index,
            directions,
            score_baseline_cache,
        )
        append_jsonl(scores_path, rows)
        score_completed += len(rows)
        _print_progress(
            "goal3-residual-score",
            runner.model_spec.name,
            score_completed,
            len(executions),
            score_started,
        )

    generations_path = run_dir / RESIDUAL_GENERATIONS_FILENAME
    completed_generations = _completed_ids(generations_path)
    baseline_cache = _residual_baseline_cache(generations_path)
    pending_generations = [
        execution
        for execution in executions
        if _decode_residual_execution(config, execution)
        and execution.id not in completed_generations
    ]
    generation_total = sum(
        _decode_residual_execution(config, execution) for execution in executions
    )
    generation_completed = generation_total - len(pending_generations)
    generation_started = time.perf_counter()
    for batch in _grouped_batches(
        pending_generations,
        config.runner.batch_size,
        key=lambda case: case.layer_index,
    ):
        rows = _generate_residual_batch(
            config,
            runner,
            batch,
            prefix_index,
            directions,
            baseline_cache,
        )
        append_jsonl(generations_path, rows)
        generation_completed += len(rows)
        _print_progress(
            "goal3-residual-decode",
            runner.model_spec.name,
            generation_completed,
            generation_total,
            generation_started,
        )


def _score_residual_batch(
    config: Goal3RunConfig,
    runner: HuggingFaceModelRunner,
    executions: list[ResidualExecution],
    prefix_index: dict[tuple[str, str | None, ActivationLocation], Goal3ReplayPrefix],
    directions: dict[str, Any],
    baseline_cache: dict[tuple[str, str, str, str], float],
) -> list[dict[str, Any]]:
    """Score factual and counterfactual answers before and after intervention."""
    cases = [execution.case for execution in executions]
    contexts = [_residual_context_ids(runner.tokenizer, case, prefix_index) for case in cases]
    missing_baselines: dict[
        tuple[str, str, str, str],
        tuple[list[int], list[int]],
    ] = {}
    for case, context in zip(cases, contexts, strict=True):
        for answer in [case.factual_answer, case.counterfactual_answer]:
            key = _residual_score_baseline_key(case, answer)
            if key not in baseline_cache:
                missing_baselines[key] = _scoring_sequence(
                    runner.tokenizer,
                    context,
                    config.replay.answer_cue,
                    answer,
                )
    if missing_baselines:
        missing_items = list(missing_baselines.items())
        scores = _score_sequences(runner.model, [item[1] for item in missing_items])
        for (key, _), score in zip(missing_items, scores, strict=True):
            baseline_cache[key] = score
    intervention_sequences = []
    for case, context in zip(cases, contexts, strict=True):
        intervention_sequences.extend(
            [
                _scoring_sequence(
                    runner.tokenizer,
                    context,
                    config.replay.answer_cue,
                    case.factual_answer,
                ),
                _scoring_sequence(
                    runner.tokenizer,
                    context,
                    config.replay.answer_cue,
                    case.counterfactual_answer,
                ),
            ]
        )
    interventions = []
    positions = []
    calibration_positions = []
    for execution, context in zip(executions, contexts, strict=True):
        intervention = _execution_intervention(execution, directions)
        position = _intervention_position(
            runner.tokenizer,
            context,
            config.replay.answer_cue,
            execution.intervention_site,
        )
        interventions.extend([intervention, intervention])
        positions.extend([position, position])
        calibration_positions.extend([len(context) - 1, len(context) - 1])
    diagnostics: list[dict[str, Any] | None] = [None] * len(intervention_sequences)
    intervention_scores = _score_sequences(
        runner.model,
        intervention_sequences,
        layer_index=cases[0].layer_index,
        interventions=interventions,
        intervention_positions=positions,
        intervention_calibration_positions=calibration_positions,
        intervention_diagnostics=diagnostics,
    )
    rows = []
    for index, execution in enumerate(executions):
        case = execution.case
        baseline_factual = baseline_cache[_residual_score_baseline_key(case, case.factual_answer)]
        baseline_counterfactual = baseline_cache[
            _residual_score_baseline_key(case, case.counterfactual_answer)
        ]
        intervention_factual, intervention_counterfactual = intervention_scores[
            index * 2 : index * 2 + 2
        ]
        baseline_preference = baseline_counterfactual - baseline_factual
        intervention_preference = intervention_counterfactual - intervention_factual
        diagnostic = diagnostics[index * 2]
        if diagnostic is None:
            raise RuntimeError(f"missing intervention diagnostics for {execution.id}")
        rows.append(
            {
                **_residual_identity(execution),
                "factual_answer": case.factual_answer,
                "counterfactual_answer": case.counterfactual_answer,
                "baseline_factual_logprob": baseline_factual,
                "baseline_counterfactual_logprob": baseline_counterfactual,
                "intervention_factual_logprob": intervention_factual,
                "intervention_counterfactual_logprob": intervention_counterfactual,
                "baseline_counterfactual_preference": baseline_preference,
                "intervention_counterfactual_preference": intervention_preference,
                "counterfactual_preference_shift": intervention_preference - baseline_preference,
                "counterfactual_shift_positive": (
                    intervention_preference - baseline_preference > 0.0
                ),
                "applied_shift_norm": diagnostic["requested_shift_norm"],
                **diagnostic,
            }
        )
    return rows


def _residual_score_baseline_key(
    case: Goal3ResidualInterventionCase,
    answer: str,
) -> tuple[str, str, str, str]:
    """Return the context-and-answer key shared by baseline likelihood scores."""
    return (case.model_name, case.example_id, case.location_kind.value, answer)


def _residual_score_baseline_cache(
    path: Path,
) -> dict[tuple[str, str, str, str], float]:
    """Recover baseline answer likelihoods from completed residual score rows."""
    if not path.exists():
        return {}
    cache = {}
    for row in read_jsonl(path):
        identity = (
            str(row["model_name"]),
            str(row["example_id"]),
            str(row["location_kind"]),
        )
        cache[(*identity, str(row["factual_answer"]))] = float(row["baseline_factual_logprob"])
        cache[(*identity, str(row["counterfactual_answer"]))] = float(
            row["baseline_counterfactual_logprob"]
        )
    return cache


def _generate_residual_batch(
    config: Goal3RunConfig,
    runner: HuggingFaceModelRunner,
    executions: list[ResidualExecution],
    prefix_index: dict[tuple[str, str | None, ActivationLocation], Goal3ReplayPrefix],
    directions: dict[str, Any],
    baseline_cache: dict[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    """Generate baseline and intervened answers for focused residual cases."""
    cases = [execution.case for execution in executions]
    contexts = [
        _generation_context(
            runner.tokenizer,
            _residual_context_ids(runner.tokenizer, case, prefix_index),
            config.replay.answer_cue,
        )
        for case in cases
    ]
    missing_by_key: dict[tuple[str, str, str], tuple[list[int], int]] = {}
    for case, context in zip(cases, contexts, strict=True):
        key = _residual_baseline_key(case)
        if key not in baseline_cache:
            missing_by_key[key] = (context, len(case.factual_answer))
    if missing_by_key:
        missing_items = list(missing_by_key.items())
        generated = _generate_contexts(
            runner,
            [item[1][0] for item in missing_items],
            config.residual.generation,
            expected_output_lengths=[item[1][1] for item in missing_items],
        )
        for (key, _), output in zip(missing_items, generated, strict=True):
            baseline_cache[key] = output
    baseline_outputs = [baseline_cache[_residual_baseline_key(case)] for case in cases]
    shifts = [_execution_shift(execution, directions) for execution in executions]
    intervention_positions = [
        len(context) - len(_encode(runner.tokenizer, config.replay.answer_cue)) - 1
        for context in contexts
    ]
    intervention_outputs = _generate_contexts(
        runner,
        contexts,
        config.residual.generation,
        layer_index=cases[0].layer_index,
        shifts=shifts,
        intervention_positions=intervention_positions,
        expected_output_lengths=[len(case.counterfactual_answer) for case in cases],
    )
    rows = []
    for execution, baseline_output, intervention_output in zip(
        executions,
        baseline_outputs,
        intervention_outputs,
        strict=True,
    ):
        case = execution.case
        baseline_parsed = parse_final_answer(baseline_output, base=10, answer_format="standard")
        intervention_parsed = parse_final_answer(
            intervention_output,
            base=10,
            answer_format="standard",
        )
        rows.append(
            {
                **_residual_identity(execution),
                "factual_answer": case.factual_answer,
                "counterfactual_answer": case.counterfactual_answer,
                "baseline_decoded_output": baseline_output,
                "baseline_parsed_answer": baseline_parsed,
                "baseline_factual_exact_match": baseline_parsed == case.factual_answer,
                "intervention_decoded_output": intervention_output,
                "intervention_parsed_answer": intervention_parsed,
                "intervention_counterfactual_exact_match": (
                    intervention_parsed == case.counterfactual_answer
                ),
                "intervention_factual_exact_match": intervention_parsed == case.factual_answer,
                "off_target_digits_preserved": _off_target_digits_preserved(
                    intervention_parsed,
                    case.factual_answer,
                    case.unchanged_output_columns_lsd,
                ),
            }
        )
    return rows


def _residual_baseline_key(case: Goal3ResidualInterventionCase) -> tuple[str, str, str]:
    """Return the context key shared by equivalent residual baselines."""
    return (case.model_name, case.example_id, case.location_kind.value)


def _residual_baseline_cache(path: Path) -> dict[tuple[str, str, str], str]:
    """Recover generated residual baselines from completed case rows."""
    if not path.exists():
        return {}
    return {
        (str(row["model_name"]), str(row["example_id"]), str(row["location_kind"])): str(
            row["baseline_decoded_output"]
        )
        for row in read_jsonl(path)
    }


def _execution_shift(
    execution: ResidualExecution,
    directions: dict[str, Any],
) -> Any:
    """Return the signed, scaled shift for one intervention execution variant."""
    case = execution.case
    direction = directions.get(case.direction_id)
    if direction is None:
        raise KeyError(f"missing fitted direction {case.direction_id}")
    gap = float(direction["class_mean_gap"])
    sign = case.counterfactual_carry - case.factual_carry
    vector = direction[execution.control_direction]
    return vector * (sign * gap * execution.intervention_scale)


def _execution_intervention(
    execution: ResidualExecution,
    directions: dict[str, Any],
) -> ResidualIntervention:
    """Build the fixed or projection-clamped intervention for one execution."""
    case = execution.case
    direction = directions.get(case.direction_id)
    if direction is None:
        raise KeyError(f"missing fitted direction {case.direction_id}")
    if execution.control_direction not in direction:
        raise KeyError(f"missing {execution.control_direction} control for {case.direction_id}")
    fixed_delta = None
    target_projection = None
    if execution.intervention_mode == "fixed_gap":
        sign = case.counterfactual_carry - case.factual_carry
        fixed_delta = sign * float(direction["class_mean_gap"])
    elif execution.intervention_mode == "projection_clamp":
        target_projection = float(
            direction[f"class_{'one' if case.counterfactual_carry else 'zero'}_projection_mean"]
        )
    else:
        raise ValueError(f"unsupported intervention mode {execution.intervention_mode}")
    return ResidualIntervention(
        carry_direction=direction["carry"],
        applied_direction=direction[execution.control_direction],
        mode=execution.intervention_mode,
        scale=execution.intervention_scale,
        fixed_projection_delta=fixed_delta,
        target_projection=target_projection,
    )


def _intervention_position(
    tokenizer: Any,
    context: list[int],
    answer_cue: str,
    site: str,
) -> int:
    """Return the context-relative token index receiving an intervention."""
    if site == "prefix_boundary":
        return len(context) - 1
    if site == "answer_cue":
        cue_ids = _encode(tokenizer, answer_cue)
        if not cue_ids:
            raise ValueError("answer_cue intervention requires at least one cue token")
        return len(context) + len(cue_ids) - 1
    raise ValueError(f"unsupported intervention site {site}")


def _residual_prefix_index(
    prefixes: list[Goal3ReplayPrefix],
) -> dict[tuple[str, str | None, ActivationLocation], Goal3ReplayPrefix]:
    """Index self-generated prefixes and shared no-reasoning prefixes."""
    return {
        (prefix.example_id, prefix.source_model_name, prefix.location_kind): prefix
        for prefix in prefixes
    }


def _residual_context_ids(
    tokenizer: Any,
    case: Goal3ResidualInterventionCase,
    prefixes: dict[tuple[str, str | None, ActivationLocation], Goal3ReplayPrefix],
) -> list[int]:
    """Return the original model's natural prefix context for one residual case."""
    source_model = (
        None if case.location_kind == ActivationLocation.PROMPT_FINAL else case.model_name
    )
    key = (case.example_id, source_model, case.location_kind)
    prefix = prefixes.get(key)
    if prefix is None:
        raise KeyError(f"missing replay prefix for residual case {case.id}")
    return _replay_context_ids(tokenizer, prefix.messages, prefix.assistant_prefix_token_ids)


def _replay_context_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    assistant_prefix_ids: list[int],
) -> list[int]:
    """Render the user turn and append an assistant reasoning prefix."""
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    else:
        prompt_ids = _encode(tokenizer, messages[-1]["content"], add_special_tokens=True)
    return _flat_token_ids(prompt_ids) + list(assistant_prefix_ids)


def _flat_token_ids(values: Any) -> list[int]:
    """Normalize tokenizer output to one flat token-ID sequence."""
    if isinstance(values, Mapping):
        values = values["input_ids"]
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    values = list(values)
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("expected token IDs for exactly one replay context")
        values = list(values[0])
    return [int(value) for value in values]


def _scoring_sequence(
    tokenizer: Any,
    context_ids: list[int],
    answer_cue: str,
    answer: str,
    *,
    answer_separator: str = " ",
) -> tuple[list[int], list[int]]:
    """Build one teacher-forced sequence and answer-token positions."""
    cue_ids = _encode(tokenizer, answer_cue)
    answer_ids = _encode(tokenizer, f"{answer_separator}{answer}")
    sequence = context_ids + cue_ids + answer_ids
    answer_positions = list(range(len(context_ids) + len(cue_ids), len(sequence)))
    return sequence, answer_positions


def _generation_context(tokenizer: Any, context_ids: list[int], answer_cue: str) -> list[int]:
    """Append the fixed answer cue to a generation context."""
    return context_ids + _encode(tokenizer, answer_cue)


def _encode(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    """Encode text as a flat list of token IDs."""
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=add_special_tokens)]


def _score_sequences(
    model: Any,
    sequences: list[tuple[list[int], list[int]]],
    *,
    layer_index: int | None = None,
    shifts: list[Any] | None = None,
    interventions: list[ResidualIntervention] | None = None,
    intervention_positions: list[int] | None = None,
    intervention_calibration_positions: list[int] | None = None,
    intervention_diagnostics: list[dict[str, Any] | None] | None = None,
) -> list[float]:
    """Return summed answer-token log probabilities without materializing full logits."""
    import torch

    input_ids, attention_mask, padding = _left_pad([sequence[0] for sequence in sequences], model)
    padded_score_positions = [
        [position + pad for position in score_positions]
        for (_, score_positions), pad in zip(sequences, padding, strict=True)
    ]
    padded_intervention_positions = (
        [position + pad for position, pad in zip(intervention_positions, padding, strict=True)]
        if intervention_positions is not None
        else None
    )
    padded_calibration_positions = (
        [
            position + pad
            for position, pad in zip(
                intervention_calibration_positions,
                padding,
                strict=True,
            )
        ]
        if intervention_calibration_positions is not None
        else padded_intervention_positions
    )
    with (
        torch.no_grad(),
        _intervention_hook(
            model,
            layer_index,
            shifts,
            padded_intervention_positions,
            interventions=interventions,
            calibration_positions=padded_calibration_positions,
            diagnostics=intervention_diagnostics,
        ),
    ):
        outputs = _backbone(model)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        predictor_states = []
        target_ids = []
        owners = []
        for row_index, positions in enumerate(padded_score_positions):
            for position in positions:
                predictor_states.append(hidden[row_index, position - 1])
                target_ids.append(input_ids[row_index, position])
                owners.append(row_index)
        logits = model.lm_head(torch.stack(predictor_states))
        token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
            1,
            torch.stack(target_ids).unsqueeze(1),
        )[:, 0]
    scores = [0.0] * len(sequences)
    for owner, value in zip(owners, token_logprobs.detach().cpu().tolist(), strict=True):
        scores[owner] += float(value)
    return scores


def _generate_contexts(
    runner: HuggingFaceModelRunner,
    contexts: list[list[int]],
    generation: Any,
    *,
    layer_index: int | None = None,
    shifts: list[Any] | None = None,
    intervention_positions: list[int] | None = None,
    expected_output_lengths: list[int] | None = None,
) -> list[str]:
    """Generate deterministic short continuations with an optional residual intervention."""
    import torch

    input_ids, attention_mask, padding = _left_pad(contexts, runner.model)
    padded_positions = (
        [position + pad for position, pad in zip(intervention_positions, padding, strict=True)]
        if intervention_positions is not None
        else None
    )
    stop_token_ids = _generation_stop_token_ids(runner.tokenizer, runner.model)
    kwargs = {
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": generation.do_sample,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "pad_token_id": runner.tokenizer.pad_token_id,
        "eos_token_id": stop_token_ids or runner.tokenizer.eos_token_id,
    }
    if not generation.do_sample:
        kwargs.pop("temperature")
        kwargs.pop("top_p")
    if expected_output_lengths is not None:
        from transformers import StoppingCriteriaList

        kwargs["stopping_criteria"] = StoppingCriteriaList(
            [
                AnswerDigitStoppingCriteria(
                    tokenizer=runner.tokenizer,
                    input_width=int(input_ids.shape[1]),
                    expected_output_lengths=expected_output_lengths,
                )
            ]
        )
    with (
        torch.no_grad(),
        _intervention_hook(
            runner.model,
            layer_index,
            shifts,
            padded_positions,
        ),
    ):
        outputs = runner.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
    input_width = int(input_ids.shape[1])
    return [
        runner.tokenizer.decode(row[input_width:], skip_special_tokens=True)
        for row in outputs.detach().cpu().tolist()
    ]


def _left_pad(sequences: list[list[int]], model: Any) -> tuple[Any, Any, list[int]]:
    """Left-pad token sequences on the model's input device."""
    import torch

    width = max(len(sequence) for sequence in sequences)
    pad_token_id = int(getattr(model.config, "pad_token_id", 0) or 0)
    padding = [width - len(sequence) for sequence in sequences]
    rows = [
        [pad_token_id] * pad + sequence for sequence, pad in zip(sequences, padding, strict=True)
    ]
    masks = [
        [0] * pad + [1] * len(sequence) for sequence, pad in zip(sequences, padding, strict=True)
    ]
    device = _model_device(model)
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        padding,
    )


@contextmanager
def _intervention_hook(
    model: Any,
    layer_index: int | None,
    shifts: list[Any] | None,
    positions: list[int] | None,
    *,
    interventions: list[ResidualIntervention] | None = None,
    calibration_positions: list[int] | None = None,
    diagnostics: list[dict[str, Any] | None] | None = None,
) -> Iterator[None]:
    """Temporarily apply fixed or clamped changes to one decoder-layer output."""
    if layer_index is None:
        yield
        return
    if positions is None or (shifts is None) == (interventions is None):
        raise ValueError("residual interventions require shifts and positions")
    intervention_count = len(shifts) if shifts is not None else len(interventions or [])
    if diagnostics is not None and len(diagnostics) != intervention_count:
        raise ValueError("intervention diagnostics must match the intervention batch")
    if calibration_positions is not None and len(calibration_positions) != intervention_count:
        raise ValueError("calibration positions must match the intervention batch")
    layer = _decoder_layers(model)[layer_index]

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        """Apply shifts only during the initial pass containing target positions."""
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != intervention_count:
            return output
        applicable = [
            index for index, position in enumerate(positions) if position < hidden.shape[1]
        ]
        if not applicable:
            return output
        updated = hidden.clone()
        for row_index in applicable:
            position = positions[row_index]
            before = hidden[row_index, position].float()
            if interventions is None:
                requested_shift = shifts[row_index].to(  # type: ignore[index]
                    device=updated.device,
                    dtype=updated.dtype,
                )
                updated[row_index, position] = hidden[row_index, position] + requested_shift
                continue
            intervention = interventions[row_index]
            carry_direction = intervention.carry_direction.to(
                device=updated.device,
                dtype=before.dtype,
            )
            applied_direction = intervention.applied_direction.to(
                device=updated.device,
                dtype=before.dtype,
            )
            calibration_position = (
                calibration_positions[row_index] if calibration_positions is not None else position
            )
            calibration_state = hidden[row_index, calibration_position].float()
            carry_before = float(calibration_state @ carry_direction)
            if intervention.mode == "fixed_gap":
                if intervention.fixed_projection_delta is None:
                    raise ValueError("fixed intervention is missing its projection delta")
                requested_delta = intervention.fixed_projection_delta * intervention.scale
            else:
                if intervention.target_projection is None:
                    raise ValueError("clamped intervention is missing its target projection")
                requested_delta = (
                    intervention.target_projection - carry_before
                ) * intervention.scale
            requested_shift_float = applied_direction * requested_delta
            requested_shift = requested_shift_float.to(dtype=updated.dtype)
            updated[row_index, position] = hidden[row_index, position] + requested_shift
            if diagnostics is not None:
                after = updated[row_index, position].float()
                realized_shift = after - before
                requested_norm = float(requested_shift_float.norm())
                realized_norm = float(realized_shift.norm())
                diagnostics[row_index] = {
                    "target_carry_projection": intervention.target_projection,
                    "calibration_carry_projection": carry_before,
                    "applied_carry_projection_before": float(before @ carry_direction),
                    "applied_carry_projection_after": float(after @ carry_direction),
                    "requested_applied_projection_delta": float(requested_delta),
                    "realized_applied_projection_delta": float(realized_shift @ applied_direction),
                    "requested_shift_norm": requested_norm,
                    "realized_shift_norm": realized_norm,
                    "realized_to_requested_norm_ratio": (
                        realized_norm / requested_norm if requested_norm > 0.0 else None
                    ),
                }
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _backbone(model: Any) -> Any:
    """Return the decoder backbone that produces normalized hidden states."""
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise TypeError("Goal 3 requires a causal LM exposing model and lm_head")
    return backbone


def _decoder_layers(model: Any) -> Any:
    """Return the causal LM's ordered decoder-layer collection."""
    backbone = _backbone(model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        decoder = getattr(backbone, "decoder", None)
        layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("Goal 3 could not locate decoder layers on the causal LM")
    return layers


def _model_device(model: Any) -> Any:
    """Return the device receiving model input IDs."""
    embedding = model.get_input_embeddings()
    return embedding.weight.device


def _replay_identity(case: Goal3ReplayCase) -> dict[str, Any]:
    """Return shared identifying fields for replay artifacts."""
    return {
        "id": case.id,
        "example_id": case.example_id,
        "problem_id": case.problem_id,
        "n_digits": case.n_digits,
        "replay_kind": case.replay_kind,
        "source_model_name": case.source_model_name,
        "receiver_model_name": case.receiver_model_name,
        "location_kind": case.location_kind.value,
        "source_answer_correct": case.metadata.get("source_answer_correct"),
    }


def _completion_audit_rows(prefixes: list[Goal3ReplayPrefix]) -> list[dict[str, Any]]:
    """Audit natural completions for trailing reasoning and answer ambiguity."""
    unique: dict[tuple[str, str], Goal3ReplayPrefix] = {}
    for prefix in prefixes:
        if prefix.source_model_name is not None:
            unique[(prefix.example_id, prefix.source_model_name)] = prefix
    rows = []
    for prefix in unique.values():
        candidates = []
        for match in re.finditer(r"[0-9][0-9,|]*", prefix.decoded_output):
            normalized = normalize_output_digits(match.group(0), base=10)
            if normalized is not None:
                candidates.append(
                    {
                        "text": match.group(0),
                        "normalized": normalized,
                        "char_start": match.start(),
                        "char_end": match.end(),
                    }
                )
        final_candidate = candidates[-1] if candidates else None
        trailing = (
            prefix.decoded_output[int(final_candidate["char_end"]) :]
            if final_candidate is not None
            else prefix.decoded_output
        )
        answer_markers = list(
            re.finditer(
                r"final answer|answer is|therefore|thus",
                prefix.decoded_output,
                flags=re.IGNORECASE,
            )
        )
        rows.append(
            {
                "example_id": prefix.example_id,
                "problem_id": prefix.problem_id,
                "n_digits": prefix.n_digits,
                "source_model_name": prefix.source_model_name,
                "expected_output": prefix.expected_output,
                "parsed_answer": prefix.parsed_answer,
                "source_answer_correct": prefix.metadata.get("source_answer_correct"),
                "numeric_candidate_count": len(candidates),
                "final_numeric_candidate": final_candidate,
                "trailing_character_count": len(trailing),
                "trailing_has_alphanumeric": bool(re.search(r"[A-Za-z0-9]", trailing)),
                "answer_marker_count": len(answer_markers),
                "last_answer_marker_char_index": (
                    answer_markers[-1].start() if answer_markers else None
                ),
                "clean_terminal_answer": (
                    final_candidate is not None and not bool(re.search(r"[A-Za-z0-9]", trailing))
                ),
            }
        )
    return rows


def _residual_identity(execution: ResidualExecution) -> dict[str, Any]:
    """Return shared identifying fields for residual artifacts."""
    case = execution.case
    return {
        "id": execution.id,
        "case_id": case.id,
        "example_id": case.example_id,
        "problem_id": case.problem_id,
        "n_digits": case.n_digits,
        "model_name": case.model_name,
        "target": case.target.value,
        "target_column_lsd": case.target_column_lsd,
        "affected_output_column_lsd": case.affected_output_column_lsd,
        "factual_carry": case.factual_carry,
        "counterfactual_carry": case.counterfactual_carry,
        "location_kind": case.location_kind.value,
        "layer_index": case.layer_index,
        "direction_id": case.direction_id,
        "intervention_scale": execution.intervention_scale,
        "intervention_mode": execution.intervention_mode,
        "intervention_site": execution.intervention_site,
        "control_direction": execution.control_direction,
    }


def _off_target_digits_preserved(
    observed: str | None,
    factual: str,
    unchanged_columns: list[int],
) -> bool | None:
    """Return whether a generated answer preserves all specified factual digits."""
    if observed is None or len(observed) != len(factual):
        return None
    observed_lsd = list(reversed(observed))
    factual_lsd = list(reversed(factual))
    return all(observed_lsd[index] == factual_lsd[index] for index in unchanged_columns)


def _completed_ids(path: Path) -> set[str]:
    """Return completed case IDs from an append-only artifact."""
    if not path.exists():
        return set()
    return {str(row["id"]) for row in read_jsonl(path)}


def _batches(rows: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield fixed-size list batches."""
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _grouped_batches(
    rows: list[Any],
    size: int,
    key: Any,
) -> Iterator[list[Any]]:
    """Yield batches that never mix values of a required grouping key."""
    groups: dict[Any, list[Any]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    for group_key in sorted(groups, key=str):
        yield from _batches(groups[group_key], size)


def _print_progress(
    stage: str,
    model_name: str,
    completed: int,
    total: int,
    started: float,
) -> None:
    """Print cumulative progress and elapsed time for one execution stage."""
    percent = 100.0 * completed / total if total else 100.0
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    print(
        f"[{stage}] model={model_name} completed={completed}/{total} "
        f"({percent:.1f}%) elapsed={elapsed_minutes:.1f}m",
        flush=True,
    )


def _write_analysis_artifacts(run_dir: Path) -> None:
    """Write paired replay effects and grouped Goal 3 summary metrics."""
    replay_scores = _read_optional_jsonl(run_dir / REPLAY_SCORES_FILENAME)
    replay_generations = _read_optional_jsonl(run_dir / REPLAY_GENERATIONS_FILENAME)
    residual_scores = _read_optional_jsonl(run_dir / RESIDUAL_SCORES_FILENAME)
    residual_generations = _read_optional_jsonl(run_dir / RESIDUAL_GENERATIONS_FILENAME)
    replay_effects = _replay_effect_rows(replay_scores)
    residual_control_contrasts = _residual_control_contrast_rows(residual_scores)
    summary = _summary_rows(
        replay_scores,
        replay_generations,
        residual_scores,
        residual_generations,
        residual_control_contrasts,
    )
    write_jsonl(run_dir / REPLAY_EFFECTS_FILENAME, replay_effects)
    write_jsonl(
        run_dir / RESIDUAL_CONTROL_CONTRASTS_FILENAME,
        residual_control_contrasts,
    )
    write_jsonl(run_dir / SUMMARY_METRICS_FILENAME, summary)


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact or return an empty list when disabled."""
    return read_jsonl(path) if path.exists() else []


def _replay_effect_rows(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subtract each receiver's no-reasoning score on the same problem."""
    baselines = {
        (row["example_id"], row["receiver_model_name"]): row
        for row in scores
        if row["location_kind"] == ActivationLocation.PROMPT_FINAL.value
    }
    rows = []
    for row in scores:
        baseline = baselines.get((row["example_id"], row["receiver_model_name"]))
        if baseline is None:
            continue
        rows.append(
            {
                **row,
                "prompt_final_logprob": baseline["expected_answer_logprob"],
                "logprob_delta_from_prompt_final": (
                    row["expected_answer_logprob"] - baseline["expected_answer_logprob"]
                ),
            }
        )
    return rows


def _residual_control_contrast_rows(
    scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare each carry effect with the mean of its orthogonal controls."""
    grouped: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[
            (
                str(row["case_id"]),
                float(row["intervention_scale"]),
                str(row.get("intervention_mode", "fixed_gap")),
                str(row.get("intervention_site", "prefix_boundary")),
            )
        ].append(row)
    rows = []
    for (case_id, scale, mode, site), group in sorted(grouped.items()):
        carry = next(
            (row for row in group if row["control_direction"] == "carry"),
            None,
        )
        orthogonals = [
            row for row in group if str(row["control_direction"]).startswith("orthogonal")
        ]
        if carry is None or not orthogonals:
            continue
        carry_shift = float(carry["counterfactual_preference_shift"])
        orthogonal_shifts = [float(row["counterfactual_preference_shift"]) for row in orthogonals]
        orthogonal_shift = float(np.mean(orthogonal_shifts))
        rows.append(
            {
                **{
                    key: carry[key]
                    for key in [
                        "case_id",
                        "example_id",
                        "problem_id",
                        "n_digits",
                        "model_name",
                        "target",
                        "target_column_lsd",
                        "affected_output_column_lsd",
                        "factual_carry",
                        "counterfactual_carry",
                        "location_kind",
                        "layer_index",
                        "direction_id",
                    ]
                },
                "id": stable_hash(
                    {
                        "case_id": case_id,
                        "intervention_scale": scale,
                        "intervention_mode": mode,
                        "intervention_site": site,
                    },
                    length=16,
                ),
                "intervention_scale": scale,
                "intervention_mode": mode,
                "intervention_site": site,
                "carry_preference_shift": carry_shift,
                "orthogonal_preference_shift": orthogonal_shift,
                "orthogonal_preference_shift_std": (
                    float(np.std(orthogonal_shifts, ddof=1)) if len(orthogonal_shifts) > 1 else 0.0
                ),
                "orthogonal_control_n": len(orthogonal_shifts),
                "carry_minus_orthogonal_shift": carry_shift - orthogonal_shift,
                "carry_shift_positive": carry_shift > 0.0,
                "orthogonal_shift_positive_rate": float(
                    np.mean([shift > 0.0 for shift in orthogonal_shifts])
                ),
            }
        )
    return rows


def _summary_rows(
    replay_scores: list[dict[str, Any]],
    replay_generations: list[dict[str, Any]],
    residual_scores: list[dict[str, Any]],
    residual_generations: list[dict[str, Any]],
    residual_control_contrasts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate raw Goal 3 metrics over preregistered grouping dimensions."""
    rows = []
    replay_generation_by_id = {row["id"]: row for row in replay_generations}
    replay_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in replay_scores:
        key = (
            row["receiver_model_name"],
            row["source_model_name"],
            row["location_kind"],
            row["n_digits"],
        )
        replay_groups[key].append(row)
    for key, group in sorted(replay_groups.items(), key=lambda item: str(item[0])):
        generated = [
            replay_generation_by_id[row["id"]]
            for row in group
            if row["id"] in replay_generation_by_id
        ]
        rows.append(
            {
                "analysis": "replay",
                "receiver_model_name": key[0],
                "source_model_name": key[1],
                "location_kind": key[2],
                "n_digits": key[3],
                "n": len(group),
                "mean_expected_answer_logprob": _mean(group, "expected_answer_logprob"),
                "generation_n": len(generated),
                "generation_exact_match": _mean(generated, "exact_match"),
            }
        )
    residual_generation_by_id = {row["id"]: row for row in residual_generations}
    residual_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in residual_scores:
        key = (
            row["model_name"],
            row["target"],
            row["target_column_lsd"],
            row["location_kind"],
            row["layer_index"],
            row["n_digits"],
            row["intervention_scale"],
            row.get("intervention_mode", "fixed_gap"),
            row.get("intervention_site", "prefix_boundary"),
            row["control_direction"],
            row["factual_carry"],
        )
        residual_groups[key].append(row)
    for key, group in sorted(residual_groups.items(), key=lambda item: str(item[0])):
        generated = [
            residual_generation_by_id[row["id"]]
            for row in group
            if row["id"] in residual_generation_by_id
        ]
        rows.append(
            {
                "analysis": "residual",
                "model_name": key[0],
                "target": key[1],
                "target_column_lsd": key[2],
                "location_kind": key[3],
                "layer_index": key[4],
                "n_digits": key[5],
                "intervention_scale": key[6],
                "intervention_mode": key[7],
                "intervention_site": key[8],
                "control_direction": key[9],
                "factual_carry": key[10],
                "n": len(group),
                "mean_counterfactual_preference_shift": _mean(
                    group,
                    "counterfactual_preference_shift",
                ),
                "counterfactual_shift_positive_rate": _mean(
                    group,
                    "counterfactual_shift_positive",
                ),
                "mean_requested_shift_norm": _mean(group, "requested_shift_norm"),
                "mean_realized_shift_norm": _mean(group, "realized_shift_norm"),
                "mean_realized_to_requested_norm_ratio": _mean(
                    group,
                    "realized_to_requested_norm_ratio",
                ),
                "mean_realized_applied_projection_delta": _mean(
                    group,
                    "realized_applied_projection_delta",
                ),
                "generation_n": len(generated),
                "intervention_counterfactual_exact_match": _mean(
                    generated,
                    "intervention_counterfactual_exact_match",
                ),
                "off_target_digits_preserved": _mean(
                    generated,
                    "off_target_digits_preserved",
                ),
            }
        )
    contrast_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in residual_control_contrasts:
        key = (
            row["model_name"],
            row["target_column_lsd"],
            row["location_kind"],
            row["layer_index"],
            row["n_digits"],
            row["intervention_scale"],
            row.get("intervention_mode", "fixed_gap"),
            row.get("intervention_site", "prefix_boundary"),
            row["factual_carry"],
        )
        contrast_groups[key].append(row)
    for key, group in sorted(contrast_groups.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                "analysis": "residual_control_contrast",
                "model_name": key[0],
                "target_column_lsd": key[1],
                "location_kind": key[2],
                "layer_index": key[3],
                "n_digits": key[4],
                "intervention_scale": key[5],
                "intervention_mode": key[6],
                "intervention_site": key[7],
                "factual_carry": key[8],
                "n": len(group),
                "mean_carry_preference_shift": _mean(group, "carry_preference_shift"),
                "mean_orthogonal_preference_shift": _mean(
                    group,
                    "orthogonal_preference_shift",
                ),
                "mean_carry_minus_orthogonal_shift": _mean(
                    group,
                    "carry_minus_orthogonal_shift",
                ),
                "mean_orthogonal_preference_shift_std": _mean(
                    group,
                    "orthogonal_preference_shift_std",
                ),
                "orthogonal_control_n": int(group[0]["orthogonal_control_n"]),
                "carry_shift_positive_rate": _mean(group, "carry_shift_positive"),
            }
        )
    return rows


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    """Return the mean of available numeric or Boolean values."""
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _artifact_counts(run_dir: Path) -> dict[str, int]:
    """Count rows in each Goal 3 JSONL artifact."""
    filenames = [
        REPLAY_SCORES_FILENAME,
        REPLAY_GENERATIONS_FILENAME,
        REPLAY_EFFECTS_FILENAME,
        RESIDUAL_SCORES_FILENAME,
        RESIDUAL_GENERATIONS_FILENAME,
        RESIDUAL_CONTROL_CONTRASTS_FILENAME,
        DIRECTION_METADATA_FILENAME,
        COMPLETION_AUDITS_FILENAME,
        SUMMARY_METRICS_FILENAME,
    ]
    return {
        filename: len(read_jsonl(run_dir / filename))
        for filename in filenames
        if (run_dir / filename).exists()
    }


def _cleanup_runner(runner: HuggingFaceModelRunner) -> None:
    """Release one model before loading the next checkpoint."""
    try:
        import torch

        del runner.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        gc.collect()
