"""Goal 2 linear-probe training and evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from carry_trace.config import Goal2ProbeConfig
from carry_trace.enums import ProbeTarget
from carry_trace.goal2_probe_analysis import (
    GOAL2_BOOTSTRAP_METRICS_FILENAME,
    GOAL2_FIGURE_METRICS_FILENAME,
    GOAL2_PROBE_ANALYSIS_SCHEMA_VERSION,
    GOAL2_SHARED_SLICE_METRICS_FILENAME,
    GOAL2_SLICE_METRICS_FILENAME,
    build_goal2_probe_analysis,
)
from carry_trace.io import ensure_dir, read_jsonl, stable_hash, utc_now_iso, write_json, write_jsonl

COT_LOCATION_KINDS = {"cot_start", "cot_1_3", "cot_2_3", "cot_end"}


def run_goal2_probes(config: Goal2ProbeConfig, show_progress: bool = False) -> Path:
    """Train and evaluate Goal 2 linear probes from saved activations."""
    config_hash = stable_hash(config.model_dump(mode="json"))
    run_id = f"{config.name}-{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    probe_dir = ensure_dir(config.output_dir / run_id)
    examples = {row["id"]: row for row in read_jsonl(config.goal2_run_dir / "dataset.jsonl")}
    records = _valid_activation_records(config, examples)
    groups = _build_probe_groups(config, examples, records, show_progress=show_progress)
    metrics, predictions = _fit_probe_groups(config, groups, show_progress=show_progress)
    (
        slice_metrics,
        shared_slice_metrics,
        figure_metrics,
        bootstrap_metrics,
    ) = build_goal2_probe_analysis(predictions)

    write_jsonl(probe_dir / "probe_metrics.jsonl", metrics)
    write_jsonl(probe_dir / "probe_predictions.jsonl", predictions)
    write_jsonl(probe_dir / GOAL2_SLICE_METRICS_FILENAME, slice_metrics)
    write_jsonl(probe_dir / GOAL2_SHARED_SLICE_METRICS_FILENAME, shared_slice_metrics)
    write_jsonl(probe_dir / GOAL2_FIGURE_METRICS_FILENAME, figure_metrics)
    write_jsonl(probe_dir / GOAL2_BOOTSTRAP_METRICS_FILENAME, bootstrap_metrics)
    write_json(
        probe_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "config_hash": config_hash,
            "config": config.model_dump(mode="json"),
            "goal2_run_dir": str(config.goal2_run_dir),
            "activation_record_count": len(records),
            "probe_group_count": len(groups),
            "fitted_probe_count": sum(1 for row in metrics if row["status"] == "fitted"),
            "prediction_count": len(predictions),
            "slice_metric_count": len(slice_metrics),
            "shared_slice_metric_count": len(shared_slice_metrics),
            "figure_metric_count": len(figure_metrics),
            "bootstrap_metric_count": len(bootstrap_metrics),
            "analysis_schema_version": GOAL2_PROBE_ANALYSIS_SCHEMA_VERSION,
            "artifact_kind": "goal2_linear_probes",
        },
    )
    return probe_dir


def _valid_activation_records(
    config: Goal2ProbeConfig,
    examples: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return uncapped activation records matching the configured probe data filters."""
    records = []
    allowed_splits = {config.train_split, config.test_split}
    allowed_prompt_modes = (
        {str(prompt_mode) for prompt_mode in config.prompt_modes}
        if config.prompt_modes is not None
        else None
    )
    for record in read_jsonl(config.goal2_run_dir / "activations.jsonl"):
        example = examples.get(str(record.get("example_id")))
        if example is None or example.get("split") not in allowed_splits:
            continue
        if (
            allowed_prompt_modes is not None
            and example.get("prompt_mode") not in allowed_prompt_modes
        ):
            continue
        if _hit_token_limit(record):
            continue
        activation_path = config.goal2_run_dir / str(record.get("activation_path", ""))
        if not activation_path.exists():
            continue
        records.append(record)
    return records


def _hit_token_limit(record: dict[str, Any]) -> bool:
    """Return whether an activation record came from a capped generation."""
    metadata = record.get("call_metadata") or {}
    return bool(metadata.get("hit_token_limit"))


def _build_probe_groups(
    config: Goal2ProbeConfig,
    examples: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    show_progress: bool = False,
) -> dict[tuple[str, str, str, str, int | None], dict[str, list[dict[str, Any]]]]:
    """Build train/test feature groups keyed by model, target, location kind, and layer."""
    groups: dict[tuple[str, str, str, str, int | None], dict[str, list[dict[str, Any]]]] = {}
    record_iter = _progress(records, show_progress, "Loading activation tensors", "record")
    for record_index, record in enumerate(record_iter, start=1):
        example = examples[str(record["example_id"])]
        split = _probe_split(config, example)
        if split is None:
            continue
        payload = _load_activation_payload(config.goal2_run_dir / str(record["activation_path"]))
        activations = payload["activations"]
        locations = payload.get("locations") or record["locations"]
        layer_indices = payload.get("layer_indices") or record["layer_indices"]
        ambiguous_location_indexes = _ambiguous_digit_location_indexes(locations)
        for location_index, location in enumerate(locations):
            if location_index >= activations.shape[0]:
                continue
            if not _use_location_for_example(example, location):
                continue
            location_lsd_index = _location_lsd_index(location)
            location_features = (
                activations[location_index].detach().to(dtype=_torch_float32()).cpu().numpy()
            )
            for target in config.targets:
                target = ProbeTarget(target)
                if (
                    config.require_unambiguous_digit_tokens
                    and target != ProbeTarget.ANY_CARRY
                    and location_index in ambiguous_location_indexes
                ):
                    continue
                labels = _target_labels(example, target)
                if not labels:
                    continue
                _append_probe_samples(
                    groups=groups,
                    split=split,
                    record=record,
                    example=example,
                    location=location,
                    layer_indices=layer_indices,
                    location_features=location_features,
                    location_lsd_index=location_lsd_index,
                    target=target,
                    labels=labels,
                    record_index=record_index,
                )
    return groups


def _append_probe_samples(
    groups: dict[tuple[str, str, str, str, int | None], dict[str, list[dict[str, Any]]]],
    split: str,
    record: dict[str, Any],
    example: dict[str, Any],
    location: dict[str, Any],
    layer_indices: list[int | str],
    location_features: np.ndarray,
    location_lsd_index: int | None,
    target: ProbeTarget,
    labels: list[tuple[int | None, int]],
    record_index: int,
) -> None:
    """Append feature rows for one location across layers and target columns."""
    for target_column_lsd, label in labels:
        same_column = (
            location_lsd_index == target_column_lsd if target_column_lsd is not None else None
        )
        for layer_offset, layer_index in enumerate(layer_indices):
            feature = location_features[layer_offset]
            key = (
                str(record["model_name"]),
                target.value,
                str(location["kind"]),
                str(layer_index),
                target_column_lsd,
            )
            groups.setdefault(key, {"train": [], "test": []})[split].append(
                {
                    "x": feature,
                    "y": int(label),
                    "metadata": _sample_metadata(
                        record=record,
                        example=example,
                        location=location,
                        layer_index=layer_index,
                        record_index=record_index,
                        target_column_lsd=target_column_lsd,
                        location_lsd_index=location_lsd_index,
                        same_column=same_column,
                    ),
                }
            )


def _use_location_for_example(example: dict[str, Any], location: dict[str, Any]) -> bool:
    """Return whether a saved location should be used for a dataset example."""
    if location.get("kind") in COT_LOCATION_KINDS:
        return example.get("prompt_mode") == "free_cot"
    return True


def _load_activation_payload(path: Path) -> dict[str, Any]:
    """Load one saved activation tensor payload from disk."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_float32() -> Any:
    """Return torch.float32 without importing torch at module import time."""
    import torch

    return torch.float32


def _probe_split(config: Goal2ProbeConfig, example: dict[str, Any]) -> str | None:
    """Return the probe split name for a dataset example."""
    if example.get("split") == config.train_split:
        return "train"
    if example.get("split") == config.test_split:
        return "test"
    return None


def _ambiguous_digit_location_indexes(locations: list[dict[str, Any]]) -> set[int]:
    """Return digit-location indexes whose token is shared by multiple digit locations."""
    token_to_indexes: dict[tuple[str, int], list[int]] = {}
    for index, location in enumerate(locations):
        if _location_lsd_index(location) is None:
            continue
        key = (str(location.get("kind")), int(location.get("absolute_token_index", -1)))
        token_to_indexes.setdefault(key, []).append(index)
    ambiguous: set[int] = set()
    for indexes in token_to_indexes.values():
        if len(indexes) > 1:
            ambiguous.update(indexes)
    return ambiguous


def _target_labels(
    example: dict[str, Any],
    target: ProbeTarget,
) -> list[tuple[int | None, int]]:
    """Return probe labels keyed by target column for one example."""
    if target == ProbeTarget.ANY_CARRY:
        return [(None, int(int(example.get("carry_count", 0)) > 0))]
    if target == ProbeTarget.CARRY_CHAIN_MEMBERSHIP:
        return _binary_column_labels(_carry_chain_membership(example))
    if target == ProbeTarget.COLUMN_POINTER:
        return _binary_column_labels(_column_pointer(example))
    values = _column_values(example, target)
    if not isinstance(values, list):
        return []
    return [(index, int(value)) for index, value in enumerate(values)]


def _column_values(example: dict[str, Any], target: ProbeTarget) -> object:
    """Return the dataset field values for a per-column probe target."""
    if target == ProbeTarget.OUTPUT_DIGIT:
        return example.get("output_digits_lsd")
    return example.get(target.value)


def _binary_column_labels(values: list[int]) -> list[tuple[int | None, int]]:
    """Return binary labels for every column in a value list."""
    return [(index, int(value)) for index, value in enumerate(values)]


def _carry_chain_membership(example: dict[str, Any]) -> list[int]:
    """Return whether each column participates in any carry transition."""
    incoming = example.get("incoming_carry")
    outgoing = example.get("outgoing_carry")
    if not isinstance(incoming, list) or not isinstance(outgoing, list):
        return []
    return [int(bool(a) or bool(b)) for a, b in zip(incoming, outgoing, strict=False)]


def _column_pointer(example: dict[str, Any]) -> list[int]:
    """Return one-hot labels for the first carry-producing column."""
    n_digits = int(example.get("n_digits", 0))
    first_carry = example.get("first_carry_position")
    return [int(first_carry == index) for index in range(n_digits)]


def _location_lsd_index(location: dict[str, Any]) -> int | None:
    """Return a location's least-significant digit index when available."""
    metadata = location.get("metadata") or {}
    value = metadata.get("lsd_index")
    return int(value) if isinstance(value, int) else None


def _sample_metadata(
    record: dict[str, Any],
    example: dict[str, Any],
    location: dict[str, Any],
    layer_index: str | int,
    record_index: int,
    target_column_lsd: int | None,
    location_lsd_index: int | None,
    same_column: bool | None,
) -> dict[str, Any]:
    """Return JSON metadata for one probe sample."""
    expected_output = example.get("expected_output") or example.get("answer")
    parsed_answer = record.get("parsed_answer")
    behavior_correct = parsed_answer == expected_output if parsed_answer is not None else None
    metadata = location.get("metadata") or {}
    return {
        "example_id": record["example_id"],
        "problem_id": example.get("problem_id", record["example_id"]),
        "model_name": record["model_name"],
        "split": example.get("split"),
        "prompt_mode": example.get("prompt_mode"),
        "digit_format": example.get("digit_format"),
        "answer_format": example.get("answer_format"),
        "slice_name": example.get("slice_name"),
        "n_digits": example.get("n_digits"),
        "carry_count": example.get("carry_count"),
        "max_carry_chain": example.get("max_carry_chain"),
        "location_name": location.get("name"),
        "location_kind": location.get("kind"),
        "layer_index": layer_index,
        "location_lsd_index": location_lsd_index,
        "target_column_lsd": target_column_lsd,
        "same_column": same_column,
        "lsd_index": metadata.get("lsd_index"),
        "behavior_correct": behavior_correct,
        "record_index": record_index,
    }


def _fit_probe_groups(
    config: Goal2ProbeConfig,
    groups: dict[tuple[str, str, str, str, int | None], dict[str, list[dict[str, Any]]]],
    show_progress: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit all probe groups and return metric and prediction rows."""
    group_items = sorted(groups.items())
    if config.n_jobs == 1 or len(group_items) <= 1:
        results = [
            _fit_one_probe_group(config, key, split_rows)
            for key, split_rows in _progress(
                group_items, show_progress, "Fitting probe groups", "probe"
            )
        ]
    else:
        results = _fit_probe_groups_parallel(config, group_items, show_progress)
    metrics = []
    predictions = []
    for metric, prediction_rows in results:
        metrics.append(metric)
        predictions.extend(prediction_rows)
    return metrics, predictions


def _fit_probe_groups_parallel(
    config: Goal2ProbeConfig,
    group_items: list[
        tuple[tuple[str, str, str, str, int | None], dict[str, list[dict[str, Any]]]]
    ],
    show_progress: bool,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Fit probe groups with a small thread pool while preserving output order."""
    results: list[tuple[dict[str, Any], list[dict[str, Any]]] | None] = [None] * len(group_items)
    with ThreadPoolExecutor(max_workers=config.n_jobs) as executor:
        futures = {
            executor.submit(_fit_one_probe_group, config, key, split_rows): index
            for index, (key, split_rows) in enumerate(group_items)
        }
        future_iter = as_completed(futures)
        if show_progress:
            future_iter = tqdm(
                future_iter,
                total=len(futures),
                desc="Fitting probe groups",
                unit="probe",
                dynamic_ncols=True,
            )
        for future in future_iter:
            results[futures[future]] = future.result()
    return [result for result in results if result is not None]


def _fit_one_probe_group(
    config: Goal2ProbeConfig,
    key: tuple[str, str, str, str, int | None],
    split_rows: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit or skip one probe group and return its metric and predictions."""
    model_name, target, location_kind, layer_index, target_column_lsd = key
    train_rows = split_rows["train"]
    test_rows = split_rows["test"]
    base_metric = _metric_base_row(
        model_name=model_name,
        target=target,
        location_kind=location_kind,
        layer_index=layer_index,
        target_column_lsd=target_column_lsd,
        train_rows=train_rows,
        test_rows=test_rows,
    )
    skip_reason = _skip_reason(config, train_rows, test_rows)
    if skip_reason is not None:
        return {**base_metric, "status": "skipped", "skip_reason": skip_reason}, []
    x_train, y_train = _xy(train_rows)
    x_test, y_test = _xy(test_rows)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.c,
            class_weight="balanced",
            max_iter=config.max_iter,
            random_state=config.random_state,
        ),
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_score = _positive_scores(classifier, x_test)
    metric = {
        **base_metric,
        "status": "fitted",
        "skip_reason": None,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": _balanced_accuracy(y_test, y_pred),
        "test_majority_baseline": _majority_baseline(y_test),
    }
    predictions = _prediction_rows(
        model_name=model_name,
        target=target,
        location_kind=location_kind,
        layer_index=layer_index,
        target_column_lsd=target_column_lsd,
        test_rows=test_rows,
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
    )
    return metric, predictions


def _progress(items: list[Any], enabled: bool, description: str, unit: str) -> Any:
    """Return a tqdm-wrapped iterable when progress reporting is enabled."""
    if not enabled:
        return items
    return tqdm(items, desc=description, unit=unit, dynamic_ncols=True)


def _metric_base_row(
    model_name: str,
    target: str,
    location_kind: str,
    layer_index: str,
    target_column_lsd: int | None,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return common metric fields for a probe group."""
    y_train = [row["y"] for row in train_rows]
    y_test = [row["y"] for row in test_rows]
    return {
        "model_name": model_name,
        "target": target,
        "target_column_lsd": target_column_lsd,
        "location_kind": location_kind,
        "layer_index": layer_index,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "train_class_count": len(set(y_train)),
        "test_class_count": len(set(y_test)),
        "train_positive_rate": _binary_positive_rate(y_train),
        "test_positive_rate": _binary_positive_rate(y_test),
    }


def _binary_positive_rate(values: list[int]) -> float | None:
    """Return the positive rate when labels are binary, otherwise null."""
    if not values or not set(values).issubset({0, 1}):
        return None
    return float(np.mean(values))


def _skip_reason(
    config: Goal2ProbeConfig,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> str | None:
    """Return why a probe group should not be fit, if applicable."""
    if len(train_rows) < config.min_train_examples:
        return "too_few_train_examples"
    if len(test_rows) < config.min_test_examples:
        return "too_few_test_examples"
    if len({row["y"] for row in train_rows}) < 2:
        return "single_train_class"
    return None


def _xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return feature and label arrays for sklearn."""
    return np.stack([row["x"] for row in rows]), np.array([row["y"] for row in rows])


def _positive_scores(classifier: Any, x_test: np.ndarray) -> list[float | None]:
    """Return binary positive-class scores when available."""
    final_estimator = classifier[-1]
    classes = list(getattr(final_estimator, "classes_", []))
    if len(classes) != 2 or 1 not in classes or not hasattr(classifier, "predict_proba"):
        return [None] * len(x_test)
    class_index = classes.index(1)
    return [float(value) for value in classifier.predict_proba(x_test)[:, class_index]]


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Return balanced accuracy when both classes appear in the test set."""
    if len(set(y_true.tolist())) < 2:
        return None
    return float(balanced_accuracy_score(y_true, y_pred))


def _majority_baseline(y_true: np.ndarray) -> float:
    """Return the majority-class baseline accuracy."""
    _, counts = np.unique(y_true, return_counts=True)
    return float(counts.max() / counts.sum())


def _prediction_rows(
    model_name: str,
    target: str,
    location_kind: str,
    layer_index: str,
    target_column_lsd: int | None,
    test_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: list[float | None],
) -> list[dict[str, Any]]:
    """Return JSONL prediction rows for a fitted probe group."""
    rows = []
    for index, test_row in enumerate(test_rows):
        rows.append(
            {
                **test_row["metadata"],
                "model_name": model_name,
                "target": target,
                "target_column_lsd": target_column_lsd,
                "location_kind": location_kind,
                "layer_index": layer_index,
                "y_true": int(y_true[index]),
                "y_pred": int(y_pred[index]),
                "y_score": y_score[index],
                "probe_correct": bool(y_true[index] == y_pred[index]),
            }
        )
    return rows
