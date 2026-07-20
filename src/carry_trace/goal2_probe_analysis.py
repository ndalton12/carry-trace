"""Analysis-ready metric aggregation for Goal 2 probe artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

GOAL2_PROBE_ANALYSIS_SCHEMA_VERSION = "goal2.probe_analysis.v1"
GOAL2_SLICE_METRICS_FILENAME = "probe_slice_metrics.jsonl"
GOAL2_SHARED_SLICE_METRICS_FILENAME = "probe_shared_slice_metrics.jsonl"
GOAL2_BOOTSTRAP_METRICS_FILENAME = "probe_bootstrap_metrics.jsonl"
GOAL2_FIGURE_METRICS_FILENAME = "probe_figure_metrics.jsonl"
GOAL2_BOOTSTRAP_REPLICATES = 1000
GOAL2_BOOTSTRAP_SEED = 20260601
GOAL2_BOOTSTRAP_TARGETS = {"outgoing_carry"}
GOAL2_TIMING_LOCATION_KINDS = {
    "prompt_final",
    "cot_start",
    "cot_1_3",
    "cot_2_3",
    "cot_end",
    "answer_digits",
}

_SLICE_KEY_FIELDS = (
    "digit_format",
    "answer_format",
    "prompt_mode",
    "n_digits",
    "position_scope",
    "model_name",
    "target",
    "target_column_lsd",
    "location_kind",
    "layer_index",
)
_BOOTSTRAP_KEY_FIELDS = (
    "digit_format",
    "answer_format",
    "prompt_mode",
    "n_digits",
    "target",
    "target_column_lsd",
    "location_kind",
    "layer_index",
    "position_scope",
)


def build_goal2_probe_analysis(
    predictions: list[dict[str, Any]],
    bootstrap_replicates: int = GOAL2_BOOTSTRAP_REPLICATES,
    random_state: int = GOAL2_BOOTSTRAP_SEED,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build all figure-facing metric tables from per-example probe predictions."""
    if not predictions:
        return [], [], [], []
    shared_example_ids = _shared_model_example_ids(predictions)
    slice_stats: dict[tuple[Any, ...], _MetricAccumulator] = {}
    shared_slice_stats: dict[tuple[Any, ...], _MetricAccumulator] = {}
    bootstrap_contributions: dict[
        tuple[Any, ...],
        dict[str, dict[str, np.ndarray]],
    ] = defaultdict(lambda: defaultdict(dict))
    model_names: set[str] = set()
    for row in predictions:
        model_names.add(str(row["model_name"]))
        _update_slice_stats(slice_stats, row)
        if str(row["example_id"]) in shared_example_ids:
            _update_slice_stats(shared_slice_stats, row)
        _update_bootstrap_contributions(bootstrap_contributions, row)
    slice_metrics = _finalize_slice_stats(slice_stats)
    shared_slice_metrics = _finalize_slice_stats(shared_slice_stats)
    figure_metrics = _build_figure_metrics(slice_metrics, shared_slice_metrics, model_names)
    bootstrap_metrics = _finalize_bootstrap_metrics(
        bootstrap_contributions,
        model_names,
        bootstrap_replicates=bootstrap_replicates,
        random_state=random_state,
    )
    return slice_metrics, shared_slice_metrics, figure_metrics, bootstrap_metrics


class _MetricAccumulator:
    """Accumulate exact classification counts for one precomputed metric slice."""

    def __init__(self) -> None:
        self.examples = 0
        self.correct = 0
        self.class_examples: Counter[int] = Counter()
        self.class_correct: Counter[int] = Counter()
        self.problem_ids: set[str] = set()

    def update(self, row: dict[str, Any]) -> None:
        """Add one prediction row to the metric counts."""
        y_true = int(row["y_true"])
        y_pred = int(row["y_pred"])
        is_correct = y_true == y_pred
        self.examples += 1
        self.correct += int(is_correct)
        self.class_examples[y_true] += 1
        self.class_correct[y_true] += int(is_correct)
        self.problem_ids.add(str(row.get("problem_id", row["example_id"])))

    def metrics(self) -> dict[str, Any]:
        """Return finalized accuracy metrics for the accumulated rows."""
        recalls = [
            self.class_correct[label] / count
            for label, count in self.class_examples.items()
            if count > 0
        ]
        return {
            "status": "fitted",
            "test_accuracy": self.correct / self.examples,
            "test_balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "test_majority_baseline": max(self.class_examples.values()) / self.examples,
            "test_examples": self.examples,
            "test_problem_count": len(self.problem_ids),
            "test_class_count": len(self.class_examples),
        }


def _shared_model_example_ids(predictions: Iterable[dict[str, Any]]) -> set[str]:
    """Return example IDs with prediction rows for every model."""
    model_examples: dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        model_examples[str(row["model_name"])].add(str(row["example_id"]))
    if len(model_examples) < 2:
        return set().union(*model_examples.values()) if model_examples else set()
    return set.intersection(*model_examples.values())


def _update_slice_stats(
    stats: dict[tuple[Any, ...], _MetricAccumulator],
    row: dict[str, Any],
) -> None:
    """Update pooled and stratified metric slices for one prediction row."""
    prompt_mode = row.get("prompt_mode")
    n_digits = row.get("n_digits")
    scopes = ["all"]
    if row.get("location_kind") == "answer_digits" and row.get("same_column") is True:
        scopes.append("same_column")
    for prompt_scope in (None, prompt_mode):
        for digit_scope in (None, n_digits):
            for position_scope in scopes:
                key = (
                    row.get("digit_format"),
                    row.get("answer_format"),
                    prompt_scope,
                    digit_scope,
                    position_scope,
                    str(row["model_name"]),
                    str(row["target"]),
                    row.get("target_column_lsd"),
                    str(row["location_kind"]),
                    str(row["layer_index"]),
                )
                stats.setdefault(key, _MetricAccumulator()).update(row)


def _finalize_slice_stats(
    stats: dict[tuple[Any, ...], _MetricAccumulator],
) -> list[dict[str, Any]]:
    """Return sorted metric rows from accumulated slice counts."""
    rows = [
        {
            **dict(zip(_SLICE_KEY_FIELDS, key, strict=True)),
            **accumulator.metrics(),
            "analysis_schema_version": GOAL2_PROBE_ANALYSIS_SCHEMA_VERSION,
        }
        for key, accumulator in stats.items()
    ]
    return sorted(rows, key=_analysis_row_sort_key)


def _build_figure_metrics(
    slice_metrics: list[dict[str, Any]],
    shared_slice_metrics: list[dict[str, Any]],
    model_names: set[str],
) -> list[dict[str, Any]]:
    """Return fully aggregated numerical inputs for every standard Goal 2 figure."""
    summary_rows, layer_rows = _build_model_figure_metrics(slice_metrics)
    shared_summary_rows, shared_layer_rows = _build_model_figure_metrics(
        shared_slice_metrics
    )
    summary_delta_rows = _build_figure_delta_metrics(shared_summary_rows, model_names)
    layer_delta_rows = _build_figure_delta_metrics(shared_layer_rows, model_names)
    rows = [*summary_rows, *layer_rows, *summary_delta_rows, *layer_delta_rows]
    return sorted(rows, key=lambda row: tuple(str(value) for value in row.values()))


def _build_model_figure_metrics(
    slice_metrics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return precomputed best-layer summaries and layer profiles for each model."""
    eligible = [
        row
        for row in slice_metrics
        if row.get("n_digits") is None and row.get("position_scope") == "all"
    ]
    best_by_column: dict[tuple[Any, ...], dict[str, Any]] = {}
    layer_sums: dict[tuple[Any, ...], list[float]] = {}
    for row in eligible:
        column_key = (
            row.get("digit_format"),
            row.get("answer_format"),
            row.get("prompt_mode"),
            row.get("model_name"),
            row.get("target"),
            row.get("target_column_lsd"),
            row.get("location_kind"),
        )
        current_best = best_by_column.get(column_key)
        if current_best is None or row["test_accuracy"] > current_best["test_accuracy"]:
            best_by_column[column_key] = row
        layer_key = (
            row.get("digit_format"),
            row.get("answer_format"),
            row.get("prompt_mode"),
            row.get("model_name"),
            row.get("target"),
            row.get("location_kind"),
            row.get("layer_index"),
        )
        weighted = layer_sums.setdefault(layer_key, [0.0, 0.0])
        weight = float(row.get("test_examples", 0))
        weighted[0] += float(row["test_accuracy"]) * weight
        weighted[1] += weight
    summary_sums: dict[tuple[Any, ...], list[float]] = {}
    for column_key, row in best_by_column.items():
        summary_key = (*column_key[:5], column_key[6])
        values = summary_sums.setdefault(summary_key, [0.0, 0.0])
        values[0] += float(row["test_accuracy"])
        values[1] += 1.0
    summary_rows = [
        {
            "figure_data_kind": "summary_model",
            "digit_format": key[0],
            "answer_format": key[1],
            "prompt_mode": key[2],
            "model_name": key[3],
            "target": key[4],
            "location_kind": key[5],
            "test_accuracy": values[0] / values[1],
            "analysis_schema_version": GOAL2_PROBE_ANALYSIS_SCHEMA_VERSION,
        }
        for key, values in summary_sums.items()
        if values[1] > 0
    ]
    layer_rows = [
        {
            "figure_data_kind": "layer_model",
            "digit_format": key[0],
            "answer_format": key[1],
            "prompt_mode": key[2],
            "model_name": key[3],
            "target": key[4],
            "location_kind": key[5],
            "layer_index": key[6],
            "test_accuracy": values[0] / values[1],
            "analysis_schema_version": GOAL2_PROBE_ANALYSIS_SCHEMA_VERSION,
        }
        for key, values in layer_sums.items()
        if values[1] > 0
    ]
    return summary_rows, layer_rows


def _build_figure_delta_metrics(
    model_rows: list[dict[str, Any]],
    model_names: set[str],
) -> list[dict[str, Any]]:
    """Return precomputed Full-minus-SFT deltas for summary or layer figure rows."""
    model_pair = _goal2_model_delta_pair(model_names)
    if model_pair is None or not model_rows:
        return []
    minuend, subtrahend = model_pair
    values_by_key: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in model_rows:
        key = (
            row.get("digit_format"),
            row.get("answer_format"),
            row.get("prompt_mode"),
            row.get("target"),
            row.get("location_kind"),
            row.get("layer_index"),
        )
        values_by_key[key][str(row["model_name"])] = float(row["test_accuracy"])
    delta_kind = (
        "summary_delta"
        if model_rows[0]["figure_data_kind"] == "summary_model"
        else "layer_delta"
    )
    rows = []
    for key, model_values in values_by_key.items():
        if minuend not in model_values or subtrahend not in model_values:
            continue
        row = {
            "figure_data_kind": delta_kind,
            "digit_format": key[0],
            "answer_format": key[1],
            "prompt_mode": key[2],
            "target": key[3],
            "location_kind": key[4],
            "minuend_model": minuend,
            "subtrahend_model": subtrahend,
            "delta_accuracy": model_values[minuend] - model_values[subtrahend],
            "analysis_schema_version": GOAL2_PROBE_ANALYSIS_SCHEMA_VERSION,
        }
        if key[5] is not None:
            row["layer_index"] = key[5]
        rows.append(row)
    return rows


def _update_bootstrap_contributions(
    contributions: dict[tuple[Any, ...], dict[str, dict[str, np.ndarray]]],
    row: dict[str, Any],
) -> None:
    """Accumulate per-problem confusion counts for paired carry bootstraps."""
    if row.get("target") not in GOAL2_BOOTSTRAP_TARGETS:
        return
    location_kind = str(row.get("location_kind"))
    if location_kind not in GOAL2_TIMING_LOCATION_KINDS:
        return
    if location_kind == "answer_digits":
        if row.get("same_column") is not True:
            return
        position_scope = "same_column"
    else:
        position_scope = "single_location"
    y_true = int(row["y_true"])
    y_pred = int(row["y_pred"])
    if y_true not in {0, 1} or y_pred not in {0, 1}:
        return
    key = (
        row.get("digit_format"),
        row.get("answer_format"),
        row.get("prompt_mode"),
        row.get("n_digits"),
        str(row["target"]),
        row.get("target_column_lsd"),
        location_kind,
        str(row["layer_index"]),
        position_scope,
    )
    model_name = str(row["model_name"])
    problem_id = str(row.get("problem_id", row["example_id"]))
    model_rows = contributions[key][model_name]
    counts = model_rows.setdefault(problem_id, np.zeros(4, dtype=np.int64))
    counts += np.array(
        [
            int(y_true == 1 and y_pred == 1),
            int(y_true == 1),
            int(y_true == 0 and y_pred == 0),
            int(y_true == 0),
        ],
        dtype=np.int64,
    )


def _finalize_bootstrap_metrics(
    contributions: dict[tuple[Any, ...], dict[str, dict[str, np.ndarray]]],
    model_names: set[str],
    bootstrap_replicates: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Return paired clustered-bootstrap metrics from per-problem counts."""
    model_pair = _goal2_model_delta_pair(model_names)
    if model_pair is None:
        return []
    minuend, subtrahend = model_pair
    rng = np.random.default_rng(random_state)
    rows = []
    for key in sorted(contributions, key=lambda value: tuple(map(str, value))):
        model_rows = contributions[key]
        if minuend not in model_rows or subtrahend not in model_rows:
            continue
        shared_problem_ids = sorted(
            set(model_rows[minuend]) & set(model_rows[subtrahend]),
            key=str,
        )
        if len(shared_problem_ids) < 2:
            continue
        minuend_values = np.stack(
            [model_rows[minuend][problem_id] for problem_id in shared_problem_ids]
        )
        subtrahend_values = np.stack(
            [model_rows[subtrahend][problem_id] for problem_id in shared_problem_ids]
        )
        minuend_accuracy = _balanced_accuracy_from_contributions(minuend_values.sum(axis=0))
        subtrahend_accuracy = _balanced_accuracy_from_contributions(
            subtrahend_values.sum(axis=0)
        )
        if not np.isfinite(minuend_accuracy) or not np.isfinite(subtrahend_accuracy):
            continue
        sample_indexes = rng.integers(
            0,
            len(shared_problem_ids),
            size=(bootstrap_replicates, len(shared_problem_ids)),
        )
        bootstrap_delta = _balanced_accuracy_from_contributions(
            minuend_values[sample_indexes].sum(axis=1)
        ) - _balanced_accuracy_from_contributions(
            subtrahend_values[sample_indexes].sum(axis=1)
        )
        bootstrap_delta = bootstrap_delta[np.isfinite(bootstrap_delta)]
        if bootstrap_delta.size == 0:
            continue
        rows.append(
            {
                **dict(zip(_BOOTSTRAP_KEY_FIELDS, key, strict=True)),
                "minuend_model": minuend,
                "subtrahend_model": subtrahend,
                "paired_problem_count": len(shared_problem_ids),
                "minuend_balanced_accuracy": float(minuend_accuracy),
                "subtrahend_balanced_accuracy": float(subtrahend_accuracy),
                "delta_balanced_accuracy": float(minuend_accuracy - subtrahend_accuracy),
                "ci_lower": float(np.quantile(bootstrap_delta, 0.025)),
                "ci_upper": float(np.quantile(bootstrap_delta, 0.975)),
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_valid_replicates": int(bootstrap_delta.size),
                "analysis_schema_version": GOAL2_PROBE_ANALYSIS_SCHEMA_VERSION,
            }
        )
    return rows


def _balanced_accuracy_from_contributions(values: np.ndarray) -> np.ndarray:
    """Return balanced accuracy from TP, positive, TN, and negative counts."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * (values[..., 0] / values[..., 1] + values[..., 2] / values[..., 3])


def _goal2_model_delta_pair(model_names: set[str]) -> tuple[str, str] | None:
    """Return the preferred two-model subtraction order."""
    ordered = sorted(model_names)
    if len(ordered) != 2:
        return None
    full_models = [model for model in ordered if "full" in model.lower()]
    sft_models = [model for model in ordered if "sft" in model.lower()]
    if len(full_models) == 1 and len(sft_models) == 1:
        return full_models[0], sft_models[0]
    return ordered[0], ordered[1]


def _analysis_row_sort_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable string sort key for an analysis metric row."""
    return tuple(str(row.get(field)) for field in _SLICE_KEY_FIELDS)
