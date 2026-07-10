"""Figure generation from saved Goal 1 artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from carry_trace.io import ensure_dir, read_jsonl

COMPARISON_ERRORBAR = ("ci", 95)
COMPARISON_ERR_KWS = {"color": "#222222", "linewidth": 1.1}
FIGURE_DPI = 300
ERROR_LOCALIZATION_FIGSIZE = (7.2, 3.8)
MODEL_PALETTE = [
    "#56B4E9",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#0072B2",
    "#F0E442",
]
HEATMAP_CMAP = "viridis"
HEATMAP_FIGSIZE = (6.4, 3.8)
COMPARISON_FIGSIZE = (6.8, 3.8)
TOKEN_BUDGET_PANEL_WIDTH = 3.1
TOKEN_BUDGET_PANEL_HEIGHT = 3.6
PROMPT_LINESTYLES = ["-", "--", ":", "-."]
GOAL1_STANDARD_DIGIT_FORMAT = "standard"
GOAL1_STANDARD_ANSWER_FORMAT = "standard"
GOAL2_LOCATION_ORDER = [
    "operand_digits",
    "question_token",
    "prompt_final",
    "cot_start",
    "cot_1_3",
    "cot_2_3",
    "cot_end",
    "answer_digits",
]
GOAL2_TIMING_LOCATION_ORDER = [
    "prompt_final",
    "cot_start",
    "cot_1_3",
    "cot_2_3",
    "cot_end",
    "answer_digits",
]
GOAL2_LAYER_PROFILE_LOCATIONS = ["prompt_final", "cot_2_3", "answer_digits"]
GOAL2_LAYER_PROFILE_MARK_EVERY = 4
GOAL2_STANDARD_DIGIT_FORMAT = "standard"
GOAL2_STANDARD_ANSWER_FORMAT = "standard"
GOAL2_DELIMITED_DIGIT_FORMAT = "delimited"
GOAL2_LSD_ANSWER_FORMAT = "lsd"
GOAL2_FIGURE_FORMAT_SPECS = [
    ("standard", GOAL2_STANDARD_DIGIT_FORMAT, GOAL2_STANDARD_ANSWER_FORMAT),
    ("delimited_digit_format", GOAL2_DELIMITED_DIGIT_FORMAT, GOAL2_STANDARD_ANSWER_FORMAT),
    ("lsd_answer_format", GOAL2_STANDARD_DIGIT_FORMAT, GOAL2_LSD_ANSWER_FORMAT),
]
GOAL2_TARGET_ORDER = [
    "any_carry",
    "incoming_carry",
    "outgoing_carry",
    "carry_chain_membership",
    "output_digit",
    "raw_sum",
    "column_pointer",
]
ERROR_LOCATION_ORDER = [
    "Before First Carry",
    "At First Carry",
    "After First Carry",
]
ERROR_LOCATION_COLORS = {
    "Before First Carry": "#999999",
    "At First Carry": "#D55E00",
    "After First Carry": "#0072B2",
}
DISPLAY_LABELS = {
    "any_carry": "Any Carry",
    "incoming_carry": "Incoming Carry",
    "outgoing_carry": "Outgoing Carry",
    "output_digit": "Output Digit",
    "raw_sum": "Raw Sum",
    "carry_chain_membership": "Carry-Chain Membership",
    "column_pointer": "Column Pointer",
    "operand_digits": "Operand Digits",
    "question_token": "Question Token",
    "prompt_final": "Prompt Final",
    "cot_start": "CoT Start",
    "cot_1_3": "CoT 1/3",
    "cot_2_3": "CoT 2/3",
    "cot_end": "CoT End",
    "answer_digits": "Answer Digits",
    "answer_only": "Answer Only",
    "free_cot": "Free CoT",
    "length_controlled_cot": "Length-Controlled CoT",
    "structured_column_cot": "Structured Column CoT",
    "standard": "Standard",
    "delimited": "Delimited",
    "lsd": "LSD",
    "no_carry": "No Carry",
    "isolated_carry": "Isolated Carry",
    "long_carry_chain": "Long Carry Chain",
    "internal_carry_chain": "Internal Carry Chain",
    "carry_distractor": "Carry Distractor",
    "many_9s_no_carry": "Many 9s No Carry",
    "random": "Random",
}
PAPER_STYLE_RC = {
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.linewidth": 0.8,
    "axes.titleweight": "semibold",
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "savefig.facecolor": "white",
}


def make_goal1_figures(
    run_dir: Path,
    output_dir: Path | None = None,
    include_token_limit_hits: bool = False,
    token_budget_by_digit_length: bool = True,
) -> list[Path]:
    """Generate Goal 1 figures, excluding token-limit hits unless requested."""
    _set_paper_style()
    output_dir = ensure_dir(output_dir or run_dir / "figures")
    examples = pd.DataFrame(read_jsonl(run_dir / "dataset.jsonl"))
    records = pd.DataFrame(read_jsonl(run_dir / "scored_calls.jsonl"))
    records = _filter_figure_records(records, include_token_limit_hits)
    merged = records.merge(
        examples[
            [
                "id",
                "prompt_mode",
                "digit_format",
                "answer_format",
                "slice_name",
                "n_digits",
                "max_carry_chain",
                "first_carry_position",
            ]
        ],
        left_on="example_id",
        right_on="id",
        how="left",
    )
    paths = [
        _accuracy_heatmap(merged, output_dir),
        *_accuracy_heatmaps_by_model(merged, output_dir),
        _standard_format_accuracy_by_digits(merged, output_dir),
        _prompt_mode_comparison(merged, output_dir),
        _digit_format_comparison(merged, output_dir),
        _answer_format_comparison(merged, output_dir),
        _slice_type_comparison(merged, output_dir),
        _error_localization_plot(merged, output_dir),
        _token_budget_curves(
            merged,
            output_dir,
            by_digit_length=token_budget_by_digit_length,
        ),
    ]
    return [path for path in paths if path is not None]


def make_goal2_probe_figures(
    probe_dir: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate Goal 2 figures from saved linear-probe artifacts."""
    _set_paper_style()
    output_dir = ensure_dir(output_dir or probe_dir / "figures")
    metrics_path = probe_dir / "probe_metrics.jsonl"
    predictions_path = probe_dir / "probe_predictions.jsonl"
    metrics = pd.DataFrame(read_jsonl(metrics_path))
    metrics = metrics[metrics["status"] == "fitted"].copy() if not metrics.empty else metrics
    if metrics.empty:
        return []
    predictions = (
        pd.DataFrame(read_jsonl(predictions_path)) if predictions_path.exists() else pd.DataFrame()
    )
    if _has_goal2_format_metadata(predictions) or _has_goal2_format_metadata(metrics):
        return _make_goal2_probe_figure_sets(metrics, predictions, output_dir)
    return _make_goal2_probe_figure_set(metrics, predictions, ensure_dir(output_dir / "standard"))


def _make_goal2_probe_figure_sets(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate Goal 2 figure sets for each supported digit/answer format condition."""
    paths = []
    for dirname, digit_format, answer_format in GOAL2_FIGURE_FORMAT_SPECS:
        format_predictions = _filter_goal2_formats(predictions, digit_format, answer_format)
        format_metrics = _goal2_prediction_metric_df(format_predictions)
        if (
            format_metrics.empty
            and format_predictions.empty
            and _has_goal2_format_metadata(metrics)
        ):
            format_metrics = _filter_goal2_formats(metrics, digit_format, answer_format)
        if format_metrics.empty:
            continue
        paths.extend(
            _make_goal2_probe_figure_set(
                format_metrics,
                format_predictions,
                ensure_dir(output_dir / dirname),
            )
        )
    return paths


def _make_goal2_probe_figure_set(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate one Goal 2 figure set from prepared metrics and predictions."""
    if metrics.empty:
        return []
    delta_predictions = _goal2_shared_model_example_predictions(predictions)
    delta_prediction_metrics = _goal2_prediction_metric_df(delta_predictions)
    delta_metrics = delta_prediction_metrics
    if delta_metrics.empty and not {"model_name", "example_id"}.issubset(predictions.columns):
        delta_metrics = metrics
    summary_dir = ensure_dir(output_dir / "summary")
    diagnostics_dir = ensure_dir(output_dir / "diagnostics")
    paths = [
        *_goal2_target_summary_matrices(metrics, summary_dir),
        _goal2_target_summary_delta_matrix(delta_metrics, summary_dir),
        *_goal2_target_summary_matrices_by_prompt_mode(predictions, summary_dir),
        *_goal2_target_summary_delta_matrices_by_prompt_mode(
            delta_predictions, summary_dir
        ),
        _goal2_reasoning_time_by_column(metrics, summary_dir),
        _goal2_reasoning_time_by_column_delta(delta_metrics, summary_dir),
        _goal2_layer_profile_by_target_free_cot(predictions, summary_dir),
        *_goal2_probe_heatmaps(metrics, diagnostics_dir),
        *_goal2_probe_delta_heatmaps(delta_metrics, diagnostics_dir),
        *_goal2_probe_timing_curves(metrics, diagnostics_dir),
    ]
    return [path for path in paths if path is not None]


def _set_paper_style() -> None:
    """Apply a consistent, paper-oriented plotting style."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=MODEL_PALETTE,
        font_scale=1.05,
        rc=PAPER_STYLE_RC,
    )


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save and close a figure with consistent publication settings."""
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_figure_with_legend_space(fig: plt.Figure, path: Path) -> Path:
    """Save and close a figure while reserving space for right-side legends."""
    fig.subplots_adjust(right=0.78, bottom=0.18, top=0.98, wspace=0.08)
    fig.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    return path


def _model_palette(df: pd.DataFrame) -> dict[object, str]:
    """Return a stable color mapping for model names present in a plot."""
    if "model_name" not in df:
        return {}
    model_names = sorted(df["model_name"].dropna().unique(), key=str)
    return {
        model_name: MODEL_PALETTE[index % len(MODEL_PALETTE)]
        for index, model_name in enumerate(model_names)
    }


def _display_label(value: object) -> str:
    """Return a human-readable display label for an artifact value."""
    text = str(value)
    if text in DISPLAY_LABELS:
        return DISPLAY_LABELS[text]
    return text.replace("_", " ").title()


def _with_display_labels(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a plotting copy with selected categorical columns display-formatted."""
    display_df = df.copy()
    for column in columns:
        if column in display_df:
            display_df[column] = display_df[column].map(_display_label)
    return display_df


def _format_legend(ax: plt.Axes) -> None:
    """Apply readable legend titles after seaborn plotting."""
    legend = ax.get_legend()
    if legend is None:
        return
    title = legend.get_title()
    title.set_text(_display_label(title.get_text()))
    for text in legend.get_texts():
        text.set_text(_display_label(text.get_text()))


def _filter_figure_records(
    records: pd.DataFrame,
    include_token_limit_hits: bool,
) -> pd.DataFrame:
    """Return records eligible for plotting under the token-limit policy."""
    if include_token_limit_hits or records.empty:
        return records
    if "generation_valid" in records:
        return records[records["generation_valid"]].copy()
    if "hit_token_limit" in records:
        return records[~records["hit_token_limit"]].copy()
    return records


def _accuracy_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "accuracy_heatmap.png",
) -> Path | None:
    """Generate an accuracy heatmap by digit length and carry-chain length."""
    if df.empty:
        return None
    table = df.pivot_table(
        index="n_digits",
        columns="max_carry_chain",
        values="parsed_answer_correct",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    sns.heatmap(
        table,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap=HEATMAP_CMAP,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_xlabel("Max carry-chain length")
    ax.set_ylabel("Digits")
    path = output_dir / filename
    return _save_figure(fig, path)


def _accuracy_heatmaps_by_model(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate one accuracy heatmap per model."""
    if df.empty or "model_name" not in df:
        return []
    paths = []
    for model_name, subset in df.groupby("model_name", dropna=False, sort=True):
        model_label = str(model_name)
        path = _accuracy_heatmap(
            subset,
            output_dir,
            filename=f"accuracy_heatmap_{_filename_slug(model_label)}.png",
        )
        if path is not None:
            paths.append(path)
    return paths


def _filename_slug(value: str) -> str:
    """Return a filesystem-safe slug for a figure filename component."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "unknown_model"


def _standard_format_accuracy_by_digits(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate model accuracy over digit lengths for standard input/output formats."""
    plot_df = _standard_format_digit_accuracy(df)
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    model_palette = _model_palette(plot_df)
    for model_name, subset in plot_df.groupby("model_name", dropna=False, sort=True):
        subset = subset.sort_values("n_digits")
        yerr = [
            subset["accuracy"] - subset["accuracy_ci_low"],
            subset["accuracy_ci_high"] - subset["accuracy"],
        ]
        ax.errorbar(
            subset["n_digits"],
            subset["accuracy"],
            yerr=yerr,
            marker="o",
            linewidth=1.7,
            capsize=3,
            color=model_palette.get(model_name, MODEL_PALETTE[0]),
            label=str(model_name),
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Digits")
    ax.set_ylabel("Parsed Answer Accuracy")
    ax.set_xticks(_digit_lengths(plot_df))
    ax.grid(axis="x", visible=False)
    ax.legend(title="Model Name")
    _format_legend(ax)
    sns.despine(fig=fig, ax=ax)
    path = output_dir / "standard_format_accuracy_by_digits.png"
    return _save_figure(fig, path)


def _standard_format_digit_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-model digit accuracy for standard digit and answer formats."""
    required = {
        "answer_format",
        "digit_format",
        "model_name",
        "n_digits",
        "parsed_answer_correct",
    }
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    standard = df[
        (df["digit_format"] == GOAL1_STANDARD_DIGIT_FORMAT)
        & (df["answer_format"] == GOAL1_STANDARD_ANSWER_FORMAT)
    ].dropna(subset=["n_digits"])
    if standard.empty:
        return pd.DataFrame()
    accuracy = (
        standard.assign(_correct=standard["parsed_answer_correct"].astype(bool))
        .groupby(["model_name", "n_digits"], dropna=False)["_correct"]
        .agg(correct="sum", total="size", accuracy="mean")
        .reset_index()
    )
    accuracy["accuracy_ci_low"] = accuracy.apply(
        lambda row: _binomial_ci_low(int(row["correct"]), int(row["total"])),
        axis=1,
    )
    accuracy["accuracy_ci_high"] = accuracy.apply(
        lambda row: _binomial_ci_high(int(row["correct"]), int(row["total"])),
        axis=1,
    )
    accuracy["n_digits"] = accuracy["n_digits"].astype(int)
    return accuracy.sort_values(["model_name", "n_digits"]).reset_index(drop=True)


def _prompt_mode_comparison(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if df.empty:
        return None
    plot_df = _with_display_labels(df, ["prompt_mode"])
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    sns.barplot(
        data=plot_df,
        x="prompt_mode",
        y="parsed_answer_correct",
        hue="model_name",
        errorbar=COMPARISON_ERRORBAR,
        palette=_model_palette(plot_df),
        capsize=0.08,
        err_kws=COMPARISON_ERR_KWS,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Prompt Mode")
    ax.set_ylabel("Parsed Answer Accuracy")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=20)
    _format_legend(ax)
    sns.despine(fig=fig, ax=ax)
    path = output_dir / "prompt_mode_comparison.png"
    return _save_figure(fig, path)


def _error_localization_plot(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate a carry-conditioned error-localization stacked bar plot."""
    plot_df = _error_localization_data(df)
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=ERROR_LOCALIZATION_FIGSIZE)
    left = pd.Series(0.0, index=plot_df.index)
    y_positions = range(len(plot_df))
    for location in ERROR_LOCATION_ORDER:
        values = plot_df[location]
        ax.barh(
            y_positions,
            values,
            left=left,
            color=ERROR_LOCATION_COLORS[location],
            edgecolor="white",
            linewidth=0.4,
            label=location,
        )
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of Incorrect Carry-Bearing Examples")
    ax.set_ylabel("")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df["group_label"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(title="Error Location", loc="center left", bbox_to_anchor=(1.02, 0.5))
    _format_legend(ax)
    sns.despine(fig=fig, ax=ax)
    path = output_dir / "error_localization.png"
    return _save_figure(fig, path)


def _error_localization_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return parseable error-location shares by model and prompt mode."""
    required = {
        "model_name",
        "prompt_mode",
        "parsed_answer_correct",
        "first_wrong_digit_lsd",
        "first_carry_position",
    }
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    errors = _localized_error_rows(df)
    if errors.empty:
        return pd.DataFrame()
    errors["error_location"] = errors.apply(_error_location, axis=1)
    grouped = (
        errors.groupby(["model_name", "prompt_mode", "error_location"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = grouped.groupby(["model_name", "prompt_mode"], dropna=False)["count"].transform("sum")
    grouped["share"] = grouped["count"] / totals
    table = grouped.pivot_table(
        index=["model_name", "prompt_mode"],
        columns="error_location",
        values="share",
        aggfunc="sum",
        fill_value=0,
    )
    table = table.reindex(columns=ERROR_LOCATION_ORDER, fill_value=0).reset_index()
    error_counts = (
        grouped.groupby(["model_name", "prompt_mode"], dropna=False)["count"]
        .sum()
        .rename("localized_error_count")
        .reset_index()
    )
    table = table.merge(error_counts, on=["model_name", "prompt_mode"], how="left")
    table["group_label"] = table.apply(
        lambda row: (
            f"{_display_label(row['model_name'])} / "
            f"{_display_label(row['prompt_mode'])} "
            f"(n={int(row['localized_error_count'])})"
        ),
        axis=1,
    )
    return table.sort_values(["model_name", "prompt_mode"]).reset_index(drop=True)


def _localized_error_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid, parseable, incorrect carry-bearing rows for localization."""
    localized = df.copy()
    if "generation_valid" in localized:
        localized = localized[localized["generation_valid"] == True]  # noqa: E712
    elif "hit_token_limit" in localized:
        localized = localized[~localized["hit_token_limit"].fillna(False).astype(bool)]
    parse_column = _parse_column(localized)
    if parse_column is not None:
        localized = localized[localized[parse_column].notna()]
    localized = localized.dropna(subset=["first_carry_position", "first_wrong_digit_lsd"])
    return localized[localized["parsed_answer_correct"] == False].copy()  # noqa: E712


def _parse_column(df: pd.DataFrame) -> str | None:
    """Return the best available scored parse column for parseability filtering."""
    if "parsed_output" in df:
        return "parsed_output"
    if "parsed_answer" in df:
        return "parsed_answer"
    return None


def _error_location(row: pd.Series) -> str:
    """Bucket an example by where the first wrong digit occurs relative to carry."""
    first_wrong = int(row["first_wrong_digit_lsd"])
    first_carry = int(row["first_carry_position"])
    if first_wrong < first_carry:
        return "Before First Carry"
    if first_wrong == first_carry:
        return "At First Carry"
    return "After First Carry"


def _token_budget_curves(
    df: pd.DataFrame,
    output_dir: Path,
    by_digit_length: bool = True,
) -> Path | None:
    """Generate correct-and-completed accuracy curves across token budgets."""
    if df.empty or "token_count_output" not in df:
        return None
    curves = _token_budget_curve_data(df, by_digit_length=by_digit_length)
    if curves.empty:
        return None
    plot_df = _with_display_labels(curves, ["prompt_mode"])
    digit_lengths = _digit_lengths(plot_df) if by_digit_length else []
    facet_values = digit_lengths or [None]
    model_palette = _model_palette(plot_df)
    prompt_styles = _prompt_linestyle_map(plot_df)
    fig, axes = plt.subplots(
        ncols=len(facet_values),
        figsize=(TOKEN_BUDGET_PANEL_WIDTH * len(facet_values), TOKEN_BUDGET_PANEL_HEIGHT),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, n_digits in zip(axes[0], facet_values, strict=True):
        subset = plot_df if n_digits is None else plot_df[plot_df["n_digits"] == n_digits]
        _plot_token_budget_lines(ax, subset, model_palette, prompt_styles)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.02)
        if n_digits is not None:
            ax.text(
                0.02,
                0.98,
                f"{n_digits} Digits",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize="medium",
                fontweight="semibold",
            )
        sns.despine(fig=fig, ax=ax)
    axes[0][0].set_ylabel("Correct and Completed")
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    fig.supxlabel("Output Token Budget", x=0.45)
    _token_budget_legends(fig, plot_df)
    path = output_dir / "token_budget_curves.png"
    return _save_figure_with_legend_space(fig, path)


def _plot_token_budget_lines(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    model_palette: dict[object, str],
    prompt_styles: dict[object, str],
) -> None:
    """Draw token-budget curves with model color and prompt-mode line style."""
    group_columns = ["model_name", "prompt_mode"]
    for group_values, subset in plot_df.groupby(group_columns, dropna=False, sort=True):
        values = group_values if isinstance(group_values, tuple) else (group_values,)
        model_name = values[0]
        prompt_mode = values[1]
        color = model_palette.get(model_name, MODEL_PALETTE[0])
        x_values = subset["token_budget"].astype(float).to_numpy()
        y_values = subset["correct_completion_rate"].astype(float).to_numpy()
        ax.plot(
            x_values,
            y_values,
            color=color,
            linestyle=prompt_styles[prompt_mode],
            linewidth=1.7,
        )
        ax.fill_between(
            x_values,
            subset["correct_completion_rate_ci_low"].astype(float).to_numpy(),
            subset["correct_completion_rate_ci_high"].astype(float).to_numpy(),
            color=color,
            alpha=0.12,
            linewidth=0,
        )


def _prompt_linestyle_map(df: pd.DataFrame) -> dict[object, str]:
    """Return a stable line-style mapping for prompt modes present in a plot."""
    prompt_modes = sorted(df["prompt_mode"].dropna().unique(), key=str)
    return {
        prompt_mode: PROMPT_LINESTYLES[index % len(PROMPT_LINESTYLES)]
        for index, prompt_mode in enumerate(prompt_modes)
    }


def _digit_lengths(df: pd.DataFrame) -> list[int]:
    """Return sorted digit lengths available for token-budget faceting."""
    if "n_digits" not in df:
        return []
    return sorted(int(value) for value in df["n_digits"].dropna().unique())


def _token_budget_legends(
    fig: plt.Figure,
    plot_df: pd.DataFrame,
) -> None:
    """Add compact figure-level legends for token-budget encodings."""
    model_palette = _model_palette(plot_df)
    model_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=_display_label(model_name))
        for model_name, color in model_palette.items()
    ]
    prompt_handles = [
        Line2D([0], [0], color="#333333", linestyle=style, linewidth=1.8, label=prompt_mode)
        for prompt_mode, style in _prompt_linestyle_map(plot_df).items()
    ]
    handles = [Line2D([], [], linestyle="none", label="Model Name"), *model_handles]
    if prompt_handles:
        handles.extend(
            [
                Line2D([], [], linestyle="none", label=""),
                Line2D([], [], linestyle="none", label="Prompt Mode"),
                *prompt_handles,
            ]
        )
    legend = fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.805, 0.54),
        handlelength=1.8,
        labelspacing=0.55,
        borderaxespad=0,
    )
    for text in legend.get_texts():
        if text.get_text() in {"Model Name", "Prompt Mode"}:
            text.set_fontweight("semibold")


def _token_budget_curve_data(
    df: pd.DataFrame,
    by_digit_length: bool = True,
) -> pd.DataFrame:
    """Return cumulative token-budget curve rows for the requested grouping."""
    required = {"model_name", "prompt_mode", "token_count_output", "parsed_answer_correct"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    group_columns = ["model_name", "prompt_mode"]
    if by_digit_length and "n_digits" in df:
        group_columns.append("n_digits")
    budgets = sorted(int(value) for value in df["token_count_output"].dropna().unique())
    for group_values, subset in df.groupby(group_columns, dropna=False, sort=True):
        values = group_values if isinstance(group_values, tuple) else (group_values,)
        model_name = values[0]
        prompt_mode = values[1]
        n_digits = values[2] if len(values) > 2 else None
        total = len(subset)
        if total == 0:
            continue
        for budget in budgets:
            completed = subset["token_count_output"] <= budget
            correct_completed = completed & subset["parsed_answer_correct"].astype(bool)
            rows.append(
                {
                    "model_name": model_name,
                    "prompt_mode": prompt_mode,
                    "n_digits": n_digits,
                    "token_budget": budget,
                    "completion_rate": completed.mean(),
                    "correct_completion_rate": correct_completed.mean(),
                    "correct_completion_rate_ci_low": _binomial_ci_low(
                        int(correct_completed.sum()),
                        total,
                    ),
                    "correct_completion_rate_ci_high": _binomial_ci_high(
                        int(correct_completed.sum()),
                        total,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _binomial_ci_low(successes: int, total: int, z_score: float = 1.96) -> float:
    """Return the lower Wilson interval bound for a binomial proportion."""
    low, _ = _binomial_ci(successes, total, z_score)
    return low


def _binomial_ci_high(successes: int, total: int, z_score: float = 1.96) -> float:
    """Return the upper Wilson interval bound for a binomial proportion."""
    _, high = _binomial_ci(successes, total, z_score)
    return high


def _binomial_ci(successes: int, total: int, z_score: float = 1.96) -> tuple[float, float]:
    """Return Wilson interval bounds for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z_squared = z_score**2
    denominator = 1 + z_squared / total
    center = proportion + z_squared / (2 * total)
    margin = z_score * (
        (proportion * (1 - proportion) + z_squared / (4 * total)) / total
    ) ** 0.5
    return max(0.0, (center - margin) / denominator), min(
        1.0,
        (center + margin) / denominator,
    )


def _digit_format_comparison(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if df.empty or "digit_format" not in df:
        return None
    plot_df = _with_display_labels(df, ["digit_format"])
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    sns.barplot(
        data=plot_df,
        x="digit_format",
        y="parsed_answer_correct",
        hue="model_name",
        errorbar=COMPARISON_ERRORBAR,
        palette=_model_palette(plot_df),
        capsize=0.08,
        err_kws=COMPARISON_ERR_KWS,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Digit Format")
    ax.set_ylabel("Parsed Answer Accuracy")
    ax.grid(axis="x", visible=False)
    _format_legend(ax)
    sns.despine(fig=fig, ax=ax)
    path = output_dir / "digit_format_comparison.png"
    return _save_figure(fig, path)


def _answer_format_comparison(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate an accuracy comparison by answer format."""
    if df.empty or "answer_format" not in df:
        return None
    plot_df = _with_display_labels(df, ["answer_format"])
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    sns.barplot(
        data=plot_df,
        x="answer_format",
        y="parsed_answer_correct",
        hue="model_name",
        errorbar=COMPARISON_ERRORBAR,
        palette=_model_palette(plot_df),
        capsize=0.08,
        err_kws=COMPARISON_ERR_KWS,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Answer Format")
    ax.set_ylabel("Parsed Answer Accuracy")
    ax.grid(axis="x", visible=False)
    _format_legend(ax)
    sns.despine(fig=fig, ax=ax)
    path = output_dir / "answer_format_comparison.png"
    return _save_figure(fig, path)


def _slice_type_comparison(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate an accuracy comparison by dataset slice type."""
    if df.empty or "slice_name" not in df:
        return None
    plot_df = _with_display_labels(df, ["slice_name"])
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    sns.barplot(
        data=plot_df,
        x="slice_name",
        y="parsed_answer_correct",
        hue="model_name",
        errorbar=COMPARISON_ERRORBAR,
        palette=_model_palette(plot_df),
        capsize=0.08,
        err_kws=COMPARISON_ERR_KWS,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Slice Type")
    ax.set_ylabel("Parsed Answer Accuracy")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=25)
    _format_legend(ax)
    sns.despine(fig=fig, ax=ax)
    path = output_dir / "slice_type_comparison.png"
    return _save_figure(fig, path)


def _goal2_target_summary_matrices(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate one compact target-by-location probe summary matrix per model."""
    plot_df = _prepare_goal2_metric_df(metrics)
    if plot_df.empty:
        return []
    paths = []
    for model_name, subset in plot_df.groupby("model_name", sort=True):
        table = _goal2_target_summary_table(subset)
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        sns.heatmap(
            table,
            annot=False,
            vmin=0,
            vmax=1,
            cmap=HEATMAP_CMAP,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Probe Accuracy"},
            ax=ax,
        )
        ax.set_xlabel("Token Location")
        ax.set_ylabel("Probe Target")
        ax.tick_params(axis="x", rotation=25)
        paths.append(
            _save_figure(
                fig,
                output_dir
                / f"goal2_target_summary_matrix_{_filename_slug(str(model_name))}.png",
            )
        )
    return paths


def _goal2_target_summary_delta_matrix(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """Generate a target-by-location delta matrix between two models."""
    plot_df = _prepare_goal2_metric_df(metrics)
    model_pair = _goal2_model_delta_pair(plot_df)
    if model_pair is None:
        return None
    minuend, subtrahend = model_pair
    minuend_table = _goal2_target_summary_table(plot_df[plot_df["model_name"] == minuend])
    subtrahend_table = _goal2_target_summary_table(plot_df[plot_df["model_name"] == subtrahend])
    delta = _aligned_delta_table(minuend_table, subtrahend_table)
    if delta.empty:
        return None
    vmin, vmax = _symmetric_heatmap_bounds(delta)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    sns.heatmap(
        delta,
        annot=False,
        center=0,
        vmin=vmin,
        vmax=vmax,
        cmap="vlag",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Delta Probe Accuracy"},
        ax=ax,
    )
    ax.set_xlabel("Token Location")
    ax.set_ylabel("Probe Target")
    ax.tick_params(axis="x", rotation=25)
    filename = f"goal2_target_summary_delta_{_goal2_delta_slug((minuend, subtrahend))}.png"
    return _save_figure(fig, output_dir / filename)


def _goal2_target_summary_matrices_by_prompt_mode(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate target-by-location summary matrices split by prompt mode."""
    plot_df = _goal2_prompt_metric_df(predictions)
    if plot_df.empty:
        return []
    paths = []
    for (prompt_mode, model_name), subset in plot_df.groupby(
        ["prompt_mode", "model_name"], sort=True
    ):
        table = _goal2_target_summary_table(subset)
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        sns.heatmap(
            table,
            annot=False,
            vmin=0,
            vmax=1,
            cmap=HEATMAP_CMAP,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Probe Accuracy"},
            ax=ax,
        )
        ax.set_xlabel("Token Location")
        ax.set_ylabel("Probe Target")
        ax.tick_params(axis="x", rotation=25)
        filename = (
            f"goal2_target_summary_matrix_{_filename_slug(str(model_name))}_"
            f"{_filename_slug(str(prompt_mode))}.png"
        )
        paths.append(_save_figure(fig, output_dir / filename))
    return paths


def _goal2_target_summary_delta_matrices_by_prompt_mode(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate target-by-location two-model delta matrices split by prompt mode."""
    plot_df = _goal2_prompt_metric_df(predictions)
    if plot_df.empty:
        return []
    paths = []
    for prompt_mode, subset in plot_df.groupby("prompt_mode", sort=True):
        model_pair = _goal2_model_delta_pair(subset)
        if model_pair is None:
            continue
        minuend, subtrahend = model_pair
        minuend_table = _goal2_target_summary_table(subset[subset["model_name"] == minuend])
        subtrahend_table = _goal2_target_summary_table(
            subset[subset["model_name"] == subtrahend]
        )
        delta = _aligned_delta_table(minuend_table, subtrahend_table)
        if delta.empty:
            continue
        vmin, vmax = _symmetric_heatmap_bounds(delta)
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        sns.heatmap(
            delta,
            annot=False,
            center=0,
            vmin=vmin,
            vmax=vmax,
            cmap="vlag",
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Delta Probe Accuracy"},
            ax=ax,
        )
        ax.set_xlabel("Token Location")
        ax.set_ylabel("Probe Target")
        ax.tick_params(axis="x", rotation=25)
        filename = (
            f"goal2_target_summary_delta_{_goal2_delta_slug(model_pair)}_"
            f"{_filename_slug(str(prompt_mode))}.png"
        )
        paths.append(_save_figure(fig, output_dir / filename))
    return paths


def _goal2_target_summary_table(subset: pd.DataFrame) -> pd.DataFrame:
    """Return a target-by-location best-layer accuracy table."""
    if subset.empty:
        return pd.DataFrame()
    best = _goal2_best_layer_by_column(subset)
    summary = (
        best.groupby(["target", "target_label", "location_order", "location_label"])
        ["test_accuracy"]
        .mean()
        .reset_index()
        .sort_values(["target", "location_order"])
    )
    table = summary.pivot_table(
        index=["target", "target_label"],
        columns=["location_order", "location_label"],
        values="test_accuracy",
        aggfunc="mean",
    )
    if table.empty:
        return table
    return _order_goal2_target_table(table)


def _order_goal2_target_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a target-summary table with readable ordered labels."""
    table = table.sort_index(
        level=0,
        key=lambda values: values.map(_target_sort_key),
    )
    table.index = [label for _, label in table.index]
    table.columns = [label for _, label in table.columns]
    return table


def _target_sort_key(value: object) -> int:
    """Return the standard sort key for a Goal 2 probe target."""
    target_order = {target: index for index, target in enumerate(GOAL2_TARGET_ORDER)}
    return target_order.get(str(value), 999)


def _goal2_reasoning_time_by_column(metrics: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate reasoning-time carry-chain decoding curves by model."""
    plot_df = _prepare_goal2_metric_df(metrics)
    plot_df = plot_df[
        (plot_df["target"] == "carry_chain_membership")
        & plot_df["target_column_lsd"].notna()
        & plot_df["location_kind"].isin(GOAL2_TIMING_LOCATION_ORDER)
    ].copy()
    if plot_df.empty:
        return None
    best = _goal2_best_layer_by_column(plot_df)
    best = best.sort_values(["model_name", "location_order"])
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    sns.lineplot(
        data=best,
        x="location_label",
        y="test_accuracy",
        hue="model_name",
        marker="o",
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Reasoning Location")
    ax.set_ylabel("Best-Layer Probe Accuracy")
    ax.tick_params(axis="x", rotation=25)
    _format_legend(ax)
    sns.despine(fig=fig, ax=ax)
    return _save_figure(fig, output_dir / "goal2_reasoning_time_by_column.png")


def _goal2_reasoning_time_by_column_delta(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """Generate a reasoning-time carry-chain model-delta curve."""
    plot_df = _prepare_goal2_metric_df(metrics)
    plot_df = plot_df[
        (plot_df["target"] == "carry_chain_membership")
        & plot_df["target_column_lsd"].notna()
        & plot_df["location_kind"].isin(GOAL2_TIMING_LOCATION_ORDER)
    ].copy()
    if plot_df.empty:
        return None
    best = _goal2_best_layer_by_column(plot_df)
    delta, model_pair = _goal2_delta_rows(
        best,
        keys=[
            "target_column_lsd",
            "location_kind",
            "location_label",
            "location_order",
        ],
        value_column="test_accuracy",
    )
    if delta.empty or model_pair is None:
        return None
    delta = (
        delta.groupby(["location_kind", "location_label", "location_order"], as_index=False)
        ["delta_accuracy"]
        .mean()
        .sort_values("location_order")
    )
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    sns.lineplot(
        data=delta,
        x="location_label",
        y="delta_accuracy",
        marker="o",
        errorbar=None,
        ax=ax,
    )
    ax.axhline(0, color="#555555", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Reasoning Location")
    ax.set_ylabel("Delta Best-Layer Probe Accuracy")
    ax.tick_params(axis="x", rotation=25)
    _format_legend(ax)
    sns.despine(fig=fig, ax=ax)
    filename = f"goal2_reasoning_time_by_column_delta_{_goal2_delta_slug(model_pair)}.png"
    return _save_figure(fig, output_dir / filename)


def _goal2_layer_profile_by_target_free_cot(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """Generate free-CoT target-faceted layer profiles for selected locations."""
    plot_df = _goal2_prompt_metric_df(predictions)
    if plot_df.empty:
        return None
    plot_df = plot_df[plot_df["prompt_mode"] == "free_cot"].copy()
    return _goal2_layer_profile_plot_from_metric_df(
        plot_df,
        output_dir,
        filename="goal2_layer_profile_by_target_free_cot.png",
    )


def _goal2_layer_profile_plot_from_metric_df(
    metrics: pd.DataFrame,
    output_dir: Path,
    filename: str,
) -> Path | None:
    """Generate target-faceted layer profiles from prepared metric rows."""
    plot_df = _goal2_layer_profile_data(metrics)
    if plot_df.empty:
        return None
    targets = (
        plot_df[["target", "target_label"]]
        .drop_duplicates()
        .sort_values("target", key=lambda values: values.map(_target_sort_key))
    )
    target_labels = list(targets["target_label"])
    ncols = min(3, len(target_labels))
    nrows = (len(target_labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.15 * ncols, 2.65 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    palette = _model_palette(plot_df)
    markers = _goal2_layer_profile_markers(plot_df)
    for ax, target_label in zip(axes.flat, target_labels, strict=False):
        subset = plot_df[plot_df["target_label"] == target_label]
        for (model_name, location_label), group in subset.groupby(
            ["model_name", "location_label"], sort=True
        ):
            group = group.sort_values("layer_order")
            ax.plot(
                group["layer_order"],
                group["test_accuracy"],
                color=palette.get(model_name, MODEL_PALETTE[0]),
                marker=markers.get(location_label, "o"),
                markevery=GOAL2_LAYER_PROFILE_MARK_EVERY,
                linewidth=1.5,
                markersize=4.5,
            )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(
            0.03,
            0.95,
            target_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="semibold",
        )
        sns.despine(fig=fig, ax=ax)
    for ax in axes.flat[len(target_labels) :]:
        ax.set_visible(False)
    fig.supxlabel("Layer", y=0.03)
    fig.supylabel("Probe Accuracy", x=0.03)
    fig.legend(
        handles=[
            Line2D([], [], linestyle="none", label="Model Name"),
            *[
                Line2D([0], [0], color=color, linewidth=1.8, label=_display_label(model_name))
                for model_name, color in palette.items()
            ],
            Line2D([], [], linestyle="none", label="Token Location"),
            *[
                Line2D(
                    [0],
                    [0],
                    color="#333333",
                    marker=markers.get(location_label, "o"),
                    linewidth=1.2,
                    markersize=4.5,
                    label=location_label,
                )
                for location_label in _available_location_labels(plot_df)
            ],
        ],
        loc="center left",
        bbox_to_anchor=(0.805, 0.54),
        handlelength=1.8,
        labelspacing=0.55,
        borderaxespad=0,
    )
    return _save_figure_with_legend_space(
        fig,
        output_dir / filename,
    )


def _goal2_layer_profile_data(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return layer-profile rows averaged over target columns."""
    plot_df = _prepare_goal2_metric_df(metrics)
    plot_df = plot_df[plot_df["location_kind"].isin(GOAL2_LAYER_PROFILE_LOCATIONS)].copy()
    if plot_df.empty:
        return plot_df
    rows = []
    for (model_name, target, target_label), subset in plot_df.groupby(
        ["model_name", "target", "target_label"], sort=True
    ):
        grouped = _goal2_weighted_metric_rows(subset, "test_accuracy")
        if grouped.empty:
            continue
        grouped["model_name"] = model_name
        grouped["target"] = target
        grouped["target_label"] = target_label
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    profile = pd.concat(rows, ignore_index=True)
    return profile.sort_values(
        ["target", "model_name", "location_order", "layer_order"],
        key=lambda values: values.map(_target_sort_key) if values.name == "target" else values,
    )


def _goal2_layer_profile_markers(plot_df: pd.DataFrame) -> dict[str, str]:
    """Return marker shapes for layer-profile locations."""
    marker_cycle = ["o", "s", "D", "^", "v", "P"]
    locations = _available_location_labels(plot_df)
    return dict(zip(locations, marker_cycle, strict=False))


def _legend_labels(legend: object) -> list[str]:
    """Return text labels from a matplotlib legend."""
    return [text.get_text() for text in legend.get_texts()]


def _available_location_labels(df: pd.DataFrame) -> list[str]:
    """Return location labels present in data using the standard location order."""
    locations = (
        df[["location_order", "location_label"]]
        .drop_duplicates()
        .sort_values("location_order")
    )
    return list(locations["location_label"])


def _goal2_location_legend(fig: plt.Figure, location_labels: list[str]) -> None:
    """Add a compact location legend to a multi-panel Goal 2 figure."""
    if not location_labels:
        return
    palette = sns.color_palette(n_colors=len(location_labels))
    handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=label)
        for label, color in zip(location_labels, palette, strict=True)
    ]
    fig.legend(
        handles=handles,
        title="Token Location",
        loc="center left",
        bbox_to_anchor=(0.805, 0.54),
        handlelength=1.8,
        labelspacing=0.55,
        borderaxespad=0,
    )


def _goal2_probe_heatmaps(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate layer-by-location Goal 2 probe heatmaps per model and target."""
    plot_df = _prepare_goal2_metric_df(metrics)
    paths = []
    for (model_name, target), subset in plot_df.groupby(["model_name", "target"], sort=True):
        table = _goal2_metric_table(subset)
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
        sns.heatmap(
            table,
            annot=False,
            vmin=0,
            vmax=1,
            cmap=HEATMAP_CMAP,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Probe Accuracy"},
            ax=ax,
        )
        ax.set_xlabel("Token Location")
        ax.set_ylabel("Layer")
        ax.tick_params(axis="x", rotation=25)
        filename = (
            f"linear_probe_heatmap_{_filename_slug(str(model_name))}_"
            f"{_filename_slug(str(target))}.png"
        )
        paths.append(_save_figure(fig, output_dir / filename))
    return paths


def _goal2_probe_delta_heatmaps(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate layer-by-location delta heatmaps between two models."""
    plot_df = _prepare_goal2_metric_df(metrics)
    model_pair = _goal2_model_delta_pair(plot_df)
    if model_pair is None:
        return []
    minuend, subtrahend = model_pair
    paths = []
    for target, subset in plot_df.groupby("target", sort=True):
        minuend_table = _goal2_metric_table(subset[subset["model_name"] == minuend])
        subtrahend_table = _goal2_metric_table(subset[subset["model_name"] == subtrahend])
        delta = _aligned_delta_table(minuend_table, subtrahend_table)
        if delta.empty:
            continue
        vmin, vmax = _symmetric_heatmap_bounds(delta)
        fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
        sns.heatmap(
            delta,
            annot=False,
            center=0,
            vmin=vmin,
            vmax=vmax,
            cmap="vlag",
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Delta Probe Accuracy"},
            ax=ax,
        )
        ax.set_xlabel("Token Location")
        ax.set_ylabel("Layer")
        ax.tick_params(axis="x", rotation=25)
        filename = (
            f"linear_probe_delta_heatmap_{_filename_slug(str(minuend))}"
            f"_minus_{_filename_slug(str(subtrahend))}_{_filename_slug(str(target))}.png"
        )
        paths.append(_save_figure(fig, output_dir / filename))
    return paths


def _goal2_metric_table(subset: pd.DataFrame) -> pd.DataFrame:
    """Return a layer-by-location mean accuracy table for Goal 2 probe metrics."""
    grouped = _goal2_weighted_metric_rows(subset, "test_accuracy")
    table = grouped.pivot_table(
        index=["layer_order", "layer_label"],
        columns=["location_order", "location_label"],
        values="test_accuracy",
        aggfunc="mean",
    )
    if table.empty:
        return table
    table.index = [label for _, label in table.index]
    table.columns = [label for _, label in table.columns]
    return table


def _goal2_weighted_metric_rows(subset: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Return layer/location metric rows averaged over target columns."""
    if subset.empty:
        return pd.DataFrame()
    keys = ["layer_order", "layer_label", "location_order", "location_label"]
    working = subset.copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])
    if working.empty:
        return pd.DataFrame(columns=[*keys, value_column])
    if "test_examples" not in working:
        return (
            working.groupby(keys, as_index=False)[value_column]
            .mean()
            .sort_values(["layer_order", "location_order"])
        )
    working["_metric_weight"] = pd.to_numeric(
        working["test_examples"],
        errors="coerce",
    ).fillna(0)
    working = working[working["_metric_weight"] > 0].copy()
    if working.empty:
        return pd.DataFrame(columns=[*keys, value_column])
    working["_weighted_metric"] = working[value_column] * working["_metric_weight"]
    grouped = (
        working.groupby(keys, as_index=False)[["_weighted_metric", "_metric_weight"]]
        .sum()
        .sort_values(["layer_order", "location_order"])
    )
    grouped[value_column] = grouped["_weighted_metric"] / grouped["_metric_weight"]
    return grouped.drop(columns=["_weighted_metric", "_metric_weight"])


def _goal2_probe_timing_curves(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate reasoning-progress probe accuracy curves using best layer per location."""
    plot_df = _prepare_goal2_metric_df(metrics)
    plot_df = plot_df[plot_df["location_kind"].isin(GOAL2_TIMING_LOCATION_ORDER)].copy()
    if plot_df.empty:
        return []
    best = (
        plot_df.groupby(
            [
                "model_name",
                "target",
                "target_column_lsd",
                "location_kind",
                "location_label",
                "location_order",
            ],
            dropna=False,
            as_index=False,
        )["test_accuracy"]
        .max()
        .sort_values(["target", "model_name", "location_order"])
    )
    paths = []
    for target, subset in best.groupby("target", sort=True):
        fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
        sns.lineplot(
            data=subset,
            x="location_label",
            y="test_accuracy",
            hue="model_name",
            marker="o",
            errorbar=None,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Reasoning Location")
        ax.set_ylabel("Best-Layer Probe Accuracy")
        ax.tick_params(axis="x", rotation=25)
        _format_legend(ax)
        sns.despine(fig=fig, ax=ax)
        paths.append(
            _save_figure(
                fig,
                output_dir / f"linear_probe_timing_{_filename_slug(str(target))}.png",
            )
        )
    return paths


def _filter_goal2_standard_formats(df: pd.DataFrame) -> pd.DataFrame:
    """Return only standard-format Goal 2 rows when format metadata is available."""
    return _filter_goal2_formats(
        df,
        GOAL2_STANDARD_DIGIT_FORMAT,
        GOAL2_STANDARD_ANSWER_FORMAT,
    )


def _filter_goal2_formats(
    df: pd.DataFrame,
    digit_format: str,
    answer_format: str,
) -> pd.DataFrame:
    """Return only rows matching the requested Goal 2 digit and answer formats."""
    if df.empty:
        return df
    filtered = df
    if "digit_format" in filtered:
        filtered = filtered[filtered["digit_format"] == digit_format]
    if "answer_format" in filtered:
        filtered = filtered[filtered["answer_format"] == answer_format]
    return filtered.copy() if filtered is not df else df


def _has_goal2_format_metadata(df: pd.DataFrame) -> bool:
    """Return whether Goal 2 rows include digit or answer format metadata."""
    return "digit_format" in df.columns or "answer_format" in df.columns


def _goal2_shared_model_example_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return prediction rows whose example IDs are present for every model."""
    if predictions.empty or not {"model_name", "example_id"}.issubset(predictions.columns):
        return predictions
    model_names = predictions["model_name"].dropna().unique()
    if len(model_names) < 2:
        return predictions
    model_examples = predictions[["model_name", "example_id"]].dropna().drop_duplicates()
    model_counts = model_examples.groupby("example_id")["model_name"].nunique()
    shared_example_ids = model_counts[model_counts == len(model_names)].index
    return predictions[predictions["example_id"].isin(shared_example_ids)].copy()


def _goal2_prediction_metric_df(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate Goal 2 probe metrics from per-example predictions."""
    scored = _goal2_scored_prediction_df(predictions)
    if scored.empty:
        return pd.DataFrame()
    rows = []
    group_columns = [
        "model_name",
        "target",
        "target_column_lsd",
        "location_kind",
        "layer_index",
    ]
    for group_values, subset in scored.groupby(group_columns, dropna=False, sort=True):
        values = group_values if isinstance(group_values, tuple) else (group_values,)
        y_true = subset["y_true"].dropna()
        if y_true.empty:
            continue
        rows.append(
            {
                "model_name": values[0],
                "target": values[1],
                "target_column_lsd": values[2],
                "location_kind": values[3],
                "layer_index": values[4],
                "status": "fitted",
                "test_accuracy": float(subset["probe_correct"].fillna(False).mean()),
                "test_majority_baseline": _series_majority_baseline(y_true),
                "test_examples": int(len(subset)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return _prepare_goal2_metric_df(pd.DataFrame(rows))


def _goal2_scored_prediction_df(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return prediction rows with required scoring columns normalized."""
    if predictions.empty:
        return pd.DataFrame()
    plot_df = predictions.copy()
    if "probe_correct" not in plot_df and {"y_true", "y_pred"}.issubset(plot_df.columns):
        plot_df["probe_correct"] = plot_df["y_true"] == plot_df["y_pred"]
    required = {
        "model_name",
        "target",
        "location_kind",
        "layer_index",
        "y_true",
        "probe_correct",
    }
    if not required.issubset(plot_df.columns):
        return pd.DataFrame()
    if "target_column_lsd" not in plot_df:
        plot_df["target_column_lsd"] = None
    return plot_df


def _goal2_prompt_metric_df(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return prompt-mode-specific probe metrics from per-example predictions."""
    plot_df = _goal2_scored_prediction_df(predictions)
    if plot_df.empty:
        return pd.DataFrame()
    required = {
        "prompt_mode",
    }
    if not required.issubset(plot_df.columns):
        return pd.DataFrame()
    rows = []
    group_columns = [
        "model_name",
        "prompt_mode",
        "target",
        "target_column_lsd",
        "location_kind",
        "layer_index",
    ]
    for group_values, subset in plot_df.groupby(group_columns, dropna=False, sort=True):
        values = group_values if isinstance(group_values, tuple) else (group_values,)
        y_true = subset["y_true"].dropna()
        if y_true.empty:
            continue
        rows.append(
            {
                "model_name": values[0],
                "prompt_mode": values[1],
                "target": values[2],
                "target_column_lsd": values[3],
                "location_kind": values[4],
                "layer_index": values[5],
                "status": "fitted",
                "test_accuracy": float(subset["probe_correct"].fillna(False).mean()),
                "test_majority_baseline": _series_majority_baseline(y_true),
                "test_examples": int(len(subset)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return _prepare_goal2_metric_df(pd.DataFrame(rows))


def _series_majority_baseline(values: pd.Series) -> float:
    """Return the majority-class baseline for a non-empty series."""
    counts = values.value_counts(dropna=True)
    if counts.empty:
        return 0.0
    return float(counts.max() / counts.sum())


def _prepare_goal2_metric_df(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return display-ready Goal 2 probe metrics."""
    plot_df = metrics.copy()
    if "target_column_lsd" not in plot_df:
        plot_df["target_column_lsd"] = None
    plot_df["target_label"] = plot_df["target"].map(_display_label)
    plot_df["layer_order"] = plot_df["layer_index"].map(_layer_sort_key)
    plot_df["layer_label"] = plot_df["layer_index"].map(_display_label)
    plot_df["location_order"] = plot_df["location_kind"].map(_location_sort_key)
    plot_df["location_label"] = plot_df["location_kind"].map(_display_label)
    return plot_df.sort_values(["target", "model_name", "location_order", "layer_order"])


def _goal2_model_delta_pair(plot_df: pd.DataFrame) -> tuple[object, object] | None:
    """Return the two-model subtraction order for Goal 2 delta figures."""
    if plot_df.empty or "model_name" not in plot_df:
        return None
    model_names = sorted(plot_df["model_name"].dropna().unique(), key=str)
    if len(model_names) != 2:
        return None
    full_models = [model for model in model_names if "full" in str(model).lower()]
    sft_models = [model for model in model_names if "sft" in str(model).lower()]
    if len(full_models) == 1 and len(sft_models) == 1:
        return full_models[0], sft_models[0]
    return model_names[0], model_names[1]


def _goal2_delta_rows(
    plot_df: pd.DataFrame,
    keys: list[str],
    value_column: str,
) -> tuple[pd.DataFrame, tuple[object, object] | None]:
    """Return matched two-model row deltas for the requested grouping keys."""
    model_pair = _goal2_model_delta_pair(plot_df)
    required = {"model_name", value_column, *keys}
    if model_pair is None or plot_df.empty or not required.issubset(plot_df.columns):
        return pd.DataFrame(), None
    minuend, subtrahend = model_pair
    grouped = (
        plot_df.groupby(["model_name", *keys], dropna=False, as_index=False)[value_column]
        .max()
        .copy()
    )
    minuend_rows = (
        grouped[grouped["model_name"] == minuend]
        .drop(columns=["model_name"])
        .rename(columns={value_column: "_minuend_value"})
    )
    subtrahend_rows = (
        grouped[grouped["model_name"] == subtrahend]
        .drop(columns=["model_name"])
        .rename(columns={value_column: "_subtrahend_value"})
    )
    delta = minuend_rows.merge(subtrahend_rows, on=keys, how="inner")
    if delta.empty:
        return delta, model_pair
    delta["delta_accuracy"] = delta["_minuend_value"] - delta["_subtrahend_value"]
    return delta.drop(columns=["_minuend_value", "_subtrahend_value"]), model_pair


def _goal2_delta_slug(model_pair: tuple[object, object]) -> str:
    """Return a filename-safe label for a Goal 2 model delta."""
    minuend, subtrahend = model_pair
    return f"{_filename_slug(str(minuend))}_minus_{_filename_slug(str(subtrahend))}"


def _aligned_delta_table(
    minuend_table: pd.DataFrame,
    subtrahend_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return a table difference over shared rows and columns only."""
    if minuend_table.empty or subtrahend_table.empty:
        return pd.DataFrame()
    minuend_aligned, subtrahend_aligned = minuend_table.align(
        subtrahend_table,
        join="inner",
        axis=None,
    )
    delta = minuend_aligned - subtrahend_aligned
    return delta.dropna(how="all").dropna(axis=1, how="all")


def _symmetric_heatmap_bounds(table: pd.DataFrame) -> tuple[float, float]:
    """Return symmetric color bounds around zero for a delta heatmap."""
    max_abs = table.abs().max().max()
    if pd.isna(max_abs) or max_abs == 0:
        max_abs = 1.0
    return -float(max_abs), float(max_abs)


def _goal2_best_layer_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return best-layer probe rows for each model, target, column, and location."""
    group_columns = [
        "model_name",
        "target",
        "target_column_lsd",
        "location_kind",
        "location_label",
        "location_order",
    ]
    best_indexes = df.groupby(group_columns, dropna=False)["test_accuracy"].idxmax()
    return df.loc[best_indexes].copy()


def _goal2_best_layer_by_prompt_and_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return best-layer probe rows for each prompt, target, column, and location."""
    group_columns = [
        "model_name",
        "prompt_mode",
        "target",
        "target_column_lsd",
        "location_kind",
        "location_label",
        "location_order",
    ]
    best_indexes = df.groupby(group_columns, dropna=False)["test_accuracy"].idxmax()
    return df.loc[best_indexes].copy()


def _layer_sort_key(value: object) -> int:
    """Return a numeric sort key for a saved layer index."""
    if value == "embedding":
        return -1
    return int(value)


def _location_sort_key(value: object) -> int:
    """Return a stable sort key for a Goal 2 activation location kind."""
    text = str(value)
    if text in GOAL2_LOCATION_ORDER:
        return GOAL2_LOCATION_ORDER.index(text)
    return len(GOAL2_LOCATION_ORDER)
