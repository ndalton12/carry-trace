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
        ax.plot(
            subset["token_budget"],
            subset["correct_completion_rate"],
            color=color,
            linestyle=prompt_styles[prompt_mode],
            linewidth=1.7,
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
                }
            )
    return pd.DataFrame(rows)


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
