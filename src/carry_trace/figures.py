"""Figure generation from saved Goal 1 artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from carry_trace.io import ensure_dir, read_jsonl


def make_goal1_figures(
    run_dir: Path,
    output_dir: Path | None = None,
    include_token_limit_hits: bool = False,
) -> list[Path]:
    """Generate Goal 1 figures, excluding token-limit hits unless requested."""
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
        _prompt_mode_comparison(merged, output_dir),
        _digit_format_comparison(merged, output_dir),
        _first_wrong_digit_plot(merged, output_dir),
        _token_count_vs_accuracy(merged, output_dir),
    ]
    return [path for path in paths if path is not None]


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


def _accuracy_heatmap(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if df.empty:
        return None
    table = df.pivot_table(
        index="n_digits",
        columns="max_carry_chain",
        values="parsed_answer_correct",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(table, annot=True, vmin=0, vmax=1, cmap="viridis", ax=ax)
    ax.set_title("Accuracy by digits and carry-chain length")
    ax.set_xlabel("Max carry-chain length")
    ax.set_ylabel("Digits")
    path = output_dir / "accuracy_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _prompt_mode_comparison(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="prompt_mode",
        y="parsed_answer_correct",
        hue="model_name",
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Prompt-mode comparison")
    ax.set_xlabel("Prompt mode")
    ax.set_ylabel("Parsed-answer accuracy")
    ax.tick_params(axis="x", rotation=20)
    path = output_dir / "prompt_mode_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _first_wrong_digit_plot(df: pd.DataFrame, output_dir: Path) -> Path | None:
    subset = df.dropna(subset=["first_wrong_digit_lsd", "first_carry_position"])
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=subset,
        x="first_carry_position",
        y="first_wrong_digit_lsd",
        hue="model_name",
        style="digit_format",
        ax=ax,
    )
    ax.set_title("First wrong digit vs first carry position")
    ax.set_xlabel("First carry-relevant column, LSD index")
    ax.set_ylabel("First wrong digit, LSD index")
    path = output_dir / "first_wrong_digit_vs_carry.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _token_count_vs_accuracy(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="token_count_output",
        y="parsed_answer_correct",
        hue="model_name",
        style="digit_format",
        ax=ax,
    )
    ax.set_title("Token count vs accuracy")
    ax.set_xlabel("Output tokens")
    ax.set_ylabel("Parsed-answer correct")
    path = output_dir / "token_count_vs_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _digit_format_comparison(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if df.empty or "digit_format" not in df:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=df,
        x="digit_format",
        y="parsed_answer_correct",
        hue="model_name",
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Digit-format comparison")
    ax.set_xlabel("Digit format")
    ax.set_ylabel("Parsed-answer accuracy")
    path = output_dir / "digit_format_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
