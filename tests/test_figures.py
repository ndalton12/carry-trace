from pathlib import Path

import pandas as pd

import carry_trace.figures as figures
from carry_trace.io import write_jsonl


def test_goal1_figures_exclude_token_limit_hits_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Goal 1 figures plot only valid generations by default."""
    run_dir = _write_figure_run(tmp_path)
    row_counts: list[int] = []

    def fake_plot(df: pd.DataFrame, output_dir: Path) -> Path:
        """Record plotted row counts without rendering a figure."""
        row_counts.append(len(df))
        return output_dir / f"plot_{len(row_counts)}.png"

    for name in (
        "_accuracy_heatmap",
        "_prompt_mode_comparison",
        "_digit_format_comparison",
        "_first_wrong_digit_plot",
        "_token_count_vs_accuracy",
    ):
        monkeypatch.setattr(figures, name, fake_plot)

    paths = figures.make_goal1_figures(run_dir)

    assert len(paths) == 5
    assert row_counts == [1, 1, 1, 1, 1]


def test_goal1_figures_can_include_token_limit_hits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Goal 1 figures can opt into plotting token-limit hits."""
    run_dir = _write_figure_run(tmp_path)
    row_counts: list[int] = []

    def fake_plot(df: pd.DataFrame, output_dir: Path) -> Path:
        """Record plotted row counts without rendering a figure."""
        row_counts.append(len(df))
        return output_dir / f"plot_{len(row_counts)}.png"

    for name in (
        "_accuracy_heatmap",
        "_prompt_mode_comparison",
        "_digit_format_comparison",
        "_first_wrong_digit_plot",
        "_token_count_vs_accuracy",
    ):
        monkeypatch.setattr(figures, name, fake_plot)

    paths = figures.make_goal1_figures(run_dir, include_token_limit_hits=True)

    assert len(paths) == 5
    assert row_counts == [2, 2, 2, 2, 2]


def _write_figure_run(tmp_path: Path) -> Path:
    """Write a minimal run directory with one valid and one capped generation."""
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "dataset.jsonl",
        [
            {
                "id": "ex-valid",
                "prompt_mode": "answer_only",
                "digit_format": "standard",
                "slice_name": "no_carry",
                "n_digits": 2,
                "max_carry_chain": 0,
                "first_carry_position": None,
            },
            {
                "id": "ex-capped",
                "prompt_mode": "answer_only",
                "digit_format": "standard",
                "slice_name": "no_carry",
                "n_digits": 2,
                "max_carry_chain": 0,
                "first_carry_position": None,
            },
        ],
    )
    write_jsonl(
        run_dir / "scored_calls.jsonl",
        [
            {
                "example_id": "ex-valid",
                "model_name": "model",
                "generation_valid": True,
                "hit_token_limit": False,
                "parsed_answer_correct": True,
                "first_wrong_digit_lsd": None,
                "token_count_output": 3,
            },
            {
                "example_id": "ex-capped",
                "model_name": "model",
                "generation_valid": False,
                "hit_token_limit": True,
                "parsed_answer_correct": False,
                "first_wrong_digit_lsd": 0,
                "token_count_output": 1024,
            },
        ],
    )
    return run_dir
