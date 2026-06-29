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

    def fake_plot(df: pd.DataFrame, output_dir: Path, *args: object, **kwargs: object) -> Path:
        """Record plotted row counts without rendering a figure."""
        row_counts.append(len(df))
        return output_dir / f"plot_{len(row_counts)}.png"

    for name in (
        "_accuracy_heatmap",
        "_prompt_mode_comparison",
        "_digit_format_comparison",
        "_error_localization_plot",
        "_token_budget_curves",
    ):
        monkeypatch.setattr(figures, name, fake_plot)

    paths = figures.make_goal1_figures(run_dir)

    assert len(paths) == 6
    assert row_counts == [1, 1, 1, 1, 1, 1]


def test_goal1_figures_can_include_token_limit_hits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Goal 1 figures can opt into plotting token-limit hits."""
    run_dir = _write_figure_run(tmp_path)
    row_counts: list[int] = []

    def fake_plot(df: pd.DataFrame, output_dir: Path, *args: object, **kwargs: object) -> Path:
        """Record plotted row counts without rendering a figure."""
        row_counts.append(len(df))
        return output_dir / f"plot_{len(row_counts)}.png"

    for name in (
        "_accuracy_heatmap",
        "_prompt_mode_comparison",
        "_digit_format_comparison",
        "_error_localization_plot",
        "_token_budget_curves",
    ):
        monkeypatch.setattr(figures, name, fake_plot)

    paths = figures.make_goal1_figures(run_dir, include_token_limit_hits=True)

    assert len(paths) == 6
    assert row_counts == [2, 2, 2, 2, 2, 2]


def test_goal1_figures_add_per_model_accuracy_heatmaps(tmp_path: Path) -> None:
    """Verify Goal 1 figures include combined and per-model accuracy heatmaps."""
    run_dir = _write_figure_run(tmp_path)
    write_jsonl(
        run_dir / "scored_calls.jsonl",
        [
            {
                "example_id": "ex-valid",
                "model_name": "olmo/3 32b instruct",
                "generation_valid": True,
                "hit_token_limit": False,
                "parsed_answer_correct": True,
                "first_wrong_digit_lsd": None,
                "token_count_output": 3,
            },
            {
                "example_id": "ex-valid",
                "model_name": "qwen-rlvr",
                "generation_valid": True,
                "hit_token_limit": False,
                "parsed_answer_correct": False,
                "first_wrong_digit_lsd": 0,
                "token_count_output": 4,
            },
        ],
    )

    paths = figures.make_goal1_figures(run_dir)
    filenames = {path.name for path in paths}

    assert "accuracy_heatmap.png" in filenames
    assert "accuracy_heatmap_olmo_3_32b_instruct.png" in filenames
    assert "accuracy_heatmap_qwen-rlvr.png" in filenames


def test_comparison_figures_use_95_percent_confidence_intervals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify comparison bar plots request 95 percent confidence intervals."""
    calls: list[dict[str, object]] = []

    def fake_barplot(*args: object, **kwargs: object) -> None:
        """Record barplot errorbar settings."""
        calls.append(kwargs)

    monkeypatch.setattr(figures.sns, "barplot", fake_barplot)
    df = pd.DataFrame(
        [
            {
                "prompt_mode": "answer_only",
                "digit_format": "standard",
                "model_name": "model",
                "parsed_answer_correct": True,
            },
            {
                "prompt_mode": "free_cot",
                "digit_format": "delimited",
                "model_name": "model",
                "parsed_answer_correct": False,
            },
        ],
    )

    figures._prompt_mode_comparison(df, tmp_path)
    figures._digit_format_comparison(df, tmp_path)

    assert [call["errorbar"] for call in calls] == [("ci", 95), ("ci", 95)]
    assert [call["err_kws"] for call in calls] == [
        {"color": "#222222", "linewidth": 1.1},
        {"color": "#222222", "linewidth": 1.1},
    ]


def test_figure_display_labels_are_human_readable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify figure categories use presentation labels instead of raw enum values."""
    calls: list[pd.DataFrame] = []

    def fake_barplot(*args: object, **kwargs: object) -> None:
        """Record plotted data passed to seaborn."""
        calls.append(kwargs["data"])

    monkeypatch.setattr(figures.sns, "barplot", fake_barplot)
    df = pd.DataFrame(
        [
            {
                "prompt_mode": "answer_only",
                "digit_format": "standard",
                "model_name": "model",
                "parsed_answer_correct": True,
            },
            {
                "prompt_mode": "free_cot",
                "digit_format": "delimited",
                "model_name": "model",
                "parsed_answer_correct": False,
            },
        ],
    )

    figures._prompt_mode_comparison(df, tmp_path)
    figures._digit_format_comparison(df, tmp_path)

    assert list(calls[0]["prompt_mode"]) == ["Answer Only", "Free CoT"]
    assert list(calls[1]["digit_format"]) == ["Standard", "Delimited"]
    assert figures._display_label("model_name") == "Model Name"


def test_token_budget_curve_data_counts_completed_and_correct() -> None:
    """Verify token-budget curves distinguish completion from correctness."""
    df = pd.DataFrame(
        [
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "token_count_output": 10,
                "parsed_answer_correct": True,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "token_count_output": 20,
                "parsed_answer_correct": False,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "token_count_output": 30,
                "parsed_answer_correct": True,
            },
        ],
    )

    curves = figures._token_budget_curve_data(df)

    assert list(curves["token_budget"]) == [10, 20, 30]
    assert list(curves["completion_rate"]) == [1 / 3, 2 / 3, 1.0]
    assert list(curves["correct_completion_rate"]) == [1 / 3, 1 / 3, 2 / 3]


def test_error_localization_data_buckets_first_wrong_digit() -> None:
    """Verify error localization uses parseable incorrect carry examples only."""
    df = pd.DataFrame(
        [
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "generation_valid": True,
                "parsed_output": "1234",
                "parsed_answer_correct": True,
                "first_wrong_digit_lsd": None,
                "first_carry_position": 2,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "generation_valid": True,
                "parsed_output": "1234",
                "parsed_answer_correct": False,
                "first_wrong_digit_lsd": 1,
                "first_carry_position": 2,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "generation_valid": True,
                "parsed_output": "1234",
                "parsed_answer_correct": False,
                "first_wrong_digit_lsd": 2,
                "first_carry_position": 2,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "generation_valid": True,
                "parsed_output": "1234",
                "parsed_answer_correct": False,
                "first_wrong_digit_lsd": 3,
                "first_carry_position": 2,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "generation_valid": True,
                "parsed_output": None,
                "parsed_answer_correct": False,
                "first_wrong_digit_lsd": 0,
                "first_carry_position": 2,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "generation_valid": False,
                "parsed_output": "1234",
                "parsed_answer_correct": False,
                "first_wrong_digit_lsd": 0,
                "first_carry_position": 2,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "generation_valid": True,
                "parsed_output": "1234",
                "parsed_answer_correct": False,
                "first_wrong_digit_lsd": 0,
                "first_carry_position": None,
            },
        ],
    )

    localized = figures._error_localization_data(df)

    assert localized.loc[0, "Before First Carry"] == 1 / 3
    assert localized.loc[0, "At First Carry"] == 1 / 3
    assert localized.loc[0, "After First Carry"] == 1 / 3
    assert localized.loc[0, "localized_error_count"] == 3
    assert localized.loc[0, "group_label"] == "Model / Answer Only (n=3)"


def test_token_budget_curve_data_groups_by_digit_length_by_default() -> None:
    """Verify token-budget curves split by digit length unless disabled."""
    df = pd.DataFrame(
        [
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "n_digits": 2,
                "token_count_output": 10,
                "parsed_answer_correct": True,
            },
            {
                "model_name": "model",
                "prompt_mode": "answer_only",
                "n_digits": 4,
                "token_count_output": 20,
                "parsed_answer_correct": False,
            },
        ],
    )

    split_curves = figures._token_budget_curve_data(df)
    pooled_curves = figures._token_budget_curve_data(df, by_digit_length=False)

    assert set(split_curves["n_digits"]) == {2, 4}
    assert list(pooled_curves["n_digits"]) == [None, None]
    assert list(pooled_curves["correct_completion_rate"]) == [1 / 2, 1 / 2]


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
