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
        "_standard_format_accuracy_by_digits",
        "_prompt_mode_comparison",
        "_digit_format_comparison",
        "_answer_format_comparison",
        "_slice_type_comparison",
        "_error_localization_plot",
        "_token_budget_curves",
    ):
        monkeypatch.setattr(figures, name, fake_plot)

    paths = figures.make_goal1_figures(run_dir)

    assert len(paths) == 9
    assert row_counts == [1, 1, 1, 1, 1, 1, 1, 1, 1]


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
        "_standard_format_accuracy_by_digits",
        "_prompt_mode_comparison",
        "_digit_format_comparison",
        "_answer_format_comparison",
        "_slice_type_comparison",
        "_error_localization_plot",
        "_token_budget_curves",
    ):
        monkeypatch.setattr(figures, name, fake_plot)

    paths = figures.make_goal1_figures(run_dir, include_token_limit_hits=True)

    assert len(paths) == 9
    assert row_counts == [2, 2, 2, 2, 2, 2, 2, 2, 2]


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
    assert "standard_format_accuracy_by_digits.png" in filenames
    assert "answer_format_comparison.png" in filenames
    assert "slice_type_comparison.png" in filenames


def test_standard_format_digit_accuracy_filters_nonstandard_formats() -> None:
    """Verify standard digit accuracy excludes format ablation rows."""
    df = pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "n_digits": 2,
                "digit_format": "standard",
                "answer_format": "standard",
                "parsed_answer_correct": True,
            },
            {
                "model_name": "model-a",
                "n_digits": 2,
                "digit_format": "standard",
                "answer_format": "standard",
                "parsed_answer_correct": False,
            },
            {
                "model_name": "model-a",
                "n_digits": 4,
                "digit_format": "delimited",
                "answer_format": "standard",
                "parsed_answer_correct": False,
            },
            {
                "model_name": "model-a",
                "n_digits": 4,
                "digit_format": "standard",
                "answer_format": "lsd",
                "parsed_answer_correct": False,
            },
            {
                "model_name": "model-b",
                "n_digits": 4,
                "digit_format": "standard",
                "answer_format": "standard",
                "parsed_answer_correct": True,
            },
        ]
    )

    accuracy = figures._standard_format_digit_accuracy(df)

    assert list(
        accuracy[["model_name", "n_digits", "correct", "total", "accuracy"]].itertuples(
            index=False
        )
    ) == [
        ("model-a", 2, 1, 2, 0.5),
        ("model-b", 4, 1, 1, 1.0),
    ]
    assert all(accuracy["accuracy_ci_low"] <= accuracy["accuracy"])
    assert all(accuracy["accuracy_ci_high"] >= accuracy["accuracy"])


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
                "answer_format": "standard",
                "slice_name": "no_carry",
                "model_name": "model",
                "parsed_answer_correct": True,
            },
            {
                "prompt_mode": "free_cot",
                "digit_format": "delimited",
                "answer_format": "lsd",
                "slice_name": "long_carry_chain",
                "model_name": "model",
                "parsed_answer_correct": False,
            },
        ],
    )

    figures._prompt_mode_comparison(df, tmp_path)
    figures._digit_format_comparison(df, tmp_path)
    figures._answer_format_comparison(df, tmp_path)
    figures._slice_type_comparison(df, tmp_path)

    assert [call["errorbar"] for call in calls] == [
        ("ci", 95),
        ("ci", 95),
        ("ci", 95),
        ("ci", 95),
    ]
    assert [call["err_kws"] for call in calls] == [
        {"color": "#222222", "linewidth": 1.1},
        {"color": "#222222", "linewidth": 1.1},
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
                "answer_format": "standard",
                "slice_name": "no_carry",
                "model_name": "model",
                "parsed_answer_correct": True,
            },
            {
                "prompt_mode": "free_cot",
                "digit_format": "delimited",
                "answer_format": "lsd",
                "slice_name": "many_9s_no_carry",
                "model_name": "model",
                "parsed_answer_correct": False,
            },
        ],
    )

    figures._prompt_mode_comparison(df, tmp_path)
    figures._digit_format_comparison(df, tmp_path)
    figures._answer_format_comparison(df, tmp_path)
    figures._slice_type_comparison(df, tmp_path)

    assert list(calls[0]["prompt_mode"]) == ["Answer Only", "Free CoT"]
    assert list(calls[1]["digit_format"]) == ["Standard", "Delimited"]
    assert list(calls[2]["answer_format"]) == ["Standard", "LSD"]
    assert list(calls[3]["slice_name"]) == ["No Carry", "Many 9s No Carry"]
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
    assert all(curves["correct_completion_rate_ci_low"] <= curves["correct_completion_rate"])
    assert all(curves["correct_completion_rate_ci_high"] >= curves["correct_completion_rate"])


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


def test_goal2_probe_figures_generate_core_outputs(tmp_path: Path) -> None:
    """Verify Goal 2 probe figures are generated from probe artifacts."""
    probe_dir = tmp_path / "probe"
    write_jsonl(
        probe_dir / "probe_metrics.jsonl",
        [
            _probe_metric("any_carry", "prompt_final", 0, 1.0),
            _probe_metric("any_carry", "cot_start", 0, 0.75),
            _probe_metric("any_carry", "cot_end", 1, 0.80),
            _probe_metric("outgoing_carry", "prompt_final", 0, 0.90, target_column_lsd=0),
            _probe_metric("outgoing_carry", "cot_start", 0, 0.80, target_column_lsd=0),
            _probe_metric("outgoing_carry", "cot_end", 1, 0.85, target_column_lsd=0),
            _probe_metric(
                "carry_chain_membership",
                "prompt_final",
                0,
                0.75,
                target_column_lsd=0,
            ),
            _probe_metric(
                "carry_chain_membership",
                "cot_start",
                0,
                0.80,
                target_column_lsd=0,
            ),
            _probe_metric(
                "carry_chain_membership",
                "cot_end",
                1,
                0.90,
                target_column_lsd=0,
            ),
        ],
    )
    write_jsonl(
        probe_dir / "probe_predictions.jsonl",
        [
            _probe_prediction(
                "model",
                "free_cot",
                "any_carry",
                "prompt_final",
                None,
                y_true=1,
                probe_correct=True,
            ),
            _probe_prediction(
                "model",
                "free_cot",
                "carry_chain_membership",
                "prompt_final",
                0,
                y_true=1,
                probe_correct=True,
            ),
            _probe_prediction(
                "model",
                "free_cot",
                "carry_chain_membership",
                "cot_2_3",
                0,
                y_true=1,
                probe_correct=True,
            ),
            _probe_prediction(
                "model",
                "free_cot",
                "outgoing_carry",
                "prompt_final",
                0,
                y_true=1,
                probe_correct=True,
            ),
            _probe_prediction(
                "model",
                "free_cot",
                "outgoing_carry",
                "cot_2_3",
                0,
                y_true=1,
                probe_correct=True,
            ),
            _probe_prediction(
                "model",
                "free_cot",
                "outgoing_carry",
                "answer_digits",
                0,
                y_true=1,
                probe_correct=True,
            ),
        ],
    )
    paths = figures.make_goal2_probe_figures(probe_dir)
    relative_paths = {path.relative_to(probe_dir / "figures").as_posix() for path in paths}

    assert "standard/diagnostics/linear_probe_heatmap_model_any_carry.png" in relative_paths
    assert "standard/diagnostics/linear_probe_timing_any_carry.png" in relative_paths
    assert "standard/summary/goal2_target_summary_matrix_model.png" in relative_paths
    assert "standard/summary/goal2_reasoning_time_by_column.png" in relative_paths
    assert "standard/summary/goal2_layer_profile_by_target_free_cot.png" in relative_paths


def test_goal2_probe_figures_generate_model_delta_outputs(tmp_path: Path) -> None:
    """Verify Goal 2 probe figures include Full-minus-SFT delta plots."""
    probe_dir = tmp_path / "probe"
    write_jsonl(
        probe_dir / "probe_metrics.jsonl",
        [
            _probe_metric(
                "any_carry",
                "prompt_final",
                0,
                0.90,
                model_name="Olmo3-Instruct-Full",
            ),
            _probe_metric(
                "any_carry",
                "prompt_final",
                0,
                0.70,
                model_name="Olmo3-Instruct-Sft",
            ),
        ],
    )

    paths = figures.make_goal2_probe_figures(probe_dir)
    relative_paths = {path.relative_to(probe_dir / "figures").as_posix() for path in paths}

    assert (
        "standard/summary/goal2_target_summary_delta_Olmo3-Instruct-Full"
        "_minus_Olmo3-Instruct-Sft.png"
    ) in relative_paths
    assert (
        "standard/diagnostics/linear_probe_delta_heatmap_Olmo3-Instruct-Full"
        "_minus_Olmo3-Instruct-Sft_any_carry.png"
    ) in relative_paths


def test_goal2_probe_figures_generate_summary_delta_outputs(tmp_path: Path) -> None:
    """Verify Goal 2 summary figures include two-model delta companions."""
    probe_dir = tmp_path / "probe"
    rows = []
    locations = [
        "prompt_final",
        "cot_start",
        "cot_1_3",
        "cot_2_3",
        "cot_end",
        "answer_digits",
    ]
    for model_name, accuracy_offset in (
        ("Olmo3-Instruct-Full", 0.1),
        ("Olmo3-Instruct-Sft", 0.0),
    ):
        for target in ("outgoing_carry", "carry_chain_membership"):
            for location in locations:
                rows.append(
                    _probe_metric(
                        target,
                        location,
                        0,
                        0.70 + accuracy_offset,
                        target_column_lsd=0,
                        model_name=model_name,
                    )
                )
    write_jsonl(probe_dir / "probe_metrics.jsonl", rows)

    paths = figures.make_goal2_probe_figures(probe_dir)
    relative_paths = {path.relative_to(probe_dir / "figures").as_posix() for path in paths}
    delta_slug = "Olmo3-Instruct-Full_minus_Olmo3-Instruct-Sft"

    assert (
        f"standard/summary/goal2_reasoning_time_by_column_delta_{delta_slug}.png"
        in relative_paths
    )


def test_goal2_probe_figures_generate_prompt_mode_split_outputs(tmp_path: Path) -> None:
    """Verify Goal 2 summary figures include prompt-mode split outputs."""
    probe_dir = tmp_path / "probe"
    write_jsonl(
        probe_dir / "probe_metrics.jsonl",
        [
            _probe_metric(
                "outgoing_carry",
                "prompt_final",
                0,
                0.80,
                target_column_lsd=0,
                model_name="Olmo3-Instruct-Full",
            ),
            _probe_metric(
                "outgoing_carry",
                "prompt_final",
                0,
                0.70,
                target_column_lsd=0,
                model_name="Olmo3-Instruct-Sft",
            ),
        ],
    )
    rows = []
    for model_name in ("Olmo3-Instruct-Full", "Olmo3-Instruct-Sft"):
        for prompt_mode in ("answer_only", "free_cot"):
            for location_kind in ("prompt_final", "answer_digits"):
                rows.extend(
                    [
                        _probe_prediction(
                            model_name,
                            prompt_mode,
                            "outgoing_carry",
                            location_kind,
                            0,
                            y_true=0,
                            probe_correct=True,
                        ),
                        _probe_prediction(
                            model_name,
                            prompt_mode,
                            "outgoing_carry",
                            location_kind,
                            0,
                            y_true=1,
                            probe_correct=(prompt_mode == "free_cot"),
                        ),
                    ]
                )
    write_jsonl(probe_dir / "probe_predictions.jsonl", rows)

    paths = figures.make_goal2_probe_figures(probe_dir)
    relative_paths = {path.relative_to(probe_dir / "figures").as_posix() for path in paths}
    delta_slug = "Olmo3-Instruct-Full_minus_Olmo3-Instruct-Sft"

    assert (
        "standard/summary/goal2_target_summary_matrix_Olmo3-Instruct-Full_answer_only.png"
        in relative_paths
    )
    assert (
        f"standard/summary/goal2_target_summary_delta_{delta_slug}_answer_only.png"
        in relative_paths
    )


def test_goal2_probe_figures_generate_format_nested_outputs(tmp_path: Path) -> None:
    """Verify Goal 2 figures are nested by digit and answer format condition."""
    probe_dir = tmp_path / "probe"
    write_jsonl(
        probe_dir / "probe_metrics.jsonl",
        [_probe_metric("any_carry", "prompt_final", 0, 0.80)],
    )
    rows = []
    format_rows = [
        ("standard", "standard", "standard"),
        ("delimited_digit_format", "delimited", "standard"),
        ("lsd_answer_format", "standard", "lsd"),
    ]
    for _, digit_format, answer_format in format_rows:
        rows.extend(
            [
                _probe_prediction(
                    "model",
                    "answer_only",
                    "any_carry",
                    "prompt_final",
                    0,
                    y_true=0,
                    probe_correct=True,
                    digit_format=digit_format,
                    answer_format=answer_format,
                ),
                _probe_prediction(
                    "model",
                    "answer_only",
                    "any_carry",
                    "prompt_final",
                    0,
                    y_true=1,
                    probe_correct=False,
                    digit_format=digit_format,
                    answer_format=answer_format,
                ),
            ]
        )
    write_jsonl(probe_dir / "probe_predictions.jsonl", rows)

    paths = figures.make_goal2_probe_figures(probe_dir)
    relative_paths = {path.relative_to(probe_dir / "figures").as_posix() for path in paths}

    for dirname, _, _ in format_rows:
        assert (
            f"{dirname}/summary/goal2_target_summary_matrix_model.png"
            in relative_paths
        )
        assert (
            f"{dirname}/diagnostics/linear_probe_heatmap_model_any_carry.png"
            in relative_paths
        )


def test_goal2_prediction_metrics_filter_nonstandard_formats() -> None:
    """Verify Goal 2 figure metrics keep only standard digit and answer formats."""
    predictions = pd.DataFrame(
        [
            _probe_prediction(
                "model",
                "answer_only",
                "outgoing_carry",
                "prompt_final",
                0,
                y_true=0,
                probe_correct=True,
            ),
            _probe_prediction(
                "model",
                "answer_only",
                "outgoing_carry",
                "prompt_final",
                0,
                y_true=1,
                probe_correct=False,
                digit_format="delimited",
            ),
            _probe_prediction(
                "model",
                "answer_only",
                "outgoing_carry",
                "prompt_final",
                0,
                y_true=1,
                probe_correct=False,
                answer_format="lsd",
            ),
        ],
    )

    metrics = figures._goal2_prediction_metric_df(
        figures._filter_goal2_standard_formats(predictions)
    )

    assert len(metrics) == 1
    assert int(metrics.loc[0, "test_examples"]) == 1
    assert float(metrics.loc[0, "test_accuracy"]) == 1.0


def test_goal2_delta_uses_full_minus_sft_raw_accuracy() -> None:
    """Verify Goal 2 delta tables subtract SFT accuracy from Full accuracy."""
    full = _probe_metric(
        "any_carry",
        "prompt_final",
        0,
        0.90,
        model_name="Olmo3-Instruct-Full",
    )
    full["test_majority_baseline"] = 0.80
    sft = _probe_metric(
        "any_carry",
        "prompt_final",
        0,
        0.70,
        model_name="Olmo3-Instruct-Sft",
    )
    sft["test_majority_baseline"] = 0.10
    metrics = pd.DataFrame([full, sft])
    plot_df = figures._prepare_goal2_metric_df(metrics)
    full, sft = figures._goal2_model_delta_pair(plot_df)
    full_table = figures._goal2_target_summary_table(plot_df[plot_df["model_name"] == full])
    sft_table = figures._goal2_target_summary_table(plot_df[plot_df["model_name"] == sft])

    delta = figures._aligned_delta_table(full_table, sft_table)

    assert round(float(delta.loc["Any Carry", "Prompt Final"]), 6) == 0.20


def test_goal2_diagnostic_heatmap_averages_target_columns() -> None:
    """Verify diagnostic heatmaps average over target columns instead of taking max."""
    metrics = pd.DataFrame(
        [
            _probe_metric(
                "outgoing_carry",
                "prompt_final",
                0,
                1.0,
                target_column_lsd=0,
            ),
            _probe_metric(
                "outgoing_carry",
                "prompt_final",
                0,
                0.0,
                target_column_lsd=1,
            ),
        ],
    )

    table = figures._goal2_metric_table(figures._prepare_goal2_metric_df(metrics))

    assert round(float(table.loc["0", "Prompt Final"]), 6) == 0.5


def test_goal2_diagnostic_delta_heatmap_uses_raw_accuracy() -> None:
    """Verify diagnostic heatmap deltas use raw accuracy, not baseline lift."""
    full = _probe_metric(
        "outgoing_carry",
        "prompt_final",
        0,
        0.8,
        model_name="Olmo3-Instruct-Full",
    )
    full["test_majority_baseline"] = 0.7
    sft = _probe_metric(
        "outgoing_carry",
        "prompt_final",
        0,
        0.6,
        model_name="Olmo3-Instruct-Sft",
    )
    sft["test_majority_baseline"] = 0.0
    plot_df = figures._prepare_goal2_metric_df(pd.DataFrame([full, sft]))
    full_model, sft_model = figures._goal2_model_delta_pair(plot_df)
    full_table = figures._goal2_metric_table(plot_df[plot_df["model_name"] == full_model])
    sft_table = figures._goal2_metric_table(plot_df[plot_df["model_name"] == sft_model])

    delta = figures._aligned_delta_table(full_table, sft_table)

    assert round(float(delta.loc["0", "Prompt Final"]), 6) == 0.2


def test_goal2_model_delta_predictions_keep_shared_examples_only() -> None:
    """Verify model-delta metrics use examples present for both models."""
    predictions = pd.DataFrame(
        [
            _probe_prediction(
                "Olmo3-Instruct-Full",
                "answer_only",
                "outgoing_carry",
                "prompt_final",
                0,
                y_true=1,
                probe_correct=True,
                example_id="shared",
            ),
            _probe_prediction(
                "Olmo3-Instruct-Sft",
                "answer_only",
                "outgoing_carry",
                "prompt_final",
                0,
                y_true=1,
                probe_correct=False,
                example_id="shared",
            ),
            _probe_prediction(
                "Olmo3-Instruct-Full",
                "answer_only",
                "outgoing_carry",
                "prompt_final",
                0,
                y_true=1,
                probe_correct=False,
                example_id="full-only",
            ),
            _probe_prediction(
                "Olmo3-Instruct-Sft",
                "answer_only",
                "outgoing_carry",
                "prompt_final",
                0,
                y_true=1,
                probe_correct=True,
                example_id="sft-only",
            ),
        ]
    )

    paired = figures._goal2_shared_model_example_predictions(predictions)
    metrics = figures._goal2_prediction_metric_df(paired)
    full, sft = figures._goal2_model_delta_pair(metrics)
    full_table = figures._goal2_target_summary_table(metrics[metrics["model_name"] == full])
    sft_table = figures._goal2_target_summary_table(metrics[metrics["model_name"] == sft])
    delta = figures._aligned_delta_table(full_table, sft_table)

    assert set(paired["example_id"]) == {"shared"}
    assert round(float(delta.loc["Outgoing Carry", "Prompt Final"]), 6) == 1.0


def test_goal2_layer_profile_uses_requested_locations_only() -> None:
    """Verify layer-profile rows use the requested reasoning locations."""
    metrics = pd.DataFrame(
        [
            _probe_metric("outgoing_carry", "prompt_final", 0, 0.80, target_column_lsd=0),
            _probe_metric("outgoing_carry", "cot_2_3", 0, 0.85, target_column_lsd=0),
            _probe_metric("outgoing_carry", "answer_digits", 0, 0.90, target_column_lsd=0),
            _probe_metric("outgoing_carry", "cot_start", 0, 0.70, target_column_lsd=0),
            _probe_metric("any_carry", "prompt_final", 0, 0.75),
        ],
    )

    profile = figures._goal2_layer_profile_data(metrics)

    assert set(profile["location_label"]) == {"Prompt Final", "CoT 2/3", "Answer Digits"}
    assert set(profile["target"]) == {"any_carry", "outgoing_carry"}


def test_goal2_layer_profile_uses_model_color_and_location_shapes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify layer-profile plots encode model and location separately."""
    calls: list[dict[str, object]] = []

    def fake_plot(*args: object, **kwargs: object) -> list[object]:
        """Record matplotlib line settings."""
        calls.append(kwargs)
        return []

    monkeypatch.setattr(figures.plt.Axes, "plot", fake_plot)
    metrics = pd.DataFrame(
        [
            _probe_metric("outgoing_carry", "cot_2_3", 0, 0.74, target_column_lsd=0),
            _probe_metric("outgoing_carry", "prompt_final", 1, 0.80, target_column_lsd=0),
            _probe_metric("outgoing_carry", "answer_digits", 0, 0.90, target_column_lsd=0),
            _probe_metric(
                "outgoing_carry",
                "prompt_final",
                0,
                0.71,
                target_column_lsd=0,
                model_name="other-model",
            ),
        ],
    )

    figures._goal2_layer_profile_plot_from_metric_df(
        figures._prepare_goal2_metric_df(metrics),
        tmp_path,
        filename="layer_profile.png",
    )

    assert calls
    assert {call["marker"] for call in calls[:3]} == {"o", "s", "D"}
    assert all(call["markevery"] == figures.GOAL2_LAYER_PROFILE_MARK_EVERY for call in calls)


def test_goal2_line_plots_do_not_use_column_confidence_intervals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Goal 2 line plots do not show non-example-level CIs."""
    calls: list[dict[str, object]] = []

    def fake_lineplot(*args: object, **kwargs: object) -> None:
        """Record lineplot errorbar settings."""
        calls.append(kwargs)

    monkeypatch.setattr(figures.sns, "lineplot", fake_lineplot)
    metrics = pd.DataFrame(
        [
            _probe_metric("outgoing_carry", "prompt_final", 0, 0.80, target_column_lsd=0),
            _probe_metric("outgoing_carry", "cot_start", 0, 0.70, target_column_lsd=0),
            _probe_metric(
                "carry_chain_membership",
                "prompt_final",
                0,
                0.75,
                target_column_lsd=0,
            ),
            _probe_metric(
                "carry_chain_membership",
                "cot_start",
                0,
                0.85,
                target_column_lsd=0,
            ),
        ],
    )

    figures._goal2_reasoning_time_by_column(metrics, tmp_path)
    figures._goal2_probe_timing_curves(metrics, tmp_path)

    assert calls
    assert calls[0]["hue"] == "model_name"
    assert "style" not in calls[0]
    assert all(call["errorbar"] is None for call in calls)


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
                "answer_format": "standard",
                "slice_name": "no_carry",
                "n_digits": 2,
                "max_carry_chain": 0,
                "first_carry_position": None,
            },
            {
                "id": "ex-capped",
                "prompt_mode": "answer_only",
                "digit_format": "standard",
                "answer_format": "standard",
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


def _probe_metric(
    target: str,
    location_kind: str,
    layer_index: int,
    accuracy: float,
    target_column_lsd: int | None = None,
    model_name: str = "model",
) -> dict[str, object]:
    """Return one fitted Goal 2 probe metric row."""
    return {
        "model_name": model_name,
        "target": target,
        "target_column_lsd": target_column_lsd,
        "location_kind": location_kind,
        "layer_index": str(layer_index),
        "status": "fitted",
        "test_accuracy": accuracy,
        "test_majority_baseline": 0.5,
        "test_examples": 2,
        "train_examples": 4,
    }


def _probe_prediction(
    model_name: str,
    prompt_mode: str,
    target: str,
    location_kind: str,
    target_column_lsd: int | None,
    y_true: int,
    probe_correct: bool,
    digit_format: str = "standard",
    answer_format: str = "standard",
    example_id: str = "example",
) -> dict[str, object]:
    """Return one Goal 2 probe prediction row with prompt metadata."""
    return {
        "example_id": example_id,
        "model_name": model_name,
        "prompt_mode": prompt_mode,
        "digit_format": digit_format,
        "answer_format": answer_format,
        "target": target,
        "target_column_lsd": target_column_lsd,
        "location_kind": location_kind,
        "layer_index": "0",
        "y_true": y_true,
        "probe_correct": probe_correct,
    }
