"""Pydantic schemas for saved datasets and run artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from carry_trace.enums import (
    ActivationLocation,
    AnswerFormat,
    DigitFormat,
    ProbeTarget,
    PromptMode,
    RunnerKind,
    SliceName,
)


class AdditionExample(BaseModel):
    """Saved dataset row for one rendered addition prompt."""
    id: str
    problem_id: str
    schema_version: str
    split: str
    seed: int
    split_seed: int | None = None
    generator_config_hash: str
    slice_name: SliceName
    base: int
    n_digits: int
    a: str
    b: str
    answer: str
    prompt_mode: PromptMode
    digit_format: DigitFormat
    digit_delimiter: str
    answer_format: AnswerFormat = AnswerFormat.STANDARD
    expected_output: str | None = None
    prompt_a: str
    prompt_b: str
    template_id: str
    prompt: str
    messages: list[dict[str, str]]
    digits_a_lsd: list[int]
    digits_b_lsd: list[int]
    raw_sum: list[int]
    incoming_carry: list[int]
    outgoing_carry: list[int]
    output_digits_lsd: list[int]
    carry_count: int
    max_carry_chain: int
    carry_positions: list[int]
    first_carry_position: int | None = None
    answer_length_change: bool = False
    match_group_id: str | None = None
    match_role: str | None = None
    target_column_lsd: int | None = None
    intervention_variable: str | None = None
    match_family: str | None = None
    match_constraints: dict[str, Any] = Field(default_factory=dict)
    partner_problem_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Goal3ReplayPrefix(BaseModel):
    """Saved natural-CoT prefix recovered from a Goal 2 activation record."""

    id: str
    schema_version: str
    source_goal2_run_id: str
    example_id: str
    problem_id: str
    split: str
    n_digits: int
    source_model_name: str | None = None
    source_model_id: str | None = None
    location_kind: ActivationLocation
    recorded_output_token_end_index: int | None = None
    replay_output_token_end_index: int | None = None
    prefix_token_source: str
    prefix_alignment_delta: int | None = None
    assistant_prefix_token_ids: list[int]
    assistant_prefix: str
    decoded_output: str
    parsed_answer: str | None = None
    expected_output: str
    prompt: str
    messages: list[dict[str, str]]
    answer_only_example_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Goal3ReplayCase(BaseModel):
    """Saved receiver assignment for one natural-CoT replay prefix."""

    id: str
    schema_version: str
    replay_prefix_id: str
    source_goal2_run_id: str
    example_id: str
    problem_id: str
    split: str
    n_digits: int
    replay_kind: str
    source_model_name: str | None = None
    receiver_model_name: str
    location_kind: ActivationLocation
    assistant_prefix_token_ids: list[int]
    assistant_prefix: str
    expected_output: str
    prompt: str
    messages: list[dict[str, str]]
    answer_only_example_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Goal3ResidualInterventionCase(BaseModel):
    """Saved specification for one probe-guided residual intervention."""

    id: str
    schema_version: str
    source_goal2_run_id: str
    example_id: str
    problem_id: str
    split: str
    n_digits: int
    model_name: str
    model_id: str
    target: ProbeTarget
    target_column_lsd: int
    affected_output_column_lsd: int
    factual_carry: int
    counterfactual_carry: int
    factual_output_digit: int
    counterfactual_output_digit: int
    factual_answer: str
    counterfactual_answer: str
    unchanged_output_columns_lsd: list[int]
    location_kind: ActivationLocation
    activation_location_name: str
    activation_location_index: int
    layer_index: int
    activation_layer_index: int
    activation_path: str
    direction_id: str
    direction_train_split: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCallRecord(BaseModel):
    """Saved model generation record for one dataset example."""
    run_id: str
    example_id: str
    model_name: str
    model_id: str
    model_revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    runner_kind: RunnerKind
    seed: int
    timestamp: str
    prompt: str
    messages: list[dict[str, str]]
    rendered_prompt: str
    input_ids: list[int]
    output_ids: list[int]
    decoded_output: str
    parsed_answer: str | None = None
    generation_config: dict[str, Any]
    token_count_input: int
    token_count_output: int
    latency_seconds: float
    git_commit: str | None = None
    exact_match: bool | None = None
    parsed_answer_correct: bool | None = None
    first_wrong_digit_lsd: int | None = None
    first_wrong_digit_msd: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
