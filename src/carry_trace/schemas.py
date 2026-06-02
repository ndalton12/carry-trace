"""Pydantic schemas for saved datasets and run artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from carry_trace.enums import DigitFormat, PromptMode, RunnerKind, SliceName


class AdditionExample(BaseModel):
    id: str
    schema_version: str
    split: str
    seed: int
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCallRecord(BaseModel):
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
    package_versions: dict[str, str]
    exact_match: bool | None = None
    parsed_answer_correct: bool | None = None
    first_wrong_digit_lsd: int | None = None
    first_wrong_digit_msd: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
