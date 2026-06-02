"""Closed value sets used by configs and saved schemas."""

from __future__ import annotations

from enum import StrEnum


class PromptMode(StrEnum):
    ANSWER_ONLY = "answer_only"
    FREE_COT = "free_cot"
    LENGTH_CONTROLLED_COT = "length_controlled_cot"
    STRUCTURED_COLUMN_COT = "structured_column_cot"


class DigitFormat(StrEnum):
    PLAIN = "plain"
    DELIMITED = "delimited"


class SliceName(StrEnum):
    NO_CARRY = "no_carry"
    ISOLATED_CARRY = "isolated_carry"
    LONG_CARRY_CHAIN = "long_carry_chain"
    INTERNAL_CARRY_CHAIN = "internal_carry_chain"
    CARRY_DISTRACTOR = "carry_distractor"
    MANY_9S_NO_CARRY = "many_9s_no_carry"
    RANDOM = "random"


class RunnerKind(StrEnum):
    FAKE = "fake"
    HF = "hf"


class TorchDType(StrEnum):
    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
