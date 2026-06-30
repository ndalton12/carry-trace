"""Closed value sets used by configs and saved schemas."""

from __future__ import annotations

from enum import StrEnum


class PromptMode(StrEnum):
    ANSWER_ONLY = "answer_only"
    FREE_COT = "free_cot"
    LENGTH_CONTROLLED_COT = "length_controlled_cot"
    STRUCTURED_COLUMN_COT = "structured_column_cot"


class DigitFormat(StrEnum):
    """Operand display formats used in rendered prompts."""
    STANDARD = "standard"
    DELIMITED = "delimited"

    @classmethod
    def _missing_(cls, value: object) -> DigitFormat | None:
        """Map legacy digit-format names to current enum values."""
        if value == "plain":
            return cls.STANDARD
        return None


class AnswerFormat(StrEnum):
    """Expected answer emission formats used in rendered prompts."""
    STANDARD = "standard"
    LSD = "lsd"

    @classmethod
    def _missing_(cls, value: object) -> AnswerFormat | None:
        """Map legacy answer-format names to current enum values."""
        if value == "lsd_delimited":
            return cls.LSD
        return None


class SliceName(StrEnum):
    """Addition problem structure slices."""
    NO_CARRY = "no_carry"
    ISOLATED_CARRY = "isolated_carry"
    LONG_CARRY_CHAIN = "long_carry_chain"
    INTERNAL_CARRY_CHAIN = "internal_carry_chain"
    CARRY_DISTRACTOR = "carry_distractor"
    MANY_9S_NO_CARRY = "many_9s_no_carry"
    RANDOM = "random"


class RunnerKind(StrEnum):
    """Model execution backends."""

    FAKE = "fake"
    HF = "hf"
    VLLM = "vllm"


class ActivationLocation(StrEnum):
    """Symbolic token locations supported by Goal 2 activation extraction."""

    OPERAND_DIGITS = "operand_digits"
    QUESTION_TOKEN = "question_token"
    PROMPT_FINAL = "prompt_final"
    COT_START = "cot_start"
    COT_1_3 = "cot_1_3"
    COT_2_3 = "cot_2_3"
    COT_END = "cot_end"
    ANSWER_DIGITS = "answer_digits"


class QuantizationKind(StrEnum):
    """Model quantization modes supported by runners."""
    NONE = "none"
    BITSANDBYTES_8BIT = "bitsandbytes_8bit"


class TorchDType(StrEnum):
    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
