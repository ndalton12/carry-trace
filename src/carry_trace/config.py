"""Configuration models and loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from carry_trace.enums import (
    AnswerFormat,
    DigitFormat,
    PromptMode,
    QuantizationKind,
    RunnerKind,
    SliceName,
    TorchDType,
)


class SplitConfig(BaseModel):
    """Per-split dataset generation settings."""

    examples_per_slice_per_length: int | None = None


class DatasetConfig(BaseModel):
    """Dataset generation configuration."""

    name: str
    seed: int = 0
    base: int = 10
    output_dir: Path = Path("data/generated")
    write_parquet: bool = True
    schema_version: str = "goal1.v1"
    splits: dict[str, SplitConfig] = Field(default_factory=lambda: {"smoke": SplitConfig()})
    digit_lengths: list[int] = Field(default_factory=lambda: [2, 3, 4])
    slices: list[SliceName] = Field(
        default_factory=lambda: [SliceName.NO_CARRY, SliceName.ISOLATED_CARRY]
    )
    prompt_modes: list[PromptMode] = Field(default_factory=lambda: [PromptMode.ANSWER_ONLY])
    digit_formats: list[DigitFormat] = Field(default_factory=lambda: [DigitFormat.STANDARD])
    answer_formats: list[AnswerFormat] = Field(default_factory=lambda: [AnswerFormat.STANDARD])
    digit_delimiter: str = "|"
    examples_per_slice_per_length: int = 1


class GenerationParams(BaseModel):
    """Text generation parameters passed to model runners."""

    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    thinking_final_answer_tokens: int | None = Field(default=None, gt=0)
    force_close_thinking: bool = False

    @model_validator(mode="after")
    def validate_thinking_cap(self) -> GenerationParams:
        """Validate that the thinking cap leaves room for both generation phases."""
        if self.force_close_thinking and self.thinking_final_answer_tokens is None:
            raise ValueError(
                "thinking_final_answer_tokens is required when force_close_thinking is true"
            )
        if (
            self.force_close_thinking
            and self.thinking_final_answer_tokens is not None
            and self.thinking_final_answer_tokens >= self.max_new_tokens
        ):
            raise ValueError("thinking_final_answer_tokens must be less than max_new_tokens")
        return self


class ModelSpec(BaseModel):
    """Model and tokenizer identifiers for one evaluated checkpoint."""

    name: str
    model_id: str
    revision: str | None = None
    tokenizer_id: str | None = None


class RunnerConfig(BaseModel):
    """Model runner backend configuration."""

    kind: RunnerKind = RunnerKind.FAKE
    device: str = "auto"
    dtype: TorchDType = TorchDType.AUTO
    batch_size: int = 1
    trust_remote_code: bool = False
    quantization: QuantizationKind = QuantizationKind.NONE


class ExperimentConfig(BaseModel):
    """Experiment run configuration."""

    name: str
    seed: int = 0
    dataset_path: Path
    output_dir: Path = Path("runs")
    max_examples: int | None = None
    splits: list[str] | None = None
    prompt_modes: list[PromptMode] | None = None
    digit_formats: list[DigitFormat] | None = None
    answer_formats: list[AnswerFormat] | None = None
    models: list[ModelSpec] = Field(default_factory=list)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    generation: GenerationParams = Field(default_factory=GenerationParams)


def load_yaml_config(path: Path) -> dict[str, object]:
    """Load a YAML config file as a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a mapping")
    return data


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset config from YAML."""
    return DatasetConfig.model_validate(load_yaml_config(path))


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config from YAML."""
    return ExperimentConfig.model_validate(load_yaml_config(path))
