"""Configuration models and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, model_validator

from carry_trace.enums import (
    ActivationLocation,
    AnswerFormat,
    DigitFormat,
    ProbeTarget,
    PromptMode,
    QuantizationKind,
    RunnerKind,
    SliceName,
    TorchDType,
)

ExampleCount = Annotated[int, Field(ge=0)]


class SplitConfig(BaseModel):
    """Per-split dataset generation settings."""

    examples_per_slice_per_length: ExampleCount | None = None
    slice_examples_per_length: dict[SliceName, ExampleCount] = Field(default_factory=dict)


class RandomSamplingConfig(BaseModel):
    """Random-slice generation settings."""

    balance_carry_count: bool = False


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
    examples_per_slice_per_length: ExampleCount = 1
    slice_examples_per_length: dict[SliceName, ExampleCount] = Field(default_factory=dict)
    random_sampling: RandomSamplingConfig = Field(default_factory=RandomSamplingConfig)


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
    tensor_parallel_size: int = Field(default=1, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    max_model_len: int | None = Field(default=None, gt=0)
    enforce_eager: bool = False


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


class ActivationExtractionConfig(BaseModel):
    """Goal 2 activation extraction settings."""

    locations: list[ActivationLocation] = Field(
        default_factory=lambda: [
            ActivationLocation.OPERAND_DIGITS,
            ActivationLocation.QUESTION_TOKEN,
            ActivationLocation.PROMPT_FINAL,
            ActivationLocation.COT_START,
            ActivationLocation.COT_1_3,
            ActivationLocation.COT_2_3,
            ActivationLocation.COT_END,
            ActivationLocation.ANSWER_DIGITS,
        ]
    )
    storage_dtype: TorchDType = TorchDType.FLOAT16
    include_embedding_layer: bool = False


class HubUploadConfig(BaseModel):
    """Optional Hugging Face Hub upload settings for run artifacts."""

    enabled: bool = False
    repo_id: str | None = None
    path_in_repo: str | None = None
    revision: str | None = None
    private: bool = False
    create_pr: bool = False
    create_repo: bool = True
    commit_message: str | None = None

    @model_validator(mode="after")
    def validate_repo_id(self) -> HubUploadConfig:
        """Require a repository ID when Hub upload is enabled."""
        if self.enabled and not self.repo_id:
            raise ValueError("upload.repo_id is required when upload.enabled is true")
        return self


class Goal2Config(BaseModel):
    """Goal 2 activation extraction run configuration."""

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
    activations: ActivationExtractionConfig = Field(default_factory=ActivationExtractionConfig)
    upload: HubUploadConfig = Field(default_factory=HubUploadConfig)


class Goal2ProbeConfig(BaseModel):
    """Goal 2 linear-probe training and evaluation configuration."""

    name: str
    goal2_run_dir: Path
    output_dir: Path = Path("runs/probes")
    train_split: str = "train_probe"
    test_split: str = "test_probe"
    targets: list[ProbeTarget] = Field(
        default_factory=lambda: [
            ProbeTarget.ANY_CARRY,
            ProbeTarget.INCOMING_CARRY,
            ProbeTarget.OUTGOING_CARRY,
            ProbeTarget.OUTPUT_DIGIT,
            ProbeTarget.RAW_SUM,
            ProbeTarget.CARRY_CHAIN_MEMBERSHIP,
            ProbeTarget.COLUMN_POINTER,
        ]
    )
    min_train_examples: int = Field(default=20, ge=1)
    min_test_examples: int = Field(default=5, ge=1)
    max_iter: int = Field(default=1000, gt=0)
    c: float = Field(default=1.0, gt=0)
    random_state: int = 0
    require_unambiguous_digit_tokens: bool = False
    n_jobs: int = Field(default=1, ge=1)


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


def load_goal2_config(path: Path) -> Goal2Config:
    """Load and validate a Goal 2 config from YAML."""
    return Goal2Config.model_validate(load_yaml_config(path))


def load_goal2_probe_config(path: Path) -> Goal2ProbeConfig:
    """Load and validate a Goal 2 linear-probe config from YAML."""
    return Goal2ProbeConfig.model_validate(load_yaml_config(path))
