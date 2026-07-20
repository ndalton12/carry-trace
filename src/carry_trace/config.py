"""Configuration models and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

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
    prompt_modes: list[PromptMode] | None = None
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


class Goal3ReplayDatasetConfig(BaseModel):
    """Goal 3 natural-CoT replay dataset settings."""

    enabled: bool = True
    locations: list[ActivationLocation] = Field(
        default_factory=lambda: [
            ActivationLocation.PROMPT_FINAL,
            ActivationLocation.COT_1_3,
            ActivationLocation.COT_2_3,
            ActivationLocation.COT_END,
        ]
    )
    crossed_models: bool = True
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_local_files_only: bool = False
    max_token_alignment_shift: int = Field(default=64, ge=0)


class Goal3ResidualDatasetConfig(BaseModel):
    """Goal 3 residual-intervention dataset settings."""

    enabled: bool = True
    targets: list[ProbeTarget] = Field(
        default_factory=lambda: [
            ProbeTarget.INCOMING_CARRY,
            ProbeTarget.OUTGOING_CARRY,
        ]
    )
    locations: list[ActivationLocation] = Field(
        default_factory=lambda: [
            ActivationLocation.PROMPT_FINAL,
            ActivationLocation.COT_2_3,
            ActivationLocation.COT_END,
        ]
    )
    layers: list[int] = Field(default_factory=lambda: [8, 16, 24])
    direction_train_split: str = "train_probe"
    target_columns_by_target: dict[ProbeTarget, dict[int, list[int]]] = Field(
        default_factory=lambda: {
            ProbeTarget.INCOMING_CARRY: {2: [1], 4: [1, 2]},
            ProbeTarget.OUTGOING_CARRY: {2: [0], 4: [0, 1]},
        }
    )
    require_stable_outgoing_carry: bool = True

    @model_validator(mode="after")
    def validate_carry_targets(self) -> Goal3ResidualDatasetConfig:
        """Restrict Goal 3 residual datasets to supported carry targets."""
        supported = {ProbeTarget.INCOMING_CARRY, ProbeTarget.OUTGOING_CARRY}
        if not self.targets or not set(self.targets).issubset(supported):
            raise ValueError("Goal 3 residual datasets require incoming_carry or outgoing_carry")
        return self


class Goal3DatasetConfig(BaseModel):
    """Goal 3 model-derived dataset bundle configuration."""

    name: str
    source_goal2_run_dir: Path
    source_dataset_path: Path | None = None
    output_dir: Path = Path("data/generated")
    schema_version: str = "goal3.natural_cot.v2"
    splits: list[str] = Field(default_factory=lambda: ["test_probe"])
    models: list[str] | None = None
    digit_lengths: list[int] = Field(default_factory=lambda: [2, 4])
    prompt_modes: list[PromptMode] = Field(default_factory=lambda: [PromptMode.FREE_COT])
    digit_formats: list[DigitFormat] = Field(default_factory=lambda: [DigitFormat.STANDARD])
    answer_formats: list[AnswerFormat] = Field(default_factory=lambda: [AnswerFormat.STANDARD])
    require_shared_models: bool = True
    include_token_limit_hits: bool = False
    replay: Goal3ReplayDatasetConfig = Field(default_factory=Goal3ReplayDatasetConfig)
    residual: Goal3ResidualDatasetConfig = Field(default_factory=Goal3ResidualDatasetConfig)


class Goal3ReplayRunConfig(BaseModel):
    """Goal 3 natural-CoT replay execution settings."""

    enabled: bool = True
    answer_cue: str = "\nFinal answer:"
    decode_locations: list[ActivationLocation] = Field(
        default_factory=lambda: [ActivationLocation.PROMPT_FINAL, ActivationLocation.COT_END]
    )
    generation: GenerationParams = Field(
        default_factory=lambda: GenerationParams(max_new_tokens=16)
    )


class Goal3ResidualRunConfig(BaseModel):
    """Goal 3 probe-guided residual intervention execution settings."""

    enabled: bool = True
    intervention_mode: Literal["fixed_gap", "projection_clamp"] = "fixed_gap"
    intervention_scales: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    intervention_sites: list[Literal["prefix_boundary", "answer_cue"]] = Field(
        default_factory=lambda: ["prefix_boundary"],
        min_length=1,
    )
    control_directions: list[Literal["carry", "orthogonal"]] = Field(
        default_factory=lambda: ["carry"]
    )
    orthogonal_control_count: int = Field(default=1, ge=1)
    max_problems_per_digit_length: int | None = Field(default=None, ge=1)
    require_shared_correct_source: bool = False
    score_locations: list[ActivationLocation] | None = None
    score_layers: list[int] | None = None
    min_train_examples: int = Field(default=20, ge=2)
    max_iter: int = Field(default=1000, gt=0)
    c: float = Field(default=1.0, gt=0.0)
    require_exact_prefix_alignment: bool = True
    decode_locations: list[ActivationLocation] = Field(
        default_factory=lambda: [ActivationLocation.COT_2_3, ActivationLocation.COT_END]
    )
    decode_layers: list[int] = Field(default_factory=lambda: [16])
    decode_intervention_scale: float = Field(default=1.0, gt=0.0)
    generation: GenerationParams = Field(
        default_factory=lambda: GenerationParams(max_new_tokens=16)
    )

    @model_validator(mode="after")
    def validate_intervention_variants(self) -> Goal3ResidualRunConfig:
        """Require unique positive scales and a carry direction for primary execution."""
        if any(scale <= 0.0 for scale in self.intervention_scales):
            raise ValueError("residual.intervention_scales must all be positive")
        if len(set(self.intervention_scales)) != len(self.intervention_scales):
            raise ValueError("residual.intervention_scales must be unique")
        if len(set(self.intervention_sites)) != len(self.intervention_sites):
            raise ValueError("residual.intervention_sites must be unique")
        if len(set(self.control_directions)) != len(self.control_directions):
            raise ValueError("residual.control_directions must be unique")
        if "carry" not in self.control_directions:
            raise ValueError("residual.control_directions must include carry")
        if self.score_locations is not None and len(set(self.score_locations)) != len(
            self.score_locations
        ):
            raise ValueError("residual.score_locations must be unique")
        if self.score_layers is not None and len(set(self.score_layers)) != len(self.score_layers):
            raise ValueError("residual.score_layers must be unique")
        if self.decode_locations and self.decode_intervention_scale not in self.intervention_scales:
            raise ValueError("residual.decode_intervention_scale must be in intervention_scales")
        if self.decode_locations and (
            self.intervention_mode != "fixed_gap" or self.intervention_sites != ["prefix_boundary"]
        ):
            raise ValueError("residual decoding currently requires fixed_gap at prefix_boundary")
        return self


class Goal3RunConfig(BaseModel):
    """Goal 3 replay and residual-intervention run configuration."""

    name: str
    seed: int = 0
    dataset_bundle_dir: Path
    output_dir: Path = Path("runs")
    models: list[ModelSpec] = Field(default_factory=list)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    replay: Goal3ReplayRunConfig = Field(default_factory=Goal3ReplayRunConfig)
    residual: Goal3ResidualRunConfig = Field(default_factory=Goal3ResidualRunConfig)

    @model_validator(mode="after")
    def validate_hf_runner(self) -> Goal3RunConfig:
        """Require Hugging Face execution for residual-stream interventions."""
        if self.runner.kind != RunnerKind.HF:
            raise ValueError("Goal 3 execution currently requires runner.kind=hf")
        return self


class Goal35DatasetConfig(BaseModel):
    """Goal 3.5 generation-only arithmetic dataset settings."""

    name: str
    output_dir: Path = Path("data/generated")
    split: str = "test_replay"
    examples_per_digit_length: int = Field(default=64, ge=1)
    digit_lengths: list[int] = Field(default_factory=lambda: [4, 6], min_length=1)
    balance_carry_count: bool = True

    @model_validator(mode="after")
    def validate_digit_lengths(self) -> Goal35DatasetConfig:
        """Require unique positive digit lengths."""
        if any(length < 1 for length in self.digit_lengths):
            raise ValueError("dataset.digit_lengths must be positive")
        if len(set(self.digit_lengths)) != len(self.digit_lengths):
            raise ValueError("dataset.digit_lengths must be unique")
        return self


class Goal35AnalysisConfig(BaseModel):
    """Goal 3.5 paired replay analysis settings."""

    bootstrap_samples: int = Field(default=10000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


class Goal35Config(BaseModel):
    """Goal 3.5 generation-only source-completion and crossed-replay settings."""

    name: str
    seed: int = 0
    output_dir: Path = Path("runs")
    schema_version: str = "goal3.5.v1"
    dataset: Goal35DatasetConfig
    models: list[ModelSpec] = Field(min_length=2, max_length=2)
    tokenizer_id: str
    tokenizer_revision: str | None = None
    tokenizer_local_files_only: bool = False
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    source_generation: GenerationParams = Field(
        default_factory=lambda: GenerationParams(
            max_new_tokens=4096,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )
    )
    replay: Goal3ReplayRunConfig = Field(
        default_factory=lambda: Goal3ReplayRunConfig(
            decode_locations=[
                ActivationLocation.PROMPT_FINAL,
                ActivationLocation.COT_2_3,
                ActivationLocation.COT_END,
            ]
        )
    )
    source_locations: list[ActivationLocation] = Field(
        default_factory=lambda: [
            ActivationLocation.COT_1_3,
            ActivationLocation.COT_2_3,
            ActivationLocation.COT_END,
        ],
        min_length=1,
    )
    require_no_token_limit_hit: bool = True
    historical_source_run_dirs: list[Path] = Field(default_factory=list)
    analysis: Goal35AnalysisConfig = Field(default_factory=Goal35AnalysisConfig)

    @model_validator(mode="after")
    def validate_goal35(self) -> Goal35Config:
        """Require two distinct HF models and the replay endpoint boundary."""
        if self.runner.kind != RunnerKind.HF:
            raise ValueError("Goal 3.5 execution currently requires runner.kind=hf")
        if not self.replay.enabled:
            raise ValueError("Goal 3.5 requires replay.enabled=true")
        if len({model.name for model in self.models}) != 2:
            raise ValueError("Goal 3.5 models must have distinct names")
        if len(set(self.source_locations)) != len(self.source_locations):
            raise ValueError("source_locations must be unique")
        if ActivationLocation.COT_END not in self.source_locations:
            raise ValueError("source_locations must include cot_end")
        supported = {
            ActivationLocation.COT_1_3,
            ActivationLocation.COT_2_3,
            ActivationLocation.COT_END,
        }
        if not set(self.source_locations).issubset(supported):
            raise ValueError("Goal 3.5 source locations must be CoT-relative")
        return self


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


def load_goal3_dataset_config(path: Path) -> Goal3DatasetConfig:
    """Load and validate a Goal 3 dataset bundle config from YAML."""
    return Goal3DatasetConfig.model_validate(load_yaml_config(path))


def load_goal3_run_config(path: Path) -> Goal3RunConfig:
    """Load and validate a Goal 3 execution config from YAML."""
    return Goal3RunConfig.model_validate(load_yaml_config(path))


def load_goal35_config(path: Path) -> Goal35Config:
    """Load and validate a Goal 3.5 generation and replay config from YAML."""
    return Goal35Config.model_validate(load_yaml_config(path))
