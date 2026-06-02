"""Configuration models and loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from carry_trace.enums import DigitFormat, PromptMode, RunnerKind, SliceName, TorchDType


class DatasetConfig(BaseModel):
    name: str
    seed: int = 0
    base: int = 10
    output_dir: Path = Path("data/generated")
    write_parquet: bool = True
    schema_version: str = "goal1.v1"
    splits: dict[str, int] = Field(default_factory=lambda: {"smoke": 1})
    digit_lengths: list[int] = Field(default_factory=lambda: [2, 3, 4])
    slices: list[SliceName] = Field(
        default_factory=lambda: [SliceName.NO_CARRY, SliceName.ISOLATED_CARRY]
    )
    prompt_modes: list[PromptMode] = Field(default_factory=lambda: [PromptMode.ANSWER_ONLY])
    digit_formats: list[DigitFormat] = Field(default_factory=lambda: [DigitFormat.PLAIN])
    digit_delimiter: str = "|"
    examples_per_slice_per_length: int = 1


class GenerationParams(BaseModel):
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


class ModelSpec(BaseModel):
    name: str
    model_id: str
    revision: str | None = None
    tokenizer_id: str | None = None


class RunnerConfig(BaseModel):
    kind: RunnerKind = RunnerKind.FAKE
    device: str = "auto"
    dtype: TorchDType = TorchDType.AUTO
    batch_size: int = 1
    trust_remote_code: bool = False


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 0
    dataset_path: Path
    output_dir: Path = Path("runs")
    max_examples: int | None = None
    prompt_modes: list[PromptMode] | None = None
    digit_formats: list[DigitFormat] | None = None
    models: list[ModelSpec] = Field(default_factory=list)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    generation: GenerationParams = Field(default_factory=GenerationParams)


def load_yaml_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a mapping")
    return data


def load_dataset_config(path: Path) -> DatasetConfig:
    return DatasetConfig.model_validate(load_yaml_config(path))


def load_experiment_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml_config(path))
