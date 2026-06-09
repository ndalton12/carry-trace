from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from carry_trace.config import DatasetConfig, GenerationParams, ModelSpec, RunnerConfig
from carry_trace.datasets import generate_dataset
from carry_trace.enums import QuantizationKind
from carry_trace.models import HuggingFaceModelRunner, has_unclosed_thinking


class DummyBatchTokenizer:
    """Tiny tokenizer that supports left-padded batch tokenization for runner tests."""

    chat_template = None
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def __call__(
        self,
        texts: str | list[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize strings into character-code tensors."""
        if isinstance(texts, str):
            texts = [texts]
        rows = [[ord(char) for char in text] for text in texts]
        max_length = max(len(row) for row in rows)
        input_rows = []
        mask_rows = []
        for row in rows:
            pad_length = max_length - len(row) if padding else 0
            input_rows.append([self.pad_token_id] * pad_length + row)
            mask_rows.append([0] * pad_length + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        }

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode character-code token IDs into a string."""
        return "".join(chr(token_id) for token_id in ids if token_id != self.pad_token_id)


class DummyBatchModel:
    """Tiny model that records batch sizes passed to generate."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        """Initialize recorded batch sizes and minimal model config."""
        self.config = SimpleNamespace(pad_token_id=0)
        self.batch_sizes: list[int] = []

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Append fixed output tokens while recording the incoming batch size."""
        batch_size = int(input_ids.shape[0])
        self.batch_sizes.append(batch_size)
        max_new_tokens = int(kwargs.get("max_new_tokens", 1))
        output = torch.full((batch_size, max_new_tokens), ord("7"), dtype=torch.long)
        return torch.cat([input_ids, output], dim=-1)


def test_has_unclosed_thinking_detects_open_think_block() -> None:
    """Check that unclosed thinking is detected only when the close marker is absent."""
    assert has_unclosed_thinking("<think>working") is True
    assert has_unclosed_thinking("<think>working</think>\nAnswer: 5") is False
    assert has_unclosed_thinking("Answer: 5") is False


def test_hf_generate_kwargs_excludes_thinking_cap_fields() -> None:
    """Check that local thinking-cap fields are not passed to Transformers generate."""
    runner = object.__new__(HuggingFaceModelRunner)
    runner.generation = GenerationParams(
        max_new_tokens=256,
        thinking_final_answer_tokens=100,
        force_close_thinking=True,
    )
    assert runner._hf_generate_kwargs() == {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }


def test_generation_params_rejects_invalid_thinking_cap() -> None:
    """Check that thinking-cap configs must reserve a valid final-answer budget."""
    with pytest.raises(ValidationError):
        GenerationParams(max_new_tokens=100, force_close_thinking=True)
    with pytest.raises(ValidationError):
        GenerationParams(
            max_new_tokens=100,
            thinking_final_answer_tokens=100,
            force_close_thinking=True,
        )


def test_quantization_config_rejects_missing_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that bitsandbytes quantization fails clearly when the package is absent."""
    runner = object.__new__(HuggingFaceModelRunner)
    runner.runner = type("Runner", (), {"quantization": QuantizationKind.BITSANDBYTES_8BIT})()
    monkeypatch.setattr("carry_trace.models.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="requires bitsandbytes"):
        runner._quantization_config()


def test_hf_runner_batches_generation_calls(tmp_path: Path) -> None:
    """Check that HF runner batch_size controls batched generate calls."""
    _, _, examples = generate_dataset(
        DatasetConfig(
            name="batching",
            seed=1,
            output_dir=tmp_path,
            write_parquet=False,
            splits={"smoke": {"examples_per_slice_per_length": 3}},
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
            digit_formats=["standard"],
            answer_formats=["standard"],
        )
    )
    runner = object.__new__(HuggingFaceModelRunner)
    runner.model_spec = ModelSpec(name="dummy", model_id="dummy")
    runner.runner = RunnerConfig(kind="hf", batch_size=2)
    runner.generation = GenerationParams(max_new_tokens=2)
    runner.git_commit = None
    runner.tokenizer = DummyBatchTokenizer()
    runner.model = DummyBatchModel()

    records = list(runner.generate(examples, run_id="run", seed=1))

    assert runner.model.batch_sizes == [2, 1]
    assert len(records) == 3
    assert {record.decoded_output for record in records} == {"77"}
    assert {record.metadata["hf_batch_size"] for record in records} == {1, 2}
