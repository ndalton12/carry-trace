from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from carry_trace.config import DatasetConfig, GenerationParams, ModelSpec, RunnerConfig
from carry_trace.datasets import generate_dataset
from carry_trace.enums import QuantizationKind
from carry_trace.models import HuggingFaceModelRunner, VllmModelRunner, has_unclosed_thinking


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


class DummyChatStopTokenizer(DummyBatchTokenizer):
    """Tiny chat tokenizer with an explicit end-of-message token."""

    chat_template = "dummy"
    im_end_token_id = 2
    im_start_token_id = 3
    special_token_ids = {im_end_token_id, im_start_token_id}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: object,
    ) -> str:
        """Render messages with ChatML-like role delimiters."""
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def convert_tokens_to_ids(self, token: str) -> int | None:
        """Return fake token IDs for ChatML control tokens."""
        if token == "<|im_end|>":
            return self.im_end_token_id
        if token == "<|im_start|>":
            return self.im_start_token_id
        return None

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode fake token IDs while optionally hiding special tokens."""
        chars = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.special_token_ids:
                continue
            if token_id == self.im_end_token_id:
                chars.append("<|im_end|>")
            elif token_id == self.im_start_token_id:
                chars.append("<|im_start|>")
            elif token_id != self.pad_token_id:
                chars.append(chr(token_id))
        return "".join(chars)


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


class DummyChatStopModel:
    """Tiny model that would continue into a new chat turn unless stopped."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        """Initialize recorded EOS kwargs and minimal model config."""
        self.config = SimpleNamespace(pad_token_id=0)
        self.eos_token_ids: list[object] = []

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Emit answer, end-of-message, and a fake next user turn."""
        self.eos_token_ids.append(kwargs.get("eos_token_id"))
        batch_size = int(input_ids.shape[0])
        generated = [
            ord("7"),
            DummyChatStopTokenizer.im_end_token_id,
            DummyChatStopTokenizer.im_start_token_id,
            *[ord(char) for char in "user\nignored"],
        ]
        eos_token_ids = kwargs.get("eos_token_id", [])
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        for index, token_id in enumerate(generated):
            if token_id in eos_token_ids:
                generated = generated[: index + 1]
                break
        output = torch.tensor([generated for _ in range(batch_size)], dtype=torch.long)
        return torch.cat([input_ids, output], dim=-1)


class DummyThinkingCapModel:
    """Tiny model that emits thinking text first and final-answer text on continuation."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        """Initialize call count and minimal model config."""
        self.config = SimpleNamespace(pad_token_id=0)
        self.calls = 0

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Emit different fixed text for first-pass and continuation generations."""
        self.calls += 1
        batch_size = int(input_ids.shape[0])
        if self.calls == 1:
            text = "thinking"
        else:
            text = " 64031"
        token_ids = [ord(char) for char in text]
        output = torch.tensor([token_ids for _ in range(batch_size)], dtype=torch.long)
        return torch.cat([input_ids, output], dim=-1)


class DummyThinkingTokenizer(DummyBatchTokenizer):
    """Tiny tokenizer whose chat template opens a think block in the prompt."""

    chat_template = "dummy"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: object,
    ) -> str:
        """Render messages with an assistant think prefix."""
        return f"user: {messages[0]['content']}\nassistant:\n<think>"


class DummySamplingParams:
    """Tiny stand-in for vLLM SamplingParams."""

    def __init__(self, **kwargs: object) -> None:
        """Store sampling parameters for assertions and fake generation."""
        self.kwargs = kwargs
        self.max_tokens = int(kwargs["max_tokens"])


class DummyVllmLLM:
    """Tiny vLLM-like engine that records prompt batch sizes."""

    def __init__(self) -> None:
        """Initialize recorded batch sizes."""
        self.batch_sizes: list[int] = []

    def generate(self, prompts: list[str], sampling_params: DummySamplingParams) -> list[object]:
        """Return fixed vLLM-like request outputs for each prompt."""
        self.batch_sizes.append(len(prompts))
        text = "7" * sampling_params.max_tokens
        token_ids = [ord(char) for char in text]
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=token_ids)])
            for _ in prompts
        ]


class DummyVllmThinkingLLM:
    """Tiny vLLM-like engine that emits thinking text then final-answer text."""

    def __init__(self, continuation_text: str) -> None:
        """Initialize continuation text and call counter."""
        self.continuation_text = continuation_text
        self.calls = 0

    def generate(self, prompts: list[str], sampling_params: DummySamplingParams) -> list[object]:
        """Return different vLLM-like outputs for first-pass and continuation calls."""
        self.calls += 1
        if self.calls == 1:
            text = "thinking"
            token_ids = [ord("t")] * sampling_params.max_tokens
        else:
            text = self.continuation_text
            token_ids = [ord(char) for char in text]
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=token_ids)])
            for _ in prompts
        ]


def test_has_unclosed_thinking_detects_open_think_block() -> None:
    """Check that unclosed thinking is detected only when the close marker is absent."""
    assert has_unclosed_thinking("<think>working") is True
    assert has_unclosed_thinking("<think>working</think>\nAnswer: 5") is False
    assert has_unclosed_thinking("<think>done</think>\n<think>again") is True
    assert has_unclosed_thinking("Answer: 5") is False


def test_hf_generate_kwargs_excludes_thinking_cap_fields() -> None:
    """Check that local thinking-cap fields are not passed to Transformers generate."""
    runner = object.__new__(HuggingFaceModelRunner)
    runner.generation = GenerationParams(
        max_new_tokens=256,
        thinking_final_answer_tokens=100,
        force_close_thinking=True,
    )
    runner.tokenizer = DummyBatchTokenizer()
    runner.model = DummyBatchModel()
    assert runner._hf_generate_kwargs() == {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "eos_token_id": 0,
        "pad_token_id": 0,
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


def test_hf_runner_stops_before_next_chat_turn(tmp_path: Path) -> None:
    """Check that HF generation stops on chat turn-end tokens."""
    _, _, examples = generate_dataset(
        DatasetConfig(
            name="chat_stop",
            seed=1,
            output_dir=tmp_path,
            write_parquet=False,
            splits={"smoke": {"examples_per_slice_per_length": 1}},
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
            digit_formats=["standard"],
            answer_formats=["standard"],
        )
    )
    runner = object.__new__(HuggingFaceModelRunner)
    runner.model_spec = ModelSpec(name="dummy", model_id="dummy")
    runner.runner = RunnerConfig(kind="hf", batch_size=1)
    runner.generation = GenerationParams(max_new_tokens=20)
    runner.git_commit = None
    runner.tokenizer = DummyChatStopTokenizer()
    runner.model = DummyChatStopModel()

    record = next(runner.generate(examples, run_id="run", seed=1))

    assert record.decoded_output == "7"
    assert DummyChatStopTokenizer.im_end_token_id in runner.model.eos_token_ids[0]


def test_vllm_runner_batches_generation_calls(tmp_path: Path) -> None:
    """Check that vLLM runner batch_size controls batched generate calls."""
    _, _, examples = generate_dataset(
        DatasetConfig(
            name="vllm_batching",
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
    runner = object.__new__(VllmModelRunner)
    runner.model_spec = ModelSpec(name="dummy", model_id="dummy")
    runner.runner = RunnerConfig(kind="vllm", batch_size=2)
    runner.generation = GenerationParams(max_new_tokens=2)
    runner.git_commit = None
    runner.tokenizer = DummyBatchTokenizer()
    runner.sampling_params_cls = DummySamplingParams
    runner.llm = DummyVllmLLM()

    records = list(runner.generate(examples, run_id="run", seed=1))

    assert runner.llm.batch_sizes == [2, 1]
    assert len(records) == 3
    assert {record.decoded_output for record in records} == {"77"}
    assert {record.metadata["vllm_batch_size"] for record in records} == {1, 2}
    assert {record.runner_kind for record in records} == {"vllm"}


def test_hf_runner_force_closes_thinking_opened_by_prompt(tmp_path: Path) -> None:
    """Check that thinking caps see think blocks opened by the rendered prompt."""
    _, _, examples = generate_dataset(
        DatasetConfig(
            name="thinking_cap",
            seed=1,
            output_dir=tmp_path,
            write_parquet=False,
            splits={"smoke": {"examples_per_slice_per_length": 1}},
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
            digit_formats=["standard"],
            answer_formats=["lsd"],
        )
    )
    runner = object.__new__(HuggingFaceModelRunner)
    runner.model_spec = ModelSpec(name="dummy", model_id="dummy")
    runner.runner = RunnerConfig(kind="hf", batch_size=1)
    runner.generation = GenerationParams(
        max_new_tokens=10,
        thinking_final_answer_tokens=2,
        force_close_thinking=True,
    )
    runner.git_commit = None
    runner.tokenizer = DummyThinkingTokenizer()
    runner.model = DummyThinkingCapModel()

    record = next(runner.generate(examples, run_id="run", seed=1))

    assert (
        "</think>\nFinal answer digits from right to left with no separators:"
        in record.decoded_output
    )
    assert record.metadata["thinking_force_closed"] is True
    assert record.parsed_answer is not None


def test_vllm_runner_force_closes_thinking_opened_by_prompt(tmp_path: Path) -> None:
    """Check that vLLM thinking caps close rendered-prompt think blocks."""
    _, _, examples = generate_dataset(
        DatasetConfig(
            name="vllm_thinking_cap",
            seed=1,
            output_dir=tmp_path,
            write_parquet=False,
            splits={"smoke": {"examples_per_slice_per_length": 1}},
            digit_lengths=[1],
            slices=["no_carry"],
            prompt_modes=["answer_only"],
            digit_formats=["standard"],
            answer_formats=["standard"],
        )
    )
    runner = object.__new__(VllmModelRunner)
    runner.model_spec = ModelSpec(name="dummy", model_id="dummy")
    runner.runner = RunnerConfig(kind="vllm", batch_size=1)
    runner.generation = GenerationParams(
        max_new_tokens=10,
        thinking_final_answer_tokens=2,
        force_close_thinking=True,
    )
    runner.git_commit = None
    runner.tokenizer = DummyThinkingTokenizer()
    runner.sampling_params_cls = DummySamplingParams
    runner.llm = DummyVllmThinkingLLM(f" {examples[0].expected_output} trailing 1124")

    record = next(runner.generate(examples, run_id="run", seed=1))

    assert "</think>\nFinal answer:" in record.decoded_output
    assert "trailing" not in record.decoded_output
    assert record.metadata["thinking_force_closed"] is True
    expected_output_length = len(examples[0].expected_output)
    assert record.metadata["thinking_stop_expected_output_digits"] == expected_output_length
    assert record.parsed_answer == examples[0].answer
