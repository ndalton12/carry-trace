"""Model runner abstractions."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterable, Iterator
from importlib.util import find_spec
from typing import Any, Protocol

from carry_trace.config import GenerationParams, ModelSpec, RunnerConfig
from carry_trace.enums import QuantizationKind, RunnerKind, TorchDType
from carry_trace.io import utc_now_iso
from carry_trace.parsing import parse_final_answer
from carry_trace.schemas import AdditionExample, ModelCallRecord

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
FORCED_THINKING_CLOSE = "</think>\nFinal answer:"
HF_GENERATE_PARAM_NAMES = {
    "max_new_tokens",
    "temperature",
    "top_p",
    "do_sample",
}


class ModelRunner(Protocol):
    def generate(
        self,
        examples: Iterable[AdditionExample],
        run_id: str,
        seed: int,
    ) -> Iterator[ModelCallRecord]:
        """Generate model-call records for examples."""
        ...


class FakeModelRunner:
    """Deterministic runner for tests and CLI smoke runs."""

    def __init__(self, model: ModelSpec, generation: GenerationParams):
        """Store fake-runner model metadata and generation settings."""
        self.model = model
        self.generation = generation
        self.git_commit = git_commit_hash()

    def generate(
        self,
        examples: Iterable[AdditionExample],
        run_id: str,
        seed: int,
    ) -> Iterator[ModelCallRecord]:
        """Yield deterministic correct records for each example."""
        for example in examples:
            started = time.perf_counter()
            decoded = f"Answer: {example.expected_output or example.answer}"
            output_ids = [ord(char) for char in decoded]
            input_ids = [ord(char) for char in example.prompt]
            yield ModelCallRecord(
                run_id=run_id,
                example_id=example.id,
                model_name=self.model.name,
                model_id=self.model.model_id,
                model_revision=self.model.revision,
                tokenizer_id=self.model.tokenizer_id,
                runner_kind=RunnerKind.FAKE,
                seed=seed,
                timestamp=utc_now_iso(),
                prompt=example.prompt,
                messages=example.messages,
                rendered_prompt=example.prompt,
                input_ids=input_ids,
                output_ids=output_ids,
                decoded_output=decoded,
                parsed_answer=parse_final_answer(
                    decoded,
                    base=example.base,
                    answer_format=example.answer_format,
                ),
                generation_config=self.generation.model_dump(mode="json"),
                token_count_input=len(input_ids),
                token_count_output=len(output_ids),
                latency_seconds=time.perf_counter() - started,
                git_commit=self.git_commit,
            )


class HuggingFaceModelRunner:
    """Hugging Face Transformers causal-LM runner."""

    def __init__(self, model: ModelSpec, runner: RunnerConfig, generation: GenerationParams):
        """Store runner settings and load the Hugging Face model."""
        self.model_spec = model
        self.runner = runner
        self.generation = generation
        self.git_commit = git_commit_hash()
        self._load()

    def _load(self) -> None:
        """Load tokenizer and model objects from Hugging Face."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_id = self.model_spec.tokenizer_id or self.model_spec.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=self.model_spec.revision,
            trust_remote_code=self.runner.trust_remote_code,
        )
        self._prepare_tokenizer_for_batching()
        dtype = _torch_dtype(torch, self.runner.dtype)
        model_kwargs: dict[str, Any] = {
            "revision": self.model_spec.revision,
            "trust_remote_code": self.runner.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        quantization_config = self._quantization_config()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if self.runner.device == "auto":
            model_kwargs["device_map"] = "auto"
        elif quantization_config is not None:
            model_kwargs["device_map"] = {"": self.runner.device}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_spec.model_id, **model_kwargs)
        if self.runner.device != "auto" and quantization_config is None:
            self.model.to(self.runner.device)
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def _prepare_tokenizer_for_batching(self) -> None:
        """Configure tokenizer padding needed for batched decoder-only generation."""
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is not None:
            return
        if self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        raise ValueError("batched Hugging Face generation requires a tokenizer pad or eos token")

    def _quantization_config(self) -> Any | None:
        """Return the configured Hugging Face quantization object, if any."""
        if self.runner.quantization == QuantizationKind.NONE:
            return None
        if self.runner.quantization == QuantizationKind.BITSANDBYTES_8BIT:
            if find_spec("bitsandbytes") is None:
                raise RuntimeError(
                    "runner.quantization=bitsandbytes_8bit requires bitsandbytes; "
                    "install with `pip install bitsandbytes` or `uv pip install bitsandbytes`"
                )
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"unsupported quantization mode {self.runner.quantization!r}")

    def generate(
        self,
        examples: Iterable[AdditionExample],
        run_id: str,
        seed: int,
    ) -> Iterator[ModelCallRecord]:
        """Generate records from a Hugging Face causal language model."""
        import torch

        torch.manual_seed(seed)
        generation_config = self.generation.model_dump(mode="json")
        generate_kwargs = self._hf_generate_kwargs()
        for batch in _batched(examples, self.runner.batch_size):
            started = time.perf_counter()
            rendered_prompts, inputs, input_ids_by_row = self._tokenize_batch(batch)
            with torch.no_grad():
                outputs = self._generate_outputs(
                    inputs,
                    input_ids_by_row=input_ids_by_row,
                    rendered_prompts=rendered_prompts,
                    generate_kwargs=generate_kwargs,
                )
            latency_seconds = (time.perf_counter() - started) / len(batch)
            for example, rendered_prompt, input_ids, (output_ids, decoded, metadata) in zip(
                batch,
                rendered_prompts,
                input_ids_by_row,
                outputs,
                strict=True,
            ):
                metadata = {**metadata, "hf_batch_size": len(batch)}
                yield ModelCallRecord(
                    run_id=run_id,
                    example_id=example.id,
                    model_name=self.model_spec.name,
                    model_id=self.model_spec.model_id,
                    model_revision=self.model_spec.revision,
                    tokenizer_id=self.model_spec.tokenizer_id,
                    tokenizer_revision=self.model_spec.revision,
                    runner_kind=RunnerKind.HF,
                    seed=seed,
                    timestamp=utc_now_iso(),
                    prompt=example.prompt,
                    messages=example.messages,
                    rendered_prompt=rendered_prompt,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    decoded_output=decoded,
                    parsed_answer=parse_final_answer(
                        decoded,
                        base=example.base,
                        answer_format=example.answer_format,
                    ),
                    generation_config=generation_config,
                    token_count_input=len(input_ids),
                    token_count_output=len(output_ids),
                    latency_seconds=latency_seconds,
                    git_commit=self.git_commit,
                    metadata=metadata,
                )

    def _tokenize(self, example: AdditionExample) -> tuple[str, dict[str, Any]]:
        """Tokenize an example prompt, applying a chat template when available."""
        rendered_prompts, inputs, _ = self._tokenize_batch([example])
        return rendered_prompts[0], inputs

    def _tokenize_batch(
        self,
        examples: list[AdditionExample],
    ) -> tuple[list[str], dict[str, Any], list[list[int]]]:
        """Tokenize a batch of examples with left padding for causal generation."""
        use_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        if use_chat_template:
            rendered_prompts = [
                self.tokenizer.apply_chat_template(
                    example.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for example in examples
            ]
            inputs = self.tokenizer(
                rendered_prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        else:
            rendered_prompts = [example.prompt for example in examples]
            inputs = self.tokenizer(rendered_prompts, padding=True, return_tensors="pt")
        input_ids_by_row = _unpadded_rows(inputs["input_ids"], inputs["attention_mask"])
        return (
            rendered_prompts,
            {key: value.to(self.model.device) for key, value in inputs.items()},
            input_ids_by_row,
        )

    def _hf_generate_kwargs(self) -> dict[str, Any]:
        """Return only generation parameters accepted by Hugging Face generate."""
        return {
            key: value
            for key, value in self.generation.model_dump(mode="json").items()
            if key in HF_GENERATE_PARAM_NAMES
        }

    def _generate_outputs(
        self,
        inputs: dict[str, Any],
        input_ids_by_row: list[list[int]],
        rendered_prompts: list[str],
        generate_kwargs: dict[str, Any],
    ) -> list[tuple[list[int], str, dict[str, Any]]]:
        """Generate output records for a batch, optionally force-closing think blocks."""
        final_answer_tokens = self.generation.thinking_final_answer_tokens
        if not self.generation.force_close_thinking or final_answer_tokens is None:
            outputs = self.model.generate(**inputs, **generate_kwargs)
            return self._decode_batch_outputs(
                outputs,
                int(inputs["input_ids"].shape[-1]),
            )

        first_pass_tokens = self.generation.max_new_tokens - final_answer_tokens
        first_kwargs = {**generate_kwargs, "max_new_tokens": first_pass_tokens}
        first_outputs = self.model.generate(**inputs, **first_kwargs)
        first_output_ids_by_row = self._generated_rows(
            first_outputs,
            int(inputs["input_ids"].shape[-1]),
        )
        records: list[tuple[list[int], str, dict[str, Any]] | None] = [None] * len(
            first_output_ids_by_row
        )
        forced_indices: list[int] = []
        continue_indices: list[int] = []
        first_texts = [
            self.tokenizer.decode(output_ids, skip_special_tokens=False)
            for output_ids in first_output_ids_by_row
        ]
        for index, (first_output_ids, first_text, rendered_prompt) in enumerate(
            zip(first_output_ids_by_row, first_texts, rendered_prompts, strict=True)
        ):
            hit_thinking_cap = len(first_output_ids) >= first_pass_tokens
            if hit_thinking_cap and has_unclosed_thinking(rendered_prompt + first_text):
                forced_indices.append(index)
            elif hit_thinking_cap:
                continue_indices.append(index)
            else:
                records[index] = (
                    first_output_ids,
                    self.tokenizer.decode(first_output_ids, skip_special_tokens=True),
                    {},
                )

        if continue_indices:
            contexts = [
                input_ids_by_row[index] + first_output_ids_by_row[index]
                for index in continue_indices
            ]
            continuation_ids_by_row = self._continue_generation_batch(
                contexts,
                final_answer_tokens,
                generate_kwargs,
            )
            for index, continuation_ids in zip(
                continue_indices,
                continuation_ids_by_row,
                strict=True,
            ):
                output_ids = first_output_ids_by_row[index] + continuation_ids
                records[index] = (
                    output_ids,
                    self.tokenizer.decode(output_ids, skip_special_tokens=True),
                    {},
                )

        if forced_indices:
            close_ids = self._encode_continuation(FORCED_THINKING_CLOSE)
            contexts = [
                input_ids_by_row[index] + first_output_ids_by_row[index] + close_ids
                for index in forced_indices
            ]
            continuation_ids_by_row = self._continue_generation_batch(
                contexts,
                final_answer_tokens,
                generate_kwargs,
            )
            for index, continuation_ids in zip(
                forced_indices,
                continuation_ids_by_row,
                strict=True,
            ):
                output_ids = first_output_ids_by_row[index] + close_ids + continuation_ids
                decoded = (
                    first_texts[index]
                    + FORCED_THINKING_CLOSE
                    + self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
                )
                records[index] = (
                    output_ids,
                    decoded,
                    {
                        "thinking_force_closed": True,
                        "thinking_first_pass_token_count": len(first_output_ids_by_row[index]),
                        "thinking_final_answer_token_budget": final_answer_tokens,
                    },
                )

        if any(record is None for record in records):
            raise RuntimeError("internal error: missing batched generation record")
        return [record for record in records if record is not None]

    def _continue_generation_batch(
        self,
        contexts: list[list[int]],
        max_new_tokens: int,
        generate_kwargs: dict[str, Any],
    ) -> list[list[int]]:
        """Continue generation from a batch of existing token contexts."""
        input_ids, attention_mask = _left_pad_contexts(
            contexts,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**generate_kwargs, "max_new_tokens": max_new_tokens},
        )
        return self._generated_rows(outputs, int(input_ids.shape[-1]))

    def _decode_batch_outputs(
        self,
        outputs: Any,
        input_width: int,
    ) -> list[tuple[list[int], str, dict[str, Any]]]:
        """Decode generated tokens for each row in a batch."""
        records = []
        for output_ids in self._generated_rows(outputs, input_width):
            records.append(
                (output_ids, self.tokenizer.decode(output_ids, skip_special_tokens=True), {})
            )
        return records

    def _encode_continuation(self, text: str) -> list[int]:
        """Encode continuation text without adding special tokens."""
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return inputs["input_ids"][0].detach().cpu().tolist()

    def _generated_rows(self, outputs: Any, input_width: int) -> list[list[int]]:
        """Extract generated token rows and remove trailing batch padding."""
        return [
            _strip_trailing_token(row, self.tokenizer.pad_token_id)
            for row in _generated_rows(outputs, input_width)
        ]


def make_runner(
    model: ModelSpec,
    runner: RunnerConfig,
    generation: GenerationParams,
) -> ModelRunner:
    """Create the configured model runner implementation."""
    if runner.kind == RunnerKind.FAKE:
        return FakeModelRunner(model, generation)
    return HuggingFaceModelRunner(model, runner, generation)


def _batched(
    examples: Iterable[AdditionExample],
    batch_size: int,
) -> Iterator[list[AdditionExample]]:
    """Yield examples in fixed-size batches."""
    if batch_size < 1:
        raise ValueError("runner.batch_size must be at least 1")
    batch: list[AdditionExample] = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _unpadded_rows(input_ids: Any, attention_mask: Any) -> list[list[int]]:
    """Return each padded input row with padding tokens removed."""
    rows = []
    for row_ids, row_mask in zip(input_ids, attention_mask, strict=True):
        rows.append(row_ids[row_mask.bool()].detach().cpu().tolist())
    return rows


def _left_pad_contexts(contexts: list[list[int]], pad_token_id: int) -> tuple[Any, Any]:
    """Left-pad token-id contexts and return input IDs plus attention masks."""
    import torch

    max_length = max(len(context) for context in contexts)
    input_rows = []
    mask_rows = []
    for context in contexts:
        pad_length = max_length - len(context)
        input_rows.append([pad_token_id] * pad_length + context)
        mask_rows.append([0] * pad_length + [1] * len(context))
    return torch.tensor(input_rows, dtype=torch.long), torch.tensor(mask_rows, dtype=torch.long)


def _generated_rows(outputs: Any, input_width: int) -> list[list[int]]:
    """Extract generated token rows from a padded batch output tensor."""
    return [row[input_width:].detach().cpu().tolist() for row in outputs]


def _strip_trailing_token(token_ids: list[int], token_id: int | None) -> list[int]:
    """Remove trailing instances of one token ID from a token-id list."""
    if token_id is None:
        return token_ids
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] == token_id:
        end -= 1
    return token_ids[:end]


def has_unclosed_thinking(text: str) -> bool:
    """Return whether decoded text has an opened but unclosed think block."""
    return text.rfind(THINK_OPEN) > text.rfind(THINK_CLOSE)


def git_commit_hash() -> str | None:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _torch_dtype(torch: Any, dtype: TorchDType | str) -> Any:
    """Map a configured dtype enum to a torch dtype object."""
    dtype = TorchDType(dtype)
    if dtype == TorchDType.AUTO:
        return None
    mapping = {
        TorchDType.FLOAT16: torch.float16,
        TorchDType.BFLOAT16: torch.bfloat16,
        TorchDType.FLOAT32: torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return mapping[dtype]
