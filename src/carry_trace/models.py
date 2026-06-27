"""Model runner abstractions."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterable, Iterator
from importlib.util import find_spec
from typing import Any, Protocol

from carry_trace.arithmetic import DIGIT_ALPHABET
from carry_trace.config import GenerationParams, ModelSpec, RunnerConfig
from carry_trace.enums import AnswerFormat, QuantizationKind, RunnerKind, TorchDType
from carry_trace.io import utc_now_iso
from carry_trace.parsing import normalize_output_digits, parse_final_answer
from carry_trace.schemas import AdditionExample, ModelCallRecord

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
HF_GENERATE_PARAM_NAMES = {
    "max_new_tokens",
    "temperature",
    "top_p",
    "do_sample",
}
CHAT_TURN_STOP_TOKENS = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_turn|>",
)


class AnswerDigitStoppingCriteria:
    """Stop generation once every batch row has emitted enough answer digits."""

    def __init__(
        self,
        tokenizer: Any,
        input_width: int,
        expected_output_lengths: list[int],
    ):
        """Store tokenizer and expected emitted-answer lengths for a continuation batch."""
        self.tokenizer = tokenizer
        self.input_width = input_width
        self.expected_output_lengths = expected_output_lengths

    def __call__(self, input_ids: Any, scores: Any, **kwargs: object) -> bool:
        """Return true when every row has emitted its expected number of digits."""
        for row, expected_length in zip(input_ids, self.expected_output_lengths, strict=True):
            generated_ids = row[self.input_width :].detach().cpu().tolist()
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            output_digits = normalize_output_digits(generated_text)
            if output_digits is None or len(output_digits) < expected_length:
                return False
        return True


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
                    answer_formats=[example.answer_format for example in batch],
                    expected_output_lengths=[
                        _expected_output_digit_count(example) for example in batch
                    ],
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
        kwargs = {
            key: value
            for key, value in self.generation.model_dump(mode="json").items()
            if key in HF_GENERATE_PARAM_NAMES
        }
        stop_token_ids = _generation_stop_token_ids(self.tokenizer, self.model)
        if stop_token_ids:
            kwargs["eos_token_id"] = (
                stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids
            )
        if self.tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = int(self.tokenizer.pad_token_id)
        return kwargs

    def _generate_outputs(
        self,
        inputs: dict[str, Any],
        input_ids_by_row: list[list[int]],
        rendered_prompts: list[str],
        answer_formats: list[Any],
        expected_output_lengths: list[int],
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
            forced_groups: dict[tuple[str, int], list[int]] = {}
            for index in forced_indices:
                close_text = _forced_thinking_close_text(answer_formats[index])
                expected_length = expected_output_lengths[index]
                forced_groups.setdefault((close_text, expected_length), []).append(index)
            for (close_text, expected_length), group_indices in forced_groups.items():
                close_ids = self._encode_continuation(close_text)
                contexts = [
                    input_ids_by_row[index] + first_output_ids_by_row[index] + close_ids
                    for index in group_indices
                ]
                continuation_ids_by_row = self._continue_generation_batch(
                    contexts,
                    final_answer_tokens,
                    generate_kwargs,
                    expected_output_lengths=[expected_length] * len(group_indices),
                )
                for index, continuation_ids in zip(
                    group_indices,
                    continuation_ids_by_row,
                    strict=True,
                ):
                    output_ids = first_output_ids_by_row[index] + close_ids + continuation_ids
                    decoded = (
                        first_texts[index]
                        + close_text
                        + self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
                    )
                    records[index] = (
                        output_ids,
                        decoded,
                        {
                            "thinking_force_closed": True,
                            "thinking_force_close_text": close_text,
                            "thinking_first_pass_token_count": len(first_output_ids_by_row[index]),
                            "thinking_final_answer_token_budget": final_answer_tokens,
                            "thinking_stop_expected_output_digits": expected_length,
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
        expected_output_lengths: list[int] | None = None,
    ) -> list[list[int]]:
        """Continue generation from a batch of existing token contexts."""
        input_ids, attention_mask = _left_pad_contexts(
            contexts,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        kwargs = {**generate_kwargs, "max_new_tokens": max_new_tokens}
        if expected_output_lengths is not None:
            from transformers import StoppingCriteriaList

            kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    AnswerDigitStoppingCriteria(
                        tokenizer=self.tokenizer,
                        input_width=int(input_ids.shape[-1]),
                        expected_output_lengths=expected_output_lengths,
                    )
                ]
            )
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
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


class VllmModelRunner:
    """vLLM offline batched inference runner."""

    def __init__(self, model: ModelSpec, runner: RunnerConfig, generation: GenerationParams):
        """Store runner settings and load the vLLM engine."""
        self.model_spec = model
        self.runner = runner
        self.generation = generation
        self.git_commit = git_commit_hash()
        self._load()

    def _load(self) -> None:
        """Load a tokenizer and initialize the vLLM engine."""
        if find_spec("vllm") is None:
            raise RuntimeError(
                "runner.kind=vllm requires vLLM; install with "
                "`pip install vllm` or `uv pip install vllm --torch-backend=auto`"
            )
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer_id = self.model_spec.tokenizer_id or self.model_spec.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=self.model_spec.revision,
            trust_remote_code=self.runner.trust_remote_code,
        )
        self.sampling_params_cls = SamplingParams
        engine_kwargs: dict[str, Any] = {
            "model": self.model_spec.model_id,
            "trust_remote_code": self.runner.trust_remote_code,
            "dtype": _vllm_dtype(self.runner.dtype),
            "tensor_parallel_size": self.runner.tensor_parallel_size,
            "skip_tokenizer_init": True,
            "enforce_eager": self.runner.enforce_eager,
        }
        if self.model_spec.revision is not None:
            engine_kwargs["revision"] = self.model_spec.revision
        if self.runner.gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = self.runner.gpu_memory_utilization
        if self.runner.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.runner.max_model_len
        quantization = self._vllm_quantization()
        if quantization is not None:
            engine_kwargs["quantization"] = quantization
        self.llm = LLM(**engine_kwargs)

    def _vllm_quantization(self) -> str | None:
        """Return the configured vLLM quantization argument, if any."""
        if self.runner.quantization == QuantizationKind.NONE:
            return None
        if self.runner.quantization == QuantizationKind.BITSANDBYTES_8BIT:
            return "bitsandbytes"
        raise ValueError(f"unsupported quantization mode {self.runner.quantization!r}")

    def generate(
        self,
        examples: Iterable[AdditionExample],
        run_id: str,
        seed: int,
    ) -> Iterator[ModelCallRecord]:
        """Generate records from a vLLM offline inference engine."""
        generation_config = self.generation.model_dump(mode="json")
        for batch in _batched(examples, self.runner.batch_size):
            started = time.perf_counter()
            rendered_prompts, input_ids_by_row = self._render_and_tokenize_batch(batch)
            outputs = self._generate_outputs(
                batch=batch,
                rendered_prompts=rendered_prompts,
                input_ids_by_row=input_ids_by_row,
            )
            latency_seconds = (time.perf_counter() - started) / len(batch)
            for example, rendered_prompt, input_ids, (output_ids, decoded, metadata) in zip(
                batch,
                rendered_prompts,
                input_ids_by_row,
                outputs,
                strict=True,
            ):
                metadata = {**metadata, "vllm_batch_size": len(batch)}
                yield ModelCallRecord(
                    run_id=run_id,
                    example_id=example.id,
                    model_name=self.model_spec.name,
                    model_id=self.model_spec.model_id,
                    model_revision=self.model_spec.revision,
                    tokenizer_id=self.model_spec.tokenizer_id,
                    tokenizer_revision=self.model_spec.revision,
                    runner_kind=RunnerKind.VLLM,
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

    def _render_and_tokenize_batch(
        self,
        examples: list[AdditionExample],
    ) -> tuple[list[str], list[list[int]]]:
        """Render prompts and tokenize them for saved call metadata."""
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
            add_special_tokens = False
        else:
            rendered_prompts = [example.prompt for example in examples]
            add_special_tokens = True
        input_ids_by_row = [
            _tokenize_text_ids(self.tokenizer, prompt, add_special_tokens=add_special_tokens)
            for prompt in rendered_prompts
        ]
        return rendered_prompts, input_ids_by_row

    def _generate_outputs(
        self,
        batch: list[AdditionExample],
        rendered_prompts: list[str],
        input_ids_by_row: list[list[int]],
    ) -> list[tuple[list[int], str, dict[str, Any]]]:
        """Generate output records for a vLLM batch, optionally force-closing think blocks."""
        final_answer_tokens = self.generation.thinking_final_answer_tokens
        if not self.generation.force_close_thinking or final_answer_tokens is None:
            request_outputs = self._generate_token_prompts(
                input_ids_by_row,
                self.generation.max_new_tokens,
            )
            return [self._decode_vllm_output(output) for output in request_outputs]

        first_pass_tokens = self.generation.max_new_tokens - final_answer_tokens
        first_outputs = self._generate_token_prompts(input_ids_by_row, first_pass_tokens)
        records: list[tuple[list[int], str, dict[str, Any]] | None] = [None] * len(batch)
        continuation_contexts: list[list[int]] = []
        continuation_indices: list[int] = []
        continuation_close_texts: list[str | None] = []

        first_decoded = [self._decode_vllm_output(output) for output in first_outputs]
        for index, ((first_ids, first_text, _), rendered_prompt, example) in enumerate(
            zip(first_decoded, rendered_prompts, batch, strict=True)
        ):
            hit_thinking_cap = len(first_ids) >= first_pass_tokens
            if hit_thinking_cap and has_unclosed_thinking(rendered_prompt + first_text):
                close_text = _forced_thinking_close_text(example.answer_format)
                close_ids = _tokenize_text_ids(
                    self.tokenizer,
                    close_text,
                    add_special_tokens=False,
                )
                continuation_contexts.append(input_ids_by_row[index] + first_ids + close_ids)
                continuation_indices.append(index)
                continuation_close_texts.append(close_text)
            elif hit_thinking_cap:
                continuation_contexts.append(input_ids_by_row[index] + first_ids)
                continuation_indices.append(index)
                continuation_close_texts.append(None)
            else:
                records[index] = (first_ids, first_text, {})

        if continuation_contexts:
            continuation_outputs = self._generate_token_prompts(
                continuation_contexts,
                final_answer_tokens,
            )
            for index, close_text, request_output in zip(
                continuation_indices,
                continuation_close_texts,
                continuation_outputs,
                strict=True,
            ):
                first_ids, first_text, _ = first_decoded[index]
                continuation_ids, continuation_text, _ = self._decode_vllm_output(request_output)
                if close_text is not None:
                    expected_length = _expected_output_digit_count(batch[index])
                    continuation_text = _trim_after_answer_digits(
                        continuation_text,
                        expected_length=expected_length,
                        base=batch[index].base,
                    )
                    continuation_ids = _tokenize_text_ids(
                        self.tokenizer,
                        continuation_text,
                        add_special_tokens=False,
                    )
                    close_ids = _tokenize_text_ids(
                        self.tokenizer,
                        close_text,
                        add_special_tokens=False,
                    )
                    records[index] = (
                        first_ids + close_ids + continuation_ids,
                        first_text + close_text + continuation_text,
                        {
                            "thinking_force_closed": True,
                            "thinking_force_close_text": close_text,
                            "thinking_first_pass_token_count": len(first_ids),
                            "thinking_final_answer_token_budget": final_answer_tokens,
                            "thinking_stop_expected_output_digits": expected_length,
                        },
                    )
                else:
                    records[index] = (
                        first_ids + continuation_ids,
                        first_text + continuation_text,
                        {},
                    )

        if any(record is None for record in records):
            raise RuntimeError("internal error: missing vLLM generation record")
        return [record for record in records if record is not None]

    def _sampling_params(self, max_tokens: int) -> Any:
        """Create vLLM sampling parameters for one generation phase."""
        temperature = self.generation.temperature if self.generation.do_sample else 0.0
        return self.sampling_params_cls(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.generation.top_p,
            skip_special_tokens=False,
        )

    def _generate_token_prompts(self, prompt_token_ids: list[list[int]], max_tokens: int) -> Any:
        """Generate from pre-tokenized prompts so vLLM does not initialize a tokenizer."""
        prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]
        return self.llm.generate(prompts, self._sampling_params(max_tokens))

    def _decode_vllm_output(self, request_output: Any) -> tuple[list[int], str, dict[str, Any]]:
        """Extract token IDs and text from a vLLM request output."""
        output = request_output.outputs[0]
        token_ids = getattr(output, "token_ids", None)
        if token_ids is None:
            text = getattr(output, "text", "")
            token_ids = _tokenize_text_ids(self.tokenizer, text, add_special_tokens=False)
        else:
            text = getattr(output, "text", "") or self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
            )
        return list(token_ids), text, {}


def make_runner(
    model: ModelSpec,
    runner: RunnerConfig,
    generation: GenerationParams,
) -> ModelRunner:
    """Create the configured model runner implementation."""
    if runner.kind == RunnerKind.FAKE:
        return FakeModelRunner(model, generation)
    if runner.kind == RunnerKind.HF:
        return HuggingFaceModelRunner(model, runner, generation)
    if runner.kind == RunnerKind.VLLM:
        return VllmModelRunner(model, runner, generation)
    raise ValueError(f"unsupported runner kind {runner.kind!r}")


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


def _expected_output_digit_count(example: AdditionExample) -> int:
    """Return the expected number of emitted answer digits for an example."""
    output = normalize_output_digits(example.expected_output or example.answer, base=example.base)
    if output is None:
        raise ValueError(f"example {example.id} has no expected output digits")
    return len(output)


def _forced_thinking_close_text(answer_format: AnswerFormat | str) -> str:
    """Return the forced visible-answer prefix for a requested answer format."""
    answer_format = AnswerFormat(answer_format)
    if answer_format == AnswerFormat.STANDARD:
        return "</think>\nFinal answer:"
    if answer_format == AnswerFormat.LSD:
        return "</think>\nFinal answer digits from right to left with no separators:"
    raise ValueError(f"unknown answer format {answer_format!r}")


def _tokenize_text_ids(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    """Tokenize one string and return a flat list of token IDs."""
    inputs = tokenizer(text, add_special_tokens=add_special_tokens)
    input_ids = inputs["input_ids"]
    if hasattr(input_ids, "detach"):
        values = input_ids.detach().cpu().tolist()
    else:
        values = input_ids
    if values and isinstance(values[0], list):
        return list(values[0])
    return list(values)


def _trim_after_answer_digits(text: str, expected_length: int, base: int = 10) -> str:
    """Return text truncated immediately after the expected number of output digits."""
    allowed = set(DIGIT_ALPHABET[:base])
    seen = 0
    for index, char in enumerate(text.upper()):
        if char in allowed:
            seen += 1
            if seen >= expected_length:
                return text[: index + 1]
    return text


def _generation_stop_token_ids(tokenizer: Any, model: Any) -> list[int]:
    """Return EOS and chat-turn stop token IDs supported by the tokenizer/model."""
    token_ids: list[int] = []

    def add_token_id(value: object) -> None:
        """Add one or more valid token IDs while preserving order."""
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_token_id(item)
            return
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            return
        if token_id >= 0 and token_id not in token_ids:
            token_ids.append(token_id)

    add_token_id(getattr(tokenizer, "eos_token_id", None))
    add_token_id(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    add_token_id(getattr(getattr(model, "config", None), "eos_token_id", None))

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        for token in CHAT_TURN_STOP_TOKENS:
            token_id = convert(token)
            if token_id is not None and token_id != unk_token_id:
                add_token_id(token_id)

    return token_ids


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


def _vllm_dtype(dtype: TorchDType | str) -> str:
    """Map a configured dtype enum to a vLLM dtype string."""
    dtype = TorchDType(dtype)
    if dtype == TorchDType.AUTO:
        return "auto"
    return dtype.value
