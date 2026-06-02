"""Model runner abstractions."""

from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Iterable, Iterator
from typing import Any, Protocol

from carry_trace.config import GenerationParams, ModelSpec, RunnerConfig
from carry_trace.enums import RunnerKind, TorchDType
from carry_trace.io import utc_now_iso
from carry_trace.parsing import parse_final_answer
from carry_trace.schemas import AdditionExample, ModelCallRecord


class ModelRunner(Protocol):
    def generate(
        self,
        examples: Iterable[AdditionExample],
        run_id: str,
        seed: int,
    ) -> Iterator[ModelCallRecord]:
        ...


class FakeModelRunner:
    """Deterministic runner for tests and CLI smoke runs."""

    def __init__(self, model: ModelSpec, generation: GenerationParams):
        self.model = model
        self.generation = generation

    def generate(
        self,
        examples: Iterable[AdditionExample],
        run_id: str,
        seed: int,
    ) -> Iterator[ModelCallRecord]:
        for example in examples:
            started = time.perf_counter()
            decoded = f"Answer: {example.answer}"
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
                parsed_answer=parse_final_answer(decoded, base=example.base),
                generation_config=self.generation.model_dump(mode="json"),
                token_count_input=len(input_ids),
                token_count_output=len(output_ids),
                latency_seconds=time.perf_counter() - started,
                package_versions=package_versions(),
            )


class HuggingFaceModelRunner:
    """Hugging Face Transformers causal-LM runner."""

    def __init__(self, model: ModelSpec, runner: RunnerConfig, generation: GenerationParams):
        self.model_spec = model
        self.runner = runner
        self.generation = generation
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_id = self.model_spec.tokenizer_id or self.model_spec.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=self.model_spec.revision,
            trust_remote_code=self.runner.trust_remote_code,
        )
        dtype = _torch_dtype(torch, self.runner.dtype)
        model_kwargs: dict[str, Any] = {
            "revision": self.model_spec.revision,
            "trust_remote_code": self.runner.trust_remote_code,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.runner.device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_spec.model_id, **model_kwargs)
        if self.runner.device != "auto":
            self.model.to(self.runner.device)
        self.model.eval()

    def generate(
        self,
        examples: Iterable[AdditionExample],
        run_id: str,
        seed: int,
    ) -> Iterator[ModelCallRecord]:
        import torch

        torch.manual_seed(seed)
        for example in examples:
            started = time.perf_counter()
            rendered_prompt, inputs = self._tokenize(example)
            input_len = int(inputs["input_ids"].shape[-1])
            generate_kwargs = self.generation.model_dump(mode="json")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
            output_ids_tensor = outputs[0][input_len:]
            output_ids = output_ids_tensor.detach().cpu().tolist()
            decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
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
                input_ids=inputs["input_ids"][0].detach().cpu().tolist(),
                output_ids=output_ids,
                decoded_output=decoded,
                parsed_answer=parse_final_answer(decoded, base=example.base),
                generation_config=generate_kwargs,
                token_count_input=input_len,
                token_count_output=len(output_ids),
                latency_seconds=time.perf_counter() - started,
                package_versions=package_versions(),
            )

    def _tokenize(self, example: AdditionExample) -> tuple[str, dict[str, Any]]:
        if getattr(self.tokenizer, "chat_template", None):
            inputs = self.tokenizer.apply_chat_template(
                example.messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            rendered_prompt = self.tokenizer.apply_chat_template(
                example.messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            rendered_prompt = example.prompt
            inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        return rendered_prompt, {key: value.to(self.model.device) for key, value in inputs.items()}


def make_runner(
    model: ModelSpec,
    runner: RunnerConfig,
    generation: GenerationParams,
) -> ModelRunner:
    if runner.kind == RunnerKind.FAKE:
        return FakeModelRunner(model, generation)
    return HuggingFaceModelRunner(model, runner, generation)


def package_versions() -> dict[str, str]:
    packages = ["carry-trace", "transformers", "torch", "accelerate", "datasets"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _torch_dtype(torch: Any, dtype: TorchDType | str) -> Any:
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
