"""Goal 2 activation extraction orchestration."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any

from carry_trace.config import ActivationExtractionConfig, Goal2Config
from carry_trace.datasets import dump_dataset_row
from carry_trace.enums import ActivationLocation, AnswerFormat, RunnerKind
from carry_trace.io import (
    append_jsonl,
    ensure_dir,
    read_json,
    read_jsonl,
    stable_hash,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from carry_trace.models import HuggingFaceModelRunner, _torch_dtype, git_commit_hash, make_runner
from carry_trace.parsing import normalize_output_digits
from carry_trace.runs import _cleanup_runner, _load_repairable_jsonl
from carry_trace.schemas import AdditionExample, ModelCallRecord


def run_goal2(config: Goal2Config) -> Path:
    """Run Goal 2 activation extraction and write local artifacts."""
    if config.runner.kind != RunnerKind.HF:
        raise ValueError("Goal 2 activation extraction currently requires runner.kind=hf")

    config_hash = stable_hash(config.model_dump(mode="json"))
    examples = _load_goal2_examples(config)
    expected_record_count = len(examples) * len(config.models)
    run_dir = _find_resumable_goal2_run_dir(
        output_dir=config.output_dir,
        run_name=config.name,
        config_hash=config_hash,
        expected_record_count=expected_record_count,
    )
    if run_dir is None:
        run_id = f"{config.name}-{utc_now_iso().replace(':', '').replace('+', 'Z')}"
        run_dir = ensure_dir(config.output_dir / run_id)
        created_at = utc_now_iso()
    else:
        run_id = _manifest_value(run_dir, "run_id") or run_dir.name
        created_at = _manifest_value(run_dir, "created_at") or utc_now_iso()

    write_jsonl(run_dir / "dataset.jsonl", [dump_dataset_row(example) for example in examples])
    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": utc_now_iso(),
        "status": "running",
        "config_hash": config_hash,
        "config": config.model_dump(mode="json"),
        "dataset_path": str(config.dataset_path),
        "example_count": len(examples),
        "expected_record_count": expected_record_count,
        "artifact_kind": "goal2_activations",
    }
    write_json(run_dir / "manifest.json", manifest)

    index_path = run_dir / "activations.jsonl"
    existing_records = _load_repairable_jsonl(index_path)
    completed = _completed_activation_keys(run_dir, existing_records)
    examples_by_id = {example.id: example for example in examples}
    for model in config.models:
        model_examples = [
            example for example in examples if _model_example_key(model, example) not in completed
        ]
        completed_for_model = len(examples) - len(model_examples)
        print(
            f"[goal2] model={model.name} pending={len(model_examples)} "
            f"completed={completed_for_model}/{len(examples)}",
            flush=True,
        )
        if not model_examples:
            continue
        runner = make_runner(model, config.runner, config.generation)
        if not isinstance(runner, HuggingFaceModelRunner):
            raise TypeError("Goal 2 activation extraction expected a HuggingFaceModelRunner")
        started = time.perf_counter()
        try:
            for call in runner.generate(model_examples, run_id=run_id, seed=config.seed):
                example = examples_by_id[call.example_id]
                row = _extract_and_save_activations(
                    runner=runner,
                    example=example,
                    call=call,
                    run_dir=run_dir,
                    activation_config=config.activations,
                )
                append_jsonl(index_path, [row])
                completed.add(_activation_key(row))
                completed_for_model += 1
                elapsed = time.perf_counter() - started
                print(
                    f"[goal2] model={model.name} example={completed_for_model}/{len(examples)} "
                    f"example_id={call.example_id} locations={row['location_count']} "
                    f"output_tokens={call.token_count_output} elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
        finally:
            _cleanup_runner(runner)

    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now_iso()
    manifest["updated_at"] = manifest["completed_at"]
    write_json(run_dir / "manifest.json", manifest)
    if config.upload.enabled and config.upload.repo_id is not None:
        manifest["upload"] = _try_upload_goal2_run(run_dir, config)
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _load_goal2_examples(config: Goal2Config) -> list[AdditionExample]:
    """Load and filter dataset examples for a Goal 2 run."""
    examples = [AdditionExample.model_validate(row) for row in read_jsonl(config.dataset_path)]
    if config.splits is not None:
        allowed = set(config.splits)
        examples = [example for example in examples if example.split in allowed]
    if config.prompt_modes is not None:
        allowed = set(config.prompt_modes)
        examples = [example for example in examples if example.prompt_mode in allowed]
    if config.digit_formats is not None:
        allowed = set(config.digit_formats)
        examples = [example for example in examples if example.digit_format in allowed]
    if config.answer_formats is not None:
        allowed = set(config.answer_formats)
        examples = [example for example in examples if example.answer_format in allowed]
    if config.max_examples is not None:
        examples = examples[: config.max_examples]
    return examples


def _extract_and_save_activations(
    runner: HuggingFaceModelRunner,
    example: AdditionExample,
    call: ModelCallRecord,
    run_dir: Path,
    activation_config: ActivationExtractionConfig,
) -> dict[str, Any]:
    """Resolve token locations, collect hidden states, and save one tensor artifact."""
    locations = resolve_activation_locations(
        example=example,
        call=call,
        tokenizer=runner.tokenizer,
        requested_locations=activation_config.locations,
    )
    tensor, layer_indices = _collect_hidden_states(
        runner=runner,
        token_ids=call.input_ids + call.output_ids,
        locations=locations,
        activation_config=activation_config,
    )
    activation_path = _activation_tensor_path(run_dir, call)
    ensure_dir(activation_path.parent)
    _save_activation_tensor(
        activation_path,
        tensor=tensor,
        layer_indices=layer_indices,
        locations=locations,
        call=call,
    )
    relative_path = activation_path.relative_to(run_dir).as_posix()
    return {
        "run_id": call.run_id,
        "example_id": call.example_id,
        "model_name": call.model_name,
        "model_id": call.model_id,
        "model_revision": call.model_revision,
        "tokenizer_id": call.tokenizer_id,
        "tokenizer_revision": call.tokenizer_revision,
        "runner_kind": call.runner_kind,
        "seed": call.seed,
        "timestamp": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "activation_path": relative_path,
        "activation_shape": list(tensor.shape),
        "activation_dtype": str(tensor.dtype).replace("torch.", ""),
        "layer_indices": layer_indices,
        "location_count": len(locations),
        "locations": locations,
        "generation_config": call.generation_config,
        "token_count_input": call.token_count_input,
        "token_count_output": call.token_count_output,
        "latency_seconds": call.latency_seconds,
        "decoded_output": call.decoded_output,
        "parsed_answer": call.parsed_answer,
        "call_metadata": call.metadata,
    }


def resolve_activation_locations(
    example: AdditionExample,
    call: ModelCallRecord,
    tokenizer: Any,
    requested_locations: list[ActivationLocation],
) -> list[dict[str, Any]]:
    """Resolve symbolic activation locations to absolute token indices."""
    requested = {ActivationLocation(location) for location in requested_locations}
    locations: list[dict[str, Any]] = []
    prompt_offsets = _token_offsets(tokenizer, call.rendered_prompt, call.input_ids)
    generated_text = tokenizer.decode(call.output_ids, skip_special_tokens=False)
    output_offsets = _token_offsets(tokenizer, generated_text, call.output_ids)
    user_span = _user_prompt_span(call.rendered_prompt, example.prompt)

    if ActivationLocation.OPERAND_DIGITS in requested and prompt_offsets is not None:
        _add_operand_digit_locations(
            locations,
            example=example,
            tokenizer=tokenizer,
            rendered_prompt=call.rendered_prompt,
            input_ids=call.input_ids,
            prompt_offsets=prompt_offsets,
            user_span=user_span,
        )
    if ActivationLocation.QUESTION_TOKEN in requested and prompt_offsets is not None:
        _add_question_token_location(
            locations,
            example=example,
            tokenizer=tokenizer,
            rendered_prompt=call.rendered_prompt,
            input_ids=call.input_ids,
            prompt_offsets=prompt_offsets,
            user_span=user_span,
        )
    if ActivationLocation.PROMPT_FINAL in requested and call.input_ids:
        locations.append(
            _location_row(
                name="prompt_final",
                kind=ActivationLocation.PROMPT_FINAL.value,
                absolute_token_index=len(call.input_ids) - 1,
                token_id=call.input_ids[-1],
                token_text=_decode_token(tokenizer, call.input_ids[-1]),
                source="prompt",
            )
        )

    answer = _final_output_digit_chars(generated_text, example.base)
    answer_start_token = _answer_start_token(answer, output_offsets)
    reasoning_count = answer_start_token if answer_start_token is not None else len(call.output_ids)
    _add_cot_locations(
        locations,
        requested=requested,
        tokenizer=tokenizer,
        input_token_count=len(call.input_ids),
        output_ids=call.output_ids,
        reasoning_count=reasoning_count,
    )
    if ActivationLocation.ANSWER_DIGITS in requested and answer is not None:
        _add_answer_digit_locations(
            locations,
            example=example,
            tokenizer=tokenizer,
            input_token_count=len(call.input_ids),
            output_ids=call.output_ids,
            output_offsets=output_offsets,
            answer_chars=answer["digit_chars"],
        )
    return locations


def _add_operand_digit_locations(
    locations: list[dict[str, Any]],
    example: AdditionExample,
    tokenizer: Any,
    rendered_prompt: str,
    input_ids: list[int],
    prompt_offsets: list[tuple[int, int]],
    user_span: tuple[int, int] | None,
) -> None:
    """Append token locations for operand digit characters in the prompt."""
    search_start, search_end = user_span or (0, len(rendered_prompt))
    a_start = rendered_prompt.find(example.prompt_a, search_start, search_end)
    b_start = rendered_prompt.find(example.prompt_b, a_start + len(example.prompt_a), search_end)
    for operand_name, operand_text, operand_start in (
        ("a", example.prompt_a, a_start),
        ("b", example.prompt_b, b_start),
    ):
        if operand_start < 0:
            continue
        digit_spans = _digit_spans(operand_text)
        for digit_span in digit_spans:
            token_index = _token_index_for_span(
                prompt_offsets,
                operand_start + digit_span["start"],
                operand_start + digit_span["end"],
            )
            if token_index is None:
                continue
            locations.append(
                _location_row(
                    name=f"operand_{operand_name}_digit_lsd_{digit_span['lsd_index']}",
                    kind=ActivationLocation.OPERAND_DIGITS.value,
                    absolute_token_index=token_index,
                    token_id=input_ids[token_index],
                    token_text=_decode_token(tokenizer, input_ids[token_index]),
                    source="prompt",
                    metadata={
                        "operand": operand_name,
                        "digit": digit_span["digit"],
                        "lsd_index": digit_span["lsd_index"],
                        "char_start": operand_start + digit_span["start"],
                        "char_end": operand_start + digit_span["end"],
                    },
                )
            )


def _add_question_token_location(
    locations: list[dict[str, Any]],
    example: AdditionExample,
    tokenizer: Any,
    rendered_prompt: str,
    input_ids: list[int],
    prompt_offsets: list[tuple[int, int]],
    user_span: tuple[int, int] | None,
) -> None:
    """Append the token location containing the question mark."""
    search_start, search_end = user_span or (0, len(rendered_prompt))
    question_index = rendered_prompt.find("?", search_start, search_end)
    if question_index < 0:
        question_index = example.prompt.find("?")
        if question_index >= 0 and user_span is not None:
            question_index += user_span[0]
    if question_index < 0:
        return
    token_index = _token_index_for_span(prompt_offsets, question_index, question_index + 1)
    if token_index is None:
        return
    locations.append(
        _location_row(
            name="question_token",
            kind=ActivationLocation.QUESTION_TOKEN.value,
            absolute_token_index=token_index,
            token_id=input_ids[token_index],
            token_text=_decode_token(tokenizer, input_ids[token_index]),
            source="prompt",
            metadata={"char_start": question_index, "char_end": question_index + 1},
        )
    )


def _add_cot_locations(
    locations: list[dict[str, Any]],
    requested: set[ActivationLocation],
    tokenizer: Any,
    input_token_count: int,
    output_ids: list[int],
    reasoning_count: int,
) -> None:
    """Append CoT-relative generated-token locations by token-count thirds."""
    if reasoning_count <= 0:
        return
    specs = [
        (ActivationLocation.COT_START, "cot_start", 0),
        (ActivationLocation.COT_1_3, "cot_1_3", (reasoning_count - 1) // 3),
        (ActivationLocation.COT_2_3, "cot_2_3", ((reasoning_count - 1) * 2) // 3),
        (ActivationLocation.COT_END, "cot_end", reasoning_count - 1),
    ]
    for location_kind, name, output_index in specs:
        if location_kind not in requested:
            continue
        locations.append(
            _location_row(
                name=name,
                kind=location_kind.value,
                absolute_token_index=input_token_count + output_index,
                token_id=output_ids[output_index],
                token_text=_decode_token(tokenizer, output_ids[output_index]),
                source="generated",
                metadata={
                    "output_token_index": output_index,
                    "reasoning_token_count": reasoning_count,
                },
            )
        )


def _add_answer_digit_locations(
    locations: list[dict[str, Any]],
    example: AdditionExample,
    tokenizer: Any,
    input_token_count: int,
    output_ids: list[int],
    output_offsets: list[tuple[int, int]] | None,
    answer_chars: list[dict[str, Any]],
) -> None:
    """Append token locations for final answer digit characters."""
    if output_offsets is None:
        return
    total_digits = len(answer_chars)
    for emitted_index, digit_span in enumerate(answer_chars):
        output_token_index = _token_index_for_span(
            output_offsets,
            int(digit_span["start"]),
            int(digit_span["end"]),
        )
        if output_token_index is None:
            continue
        lsd_index = _answer_lsd_index(example.answer_format, emitted_index, total_digits)
        locations.append(
            _location_row(
                name=f"answer_digit_lsd_{lsd_index}",
                kind=ActivationLocation.ANSWER_DIGITS.value,
                absolute_token_index=input_token_count + output_token_index,
                token_id=output_ids[output_token_index],
                token_text=_decode_token(tokenizer, output_ids[output_token_index]),
                source="generated",
                metadata={
                    "digit": digit_span["digit"],
                    "emitted_index": emitted_index,
                    "lsd_index": lsd_index,
                    "output_token_index": output_token_index,
                    "char_start": digit_span["start"],
                    "char_end": digit_span["end"],
                },
            )
        )


def _collect_hidden_states(
    runner: HuggingFaceModelRunner,
    token_ids: list[int],
    locations: list[dict[str, Any]],
    activation_config: ActivationExtractionConfig,
) -> tuple[Any, list[int | str]]:
    """Run a teacher-forced forward pass and return selected hidden states."""
    import torch

    if not token_ids:
        raise ValueError("cannot extract activations for an empty token sequence")
    device = _model_device(runner.model)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        outputs = runner.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = list(outputs.hidden_states)
    layer_states, layer_indices = _selected_layer_states(
        hidden_states,
        include_embedding_layer=activation_config.include_embedding_layer,
    )
    storage_dtype = _torch_dtype(torch, activation_config.storage_dtype)
    rows = []
    for location in locations:
        token_index = int(location["absolute_token_index"])
        rows.append(torch.stack([state[0, token_index] for state in layer_states]))
    if rows:
        tensor = torch.stack(rows).detach()
        if storage_dtype is not None:
            tensor = tensor.to(dtype=storage_dtype)
        tensor = tensor.cpu().contiguous()
    else:
        hidden_size = int(getattr(runner.model.config, "hidden_size", 0))
        dtype = storage_dtype or layer_states[0].dtype
        tensor = torch.empty((0, len(layer_states), hidden_size), dtype=dtype)
    return tensor, layer_indices


def _selected_layer_states(
    hidden_states: list[Any],
    include_embedding_layer: bool,
) -> tuple[list[Any], list[int | str]]:
    """Return hidden-state tensors and public layer labels to save."""
    if include_embedding_layer:
        return hidden_states, ["embedding", *range(len(hidden_states) - 1)]
    return hidden_states[1:], list(range(len(hidden_states) - 1))


def _save_activation_tensor(
    path: Path,
    tensor: Any,
    layer_indices: list[int | str],
    locations: list[dict[str, Any]],
    call: ModelCallRecord,
) -> None:
    """Save one activation tensor file with lightweight metadata."""
    import torch

    torch.save(
        {
            "activations": tensor,
            "layer_indices": layer_indices,
            "locations": locations,
            "example_id": call.example_id,
            "model_name": call.model_name,
            "model_id": call.model_id,
            "model_revision": call.model_revision,
        },
        path,
    )


def _try_upload_goal2_run(run_dir: Path, config: Goal2Config) -> dict[str, str | None]:
    """Upload a completed Goal 2 run directory to Hugging Face without failing the run."""
    try:
        result = upload_goal2_run_to_hub(
            run_dir,
            repo_id=str(config.upload.repo_id),
            path_in_repo=config.upload.path_in_repo,
            private=config.upload.private,
            revision=config.upload.revision,
            create_pr=config.upload.create_pr,
            create_repo=config.upload.create_repo,
            commit_message=config.upload.commit_message,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc), "commit_url": None}
    return {"status": "complete", **result}


def upload_goal2_run_to_hub(
    run_dir: Path,
    repo_id: str,
    *,
    path_in_repo: str | None = None,
    private: bool = False,
    revision: str | None = None,
    create_pr: bool = False,
    create_repo: bool = True,
    commit_message: str | None = None,
) -> dict[str, str | None]:
    """Upload one Goal 2 activation run directory to a HF dataset repo."""
    from huggingface_hub import HfApi

    upload_path = _validate_hub_path(path_in_repo or run_dir.name)
    token = os.environ.get("HF_TOKEN")
    api = HfApi()
    if create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=run_dir,
        path_in_repo=upload_path,
        revision=revision,
        create_pr=create_pr,
        token=token,
        commit_message=commit_message or f"Upload carry-trace Goal 2 run {run_dir.name}",
        ignore_patterns=[".DS_Store"],
    )
    return {
        "repo_id": repo_id,
        "run_dir": str(run_dir),
        "path_in_repo": upload_path,
        "commit_url": str(getattr(commit_info, "commit_url", "")) or None,
    }


def _find_resumable_goal2_run_dir(
    output_dir: Path,
    run_name: str,
    config_hash: str,
    expected_record_count: int,
) -> Path | None:
    """Return the newest incomplete Goal 2 run directory for the same config hash."""
    candidates = sorted(
        output_dir.glob(f"{run_name}-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("config_hash") != config_hash or manifest.get("completed_at"):
            continue
        records = _load_repairable_jsonl(run_dir / "activations.jsonl")
        completed = _completed_activation_keys(run_dir, records)
        if len(completed) <= expected_record_count:
            return run_dir
    return None


def _completed_activation_keys(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> set[tuple[object, object, object, object]]:
    """Return completed activation keys whose tensor files are present."""
    completed = set()
    for record in records:
        path = record.get("activation_path")
        if not isinstance(path, str) or not (run_dir / path).exists():
            continue
        completed.add(_activation_key(record))
    return completed


def _activation_key(record: dict[str, Any]) -> tuple[object, object, object, object]:
    """Return the resume key for a saved activation record."""
    return (
        record.get("model_name"),
        record.get("model_id"),
        record.get("model_revision"),
        record.get("example_id"),
    )


def _model_example_key(
    model: object,
    example: AdditionExample,
) -> tuple[object, object, object, str]:
    """Return the resume key expected for one model and example pair."""
    return (
        getattr(model, "name", None),
        getattr(model, "model_id", None),
        getattr(model, "revision", None),
        example.id,
    )


def _manifest_value(run_dir: Path, key: str) -> str | None:
    """Return a string manifest value when present."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    value = read_json(manifest_path).get(key)
    return value if isinstance(value, str) else None


def _activation_tensor_path(run_dir: Path, call: ModelCallRecord) -> Path:
    """Return the local tensor path for one activation record."""
    model_slug = _filename_slug(call.model_name)
    return run_dir / "activations" / model_slug / f"{call.example_id}.pt"


def _filename_slug(value: str) -> str:
    """Return a filesystem-safe slug for artifact paths."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "unknown"


def _location_row(
    name: str,
    kind: str,
    absolute_token_index: int,
    token_id: int,
    token_text: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-serializable activation location record."""
    return {
        "name": name,
        "kind": kind,
        "absolute_token_index": absolute_token_index,
        "token_id": int(token_id),
        "token_text": token_text,
        "source": source,
        "metadata": metadata or {},
    }


def _token_offsets(
    tokenizer: Any,
    text: str,
    expected_ids: list[int],
) -> list[tuple[int, int]] | None:
    """Return token character offsets when they align with expected token IDs."""
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    if input_ids != expected_ids or len(offsets) != len(expected_ids):
        return None
    return offsets


def _user_prompt_span(rendered_prompt: str, prompt: str) -> tuple[int, int] | None:
    """Return the character span of the original user prompt inside the rendered prompt."""
    start = rendered_prompt.find(prompt)
    if start < 0:
        return None
    return start, start + len(prompt)


def _digit_spans(text: str) -> list[dict[str, Any]]:
    """Return digit character spans with least-significant-digit indices."""
    digit_positions = [
        {"digit": char, "start": index, "end": index + 1}
        for index, char in enumerate(text)
        if char.isalnum()
    ]
    total = len(digit_positions)
    for emitted_index, row in enumerate(digit_positions):
        row["lsd_index"] = total - emitted_index - 1
    return digit_positions


def _token_index_for_span(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> int | None:
    """Return the first token index whose character span overlaps a target span."""
    for index, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        if token_start < end and token_end > start:
            return index
    return None


def _final_output_digit_chars(text: str, base: int) -> dict[str, Any] | None:
    """Return digit character spans for the final answer-like output sequence."""
    from carry_trace.arithmetic import DIGIT_ALPHABET

    allowed = DIGIT_ALPHABET[:base]
    token_re = re.compile(rf"[{re.escape(allowed)},|]+(?:_{base})?", re.IGNORECASE)
    matches = list(token_re.finditer(text.upper()))
    for match in reversed(matches):
        parsed = normalize_output_digits(match.group(0), base=base)
        if parsed is None:
            continue
        digit_chars = [
            {
                "digit": char,
                "start": match.start() + index,
                "end": match.start() + index + 1,
            }
            for index, char in enumerate(match.group(0).upper())
            if char in allowed
        ]
        if digit_chars:
            return {"span": match.span(), "digit_chars": digit_chars}
    return None


def _answer_start_token(
    answer: dict[str, Any] | None,
    output_offsets: list[tuple[int, int]] | None,
) -> int | None:
    """Return the output token index containing the first final-answer digit."""
    if answer is None or output_offsets is None:
        return None
    first_digit = answer["digit_chars"][0]
    return _token_index_for_span(output_offsets, int(first_digit["start"]), int(first_digit["end"]))


def _answer_lsd_index(
    answer_format: AnswerFormat | str,
    emitted_index: int,
    total_digits: int,
) -> int:
    """Return the answer digit's canonical LSD index from its emitted index."""
    answer_format = AnswerFormat(answer_format)
    if answer_format == AnswerFormat.LSD:
        return emitted_index
    return total_digits - emitted_index - 1


def _decode_token(tokenizer: Any, token_id: int) -> str:
    """Decode a single token ID for location metadata."""
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _decode_token_from_ids(token_ids: list[int], token_index: int) -> str:
    """Return a fallback token text for prompt-token metadata."""
    return str(token_ids[token_index])


def _model_device(model: Any) -> Any:
    """Return the device that should receive teacher-forced input IDs."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _validate_hub_path(path_in_repo: str) -> str:
    """Validate and normalize a relative Hugging Face repository path."""
    path = PurePosixPath(path_in_repo)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path_in_repo must be a relative path without '..': {path_in_repo}")
    return str(path)
