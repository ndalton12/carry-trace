# Config Reference

`carry-trace` uses YAML config files for datasets and experiment runs. Configs
are validated before work starts. Fields with closed value sets use enums, so
invalid slices, prompt modes, digit formats, runner kinds, and torch dtypes fail
during config loading.

## Dataset Configs

Dataset configs are passed to:

```bash
uv run carry-trace dataset generate --config configs/datasets/goal1_smoke.yaml
```

Example:

```yaml
name: goal1_smoke
seed: 13
base: 10
output_dir: data/generated
write_parquet: true
schema_version: goal1.v1
splits:
  smoke: 0
digit_lengths: [2, 3]
slices: [no_carry, isolated_carry]
prompt_modes: [answer_only, structured_column_cot]
digit_formats: [plain, delimited]
digit_delimiter: "|"
examples_per_slice_per_length: 1
```

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Dataset directory name under `output_dir`. |
| `seed` | integer | Base random seed for deterministic generation. |
| `base` | integer | Arithmetic base. Goal 1 currently supports `10`. |
| `output_dir` | path | Parent directory for generated dataset artifacts. |
| `write_parquet` | boolean | Also write `examples.parquet` beside JSONL. |
| `schema_version` | string | Saved row schema version. Keep `goal1.v1` for current runs. |
| `splits` | map string to integer | Split names and seed offsets. |
| `digit_lengths` | list of integers | Operand digit lengths to generate. |
| `slices` | list of `SliceName` | Addition structure slices to include. |
| `prompt_modes` | list of `PromptMode` | Prompt styles crossed with each generated problem. |
| `digit_formats` | list of `DigitFormat` | Operand display formats crossed with prompt modes. |
| `digit_delimiter` | string | Delimiter used when `digit_format` is `delimited`. |
| `examples_per_slice_per_length` | integer | Replicates per split, digit length, and slice before prompt and digit-format expansion. |

Allowed `SliceName` values:

| Value | Meaning |
| --- | --- |
| `no_carry` | Columns are generated so no carry occurs. |
| `isolated_carry` | Exactly one non-propagating carry. |
| `long_carry_chain` | Carry propagates across all generated digits, e.g. `999 + 1`. |
| `internal_carry_chain` | Carry starts inside the number rather than only at the final answer boundary. |
| `carry_distractor` | Surface digits suggest carry-like difficulty but generated labels may have no carry. |
| `many_9s_no_carry` | Many 9s without carry, useful for shortcut controls. |
| `random` | Unconstrained random operands. Mostly useful for debugging. |

Allowed `PromptMode` values:

| Value | Meaning |
| --- | --- |
| `answer_only` | Ask for only the answer. |
| `free_cot` | Ask the model to think step by step. |
| `length_controlled_cot` | Ask for exactly four short steps. |
| `structured_column_cot` | Ask for right-to-left column reasoning with digit and carry. |

Allowed `DigitFormat` values:

| Value | Meaning |
| --- | --- |
| `plain` | Show operands normally, e.g. `4879 + 2568`. |
| `delimited` | Insert `digit_delimiter` between displayed digits, e.g. `4|8|7|9 + 2|5|6|8`. |

`digit_format` is independent of `prompt_mode`: every selected prompt mode is
crossed with every selected digit format. Arithmetic fields such as `a`, `b`,
`answer`, digit arrays, and carry labels remain plain; rendered prompt operands
are saved separately as `prompt_a` and `prompt_b`.

## Experiment Configs

Experiment configs are passed to:

```bash
uv run carry-trace run goal1 --config configs/experiments/goal1_smoke.yaml
```

Example:

```yaml
name: goal1_smoke_fake
seed: 13
dataset_path: data/generated/goal1_smoke/examples.jsonl
output_dir: runs
max_examples: 8
prompt_modes: [answer_only, structured_column_cot]
digit_formats: [plain, delimited]
runner:
  kind: fake
  device: auto
  dtype: auto
  batch_size: 1
  trust_remote_code: false
models:
  - name: olmo3-think-fake
    model_id: allenai/Olmo-3-7B-Think
generation:
  max_new_tokens: 128
  temperature: 0.0
  top_p: 1.0
  do_sample: false
```

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Run-name prefix. |
| `seed` | integer | Generation seed passed to the runner. |
| `dataset_path` | path | JSONL dataset produced by a dataset config. |
| `output_dir` | path | Parent directory for run artifacts. |
| `max_examples` | integer or null | Optional cap after filtering. |
| `prompt_modes` | list of `PromptMode` or null | Optional filter over dataset prompt modes. |
| `digit_formats` | list of `DigitFormat` or null | Optional filter over dataset digit formats. |
| `runner` | object | Model execution backend settings. |
| `models` | list of model specs | Checkpoints to run. |
| `generation` | object | Hugging Face generation parameters saved with every call. |

Allowed runner values:

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `runner.kind` | `fake`, `hf` | Deterministic test runner or Hugging Face Transformers runner. |
| `runner.dtype` | `auto`, `float16`, `bfloat16`, `float32` | Torch dtype for model loading. |
| `runner.device` | string | Device string such as `auto`, `cpu`, `mps`, `cuda`, or `cuda:0`. |

Model specs:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Local label used in metrics and figures. |
| `model_id` | string | Hugging Face model ID. |
| `revision` | string or null | Optional model revision. |
| `tokenizer_id` | string or null | Optional separate tokenizer ID. Defaults to `model_id`. |

Generation parameters:

| Field | Type | Meaning |
| --- | --- | --- |
| `max_new_tokens` | integer | Maximum generated tokens. |
| `temperature` | float | Sampling temperature. |
| `top_p` | float | Nucleus sampling threshold. |
| `do_sample` | boolean | Whether to sample. Use `false` for deterministic greedy decoding. |

## Artifacts

Generated datasets write:

- `examples.jsonl`
- `examples.parquet` when `write_parquet: true`
- `manifest.json`

Goal 1 runs write:

- `dataset.jsonl`
- `calls.jsonl`
- `scored_calls.jsonl`
- `metrics_summary.csv`
- `manifest.json`
- run-local `figures/` after `carry-trace figures goal1`
