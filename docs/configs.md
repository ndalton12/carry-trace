# Config Reference

`carry-trace` uses YAML config files for datasets and experiment runs. Configs
are validated before work starts. Fields with closed value sets use enums, so
invalid slices, prompt modes, digit formats, answer formats, runner kinds, and
torch dtypes fail during config loading.

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
  smoke:
    examples_per_slice_per_length: 1
    slice_examples_per_length:
      random: 8
digit_lengths: [2, 3]
slices: [no_carry, isolated_carry]
prompt_modes: [answer_only, structured_column_cot]
digit_formats: [standard, delimited]
answer_formats: [standard, lsd]
digit_delimiter: "|"
random_sampling:
  balance_carry_count: true
```

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Dataset directory name under `output_dir`. |
| `seed` | integer | Base random seed for deterministic generation. |
| `base` | integer | Arithmetic base. Goal 1 currently supports `10`. |
| `output_dir` | path | Parent directory for generated dataset artifacts. |
| `write_parquet` | boolean | Also write `examples.parquet` beside JSONL. |
| `schema_version` | string | Saved row schema version. Keep `goal1.v1` for current runs. |
| `splits` | map string to split config | Split names and optional per-split generation settings. Split RNG seeds are derived from `seed` and split name. |
| `digit_lengths` | list of integers | Operand digit lengths to generate. |
| `slices` | list of `SliceName` | Addition structure slices to include for the standard input/output condition. |
| `prompt_modes` | list of `PromptMode` | Prompt styles crossed with each generated problem. |
| `digit_formats` | list of `DigitFormat` | Operand display formats to include. Non-standard formats are generated only on `random` examples. |
| `answer_formats` | list of `AnswerFormat` | Expected answer emission formats to include. Non-standard formats are generated only on `random` examples. |
| `digit_delimiter` | string | Delimiter used when `digit_format` is `delimited`. |
| `examples_per_slice_per_length` | integer | Default replicates per split, digit length, and generated condition before prompt expansion. |
| `slice_examples_per_length` | map `SliceName` to integer | Dataset-level replicate overrides for particular slices. |
| `random_sampling` | object | Optional generation strategy settings for `random` slices. |

Split config fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `examples_per_slice_per_length` | integer or null | Optional split-specific replicate count. Falls back to the dataset-level default when omitted. |
| `slice_examples_per_length` | map `SliceName` to integer | Split-specific replicate overrides for particular slices. |

Replicate-count precedence is: split-specific slice override, then split
default, then dataset-level slice override, then dataset default.

Random sampling fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `balance_carry_count` | boolean | If true, `random` examples are generated with approximately even total carry counts per split, digit length, and format condition. This balances only the number of carry-producing columns, not the exact carry positions. |

Allowed `SliceName` values:

| Value | Meaning |
| --- | --- |
| `no_carry` | No column produces outgoing carry. |
| `isolated_carry` | Exactly one carry-producing column, and it does not propagate. |
| `long_carry_chain` | Carry propagates across all generated digits, e.g. `999 + 1`; often changes answer length. |
| `internal_carry_chain` | Carry starts in low-order digits and stops before the most-significant digit, e.g. `1099 + 1`. |
| `carry_distractor` | Alternating carry-suggestive surface pattern with some local carry activity but no long chain. |
| `many_9s_no_carry` | Many 9s appear, but no column carries; useful as a shortcut-control slice. |
| `random` | Unconstrained random operands. Used for format ablations and broad sanity checks. |

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
| `standard` | Show operands normally, e.g. `4879 + 2568`. |
| `delimited` | Insert `digit_delimiter` between displayed digits, e.g. `4|8|7|9 + 2|5|6|8`. |

Allowed `AnswerFormat` values:

| Value | Meaning |
| --- | --- |
| `standard` | Ask for the conventional most-significant-first answer, e.g. `6912`. |
| `lsd` | Ask for answer digits least-significant first with no separators, e.g. `2196` for the normal answer `6912`. |

Generation uses standard formats for the full arithmetic frontier and isolates
format ablations on fresh `random` examples:

- `digit_format=standard` and `answer_format=standard` is crossed with every
  selected slice, digit length, and prompt mode.
- `digit_format=delimited` and `answer_format=standard` is generated only for
  `random` examples, crossed with digit length and prompt mode.
- `digit_format=standard` and `answer_format=lsd` is generated only for
  `random` examples, crossed with digit length and prompt mode.
- `digit_format=delimited` and `answer_format=lsd` is intentionally excluded.

Arithmetic fields such as `a`, `b`, `answer`, digit arrays, and carry labels
remain canonical; rendered prompt operands are saved as `prompt_a` and
`prompt_b`, and the requested emitted answer is saved as `expected_output`.

Dataset rows also support optional Goal 3 matching metadata:

| Field | Meaning |
| --- | --- |
| `problem_id` | Stable arithmetic-problem ID shared by rendered prompt variants. |
| `match_group_id` | Stable ID for a matched intervention group. |
| `match_role` | Role within a matched group, such as `clean`, `corrupt`, or `control`. |
| `target_column_lsd` | Target column for a probe or intervention, with ones place as `0`. |
| `intervention_variable` | Intended causal variable, such as `incoming_carry` or `outgoing_carry`. |
| `match_family` | Matching design family, such as `carry_interchange`. |
| `match_constraints` | JSON object describing held-fixed and varied constraints. |
| `partner_problem_ids` | Other arithmetic problem IDs in the matched group. |

Ordinary Goal 1 rows include `problem_id` but omit empty matching fields from
JSONL and Parquet artifacts.

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
splits: [smoke]
prompt_modes: [answer_only, structured_column_cot]
digit_formats: [standard, delimited]
answer_formats: [standard, lsd]
runner:
  kind: fake
  device: auto
  dtype: auto
  batch_size: 1
  trust_remote_code: false
  quantization: none
  tensor_parallel_size: 1
  gpu_memory_utilization: null
  max_model_len: null
  enforce_eager: false
models:
  - name: olmo3-think-fake
    model_id: allenai/Olmo-3-7B-Think
generation:
  max_new_tokens: 128
  temperature: 0.0
  top_p: 1.0
  do_sample: false
  thinking_final_answer_tokens: null
  force_close_thinking: false
```

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Run-name prefix. |
| `seed` | integer | Generation seed passed to the runner. |
| `dataset_path` | path | JSONL dataset produced by a dataset config. |
| `output_dir` | path | Parent directory for run artifacts. |
| `max_examples` | integer or null | Optional cap after filtering. When omitted, all filtered examples are used. |
| `splits` | list of strings or null | Optional filter over dataset split names. |
| `prompt_modes` | list of `PromptMode` or null | Optional filter over dataset prompt modes. |
| `digit_formats` | list of `DigitFormat` or null | Optional filter over dataset digit formats. |
| `answer_formats` | list of `AnswerFormat` or null | Optional filter over dataset answer formats. |
| `runner` | object | Model execution backend settings. |
| `models` | list of model specs | Checkpoints to run. |
| `generation` | object | Hugging Face generation parameters saved with every call. |

Allowed runner values:

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `runner.kind` | `fake`, `hf`, `vllm` | Deterministic test runner, Hugging Face Transformers runner, or vLLM offline inference runner. |
| `runner.dtype` | `auto`, `float16`, `bfloat16`, `float32` | Torch dtype for model loading. |
| `runner.device` | string | Device string such as `auto`, `cpu`, `mps`, `cuda`, or `cuda:0`. |
| `runner.quantization` | `none`, `bitsandbytes_8bit` | Optional model-weight quantization. With `hf`, `bitsandbytes_8bit` uses Hugging Face `BitsAndBytesConfig(load_in_8bit=True)`, i.e. LLM.int8-style quantization, not FP8. With `vllm`, it passes `quantization="bitsandbytes"` to vLLM. |
| `runner.tensor_parallel_size` | positive integer | vLLM tensor-parallel GPU count. Ignored by `fake` and `hf`. |
| `runner.gpu_memory_utilization` | float or null | Optional vLLM GPU memory utilization fraction, e.g. `0.9`. Ignored by `fake` and `hf`. |
| `runner.max_model_len` | integer or null | Optional vLLM maximum model context length. Ignored by `fake` and `hf`. |
| `runner.enforce_eager` | boolean | Optional vLLM eager-mode switch. Use `true` to avoid Torch Dynamo / compile-backend failures at the cost of some throughput. Ignored by `fake` and `hf`. |

The `vllm` runner is an optional backend dependency. Install it in the GPU
runtime, for example with `pip install vllm` or `uv pip install vllm
--torch-backend=auto`.

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
| `thinking_final_answer_tokens` | integer or null | Optional budget reserved for final-answer generation after an enforced thinking cap. |
| `force_close_thinking` | boolean | When `true`, generate first for `max_new_tokens - thinking_final_answer_tokens`; if that output hit the cap and contains `<think>` without `</think>`, append `</think>\nFinal answer:` to the model context and continue for the reserved final-answer budget. Non-thinking outputs continue normally up to the full `max_new_tokens` budget. |

## Artifacts

Generated datasets write:

- `examples.jsonl`
- `examples.parquet` when `write_parquet: true`
- `manifest.json`

Goal 1 runs write:

- `dataset.jsonl`
- `calls.jsonl`, appended incrementally after each completed model call
- `scored_calls.jsonl`
- `metrics_summary.csv`
- `manifest.json`
- run-local `figures/` after `carry-trace figures goal1`

If a Goal 1 run exits before scoring completes, its manifest remains
`status: running`. Re-running the same config automatically resumes the newest
incomplete run with the same config hash, skips completed `(model, example)`
calls already present in `calls.jsonl`, appends missing calls, then scores the
completed artifact set.

Generation records include `metadata.hit_token_limit` when a model exhausts
its generation budget. Scored records expose this as top-level
`hit_token_limit` and `generation_valid`; summaries also include token-limit
hit counts/rates and valid-only accuracy/token averages for analysis that wants
to exclude capped outputs.
