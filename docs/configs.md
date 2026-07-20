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

## Goal 3 Dataset Bundles

Goal 3 dataset bundles are derived from a completed Goal 2 activation run:

```bash
uv run carry-trace dataset goal3 \
  --config configs/datasets/goal3_olmo3_natural_cot.yaml
```

The derivation reuses the original natural free-CoT output and the
`output_token_index` token anchors saved for Goal 2 activation locations. New
Goal 2 runs provide the exact generated token IDs. For legacy runs that omit
them, the command retokenizes the decoded output and aligns each boundary to
the nearest matching recorded token ID; every prefix records the alignment
method and index offset. Terminal control tokens are not included in assistant
prefills. It does not regenerate, normalize, or impose a structured CoT. The
source dataset is joined
by `example_id` to recover prompts, arithmetic labels, `problem_id`, and the
native answer-only example for the same problem.

Top-level Goal 3 dataset fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Output dataset-bundle directory under `output_dir`. |
| `source_goal2_run_dir` | path | Completed Goal 2 activation run containing `manifest.json` and `activations.jsonl`. |
| `source_dataset_path` | path or null | Original Goal 2 dataset. Defaults to `dataset_path` in the source manifest. |
| `output_dir` | path | Parent directory for the derived bundle. |
| `schema_version` | string | Schema version written to every Goal 3 row. |
| `splits` | list of strings | Source example splits to include. |
| `models` | list of strings or null | Source model names. Null infers all filtered models in the activation run. |
| `digit_lengths` | list of integers | Source operand lengths to include. The proposed primary analysis uses 2 and 4. |
| `prompt_modes` | list of `PromptMode` | Source prompt modes. The natural-CoT design uses `free_cot`. |
| `digit_formats` | list of `DigitFormat` | Source operand formats. The primary design uses `standard`. |
| `answer_formats` | list of `AnswerFormat` | Source answer formats. The primary design uses `standard`. |
| `require_shared_models` | boolean | Keep only examples with an eligible activation record for every requested model. |
| `include_token_limit_hits` | boolean | Include capped generations. Defaults to false. |
| `replay` | object | Goal 3A/3B natural-CoT prefix and receiver-crossing settings. |
| `residual` | object | Goal 3C residual-intervention specification settings. |

Replay fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `enabled` | boolean | Materialize `replay_prefixes.jsonl` and `replay_cases.jsonl`. |
| `locations` | list of `ActivationLocation` | Goal 2 token boundaries to materialize. `prompt_final` produces one empty, no-reasoning prefix per example. |
| `crossed_models` | boolean | Cross every nonempty source-model prefix with every requested receiver model. When false, emit self-replay cases only. |
| `tokenizer_id` | string or null | Optional tokenizer override. Otherwise use the tokenizer ID saved on each activation record. |
| `tokenizer_revision` | string or null | Optional tokenizer revision override. |
| `tokenizer_local_files_only` | boolean | Require the tokenizer to be present in the local Hugging Face cache. |
| `max_token_alignment_shift` | integer | Maximum index shift allowed when aligning a legacy decoded output to its recorded Goal 2 token anchor. |

Residual-intervention fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `enabled` | boolean | Materialize `residual_intervention_cases.jsonl`. |
| `targets` | list of `ProbeTarget` | Probe-guided intervention variables. Supported values are `incoming_carry` and `outgoing_carry`. |
| `locations` | list of `ActivationLocation` | Natural Goal 2 positions at which later runners should intervene. |
| `layers` | list of integers | Saved residual layers to reference. The proposed 8/16/24 set is a robustness grid, not best-layer selection. |
| `direction_train_split` | string | Source split later used to estimate the column-specific residual direction. |
| `target_columns_by_target` | map target to digit-length maps | Eligible least-significant-first carry columns for each target and operand length. Incoming carry at column `c` affects output `c`; outgoing carry from `c` affects output `c+1`. |
| `require_stable_outgoing_carry` | boolean | Require the factual and flipped carry to produce the same carry out of the affected output column, keeping the counterfactual local to one digit. |

`outgoing_carry[c]` and `incoming_carry[c+1]` are the same arithmetic label.
With the global token locations in the proposed config, outgoing carry is an
equivalence/control target; it is not an independent source of evidence.

The bundle contains:

- `replay_prefixes.jsonl`: unique natural assistant prefixes with replay token
  IDs, original output, source-model metadata, recorded and replay boundary
  indices, and the Goal 2 location;
- `replay_cases.jsonl`: no-reasoning, self-replay, and crossed-replay receiver
  assignments for Goals 3A and 3B;
- `residual_intervention_cases.jsonl`: factual/counterfactual carry labels,
  carry and affected-output columns, target digits and answers, activation
  tensor coordinates, direction IDs, and expected unchanged output columns for
  Goal 3C;
- `manifest.json`: source paths, filters, model names, config hash, and counts.

The analysis and interpretation plan for these artifacts is in
`docs/goal3-plan.md`.

## Goal 3 Execution Configs

Run the derived bundle with:

```bash
uv run carry-trace run goal3 \
  --config configs/experiments/goal3_olmo3_natural_cot.yaml
```

Top-level fields are `name`, `seed`, `dataset_bundle_dir`, `output_dir`,
`models`, `runner`, `replay`, and `residual`. Goal 3 currently requires the
Hugging Face runner because residual interventions use decoder-layer hooks.

Replay execution fields:

| Field | Meaning |
| --- | --- |
| `enabled` | Run Goals 3A/3B replay scoring. |
| `answer_cue` | Fixed assistant text inserted before answer scoring or generation. |
| `decode_locations` | Prefix locations that receive deterministic autoregressive decoding. All locations are still teacher-force scored. |
| `generation` | Short deterministic decoding parameters. |

Residual execution fields:

| Field | Meaning |
| --- | --- |
| `enabled` | Run Goal 3C direction fitting and interventions. |
| `intervention_mode` | `fixed_gap` adds the train-set class-mean gap; `projection_clamp` moves the live carry projection toward the counterfactual class mean. |
| `intervention_scales` | Positive intervention multipliers. Fixed-gap mode scales the class-mean gap; clamp mode interpolates toward the counterfactual mean, with `1.0` giving an exact clamp. |
| `intervention_sites` | Apply at the saved `prefix_boundary`, the final fixed `answer_cue` token, or both. Answer-cue clamps retain the prefix-boundary calibration magnitude. |
| `control_directions` | Intervention vectors to execute. `carry` is primary; `orthogonal` is a deterministic norm-matched control. |
| `orthogonal_control_count` | Number of deterministic orthogonal replicates. Their effects are averaged in the paired contrast artifact. |
| `max_problems_per_digit_length` | Optional deterministic per-length problem cap used by the pilot. Null keeps the full population. |
| `require_shared_correct_source` | Keep only selected problems with correct, cleanly terminated source completions from every intervention model. |
| `score_locations`, `score_layers` | Optional residual-case scoring subset. Null uses every location or layer in the dataset bundle. |
| `min_train_examples`, `max_iter`, `c` | Train-split logistic-direction fitting settings. |
| `require_exact_prefix_alignment` | Exclude legacy cases whose replay token is not the exact saved activation token. |
| `decode_locations`, `decode_layers` | Focused subset receiving baseline and intervened autoregressive decoding. All eligible cases receive teacher-forced sequence scoring. |
| `decode_intervention_scale` | Carry-direction scale used for focused decoding. Controls are not decoded. |
| `generation` | Short deterministic intervention-decoding parameters. |

The run writes raw replay and residual scores, reduced generation records,
paired replay effects, fitted direction tensors and metadata, completion audits,
grouped summary metrics, and a resumable manifest. Whole-answer factual and
counterfactual sequence probabilities are used instead of per-digit-token
probabilities, so merged number tokens do not invalidate Goal 3C.

The calibration pilot uses the same command with
`configs/experiments/goal3_olmo3_natural_cot_pilot.yaml`. It deterministically
selects five problems per digit length, disables replay and decoding, crosses
three intervention scales with carry and orthogonal directions, and writes
`residual_control_contrasts.jsonl` with paired carry-minus-control effects.

The clamp diagnostic uses
`configs/experiments/goal3_olmo3_natural_cot_clamp_pilot.yaml`. It filters the
same selected problems to shared correct, clean completions; clamps projections
at layers 16 and 24; compares prefix-boundary and answer-cue application; and
averages three norm-matched orthogonal controls. Raw residual rows include the
requested and realized BF16 projection changes and shift norms.

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

Upload one generated dataset directory to a shared Hugging Face dataset repo:

```bash
uv run carry-trace dataset upload \
  --dataset-dir data/generated/goal1_paper_like \
  --repo-id <user-or-org>/carry-trace-datasets
```

The upload is nested by default. For the command above, the Hugging Face dataset
repo receives:

- `goal1_paper_like/examples.jsonl`
- `goal1_paper_like/examples.parquet`
- `goal1_paper_like/manifest.json`

Set `HF_TOKEN` or pass `--token`; use `--path-in-repo` only when you want a
different subdirectory name.

Goal 1 runs write:

- `dataset.jsonl`
- `calls.jsonl`, appended incrementally after each completed model call
- `scored_calls.jsonl`
- `metrics_summary.csv`
- `manifest.json`
- run-local `figures/` after `carry-trace figures goal1`

Goal 1 figures exclude token-limit hits by default and therefore align with the
valid-generation metrics in `metrics_summary.csv`, such as
`parsed_accuracy_valid`. To include capped generations in diagnostic figures,
pass `--include-token-limit-hits`.
The combined `accuracy_heatmap.png` averages over all models in the run; the
figure command also writes `accuracy_heatmap_<model-name>.png` files split out
by model.
The comparison figures include `prompt_mode_comparison.png`,
`digit_format_comparison.png`, `answer_format_comparison.png`, and
`slice_type_comparison.png`.
`token_budget_curves.png` replaces the raw token-count scatter with the fraction
of examples both correct and completed within each output-token budget. By
default, this figure uses one panel per digit length; pass
`--no-token-budget-by-digit-length` to pool digit lengths in this figure.
Prompt-mode and digit-format comparison bars show 95% confidence intervals.
All figures use a shared paper-style theme, colorblind palette, white grid,
300 DPI export, and display-formatted labels instead of raw enum names.

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

## Goal 2 Activation Extraction

Goal 2 runs use:

```bash
uv run carry-trace run goal2 --config configs/experiments/goal2_olmo3_7b_probe_smoke.yaml
```

The runner currently requires `runner.kind: hf` because hidden-state extraction
uses Hugging Face `output_hidden_states=True`. Each example is generated first,
then a teacher-forced forward pass over `prompt + generated_output` saves only
the configured token locations across all transformer layers.

Activation location options:

- `operand_digits`
- `question_token`
- `prompt_final`
- `cot_start`
- `cot_1_3`
- `cot_2_3`
- `cot_end`
- `answer_digits`

Activation storage settings:

| Field | Type | Meaning |
| --- | --- | --- |
| `storage_dtype` | `auto`, `float16`, `bfloat16`, or `float32` | Dtype used when saving activation tensors. `auto` keeps the model hidden-state dtype; `float16` is the default and is usually the best storage/runtime tradeoff. |
| `include_embedding_layer` | boolean | When `false`, save only transformer-layer outputs. When `true`, also save the embedding hidden state as layer `embedding`. |

Goal 2 runs write:

- `dataset.jsonl`
- `activations.jsonl`, appended incrementally after each saved tensor
- `activations/<model-name>/<example-id>.pt`
- `manifest.json`

Each tensor file stores `activations` with shape
`[locations, layers, hidden_size]`; by default `layers` excludes the embedding
state and includes every transformer layer. Re-running the same config resumes
the newest incomplete run with the same config hash and skips activation rows
whose tensor files already exist.

Set `upload.enabled: true` and `upload.repo_id: <user-or-org>/<dataset-repo>`
to upload the completed run directory to a Hugging Face dataset repo after local
extraction finishes. Upload is best-effort: a Hub failure is recorded in the
local manifest without invalidating the completed local run.

## Goal 2 Linear Probes

Goal 2 probes use saved activation runs:

```bash
uv run carry-trace probe goal2 --config configs/probes/goal2_linear_smoke.yaml
```

The probe runner trains logistic-regression linear probes from saved hidden
states, using dataset split names to separate train and test rows. By default
it uses `train_probe` for training and `test_probe` for evaluation.
Aggregate targets train one probe per `model_name x target x location_kind x
layer_index`. Column-specific targets train one probe per `model_name x target
x target_column_lsd x location_kind x layer_index`, so every saved location can
be tested for whether column `i` is decodable.

Probe config fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Probe run-name prefix. |
| `goal2_run_dir` | path | Completed Goal 2 activation run directory. |
| `output_dir` | path | Parent directory for probe artifacts. Defaults to `runs/probes`. |
| `train_split` | string | Dataset split used to train probes. Defaults to `train_probe`. |
| `test_split` | string | Dataset split used to evaluate probes. Defaults to `test_probe`. |
| `prompt_modes` | list of `PromptMode` or null | Optional prompt-mode filter applied before fitting. Use separate answer-only and free-CoT runs when prompt-specific probes are required. |
| `targets` | list of `ProbeTarget` | Probe targets. Defaults to all supported targets. |
| `min_train_examples` | integer | Minimum train samples required to fit a probe group. |
| `min_test_examples` | integer | Minimum test samples required to evaluate a probe group. |
| `max_iter` | integer | Maximum iterations for sklearn logistic regression. |
| `c` | float | Inverse regularization strength for sklearn logistic regression. |
| `random_state` | integer | Logistic-regression random state. |
| `require_unambiguous_digit_tokens` | boolean | When true, column-specific targets skip digit locations where multiple digit locations resolved to the same token. Defaults to `false`. |
| `n_jobs` | integer | Number of thread-pool workers used to fit independent probe groups. Defaults to `1`. |

Allowed `ProbeTarget` values:

| Value | Meaning |
| --- | --- |
| `any_carry` | Binary label for whether the problem has any carry-producing column. Usable at every saved location. |
| `incoming_carry` | Binary per-column label `incoming_carry[i]`. |
| `outgoing_carry` | Binary per-column label `outgoing_carry[i]`. |
| `output_digit` | Multiclass per-column label `output_digits_lsd[i]`, including the final carry digit when present. |
| `raw_sum` | Multiclass per-column label `raw_sum[i]`, the operand digit sum before incoming carry. |
| `carry_chain_membership` | Binary per-column label for whether column `i` has either incoming or outgoing carry. |
| `column_pointer` | Binary per-column one-hot label for `i == first_carry_position`; all zeros when no carry occurs. |

Probe training excludes activation records whose generation metadata has
`hit_token_limit: true`, so capped outputs do not enter train or test metrics.
CoT-derived locations (`cot_start`, `cot_1_3`, `cot_2_3`, and `cot_end`) are
used only for examples whose `prompt_mode` is `free_cot`; they are ignored for
answer-only examples even if the model generated extra text.
For column-specific targets, the optional ambiguity guard drops digit-indexed
samples where multiple digit locations share one tokenizer token. This avoids
training the same activation against potentially conflicting column labels when
the tokenizer merged multiple digits into one token. The default is `false`
because merged digit tokens are part of the tokenization effect we usually want
to measure. Probe artifacts record `problem_id`, `target_column_lsd`,
`location_lsd_index`, and `same_column` so analyses can use problem-level
pairing and separate same-column, cross-column, and non-column locations.

Goal 2 probe runs write:

- `probe_metrics.jsonl`
- `probe_predictions.jsonl`
- `probe_slice_metrics.jsonl`: exact test metrics pooled and stratified by
  prompt mode, digit length, format, and answer-position scope.
- `probe_shared_slice_metrics.jsonl`: the same metrics restricted to examples
  available for every model.
- `probe_figure_metrics.jsonl`: best-layer summaries, layer profiles, and
  paired model deltas fully aggregated for plotting.
- `probe_bootstrap_metrics.jsonl`: fixed-layer paired balanced-accuracy
  comparisons with problem-clustered confidence intervals.
- `manifest.json`
- run-local `figures/` after `carry-trace figures goal2`

All statistical aggregation happens during probe generation. For these new
artifacts, the figure command reads `probe_figure_metrics.jsonl` and
`probe_bootstrap_metrics.jsonl` and only performs display reshaping and
rendering; it does not reopen or aggregate the large prediction file. Legacy
probe artifacts without the analysis tables retain a compatibility path.

Goal 2 probe figures are generated with:

```bash
uv run carry-trace figures goal2 --probe-id <probe-run-id>
```

The figure command organizes Goal 2 plots into subdirectories under
`figures/`.

`figures/summary/` contains compact paper-facing summaries:

- `goal2_target_summary_matrix_<model>.png`: target-by-location matrix using
  raw accuracy, averaged over columns after selecting the best layer.
- `goal2_target_summary_delta_<model_a>_minus_<model_b>.png`: two-model
  target-by-location difference in raw accuracy. When the
  models are identifiable as Full and SFT, this uses Full minus SFT.
- `goal2_carry_column_facets.png`: outgoing-carry decoding over layers, faceted
  by target column.
- `goal2_carry_column_facets_delta_<model_a>_minus_<model_b>.png`: two-model
  outgoing-carry decoding difference over layers, faceted by target column.
- `goal2_reasoning_time_by_column.png`: carry-chain decoding across prompt/CoT
  locations, grouped by low/mid/high columns.
- `goal2_reasoning_time_by_column_delta_<model_a>_minus_<model_b>.png`:
  two-model carry-chain decoding difference across prompt/CoT locations.
- `goal2_layer_profile_by_target_free_cot.png`: target-faceted free-CoT layer
  profiles at `prompt_final`, `cot_2_3`, and `answer_digits`, with model
  encoded by color and location encoded by marker shape.

`figures/diagnostics/` contains diagnostic per-target layer-location heatmaps,
two-model delta heatmaps, layer trajectories, and timing curves:

- `linear_probe_delta_heatmap_<model_a>_minus_<model_b>_<target>.png`:
  per-target layer-by-location difference in raw accuracy.

`figures/inference/` contains fixed-layer, paired model comparisons for
outgoing-carry probes:

- `paired_bootstrap_balanced_accuracy_delta_outgoing_carry_<prompt_mode>_<n>digits.png`:
  layer profiles of the paired balanced-accuracy difference, faceted by carry
  column. Answer-digit results include only rows whose answer position matches
  the probed carry column.

Exact Full-minus-SFT values and percentile 95% intervals from 1,000 paired
`problem_id` cluster-bootstrap resamples are stored in
`probe_bootstrap_metrics.jsonl`. Rows remain separate by prompt mode, digit
length, target column, token location, and layer.

The bootstrap compares only problem IDs available for both models. If one
model has capped generations, the resulting free-CoT estimand is therefore
conditional on the shared uncapped subset; use an uncensored run for a full
population comparison.

Probe heatmaps intentionally omit numeric cell annotations because the expanded
per-column target set makes annotated cells too busy for paper figures.
