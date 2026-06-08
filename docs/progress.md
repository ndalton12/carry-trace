# Progress

## Current Status

- Implemented the Goal 1 scaffold for reproducible behavioral addition experiments.
- Default checkpoint comparison is OLMo 3 Think vs OLMo 3 Instruct.
- Default executable run is a fake-runner smoke test so the repo can validate without model downloads.
- Verified dataset generation, fake-runner Goal 1 execution, metrics, and figure generation locally.
- Added `digit_format` as an independent Goal 1 condition with `plain` and `delimited` variants.
- Replaced free-form config lists with enums for slices, prompt modes, digit formats, runner kind, and torch dtype.
- Added tokenizer inspection that scans decoded vocabulary entries for digit tokens and split-digit guarantees.
- Added `answer_format` as an independent Goal 1 condition with `standard` and `lsd` variants.
- Changed dataset splits to use named split configs with internally derived deterministic split seeds and optional per-split replicate counts.
- Added Goal 3-ready dataset metadata: populated `problem_id` plus optional matching fields that are omitted from ordinary Goal 1 artifacts when empty.
- Added experiment-level split filtering; omitted filters, including `max_examples`, use all rows remaining after earlier filters.
- Added optional Hugging Face thinking-cap generation for thinking checkpoints, with reserved final-answer tokens after a forced `</think>` close.

## Decisions

- Use uv and a `src/` Python package.
- Use Typer for command-line hooks.
- Use Pydantic schemas for dataset rows and model-call artifacts.
- Store generated datasets under `data/generated/`, run artifacts under `runs/`, and figures under run-local `figures/`.
- Treat OLMo 3 tokenizers as GPT-2-style byte-level BPE via Hugging Face `AutoTokenizer`, based on `tokenizer_class: GPT2Tokenizer`.
- Keep delimited operands prompt-only: rows store plain `a`, `b`, and `answer`, plus rendered `prompt_a` and `prompt_b`.
- Document config fields and enum values in `docs/configs.md`.
- For the delimiter prompt condition, use compact pipe-separated digits such as `4|8|7|9`: vocabulary inspection found no OLMo token surfaces that span multiple digits under this grammar, it is more compact than ` | ` spacing, and it provides clearer operand grouping than plain spaces.
- Do not rely on unsplit operands for digit-level token alignment: OLMo has bare single tokens for all 1-, 2-, and 3-digit ASCII strings, but not for 4-digit strings in the inspected vocabulary.
- A one-off OLMo vocab scan found pipe-only tokens such as `|`, ` |`, `||`, and `||||`, but no digit-containing pipe tokens such as `|7`, `7|`, or `4|8`, supporting compact `|` as a digit delimiter.
- Use `lsd` as the output-order ablation: canonical answers remain normal, while `expected_output` asks for least-significant-first digits with no separators.
- Use named split configs instead of manual seed offsets. Split RNG seeds are internal and reproducible, and split-specific replicate counts control dataset size.
- Use `problem_id` for arithmetic-problem joins and reserve optional matching metadata for future clean/corrupt/control intervention groups.
- Use experiment `splits` to select intended data partitions at run time instead of relying on separate dataset files.
- For OLMo Think runtime control, optionally reserve final-answer tokens and force-close an unclosed `<think>` block only when the first generation phase hits the thinking cap.

## Next Work

- Run real OLMo 3 smoke generations on suitable hardware.
- Expand Goal 1 sweeps after validating parsing and runtime cost.
- Start Goal 2 by adding activation cache site definitions that refer back to saved dataset IDs.
