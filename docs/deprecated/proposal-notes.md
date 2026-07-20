# Proposal Notes

## Goal 1 Focus

Goal 1 establishes the behavioral frontier for synthetic addition across
direct-answer and elicited-reasoning prompt regimes. It asks whether elicited
reasoning improves uniformly or shows jagged gains concentrated around
particular carry structures.

## Current Framing

We study how elicited explicit reasoning changes arithmetic behavior and
carry-state representations in instruction-following models.

## Implemented Prompt Modes

- `answer_only`
- `free_cot`
- `length_controlled_cot`
- `structured_column_cot`

## Implemented Digit Formats

- `standard`: operands are shown normally.
- `delimited`: operands are shown with `|` between digits, while labels and expected answers stay plain.

## Format Ablation Policy

Standard input/output examples use the full configured addition-slice frontier.
Non-standard input or output formats use fresh `random` examples only, and the
combined `delimited` input plus `lsd` output condition is excluded.

## Config Policy

User-facing config fields with closed value sets are represented as enums in code
and documented in `docs/configs.md`. Invalid slices, prompt modes, digit formats,
runner kinds, and torch dtypes should fail during config validation.

## Implemented Addition Slices

- `no_carry`
- `isolated_carry`
- `long_carry_chain`
- `internal_carry_chain`
- `carry_distractor`
- `many_9s_no_carry`

## Future-Compatible Constraints

- Dataset rows include stable example IDs, split names, prompt metadata, carry
  labels, and digit arrays for later probing and interventions.
- Model-call records save exact prompts, chat messages, token IDs, decoded
  outputs, parsed answers, generation settings, timing, and git commit hash.
- Because natural Think termination is budget-dependent and non-uniform across
  conditions, we do not use forced-closed generations for primary mechanistic
  conclusions; instead, we report termination statistics and analyze the
  naturally closed subset where coverage permits.
- Base-k is not part of Goal 1 execution, but parsing and digit utilities support
  non-decimal alphabets for future extension.

## Tokenization Notes

- OLMo 3 tokenization should avoid unsplit multi-digit operands when experiments
  need one digit per token: bare 1-, 2-, and 3-digit ASCII strings are vocabulary
  tokens, while bare 4-digit strings were not observed as single tokens.
- Vocabulary-level inspection found that space-separated digits, pipe-separated
  digits, and ` | `-separated digits have no token surfaces that span multiple
  digits, so these separator styles are valid for arbitrary-length one-token-per-
  digit addition prompts.
- Include an `lsd` answer-format ablation to test whether standard
  left-to-right answer emission is a bottleneck for right-to-left carry
  computation. The canonical answer remains saved normally, while
  `expected_output` records the requested emitted form.
