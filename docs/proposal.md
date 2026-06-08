# Carry Trace Research Proposal

Working title: **From Narration to Computation: Causal Development of Latent Algorithmic State in Reasoning-Tuned Language Models**

## Summary

This project studies how reasoning post-training changes arithmetic computation
inside language models. Addition is used as a controlled substrate because the
ground-truth latent variables are known: incoming carry, outgoing carry, output
digit, raw column sum, carry-chain length, and column position.

The central question is not whether chain-of-thought is broadly "faithful." The
more precise question is:

> Which parts of the visible reasoning trace are computational state, which are
> control signals, and which are narration, and how does post-training change
> that division of labor?

The current implementation focuses on Goal 1. The repo is intentionally being
structured so later probing, activation patching, CoT perturbation, and base-k
extensions can reuse the same dataset IDs, prompt metadata, saved model calls,
and carry labels.

## Core Research Questions

1. **Latent algorithmic state**: Does reasoning post-training increase the
   presence and causal use of column-indexed carry state?
2. **Temporal ordering**: Is carry state computed before it is verbalized,
   only after it is verbalized, or only near final answer emission?
3. **Chain-of-thought function**: Are reasoning tokens scratchpad state,
   control policy, attention anchors, semantic reports, or rationalization?
4. **Training-stage development**: How do these mechanisms change from base to
   instruction-tuned to thinking/reasoning checkpoints?
5. **Algorithmic generality**: Do the mechanisms generalize beyond base-10
   addition to base-k arithmetic?

## Core Experimental Object

Use synthetic addition problems with exact labels for all relevant latent
variables.

Example:

```text
 4879
+2568
= ?
```

Canonical labels are least-significant-digit first:

```python
digits_a_lsd = [9, 7, 8, 4]
digits_b_lsd = [8, 6, 5, 2]
raw_sum = [17, 13, 13, 6]
incoming_carry = [0, 1, 1, 1]
outgoing_carry = [1, 1, 1, 0]
output_digits_lsd = [7, 4, 4, 7]
answer = "7447"
```

Important variables:

- `raw_sum_i`: local column sum before incoming carry.
- `incoming_carry_i`: carry entering column `i`.
- `outgoing_carry_i`: carry leaving column `i`.
- `output_digit_i`: answer digit at column `i`.
- `carry_count`: number of carry-producing columns.
- `max_carry_chain`: longest consecutive carry propagation.
- `carry_positions`: columns where outgoing carry is nonzero.
- `column_pointer`: which column the model appears to be processing.
- `answer_prefix_correctness`: whether emitted answer prefix is currently
  correct.

Column binding matters: a model can know that a carry exists but fail to bind it
to the right answer digit.

## Current Dataset Axes

Implemented or planned config axes should stay independent unless a later goal
explicitly justifies coupling them.

- `prompt_mode`
  - `answer_only`
  - `free_cot`
  - `length_controlled_cot`
  - `structured_column_cot`
- `digit_format`
  - `plain`: operands shown normally, e.g. `4879 + 2568`
  - `delimited`: operands shown with `|`, e.g. `4|8|7|9 + 2|5|6|8`
- `answer_format`
  - `standard`: conventional most-significant-first answer
  - `lsd`: least-significant-first digits with no separators
- addition slice
  - `no_carry`
  - `isolated_carry`
  - `long_carry_chain`
  - `internal_carry_chain`
  - `carry_distractor`
  - `many_9s_no_carry`
  - `random`

Slice meanings:

| Slice | Carry pattern | Purpose |
| --- | --- | --- |
| `no_carry` | No column produces outgoing carry. | Baseline for ordinary digit-wise addition without carry state. |
| `isolated_carry` | Exactly one carry-producing column, and it does not propagate. | Tests whether the model handles a single local carry event. |
| `long_carry_chain` | Carry propagates through every generated digit, e.g. `9999 + 1`. | Tests full-chain propagation and answer-length changes. |
| `internal_carry_chain` | Carry chain starts in the low-order digits but stops before the most-significant digit, e.g. `1099 + 1`. | Tests propagation inside the number without making the whole problem a boundary case like all 9s. |
| `carry_distractor` | Alternating 9-like surface pattern with some local carry activity but no long chain, e.g. LSD patterns `a=[9,0,...]`, `b=[0,1,...]`. | Tests whether models overreact to carry-suggestive surface digits. |
| `many_9s_no_carry` | Many 9s appear, but no column carries, e.g. LSD patterns `a=[9,0,9,...]`, `b=[0,8,0,...]`. | Control for “many 9s” heuristics without actual carry state. |
| `random` | Unconstrained random operands. | Broad background distribution, mostly for debugging or sanity checks. |

Canonical arithmetic fields should remain plain and stable (`a`, `b`, `answer`,
digit arrays, carry labels). Rendered prompt/output variants should be stored in
separate fields such as `prompt_a`, `prompt_b`, and `expected_output`.

Rows should distinguish arithmetic identity from rendered-example identity:
`problem_id` identifies the underlying addition problem, while `id` identifies a
specific rendered prompt/output condition. Goal 3 matched-pair metadata should
use optional fields such as `match_group_id`, `match_role`,
`target_column_lsd`, `intervention_variable`, `match_family`,
`match_constraints`, and `partner_problem_ids`; ordinary Goal 1 rows may omit
empty matching fields from artifacts.

## Checkpoint Focus

Default Goal 1 comparison:

- Thinking: `allenai/Olmo-3-7B-Think`
- Non-thinking: `allenai/Olmo-3-7B-Instruct`

The code should continue to make checkpoint substitution easy through config.
Future training-stage analyses may include base, instruct, Think-SFT,
Think-RLVR, or other staged checkpoints if available.

Tokenizer note:

- OLMo 3 uses Hugging Face `GPT2Tokenizer`, so treat it as GPT-2-style
  byte-level BPE through `AutoTokenizer`, not SentencePiece.
- Tokenization-sensitive analyses should prefer delimiter conditions when they
  require one visible digit per token or stable digit boundaries.

## Goal 1: Behavioral Frontier Across Training Stages

Purpose:

Establish where each checkpoint succeeds and fails. This is the behavioral
foundation for all mechanistic work.

Main question:

Do reasoning-tuned checkpoints improve uniformly, or do they show jagged gains
concentrated on specific carry structures, answer formats, digit formats, and
prompt modes?

Implementation requirements:

- Generate reusable synthetic datasets with stable IDs and exact labels.
- Run each checkpoint across prompt modes, digit formats, answer formats, digit
  lengths, and carry slices.
- Save exact local model-call artifacts: prompt, messages, rendered prompt,
  token IDs, generation config, decoded output, parsed answer, timing, package
  versions, model ID, model revision, seed, and timestamp.
- Produce figures only from saved artifacts.

Primary metrics:

- exact match
- parsed-answer accuracy
- first wrong digit from LSD and MSD views
- carry-specific error rate
- no-carry and carry-heavy error rates
- token count / generation length
- latency

Expected figures:

- Accuracy heatmap: digits x carry-chain length.
- Error localization: first wrong digit vs first carry-relevant column.
- Prompt-mode comparison across checkpoints.
- Digit-format and answer-format comparisons.
- Token count vs accuracy.

Minimum publishable result:

A clean behavioral characterization of how reasoning training changes arithmetic
failure modes. This is likely workshop-level by itself unless checkpoint-stage
differences are unusually surprising.

Current repo status:

- Goal 1 scaffold exists.
- Configs are enum-validated and documented in `docs/configs.md`.
- Dataset and run artifacts are local by default.
- Fake runner supports cheap smoke tests without model downloads.

## Goal 2: Decode Latent Algorithmic State

Purpose:

Test whether carry-related state is represented internally, and how this changes
across checkpoints and generation time.

Main question:

Does post-training make carry state more explicit, earlier, or more robustly
represented?

Activation sites to start with:

- residual stream pre-layer
- residual stream post-layer
- attention output
- MLP output

Suggested tooling:

- Use TransformerLens if OLMo architecture support is practical.
- Otherwise implement PyTorch hooks manually around Hugging Face models.

Store activations compressed by:

- checkpoint
- dataset example ID
- prompt mode
- digit format
- answer format
- layer
- component
- token position

Primary probe targets:

- `incoming_carry_i`
- `outgoing_carry_i`
- `output_digit_i`
- `raw_sum_i`
- `carry_chain_membership_i`
- `column_pointer_i`

Probe policy:

- Keep linear probes primary.
- Use logistic regression with L2 regularization and balanced classes.
- Use small MLP probes only as secondary analysis.

Controls:

- shuffled labels
- cross-template generalization
- train on short lengths, test on longer lengths
- train on one carry position, test on another
- matched carry/no-carry pairs
- digit identity controls
- answer-length controls

Temporal analysis:

Probe across both layer depth and generation token position. Important positions
include prompt end, first reasoning sentence, first carry mention, pre-final
answer, and answer digit positions.

Expected figures:

- Carry decodability vs layer by checkpoint.
- Layer x generation-time carry decodability heatmaps.
- Probe generalization across templates and digit lengths.
- Incoming-carry vs outgoing-carry decodability.

Minimum publishable result for Goals 1-2:

Reasoning checkpoints improve arithmetic unevenly, and those improvements
correspond to stronger, earlier, and more robust carry-state representations.
Reviewers may still object that decodability does not imply causal use, so Goal
3 is the main elevation.

## Goal 3: Causal Specificity Via Carry-State Interventions

Purpose:

Move beyond "carry is decodable" to "carry is causally used."

Main question:

Can we intervene on internal carry state and selectively alter the corresponding
output digit?

### Goal 3A: Activation Patching On Matched Pairs

Construct matched problem pairs that differ in a target carry variable while
controlling digit length, local digit identities where possible, answer length,
surface format, and prompt style.

Patch activations from a clean run into a corrupted run and measure:

- logit difference between correct and incorrect answer
- probability of correct full answer
- target digit probability
- first-wrong-digit repair rate

Critical measurement:

Do not rely only on full-answer accuracy. Measure whether the intervention
changes the specific downstream digit predicted by the target carry variable.

Expected figure:

Patching effect by layer and token position, separated by carry-heavy cases,
no-carry controls, and irrelevant-column controls.

### Goal 3B: Carry-State Interchange

This is the central high-value causal experiment.

Design:

Find two problems that are identical or closely matched except for incoming carry
at a target column. Example target column in base 10:

```text
7 + 2, incoming carry = 0 -> output digit 9
7 + 2, incoming carry = 1 -> output digit 0, outgoing carry 1
```

Patch the candidate carry representation from the carry case into the no-carry
case.

Predicted result:

- target digit changes in the carry-predicted direction
- downstream carry may propagate
- unrelated digits remain mostly unchanged

Candidate sites should be selected on a dev split, then evaluated on held-out
test examples.

Controls:

- irrelevant-column patch
- no-carry examples
- shuffled activation vector
- random same-layer token position
- digit-identity-matched but carry-mismatched examples
- carry-matched but digit-mismatched examples

Strong evidence criterion:

The intervention should change the target digit in the predicted direction,
affect carry-relevant cases more than no-carry controls, preserve unrelated
digits, replicate across templates, and become stronger or cleaner across
training stages.

Conference-shaped claim:

Reasoning post-training increases the causal manipulability and
column-specificity of latent carry state.

## Goal 4: Chain-of-Thought / State Coupling

Purpose:

Determine whether reasoning tokens are scratchpad state, control tokens,
semantic reports, or post-hoc narration.

Main question:

How tightly is visible reasoning coupled to latent algorithmic computation?

### Goal 4A: Temporal Precedence

Identify spans such as:

- `carry`
- `carried`
- `carry the 1`
- `add the carry`
- `write down`

Compare carry decodability before, during, and after carry mentions, plus
pre-final-answer positions.

Interpretations:

- carry state before text: CoT may report latent computation
- carry state after text: CoT may mediate computation
- carry state only at answer: CoT may be weakly coupled
- carry state appears but text is wrong: CoT may be misleading or decorative

### Goal 4B: Narration Intervention Ladder

Run systematic CoT perturbations:

- free CoT
- answer-only
- correct structured CoT
- paraphrased correct CoT
- symbolic CoT
- nonsense arithmetic-shaped CoT
- carry-shuffled CoT
- heuristic narration
- forced no-carry narration

Measurements:

- accuracy
- target digit accuracy
- carry error rate
- latent carry decodability
- causal site overlap
- patching transferability

Pathway invariance analysis:

Compare top-k causal sites across CoT conditions using overlap and rank
correlation of effect sizes.

Interpretation examples:

- same pathway and stable accuracy: CoT likely not central
- different pathway and stable accuracy: CoT changes strategy
- same pathway and accuracy drops: CoT disrupts inputs but not mechanism
- corrupted CoT flips latent carry: CoT acts as scratchpad or strong mediator
- corrupted CoT does not flip latent carry: CoT is ignored or post-hoc

## Goal 5: Training-Stage Developmental Analysis

Purpose:

Turn the project from an arithmetic case study into a study of how reasoning
models develop across training stages.

Main question:

How does post-training reshape the relationship between latent computation, CoT,
and answer production?

Compare each checkpoint on:

- behavioral accuracy
- carry probe AUC / accuracy
- carry probe generalization
- causal patching effect
- carry interchange effect
- CoT pathway invariance
- CoT corruption sensitivity

Possible developmental patterns:

1. **SFT teaches narration, RLVR teaches computation**
   - Base: weak carry state, poor accuracy
   - SFT: more carry words, modest accuracy, weak causal carry state
   - RLVR: stronger causal carry state and better digit-specific interventions

2. **CoT improves control, not internal algorithmic state**
   - Reasoning checkpoints use different pathways under different narrations,
     but carry-state interventions remain diffuse or weak.

3. **Latent computation precedes narration across stages**
   - Carry state appears before CoT mentions, and CoT corruption barely affects
     causal pathways.

4. **No clean localization**
   - Carry is decodable but interventions are weak or distributed.
   - This can still be publishable as a careful negative result if methodology
     is strong.

Expected summary panel:

Checkpoint x metrics for accuracy, carry decodability, carry causality, CoT
coupling, and base-k transfer.

## Goal 6: Base-k Arithmetic Extension

Purpose:

Test whether the model has learned a flexible carry algorithm or a
decimal-specific heuristic.

Main question:

Do identified carry mechanisms transfer when the carry threshold changes?

Candidate bases:

- base 6
- base 8
- base 10
- base 12
- base 16

Base-k setup:

- For base <= 10, use digits `0` through `9` as needed.
- For base 12 or 16, use `0` through `9` plus `A`, `B`, `C`, etc.
- Prompts must explicitly state the base and valid digits.

Example:

```text
We are doing arithmetic in base 8.
In base 8, the digits are 0 through 7.
What is 637_8 + 145_8?
Think step by step, then give the answer in base 8.
```

Conditions:

- explicit base instruction
- few-shot base instruction
- symbolic algorithm instruction
- unfamiliar digit symbols

Hypotheses:

1. **Algorithmic transfer**
   - Analogous carry representations appear in base-k.
   - Decodability and causal interventions transfer across bases.

2. **Decimal specialization**
   - Base-10 works much better than unfamiliar bases.
   - Base-10 carry circuits do not transfer cleanly.

3. **Reasoning training improves rule binding**
   - Thinking checkpoints bind the explicit rule "carry if total >= k" better
     than non-thinking checkpoints.
   - Carry representations emerge after the base rule is stated.

Base-k causal test:

Repeat carry-state interchange. Example in base 8:

```text
5 + 2, incoming carry = 0 -> output 7
5 + 2, incoming carry = 1 -> output 0, carry 1
```

Patch incoming-carry representation from the second case into the first and test
whether the target digit changes from `7` to `0`.

Base-k should remain optional until base-k behavioral accuracy is high enough to
make interventions interpretable.

## Priority Ladder

Tier 1: necessary core

- Goal 1: behavioral frontier
- Goal 2: carry-state probing
- Deliverable: arithmetic improvements correspond to stronger and earlier carry
  representations.
- Likely venue: workshop, or borderline main-conference if findings are
  surprising.

Tier 2: conference-critical

- Goal 3: causal carry-state interventions
- Deliverable: carry state is causally manipulable and column-specific.
- Likely venue: credible ICLR/ICML/NeurIPS submission if clean.

Tier 3: strong conference story

- Goal 4: CoT/state coupling
- Goal 5: training-stage development
- Deliverable: reasoning post-training changes the causal coupling between
  latent algorithmic state, visible reasoning traces, and answer production.

Tier 4: high-upside extension

- Goal 6: base-k arithmetic
- Deliverable: test whether the mechanism reflects flexible algorithmic
  computation or decimal-specific specialization.

## Expected Final Figures

1. Behavioral frontier: accuracy by digit length and carry-chain length across
   checkpoints and prompt modes.
2. Carry-state emergence: layerwise carry decodability across checkpoints.
3. Temporal ordering: carry decodability over layer x generation position,
   aligned to explicit carry mentions.
4. Causal carry intervention: patching and interchange effects on target answer
   digits.
5. CoT coupling: pathway overlap and accuracy under narration perturbations.
6. Base-k extension: base-k accuracy and carry-state decodability/causality
   compared with base 10.

## Key Risks And Mitigations

Risk: probes decode shortcuts.

Mitigation: matched pairs, cross-template tests, digit controls, answer-length
controls, and train/test splits across lengths.

Risk: patching effects are diffuse.

Mitigation: report localization honestly and prioritize target-digit effects over
full-answer accuracy.

Risk: CoT interventions alter token budget or formatting.

Mitigation: use length-controlled conditions and matched final-answer formats.

Risk: base-k performance is too weak.

Mitigation: treat base-k as optional; start with explicit and few-shot
instructions, then only mechanistically analyze bases with interpretable
behavioral performance.

Risk: OLMo checkpoints are difficult to instrument.

Mitigation: start with residual stream probes and patching; only move to
head/MLP localization after infrastructure is stable.

## Handoff Notes For Future Agents

- Read `docs/progress.md` first for current implementation state.
- Read `docs/configs.md` before changing configs.
- Do not collapse `prompt_mode`, `digit_format`, and `answer_format`; they are
  independent experimental axes.
- Preserve canonical arithmetic fields. Add rendered variants in separate fields.
- Keep all generated datasets and runs reproducible from config files.
- Save raw model inputs and outputs exactly; never rely only on aggregate
  metrics.
- For Goal 2+, use saved dataset IDs as the join key between behavior,
  activations, probes, and interventions.
- For Goal 3, use `problem_id` and explicit matching metadata rather than split
  names to identify clean/corrupt/control intervention groups.
- Keep Goal 1 behavior stable while adding later mechanistic modules.
