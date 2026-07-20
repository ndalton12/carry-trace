# Goal 3: Natural-CoT Causal Coupling

## Objective

Goal 3 tests whether the SFT and Full checkpoints differ in how causally their
naturally generated reasoning influences arithmetic output. It deliberately
keeps the Goal 1 prompts, standard number format, natural free-CoT completions,
and Goal 2 activation locations rather than imposing a new structured
scratchpad.

Because `Olmo-3-7B-Instruct` includes both DPO and RLVR after the released SFT
checkpoint, an SFT-versus-Full comparison identifies the effect of post-SFT
training as a whole. It cannot isolate RLVR from DPO without the intermediate
DPO checkpoint.

## Goal 3A: Natural-CoT Mediation

Use naturally completed, standard-format free-CoT generations shared by both
models. Reuse the token boundaries recorded during Goal 2 at `cot_1_3`,
`cot_2_3`, and `cot_end`, plus a no-reasoning baseline at `prompt_final`.

For each prefix, replay the original user prompt with the recorded natural CoT
prefix as an assistant prefill and measure:

- correct-answer sequence log probability;
- deterministic generated-answer accuracy;
- change relative to the no-reasoning prefix;
- sensitivity to naturally correct versus naturally incorrect reasoning.

The complete natural output and both the recorded and replay prefix token
counts remain in the dataset so replay runners can audit the derivation.
All prefixes receive teacher-forced sequence scoring. The full replay-plus-
residual run restricts deterministic decoding to `prompt_final` and `cot_end`.
The replay-only run also decodes `cot_2_3`, which provides one intermediate
behavioral measurement after substantial reasoning without paying for the
often incomplete and noisy `cot_1_3` continuations. The runner appends a fixed
answer cue after `prompt_final`, `cot_1_3`, and `cot_2_3`. A `cot_end` prefix
already terminates immediately before the final numeric candidate, so scoring
and generation continue directly with answer digits rather than inserting a
second answer cue or synthetic space.
`completion_audits.jsonl` records numeric-candidate and trailing-text signals
so unusual answer-then-more-reasoning completions can be separated in a
sensitivity analysis.

## Goal 3B: Crossed CoT Replay

For each arithmetic problem, cross the source of the natural reasoning prefix
with the receiving checkpoint:

| Receiver | Reasoning source |
| --- | --- |
| SFT | SFT CoT |
| SFT | Full CoT |
| Full | SFT CoT |
| Full | Full CoT |

The receiver-by-source interaction separates reasoning-content quality from a
checkpoint's sensitivity to the same reasoning text. Each replay case also
links to the native answer-only example for the same problem and format.

## Controlled Carry-Statement Intervention

This is the primary planned causal extension to natural-CoT replay. It tests
whether an explicit statement about one incoming carry selectively controls the
answer digit that consumes that carry, and whether SFT and Full differ in their
sensitivity to the same statement. It does not require a new Goal 1 or Goal 2
run, activation extraction, or an additional checkpoint. The comparison is
interpreted as SFT versus post-SFT training as a whole, not as an isolated RLVR
effect.

The controlled statement is placed directly after the user prompt as a short
assistant prefill, replacing rather than following the natural chain of
thought. Appending it to a natural completion would confound the intervention
with carry claims, calculations, and possible answers already present in that
completion. See [Controlled Carry-Statement Intervention](goal3-controlled-carry-intervention.md)
for the standalone experiment specification.

The existing bundle contains 100 shared problems, of which 87 have correct
natural completions from both checkpoints. Correct natural completions mention
carry language in 27 SFT traces and 34 Full traces, but only 16 shared problems
have explicit carry language in both traces. Natural carry-mention analysis is
therefore descriptive; the controlled intervention uses standardized text so
all eligible problem-column cases can contribute to the causal estimate.

### Case Construction

Reuse the two- and four-digit test problems. Create one case for each eligible
incoming carry at the tens column for two-digit problems and at the tens or
hundreds column for four-digit problems. Prefer the same locality restriction
as the residual experiment: flipping the incoming carry must leave the outgoing
carry from the affected column unchanged. The factual and counterfactual
answers then differ only at the target digit.

For each case, construct otherwise identical assistant prefixes containing one
of the following statements immediately before a fixed final-answer cue:

1. factual carry, for example `The incoming carry to the tens column is 1.`;
2. counterfactual carry, with only the stated carry value changed;
3. carry omitted, while retaining the surrounding column context;
4. a meaning-preserving factual paraphrase, such as
   `Add one carried from the previous column.`;
5. a length-matched irrelevant-column numerical edit.

The exact wording and answer cue are fixed before evaluation. Prefixes must not
contain the full factual or counterfactual answer. The primary dataset includes
all eligible cases; the 87 shared-correct problems form a preregistered
sensitivity subset rather than a requirement, because these controlled prefixes
do not depend on a natural source completion.

### Outcomes And Estimand

Teacher-force both the factual and locally counterfactual full answers under
every prefix. For prefix condition `z`, define

```text
L(z) = log P(counterfactual answer | z) - log P(factual answer | z)
```

The primary visible-carry effect is

```text
CE_text = L(counterfactual carry statement) - L(factual carry statement)
```

A positive value means that changing the visible carry statement shifts the
receiver toward the carry-consistent counterfactual answer. Compare `CE_text`
between SFT and Full on the same cases. Whole-answer sequence probabilities are
primary because a tokenizer token may contain multiple answer digits.

Secondary outcomes are deterministic counterfactual-answer rate, target-digit
change rate, and off-target-digit preservation. The paraphrase must reproduce
the factual-statement effect, while omission and irrelevant-column edits serve
as controls. Report problem-clustered bootstrap intervals and separate two- and
four-digit results before pooling.

### Execution Gate

Implement and run this teacher-forced text experiment after Goals 3A and 3B.
It is expected to require roughly 100--150 problem-column cases crossed with
five prefixes, two receivers, and two scored answers, without long generation
or activation storage. Run the probe-guided residual grid only if the visible
carry effect is target-specific, exceeds the irrelevant-column control, and
shows a stable SFT-versus-Full difference. Otherwise keep the residual result
exploratory and make the controlled text intervention the final causal result.

## Goal 3C: Probe-Guided Residual Intervention

Use the held-out natural free-CoT examples and the same activation locations as
Goal 2. For each eligible target column, materialize an intervention case that
records:

- the factual and counterfactual incoming or outgoing carry;
- the carry column and affected output column;
- the factual and counterfactual target output digit and answer;
- the model, activation tensor, activation location, and layer;
- the column-specific probe target and direction-training split;
- the expected unchanged off-target digits.

The primary intervention changes only the residual component aligned with a
column-specific carry direction. Each direction is a standardized logistic
probe normal, oriented from carry zero to carry one and normalized to unit
length. A scale-one intervention moves the activation by the projected class
mean gap toward the counterfactual class. Full-vector cross-model patches are
not part of the primary design. Interventions are performed within each model
and their effects are compared afterward.

Default causal locations are `prompt_final`, `cot_2_3`, and `cot_end`. The
current config uses layers 16 and 24 as preregistered robustness points, not as
a search for a best layer.

Incoming-carry cases exclude target column 0, where incoming carry is always
zero. Outgoing-carry cases target carry from column `c` and evaluate output
column `c+1`. Cases can require that flipping the carry leaves the carry out of
the affected output column unchanged. That restriction makes the arithmetic
counterfactual local: only the affected digit should change.

By construction, outgoing carry from column `c` equals incoming carry to
column `c+1`. At the shared global locations used here, those two probe targets
therefore have identical labels after shifting the column index. Outgoing carry
is treated as an equivalence/control analysis, not independent evidence for a
second latent variable. It becomes distinct only if later work uses
column-aligned source locations or explicitly tests carry transmission timing.

The primary outcome is the intervention-induced change in
`log P(counterfactual answer) - log P(factual answer)`. Whole-answer sequence
probabilities remain valid when multiple digits share one tokenizer token.
The current configuration uses teacher-forced scoring without residual
intervention decoding. Cases whose legacy replay boundary does not exactly
match the saved activation token are excluded from 3C. Model differences are
paired on the same arithmetic cases and uncertainty is clustered by problem or
match group.

## Execution

```bash
uv run carry-trace run goal3 \
  --config configs/experiments/goal3_olmo3_natural_cot.yaml
```

The run scores all 1,400 replay assignments without autoregressive decoding,
decodes the 600 replay endpoint assignments, scores the exactly aligned
incoming-carry intervention grid, and does not decode residual interventions.
Runs are append-only and resumable by config hash; the two checkpoints are
loaded sequentially.

To rerun only Goals 3A and 3B without repeating residual interventions, use:

```bash
uv run carry-trace run goal3 \
  --config configs/experiments/goal3_olmo3_natural_cot_replay.yaml
```

This replay-only configuration teacher-force scores all 1,400 assignments and
decodes the 600 assignments at `prompt_final` and `cot_end`. The two partial-CoT
locations remain score-only.

### Goal 3C Calibration Pilot

Before the full run, execute:

```bash
uv run carry-trace run goal3 \
  --config configs/experiments/goal3_olmo3_natural_cot_pilot.yaml
```

The pilot deterministically selects five 2-digit and five 4-digit problems,
uses all preregistered layers and locations, and performs teacher-forced scoring
without autoregressive decoding. Each case is crossed with scales 0.5, 1.0, and
2.0 and with the carry direction and a fixed orthogonal unit-vector control.
The control uses the same sign and class-gap magnitude as the carry shift.

Use `residual_control_contrasts.jsonl` and the corresponding summary rows to
choose the smallest scale with a consistently positive counterfactual shift
that exceeds the orthogonal control without an isolated layer/location spike.
This is a calibration rule, not a model-difference hypothesis test. After
choosing the scale, update `intervention_scales` and
`decode_intervention_scale` in the full config before launching it.

### Goal 3C Clamp Diagnostic

The fixed-gap pilot did not show a stable dose response or model difference,
so run the following teacher-forced diagnostic before the full experiment:

```bash
uv run carry-trace run goal3 \
  --config configs/experiments/goal3_olmo3_natural_cot_clamp_pilot.yaml
```

The diagnostic starts from the same deterministic ten-problem pilot subset and
then requires correct, cleanly terminated source completions from both models.
The current bundle retains seven problems, 108 residual cases, and 864 scored
executions. It uses layers 16 and 24, all three residual locations, no decoding,
and one exact carry-projection clamp plus three deterministic orthogonal
controls at both the prefix boundary and final answer-cue token.

For a clamp at residual state `h`, carry direction `d`, and counterfactual
train-class projection mean `m`, the requested change is
`(m - h dot d) d`. The answer-cue condition computes this magnitude from the
original prefix boundary and applies the same vector at the cue, avoiding a
comparison between cue-token states and prefix-trained class means. Artifacts
record requested and realized BF16 shift norms and projection changes.

Use a successful prefix-boundary clamp to replace fixed-gap addition in the
full Goal 3C run. An effect only at the answer cue is a positive-control result,
not evidence that the saved CoT boundary is causally coupled to the answer. If
the clamp is numerically realized but does not exceed the averaged orthogonal
controls, keep Goal 3C exploratory and base the primary Goal 3 result on replay.

## Current Results

### Goal 3A/3B: Natural-CoT Replay

The initial replay run used 100 standard-format problems with valid CoT
completions from both models: 64 two-digit and 36 four-digit problems. Full's
original CoTs were correct on 97% of them, compared with 90% for SFT.

Natural reasoning text clearly influenced the answer. On the ten problems
where SFT produced a clean but incorrect CoT, both receivers answered all ten
correctly without the CoT. After replaying the incorrect SFT CoT, SFT answered
none correctly and Full answered one correctly. Both receivers repeated the
source's exact wrong answer in seven cases.

Crossed replay did not show that Full uses the same CoT text differently from
SFT. The receiver-by-source interaction at `cot_end` was approximately zero
(0.01 answer-sequence log probability, 95% bootstrap CI [-0.43, 0.45]). The
plain interpretation is that Full generated more accurate reasoning, but the
two receivers were similarly sensitive to supplied reasoning.

These replay numbers are provisional. The initial runner inserted a second
`Final answer:` cue after `cot_end` prefixes that already stopped immediately
before the answer. The corrected runner now continues directly into the answer
digits, and the replay-only rerun should replace the initial endpoint results.

### Goal 3C: Carry-Projection Clamp

The clamp worked mechanically: it moved the residual-stream projection to the
requested counterfactual carry-class mean with negligible numerical error. It
nevertheless did not move answer preference more than matched orthogonal
controls in a reliable way.

The full run retained 73 shared-correct problems and 1,040 carry-versus-control
contrasts. Averaged by problem, the carry-minus-control effect was 0.004 for
SFT (95% bootstrap CI [-0.003, 0.012]) and -0.002 for Full ([-0.009, 0.005]).
No location or layer showed a stable effect, and there was no evidence that
Full was more sensitive than SFT.

The result is therefore a causal null: carry state is decodable from the
residual stream, but the linear probe direction is not a sufficient causal
control axis for the arithmetic answer. This does not rule out a distributed
or multi-token carry mechanism.

## Dataset Bundle

The Goal 3 dataset command derives a versioned bundle from a completed Goal 2
activation run:

- `replay_prefixes.jsonl`: unique natural reasoning prefixes from source model
  generations at recorded Goal 2 token locations;
- `replay_cases.jsonl`: no-reasoning, self-replay, and crossed-replay receiver
  assignments for Goals 3A and 3B;
- `residual_intervention_cases.jsonl`: Goal 3C evaluation specifications tied
  to existing activation tensors;
- `manifest.json`: source paths, filters, config hash, counts, and exclusions.

The bundle is model-output-derived. Rebuilding it requires the source Goal 2
run and its original dataset. New Goal 2 records contain the exact generated
token IDs. Legacy records are retokenized and aligned to the nearest matching
recorded boundary token; the source, index shift, and original location are
saved for audit. Terminal control tokens such as `<|im_end|>` are excluded from
assistant replay prefixes. The command does not regenerate or normalize natural
CoT text.

## Initial Scope

The initial paper-facing analysis uses 2- and 4-digit, standard-input,
standard-answer, free-CoT examples from the shared naturally terminated subset.
Six-digit free-CoT examples remain out of the primary analysis because the Full
checkpoint has low and biased natural-termination coverage at the current token
budget.
