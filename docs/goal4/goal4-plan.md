# Goal 4: Reasoning Dynamics Across Natural Chains of Thought

## Status

This document specifies the proposed Goal 4 experiments. It is a plan, not an
implementation or a record of completed results.

## Objective

Goal 4 asks when arithmetic answers become available during a natural chain of
thought (CoT), which reasoning steps change the answer, and whether the SFT and
Full OLMo 3 checkpoints move through different internal representation
geometries while reasoning.

The main comparison is between:

- `Olmo-3-7B-Instruct-SFT`, the supervised-fine-tuned checkpoint; and
- `Olmo-3-7B-Instruct`, called Full here, which received additional DPO and
  RLVR training after SFT.

The comparison therefore identifies the combined effect of post-SFT training.
It cannot attribute a difference specifically to RLVR without an intermediate
DPO checkpoint.

Goal 4 has four connected parts:

1. measure answer availability at every natural reasoning boundary;
2. estimate which natural reasoning units change the answer;
3. measure effective-rank and spectral dynamics at the same boundaries; and
4. study whether long, token-limited Full traces already contain a recoverable
   answer before they stop.

The behavioral and geometric measurements are deliberately paired. Effective
rank is treated as a diagnostic of representation complexity, not as a
standalone measure of reasoning quality or a causal mechanism.

## Motivation From Earlier Goals

Goal 1 collected natural free-CoT addition completions and established the
basic behavioral comparison across arithmetic conditions.

Goal 2 trained linear probes on residual-stream activations. Carry and output
digit information was decodable across a broad range of layers and CoT
locations. There was no narrow set of clearly best layers and no simple,
large SFT-versus-Full decodability gap. This suggests that another global probe
accuracy comparison is unlikely to explain how reasoning is used over time.

Goals 3 and 3.5 replayed natural CoTs into both checkpoints. Incorrect CoTs
often caused the receiver to repeat the source's wrong answer, showing that the
generated reasoning is causally coupled to the arithmetic output. Full was
more likely than SFT to recover from incorrect SFT-generated reasoning, but
both receivers were strongly influenced by it. Endpoint replay did not show a
clear general receiver-by-source interaction for correct CoTs. A partial replay
at two-thirds of the CoT showed a possible log-probability interaction, but the
location was an arbitrary token fraction and the endpoint result did not
corroborate it.

Goal 3 also moved a residual-stream activation along a decoded carry direction.
The intervention was numerically successful but did not affect the answer more
than matched orthogonal controls. This showed that a linearly decodable
direction was not, by itself, a specific causal control axis.

Goal 3.5 exposed a separate behavior that matters for Goal 4. With a
4,096-token source-generation budget:

| Digits | Model | Requested | Token-limit hits | Usable for replay |
| --- | --- | ---: | ---: | ---: |
| 4 | SFT | 64 | 1 | 62 |
| 4 | Full | 64 | 23 | 41 |
| 6 | SFT | 64 | 0 | 64 |
| 6 | Full | 64 | 41 | 21 |

Discarding the capped Full traces would condition the analysis on unusually
short or easy Full generations. Goal 4 therefore retains them for every
measurement that can be made before the cutoff and treats their unobserved
endpoints as censored.

## Research Questions

Goal 4 addresses five questions.

1. At which reasoning boundary can each checkpoint first produce the correct
   answer?
2. Which naturally generated reasoning units increase or decrease the
   probability of the correct answer?
3. Does Full extract different value from exactly the same CoT text as SFT?
4. Does activation geometry expand, compress, or otherwise reorganize as the
   answer becomes stable?
5. When Full reaches the token limit, did it already have access to the correct
   answer, was it still changing its answer, or was it failing to converge?

## Terminology

**Source model** is the checkpoint that originally generated a CoT.

**Receiver model** is the checkpoint through which that fixed CoT prefix is
replayed and measured. Crossing source and receiver gives four conditions:

| CoT source | Receiver |
| --- | --- |
| SFT | SFT |
| SFT | Full |
| Full | SFT |
| Full | Full |

**Reasoning boundary** is the token position immediately after a complete
reasoning unit. Most units are ordinary sentences, but Markdown list items,
headings, table rows, code lines, and standalone equations can also be complete
units in model-generated arithmetic reasoning.

**Answer commitment** is the first observed boundary at which deterministic
short-answer generation is correct and remains correct at every later observed
boundary. For a capped trace, this is called observed commitment through the
cutoff; the experiment cannot claim that it would remain stable after the
unobserved continuation.

**Right-censored** means that the final event was not observed before the
generation limit. A capped trace tells us that no terminal answer was observed
within 4,096 tokens, but not what would have happened with unlimited
generation.

## Data

### Problem Set

Use 64 standard-format, random, carry-balanced problems at each of 2, 4, and 6
digits:

- reuse the 64 two-digit problems and source completions from Goal 3;
- reuse the 64 four-digit problems and source calls from Goal 3.5; and
- reuse the 64 six-digit problems and source calls from Goal 3.5.

Both checkpoints were prompted with the same arithmetic problems. All source
calls are retained, including incorrect and token-limited completions.

The arithmetic target is known directly from the dataset, so correct-answer
scoring does not require the source CoT itself to finish correctly. Results are
reported separately for 2, 4, and 6 digits. They are not pooled into one main
number because digit length changes problem difficulty, CoT length, and
completion coverage.

### Analysis Cohorts

The following cohorts answer different questions and must not be silently
mixed:

1. **All observed prefixes:** every source trace, including capped traces, up
   to its last complete reasoning boundary. This is the primary trajectory
   cohort.
2. **Completed traces:** traces with a parseable final-answer boundary and no
   token-limit hit. This cohort supports endpoint and stable-commitment
   analyses.
3. **Shared completed traces:** problems completed by both source models. This
   supports a balanced source-model comparison but is explicitly conditional
   on both models finishing.
4. **Incorrect completed traces:** completed source CoTs whose terminal answer
   is wrong. This cohort supports error-uptake and recovery analyses.
5. **Extended six-digit traces:** the small prespecified continuation sample
   described below. This cohort is reported separately.

Every figure and table reports the number of unique problems, source traces,
and boundaries contributing to it.

## Defining Reasoning Boundaries

Natural CoTs contain prose, lists, equations, tables, and Markdown formatting.
A plain sentence tokenizer would miss many meaningful arithmetic steps.

The boundary parser should:

- split prose after sentence-ending punctuation;
- treat a completed Markdown list item, heading, table row, code line, or
  standalone equation as a reasoning unit;
- ignore formatting-only lines such as separators and code fences;
- map each text boundary back to an exact assistant token boundary; and
- retain the original text span for audit and qualitative analysis.

The primary reasoning region stops immediately before the first recognized
terminal answer candidate. A completion that emits an answer and then resumes
reasoning is retained but marked for a separate sensitivity analysis.

Before running the full experiment, manually audit a fixed sample of 20 traces
covering both models and all digit lengths. The audit checks that arithmetic
steps, table rows, and answer boundaries are being split sensibly. Parser rules
are then frozen before outcome analysis.

## Goal 4A: Answer Emergence At Every Boundary

At every observed reasoning boundary, replay the source prefix into both
receivers. Append one fixed final-answer cue and measure:

- teacher-forced log probability of the complete correct answer;
- correct-answer log probability per answer token;
- deterministic short-answer exact match; and
- the deterministic answer text.

If the prefix already ends in a recognized final-answer cue, do not append a
second cue. This avoids the duplicate-cue problem found in the first Goal 3
endpoint replay.

Whole-answer sequence scoring remains primary because one tokenizer token can
contain more than one decimal digit.

### Main Outcomes

- accuracy by reasoning boundary;
- change in correct-answer log probability over the CoT;
- first correct boundary;
- first observed stable correct boundary;
- number of reasoning tokens after observed commitment; and
- number and size of answer reversals.

All problems receive the short-answer measurements at every observed boundary.
Long natural continuation is reserved for the selected Goal 4B cases so that
the main trajectory experiment remains computationally manageable.

## Goal 4B: Value Of Natural Reasoning Units

For a reasoning unit ending at boundary `s`, define its answer value as:

```text
sentence_value(s) =
    log P(correct answer | prefix through s)
  - log P(correct answer | prefix before s)
```

A positive value means that adding the naturally generated unit made the
correct answer more likely. A negative value means that it moved the receiver
away from the correct answer.

Because the same source text is replayed into both receivers, the
Full-minus-SFT difference in sentence value directly tests whether the two
checkpoints use that text differently.

### Selected Natural Continuations

Autoregressive continuation is more expensive than short-answer scoring, so it
is run only around prespecified informative boundaries:

- the first incorrect arithmetic statement in an incorrect CoT;
- the largest positive answer-value unit in a correct CoT;
- the largest negative answer-value unit; and
- one low-absolute-value unit matched on digit length and approximate CoT
  progress.

For each selected unit, generate from both the prefix immediately before it and
the prefix immediately after it. Compare final accuracy, correction of an
existing error, repetition of the source answer, and tokens to termination.
These are natural prefixes from typical model use; no synthetic carry statement
or artificial scratchpad is inserted.

Run this continuation analysis on at most 16 source problems per digit length,
selected deterministically after applying the boundary categories above.
Use a 1,024-token continuation budget and report token-limit hits. The dense
short-answer analysis remains the source of population-level estimates.

## Goal 4C: Effective-Rank And Spectral Dynamics

### Why Measure Rank

Recent work links changes in activation geometry with compression,
generalization, and post-training dynamics. The relevant literature does not
support the rule that lower rank always means a better model. It instead
supports measuring the complete spectral trajectory on matched data and
relating it to independent behavior. See the local
[rank and compressibility review](../paper_summary/rank_compressibility_generalization_transformers.md).

Goal 4 therefore asks whether rank changes at behaviorally meaningful points,
not whether either checkpoint has one universally better rank value.

### Activation Collection

For each source CoT, run one teacher-forced forward pass through each receiver.
In a causal decoder, the hidden state at a token in this pass is the same state
the receiver obtains from the prefix ending at that token. One pass therefore
provides activations for every sentence boundary without regenerating the CoT
once per boundary.

Collect the residual stream after each transformer block at:

- every reasoning boundary;
- every token needed for a trailing 32-token local window; and
- the prompt-final and answer-digit positions.

The source text and token positions are identical across receivers. This paired
same-text comparison controls for wording, sentence length, and tokenization.
The native SFT-to-SFT and Full-to-Full conditions remain useful descriptions of
ordinary model behavior, but they are not the only checkpoint comparison.

### Rank Object 1: Local Boundary-Window Rank

For each problem, boundary, layer, and receiver, form a matrix from the 32
assistant-token activations ending at that boundary:

```text
W[i, s, l] has shape 32 x hidden_size
```

Center the 32 rows before taking the singular values. After centering, the
maximum possible rank is 31. Normalize scalar rank measures by 31 so they are
comparable across conditions.

Only boundaries with at least 32 preceding assistant tokens receive this
measurement. Do not pad early boundaries with prompt tokens, because that
would change what the window represents.

This rank measures the dimensional spread of the model's recent internal
trajectory within one CoT. It can be calculated at every eligible reasoning
boundary without aligning sentence numbers across different traces.

### Rank Object 2: Population Boundary-State Rank

At an aligned stage, stack one boundary activation from every included problem:

```text
H[s, l] has shape number_of_problems x hidden_size
```

Center across problems before calculating the spectrum. This rank measures how
many dimensions distinguish arithmetic problems at that reasoning stage.

Population stages are aligned in three ways:

- ordinal sentence number over the early region where a fixed shared problem
  set remains observed;
- relative CoT progress using the nearest real reasoning boundary; and
- position relative to observed answer commitment, such as two units before
  through two units after commitment.

The same problem IDs and same sample count must be used for every point in a
direct Full-versus-SFT comparison. Raw effective ranks with different sample
counts are not directly compared.

### Rank Object 3: Phase Token-Cloud Rank

As a higher-sample complement, divide each observed CoT into early, middle, and
late reasoning phases. Select 16 evenly spaced assistant tokens per problem in
each phase and stack them across problems. This produces up to 1,024 token
vectors per phase for 64 problems.

This measurement is closer to corpus-level activation-rank work and is less
limited by the 64-problem sample size. Tokens from the same problem are still
statistically dependent, which is why uncertainty is resampled by problem
rather than by token.

### Spectral Measurements

Use normalized entropy effective rank as the headline scalar:

```text
p_j = singular_value_j / sum(singular_values)
effective_rank = exp(-sum(p_j * log(p_j)))
normalized_effective_rank = effective_rank / maximum_possible_rank
```

Also calculate and retain:

- stable rank;
- participation ratio;
- dimensions needed for 50%, 90%, and 95% explained variance;
- variance share of the largest singular direction; and
- the complete singular-value spectrum.

For the primary preprocessing, RMS-normalize each residual vector using the
model's configured epsilon and then center each hidden feature across the rows
of the activation matrix. Repeat the headline analysis on centered raw
residual activations as a sensitivity check. A further sensitivity analysis
subtracts or fits out the layer-0 token representation to test how much of the
spectrum is explained by lexical and positional input structure.

### OLMo 3 Layer Bands

The released [OLMo 3 7B model
configuration](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/config.json)
has 32 transformer blocks, indexed `0` through `31`, with a repeating
four-block pattern: three sliding-attention blocks followed by one
full-attention block. The embedding output is recorded separately and is not a
transformer block.

Use four equal, architecture-aligned bands:

| Band | Transformer blocks | Full-attention blocks |
| --- | --- | --- |
| Early | 0-7 | 3, 7 |
| Lower middle | 8-15 | 11, 15 |
| Upper middle | 16-23 | 19, 23 |
| Late | 24-31 | 27, 31 |

Each band contains eight blocks and exactly two complete attention cycles.
This keeps band averages equally weighted and avoids choosing boundaries after
seeing the results.

The complete layer-by-boundary heatmap is the primary display. Band averages
are secondary summaries used to give one stable estimate per broad depth
region and to avoid presenting hundreds of individual layer-boundary tests as
independent discoveries. A localized layer effect can be highlighted only if
it is confirmed on held-out problems or a prespecified replication subset.

### Connecting Geometry To Behavior

The primary geometric contrasts are:

- Full minus SFT rank on the exact same source prefix;
- rank immediately before versus after observed answer commitment;
- rank change around high-value and low-value reasoning units;
- rank change around the first incorrect arithmetic statement; and
- rank trajectory for capped versus completed Full traces over their common
  observed region.

Possible interpretations are conditional:

- expansion followed by compression around commitment would be consistent
  with building and then consolidating an answer representation;
- an early stable low-rank state followed by behaviorally inert reasoning would
  be consistent with explanation after computation;
- a difference that appears on native CoTs but disappears on identical source
  text would be attributable mainly to trace content or length; and
- a difference that persists on identical text would be stronger evidence that
  post-SFT training changed internal processing.

Rank changes without corresponding behavioral changes remain descriptive
geometry, not evidence that rank controls the answer.

## Goal 4D: Small Six-Digit Extended-Budget Analysis

### Motivation

In Goal 3.5, 41 of 64 six-digit Full generations reached the 4,096-token limit.
These traces are useful through their cutoff, but they do not tell us whether
Full would eventually answer or whether it was already able to answer while
continuing to reason.

### Prespecified Rescue Sample

Select 12 of the 41 capped six-digit Full traces using seed `20260720`.
Allocate the 12 slots proportionally across the observed carry-count strata,
with at least one trace from each stratum when the number of slots permits.
Within each stratum, select problem IDs randomly using the fixed seed.

Continue from the exact saved assistant token prefix rather than generating a
new trace from the original prompt. Use the original source-generation
settings:

```yaml
additional_max_new_tokens: 4096
temperature: 0.6
top_p: 0.95
do_sample: true
```

The cumulative generated-token ceiling is therefore 8,192. The released model
configuration specifies 65,536 maximum positions, so the prompt plus this
generation budget remains comfortably within the supported context. The
extension has a worst-case cost of 49,152 additional generated tokens. Do not
recursively extend traces that remain unfinished at 8,192 generated tokens;
mark them as censored at the new limit.

Restarting sampled generation from a saved prefix may not reproduce the exact
random continuation that one uninterrupted call would have produced, because
the original random-number-generator state was not retained. It is still a
valid sample from the continuation distribution conditioned on the exact
observed prefix. The rescue cohort must therefore remain separately labeled.

### Rescue Outcomes

For each of the 12 traces, report:

- whether a parseable terminal answer appears by 8,192 tokens;
- whether that answer is correct;
- additional tokens to termination;
- forced short-answer accuracy at the original cutoff and later boundaries;
- whether observed answer commitment preceded terminal generation; and
- local rank trajectories over the added reasoning, with boundary activations
  retained for a separately labeled 12-trace pooled spectrum.

Because `n = 12` is small, report the individual trace outcomes, counts, and an
exact binomial interval for completion and accuracy rates. Do not use this
sample to replace the original `41/64` token-limit rate or make a precise
population claim. Its purpose is to distinguish plausible failure modes and
decide whether a larger continuation study would be worthwhile.

## Uncertainty And Bootstrapping

Many measurements come from each arithmetic problem: multiple tokens,
sentences, layers, and both receiver models. These measurements are correlated.
For example, 100 token activations from one problem are not equivalent to 100
independent arithmetic problems.

Uncertainty is therefore estimated by bootstrapping whole problems:

1. Start with the fixed set of problem IDs in an analysis cohort.
2. Sample the same number of problem IDs with replacement.
3. When a problem is selected, include all of its relevant tokens, boundaries,
   source conditions, and both receiver measurements as one block.
4. Recompute the rank statistic or behavioral contrast.
5. Repeat 10,000 times and take the central 95% interval.

The same bootstrap draw is used for SFT and Full. This preserves their paired
comparison and preserves the dependence among tokens from one trace.
Resampling individual tokens would incorrectly treat nearby tokens from one
CoT as independent evidence and would produce confidence intervals that are
too narrow.

For population rank, a bootstrap replicate can contain the same problem more
than once. Rather than physically duplicating identical activation rows,
assign each selected problem its bootstrap multiplicity as a weight. Calculate
the spectrum of the resulting weighted, centered activation matrix, or the
mathematically equivalent weighted covariance and convert its eigenvalues back
to singular values. Local-window rank is calculated within each trace first
and then averaged using the problem-level bootstrap weights.

Also run a sample-size stability analysis at 32, 48, and 64 problems. A rank
pattern is considered stable only if its direction and broad layer/time shape
do not depend on using the maximum sample size.

## Primary Figures And Tables

1. **Answer emergence:** correct-answer probability and deterministic accuracy
   over normalized reasoning progress, separated by digit length and receiver.
2. **Commitment alignment:** answer probability from two reasoning units before
   through two units after observed commitment.
3. **Sentence value:** distribution of answer-value changes for the two
   receivers on identical source text.
4. **Rank heatmaps:** layer by reasoning progress for each receiver, with
   separate source-model panels.
5. **Rank-difference heatmaps:** paired Full-minus-SFT normalized effective rank
   on identical prefixes.
6. **Commitment geometry:** effective-rank trajectory around observed answer
   commitment, with layer-band summaries.
7. **Error transition:** behavior and rank immediately before and after the
   first incorrect arithmetic statement.
8. **Six-digit censoring table:** original completion coverage, forced-answer
   state at the cutoff, and the 12-trace extended-budget outcomes.

Every rank figure should identify the activation object being measured:
boundary state, local 32-token window, or phase token cloud. These quantities
must not be labeled interchangeably as one generic internal rank.

## Main Claims The Experiment Can Support

With positive results, Goal 4 could support claims such as:

- the checkpoints differ in when a correct arithmetic answer becomes
  available during natural reasoning;
- particular natural reasoning units have different behavioral value for SFT
  and Full;
- post-SFT training changes the spectral trajectory induced by identical CoT
  text; or
- Full often continues reasoning after a correct answer is already
  recoverable.

Goal 4 cannot by itself show that a change in effective rank causes reasoning
quality, that one rank value universally measures generalization, or that an
observed SFT-versus-Full difference is caused specifically by RLVR.

## Implementation Order

1. Implement and audit reasoning-boundary extraction.
2. Derive the 2-, 4-, and 6-digit Goal 4 bundle from existing Goal 3 and 3.5
   artifacts.
3. Run short-answer scoring and generation at every observed boundary.
4. Collect same-text receiver activations and calculate the three spectral
   objects.
5. Select and run the limited natural-continuation cases.
6. Run the 12-trace six-digit extended-budget cohort.
7. Freeze analysis cohorts and generate the preregistered figures and tables.

This order obtains the high-coverage behavioral and rank results before paying
for any additional long generation.
