# Goal 3.5: Expanded Natural-CoT Replay

## Objective

Goal 3.5 expands the crossed natural-CoT replay analysis without collecting
residual-stream activations. It tests harder four- and six-digit problems while
recording model-specific completion coverage explicitly.

The proposed run generates 64 random carry-balanced problems at each digit
length. Both checkpoints receive the same standard-format free-CoT prompts.
Natural source generations use the Goal 1 sampling settings with a larger
4,096-token budget. Replay then crosses each eligible source CoT with both
receiving checkpoints.

## Selection

Every source call is retained in `source_calls.jsonl`, including token-limit
hits and completions without a usable final-answer boundary. A problem enters
crossed replay only when both models finish within the token budget and provide
all requested CoT boundaries. Correctness and cleanliness are not selection
criteria; they define sensitivity-analysis subsets afterward.

`completion_coverage.jsonl` reports requested calls, token-limit rates, usable
boundaries, shared selection, source accuracy, and clean-termination rates by
model and digit length.

The incorrect-CoT analysis is additionally augmented with historical Goal 1
and Goal 2 completions. Historical imports are limited to four- and six-digit
`free_cot` examples from the `random` slice with standard digit and answer
formats. Only incorrect, non-truncated completions with a clean terminal answer
and recoverable `cot_end` boundary are imported. They are replayed at `cot_end`
into both receivers and do not enter the original paired source-by-receiver
interaction analysis.

## Metrics

`replay_primary_metrics.jsonl` stores problem-paired estimates and bootstrap
confidence intervals for:

- self-replay change from the no-reasoning baseline;
- Full-source minus SFT-source effects within each receiver;
- the receiver-by-source interaction;
- exact-match versions of decoded replay effects;
- copying and correction rates for naturally incorrect clean CoTs.

Metrics are written for all shared problems, problems where both source CoTs
are correct, and problems where both are correct and clean. Each analysis is
reported overall and by digit length.

Incorrect-CoT metrics are also reported separately for the native Goal 3.5
sample, the pooled historical sample, and each historical source run. With the
configured local artifacts, preprocessing finds 130 historical error CoTs from
125 distinct problems.

## Execution

```bash
uv run carry-trace run goal35 \
  --config configs/experiments/goal35_olmo3_natural_cot_replay.yaml
```

The command generates the dataset, gathers source completions, derives replay
prefixes, executes teacher-forced scoring and configured deterministic replay
decoding, and writes the final metrics. Source calls and replay artifacts are
append-only and resume from the latest incomplete run with the same config
hash. Progress is printed for every source completion and replay batch.

To append historical cases to an already completed Goal 3.5 run, select that
directory explicitly:

```bash
uv run carry-trace run goal35 \
  --config configs/experiments/goal35_olmo3_natural_cot_replay.yaml \
  --resume-run-dir runs/goal35_olmo3_natural_cot_replay-2026-07-18T145827.521405Z0000
```

Existing source calls, replay scores, and replay generations are reused by
stable ID. Only the imported historical baselines and endpoint replays are
appended.
