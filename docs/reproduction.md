# Exact Data Used
Exact dataset available [here](https://huggingface.co/datasets/nialldalton12/carry-trace/tree/main)

# Goal 1

Run
```
uv run carry-trace dataset generate --config configs/datasets/goal1_paper_like.yaml
uv run carry-trace run goal1 --config configs/experiments/goal1_olmo3_real.yaml
```

Runs are resumable automagically using a hash of the config above and corresponding manifest, since they will take a while to run.

To produce figures:
```
uv run carry-trace figures goal1 --run-id <directory within runs/>
```

# Goal 2

Run

```
uv run carry-trace dataset generate --config configs/datasets/goal2_paper.yaml
uv run carry-trace run goal2 --config configs/experiments/goal2_olmo3_real.yaml
uv run carry-trace probe goal2 --config configs/probes/goal2_probes_real.yaml
```

For prompt-specific standard-format probes, fit the two prompt modes in
separate artifacts:

```
uv run carry-trace probe goal2 --config configs/probes/goal2_probes_real_standard_only_answer_only.yaml
uv run carry-trace probe goal2 --config configs/probes/goal2_probes_real_standard_only_free_cot.yaml
```

To produce figures:
```
uv run carry-trace figures goal2 --probe-id <directory within runs/probes/>
```

# Goal 3

Derive the natural-CoT replay and intervention bundle, then execute the reduced
Goal 3A-3C design:

```
uv run carry-trace dataset goal3 --config configs/datasets/goal3_olmo3_natural_cot.yaml
uv run carry-trace run goal3 --config configs/experiments/goal3_olmo3_natural_cot_pilot.yaml
uv run carry-trace run goal3 --config configs/experiments/goal3_olmo3_natural_cot_clamp_pilot.yaml
uv run carry-trace run goal3 --config configs/experiments/goal3_olmo3_natural_cot.yaml
```

Goal 3 runs are resumable by config hash and load SFT and Full sequentially.

# Goal 3.5

Goal 3.5 expands crossed natural-CoT replay to 64 four-digit and 64 six-digit
problems with a 4,096-token source-generation budget. It does not collect
activations or run residual interventions.

No separate dataset-generation command is required. The run command
deterministically generates the configured dataset under
`data/generated/goal35_olmo3_natural_cot_replay/` before gathering source
completions:

```
uv run carry-trace run goal35 --config configs/experiments/goal35_olmo3_natural_cot_replay.yaml
```

The maximum workload before shared-completion filtering is 256 long source
generations, 1,792 teacher-forced replay scores, and 1,280 short replay
generations. Every source call is retained, including token-limit hits. Only
problems with usable CoT boundaries from both models enter crossed replay.

The run prints progress for each source completion and replay batch. It resumes
the newest incomplete run with the same config hash. Main outputs include
`completion_coverage.jsonl`, `replay_scores.jsonl`,
`replay_generations.jsonl`, and `replay_primary_metrics.jsonl`.

The default config also imports incorrect clean `cot_end` completions from the
listed historical Goal 1/2 runs. Imports are restricted to random-slice,
standard-format, four- and six-digit free-CoT examples and are reported as a
separate historical cohort. Sync the required lightweight run metadata with:

```bash
scripts/sync_to_runpod.sh --include-goal35-history \
  root@<host>:/workspace/carry-trace/ <port>
```

With the configured historical artifacts, this adds 510 endpoint replay cases
without adding any long source generations. Selection details are written to
`historical_import_status.jsonl`.

To reuse the source calls and replay outputs from the completed first Goal 3.5
run on RunPod, append only the new historical cases with:

```bash
uv run carry-trace run goal35 \
  --config configs/experiments/goal35_olmo3_natural_cot_replay.yaml \
  --resume-run-dir runs/goal35_olmo3_natural_cot_replay-2026-07-18T145827.521405Z0000
```

Generate the Goal 3.5 paper figures and plain-text tables with:

```bash
uv run carry-trace figures goal35 \
  --run-id runs/goal35_olmo3_natural_cot_replay-2026-07-18T145827.521405Z0000
```

Outputs are written under the run's `figures/` directory. Tables use `.txt`;
figures use `.png`.
