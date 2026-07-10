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

To produce figures:
```
uv run carry-trace figures goal2 --probe-id <directory within runs/probes/>
```
