# Goal 1

Run
```
uv run carry-trace dataset generate --config configs/datasets/goal1_paper_like.yaml
uv run carry-trace run goal1 --config configs/experiments/goal1_olmo3_real.yaml
```

Runs are resumable automagically using a hash of the config above and corresponding manifest, since they will take a while to run.
Exact dataset available [here](https://huggingface.co/datasets/nialldalton12/carry-trace/tree/main)
