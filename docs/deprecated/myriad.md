# Myriad Smoke Jobs

This repo includes an SGE jobscript for a small OLMo 3.1 32B Instruct smoke run on UCL Myriad.

## Files

- `scripts/myriad_olmo31_32b_instruct_smoke.sh`: SGE `qsub` jobscript.
- `scripts/sync_to_myriad.sh`: local rsync helper that excludes gitignored files, then explicitly syncs `data/`.
- `configs/datasets/myriad_olmo31_32b_instruct_smoke.yaml`: 12-example smoke dataset.
- `configs/experiments/myriad_olmo31_32b_instruct_smoke.yaml`: vLLM run using `allenai/Olmo-3.1-32B-Instruct`.

## Sync From Local

From your local repo checkout:

```bash
scripts/sync_to_myriad.sh myriad:/home/ucabnd3/Scratch/carry-trace/
```

The first sync pass uses `.gitignore` as an rsync filter and excludes `.git/`, so local runs, virtualenvs, and caches are not copied. The script then runs a second explicit `rsync` for `data/`, so ignored generated datasets under `data/generated/` are copied too.

The `myriad` SSH alias is defined in `~/.ssh/config` and routes through the UCL SSH gateway. Keep the trailing slash on the remote path so rsync treats it as the repo directory.

To sync to a different Scratch location:

```bash
scripts/sync_to_myriad.sh myriad:/home/ucabnd3/Scratch/projects/carry-trace/
```

## Submit

SSH to Myriad, enter the synced repo, then submit from the repo root:

```bash
ssh myriad
cd /home/ucabnd3/Scratch/carry-trace
qsub scripts/myriad_olmo31_32b_instruct_smoke.sh
```

The script requests one GPU on Myriad `U/V` nodes because those nodes have A100 80GB GPUs. Myriad `L` nodes have A100 40GB GPUs, which are unlikely to fit OLMo 32B BF16 without quantization.

The script always uses `uv`. If `uv` is not already on `PATH`, it installs a local copy into `.uv-bin` inside the repo before running `uv sync`.

The smoke experiment uses vLLM with:

```yaml
batch_size: 2
dtype: bfloat16
gpu_memory_utilization: 0.95
max_model_len: 4096
max_new_tokens: 2048
temperature: 0.6
top_p: 0.95
do_sample: true
```

If dependency or model downloads are slow or blocked on compute nodes, warm the environment/cache on a login node before submitting:

```bash
cd /home/ucabnd3/Scratch/carry-trace
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.uv-bin" sh
  export PATH="$PWD/.uv-bin:$PATH"
fi
uv sync
uv pip install "vllm==0.10.2" "hf_transfer>=0.1.8"
export HF_HOME="$HOME/Scratch/huggingface"
uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('allenai/Olmo-3.1-32B-Instruct')"
```

If the scheduler rejects `#$ -ac allow=UV`, change it to a single available 80GB type such as `#$ -ac allow=V` or `#$ -ac allow=U`.

## Monitor

After `qsub`, Myriad prints a job ID. Check queue state with:

```bash
qstat
qstat -u "$USER"
qstat -j <job_id>
```

If you only want the job IDs and names:

```bash
qstat | grep carry_olmo32_smoke
```

Because the jobscript uses `#$ -cwd` and `#$ -j y`, SGE writes the combined stdout/stderr log in the repo directory with a name like:

```text
carry_olmo32_smoke.o<job_id>
```

Follow it while the job runs:

```bash
tail -f carry_olmo32_smoke.o<job_id>
```

## Outputs

The job writes a timestamped artifact directory under:

```text
runs/myriad_jobs/
```

The most useful files are:

- `full_outputs.txt`: human-readable prompts, rendered chat prompts, decoded outputs, parsed answers, and correctness.
- `full_outputs.jsonl`: machine-readable full outputs.
- `full_outputs.csv`: tabular full outputs.
- `metrics_summary.csv`: aggregate Goal 1 metrics.
- `calls.jsonl`, `scored_calls.jsonl`, `dataset.jsonl`: raw carry-trace run artifacts.

The job also creates a `.tar.gz` archive beside the artifact directory.

The SGE log prints the exact artifact paths at the end:

```text
artifact_dir=/home/ucabnd3/Scratch/carry-trace/runs/myriad_jobs/...
artifact_tar=/home/ucabnd3/Scratch/carry-trace/runs/myriad_jobs/....tar.gz
```

Inspect outputs on Myriad:

```bash
less runs/myriad_jobs/<artifact_dir_name>/full_outputs.txt
column -s, -t < runs/myriad_jobs/<artifact_dir_name>/metrics_summary.csv | less -S
```

Copy the archived outputs back to your local machine:

```bash
scp myriad:/home/ucabnd3/Scratch/carry-trace/runs/myriad_jobs/<artifact_dir_name>.tar.gz .
tar -xzf <artifact_dir_name>.tar.gz
```

Or copy just the human-readable output dump:

```bash
scp myriad:/home/ucabnd3/Scratch/carry-trace/runs/myriad_jobs/<artifact_dir_name>/full_outputs.txt .
```
