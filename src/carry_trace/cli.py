"""Command line interface for carry-trace."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.json import JSON

from carry_trace.config import load_dataset_config, load_experiment_config, load_goal2_config
from carry_trace.datasets import generate_dataset, upload_dataset_to_hub
from carry_trace.figures import make_goal1_figures
from carry_trace.goal2 import run_goal2
from carry_trace.inspect import inspect_tokenizer
from carry_trace.runs import run_goal1

app = typer.Typer(help="Reproducible carry-trace experiments.")
dataset_app = typer.Typer(help="Dataset commands.")
run_app = typer.Typer(help="Experiment run commands.")
figures_app = typer.Typer(help="Figure commands.")
inspect_app = typer.Typer(help="Inspection commands.")
console = Console()

app.add_typer(dataset_app, name="dataset")
app.add_typer(run_app, name="run")
app.add_typer(figures_app, name="figures")
app.add_typer(inspect_app, name="inspect")


@dataset_app.command("generate")
def dataset_generate(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Generate a reusable synthetic addition dataset."""
    dataset_config = load_dataset_config(config)
    jsonl_path, manifest_path, rows = generate_dataset(dataset_config)
    console.print(f"Wrote {len(rows)} examples to {jsonl_path}")
    console.print(f"Wrote manifest to {manifest_path}")


@dataset_app.command("upload")
def dataset_upload(
    dataset_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    repo_id: Annotated[str, typer.Option(help="Hugging Face dataset repo ID.")],
    path_in_repo: Annotated[
        str | None,
        typer.Option(help="HF repo subdirectory. Defaults to the local dataset directory name."),
    ] = None,
    revision: Annotated[str | None, typer.Option(help="Target branch or revision.")] = None,
    private: Annotated[
        bool,
        typer.Option(help="Create the dataset repo as private if needed."),
    ] = False,
    create_pr: Annotated[
        bool,
        typer.Option(help="Open a pull request instead of committing."),
    ] = False,
    token: Annotated[
        str | None,
        typer.Option(envvar="HF_TOKEN", help="Hugging Face token. Defaults to HF_TOKEN."),
    ] = None,
    commit_message: Annotated[str | None, typer.Option()] = None,
    create_repo: Annotated[
        bool,
        typer.Option(help="Create the Hugging Face dataset repo if it does not exist."),
    ] = True,
) -> None:
    """Upload a generated dataset directory into a HF dataset repo subdirectory."""
    result = upload_dataset_to_hub(
        dataset_dir,
        repo_id,
        path_in_repo=path_in_repo,
        private=private,
        revision=revision,
        create_pr=create_pr,
        token=token,
        commit_message=commit_message,
        create_repo=create_repo,
    )
    console.print(f"Uploaded {result['dataset_dir']} to {repo_id}/{result['path_in_repo']}")
    if result["commit_url"]:
        console.print(f"Commit: {result['commit_url']}")


@run_app.command("goal1")
def run_goal1_command(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Run Goal 1 behavioral evaluation."""
    experiment_config = load_experiment_config(config)
    run_dir = run_goal1(experiment_config)
    console.print(f"Wrote run artifacts to {run_dir}")


@run_app.command("goal2")
def run_goal2_command(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Run Goal 2 activation extraction."""
    goal2_config = load_goal2_config(config)
    run_dir = run_goal2(goal2_config)
    console.print(f"Wrote activation artifacts to {run_dir}")


@figures_app.command("goal1")
def figures_goal1(
    run_id: Annotated[str, typer.Option(help="Run directory name under runs/, or a full path.")],
    runs_dir: Annotated[Path, typer.Option()] = Path("runs"),
    output_dir: Annotated[Path | None, typer.Option()] = None,
    include_token_limit_hits: Annotated[
        bool,
        typer.Option(help="Include generations that exhausted the token budget."),
    ] = False,
    token_budget_by_digit_length: Annotated[
        bool,
        typer.Option(
            "--token-budget-by-digit-length/--no-token-budget-by-digit-length",
            help="Facet token-budget curves by digit length.",
        ),
    ] = True,
) -> None:
    """Generate Goal 1 figures from saved run artifacts."""
    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = runs_dir / run_id
    paths = make_goal1_figures(
        run_dir,
        output_dir,
        include_token_limit_hits=include_token_limit_hits,
        token_budget_by_digit_length=token_budget_by_digit_length,
    )
    for path in paths:
        console.print(f"Wrote {path}")


@inspect_app.command("tokenizer")
def inspect_tokenizer_command(
    model_id: Annotated[str, typer.Option()] = "allenai/Olmo-3-7B-Think",
    revision: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Inspect a Hugging Face tokenizer configuration."""
    info = inspect_tokenizer(model_id, revision)
    console.print(JSON.from_data(info))
