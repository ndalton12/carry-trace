"""Command line interface for carry-trace."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.json import JSON

from carry_trace.config import load_dataset_config, load_experiment_config
from carry_trace.datasets import generate_dataset
from carry_trace.figures import make_goal1_figures
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


@run_app.command("goal1")
def run_goal1_command(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Run Goal 1 behavioral evaluation."""
    experiment_config = load_experiment_config(config)
    run_dir = run_goal1(experiment_config)
    console.print(f"Wrote run artifacts to {run_dir}")


@figures_app.command("goal1")
def figures_goal1(
    run_id: Annotated[str, typer.Option(help="Run directory name under runs/, or a full path.")],
    runs_dir: Annotated[Path, typer.Option()] = Path("runs"),
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Generate Goal 1 figures from saved run artifacts."""
    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = runs_dir / run_id
    paths = make_goal1_figures(run_dir, output_dir)
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
