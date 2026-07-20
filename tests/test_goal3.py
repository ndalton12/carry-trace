from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from carry_trace.cli import app
from carry_trace.config import Goal3RunConfig
from carry_trace.enums import ActivationLocation
from carry_trace.goal3 import (
    ResidualIntervention,
    _completion_audit_rows,
    _fit_one_direction,
    _generate_replay_batch,
    _off_target_digits_preserved,
    _orthogonal_direction,
    _replay_context_ids,
    _residual_control_contrast_rows,
    _residual_executions,
    _run_replay_model,
    _score_replay_batch,
    _score_sequences,
)
from carry_trace.schemas import Goal3ReplayCase, Goal3ReplayPrefix


def test_goal3_run_config_requires_hf_runner(tmp_path) -> None:
    """Verify Goal 3 execution rejects runners without residual-hook support."""
    with pytest.raises(ValueError, match="runner.kind=hf"):
        Goal3RunConfig(name="bad", dataset_bundle_dir=tmp_path)


def test_goal3_run_cli_loads_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the Goal 3 run command loads its config and reports the run path."""
    config_path = tmp_path / "goal3.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: goal3-test",
                f"dataset_bundle_dir: {tmp_path}",
                "runner:",
                "  kind: hf",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    monkeypatch.setattr("carry_trace.cli.run_goal3", lambda config: run_dir)

    result = CliRunner().invoke(app, ["run", "goal3", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Wrote Goal 3 artifacts" in result.output
    assert run_dir.name in result.output


def test_fit_one_direction_returns_positive_class_gap(tmp_path) -> None:
    """Verify fitted directions point from carry zero toward carry one."""
    config = Goal3RunConfig(
        name="test",
        dataset_bundle_dir=tmp_path,
        runner={"kind": "hf"},
        residual={"min_train_examples": 2},
    )
    rows = [
        {"x": [-2.0, 0.0], "y": 0, "metadata": {"split": "train_probe"}},
        {"x": [-1.0, 0.0], "y": 0, "metadata": {"split": "train_probe"}},
        {"x": [1.0, 0.0], "y": 1, "metadata": {"split": "train_probe"}},
        {"x": [2.0, 0.0], "y": 1, "metadata": {"split": "train_probe"}},
    ]

    direction, metadata = _fit_one_direction(
        config,
        ("model", "incoming_carry", "cot_end", "16", 1),
        "direction-id",
        rows,
    )

    assert direction[0] > 0
    assert metadata["class_mean_gap"] > 0


def test_pilot_variants_have_unique_ids_and_orthogonal_controls(tmp_path) -> None:
    """Verify pilot scales and controls produce unique, norm-matched assignments."""
    torch = pytest.importorskip("torch")
    config = Goal3RunConfig(
        name="pilot",
        dataset_bundle_dir=tmp_path,
        runner={"kind": "hf"},
        replay={"enabled": False},
        residual={
            "intervention_scales": [0.5, 1.0, 2.0],
            "control_directions": ["carry", "orthogonal"],
            "decode_locations": [],
            "decode_layers": [],
        },
    )
    case = SimpleNamespace(id="case", layer_index=16)

    executions = _residual_executions(config, [case])
    carry = torch.tensor([1.0, 0.0, 0.0])
    orthogonal = _orthogonal_direction(carry, "direction", seed=13)

    assert len(executions) == 6
    assert len({execution.id for execution in executions}) == 6
    assert torch.linalg.vector_norm(orthogonal).item() == pytest.approx(1.0)
    assert torch.dot(carry, orthogonal).item() == pytest.approx(0.0, abs=1e-6)


def test_clamp_pilot_expands_sites_and_orthogonal_controls(tmp_path) -> None:
    """Verify clamp pilots cross two sites with three orthogonal controls."""
    config = Goal3RunConfig(
        name="clamp-pilot",
        dataset_bundle_dir=tmp_path,
        runner={"kind": "hf"},
        replay={"enabled": False},
        residual={
            "intervention_mode": "projection_clamp",
            "intervention_sites": ["prefix_boundary", "answer_cue"],
            "control_directions": ["carry", "orthogonal"],
            "orthogonal_control_count": 3,
            "decode_locations": [],
            "decode_layers": [],
        },
    )
    case = SimpleNamespace(id="case", layer_index=16)

    executions = _residual_executions(config, [case])

    assert len(executions) == 8
    assert len({execution.id for execution in executions}) == 8
    assert {execution.control_direction for execution in executions} == {
        "carry",
        "orthogonal_1",
        "orthogonal_2",
        "orthogonal_3",
    }


def test_residual_control_contrasts_pair_same_case_and_scale() -> None:
    """Verify pilot contrasts subtract orthogonal effects from carry effects."""
    identity = {
        "case_id": "case",
        "example_id": "example",
        "problem_id": "problem",
        "n_digits": 2,
        "model_name": "model",
        "target": "incoming_carry",
        "target_column_lsd": 1,
        "affected_output_column_lsd": 1,
        "factual_carry": 0,
        "counterfactual_carry": 1,
        "location_kind": "cot_end",
        "layer_index": 16,
        "direction_id": "direction",
        "intervention_scale": 1.0,
    }
    scores = [
        {
            **identity,
            "control_direction": "carry",
            "counterfactual_preference_shift": 1.25,
        },
        {
            **identity,
            "control_direction": "orthogonal_1",
            "counterfactual_preference_shift": 0.0,
        },
        {
            **identity,
            "control_direction": "orthogonal_2",
            "counterfactual_preference_shift": 0.25,
        },
        {
            **identity,
            "control_direction": "orthogonal_3",
            "counterfactual_preference_shift": 0.5,
        },
    ]

    contrast = _residual_control_contrast_rows(scores)[0]

    assert contrast["carry_minus_orthogonal_shift"] == pytest.approx(1.0)
    assert contrast["orthogonal_control_n"] == 3
    assert contrast["carry_shift_positive"] is True


def test_run_replay_model_batches_cases_without_layer_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify replay orchestration batches cases without residual-layer grouping."""
    config = Goal3RunConfig(
        name="test",
        dataset_bundle_dir=tmp_path,
        runner={"kind": "hf", "batch_size": 2},
    )
    case = Goal3ReplayCase(
        id="case",
        schema_version="goal3.natural_cot.v2",
        replay_prefix_id="prefix",
        source_goal2_run_id="run",
        example_id="example",
        problem_id="problem",
        split="test_probe",
        n_digits=2,
        replay_kind="no_reasoning",
        receiver_model_name="model",
        location_kind="prompt_final",
        assistant_prefix_token_ids=[],
        assistant_prefix="",
        expected_output="83",
        prompt="What is 19 + 64?",
        messages=[{"role": "user", "content": "What is 19 + 64?"}],
    )
    runner = SimpleNamespace(model_spec=SimpleNamespace(name="model"))
    monkeypatch.setattr(
        "carry_trace.goal3._score_replay_batch",
        lambda config, runner, cases: [{"id": item.id} for item in cases],
    )
    monkeypatch.setattr(
        "carry_trace.goal3._generate_replay_batch",
        lambda config, runner, cases: [{"id": item.id} for item in cases],
    )

    _run_replay_model(config, tmp_path, runner, [case])

    assert (tmp_path / "replay_scores.jsonl").exists()
    assert (tmp_path / "replay_generations.jsonl").exists()


def test_replay_context_ids_accepts_mapping_chat_template_output() -> None:
    """Verify replay contexts accept a BatchEncoding-like mapping."""

    class MappingChatTokenizer:
        """Return one nested input-ID row from the chat template."""

        chat_template = "dummy"

        def apply_chat_template(self, messages, **kwargs):
            """Return a mapping matching Hugging Face tokenizer output."""
            del messages, kwargs
            return {"input_ids": [[11, 12]]}

    context = _replay_context_ids(
        MappingChatTokenizer(),
        [{"role": "user", "content": "What is 19 + 64?"}],
        [13, 14],
    )

    assert context == [11, 12, 13, 14]


def test_cot_end_replay_continues_directly_into_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cot-end scoring and decoding omit the synthetic answer cue and space."""

    class CharacterTokenizer:
        """Encode text as character code points for transparent assertions."""

        chat_template = None

        def encode(self, text, add_special_tokens=False):
            """Return deterministic character-level token IDs."""
            del add_special_tokens
            return [ord(character) for character in text]

    config = Goal3RunConfig(
        name="test",
        dataset_bundle_dir=tmp_path,
        runner={"kind": "hf"},
        residual={"enabled": False},
    )
    prompt_case = Goal3ReplayCase(
        id="prompt",
        schema_version="goal3.natural_cot.v2",
        replay_prefix_id="prompt-prefix",
        source_goal2_run_id="run",
        example_id="example",
        problem_id="problem",
        split="test_probe",
        n_digits=2,
        replay_kind="no_reasoning",
        receiver_model_name="model",
        location_kind="prompt_final",
        assistant_prefix_token_ids=[],
        assistant_prefix="",
        expected_output="83",
        prompt="What is 19 + 64?",
        messages=[{"role": "user", "content": "What is 19 + 64?"}],
    )
    cot_end_case = prompt_case.model_copy(
        update={
            "id": "cot-end",
            "replay_prefix_id": "cot-prefix",
            "replay_kind": "self",
            "source_model_name": "model",
            "location_kind": ActivationLocation.COT_END,
            "assistant_prefix_token_ids": [91],
            "assistant_prefix": "\\boxed{",
        }
    )
    tokenizer = CharacterTokenizer()
    runner = SimpleNamespace(tokenizer=tokenizer, model=object())
    scored_sequences = []
    generated_contexts = []
    monkeypatch.setattr(
        "carry_trace.goal3._score_sequences",
        lambda model, sequences: scored_sequences.extend(sequences) or [0.0] * len(sequences),
    )
    monkeypatch.setattr(
        "carry_trace.goal3._generate_contexts",
        lambda runner, contexts, generation, expected_output_lengths: (
            generated_contexts.extend(contexts) or ["83"] * len(contexts)
        ),
    )

    _score_replay_batch(config, runner, [prompt_case, cot_end_case])
    _generate_replay_batch(config, runner, [prompt_case, cot_end_case])

    prompt_context = tokenizer.encode(prompt_case.prompt)
    cue = tokenizer.encode(config.replay.answer_cue)
    assert scored_sequences[0][0] == prompt_context + cue + tokenizer.encode(" 83")
    assert scored_sequences[1][0] == prompt_context + [91] + tokenizer.encode("83")
    assert generated_contexts[0] == prompt_context + cue
    assert generated_contexts[1] == prompt_context + [91]


def test_score_sequences_applies_residual_shift() -> None:
    """Verify intervention hooks alter the requested answer-token probability."""
    torch = pytest.importorskip("torch")

    class Layer(torch.nn.Module):
        """Return hidden states unchanged."""

        def forward(self, hidden):
            """Pass hidden states through unchanged."""
            return hidden

    class Backbone(torch.nn.Module):
        """Provide a minimal decoder backbone."""

        def __init__(self):
            """Create deterministic embeddings and one decoder layer."""
            super().__init__()
            self.embed = torch.nn.Embedding(5, 2)
            self.layers = torch.nn.ModuleList([Layer()])
            with torch.no_grad():
                self.embed.weight.zero_()

        def forward(self, input_ids, **kwargs):
            """Return final hidden states in a causal-LM-like result."""
            del kwargs
            hidden = self.embed(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return SimpleNamespace(last_hidden_state=hidden)

    class TinyLM(torch.nn.Module):
        """Provide the model interface used by Goal 3 scoring."""

        def __init__(self):
            """Create a backbone and deterministic language-model head."""
            super().__init__()
            self.model = Backbone()
            self.lm_head = torch.nn.Linear(2, 5, bias=False)
            self.config = SimpleNamespace(pad_token_id=0)
            with torch.no_grad():
                self.lm_head.weight.zero_()
                self.lm_head.weight[3, 0] = 2.0

        def get_input_embeddings(self):
            """Return the model token embedding table."""
            return self.model.embed

    model = TinyLM()
    sequence = [([1, 2, 3], [2])]

    baseline = _score_sequences(model, sequence)[0]
    intervened = _score_sequences(
        model,
        sequence,
        layer_index=0,
        shifts=[torch.tensor([1.0, 0.0])],
        intervention_positions=[1],
    )[0]
    diagnostics = [None]
    clamped = _score_sequences(
        model,
        sequence,
        layer_index=0,
        interventions=[
            ResidualIntervention(
                carry_direction=torch.tensor([1.0, 0.0]),
                applied_direction=torch.tensor([1.0, 0.0]),
                mode="projection_clamp",
                scale=1.0,
                target_projection=1.0,
            )
        ],
        intervention_positions=[1],
        intervention_diagnostics=diagnostics,
    )[0]

    assert intervened > baseline
    assert clamped > baseline
    assert diagnostics[0]["applied_carry_projection_after"] == pytest.approx(1.0)
    assert diagnostics[0]["realized_to_requested_norm_ratio"] == pytest.approx(1.0)

    with torch.no_grad():
        model.model.embed.weight[3, 0] = 5.0
    cue_diagnostics = [None]
    _score_sequences(
        model,
        sequence,
        layer_index=0,
        interventions=[
            ResidualIntervention(
                carry_direction=torch.tensor([1.0, 0.0]),
                applied_direction=torch.tensor([1.0, 0.0]),
                mode="projection_clamp",
                scale=1.0,
                target_projection=1.0,
            )
        ],
        intervention_positions=[2],
        intervention_calibration_positions=[1],
        intervention_diagnostics=cue_diagnostics,
    )

    assert cue_diagnostics[0]["requested_applied_projection_delta"] == pytest.approx(1.0)
    assert cue_diagnostics[0]["applied_carry_projection_after"] == pytest.approx(6.0)


def test_off_target_digit_preservation_uses_lsd_columns() -> None:
    """Verify off-target checks compare least-significant-first columns."""
    assert _off_target_digits_preserved("173", "183", [0, 2]) is True
    assert _off_target_digits_preserved("174", "183", [0, 2]) is False
    assert _off_target_digits_preserved(None, "183", [0, 2]) is None


def test_completion_audit_flags_reasoning_after_answer() -> None:
    """Verify completion audits identify substantive text after a numeric answer."""
    prefix = Goal3ReplayPrefix(
        id="prefix",
        schema_version="goal3.natural_cot.v2",
        source_goal2_run_id="run",
        example_id="example",
        problem_id="problem",
        split="test_probe",
        n_digits=2,
        source_model_name="model",
        source_model_id="model-id",
        location_kind="cot_end",
        prefix_token_source="recorded",
        assistant_prefix_token_ids=[1],
        assistant_prefix="Answer: 83 but let me verify",
        decoded_output="Answer: 83 but let me verify.",
        parsed_answer="83",
        expected_output="83",
        prompt="What is 19 + 64?",
        messages=[{"role": "user", "content": "What is 19 + 64?"}],
        metadata={"source_answer_correct": True},
    )

    audit = _completion_audit_rows([prefix])[0]

    assert audit["trailing_has_alphanumeric"] is True
    assert audit["clean_terminal_answer"] is False
