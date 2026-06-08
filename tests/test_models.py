import pytest
from pydantic import ValidationError

from carry_trace.config import GenerationParams
from carry_trace.models import HuggingFaceModelRunner, has_unclosed_thinking


def test_has_unclosed_thinking_detects_open_think_block() -> None:
    """Check that unclosed thinking is detected only when the close marker is absent."""
    assert has_unclosed_thinking("<think>working") is True
    assert has_unclosed_thinking("<think>working</think>\nAnswer: 5") is False
    assert has_unclosed_thinking("Answer: 5") is False


def test_hf_generate_kwargs_excludes_thinking_cap_fields() -> None:
    """Check that local thinking-cap fields are not passed to Transformers generate."""
    runner = object.__new__(HuggingFaceModelRunner)
    runner.generation = GenerationParams(
        max_new_tokens=256,
        thinking_final_answer_tokens=100,
        force_close_thinking=True,
    )
    assert runner._hf_generate_kwargs() == {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }


def test_generation_params_rejects_invalid_thinking_cap() -> None:
    """Check that thinking-cap configs must reserve a valid final-answer budget."""
    with pytest.raises(ValidationError):
        GenerationParams(max_new_tokens=100, force_close_thinking=True)
    with pytest.raises(ValidationError):
        GenerationParams(
            max_new_tokens=100,
            thinking_final_answer_tokens=100,
            force_close_thinking=True,
        )
