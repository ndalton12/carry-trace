from carry_trace.prompts import format_operand, render_prompt


def test_delimited_digit_format_is_prompt_only() -> None:
    problem = {"a": "4879", "b": "2568"}
    prompt, template_id, messages, prompt_a, prompt_b = render_prompt(
        problem,
        "answer_only",
        digit_format="delimited",
    )
    assert prompt_a == "4|8|7|9"
    assert prompt_b == "2|5|6|8"
    assert "4|8|7|9 + 2|5|6|8" in prompt
    assert template_id == "answer_only_delimited_v1"
    assert messages == [{"role": "user", "content": prompt}]


def test_plain_digit_format_keeps_operand_text() -> None:
    assert format_operand("4879", "plain") == "4879"
