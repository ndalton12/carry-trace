from carry_trace.prompts import format_operand, render_prompt


def test_delimited_digit_format_is_prompt_only() -> None:
    problem = {"a": "4879", "b": "2568", "answer": "7447"}
    prompt, template_id, messages, prompt_a, prompt_b, expected_output = render_prompt(
        problem,
        "answer_only",
        digit_format="delimited",
    )
    assert prompt_a == "4|8|7|9"
    assert prompt_b == "2|5|6|8"
    assert "4|8|7|9 + 2|5|6|8" in prompt
    assert template_id == "answer_only_delimited_standard_v1"
    assert expected_output == "7447"
    assert messages == [{"role": "user", "content": prompt}]


def test_plain_digit_format_keeps_operand_text() -> None:
    assert format_operand("4879", "plain") == "4879"


def test_lsd_delimited_answer_format_sets_expected_output() -> None:
    problem = {"a": "1234", "b": "5678", "answer": "6912"}
    prompt, _, _, _, _, expected_output = render_prompt(
        problem,
        "answer_only",
        answer_format="lsd_delimited",
    )
    assert "right to left" in prompt
    assert expected_output == "2|1|9|6"
