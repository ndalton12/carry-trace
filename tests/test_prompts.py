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
    assert "1|2|3 means 123" in prompt
    assert "4|8|7|9 + 2|5|6|8" in prompt
    assert template_id == "answer_only_delimited_standard_v1"
    assert expected_output == "7447"
    assert messages == [{"role": "user", "content": prompt}]


def test_standard_digit_format_keeps_operand_text() -> None:
    assert format_operand("4879", "standard") == "4879"


def test_lsd_answer_format_sets_expected_output() -> None:
    problem = {"a": "1234", "b": "5678", "answer": "6912"}
    prompt, _, _, _, _, expected_output = render_prompt(
        problem,
        "answer_only",
        answer_format="lsd",
    )
    assert "right to left" in prompt
    assert "no separators" in prompt
    assert expected_output == "2196"


def test_non_decimal_standard_prompt_specifies_answer_base() -> None:
    problem = {"a": "1234", "b": "456", "answer": "2023", "base": 7}
    prompt, _, _, _, _, expected_output = render_prompt(problem, "answer_only")
    assert "In base 7, what is 1234 + 456?" in prompt
    assert "Give only the answer in base 7" in prompt
    assert expected_output == "2023"


def test_non_decimal_lsd_prompt_specifies_answer_base() -> None:
    problem = {"a": "6666", "b": "1", "answer": "10000", "base": 7}
    prompt, _, _, _, _, expected_output = render_prompt(
        problem,
        "free_cot",
        answer_format="lsd",
    )
    assert "In base 7, what is 6666 + 1?" in prompt
    assert "answer digits in base 7 from right to left" in prompt
    assert expected_output == "00001"
