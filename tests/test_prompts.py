import pytest

from carry_trace.enums import AnswerFormat, DigitFormat, PromptMode
from carry_trace.prompts import format_operand, render_prompt


def print_prompt(label: str, prompt: str) -> None:
    """Print a rendered prompt for manual confirmation during pytest -s runs."""
    print(f"\n--- {label} ---\n{prompt}\n--- end prompt ---")


@pytest.mark.parametrize(
    ("prompt_mode", "expected_prompt"),
    [
        (
            PromptMode.ANSWER_ONLY,
            "What is 4879 + 2568? Give only the answer.",
        ),
        (
            PromptMode.FREE_COT,
            "What is 4879 + 2568? Solve the problem step by step.",
        ),
        (
            PromptMode.LENGTH_CONTROLLED_COT,
            "What is 4879 + 2568? Solve this in exactly four short steps.",
        ),
        (
            PromptMode.STRUCTURED_COLUMN_COT,
            "What is 4879 + 2568? Solve column by column from right to left. "
            "For each column, state the digit and carry.",
        ),
    ],
)
def test_standard_base10_prompt_mode_happy_paths(
    prompt_mode: PromptMode,
    expected_prompt: str,
) -> None:
    problem = {"a": "4879", "b": "2568", "answer": "7447"}
    prompt, template_id, messages, prompt_a, prompt_b, expected_output = render_prompt(
        problem,
        prompt_mode,
    )
    print_prompt(f"base10 standard {prompt_mode.value}", prompt)
    assert prompt == expected_prompt
    if prompt_mode != PromptMode.ANSWER_ONLY:
        assert "give only the answer" not in prompt.lower()
    assert prompt_a == "4879"
    assert prompt_b == "2568"
    assert template_id == f"{prompt_mode.value}_standard_standard_v1"
    assert expected_output == "7447"
    assert messages == [{"role": "user", "content": prompt}]


def test_standard_digit_format_keeps_operand_text() -> None:
    assert format_operand("4879", "standard") == "4879"


def test_delimited_digit_format_adds_delimiters_to_prompt_operands_only() -> None:
    problem = {"a": "4879", "b": "2568", "answer": "7447"}
    prompt, _, _, prompt_a, prompt_b, expected_output = render_prompt(
        problem,
        "answer_only",
        digit_format="delimited",
    )
    print_prompt("delimited operand formatting", prompt)
    assert prompt_a == "4|8|7|9"
    assert prompt_b == "2|5|6|8"
    assert expected_output == "7447"


@pytest.mark.parametrize("prompt_mode", list(PromptMode))
@pytest.mark.parametrize("digit_format", list(DigitFormat))
@pytest.mark.parametrize("answer_format", list(AnswerFormat))
def test_base10_prompt_mode_format_cross_product(
    prompt_mode: PromptMode,
    digit_format: DigitFormat,
    answer_format: AnswerFormat,
) -> None:
    problem = {"a": "1234", "b": "5678", "answer": "6912"}
    prompt, template_id, messages, prompt_a, prompt_b, actual_expected_output = render_prompt(
        problem,
        prompt_mode,
        digit_format=digit_format,
        answer_format=answer_format,
    )
    print_prompt(
        f"base10 {prompt_mode.value} {digit_format.value} {answer_format.value}",
        prompt,
    )

    if digit_format == DigitFormat.STANDARD:
        assert prompt_a == "1234"
        assert prompt_b == "5678"
        assert "1|2|3 means 123" not in prompt
        assert "without delimiters" not in prompt
    else:
        assert prompt_a == "1|2|3|4"
        assert prompt_b == "5|6|7|8"
        assert "The | symbol separates digits inside a number" in prompt
        assert "What is 1|2|3|4 + 5|6|7|8?" in prompt
        assert prompt.count("without delimiters") == 1

    if answer_format == AnswerFormat.STANDARD:
        assert actual_expected_output == "6912"
        assert "least significant digit format" not in prompt
        assert "no separators" not in prompt
    else:
        assert actual_expected_output == "2196"
        assert "least significant digit format" in prompt
        assert "right to left with no separators" in prompt

    if prompt_mode == PromptMode.ANSWER_ONLY:
        if answer_format == AnswerFormat.STANDARD:
            assert "Give only the answer." in prompt
        else:
            assert "Solve the problem step by step" not in prompt
    else:
        assert "give only the answer" not in prompt.lower()

    if prompt_mode == PromptMode.FREE_COT:
        assert "Solve the problem step by step." in prompt
    if prompt_mode == PromptMode.LENGTH_CONTROLLED_COT:
        assert "Solve this in exactly four short steps." in prompt
    if prompt_mode == PromptMode.STRUCTURED_COLUMN_COT:
        assert "Solve column by column from right to left." in prompt
        assert "For each column, state the digit and carry." in prompt

    assert "base 10" not in prompt
    assert template_id == f"{prompt_mode.value}_{digit_format.value}_{answer_format.value}_v1"
    assert messages == [{"role": "user", "content": prompt}]


@pytest.mark.parametrize(
    (
        "label",
        "problem",
        "prompt_mode",
        "answer_format",
        "expected_prompt_parts",
        "expected_output",
    ),
    [
        (
            "base7 standard answer_only",
            {"a": "1234", "b": "456", "answer": "2023", "base": 7},
            PromptMode.ANSWER_ONLY,
            AnswerFormat.STANDARD,
            [
                "In base 7, what is 1234 + 456?",
                "Give only the answer in base 7.",
            ],
            "2023",
        ),
        (
            "base7 standard free_cot",
            {"a": "1234", "b": "456", "answer": "2023", "base": 7},
            PromptMode.FREE_COT,
            AnswerFormat.STANDARD,
            [
                "In base 7, what is 1234 + 456?",
                "Solve the problem step by step.",
                "Use base 7 for your answer.",
            ],
            "2023",
        ),
        (
            "base7 lsd free_cot",
            {"a": "6666", "b": "1", "answer": "10000", "base": 7},
            PromptMode.FREE_COT,
            AnswerFormat.LSD,
            [
                "In base 7, what is 6666 + 1?",
                "Solve the problem step by step.",
                "Use least significant digit format for the final answer in base 7: "
                "write digits from right to left with no separators.",
            ],
            "00001",
        ),
    ],
)
def test_base7_prompt_format_conditions(
    label: str,
    problem: dict[str, object],
    prompt_mode: PromptMode,
    answer_format: AnswerFormat,
    expected_prompt_parts: list[str],
    expected_output: str,
) -> None:
    prompt, template_id, messages, _, _, actual_expected_output = render_prompt(
        problem,
        prompt_mode,
        answer_format=answer_format,
    )
    print_prompt(label, prompt)
    for expected_part in expected_prompt_parts:
        assert expected_part in prompt
    if prompt_mode != PromptMode.ANSWER_ONLY:
        assert "give only the answer" not in prompt.lower()
    assert "delimiters" not in prompt
    assert template_id == f"{prompt_mode.value}_standard_{answer_format.value}_v1"
    assert actual_expected_output == expected_output
    assert messages == [{"role": "user", "content": prompt}]
