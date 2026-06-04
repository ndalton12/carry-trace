"""Prompt templates for Goal 1 behavioral experiments."""

from carry_trace.enums import AnswerFormat, DigitFormat, PromptMode


def render_prompt(
    problem: dict[str, object],
    prompt_mode: PromptMode | str,
    digit_format: DigitFormat | str = DigitFormat.PLAIN,
    digit_delimiter: str = "|",
    answer_format: AnswerFormat | str = AnswerFormat.STANDARD,
    answer_delimiter: str = "|",
) -> tuple[str, str, list[dict[str, str]], str, str, str]:
    """Render a prompt and expected output for one arithmetic example."""
    prompt_mode = PromptMode(prompt_mode)
    digit_format = DigitFormat(digit_format)
    answer_format = AnswerFormat(answer_format)

    a = format_operand(str(problem["a"]), digit_format, digit_delimiter)
    b = format_operand(str(problem["b"]), digit_format, digit_delimiter)
    answer_instruction = answer_format_instruction(answer_format, answer_delimiter)
    if prompt_mode == PromptMode.ANSWER_ONLY:
        content = f"What is {a} + {b}? {answer_instruction}"
    elif prompt_mode == PromptMode.FREE_COT:
        content = f"What is {a} + {b}? Think step by step, then {answer_instruction.lower()}"
    elif prompt_mode == PromptMode.LENGTH_CONTROLLED_COT:
        content = (
            f"What is {a} + {b}? Solve this in exactly four short steps, "
            f"then {answer_instruction.lower()}"
        )
    else:
        content = (
            f"What is {a} + {b}? Solve column by column from right to left. "
            f"For each column, state the digit and carry, then {answer_instruction.lower()}"
        )

    messages = [{"role": "user", "content": content}]
    template_id = f"{prompt_mode.value}_{digit_format.value}_{answer_format.value}_v1"
    expected_output = format_expected_output(
        str(problem["answer"]),
        answer_format,
        answer_delimiter,
    )
    return content, template_id, messages, a, b, expected_output


def format_operand(
    value: str,
    digit_format: DigitFormat | str,
    digit_delimiter: str = "|",
) -> str:
    """Format an operand for display in a rendered prompt."""
    digit_format = DigitFormat(digit_format)
    if digit_format == DigitFormat.PLAIN:
        return value
    if digit_format == DigitFormat.DELIMITED:
        return digit_delimiter.join(value)
    raise ValueError(f"unknown digit format {digit_format!r}")


def answer_format_instruction(
    answer_format: AnswerFormat | str,
    answer_delimiter: str = "|",
) -> str:
    """Return the prompt instruction for the requested answer format."""
    answer_format = AnswerFormat(answer_format)
    if answer_format == AnswerFormat.STANDARD:
        return "Give only the answer."
    if answer_format == AnswerFormat.LSD_DELIMITED:
        return (
            "Give only the answer digits from right to left, "
            f"separated by {answer_delimiter}."
        )
    raise ValueError(f"unknown answer format {answer_format!r}")


def format_expected_output(
    answer: str,
    answer_format: AnswerFormat | str,
    answer_delimiter: str = "|",
) -> str:
    """Format the expected model output for the requested answer format."""
    answer_format = AnswerFormat(answer_format)
    if answer_format == AnswerFormat.STANDARD:
        return answer
    if answer_format == AnswerFormat.LSD_DELIMITED:
        return answer_delimiter.join(reversed(answer))
    raise ValueError(f"unknown answer format {answer_format!r}")
