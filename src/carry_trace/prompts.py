"""Prompt templates for Goal 1 behavioral experiments."""

from carry_trace.enums import AnswerFormat, DigitFormat, PromptMode


def render_prompt(
    problem: dict[str, object],
    prompt_mode: PromptMode | str,
    digit_format: DigitFormat | str = DigitFormat.STANDARD,
    digit_delimiter: str = "|",
    answer_format: AnswerFormat | str = AnswerFormat.STANDARD,
) -> tuple[str, str, list[dict[str, str]], str, str, str]:
    """Render a prompt and expected output for one arithmetic example."""
    prompt_mode = PromptMode(prompt_mode)
    digit_format = DigitFormat(digit_format)
    answer_format = AnswerFormat(answer_format)

    base = int(problem.get("base", 10))
    a = format_operand(str(problem["a"]), digit_format, digit_delimiter)
    b = format_operand(str(problem["b"]), digit_format, digit_delimiter)
    prefix = digit_format_instruction(digit_format, digit_delimiter)
    question = addition_question(a, b, base)
    answer_instruction = answer_format_instruction(answer_format, base)
    if prompt_mode == PromptMode.ANSWER_ONLY:
        content = f"{prefix}{question} {answer_instruction}"
    elif prompt_mode == PromptMode.FREE_COT:
        content = (
            f"{prefix}{question} Solve the problem step by step, "
            f"then {answer_instruction.lower()}"
        )
    elif prompt_mode == PromptMode.LENGTH_CONTROLLED_COT:
        content = (
            f"{prefix}{question} Solve this in exactly four short steps, "
            f"then {answer_instruction.lower()}"
        )
    else:
        content = (
            f"{prefix}{question} Solve column by column from right to left. "
            f"For each column, state the digit and carry, then {answer_instruction.lower()}"
        )

    messages = [{"role": "user", "content": content}]
    template_id = f"{prompt_mode.value}_{digit_format.value}_{answer_format.value}_v1"
    expected_output = format_expected_output(
        str(problem["answer"]),
        answer_format,
    )
    return content, template_id, messages, a, b, expected_output


def addition_question(a: str, b: str, base: int = 10) -> str:
    """Return the addition question text, including base wording when needed."""
    if base == 10:
        return f"What is {a} + {b}?"
    return f"In base {base}, what is {a} + {b}?"


def format_operand(
    value: str,
    digit_format: DigitFormat | str,
    digit_delimiter: str = "|",
) -> str:
    """Format an operand for display in a rendered prompt."""
    digit_format = DigitFormat(digit_format)
    if digit_format == DigitFormat.STANDARD:
        return value
    if digit_format == DigitFormat.DELIMITED:
        return digit_delimiter.join(value)
    raise ValueError(f"unknown digit format {digit_format!r}")


def digit_format_instruction(
    digit_format: DigitFormat | str,
    digit_delimiter: str = "|",
) -> str:
    """Return explanatory prompt text for the operand digit format."""
    digit_format = DigitFormat(digit_format)
    if digit_format == DigitFormat.STANDARD:
        return ""
    if digit_format == DigitFormat.DELIMITED:
        return (
            f"The {digit_delimiter} symbol separates digits inside a number; "
            f"for example, 1{digit_delimiter}2{digit_delimiter}3 means 123. "
        )
    raise ValueError(f"unknown digit format {digit_format!r}")


def answer_format_instruction(answer_format: AnswerFormat | str, base: int = 10) -> str:
    """Return the prompt instruction for the requested answer format."""
    answer_format = AnswerFormat(answer_format)
    base_phrase = "" if base == 10 else f" in base {base}"
    if answer_format == AnswerFormat.STANDARD:
        return f"Give only the answer{base_phrase}; use standard formatting without delimiters."
    if answer_format == AnswerFormat.LSD:
        digit_phrase = "answer digits" if base == 10 else f"answer digits in base {base}"
        return (
            f"Give only the {digit_phrase} from right to left with no separators; "
            "for example, if the normal answer is 6912, write 2196. "
            "This is known as the least significant digit format."
        )
    raise ValueError(f"unknown answer format {answer_format!r}")


def format_expected_output(
    answer: str,
    answer_format: AnswerFormat | str,
) -> str:
    """Format the expected model output for the requested answer format."""
    answer_format = AnswerFormat(answer_format)
    if answer_format == AnswerFormat.STANDARD:
        return answer
    if answer_format == AnswerFormat.LSD:
        return "".join(reversed(answer))
    raise ValueError(f"unknown answer format {answer_format!r}")
