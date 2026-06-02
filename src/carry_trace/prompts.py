"""Prompt templates for Goal 1 behavioral experiments."""

from carry_trace.enums import DigitFormat, PromptMode


def render_prompt(
    problem: dict[str, object],
    prompt_mode: PromptMode | str,
    digit_format: DigitFormat | str = DigitFormat.PLAIN,
    digit_delimiter: str = "|",
) -> tuple[str, str, list[dict[str, str]], str, str]:
    prompt_mode = PromptMode(prompt_mode)
    digit_format = DigitFormat(digit_format)

    a = format_operand(str(problem["a"]), digit_format, digit_delimiter)
    b = format_operand(str(problem["b"]), digit_format, digit_delimiter)
    if prompt_mode == PromptMode.ANSWER_ONLY:
        content = f"What is {a} + {b}? Give only the answer."
    elif prompt_mode == PromptMode.FREE_COT:
        content = f"What is {a} + {b}? Think step by step, then give the answer."
    elif prompt_mode == PromptMode.LENGTH_CONTROLLED_COT:
        content = (
            f"What is {a} + {b}? Solve this in exactly four short steps, then give the answer."
        )
    else:
        content = (
            f"What is {a} + {b}? Solve column by column from right to left. "
            "For each column, state the digit and carry, then give the final answer."
        )

    messages = [{"role": "user", "content": content}]
    return content, f"{prompt_mode.value}_{digit_format.value}_v1", messages, a, b


def format_operand(
    value: str,
    digit_format: DigitFormat | str,
    digit_delimiter: str = "|",
) -> str:
    digit_format = DigitFormat(digit_format)
    if digit_format == DigitFormat.PLAIN:
        return value
    if digit_format == DigitFormat.DELIMITED:
        return digit_delimiter.join(value)
    raise ValueError(f"unknown digit format {digit_format!r}")
