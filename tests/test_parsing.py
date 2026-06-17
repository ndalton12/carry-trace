from carry_trace.parsing import parse_final_answer, parse_final_output_digits


def test_parse_plain_answer() -> None:
    assert parse_final_answer("7447") == "7447"


def test_parse_answer_prefix() -> None:
    assert parse_final_answer("Reasoning...\nAnswer: 7,447") == "7447"


def test_parse_boxed_answer() -> None:
    assert parse_final_answer("Therefore \\boxed{7447}.") == "7447"


def test_parse_base_suffix() -> None:
    assert parse_final_answer("final answer: 123_8", base=8) == "123"


def test_parse_delimited_answer() -> None:
    assert parse_final_answer("Answer: 1|2|3") == "123"


def test_parse_final_numeric_string() -> None:
    assert parse_final_answer("First 12, then 34, so 46.") == "46"


def test_parse_lsd_final_answer_after_normal_answer() -> None:
    text = """
    So the sum is 110,888,433.

    Normal answer: 110,888,433
    Digits (right to left): 3 3 4 8 8 8 0 1 1

    So the final answer in least significant digit format is:

    334888011
    """
    assert parse_final_output_digits(text) == "334888011"
    assert parse_final_answer(text, answer_format="lsd") == "110888433"
