from carry_trace.parsing import parse_final_answer


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
