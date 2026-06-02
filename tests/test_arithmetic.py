from random import Random

from carry_trace.arithmetic import (
    add_lsd_digits,
    digits_lsd_to_str,
    generate_problem,
    int_to_digits_lsd,
    str_to_digits_lsd,
)


def test_digit_round_trip_decimal() -> None:
    digits = int_to_digits_lsd(4879)
    assert digits == [9, 7, 8, 4]
    assert digits_lsd_to_str(digits) == "4879"
    assert str_to_digits_lsd("4,879") == digits


def test_addition_labels_include_carry_state() -> None:
    labels = add_lsd_digits([9, 7, 8, 4], [8, 6, 5, 2], base=10)
    assert labels["incoming_carry"] == [0, 1, 1, 1]
    assert labels["outgoing_carry"] == [1, 1, 1, 0]
    assert labels["output_digits_lsd"] == [7, 4, 4, 7]
    assert labels["carry_count"] == 3
    assert labels["max_carry_chain"] == 3


def test_slice_generation_is_deterministic() -> None:
    rng_a = Random(7)
    rng_b = Random(7)
    problem_a = generate_problem(3, rng_a, slice_name="isolated_carry")
    problem_b = generate_problem(3, rng_b, slice_name="isolated_carry")
    assert problem_a == problem_b
    assert problem_a["carry_count"] == 1
    assert problem_a["max_carry_chain"] == 1


def test_long_carry_chain_slice() -> None:
    problem = generate_problem(4, Random(0), slice_name="long_carry_chain")
    assert problem["a"] == "9999"
    assert problem["b"] == "1"
    assert problem["answer"] == "10000"
    assert problem["max_carry_chain"] == 4
