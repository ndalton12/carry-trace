"""Arithmetic generation and labeling helpers."""

from __future__ import annotations

from itertools import zip_longest
from random import Random

from carry_trace.enums import SliceName

DIGIT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def int_to_digits_lsd(value: int, base: int = 10, min_width: int = 0) -> list[int]:
    """Return integer digits least-significant first."""
    if base < 2 or base > len(DIGIT_ALPHABET):
        raise ValueError(f"base must be between 2 and {len(DIGIT_ALPHABET)}")
    if value < 0:
        raise ValueError("value must be non-negative")
    if value == 0:
        digits = [0]
    else:
        digits = []
        while value:
            digits.append(value % base)
            value //= base
    while len(digits) < min_width:
        digits.append(0)
    return digits


def digits_lsd_to_int(digits: list[int], base: int = 10) -> int:
    value = 0
    place = 1
    for digit in digits:
        if digit < 0 or digit >= base:
            raise ValueError(f"digit {digit} is invalid for base {base}")
        value += digit * place
        place *= base
    return value


def digits_lsd_to_str(digits: list[int], base: int = 10) -> str:
    trimmed = list(digits)
    while len(trimmed) > 1 and trimmed[-1] == 0:
        trimmed.pop()
    return "".join(DIGIT_ALPHABET[d] for d in reversed(trimmed))


def str_to_digits_lsd(text: str, base: int = 10) -> list[int]:
    cleaned = text.strip().upper().replace(",", "")
    if not cleaned:
        raise ValueError("empty digit string")
    digits = []
    for char in reversed(cleaned):
        if char not in DIGIT_ALPHABET[:base]:
            raise ValueError(f"digit {char!r} is invalid for base {base}")
        digits.append(DIGIT_ALPHABET.index(char))
    return digits


def add_lsd_digits(a_digits: list[int], b_digits: list[int], base: int = 10) -> dict[str, object]:
    carry = 0
    incoming: list[int] = []
    outgoing: list[int] = []
    raw_sum: list[int] = []
    output: list[int] = []

    for a_digit, b_digit in zip_longest(a_digits, b_digits, fillvalue=0):
        incoming.append(carry)
        local_sum = a_digit + b_digit
        raw_sum.append(local_sum)
        total = local_sum + carry
        output.append(total % base)
        carry = total // base
        outgoing.append(carry)

    if carry:
        output.append(carry)

    carry_positions = [idx for idx, value in enumerate(outgoing) if value > 0]
    return {
        "raw_sum": raw_sum,
        "incoming_carry": incoming,
        "outgoing_carry": outgoing,
        "output_digits_lsd": output,
        "carry_count": len(carry_positions),
        "max_carry_chain": max_carry_chain(outgoing),
        "carry_positions": carry_positions,
    }


def max_carry_chain(outgoing_carry: list[int]) -> int:
    best = 0
    current = 0
    for value in outgoing_carry:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def make_problem(a: int, b: int, n_digits: int, base: int = 10) -> dict[str, object]:
    a_digits = int_to_digits_lsd(a, base=base)
    b_digits = int_to_digits_lsd(b, base=base)
    labels = add_lsd_digits(a_digits, b_digits, base=base)
    answer_digits = labels["output_digits_lsd"]
    if not isinstance(answer_digits, list):
        raise TypeError("output_digits_lsd must be a list")
    answer = digits_lsd_to_str(answer_digits, base=base)
    return {
        "base": base,
        "n_digits": n_digits,
        "a": digits_lsd_to_str(a_digits, base=base),
        "b": digits_lsd_to_str(b_digits, base=base),
        "answer": answer,
        "digits_a_lsd": a_digits,
        "digits_b_lsd": b_digits,
        "answer_length_change": len(answer) > max(
            len(digits_lsd_to_str(a_digits, base=base)),
            len(digits_lsd_to_str(b_digits, base=base)),
        ),
        **labels,
    }


def generate_problem(
    n_digits: int,
    rng: Random,
    base: int = 10,
    slice_name: SliceName | str = SliceName.RANDOM,
) -> dict[str, object]:
    """Generate a labeled addition problem for a named behavioral slice."""
    slice_name = SliceName(slice_name)
    if n_digits < 1:
        raise ValueError("n_digits must be positive")
    if base != 10:
        raise ValueError("Goal 1 generator currently supports base 10 only")

    if slice_name == SliceName.NO_CARRY:
        return _generate_no_carry(n_digits, rng, base)
    if slice_name == SliceName.ISOLATED_CARRY:
        return _generate_by_predicate(
            n_digits,
            rng,
            base,
            lambda row: row["carry_count"] == 1 and row["max_carry_chain"] == 1,
            slice_name.value,
        )
    if slice_name == SliceName.LONG_CARRY_CHAIN:
        return make_problem(int("9" * n_digits), 1, n_digits=n_digits, base=base)
    if slice_name == SliceName.INTERNAL_CARRY_CHAIN:
        if n_digits < 3:
            return make_problem(int("9" * n_digits), 1, n_digits=n_digits, base=base)
        a = int("1" + "0" + ("9" * (n_digits - 2)))
        return make_problem(a, 1, n_digits=n_digits, base=base)
    if slice_name == SliceName.CARRY_DISTRACTOR:
        return _make_pattern_problem(n_digits, a_pattern=[9, 0], b_pattern=[0, 1], base=base)
    if slice_name == SliceName.MANY_9S_NO_CARRY:
        return _make_pattern_problem(n_digits, a_pattern=[9, 0, 9], b_pattern=[0, 8, 0], base=base)
    if slice_name == SliceName.RANDOM:
        return _generate_random(n_digits, rng, base)
    raise ValueError(f"unknown slice {slice_name!r}")


def _generate_random(n_digits: int, rng: Random, base: int) -> dict[str, object]:
    lower = 0 if n_digits == 1 else base ** (n_digits - 1)
    upper = (base**n_digits) - 1
    return make_problem(rng.randint(lower, upper), rng.randint(lower, upper), n_digits, base)


def _generate_by_predicate(
    n_digits: int,
    rng: Random,
    base: int,
    predicate: object,
    slice_name: str,
) -> dict[str, object]:
    for _ in range(10_000):
        row = _generate_random(n_digits, rng, base)
        if callable(predicate) and predicate(row):
            return row
    raise RuntimeError(f"could not generate slice {slice_name!r} for {n_digits} digits")


def _generate_no_carry(n_digits: int, rng: Random, base: int) -> dict[str, object]:
    a_digits = []
    b_digits = []
    for idx in range(n_digits):
        max_a = base - 1
        if idx == n_digits - 1 and n_digits > 1:
            a_digit = rng.randint(1, max_a)
        else:
            a_digit = rng.randint(0, max_a)
        b_digit = rng.randint(0, base - 1 - a_digit)
        if idx == n_digits - 1 and n_digits > 1 and b_digit == 0:
            b_digit = rng.randint(0, base - 1 - a_digit)
        a_digits.append(a_digit)
        b_digits.append(b_digit)
    return make_problem(
        digits_lsd_to_int(a_digits, base),
        digits_lsd_to_int(b_digits, base),
        n_digits,
        base,
    )


def _make_pattern_problem(
    n_digits: int,
    a_pattern: list[int],
    b_pattern: list[int],
    base: int,
) -> dict[str, object]:
    a_digits = [a_pattern[idx % len(a_pattern)] for idx in range(n_digits)]
    b_digits = [b_pattern[idx % len(b_pattern)] for idx in range(n_digits)]
    if n_digits > 1 and a_digits[-1] == 0:
        a_digits[-1] = 1
    return make_problem(
        digits_lsd_to_int(a_digits, base),
        digits_lsd_to_int(b_digits, base),
        n_digits,
        base,
    )
