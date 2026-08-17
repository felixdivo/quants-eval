from quants_ablation.metrics_utils import parse_binary, parse_multi


def test_binary_parser_requires_unambiguous_answer():
    assert parse_binary("Yes") == 1
    assert parse_binary("The answer is no.") == 0
    assert parse_binary("yes or no") is None
    assert parse_binary("unknown") is None


def test_multi_parser_requires_one_standalone_uppercase_option():
    assert parse_multi("A") == 0
    assert parse_multi("The answer is C.") == 2
    assert parse_multi("A or B") is None
    assert parse_multi("answer") is None
