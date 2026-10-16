"""Tests for minimization config expansion and evaluation helpers."""

import pytest

from maria_nifty.minimize import (
    check_dict,
    eval_string,
    extend_reps,
    get_reps,
    parse_minimization,
)


class TestEvalString:
    def test_simple_list(self):
        assert eval_string("[kl]") == ["kl"]

    def test_repeated_list(self):
        assert eval_string("2*[kl]") == ["kl", "kl"]

    def test_concatenation(self):
        assert eval_string("[kl]+[linear]") == ["kl", "linear"]

    def test_compact_notation(self):
        assert eval_string("2*[kl]+[linear]") == ["kl", "kl", "linear"]

    def test_integer_constant(self):
        assert eval_string("[100]") == [100]

    def test_none_is_preserved(self):
        result = eval_string("[None]")
        assert result == [None]

    def test_null_is_none(self):
        result = eval_string("[null]")
        assert result == [None]


class TestExtendReps:
    def test_pads_with_last_element(self):
        assert extend_reps([1, 2], 4) == [1, 2, 2, 2]

    def test_trims_to_length(self):
        assert extend_reps([1, 2, 3, 4, 5], 3) == [1, 2, 3]

    def test_exact_length_unchanged(self):
        assert extend_reps([1, 2, 3], 3) == [1, 2, 3]

    def test_custom_pad_value(self):
        assert extend_reps([1], 3, add_val=0) == [1, 0, 0]

    def test_none_pad_value(self):
        assert extend_reps(["kl"], 3, add_val=None) == ["kl", None, None]


class TestCheckDict:
    def test_returns_filtered_dict(self):
        result = check_dict({"a": 1, "b": 2, "c": 3}, needed_keys=["a"], optional_keys=["b"])
        assert result == {"a": 1, "b": 2}

    def test_missing_required_key_raises(self):
        with pytest.raises(ValueError, match="missing"):
            check_dict({"a": 1}, needed_keys=["a", "b"])

    def test_extra_keys_stripped(self):
        result = check_dict({"a": 1, "z": 99}, needed_keys=["a"])
        assert "z" not in result


class TestGetReps:
    def test_scalar_expanded_to_list(self):
        sub = {"n_total_iterations": 3, "sample_mode": "linear"}
        result = get_reps(sub, 3)
        assert result["sample_mode"] == ["linear", "linear", "linear"]

    def test_list_extended(self):
        sub = {"n_total_iterations": 4, "sample_mode": ["kl", "linear"]}
        result = get_reps(sub, 4)
        assert result["sample_mode"] == ["kl", "linear", "linear", "linear"]

    def test_string_expression_expanded(self):
        sub = {"n_total_iterations": 3, "sample_mode": "2*[kl]+[linear]"}
        result = get_reps(sub, 3)
        assert result["sample_mode"] == ["kl", "kl", "linear"]


class TestParseMinimization:
    def test_minimal_config(self):
        cfg = {
            "n_total_iterations": 2,
            "n_samples": 4,
            "sample_mode": "kl",
        }
        result = parse_minimization(cfg)
        assert callable(result["sample_mode"]) or isinstance(result["sample_mode"], str)
        n_total_iterations = result["n_total_iterations"]
        assert (n_total_iterations(0) if callable(n_total_iterations) else n_total_iterations) == 2

    def test_missing_required_key_raises(self):
        with pytest.raises(ValueError):
            parse_minimization({"n_total_iterations": 2})
