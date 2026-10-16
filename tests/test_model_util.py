"""Tests for maria_nifty/model/util.py — no JAX required."""

import math

import numpy as np
import pytest

from maria_nifty.model.util import (
    ARCMIN2RAD,
    AS2RAD,
    DEG2RAD,
    check_type,
    flatten_list,
    is_val,
    str2rad,
    to_shape,
)


class TestStr2Rad:
    def test_zero(self):
        assert str2rad("0deg") == pytest.approx(0.0)

    def test_rad(self):
        assert str2rad("1rad") == pytest.approx(1.0)

    def test_deg_180(self):
        assert str2rad("180deg") == pytest.approx(math.pi)

    def test_deg_90(self):
        assert str2rad("90deg") == pytest.approx(math.pi / 2)

    def test_amin(self):
        assert str2rad("60amin") == pytest.approx(DEG2RAD)

    def test_as(self):
        assert str2rad("1as") == pytest.approx(AS2RAD)

    def test_mas(self):
        assert str2rad("1mas") == pytest.approx(AS2RAD * 1e-3)

    def test_muas(self):
        assert str2rad("1muas") == pytest.approx(AS2RAD * 1e-6)

    def test_bare_float(self):
        assert str2rad("1.5") == pytest.approx(1.5)

    def test_negative(self):
        assert str2rad("-90deg") == pytest.approx(-math.pi / 2)

    def test_unknown_unit_raises(self):
        with pytest.raises(RuntimeError, match="Unit not understood"):
            str2rad("1km")

    def test_float_string_no_unit(self):
        assert str2rad("3.14159") == pytest.approx(3.14159)

    def test_muas_smaller_than_mas(self):
        assert str2rad("1muas") < str2rad("1mas") < str2rad("1as") < str2rad("1amin") < str2rad("1deg")

    def test_constants_consistent(self):
        assert pytest.approx(ARCMIN2RAD / 60) == AS2RAD
        assert pytest.approx(DEG2RAD / 60) == ARCMIN2RAD


class TestCheckType:
    def test_scalar_pass(self):
        check_type(1, int)

    def test_scalar_wrong_type(self):
        with pytest.raises(TypeError):
            check_type("hello", int)

    def test_float_in_union(self):
        check_type(1.0, (int, float))
        check_type(1, (int, float))

    def test_tuple_of_int(self):
        check_type((1, 2, 3), tuple, int)

    def test_tuple_wrong_element(self):
        with pytest.raises(TypeError):
            check_type((1, 2.0), tuple, int)

    def test_nested_tuple_of_float(self):
        check_type(((1.0, 2.0), (3.0,)), tuple, tuple, float)

    def test_nested_wrong_element(self):
        with pytest.raises(TypeError):
            check_type(((1.0,), (2,)), tuple, tuple, float)

    def test_none_in_union(self):
        check_type(None, (int, type(None)))


class TestFlattenList:
    def test_flat_list(self):
        assert flatten_list([1, 2, 3]) == [1, 2, 3]

    def test_nested_one_level(self):
        assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_nested_two_levels(self):
        assert flatten_list([[[1]], [2, [3]]]) == [1, 2, 3]

    def test_strings_not_flattened(self):
        assert flatten_list(["ab", "cd"]) == ["ab", "cd"]

    def test_mixed(self):
        assert flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]

    def test_empty(self):
        assert flatten_list([]) == []


class TestToShape:
    def test_none_becomes_nan(self):
        # None is treated as a non-iterable scalar and cast to NaN via float64.
        result = to_shape(None, (2,))
        assert result.shape == (2,)
        assert np.all(np.isnan(result))

    def test_scalar_broadcast(self):
        result = to_shape(1.0, (3,))
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])

    def test_string_degrees(self):
        result = to_shape("180deg", (2,))
        np.testing.assert_allclose(result, [math.pi, math.pi])

    def test_reshape_exact(self):
        result = to_shape([1.0, 2.0, 3.0, 4.0], (2, 2))
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_dtype_int(self):
        result = to_shape(1, (2,), dtype="int64")
        assert result.dtype == np.int64

    def test_nested_list_flattened(self):
        result = to_shape([[1.0, 2.0]], (2,))
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_2d_target_shape(self):
        result = to_shape(5.0, (3, 4))
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, np.full((3, 4), 5.0))


class TestIsVal:
    def test_all_zeros_false(self):
        assert not is_val(np.zeros((3, 3)))

    def test_nonzero_true(self):
        assert is_val(np.array([0.0, 1.0]))

    def test_all_nan_false(self):
        assert not is_val(np.array([float("nan"), float("nan")]))

    def test_nan_with_value_true(self):
        assert is_val(np.array([float("nan"), 1.0]))

    def test_scalar_zero_false(self):
        assert not is_val(np.array(0.0))
