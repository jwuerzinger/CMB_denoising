"""Tests for maria_nifty/plot/util.py — pure Python and NumPy."""

import numpy as np
import pytest

from maria_nifty.plot.util import rows_and_cols, to_shape


class TestRowsAndCols:
    def test_perfect_square(self):
        rows, cols = rows_and_cols(9)
        assert rows == 3
        assert cols == 3

    def test_non_square_product_covers_nums(self):
        for n in [2, 5, 7, 10, 17]:
            rows, cols = rows_and_cols(n)
            assert rows * cols >= n

    def test_near_square_shape(self):
        rows, cols = rows_and_cols(6)
        assert rows * cols >= 6
        assert rows == 3
        assert cols == 2

    def test_single_item(self):
        rows, cols = rows_and_cols(1)
        assert rows == 1
        assert cols == 1

    def test_fixed_rows(self):
        rows, cols = rows_and_cols(12, rows=3)
        assert rows == 3
        assert cols == 4

    def test_fixed_cols(self):
        rows, cols = rows_and_cols(12, cols=4)
        assert rows == 3
        assert cols == 4

    def test_fixed_rows_caps_to_nums(self):
        rows, cols = rows_and_cols(2, rows=10)
        assert rows == 2
        assert cols == 1

    def test_fixed_cols_caps_to_nums(self):
        rows, cols = rows_and_cols(2, cols=10)
        assert cols == 2
        assert rows == 1

    def test_fixed_rows_and_cols(self):
        rows, cols = rows_and_cols(8, rows=2, cols=4)
        assert rows * cols >= 8


class TestToShape:
    def test_scalar_broadcast_to_1d(self):
        result = to_shape(5, shape=(3,))
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [5, 5, 5])

    def test_list_reshape_to_2d(self):
        result = to_shape([1, 2, 3, 4], shape=(2, 2))
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_truncate_excess_elements(self):
        result = to_shape([1, 2, 3, 4, 5], shape=(2,))
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [1, 2])

    def test_pad_with_last_value_by_default(self):
        result = to_shape([1, 2], shape=(4,))
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, [1, 2, 2, 2])

    def test_pad_with_explicit_default(self):
        result = to_shape([1], shape=(3,), default=0)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_dtype_cast(self):
        result = to_shape([1, 2, 3], shape=(3,), dtype=float)
        assert result.dtype == float

    def test_auto_shape_from_rows_cols(self):
        result = to_shape([1, 2, 3, 4], rows=2, cols=2)
        assert result.shape == (2, 2)

    def test_exact_broadcast_single_to_grid(self):
        result = to_shape(7, shape=(2, 3))
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, np.full((2, 3), 7))

    def test_2d_input_batched(self):
        arr = np.ones((3, 4))
        result = to_shape(arr, shape=(2,))
        assert result.shape[0] == 2

    def test_max_axes_zero_raises(self):
        with pytest.raises(ValueError):
            to_shape([1, 2], shape=(2,), max_axes=0)
