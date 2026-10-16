"""Tests for maria_nifty/model/prior.py — pure Python and JAX."""

import numpy as np
import pytest

from maria_nifty.model.prior import _centered_coos, _prior_or_const

# ---------------------------------------------------------------------------
# _prior_or_const — pure Python (constant branches only)
# ---------------------------------------------------------------------------


class TestPriorOrConst:
    def test_float_returned_unchanged(self):
        result = _prior_or_const(1.5, {}, "test_name")
        assert result == 1.5

    def test_int_returned_unchanged(self):
        result = _prior_or_const(2, {}, "test_name")
        assert result == 2

    def test_zero_float_returned_unchanged(self):
        result = _prior_or_const(0.0, {}, "test_name")
        assert result == 0.0

    def test_negative_float_returned_unchanged(self):
        result = _prior_or_const(-3.14, {}, "test_name")
        assert result == pytest.approx(-3.14)

    def test_ptree_unchanged_for_scalar(self):
        ptree = {"existing_key": 1}
        _prior_or_const(5.0, ptree, "test_name")
        assert ptree == {"existing_key": 1}

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError):
            _prior_or_const("invalid", {}, "test_name")

    def test_list_wrong_length_raises_type_error(self):
        # A list of length 1 is not (int, float) and not a 2-element spec.
        with pytest.raises(TypeError):
            _prior_or_const([1.0], {}, "test_name")

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            _prior_or_const(None, {}, "test_name")


# ---------------------------------------------------------------------------
# _centered_coos — JAX
# ---------------------------------------------------------------------------


class TestCenteredCoos:
    def test_output_shape(self):
        result = _centered_coos(np.array([4, 6]), np.array([1.0, 1.0]))
        assert result.shape == (2, 4, 6)

    def test_center_pixel_at_origin(self):
        # For a 3×3 grid with unit spacing the center pixel (1,1) maps to (0, 0).
        result = _centered_coos(np.array([3, 3]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(np.array(result[:, 1, 1]), [0.0, 0.0], atol=1e-6)

    def test_corner_pixel_symmetric(self):
        # With unit spacing the grid spans from –1 to +1 for a 3×3 array.
        result = _centered_coos(np.array([3, 3]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(np.array(result[:, 0, 0]), [-1.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(np.array(result[:, 2, 2]), [1.0, 1.0], atol=1e-6)

    def test_distances_scale_coordinates(self):
        # Halving the distances halves the coordinate range.
        result_full = _centered_coos(np.array([3, 3]), np.array([1.0, 1.0]))
        result_half = _centered_coos(np.array([3, 3]), np.array([0.5, 0.5]))
        np.testing.assert_allclose(
            np.array(result_half),
            np.array(result_full) * 0.5,
            rtol=1e-6,
        )

    def test_asymmetric_grid(self):
        result = _centered_coos(np.array([3, 5]), np.array([1.0, 1.0]))
        assert result.shape == (2, 3, 5)
        # First axis (rows): center row index = 1 → 0.0
        np.testing.assert_allclose(float(result[0, 1, 2]), 0.0, atol=1e-6)
        # Second axis (cols): center col index = 2 → 0.0
        np.testing.assert_allclose(float(result[1, 1, 2]), 0.0, atol=1e-6)

    def test_even_grid_no_exact_center(self):
        # For a 4×4 grid there is no integer center; coordinates are half-integer.
        result = _centered_coos(np.array([4, 4]), np.array([1.0, 1.0]))
        # Pixel (1,1) should be at (-0.5, -0.5) for a 4×4 unit grid.
        np.testing.assert_allclose(np.array(result[:, 1, 1]), [-0.5, -0.5], atol=1e-6)
        np.testing.assert_allclose(np.array(result[:, 2, 2]), [0.5, 0.5], atol=1e-6)

    def test_single_pixel(self):
        result = _centered_coos(np.array([1, 1]), np.array([2.0, 3.0]))
        assert result.shape == (2, 1, 1)
        np.testing.assert_allclose(np.array(result[:, 0, 0]), [0.0, 0.0], atol=1e-6)
