"""Tests for maria_nifty/model/correlation.py — pure Python and JAX."""

import jax.numpy as jnp
import numpy as np
import pytest

from maria_nifty.model.correlation import (
    _make_space_ops,
    _make_time_ops,
    atmos2tods,
    slice_array,
    upsample,
    vmap_atmos2tods,
)

# ---------------------------------------------------------------------------
# _make_space_ops — pure Python (slicers may use JAX internally)
# ---------------------------------------------------------------------------


class TestMakeSpaceOps:
    def test_no_padding_returns_original_shape(self):
        p_shape, _ = _make_space_ops((10, 20), 1.0)
        assert p_shape == (10, 20)

    def test_padding_expands_shape(self):
        p_shape, _ = _make_space_ops((10, 20), 2.0)
        assert p_shape == (20, 40)

    def test_partial_padding(self):
        p_shape, _ = _make_space_ops((10, 10), 1.5)
        assert p_shape == (15, 15)

    def test_no_padding_slicer_is_identity(self):
        _, slicer = _make_space_ops((4, 4), 1.0)
        arr = np.ones((4, 4))
        np.testing.assert_array_equal(slicer(arr), arr)

    def test_padding_slicer_trims_to_original_shape(self):
        original_shape = (4, 6)
        _, slicer = _make_space_ops(original_shape, 2.0)
        padded = jnp.ones((8, 12))
        result = slicer(padded)
        assert result.shape == original_shape


# ---------------------------------------------------------------------------
# _make_time_ops — pure Python (slicers/upsamplers may use JAX internally)
# ---------------------------------------------------------------------------


class TestMakeTimeOps:
    def test_no_padding_no_downsampling(self):
        p_steps, _, _ = _make_time_ops(100, 1.0, 1)
        assert p_steps == 100

    def test_padding_expands_steps(self):
        p_steps, _, _ = _make_time_ops(100, 2.0, 1)
        assert p_steps == 200

    def test_downsampling_reduces_steps(self):
        p_steps, _, _ = _make_time_ops(100, 1.0, 4)
        assert p_steps == 25

    def test_downsampling_and_padding(self):
        p_steps, _, _ = _make_time_ops(100, 2.0, 4)
        assert p_steps == 50

    def test_uneven_downsampling_raises(self):
        with pytest.raises(ValueError, match="downsampling must divide"):
            _make_time_ops(100, 1.0, 3)

    def test_no_downsampling_upsampler_is_identity(self):
        _, _, upsampler = _make_time_ops(100, 1.0, 1)
        arr = np.ones((5, 100))
        np.testing.assert_array_equal(upsampler(arr), arr)

    def test_downsampling_upsampler_restores_length(self):
        _, _, upsampler = _make_time_ops(100, 1.0, 4)
        arr = jnp.ones((3, 25))
        result = upsampler(arr)
        assert result.shape == (3, 100)

    def test_padding_slicer_trims_time_axis(self):
        _, slicer, _ = _make_time_ops(50, 2.0, 1)
        arr = jnp.ones((3, 100))
        result = slicer(arr)
        assert result.shape == (3, 50)


# ---------------------------------------------------------------------------
# upsample — JAX
# ---------------------------------------------------------------------------


class TestUpsample:
    def test_factor_one_identity(self):
        arr = jnp.array([1.0, 2.0, 3.0])
        result = upsample(arr, 1, axis=0)
        np.testing.assert_array_equal(np.array(result), [1.0, 2.0, 3.0])

    def test_factor_two_1d(self):
        arr = jnp.array([1.0, 2.0, 3.0])
        result = upsample(arr, 2, axis=0)
        np.testing.assert_array_equal(np.array(result), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    def test_factor_two_2d_axis0(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = upsample(arr, 2, axis=0)
        assert result.shape == (4, 2)
        np.testing.assert_array_equal(np.array(result), [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])

    def test_factor_three_last_axis(self):
        arr = jnp.ones((5, 10))
        result = upsample(arr, 3, axis=-1)
        assert result.shape == (5, 30)

    def test_invalid_factor_zero_raises(self):
        arr = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError):
            upsample(arr, 0, axis=0)

    def test_invalid_factor_float_raises(self):
        arr = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError):
            upsample(arr, 2.5, axis=0)


# ---------------------------------------------------------------------------
# slice_array — JAX
# ---------------------------------------------------------------------------


class TestSliceArray:
    def test_1d_slice(self):
        arr = jnp.arange(10.0)
        result = slice_array(arr, 2, 5, axis=0)
        np.testing.assert_array_equal(np.array(result), [2.0, 3.0, 4.0])

    def test_2d_slice_axis0(self):
        arr = jnp.arange(12.0).reshape(4, 3)
        result = slice_array(arr, 1, 3, axis=0)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(np.array(result), [[3, 4, 5], [6, 7, 8]])

    def test_2d_slice_axis1(self):
        arr = jnp.arange(12.0).reshape(3, 4)
        result = slice_array(arr, 1, 3, axis=1)
        assert result.shape == (3, 2)

    def test_full_slice_identity(self):
        arr = jnp.arange(6.0)
        result = slice_array(arr, 0, 6, axis=0)
        np.testing.assert_array_equal(np.array(result), np.arange(6.0))


# ---------------------------------------------------------------------------
# atmos2tods — JAX
# ---------------------------------------------------------------------------


class TestAtmos2Tods:
    def test_output_shape(self):
        grid = jnp.ones((5, 5))
        result = atmos2tods(
            grid,
            jnp.array([5, 5]),
            jnp.zeros(2),
            jnp.array([1.0, 1.0]),
            jnp.zeros((10, 2)),
        )
        assert result.shape == (10,)

    def test_uniform_grid_returns_constant(self):
        grid = jnp.ones((7, 7)) * 3.14
        out_coords = jnp.array([[0.0, 0.0], [0.5, -0.5], [-1.0, 1.0]])
        result = atmos2tods(
            grid,
            jnp.array([7, 7]),
            jnp.zeros(2),
            jnp.array([1.0, 1.0]),
            out_coords,
        )
        np.testing.assert_allclose(np.array(result), 3.14, atol=1e-5)

    def test_single_sample(self):
        grid = jnp.arange(25.0).reshape(5, 5)
        # Query at grid center (2, 2) -> should return value 12.0
        center_coord = jnp.array([[2.0 * 1.0, 2.0 * 1.0]])  # physical center coords
        result = atmos2tods(
            grid,
            jnp.array([5, 5]),
            jnp.array([2.0, 2.0]),  # center at pixel (2,2) * spacing (1,1)
            jnp.array([1.0, 1.0]),
            center_coord,
        )
        assert result.shape == (1,)

    def test_non_square_grid(self):
        grid = jnp.ones((4, 8)) * 2.5
        result = atmos2tods(
            grid,
            jnp.array([4, 8]),
            jnp.zeros(2),
            jnp.array([1.0, 1.0]),
            jnp.zeros((6, 2)),
        )
        assert result.shape == (6,)
        np.testing.assert_allclose(np.array(result), 2.5, atol=1e-5)


# ---------------------------------------------------------------------------
# vmap_atmos2tods — JAX (vectorized over leading axis)
# ---------------------------------------------------------------------------


class TestVmapAtmos2Tods:
    def test_output_shape(self):
        # in_array shape: (ny, nx, n_groups)
        grid = jnp.ones((5, 5, 3)) * 1.0
        result = vmap_atmos2tods(
            grid,
            jnp.array([5, 5]),
            jnp.zeros(2),
            jnp.array([1.0, 1.0]),
            jnp.zeros((10, 2)),
        )
        # Expected: (n_samples, n_groups)
        assert result.shape == (10, 3)

    def test_uniform_grids_return_constant(self):
        n_groups = 4
        grid = jnp.ones((7, 7, n_groups)) * 2.0
        result = vmap_atmos2tods(
            grid,
            jnp.array([7, 7]),
            jnp.zeros(2),
            jnp.array([1.0, 1.0]),
            jnp.zeros((5, 2)),
        )
        np.testing.assert_allclose(np.array(result), 2.0, atol=1e-5)

    def test_consistent_with_single_group(self):
        # vmap over 1 group should match a direct atmos2tods call.
        grid_single = jnp.ones((5, 5))
        grid_stacked = grid_single[:, :, None]
        out_coords = jnp.zeros((8, 2))
        params = dict(
            in_shp=jnp.array([5, 5]),
            in_cen=jnp.zeros(2),
            in_dis=jnp.array([1.0, 1.0]),
            out_coords=out_coords,
        )
        single = atmos2tods(grid_single, **params)
        stacked = vmap_atmos2tods(grid_stacked, **params)
        np.testing.assert_allclose(np.array(stacked[:, 0]), np.array(single), atol=1e-6)
