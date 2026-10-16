"""Tests for model/response.py and model/jax_filter.py — pure Python and JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.ndimage import map_coordinates

from maria_nifty.model.jax_filter import _gaussian_kernel1d, gaussian_filter
from maria_nifty.model.response import (
    apply_pointing_matrix,
    build_pointing_matrix,
    map2tods,
    offsets2coords,
    project_map,
)

# ---------------------------------------------------------------------------
# offsets2coords — pure NumPy
# ---------------------------------------------------------------------------


class TestOffsets2Coords:
    def test_output_shape(self):
        n_det, n_time = 4, 100
        offsets = np.zeros((n_det, n_time, 2))
        coords = offsets2coords(offsets, shape=(10, 10), fov=(0.1, 0.1))
        assert coords.shape == (2, n_det, n_time)

    def test_zero_offsets_land_at_map_center(self):
        # For zero offsets and zero center: pixel index = 0.5 * (shape - 1)
        offsets = np.zeros((2, 3, 2))
        coords = offsets2coords(offsets, shape=(11, 11), fov=(0.11, 0.11))
        np.testing.assert_allclose(coords, 5.0)

    def test_ra_offset_shifts_axis0_only(self):
        # offsets[..., 0] (ra) indexes array axis 0 (see `shape` docstring);
        # a ra-only shift must move coords[0] and leave coords[1] unchanged.
        n_det, n_time = 2, 2
        offsets_ra = np.zeros((n_det, n_time, 2))
        offsets_ra[:, :, 0] = 0.05
        shape, fov = (10, 10), (0.1, 0.1)
        c_zero = offsets2coords(np.zeros_like(offsets_ra), shape, fov)
        c_ra = offsets2coords(offsets_ra, shape, fov)
        assert not np.allclose(c_ra[0], c_zero[0])  # axis 0 (ra) shifted
        np.testing.assert_allclose(c_ra[1], c_zero[1])  # axis 1 (dec) unchanged

    def test_dec_offset_shifts_axis1_only(self):
        # offsets[..., 1] (dec) indexes array axis 1; a dec-only shift must
        # move coords[1] and leave coords[0] unchanged.
        n_det, n_time = 2, 2
        offsets_dec = np.zeros((n_det, n_time, 2))
        offsets_dec[:, :, 1] = 0.03
        shape, fov = (10, 10), (0.1, 0.1)
        c_zero = offsets2coords(np.zeros_like(offsets_dec), shape, fov)
        c_dec = offsets2coords(offsets_dec, shape, fov)
        np.testing.assert_allclose(c_dec[0], c_zero[0])  # axis 0 (ra) unchanged
        assert not np.allclose(c_dec[1], c_zero[1])  # axis 1 (dec) shifted

    def test_center_parameter_shifts_coords(self):
        n_det, n_time = 2, 2
        offsets = np.zeros((n_det, n_time, 2))
        shape, fov = (10, 10), (0.1, 0.1)
        c0 = offsets2coords(offsets, shape, fov, center=(0, 0))
        c1 = offsets2coords(offsets, shape, fov, center=(0.01, 0))
        assert not np.allclose(c0[0], c1[0])  # ra component of center shifts axis 0

    def test_coords_within_map_for_small_offsets(self):
        rng = np.random.default_rng(42)
        shape = (20, 30)
        fov = (0.2, 0.3)
        offsets = rng.uniform(-0.05, 0.05, (3, 5, 2))
        coords = offsets2coords(offsets, shape=shape, fov=fov)
        assert coords.shape == (2, 3, 5)
        assert np.all(coords >= 0)
        assert np.all(coords < max(shape))

    def test_non_square_map_shape(self):
        offsets = np.zeros((2, 4, 2))
        coords = offsets2coords(offsets, shape=(10, 20), fov=(0.1, 0.2))
        assert coords.shape == (2, 2, 4)

    def test_non_square_map_samples_correct_pixel(self):
        # Regression test for a ra/dec <-> array-axis mismatch: build a map
        # array the same way DiffuseModel does (shape passed straight
        # through, so axis 0 has size shape[0] and is the ra axis), place a
        # marker at one pixel, and confirm the offset that should land on
        # that pixel actually retrieves it.
        shape = (3, 7)
        fov = (3.0, 7.0)  # distances == (1.0, 1.0)
        m = np.zeros(shape)
        m[0, 5] = 100.0
        # pixel_index = offset / distance + 0.5 * (shape - 1)
        # ra_idx=0  -> offset_ra  = (0 - 1.0) * 1.0 = -1.0
        # dec_idx=5 -> offset_dec = (5 - 3.0) * 1.0 = 2.0
        offsets = np.array([[[-1.0, 2.0]]])
        coords = offsets2coords(offsets, shape, fov)
        sampled = map2tods(jnp.asarray(m), coords, sigma_pixels=0)
        np.testing.assert_allclose(np.asarray(sampled).ravel(), [100.0])


# ---------------------------------------------------------------------------
# _gaussian_kernel1d — JAX
# ---------------------------------------------------------------------------


class TestGaussianKernel1d:
    def test_sums_to_one(self):
        kernel = _gaussian_kernel1d(2.0, 0, 8)
        np.testing.assert_allclose(float(jnp.sum(kernel)), 1.0, atol=1e-6)

    def test_symmetric(self):
        kernel = _gaussian_kernel1d(2.0, 0, 8)
        np.testing.assert_allclose(np.array(kernel), np.array(kernel[::-1]), atol=1e-6)

    def test_length(self):
        kernel = _gaussian_kernel1d(1.0, 0, 5)
        assert len(kernel) == 11  # 2 * radius + 1

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match="order must be non-negative"):
            _gaussian_kernel1d(1.0, -1, 4)

    def test_first_derivative_sums_to_zero(self):
        # The derivative of a Gaussian integrates to zero.
        kernel = _gaussian_kernel1d(2.0, 1, 8)
        np.testing.assert_allclose(float(jnp.sum(kernel)), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# gaussian_filter — JAX
# ---------------------------------------------------------------------------


class TestGaussianFilter:
    def test_shape_preserved(self):
        arr = jnp.ones((10, 10))
        result = gaussian_filter(arr, sigma=1.0)
        assert result.shape == (10, 10)

    def test_uniform_interior_unchanged(self):
        # Constant padding attenuates the values close to the boundary.
        arr = jnp.ones((20, 20)) * 3.0
        result = gaussian_filter(arr, sigma=2.0)
        np.testing.assert_allclose(np.array(result[8:12, 8:12]), 3.0, atol=2e-3)

    def test_near_zero_sigma_is_identity(self):
        # sigma below the internal threshold (1e-15) skips the filter entirely.
        arr = jnp.arange(9.0).reshape(3, 3)
        result = gaussian_filter(arr, sigma=1e-16)
        np.testing.assert_allclose(np.array(result), np.array(arr), atol=1e-6)

    def test_1d_array(self):
        arr = jnp.ones((50,))
        result = gaussian_filter(arr, sigma=2.0)
        np.testing.assert_allclose(np.array(result[10:40]), 1.0, atol=1e-5)

    def test_larger_sigma_spreads_more(self):
        # Convolving a delta with a wider kernel should produce a wider result.
        arr = jnp.zeros((30,)).at[15].set(1.0)
        r1 = np.array(gaussian_filter(arr, sigma=1.0))
        r2 = np.array(gaussian_filter(arr, sigma=3.0))
        # The wider kernel should lower the peak and raise the tails.
        assert float(r2[15]) < float(r1[15])
        assert float(r2[10]) > float(r1[10])


# ---------------------------------------------------------------------------
# map2tods — JAX
# ---------------------------------------------------------------------------


class TestMap2Tods:
    def test_output_shape(self):
        n_det, n_time = 3, 5
        shape = (10, 10)
        offsets = np.zeros((n_det, n_time, 2))
        coords = jnp.array(offsets2coords(offsets, shape=shape, fov=(0.1, 0.1)))
        result = map2tods(jnp.ones(shape), coords, 0.0)
        assert result.shape == (n_det, n_time)

    def test_uniform_map_returns_uniform(self):
        n_det, n_time = 2, 4
        shape = (10, 10)
        rng = np.random.default_rng(0)
        offsets = rng.uniform(-0.03, 0.03, (n_det, n_time, 2))
        coords = jnp.array(offsets2coords(offsets, shape=shape, fov=(0.1, 0.1)))
        result = map2tods(jnp.ones(shape) * 7.0, coords, 0.0)
        np.testing.assert_allclose(np.array(result), 7.0, atol=1e-4)

    def test_smoothing_leaves_uniform_map_unchanged(self):
        n_det, n_time = 2, 3
        shape = (10, 10)
        offsets = np.zeros((n_det, n_time, 2))
        coords = jnp.array(offsets2coords(offsets, shape=shape, fov=(0.1, 0.1)))
        result = map2tods(jnp.ones(shape), coords, 2.0)
        np.testing.assert_allclose(np.array(result), 1.0, atol=3e-2)

    def test_smoothing_spreads_point_source(self):
        # A delta map smoothed with sigma>0 should differ from the unsmoothed version.
        shape = (20, 20)
        input_map = jnp.zeros(shape).at[10, 10].set(1.0)
        offsets = np.zeros((1, 5, 2))
        coords = jnp.array(offsets2coords(offsets, shape=shape, fov=(0.2, 0.2)))
        result_no_smooth = np.array(map2tods(input_map, coords, 0.0))
        result_smooth = np.array(map2tods(input_map, coords, 2.0))
        assert not np.allclose(result_no_smooth, result_smooth)

    def test_mode_nearest_edge_clamps_like_pointing_matrix(self):
        # map2tods(order=0, mode="nearest") is the dense equivalent of
        # kind="pointing_matrix" -- exercised here via the config-facing
        # mode/order kwargs rather than calling map_coordinates directly.
        shape = (10, 8)
        rng = np.random.default_rng(6)
        m = jnp.asarray(rng.standard_normal(shape))
        coords = jnp.array([[shape[0] + 3.0], [3.0]])

        zero_filled = np.asarray(map2tods(m, coords, 0.0))  # default order=1, mode="constant"
        edge_clamped = np.asarray(map2tods(m, coords, 0.0, order=0, mode="nearest"))
        assert zero_filled[0] == 0.0
        assert edge_clamped[0] == m[shape[0] - 1, 3]

    def test_cval_used_for_constant_mode(self):
        shape = (10, 8)
        m = jnp.ones(shape)
        coords = jnp.array([[shape[0] + 3.0], [3.0]])  # out of range -> filled with cval
        result = np.asarray(map2tods(m, coords, 0.0, mode="constant", cval=-5.0))
        np.testing.assert_allclose(result, -5.0)


# ---------------------------------------------------------------------------
# build_pointing_matrix / apply_pointing_matrix / project_map — JAX + scipy.sparse
# ---------------------------------------------------------------------------


class TestPointingMatrix:
    def test_agrees_with_nearest_mode_map_coordinates_on_interior_offsets(self):
        # build_pointing_matrix is nearest-neighbor + edge-clamp -- its dense
        # equivalent is map_coordinates(order=0, mode="nearest"), not the
        # default order=1/mode="constant" used by kind="map_coordinates".
        rng = np.random.default_rng(1)
        shape = (12, 9)
        m = jnp.asarray(rng.standard_normal(shape))
        offsets = rng.uniform(-0.03, 0.03, (5, 7, 2))
        coords = offsets2coords(offsets, shape=shape, fov=(0.1, 0.08))

        ref = np.asarray(map_coordinates(m, coords, order=0, mode="nearest"))
        pointing = build_pointing_matrix(coords, shape)
        out = np.asarray(apply_pointing_matrix(m, pointing, coords.shape[1:]))
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_boundary_edge_clamps_unlike_map_coordinates(self):
        # Regression test for the boundary-condition finding: build_pointing_matrix
        # clamps out-of-range samples to the nearest edge pixel (matching maria's
        # own forward model), unlike kind="map_coordinates"'s zero-fill. Includes
        # samples just past the edge and samples far outside the map.
        shape = (10, 8)
        rng = np.random.default_rng(2)
        m = jnp.asarray(rng.standard_normal(shape))
        coords = np.array(
            [
                [-0.6, -0.4, shape[0] - 1.0, shape[0] - 0.6, shape[0] + 5.0, 3.2],
                [3.0, 3.0, 3.0, 3.0, 3.0, -8.0],
            ]
        )

        ref = np.asarray(map_coordinates(m, coords, order=0, mode="nearest"))
        zero_fill = np.asarray(map_coordinates(m, coords, order=1, mode="constant"))
        pointing = build_pointing_matrix(coords, shape)
        out = np.asarray(apply_pointing_matrix(m, pointing, coords.shape[1:]))

        np.testing.assert_allclose(out, ref, atol=1e-5)
        assert zero_fill[-2] == 0.0 and zero_fill[-1] == 0.0  # far-out-of-range: map_coordinates zero-fills...
        assert out[-2] != 0.0 and out[-1] != 0.0  # ...but pointing_matrix edge-clamps to a real pixel value

    def test_gradient_wrt_map_matches_nearest_mode_map_coordinates(self):
        rng = np.random.default_rng(3)
        shape = (10, 10)
        m = jnp.asarray(rng.standard_normal(shape))
        offsets = rng.uniform(-0.03, 0.03, (4, 6, 2))
        coords = offsets2coords(offsets, shape=shape, fov=(0.1, 0.1))
        pointing = build_pointing_matrix(coords, shape)

        grad_ref = jax.grad(lambda mm: jnp.sum(map_coordinates(mm, coords, order=0, mode="nearest") ** 2))(m)
        grad_sparse = jax.grad(lambda mm: jnp.sum(apply_pointing_matrix(mm, pointing, coords.shape[1:]) ** 2))(m)
        np.testing.assert_allclose(np.asarray(grad_sparse), np.asarray(grad_ref), atol=1e-4)

    def test_map2tods_dispatches_pointing_matrix_kind(self):
        rng = np.random.default_rng(4)
        shape = (10, 10)
        m = jnp.asarray(rng.standard_normal(shape))
        offsets = rng.uniform(-0.03, 0.03, (3, 5, 2))
        coords = offsets2coords(offsets, shape=shape, fov=(0.1, 0.1))
        pointing = build_pointing_matrix(coords, shape)

        ref = np.asarray(map_coordinates(m, coords, order=0, mode="nearest"))
        out = np.asarray(map2tods(m, coords, 0.0, kind="pointing_matrix", pointing=pointing))
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_single_nonzero_per_row_with_unit_weight(self):
        rng = np.random.default_rng(5)
        shape = (14, 11)
        offsets = rng.uniform(-0.05, 0.05, (6, 9, 2))
        coords = offsets2coords(offsets, shape=shape, fov=(0.15, 0.12))
        pointing = build_pointing_matrix(coords, shape)

        n_samples = int(np.prod(coords.shape[1:]))
        indptr = np.asarray(pointing.indptr)
        data = np.asarray(pointing.data)
        assert pointing.shape == (n_samples, shape[0] * shape[1])
        np.testing.assert_array_equal(np.diff(indptr), 1)  # exactly one nonzero per row
        np.testing.assert_allclose(data, 1.0)  # every weight is exactly 1

    def test_project_map_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown response kind"):
            project_map(jnp.ones((5, 5)), jnp.zeros((2, 3)), kind="bogus")
