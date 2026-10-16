"""Tests for maria_nifty/model/space.py — pure Python and JAX."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from maria_nifty.model.space import GridSpace, PointSpace, jax_rotate, space_coos, space_lims

# ---------------------------------------------------------------------------
# space_lims — pure NumPy
# ---------------------------------------------------------------------------


class TestSpaceLims:
    def test_symmetric_unit_fov(self):
        result = space_lims(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.zeros(2))
        np.testing.assert_allclose(result, [[-0.5, 0.5], [-0.5, 0.5]])

    def test_off_center(self):
        result = space_lims(np.array([2.0, 2.0]), np.array([1.0, 1.0]), np.zeros(2))
        np.testing.assert_allclose(result, [[0.0, 2.0], [0.0, 2.0]])

    def test_output_shape(self):
        result = space_lims(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.zeros(2))
        assert result.shape == (2, 2)

    def test_asymmetric_fov(self):
        result = space_lims(np.array([2.0, 4.0]), np.array([0.0, 0.0]), np.zeros(2))
        np.testing.assert_allclose(result[0], [-1.0, 1.0])
        np.testing.assert_allclose(result[1], [-2.0, 2.0])

    def test_shift_translates_limits(self):
        result = space_lims(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.1, 0.2]))
        np.testing.assert_allclose(result, [[-0.4, 0.6], [-0.3, 0.7]])


# ---------------------------------------------------------------------------
# GridSpace — pure Python (no shift, no JAX)
# ---------------------------------------------------------------------------


class TestGridSpaceBuild:
    def test_build_from_fov(self):
        gs = GridSpace.build(shape=[10, 10], fov=[0.1, 0.1])
        assert gs.shape == (10, 10)
        np.testing.assert_allclose(gs.fov, [0.1, 0.1], rtol=1e-6)

    def test_build_from_distances(self):
        gs = GridSpace.build(shape=[8, 8], distances=[0.02, 0.02])
        assert gs.shape == (8, 8)
        np.testing.assert_allclose(gs.distances, (0.02, 0.02))

    def test_default_unit_spacing(self):
        gs = GridSpace.build(shape=[5, 5])
        np.testing.assert_allclose(gs.distances, (0.2, 0.2), rtol=1e-6)

    def test_center(self):
        gs = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0], center=[0.1, 0.2])
        np.testing.assert_allclose(gs.center, (0.1, 0.2))

    def test_rotation_wrapped_to_0_2pi(self):
        gs = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0], rotation=3 * math.pi)
        assert 0 <= gs.rotation <= 2 * math.pi

    def test_repr(self):
        gs = GridSpace.build(shape=[10, 10], fov=[0.1, 0.1])
        r = repr(gs)
        assert "GridSpace" in r
        assert "shape=" in r


class TestGridSpaceProperties:
    @pytest.fixture
    def gs(self):
        return GridSpace.build(shape=[10, 20], distances=[0.01, 0.02])

    def test_ndim(self, gs):
        assert gs.ndim == 2

    def test_size(self, gs):
        assert gs.size == 200

    def test_shp(self, gs):
        np.testing.assert_array_equal(gs.shp, [10, 20])

    def test_dis(self, gs):
        np.testing.assert_allclose(gs.dis, [0.01, 0.02])

    def test_fov(self, gs):
        np.testing.assert_allclose(gs.fov, [0.1, 0.4], rtol=1e-6)

    def test_to_dict_contains_shape_and_fov(self, gs):
        d = gs.to_dict()
        assert "shape" in d
        assert "fov" in d

    def test_to_dict_distances_key(self, gs):
        d = gs.to_dict(keys=("shape", "distances"))
        assert "distances" in d
        assert "fov" not in d


class TestGridSpaceManipulation:
    def test_refine_doubles_resolution(self):
        gs = GridSpace.build(shape=[10, 10], distances=[0.01, 0.01])
        refined = gs.refine(2)
        assert refined.shape == (20, 20)
        np.testing.assert_allclose(refined.fov, gs.fov, rtol=1e-6)

    def test_multiply_fov_doubles_pixel_count(self):
        gs = GridSpace.build(shape=[10, 10], distances=[0.01, 0.01])
        larger = gs.multiply_fov(2)
        assert larger.shape == (20, 20)
        np.testing.assert_allclose(larger.distances, gs.distances)

    def test_update_changes_shape(self):
        gs = GridSpace.build(shape=[10, 10], fov=[0.1, 0.1])
        gs2 = gs.update(shape=[20, 20])
        assert gs2.shape == (20, 20)
        np.testing.assert_allclose(gs2.fov, gs.fov, rtol=1e-5)


# ---------------------------------------------------------------------------
# PointSpace — pure Python (no shift, no JAX)
# ---------------------------------------------------------------------------


class TestPointSpaceBuild:
    def test_single_point(self):
        ps = PointSpace.build(coordinates=[0.1, 0.2])
        assert ps.n_copies == 1
        np.testing.assert_allclose(ps.coos, [0.1, 0.2])

    def test_multiple_copies(self):
        ps = PointSpace.build(coordinates=[[0.1, 0.2], [0.3, 0.4]], n_copies=2)
        assert ps.n_copies == 2
        np.testing.assert_allclose(ps.coos, [[0.1, 0.2], [0.3, 0.4]])

    def test_shape_property(self):
        ps = PointSpace.build(coordinates=[0.0, 0.0])
        assert ps.shape == (1, 1)

    def test_ndim(self):
        ps = PointSpace.build(coordinates=[0.0, 0.0])
        assert ps.ndim == 2

    def test_repr(self):
        ps = PointSpace.build(coordinates=[0.1, 0.2])
        assert "PointSpace" in repr(ps)


# ---------------------------------------------------------------------------
# JAX tests
# ---------------------------------------------------------------------------


class TestJaxRotate:
    def test_identity(self):
        xy = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        result = jax_rotate(xy, 0.0)
        np.testing.assert_allclose(np.array(result), [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)

    def test_90_degrees(self):
        # CCW rotation by pi/2: (1, 0) -> (0, 1)
        xy = jnp.array([[1.0, 0.0]])
        result = jax_rotate(xy, math.pi / 2)
        np.testing.assert_allclose(np.array(result), [[0.0, 1.0]], atol=1e-6)

    def test_180_degrees(self):
        xy = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        result = jax_rotate(xy, math.pi)
        np.testing.assert_allclose(np.array(result), [[-1.0, 0.0], [0.0, -1.0]], atol=1e-6)

    def test_preserves_norm(self):
        xy = jnp.array([[3.0, 4.0]])
        result = jax_rotate(xy, 1.23)
        original_norm = float(jnp.sqrt(jnp.sum(xy**2)))
        result_norm = float(jnp.sqrt(jnp.sum(result**2)))
        assert abs(original_norm - result_norm) < 1e-5


class TestSpaceCoos:
    def test_output_shape(self):
        result = space_coos(jnp.array([4, 6]), jnp.array([0.1, 0.1]), jnp.zeros(2), 0.0)
        assert result.shape == (2, 4, 6)

    def test_center_pixel_at_origin(self):
        # 3x3 grid, unit spacing, zero center: middle pixel (1,1) -> (0, 0)
        result = space_coos(jnp.array([3, 3]), jnp.array([1.0, 1.0]), jnp.zeros(2), 0.0)
        np.testing.assert_allclose(np.array(result[:, 1, 1]), [0.0, 0.0], atol=1e-6)

    def test_corner_pixel_values(self):
        result = space_coos(jnp.array([3, 3]), jnp.array([1.0, 1.0]), jnp.zeros(2), 0.0)
        np.testing.assert_allclose(np.array(result[:, 0, 0]), [-1.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(np.array(result[:, 2, 2]), [1.0, 1.0], atol=1e-6)

    def test_center_offset_shifts_all_pixels(self):
        result = space_coos(jnp.array([3, 3]), jnp.array([1.0, 1.0]), jnp.array([5.0, 5.0]), 0.0)
        np.testing.assert_allclose(np.array(result[:, 1, 1]), [5.0, 5.0], atol=1e-6)


class TestGridSpaceCoos:
    def test_coos_shape(self):
        gs = GridSpace.build(shape=[5, 10], distances=[0.1, 0.1])
        assert gs.coos.shape == (2, 5, 10)

    def test_center_pixel_at_zero_center(self):
        gs = GridSpace.build(shape=[5, 5], distances=[1.0, 1.0])
        np.testing.assert_allclose(np.array(gs.coos[:, 2, 2]), [0.0, 0.0], atol=1e-5)

    def test_coos_with_nonzero_center(self):
        gs = GridSpace.build(shape=[5, 5], distances=[1.0, 1.0], center=[2.0, 3.0])
        np.testing.assert_allclose(np.array(gs.coos[:, 2, 2]), [2.0, 3.0], atol=1e-5)


class TestGridSpaceEquality:
    def test_equal_spaces(self):
        gs1 = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0])
        gs2 = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0])
        assert gs1 == gs2

    def test_different_shape_not_equal(self):
        gs1 = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0])
        gs2 = GridSpace.build(shape=[20, 20], fov=[1.0, 1.0])
        assert gs1 != gs2

    def test_different_fov_not_equal(self):
        gs1 = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0])
        gs2 = GridSpace.build(shape=[10, 10], fov=[2.0, 2.0])
        assert gs1 != gs2

    def test_compare_non_gridspace_raises(self):
        gs = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0])
        with pytest.raises(ValueError):
            gs.__eq__("not a GridSpace")


class TestGridSpaceMultiCopy:
    def test_n_copies_stored(self):
        gs = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0], center=[[0.0, 0.0], [0.1, 0.1]], n_copies=2)
        assert gs.n_copies == 2

    def test_multi_copy_center_shape(self):
        gs = GridSpace.build(shape=[10, 10], fov=[1.0, 1.0], center=[[0.0, 0.0], [0.1, 0.1]], n_copies=2)
        assert gs.cen.shape == (2, 2)
