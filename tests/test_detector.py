"""Tests for detector geometry and grouping helpers."""

import numpy as np
import pytest

from maria_nifty.detector import (
    DetectorGroups,
    detector_mask,
    hex_ring_clockwise,
    hex_ring_linear,
    n_coords,
)


class TestNCoords:
    def test_order_0(self):
        assert n_coords(0) == 1

    def test_order_1(self):
        assert n_coords(1) == 7

    def test_order_2(self):
        assert n_coords(2) == 19

    def test_formula(self):
        for o in range(5):
            assert n_coords(o) == 1 + 6 * sum(range(1, o + 1))


class TestHexRings:
    def test_clockwise_order_0(self):
        coords = hex_ring_clockwise(0)
        assert coords.shape == (1, 2)

    def test_clockwise_order_1(self):
        coords = hex_ring_clockwise(1)
        assert coords.shape == (7, 2)

    def test_linear_order_0(self):
        coords = hex_ring_linear(0)
        assert coords.shape == (1, 2)

    def test_linear_order_1(self):
        coords = hex_ring_linear(1)
        assert coords.shape == (7, 2)

    def test_same_set_of_coords(self):
        # Clockwise and linear orderings cover the same set of points.
        for order in range(1, 4):
            cw = hex_ring_clockwise(order)
            ln = hex_ring_linear(order)
            assert cw.shape == ln.shape
            assert np.allclose(np.sort(cw, axis=0), np.sort(ln, axis=0))

    def test_center_kwarg(self):
        offset = (1.0, 2.0)
        coords_centered = hex_ring_clockwise(1, center=offset)
        coords_default = hex_ring_clockwise(1)
        diff = coords_centered - coords_default
        assert np.allclose(diff, np.array(offset))


class TestDetectorMask:
    def test_all_unmasked_by_default(self):
        m = detector_mask(5)
        assert m.dtype == bool
        assert m.shape == (5,)
        assert not m.any()

    def test_mask_via_indices(self):
        m = detector_mask(5, mask_indices=[1, 3])
        assert m[1] and m[3]
        assert not m[0] and not m[2] and not m[4]

    def test_mask_direct(self):
        raw = np.array([False, True, False, True, False])
        m = detector_mask(5, mask=raw)
        assert np.array_equal(m, raw)

    def test_both_raises(self):
        with pytest.raises(ValueError):
            detector_mask(5, mask=[True, False, False, False, False], mask_indices=[0])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            detector_mask(5, mask=[True, False])

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError):
            detector_mask(5, mask_indices=[10])


class TestDetectorGroups:
    def test_build_from_hex_order_1(self):
        coords = hex_ring_clockwise(1)
        groups = DetectorGroups.build(coords, np.arange(len(coords)))
        assert groups.number >= 1

    def test_mask_applied(self):
        coords = hex_ring_clockwise(1)
        n = len(coords)
        mask = np.zeros(n, dtype=bool)
        mask[0] = True
        groups = DetectorGroups.build(coords, np.arange(n), mask=mask)
        assert groups.n_masked == 1
        assert groups.indices[0] == -1

    def test_number_of_groups_upper_bound(self):
        coords = hex_ring_clockwise(2)
        n = len(coords)
        groups = DetectorGroups.build(coords, np.arange(n), order=2)
        assert groups.number <= n
