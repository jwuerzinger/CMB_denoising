"""Tests for maria_nifty/model/polynomial.py — JAX required."""

import jax.numpy as jnp
import numpy as np
import pytest

from maria_nifty.model.polynomial import SpacePolynomialModel, TimePolynomialModel

# ---------------------------------------------------------------------------
# TimePolynomialModel._build_basis — static method, JAX
# ---------------------------------------------------------------------------


class TestTimePolynomialBuildBasis:
    """Verify the Vandermonde-style temporal basis matrix."""

    def test_order_zero_all_ones(self):
        t = jnp.linspace(0.0, 1.0, 10)
        basis = TimePolynomialModel._build_basis(t, 0)
        assert basis.shape == (10, 1)
        np.testing.assert_allclose(np.array(basis[:, 0]), 1.0)

    def test_order_one_shape(self):
        t = jnp.linspace(0.0, 1.0, 20)
        basis = TimePolynomialModel._build_basis(t, 1)
        assert basis.shape == (20, 2)

    def test_order_one_constant_column(self):
        t = jnp.linspace(0.0, 1.0, 15)
        basis = TimePolynomialModel._build_basis(t, 1)
        np.testing.assert_allclose(np.array(basis[:, 0]), 1.0)

    def test_order_one_linear_column(self):
        t = jnp.linspace(0.0, 1.0, 15)
        basis = TimePolynomialModel._build_basis(t, 1)
        np.testing.assert_allclose(np.array(basis[:, 1]), np.array(t), rtol=1e-6)

    def test_order_two_quadratic_column(self):
        t = jnp.linspace(0.0, 1.0, 15)
        basis = TimePolynomialModel._build_basis(t, 2)
        assert basis.shape == (15, 3)
        np.testing.assert_allclose(np.array(basis[:, 2]), np.array(t**2), rtol=1e-5)

    def test_order_three_shape(self):
        t = jnp.linspace(0.0, 1.0, 8)
        basis = TimePolynomialModel._build_basis(t, 3)
        assert basis.shape == (8, 4)

    def test_basis_at_zero_and_one(self):
        t = jnp.array([0.0, 1.0])
        basis = TimePolynomialModel._build_basis(t, 3)
        # At t=0: all columns are 0^d, which is 1 for d=0 and 0 for d>0.
        np.testing.assert_allclose(np.array(basis[0]), [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        # At t=1: all columns are 1^d = 1.
        np.testing.assert_allclose(np.array(basis[1]), [1.0, 1.0, 1.0, 1.0], atol=1e-6)

    def test_n_columns_equals_order_plus_one(self):
        t = jnp.linspace(0.0, 1.0, 5)
        for order in range(6):
            basis = TimePolynomialModel._build_basis(t, order)
            assert basis.shape[1] == order + 1


# ---------------------------------------------------------------------------
# SpacePolynomialModel._build_basis — instance method (self unused), JAX
# ---------------------------------------------------------------------------


class TestSpacePolynomialBuildBasis:
    """Verify the 2D polynomial spatial basis matrix."""

    # The n_terms formula is (order+1)*(order+2)//2.
    @pytest.mark.parametrize(
        "order, n_terms",
        [(0, 1), (1, 3), (2, 6), (3, 10), (4, 15)],
    )
    def test_n_terms_formula(self, order, n_terms):
        coords = jnp.zeros((5, 2))
        basis = SpacePolynomialModel._build_basis(None, coords, order)
        assert basis.shape == (5, n_terms)

    def test_order_zero_constant_basis(self):
        # Any coordinate, order 0: single column of ones.
        coords = jnp.array([[0.1, 0.2], [0.3, -0.4], [-0.5, 0.6]])
        basis = SpacePolynomialModel._build_basis(None, coords, 0)
        assert basis.shape == (3, 1)
        np.testing.assert_allclose(np.array(basis[:, 0]), 1.0)

    def test_order_one_constant_column(self):
        coords = jnp.array([[0.1, 0.2], [-0.3, 0.4]])
        basis = SpacePolynomialModel._build_basis(None, coords, 1)
        np.testing.assert_allclose(np.array(basis[:, 0]), 1.0)

    def test_order_one_x_column(self):
        # Second term (d=1, i=1, j=0) is x^1 * y^0 = x.
        coords = jnp.array([[0.5, 0.2], [-0.3, 0.7]])
        basis = SpacePolynomialModel._build_basis(None, coords, 1)
        np.testing.assert_allclose(np.array(basis[:, 1]), np.array(coords[:, 1]), rtol=1e-6)

    def test_order_one_y_column(self):
        # Third term (d=1, i=0, j=1) is x^0 * y^1 = y.
        coords = jnp.array([[0.5, 0.2], [-0.3, 0.7]])
        basis = SpacePolynomialModel._build_basis(None, coords, 1)
        np.testing.assert_allclose(np.array(basis[:, 2]), np.array(coords[:, 0]), rtol=1e-6)

    def test_origin_coordinates_give_ones_in_constant(self):
        # At the origin all terms except the constant are zero.
        coords = jnp.zeros((4, 2))
        basis = SpacePolynomialModel._build_basis(None, coords, 2)
        np.testing.assert_allclose(np.array(basis[:, 0]), 1.0)
        # All non-constant terms are zero at the origin.
        np.testing.assert_allclose(np.array(basis[:, 1:]), 0.0, atol=1e-7)

    def test_output_shape_for_n_detectors(self):
        n_dets = 37
        coords = jnp.ones((n_dets, 2))
        basis = SpacePolynomialModel._build_basis(None, coords, 2)
        assert basis.shape[0] == n_dets

    def test_matches_n_terms_classmethod_formula(self):
        # Verify _build_basis column count matches the formula used in build().
        for order in range(5):
            n_terms_formula = (order + 1) * (order + 2) // 2
            coords = jnp.zeros((3, 2))
            basis = SpacePolynomialModel._build_basis(None, coords, order)
            assert basis.shape[1] == n_terms_formula
