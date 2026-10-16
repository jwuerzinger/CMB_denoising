"""Polynomial atmosphere models in time and detector space."""

import jax.numpy as jnp
from nifty.re import Model

from ..detector import Detector
from .prior import normal_prior


def poly2d_basis(coords, order):
    """2D monomial basis up to total degree ``order``, as ``(n_points, n_terms)``.

    ``coords`` is ``(n_points, 2)``; terms are ordered by ascending total
    degree ``d``, and within a degree by ascending power of the first axis:
    ``[1, y, x, y^2, xy, x^2, ...]``.  ``n_terms == (order+1)(order+2)//2``.
    This is the single definition of the ordering, shared by
    :class:`SpacePolynomialModel` (detector space) and the fused
    focal-plane term assembled in :mod:`maria_nifty.fuse`.
    """
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    terms = []
    for d in range(int(order) + 1):
        for i in range(d + 1):
            terms.append((x**i) * (y ** (d - i)))
    return jnp.concatenate(terms, axis=1)


def poly2d_n_terms(order):
    """Number of 2D monomial terms up to total degree ``order``."""
    return (int(order) + 1) * (int(order) + 2) // 2


class TimePolynomialModel(Model):
    """Time-domain polynomial atmosphere model."""

    def __init__(self, model, detector, t_steps, order):
        self.model = model
        self.t_steps = int(t_steps)
        self.order = int(order)
        self.detector = detector

        t_space = jnp.linspace(0.0, 1.0, self.t_steps)
        self.temporal_basis = self._build_basis(t_space, self.order)

        super().__init__(domain=model.domain, init=model.init)

    @staticmethod
    def _build_basis(t_space, order):
        """Stack monomial columns ``[1, t, t², …, t^order]`` into a basis matrix.

        Parameters
        ----------
        t_space : jnp.ndarray, shape (n_time,)
            Normalized time samples in ``[0, 1]``.
        order : int
            Polynomial order; produces ``order + 1`` columns.

        Returns
        -------
        jnp.ndarray, shape (n_time, order + 1)
            Vandermonde-style basis matrix.
        """
        basis_terms = [t_space**d for d in range(order + 1)]
        return jnp.stack(basis_terms, axis=1)

    def __call__(self, x):
        """Evaluate the time polynomial baseline and apply detector masking."""
        coeff = self.model(x)
        output = self.temporal_basis @ coeff.T
        baseline = output.T

        if self.detector.n_masked > 0:
            baseline = baseline[~self.detector.mask]

        return baseline

    @classmethod
    def build(cls, detector, t_steps, *, prefix="poly-1d", order=1, coeff_prior=None, **params):
        """Build a time-domain polynomial atmosphere model.

        Parameters
        ----------
        detector : Detector
            Detector instance used to apply masking.
        t_steps : int
            Number of time samples in the temporal basis.
        prefix : str, optional
            Prefix applied to the polynomial coefficients.
        order : int, optional
            Polynomial order of the time basis.
        coeff_prior : dict, optional
            Prior configuration for the coefficients.

        Returns
        -------
        TimePolynomialModel
            A masked time-domain polynomial atmosphere model.
        """
        if not isinstance(detector, Detector):
            raise ValueError("Expected `detector` to be an instance of `Detector`.")
        if not isinstance(t_steps, int) or t_steps <= 0:
            raise ValueError("Expected `t_steps` to be a positive integer.")
        if not isinstance(order, int) or order < 0:
            raise ValueError("Expected `order` to be a non-negative integer.")

        prior = {} if coeff_prior is None else dict(coeff_prior)
        model = normal_prior(
            prefix=prefix.strip() + " " + "coeffs",
            shape=(detector.full_size, int(order) + 1),
            mean=float(prior.get("mean", params.pop("mean", 0.0))),
            std=float(prior.get("std", params.pop("std", 1.0))),
        )
        return cls(model, detector, t_steps, order)


class SpacePolynomialModel(Model):
    """Space-domain polynomial atmosphere model."""

    def __init__(self, model, detector, order):
        self.model = model
        self.detector = detector
        self.order = int(order)

        # Precompute spatial basis
        coords_norm = detector.coordinates / jnp.max(jnp.abs(detector.coordinates), axis=0, keepdims=True)
        self.spatial_basis = self._build_basis(coords_norm, self.order)  # (n_dets, n_terms)

        super().__init__(domain=model.domain, init=model.init)

    def _build_basis(self, coords, order):
        """Create a 2D polynomial basis up to the requested order."""
        return poly2d_basis(coords, order)  # (n_dets, n_terms)

    def __call__(self, x):
        """Evaluate the spatial polynomial and apply detector masking."""
        coeff = self.model(x)  # (n_terms, 1)
        output = self.spatial_basis @ coeff  # (n_dets, 1)

        if self.detector.n_masked > 0:
            output = output[~self.detector.mask]

        return output

    @classmethod
    def build(cls, detector, *, prefix="poly-2d", order=1, coeff_prior=None, **params):
        """Build a 2D spatial polynomial atmosphere model.

        Parameters
        ----------
        detector : Detector
            Detector instance used to normalize coordinates and apply masks.
        prefix : str, optional
            Prefix applied to the polynomial coefficients.
        order : int, optional
            Total polynomial order of the spatial basis.
        coeff_prior : dict, optional
            Prior configuration for the coefficient field.

        Returns
        -------
        SpacePolynomialModel
            A masked 2D spatial polynomial atmosphere model.
        """
        if not isinstance(detector, Detector):
            raise ValueError("Expected `detector` to be an instance of `Detector`.")
        if not isinstance(order, int) or order < 0:
            raise ValueError("Expected `order` to be a non-negative integer.")

        n_terms = poly2d_n_terms(order)

        prior = {} if coeff_prior is None else dict(coeff_prior)
        model = normal_prior(
            prefix=prefix.strip() + " " + "coeffs",
            shape=(n_terms, 1),
            mean=float(prior.get("mean", params.pop("mean", 0.0))),
            std=float(prior.get("std", params.pop("std", 1.0))),
        )

        return cls(model, detector, order)
