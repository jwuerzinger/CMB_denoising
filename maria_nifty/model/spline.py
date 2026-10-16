"""B-spline atmosphere models projected onto detector time streams."""

import jax.numpy as jnp
import numpy as np
from maria import utils as maria_utils
from nifty.re import Model

from ..detector import Detector
from .prior import normal_prior


class TimeSplineModel(Model):
    """Time-domain atmosphere model using a B-spline basis."""

    def __init__(self, model, detector, basis):
        self.model = model
        self.basis = jnp.asarray(basis, dtype=jnp.float64)
        self.detector = detector

        super().__init__(domain=self.model.domain, init=self.model.init)

    def __call__(self, x):
        """Evaluate the spline basis and return the masked detector stream."""
        coeff = self.model(x)
        baseline = coeff @ self.basis

        if self.detector.n_masked > 0:
            baseline = baseline[~self.detector.mask]

        return baseline

    @classmethod
    def build(
        cls,
        *,
        detector,
        time,
        prefix="spline",
        knot_spacing,
        order=3,
        remove_el_gradient=False,
        boresight_el=None,
        coeff_prior=None,
    ):
        """Build a time spline model from time samples and knot spacing.

        Parameters
        ----------
        detector : Detector
            Detector instance used to apply masking.
        time : array-like
            One-dimensional time axis used to construct the spline basis.
        prefix : str, optional
            Prefix applied to the spline coefficient prior.
        knot_spacing : float
            Spacing between spline knots in the same units as `time`.
        order : int, optional
            Spline order used by the basis construction.
        remove_el_gradient : bool, optional
            Whether to split the basis into elevation-weighted components.
        boresight_el : array-like, optional
            Boresight elevation samples required when removing the elevation
            gradient.
        coeff_prior : dict, optional
            Prior configuration for the spline coefficients.

        Returns
        -------
        TimeSplineModel
            A masked spline model for time-domain atmosphere structure.
        """
        if not isinstance(detector, Detector):
            raise ValueError("Expected `detector` to be an instance of `Detector`.")

        t = np.asarray(time, dtype=np.float64)
        if t.ndim != 1:
            raise ValueError("Expected `time` to be 1D.")

        basis = maria_utils.signal.bspline_basis(
            t,
            spacing=float(knot_spacing),
            order=int(order),
        )

        if remove_el_gradient:
            if boresight_el is None:
                raise ValueError("`boresight_el` is required when `remove_el_gradient=True`.")
            el = np.asarray(boresight_el, dtype=np.float64)
            if el.shape != t.shape:
                raise ValueError("Expected `boresight_el` to match `time` shape.")
            el_ptp = el.max() - el.min()
            if el_ptp == 0:
                raise ValueError("Cannot build spline elevation gradient basis for constant elevation.")
            rel_el = (el - el.min()) / el_ptp
            basis = np.concatenate([basis * rel_el[None], basis * (1 - rel_el)[None]], axis=0)

        prior = {"mean": 0.0, "std": 0.05} if coeff_prior is None else dict(coeff_prior)
        prior_mean = float(prior.get("mean", 0.0))
        prior_std = float(prior.get("std", 0.05))

        model = normal_prior(
            prefix.strip() + " " + "coeffs",
            (detector.full_size, basis.shape[0]),
            mean=prior_mean,
            std=prior_std,
        )

        return cls(
            model=model,
            detector=detector,
            basis=basis,
        )
