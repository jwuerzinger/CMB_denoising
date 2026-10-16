"""Atmosphere model consisting of a base model and optional additive components."""

import re
from functools import reduce
from operator import or_

import numpy as np
from nifty.re import Model

from ..detector import Detector
from .correlation import SpaceTimeCorrelationModel, TimeCorrelationModel
from .polynomial import SpacePolynomialModel, TimePolynomialModel
from .spline import TimeSplineModel


class AtmosphereModel(Model):
    """Combine a base atmosphere model with optional additive components."""

    def __init__(self, base_model, additive_models=None, mode=None):
        self.base_model = base_model
        self.additive_models = [] if additive_models is None else list(additive_models)
        self.mode = mode

        models = [self.base_model, *self.additive_models]
        super().__init__(
            domain=reduce(or_, [m.domain for m in models]),
            init=reduce(or_, [m.init for m in models]),
        )

    def __call__(self, x):
        """Evaluate the atmosphere model and all additive components."""
        out = self.base_model(x)
        for model in self.additive_models:
            out += model(x)
        return out

    @staticmethod
    def _interpret_mode(mode):
        """Normalize and validate the atmosphere mode specification."""
        if mode is None:
            raise ValueError("`mode` is required.")

        allowed = {"spline", "gp-1d", "gp-3d", "poly-1d", "poly-2d"}
        base_allowed = {"spline", "gp-1d", "gp-3d"}

        tokens = [t for t in re.split(r"[+,\s]+", str(mode).strip().lower()) if t]
        if not tokens:
            raise ValueError("`mode` is empty.")

        unknown = [t for t in tokens if t not in allowed]
        if unknown:
            raise ValueError(f"Unknown mode token(s): {unknown}.")

        # Deduplicate while preserving order.
        tokens = list(dict.fromkeys(tokens))
        bases = [t for t in tokens if t in base_allowed]
        if len(bases) != 1:
            raise ValueError("Use exactly one base mode: `spline`, `gp-1d`, or `gp-3d`.")

        if "poly-1d" in tokens and "poly-2d" in tokens:
            raise ValueError("Use either `poly-1d` or `poly-2d`, not both.")

        out = [bases[0]]
        if "poly-1d" in tokens:
            out.append("poly-1d")
        if "poly-2d" in tokens:
            out.append("poly-2d")
        return "+".join(out)

    @classmethod
    def build(
        cls,
        detector,
        tod_data,
        *,
        mode,
        prefix="atm",
        time=None,
        boresight_el=None,
        spline_params=None,
        gp_params=None,
        poly_params=None,
    ):
        """Build an atmosphere model from detector data and config blocks.

        Parameters
        ----------
        detector : Detector
            Detector instance used to align atmosphere components to the TODs.
        tod_data : numpy.ndarray
            TOD array used to derive model shapes and optional grouping.
        mode : str
            Atmosphere mode string, for example ``spline``, ``gp-1d``,
            ``gp-3d``, ``poly-1d``, or ``poly-2d``. Composite modes may be
            expressed with ``+`` separators.
        prefix : str, optional
            Prefix applied to all sub-model names.
        time : array-like, optional
            Time axis required for spline-based atmosphere models.
        boresight_el : array-like, optional
            Boresight elevation samples used by some spline configurations.
        spline_params : dict, optional
            Configuration for the spline atmosphere component.
        gp_params : dict, optional
            Configuration for Gaussian-process atmosphere components.
        poly_params : dict, optional
            Configuration for polynomial atmosphere components.

        Returns
        -------
        AtmosphereModel
            A combined atmosphere model with a base component and optional
            additive polynomial components.
        """
        if not isinstance(detector, Detector):
            raise ValueError("Expected `detector` to be an instance of `Detector`")
        if not isinstance(tod_data, np.ndarray):
            raise ValueError("Expected `tod_data` to be a numpy array")

        spline_params = {} if spline_params is None else dict(spline_params)
        gp_params = {} if gp_params is None else dict(gp_params)
        poly_params = {} if poly_params is None else dict(poly_params)

        mode = cls._interpret_mode(mode)

        base_model = None
        additive_models = []

        if "spline" in mode:
            if time is None:
                raise ValueError("`time` is required for atmosphere mode `spline`.")

            sp_prefix = spline_params.pop("prefix", "spline")
            base_model = TimeSplineModel.build(
                detector=detector,
                time=time,
                boresight_el=boresight_el,
                prefix=prefix.strip() + " " + sp_prefix.strip(),
                **spline_params,
            )

        elif "gp-3d" in mode:
            g3_prefix = gp_params.pop("prefix", "gp")
            base_model = SpaceTimeCorrelationModel.build(
                detector,
                tod_data,
                prefix=prefix.strip() + " " + g3_prefix.strip(),
                **gp_params,
            )
        elif "gp-1d" in mode:
            g1_prefix = gp_params.pop("prefix", "gp")
            base_model = TimeCorrelationModel.build(
                detector,
                tod_data,
                prefix=prefix.strip() + " " + g1_prefix.strip(),
                **gp_params,
            )

        if "poly-1d" in mode:
            p1_prefix = poly_params.pop("prefix", "poly")
            additive_models.append(
                TimePolynomialModel.build(
                    detector,
                    tod_data.shape[1],
                    prefix=prefix.strip() + " " + p1_prefix.strip(),
                    **poly_params,
                )
            )
        elif "poly-2d" in mode:
            p2_prefix = poly_params.pop("prefix", "poly")
            additive_models.append(
                SpacePolynomialModel.build(
                    detector,
                    prefix=prefix.strip() + " " + p2_prefix.strip(),
                    **poly_params,
                )
            )

        return cls(base_model=base_model, additive_models=additive_models, mode=mode)
