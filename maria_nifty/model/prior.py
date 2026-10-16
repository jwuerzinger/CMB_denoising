"""Prior-construction helpers for correlated fields and Gaussian envelopes."""

from collections.abc import Mapping
from functools import partial

import jax.numpy as jnp
import numpy as np
from nifty.re import (
    CorrelatedFieldMaker,
    Model,
    NormalPrior,
    VModel,
    WrappedCall,
    lognormal_prior,
    random_like,
)
from nifty.re import normal_prior as nifty_normal_prior

FLUCTUATION_PARAMS = (
    "shape",
    "distances",
    "fluctuations",
    "loglogavgslope",
    "flexibility",
    "asperity",
    "harmonic_type",
    "non_parametric_kind",
)


def _cfm_fluctuations(
    cfm: CorrelatedFieldMaker,
    prefix,
    *,
    shape,
    distances=None,
    fluctuations,
    loglogavgslope,
    flexibility=None,
    asperity=None,
    harmonic_type="fourier",
    non_parametric_kind="power",
):
    """Add fluctuation parameters to a correlated-field maker.

    Parameters
    ----------
    cfm : CorrelatedFieldMaker
        The maker to extend with a new fluctuation axis.
    prefix : str
        Name prefix for this fluctuation component.
    shape : tuple of int
        Grid shape of the fluctuation domain.
    distances : array-like or None, optional
        Physical pixel size; defaults to ``1 / shape``.
    fluctuations : tuple or float
        Prior (or fixed value) for the overall fluctuation amplitude.
    loglogavgslope : tuple or float
        Prior on the power-law slope of the power spectrum.
    flexibility, asperity : tuple or float or None, optional
        Additional spectral shape priors (NIFTy.re conventions).
    harmonic_type : str, optional
        Harmonic basis type passed to NIFTy.re (default ``"fourier"``).
    non_parametric_kind : str, optional
        Non-parametric model type passed to NIFTy.re (default ``"power"``).
    """
    distances = 1 / np.array(shape) if distances is None else distances

    cfm.add_fluctuations(
        shape,
        distances,
        fluctuations,
        loglogavgslope,
        flexibility,
        asperity,
        prefix,
        harmonic_type,
        non_parametric_kind,
    )
    return cfm


def correlated_field_prior(
    prefix,
    *,
    offset_mean=0,
    offset_std,
    n_copies=1,
    in_axes=0,
    out_axes=0,
    **fluctuations,
):
    """Build a correlated-field prior with optional vectorized copies.

    Parameters
    ----------
    prefix : str
        Name prefix for all latent variables of this prior.
    offset_mean : float, optional
        Mean of the overall amplitude offset (default 0).
    offset_std : tuple or float
        Prior on the amplitude offset standard deviation.
    n_copies : int, optional
        Number of independent copies stacked via ``VModel`` (default 1).
    in_axes, out_axes : int or list of str, optional
        Axis specifications passed to ``VModel`` when ``n_copies > 1``.
    **fluctuations : dict
        Fluctuation configurations.  Keyword arguments whose names appear in
        ``FLUCTUATION_PARAMS`` are treated as a single global fluctuation axis.
        All other keyword arguments are expected to be dicts defining named
        per-axis fluctuations (e.g. ``space=dict(...), time=dict(...)``).

    Returns
    -------
    nifty.re.Model
        The assembled correlated-field model (or ``VModel`` if ``n_copies > 1``).
    """
    prefix = prefix.strip() + " "

    cfm = CorrelatedFieldMaker(prefix)
    cfm.set_amplitude_total_offset(offset_mean, offset_std)

    global_fluc = {fk: fv for fk, fv in fluctuations.items() if fk in FLUCTUATION_PARAMS}
    named_fluc = {fk: fv for fk, fv in fluctuations.items() if fk not in FLUCTUATION_PARAMS}

    if global_fluc:
        _cfm_fluctuations(cfm, prefix="", **global_fluc)

    for fk, fv in named_fluc.items():
        if not isinstance(fv, dict):
            raise ValueError(f"Expected fluctuation parameters for '{fk}' to be a dict.")
        fv_cfg = dict(fv)
        pf = fv_cfg.pop("prefix", fk)
        pf = pf.strip() + " "
        _cfm_fluctuations(cfm, prefix=pf, **fv_cfg)

    model = cfm.finalize()

    if n_copies > 1:
        if isinstance(in_axes, list):
            in_axes = [a if a.startswith(prefix) else prefix + a for a in in_axes]
        if isinstance(out_axes, list):
            out_axes = [a if a.startswith(prefix) else prefix + a for a in out_axes]
        return VModel(model, n_copies, in_axes, out_axes)
    else:
        return model


def _prior_or_const(value, ptree, name, prior=lognormal_prior):
    """Convert a constant or prior specification into a call-compatible value."""
    if isinstance(value, tuple | list) and len(value) == 2:
        value = prior(*value)
        value = WrappedCall(value, name=name)
        ptree.update(value.domain)
    elif not isinstance(value, int | float):
        raise TypeError(f"`{value}` must be of type `tuple`, `list`, `int` or `float`")
    return value


def _centered_coos(shp, dis):
    """Return centered coordinates for a 2D grid."""
    coos = jnp.indices(shp).astype(float)
    coos_T = coos.T.reshape(-1, 2)
    coos_T -= 0.5 * (shp - 1)
    coos_T *= dis
    return coos_T.reshape(coos.T.shape).T


def gaussian_prior(
    *,
    prefix: str,
    shape: np.ndarray,
    distances: np.ndarray,
    cov_x: tuple | float,
    cov_y: tuple | float,
    scale: tuple | float = 1.0,
    theta: tuple | float = 0.0,
    n_copies: int = 1,
) -> Model:
    """Build an anisotropic Gaussian prior over a centered 2D grid.

    Parameters
    ----------
    prefix : str
        Name prefix for latent variables.
    shape : array-like of int
        Grid shape ``(ny, nx)``.
    distances : array-like of float
        Physical pixel size ``(dy, dx)``.
    cov_x, cov_y : tuple of (mean, std) or float
        Prior (or fixed value) for the Gaussian widths along each axis before
        rotation.  A two-element tuple is interpreted as the ``(mean, std)``
        moments of a lognormal prior (strictly positive, as a width must be);
        a scalar is used as a fixed constant.
    scale : tuple or float, optional
        Amplitude scale factor (default 1).
    theta : tuple or float, optional
        Counter-clockwise rotation angle in radians (default 0).
    n_copies : int, optional
        Number of independent copies stacked via ``VModel`` (default 1).

    Returns
    -------
    nifty.re.Model
        An anisotropic Gaussian map model with learnable or fixed parameters.
    """
    prefix = prefix.strip() + " "

    ptree = {}
    # cov_x/cov_y are widths, so must stay positive -- lognormal_prior (unlike
    # nifty_normal_prior) enforces that while still moment-matching to the
    # given (mean, std), so existing config values carry over unchanged.
    cov_x = _prior_or_const(cov_x, ptree, prefix + "cov_x", lognormal_prior)
    cov_y = _prior_or_const(cov_y, ptree, prefix + "cov_y", lognormal_prior)
    scale = _prior_or_const(scale, ptree, prefix + "scale", nifty_normal_prior)
    theta = _prior_or_const(theta, ptree, prefix + "theta", nifty_normal_prior)

    coordinates = _centered_coos(np.array(shape), np.array(distances))

    def gaussian(primals: Mapping) -> jnp.ndarray:
        cx = cov_x(primals) if prefix + "cov_x" in primals else cov_x
        cy = cov_y(primals) if prefix + "cov_y" in primals else cov_y
        sc = scale(primals) if prefix + "scale" in primals else scale
        th = theta(primals) if prefix + "theta" in primals else theta

        x, y = coordinates

        # Coefficients of the rotated 2D Gaussian: exp(-(a*x² + 2*b*x*y + c*y²)).
        # theta rotates the principal axes counter-clockwise; cov_x/cov_y are the
        # axis-aligned widths before rotation.
        a = jnp.cos(th) ** 2 / (2 * cx**2) + jnp.sin(th) ** 2 / (2 * cy**2)
        b = -jnp.sin(2 * th) / (4 * cx**2) + jnp.sin(2 * th) / (4 * cy**2)
        c = jnp.sin(th) ** 2 / (2 * cx**2) + jnp.cos(th) ** 2 / (2 * cy**2)
        return sc * jnp.exp(-(a * x**2 + 2 * b * x * y + c * y**2))

    init = {k: partial(random_like, primals=v) for k, v in ptree.items()}
    model = Model(gaussian, domain=ptree.copy(), init=init)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model


def normal_prior(
    prefix,
    shape,
    *,
    mean,
    std,
    n_copies=1,
):
    """Build a normal prior with optional vectorized copies.

    Parameters
    ----------
    prefix : str
        Name for the latent variable.
    shape : tuple of int or int
        Shape of the sampled array.
    mean : float
        Prior mean.
    std : float
        Prior standard deviation.
    n_copies : int, optional
        Number of independent copies stacked via ``VModel`` (default 1).

    Returns
    -------
    nifty.re.Model
        A ``NormalPrior`` model (or ``VModel`` if ``n_copies > 1``).
    """
    model = NormalPrior(mean, std, shape=shape, name=prefix)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model
