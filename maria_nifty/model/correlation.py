"""Gaussian process-based models for spatial and temporal correlations."""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.lax import slice as jax_slice
from jax.scipy.ndimage import map_coordinates
from nifty.re import Model

from ..detector import Detector
from ..minimize import check_dict
from .prior import correlated_field_prior


def upsample(array, factor, axis=-1):
    """Repeat samples along one axis by an integer factor."""
    if factor == 1:
        return array
    if isinstance(factor, int) and factor > 1:
        return jnp.repeat(array, factor, axis=axis)
    raise ValueError(f"Invalid zoom factor `{factor}`")


def upsample_linear(array, factor, axis=-1):
    """Linearly interpolate samples along one axis by an integer factor."""
    if factor == 1:
        return array
    if not (isinstance(factor, int) and factor > 1):
        raise ValueError(f"Invalid zoom factor `{factor}`")

    axis = axis % array.ndim
    moved = jnp.moveaxis(array, axis, -1)
    n = moved.shape[-1]
    if n == 0:
        return array

    coords_1d = np.linspace(0, n - 1, n * factor)

    def interp_1d(x):
        return map_coordinates(x, coords_1d[None, :], order=1, mode="nearest")

    flat = moved.reshape((-1, n))
    out = vmap(interp_1d)(flat)
    out = out.reshape(moved.shape[:-1] + (coords_1d.size,))
    return jnp.moveaxis(out, -1, axis)


def slice_array(array, start, limit, axis=0):
    """Slice a JAX array along one axis with fixed start and limit."""
    start_indices = [0] * array.ndim
    limit_indices = list(array.shape)
    start_indices[axis] = start
    limit_indices[axis] = limit
    return jax_slice(array, tuple(start_indices), tuple(limit_indices))


def atmos2tods(in_array, in_shp, in_cen, in_dis, out_coords):
    """Project atmosphere grid values to TOD sample coordinates.

    Parameters
    ----------
    in_array : jnp.ndarray, shape (ny, nx)
        Atmosphere map on a regular grid.
    in_shp : array-like of int
        Grid shape ``(ny, nx)``.
    in_cen : array-like of float
        Physical center of the grid ``(cy, cx)``.
    in_dis : array-like of float
        Physical pixel size ``(dy, dx)``.
    out_coords : jnp.ndarray, shape (n_samples, 2)
        Physical ``(y, x)`` coordinates of each TOD sample.

    Returns
    -------
    jnp.ndarray, shape (n_samples,)
        Bi-linearly interpolated atmosphere values at the TOD positions.
    """
    rel_coords = out_coords.copy()
    rel_coords -= in_cen
    rel_coords /= in_dis
    rel_coords += 0.5 * (jnp.array(in_shp) - 1)
    return map_coordinates(in_array, rel_coords.T, order=1)


def vmap_atmos2tods(in_array, in_shp, in_cen, in_dis, out_coords):
    """Vectorize atmosphere-to-TOD projection over a leading axis.

    Applies :func:`atmos2tods` independently to each slice along the last
    axis of ``in_array`` (one slice per detector group or time frame).

    Parameters
    ----------
    in_array : jnp.ndarray, shape (ny, nx, n_groups)
        Atmosphere maps stacked along the last axis, one per group.
    in_shp, in_cen, in_dis, out_coords
        Forwarded to :func:`atmos2tods`; see that function for details.

    Returns
    -------
    jnp.ndarray, shape (n_samples, n_groups)
        Interpolated values for each group at all TOD sample positions.
    """
    return vmap(atmos2tods, in_axes=(-1, None, None, None, None), out_axes=-1)(
        in_array, in_shp, in_cen, in_dis, out_coords
    )


def _make_space_ops(shape, padding):
    """Return padded space shape and a slicer for the original extent."""
    slicer = lambda x: x
    p_shape = tuple(shape)

    if padding > 1.0:
        p_shape = tuple(int(padding * s) for s in shape)
        p_min = tuple((p - s) // 2 for p, s in zip(p_shape, shape))
        p_max = tuple(s + (p - s) // 2 for p, s in zip(p_shape, shape))
        slicer = lambda x: jax_slice(x, p_min, p_max)

    return p_shape, slicer


def _make_time_ops(t_steps, padding, downsampling):
    """Return padded time length, slicer, and upsampler for time models."""
    slicer = upsampler = lambda x: x

    d_steps = int(t_steps)
    if downsampling > 1:
        if d_steps % downsampling != 0:
            raise ValueError("downsampling must divide time_steps evenly.")
        d_steps //= downsampling
        upsampler = partial(upsample_linear, factor=downsampling, axis=-1)

    p_steps = d_steps
    if padding > 1.0:
        p_steps = int(padding * d_steps)
        slicer = lambda x: slice_array(x, 0, d_steps, axis=-1)

    return p_steps, slicer, upsampler


class SpaceCorrelationModel(Model):
    """Correlated-field model restricted to the spatial domain."""

    def __init__(self, model, slicer):
        self.model = model
        self.slicer = slicer

        super().__init__(domain=self.model.domain, init=self.model.init)

    def __call__(self, x):
        """Evaluate the model and remove any padded spatial border."""
        return self.slicer(self.model(x))

    @classmethod
    def build(cls, *, prefix="gp-2d", shape, cfm_params, padding=1.0):
        """Build a 2D spatial correlated-field model.

        Parameters
        ----------
        prefix : str, optional
            Prefix applied to the correlated-field prior.
        shape : tuple of int
            Spatial output shape of the model.
        cfm_params : dict
            Configuration for :func:`correlated_field_prior`.
        padding : float, optional
            Padding factor applied before slicing back to the requested shape.

        Returns
        -------
        SpaceCorrelationModel
            A spatial correlated-field wrapper with optional padding removal.
        """
        p_shape, slicer = _make_space_ops(shape, float(padding))

        cfm_cfg = dict(cfm_params)
        if "space" in cfm_cfg:
            cfm_cfg["space"] = dict(cfm_cfg["space"])
            cfm_cfg["space"] |= {"shape": p_shape}
        else:
            cfm_cfg |= {"shape": p_shape}

        model = correlated_field_prior(prefix, **cfm_cfg)
        return cls(model, slicer)


class TimeCorrelationModel(Model):
    """Correlated-field model for time-only atmosphere structure."""

    def __init__(self, model, detector, slicer, upsampler):
        self.model = model
        self.detector = detector
        self.slicer = slicer
        self.upsampler = upsampler

        super().__init__(domain=self.model.domain, init=self.model.init)

    def __call__(self, x):
        """Evaluate the model, upsample in time, and apply detector masking."""
        val = self.model(x)
        val = self.slicer(val)
        val = self.upsampler(val)

        group_indices = self.detector.groups.indices
        if self.detector.n_groups > 1:
            # Clamp masked detectors (group index -1) to 0 so JAX array
            # indexing stays in-bounds; their values are zeroed out below.
            safe_indices = np.where(group_indices < 0, 0, group_indices)
            val = val[safe_indices]
        elif self.detector.n_groups == 1:
            val = jnp.broadcast_to(val, (self.detector.full_size,) + val.shape)
        else:
            val = jnp.zeros((self.detector.full_size,) + val.shape)

        masked = group_indices < 0
        if np.any(masked):
            # Zero out detectors that were masked and indexed via the 0-clamp above.
            val = val.at[masked].set(0.0)
        if self.detector.n_masked > 0:
            val = val[~self.detector.mask]
        return val

    @classmethod
    def build(cls, detector, tod_data, *, prefix="gp-1d", cfm_params, padding=1.0, downsampling=1):
        """Build a 1D time-domain correlated-field model for the detector data.

        Parameters
        ----------
        detector : Detector
            Detector instance that defines masking and grouping.
        tod_data : numpy.ndarray
            TOD array used to infer the temporal model length.
        prefix : str, optional
            Prefix applied to the correlated-field prior.
        cfm_params : dict
            Configuration for :func:`correlated_field_prior`.
        padding : float, optional
            Padding factor applied to the temporal axis before slicing back.
        downsampling : int, optional
            Downsampling factor applied to the underlying temporal model.

        Returns
        -------
        TimeCorrelationModel
            A time-domain correlated-field wrapper that honors detector masks.
        """
        if not isinstance(detector, Detector):
            raise ValueError("Expected `detector` to be an instance of `Detector`")
        if not isinstance(tod_data, np.ndarray):
            raise ValueError("Expected `tod_data` to be a numpy array")

        p_steps, slicer, upsampler = _make_time_ops(tod_data.shape[1], float(padding), int(downsampling))

        cfm_cfg = dict(cfm_params)
        if "time" in cfm_cfg:
            cfm_cfg["time"] = dict(cfm_cfg["time"])
            cfm_cfg["time"] |= {"shape": p_steps}
        else:
            cfm_cfg |= {"shape": p_steps}

        model = correlated_field_prior(prefix, n_copies=max(detector.n_groups, 1), **cfm_cfg)
        return cls(model, detector, slicer, upsampler)


class SpaceTimeCorrelationModel(Model):
    """Correlated-field model with time evolution and spatial interpolation."""

    def __init__(self, model, detector, slicer, upsampler, space_dis):
        self.model = model
        self.detector = detector
        self.slicer = slicer
        self.upsampler = upsampler
        self.space_cen = np.array([0.0, 0.0])
        self.space_dis = np.array(space_dis)

        super().__init__(domain=self.model.domain, init=self.model.init)

    def __call__(self, x):
        """Evaluate, upsample, project to detector coordinates, and mask."""
        val = self.model(x)
        val = self.slicer(val)
        val = self.upsampler(val)
        val = vmap_atmos2tods(
            val,
            val.shape[:-1],
            self.space_cen,
            self.space_dis,
            self.detector.coordinates,
        )
        if self.detector.n_masked > 0:
            val = val.at[self.detector.mask].set(0.0)
            val = val[~self.detector.mask]
        return val

    @classmethod
    def build(cls, detector, tod_data, *, prefix="gp-3d", cfm_params, padding=1.0, downsampling=1):
        """Build a 3D space-time correlated-field model for detector TODs.

        Parameters
        ----------
        detector : Detector
            Detector instance that defines mask handling and coordinates.
        tod_data : numpy.ndarray
            TOD array used to infer the temporal model length.
        prefix : str, optional
            Prefix applied to the correlated-field prior.
        cfm_params : dict
            Configuration for :func:`correlated_field_prior`.
        padding : float, optional
            Padding factor applied to the temporal axis before slicing back.
        downsampling : int, optional
            Downsampling factor applied to the underlying temporal model.

        Returns
        -------
        SpaceTimeCorrelationModel
            A combined time-space atmosphere model projected to detector TODs.
        """
        if not isinstance(detector, Detector):
            raise ValueError("Expected `detector` to be an instance of `Detector`")
        if not isinstance(tod_data, np.ndarray):
            raise ValueError("Expected `tod_data` to be a numpy array")

        check_dict(cfm_params, ["offset_std", "space", "time"], ["offset_mean"])

        p_steps, slicer, upsampler = _make_time_ops(tod_data.shape[1], float(padding), int(downsampling))

        cfm_cfg = dict(cfm_params)
        cfm_cfg["time"] = dict(cfm_cfg["time"])
        cfm_cfg["space"] = dict(cfm_cfg["space"])
        cfm_cfg["time"] |= {"shape": p_steps}

        model = correlated_field_prior(prefix, **cfm_cfg)

        space_dis = cfm_cfg["space"].get("distances", None)
        if space_dis is None:
            space_dis = 1 / np.array(cfm_cfg["space"]["shape"])

        return cls(model, detector, slicer, upsampler, space_dis)
