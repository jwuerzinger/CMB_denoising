"""Forward-response models that convert maps and atmosphere into TOD space."""

from functools import partial, reduce
from operator import or_

import numpy as np
import scipy.sparse
from jax import jit
from jax.experimental.sparse import BCSR, bcsr_dot_general
from jax.scipy.ndimage import map_coordinates
from nifty.re import Model, Vector

from .jax_filter import gaussian_filter
from .map import MapModel


def offsets2coords(offsets, shape, fov, center=(0, 0)):
    """Convert detector offsets into map pixel coordinates for interpolation.

    Parameters
    ----------
    offsets : array-like, shape (n_det, n_time, 2)
        Sky offsets of each detector sample in radians; last axis is (ra, dec).
    shape : tuple of int
        Pixel grid shape ``(nx, ny)`` of the target map; this is the literal
        shape of the array produced for this space (see
        :class:`maria_nifty.model.map.DiffuseModel`, which passes ``shape``
        straight through), so axis 0 of that array is the ra axis and axis 1
        is the dec axis.
    fov : array-like of float
        Field of view ``(fov_x, fov_y)`` in radians.
    center : tuple of float, optional
        Map center offset applied before coordinate conversion.

    Returns
    -------
    numpy.ndarray, shape (2, n_det, n_time)
        Pixel coordinates suitable for :func:`jax.scipy.ndimage.map_coordinates`;
        coords[0] indexes array axis 0 (ra), coords[1] indexes axis 1 (dec).
    """
    center = np.array(center)
    distances = np.array(fov) / np.array(shape)
    coords = np.array(offsets) - center
    coords /= np.array(distances)
    coords += 0.5 * (np.array(shape) - 1)
    coords = coords.transpose(2, 0, 1)
    return coords


def build_pointing_matrix(coords, shape):
    """Build a fixed sparse nearest-neighbor pointing matrix, reproducing
    exactly maria's own real forward-model projection (see
    ``maria.map.ProjectedMap.pointing_matrix`` / ``maria/sim/map.py``,
    ``regular_digitization`` in ``maria/utils/__init__.py``): each sample is
    rounded to its single nearest pixel (not bilinearly split across 4
    neighbors) and that index is *clamped* into ``[0, shape)`` per axis
    (edge-clamp) rather than zero-filled -- an out-of-map sample always reads
    the nearest edge pixel's value, exactly like maria's edge-padded map.
    This is deliberately different from ``jax.scipy.ndimage.map_coordinates``'s
    default ``mode="constant"`` zero-fill; the dense equivalent of this
    function is ``map_coordinates(..., order=0, mode="nearest")``.

    Parameters
    ----------
    coords : array-like, shape (2, *sample_shape)
        Pixel coordinates as returned by :func:`offsets2coords`.
    shape : tuple of int
        Pixel grid shape ``(nx, ny)`` of the target map.

    Returns
    -------
    P : jax.experimental.sparse.BCSR, shape (n_samples, nx * ny)
        One unit-weight nonzero per row. Differentiable w.r.t. the dense map
        it's applied to (via :func:`apply_pointing_matrix`), not w.r.t. its
        own index structure. ``n_samples == prod(coords.shape[1:])`` --
        callers reshape :func:`apply_pointing_matrix`'s output using
        ``coords.shape[1:]`` themselves (not returned here: a plain Python
        shape tuple can't safely travel through a jitted call alongside a
        traced ``P``).
    """
    coords = np.asarray(coords)
    sample_shape = coords.shape[1:]
    n_samples = int(np.prod(sample_shape))
    nx, ny = shape

    x = coords[0].reshape(-1).astype(np.float64)
    y = coords[1].reshape(-1).astype(np.float64)
    xi = np.clip(np.round(x), 0, nx - 1).astype(np.int64)
    yi = np.clip(np.round(y), 0, ny - 1).astype(np.int64)

    rows = np.arange(n_samples)
    cols = xi * ny + yi
    data = np.ones(n_samples, dtype=np.float32)

    p_scipy = scipy.sparse.csr_array((data, (rows, cols)), shape=(n_samples, nx * ny))
    return BCSR.from_scipy_sparse(p_scipy)


def apply_pointing_matrix(smoothed_map, pointing, sample_shape):
    """``P @ smoothed_map.ravel()``, reshaped back to ``sample_shape``.

    ``sample_shape`` is taken as an explicit argument (rather than bundled
    with ``pointing``) so it stays a concrete Python tuple under ``jit`` --
    a plain tuple of ints bundled into the same pytree as a traced ``BCSR``
    would itself be treated as traced leaves and become un-reshapable.
    """
    out = bcsr_dot_general(pointing, smoothed_map.reshape(-1), dimension_numbers=(((1,), (0,)), ((), ())))
    return out.reshape(sample_shape)


def project_map(smoothed_map, coords, order=1, *, kind="map_coordinates", mode="constant", cval=0.0, pointing=None):
    """Dispatch a smoothed map's projection to TOD space by ``kind``.

    ``order``/``mode``/``cval`` are :func:`jax.scipy.ndimage.map_coordinates`'s
    own arguments, used only when ``kind="map_coordinates"``.
    ``pointing_matrix`` currently takes no tunable arguments of its own --
    it's always nearest-neighbor with edge-clamped boundaries (see
    :func:`build_pointing_matrix`), by design, to match maria's own forward
    model exactly.
    """
    if kind == "map_coordinates":
        return map_coordinates(smoothed_map, coords, order, mode=mode, cval=cval)
    if kind == "pointing_matrix":
        return apply_pointing_matrix(smoothed_map, pointing, coords.shape[1:])
    raise ValueError(f"Unknown response kind {kind!r}; expected 'map_coordinates' or 'pointing_matrix'.")


@partial(jit, static_argnames=["sigma_pixels", "order", "kind", "mode", "cval"])
def map2tods(
    input_map, coords, sigma_pixels, order=1, kind="map_coordinates", mode="constant", cval=0.0, pointing=None
):
    """Project a sky map into TOD space with optional Gaussian smoothing.

    Parameters
    ----------
    input_map : jnp.ndarray, shape (nx, ny)
        Sky map to project.
    coords : jnp.ndarray, shape (2, n_det, n_time)
        Pixel coordinates as returned by :func:`offsets2coords`.
    sigma_pixels : float
        Beam smoothing width in pixels; no smoothing applied when ≤ 0.
    order : int, optional
        Interpolation order passed to :func:`jax.scipy.ndimage.map_coordinates`
        (``kind="map_coordinates"`` only).
    kind : {"map_coordinates", "pointing_matrix"}, optional
        Projection backend; see :func:`project_map`.
    mode, cval : optional
        Boundary-handling arguments passed to
        :func:`jax.scipy.ndimage.map_coordinates` (``kind="map_coordinates"``
        only) -- e.g. ``mode="nearest"`` for edge-clamp instead of the
        default zero-fill (``mode="constant"``, ``cval=0.0``).
    pointing : optional
        Sparse pointing matrix from :func:`build_pointing_matrix`, required
        when ``kind="pointing_matrix"``.

    Returns
    -------
    jnp.ndarray, shape (n_det, n_time)
        Interpolated TOD samples.
    """
    smoothed_map = input_map if sigma_pixels <= 0.0 else gaussian_filter(input_map, sigma=sigma_pixels)

    return project_map(smoothed_map, coords, order, kind=kind, mode=mode, cval=cval, pointing=pointing)


class MapResponse(Model):
    """Map-only response model that projects map reconstructions to TODs."""

    def __init__(self, model, offsets, sigma, mask=None, kind="map_coordinates", order=1, mode="constant", cval=0.0):
        if not isinstance(model, MapModel):
            raise ValueError("model has to be of type `MapModel`")
        self.model = model
        self.space = model.space
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.offsets = np.asarray(offsets)

        if self.mask is not None:
            if self.mask.shape != (self.offsets.shape[0],):
                n_valid = int(np.count_nonzero(~self.mask))
                if n_valid == self.offsets.shape[0]:
                    # Offsets already contain only unmasked detectors.
                    self.mask = None
                else:
                    raise ValueError("Expected `mask` to match detector axis shape `(n_detectors,)`.")
            if self.mask is not None:
                self.offsets = self.offsets[~self.mask]

        self.sigma = float(sigma)
        self.coords = offsets2coords(self.offsets, self.space.shape, self.space.fov, self.space.center)
        self.kind = str(kind)
        self.order = int(order)
        self.mode = str(mode)
        self.cval = float(cval)
        self.pointing = None if self.kind != "pointing_matrix" else build_pointing_matrix(self.coords, self.space.shape)

        super().__init__(domain=self.model.domain, init=self.model.init)

    def __call__(self, x):
        """Evaluate the map response for the given model parameters."""
        return map2tods(
            self.model(x),
            self.coords,
            self.sigma,
            order=self.order,
            kind=self.kind,
            mode=self.mode,
            cval=self.cval,
            pointing=self.pointing,
        )


class SumResponse(Model):
    """Combine atmosphere and map responses into one TOD-space model."""

    def __init__(self, atm_model, map_model, map_response):
        self.atm_model = atm_model
        self.map_model = map_model
        self.map_response = map_response

        # Build the domain from the components actually evaluated in __call__.
        models = [m for m in (self.atm_model, self.map_response) if m is not None]
        if len(models) == 0:
            raise ValueError("At least one response component must be provided.")

        super().__init__(
            domain=Vector(reduce(or_, [m.domain for m in models])),
            init=reduce(or_, [m.init for m in models]),
        )

    def __call__(self, x):
        """Evaluate the combined response, summing all available components."""
        res = self.atm_model(x) if self.atm_model is not None else None
        if self.map_response is not None:
            map_res = self.map_response(x)
            res = map_res if res is None else res + map_res
        if res is None:
            raise RuntimeError("No response component available.")
        return res

    @classmethod
    def build(
        cls,
        atm_model=None,
        map_model=None,
        *,
        offsets,
        sigma=None,
        detector=None,
        kind="map_coordinates",
        order=1,
        mode="constant",
        cval=0.0,
    ):
        """Build a sum response from atmosphere and/or map components.

        Parameters
        ----------
        atm_model : Model, optional
            Atmosphere response model. May be omitted for map-only runs.
        map_model : MapModel, optional
            Map model used to construct the map response. May be omitted for
            atmosphere-only runs.
        offsets : array-like
            Detector offsets used to map sky coordinates into TOD space.
        sigma : float, optional
            Gaussian smoothing width in map-pixel units. If omitted, a default
            is inferred from the map field of view.
        detector : object, optional
            Detector-like object whose mask is reused for the map response.
        kind : {"map_coordinates", "pointing_matrix"}, optional
            Map projection backend; see :func:`project_map`.
        order, mode, cval : optional
            :func:`jax.scipy.ndimage.map_coordinates`'s own arguments, used
            only when ``kind="map_coordinates"``.

        Returns
        -------
        SumResponse
            Combined response model for atmosphere and map components.
        """
        if atm_model is None and map_model is None:
            raise ValueError("At least one response component must be provided.")

        map_response = None
        if map_model is not None:
            if detector is not None:
                det_mask = getattr(detector, "mask", None)
            else:
                det_mask = getattr(getattr(atm_model, "detector", None), "mask", None)
            if sigma is None:
                sigma = 12.362925440458655 * 0.00155586 / map_model.space.fov[0]
            map_response = MapResponse(
                map_model, offsets, sigma, mask=det_mask, kind=kind, order=order, mode=mode, cval=cval
            )

        return cls(atm_model, map_model, map_response)
