from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.lax import fori_loop
from jax.scipy.ndimage import map_coordinates as jax_map_coordinates
from nifty.re import Model

from .prior import normal_prior
from .util import check_type, is_val, to_shape


class GridSpace:
    """A regular 2D pixel grid anchored to a sky location.

    Use :meth:`build` to construct instances from physical parameters such as
    field of view or pixel spacing.  Direct construction requires all parameters
    to already be in normalized form (tuples of floats, radians).
    """

    def __init__(self, shape, distances=(1.0, 1.0), center=(0.0, 0.0), rotation=0.0, shift=None, n_copies=1):
        check_type(shape, tuple, int)
        check_type(distances, tuple, float)
        check_type(center, tuple, (tuple, float), float)
        check_type(rotation, (tuple, float), float)
        check_type(shift, (Model, type(None)))
        check_type(n_copies, int)

        self.shape = shape
        self.distances = distances
        self.center = center
        self.rotation = rotation
        self.shift = shift if shift else lambda x: np.zeros_like(center)
        self.n_copies = n_copies

    def __repr__(self):
        return (
            f"GridSpace(shape={self.shape}, distances={self.distances}, center={self.center}, rotation={self.rotation})"
        )

    def __eq__(self, other):
        if not isinstance(other, GridSpace):
            raise ValueError("Can only compare equality for another GridSpace.")
        return self.shape == other.shape and np.all(self.coos == other.coos)

    # def __contains__(self, other):
    #     if not isinstance(other, (GridSpace, PointSpace)):
    #         raise ValueError("Can only check containment for a GridSpace or PointSpace.")
    #     s_lims = self.lims.reshape(-1,2,2).T
    #     o_lims = other.lims.reshape(-1,2,2).T
    #     return np.all(s_lims[0] <= o_lims[0]) and np.all(s_lims[1] >= o_lims[1])

    def __mul__(self, other):
        return self.multiply_shape(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.multiply_shape(1 / other)

    @classmethod
    def build(
        cls,
        shape,
        *,
        distances=None,
        fov=None,
        center=(0.0, 0.0),
        rotation=0.0,
        shift=None,
        n_copies=1,
        prefix="space",
    ):
        """Build a GridSpace from shape, extent, and optional shift parameters.

        Parameters
        ----------
        shape : int or tuple of int
            Pixel grid shape of the space.
        distances : float or tuple of float, optional
            Pixel spacing in each axis. If omitted, it is inferred from `fov`
            or falls back to unit spacing.
        fov : float or tuple of float, optional
            Field of view of the space. Used to derive `distances` when
            explicit spacing is not given.
        center : float or tuple of float, optional
            Sky center of the space. When `n_copies > 1`, one center is used
            per copy.
        rotation : float or tuple of float, optional
            Rotation angle(s) in radians.
        shift : dict or None, optional
            Optional shift prior configuration used to model the exact center.
        n_copies : int, optional
            Number of independent copies represented by the space.
        prefix : str, optional
            Prefix used for the optional shift model.

        Returns
        -------
        GridSpace
            A fully configured grid space instance.
        """
        n_c = to_shape(n_copies, (), "int64")
        if not is_val(n_c):
            n_c = 1
        n_copies = int(n_c)

        shp = to_shape(shape, (2,), "int64")
        dis = to_shape(distances, (2,), "float64")
        fov = to_shape(fov, (2,), "float64")
        cen = to_shape(center, (n_copies, 2), "float64")
        rot = to_shape(rotation, (n_copies,), "float64")

        shape = tuple(shp.tolist())

        if is_val(dis):
            distances = tuple(dis.tolist())
        elif is_val(fov):
            distances = tuple((fov / shp).tolist())
        else:
            distances = tuple((1 / shp).tolist())

        if not is_val(cen):
            cen = np.zeros_like(cen)
        center = tuple(map(tuple, cen.tolist()))

        if not is_val(rot):
            rot = np.zeros_like(rot)
        rotation = tuple((rot % (2 * np.pi)).tolist())

        if shift:
            sh_cfg = dict(shift)
            sh_prefix = sh_cfg.pop("prefix", "shift")
            shift = normal_prior(
                prefix=prefix.strip() + " " + sh_prefix.strip(),
                shape=(n_copies, 2) if n_copies > 1 else (2,),
                n_copies=1,
                **shift,
            )

        if n_copies == 1:
            center = center[0]
            rotation = rotation[0]

        return cls(shape, distances, center, rotation, shift, n_copies)

    @property
    def shp(self):
        return np.array(self.shape)

    @property
    def dis(self):
        return np.array(self.distances)

    @property
    def fov(self):
        return self.shp * self.dis

    @property
    def cen(self):
        return np.array(self.center)

    @property
    def rot(self):
        return np.array(self.rotation)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return self.shp.prod()

    # @property
    # def dvol(self):
    #     return self.dis.prod()

    @property
    def coos(self):
        if self.n_copies == 1:
            return space_coos(self.shp, self.dis, self.cen, self.rot)
        else:
            return vmap(space_coos, in_axes=(None, None, 0, 0))(self.shp, self.dis, self.cen, self.rot)

    @property
    def lims(self):
        if self.n_copies == 1:
            return space_lims(self.fov, self.cen)
        else:
            return vmap(space_lims, in_axes=(None, 0))(self.fov, self.cen)

    @property
    def domain(self):
        return self.shift.domain if isinstance(self.shift, Model) else {}

    @property
    def init(self):
        return self.shift.init if isinstance(self.shift, Model) else {}

    def update(self, **kwargs):
        """Update the space by changing some of its parameters."""
        dct = self.to_dict()
        dct.update(kwargs)
        return GridSpace.build(**dct)

    def refine(self, factor):
        """Multiply the resolution of the space by a factor."""
        check_type(factor, (int, float))
        shape = tuple(int(round(s * factor)) for s in self.shape)
        distances = tuple(float(f / s) for f, s in zip(self.fov, shape))
        return GridSpace(shape, distances, self.center, self.rotation)

    def multiply_fov(self, factor):
        """Multiply the field of view of the space by a factor."""
        check_type(factor, (int, float))
        shape = tuple(int(round(s * factor)) for s in self.shape)
        return GridSpace(shape, self.distances, self.center, self.rotation)

    def to_dict(self, keys=("shape", "fov", "center", "rotation", "n_copies")):
        """Convert the space to a dictionary ({shape: [sx, sy], ...})."""
        dct = {}
        if "shape" in keys:
            dct["shape"] = self.shp.tolist()
        if "distances" in keys:
            dct["distances"] = self.dis.tolist()
        elif "fov" in keys:
            dct["fov"] = self.fov.tolist()
        if "center" in keys and is_val(self.cen):
            dct["center"] = self.cen.tolist()
        if "rotation" in keys and is_val(self.rot):
            dct["rotation"] = self.rot.tolist()
        if "n_copies" in keys and self.n_copies != 1:
            dct["n_copies"] = int(self.n_copies)
        return dct


def space_coos(shp, dis, cen, rot, shift=0):
    """Return physical sky coordinates for every pixel in a grid space.

    Parameters
    ----------
    shp : array-like of int, shape (2,)
        Grid shape ``(nx, ny)``.
    dis : array-like of float, shape (2,)
        Pixel size ``(dx, dy)`` in radians.
    cen : array-like of float, shape (2,)
        Sky center of the grid in radians.
    rot : float
        Rotation angle in radians.
    shift : array-like of float or 0, optional
        Additional translation applied after rotation.

    Returns
    -------
    jnp.ndarray, shape (2, nx, ny)
        Physical coordinates of each pixel.
    """
    coos = jnp.indices(shp).astype(float)
    coos_T = coos.T.reshape(-1, 2)
    coos_T -= 0.5 * (shp - 1)
    coos_T *= dis
    coos_T = jax_rotate(coos_T, rot)
    coos_T += cen + shift
    return coos_T.reshape(coos.T.shape).T


def space_lims(fov, cen, shift=0):
    """Return the [min, max] extent of a grid space along each axis.

    Parameters
    ----------
    fov : array-like of float, shape (2,)
        Field of view ``(fov_x, fov_y)`` in radians.
    cen : array-like of float, shape (2,)
        Sky center in radians.
    shift : array-like of float or 0, optional
        Additional translation.

    Returns
    -------
    numpy.ndarray, shape (2, 2)
        Array ``[[xmin, xmax], [ymin, ymax]]`` in radians.
    """
    return fov[:, None] / 2 * np.array([-1, 1]) + cen[:, None] + shift[:, None]


def jax_rotate(xy, phi):
    """Rotate 2D coordinates counter-clockwise by angle ``phi``.

    Parameters
    ----------
    xy : jnp.ndarray, shape (..., 2)
        Coordinate array to rotate.
    phi : float
        Rotation angle in radians.

    Returns
    -------
    jnp.ndarray, shape (..., 2)
        Rotated coordinates.
    """
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    R = jnp.array([[c, s], [-s, c]])
    return xy @ R


class PointSpace:
    """A collection of discrete point-source locations on the sky.

    Use :meth:`build` to construct instances from raw coordinate inputs.
    """

    def __init__(self, coordinates, shift=None, n_copies=1):
        check_type(coordinates, tuple, (tuple, float), float)
        check_type(shift, (Model, type(None)))
        check_type(n_copies, int)

        self.coordinates = coordinates
        self.shift = shift if shift else lambda x: np.zeros_like(coordinates)
        self.n_copies = n_copies

    def __repr__(self):
        return f"PointSpace(coordinates={self.coordinates})"

    def __len__(self):
        return self.n_copies

    @classmethod
    def build(
        cls,
        coordinates,
        *,
        shift=None,
        n_copies=1,
        prefix="space",
    ):
        """Build a PointSpace object from coordinates and optional shifts.

        Parameters
        ----------
        coordinates : array-like
            Point-source coordinates as one or more ``(x, y)`` pairs.
        shift : dict or None, optional
            Optional shift prior configuration used to refine the coordinates.
        n_copies : int, optional
            Number of point-source copies represented by the space.
        prefix : str, optional
            Prefix used for the optional shift model.

        Returns
        -------
        PointSpace
            A configured point-space instance.
        """
        n_c = to_shape(n_copies, (), "int64")
        if not is_val(n_c):
            n_c = 1
        n_copies = int(n_c)

        coos = to_shape(coordinates, (n_copies, 2), "float64")
        coordinates = tuple(map(tuple, coos.tolist()))

        if shift:
            sh_cfg = dict(shift)
            sh_prefix = sh_cfg.pop("prefix", "shift")
            shift = normal_prior(
                prefix=prefix.strip() + " " + sh_prefix.strip(),
                shape=(n_copies, 2) if n_copies > 1 else (2,),
                n_copies=1,
                **shift,
            )

        if n_copies == 1:
            coordinates = coordinates[0]

        return cls(coordinates, shift, n_copies)

    @property
    def shape(self):
        return (1, 1)

    @property
    def shp(self):
        return np.array(self.shape)

    @property
    def coos(self):
        return np.array(self.coordinates)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def cen(self):
        return np.array(self.coordinates)

    @property
    def rot(self):
        return 0.0 if self.n_copies == 1 else np.zeros(self.n_copies)


def map2space(in_space, out_space, order=1):
    """Return a callable that interpolates a signal from one space to another.

    Parameters
    ----------
    in_space : GridSpace or PointSpace
        Space of the input signal.
    out_space : GridSpace or PointSpace
        Target space of the output signal.
    order : int, optional
        Interpolation order (default 1 = bilinear).

    Returns
    -------
    callable
        A function ``f(signal, *, x)`` that maps ``signal`` from ``in_space``
        to ``out_space``.  Returns an identity lambda when the spaces are equal.
    """
    if out_space == in_space:
        return lambda res, x: res

    map_func = partial(
        map_coordinates_nd,
        in_copies=in_space.n_copies,
        out_copies=out_space.n_copies,
        out_shape=out_space.shape,
        rel_coos=partial(rel_space_coos, in_space=in_space, out_space=out_space),
        order=order,
    )
    return map_func


def rel_space_coos(in_space, out_space, x=None):
    """Get the relative coordinates of the output space in the input space."""

    def rel_coos(out_coos, out_shift, in_cen, in_shift, in_rot=0):
        rel_coos_T = out_coos.T.reshape(-1, 2)
        rel_coos_T += out_shift
        rel_coos_T -= in_cen + in_shift
        rel_coos_T = jax_rotate(rel_coos_T, -in_rot)
        if isinstance(in_space, GridSpace):
            rel_coos_T /= in_space.dis
            rel_coos_T += 0.5 * (jnp.array(in_space.shape) - 1)
        else:
            rel_coos_T /= out_space.dis
            rel_coos_T -= jnp.diag(rel_coos_T[jnp.abs(rel_coos_T).argmin(axis=0)])
        return rel_coos_T.reshape(out_coos.T.shape).T

    match (in_space.n_copies, out_space.n_copies):
        case (1, 1):
            return rel_coos(out_space.coos, out_space.shift(x), in_space.cen, in_space.shift(x), in_space.rot)
        case (1, n_out):
            return vmap(rel_coos, in_axes=(0, 0, None, None, None))(
                out_space.coos, out_space.shift(x), in_space.cen, in_space.shift(x), in_space.rot
            )
        case (n_in, 1):
            return vmap(rel_coos, in_axes=(None, None, 0, 0, 0))(
                out_space.coos, out_space.shift(x), in_space.cen, in_space.shift(x), in_space.rot
            )
        case (n_in, n_out) if n_in == n_out:
            return vmap(rel_coos, in_axes=(0, 0, 0, 0, 0))(
                out_space.coos, out_space.shift(x), in_space.cen, in_space.shift(x), in_space.rot
            )
        case _:
            raise ValueError("Number of copies for input space and output space must match, or one of them must be 1.")


def map_coordinates_2d(in_array, out_array, rel_coos, order=1):
    out_array += jax_map_coordinates(in_array, rel_coos, order)
    return out_array


def map_coordinates_nd(in_array, in_copies, out_copies, out_shape, rel_coos, order=1, x=None):
    if isinstance(in_array, np.ndarray):
        np_type = True
        device = jax.devices("cpu")[0]
    else:
        np_type = False
        device = jax.devices()[0]

    with jax.default_device(device):
        in_array = jnp.asarray(in_array)
        rel_coos = jnp.asarray(rel_coos(x=x))

        in_shape = in_array.shape[-2:]
        ptf_shape = in_array.shape[:-2] if in_copies == 1 else in_array.shape[1:-2]
        sub_shape = (int(np.prod(ptf_shape)),) if len(ptf_shape) > 0 else ()

        map_func = map_coordinates_2d if sub_shape == () else vmap(map_coordinates_2d, in_axes=(0, 0, None, None))

        match (in_copies, out_copies):
            # (p, t, f, x, y) -> (p, t, f, x', y')
            case (1, 1):
                in_array = in_array.reshape(sub_shape + in_shape)
                sub_array = jnp.zeros(sub_shape + out_shape, dtype=in_array.dtype)
                sub_array = map_func(in_array, sub_array, rel_coos, order)
                out_array = sub_array.reshape(ptf_shape + out_shape)

            # (n_in, p, t, f, x, y) -> (p, t, f, x', y')
            case (n_in, 1):
                in_array = in_array.reshape((n_in,) + sub_shape + in_shape)
                sub_array = jnp.zeros(sub_shape + out_shape, dtype=in_array.dtype)
                loop_over_n_in = lambda i, o: map_func(in_array[i], o, rel_coos[i], order)
                sub_array = fori_loop(0, n_in, loop_over_n_in, sub_array)
                out_array = sub_array.reshape(ptf_shape + out_shape)

            # (n_in, p, t, f, x, y) -> (n_out, p, t, f, x', y')
            case (1, n_out):
                in_array = in_array.reshape(sub_shape + in_shape)
                sub_array = jnp.zeros((n_out,) + sub_shape + out_shape, dtype=in_array.dtype)
                loop_over_n_out = lambda i, o: o.at[i].set(map_func(in_array, o[i], rel_coos[i], order))
                sub_array = fori_loop(0, n_out, loop_over_n_out, sub_array)
                out_array = sub_array.reshape((n_out,) + ptf_shape + out_shape)

            # (n_in, p, t, f, x, y) -> (n_out, p, t, f, x', y') if n_in == n_out
            case (n_in, n_out) if n_in == n_out:
                in_array = in_array.reshape((n_in,) + sub_shape + in_shape)
                sub_array = jnp.zeros((n_out,) + sub_shape + out_shape, dtype=in_array.dtype)
                loop_over_n_out = lambda i, o: o.at[i].set(map_func(in_array[i], o[i], rel_coos[i], order))
                sub_array = fori_loop(0, n_out, loop_over_n_out, sub_array)
                out_array = sub_array.reshape((n_out,) + ptf_shape + out_shape)

            case _:
                raise ValueError(
                    "Number of copies for input space and output space must match, or one of them must be 1."
                )

        if np_type:
            return np.asarray(out_array)

    return out_array
