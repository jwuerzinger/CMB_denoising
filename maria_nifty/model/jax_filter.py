"""JAX-compatible Gaussian filtering helpers used by response models."""

import numbers
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.signal import correlate
from scipy.ndimage import _ni_support


def vmap_correlate(input, weights, axis=-1, mode="same"):
    """Apply 1D correlation along an axis using a vectorized JAX correlate.

    Parameters
    ----------
    input : jnp.ndarray
        Input array of arbitrary shape.
    weights : jnp.ndarray, shape (k,)
        1D convolution kernel.
    axis : int, optional
        Axis along which to apply the correlation (default ``-1``).
    mode : str, optional
        Correlation mode forwarded to :func:`jax.scipy.signal.correlate`.

    Returns
    -------
    jnp.ndarray
        Array with the same shape as ``input``.
    """
    input_moved = jnp.moveaxis(input, axis, -1)
    input_flat = input_moved.reshape(-1, input_moved.shape[-1])
    result_flat = jax.vmap(correlate, in_axes=(0, None, None))(input_flat, weights, mode)
    result_moved = result_flat.reshape(input_moved.shape)
    result = jnp.moveaxis(result_moved, -1, axis)
    return result


def _gaussian_kernel1d(sigma, order, radius):
    """Compute a 1D Gaussian convolution kernel with optional derivatives.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    order : int
        Derivative order; 0 returns the plain Gaussian kernel.
    radius : int
        Half-width of the kernel; output has length ``2 * radius + 1``.

    Returns
    -------
    jnp.ndarray, shape (2 * radius + 1,)
        Normalized Gaussian kernel (or its ``order``-th derivative).
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = jnp.arange(order + 1)
    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = jnp.zeros(order + 1).at[0].set(1)
        D = jnp.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = jnp.diag(jnp.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def gaussian_filter1d(input, sigma, axis=-1, order=0, mode="constant", cval=0.0, truncate=4.0, *, radius=None):
    """Apply a 1D Gaussian filter along the requested axis.

    Parameters
    ----------
    input : jnp.ndarray
        Input array.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.
    axis : int, optional
        Axis to filter (default ``-1``).
    order : int, optional
        Derivative order of the filter (default 0 = plain smoothing).
    mode : str, optional
        Border handling; only ``"constant"`` is currently implemented.
    cval : float, optional
        Constant fill value used at borders (only effective when ``mode="constant"``).
    truncate : float, optional
        Truncate the kernel at this many standard deviations.
    radius : int or None, optional
        Explicit kernel half-width; overrides ``truncate`` when given.

    Returns
    -------
    jnp.ndarray
        Filtered array with the same shape as ``input``.
    """
    if mode != "constant" and cval != 0.0:
        raise NotImplementedError("Only constant mode with cval=0.0 is implemented for now.")

    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError("Radius must be a nonnegative integer.")
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return vmap_correlate(input, weights, axis, "same")


@partial(jax.jit, static_argnames=("sigma", "order", "mode", "cval", "truncate", "radius", "axes"))
def gaussian_filter(input, sigma, order=0, mode="constant", cval=0.0, truncate=4.0, *, radius=None, axes=None):
    """Apply a multidimensional Gaussian filter to a JAX array.

    JIT-compiled drop-in for :func:`scipy.ndimage.gaussian_filter`.

    Parameters
    ----------
    input : jnp.ndarray
        Input array to filter.
    sigma : float or sequence of float
        Standard deviation(s) of the Gaussian kernel; one value per axis.
    order : int or sequence of int, optional
        Derivative order(s) per axis (default 0).
    mode : str or sequence of str, optional
        Border handling per axis; only ``"constant"`` is implemented.
    cval : float, optional
        Constant fill value at borders.
    truncate : float, optional
        Truncate the kernel at this many standard deviations.
    radius : int or sequence of int or None, optional
        Explicit kernel half-widths; override ``truncate`` when given.
    axes : sequence of int or None, optional
        Axes to filter; defaults to all axes.

    Returns
    -------
    jnp.ndarray
        Filtered array with the same shape as ``input``.
    """
    if mode != "constant" and cval != 0.0:
        raise NotImplementedError("Only constant mode with cval=0.0 is implemented for now.")

    output = jnp.asarray(input)

    axes = _ni_support._check_axes(axes, input.ndim)

    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    radiuses = _ni_support._normalize_sequence(radius, num_axes)
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii]) for ii in range(num_axes) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode, radius in axes:
            output = gaussian_filter1d(output, sigma, axis, order, mode, cval, truncate, radius=radius)

    return output
