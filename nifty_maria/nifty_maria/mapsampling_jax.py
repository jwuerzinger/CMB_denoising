"""
Module collecting loose set of jax-ified and jit-ed mapsampling functions.
"""

import jax
import numbers
import jax.numpy as jnp
from functools import partial
from jax.scipy.signal import correlate
from scipy.ndimage import _ni_support


def vmap_correlate(input, weights, axis=-1, mode='same'):
    """
    Applies a 1D correlation to the input array along the specified axis using JAX's vmap.
    """
    input_moved = jnp.moveaxis(input, axis, -1)
    input_flat = input_moved.reshape(-1, input_moved.shape[-1])
    result_flat = jax.vmap(correlate, in_axes=(0, None, None))(input_flat, weights, mode)
    result_moved = result_flat.reshape(input_moved.shape)
    result = jnp.moveaxis(result_moved, -1, axis)
    return result


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = jnp.arange(order + 1)
    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius+1)
    phi_x = jnp.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = jnp.zeros(order + 1)
        q[0] = 1
        D = jnp.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = jnp.diag(jnp.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x
    

def gaussian_filter1d(input, sigma, axis=-1, order=0, mode="constant", cval=0.0, truncate=4.0, *, radius=None):
    """
    1D-Gauss-Filter.
    """
    if mode != "constant" and cval != 0.0:
        raise NotImplementedError("Only constant mode with cval=0.0 is implemented for now.")

    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return vmap_correlate(input, weights, axis, 'same')
    

@partial(jax.jit, static_argnames=('sigma', 'order', 'mode', 'cval', 'truncate', 'radius', 'axes'))
def gaussian_filter(input, sigma, order=0, mode="constant", cval=0.0, truncate=4.0, *, radius=None, axes=None):
    """
    Multidimensional Gaussian filter.
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
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
            for ii in range(num_axes) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode, radius in axes:
            output = gaussian_filter1d(output, sigma, axis, order, mode, cval, truncate, radius=radius)
     
    return output


@partial(jax.jit, static_argnames=['instrument', 'sigma_pixels'])
def sample_maps(sim_truthmap, instrument, offsets, sigma_pixels, x_side, y_side, pW_per_K_RJ):

    sim_truthmap_smoothed = gaussian_filter(sim_truthmap, sigma=sigma_pixels)
    pbar = instrument.bands

    for band in pbar:
        band_mask = instrument.dets.band_name == band.name

        sim_truthmap_smoothed = jax.numpy.flip(sim_truthmap_smoothed, axis=0)

        samples_K_RJ = jax.scipy.interpolate.RegularGridInterpolator(
            (y_side[::-1], x_side),
            sim_truthmap_smoothed[::-1],
            bounds_error=False,
            fill_value=0,
            method="linear",
        )((offsets[band_mask, ..., 1], offsets[band_mask, ..., 0]))

        loading = pW_per_K_RJ * samples_K_RJ
    
    return loading


# @jax.jit
# def separably_filter_2d(data, F, tol=1e-2, return_filter=False):
#     """
#     This is more efficient than 2d convolution
#     """

#     if F.ndim != 2:
#         raise ValueError("F must be two-dimensional.")

#     u, s, v = jax.numpy.linalg.svd(F)
#     effective_filter = 0
#     filtered_image = 0

#     for m in range(len(F)):
#         effective_filter += s[m] * u[:, m : m + 1] @ v[m : m + 1]
        
#         u_kernel = jax.numpy.broadcast_to(u[:, m], (1, u[:, m].size))
#         v_kernel = jax.numpy.broadcast_to(v[m], (1, v[m].size))
#         # v_kernel = jax.numpy.broadcast_to(u[:, m], (1, v[m].size))

#         filtered_image += s[m] * jax.scipy.signal.convolve(
#             jax.scipy.signal.convolve(data[0], u_kernel.T, mode='same', precision='default'), 
#             v_kernel, mode='same', precision='default'
#             )

#         # if np.abs(F - effective_filter).mean() < tol: break # may not be necessary?!

#     # re-add 0th dim:
#     filtered_image = filtered_image[None, :, :]

#     return (filtered_image, effective_filter) if return_filter else filtered_image

# @jax.jit
# def construct_beam_filter(fwhm, res, buffer=1):
#     """
#     Make a beam filter for an image.
#     """

#     # if beam_profile is None:
#     #     # beam_profile = lambda r, r0: np.where(r <= r0, 1., 0.)

#     #     # a top hat
#     #     def beam_profile(r, r0):
#     #         return np.exp(-((r / r0) ** 16))

#     filter_width = buffer * fwhm

#     # n_side = jax.numpy.maximum(filter_width / res, 3).astype(int) # n_side = 27 -> having this as var breaks jit...
#     # filter_side = jax.numpy.linspace(-filter_width / 2, filter_width / 2, n_side)

#     filter_side = jax.numpy.linspace(-filter_width / 2, filter_width / 2, 27)

#     X, Y = jax.numpy.meshgrid(filter_side, filter_side, indexing="ij")
#     R = jax.numpy.sqrt(jax.numpy.square(X) + jax.numpy.square(Y))
#     F = jax.numpy.exp(-((R / (fwhm / 2)) ** 16))

#     return F / F.sum()

# from maria.constants import k_B
# from maria import beam

# import numpy as np

# # @jax.jit
# # def sample_maps(sim_truthmap, dx, dy, resolution, x_side, y_side):
# def sample_maps(sim_truthmap, instrument, offsets, resolution, x_side, y_side, pW_per_K_RJ):

#     # NEW
#     sim_truthmap = sim_truthmap[None, None, :]
#     # (dx, dy) = (offsets[:, :, 0], offsets[:, :, 1])

#     # data_map = jax.numpy.array(1e-16 * np.random.standard_normal(size=dx.shape))
#     pbar = instrument.bands

#     for band in pbar:
#         band_mask = instrument.dets.band_name == band.name
        
#         dx = offsets[band_mask, ..., 0]
#         dy = offsets[band_mask, ..., 1]
#         data_map = jax.numpy.array(1e-16 * np.random.standard_normal(size=dx.shape))
#         # print("BAND:", band)

#         # nu = jax.numpy.linspace(band.nu_min, band.nu_max, 64)
#         nu = jax.numpy.linspace(60., 120., 64)

#         TRJ = jax.scipy.interpolate.RegularGridInterpolator(
#             (jax.numpy.array([100.]),),
#             sim_truthmap,
#             fill_value=None,
#             bounds_error=False,
#             method='nearest',
#         )(nu)
        
#         # print("TRJ shape:", TRJ.shape)

#         TRJ = jax.numpy.reshape(TRJ, (TRJ.shape[1], TRJ.shape[0], TRJ.shape[2], TRJ.shape[3]))
#         nu_passband = jax.numpy.exp(jax.numpy.log(0.5) * (2 * (nu - 90.) / 30.) ** 2)

#         power_map = (
#             1e12
#             * k_B
#             * jax.numpy.trapezoid(nu_passband[:, None, None] * TRJ, axis=1, x=1e9 * nu)
#         )

#         # print("POWER_MAP:", power_map)
#         # nu is in GHz, f is in Hz
#         nu_fwhm = beam.compute_angular_fwhm(
#             # fwhm_0=sim_truthmap.instrument.dets.primary_size.mean(),
#             fwhm_0=instrument.dets.primary_size.mean(),
#             z=np.inf,
#             # f=1e9 * band.center,
#             nu = band.center
#         ) 
        
#         nu_map_filter = construct_beam_filter(fwhm=nu_fwhm, res=resolution)
#         filtered_power_map = separably_filter_2d(power_map, nu_map_filter)

#         # assume no time-dim for now. TODO: Add time dim!
#         map_power = jax.scipy.interpolate.RegularGridInterpolator(
#             # Need to invert x_side and y_side for jax interpolation:
#             # (jax.numpy.flip(x_side), jax.numpy.flip(y_side)), # length N=2 sequence of arrays with grid coords
#             (y_side[::-1], x_side),
#             # jax.numpy.flip(filtered_power_map), # N=2-dimensional array specifying grid values (1000, 1000)
#             # (x_side, y_side), # length N=2 sequence of arrays with grid coords
#             filtered_power_map[0], # N=2-dimensional array specifying grid values (1000, 1000)
#             fill_value=0.,
#             bounds_error=False,
#             method="linear",
#         # )((jax.numpy.array(dx[band_mask]), jax.numpy.array(dy[band_mask])))
#         )((jax.numpy.array(dy[band_mask]), jax.numpy.array(dx[band_mask])))

#         # jax.debug.print("Total map power: {pwr}", pwr=map_power.sum())

#         # data["map"][band_mask] += map_power
#         # print("MAP_POWER:", map_power)
        
#         data_map = data_map.at[band_mask].add(map_power)
        
#         data_map = pW_per_K_RJ * data_map * (6.0/2.5)
        
#     # return sim_truthmap.data["map"]
#     return data_map