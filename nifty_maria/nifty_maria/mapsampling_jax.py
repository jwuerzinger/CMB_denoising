"""
Module collecting loose set of jax-ified and jit-ed mapsampling functions.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import correlate2d
jax.config.update("jax_enable_x64", True)

from functools import partial

@partial(jax.jit, static_argnames=['radius'])
def gaussian_kernel2d(sigma, radius):
    '''
    Re-implementation of scipy.ndimage.gaussian_kernel2d
    
    Parameters
    ----------
    sigma : float
        standard deviation of the gaussian kernel
    radius : int
        radius of the kernel

    Returns
    -------
    y : ndarray
        2D gaussian kernel
    '''
    x, y = jnp.meshgrid(jnp.arange(-radius, radius+1),
                        jnp.arange(-radius, radius+1))
    dst = jnp.sqrt(x**2+y**2)
    normal = 1/(2 * jnp.pi * sigma**2)
    return jnp.exp(-(dst**2 / (2.0 * sigma**2))) * normal
 
@partial(jax.jit, static_argnames=['radius'])
def gaussian_filter2d(x, sigma, radius=5):
    '''
    Re-implementation of scipy.ndimage.gaussian_filter2d

    Parameters
    ----------
    x : ndarray
        2D array to be filtered
    sigma : float
        standard deviation of the gaussian kernel
    radius : int
        radius of the kernel. Should be something like `int(4*sigma + 0.5)`. Default is 5.
    
    Returns
    -------
    y : ndarray
        2D gaussian filter
    '''
    def true_branch(x, sigma):
        k = gaussian_kernel2d(sigma, radius)
        y = correlate2d(x, k, 'same')
        return y

    def false_branch(x, sigma):
        return x

    return lax.cond(sigma > 0, true_branch, false_branch, x, sigma)

@partial(jax.jit, static_argnames=['instrument'])
def sample_maps(sim_truthmap, instrument, offsets, resolution, x_side, y_side, pW_per_K_RJ):

    # sigma_rad = instrument.dets.fwhm[0]/ jax.numpy.sqrt(8 * jax.numpy.log(2))
    # sigma_pixels = sigma_rad/resolution
    # sim_truthmap_smoothed = gaussian_filter2d(sim_truthmap, sigma_pixels, radius=16)
    sim_truthmap_smoothed = sim_truthmap
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