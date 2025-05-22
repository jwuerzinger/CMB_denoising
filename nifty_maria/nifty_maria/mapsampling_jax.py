"""
Module collecting loose set of jax-ified and jit-ed mapsampling functions.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import correlate2d
jax.config.update("jax_enable_x64", True)

from functools import partial

# jax compatible rewrite if beams.separably_filter
@jax.jit
def separably_filter_2d(data, F, tol=1e-2, return_filter=False):
    """
    This is more efficient than 2d convolution
    """

    if F.ndim != 2:
        raise ValueError("F must be two-dimensional.")

    u, s, v = jax.numpy.linalg.svd(F)
    effective_filter = 0
    filtered_image = 0

    for m in range(len(F)):
        effective_filter += s[m] * u[:, m : m + 1] @ v[m : m + 1]
        
        u_kernel = jax.numpy.broadcast_to(u[:, m], (1, u[:, m].size))
        v_kernel = jax.numpy.broadcast_to(v[m], (1, v[m].size))
        # v_kernel = jax.numpy.broadcast_to(u[:, m], (1, v[m].size))

        filtered_image += s[m] * jax.scipy.signal.convolve(
            jax.scipy.signal.convolve(data[0], u_kernel.T, mode='same', precision='default'), 
            v_kernel, mode='same', precision='default'
            )

        # if np.abs(F - effective_filter).mean() < tol: break # may not be necessary?!

    # re-add 0th dim:
    filtered_image = filtered_image[None, :, :]

    return (filtered_image, effective_filter) if return_filter else filtered_image

@jax.jit
def construct_beam_filter(fwhm, res, buffer=1):
    """
    Make a beam filter for an image.
    """

    # if beam_profile is None:
    #     # beam_profile = lambda r, r0: np.where(r <= r0, 1., 0.)

    #     # a top hat
    #     def beam_profile(r, r0):
    #         return np.exp(-((r / r0) ** 16))

    filter_width = buffer * fwhm

    # n_side = jax.numpy.maximum(filter_width / res, 3).astype(int) # n_side = 27 -> having this as var breaks jit...
    # filter_side = jax.numpy.linspace(-filter_width / 2, filter_width / 2, n_side)

    filter_side = jax.numpy.linspace(-filter_width / 2, filter_width / 2, 27)

    X, Y = jax.numpy.meshgrid(filter_side, filter_side, indexing="ij")
    R = jax.numpy.sqrt(jax.numpy.square(X) + jax.numpy.square(Y))
    F = jax.numpy.exp(-((R / (fwhm / 2)) ** 16))

    return F / F.sum()

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

    # print("MARKER:", sim_truthmap.shape)
    # sim_truthmap = sim_truthmap[0, 0, :]
    # sim_truthmap = sim_truthmap[..., -2:]
    # sim_truthmap = sim_truthmap.reshape(-1, *sim_truthmap.shape[-1:])
    # print("MARKER:", sim_truthmap.dtype)

    sigma_rad = instrument.dets.fwhm[0]/ jax.numpy.sqrt(8 * jax.numpy.log(2))
    sigma_pixels = sigma_rad/resolution
    # sim_truthmap_smoothed = jax.scipy.ndimage.gaussian_filter(sim_truthmap, sigma=(sigma_pixels, sigma_pixels), axes=(-2, -1))
    sim_truthmap_smoothed = gaussian_filter2d(sim_truthmap, sigma_pixels, radius=16)
    # data_map = jax.numpy.array(1e-16 * np.random.standard_normal(size=offsets.shape[:-1]))
    pbar = instrument.bands

    for band in pbar:
        band_mask = instrument.dets.band_name == band.name

        samples_K_RJ = jax.scipy.interpolate.RegularGridInterpolator(
            (y_side[::-1], x_side),
            sim_truthmap_smoothed[::-1],
            bounds_error=False,
            fill_value=0,
            method="nearest",
        )((offsets[band_mask, ..., 0], offsets[band_mask, ..., 1]))

        loading = pW_per_K_RJ * samples_K_RJ

    return loading