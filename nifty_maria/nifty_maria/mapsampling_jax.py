"""
Module collecting loose set of jax-ified and jit-ed mapsampling functions.
"""

import jax
import numpy as np

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

from maria.units.constants import k_B
from maria.instrument import beam
import maria

instrument = maria.get_instrument('MUSTANG-2')

from maria.instrument import Band

def get_mini():
    
    f090 = Band(center=92, # in GHz
                width=40.0,
                knee=1,
                sensitivity=6e-5) # in K sqrt(s)

    # array = {"field_of_view": 1.0, "bands": [f090], "primary_size": 50, "beam_spacing": 2} # AtLAST
    array = {"field_of_view": 1.0, "bands": [f090], "primary_size": 4, "beam_spacing": 8} # Dummy setup with less detectors

    global instrument
    # instrument = maria.get_instrument(array=array, primary_size=50, beam_spacing = 2)
    instrument = maria.get_instrument(array=array)

    return instrument

def get_atlast():
    
    f090 = Band(center=92, # in GHz
                width=40.0,
                knee=1,
                sensitivity=6e-5) # in K sqrt(s)

    array = {"field_of_view": 1.0, "bands": [f090], "primary_size": 50, "beam_spacing": 2} # AtLAST

    global instrument
    # instrument = maria.get_instrument(array=array, primary_size=50, beam_spacing = 2)
    instrument = maria.get_instrument(array=array)

    return instrument

def get_dummy():
    
    f090 = Band(center=92, # in GHz
                width=40.0,
                knee=1,
                sensitivity=6e-5) # in K sqrt(s)

    # array = {"field_of_view": 1.0, "bands": [f090], "primary_size": 50, "beam_spacing": 2} # AtLAST
    array = {"field_of_view": 1.0, "bands": [f090], "primary_size": 50, "beam_spacing": 4} # Dummy setup with less detectors

    global instrument
    # instrument = maria.get_instrument(array=array, primary_size=50, beam_spacing = 2)
    instrument = maria.get_instrument(array=array)

    return instrument

@jax.jit
def sample_maps(sim_truthmap, dx, dy, resolution, x_side, y_side):

    data_map = jax.numpy.array(1e-16 * np.random.standard_normal(size=dx.shape))
    pbar = instrument.bands

    for band in pbar:
        band_mask = instrument.dets.band_name == band.name

        nu = jax.numpy.linspace(band.nu_min, band.nu_max, 64)

        TRJ = jax.scipy.interpolate.RegularGridInterpolator(
            (jax.numpy.array([100.]),),
            sim_truthmap,
            fill_value=None,
            bounds_error=False,
            method='nearest',
        )(nu)

        TRJ = jax.numpy.reshape(TRJ, (TRJ.shape[1], TRJ.shape[0], TRJ.shape[2], TRJ.shape[3]))
        nu_passband = jax.numpy.exp(jax.numpy.log(0.5) * (2 * (nu - 90.) / 30.) ** 2)

        power_map = (
            1e12
            * k_B
            * jax.numpy.trapezoid(nu_passband[:, None, None] * TRJ, axis=1, x=1e9 * nu)
        )

        # nu is in GHz, f is in Hz
        nu_fwhm = beam.compute_angular_fwhm(
            # fwhm_0=sim_truthmap.instrument.dets.primary_size.mean(),
            fwhm_0=instrument.dets.primary_size.mean(),
            z=np.inf,
            # f=1e9 * band.center,
            nu = band.center
        ) 
        
        nu_map_filter = construct_beam_filter(fwhm=nu_fwhm, res=resolution)
        filtered_power_map = separably_filter_2d(power_map, nu_map_filter)

        # assume no time-dim for now. TODO: Add time dim!
        map_power = jax.scipy.interpolate.RegularGridInterpolator(
            # Need to invert x_side and y_side for jax interpolation:
            # (jax.numpy.flip(x_side), jax.numpy.flip(y_side)), # length N=2 sequence of arrays with grid coords
            # jax.numpy.flip(filtered_power_map), # N=2-dimensional array specifying grid values (1000, 1000)
            (x_side, y_side), # length N=2 sequence of arrays with grid coords
            filtered_power_map[0], # N=2-dimensional array specifying grid values (1000, 1000)
            fill_value=0.,
            bounds_error=False,
            method="linear",
        )((jax.numpy.array(dx[band_mask]), jax.numpy.array(dy[band_mask])))

        # jax.debug.print("Total map power: {pwr}", pwr=map_power.sum())

        # data["map"][band_mask] += map_power
        data_map = data_map.at[band_mask].add(map_power)
        
    # return sim_truthmap.data["map"]
    return data_map