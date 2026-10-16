"""Self-contained map-reconstruction demo with NIFTy.

1. Load maria's ``cluster2.fits`` sky map (the map of ``steering/config/sim_mustang.yml``)
   and the pointing of a MUSTANG-2 daisy scan from maria.
2. Project the map into time-ordered data (TODs) with the maria-nifty
   ``map_coordinates`` response (redefined below) and add white noise.
3. Reconstruct the map with a correlated-field (Gaussian process) prior and a
   Gaussian likelihood via variational inference (``nifty.re.optimize_kl``).

Nothing is imported from ``maria_nifty``. Run from the repository root:

    pixi run -e cpu python demo/demo.py

Outputs go to ``ODIR`` (set below): ``files/`` holds the NIFTy checkpoints and
``results.npz``, ``plots/`` the figures (see ``plot.py``, which can also redraw
them from ``results.npz`` alone).
"""

import os

# CPU by default; set e.g. CUDA_VISIBLE_DEVICES=0 DEMO_DEVICE=gpu to run on a GPU.
if os.environ.get("DEMO_DEVICE", "cpu") == "cpu":
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("MARIA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "maria-data"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import maria  # noqa: E402
import nifty.re as jft  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.ndimage import map_coordinates  # noqa: E402
from jax.scipy.signal import convolve2d  # noqa: E402
from maria.coords import Coordinates  # noqa: E402
from maria.utils import get_rotation_matrix_2d  # noqa: E402
from plot import plot_all  # noqa: E402

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Settings
# =============================================================================

# Sky map, as in steering/config/sim_mustang.yml except for the scale.
MAP_FILE = "maps/cluster2.fits"  # fetched from the maria-data repository
MAP_SCALE = 0.5  # -> ~3 mK_RJ peak decrement, comparable to RXJ1347
MAP_CENTER = [144.0, 12.0]  # ra, dec in degrees; also the scan center
MAP_WIDTH = 0.08714  # degrees
MAP_RES = 8.714e-05  # degrees per pixel of the loaded map (1024 x 1024)
DOWNSAMPLE = 2  # block-average to 512 x 512 pixels for the reconstruction

# Scan, as in sim_mustang.yml except for the lower sample rate.
SCAN = dict(
    scan_pattern="daisy",
    scan_options={"radius": 0.0333, "speed": 0.008333},  # degrees, degrees / s
    duration=500,  # seconds
    sample_rate=20,  # Hz
    start_time="2024-02-10T06:00:00.000",
)

# The N_DETECTORS detectors closest to the array center are used (1: the central
# detector, 7: one small hexagon, 217: the full MUSTANG-2 array).
N_DETECTORS = 1
NOISE_STD = 0.001  # white noise per TOD sample, K_RJ (real MUSTANG-2 data: ~0.02)

# Variational inference (MGVI); cf. the `minimization` section of the configs in steering/config/.
N_ITERATIONS = 4
N_SAMPLES = 4  # mirrored -> 16 posterior samples
DRAW_LINEAR_KWARGS = dict(cg_name="SL", cg_kwargs=dict(absdelta=1e-11, maxiter=500))
KL_KWARGS = dict(minimize_kwargs=dict(name="M", absdelta=1e-12, cg_kwargs=dict(name=None), maxiter=10))

SEED = 42

# Output folder; change it to keep the results of a previous run.
ODIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "test_1")
FILES_DIR = os.path.join(ODIR, "files")  # NIFTy checkpoints and results.npz
PLOTS_DIR = os.path.join(ODIR, "plots")


# =============================================================================
# Response: map -> TODs
# =============================================================================


def gaussian_filter(image, sigma, truncate=4.0):
    """Zero-padded Gaussian smoothing of a 2D image (``sigma`` in pixels)."""
    radius = int(truncate * sigma + 0.5)
    x = jnp.arange(-radius, radius + 1)
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    image = convolve2d(image, kernel[:, None], mode="same")  # separable: one pass per axis
    return convolve2d(image, kernel[None, :], mode="same")


def offsets2coords(offsets, shape, fov):
    """Sky offsets ``(n_det, n_time, 2)`` in rad -> pixel coordinates ``(2, n_det, n_time)``.

    Map axis 0 is x (first offset component), axis 1 is y; offset zero is the map center.
    """
    pixel_size = np.asarray(fov) / np.asarray(shape)
    coords = np.asarray(offsets) / pixel_size + 0.5 * (np.asarray(shape) - 1)
    return coords.transpose(2, 0, 1)


def map2tods(sky, coords, sigma_pixels):
    """Beam-smooth the map and interpolate it bilinearly at the detector positions.

    Samples outside the map read zero.
    """
    smoothed = gaussian_filter(sky, sigma_pixels)
    return map_coordinates(smoothed, coords, order=1, mode="constant", cval=0.0)


# =============================================================================
# Inputs from maria
# =============================================================================


def load_true_map():
    """The sky map in K_RJ on the reconstruction grid, and its field of view in rad.

    Axis 0 of the returned map is x, axis 1 is y.
    """
    sky_map = maria.map.load(
        filename=maria.io.fetch(MAP_FILE),
        resolution=MAP_RES,
        width=MAP_WIDTH,
        center=MAP_CENTER,
        frame="ra_dec",
        nu=93.0e9,
        units="Jy/pixel",
    ).to("K_RJ")

    true_map = MAP_SCALE * np.asarray(sky_map.data, dtype=float).reshape(sky_map.data.shape[-2:]).T  # (y, x) -> (x, y)
    n = true_map.shape[0] // DOWNSAMPLE
    true_map = true_map.reshape(n, DOWNSAMPLE, n, DOWNSAMPLE).mean(axis=(1, 3))
    return true_map, np.radians([MAP_WIDTH, MAP_WIDTH])


def simulate_scan():
    """Pointing and beam of the daisy scan (no maria simulation is run).

    The boresight of the scan plan is broadcast to the selected detectors of the
    (rolled) array, as in ``maria.sim.Observation``.

    Returns
    -------
    offsets : numpy.ndarray, shape (n_det, n_time, 2)
        Sky offsets (x, y) of every sample from the map center, in rad.
    det_offsets : numpy.ndarray, shape (n_det, 2)
        Focal-plane positions of the detectors, in rad.
    beam_sigma : float
        Gaussian beam width, in rad.
    """
    instrument = maria.get_instrument("mustang-2")
    plan = maria.get_plan(scan_center=MAP_CENTER, frame="ra_dec", **SCAN)
    site = maria.get_site("green_bank")

    det_offsets = np.asarray(instrument.dets.offsets, dtype=float)
    det_offsets = det_offsets[np.argsort(np.hypot(*det_offsets.T), kind="stable")[:N_DETECTORS]]

    boresight = Coordinates(
        t=plan.time, phi=plan.phi, theta=plan.theta, earth_location=site.earth_location, frame=plan.frame
    )
    rotation = get_rotation_matrix_2d(np.radians(plan.roll))
    coords = boresight.broadcast(det_offsets @ rotation.T, frame="az/el")
    offsets = np.asarray(coords.offsets(frame="ra_dec", center=np.radians(MAP_CENTER)), dtype=float)
    offsets = offsets.reshape(N_DETECTORS, -1, 2)  # maria drops the detector axis for a single detector

    beam_sigma = float(instrument.dets.fwhm[0].rad) / np.sqrt(8 * np.log(2))
    return offsets, det_offsets, beam_sigma


# =============================================================================
# Model
# =============================================================================


def build_sky_model(shape):
    """2D Gaussian process with a power-law power spectrum, in K_RJ."""
    cfm = jft.CorrelatedFieldMaker("sky ")
    cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(1e-3, 5e-4))
    cfm.add_fluctuations(
        shape,
        distances=1.0 / shape[0],
        fluctuations=(1e-3, 5e-4),
        loglogavgslope=(-4.0, 1.0),
        flexibility=None,
        asperity=None,
        prefix="",
        non_parametric_kind="power",
    )
    return cfm.finalize()


# =============================================================================
# Main
# =============================================================================


def main():
    os.makedirs(FILES_DIR, exist_ok=True)
    key = jax.random.PRNGKey(SEED)

    # 1. inputs from maria
    true_map, fov = load_true_map()
    offsets, det_offsets, beam_sigma = simulate_scan()
    sigma_pixels = float(beam_sigma / (fov[0] / true_map.shape[0]))
    coords = jnp.asarray(offsets2coords(offsets, true_map.shape, fov))
    print(f"map {true_map.shape}, beam sigma {sigma_pixels:.2f} px")
    print(f"TODs: {offsets.shape[0]} detector(s) x {offsets.shape[1]} samples")

    # 2. synthetic data: response of the true map plus white noise
    key, k_noise = jax.random.split(key)
    signal = map2tods(jnp.asarray(true_map), coords, sigma_pixels)
    data = signal + NOISE_STD * jax.random.normal(k_noise, signal.shape)

    # 3. forward model and Gaussian likelihood
    sky = build_sky_model(true_map.shape)
    forward = jft.Model(lambda x: map2tods(sky(x), coords, sigma_pixels), domain=sky.domain, init=sky.init)
    likelihood = jft.Gaussian(data, noise_cov_inv=lambda x: x / NOISE_STD**2).amend(forward)

    # 4. variational inference, starting close to the prior mean
    key, k_init, k_kl = jax.random.split(key, 3)
    pos_init = jax.tree_util.tree_map(lambda x: 0.1 * x, jft.Vector(likelihood.init(k_init)))
    samples, _ = jft.optimize_kl(
        likelihood,
        pos_init,
        n_total_iterations=N_ITERATIONS,
        n_samples=N_SAMPLES,
        key=k_kl,
        draw_linear_kwargs=DRAW_LINEAR_KWARGS,
        kl_kwargs=KL_KWARGS,
        sample_mode="linear_resample",
        odir=FILES_DIR,
        resume=False,
    )

    # 5. save the posterior products and plot them
    fname = os.path.join(FILES_DIR, "results.npz")
    np.savez(
        fname,
        true_map=true_map,
        map_samples=np.stack([np.asarray(sky(s)) for s in samples]),
        data=np.asarray(data),
        signal=np.asarray(signal),
        tod_mean=np.mean([np.asarray(forward(s)) for s in samples], axis=0),
        offsets=offsets,
        det_offsets=det_offsets,
        fov=fov,
        sigma_pixels=sigma_pixels,
        sample_rate=SCAN["sample_rate"],
        noise_std=NOISE_STD,
    )
    print(f"saved {fname}")
    plot_all(fname, PLOTS_DIR)


if __name__ == "__main__":
    main()
