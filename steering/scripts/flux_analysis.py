"""Flux/quantity measurements on an existing maria-nifty reconstruction.

Step 1: load a reconstruction -- either a ``joint_fit`` run dir
(``--fit_dir``: ``config.yaml`` + ``last.pkl``, the map model is rebuilt and
evaluated per sample) or a ``save_fits.py`` export (``--fits_dir``: the
per-sample maps are read straight from ``map_samples.fits``, no model or
``last.pkl`` needed) -- and compute the integrated flux of the
reconstructed map, in both native K_RJ and Jy, using the same
Rayleigh-Jeans conversion as ``flux_uncertainty.py`` and ``decrement_flux.py``
(``S_nu = 2 k_B nu^2 T_RJ / c^2 * Omega``, maria's own convention). The
quantity is evaluated on every stored posterior sample and reported as
mean +/- std across samples, the same propagation ``flux_uncertainty.py``
uses.

If the map model has more than one component (e.g. ``s1`` diffuse + ``s2``
compact source), the integrated flux is reported for each component
separately as well as for the full (summed) map -- with one component the two
are the same thing.

Step 2 (optional, ``--alma_compact_fits``/``--alma_decrement_fits``): compare
the compact source and the SZ decrement against ALMA (Kitayama et al. 2016,
arXiv:1607.08833, "K16"): the Fig. 1 bottom-left map (7-m array *dirty* map,
before AGN subtraction) for the compact source, the point-source-subtracted,
primary-beam-corrected, 5"-smoothed Fig. 4 map for the decrement. Each ALMA
map is measured in its own Jy/beam units; the matching NIFTy component
(``--compact_comp``/``--decrement_comp``) is first convolved with that ALMA
map's own (elliptical) restoring beam, so both peaks are quoted per the same
beam. Aperture fluxes are summed within a circle of the same radius in both
maps. K16's own published numbers are printed alongside:

* compact source: the AGN flux from a point-source fit to the 12-m long
  baselines (4.16 mJy at 86 GHz, 3.96 mJy at 98 GHz, alpha = -0.38, Sect. 3.2),
  power-law interpolated to ``--freq_ghz``. The Fig. 1 dirty-map peak is only
  indicative -- it still contains the (negative) SZ signal under the AGN, and
  no aperture flux is quoted for it (a dirty beam integrates to ~0).
* decrement: the peak -0.286 +/- 0.017 +/- 0.017 mJy/beam at the south-east
  peak on the 5" map, and -12.4 +/- 0.2 +/- 0.8 mJy within r < 40" of that
  same south-east peak (Sect. 3.4).

Map positions are tied to the AGN, so no absolute astrometry is needed for
the NIFTy map: in NIFTy the AGN is the compact component's posterior-mean
peak, and the decrement aperture sits at K16's SE-peak-minus-AGN sky offset
(``--agn_radec``/``--sze_peak_radec``) from it. In ALMA both positions are
looked up through the FITS file's own WCS.

The ALMA decrement map is interferometric (shortest baseline 2.1 klambda,
K16: ~80% of the flux lost at the largest scales, accurate only for 5"-40"),
so its aperture flux is expected to fall short of NIFTy's -- the peak at the
shared 5" beam is the fairer comparison. NIFTy is converted at
``--freq_ghz``; ALMA is quoted as-is at its own ~92 GHz.

Note on sample counts: NIFTy stores ``2 * n_samples`` states as antithetic
(mirrored) pairs, so the *effective* number of independent draws is
``n_samples``, not the stored count -- see ``flux_uncertainty.py``'s
docstring for the same caveat.

Example::

    pixi run -e cpu python scripts/flux_analysis.py \\
        --fit_dir runs/RXJ1347/AGBT17A_358_test/10tods_10amin_8s_nl/run_4/joint_fit \\
        --alma_compact_fits K16_Fig1bottomleft_bunitx1000.fits \\
        --alma_decrement_fits K16_Fig4_bunitx1000.fits

    # the same, from save_fits.py's FITS export instead of last.pkl
    pixi run -e cpu python scripts/flux_analysis.py \\
        --fits_dir runs/RXJ1347/AGBT17A_358_test/10tods_10amin_8s_nl/plot_4/save_fits \\
        --alma_compact_fits K16_Fig1bottomleft_bunitx1000.fits \\
        --alma_decrement_fits K16_Fig4_bunitx1000.fits
"""

import os

import click

K_B = 1.380649e-23  # J/K
C = 2.99792458e8  # m/s
JY = 1e-26  # W / m^2 / Hz
ARCSEC = 3.141592653589793 / (180 * 3600)  # rad

# Kitayama et al. (2016, arXiv:1607.08833) published values.
K16_AGN = ((86.0, 4.16), (98.0, 3.96))  # (GHz, mJy), 12-m point-source fit, Sect. 3.2
K16_AGN_ERR = (0.03, 0.25)  # mJy, (stat, sys)
K16_AGN_ALPHA = -0.38
K16_SZ_PEAK = (-0.286, 0.017, 0.017)  # mJy/beam (value, stat, sys) at the SE peak, 5" beam
K16_SZ_INT = (-12.4, 0.2, 0.8)  # mJy (value, stat, sys) within r < 40" of the SE peak
K16_NOISE = 0.017  # mJy/beam, Fig. 4 1-sigma


def rj_to_flux(T_RJ, nu, solid_angle):
    """S_nu [Jy] = 2 k_B nu^2 T_RJ / c^2 * solid_angle, per maria's own convention."""
    return 2 * K_B * nu**2 * T_RJ / C**2 * solid_angle / JY


def load_alma(path, data_scale):
    """An ALMA/CASA FITS map as ``(data [Jy/beam], beam, om_pix [sr], px [arcsec], freq [Hz], wcs)``.

    ``data`` stays in the FITS file's own orientation; only circular
    apertures and peaks are measured on it, so the axis convention does not
    matter.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data, dtype=float)
        wcs = WCS(hdr).celestial
    data = data.reshape(data.shape[-2:]) * float(data_scale)
    freq_hz = float(hdr["RESTFRQ"])
    for i in range(1, int(hdr.get("NAXIS", 0)) + 1):
        if str(hdr.get(f"CTYPE{i}", "")).upper().startswith("FREQ"):
            freq_hz = float(hdr[f"CRVAL{i}"])
    beam = {
        "bmaj": float(hdr["BMAJ"]) * 3600,  # arcsec FWHM
        "bmin": float(hdr["BMIN"]) * 3600,
        "bpa": float(hdr["BPA"]),  # deg, east of north
    }
    px_arcsec = abs(float(hdr["CDELT1"])) * 3600, abs(float(hdr["CDELT2"])) * 3600
    om_pix = px_arcsec[0] * px_arcsec[1] * ARCSEC**2
    return data, beam, om_pix, px_arcsec, freq_hz, wcs


def beam_solid_angle(beam):
    """Gaussian beam solid angle [sr] from its FWHM major/minor axes [arcsec]."""
    import numpy as np

    return np.pi / (4 * np.log(2)) * beam["bmaj"] * beam["bmin"] * ARCSEC**2


def beam_kernel(beam, px_arcsec):
    """Unit-peak elliptical Gaussian on a NIFTy ``[i, j]`` grid (``i`` -> +RA/east, ``j`` -> +Dec/north).

    Unit *peak* (not unit sum), so convolving a K_RJ-per-pixel map with it and
    dividing by the kernel's sum gives the beam-weighted mean temperature --
    i.e. the map as seen through that beam, still in K_RJ.
    """
    import numpy as np

    sig_maj = beam["bmaj"] / (2 * np.sqrt(2 * np.log(2))) / px_arcsec
    sig_min = beam["bmin"] / (2 * np.sqrt(2 * np.log(2))) / px_arcsec
    half = int(np.ceil(4 * sig_maj))
    x, y = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1), indexing="ij")
    pa = np.radians(beam["bpa"])
    # major axis points PA east of north: (east, north) = (sin PA, cos PA)
    u = x * np.sin(pa) + y * np.cos(pa)
    v = x * np.cos(pa) - y * np.sin(pa)
    return np.exp(-0.5 * ((u / sig_maj) ** 2 + (v / sig_min) ** 2))


def load_from_fit(fit_dir):
    """Per-sample full/component maps from a fit dir's ``config.yaml`` + ``last.pkl``.

    Returns ``(full_maps, comp_maps, px_rad, n_stored, n_eff)``: lists of
    model-orientation ``[i, j]`` arrays (``comp_maps`` keyed by the config's
    ``source_params`` names, empty if they can't be matched to the model's
    components), the pixel size in rad, and the stored/effective sample counts.
    """
    import numpy as np

    import maria_nifty as mani

    config = mani.yaml_load(os.path.join(fit_dir, "config.yaml"))
    samples = list(mani.load_samples(fit_dir))
    map_model = mani.MapModel.build(**config["map"])

    full_maps = [np.asarray(map_model(s), dtype=float) for s in samples]
    comp_names = list((config["map"].get("source_params") or {}).keys())
    comps = list(map_model.models)
    comp_maps = {}
    if len(comp_names) == len(comps):
        for name, comp in zip(comp_names, comps):
            comp_maps[name] = [np.asarray(comp(s, map=True), dtype=float) for s in samples]

    px_rad = float(np.asarray(map_model.space.distances)[0])
    return full_maps, comp_maps, px_rad, len(samples), int(config["minimization"]["n_samples"])


def load_from_fits(fits_dir):
    """Same as :func:`load_from_fit`, read from ``save_fits.py``'s ``map_samples.fits``.

    The ``FULL`` and per-component cubes hold exactly ``map_model(s)`` and
    ``comp(s, map=True)`` for every stored state, so nothing is re-evaluated.
    The effective draw count comes from the copied ``config.yaml`` when
    present, else from ``ANTITHET`` (mirrored pairs -> ``NSAMP / 2``).
    """
    from astropy.io import fits

    import maria_nifty as mani

    with fits.open(os.path.join(fits_dir, "map_samples.fits")) as hdul:
        hdr = hdul[0].header
        names = [hdr[f"COMP{i}"] for i in range(1, int(hdr["NCOMP"]) + 1)]
        full_maps = list(mani.from_fits_data(hdul["FULL"].data))
        comp_maps = {name: list(mani.from_fits_data(hdul[name.upper()].data)) for name in names}
        px_rad = float(hdr["PIXX"])
        n_stored = int(hdr["NSAMP"])
        antithetic = bool(hdr.get("ANTITHET", False))

    config_path = os.path.join(fits_dir, "config.yaml")
    n_eff = None
    if os.path.isfile(config_path):
        n_eff = (mani.yaml_load(config_path).get("minimization") or {}).get("n_samples")
    if n_eff is None:
        n_eff = n_stored // 2 if antithetic else n_stored
    return full_maps, comp_maps, px_rad, n_stored, int(n_eff)


def radial_mask(shape, center, radius_px):
    import numpy as np

    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius_px**2


@click.command()
@click.option("--fit_dir", default=None, help="A joint_fit run dir (config.yaml + last.pkl).")
@click.option(
    "--fits_dir",
    default=None,
    help="A save_fits.py export (plot_<run>/save_fits, with map_samples.fits) -- alternative to --fit_dir.",
)
@click.option("--freq_ghz", default=90.0, show_default=True, help="Effective observing frequency in GHz.")
@click.option(
    "--alma_compact_fits",
    default=None,
    help="ALMA map to compare the compact source against (e.g. K16_Fig1bottomleft_bunitx1000.fits).",
)
@click.option(
    "--alma_decrement_fits",
    default=None,
    help="ALMA map to compare the SZ decrement against (e.g. the point-source-subtracted K16_Fig4_bunitx1000.fits).",
)
@click.option(
    "--alma_scale",
    default=1e-3,
    show_default=True,
    help="Multiply the ALMA data by this on load -- 1e-3 for the *_bunitx1000 files, whose pixels are "
    "mJy/beam while BUNIT still says Jy/beam.",
)
@click.option("--compact_comp", default="s2", show_default=True, help="NIFTy map component of the compact source.")
@click.option("--decrement_comp", default="s1", show_default=True, help="NIFTy map component of the decrement.")
@click.option(
    "--compact_radius_arcsec",
    default=15.0,
    show_default=True,
    help="Aperture radius (arcsec) for the compact source's integrated flux.",
)
@click.option(
    "--decrement_radius_arcsec",
    default=40.0,
    show_default=True,
    help="Aperture radius (arcsec) for the decrement's integrated flux (K16 use 40\").",
)
@click.option(
    "--agn_radec",
    default="13h47m30.62s -11d45m09.5s",
    show_default=True,
    help="Sky position of the compact source (K16 Sect. 3.4).",
)
@click.option(
    "--sze_peak_radec",
    default="13h47m31.02s -11d45m18.4s",
    show_default=True,
    help="Sky position of the SZ south-east peak, the centre of the decrement aperture (K16 Sect. 3.4).",
)
@click.option("--cuda_device", default="", help="CUDA device selection; empty forces CPU mode.")
def main(
    fit_dir,
    fits_dir,
    freq_ghz,
    alma_compact_fits,
    alma_decrement_fits,
    alma_scale,
    compact_comp,
    decrement_comp,
    compact_radius_arcsec,
    decrement_radius_arcsec,
    agn_radec,
    sze_peak_radec,
    cuda_device,
):
    """Print the integrated flux (K_RJ and Jy) of a reconstruction's posterior-mean map."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import numpy as np
    from nifty.re import logger

    if (fit_dir is None) == (fits_dir is None):
        raise click.UsageError("Pass exactly one of --fit_dir or --fits_dir.")
    if fit_dir is not None:
        full_maps, comp_maps, px_rad, n_stored, n_eff = load_from_fit(fit_dir)
    else:
        full_maps, comp_maps, px_rad, n_stored, n_eff = load_from_fits(fits_dir)
    om_pix = px_rad**2
    nu = float(freq_ghz) * 1e9

    logger.info(
        f"\n{fit_dir or fits_dir}\n  stored samples {n_stored} (effective independent draws {n_eff})\n"
        f"  pixel solid angle {om_pix * (180 * 3600 / np.pi) ** 2:.4f} arcsec^2  @ {freq_ghz:.1f} GHz"
    )

    print(f"\n{'component':14s} {'T_RJ [uK]':>24s} {'flux [mJy]':>24s}")
    print(f"{'':14s} {'mean':>11s} {'std':>12s} {'mean':>11s} {'std':>12s}")
    print("-" * 66)

    def report(label, maps):
        sums_K = np.array([float(m.sum()) for m in maps])
        fluxes_jy = rj_to_flux(sums_K, nu, om_pix)
        t_mean, t_std = float(sums_K.mean()), float(sums_K.std(ddof=1)) if sums_K.size > 1 else float("nan")
        f_mean, f_std = float(fluxes_jy.mean()), float(fluxes_jy.std(ddof=1)) if fluxes_jy.size > 1 else float("nan")
        print(f"{label:14s} {t_mean * 1e6:>11.4f} {t_std * 1e6:>12.4f} {f_mean * 1e3:>11.4f} {f_std * 1e3:>12.4f}")
        return t_mean, t_std, f_mean, f_std

    for name, maps in comp_maps.items():
        report(name, maps)
    report("full map" if comp_maps else "map", full_maps)

    if alma_compact_fits or alma_decrement_fits:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from scipy.signal import fftconvolve

        if compact_comp not in comp_maps:
            raise click.UsageError(f"--compact_comp {compact_comp!r} not in map components {list(comp_maps)}")
        px_arcsec = px_rad / ARCSEC
        agn = SkyCoord(agn_radec, frame="fk5")
        sze_peak = SkyCoord(sze_peak_radec, frame="fk5")
        d_east, d_north = (a.to_value(u.arcsec) for a in agn.spherical_offsets_to(sze_peak))

        # NIFTy positions: the AGN is the compact component's posterior-mean
        # peak; the SE peak sits at K16's sky offset from it (i -> east, j -> north).
        compact_mean = np.mean(comp_maps[compact_comp], axis=0)
        nifty_agn = np.array(np.unravel_index(np.argmax(compact_mean), compact_mean.shape), dtype=float)
        nifty_sze = nifty_agn + np.array([d_east, d_north]) / px_arcsec

        def alma_pixel(wcs, coord):
            x, y = wcs.world_to_pixel(coord)
            return np.array([float(y), float(x)])  # data[row=y, col=x]

        def ms(a):
            return float(np.mean(a)) * 1e3, float(np.std(a, ddof=1)) * 1e3 if len(a) > 1 else float("nan")

        def nifty_through_beam(comp, beam):
            kern = beam_kernel(beam, px_arcsec)
            return [fftconvolve(m, kern, mode="same") / float(kern.sum()) for m in comp_maps[comp]]

        def header(label, comp, path, beam, freq_alma, radius_arcsec, centre_label):
            print(
                f"\n{label}: NIFTy {comp} vs. {os.path.basename(path)}\n"
                f'  ALMA beam {beam["bmaj"]:.2f}" x {beam["bmin"]:.2f}" (PA {beam["bpa"]:.1f} deg) @ '
                f"{freq_alma / 1e9:.2f} GHz; NIFTy at {freq_ghz:.1f} GHz, smoothed to that beam\n"
                f'  aperture r < {radius_arcsec:.1f}" around the {centre_label}'
            )
            print(f"  {'':34s} {'ALMA map':>10s} {'K16 published':>26s} {'NIFTy':>22s}")

        if alma_compact_fits:
            data, beam, om_pix_a, px_a, freq_a, wcs = load_alma(alma_compact_fits, alma_scale)
            om_beam = beam_solid_angle(beam)
            ap = radial_mask(data.shape, alma_pixel(wcs, agn), compact_radius_arcsec / px_a[0]) & np.isfinite(data)
            alma_peak = float(np.nanmax(np.where(ap, data, np.nan)))
            row, col = np.unravel_index(np.nanargmax(np.where(ap, data, np.nan)), data.shape)
            sep = wcs.pixel_to_world(col, row).separation(agn).to_value(u.arcsec)

            sm = nifty_through_beam(compact_comp, beam)
            nap = radial_mask(compact_mean.shape, nifty_agn, compact_radius_arcsec / px_arcsec)
            peaks = rj_to_flux(np.array([m[nap].max() for m in sm]), nu, om_beam)
            ints = rj_to_flux(np.array([m[nap].sum() for m in comp_maps[compact_comp]]), nu, om_pix)
            tot = rj_to_flux(np.array([m.sum() for m in comp_maps[compact_comp]]), nu, om_pix)

            (f1, s1), (f2, s2) = K16_AGN
            alpha = np.log(s2 / s1) / np.log(f2 / f1)
            k16_flux = s1 * (float(freq_ghz) / f1) ** alpha

            header("Compact source", compact_comp, alma_compact_fits, beam, freq_a, compact_radius_arcsec, "AGN")
            pm, ps = ms(peaks)
            print(
                f"  {'peak [mJy/beam] (dirty map)':34s} {alma_peak * 1e3:>10.4f} {'':>26s} {pm:>11.4f} +/- {ps:<6.4f}"
            )
            im, is_ = ms(ints)
            print(f"  {'aperture flux [mJy]':34s} {'':>10s} {'':>26s} {im:>11.4f} +/- {is_:<6.4f}")
            tm, ts = ms(tot)
            k16 = f"{k16_flux:.2f} +/- {K16_AGN_ERR[0]:.2f} +/- {K16_AGN_ERR[1]:.2f}"
            print(f"  {'total flux [mJy]':34s} {'':>10s} {k16:>26s} {tm:>11.4f} +/- {ts:<6.4f}")
            print(
                f"  (ALMA-map peak {sep:.1f}\" from K16's AGN position; K16 flux is the 12-m point-source fit, "
                f"alpha = {alpha:.2f} interpolated to {freq_ghz:.1f} GHz)"
            )

        if alma_decrement_fits:
            if decrement_comp not in comp_maps:
                raise click.UsageError(f"--decrement_comp {decrement_comp!r} not in map components {list(comp_maps)}")
            data, beam, om_pix_a, px_a, freq_a, wcs = load_alma(alma_decrement_fits, alma_scale)
            om_beam = beam_solid_angle(beam)
            c_sze = alma_pixel(wcs, sze_peak)
            ap = radial_mask(data.shape, c_sze, decrement_radius_arcsec / px_a[0]) & np.isfinite(data)
            at_peak = float(data[int(round(c_sze[0])), int(round(c_sze[1]))])
            alma_min = float(np.nanmin(np.where(ap, data, np.nan)))
            alma_int = float(np.nansum(data[ap])) * om_pix_a / om_beam

            sm = nifty_through_beam(decrement_comp, beam)
            nap = radial_mask(compact_mean.shape, nifty_sze, decrement_radius_arcsec / px_arcsec)
            ij = tuple(int(round(v)) for v in nifty_sze)
            at_peaks = rj_to_flux(np.array([m[ij] for m in sm]), nu, om_beam)
            mins = rj_to_flux(np.array([m[nap].min() for m in sm]), nu, om_beam)
            ints = rj_to_flux(np.array([m[nap].sum() for m in comp_maps[decrement_comp]]), nu, om_pix)

            header(
                "SZ decrement",
                decrement_comp,
                alma_decrement_fits,
                beam,
                freq_a,
                decrement_radius_arcsec,
                f'SE peak ({d_east:+.1f}" E, {d_north:+.1f}" N of the AGN)',
            )
            k16_p = f"{K16_SZ_PEAK[0]:.3f} +/- {K16_SZ_PEAK[1]:.3f} +/- {K16_SZ_PEAK[2]:.3f}"
            k16_i = f"{K16_SZ_INT[0]:.1f} +/- {K16_SZ_INT[1]:.1f} +/- {K16_SZ_INT[2]:.1f}"
            pm, ps = ms(at_peaks)
            print(f"  {'at SE peak [mJy/beam]':34s} {at_peak * 1e3:>10.4f} {k16_p:>26s} {pm:>11.4f} +/- {ps:<6.4f}")
            mm, mss = ms(mins)
            print(
                f"  {'minimum in aperture [mJy/beam]':34s} {alma_min * 1e3:>10.4f} {'':>26s} {mm:>11.4f} +/- {mss:<6.4f}"
            )
            im, is_ = ms(ints)
            print(f"  {'aperture flux [mJy]':34s} {alma_int * 1e3:>10.4f} {k16_i:>26s} {im:>11.4f} +/- {is_:<6.4f}")
            print(
                f'  (K16 1-sigma {K16_NOISE:.3f} mJy/beam; ALMA misses scales >~40", so its aperture flux is a '
                f"lower bound on the true |decrement|)"
            )

        print("\nK16 errors are (stat, sys); NIFTy's are the posterior std.")

    print(
        f"\nCaveat: {n_stored} stored samples are {n_eff} antithetic pairs, so each std above rests on "
        f"~{n_eff} independent draws\nand is itself uncertain by roughly "
        f"{100 / np.sqrt(2 * max(n_eff - 1, 1)):.0f}%."
    )


if __name__ == "__main__":
    main()
