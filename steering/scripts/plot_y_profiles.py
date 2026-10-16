"""Compton-y profiles of one reconstruction, from its FITS export.

The single-run successor of ``profile_compare.py``, reading the per-sample maps
from ``save_fits.py``'s ``map_samples.fits`` (no model or ``last.pkl`` needed).
One map component (``--component``, default the diffuse ``s1``) is converted
from K_RJ to Compton y at ``--freq_ghz`` and its zero level removed per sample
(median beyond ``--zero_radius``; the correlated field's ``zeromode`` is
unconstrained by the data, so everything here is differential).

Writes ``y_profiles.png`` to ``<fits_dir>/../y_profiles/`` (i.e.
``plot_<run>/y_profiles/``) unless ``--out_dir`` is given. Three panels: the
azimuthally averaged ``dy(r)`` with a band spanning the posterior samples, the
cumulative cylindrically integrated ``Y(<r) = int y dOmega``, and ``Y(<r)``
relative to the Di Mascolo+2019 joint ALMA+Bolocam+Planck gNFW pressure fit.
That model is drawn both absolute and with the data's own zero level (a finite
field cannot see the model's outer wing, so the reconstruction is pushed toward
zero at its edge; offsetting the model the same way keeps the comparison
unbiased). Their model is elliptical (eps = 0.648); integrating it as spherical
overestimates Y by 1/eps, so the spherical profile is scaled by eps -- this
reproduces their published ``Y_cyl(15')`` to 0.9%, printed as a self-check on
every run. The Planck ``Y(r<15')`` (Di Mascolo+2019) is shown as a band; its
15' aperture lies off the plot. With ``--fit_profile`` a gNFW is also fitted to
every sample's profile and its mean drawn dash-dotted.

Example::

    pixi run -e cpu python scripts/plot_y_profiles.py \\
        --fits_dir runs/RXJ1347/AGBT17A_358_test/10tods_10amin_8s_nl/plot_4/save_fits
"""

import os

import click

# CMB / tSZ constants
H = 6.62607015e-34
K_B = 1.380649e-23
T_CMB = 2.72548
C = 2.99792458e8

# Di Mascolo+2019 Planck y-map: Y(r<15').
PLANCK_Y_ARCMIN2 = 3.24e-3
PLANCK_Y_ERR = 0.54e-3

# Di Mascolo+2019 joint gNFW fit (their Fig. 6 corner plot, RXJ1347 at z=0.451).
# alpha/beta held at the Arnaud+2010 cool-core values; gamma, P0, rs fitted.
GNFW_P0 = 0.157  # keV cm^-3 (table lists 1.57 in units of 1e-1)
GNFW_RS_AMIN = 2.39
GNFW_GAMMA = 0.563
GNFW_ALPHA = 1.2223
GNFW_BETA = 5.4905
GNFW_EPS = 0.648  # axis ratio of the elliptical fit
KPC_PER_ASEC = 5.94  # their stated scale at z=0.451
SIGMA_T = 6.6524587321e-25  # cm^2
MEC2 = 510.99895  # keV
KPC_CM = 3.0856775814913673e21


def k_rj_per_y(nu):
    """K_RJ corresponding to one unit of Compton y at frequency ``nu`` [Hz]."""
    import numpy as np

    x = H * nu / (K_B * T_CMB)
    g = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4.0
    dI_dT = (2 * H**2 * nu**4) / (K_B * T_CMB**2 * C**2) * np.exp(x) / (np.exp(x) - 1) ** 2
    return g * T_CMB * dI_dT / (2 * K_B * nu**2 / C**2)


def gnfw_y(theta_amin, p0=GNFW_P0, rs_amin=GNFW_RS_AMIN, gamma=GNFW_GAMMA):
    """Azimuthally averaged Compton y of a gNFW pressure profile.

    Line-of-sight integral of the electron pressure,
    ``y = sigma_T/(m_e c^2) * int P_e dl``, on a spherical profile, scaled by
    Di Mascolo's fitted axis ratio ``eps`` so the enclosed Y matches their
    elliptical model. Defaults are their best fit.
    """
    import numpy as np

    kpc_per_amin = KPC_PER_ASEC * 60.0
    rs = float(rs_amin) * kpc_per_amin

    def p_e(r_kpc):
        x = np.maximum(r_kpc, 1e-3) / rs
        return float(p0) * x ** (-float(gamma)) * (1 + x**GNFW_ALPHA) ** ((float(gamma) - GNFW_BETA) / GNFW_ALPHA)

    los = np.linspace(0.0, 60.0 * rs, 4000)
    theta = np.atleast_1d(np.asarray(theta_amin, dtype=float))

    def y_at(t):
        r = np.sqrt((t * kpc_per_amin) ** 2 + los**2)
        return 2.0 * np.trapezoid(p_e(r), los) * KPC_CM * SIGMA_T / MEC2

    return GNFW_EPS * np.array([y_at(t) for t in theta.ravel()]).reshape(theta.shape)


def cumulative_y(theta_amin, y):
    """Cylindrically integrated ``Y(<theta)`` [arcmin^2] of a radial profile ``y(theta)``."""
    import numpy as np

    th = np.asarray(theta_amin, dtype=float)
    return np.array([np.trapezoid(2 * np.pi * th[: i + 1] * y[: i + 1], th[: i + 1]) for i in range(th.size)])


def fit_gnfw(mid, prof, sel, r_hi, fix_rs):
    """Fit a gNFW (zero level matched at ``r_hi`` [arcsec]) to one radial profile.

    Fits P0 and gamma (and r_s unless ``fix_rs``) with alpha/beta held at the
    Arnaud+2010 cool-core values, as Di Mascolo do. Returns ``(p0, rs_amin,
    gamma)``; raises on a failed fit.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    def m_fix(th, p0, gamma):
        v = gnfw_y(th / 60.0, p0, GNFW_RS_AMIN, gamma)
        return v - np.interp(r_hi, th, v)

    def m_free(th, p0, rs, gamma):
        v = gnfw_y(th / 60.0, p0, rs, gamma)
        return v - np.interp(r_hi, th, v)

    if fix_rs:
        popt, _ = curve_fit(
            m_fix, mid[sel], prof[sel], p0=[GNFW_P0, GNFW_GAMMA], bounds=([1e-3, 0.0], [5.0, 2.0]), maxfev=20000
        )
        return float(popt[0]), GNFW_RS_AMIN, float(popt[1])
    popt, _ = curve_fit(
        m_free,
        mid[sel],
        prof[sel],
        p0=[GNFW_P0, GNFW_RS_AMIN, GNFW_GAMMA],
        bounds=([1e-3, 0.2, 0.0], [5.0, 20.0, 2.0]),
        maxfev=20000,
    )
    return tuple(float(v) for v in popt)


def load_component(fits_dir, component):
    """One component's per-sample maps (K_RJ, model ``[i, j]`` orientation) from ``map_samples.fits``.

    Returns ``(cube, px_rad, n_stored, n_eff)``; ``n_eff`` is the number of independent draws (the copied ``config.yaml``'s
    ``n_samples``, else ``NSAMP / 2`` for mirrored pairs).
    """
    from astropy.io import fits

    import maria_nifty as mani

    with fits.open(os.path.join(fits_dir, "map_samples.fits")) as hdul:
        hdr = hdul[0].header
        names = [hdr[f"COMP{i}"] for i in range(1, int(hdr["NCOMP"]) + 1)]
        if component not in names:
            raise click.UsageError(f"--component {component!r} not in map components {names}")
        cube = mani.from_fits_data(hdul[component.upper()].data)
        px_rad = float(hdr["PIXX"])
        n_stored = int(hdr["NSAMP"])
        antithetic = bool(hdr.get("ANTITHET", False))

    n_eff = None
    config_path = os.path.join(fits_dir, "config.yaml")
    if os.path.isfile(config_path):
        n_eff = (mani.yaml_load(config_path).get("minimization") or {}).get("n_samples")
    if n_eff is None:
        n_eff = n_stored // 2 if antithetic else n_stored
    return cube, px_rad, n_stored, int(n_eff)


@click.command()
@click.option("--fits_dir", required=True, help="A save_fits.py export (plot_<run>/save_fits, with map_samples.fits).")
@click.option(
    "--out_dir", default=None, help="Output folder; defaults to <fits_dir>/../y_profiles, i.e. plot_<run>/y_profiles/."
)
@click.option("--component", default="s1", show_default=True, help="Map component to profile (s1 = diffuse).")
@click.option("--label", default=None, help="Legend label for the run (defaults to the run folder name).")
@click.option(
    "--freq_ghz",
    default=90.0,
    show_default=True,
    help="Frequency for the K_RJ->y conversion. Default 90 GHz, from the TOD headers' own "
    "WAVE = 0.00333 m (= 90.03 GHz) -- not maria's band_center of 86.21 GHz, which is a "
    "tau-weighted mean of its modelled passband rather than an observing frequency.",
)
@click.option(
    "--zero_radius",
    default=0.0,
    show_default=True,
    help="Zero level: median beyond this radius; 0 = 0.93 * half-width.",
)
@click.option("--n_bins", default=40, show_default=True, help="Number of radial bins.")
@click.option(
    "--fit_rmin", default=8.0, show_default=True, help="Inner radius [arcsec] for the gNFW fit (excludes the beam)."
)
@click.option(
    "--fit_profile",
    default=True,
    show_default=True,
    help="Fit a gNFW to every sample's profile (about the field centre) and report P0, rs, gamma against "
    "Di Mascolo+2019; the mean fit is drawn dash-dotted in y_profiles.png. The fit uses only the radii "
    "actually constrained, so extrapolation beyond the field is the model's job rather than the map's.",
)
@click.option(
    "--fit_rmax",
    default=0.0,
    show_default=True,
    help="Outer radius [arcsec] for the profile fit; 0 uses the zero-level radius, beyond which the "
    "profile is set by the zero-level choice rather than by the data.",
)
@click.option(
    "--profile_fix_rs",
    default=False,
    show_default=True,
    help="Hold r_s at Di Mascolo's 2.39 arcmin in the profile fit and fit only P0 and gamma. Their r_s sits "
    "near the edge of a few-arcmin field, so left free it trades against P0 along that degeneracy.",
)
@click.option("--background", default="black", show_default=True, help="Figure background.")
@click.option("--grid", default=False, show_default=True, help="Draw grid lines.")
def main(
    fits_dir,
    out_dir,
    component,
    label,
    freq_ghz,
    zero_radius,
    n_bins,
    fit_rmin,
    fit_profile,
    fit_rmax,
    profile_fix_rs,
    background,
    grid,
):
    """Plot the y(r) and Y(<r) profiles of one run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from nifty.re import logger

    background = str(background).lower()
    plt.style.use("dark_background" if background == "black" else "default")
    for key in ("figure.facecolor", "axes.facecolor", "savefig.facecolor", "savefig.edgecolor"):
        plt.rcParams[key] = background
    plt.rcParams["axes.grid"] = bool(grid)

    AMIN2SR = (np.pi / 180 / 60) ** 2

    fits_dir = os.path.abspath(fits_dir)
    out_dir = out_dir or os.path.join(os.path.dirname(fits_dir), "y_profiles")
    os.makedirs(out_dir, exist_ok=True)
    if label is None:
        # .../<run folder>/plot_<N>/save_fits -> "<run folder> plot_<N>"
        plot_dir = os.path.dirname(fits_dir)
        label = f"{os.path.basename(os.path.dirname(plot_dir))} {os.path.basename(plot_dir)}"

    cube, px_rad, n_stored, n_eff = load_component(fits_dir, component)
    factor = k_rj_per_y(float(freq_ghz) * 1e9)
    px = px_rad * 180 * 3600 / np.pi
    om_pix = px_rad**2
    n = cube.shape[1]
    ii, jj = np.mgrid[0:n, 0:n]
    c0 = (n - 1) / 2.0
    r = np.hypot(ii - c0, jj - c0) * px
    half = 0.5 * n * px
    r_zero = float(zero_radius) if float(zero_radius) > 0 else 0.93 * half
    edges = np.linspace(0, 0.98 * half, int(n_bins) + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])

    def radial_profile(y):
        return np.array([float(y[(r >= a) & (r < b)].mean()) for a, b in zip(edges[:-1], edges[1:])])

    # Compton y per sample, zero level removed (median beyond r_zero).
    ymaps = [m / factor - float(np.median(m[r > r_zero] / factor)) for m in cube]

    logger.info(
        f"\n{fits_dir}\n  component {component}, pixel {px:.3f}\", field {2 * half / 60:.2f}', "
        f"{n_stored} stored samples ({n_eff} effective)\n"
        f'  zero level: median beyond {r_zero:.0f}"'
    )

    # Self-check: does the model reproduce Di Mascolo's published Planck Y?
    th_chk = np.linspace(1e-3, 15.0, 3000)
    y_chk = float(cumulative_y(th_chk, gnfw_y(th_chk))[-1])
    logger.info(
        f"  gNFW self-check: model Y_cyl(15') = {y_chk:.4e} arcmin^2 vs published "
        f"{PLANCK_Y_ARCMIN2:.3e} (ratio {y_chk / PLANCK_Y_ARCMIN2:.4f})"
    )

    prof = np.array([radial_profile(y) for y in ymaps])
    cum = np.array([[float(y[r < b].sum() * om_pix) for b in edges[1:]] for y in ymaps])
    col = plt.colormaps.get_cmap("cmb")(0.15)

    fig, (ax_y, ax_Y, ax_r) = plt.subplots(1, 3, figsize=(21, 6), dpi=150)
    label_fs, tick_fs = 14.0, 12.0

    if fit_profile:
        r_hi = float(fit_rmax) if float(fit_rmax) > 0 else r_zero
        sel = (mid >= float(fit_rmin)) & (mid <= r_hi)
        fits = []
        for row in prof:
            try:
                fits.append(fit_gnfw(mid, row, sel, r_hi, profile_fix_rs))
            except Exception as exc:  # noqa: BLE001 - report and carry on
                logger.warning(f"  gNFW profile fit failed for one sample: {exc}")
        if fits:
            fa = np.asarray(fits)
            sdf = fa.std(axis=0, ddof=1) if fa.shape[0] > 1 else np.full(3, np.nan)
            logger.info(
                f'  gNFW fit to the profile over {fit_rmin:.0f}-{r_hi:.0f}"'
                f"{' (r_s fixed)' if profile_fix_rs else ''}:\n"
                f"    P0    = {fa[:, 0].mean():.4f} +/- {sdf[0]:.4f} keV/cm3   (Di Mascolo {GNFW_P0:.3f})\n"
                f"    rs    = {fa[:, 1].mean():.3f} +/- {sdf[1]:.3f} arcmin    (Di Mascolo {GNFW_RS_AMIN:.2f})\n"
                f"    gamma = {fa[:, 2].mean():.3f} +/- {sdf[2]:.3f}           (Di Mascolo {GNFW_GAMMA:.3f})"
            )
            yf = gnfw_y(mid / 60.0, *fa.mean(axis=0))
            ax_y.plot(mid, yf - np.interp(r_hi, mid, yf), color=col, ls="-.", lw=1.3, alpha=0.9, label="gNFW fit")

    ax_y.plot(mid, prof.mean(0), color=col, lw=2.0, label=label)
    ax_y.fill_between(mid, prof.min(0), prof.max(0), color=col, alpha=0.25, lw=0)
    ax_Y.plot(mid, cum.mean(0) / AMIN2SR, color=col, lw=2.0, label=label)
    ax_Y.fill_between(mid, cum.min(0) / AMIN2SR, cum.max(0) / AMIN2SR, color=col, alpha=0.25, lw=0)

    # Third panel: recovered fraction of the gNFW model, radius by radius, with
    # the model carrying the same zero level as the data.
    y_m = gnfw_y(mid / 60.0)
    y_m = y_m - float(np.interp(r_zero, mid, y_m))
    mdl = cumulative_y(mid / 60.0, y_m)
    good = mdl > 0
    ax_r.plot(mid[good], (cum.mean(0) / AMIN2SR)[good] / mdl[good], color=col, lw=2.0, label=label)
    ax_r.fill_between(
        mid[good],
        (cum.min(0) / AMIN2SR)[good] / mdl[good],
        (cum.max(0) / AMIN2SR)[good] / mdl[good],
        color=col,
        alpha=0.25,
        lw=0,
    )

    # gNFW model curves, absolute and with the data's zero level (see docstring).
    th_mdl = np.linspace(1.0, max(ax_y.get_xlim()[1], 60.0), 300)
    y_mdl = gnfw_y(th_mdl / 60.0)
    off = float(np.interp(r_zero, th_mdl, y_mdl))
    logger.info(f'  model zero level: y({r_zero:.0f}") = {off:.3e} subtracted to match the data')
    ax_y.plot(th_mdl, y_mdl, color="0.45", ls=":", lw=1.5, label="gNFW (absolute)")
    ax_y.plot(th_mdl, y_mdl - off, color="0.85", ls="--", lw=1.9, label="gNFW, data's zero level")
    ax_Y.plot(th_mdl, cumulative_y(th_mdl / 60.0, y_mdl), color="0.45", ls=":", lw=1.5, label="gNFW (absolute)")
    ax_Y.plot(
        th_mdl, cumulative_y(th_mdl / 60.0, y_mdl - off), color="0.85", ls="--", lw=1.9, label="gNFW, data's zero level"
    )

    ax_Y.axhline(PLANCK_Y_ARCMIN2, color="0.6", ls="--", lw=1.3)
    ax_Y.axhspan(PLANCK_Y_ARCMIN2 - PLANCK_Y_ERR, PLANCK_Y_ARCMIN2 + PLANCK_Y_ERR, color="0.6", alpha=0.18, lw=0)
    ax_Y.annotate(
        f"Planck $Y(r{{<}}15^\\prime)$ = {PLANCK_Y_ARCMIN2:.2e} arcmin$^2$\n(aperture 15$^\\prime$, off this plot)",
        xy=(0.02, PLANCK_Y_ARCMIN2),
        xycoords=("axes fraction", "data"),
        xytext=(0, 8),
        textcoords="offset points",
        color="0.75",
        fontsize=9,
    )

    ax_y.set_xlabel("radius [arcsec]", fontsize=label_fs)
    ax_y.set_ylabel(r"differential Compton $\Delta y$", fontsize=label_fs)
    ax_y.axhline(0, color="0.4", lw=0.8)
    ax_Y.set_xlabel("radius [arcsec]", fontsize=label_fs)
    ax_Y.set_ylabel(r"cumulative $Y(<r)$ [arcmin$^2$]", fontsize=label_fs)
    ax_r.axhline(1.0, color="0.8", ls="--", lw=1.8)
    ax_r.set_xlabel("radius [arcsec]", fontsize=label_fs)
    ax_r.set_ylabel(r"$Y(<r)$ / gNFW model", fontsize=label_fs)
    ax_r.set_ylim(0, 2.0)
    for ax in (ax_y, ax_Y, ax_r):
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(bool(grid))
        ax.set_xmargin(0)
        ax.legend(fontsize=10, loc="best")

    fname = os.path.join(out_dir, "y_profiles.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  wrote {fname}")

    logger.info(
        f"\nCaveat: {n_stored} stored samples are {n_eff} independent draws, so each band/std above is "
        f"itself uncertain by roughly {100 / np.sqrt(2 * max(n_eff - 1, 1)):.0f}%."
    )


if __name__ == "__main__":
    main()
