"""FITS export helpers for reconstructed maps and map components.

The export functions turn posterior samples or single positions into FITS
products that preserve the space metadata where possible.
"""

import os

import jax
import nifty.re as jft
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from maria.coords.transforms import unjitted_offsets_to_phi_theta

from .model.util import DEG2RAD


def _mean_std_from_samples(eval_fn, samples):
    """Evaluate a callable on samples and return mean/std arrays."""
    if isinstance(samples, jft.Samples):
        mean, std = jft.mean_and_std(tuple(eval_fn(s) for s in samples))
        return np.asarray(mean), np.asarray(std)

    arr = np.asarray(eval_fn(samples))
    return arr, np.zeros_like(arr)


def _iter_map_components(map_model, names=None):
    """Named map components in a stable order: background (if any), then sources.

    ``names`` labels ``map_model.sources`` (e.g. the config's ``source_params``
    keys, see :func:`map_component_names`); defaults to ``s1, s2, ...``.
    """
    sources = tuple(getattr(map_model, "sources", ()))
    if names is None:
        names = [f"s{i + 1}" for i in range(len(sources))]
    if len(names) != len(sources):
        raise ValueError(f"Got {len(names)} component names for {len(sources)} map sources.")
    out = [] if getattr(map_model, "background", None) is None else [("bg", map_model.background)]
    return out + list(zip(names, sources))


def map_component_names(config):
    """Source names in ``map_model.sources`` order, from the config's ``map.source_params``.

    Mirrors :meth:`maria_nifty.model.map.MapModel.build`: a single unnamed
    source (``source_params`` holding ``space_params`` directly) is ``s1``.
    """
    source_params = (config.get("map") or {}).get("source_params") or {}
    if "space_params" in source_params:
        return ["s1"]
    return [str(k) for k in source_params]


def _positions(samples):
    """All stored posterior states (a single position counts as one)."""
    return list(samples) if isinstance(samples, jft.Samples) else [samples]


def _is_antithetic(samples):
    """Whether ``samples`` holds NIFTy's mirrored pairs, zipped as ``[s1, -s1, s2, -s2, ...]``."""
    if not isinstance(samples, jft.Samples) or len(samples) < 2 or len(samples) % 2:
        return False
    leaves = jax.tree_util.tree_leaves(samples._samples)
    return all(np.allclose(np.asarray(leaf[0::2]), -np.asarray(leaf[1::2])) for leaf in leaves)


def _component_cards(hdr, names):
    """Write ``NCOMP`` and ``COMP1``, ``COMP2``, ... (1-based, matching HDU order) into ``hdr``."""
    hdr["NCOMP"] = (len(names), "Number of map components")
    for i, name in enumerate(names, start=1):
        hdr[f"COMP{i}"] = (name[:60], f"Component {i} name")


def _grid_cards(hdr, space, cphi0, ctheta0):
    """Write the local tangent-plane grid geometry into ``hdr``.

    ``PIXX``/``PIXY`` (``space.distances``) and ``CENX``/``CENY``
    (``space.center``) in radians, plus the absolute reference ``CPHI0``/
    ``CTHETA0`` of the local ``(0, 0)`` origin -- enough to rebuild the plots'
    arcmin extent and sky ticks without decoding the CAR WCS.
    """
    if space is None:
        return
    distances = np.asarray(space.distances, dtype=float).reshape(-1)
    center = np.asarray(space.center, dtype=float).reshape(-1)
    hdr["PIXX"] = (float(distances[0]), "[rad] pixel size along model axis 0")
    hdr["PIXY"] = (float(distances[1]), "[rad] pixel size along model axis 1")
    hdr["CENX"] = (float(center[0]), "[rad] local grid center, model axis 0")
    hdr["CENY"] = (float(center[1]), "[rad] local grid center, model axis 1")
    if cphi0 is not None:
        hdr["CPHI0"] = (float(cphi0), "[rad] RA of the local (0, 0) offset origin")
        hdr["CTHETA0"] = (float(ctheta0), "[rad] Dec of the local (0, 0) offset origin")


def _bunit(datasets):
    """The TODs' native unit string (e.g. ``"K_RJ"``), or ``""`` if unknown."""
    if not datasets:
        return ""
    return str(getattr(datasets[0].get("tods"), "units", None) or "")


def _to_fits_data(array):
    """Convert an array into FITS image orientation and dtype.

    ``array`` is indexed ``[i, j]`` matching ``GridSpace.coos`` (``i`` along
    the space's first/"x" axis, ``j`` along its second/"y" axis; see
    ``space_coos`` in ``model/space.py``). With zero rotation, increasing
    ``i`` increases RA and increasing ``j`` increases Dec (verified
    numerically against ``unjitted_offsets_to_phi_theta``). FITS wants RA on
    the fast-varying (last, "NAXIS1") axis with pixel index *decreasing* RA
    (the standard east-left sky convention), and Dec on the slow ("NAXIS2")
    axis increasing with pixel index -- i.e. ``written[j, n] = array[nx-1-n, j]``,
    which is ``array[::-1, :].T``.
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D map for FITS export, got shape {arr.shape}")
    return arr[::-1, :].T


def from_fits_data(array):
    """Inverse of :func:`_to_fits_data` on the last two axes.

    Turns a FITS image (or a ``(n, ny, nx)`` cube of them) written by this
    module back into the model's own ``[i, j]`` indexing (``i`` -> +RA,
    ``j`` -> +Dec), i.e. ``written.T[::-1, :]`` per frame.
    """
    arr = np.asarray(array, dtype=float)
    return np.flip(np.swapaxes(arr, -1, -2), axis=-2)


def _absolute_reference(datasets):
    """Absolute (ra, dec) in radians of a map model's local (0, 0) origin.

    Every dataset dict from ``load_tod_datasets`` carries the same
    ``center_phi_theta`` value (see its docstring) -- the exact reference
    point used to build the local offsets the model was fit against, not a
    re-derived approximation of it. Returns ``(None, None)`` if `datasets`
    is falsy, so callers can skip the WCS gracefully when it isn't available.
    """
    if not datasets:
        return None, None
    return datasets[0]["center_phi_theta"]


def _wcs_header(space, cphi0, ctheta0):
    """Build a FITS header with RA/Dec WCS metadata for `space`, if possible.

    Parameters
    ----------
    space : GridSpace or None
        The map model's output space (pixel grid).
    cphi0, ctheta0 : float or None
        Absolute (ra, dec) in radians of the space's local ``(0, 0)``
        origin (see :func:`_absolute_reference`). If either is ``None``,
        only non-astrometric shape/pixel-size metadata is written.

    Notes
    -----
    Follows the same conventions as reference MUSTANG-2 map FITS files
    (``RA---CAR``/``DEC--CAR``, treating the local flat-sky offsets as
    literal (RA, Dec) degrees with no ``cos(dec)`` scaling): exact at the
    ``CRPIX = (1, 1)`` reference pixel (computed via maria's own
    ``unjitted_offsets_to_phi_theta``, not the linear CAR approximation),
    and accurate to a negligible (sub-arcsec, for few-arcmin fields)
    residual elsewhere. See ``_to_fits_data`` for the axis convention this
    depends on (``CRPIX = (1, 1)`` is the written array's ``[0, 0]`` entry,
    i.e. model pixel ``(i, j) = (nx - 1, 0)``).
    """
    hdr = fits.Header()
    if space is None:
        return hdr

    shape = np.asarray(getattr(space, "shape", (np.nan, np.nan)), dtype=float)
    distances = np.asarray(getattr(space, "distances", (np.nan, np.nan)), dtype=float)
    center = np.asarray(getattr(space, "center", (0.0, 0.0)), dtype=float)
    if center.ndim > 1:
        center = center[0]
    rotation = np.asarray(getattr(space, "rotation", 0.0), dtype=float)
    rotation = float(rotation.reshape(-1)[0]) if rotation.ndim > 0 else float(rotation)

    # NAXIS1/NAXIS2 are intentionally not set here: the HDU constructor derives
    # them from the array's actual shape (see `_to_fits_data`) and inserts them
    # right after NAXIS, as FITS requires. Setting them ourselves ahead of
    # WCSAXES/other header cards violates that mandatory ordering and makes
    # `HDUList.writeto` raise a VerifyError ("NAXIS1' card at the wrong place").

    if cphi0 is None or ctheta0 is None:
        return hdr

    nx, ny = shape
    dx, dy = distances
    cx, cy = float(center[0]), float(center[1])
    c, s = np.cos(rotation), np.sin(rotation)

    # CRPIX = (1, 1) is the written array's [0, 0] entry, i.e. model pixel
    # (i, j) = (nx - 1, 0) (see _to_fits_data's axis-convention docstring).
    x0_ref = ((nx - 1) - 0.5 * (nx - 1)) * dx
    y0_ref = (0 - 0.5 * (ny - 1)) * dy
    x_ref = x0_ref * c - y0_ref * s + cx
    y_ref = x0_ref * s + y0_ref * c + cy
    ra_ref, dec_ref = (float(v) for v in unjitted_offsets_to_phi_theta(np.array([x_ref, y_ref]), cphi0, ctheta0))

    w = WCS(naxis=2)
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crval = [np.degrees(ra_ref), np.degrees(dec_ref)]
    w.wcs.radesys = "ICRS"
    if rotation == 0.0:
        w.wcs.cdelt = [-dx / DEG2RAD, dy / DEG2RAD]
    else:
        w.wcs.cd = [
            [-c * dx / DEG2RAD, -s * dy / DEG2RAD],
            [-s * dx / DEG2RAD, c * dy / DEG2RAD],
        ]

    hdr.update(w.to_header())
    hdr["MJDREF"] = (0.0, "[d] MJD of fiducial time")
    return hdr


def map2fits(
    sum_response,
    samples,
    filename: str | None = None,
    *,
    datasets: list[dict] | None = None,
    odir: str | None = None,
    overwrite: bool = True,
    default_filename: str = "map_mean.fits",
    mode: str = "mef",
    component_names: list[str] | None = None,
) -> None:
    """Write full map reconstruction and all map components into one FITS file.

    Parameters
    ----------
    sum_response : Model
        Sum response containing a `map_model`.
    samples : jft.Samples or jft.Vector-like
        Posterior samples or a single position.
    filename : str, optional
        Full output path to the FITS file. If omitted, `odir/default_filename` is used.
    datasets : list of dict, optional
        The dataset list from :func:`maria_nifty.util.load_tod_datasets` (any
        entry; they all carry the same ``center_phi_theta``). Supplies the
        absolute sky reference needed for a real RA/Dec WCS (``CRVAL``,
        etc.) -- without it, headers only carry ``NAXIS``/``WCSAXES``.
    odir : str, optional
        Output directory used when `filename` is not given.
    overwrite : bool, optional
        Whether to overwrite an existing file.
    default_filename : str, optional
        Basename used when `filename` is not given.
    mode : str, optional
        Output format: ``"cube"`` (single 3D image + frame table) or
        ``"mef"`` (multi-extension FITS, one HDU per component).
    component_names : list of str, optional
        Names of ``map_model.sources`` in order (see :func:`map_component_names`);
        defaults to ``s1, s2, ...``. A background, if present, is named ``bg``.

    Notes
    -----
    Writes the file as a side effect; returns nothing.  All map frames are
    stored as ``float32``.  A real RA/Dec WCS (``CRPIX``/``CDELT``/``CTYPE``/
    ``CRVAL``/``CUNIT``/``RADESYS``, matching reference MUSTANG-2 map FITS
    files) is embedded in each HDU header when ``sum_response`` exposes a
    space object and ``datasets`` is given; see ``_wcs_header``.
    """
    if filename is None:
        out_dir = "." if odir is None else odir
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, default_filename)

    map_model = getattr(sum_response, "map_model", None)
    if map_model is None:
        raise ValueError("sum_response.map_model is None; cannot export map reconstruction")

    mode = str(mode).strip().lower()
    if mode not in {"cube", "mef"}:
        raise ValueError("mode has to be one of: cube, mef")

    cphi0, ctheta0 = _absolute_reference(datasets)

    full_mean, _ = _mean_std_from_samples(lambda s: map_model(s), samples)

    components = _iter_map_components(map_model, component_names)
    component_names = [name for name, _ in components]
    bunit = _bunit(datasets)
    map_response = getattr(sum_response, "map_response", None)

    primary_header = _wcs_header(getattr(map_model, "space", None), cphi0, ctheta0)
    primary_header["EXTNAME"] = "FULL_MEAN"
    primary_header["BTYPE"] = "reconstruction mean"
    primary_header["COMPNAME"] = "full"
    if bunit:
        primary_header["BUNIT"] = bunit
    primary_header["NSAMP"] = (len(_positions(samples)), "Posterior states averaged (incl. mirrored)")
    if map_response is not None:
        primary_header["BEAMSIG"] = (float(map_response.sigma), "Beam sigma in map pixels")
    _grid_cards(primary_header, getattr(map_model, "space", None), cphi0, ctheta0)
    _component_cards(primary_header, component_names)

    frame_names = ["full_mean"]
    frame_types = ["mean"]
    frame_components = ["full"]
    frame_arrays = [_to_fits_data(full_mean)]

    for comp_name, comp_model in components:
        mean_c, _ = _mean_std_from_samples(lambda s, m=comp_model: m(s, map=True), samples)
        frame_names.append(f"{comp_name}_mean")
        frame_types.append("mean")
        frame_components.append(comp_name)
        frame_arrays.append(_to_fits_data(mean_c))

    if mode == "cube":
        shapes = [arr.shape for arr in frame_arrays]
        if any(sh != shapes[0] for sh in shapes):
            raise ValueError("Cannot build cube because frame shapes differ; use mode='mef' instead.")

        cube = np.stack(frame_arrays, axis=0).astype(np.float32)
        cube_header = _wcs_header(getattr(map_model, "space", None), cphi0, ctheta0)
        cube_header["EXTNAME"] = "MAP_CUBE"
        cube_header["BTYPE"] = "map reconstruction frames"
        cube_header["COMPNAME"] = "multiple"
        cube_header["NFRAME"] = (int(cube.shape[0]), "Number of frames in cube")
        for i, name in enumerate(frame_names):
            key = f"FRM{i}"
            if len(key) <= 8:
                cube_header[key] = (name[:60], f"Frame {i} name")

        frame_table = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="INDEX", format="J", array=np.arange(len(frame_names), dtype=np.int32)),
                fits.Column(name="NAME", format="64A", array=np.asarray(frame_names, dtype="S64")),
                fits.Column(name="TYPE", format="16A", array=np.asarray(frame_types, dtype="S16")),
                fits.Column(name="COMPONENT", format="64A", array=np.asarray(frame_components, dtype="S64")),
            ],
            name="FRAMEINFO",
        )

        hdus = [fits.PrimaryHDU(data=cube, header=cube_header), frame_table]
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
        return

    hdus = [fits.PrimaryHDU(data=frame_arrays[0], header=primary_header)]

    for comp_name, arr in zip(frame_components[1:], frame_arrays[1:]):
        cmean_hdr = _wcs_header(getattr(map_model, "space", None), cphi0, ctheta0)
        cmean_hdr["EXTNAME"] = f"{comp_name[:40].upper()}_MEAN"
        cmean_hdr["BTYPE"] = f"component mean: {comp_name}"
        cmean_hdr["COMPNAME"] = comp_name
        if bunit:
            cmean_hdr["BUNIT"] = bunit
        hdus.append(fits.ImageHDU(data=arr, header=cmean_hdr))

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


def map_samples2fits(
    sum_response,
    samples,
    filename: str,
    *,
    datasets: list[dict] | None = None,
    component_names: list[str] | None = None,
    prior_positions=(),
    overwrite: bool = True,
) -> None:
    """Write every posterior state (and optional prior draws) of the map into one FITS file.

    Layout: a header-only primary HDU (``NSAMP``, ``ANTITHET``, ``NPRIOR``,
    ``COMP1``, ... and the WCS), then one ``(n_samp, ny, nx)`` cube per
    quantity -- ``FULL`` (the summed map) and one per component (``S1``,
    ``S2``, ...) -- followed by ``PRIOR_FULL``/``PRIOR_<COMP>`` cubes of shape
    ``(n_prior, ny, nx)`` when ``prior_positions`` is non-empty. Every frame
    uses :func:`_to_fits_data`'s orientation and the same WCS as
    :func:`map2fits`, so frame ``i`` of ``FULL`` is directly comparable to
    ``map_mean.fits``'s primary HDU.

    NIFTy stores mirrored sample pairs zipped as ``[s1, -s1, s2, -s2, ...]``;
    ``ANTITHET`` records whether that holds, in which case the number of
    independent draws is ``NSAMP / 2``.
    """
    map_model = getattr(sum_response, "map_model", None)
    if map_model is None:
        raise ValueError("sum_response.map_model is None; cannot export map samples")

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    cphi0, ctheta0 = _absolute_reference(datasets)
    space = getattr(map_model, "space", None)
    components = _iter_map_components(map_model, component_names)
    positions = _positions(samples)
    prior_positions = list(prior_positions)
    bunit = _bunit(datasets)

    def cube_hdu(name, fn, pos, btype):
        hdr = _wcs_header(space, cphi0, ctheta0)
        hdr["EXTNAME"] = name
        hdr["BTYPE"] = btype
        if bunit:
            hdr["BUNIT"] = bunit
        cube = np.stack([_to_fits_data(fn(p)) for p in pos], axis=0)
        return fits.ImageHDU(data=cube, header=hdr)

    primary = _wcs_header(space, cphi0, ctheta0)
    primary["NSAMP"] = (len(positions), "Posterior states stored (incl. mirrored)")
    primary["ANTITHET"] = (_is_antithetic(samples), "States are mirrored pairs [s1,-s1,s2,-s2,...]")
    primary["NPRIOR"] = (len(prior_positions), "Prior draws stored")
    if bunit:
        primary["BUNIT"] = bunit
    _component_cards(primary, [name for name, _ in components])
    _grid_cards(primary, space, cphi0, ctheta0)
    hdus = [fits.PrimaryHDU(header=primary)]

    groups = [("", positions, "posterior sample")]
    if prior_positions:
        groups.append(("PRIOR_", prior_positions, "prior sample"))
    for prefix, pos, kind in groups:
        hdus.append(cube_hdu(f"{prefix}FULL", map_model, pos, f"{kind}: full map"))
        for comp_name, comp_model in components:
            hdus.append(
                cube_hdu(
                    f"{prefix}{comp_name[:40].upper()}",
                    lambda p, m=comp_model: m(p, map=True),
                    pos,
                    f"{kind}: {comp_name}",
                )
            )

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


def _sample_mean(fn, positions):
    """Mean of ``fn`` over ``positions``, accumulated one state at a time."""
    total = None
    for p in positions:
        val = np.atleast_2d(np.asarray(fn(p), dtype=np.float64))
        total = val if total is None else total + val
    return total / len(positions)


def tods2fits(
    datasets: list[dict],
    samples,
    filename: str,
    *,
    prior_positions=None,
    labels: list[str] | None = None,
    header_cards: dict | None = None,
    overwrite: bool = True,
) -> None:
    """Write the observed TODs, posterior-mean TOD components and metadata per dataset.

    One group of HDUs per dataset ``k = 01..N`` (in ``datasets`` order, i.e.
    after :func:`maria_nifty.build_detector` dropped masked detectors):

    * ``Dkk`` -- ``(n_chan, n_det, n_t)`` float32 cube; ``CHAN0 = DATA``, then
      ``ATM_MEAN`` / ``MAP_MEAN`` (whichever the model has) and
      ``MODEL_MEAN`` (their sum). ``CHANi`` header cards name each channel.
    * ``Dkk_MASK`` -- ``(n_det, n_t)`` uint8, 1 where a sample is flagged
      (``bad_mask``, e.g. from ``flag_spikes.py``).
    * ``Dkk_DETS`` -- table over all ``detector.full_size`` detectors:
      ``X``/``Y`` (``detector.coordinates``) and ``MASKED``; the rows of
      ``Dkk`` are the unmasked detectors in table order. Header carries
      ``FS``, ``TSTART``/``TEND`` (the TOD time axis, s), ``NT`` and ``LABEL``.
    * ``Dkk_PRIOR`` -- ``(n_chan, n_det, n_t)``, the ``ATM``/``MAP`` response
      at ``prior_positions[k - 1]`` (one latent per dataset; only written if given).
    * ``Dkk_PNT`` -- ``(2, n_det, n_t)`` float32, the map response's detector
      pointing in map-pixel coordinates (``map_response.coords``, see
      :func:`maria_nifty.model.response.offsets2coords`), with its projection
      settings ``KIND``/``ORDER``/``MODE``/``CVAL`` -- enough to push any map on
      the same grid through :func:`maria_nifty.model.response.map2tods`. Only
      written if the model has a map response.

    ``header_cards`` (e.g. ``noise_std``, the PRNG ``key``) go into the
    primary header together with ``NDATA``, ``NSAMP``, ``ANTITHET``,
    ``BUNIT`` and the absolute map reference ``CPHI0``/``CTHETA0`` (rad).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    positions = _positions(samples)
    bunit = _bunit(datasets)
    cphi0, ctheta0 = _absolute_reference(datasets)

    primary = fits.Header()
    primary["NDATA"] = (len(datasets), "Number of datasets")
    primary["NSAMP"] = (len(positions), "Posterior states averaged (incl. mirrored)")
    primary["ANTITHET"] = (_is_antithetic(samples), "States are mirrored pairs [s1,-s1,s2,-s2,...]")
    if bunit:
        primary["BUNIT"] = bunit
    if cphi0 is not None:
        primary["CPHI0"] = (float(cphi0), "[rad] RA of the local (0, 0) offset origin")
        primary["CTHETA0"] = (float(ctheta0), "[rad] Dec of the local (0, 0) offset origin")
    for key, val in (header_cards or {}).items():
        primary[key] = val
    hdus = [fits.PrimaryHDU(header=primary)]

    def image_hdu(data, name, channels=(), with_unit=True):
        hdr = fits.Header()
        hdr["EXTNAME"] = name
        if bunit and with_unit:
            hdr["BUNIT"] = bunit
        for i, chan in enumerate(channels):
            hdr[f"CHAN{i}"] = chan
        return fits.ImageHDU(data=data, header=hdr)

    for k, d in enumerate(datasets, start=1):
        model = d["model"]
        ext = f"D{k:02d}"

        chans = [("DATA", np.atleast_2d(np.asarray(d["tod_data"])))]
        if getattr(model, "atm_model", None) is not None:
            chans.append(("ATM_MEAN", _sample_mean(model.atm_model, positions)))
        if getattr(model, "map_response", None) is not None:
            chans.append(("MAP_MEAN", _sample_mean(model.map_response, positions)))
        chans.append(("MODEL_MEAN", _sample_mean(model, positions)))
        cube = np.stack([arr for _, arr in chans], axis=0).astype(np.float32)
        hdus.append(image_hdu(cube, ext, [name for name, _ in chans]))

        bad_mask = d.get("bad_mask")
        bad_mask = np.zeros(cube.shape[1:], dtype=bool) if bad_mask is None else np.asarray(bad_mask, dtype=bool)
        hdus.append(image_hdu(bad_mask.astype(np.uint8), f"{ext}_MASK", with_unit=False))

        detector = d["detector"]
        coords = np.asarray(detector.coordinates, dtype=np.float64)
        dets = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="X", format="D", array=coords[:, 0]),
                fits.Column(name="Y", format="D", array=coords[:, 1]),
                fits.Column(name="MASKED", format="L", array=np.asarray(detector.mask, dtype=bool)),
            ],
            name=f"{ext}_DETS",
        )
        tods = d.get("tods")
        time = np.asarray(d.get("time", []), dtype=np.float64)
        if tods is not None and getattr(tods, "fs", None):
            dets.header["FS"] = (float(tods.fs), "[Hz] sample rate")
        if time.size:
            dets.header["TSTART"] = (float(time[0]), "[s] time of first sample (TOD time axis)")
            dets.header["TEND"] = (float(time[-1]), "[s] time of last sample (TOD time axis)")
        dets.header["NT"] = (int(cube.shape[-1]), "Time samples")
        if labels is not None:
            dets.header["LABEL"] = str(labels[k - 1])[:68]
        hdus.append(dets)

        map_response = getattr(model, "map_response", None)
        if map_response is not None:
            pnt = image_hdu(np.asarray(map_response.coords, dtype=np.float32), f"{ext}_PNT", with_unit=False)
            pnt.header["KIND"] = (str(map_response.kind), "map2tods projection kind")
            pnt.header["ORDER"] = (int(map_response.order), "map2tods interpolation order")
            pnt.header["MODE"] = (str(map_response.mode), "map2tods boundary mode")
            pnt.header["CVAL"] = (float(map_response.cval), "map2tods fill value")
            hdus.append(pnt)

        if prior_positions is not None:
            prior_position = prior_positions[k - 1]
            prior = []
            if getattr(model, "atm_model", None) is not None:
                prior.append(("ATM", np.atleast_2d(np.asarray(model.atm_model(prior_position)))))
            if getattr(model, "map_response", None) is not None:
                prior.append(("MAP", np.atleast_2d(np.asarray(model.map_response(prior_position)))))
            if prior:
                pcube = np.stack([arr for _, arr in prior], axis=0).astype(np.float32)
                hdus.append(image_hdu(pcube, f"{ext}_PRIOR", [name for name, _ in prior]))

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
