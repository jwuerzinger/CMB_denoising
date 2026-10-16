"""Config/host-side assembly for the fused forward model in :mod:`.model.fused`.

Concatenates per-dataset TODs/coords onto one flat detector axis and builds
the :class:`FusedResponse` inputs. Only the spline atmosphere mode is
supported, optionally plus a per-dataset focal-plane polynomial
(``atmosphere.poly_params``) that soaks up instrument-fixed large-scale
residual ("bowling") without handing that freedom to the map.
"""

import jax.numpy as jnp
import numpy as np
from maria import utils as maria_utils
from nifty.re import Gaussian, logger

from .model.fused import FusedDatasetAtm, FusedResponse
from .model.map import MapModel
from .model.polynomial import poly2d_basis, poly2d_n_terms
from .model.prior import normal_prior
from .model.response import MapResponse, SumResponse, offsets2coords

COEFF_KEY = "atm coeffs"
POLY_KEY = "atm poly coeffs"

# Constant buffers (geometry / basis / masks) are stored at this precision; the
# latent map and coefficients stay float64. float32 ⊕ float64 -> float64 under
# x64, so accumulation/optimization stays float64 while these buffers halve.
SMALL_DTYPE = np.float32


def _has_atmosphere(config):
    """Whether the config carries a (non-empty) ``atmosphere`` section."""
    return len(config.get("atmosphere") or {}) > 0


def _spline_params(config):
    """Extract spline params from the atmosphere config (mode must be spline)."""
    atm = dict(config.get("atmosphere", {}))
    mode = str(atm.get("mode", "spline")).strip().lower()
    if mode != "spline":
        raise ValueError(f"Fused model currently supports only `spline` atmosphere, not `{mode}`.")
    sp = dict(atm.get("spline_params", atm.get("spline", {})))
    coeff_prior = dict(sp.get("coeff_prior", {}))
    return {
        "knot_spacing": float(sp["knot_spacing"]),
        "order": int(sp.get("order", 3)),
        "remove_el_gradient": bool(sp.get("remove_el_gradient", False)),
        "mean": float(coeff_prior.get("mean", 0.0)),
        "std": float(coeff_prior.get("std", 0.05)),
        # Per-detector scaling (vs. one absolute std) matters because one
        # dataset's atmosphere amplitude can be 10-100x another's.
        "scale_std_with_data": bool(coeff_prior.get("scale_with_data", False)),
    }


def _poly_params(config):
    """Focal-plane polynomial params from ``atmosphere.poly_params``, or ``None``.

    Returns ``{"order", "mean", "std"}``.  Absent section (or ``order`` below
    zero) disables the term.
    """
    atm = dict(config.get("atmosphere", {}))
    pp = atm.get("poly_params", atm.get("poly"))
    if not pp:
        return None
    pp = dict(pp)
    order = int(pp.get("order", 1))
    if order < 0:
        return None
    coeff_prior = dict(pp.get("coeff_prior", {}))
    return {
        "order": order,
        "mean": float(coeff_prior.get("mean", 0.0)),
        "std": float(coeff_prior.get("std", 1.0)),
    }


def _build_poly_basis(datasets, order, ndet, det_start, n_pad, small_dtype):
    """Per-detector focal-plane monomial basis, ``(n_pad, n_terms)``.

    Each dataset's detector coordinates are normalized by that dataset's own
    max ``|coordinate|`` per axis (matching
    :class:`~maria_nifty.model.polynomial.SpacePolynomialModel`), so every
    focal plane maps onto the same ``[-1, 1]`` box and the coefficient prior
    means the same thing for all datasets.  Padded rows stay zero, so their
    contribution vanishes regardless of the coefficient values.
    """
    n_terms = poly2d_n_terms(order)
    out = np.zeros((n_pad, n_terms), dtype=small_dtype)
    for i, d in enumerate(datasets):
        det = d.get("detector")
        if det is None:
            raise ValueError(f"Dataset {i}: `atmosphere.poly_params` needs `detector` (run build_detector first).")
        coords = np.asarray(det.coordinates, dtype=np.float64)
        mask = getattr(det, "mask", None)
        if mask is not None and np.any(mask) and coords.shape[0] != ndet[i]:
            coords = coords[~np.asarray(mask, dtype=bool)]
        if coords.shape[0] != ndet[i]:
            raise ValueError(
                f"Dataset {i}: detector coordinates ({coords.shape[0]}) do not match the TOD's "
                f"detector count ({ndet[i]}); cannot build the focal-plane polynomial basis."
            )
        scale = np.max(np.abs(coords), axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        basis = np.asarray(poly2d_basis(jnp.asarray(coords / scale), order))
        s0 = int(det_start[i])
        out[s0 : s0 + ndet[i]] = basis.astype(small_dtype)
    return out


def build_fused_arrays(config, datasets, map_model, batch=256, small_dtype=SMALL_DTYPE, verbose=True):
    """Concatenate per-dataset TODs/coords onto a flat, batch-padded detector axis.

    A single shared B-spline basis is built on the longest time grid; shorter
    datasets are masked via ``time_valid``. ``map_model=None`` -> no coords
    (atmosphere-only).
    """
    has_atm = _has_atmosphere(config)
    sp = _spline_params(config) if has_atm else None
    el_grad = sp["remove_el_gradient"] if has_atm else False
    space = None if map_model is None else map_model.space

    ndet = [int(np.asarray(d["tod_data"]).shape[0]) for d in datasets]
    ntime = [int(np.asarray(d["tod_data"]).shape[1]) for d in datasets]
    m0 = len(datasets)

    # Shared basis on the longest time grid; a common cadence across datasets
    # is assumed, so column j is the same elapsed time everywhere.
    i_ref = int(np.argmax(ntime))
    big_t = ntime[i_ref]

    if has_atm:
        t_ref = np.asarray(datasets[i_ref]["tods"].time[:big_t], dtype=np.float64)
        t_ref = t_ref - t_ref[0]
        basis = maria_utils.signal.bspline_basis(t_ref, spacing=sp["knot_spacing"], order=sp["order"])  # (K, T)
        n_knots = int(basis.shape[0])
        big_c = 2 * n_knots if el_grad else n_knots
    else:
        basis = None
        n_knots = 0
        big_c = 0

    # Flat detector-axis layout, padded up to a multiple of `batch`.
    n = int(sum(ndet))
    n_chunks = (n + batch - 1) // batch
    n_pad = n_chunks * batch
    det_start = np.concatenate([[0], np.cumsum(ndet)[:-1]]).astype(int)

    tod = np.zeros((n_pad, big_t), dtype=np.float64)
    inv_var = np.zeros((n_pad, big_t), dtype=np.float64)
    coords = None if space is None else np.zeros((n_pad, 2, big_t), dtype=small_dtype)
    rel_el = np.zeros((m0, big_t), dtype=small_dtype) if el_grad else None
    time_valid = np.zeros((m0, big_t), dtype=small_dtype)
    det_to_dataset = np.zeros((n_pad,), dtype=np.int32)
    det_weight = np.zeros((n_pad,), dtype=small_dtype)
    default_inv_var = float(config["noise_std"]) ** -2 if config.get("noise_std") else None

    for i, d in enumerate(datasets):
        nv, nt = ndet[i], ntime[i]
        s = int(det_start[i])
        tod[s : s + nv, :nt] = np.asarray(d["tod_data"], dtype=np.float64)
        if d.get("noise_cov_inv") is not None:
            inv_var[s : s + nv, :nt] = np.asarray(d["noise_cov_inv"], dtype=np.float64)
        elif d.get("bad_mask") is not None and np.any(d["bad_mask"]):
            # No dedicated noise estimate: global scalar noise_std everywhere
            # except the masked samples, which get a near-zero weight.
            if default_inv_var is None:
                raise ValueError(f"Dataset {i}: has `bad_mask` but no `noise_std` fallback in config.")
            tile = np.full((nv, nt), default_inv_var)
            tile[d["bad_mask"]] = float(config.get("bad_sample_inv_var", 1.0e-6))
            inv_var[s : s + nv, :nt] = tile
        elif default_inv_var is not None:
            inv_var[s : s + nv, :nt] = default_inv_var
        else:
            raise ValueError(f"Dataset {i}: no `noise_cov_inv` and no `noise_std` fallback in config.")
        det_to_dataset[s : s + nv] = i
        det_weight[s : s + nv] = 1.0
        time_valid[i, :nt] = 1.0
        if el_grad:
            if d.get("boresight_el") is None:
                raise ValueError(
                    f"Dataset {i}: no `boresight_el` available (elevation is required for `remove_el_gradient`)."
                )
            el = np.asarray(d["boresight_el"][:nt], dtype=np.float64)
            ptp = el.max() - el.min()
            if ptp == 0:
                raise ValueError(f"Dataset {i}: constant elevation; cannot build elevation gradient.")
            rel_el[i, :nt] = (el - el.min()) / ptp
        if coords is not None:
            c = offsets2coords(d["offsets"], space.shape, space.fov, space.center)  # (2, nv, nt)
            coords[s : s + nv, :, :nt] = np.transpose(c, (1, 0, 2)).astype(small_dtype)

    if basis is not None:
        basis = basis.astype(small_dtype)

    pp = _poly_params(config) if has_atm else None
    poly_basis = None if pp is None else _build_poly_basis(datasets, pp["order"], ndet, det_start, n_pad, small_dtype)
    n_poly_terms = 0 if pp is None else poly2d_n_terms(pp["order"])

    if verbose:
        logger.info(
            f"\nFused datasets: M={m0}, N={n} dets (padded to {n_pad}, batch={batch}, "
            f"{n_chunks} chunks), T={big_t}, K={n_knots}, C={big_c}; stacked tod {tod.shape}"
            + ("" if pp is None else f"\n  focal-plane polynomial: order={pp['order']}, {n_poly_terms} terms/dataset")
        )
    return {
        "tod": tod,
        "inv_var": inv_var,
        "coords": coords,
        "basis": basis,
        "poly_basis": poly_basis,
        "n_poly_terms": n_poly_terms,
        "rel_el": rel_el,
        "time_valid": time_valid,
        "det_to_dataset": det_to_dataset,
        "det_weight": det_weight,
        "M0": m0,
        "N": n,
        "N_pad": n_pad,
        "batch": batch,
        "n_chunks": n_chunks,
        "T": big_t,
        "K": n_knots,
        "C": big_c,
        "el_grad": el_grad,
        "ndet": ndet,
        "ntime": ntime,
        "det_start": det_start.tolist(),
    }


def build_fused_plot_models(datasets, fused, map_model, coeff_model, sigma, order=1, mode="constant", cval=0.0):
    """Attach a per-dataset ``SumResponse`` view (``d["model"]``), for plotting/FITS only."""
    basis = None if fused["basis"] is None else np.asarray(fused["basis"])  # (K, T)
    n_knots = fused["K"]
    el_grad = fused["el_grad"]
    rel_all = None if not el_grad else np.asarray(fused["rel_el"])
    for i, d in enumerate(datasets):
        nd, nt = fused["ndet"][i], fused["ntime"][i]
        start = fused["det_start"][i]
        rel = None if rel_all is None else rel_all[i, :nt]
        # map-only when there is no coefficient model (no atmosphere)
        atm = None if coeff_model is None else FusedDatasetAtm(coeff_model, basis, start, nd, n_knots, nt, rel_el=rel)
        # atmosphere-only when there is no map model
        map_response = (
            None
            if map_model is None
            else MapResponse(map_model, d["offsets"], sigma, order=order, mode=mode, cval=cval)
        )
        d["model"] = SumResponse(atm, map_model, map_response)
    return datasets


def setup_fused_forward_model(config, datasets, verbose=True, *, batch=None, sequential=True, unroll=1):
    """Assemble a single batched Gaussian likelihood over all datasets.

    Returns ``(likelihood, response, fused)``. ``sequential=True`` scans over
    ``batch``-sized chunks (memory-bound); ``False`` vmaps densely (faster).
    """
    has_atm = _has_atmosphere(config)
    # No "map" section -> atmosphere-only fused model (no map projection).
    map_model = MapModel.build(**config["map"]) if "map" in config else None
    if not has_atm and map_model is None:
        raise ValueError("Fused model needs an `atmosphere` section, a `map` section, or both.")

    response_cfg = config.get("response", {})
    sigma = response_cfg.get("sigma")
    if sigma is None and map_model is not None:
        sigma = 12.362925440458655 * 0.00155586 / map_model.space.fov[0]
    sigma = sigma or 0.0
    order = int(response_cfg.get("order", 1))
    mode = str(response_cfg.get("mode", "constant"))
    cval = float(response_cfg.get("cval", 0.0))

    if batch is None:
        batch = int(config.get("fused_batch", 256))

    fused = build_fused_arrays(config, datasets, map_model, batch=batch, verbose=verbose)

    coeff_model = None
    if has_atm:
        sp = _spline_params(config)
        std = sp["std"]
        if sp["scale_std_with_data"]:
            # Per-detector prior width, scaled by each detector's own
            # raw-signal std. Padded rows are all-zero (std=0), which is
            # harmless since their coefficients never enter the likelihood.
            std = std * np.std(fused["tod"], axis=1, keepdims=True)  # (N_pad, 1), broadcasts over C
        coeff_model = normal_prior(COEFF_KEY, (fused["N_pad"], fused["C"]), mean=sp["mean"], std=std)

    # One independent coefficient set per dataset (per TOD), so the same
    # focal-plane pattern lands at a different parallactic angle in each.
    poly_model = None
    pp = _poly_params(config) if has_atm else None
    if pp is not None:
        poly_model = normal_prior(POLY_KEY, (fused["M0"], fused["n_poly_terms"]), mean=pp["mean"], std=pp["std"])

    # Per-dataset SumResponse views for plotting/FITS.
    build_fused_plot_models(datasets, fused, map_model, coeff_model, sigma, order=order, mode=mode, cval=cval)

    response = FusedResponse(
        map_model,
        coeff_model,
        fused["basis"],
        fused["coords"],
        poly_basis=fused["poly_basis"],
        poly_model=poly_model,
        rel_el=fused["rel_el"],
        time_valid=fused["time_valid"],
        det_to_dataset=fused["det_to_dataset"],
        det_weight=fused["det_weight"],
        n_knots=fused["K"],
        batch=fused["batch"],
        sigma=sigma,
        sequential=sequential,
        unroll=unroll,
        order=order,
        mode=mode,
        cval=cval,
    )

    # Padded/short regions carry zero residual (data zero-padded, forward
    # masks its output) and zero inv_var, so they contribute nothing regardless.
    inv_var = fused["inv_var"]
    noise_cov_inv = lambda x, _inv=inv_var: _inv * x  # noqa: E731

    likelihood = Gaussian(fused["tod"], noise_cov_inv).amend(response)
    if verbose:
        dom = likelihood.domain
        keys = dom.tree.keys() if hasattr(dom, "tree") else dom.keys()
        logger.info(f"  fused likelihood domain keys: {sorted(keys)}")
    return likelihood, response, fused
