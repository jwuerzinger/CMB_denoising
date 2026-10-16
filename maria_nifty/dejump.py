"""Jump removal and spike detection for TODs, built on minkasi.

* :func:`dejump` -- used by ``steering/scripts/remove_jumps.py`` on the raw
  TODs: jumps are fit-and-subtracted
  (:func:`minkasi.tods.processing.fit_jumps_from_cm`), *repairing* the data.
* :func:`find_spike_mask` -- used by ``steering/scripts/flag_spikes.py``:
  spikes are only detected (there's no "before/after level" to reconcile for
  a single-sample outlier) and returned as a mask for the caller to apply
  downstream (e.g. as a per-sample inverse-noise-variance inflation).
"""

import numpy as np
from scipy.ndimage import binary_dilation

# Config section ``dejump`` (read by remove_jumps.py).
DEJUMP_DEFAULTS = {
    "poly_order": 3,  # per-detector polynomial drift order (fit_cm_plus_poly's `ord`)
    "cm_order": 1,  # common-mode coupling order (fit_cm_plus_poly/fit_jumps_from_cm's `cm_order`)
    "niter": 2,  # fit_cm_plus_poly refit iterations -- a single pass leaves structured
    # residual (real atmospheres rarely fit a low-order polynomial perfectly), which
    # otherwise leaks into find_jumps as spurious same-timestamp candidates across
    # every detector (harmless in itself -- the joint fit estimates ~0 height for
    # detectors with no real jump there -- but noisy and worth reducing)
    "jump_width": 0.2,  # matched-step half-window width [seconds] (find_jumps' `width`)
    "jump_pad": 2,  # end-of-TOD samples-in-units-of-width to ignore (find_jumps' `pad`)
    "jump_thresh": 7.0,  # threshold in units of filtered-data MAD (find_jumps' `thresh`)
    "jump_rat": 0.5,  # opposite-sign-neighbor-jump rejection ratio (find_jumps' `rat`)
    "jump_merge_window": 1.0,  # merge candidate indices closer than this [seconds]
}

# Config section ``despike`` (read by flag_spikes.py).
DESPIKE_DEFAULTS = {
    "spike_inner": 1.0,  # inner DoG sigma [samples] (find_spikes' `inner`)
    "spike_outer": 10.0,  # outer DoG sigma [samples] (find_spikes' `outer`)
    "spike_thresh": 8.0,  # threshold in units of filtered-data MAD (find_spikes' `thresh`)
    "spike_margin": 0.2,  # mask window half-width around a detected spike [seconds]
}


def _to_samples(seconds, fs):
    """Convert a second-valued window into a sample count (>= 1)."""
    return max(1, int(round(float(seconds) * fs)))


def _merge_nearby(indices, gap):
    """Collapse a sorted-or-not list of sample indices to one per cluster."""
    if not indices:
        return []
    idx_sorted = sorted(int(x) for x in indices)
    merged = [idx_sorted[0]]
    for idx in idx_sorted[1:]:
        if idx - merged[-1] > gap:
            merged.append(idx)
    return merged


def dejump(dat, fs, cfg):
    """Fit and subtract the jumps in raw TOD data, before any atmosphere fit.

    Estimates a per-detector polynomial drift plus a shared common-mode
    template (``minkasi.tods.processing.fit_cm_plus_poly``), detects jump
    locations on the common-mode-subtracted residual via ``find_jumps``, then
    jointly fits and subtracts the per-detector jump step heights via
    ``fit_jumps_from_cm``. This *repairs* the samples instead of excluding them.

    Parameters
    ----------
    dat : array (n_det, n_time)
        Raw per-detector timestreams (atmosphere still present).
    fs : float
        Sampling rate [Hz].
    cfg : dict
        See :data:`DEJUMP_DEFAULTS`.

    Returns
    -------
    dat_clean : ndarray (n_det, n_time)
        ``dat`` with detected jump steps fit-and-subtracted.
    n_jumps : ndarray (n_det,) int
        Number of jumps removed per detector (informational/logging).
    """
    from minkasi.tods.processing import find_jumps, fit_cm_plus_poly, fit_jumps_from_cm

    dat = np.asarray(dat, dtype=float)
    poly_order = int(cfg["poly_order"])
    cm_order = int(cfg["cm_order"])

    dd, _pred2, cm = fit_cm_plus_poly(dat, ord=poly_order, cm_ord=cm_order, niter=int(cfg["niter"]), full_out=True)

    width = _to_samples(cfg["jump_width"], fs)
    jumps = find_jumps(
        dd, width=width, pad=int(cfg["jump_pad"]), thresh=float(cfg["jump_thresh"]), rat=float(cfg["jump_rat"])
    )

    # find_jumps can return several duplicate/adjacent indices for the same
    # physical discontinuity; merge to one representative per event so
    # fit_jumps_from_cm's per-segment indicator columns don't collapse.
    merge_gap = _to_samples(cfg["jump_merge_window"], fs)
    jumps = [_merge_nearby(j, merge_gap) for j in jumps]
    n_jumps = np.array([len(j) for j in jumps])

    if n_jumps.sum() > 0:
        dat_clean = fit_jumps_from_cm(dat, jumps, cm, cm_order=cm_order, poly_order=poly_order)
    else:
        dat_clean = dat.copy()
    return dat_clean, n_jumps


def find_spike_mask(dat, fs, cfg):
    """Detect narrow impulsive glitches (spikes) and return them as a mask.

    Spikes have no meaningful "before/after level" to fit and subtract, so
    ``minkasi.tods.processing.find_spikes`` (a difference-of-Gaussians filter,
    thresholded per detector against that detector's own median absolute
    deviation) only detects them; the mask is then widened by
    ``spike_margin``. Run it on dejumped data, so a leftover step isn't picked
    up as a run of "spikes" at its edge.

    Parameters
    ----------
    dat : array (n_det, n_time)
        Per-detector timestreams.
    fs : float
        Sampling rate [Hz].
    cfg : dict
        See :data:`DESPIKE_DEFAULTS`.

    Returns
    -------
    bad_mask : ndarray (n_det, n_time) bool
        True at detected spike samples, widened by ``spike_margin``.
    """
    from minkasi.tods.processing import find_spikes

    dat = np.asarray(dat, dtype=float)
    spikes, _ = find_spikes(
        dat, inner=float(cfg["spike_inner"]), outer=float(cfg["spike_outer"]), thresh=float(cfg["spike_thresh"])
    )
    spike_margin = _to_samples(cfg["spike_margin"], fs)
    bad_mask = np.zeros(dat.shape, dtype=bool)
    for i, idx in enumerate(spikes):
        if len(idx):
            bad_mask[i, np.asarray(idx, dtype=int)] = True
    if bad_mask.any():
        bad_mask = binary_dilation(bad_mask, structure=np.ones((1, 2 * spike_margin + 1), dtype=bool))
    return bad_mask
