"""Histogram plots for model coefficient distributions."""

import os

import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np


def _extract_keys(obj):
    """Return a list of keys from a mapping-like object."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.keys())

    if hasattr(obj, "keys"):
        try:
            return list(obj.keys())
        except Exception:
            pass

    tree = getattr(obj, "tree", None)
    if isinstance(tree, dict):
        return list(tree.keys())

    return []


def _extract_value(container, key):
    """Look up a key in dict-like containers or sample trees."""
    if isinstance(container, dict):
        return container[key]

    if hasattr(container, "__getitem__"):
        try:
            return container[key]
        except Exception:
            pass

    tree = getattr(container, "tree", None)
    if isinstance(tree, dict) and key in tree:
        return tree[key]

    raise KeyError(f"Could not extract key '{key}' from sample container")


def _as_sample_list(samples):
    """Normalize a sample container into a plain list of samples."""
    if samples is None:
        return []

    if isinstance(samples, jft.Vector):
        return [samples]

    try:
        return list(samples)
    except TypeError as exc:
        raise TypeError("samples has to be iterable or a jft.Vector") from exc


def _infer_temporal_coeff_key(sample, model):
    """Infer the sample key that stores temporal polynomial coefficients."""
    if model is None:
        raise ValueError("model is required to resolve temporal polynomial coefficient keys")

    atm_model = getattr(model, "atm_model", None)
    if atm_model is None:
        raise ValueError("model has no atm_model; cannot locate temporal polynomial coefficients")

    # New atmosphere API stores additive components in `additive_models`.
    tmp_model = getattr(atm_model, "tmp_model", None)
    if tmp_model is None:
        additive_models = list(getattr(atm_model, "additive_models", []))
        for additive in additive_models:
            if additive.__class__.__name__ == "TimePolynomialModel":
                tmp_model = additive
                break

    if tmp_model is None:
        raise ValueError("No temporal polynomial model present in atm_model.")

    sample_keys = _extract_keys(sample)
    if not sample_keys:
        raise ValueError("Could not inspect keys of first sample")

    preferred = []
    tmp_prior = getattr(tmp_model, "model", None)
    preferred.extend(_extract_keys(getattr(tmp_prior, "domain", None)))
    preferred.extend(_extract_keys(getattr(tmp_model, "domain", None)))

    for key in preferred:
        if "temporal coeffs" in str(key) and key in sample_keys:
            return key

    fallback = [
        k for k in sample_keys if (("temporal coeffs" in str(k)) or (("poly" in str(k)) and ("coeffs" in str(k))))
    ]
    if len(fallback) == 1:
        return fallback[0]
    if len(fallback) > 1:
        return sorted(fallback)[0]

    raise KeyError("Could not find temporal polynomial coefficient key in samples")


def coefficient_histograms(
    model=None,
    samples=None,
    odir=None,
):
    """Create histograms of atmosphere temporal polynomial coefficient samples."""
    sample_list = _as_sample_list(samples)
    if len(sample_list) == 0:
        raise ValueError("No samples provided")

    try:
        coeff_key = _infer_temporal_coeff_key(sample_list[0], model)
    except ValueError as exc:
        # No temporal polynomial component is a valid setup; skip this plot.
        if "No temporal polynomial model present" in str(exc):
            return
        raise

    coeff_matrices = []
    for sample in sample_list:
        temporal_coeffs = np.asarray(_extract_value(sample, coeff_key))

        if temporal_coeffs.ndim == 0:
            temporal_coeffs = temporal_coeffs.reshape(1, 1)
        elif temporal_coeffs.ndim == 1:
            temporal_coeffs = temporal_coeffs[None, :]
        else:
            temporal_coeffs = temporal_coeffs.reshape(-1, temporal_coeffs.shape[-1])

        coeff_matrices.append(temporal_coeffs)

    all_coeffs = np.concatenate(coeff_matrices, axis=0)
    n_coefficients = all_coeffs.shape[1]

    n_cols = min(n_coefficients, 5)
    n_rows = int(np.ceil(n_coefficients / n_cols))
    figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel() if hasattr(axes, "ravel") else np.array([axes])

    for i in range(n_coefficients):
        ax = axes[i]
        coeff_values = all_coeffs[:, i]
        ax.hist(coeff_values, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Coefficient Value", fontsize=10)
        ax.set_ylabel("Counts" if i == 0 else "", fontsize=10)
        ax.set_title(f"Coefficient {i}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        mean_val = np.mean(coeff_values)
        std_val = np.std(coeff_values)
        ax.text(
            0.98,
            0.97,
            f"mean={mean_val:.3f}\nstd={std_val:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    for i in range(n_coefficients, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Polynomial Model: Coefficient Distributions", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()

    if odir:
        os.makedirs(odir, exist_ok=True)
        histogram_file = f"{odir}/polynomial_coeffs.png"
        fig.savefig(histogram_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
