"""Fused multi-dataset forward model — flat detector axis, chunked scan.

Datasets are concatenated along one flat detector axis (N = sum of per-dataset
detector counts) instead of a padded (M, D, T) dataset axis, since detector
counts vary far more than time length. The axis is split into `batch`-sized
chunks and scanned with a rematerialized `smap`, so peak activation memory
scales with `batch`, not with N.

A single B-spline `basis` (K, T) is shared across datasets; elevation is a
per-dataset post-basis factor (`rel_el`) instead of a doubled basis, so short
datasets can reuse the same knot grid.

An optional per-dataset focal-plane polynomial (`poly_basis`/`poly_model`)
adds a time-constant, per-detector offset built from a 2D monomial basis in
focal-plane coordinates, with coefficients drawn independently per dataset.
Because each TOD has its own parallactic angle, one focal-plane pattern
projects onto a differently-rotated sky pattern in every TOD, while the map
is a single shared object -- so this term is separable from a large-scale map
mode in a way a sky-fixed (map-side) polynomial is not.

`basis`/`coords`/masks are float32; the map and coefficients stay float64
(float32 + float64 -> float64 under x64), so only the constant buffers shrink.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from nifty.re import Model, Vector
from nifty.re.custom_map import smap

from .jax_filter import gaussian_filter


class FusedResponse(Model):
    """Batched flat-detector forward model: stacked spline atmosphere + shared map.

    `map_model`/`coeff_model` may not both be `None`; supplying only one gives
    an atmosphere-only or map-only model. `sequential=True` scans over
    `batch`-sized chunks with rematerialization (low memory); `False` vmaps
    densely (faster, all detectors resident).
    """

    # Dynamic (non-static) pytree leaves, so these large arrays flow through
    # jit as traced inputs rather than being baked in as compile-time constants.
    basis: Any = dataclasses.field(metadata=dict(static=False))
    coords: Any = dataclasses.field(metadata=dict(static=False))
    rel_el: Any = dataclasses.field(metadata=dict(static=False))
    time_valid: Any = dataclasses.field(metadata=dict(static=False))
    det_to_dataset: Any = dataclasses.field(metadata=dict(static=False))
    det_weight: Any = dataclasses.field(metadata=dict(static=False))
    poly_basis: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(
        self,
        map_model,
        coeff_model,
        basis,
        coords=None,
        *,
        poly_basis=None,
        poly_model=None,
        rel_el=None,
        time_valid,
        det_to_dataset,
        det_weight,
        n_knots,
        batch,
        sigma=0.0,
        sequential=True,
        unroll=1,
        order=1,
        mode="constant",
        cval=0.0,
    ):
        if map_model is None and coeff_model is None:
            raise ValueError("FusedResponse needs a `map_model`, a `coeff_model`, or both.")
        self.map_model = map_model  # None -> atmosphere-only
        self.coeff_model = coeff_model  # None -> map-only
        # Optional per-dataset focal-plane polynomial (both None -> disabled).
        if (poly_basis is None) != (poly_model is None):
            raise ValueError("`poly_basis` and `poly_model` must be given together.")
        self.poly_model = poly_model
        self.poly_basis = None if poly_basis is None else jnp.asarray(poly_basis)
        self.basis = None if basis is None else jnp.asarray(basis)
        self.coords = None if coords is None else jnp.asarray(coords)
        self.rel_el = None if rel_el is None else jnp.asarray(rel_el)
        self.time_valid = jnp.asarray(time_valid)
        self.det_to_dataset = jnp.asarray(det_to_dataset)
        self.det_weight = jnp.asarray(det_weight)
        self.n_knots = int(n_knots)
        self.batch = int(batch)
        self.sigma = float(sigma)
        self.sequential = bool(sequential)
        self.unroll = int(unroll)
        self.order = int(order)  # map_coordinates' own args (e.g. order=0, mode="nearest")
        self.mode = str(mode)
        self.cval = float(cval)
        if map_model is None:
            dom, ini = coeff_model.domain, coeff_model.init
        elif coeff_model is None:
            dom, ini = map_model.domain, map_model.init
        else:
            dom, ini = map_model.domain | coeff_model.domain, map_model.init | coeff_model.init
        if self.poly_model is not None:
            dom, ini = dom | self.poly_model.domain, ini | self.poly_model.init
        super().__init__(domain=Vector(dom), init=ini)

    def _atm(self, cf, ds):
        """Spline atmosphere for one detector block: ``(b, C) -> (b, T)``."""
        b = self.basis  # (K, T)
        if self.rel_el is None:
            return cf @ b
        k = self.n_knots
        rel = self.rel_el[ds]  # (b, T)
        return (cf[:, :k] @ b) * rel + (cf[:, k:] @ b) * (1.0 - rel)

    def _poly(self, x):
        """Per-detector focal-plane polynomial offset: ``(N,)`` or ``None``.

        Time-constant, so it is evaluated once outside the detector scan and
        broadcast over the time axis inside it.  ``poly_model`` holds one
        independent coefficient row per dataset; ``det_to_dataset`` gathers
        each detector's row.
        """
        if self.poly_model is None:
            return None
        pc = self.poly_model(x)  # (M, n_terms)
        return (self.poly_basis * pc[self.det_to_dataset]).sum(axis=-1)  # (N,)

    def __call__(self, x):
        """Evaluate the stacked atmosphere (+ map projection) -> ``(N, T)``."""
        coeff = None if self.coeff_model is None else self.coeff_model(x)  # (N, C) or None
        n = self.det_to_dataset.shape[0] if coeff is None else coeff.shape[0]

        sky_map = None
        if self.map_model is not None:
            map_field = self.map_model(x)
            sky_map = map_field if self.sigma <= 0.0 else gaussian_filter(map_field, sigma=self.sigma)

        nb = self.batch
        n_chunks = n // nb
        ds_c = self.det_to_dataset.reshape(n_chunks, nb)
        dw_c = self.det_weight.reshape(n_chunks, nb)
        coeff_c = None if coeff is None else coeff.reshape(n_chunks, nb, coeff.shape[1])
        # Time-constant focal-plane offset, evaluated once and broadcast over
        # time inside the scan (zeros when the term is disabled).
        pv = self._poly(x)
        pv = jnp.zeros((n,), dtype=self.det_weight.dtype) if pv is None else pv
        pv_c = pv.reshape(n_chunks, nb)

        if self.sequential:
            # Rematerialized scan over detector chunks: recompute in backward
            # instead of storing the (b, T, 2) map_coordinates adjoint.
            if self.map_model is None:

                def one(cf_ds_dw_pv):
                    cf, ds, dw, pvb = cf_ds_dw_pv
                    tv = self.time_valid[ds]  # (b, T)
                    return (self._atm(cf, ds) + pvb[:, None]) * tv * dw[:, None]

                out = smap(jax.checkpoint(one), unroll=self.unroll)((coeff_c, ds_c, dw_c, pv_c))
                return out.reshape(n, -1)

            coords_c = self.coords.reshape(n_chunks, nb, 2, self.coords.shape[-1])

            if self.coeff_model is None:

                def one(ds_dw_co_pv):
                    ds, dw, co, pvb = ds_dw_co_pv
                    # (2,b,T) -> (b, T)
                    proj = map_coordinates(sky_map, co.transpose(1, 0, 2), self.order, mode=self.mode, cval=self.cval)
                    tv = self.time_valid[ds]
                    return (proj + pvb[:, None]) * tv * dw[:, None]

                out = smap(jax.checkpoint(one), unroll=self.unroll)((ds_c, dw_c, coords_c, pv_c))
                return out.reshape(n, -1)

            def one(cf_ds_dw_co_pv):
                cf, ds, dw, co, pvb = cf_ds_dw_co_pv
                # (2,b,T) -> (b, T)
                proj = map_coordinates(sky_map, co.transpose(1, 0, 2), self.order, mode=self.mode, cval=self.cval)
                tv = self.time_valid[ds]
                return (proj + self._atm(cf, ds) + pvb[:, None]) * tv * dw[:, None]

            out = smap(jax.checkpoint(one), unroll=self.unroll)((coeff_c, ds_c, dw_c, coords_c, pv_c))
            return out.reshape(n, -1)

        # Dense (jax.vmap) path — faster but materializes all N detectors at once.
        tv_full = self.time_valid[self.det_to_dataset]  # (N, T)
        dw = self.det_weight[:, None]
        pvd = pv[:, None]  # (N, 1), zeros when the poly term is disabled
        atm = None if coeff is None else self._atm(coeff, self.det_to_dataset)
        if self.map_model is None:
            return (atm + pvd) * tv_full * dw
        proj = jax.vmap(lambda co: map_coordinates(sky_map, co, self.order, mode=self.mode, cval=self.cval))(
            self.coords
        )  # (N, T)
        if atm is None:
            return (proj + pvd) * tv_full * dw
        return (atm + proj + pvd) * tv_full * dw


class FusedDatasetAtm(Model):
    """Per-dataset view into the stacked spline coefficients, for plotting only."""

    def __init__(self, coeff_model, basis, det_start, n_det, n_knots, n_time, rel_el=None):
        self.coeff_model = coeff_model
        self.basis = jnp.asarray(basis)  # (K, T) shared
        self.det_start = int(det_start)
        self.n_det = int(n_det)
        self.n_knots = int(n_knots)
        self.n_time = int(n_time)
        self.rel_el = None if rel_el is None else jnp.asarray(rel_el)  # (n_time,)
        super().__init__(domain=coeff_model.domain, init=coeff_model.init)

    def __call__(self, x):
        coeff = self.coeff_model(x)  # (N, C)
        c = coeff[self.det_start : self.det_start + self.n_det]  # (n_det, C)
        b = self.basis[:, : self.n_time]  # (K, n_time)
        if self.rel_el is None:
            return c @ b
        k = self.n_knots
        rel = self.rel_el[None, :]  # (1, n_time)
        return (c[:, :k] @ b) * rel + (c[:, k:] @ b) * (1.0 - rel)
