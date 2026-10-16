"""Map model consisting of different diffuse and point-source components."""

from functools import reduce
from operator import or_

import jax.nn
import jax.numpy as jnp
import numpy as np
from nifty.re import Model

from .correlation import SpaceCorrelationModel
from .prior import gaussian_prior, normal_prior
from .space import GridSpace, PointSpace, map2space


class MapModel(Model):
    """Combine background and source models into a single map model."""

    def __init__(self, space, source_space=None, background=None, *sources):
        self.space = space
        self.source_space = source_space
        self.background = background
        self.sources = sources
        self.models = models = (self.background,) + self.sources if self.background is not None else self.sources
        self.set_out_space(space)

        super().__init__(
            domain=reduce(or_, [m.domain for m in models]),
            init=reduce(or_, [m.init for m in models]),
        )

    def __call__(self, x):
        """Evaluate the full map model and sum all components."""
        out = self.models[0](x, map=True)
        for m in self.models[1:]:
            out += m(x, map=True)
        return out

    @classmethod
    def build(cls, *, prefix="map", space_params=None, background_params=None, source_params=None):
        """Build a map model from space, background, and source parameters.

        Parameters
        ----------
        prefix : str, optional
            Prefix applied to all component models.
        space_params : dict, optional
            Parameters for the global output space.
        background_params : dict, optional
            Configuration for the diffuse background model.
        source_params : dict, optional
            Mapping of source names to point or diffuse source configurations.

        Returns
        -------
        MapModel
            A composite map model containing background and/or source terms.
        """
        background = None
        if background_params is not None:
            bg_cfg = dict(background_params)
            bg_prefix = prefix.strip() + " " + bg_cfg.pop("prefix", "bg").strip()
            background = DiffuseModel.build(prefix=bg_prefix, **bg_cfg)

        sources = ()
        if source_params is not None:
            if "space_params" in source_params:
                source_params = {"s1": source_params}

            for name, params in source_params.items():
                si_cfg = dict(params)
                si_prefix = prefix.strip() + " " + si_cfg.pop("prefix", name).strip()
                if "point_params" in params:
                    sources += (PointModel.build(prefix=si_prefix, **si_cfg),)
                elif "cfm_params" in params:
                    sources += (DiffuseModel.build(prefix=si_prefix, **si_cfg),)
                else:
                    raise ValueError(f"Could not determine model type for source '{name}'.")

        if len(sources) == 0:
            raise ValueError("At least one source must be specified in 'source_params'.")

        space, source_space = None, None
        if space_params is not None:
            space = source_space = GridSpace.build(**space_params)
        if background is not None:
            models = (background,) + sources if background is not None else sources
            spaces = tuple(m.space for m in models if isinstance(m.space, GridSpace))

            space_dict = background.space.to_dict(keys=("shape", "distances", "center"))
            space_dict["distances"] = np.min(np.stack([s.dis for s in spaces], axis=0), axis=0)
            space_dict["shape"] = tuple(int(fi / di) for fi, di in zip(background.space.fov, space_dict["distances"]))
            space = GridSpace.build(**space_dict)

        if space is None:
            raise ValueError("Either space_params or a background model must be specified.")

        return cls(space, source_space, background, *sources)

    def set_out_space(self, out_space, order=0):
        """Update all submodels to use a new output space."""
        for m in self.models:
            m.set_out_space(out_space, order)

    @property
    def diffuse(self):
        """Tuple of diffuse source components in this map model."""
        return tuple(c for c in self.sources if isinstance(c, DiffuseModel))

    @property
    def points(self):
        """Tuple of point-source components in this map model."""
        return tuple(c for c in self.sources if isinstance(c, PointModel))

    @property
    def source_model(self):
        """A source-only sub-model (background excluded) on the source space."""
        out_space = self.source_space if self.source_space is not None else self.space
        return MapModel(out_space, self.source_space, None, *self.sources)


class DiffuseModel(Model):
    """Diffuse map component built from a correlated field and optional envelope."""

    _NONLINEARITIES = {
        "exp": jnp.exp,
        "softplus": jax.nn.softplus,
    }

    def __init__(self, space, correlation_model, gaussian_model=None, *, nonlinearity=None, negative=False):
        if nonlinearity is not None and nonlinearity not in self._NONLINEARITIES:
            raise ValueError(f"Unknown nonlinearity {nonlinearity!r}; expected one of {sorted(self._NONLINEARITIES)}")
        self.space = space
        self.correlation_model = correlation_model
        self.gaussian_model = gaussian_model
        self.nonlinearity = nonlinearity
        self.negative = negative
        self.map_func = lambda res, x: res

        models = [m for m in (self.space, self.correlation_model, self.gaussian_model) if isinstance(m, Model)]
        super().__init__(
            domain=models[0].domain if len(models) == 1 else (models[0].domain | models[1].domain),
            init=models[0].init if len(models) == 1 else (models[0].init | models[1].init),
        )

    def __call__(self, x, *, map=False):
        """Evaluate the diffuse component and optionally map it to output space."""
        out = self.correlation_model(x)
        if self.nonlinearity is not None:
            out = self._NONLINEARITIES[self.nonlinearity](out)
            if self.negative:
                out = -out
        if self.gaussian_model is not None:
            gmx = self.gaussian_model(x)
            out *= gmx[None] if out.ndim == gmx.ndim + 1 else gmx
        if map:
            out = self.map_func(out, x=x)
        return out

    @classmethod
    def build(
        cls,
        *,
        prefix="diffuse",
        space_params,
        cfm_params,
        padding=1.0,
        gm_params=None,
        nonlinearity=None,
        negative=False,
    ):
        """Build a diffuse model from space, correlated field, and envelope configs.

        Parameters
        ----------
        prefix : str, optional
            Prefix applied to the diffuse component.
        space_params : dict
            Parameters passed to :class:`GridSpace` construction.
        cfm_params : dict
            Parameters passed to :func:`correlated_field_prior`.
        padding : float, optional
            Padding factor used for the internal correlated-field model.
        gm_params : dict, optional
            Optional Gaussian-envelope configuration applied to the field.
        nonlinearity : {None, "exp", "softplus"}, optional
            If given, apply this strictly-positive, monotonically increasing
            function to the raw correlated-field realization before applying
            the Gaussian envelope, giving a strictly one-signed field instead
            of one that can freely cross zero. ``"exp"`` is the classic
            log-normal-like transform (heavy-tailed: rare large fluctuations
            in the underlying Gaussian field blow up multiplicatively).
            ``"softplus"`` (``log(1 + exp(x))``) is a tamer alternative --
            still positive and monotonic, but grows only linearly for large
            ``x`` instead of exponentially, so it is much less prone to the
            wild outlier draws ``"exp"`` can produce for the same prior scale.
        negative : bool, optional
            If True (and ``nonlinearity`` is set), negate the transformed
            field so it is strictly non-positive instead of non-negative
            (e.g. for a component that must be a decrement).

        Returns
        -------
        DiffuseModel
            A diffuse component with optional Gaussian modulation.
        """
        space = GridSpace.build(**space_params)

        cfm_cfg = dict(cfm_params)
        corr_prefix = cfm_cfg.pop("prefix", "cfm")
        correlation_model = SpaceCorrelationModel.build(
            prefix=prefix.strip() + " " + corr_prefix.strip(),
            shape=space.shape,
            cfm_params=cfm_cfg,
            padding=padding,
        )
        # The envelope multiplies the field *after* SpaceCorrelationModel has
        # sliced the padded border away, so it must live on the unpadded shape.
        # These coincide only when padding == 1.0; sizing it from the padded
        # target instead fails to broadcast as soon as padding > 1 is combined
        # with gm_params. Normalizing distances by the same shape also keeps
        # cov_x/cov_y in units of the field width, unchanged by padding.
        g_shape = np.array(space.shape)

        gaussian_model = None
        if gm_params is not None:
            gm_cfg = dict(gm_params)
            gm_prefix = gm_cfg.pop("prefix", "gss")
            gaussian_model = gaussian_prior(
                prefix=prefix.strip() + " " + gm_prefix.strip(),
                shape=g_shape,
                distances=1 / g_shape,
                **gm_cfg,
            )

        return cls(space, correlation_model, gaussian_model, nonlinearity=nonlinearity, negative=negative)

    def set_out_space(self, out_space, order=0):
        """Update the mapping used to project to the output space."""
        self.map_func = map2space(self.space, out_space, order)


class PointModel(Model):
    """Point-source map component projected from a point-space prior."""

    def __init__(self, space, point_model):
        self.space = space
        self.point_model = point_model
        self.map_func = lambda res, x: res

        models = [m for m in (self.space, self.point_model) if isinstance(m, Model)]
        super().__init__(
            domain=models[0].domain if len(models) == 1 else (models[0].domain | models[1].domain),
            init=models[0].init if len(models) == 1 else (models[0].init | models[1].init),
        )

    def __call__(self, x, *, map=False):
        """Evaluate the point-source component and optionally project it."""
        out = self.point_model(x)
        if map:
            out = self.map_func(out, x=x)
        return out

    @classmethod
    def build(cls, *, prefix="point", space_params, point_params):
        """Build a point-source model from space and prior configuration.

        Parameters
        ----------
        prefix : str, optional
            Prefix applied to the point-source component.
        space_params : dict
            Parameters passed to :class:`PointSpace` construction.
        point_params : dict
            Parameters passed to :func:`normal_prior` for the point fluxes.

        Returns
        -------
        PointModel
            A point-source map component.
        """
        space = PointSpace.build(**space_params)

        pm_cfg = dict(point_params)
        pm_prefix = pm_cfg.pop("prefix", "pnt")
        point_model = normal_prior(prefix=prefix.strip() + " " + pm_prefix.strip(), shape=space.shape, **pm_cfg)

        return cls(space, point_model)

    def set_out_space(self, out_space, order=0):
        """Update the mapping used to project point sources to output space."""
        self.map_func = map2space(self.space, out_space, order)
