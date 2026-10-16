"""Shared reconstruction core for the atmosphere/map/joint fit stage scripts.

Builds the forward model (respecting a config's ``fused`` flag), runs a KL fit
with periodic callback plots, and writes ``last.pkl`` under ``rdir``. Used by
``steering/scripts/atmosphere_fit.py`` (map-free), ``map_fit.py``
(atmosphere-free), ``joint_fit.py`` (both, optionally warm-started from an
atmosphere-only fit via :func:`seed_joint_from_atmosphere`), and
``clean_tods.py`` (map-free, which evaluates the fit to subtract it from the
raw TOD instead of just checkpointing it).
"""

import os

from .fuse import setup_fused_forward_model
from .minimize import parse_minimization
from .plot import map_plot, signal_cbar_label, tod_plot
from .util import setup_forward_model, setup_sample_distribution


def load_samples(rdir):
    """Load posterior samples from a prior stage's ``last.pkl`` checkpoint."""
    import pickle

    with open(os.path.join(rdir, "last.pkl"), "rb") as f:
        samples, _ = pickle.load(f)
    return samples


def build_model(config, datasets, verbose=True):
    """Assemble the (fused or summed) forward model for ``datasets``.

    Parameters
    ----------
    config : dict
        Reconstruction config; only the sections relevant to the desired
        model should be present (e.g. drop ``map``/``atmosphere`` beforehand
        to exclude a component regardless of what a caller's on-disk config
        happens to contain -- see :func:`run_kl_fit`).
    datasets : list of dict
        As returned by :func:`maria_nifty.build_detector`.

    Returns
    -------
    likelihood : nifty.re.Likelihood
    datasets : list of dict
        Same list, each entry carrying a ``"model"`` (``SumResponse``) key.
        ``map_model`` and ``atm_model`` are both reachable uniformly via
        ``datasets[0]["model"]`` regardless of ``fused``.
    response : Model or None
        The shared ``FusedResponse`` when ``config["fused"]``, else ``None``
        -- only needed to draw a fused prior (``response.init(key=...)``).
    """
    fused = bool(config.get("fused", False))
    if fused:
        likelihood, response, _ = setup_fused_forward_model(config, datasets, verbose)
    else:
        likelihood, datasets = setup_forward_model(config, datasets, verbose)
        response = None
    return likelihood, datasets, response


def draw_init_position(datasets, response, fused, rng):
    """Draw a fresh prior latent for the current model. Returns ``(init_pos, rng)``."""
    import jax
    import nifty.re as jft

    if fused:
        rng, key = jax.random.split(rng)
        return 0.1 * jft.Vector(response.init(key=key)), rng
    init_tree = {}
    for d in datasets:
        rng, key = jax.random.split(rng)
        init_tree |= d["model"].init(key=key)
    return 0.1 * jft.Vector(init_tree), rng


def seed_joint_from_atmosphere(samples_atm, map_model, rng, scale=0.1):
    """Carry atmosphere-only samples into a joint (atmosphere+map) latent.

    The atmosphere mean and its per-sample residuals are carried over
    verbatim, while a freshly-drawn random map latent is injected for every
    sample (the map mean latent starts at zero, so the per-sample randomness
    lives in the residuals). Used by ``joint_fit.py``'s optional warm start
    from ``atmosphere_fit.py``'s ``last.pkl``.

    Parameters
    ----------
    samples_atm : jft.Samples
        Posterior samples from an atmosphere-only fit.
    map_model : MapModel
        The joint model's map model (for its ``.init()``).
    rng : jax.random.PRNGKey
    scale : float
        Std of the freshly-drawn map latent.

    Returns
    -------
    init_samples : jft.Samples
    rng : jax.random.PRNGKey
    """
    import jax
    import jax.numpy as jnp
    import nifty.re as jft

    n = len(samples_atm)
    rng, k0 = jax.random.split(rng)
    map0 = dict(map_model.init(key=k0))

    pos_atm = samples_atm.pos.tree
    pos_tree = dict(pos_atm)
    for k, v in map0.items():
        pos_tree[k] = jnp.zeros_like(v)

    if n == 0:
        # No residual samples (point estimate): seed a single random map mean.
        rng, ks = jax.random.split(rng)
        for k, v in map_model.init(key=ks).items():
            pos_tree[k] = scale * jnp.asarray(v)
        return jft.Vector(pos_tree), rng

    abs_atm = samples_atm.samples.tree  # absolute atmosphere samples, axis 0 = sample
    delta_tree = {k: abs_atm[k] - pos_atm[k][None, ...] for k in pos_atm}

    stacks = {k: [] for k in map0}
    for _ in range(n):
        rng, ki = jax.random.split(rng)
        mi = map_model.init(key=ki)
        for k in map0:
            stacks[k].append(scale * jnp.asarray(mi[k]))
    for k in map0:
        delta_tree[k] = jnp.stack(stacks[k], axis=0)

    init_samples = jft.Samples(pos=jft.Vector(pos_tree), samples=jft.Vector(delta_tree), keys=samples_atm.keys)
    return init_samples, rng


def run_kl_fit(
    config,
    datasets,
    *,
    rdir,
    pdir,
    with_map,
    with_atmosphere,
    init_samples=None,
    plot_kind="map",
    warm_start_atm=None,
    resume=False,
    verbose=True,
):
    """Run a KL fit and write ``last.pkl`` + periodic callback plots.

    Parameters
    ----------
    config : dict
        Loaded reconstruction config.
    datasets : list of dict
        As returned by :func:`maria_nifty.build_detector`.
    rdir, pdir : str
        This stage's run/plot directories (see :func:`maria_nifty.cli.get_stage_dirs`).
    with_map, with_atmosphere : bool
        Which components to include in the built forward model -- ``map``/
        ``atmosphere`` sections are dropped from a local copy of ``config``
        otherwise, regardless of whether they're present in ``config``.
    init_samples : jft.Samples or jft.Vector, optional
        Starting position/samples. A fresh prior is drawn when omitted (and
        ``warm_start_atm`` is also omitted).
    plot_kind : {"map", "atmosphere"}
        ``"map"`` plots ``map_plot`` + ``tod_plot`` each callback;
        ``"atmosphere"`` plots ``tod_plot`` only.
    warm_start_atm : jft.Samples, optional
        Posterior samples from a prior atmosphere-only fit (e.g.
        ``atmosphere_fit.py``'s ``last.pkl``). When given (only meaningful
        with ``with_map=True, with_atmosphere=True``), seeds the joint fit's
        starting position via :func:`seed_joint_from_atmosphere` instead of a
        fresh prior draw -- needs the joint model's ``map_model``, so this is
        resolved here rather than by the caller.
    resume : bool
        Resume from ``rdir/last.pkl`` if it exists.

    Returns
    -------
    samples : jft.Samples
    datasets : list of dict
        Same list, each entry carrying the built ``"model"``.
    """
    import jax
    import nifty.re as jft
    from nifty.re import logger  # noqa: F401  (kept available for callers that want it)

    model_config = dict(config)
    if not with_map:
        model_config.pop("map", None)
    if not with_atmosphere:
        model_config.pop("atmosphere", None)
    if not with_map and not with_atmosphere:
        raise ValueError("run_kl_fit needs at least one of with_map / with_atmosphere.")

    fused = bool(config.get("fused", False))
    likelihood, datasets, response = build_model(model_config, datasets, verbose)

    rng = jax.random.PRNGKey(config.get("key", 123))
    if warm_start_atm is not None:
        init_samples, rng = seed_joint_from_atmosphere(warm_start_atm, datasets[0]["model"].map_model, rng)
    elif init_samples is None:
        init_samples, rng = draw_init_position(datasets, response, fused, rng)

    clear_every = config.get("clear_every", False)
    plot_every = config.get("plot_every", 1)
    # `plot_tods` only gates the (expensive, per-dataset) TOD plots below -- the
    # single map plot is cheap and always drawn (when plot_kind == "map")
    # regardless of it, so a large `plot_tods: 0` run (set to skip
    # per-detector plotting) still gets a map callback to watch convergence.
    # `plot_every` remains the master on/off switch for periodic plotting.
    plot_tods = int(config.get("plot_tods", len(datasets)))
    plot_datasets = datasets[:plot_tods]

    def plot_iteration(nit, samples):
        pdir_it = os.path.join(pdir, "callback", f"it_{nit:02d}")
        if plot_kind == "map":
            pdir_it_map = os.path.join(pdir_it, "map")
            cbar_scale, cbar_label = signal_cbar_label(datasets[0]["tods"], target_unit=config.get("cbar_unit"))
            map_plot(
                datasets[0]["model"],
                datasets[0]["offsets"],
                samples,
                odir=pdir_it_map,
                center_phi_theta=datasets[0]["center_phi_theta"],
                cbar_label=cbar_label,
                cbar_scale=cbar_scale,
            )
        for i, d in enumerate(plot_datasets):
            pdir_it_tod_i = os.path.join(pdir_it, f"tod_{i + 1:02d}")
            tod_plot(d["tod_data"], d["model"], samples, d["mask"], odir=pdir_it_tod_i)

    if plot_every:
        plot_iteration(0, init_samples)

    def callback(samples, state):
        if plot_every and state.nit % plot_every == 0:
            plot_iteration(state.nit, samples)
        if clear_every and state.nit % clear_every == 0:
            jax.clear_caches()

    mini_dct = parse_minimization(config["minimization"])
    device_kwargs, mini_dct = setup_sample_distribution(config, mini_dct, verbose=verbose)

    rng, key = jax.random.split(rng)
    samples, *_ = jft.optimize_kl(
        likelihood=likelihood,
        position_or_samples=init_samples,
        key=key,
        resume=os.path.join(rdir, "last.pkl") if resume else False,
        callback=callback,
        odir=rdir,
        **device_kwargs,
        **mini_dct,
    )
    return samples, datasets
