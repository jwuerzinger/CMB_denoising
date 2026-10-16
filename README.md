# CMB_denoising

Gaussian process-based imaging of (sub-)millimeter single-dish telescope data combining
[Maria](https://github.com/thomaswmorris/maria) (telescope simulation and data pipeline),
[NIFTy.re](https://ift.pages.mpcdf.de/nifty/) (numerical information field theory), and
[JAX](https://jax.readthedocs.io/) (GPU-accelerated automatic differentiation).

Designed for analyzing time-ordered data (TODs) from (sub-)millimeter single-dish
telescopes such as MUSTANG-2, enabling joint reconstruction of sky maps and atmospheric
fluctuations within a principled Bayesian framework.

- [maria_nifty/](maria_nifty/) – the library (models, fitting, plotting)
- [steering/](steering/) – scripts and YAML configs for simulation, preprocessing, fitting and plotting
- [tests/](tests/) – unit tests (`pixi run test`)
- [demo/](demo/) – a small self-contained demo

If you're using this repo for your own research, please cite https://arxiv.org/abs/2509.01600.
Don't hesitate to reach out directly or open an issue in case you spot any problems!


## Installation

Clone the repo:

```sh
git clone https://github.com/jwuerzinger/CMB_denoising.git
cd CMB_denoising
```

### Recommended: Pixi

[Pixi](https://pixi.sh/) manages both conda and PyPI dependencies in a fully reproducible
environment. Run the commands below from the repo root.

**GPU (CUDA 12.6):**
```sh
pixi install
```

**CPU only:**
```sh
pixi install -e cpu
```

### Pip fallback

Requires Python ≥ 3.11. Install the package and its core dependencies:

```sh
pip install .
```

`maria` and `minkasi` are not on PyPI and must be installed from their Git repositories
(see `pixi.toml` for the pinned revisions).


## Quick start

This example reconstructs the galaxy cluster RX J1347.5-1145 from MUSTANG-2 observations.

### 1. Download the data

The MUSTANG-2 time-ordered data (TODs) for RX J1347 are publicly available from
[sievers/M2-TODs](https://github.com/sievers/M2-TODs) as compressed FITS files
(about 525 MB). Download them and unpack them into `steering/data/RXJ1347/`, which is
where the RX J1347 configs in `steering/config/` look for them. Afterwards,
delete the `M2-TODs` clone, so that it doesn't end up in the repo by accident:

```sh
git clone --depth 1 https://github.com/sievers/M2-TODs.git
mkdir -p steering/data/RXJ1347
cp M2-TODs/RXJ1347/*.fits.bz2 steering/data/RXJ1347/
bunzip2 steering/data/RXJ1347/*.fits.bz2
rm -rf M2-TODs
```

The folder now holds the `Signal_TOD-AGBT*.fits` files, ready to use. Since
`steering/data/` is git-ignored, the data will never be committed by accident.

### 2. Run the reconstruction

`joint_fit.py` reconstructs the sky map and the atmosphere together. The demo config
`steering/config/demo_2s.yml` uses four of the scans and a short iteration schedule,
which makes it a good first run; its comments explain every setting. Run it from the
`steering/` folder, because the paths in the config files are relative to it:

```sh
cd steering
pixi run python scripts/joint_fit.py --config config/demo_2s.yml --cuda_device 0
```

To run on a CPU instead, use `pixi run -e cpu` and leave out `--cuda_device`.

Where the results end up is set by `odir` and `run` in the config:

- `<odir>/run_<run>/joint_fit/` holds the posterior samples (`last.pkl`), a copy of
  the config used, and a convergence summary (`minisanity.txt`).
- `<odir>/plot_<run>/joint_fit/` collects diagnostic plots, refreshed every
  `plot_every` iterations while the fit runs.

To start a fresh run with the same settings, increase `run:` in the config.


## Workflow

The scripts below cover the full chain from raw TODs to final figures. They all live in
`steering/scripts/` and are run from `steering/`. Every script lists its options with
`--help`. Scripts that run JAX models select a GPU with `--cuda_device 0` (or a
comma-separated list) and fall back to the CPU when the option is left out.

### Simulation

Generate synthetic TODs, for example to test a reconstruction against a known sky.

| Script | Purpose |
|---|---|
| `simulate.py` | Simulates TODs with Maria from a config and saves the true sky and atmosphere for comparison |

`steering/config/sim_mustang.yml` (MUSTANG-2 at Green Bank) and
`steering/config/sim_atlast.yml` (a 1° AtLAST array at Chajnantor) are example configs.
The instrument is either a pre-defined Maria instrument or a custom one given in the
config, with its own array and bands:

```bash
cd steering
pixi run python scripts/simulate.py --config config/sim_mustang.yml --cuda_device 0
```

### Preprocessing

Optional cleanup of the raw TODs before fitting. The stages are chained: each one
writes its cleaned TODs together with a new config,
`<odir>/run_<run>/<stage>/config.yaml`, which you pass as `--config` to the next stage.

| Script | Purpose |
|---|---|
| `remove_jumps.py` | Finds the jumps in the raw TODs and subtracts them, using Minkasi |
| `clean_tods.py` | Subtracts a quick atmosphere-only fit from the TODs |
| `flag_spikes.py` | Detects and masks spikes, drops badly affected detectors, and estimates the noise level of every detector |

### Reconstruction

Fit the data, then export the result. A joint fit is the usual choice; an atmosphere-only
fit can serve as its starting point via `--warm_start`.

| Script | Purpose |
|---|---|
| `bin_mapper.py` | Makes a simple binned map with Maria's bin mapper, as a quick baseline |
| `atmosphere_fit.py` | Fits the atmosphere only, without a sky map |
| `map_fit.py` | Fits the sky map only, for TODs whose atmosphere has already been removed |
| `joint_fit.py` | Fits the sky map and the atmosphere together |
| `save_fits.py` | Exports a finished fit to FITS files |

`save_fits.py` writes the posterior of a fit to `<odir>/plot_<run>/save_fits/`:

- `map_mean.fits` – the mean map and the mean of every map component
- `map_samples.fits` – every posterior sample of the map, plus draws from the prior
- `tod_mean.fits` – the data and the mean reconstructed TODs of every scan

The plotting and analysis scripts below work from these files alone, so they need
neither the checkpoint nor the raw TODs.

### Plotting

| Script | Purpose |
|---|---|
| `plot_tods.py` | Gives a first look at the raw TODs and the detector layout |
| `plot_from_fits.py` | Draws the final figures (maps, uncertainties, samples, power spectra, TODs) from a `save_fits.py` export |

To compare the reconstruction against an existing map, pass that map to
`plot_from_fits.py` with `--fits_map`.


## Configuration

Reconstructions are controlled by YAML files with `---`-delimited sections.

| Section | Purpose |
|---|---|
| `odir` | Output directory for results and plots |
| `data` | TOD file paths with optional per-detector masks |
| `detector` | Detector geometry (derived from TODs or a hex-ring layout) |
| `map` | Sky-map model: spatial grid, correlated-field priors, point sources |
| `atmosphere` | Atmosphere model: `spline`, `gp-1d`, or `gp-3d` |
| `response` | Beam convolution parameters |
| `noise_std` | Per-sample noise standard deviation |
| `minimization` | Iteration schedule and per-step solver settings |

Annotated examples are in `steering/config/demo_*.yml`.

---

## Dependencies

| Dependency | Purpose |
|---|---|
| [JAX](https://jax.readthedocs.io/) | GPU/CPU array operations and automatic differentiation |
| [NIFTy.re](https://ift.pages.mpcdf.de/nifty/) | Information field theory and KL minimization |
| [Maria](https://github.com/thomaswmorris/maria) | Telescope simulation, TOD loading, and beam models |
| [Minkasi](https://github.com/sievers/minkasi) | Auxiliary TOD processing utilities |
| SciPy, NumPy, Matplotlib, PyYAML | Numerics, visualization, configuration |

## References

- maria: https://arxiv.org/abs/2402.10731
- NIFTy.re: https://arxiv.org/abs/2402.16683
- maria+NIFTy.re conference proceeding: https://arxiv.org/abs/2509.01600
