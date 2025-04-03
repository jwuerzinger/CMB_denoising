# CMB_denoising

Repo for reconstruction and denoising of (simulated) (sub-mm) observatios of ground-based single-dish telescopes. This repo uses [maria](https://thomaswmorris.com/maria/index.html) for telescope simulations and the [JAX](https://docs.jax.dev/en/latest/index.html)-based  re-envisioned [NIFTy.re](https://ift.pages.mpcdf.de/nifty/).

If you're using this repo for your own research, please cite **ADD LINK TO PAPER HERE**.

Don't hesitate to reach out to me directly or open an issue in case you spot any issues!

## Installation

I will conserve my conda environment and supply automated installation instructions here, once the server lets me..
Until then, please follow these manual instructions:

The `maria` and `NIFTy` repos are both added as submodules to this repo. After cloning this repo with submodules with
```bash
git clone --recurse-submodules git@github.com:jwuerzinger/CMB_denoising.git
```
if you're using ssh authentification or 
```bash
git clone --recurse-submodules https://github.com/jwuerzinger/CMB_denoising.git
```
if you want to authenticate with https.

Install both submodules with
```bash
pip install -e maria/
pip install -e nifty/[re]
```

If necessary, there are dedicated installation instructions for `maria` [here](https://thomaswmorris.com/maria/installation.html).

Next, install this package with:

```bash
pip install -e nifty_maria/
```

## Running

After installing, you can use the steering script in [nifty_maria/steering/steering_full.py](nifty_maria/steering/steering_full.py).
To do this, simply call:
```bash
cd nifty_maria/steering
python steering_full.py --config CONFIG --fit_atmos BOOL --fit_map BOOL
```

Where `CONFIG` should be either `mustang`, `atlast` or the path to a custom steering yaml file. The default yaml files for `mustang` and `atlast` are automatically read and can be found [here](nifty_maria/nifty_maria/configs/).
The value for `BOOL` is either `True` or `False`, depending on whether you want to fit the atmosphere/map. By default, a fit to simulated `mustang` data for both the map and atmosphere is performed.

This script starts with a reconstruction of two gaussian process-based correlated field models: One two-dimensional model for the map and one one-dimensional model for the atmosphere TOD directly. The array is then automatically split horizontally/vertically, thus doubling the numbers of modelled atmosphere TODs `n_sub` with every iterations, until `n_sub=64` for AtLAST and `n_sub=128` for MUSTANG-2. In the case of MUSTANG-2, a final step in the fit is performed where every detector's atmosphere TOD response is modelled individually.

## Structure

The [maria](maria/) and [nifty](nifty/) python packages are added as submodules.

The [tutorials](tutorials/) folder contains some simple tutorial notebooks for using maria.
They may be outdated. Also check: maria/docs/source/tutorials for up-to-date tutorials specific to `maria`. 

[nifty_maria](nifty_maria/) contains code for reconstructing maria data with the jax implementation of [nifty](nifty/).
- [nifty_maria/nifty_maria/](nifty_maria/nifty_maria/): jax-based rewrite of maria's map sampling as well as general steering code.
- [nifty_maria/steering/](nifty_maria/steering/): Test notebooks for testing improved steering notebooks.

## References:

- maria: https://arxiv.org/abs/2402.10731
- NIFTy.re: https://arxiv.org/abs/2402.16683