# CMB_denoising

Draft repo for the ODSL CMB denoising project. 

Follow the [maria installation instructions](https://thomaswmorris.com/maria/installation.html) to get startet.

The [maria](maria/) python package is added as a submodule.

The [tutorials](tutorials/) folder contains some simple tutorial notebooks for using maria.
They may be outdated. Also check: maria/docs/source/tutorials. 

Check the [transformer_maria](transformer_maria/) folder for a simple first (unsuccessful) transformer-based attempt at reconstructing maria data.

[nifty_maria](nifty_maria/) contains code for reconstructing maria data with the jax implementation of [nifty](nifty) (also added as a submodule). There are step-by-step python notebooks which build up to a simultaneous fit for disentangling astronomical signals from atmosphere contributions:
- [nifty_maria/python/](nifty_maria/python/): jax-based rewrite of maria's map sampling.
- [nifty_maria/map/](nifty_maria/map/): Reconstruction of astronomical signals without atmosphere or CMB contributions.
- [nifty_maria/atmosphere_mapsampling/](nifty_maria/atmosphere_mapsampling/): Reconstruction of (static) atmosphere contributions using nifty's 2D correlated field model (i.e. atmosphere "images"). Moderately successful and computationally very heavy!
- [nifty_maria/atmosphere_tods/](nifty_maria/atmosphere_tods/): Reconstruction of 1D atmosphere time-series directly without modelling detector response. More lightweight/successful, **but**: brings some loss of generality/expressivity since all atmosphere TODs are assumed to be identical, barring pixel fluctuations.
- [nifty_maria/simultaneous_fit/](nifty_maria/simultaneous_fit/): Simultaneous reconstruction of map (image) and atmosphere (time-series). Most successful approach thus far. Better reco than maria baseline.