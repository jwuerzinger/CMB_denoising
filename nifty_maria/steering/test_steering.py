#!/usr/bin/env python
"""
Script for steering nifty_maria fits. Takes atlast/mustang config as input and automatically sets necessary maria and nifty configuration downstream. Supports fitting atmosphere and map separately for fit parameter optimisation.
"""

import click
from datetime import date
import os

def parse_comma_separated(ctx, param, value):
    if value:
        return [x.strip() for x in value.split(',')]
    return []

@click.command(help=__doc__)
@click.option('--config', default='test', help='Config for fit. Supported values are: "atlast", "mustang" or the path to a custom steering yaml. Defaults to atlast.')
@click.option('--fit_atmos', default=True, type=bool, help='Boolean for fitting atmosphere. Defaults to True.')
@click.option('--fit_map', default=True, type=bool, help='Boolean for fitting map. Defaults to True.')
@click.option('--nit_glob', default=1, type=int, help='Number of global iterations. Defaults to 30.')
@click.option('--nit_sl', default=1, type=int, help='Maximum number of linear sampling iterations per global iteration. Defaults to 2000.')
@click.option('--nit_sn', default=1, type=int, help='Maximum number of nonlinear sampling iterations per global iteration. Defaults to 20.')
@click.option('--nit_m', default=1, type=int, help='Maximum number of minimisation iterations per global iteration. Defaults to 200.')
@click.option('--printevery', default=1, type=int, help='Number of global iterations between plotting & printing results. Defaults to 5.')
@click.option('--cudadevice', default='3', type=str, help='CUDA device to run on. Defaults to "3".')
def main(config, fit_atmos, fit_map, nit_glob, nit_sl, nit_sn, nit_m, printevery, cudadevice):
    if config not in ['mustang', 'atlast', 'test']: raise ValueError("Unsupported config provided! Please choose between mustang/atlast.")

    os.environ['CUDA_VISIBLE_DEVICES'] = cudadevice

    # Set up results directory name
    atmosstr = "_atmos" if fit_atmos else ""
    mapstr = "_map" if fit_map else ""
    plotsdir = f"runs/{config}{atmosstr}{mapstr}_{date.today()}"
    print(f"Saving results in {plotsdir}")
    
    # Initialise fit config
    from nifty_maria.FitHandler import FitHandler
    fit = FitHandler(config=config, fit_atmos=fit_atmos, fit_map=fit_map, plotsdir=plotsdir, nit_sl=nit_sl, nit_sn=nit_sn, nit_m=nit_m)

    # Simulate TODs with maria
    fit.simulate()

    # Run maria reconstruction
    fit.reco_maria()

    # Run Jax sampling & configure inputs for GP
    fit.sample_jax_tods(use_truth_slope=False)

    if fit_atmos and fit_map: n_splits = [0]
    elif fit_atmos and not fit_map: n_splits = [0]
    else: n_splits = [-1]

    fit.init_gps(n_split=n_splits[0])
    firstiter = True
    for i in n_splits:
        print(f"Fit iteration: i = {i}")
        
        # Initialise, use samples after 0th iter:
        if firstiter: firstiter = False
        else: fit.init_gps(n_split=i, samples=samples)
        fit.plotsdir = f"{plotsdir}/nsub_{fit.n_sub}"
        os.makedirs(fit.plotsdir, exist_ok=True)
        fit.plot_subdets()
        
        # Show prior sample (only in 0th iter)
        if i == 0: prior_s = fit.draw_prior_sample()

        # Perform fit
        if i == -1: samples, state = fit.perform_fit(nit_glob=nit_glob, printevery=printevery)
        else: samples, state = fit.perform_fit(nit_glob=nit_glob, printevery=printevery)
        
        # Show results:
        fit.plot_results(samples)

        # if config == 'mustang': fit.make_atmosphere_det_gif(samples, figname=f"mustang_{fit.n_sub}_atmosphere_comp.gif", tmax=1000)
        # else: fit.make_atmosphere_det_gif(samples, figname=f"atlast_{fit.n_sub}_atmosphere_comp.gif", tmax=1000)
    
if __name__ == "__main__":
    main()