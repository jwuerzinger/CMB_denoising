#!/usr/bin/env python

import click
from datetime import date
from nifty_maria.FitHandler import FitHandler

@click.command()
@click.option('--config', default='atlast', help='Config for fit. Supported values are: atlas/mustang.')
@click.option('--fit_atmos', default=True, type=bool, help='Boolean for fitting atmosphere.')
@click.option('--fit_map', default=True, type=bool, help='Boolean for fitting map.')

def main(config, fit_atmos, fit_map):
    if config not in ['mustang', 'atlast']: raise ValueError("Unsupported config provided! Please choose between mustang/atlast.")

    # Set up results directory name
    atmosstr = "_atmos" if fit_atmos else ""
    mapstr = "_map" if fit_map else ""
    plotsdir = f"runs/{config}{atmosstr}{mapstr}_{date.today()}"
    print(f"Saving results in {plotsdir}")
    
    # Initialise fit config
    fit = FitHandler(config=config, fit_atmos=fit_atmos, fit_map=fit_map, plotsdir=plotsdir)

    # Simulate TODs with maria
    fit.simulate()

    # Run maria reconstruction
    fit.reco_maria()

    # Run Jax sampling & configure inputs for GP
    fit.sample_jax_tods(use_truth_slope=False)

    if config == 'mustang': n_splits = list(range(8)) + [-1]
    else: n_splits = list(range(7))
    print(f"Will run with n_splits: {n_splits}")

    for i in n_splits:
        print(f"Fit iteration: i = {i}")
        
        # Initialise, use samples after 0th iter:
        if i == 0: fit.init_gps(n_split=i)
        else: fit.init_gps(n_split=i, samples=samples)
        fit.plot_subdets()
        
        # Show prior sample (only in 0th iter)
        if i == 0: prior_s = fit.draw_prior_sample()

        # Perform fit
        if i == -1: samples, state = fit.perform_fit(n_it=30, printevery=1)
        else: samples, state = fit.perform_fit(n_it=30, printevery=5)
        
        # Show results:
        fit.printfitresults(samples)
        fit.plotfitresults(samples)
        fit.plotpowerspectrum(samples)
        
        # Show reco comparison before expanding to full det:
        fit.plotrecos(samples)

        if config == 'mustang': fit.make_atmosphere_det_gif(samples, figname=f"mustang_{fit.n_sub}_atmosphere_comp.gif", tmax=1000)
        else: fit.make_atmosphere_det_gif(samples, figname=f"atlast_{fit.n_sub}_atmosphere_comp.gif", tmax=100)
    
if __name__ == "__main__":
    main()