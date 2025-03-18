from nifty_maria.FitHandler import FitHandler

# fit = FitHandler(config='mustang', fit_atmos=True, fit_map=True, plotsdir='mustang_fullrate_smoothed_atmos_23-02-25')
fit = FitHandler(config='atlast', fit_atmos=True, fit_map=True, plotsdir='atlast_fullrate_downsampled_atmos_2_07-03-2025')

fit.simulate()

fit.reco_maria()

fit.sample_jax_tods(use_truth_slope=False)

# n_splits = list(range(8)) + [-1]
# n_splits = list(range(14)) + [-1]
n_splits = list(range(7))
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

    # fit.make_atmosphere_det_gif(samples, figname=f"mustang_{fit.n_sub}_atmosphere_comp.gif", tmax=1000)
    fit.make_atmosphere_det_gif(samples, figname=f"atlast_{fit.n_sub}_atmosphere_comp.gif", tmax=100)