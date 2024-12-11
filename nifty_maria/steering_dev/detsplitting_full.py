from nifty_maria.FitHandler import FitHandler

fit = FitHandler(config='mustang', fit_atmos=True, fit_map=True, plotsdir='mustang_fullrun')
# fit = FitHandler(config='atlast_debug', fit_atmos=True, fit_map=True)
# fit = FitHandler(config='atlast', fit_atmos=True, fit_map=True)

fit.simulate()

fit.reco_maria()

fit.sample_jax_tods(use_truth_slope=False)

n_splits = list(range(8)) + [-1]
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
    samples, state = fit.perform_fit(n_it=30, printevery=5)
    
    # Show results:
    fit.printfitresults(samples)
    fit.plotfitresults(samples)
    fit.plotpowerspectrum(samples)
    
# Show reco comparison before expanding to full det:
fit.plotrecos(samples)

fit.make_atmosphere_det_gif(samples, figname='atmosphere_comp.gif', tmax=2500)