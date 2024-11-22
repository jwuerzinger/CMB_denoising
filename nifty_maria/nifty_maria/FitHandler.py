'''Module to collect fit config for nifty-maria fits.'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Set JAX to preallocate 90% of the GPU memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

# Disable default memory preallocation strategy for more control
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Use platform-specific memory allocation for CUDA
# # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import matplotlib.pyplot as plt
import maria

import jax.numpy as jnp
import jax
import jax.random as random

import nifty_maria.mapsampling_jax
from nifty_maria.mapsampling_jax import sample_maps
from nifty_maria.modified_CFM import CFM

import nifty8.re as jft

class FitHandler:
    def __init__(self, fit_map=True, fit_atmos=True, config='atlast_debug', noiselevel=1.0):
        
        print("Initialising...")
        self.fit_map = fit_map
        self.fit_atmos = fit_atmos
        self.config = config
        self.noiselevel = noiselevel
        
        if self.config == 'mustang':
            raise ValueError("Mustang config not implemented yet.")
        elif self.config == 'atlast':
            raise ValueError("AtLAST config not implemented yet.")
        elif self.config == 'atlast_debug':
            map_filename = maria.io.fetch("maps/cluster.fits")
        
        self.input_map = maria.map.read_fits(
            nu=150,
            filename=map_filename,  # filename
            # resolution=8.714e-05,  # pixel size in degrees
            width=1.,
            index=0,  # index for fits file
            # center=(150, 10),  # position in the sky
            center=(300, -10),  # position in the sky
            units="Jy/pixel",  # Units of the input map
        )

        # input_map.data *= 1e3
        self.input_map.data *= 1e5
        self.input_map.to(units="K_RJ").plot()
        
        self.plan = maria.get_plan(
            scan_pattern="daisy",
            scan_options={"radius": 0.25, "speed": 0.5}, # in degrees
            duration=60, # in seconds
            # duration=3, # in seconds
            # sample_rate=225, # in Hz
            sample_rate=20, # in Hz
            # sample_rate=100, # in Hz
            # sample_rate=50,
            start_time = "2022-08-10T06:00:00",
            scan_center=(300.0, -10.0),
            frame="ra_dec"
        )
        self.plan.plot()
        
        # instrument = maria.get_instrument('MUSTANG-2')
        self.instrument = nifty_maria.mapsampling_jax.get_atlast()
        self.instrument.plot()
        
        # jax init:
        seed = 42
        self.key = random.PRNGKey(seed)
        
        return

    def simulate(self):
        self.sim_truthmap = maria.Simulation(
            self.instrument, 
            plan=self.plan,
            site="llano_de_chajnantor", 
            map=self.input_map,
            # noise=False,
            atmosphere="2d",
            # cmb="generate",
        )

        self.tod_truthmap = self.sim_truthmap.run()
        
        dx, dy = self.sim_truthmap.coords.offsets(frame=self.sim_truthmap.map.frame, center=self.sim_truthmap.map.center)
        self.dx = dx.compute()
        self.dy = dy.compute()
        
        return 
    
    def reco_maria(self):
        from maria.mappers import BinMapper

        mapper_truthmap = BinMapper(
            center=(300.0, -10.0),
            frame="ra_dec",
            width=1.,
            height=1.,
            resolution=np.rad2deg(self.instrument.dets.fwhm[0]) * 3600,
            map_postprocessing={"gaussian_filter": {"sigma": 0} }
        )
        mapper_truthmap.add_tods(self.tod_truthmap)
        output_truthmap = mapper_truthmap.run()

        mapdata_truth = np.float64(self.sim_truthmap.map.data)
        self.mapdata_truth = np.nan_to_num(mapdata_truth, nan=np.nanmean(mapdata_truth)) # replace nan value by img mean

        print("mapdata_truth shape:", self.mapdata_truth.shape)
        print("mapdata_truth mean:", self.mapdata_truth.mean())

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        im0 = axes[0].imshow(output_truthmap.data[0].T)
        fig.colorbar(im0)
        axes[0].title.set_text("Noisy image (Mapper output)")

        im1 = axes[1].imshow(self.mapdata_truth[0,0])
        fig.colorbar(im1)
        axes[1].title.set_text("True Image")

        plt.show()
        
        return
    
    def sample_jax_tods(self, use_truth_slope=False):
        
        # Sample map with jax function and plot comparison
        self.jax_tods_map = sample_maps(self.mapdata_truth, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)

        fig, axes = plt.subplots(3, 1, figsize=(16, 8))

        for i in [0, 10, 100, 200]:
            im0 = axes[0].plot(self.jax_tods_map[i], label=i)

            tods_map = np.float64(self.tod_truthmap.get_field('map').compute())
            im1 = axes[1].plot(tods_map[i], label=i)

            im2 = axes[2].plot(self.jax_tods_map[i] - tods_map[i], label=i)
            
        axes[0].title.set_text(f'JAX map, TOD0-{i}')
        axes[0].legend()
        axes[1].title.set_text(f'True map, TOD0-{i}')
        axes[1].legend()
        axes[2].title.set_text(f'jax map - true map, TOD0-{i}')
        axes[2].legend()

        plt.show()
        
        # Next, create tod data for training

        # Add n TODs for atmos:
        n = self.instrument.n_dets

        self.jax_tods_atmos = self.tod_truthmap.get_field('atmosphere')
        # noised_jax_tod = np.float64(jax_tods_map) + np.float64(jax_tods_atmos) + np.float64(tod_truthmap.components['noise']*noiselevel)

        # Map + atmos
        self.noised_jax_tod = np.float64(self.jax_tods_map) + np.float64(self.tod_truthmap.get_field('noise')*self.noiselevel)
        if self.fit_atmos:
            self.noised_jax_tod[:n] += np.float64(self.jax_tods_atmos[:n]) 
        # denoised_jax_tod = noised_jax_tod - np.float64(tod_truthmap.get_field('noise')*noiselevel)

        # atmos-only:
        # noised_jax_tod = np.float64(jax_tods_atmos[:n]) + np.float64(tod_truthmap.get_field('noise')*noiselevel)[:n]

        # re-subtract noise:
        self.denoised_jax_tod = self.noised_jax_tod - np.float64(self.tod_truthmap.get_field('noise')*self.noiselevel)[:n]

        print("Noise stddev:", np.std(self.tod_truthmap.get_field('noise').compute()))

        fig, axes = plt.subplots(3, 1, figsize=(16, 8))

        for i in range(0, n, n//10 if n//10 != 0 else 1):
            im0 = axes[0].plot(self.jax_tods_map[i], label=i)
            im1 = axes[1].plot(self.jax_tods_atmos[i], label=i)
            im2 = axes[2].plot(self.noised_jax_tod[i], label=i)
            
        axes[0].title.set_text(f'JAX MAP TOD0-{i}')
        axes[0].legend()
        axes[1].title.set_text(f'Atmosphere TOD0-{i}')
        axes[1].legend()
        axes[2].title.set_text(f'Total TOD0-{i}, noise={self.noiselevel}')
        axes[2].legend()

        plt.show()
        
        slopes_tod_truth = (self.jax_tods_atmos) / (self.jax_tods_atmos[0])
        slopes_tod_truth = np.float64(slopes_tod_truth.mean(axis=1))
        slopes_tod = self.noised_jax_tod / self.noised_jax_tod[0]
        slopes_tod = np.float64(slopes_tod.mean(axis=1))
        
        offset_tod_truth = np.float64(self.jax_tods_atmos.mean(axis=1))
        offset_tod = np.float64(self.noised_jax_tod.mean(axis=1))
        
        if use_truth_slope:
            self.slopes_tod = slopes_tod_truth
            self.offset_tod = offset_tod_truth
        else:
            self.slopes_tod = slopes_tod
            self.offset_tod = offset_tod
        
        # Get simplified atmosphere tods for validation
        self.atmos_tod_simplified = (self.jax_tods_atmos - self.offset_tod[:, None])/self.slopes_tod[:, None]
        
        return 
        
    def init_gps(self):
        
        # padding_atmos = 2000
        # padding_atmos = 5000
        self.padding_atmos = 10000
        self.dims_atmos = ( (self.jax_tods_atmos.shape[1] + self.padding_atmos), )
        # dims_atmos = ( (jax_tods_atmos.shape[1] - 200 + padding_atmos), )

        # correlated field zero mode GP offset and stddev
        # cf_zm_tod = dict(offset_mean=jax_tods_atmos.mean().compute(), offset_std=(0.0002, 0.0001))
        # cf_zm_tod = dict(offset_mean=0.0, offset_std=(1e-5, 0.99e-5))
        # cf_zm_tod = dict(offset_mean=0.0, offset_std=(6e-6, 5e-6))
        self.cf_zm_tod = dict(offset_mean=0.0, offset_std=(5e-5, 4e-5))

        # correlated field fluctuations (mostly don't need tuning)
        # fluctuations: y-offset in power spectrum in fourier space (zero mode)
        # loglogavgslope: power-spectrum slope in log-log space in frequency domain (Fourier space) Jakob: -4 -- -2
        # flexibility=(1.5e0, 5e-1), # deviation from simple power-law
        # asperity=(5e-1, 5e-2), # small scale features in power-law
        self.cf_fl_tod = dict(
            # fluctuations=(0.0015, 0.0001),
            # loglogavgslope=(-2.45, 0.1), 
            fluctuations=(0.01, 0.003),
            loglogavgslope=(-2.2, 0.2), 
            flexibility=None,
            asperity=None,
        )

        # put together in correlated field model
        # Custom CFM:
        cfm_tod = CFM("combcf ")
        cfm_tod.set_amplitude_total_offset(**self.cf_zm_tod)
        cfm_tod.add_fluctuations(
            self.dims_atmos, distances=1.0 / self.dims_atmos[0], **self.cf_fl_tod, prefix="tod ", non_parametric_kind="power"
        )
        # gp_tod = cfm_tod.finalize(n)
        self.gp_tod = cfm_tod.finalize(1)
        # nsub = 2 # 1, 2, 3 already breaks
        # gp_tod = cfm_tod.finalize(nsub) 
        
        
        # padding_map = 400
        self.padding_map = 10
        # dims_map = (1024 + padding_map, 1024 + padding_map)
        self.dims_map = (1000 + self.padding_map, 1000 + self.padding_map)

        # Map model

        # correlated field zero mode GP offset and stddev
        # cf_zm_map = dict(offset_mean=mapdata_truth.mean(), offset_std=(1e-8, 1e-7))
        # # correlated field fluctuations (mostly don't need tuning)
        # cf_fl_map = dict(
        #     fluctuations=(5.6e-5, 1e-6), # fluctuations: y-offset in power spectrum in fourier space (zero mode)
        #     loglogavgslope=(-3.7, 0.1),
        #     flexibility=None,
        #     asperity=None,
        # )
        # Dummy map:
        self.cf_zm_map = dict(offset_mean=self.mapdata_truth.mean(), offset_std=(2e-6, 1e-7))
        # correlated field fluctuations (mostly don't need tuning)
        self.cf_fl_map = dict(
            fluctuations=(1e-4, 1e-5), # fluctuations: y-offset in power spectrum in fourier space (zero mode)
            loglogavgslope=(-3.0, 0.1),
            flexibility=None,
            asperity=None,
        )

        # put together in correlated field model
        cfm_map = jft.CorrelatedFieldMaker("cfmap")
        cfm_map.set_amplitude_total_offset(**self.cf_zm_map)
        cfm_map.add_fluctuations(
            self.dims_map, distances=1.0 / self.dims_map[0], **self.cf_fl_map, prefix="ax1", non_parametric_kind="power"
        )
        self.gp_map = cfm_map.finalize()
        
        from nifty_maria.SignalModels import Signal_TOD_combined
        
        if self.noiselevel == 0.0: noise_cov_inv_tod = lambda x: 1e-8**-2 * x
        elif self.noiselevel == 0.1: noise_cov_inv_tod = lambda x: 1e-4**-2 * x
        elif self.noiselevel == 0.5: noise_cov_inv_tod = lambda x: 1e-4**-2 * x
        # elif self.noiselevel == 1.0: noise_cov_inv_tod = lambda x: 2.5e-4**-2 * x
        elif self.noiselevel == 1.0: noise_cov_inv_tod = lambda x: 1.9e-4**-2 * x
        
        from maria.units import Angle

        test = Angle(self.instrument.dets.offsets)
        pos = getattr(test, test.units).T

        print(pos.shape)

        posmask_ud = jnp.array((pos[1] >= (pos[1].max() + pos[1].min())/2))
        posmask_lr = jnp.array((pos[0] >= (pos[0].max() + pos[0].min())/2))

        posmask_up = posmask_ud
        posmask_down = ~posmask_ud
        posmask_left = posmask_lr
        posmask_right = ~posmask_lr
        
        self.signal_response_tod = Signal_TOD_combined(self.gp_tod, self.offset_tod, self.slopes_tod, self.gp_map, self.dims_map, self.padding_map, self.dims_atmos, self.padding_atmos, posmask_up, posmask_down, self.sim_truthmap, self.dx, self.dy)
        self.lh = jft.Gaussian( self.noised_jax_tod, noise_cov_inv_tod).amend(self.signal_response_tod)
        
        print(self.lh)
        
        return 
    
    def draw_prior_sample(self):
        self.key, sub = jax.random.split(self.key)
        xi = jft.random_like(sub, self.signal_response_tod.domain)
        res = self.signal_response_tod(xi)
        n = self.instrument.n_dets
        print(res.shape)

        fig, axes = plt.subplots(1, 1, figsize=(16, 4))

        for i in range(0, n, n//10 if n//10 != 0 else 1):
            axes.plot( np.arange(0, res.shape[1]), res[i], label=i)

        axes.title.set_text(f'all')
        axes.legend()
        
        return 
    
    def perform_fit(self, n_it=1, fit_type = 'map', printevery = 2):
        
        if self.noiselevel == 0.0: delta = 1e-4
        elif self.noiselevel == 0.1: delta = 1e-10
        elif self.noiselevel == 0.5: delta = 1e-10
        elif self.noiselevel == 1.0: delta = 1e-4

        if fit_type == 'map':
            n_samples = 0 # no samples -> maximum aposteriory posterior
            sample_mode = 'nonlinear_resample'
        else:
            n_samples = 4
            sample_mode = lambda x: "nonlinear_resample" if x >= 1 else "linear_resample"

        self.key, k_i, k_o = random.split(self.key, 3)

        def callback(samples, opt_state):
            iter = opt_state[0]
            # printevery = 1 # 3
            n = self.instrument.n_dets
            if iter % printevery != 0: return

            fig_tods, axes_tods = plt.subplots(2, 1, figsize=(16, 6))
            mean, std = jft.mean_and_std(tuple(self.signal_response_tod(s) for s in samples))

            for i in range(0, n, n//10 if n//10 != 0 else 1):
                axes_tods[0].plot(np.arange(0, mean.shape[1]), mean[i], label=f"tod{i}")
                axes_tods[0].plot(self.denoised_jax_tod[i], label=f"truth{i}")
                axes_tods[1].plot(np.arange(0, mean.shape[1]), mean[i] - self.denoised_jax_tod[i], label=f"tod{i}")

            axes_tods[0].title.set_text('total mean pred. & truth (no noise)')
            axes_tods[0].legend()
            axes_tods[1].title.set_text('total mean pred. - truth (no noise)')
            axes_tods[1].legend()

            if self.fit_atmos:
                fig_tods, axes_tods = plt.subplots(2, 1, figsize=(16, 6))

                preds = []
                for x in samples:
                    x_tod = {k: x[k] for k in x if 'comb' in k}
                    res_tods = self.gp_tod(x_tod)

                    # From TOD-only fit:
                    # preds += [res_tods[:, padding_atmos//2:-padding_atmos//2] * slopes_truth[:, None] + offset_tod_truth[:, None], ]
                    preds += [res_tods[:, self.padding_atmos//2:-self.padding_atmos//2], ]

                mean_atmos, std = jft.mean_and_std(tuple(preds))

                for i in range(0, n, n//10 if n//10 != 0 else 1):
                    axes_tods[0].plot(np.arange(0, mean_atmos.shape[1]), mean_atmos[i], label=f"tod{i}")
                    # axes_tods[0].plot(denoised_jax_tod[i], label=f"truth{i}")
                    axes_tods[0].plot(self.atmos_tod_simplified[i], label=f"truth{i}")
                    axes_tods[1].plot(np.arange(0, mean_atmos.shape[1]), mean_atmos[i] - self.atmos_tod_simplified[i], label=f"tod{i}")
                    # axes_tods[1].plot(np.arange(0, mean_atmos.shape[1]), mean[i] - mean_atmos[i], label=f"tod{i}")

                axes_tods[0].title.set_text('mean atmos pred. & simplified truth (no noise)')
                axes_tods[0].legend()
                axes_tods[1].title.set_text('mean atmos pred. - simplified truth (no noise)')
                axes_tods[1].legend()

            fig_map, axes_map = plt.subplots(1, 3, figsize=(16, 6))

            mean_map, _ = jft.mean_and_std(tuple(self.gp_map(s)[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in samples))
            # mean_map, _ = jft.mean_and_std(tuple(gp_map(s) for s in samples))

            im0 = axes_map[0].imshow(mean_map)
            axes_map[0].title.set_text('mean map pred.')
            fig_map.colorbar(im0)

            im1 = axes_map[1].imshow(mean_map - self.mapdata_truth[0, 0])
            axes_map[1].title.set_text('mean map - truth')
            fig_map.colorbar(im1)

            im2 = axes_map[2].imshow(self.mapdata_truth[0, 0])
            axes_map[2].title.set_text('truth')
            fig_map.colorbar(im2)

            plt.show()
            
            return

        samples, state = jft.optimize_kl(
            self.lh, # likelihood
            0.1*jft.Vector(self.lh.init(k_i)), # initial position in model space (initialisation)
            n_total_iterations=n_it, # no of optimisation steps (global)
            n_samples=n_samples, # draw samples
            key=k_o, # random jax init
            draw_linear_kwargs=dict( # sampling parameters
                cg_name="SL",
                cg_kwargs=dict(absdelta=delta * jft.size(self.lh.domain) / 10.0, maxiter=60),
            ),
            nonlinearly_update_kwargs=dict( # map from multivariate gaussian to more compl. distribution (coordinate transformations)
                minimize_kwargs=dict(
                    name="SN",
                    xtol=delta,
                    cg_kwargs=dict(name=None),
                    maxiter=5,
                )
            ),
            kl_kwargs=dict( # shift transformed multivar gauss to best match true posterior
                minimize_kwargs=dict(
                    # name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=60 # map
                    name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=20 # map
                )
            ),
            sample_mode=sample_mode, # how steps are combined (samples + nonlin + KL),
            callback=callback if fit_type != 'map' else None
        )
        
        return samples, state
    
    def printfitresults(self, samples):
        
        print("Fit Results (res, init, std)")

        if self.fit_atmos:
            print("\nTODs:")
            print(f"\tfluctuations: {jft.LogNormalPrior(*self.cf_fl_tod['fluctuations'])(samples.pos['combcf tod fluctuations'])}, {self.cf_fl_tod['fluctuations'][0]}, {self.cf_fl_tod['fluctuations'][1]}")
            print(f"\tloglogvarslope: {jft.NormalPrior(*self.cf_fl_tod['loglogavgslope'])(samples.pos['combcf tod loglogavgslope'])}, {self.cf_fl_tod['loglogavgslope'][0]}, {self.cf_fl_tod['loglogavgslope'][1]}")
            print(f"\tzeromode std (LogNormal): {jft.LogNormalPrior(*self.cf_zm_tod['offset_std'])(samples.pos['combcf zeromode'])}, {self.cf_zm_tod['offset_std'][0]}, {self.cf_zm_tod['offset_std'][1]}")
    
        if self.fit_map:
            print("map:")
            print(f"\tfluctuations: {jft.LogNormalPrior(*self.cf_fl_map['fluctuations'])(samples.pos['cfmapax1fluctuations'])}, {self.cf_fl_map['fluctuations'][0]}, {self.cf_fl_map['fluctuations'][1]}")
            print(f"\tloglogvarslope: {jft.NormalPrior(*self.cf_fl_map['loglogavgslope'])(samples.pos['cfmapax1loglogavgslope'])}, {self.cf_fl_map['loglogavgslope'][0]}, {self.cf_fl_map['loglogavgslope'][1]}")
            print(f"\tzeromode std (LogNormal): {jft.LogNormalPrior(*self.cf_zm_map['offset_std'])(samples.pos['cfmapzeromode'])}, {self.cf_zm_map['offset_std'][0]}, {self.cf_zm_map['offset_std'][1]}")
         
        return
    
    def plotfitresults(self, samples):
        
        res = self.signal_response_tod(samples.pos)
        n = self.instrument.n_dets

        fig, axes = plt.subplots(3, 1, figsize=(16, 8))

        for i in range(0, n, n//10 if n//10 != 0 else 1):
            im0 = axes[0].plot(np.arange(0, res.shape[1]), res[i], label=f"tod{i}")
            im1 = axes[1].plot(np.arange(0, res.shape[1]), res[i] - self.noised_jax_tod[i], label=f"tod{i}")
            im2 = axes[2].plot(self.noised_jax_tod[i], label=f"truth{i}")

        axes[0].title.set_text('MAP - best fit image')
        axes[0].legend()
        axes[1].title.set_text('MAP - map truth')
        axes[1].legend()
        axes[2].title.set_text('truth')
        axes[2].legend()

        plt.show()
         
        # plot maximum of posterior (mode)
        sig_map = self.gp_map(samples.pos)[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] # when splitting up in different field models
        # sig_map = gp_map(samples.pos)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        im0 = axes[0].imshow(sig_map)
        axes[0].title.set_text('MAP - best fit image')
        fig.colorbar(im0)

        im1 = axes[1].imshow( sig_map - self.mapdata_truth[0, 0] )
        axes[1].title.set_text('MAP - map truth')
        # im1 = axes[1].imshow( (sig_map - mapdata_truth) )
        # axes[1].title.set_text('diff prediction - map truth')
        fig.colorbar(im1)

        plt.show()
        
        return 
    
    def plotpowerspectrum(self, samples):
        
        import scipy as sp

        # mean, std = jft.mean_and_std(tuple(signal_response_tod(s) for s in samples))
        from itertools import cycle

        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        gp_map_nopad = jax.numpy.broadcast_to(self.gp_map(samples.pos), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)

        if not self.fit_atmos:
            components = [res_map, self.tod_truthmap.get_field('map')]
            labels = ['map', 'true map']
        else:
            
            x_tod = {k: samples.pos[k] for k in samples.pos if 'comb' in k}
            res_tods = self.gp_tod(x_tod)[:, self.padding_atmos//2:-self.padding_atmos//2]

            components = [self.signal_response_tod(samples.pos), res_map, self.tod_truthmap.get_field('map'), res_tods, self.tod_truthmap.get_field('atmosphere')]
            labels = ['pred. total', 'pred. map', 'true map', 'pred. atmos', 'true atmos']
            linestyles = ['-', '-', '--', '-', '--']

        fig_tods, axes_tods = plt.subplots(1, 1, figsize=(16, 6))
        for i in range(len(components)):
            
            f, ps = sp.signal.periodogram(components[i], fs=self.tod_truthmap.fs, window="tukey")

            f_bins = np.geomspace(f[1], f[-1], 256)
            f_mids = np.sqrt(f_bins[1:] * f_bins[:-1])

            binned_ps = sp.stats.binned_statistic(
                f, ps.mean(axis=0), bins=f_bins, statistic="mean"
            )[0]

            use = binned_ps > 0

            if linestyles[i] == '-': color = next(colors)
            axes_tods.plot(
                f_mids[use],
                binned_ps[use],
                lw=1.4,
                color=color,
                label=labels[i],
                linestyle=linestyles[i]
            )
            
        axes_tods.set_xlabel('Frequency [Hz]')
        axes_tods.set_ylabel(f"[{self.tod_truthmap.units}$^2$/Hz]")
        axes_tods.set_xlim(f_mids.min(), f_mids.max())
        axes_tods.loglog()
        axes_tods.legend()
        
        return