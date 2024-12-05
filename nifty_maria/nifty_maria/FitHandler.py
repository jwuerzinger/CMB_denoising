'''
Module to collect fit config for nifty-maria fits.
'''
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
from nifty8.re.optimize_kl import OptimizeVIState

class FitHandler:
    '''
    A steering class to handle nifty fits using data generated with maria.

    Attributes:
        fit_map (bool): Perform fit of map if True.
        fit_atmos (bool): Perform fit of atmosphere if True.
        config (str): The detector configuraion to run on. Options are: "mustang", "atlast" and "atlast_debug".
        noiselevel (float): The fraction of noise to add.
        input_map (Map): The input map of the sky to use in simulation.
        plan (Plan): The scanning pattern to simulate.
        instrument (Instrument): The detector used during simulation.
        params (dict): The dictionary containing fit parameters (initial values, stdev) for both atmosphere and map GPs used in the nifty fit.
        key (KeyArray): Pseudo random number key for jax.
        
    Dynamic Attributes (Added by Methods):    
        sim_truthmap (Simulation): Simulation object containing instrument, plan, site, input map and parameters for noise, atmosphere and cmb simulation. Added by FitHandler.simulate().
        tod_truthmap (TOD): TOD object containing simulated time-stream data. Added by FitHandler.simulate().
        dx (array): Array with detector offsets in x-direction. Added by FitHandler.simulate().
        dy (array): Array with detector offsets in y-direction. Added by FitHandler.simulate().
        output_truthmap (Map): Noised Map object obtained by reconstruction without postprocessing. Added by FitHandler.reco_maria().
        mapdata_truth (array): Array with true simulated map. Added by FitHandler.reco_maria().
        output_map (Map): Map opbject obtained by reconstruction with postprocessing. Added by FitHandler.reco_maria().
        jax_tods_map (array): Array with map TODs generated with jax. Added by FitHandler.sample_jax_tods().
        jax_tods_atmos (array): Array with atmosphere TODs generated with jax. Added by FitHandler.sample_jax_tods().
        noised_jax_tod (array): Array with total noised TODs generated with jax. Added by FitHandler.sample_jax_tods().
        denoised_jax_tod (array): Array with total denoised TODs generated with jax. Added by FitHandler.sample_jax_tods().
        slopes_tod (array): Array with slopes between atmosphere TODs. Taken from data if use_truth_slope=False, otherwise taken from the true atmosphere TODs. Added by FitHandler.sample_jax_tods(). 
        offset_tod (array): Array with offsets between atmosphere TODs. Taken from data if use_truth_slope=False, otherwise taken from the true atmosphere TODs. Added by FitHandler.sample_jax_tods(). 
        atmos_tod_simplified (array): Array with simplified atmosphere TODs, removing offsets and slopes. Added by FitHandler.sample_jax_tods(). 
        n_sub (int): Integer determining how many GPs are used to simulate sub-detector atmosphere responses. Simulates all sub-detectors if -1. Added by FitHandler.init_gps().
        initial_pos (jft.Vector): Vector with initial position to be used in fit. Added by FitHandler.init_gps().
        padding_atmos (int): Integer describing atmosphere padding. Added by FitHandler.init_gps().
        dims_atmos (tuple[int,]): Tuple with atmosphere dimensions. Added by FitHandler.init_gps().
        cf_zm_tod (dict): Dictionary with atmosphere zeromode parameters. Added by FitHandler.init_gps().
        cf_fl_tod (dict): Dictionary with atmosphere fluctuation parameters. Added by FitHandler.init_gps().
        gp_tod (nifty_maria.modified_CFM.CFM): Modified CorrelatedFieldMaker object describing GP for atmosphere TODs. Added by FitHandler.init_gps().
        padding_map (int): Integer describing map padding. Added by FitHandler.init_gps().
        dims_map (tuple[int,int]): Tuple with map dimensions. Added by FitHandler.init_gps().
        cf_zm_map (dict): Dictionary with map zeromode parameters. Added by FitHandler.init_gps().
        cf_fl_map (dict): Dictionary with map fluctuation parameters. Added by FitHandler.init_gps().
        gp_map (jft.CorrelatedFieldMaker): CorrelatedFieldMaker object describing GP for map. Added by FitHandler.init_gps().
        signal_response_tod (jft.Model): Signal model containing forward model. Added by FitHandler.init_gps().
        lh (jft.Gaussian): Gaussian Likelihood used in optimisation. Added by FitHandler.init_gps().
        
    Example:
        Setup for simple fit to mustang data:
        >>> fit = FitHandler(config='mustang')
        >>> fit.simulate()
        >>> fit.reco_maria()
        >>> fit.sample_jax_tods()
        >>> fit.init_gps()
        >>> samples, state = fit.perform_fit(fit_type = 'map')
        >>> fit.printfitresults(samples)
        >>> fit.plotfitresults(samples)
    '''
    def __init__(self, fit_map: bool = True, fit_atmos: bool = True, config: str = 'atlast_debug', noiselevel: int = 1.0) -> None:
        '''
        Initialises the FitHandler with base attributes.
        
        Args:
            fit_map (bool, optional): Perform fit of map if True. Defaults to True.
            fit_atmos (bool, optional): Perform fit of atmosphere if True. Defaults to True.
            config (str, optional): The detector configuraion to run on. Options are: 'mustang', 'atlast' and 'atlast_debug'. Defaults to 'atlast_debug'.
            noiselevel (float, optional): The fraction of noise to add. Defaults to 1.0.
            
        Raises:
            ValueError: If invalid configuration is used.
        '''
        
        print("Initialising...")
        self.fit_map = fit_map
        self.fit_atmos = fit_atmos
        self.config = config
        self.noiselevel = noiselevel
        
        if self.config == 'mustang':
            map_filename = maria.io.fetch("maps/cluster.fits")

            # load in the map from a fits file
            self.input_map = maria.map.read_fits(filename=map_filename, #filename
                                            resolution=8.714e-05, #pixel size in degrees
                                            index=0, #index for fits file
                                            center=(150, 10), # position in the sky
                                            units='Jy/pixel' # Units of the input map 
                                        )

            self.input_map.to(units="K_RJ").plot()
            
            #load the map into maria
            self.plan = maria.get_plan(scan_pattern="daisy", # scanning pattern
                                scan_options={"radius": 0.05, "speed": 0.01}, # in degrees
                                duration=600, # integration time in seconds
                                # duration=60,
                                #   duration=300, # integration time in seconds
                                sample_rate=50, # in Hz
                                scan_center=(150, 10), # position in the sky
                                frame="ra_dec")

            self.plan.plot()
            
            self.instrument = nifty_maria.mapsampling_jax.instrument
            self.instrument.plot()
            
            self.params = { 
                'tod_offset' : (1e-5, 0.99e-5),
                'tod_fluct' : (0.0015, 0.0001),
                'tod_loglog' : (-2.45, 0.1),
                'map_offset' : (1e-8, 1e-7),
                'map_fluct' : (5.6e-5, 1e-6),
                'map_loglog' : (-2.5, 0.1),
                'noise' : lambda x: 2.5e-4**-2 * x, # TODO: generalize!
            }
            
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
            
            # self.instrument = maria.get_instrument('MUSTANG-2')
            # self.instrument = nifty_maria.mapsampling_jax.get_atlast()
            self.instrument = nifty_maria.mapsampling_jax.get_dummy()
            self.instrument.plot()
            
            self.params = {
                'tod_offset' : (5e-5, 4e-5),
                'tod_fluct' : (0.01, 0.003),
                'tod_loglog' : (-2.2, 0.2),
                'map_offset' : (2e-6, 1e-7),
                'map_fluct' : (1e-4, 1e-5),
                'map_loglog' : (-3.0, 0.1),
                'noise' : lambda x: 1.9e-4**-2 * x,
            }
        
        else:
            raise ValueError("Unknown fit config!")
        
        # jax init:
        seed = 42
        self.key = random.PRNGKey(seed)

    def simulate(self) -> None:
        '''
        Performs maria simulation and decorates self with simulation parameters.
        
        Dynamic Attributes (Added by Methods):
            sim_truthmap (Simulation): Simulation object containing instrument, plan, site, input map and parameters for noise, atmosphere and cmb simulation. Added by FitHandler.simulate().
            tod_truthmap (TOD): TOD object containing simulated time-stream data. Added by FitHandler.simulate().
            dx (array): Array with detector offsets in x-direction. Added by FitHandler.simulate().
            dy (array): Array with detector offsets in y-direction. Added by FitHandler.simulate().
        '''
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
    
    def reco_maria(self) -> None:
        '''
        Performs maria reconstruction and decorates self with reconstructed maps.
        
        Dynamic Attributes (Added by Methods):
            output_truthmap (Map): Noised Map object obtained by reconstruction without postprocessing. Added by FitHandler.reco_maria().
            mapdata_truth (array): Array with true simulated map. Added by FitHandler.reco_maria().
            output_map (Map): Map opbject obtained by reconstruction with postprocessing. Added by FitHandler.reco_maria().
        '''
        from maria.mappers import BinMapper

        mapper_truthmap = BinMapper(
            # center=(300.0, -10.0),
            center=(150., 10.),
            frame="ra_dec",
            width=1.,
            height=1.,
            resolution=np.rad2deg(self.instrument.dets.fwhm[0]) * 3600,
            map_postprocessing={"gaussian_filter": {"sigma": 0} }
        )
        mapper_truthmap.add_tods(self.tod_truthmap)
        self.output_truthmap = mapper_truthmap.run()

        mapdata_truth = np.float64(self.sim_truthmap.map.data)
        self.mapdata_truth = np.nan_to_num(mapdata_truth, nan=np.nanmean(mapdata_truth)) # replace nan value by img mean

        print("mapdata_truth shape:", self.mapdata_truth.shape)
        print("mapdata_truth mean:", self.mapdata_truth.mean())

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        im0 = axes[0].imshow(self.output_truthmap.data[0].T)
        fig.colorbar(im0)
        axes[0].title.set_text("Noisy image (Mapper output)")

        im1 = axes[1].imshow(self.mapdata_truth[0,0])
        fig.colorbar(im1)
        axes[1].title.set_text("True Image")

        plt.show()
        
        # Run proper mapmaker
        mapper = BinMapper(center=(150, 10),
                   frame="ra_dec",
                   width=0.1,
                   height=0.1,
                   resolution=2e-4,
                   tod_preprocessing={
                        "window": {"name": "hamming"},
                        "remove_modes": {"modes_to_remove": [0]},
                        "despline": {"knot_spacing": 10},
                    },
                    map_postprocessing={
                        "gaussian_filter": {"sigma": 1},
                        "median_filter": {"size": 1},
                    },
                )
        
        mapper.add_tods(self.tod_truthmap)
        self.output_map = mapper.run()
        
        return
    
    def sample_jax_tods(self, use_truth_slope: bool = False) -> None:
        '''
        Sample TODs using jax map sampling, make plots comparing to TODs generated with maria and decorate self with simulated TODs.
        
        Args:
            use_truth_slope (bool): Boolean determining how slopes and offsets used for detrending atmosphere TODs are determined. If True, slopes and offsets are taken from simulated atmosphere TODs, otherwise the total simulated data (including noise) is used. Defaults to False.
        
        Dynamic Attributes (Added by Methods):
            jax_tods_map (array): Array with map TODs generated with jax. Added by FitHandler.sample_jax_tods().
            jax_tods_atmos (array): Array with atmosphere TODs generated with jax. Added by FitHandler.sample_jax_tods().
            noised_jax_tod (array): Array with total noised TODs generated with jax. Added by FitHandler.sample_jax_tods().
            denoised_jax_tod (array): Array with total denoised TODs generated with jax. Added by FitHandler.sample_jax_tods().
            slopes_tod (array): Array with slopes between atmosphere TODs. Taken from data if use_truth_slope=False, otherwise taken from the true atmosphere TODs. Added by FitHandler.sample_jax_tods(). 
            offset_tod (array): Array with offsets between atmosphere TODs. Taken from data if use_truth_slope=False, otherwise taken from the true atmosphere TODs. Added by FitHandler.sample_jax_tods(). 
            atmos_tod_simplified (array): Array with simplified atmosphere TODs, removing offsets and slopes. Added by FitHandler.sample_jax_tods(). 
        '''
        
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
        self.noised_jax_tod = np.float64(self.tod_truthmap.get_field('noise')*self.noiselevel)
        if self.fit_atmos:
            self.noised_jax_tod += np.float64(self.jax_tods_atmos)
        if self.fit_map:
            self.noised_jax_tod += np.float64(self.jax_tods_map)
        # denoised_jax_tod = noised_jax_tod - np.float64(tod_truthmap.get_field('noise')*noiselevel)

        # atmos-only:
        # noised_jax_tod = np.float64(jax_tods_atmos[:n]) + np.float64(tod_truthmap.get_field('noise')*noiselevel)[:n]

        # re-subtract noise:
        self.denoised_jax_tod = self.noised_jax_tod - np.float64(self.tod_truthmap.get_field('noise')*self.noiselevel)

        print("Noise stddev:", np.std(self.tod_truthmap.get_field('noise').compute()))

        fig, axes = plt.subplots(3, 1, figsize=(16, 8))

        for i in range(0, n, n//10 if n//10 != 0 else 1):
            im0 = axes[0].plot(self.jax_tods_map[i], label=i)
            im1 = axes[1].plot(self.jax_tods_atmos[i], label=i)
            im2 = axes[2].plot(self.noised_jax_tod[i], label=i)
            
        axes[0].title.set_text(f'JAX MAP TOD0-{i}')
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[1].title.set_text(f'Atmosphere TOD0-{i}')
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[2].title.set_text(f'Total TOD0-{i}, noise={self.noiselevel}')
        axes[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

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
        
    def init_gps(self, n_sub: int = 1, samples: jft.evi.Samples = None) -> None:
        '''
        Initialise atmosphere and map GPs. If n_sub and samples are provided, split atmos GPs in n_sub from samples.
        
        Args:
            n_sub (int): Integer determining how many GPs are used to simulate sub-detector atmosphere responses. Simulates all sub-detectors if -1. Defaults to 1.
            samples (jft.evi.Samples): Samples obtained in previous fit to use for initialisation. If None, a random initialisation is performed. Defaults to None.
        
        Dynamic Attributes (Added by Methods):
            n_sub (int): Integer determining how many GPs are used to simulate sub-detector atmosphere responses. Simulates all sub-detectors if -1. Added by FitHandler.init_gps().
            initial_pos (jft.Vector): Vector with initial position to be used in fit. Added by FitHandler.init_gps().
            padding_atmos (int): Integer describing atmosphere padding. Added by FitHandler.init_gps().
            dims_atmos (tuple[int,]): Tuple with atmosphere dimensions. Added by FitHandler.init_gps().
            cf_zm_tod (dict): Dictionary with atmosphere zeromode parameters. Added by FitHandler.init_gps().
            cf_fl_tod (dict): Dictionary with atmosphere fluctuation parameters. Added by FitHandler.init_gps().
            gp_tod (nifty_maria.modified_CFM.CFM): Modified CorrelatedFieldMaker object describing GP for atmosphere TODs. Added by FitHandler.init_gps().
            padding_map (int): Integer describing map padding. Added by FitHandler.init_gps().
            dims_map (tuple[int,int]): Tuple with map dimensions. Added by FitHandler.init_gps().
            cf_zm_map (dict): Dictionary with map zeromode parameters. Added by FitHandler.init_gps().
            cf_fl_map (dict): Dictionary with map fluctuation parameters. Added by FitHandler.init_gps().
            gp_map (jft.CorrelatedFieldMaker): CorrelatedFieldMaker object describing GP for map. Added by FitHandler.init_gps().
            signal_response_tod (jft.Model): Signal model containing forward model. Added by FitHandler.init_gps().
            lh (jft.Gaussian): Gaussian Likelihood used in optimisation. Added by FitHandler.init_gps().
        
        Raises:
            ValueError: If invalid number of subdetectors n_sub or invalid combination fo n_sub and samples is supplied.
        '''

        self.n_sub = n_sub
        
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
        
        posmask_ul = posmask_up & posmask_left
        posmask_ur = posmask_up & posmask_right
        posmask_dl = posmask_down & posmask_left
        posmask_dr = posmask_down & posmask_right
        
        if samples is not None and self.n_sub != 1:
            print(f"Will initialise GPs for atmosphere in {self.n_sub} parts based on samples.")
            
            if self.n_sub == 2:
                initial_pos = {}
                for k in samples.pos:
                    if k == 'combcf xi':
                        initial_pos[k] = jnp.broadcast_to(
                            samples.pos['combcf xi'],
                            (2, samples.pos['combcf xi'].shape[1])
                        ) # only works for one TOD!

                    else:
                        initial_pos[k] = samples.pos[k]

                self.initial_pos = jft.Vector(initial_pos)
                
            elif self.n_sub == 4:
                initial_pos = {}
                
                for k in samples.pos:
                    if k == 'combcf xi':
                        initial_pos[k] = jax.numpy.empty( (4, samples.pos['combcf xi'].shape[1]) )
                        initial_pos[k] = initial_pos[k].at[0].set( samples.pos['combcf xi'][0] )
                        initial_pos[k] = initial_pos[k].at[1].set( samples.pos['combcf xi'][0] )
                        initial_pos[k] = initial_pos[k].at[2].set( samples.pos['combcf xi'][1] )
                        initial_pos[k] = initial_pos[k].at[3].set( samples.pos['combcf xi'][1] )

                    else:
                        initial_pos[k] = samples.pos[k]

                self.initial_pos = jft.Vector(initial_pos)
            
            elif self.n_sub == -1:
                initial_pos = {}
                for k in samples.pos:
                    if k == 'combcf xi':
                        initial_pos[k] = jax.numpy.empty( (self.instrument.n_dets, samples.pos['combcf xi'].shape[1]) )
                        initial_pos[k] = initial_pos[k].at[posmask_up & posmask_left].set( samples.pos['combcf xi'][0] )
                        initial_pos[k] = initial_pos[k].at[posmask_down & posmask_left].set( samples.pos['combcf xi'][1] )
                        initial_pos[k] = initial_pos[k].at[posmask_up & posmask_right].set( samples.pos['combcf xi'][2] )
                        initial_pos[k] = initial_pos[k].at[posmask_down & posmask_right].set( samples.pos['combcf xi'][3] )

                    else:
                        initial_pos[k] = samples.pos[k]

                self.initial_pos = jft.Vector(initial_pos)
            
            else: raise ValueError("Only 1, 2, 4 and -1 (all) subdets implemented for now.")
            
        elif samples is not None and self.n_sub==1:
            raise ValueError("Samples can only be used for initialisation if n_sub is not 1!")
        
        else:
            self.initial_pos = None
        
        if self.fit_atmos:
            # padding_atmos = 2000
            # padding_atmos = 5000
            self.padding_atmos = 10000
            self.dims_atmos = ( (self.jax_tods_atmos.shape[1] + self.padding_atmos), )
            # dims_atmos = ( (jax_tods_atmos.shape[1] - 200 + padding_atmos), )

            # correlated field zero mode GP offset and stddev
            # cf_zm_tod = dict(offset_mean=jax_tods_atmos.mean().compute(), offset_std=(0.0002, 0.0001))
            # cf_zm_tod = dict(offset_mean=0.0, offset_std=(1e-5, 0.99e-5))
            # cf_zm_tod = dict(offset_mean=0.0, offset_std=(6e-6, 5e-6))
            # self.cf_zm_tod = dict(offset_mean=0.0, offset_std=(5e-5, 4e-5))
            self.cf_zm_tod = dict(offset_mean=0.0, offset_std=self.params['tod_offset'])

            # correlated field fluctuations (mostly don't need tuning)
            # fluctuations: y-offset in power spectrum in fourier space (zero mode)
            # loglogavgslope: power-spectrum slope in log-log space in frequency domain (Fourier space) Jakob: -4 -- -2
            # flexibility=(1.5e0, 5e-1), # deviation from simple power-law
            # asperity=(5e-1, 5e-2), # small scale features in power-law
            self.cf_fl_tod = dict(
                # # fluctuations=(0.0015, 0.0001),
                # # loglogavgslope=(-2.45, 0.1), 
                # fluctuations=(0.01, 0.003),
                # loglogavgslope=(-2.2, 0.2), 
                fluctuations=self.params['tod_fluct'],
                loglogavgslope=self.params['tod_loglog'],
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

            if self.n_sub == -1: self.gp_tod = cfm_tod.finalize(self.instrument.n_dets)
            else: self.gp_tod = cfm_tod.finalize(self.n_sub)
            
            print("Initialised gp_tod:", self.gp_tod)
        
        if self.fit_map:
            self.padding_map = 10
            # self.padding_map = 0
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
            # self.cf_zm_map = dict(offset_mean=self.mapdata_truth.mean(), offset_std=(2e-6, 1e-7))
            self.cf_zm_map = dict(offset_mean=self.mapdata_truth.mean(), offset_std=self.params['map_offset'])
            # correlated field fluctuations (mostly don't need tuning)
            self.cf_fl_map = dict(
                # fluctuations=(1e-4, 1e-5), # fluctuations: y-offset in power spectrum in fourier space (zero mode)
                # loglogavgslope=(-3.0, 0.1),
                fluctuations=self.params['map_fluct'], # fluctuations: y-offset in power spectrum in fourier space (zero mode)
                loglogavgslope=self.params['map_loglog'],
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
        from nifty_maria.SignalModels import Signal_TOD_atmosonly
        from nifty_maria.SignalModels import Signal_TOD_combined_nomappadding
        from nifty_maria.SignalModels import Signal_TOD_maponly_nomappadding
        from nifty_maria.SignalModels import Signal_TOD_combined_fourTODs
        from nifty_maria.SignalModels import Signal_TOD
        
        if self.noiselevel == 0.0: noise_cov_inv_tod = lambda x: 1e-8**-2 * x
        elif self.noiselevel == 0.1: noise_cov_inv_tod = lambda x: 1e-4**-2 * x
        elif self.noiselevel == 0.5: noise_cov_inv_tod = lambda x: 1e-4**-2 * x
        elif self.noiselevel == 1.0: noise_cov_inv_tod = self.params['noise']
        
        print("noise_cov_inv_tod", noise_cov_inv_tod)
        # elif self.noiselevel == 1.0: noise_cov_inv_tod = lambda x: 2.5e-4**-2 * x 
        # elif self.noiselevel == 1.0: noise_cov_inv_tod = lambda x: 1.9e-4**-2 * x
        
        #TODO Add split by dets and clean up!
        if self.fit_map and self.fit_atmos:
            if self.padding_map > 0:
                if self.n_sub == 1 or self.n_sub == 2:
                    print("Initialising model: Signal_TOD_combined")
                    self.signal_response_tod = Signal_TOD_combined(self.gp_tod, self.offset_tod, self.slopes_tod, self.gp_map, self.dims_map, self.padding_map, self.dims_atmos, self.padding_atmos, posmask_up, posmask_down, self.sim_truthmap, self.dx, self.dy)
                elif self.n_sub == 4:
                    print("Initialising model: Signal_TOD_combined_fourTODs")
                    self.signal_response_tod = Signal_TOD_combined_fourTODs(self.gp_tod, self.offset_tod, self.slopes_tod, self.gp_map, self.dims_map, self.padding_map, self.dims_atmos, self.padding_atmos, posmask_ul, posmask_ur, posmask_dl, posmask_dr, self.sim_truthmap, self.dx, self.dy)
                else:
                    print("Initialising model: Signal_TOD")
                    self.signal_response_tod = Signal_TOD(self.gp_tod, self.offset_tod, self.slopes_tod, self.gp_map, self.dims_map, self.padding_map, self.dims_atmos, self.padding_atmos, self.sim_truthmap, self.dx, self.dy)
            else:
                print("Initialising model: Signal_TOD_combined_nomappadding")
                self.signal_response_tod = Signal_TOD_combined_nomappadding(self.gp_tod, self.offset_tod, self.slopes_tod, self.gp_map, self.dims_map, self.dims_atmos, self.padding_atmos, posmask_up, posmask_down, self.sim_truthmap, self.dx, self.dy)
        elif self.fit_map and not self.fit_atmos:
            if self.padding_map > 0:
                raise ValueError("No model implemented for map-only with padding!")
            else:
                print("Initialising model: Signal_TOD_maponly_nomappadding")
                self.signal_response_tod = Signal_TOD_maponly_nomappadding(self.gp_map, self.dims_map, self.sim_truthmap, self.dx, self.dy)
        elif not self.fit_map and self.fit_atmos:
            print("Initialising model: Signal_TOD_atmosonly")
            self.signal_response_tod = Signal_TOD_atmosonly(self.gp_tod, self.offset_tod, self.slopes_tod, self.dims_atmos, self.padding_atmos, posmask_up, posmask_down)
        else:
            raise ValueError("Invalid combination: Need to fit atmosphere and/or map!")

        self.lh = jft.Gaussian( self.noised_jax_tod, noise_cov_inv_tod).amend(self.signal_response_tod)
        
        print(self.lh)
        
        return 
    
    def draw_prior_sample(self) -> jax.Array:
        '''
        Draws sample from prior model and makes a plot of sample drawn. Returns array with corresponding signal response. 
        
        Returns:
            jax.Array: A jax array containing signal response.
        '''
        self.key, sub = jax.random.split(self.key)
        xi = jft.random_like(sub, self.signal_response_tod.domain)
        res = self.signal_response_tod(xi)
        n = self.instrument.n_dets
        print(res.shape)

        fig, axes = plt.subplots(1, 1, figsize=(16, 4))

        for i in range(0, n, n//10 if n//10 != 0 else 1):
            axes.plot( np.arange(0, res.shape[1]), res[i], label=i)

        axes.title.set_text(f'all')
        axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        return res
    
    def perform_fit(self, n_it: int = 1, fit_type: str = 'full', printevery: int = 2) -> tuple[jft.evi.Samples, OptimizeVIState]:
        '''
        Performs nifty fit based on prior initialisation.
        
        Args:
            n_it (int): Integer value determining number of global iterations during optimisation. Defaults to 1.
            fit_type (str): String determining fit configuration. Options are 'full' for full MGVI fit and 'map' for maximum aposteriori. Defaults to 'full'.
            printevery (int): Integer determining interval for plotting intermediate fit results. Defaults to 2.
        
        Returns:
            :tuple[jft.evi.Samples, jft.optimize_kl.OptimizeVIState]: A tuple containing:
            - jft.evi.Samples: The samples obtained after fit has been performed
            - jft.optimize_kl.OptimizeVIState: The optimisation state after fit has been performed.
        '''
        if self.noiselevel == 0.0: delta = 1e-4
        elif self.noiselevel == 0.1: delta = 1e-10
        elif self.noiselevel == 0.5: delta = 1e-10
        elif self.noiselevel == 1.0: delta = 1e-4

        if fit_type == 'map':
            print("Running map fit!")
            n_samples = 0 # no samples -> maximum aposteriory posterior
            sample_mode = 'nonlinear_resample'
        elif fit_type == 'full':
            print("Running full fit!")
            n_samples = 4
            sample_mode = lambda x: "nonlinear_resample" if x >= 1 else "linear_resample"
        else:
            raise ValueError(f"fit_type {fit_type} not supported!")

        self.key, k_i, k_o = random.split(self.key, 3)

        def callback(samples: jft.evi.Samples, opt_state: OptimizeVIState) -> None:
            '''
            Callback function to be used for plotting fit status during optimisation.
            
            Args:
                samples (jft.evi.Samples): Samples to perform plots for.
                opt_state (jft.optimize_kl.OptimizeVIState): Optimisation state to plot.
            '''
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

            fig_tods.suptitle(f"n_sub = {self.n_sub}, iter: {iter}")
            axes_tods[0].title.set_text('total mean pred. & truth (no noise)')
            axes_tods[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            axes_tods[1].title.set_text('total mean pred. - truth (no noise)')
            axes_tods[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

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

                fig_tods.suptitle(f"n_sub = {self.n_sub}, iter: {iter}")
                axes_tods[0].title.set_text('mean atmos pred. & simplified truth (no noise)')
                axes_tods[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                axes_tods[1].title.set_text('mean atmos pred. - simplified truth (no noise)')
                axes_tods[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

            if self.fit_map:
                fig_map, axes_map = plt.subplots(1, 3, figsize=(16, 6))

                if self.padding_map > 0:
                    mean_map, _ = jft.mean_and_std(tuple(self.gp_map(s)[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in samples))
                else:
                    mean_map, _ = jft.mean_and_std(tuple(self.gp_map(s) for s in samples))

                im0 = axes_map[0].imshow(mean_map)
                axes_map[0].title.set_text('mean map pred.')
                fig_map.colorbar(im0)

                im1 = axes_map[1].imshow(mean_map - self.mapdata_truth[0, 0])
                axes_map[1].title.set_text('mean map - truth')
                fig_map.colorbar(im1)

                im2 = axes_map[2].imshow(self.mapdata_truth[0, 0])
                axes_map[2].title.set_text('truth')
                fig_map.colorbar(im2)

                fig_map.suptitle(f"n_sub = {self.n_sub}, iter: {iter}")
            
            plt.show()
            
            return

        if self.initial_pos is None:
            # self.initial_pos = 0.1*jft.Vector(self.lh.init(k_i))
            self.initial_pos = jft.Vector(self.lh.init(k_i))

        samples, state = jft.optimize_kl(
            self.lh, # likelihood
            # 0.1*jft.Vector(self.lh.init(k_i)), # initial position in model space (initialisation)
            self.initial_pos,
            n_total_iterations=n_it, # no of optimisation steps (global)
            n_samples=n_samples, # draw samples
            key=k_o, # random jax init
            draw_linear_kwargs=dict( # sampling parameters
                cg_name="SL",
                # cg_kwargs=dict(absdelta=delta * jft.size(self.lh.domain) / 10.0, maxiter=60),
                cg_kwargs=dict(absdelta=delta * jft.size(self.lh.domain) / 10.0, maxiter=20), #TODO: fine-tune!
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
    
    def printfitresults(self, samples: jft.evi.Samples) -> None:
        '''
        Prints optimised GP parameters and initial parameters for map and atmosphere GPs.
        
        Args:
            samples (jft.evi.Samples): Samples to print fit results for.
        '''
        print(f"Fit Results (res, init, std) for n_sub = {self.n_sub}")

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
    
    def plotfitresults(self, samples: jft.evi.Samples) -> None:
        '''
        Plots predictions made by optimised GP and compares with truth.
        
        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
        '''
        res = self.signal_response_tod(samples.pos)
        n = self.instrument.n_dets

        fig, axes = plt.subplots(3, 1, figsize=(16, 8))

        for i in range(0, n, n//10 if n//10 != 0 else 1):
            im0 = axes[0].plot(np.arange(0, res.shape[1]), res[i], label=f"tod{i}")
            im1 = axes[1].plot(np.arange(0, res.shape[1]), res[i] - self.noised_jax_tod[i], label=f"tod{i}")
            im2 = axes[2].plot(self.noised_jax_tod[i], label=f"truth{i}")

        fig.suptitle(f"n_sub = {self.n_sub}")
        axes[0].title.set_text('MAP - best fit image')
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[1].title.set_text('MAP - map truth')
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[2].title.set_text('truth')
        axes[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.show()
        
        if self.fit_map: 
            # plot maximum of posterior (mode)
            if self.padding_map > 0:
                sig_map = self.gp_map(samples.pos)[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] # when splitting up in different field models
            else:
                sig_map = self.gp_map(samples.pos)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            im0 = axes[0].imshow(sig_map)
            axes[0].title.set_text('MAP - best fit image')
            fig.colorbar(im0)

            im1 = axes[1].imshow( sig_map - self.mapdata_truth[0, 0] )
            axes[1].title.set_text('MAP - map truth')
            # im1 = axes[1].imshow( (sig_map - mapdata_truth) )
            # axes[1].title.set_text('diff prediction - map truth')
            fig.colorbar(im1)

            fig.suptitle(f"n_sub = {self.n_sub}")

            plt.show()
        
        return 
    
    def plotpowerspectrum(self, samples: jft.evi.Samples) -> None:
        '''
        Plots power spectrum of predictions made by optimised GP and compares with truth.
        
        Args:
            samples (jft.evi.Samples): Samples to plot power spectrum for.
        '''
        import scipy as sp

        # mean, std = jft.mean_and_std(tuple(signal_response_tod(s) for s in samples))
        from itertools import cycle

        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        if self.padding_map > 0:
            gp_map_nopad = jax.numpy.broadcast_to(self.gp_map(samples.pos), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        else:
            gp_map_nopad = jax.numpy.broadcast_to(self.gp_map(samples.pos), (1, 1, self.dims_map[0], self.dims_map[1]))
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)

        if not self.fit_atmos:
            components = [self.signal_response_tod(samples.pos), res_map, self.tod_truthmap.get_field('map')]
            labels = ['pred. total', 'pred. map', 'true map']
            linestyles = ['-', '-', '--']
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
            
        fig_tods.suptitle(f"n_sub = {self.n_sub}")
        axes_tods.set_xlabel('Frequency [Hz]')
        axes_tods.set_ylabel(f"[{self.tod_truthmap.units}$^2$/Hz]")
        axes_tods.set_xlim(f_mids.min(), f_mids.max())
        axes_tods.loglog()
        axes_tods.legend()
        
        return
    
    def plotrecos(self, samples: jft.evi.Samples) -> None:
        '''
        Plots comparison between maria and nifty reconstructions of the map with the true map.
        
        Args:
            samples (jft.evi.Samples): Samples to make comparison for.
        '''
        from skimage.transform import resize

        # Compare nifty vs maria
        if self.padding_map > 0:
            sig_map = self.gp_map(samples.pos)[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] # when splitting up in different field models
        else:
            sig_map = self.gp_map(samples.pos)
        # sig_map = self.gp_map(samples.pos)
        # mincol = -0.0012
        # maxcol = 0.
        mincol = None
        maxcol = None

        cmb_cmap = plt.get_cmap('cmb')
        fig, axes = plt.subplots(3, 2, figsize=(16, 16))

        im0 = axes[0,0].imshow( self.mapdata_truth[0,0] , cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[0,0].title.set_text('truth')
        fig.colorbar(im0)

        im1 = axes[0,1].imshow(self.output_truthmap.data[0,0], cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        fig.colorbar(im1)
        axes[0,1].title.set_text("Noisy image (Mapper output)")

        im2 = axes[1,0].imshow(self.output_map.data[0, 0], cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[1,0].title.set_text('maria mapper')
        fig.colorbar(im2)

        truth_rescaled = resize(self.mapdata_truth[0,0], (500, 500), anti_aliasing=True)
        im3 = axes[1,1].imshow((self.output_map.data[0, 0] - truth_rescaled), cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[1,1].title.set_text('maria - truth')
        fig.colorbar(im3)

        im3 = axes[2,0].imshow(sig_map, cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[2,0].title.set_text('best fit image')
        fig.colorbar(im3)

        im4 = axes[2,1].imshow((sig_map - self.mapdata_truth[0,0]), cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[2,1].title.set_text('best fit - truth')
        fig.colorbar(im4)

        plt.show()
        
        return 
        
    def make_atmosphere_det_gif(self, samples: jft.evi.Samples, figname: str = 'atmosphere_comp.gif', tmax: int = -1, num_frames: int = 100) -> None:
        '''
        Makes gif of simplified atmosphere prediction and truth in 2D detector layout. Does nothing if self.fit_atmos == False.
        
        Args:
            samples (jft.evi.Samples): Samples to make atmosphere prediction plot for.
            figname (str, optional): Location to save gif in. Defaults to 'atmosphere_comp.gif'
            tmax (int, optional): Maximum timestep to consider. If -1, will loop over all timesteps. Defaults to -1.
            num_frames (int, optional): Number of total frames to plot. Defaults to 100.
        '''
        
        if not self.fit_atmos:
            print("Not fitting atmosphere, skipping plot..")
            return 
        
        import matplotlib.pyplot as plt
        from PIL import Image
        import io

        # Generate and capture individual frames
        tmax = self.atmos_tod_simplified.shape[1] if tmax == -1 else tmax
        nskip = tmax//num_frames

        # Create a list to hold the frames
        frames = []

        for i in range(0, tmax, nskip):
            
            print(f"Making plot {i} out of {tmax}.")
            
            # fig = plot_time(instrument, dataarr, timestep=i, addtext=label)
            fig = self.plot_atmosphere_det(samples, timestep=i)
            
            # Capture the plot as an image in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            
            # Create an image from the buffer
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img)

        # Save the frames as a GIF
        # frames[0].save('testgif_atmosphere.gif', save_all=True, append_images=frames[1:], duration=1, loop=0)
        # frames[0].save(f'{label}_mustang_new.gif', save_all=True, append_images=frames[1:], duration=1, loop=0)
        frames[0].save(figname, save_all=True, append_images=frames[1:], duration=1, loop=0)
        
        return 
    
    def plot_atmosphere_det(self, samples: jft.evi.Samples, timestep: int = 0, z: float = np.inf) -> plt.Figure:
        '''
        Plots simplified atmosphere prediction and truth in 2D detector layout. Returns figure. Does nothing if self.fit_atmosphere == False.
        
        Args:
            samples (jft.evi.Samples): Samples to make atmosphere prediction plot for.
            timestep (int, optional): Timestep to make plot for.
            z (float, optional): Gaussian beam distance in instrument. Defaults to np.inf.
        
        Returns:
            plt.Figure: The produced figure object.
        
        Raises:
            ValueError: If invalid n_sub value is supplied.
        '''
        
        if not self.fit_atmos:
            print("Not fitting atmosphere, skipping plot..")
            return 
        
        from maria.units import Angle

        cmb_cmap = plt.get_cmap('cmb')

        x_tod = {k: samples.pos[k] for k in samples.pos if 'comb' in k}
        best_fit_atmos = self.gp_tod(x_tod)[:, self.padding_atmos//2:-self.padding_atmos//2]

        # re-define masks. TODO: automate & define globally.
        test = Angle(self.instrument.dets.offsets)
        pos = getattr(test, test.units).T

        posmask_ud = jnp.array((pos[1] >= (pos[1].max() + pos[1].min())/2))
        posmask_lr = jnp.array((pos[0] >= (pos[0].max() + pos[0].min())/2))

        posmask_up = posmask_ud
        posmask_down = ~posmask_ud
        posmask_left = posmask_lr
        posmask_right = ~posmask_lr

        col = np.zeros(posmask_right.shape)
        if self.n_sub == 4:
            col[posmask_up & posmask_left] = best_fit_atmos[0, timestep]
            col[posmask_down & posmask_left] = best_fit_atmos[1, timestep]
            col[posmask_up & posmask_right] = best_fit_atmos[2, timestep]
            col[posmask_down & posmask_right] = best_fit_atmos[3, timestep]
        elif self.n_sub == 2:
            col[posmask_up] = best_fit_atmos[0, timestep]
            col[posmask_down] = best_fit_atmos[1, timestep]
        elif self.n_sub == 1:
            col[:] = best_fit_atmos[0, timestep]
        elif self.n_sub == -1:
            col = best_fit_atmos[:, timestep]
        else:
            raise ValueError(f"Value for n_sub {self.n_sub} is not supported!")

        fig, ax = plt.subplots(1, 3, figsize=(8*3, 6))

        true_atmos = self.atmos_tod_simplified[:, timestep].compute()

        self.plot_instrument(fig, ax[0], col, cmb_cmap, z=z)
        self.plot_instrument(fig, ax[1], true_atmos, cmb_cmap, z=z)
        self.plot_instrument(fig, ax[2], col - true_atmos, cmb_cmap, z=z)

        time = self.tod_truthmap.time - self.tod_truthmap.time[0]

        fig.suptitle(f"n_sub = {self.n_sub}, time = {time[timestep]:.2f} s")
        ax[0].title.set_text("pred. simpl. atmosphere")
        ax[1].title.set_text("true simpl. atmosphere")
        ax[2].title.set_text("pred.-true simpl. atmosphere")
        
        # plt.show()
        
        return fig
        
    def plot_subdets(self, z: float = np.inf) -> None:
        '''
        Plots detector with n_sub subdetectors highlighted in color.
        
        Args:
            z (float, optional): Gaussian beam distance in instrument. Defaults to np.inf.

        Raises:
            ValueError: If invalid n_sub value is supplied.
        '''
        from matplotlib.collections import EllipseCollection
        from maria.units import Angle

        # cmb_cmap = plt.get_cmap('cmb')
        cmb_cmap = plt.get_cmap('viridis')

        # re-define masks. TODO: automate & define globally.
        test = Angle(self.instrument.dets.offsets)
        pos = getattr(test, test.units).T

        posmask_ud = jnp.array((pos[1] >= (pos[1].max() + pos[1].min())/2))
        posmask_lr = jnp.array((pos[0] >= (pos[0].max() + pos[0].min())/2))

        posmask_up = posmask_ud
        posmask_down = ~posmask_ud
        posmask_left = posmask_lr
        posmask_right = ~posmask_lr

        col = np.zeros(posmask_right.shape)
        if self.n_sub == 4:
            col[posmask_up & posmask_left] = 1.0
            col[posmask_up & posmask_right] = 0.25
            col[posmask_down & posmask_left] = 0.5
            col[posmask_down & posmask_right] = 0.75
        elif self.n_sub == 2:
            col[posmask_up] = 0.25
            col[posmask_down] = 1.0
        elif self.n_sub == 1:
            col = np.ones(self.instrument.n_dets)
        elif self.n_sub == -1:
            col = np.linspace(0, 1, self.instrument.n_dets)
        else:
            raise ValueError(f"Value for n_sub {self.n_sub} is not supported!")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=160)

        self.plot_instrument(fig, ax, col, cmb_cmap, z=z)

        fig.suptitle(f"n_sub = {self.n_sub}")
        plt.show()
        
        return 
    
    def plot_instrument(self, fig: plt.Figure, ax: plt.Axes, col: np.ndarray, cmb_cmap: plt.Colormap, z: float = np.inf) -> None:
        '''
        Plots detector with a given color values col into figure fig and axis ax.
        
        Args:
            fig (plt.Figure): Figure to plot into.
            ax (plt.Axes): Axes to plot into.
            col (np.ndarray): Numpy array containing colors to be plotted.
            cmb_cmap (plt.Colormap): Colormap to use for plotting.
            z (float, optional): Gaussian beam distance in instrument. Defaults to np.inf.
        '''
        import matplotlib as mpl
        from matplotlib.collections import EllipseCollection
        from matplotlib.patches import Patch
        # from matplotlib.colors import Normalize
        from maria.units import Angle

        # norm = Normalize(vmin = np.min(col), vmax=np.max(col))

        fwhms = Angle(self.instrument.dets.angular_fwhm(z=z))
        offsets = Angle(self.instrument.dets.offsets)

        i = 0

        for ia, array in enumerate(self.instrument.arrays):
            array_mask = self.instrument.dets.array_name == array.name

            for ib, band in enumerate(array.dets.bands):
                band_mask = self.instrument.dets.band_name == band.name
                mask = array_mask & band_mask

                collection = EllipseCollection(
                        widths=getattr(fwhms, offsets.units)[mask],
                        heights=getattr(fwhms, offsets.units)[mask],
                        angles=0,
                        units="xy",
                        # facecolors=cmb_cmap(col),
                        # facecolors=cmb_cmap(norm(col)),
                        edgecolors="k",
                        lw=1e-1,
                        # alpha=0.5,
                        offsets=getattr(offsets, offsets.units)[mask],
                        transOffset=ax.transData,
                    )
                
                vmin = np.min(col)*(1-1e-5) if np.min(col) >= 0. else np.min(col)*(1+1e-5)
                vmax = np.max(col)*(1+1e-5) if np.min(col) >= 0. else np.max(col)*(1-1e-5)
                
                collection.set_clim(vmin, vmax)
                collection.set_cmap(cmb_cmap)
                collection.set_array(col.ravel())
                ax.add_collection(collection)

                scatter = ax.scatter(
                    *getattr(offsets, offsets.units)[band_mask].T,
                    # label=band.name,
                    s=2.0,
                    # color=col,
                    c=col,
                    cmap=cmb_cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                i += 1

        fig.colorbar(scatter)
        # fig.colorbar(collection)
        # fig.suptitle(f"n_sub = {self.n_sub}")
        ax.set_xlabel(rf"$\theta_x$ offset ({offsets.units})")
        ax.set_ylabel(rf"$\theta_y$ offset ({offsets.units})")

        xls, yls = ax.get_xlim(), ax.get_ylim()
        cen_x, cen_y = np.mean(xls), np.mean(yls)
        wid_x, wid_y = np.ptp(xls), np.ptp(yls)
        radius = 0.5 * np.maximum(wid_x, wid_y)

        margin = getattr(fwhms, offsets.units).max()

        ax.set_xlim(cen_x - radius - margin, cen_x + radius + margin)
        ax.set_ylim(cen_y - radius - margin, cen_y + radius + margin)
        
        return 