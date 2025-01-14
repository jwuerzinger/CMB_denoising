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
from nifty_maria.plotting import Plotter

import nifty8.re as jft
from nifty8.re.optimize_kl import OptimizeVIState

class FitHandler(Plotter):
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
    def __init__(self, fit_map: bool = True, fit_atmos: bool = True, config: str = 'atlast_debug', noiselevel: int = 1.0, plotsdir: str = None) -> None:
        '''
        Initialises the FitHandler with base attributes.
        
        Args:
            fit_map (bool, optional): Perform fit of map if True. Defaults to True.
            fit_atmos (bool, optional): Perform fit of atmosphere if True. Defaults to True.
            config (str, optional): The detector configuraion to run on. Options are: 'mustang', 'atlast' and 'atlast_debug'. Defaults to 'atlast_debug'.
            noiselevel (float, optional): The fraction of noise to add. Defaults to 1.0.
            plotsdir (str, optional): Directory to save results in. Defaults to None, resulting in plt.show().
            
        Raises:
            ValueError: If invalid configuration is used.
        '''
        
        print("Initialising...")
        self.fit_map = fit_map
        self.fit_atmos = fit_atmos
        self.config = config
        self.noiselevel = noiselevel
        self.plotsdir = plotsdir
        
        if self.plotsdir is not None: os.system(f"mkdir {self.plotsdir}")
        
        if self.config == 'mustang':
            self.scan_center = (150, 10)
            map_filename = maria.io.fetch("maps/cluster.fits")

            # load in the map from a fits file
            self.input_map = maria.map.read_fits(filename=map_filename, #filename
                                            resolution=8.714e-05, #pixel size in degrees
                                            index=0, #index for fits file
                                            center=self.scan_center, # position in the sky
                                            units='Jy/pixel' # Units of the input map 
                                        )

            self.input_map.to(units="K_RJ").plot()
            
            #load the map into maria
            self.plan = maria.get_plan(scan_pattern="daisy", # scanning pattern
                                scan_options={"radius": 0.05, "speed": 0.01}, # in degrees
                                duration=600, # integration time in seconds
                                sample_rate=50, # in Hz
                                scan_center=self.scan_center, # position in the sky
                                frame="ra_dec")

            self.plan.plot()
            
            self.instrument = nifty_maria.mapsampling_jax.instrument
            self.instrument.plot()
            
            noiseval = 2.5e-4 if self.noiselevel == 1.0 else 1e-7
            print(f"Running with noise value: {noiseval}")
            self.params = { 
                'tod_offset' : (1e-5, 0.99e-5),
                'tod_fluct' : (0.0015, 0.0001),
                'tod_loglog' : (-2.45, 0.1),
                'map_offset' : (1e-8, 1e-7),
                'map_fluct' : (5.6e-5, 1e-6),
                'map_loglog' : (-2.5, 0.1),
                # 'noise' : lambda x: 2.5e-4**-2 * x, # TODO: generalize!
                'noise' : lambda x: noiseval**-2 * x,
            }
            
        elif self.config == 'atlast':
            self.scan_center = (300, -10)
            map_filename = maria.io.fetch("maps/big_cluster.fits")
            
            self.input_map = maria.map.read_fits(filename=map_filename,
                                width=1., #degrees
                                index=1,
                                center=self.scan_center, #RA and Dec in degrees
                                units ='Jy/pixel'
                               )
            self.input_map.to(units="K_RJ").plot()
            
            # Default AtLAST plan: # TODO: make this work!
            # self.plan = maria.get_plan(scan_pattern="daisy",
            #           scan_options={"radius": 0.25, "speed": 0.5}, # in degrees
            #           duration=60, # in seconds
            #           sample_rate=225, # in Hz
            #           start_time = "2022-08-10T06:00:00",
            #           scan_center=self.scan_center,
            #           frame="ra_dec")
            
            # For debugging:
            self.plan = maria.get_plan(scan_pattern="daisy",
                      scan_options={"radius": 0.25, "speed": 0.5}, # in degrees
                      duration=60, # in seconds
                    # duration=6, # TODO: change back!
                      sample_rate=225, # in Hz
                    #   sample_rate=25, # in Hz # TODO: change back!
                      start_time = "2022-08-10T06:00:00",
                      scan_center=self.scan_center,
                      frame="ra_dec")
            
            self.plan.plot()

            self.instrument = nifty_maria.mapsampling_jax.get_atlast()
            self.instrument.plot()
            
            self.params = {
                # 'tod_offset' : (5e-5, 4e-5),
                # 'tod_fluct' : (0.01, 0.003),
                # 'tod_loglog' : (-2.2, 0.2),
                # 'map_offset' : (1e-8, 1e-7),
                # 'map_fluct' : (5.5e-5, 1e-6),
                # 'map_loglog' : (-3.7, 0.1),
                'tod_offset' : (5e-5, 1e-5),
                'tod_fluct' : (0.073, 0.003),
                'tod_loglog' : (-2.8, 0.2),
                'map_offset' : (5.3e-7, 1e-7),
                'map_fluct' : (6.1e-5, 1e-6),
                'map_loglog' : (-3.6, 0.1),
                # TODO: generalize!
                'noise' : lambda x: 0.00028**-2 * x, # For 225 Hz
                # 'noise' : lambda x: 1.9e-4**-2 * x, # For 100 Hz
                # 'noise' : lambda x: 1.0e-4**-2 * x, # for 25Hz
            }
            
        elif self.config == 'atlast_debug':
            self.scan_center = (300, -10)
            map_filename = maria.io.fetch("maps/cluster.fits")
        
        
            self.input_map = maria.map.read_fits(
                nu=150,
                filename=map_filename,  # filename
                # resolution=8.714e-05,  # pixel size in degrees
                width=1.,
                index=0,  # index for fits file
                # center=(150, 10),  # position in the sky
                center=self.scan_center,  # position in the sky
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
                scan_center=self.scan_center,
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
            site="llano_de_chajnantor", # green_bank
            map=self.input_map,
            # noise=False,
            atmosphere="2d",
            # cmb="generate",
        )

        self.tod_truthmap = self.sim_truthmap.run()
        
        # Plot TODs:
        self.tod_truthmap.plot()
        
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

        cmb_cmap = plt.get_cmap('cmb')
        
        mapper_truthmap = BinMapper(
            # center=(300.0, -10.0),
            center=self.scan_center,
            frame="ra_dec",
            width= 0.1 if self.config == 'mustang' else 1.,
            height= 0.1 if self.config == 'mustang' else 1.,
            resolution=np.degrees(np.nanmin(self.instrument.dets.fwhm[0]))/4.,
            map_postprocessing={"gaussian_filter": {"sigma": 0} }
        )
        mapper_truthmap.add_tods(self.tod_truthmap)
        self.output_truthmap = mapper_truthmap.run()

        mapdata_truth = np.float64(self.sim_truthmap.map.data)
        self.mapdata_truth = np.nan_to_num(mapdata_truth, nan=np.nanmean(mapdata_truth)) # replace nan value by img mean

        print("mapdata_truth shape:", self.mapdata_truth.shape)
        print("mapdata_truth mean:", self.mapdata_truth.mean())

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        im0 = axes[0].imshow(self.output_truthmap.data[0].T, cmap=cmb_cmap)
        fig.colorbar(im0)
        axes[0].title.set_text("Noisy image (Mapper output)")

        im1 = axes[1].imshow(self.mapdata_truth[0,0], cmap=cmb_cmap)
        fig.colorbar(im1)
        axes[1].title.set_text("True Image")

        if self.plotsdir is None: plt.show()
        else: 
            plt.savefig(f"{self.plotsdir}/reco_maria.png")
            plt.close()
        
        # Run proper mapmaker TODO: customize per fit
        if self.config == 'mustang':
            mapper = BinMapper(self.scan_center,
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
        elif self.config == 'atlast': # TODO: optimise!
            mapper = BinMapper(self.scan_center,
                    frame="ra_dec",
                    width=1.,
                    height=1.,
                    resolution=np.degrees(np.nanmin(self.instrument.dets.fwhm[0]))/4.,
                    tod_preprocessing={
                            # "window": {"name": "hamming"},
                            "window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
                            "remove_modes": {"modes_to_remove": [0]},
                            "despline": {"knot_spacing": 5},
                        },
                        map_postprocessing={
                            "gaussian_filter": {"sigma": 1},
                            "median_filter": {"size": 1},
                        },
                    )
        
        mapper.add_tods(self.tod_truthmap)
        self.output_map = mapper.run()
        
        self.output_map.plot()
        
        return
    
    def sample_jax_tods(self, use_truth_slope: bool = False) -> None:
        '''
        Sample TODs using jax map sampling, make plots comparing to TODs generated with maria and decorate self with simulated TODs.
        
        Args:
            use_truth_slope (bool, optional): Boolean determining how slopes and offsets used for detrending atmosphere TODs are determined. If True, slopes and offsets are taken from simulated atmosphere TODs, otherwise the total simulated data (including noise) is used. Defaults to False.
        
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

        n = self.instrument.n_dets
        for i in range(0, n, n//10 if n//10 != 0 else 1):
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

        if self.plotsdir is None: plt.show()
        else: 
            plt.savefig(f"{self.plotsdir}/jax_map_agreement.png")
            plt.close()

        self.jax_tods_atmos = self.tod_truthmap.get_field('atmosphere')
        # noised_jax_tod = np.float64(jax_tods_map) + np.float64(jax_tods_atmos) + np.float64(tod_truthmap.components['noise']*noiselevel)

        # Map + atmos
        self.noised_jax_tod = np.float64(self.tod_truthmap.get_field('noise')*self.noiselevel)
        if self.fit_atmos:
            self.noised_jax_tod += np.float64(self.jax_tods_atmos)
        if self.fit_map:
            self.noised_jax_tod += np.float64(self.jax_tods_map)
            
        # In atmos-only fit: only consider 0th TOD:
        if self.fit_atmos and not self.fit_map:
            print("Only considering 0th TOD!")
            self.noised_jax_tod = np.float64(self.tod_truthmap.get_field('noise')*self.noiselevel)[0] + np.float64(self.jax_tods_atmos)[0]
            self.noised_jax_tod = self.noised_jax_tod[None, :]
        
            self.denoised_jax_tod = self.noised_jax_tod - np.float64(self.tod_truthmap.get_field('noise')*self.noiselevel)[0]
        
        else:
            self.denoised_jax_tod = self.noised_jax_tod - np.float64(self.tod_truthmap.get_field('noise')*self.noiselevel)

        slopes_tod_truth = (self.jax_tods_atmos) / (self.jax_tods_atmos[0])
        slopes_tod_truth = np.float64(slopes_tod_truth.mean(axis=1))
        slopes_tod = self.noised_jax_tod / self.noised_jax_tod[0]
        slopes_tod = np.float64(slopes_tod.mean(axis=1))
        
        offset_tod_truth = np.float64(self.jax_tods_atmos.mean(axis=1))
        offset_tod = np.float64(self.noised_jax_tod.mean(axis=1))
        
        if self.fit_atmos and not self.fit_map:
            slopes_tod = slopes_tod[None]
            offset_tod = offset_tod[None]
            
        if use_truth_slope:
            self.slopes_tod = slopes_tod_truth
            self.offset_tod = offset_tod_truth
        else:
            self.slopes_tod = slopes_tod
            self.offset_tod = offset_tod
        
        # Get simplified atmosphere tods for validation
        self.atmos_tod_simplified = (self.jax_tods_atmos - self.offset_tod[:, None])/self.slopes_tod[:, None]

        print("Noise stddev:", np.std(self.tod_truthmap.get_field('noise').compute()))

        fig, axes = plt.subplots(3, 1, figsize=(16, 8))

        n = self.noised_jax_tod.shape[0]
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

        if self.plotsdir is None: plt.show()
        else: 
            plt.savefig(f"{self.plotsdir}/input_TODs.png")
            plt.close()
        
        return 
        
    def init_gps(self, n_split: int = 0, samples: jft.evi.Samples = None) -> None:
        '''
        Initialise atmosphere and map GPs. If n_sub and samples are provided, split atmos GPs in n_sub from samples.
        
        Args:
            n_split (int, optional): Integer determining how many subdetectors to split into. Determines how many GPs are used to simulate sub-detector atmosphere responses with n_sub = 2**n_split. Simulates all sub-detectors if -1. Defaults to 0, resulting in one subdetector GP.
            samples (jft.evi.Samples, optional): Samples obtained in previous fit to use for initialisation. If None, a random initialisation is performed. Defaults to None.
        
        Dynamic Attributes (Added by Methods):
            n_split (int): Integer determining how many splittings to apply. Also determines self.n_sub via self.n_sub=2**self.n_split (-1 if n_split = -1).
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
            ValueError: If invalid number of splittings n_split or invalid combination fo n_sub and samples is supplied.
        '''

        self.n_split = n_split
        if self.n_split >= 0:
            self.n_sub = 2**self.n_split
        elif self.n_split == -1:
            self.n_sub = -1
        else:
            raise ValueError(f"Invalid splitting {self.n_split} supplied!")
        
        from maria.units import Angle

        test = Angle(self.instrument.dets.offsets)
        pos = getattr(test, test.units).T

        # TODO: generalise splitting here:
        if samples is not None and self.n_split >= 1:
            if self.n_sub != samples.pos['combcf xi'].shape[0]*2:
                raise ValueError("Only two-fold splitting is supported for now!")
            
            initial_pos = {}
            for k in samples.pos:
                if k == 'combcf xi':
                    # broadcast previous fit results to new ones!
                    initial_pos[k] = jnp.empty( (self.n_sub, samples.pos['combcf xi'].shape[1]) )
                    for i in range(self.n_sub): # TODO: vectorize
                        initial_pos[k] = initial_pos[k].at[i].set( samples.pos['combcf xi'][i//2] )
                else:
                    initial_pos[k] = samples.pos[k]
        
            self.initial_pos = jft.Vector(initial_pos)
        elif samples is not None and self.n_split == -1:
            initial_pos = {}
            for k in samples.pos:
                if k == 'combcf xi':
                    def apply_mask(mask, res_tod):
                        return jnp.where(mask[:, None], res_tod, jnp.zeros_like(res_tod))

                    initial_pos[k] = jax.vmap(apply_mask, in_axes=(0, 0))(self.masklist, samples.pos['combcf xi'])
                    initial_pos[k] = jnp.sum(initial_pos[k], axis=0)
                    # initial_pos[k] = jax.numpy.empty( (self.instrument.n_dets, samples.pos['combcf xi'].shape[1]) )
                    # initial_pos[k] = initial_pos[k].at[posmask_up & posmask_left].set( samples.pos['combcf xi'][0] )
                    # initial_pos[k] = initial_pos[k].at[posmask_down & posmask_left].set( samples.pos['combcf xi'][1] )
                    # initial_pos[k] = initial_pos[k].at[posmask_up & posmask_right].set( samples.pos['combcf xi'][2] )
                    # initial_pos[k] = initial_pos[k].at[posmask_down & posmask_right].set( samples.pos['combcf xi'][3] )
                else:
                    initial_pos[k] = samples.pos[k]
                    
            self.initial_pos = jft.Vector(initial_pos)

        else:
            self.initial_pos = None 
        
        # Define makslist (only if n_split is not -1)
        if self.n_split != -1:
            # Include tiny offset to avoid double-counting of dets:            
            min_x, max_x = 1.001*np.float64(pos[0].min()), 1.002*np.float64(pos[0].max())
            min_y, max_y = 1.001*np.float64(pos[1].min()), 1.002*np.float64(pos[1].max())

            # Compute the step size for each square
            n_sub_x = 2**(n_split//2)
            n_sub_y = 2**((n_split+1)//2)
            x_step = (max_x - min_x) / n_sub_x
            y_step = (max_y - min_y) / n_sub_y

            self.masklist = []
            for y_i in range(0, n_sub_y):
                yval = min_y + y_i * y_step
                for x_i in range(0, n_sub_x):
                    xval = min_x + x_i * x_step

                    posmask_x = jnp.array( (pos[0] > xval) & (pos[0] <= xval + x_step) )
                    posmask_y = jnp.array( (pos[1] > yval) & (pos[1] <= yval + y_step) )
                    
                    self.masklist.append( (posmask_x & posmask_y) )

            self.masklist = jnp.array(self.masklist)
        
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
            else: 
                if self.n_sub > self.instrument.n_dets: raise ValueError(f"ERROR: self.n_sub = {self.n_sub} is not allowed to be larger than self.instrument.n_dets = {self.instrument.n_dets}!")
                self.gp_tod = cfm_tod.finalize(self.n_sub)
            
            print("Initialised gp_tod:", self.gp_tod)
        
        if self.fit_map:
            self.padding_map = 10
            # self.padding_map = 0
            if self.config == 'atlast':
                self.dims_map = (1024 + self.padding_map, 1024 + self.padding_map)
            else:
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
        
        # TODO: Define signal model with generalised masking!
        from nifty_maria.SignalModels import Signal_TOD_general, Signal_TOD_alldets, Signal_TOD_alldets_maponly, Signal_TOD_atmos
        
        if self.noiselevel == 0.0: noise_cov_inv_tod = lambda x: 1e-8**-2 * x
        elif self.noiselevel == 0.1: noise_cov_inv_tod = lambda x: 1e-4**-2 * x
        elif self.noiselevel == 0.5: noise_cov_inv_tod = lambda x: 1e-4**-2 * x
        elif self.noiselevel == 1.0: noise_cov_inv_tod = self.params['noise']
        
        print("noise_cov_inv_tod", noise_cov_inv_tod)
        # elif self.noiselevel == 1.0: noise_cov_inv_tod = lambda x: 2.5e-4**-2 * x 
        # elif self.noiselevel == 1.0: noise_cov_inv_tod = lambda x: 1.9e-4**-2 * x
        
        if self.n_sub >= 1:
            if self.fit_atmos and self.fit_map:
                print("Initialising: Signal_TOD_general!!")
                self.signal_response_tod = Signal_TOD_general(self.gp_tod, self.offset_tod, self.slopes_tod, self.gp_map, self.dims_map, self.padding_map, self.dims_atmos, self.padding_atmos, self.masklist, self.sim_truthmap, self.dx, self.dy)
            elif self.fit_atmos and not self.fit_map:
                print("Initialising: Signal_TOD_atmos!!")
                self.signal_response_tod = Signal_TOD_atmos(self.gp_tod, self.offset_tod, self.slopes_tod, self.dims_atmos, self.padding_atmos)
            else:
                raise ValueError("Config not supported!")
        elif self.n_sub == -1:
            if self.fit_atmos and self.fit_map:
                print("Initialising: Signal_TOD_alldets")
                self.signal_response_tod = Signal_TOD_alldets(self.gp_tod, self.offset_tod, self.slopes_tod, self.gp_map, self.dims_map, self.padding_map, self.dims_atmos, self.padding_atmos, self.sim_truthmap, self.dx, self.dy)
            elif not self.fit_atmos and self.fit_map:
                print("Initialising: Signal_TOD_alldets_atmosonly")
                self.signal_response_tod = Signal_TOD_alldets_maponly(self.gp_map, self.dims_map, self.padding_map, self.sim_truthmap, self.dx, self.dy)
            else:
                raise ValueError("Config not supported!")
        else:
            raise ValueError("Number of subdetectors not supported!")

        self.lh = jft.Gaussian( self.noised_jax_tod, noise_cov_inv_tod).amend(self.signal_response_tod)
        
        print("Initialised Likelihood:", self.lh)
        
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
        # n = self.instrument.n_dets
        n = self.noised_jax_tod.shape[0]

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
            printevery (int, optional): Integer determining interval for plotting intermediate fit results. Defaults to 2.
        
        Returns:
            :tuple[jft.evi.Samples, jft.optimize_kl.OptimizeVIState]: A tuple containing:
            - jft.evi.Samples: The samples obtained after fit has been performed
            - jft.optimize_kl.OptimizeVIState: The optimisation state after fit has been performed.
        '''
        # Promote printevery to argument to be accessible:
        self.printevery = printevery
        
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
            # sample_mode = lambda x: "nonlinear_resample" if x >= 1 else "linear_resample"
            sample_mode = "nonlinear_resample"
        else:
            raise ValueError(f"fit_type {fit_type} not supported!")

        self.key, k_i, k_o = random.split(self.key, 3)

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
                cg_kwargs=dict(absdelta=delta * jft.size(self.lh.domain) / 10.0, maxiter=1000),
                # cg_kwargs=dict(absdelta=delta * jft.size(self.lh.domain) / 10.0, maxiter=20), #TODO: fine-tune!
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
                    name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=100 # map
                    # name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=20 # map
                )
            ),
            sample_mode=sample_mode, # how steps are combined (samples + nonlin + KL),
            callback=self.callback if fit_type != 'map' else None
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