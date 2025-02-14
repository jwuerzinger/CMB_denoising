'''
Module to collect maria steering configs and code.
'''

import numpy as np
import maria
import matplotlib.pyplot as plt

import nifty_maria.mapsampling_jax

class MariaSteering:
    def __init__(self, fit_map: bool, fit_atmos: bool, config: str, noiselevel: int) -> None:
        '''
        Initialize the steering handler.
        
        Args:
            fit_map (bool): Perform fit of map if True.
            fit_atmos (bool): Perform fit of atmosphere if True.
            config (str): The detector configuraion to run on. Options are: 'mustang', 'atlast' and 'atlast_debug'.
            noiselevel (float): The fraction of noise to add.
        
        Raises:
            ValueError: If invalid configuration is used.
        '''
        self.config = config
        self.fit_map = fit_map
        self.fit_atmos = fit_atmos
        self.noiselevel = noiselevel
        
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
                # 'tod_offset' : (1e-5, 0.99e-5),
                'tod_offset' : (5e-5, 0.99e-5),
                # 'tod_fluct' : (0.0015, 0.0001),
                'tod_fluct' : (0.005, 0.0001),
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
                    #   sample_rate=225, # in Hz
                      sample_rate=25, # in Hz # TODO: change back!
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
        
        return
    
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