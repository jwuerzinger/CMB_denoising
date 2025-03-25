"""
Module to collect maria steering configs and code.
"""

import numpy as np
import maria
import matplotlib.pyplot as plt

import nifty_maria.mapsampling_jax

class MariaSteering:
    """Subclass for steering maria code."""
    def __init__(self) -> None:
        """
        Initialize the steering handler.
        """
        
        maria_params = self.confdict['maria_params']
        self.scan_center = tuple(maria_params['scan_center'])
        map_filename = maria.io.fetch(maria_params['map_filename'])
        # load in the map from a fits file
        self.input_map = maria.map.read_fits(filename=map_filename, #filename
                                        resolution=maria_params['resolution'], #pixel size in degrees
                                        width=maria_params['width'],
                                        index=maria_params['index'], #index for fits file
                                        center=self.scan_center, # position in the sky
                                        units='Jy/pixel' # Units of the input map 
                                    )

        self.input_map.to(units="K_RJ").plot()

        self.plan = maria.get_plan(scan_pattern="daisy", # scanning pattern
                                scan_options=maria_params['scan_options'], # in degrees
                                duration=maria_params['duration'], # integration time in seconds
                                sample_rate=maria_params['sample_rate'], # in Hz
                                scan_center=self.scan_center, # position in the sky
                                start_time=maria_params['start_time'],
                                frame="ra_dec"
                                )

        self.plan.plot()
                
        self.instrument = nifty_maria.mapsampling_jax.instrument
        self.instrument.plot()
        
        return
    
    def simulate(self) -> None:
        """
        Performs maria simulation and decorates self with simulation parameters.
        
        Dynamic Attributes (Added by Methods):
            sim_truthmap (Simulation): Simulation object containing instrument, plan, site, input map and parameters for noise, atmosphere and cmb simulation. Added by FitHandler.simulate().
            tod_truthmap (TOD): TOD object containing simulated time-stream data. Added by FitHandler.simulate().
            dx (array): Array with detector offsets in x-direction. Added by FitHandler.simulate().
            dy (array): Array with detector offsets in y-direction. Added by FitHandler.simulate().
        """
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
        """
        Performs maria reconstruction and decorates self with reconstructed maps.
        
        Dynamic Attributes (Added by Methods):
            output_truthmap (Map): Noised Map object obtained by reconstruction without postprocessing. Added by FitHandler.reco_maria().
            mapdata_truth (array): Array with true simulated map. Added by FitHandler.reco_maria().
            output_map (Map): Map opbject obtained by reconstruction with postprocessing. Added by FitHandler.reco_maria().
        """
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