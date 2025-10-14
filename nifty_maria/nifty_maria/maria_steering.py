"""
Module to collect maria steering configs and code.
"""

import numpy as np
import matplotlib.pyplot as plt
import maria
from maria.constants import k_B
from maria.instrument import Band

class MariaSteering:
    """Subclass for steering maria code."""
    def __init__(self) -> None:
        """
        Initialize the steering handler.
        """
        
        maria_params = self.confdict['maria_params']
        self.maria_params = maria_params
        self.scan_center = tuple(maria_params['scan_center'])
        if self.config == 'mustang' or self.config == 'test':
            map_filename = maria.io.fetch(maria_params['map_filename'])
        else:
            map_filename = maria_params['map_filename']
        # load in the map from a fits file
        
        self.input_map = maria.map.load(filename=map_filename, #filename
                                        nu = maria_params['nu'],
                                        resolution=maria_params['resolution'], #pixel size in degrees
                                        width=maria_params['width'],
                                        center=self.scan_center, # position in the sky
                                        frame="ra_dec",
                                        units=maria_params['units'] # Units of the input map 
                                    )

        if self.config == 'mustang' or self.config == 'test':
            self.input_map.data = self.input_map.data/8.

        if self.config == 'atlast':
            self.input_map.data = self.input_map.data *5e1

        self.input_map.to(units="K_RJ").plot()
        plt.savefig(fname=f"{self.plotsdir}/input_map.png")

        self.plan = maria.get_plan(scan_pattern="daisy", # scanning pattern
                                scan_options=maria_params['scan_options'], # in degrees
                                duration=maria_params['duration'], # integration time in seconds
                                sample_rate=maria_params['sample_rate'], # in Hz
                                scan_center=self.scan_center, # position in the sky
                                start_time=maria_params['start_time'],
                                frame="ra_dec"
                                )

        self.plan.plot()
                
        # self.instrument = nifty_maria.mapsampling_jax.instrument
        # self.instrument = maria.get_instrument('MUSTANG-2')
        self.instrument = self.get_instrument()
        self.instrument.plot()    
        return
    
    def get_instrument(self) -> maria.Instrument:
        """
        Retrieves instrument object. Either AtLAST or MUSTANG-2 for now. Custom objects to be added later.
        """
        
        if self.config == 'mustang': return maria.get_instrument('MUSTANG-2')
        elif self.config == 'test': return maria.get_instrument('MUSTANG-2')
        elif self.config == 'atlast': return self.get_atlast()
    
    def get_atlast(self):
    
        f090 = Band(center=self.maria_params['nu'], # in Hz
                    width=self.maria_params['nu_width'],
                    NET_RJ=self.maria_params['NET_RJ'])

        array = {"field_of_view": self.maria_params['field_of_view'], 
                 "bands": [f090], 
                 "primary_size": 50, 
                 "beam_spacing": self.maria_params['beam_spacing'],  
                 "shape": "circle"} # AtLAST

        # global instrument
        # instrument = maria.get_instrument(array=array, primary_size=50, beam_spacing = 2)
        instrument = maria.get_instrument(array=array)

        return instrument
    
    
    def simulate(self) -> None:
        """
        Performs maria simulation and decorates self with simulation parameters.
        
        Dynamic Attributes (Added by Methods):
            sim_truthmap (Simulation): Simulation object containing instrument, plan, site, input map and parameters for noise, atmosphere and cmb simulation. Added by FitHandler.simulate().
            tod_truthmap (TOD): TOD object containing simulated time-stream data. Added by FitHandler.simulate().
            dx (array): Array with detector offsets in x-direction. Added by FitHandler.simulate().
            dy (array): Array with detector offsets in y-direction. Added by FitHandler.simulate().
        """
        site = maria.get_site(self.maria_params['site'])
        print(site)
        site.plot()

        self.sim_truthmap = maria.Simulation(
            self.instrument, 
            plans=self.plan,
            site= self.maria_params['site'],
            map=self.input_map,
            atmosphere="2d",
        )

        self.tod_truthmap = self.sim_truthmap.run()[0]
        
        # Plot TODs:
        self.tod_truthmap.plot()
        self.offsets = self.sim_truthmap.obs_list[0].coords.offsets(frame=self.sim_truthmap.map.frame,
                                                                    center=(self.sim_truthmap.map.center[0].rad, self.sim_truthmap.map.center[1].rad))
        
        spectrum_kwargs = {
                "spectrum": self.sim_truthmap.obs_list[0].atmosphere.spectrum,
                "zenith_pwv": self.sim_truthmap.obs_list[0].atmosphere.weather.pwv,
                "base_temperature": self.sim_truthmap.obs_list[0].atmosphere.weather.temperature[0],
                "elevation": self.sim_truthmap.obs_list[0].coords.el
        }
        
        band = self.instrument.bands[0] # TODO: should actually vmap over bands!

        import inspect
        print("Band loaded from:", inspect.getfile(Band))

        self.pW_per_K_RJ = 1e12 * k_B * band.compute_nu_integral(**spectrum_kwargs)

        # Ad hoc fix for atmosphere & noise units:
        # self.tod_truthmap.data['atmosphere'] *= self.pW_per_K_RJ
        # self.tod_truthmap.data['noise'] *= self.pW_per_K_RJ
        
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
        
        # mapper_truthmap = BinMapper(
        #     center=self.scan_center,
        #     frame="ra_dec",
        #     # width= 0.1 if self.config == 'mustang' else 1.,
        #     width = self.maria_params['width'],
        #     # height= 0.1 if self.config == 'mustang' else 1.,
        #     height = self.maria_params['width'],
        #     resolution=np.degrees(np.nanmin(self.instrument.dets.fwhm[0]))/4.,
        #     map_postprocessing={"gaussian_filter": {"sigma": 1} }
        # )
        # mapper_truthmap.add_tods(self.tod_truthmap)
        # self.output_truthmap = mapper_truthmap.run()

        mapdata_truth = self.input_map.to(units="K_RJ").data.compute()
        self.mapdata_truth = np.nan_to_num(mapdata_truth, nan=np.nanmean(mapdata_truth)) # replace nan value by img mean
        self.mapdata_truth = np.float64(self.mapdata_truth)
        self.mapdata_truth = self.mapdata_truth.reshape(-1, *self.mapdata_truth.shape[-1:])

        print("mapdata_truth shape:", self.mapdata_truth.shape)
        print("mapdata_truth mean:", self.mapdata_truth.mean())

        # fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # im0 = axes[0].imshow(self.output_truthmap.data[0].T, cmap=cmb_cmap)
        # fig.colorbar(im0)
        # axes[0].title.set_text("Noisy image (Mapper output)")

        # im1 = axes[1].imshow(self.mapdata_truth[0,0], cmap=cmb_cmap)
        # fig.colorbar(im1)
        # axes[1].title.set_text("True Image")

        # if self.plotsdir is None: plt.show()
        # else: 
        #     plt.savefig(f"{self.plotsdir}/reco_maria.png")
        #     plt.close()
        
        # Run proper mapmaker TODO: customize per fit
        if self.config == 'mustang' or self.config == 'test':
            mapper = BinMapper(self.scan_center,
                    frame="ra_dec",
                    width = self.maria_params['width'],
                    # width= 0.1 if self.config == 'mustang' else 1.,
                    height = self.maria_params['width'],
                    # height= 0.1 if self.config == 'mustang' else 1.,
                    resolution=self.instrument.dets.fwhm.deg[0]/4.,
                    tod_preprocessing={
                            "window": {"name": "hamming"},
                            "remove_modes": {"modes_to_remove": [0]},
                            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
                        },
                    map_postprocessing={
                            "gaussian_filter": {"sigma": 1},
                            "median_filter": {"size": 1},
                        },
                    units = "uK_RJ",
                    )
        elif self.config == 'atlast': # TODO: optimise!
            mapper = BinMapper(self.scan_center,
                    frame="ra_dec",
                    width=self.maria_params['width'],
                    height=self.maria_params['width'],
                    resolution=np.degrees(np.nanmin(self.instrument.dets.fwhm[0]))/4.,
                    tod_preprocessing={
                            # "window": {"name": "hamming"},
                            "window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
                            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
                            "remove_modes": {"modes_to_remove": [0]},
                        },
                        map_postprocessing={
                            "gaussian_filter": {"sigma": 1},
                            "median_filter": {"size": 1},
                        },
                        units = "uK_RJ",
                    )
        
        mapper.add_tods(self.tod_truthmap)
        self.output_map = mapper.run()
        
        # self.output_map.plot(filename=f"{self.plotsdir}/reco_maria.png")
        self.output_map.plot()
        plt.savefig(fname=f"{self.plotsdir}/reco_maria.png")
        
        return