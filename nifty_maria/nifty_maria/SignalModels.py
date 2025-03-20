"""
Module collecting nifty signal models used in maria fits.
"""
import nifty8.re as jft
import jax.numpy as jnp

from nifty_maria.mapsampling_jax import sample_maps

class Signal_TOD_alldets(jft.Model):
    """Signal Model (map + atmos) for all subdets"""
    def __init__(self, gp_tod, offset_tod_truth, slopes_truth, gp_map, dims_map, padding_map, dims_atmos, padding_atmos, sim_truthmap, dx, dy, downsampling_factor):
        self.gp_tod = gp_tod
        self.gp_map = gp_map
        self.offset_tod_truth = offset_tod_truth[:, None]
        self.slopes_truth = slopes_truth[:, None]

        self.dims_map = dims_map
        self.padding_map = padding_map
        self.dims_atmos = dims_atmos
        self.padding_atmos = padding_atmos
        
        self.sim_truthmap = sim_truthmap
        self.dx = dx
        self.dy = dy
        
        self.downsampling_factor = downsampling_factor

        super().__init__(init = self.gp_tod.init | self.gp_map.init,
                        domain = self.gp_tod.domain | self.gp_map.domain )


    def __call__(self, x):    
        x_tod = {k: x[k] for k in x if 'comb' in k}
        res_tods = self.gp_tod(x_tod)

        res_tods = res_tods[:, self.padding_atmos//2:-self.padding_atmos//2]
        res_tods_fulldet = res_tods

        # From TOD-only fit:
        res_tods_offset = jnp.repeat(res_tods_fulldet, self.downsampling_factor, axis=1) * self.slopes_truth + self.offset_tod_truth

        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        modified_res_map = res_map + res_tods_offset

        return modified_res_map

class Signal_TOD_atmos(jft.Model):
    """Signal Model (atmos only) for 0th subdet"""
    def __init__(self, gp_tod, offset_tod_truth, slopes_truth, dims_atmos, padding_atmos, downsampling_factor):
        self.gp_tod = gp_tod
        self.offset_tod_truth = offset_tod_truth[:, None]
        self.slopes_truth = slopes_truth[:, None]

        self.dims_atmos = dims_atmos
        self.padding_atmos = padding_atmos
        
        self.downsampling_factor = downsampling_factor
        
        super().__init__(init = self.gp_tod.init,
                        domain = self.gp_tod.domain )


    def __call__(self, x):    
        x_tod = {k: x[k] for k in x if 'comb' in k}
        res_tods = self.gp_tod(x_tod)

        res_tods = res_tods[:, self.padding_atmos//2:-self.padding_atmos//2]
        res_tods_fulldet = res_tods

        # From TOD-only fit:
        res_tods_offset = jnp.repeat(res_tods_fulldet, self.downsampling_factor, axis=1) * self.slopes_truth + self.offset_tod_truth

        return res_tods_offset
    
class Signal_TOD_alldets_maponly(jft.Model):
    """Signal Model (map only) for all subdets"""
    def __init__(self, gp_map, dims_map, padding_map, sim_truthmap, dx, dy):
        self.gp_map = gp_map

        self.dims_map = dims_map
        self.padding_map = padding_map
        
        self.sim_truthmap = sim_truthmap
        self.dx = dx
        self.dy = dy

        super().__init__(init = self.gp_map.init,
                        domain = self.gp_map.domain )

    def __call__(self, x):    
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        modified_res_map = res_map

        return modified_res_map

class Signal_TOD_general(jft.Model):
    """Signal Model (map + atmos) for masked subdets"""
    def __init__(self, gp_tod, offset_tod_truth, slopes_truth, gp_map, dims_map, padding_map, dims_atmos, padding_atmos, masklist, sim_truthmap, dx, dy, downsampling_factor):
        self.gp_tod = gp_tod
        self.gp_map = gp_map
        self.offset_tod_truth = offset_tod_truth[:, None]
        self.slopes_truth = slopes_truth[:, None]

        self.dims_map = dims_map
        self.padding_map = padding_map
        self.dims_atmos = dims_atmos
        self.padding_atmos = padding_atmos
        self.masklist = masklist
        
        self.sim_truthmap = sim_truthmap
        self.dx = dx
        self.dy = dy
        
        self.downsampling_factor = downsampling_factor

        super().__init__(init = self.gp_tod.init | self.gp_map.init,
                        domain = self.gp_tod.domain | self.gp_map.domain )


    def __call__(self, x):
        x_tod = {k: x[k] for k in x if 'comb' in k}
        res_tods = self.gp_tod(x_tod)

        res_tods = res_tods[:, self.padding_atmos//2:-self.padding_atmos//2]
        
        # sum over masklist and multiply with res_dots using einsum:
        res_tods_fulldet = jnp.einsum("ai,aj->ij", self.masklist, res_tods)

        # Correct (upsampled) tods by offset and slope
        res_tods_offset = jnp.repeat(res_tods_fulldet, self.downsampling_factor, axis=1) * self.slopes_truth + self.offset_tod_truth

        # Sample map and add
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        modified_res_map = res_map + res_tods_offset

        return modified_res_map