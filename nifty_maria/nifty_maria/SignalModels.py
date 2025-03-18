'''
Module collecting nifty signal models used in maria fits.
'''
import nifty8.re as jft
import jax
import jax.numpy as jnp

from nifty_maria.mapsampling_jax import sample_maps

# Signal Model (map + atmos) for all subdets:
class Signal_TOD_alldets(jft.Model):
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

        # super().__init__(init = self.gp_tod.init | self.gp_map.init | self.offset_tod.init,
        #                  domain = self.gp_tod.domain | self.gp_map.domain | self.offset_tod.domain)
        super().__init__(init = self.gp_tod.init | self.gp_map.init,
                        domain = self.gp_tod.domain | self.gp_map.domain )


    def __call__(self, x):    
        x_tod = {k: x[k] for k in x if 'comb' in k}
        res_tods = self.gp_tod(x_tod)
        # res_tods = self.gp_tod(x['tod'])

        res_tods = res_tods[:, self.padding_atmos//2:-self.padding_atmos//2]

        # Repeat gps for nested circle subdets:
        # res_tods_fulldet = jax.numpy.repeat(res_tods_nopad, -(-self.slopes_truth.shape[0]//res_tods_nopad.shape[0]), axis=0)[:self.slopes_truth.shape[0]]

        # For half-circles use mask:
        # res_tods_fulldet = jax.numpy.zeros( (self.slopes_truth.shape[0], res_tods.shape[1]) )
        # res_tods_fulldet = res_tods_fulldet.at[posmask_up].set( res_tods[0] )
        # res_tods_fulldet = res_tods_fulldet.at[posmask_down].set( res_tods[-1] )
        res_tods_fulldet = res_tods

        # From TOD-only fit:
        # res_tods_offset = res_tods_fulldet * self.slopes_truth + self.offset_tod_truth
        res_tods_offset = jnp.repeat(res_tods_fulldet, self.downsampling_factor, axis=1) * self.slopes_truth + self.offset_tod_truth

        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, dims_map[0], dims_map[1]))[:, padding_map//2:-padding_map//2, padding_map//2:-padding_map//2], dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, 1, dims_map[0], dims_map[1])), dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        # modified_res_map = res_map.at[:n, :].add(res_tods_offset)
        modified_res_map = res_map + res_tods_offset

        return modified_res_map

# Signal Model (atmos only) for 0th subdet:
class Signal_TOD_atmos(jft.Model):
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
        # res_tods = self.gp_tod(x['tod'])

        res_tods = res_tods[:, self.padding_atmos//2:-self.padding_atmos//2]
        res_tods_fulldet = res_tods

        # From TOD-only fit:
        # res_tods_offset = res_tods_fulldet * self.slopes_truth + self.offset_tod_truth
        res_tods_offset = jnp.repeat(res_tods_fulldet, self.downsampling_factor, axis=1) * self.slopes_truth + self.offset_tod_truth

        return res_tods_offset
    
# Signal Model (map only) for all subdets:
class Signal_TOD_alldets_maponly(jft.Model):
    def __init__(self, gp_map, dims_map, padding_map, sim_truthmap, dx, dy):
        # self.gp_tod = gp_tod
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

# Signal Model (map + atmos) for masked subdets:
class Signal_TOD_general(jft.Model):
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
        ## Opt1: set (only works in for loop & with masklist as list)
        # res_tods_fulldet = jnp.zeros( (self.slopes_truth.shape[0], res_tods.shape[1]) )
        # for i in range(len(self.masklist)):
        #     res_tods_fulldet = res_tods_fulldet.at[self.masklist[i]].set( res_tods[i] )
        # res_tods_fulldet = res_tods_fulldet.at[self.masklist[0]].set( res_tods[0] )
        # res_tods_fulldet = res_tods_fulldet.at[self.masklist[-1]].set( res_tods[-1] )

        ## Opt2: where in loop
        # res_tods_fulldet = jnp.zeros( (self.slopes_truth.shape[0], res_tods.shape[1]) )
        # res_tods_fulldet_loop = jnp.zeros( (self.slopes_truth.shape[0], res_tods.shape[1]) )
        # for i in range(len(self.masklist)):
        #     res_tods_fulldet_loop = jnp.where(self.masklist[i][:, None], res_tods[i], res_tods_fulldet_loop)

        ## Opt3: Try vectorizing Opt2 (not numerically identical since order is different):
        # Define the operation that replaces the for loop using jax.vmap
        # def apply_mask(mask, res_tod):
        #     print("mask.shape, res_tod.shape:", mask.shape, res_tod.shape)
        #     # return jnp.where(mask[:, None], res_tod, jnp.zeros_like(res_tod))
        #     return mask[:, None] * res_tod
        #     # return jax.lax.select(mask[:, None], res_tod, jnp.zeros_like(res_tod))
        #     # return jax.lax.dot(mask, res_tod) 

        # Vectorize the operation over the masklist and res_tods
        # print("pre-vmap dims: self.masklist, res_tods", self.masklist.shape, res_tods.shape)
        # res_tods_fulldet = jax.vmap(apply_mask, in_axes=(0, 0))(self.masklist, res_tods)
        # # Sum the results across the first axis
        # res_tods_fulldet = jnp.sum(res_tods_fulldet, axis=0)
        
        # sum over masklist and multiply with res_dots using einsum:
        res_tods_fulldet = jnp.einsum("ai,aj->ij", self.masklist, res_tods)
        # print("post einsum shape:", res_tods_fulldet.shape)

        # Correct tods by offset and slope
        # res_tods_offset = res_tods_fulldet * self.slopes_truth + self.offset_tod_truth
        res_tods_offset = jnp.repeat(res_tods_fulldet, self.downsampling_factor, axis=1) * self.slopes_truth + self.offset_tod_truth

        # Sample map and add
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        modified_res_map = res_map + res_tods_offset

        return modified_res_map