'''
Module collecting nifty signal models used in maria fits.
'''
import nifty8.re as jft
import jax.numpy as jnp

from nifty_maria.mapsampling_jax import sample_maps

# Signal model for 1 or 2 subdets
class Signal_TOD_combined(jft.Model):
    def __init__(self, gp_tod, offset_tod_truth, slopes_truth, gp_map, dims_map, padding_map, dims_atmos, padding_atmos, posmask_up, posmask_down, sim_truthmap, dx, dy):
        self.gp_tod = gp_tod
        self.gp_map = gp_map
        self.offset_tod_truth = offset_tod_truth[:, None]
        self.slopes_truth = slopes_truth[:, None]

        self.dims_map = dims_map
        self.padding_map = padding_map
        self.dims_atmos = dims_atmos
        self.padding_atmos = padding_atmos
        self.posmask_up = posmask_up
        self.posmask_down = posmask_down
        
        self.sim_truthmap = sim_truthmap
        self.dx = dx
        self.dy = dy

        # super().__init__(init = self.gp_tod.init | self.gp_map.init | self.offset_tod.init,
        #                  domain = self.gp_tod.domain | self.gp_map.domain | self.offset_tod.domain)
        super().__init__(init = self.gp_tod.init | self.gp_map.init,
                        domain = self.gp_tod.domain | self.gp_map.domain )


    def __call__(self, x):
        x_tod = {k: x[k] for k in x if 'comb' in k}
        res_tods = self.gp_tod(x_tod)
        # res_tods = self.gp_tod(x['tod'])

        res_tods = res_tods[:, self.padding_atmos//2:-self.padding_atmos//2]

        # For half-circles use mask:
        # res_tods_fulldet = jnp.zeros( (self.slopes_truth.shape[0], res_tods.shape[1]) )
        # res_tods_fulldet = res_tods_fulldet.at[self.posmask_up].set( res_tods[0] )
        # res_tods_fulldet = res_tods_fulldet.at[self.posmask_down].set( res_tods[-1] )

        # From TOD-only fit:
        # res_tods_offset = res_tods_fulldet * self.slopes_truth + self.offset_tod_truth
        res_tods_offset = res_tods * self.slopes_truth + self.offset_tod_truth

        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, dims_map[0], dims_map[1]))[:, padding_map//2:-padding_map//2, padding_map//2:-padding_map//2], dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, 1, dims_map[0], dims_map[1])), dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        # gp_map_nopad = jax.numpy.broadcast_to(self.gp_map(x), (1, 1, dims_map[0], dims_map[1]))
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        # modified_res_map = res_map.at[:n, :].add(res_tods_offset)
        modified_res_map = res_map + res_tods_offset

        return modified_res_map

class Signal_TOD_atmosonly(jft.Model):
    def __init__(self, gp_tod, offset_tod_truth, slopes_truth, dims_atmos, padding_atmos, posmask_up, posmask_down):
        self.gp_tod = gp_tod
        self.offset_tod_truth = offset_tod_truth[:, None]
        self.slopes_truth = slopes_truth[:, None]

        self.dims_atmos = dims_atmos
        self.padding_atmos = padding_atmos
        self.posmask_up = posmask_up
        self.posmask_down = posmask_down
        
        super().__init__(init = self.gp_tod.init,
                        domain = self.gp_tod.domain)


    def __call__(self, x):
        x_tod = {k: x[k] for k in x if 'comb' in k}
        res_tods = self.gp_tod(x_tod)
        # res_tods = self.gp_tod(x['tod'])

        res_tods = res_tods[:, self.padding_atmos//2:-self.padding_atmos//2]

        res_tods_offset = res_tods * self.slopes_truth + self.offset_tod_truth

        modified_res_map = res_tods_offset

        return modified_res_map
class Signal_TOD_combined_nomappadding(jft.Model):
    def __init__(self, gp_tod, offset_tod_truth, slopes_truth, gp_map, dims_map, dims_atmos, padding_atmos, posmask_up, posmask_down, sim_truthmap, dx, dy):
        self.gp_tod = gp_tod
        self.gp_map = gp_map
        self.offset_tod_truth = offset_tod_truth[:, None]
        self.slopes_truth = slopes_truth[:, None]

        self.dims_map = dims_map
        self.dims_atmos = dims_atmos
        self.padding_atmos = padding_atmos
        self.posmask_up = posmask_up
        self.posmask_down = posmask_down
        
        self.sim_truthmap = sim_truthmap
        self.dx = dx
        self.dy = dy

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

        # For half-circles use mask: TODO: fix for 1/2 subdets!
        # res_tods_fulldet = jnp.zeros( (self.slopes_truth.shape[0], res_tods.shape[1]) )
        # res_tods_fulldet = res_tods_fulldet.at[self.posmask_up].set( res_tods[0] )
        # res_tods_fulldet = res_tods_fulldet.at[self.posmask_down].set( res_tods[-1] )

        # From TOD-only fit:
        # res_tods_offset = res_tods_fulldet * self.slopes_truth + self.offset_tod_truth
        res_tods_offset = res_tods * self.slopes_truth + self.offset_tod_truth
        # res_tods_offset = res_tods_fulldet

        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, dims_map[0], dims_map[1]))[:, padding_map//2:-padding_map//2, padding_map//2:-padding_map//2], dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, 1, dims_map[0], dims_map[1])), dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        # gp_map_nopad = jax.numpy.broadcast_to(self.gp_map(x), (1, 1, dims_map[0], dims_map[1]))
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        # modified_res_map = res_map.at[:n, :].add(res_tods_offset)
        modified_res_map = res_map + res_tods_offset

        return modified_res_map
    
class Signal_TOD_maponly_nomappadding(jft.Model):
    def __init__(self, gp_map, dims_map, sim_truthmap, dx, dy):
        self.gp_map = gp_map

        self.dims_map = dims_map
        
        self.sim_truthmap = sim_truthmap
        self.dx = dx
        self.dy = dy

        super().__init__(init = self.gp_map.init, domain = self.gp_map.domain )

    def __call__(self, x):
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)

        return res_map

# Signal model for four subdets:
class Signal_TOD_combined_fourTODs(jft.Model):
    def __init__(self, gp_tod, offset_tod_truth, slopes_truth, gp_map, dims_map, padding_map, dims_atmos, padding_atmos, posmask_ul, posmask_ur, posmask_dl, posmask_dr, sim_truthmap, dx, dy):
        self.gp_tod = gp_tod
        self.gp_map = gp_map
        self.offset_tod_truth = offset_tod_truth[:, None]
        self.slopes_truth = slopes_truth[:, None]

        self.dims_map = dims_map
        self.padding_map = padding_map
        self.dims_atmos = dims_atmos
        self.padding_atmos = padding_atmos
        self.posmask_ul = posmask_ul
        self.posmask_ur = posmask_ur
        self.posmask_dl = posmask_dl
        self.posmask_dr = posmask_dr
        
        self.sim_truthmap = sim_truthmap
        self.dx = dx
        self.dy = dy

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
        res_tods_fulldet = jnp.zeros( (self.slopes_truth.shape[0], res_tods.shape[1]) )
        res_tods_fulldet = res_tods_fulldet.at[self.posmask_ul].set( res_tods[0] )
        res_tods_fulldet = res_tods_fulldet.at[self.posmask_ur].set( res_tods[1] )
        res_tods_fulldet = res_tods_fulldet.at[self.posmask_dl].set( res_tods[2] )
        res_tods_fulldet = res_tods_fulldet.at[self.posmask_dr].set( res_tods[3] )

        # From TOD-only fit:
        res_tods_offset = res_tods_fulldet * self.slopes_truth + self.offset_tod_truth
        # res_tods_offset = res_tods_fulldet

        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, dims_map[0], dims_map[1]))[:, padding_map//2:-padding_map//2, padding_map//2:-padding_map//2], dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, 1, dims_map[0], dims_map[1])), dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        # gp_map_nopad = jax.numpy.broadcast_to(self.gp_map(x), (1, 1, dims_map[0], dims_map[1]))
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        # modified_res_map = res_map.at[:n, :].add(res_tods_offset)
        modified_res_map = res_map + res_tods_offset

        return modified_res_map

# Signal Model for all subdets:
class Signal_TOD(jft.Model):
    def __init__(self, gp_tod, offset_tod_truth, slopes_truth, gp_map, dims_map, padding_map, dims_atmos, padding_atmos, sim_truthmap, dx, dy):
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
        res_tods_offset = res_tods_fulldet * self.slopes_truth + self.offset_tod_truth
        # res_tods_offset = res_tods_fulldet

        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, dims_map[0], dims_map[1]))[:, padding_map//2:-padding_map//2, padding_map//2:-padding_map//2], dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        # res_map = sample_maps(jax.numpy.broadcast_to(self.gp_map(x), (1, 1, dims_map[0], dims_map[1])), dx, dy, sim_truthmap.map.resolution, sim_truthmap.map.x_side, sim_truthmap.map.y_side)
        gp_map_nopad = jnp.broadcast_to(self.gp_map(x), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
        res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
        # modified_res_map = res_map.at[:n, :].add(res_tods_offset)
        modified_res_map = res_map + res_tods_offset

        return modified_res_map