"""
Module to colloct plotting functions for nifty-maria fits.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nifty8.re as jft
from nifty8.re.optimize_kl import OptimizeVIState
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from nifty_maria.mapsampling_jax import sample_maps

class Plotter:
    """Subclass containing plotting functionalities."""
    def callback(self, samples: jft.evi.Samples, opt_state: OptimizeVIState) -> None:
        """
        Callback function to be used for plotting fit status during optimisation.
        
        Args:
            samples (jft.evi.Samples): Samples to perform plots for.
            opt_state (jft.optimize_kl.OptimizeVIState): Optimisation state to plot.
        """
        cmb_cmap = plt.get_cmap('cmb')
        
        iter = opt_state[0]
        n = self.noised_jax_tod.shape[0]
        if iter % self.printevery != 0: return

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
        if self.plotsdir is None: plt.show()
        else:
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_iter_{iter}_TODagreement.png")
            plt.close()

        if self.fit_atmos:
            fig_tods, axes_tods = plt.subplots(2, 1, figsize=(16, 6))

            preds = []
            for x in samples:
                x_tod = {k: x[k] for k in x if 'comb' in k}
                res_tods = self.gp_tod(x_tod)

                preds += [res_tods[:, self.padding_atmos//2:-self.padding_atmos//2], ]

            mean_atmos, std = jft.mean_and_std(tuple(preds))

            for i in range(0, n, n//10 if n//10 != 0 else 1):
                axes_tods[0].plot(np.arange(0, mean_atmos.shape[1]*self.downsampling_factor), jnp.repeat(mean_atmos[i], self.downsampling_factor), label=f"tod{i}")
                axes_tods[0].plot(self.atmos_tod_simplified[i], label=f"truth{i}")
                axes_tods[1].plot(np.arange(0, mean_atmos.shape[1]*self.downsampling_factor), jnp.repeat(mean_atmos[i], self.downsampling_factor) - jnp.array(self.atmos_tod_simplified[i]), label=f"tod{i}")

            fig_tods.suptitle(f"n_sub = {self.n_sub}, iter: {iter}")
            axes_tods[0].title.set_text('mean atmos pred. & simplified truth (no noise)')
            axes_tods[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            axes_tods[1].title.set_text('mean atmos pred. - simplified truth (no noise)')
            
            if self.plotsdir is None: plt.show()
            else:
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_iter_{iter}_simplified_atmos.png")
                plt.close()

        if self.fit_map:
            fig_map, axes_map = plt.subplots(1, 3, figsize=(16, 6))

            if self.padding_map > 0:
                mean_map, _ = jft.mean_and_std(tuple(self.gp_map(s)[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in samples))
            else:
                mean_map, _ = jft.mean_and_std(tuple(self.gp_map(s) for s in samples))

            im0 = axes_map[0].imshow(mean_map, cmap=cmb_cmap)
            axes_map[0].title.set_text('mean map pred.')
            fig_map.colorbar(im0)

            im1 = axes_map[1].imshow(mean_map - self.mapdata_truth, cmap=cmb_cmap)
            axes_map[1].title.set_text('mean map - truth')
            fig_map.colorbar(im1)

            im2 = axes_map[2].imshow(self.mapdata_truth, cmap=cmb_cmap)
            axes_map[2].title.set_text('truth')
            fig_map.colorbar(im2)

            fig_map.suptitle(f"n_sub = {self.n_sub}, iter: {iter}")
        
            if self.plotsdir is None: plt.show()
            else:
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_iter_{iter}_map.png")
                plt.close()
        
        return
        
    def plotfitresults(self, samples: jft.evi.Samples) -> None:
        """
        Plots predictions made by optimised GP and compares with truth.
        
        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
        """
        cmb_cmap = plt.get_cmap('cmb')
        
        res = jft.mean(tuple(self.signal_response_tod(s) for s in samples))
        n = self.noised_jax_tod.shape[0]

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

        if self.plotsdir is None: plt.show()
        else:
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_TODagreement_final.png")
            plt.close()
        
        if self.fit_map: 
            # plot maximum of posterior (mode)
            if self.padding_map > 0:
                sig_map = jft.mean(tuple(self.gp_map(s)[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in samples))
            else:
                sig_map = jft.mean(tuple(self.gp_map(s) for s in samples))

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            im0 = axes[0].imshow(sig_map, cmap=cmb_cmap)
            axes[0].title.set_text('MAP - best fit image')
            fig.colorbar(im0)

            im1 = axes[1].imshow( sig_map - self.mapdata_truth, cmap=cmb_cmap)
            axes[1].title.set_text('MAP - map truth')
            fig.colorbar(im1)

            fig.suptitle(f"n_sub = {self.n_sub}")

            if self.plotsdir is None: plt.show()
            else:
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_mapagreement_final.png")
                plt.close()
        
        return 
    
    def plotsamples(self, samples: jft.evi.Samples) -> None:
        """
        Plots samples of the optimised GP
        
        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 8))

        for k in range(0, 3):
            idxk = np.random.randint(0, self.noised_jax_tod.shape[0])
            todk = []
            for i in range(len(samples)):
                ti = self.signal_response_tod(samples[i])
                axes[k].plot(np.arange(0, ti.shape[1]), ti[idxk], label=f"tod {idxk} - sample {i}", c='gray')
                todk.append(ti)

            tmean = jft.mean(todk)
            axes[k].plot(np.arange(0, tmean.shape[1]), tmean[idxk], label=f"tod {idxk} - mean", c='red')

            axes[k].title.set_text(f'tod {idxk} - samples and mean')
            axes[k].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.suptitle(f"n_sub = {self.n_sub}")

        if self.plotsdir is None: plt.show()
        else:
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_samples_tod.png")
            plt.close()

        if self.fit_map: 
            cmb_cmap = plt.get_cmap('cmb')

            fig, axes = plt.subplots(1, len(samples), figsize=(16, 8))

            sig_maps = tuple(self.gp_map(s) for s in samples)
            if self.padding_map > 0:
                sig_maps = tuple(s[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in sig_maps)
            
            vmin = min(s.min() for s in sig_maps)
            vmax = max(s.max() for s in sig_maps)

            for i,s in enumerate(sig_maps):
                im = axes[i].imshow(s, cmap=cmb_cmap, vmin=vmin, vmax=vmax)
                axes[i].title.set_text(f'signal map - sample {i}')

                div = make_axes_locatable(axes[i])
                cax = div.append_axes('right', size='3%', pad='2%')
                fig.colorbar(im, cax)

            fig.tight_layout()
            fig.suptitle(f"n_sub = {self.n_sub}")

            if self.plotsdir is None: plt.show()
            else:
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_samples_map.png")
                plt.close()
        
        return 
    
    def plotpowerspectrum(self, samples: jft.evi.Samples) -> None:
        """
        Plots power spectrum of predictions made by optimised GP and compares with truth.
        
        Args:
            samples (jft.evi.Samples): Samples to plot power spectrum for.
        """
        import scipy as sp
        from itertools import cycle

        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        components = [jft.mean(tuple(self.signal_response_tod(s) for s in samples))]
        labels = ['pred. total']
        linestyles = ['-']

        if self.fit_map:
            if self.padding_map > 0:
                # gp_map_nopad = jnp.broadcast_to(jft.mean(tuple(self.gp_map(s) for s in samples)), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
                gp_map_nopad = jft.mean(tuple(self.gp_map(s) for s in samples))[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
            else:
                # gp_map_nopad = jnp.broadcast_to(jft.mean(tuple(self.gp_map(s) for s in samples)), (1, 1, self.dims_map[0], self.dims_map[1]))
                gp_map_nopad = jft.mean(tuple(self.gp_map(s) for s in samples))

            # res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
            res_map = sample_maps(gp_map_nopad, self.instrument, self.offsets, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side, self.pW_per_K_RJ)
            
            components += [res_map, self.tod_truthmap.data['map']]
            labels += ['pred. map', 'true map']
            linestyles += ['-', '--']

        if self.fit_atmos:
            
            res_tods = ()
            for s in samples:
                x_tod = {k: s[k] for k in s if 'comb' in k}
                res_tod = self.gp_tod(x_tod)[:, self.padding_atmos//2:-self.padding_atmos//2]
                res_tod = jnp.repeat(res_tod, self.downsampling_factor)[None, :]
                res_tods += (res_tod,)
                
            res_tods = jft.mean(res_tods)
            
            components += [res_tods, self.tod_truthmap.data['atmosphere']]
            labels += ['pred. atmos', 'true atmos']
            linestyles += ['-', '--']

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
        
        if self.plotsdir is None: plt.show()
        else:
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_powerspectrum.png")
            plt.close()
        
        return
    
    def plotrecos(self, samples: jft.evi.Samples) -> None:
        """
        Plots comparison between maria and nifty reconstructions of the map with the true map.
        
        Args:
            samples (jft.evi.Samples): Samples to make comparison for.
        """
        from skimage.transform import resize

        # Compare nifty vs maria
        if self.padding_map > 0:
            sig_map = jft.mean(tuple(self.gp_map(s)[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in samples)) # when splitting up in different field models
        else:
            sig_map = jft.mean(tuple(self.gp_map(s) for s in samples))

        # mincol = -0.0012
        # maxcol = 0.
        mincol = None
        maxcol = None

        cmb_cmap = plt.get_cmap('cmb')
        fig, axes = plt.subplots(3, 2, figsize=(16, 16))

        im0 = axes[0,0].imshow( self.mapdata_truth , cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[0,0].title.set_text('truth')
        fig.colorbar(im0)

        # im1 = axes[0,1].imshow(self.output_truthmap.data[0,0], cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        # fig.colorbar(im1)
        # axes[0,1].title.set_text("Noisy image (Mapper output)")

        slice_2d = self.output_map.data[(0,) * (self.output_map.data.ndim - 2) + (...,)]
        im2 = axes[1,0].imshow(slice_2d, cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[1,0].title.set_text('maria mapper')
        fig.colorbar(im2)

        truth_rescaled = resize(self.mapdata_truth, slice_2d.shape, anti_aliasing=True)
        im3 = axes[1,1].imshow((slice_2d - truth_rescaled), cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[1,1].title.set_text('maria - truth')
        fig.colorbar(im3)

        im3 = axes[2,0].imshow(sig_map, cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[2,0].title.set_text('best fit image')
        fig.colorbar(im3)

        im4 = axes[2,1].imshow((sig_map - self.mapdata_truth), cmap=cmb_cmap, vmin=mincol, vmax=maxcol)
        axes[2,1].title.set_text('best fit - truth')
        fig.colorbar(im4)

        if self.plotsdir is None: plt.show()
        else:
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_reco_comp.png")
            plt.close()
        
        return 
        
    def make_atmosphere_det_gif(self, samples: jft.evi.Samples, figname: str = 'atmosphere_comp.gif', tmax: int = -1, num_frames: int = 100) -> None:
        """
        Makes gif of simplified atmosphere prediction and truth in 2D detector layout. Does nothing if self.fit_atmos == False.
        
        Args:
            samples (jft.evi.Samples): Samples to make atmosphere prediction plot for.
            figname (str, optional): Location to save gif in. Defaults to 'atmosphere_comp.gif'
            tmax (int, optional): Maximum timestep to consider. If -1, will loop over all timesteps. Defaults to -1.
            num_frames (int, optional): Number of total frames to plot. Defaults to 100.
        """
        
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
        if self.plotsdir is None: frames[0].save(figname, save_all=True, append_images=frames[1:], duration=1, loop=0)
        else: frames[0].save(f"{self.plotsdir}/{figname}", save_all=True, append_images=frames[1:], duration=1, loop=0)
        
        return 
    
    def plot_atmosphere_det(self, samples: jft.evi.Samples, timestep: int = 0, z: float = np.inf) -> plt.Figure:
        """
        Plots simplified atmosphere prediction and truth in 2D detector layout. Returns figure. Does nothing if self.fit_atmosphere == False.
        
        Args:
            samples (jft.evi.Samples): Samples to make atmosphere prediction plot for.
            timestep (int, optional): Timestep to make plot for.
            z (float, optional): Gaussian beam distance in instrument. Defaults to np.inf.
        
        Returns:
            plt.Figure: The produced figure object.
        
        """
        
        if not self.fit_atmos:
            print("Not fitting atmosphere, skipping plot..")
            return 
        
        from maria.units import Angle

        cmb_cmap = plt.get_cmap('cmb')
        
        best_fit_atmos = ()
        for s in samples:
            x_tod = {k: s[k] for k in s if 'comb' in k}
            best_fit_atmos += (self.gp_tod(x_tod)[:, self.padding_atmos//2:-self.padding_atmos//2], )
            
        best_fit_atmos = jft.mean(best_fit_atmos)

        test = Angle(self.instrument.dets.offsets)
        pos = getattr(test, test.units).T

        col = np.zeros(pos[0].shape)
        
        if self.n_split == -1:
            col = best_fit_atmos[:, timestep]
        else:
            for i in range(len(self.masklist)):
                col[self.masklist[i]] = best_fit_atmos[i, timestep]

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
        
        return fig
    
    def plot_subdets(self, z: float = np.inf) -> None:
        """
        Plots detector with n_sub subdetectors highlighted in color.
        
        Args:
            z (float, optional): Gaussian beam distance in instrument. Defaults to np.inf.

        Raises:
            ValueError: If invalid n_sub value is supplied.
        """
        from maria.units import Angle

        instrument = self.instrument

        cmb_cmap = plt.get_cmap('cmb')

        test = Angle(instrument.dets.offsets)
        pos = getattr(test, test.units).T
        
        col = np.zeros(pos[0].shape)
        if self.n_sub > 0:
            for i in range(len(self.masklist)):
                col[self.masklist[i]] = i+1
        else: col = np.linspace(0, 1, self.instrument.n_dets)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=160)

        self.plot_instrument(fig, ax, col, cmb_cmap, z=z)

        fig.suptitle(f"n_sub = {self.n_sub}")

        if self.plotsdir is None: plt.show()
        else: 
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_detector.png")
            plt.close()
        
        return 
    
    def plot_instrument(self, fig: plt.Figure, ax: plt.Axes, col: np.ndarray, cmb_cmap: plt.Colormap, z: float = np.inf) -> None:
        """
        Plots detector with a given color values col into figure fig and axis ax.
        
        Args:
            fig (plt.Figure): Figure to plot into.
            ax (plt.Axes): Axes to plot into.
            col (np.ndarray): Numpy array containing colors to be plotted.
            cmb_cmap (plt.Colormap): Colormap to use for plotting.
            z (float, optional): Gaussian beam distance in instrument. Defaults to np.inf.
        """
        from matplotlib.collections import EllipseCollection
        from maria.units import Angle

        fwhms = Angle(self.instrument.dets.angular_fwhm(z=z))
        offsets = Angle(self.instrument.dets.offsets)

        i = 0

        for ia, array in enumerate(self.instrument.arrays):
            array_mask = self.instrument.dets.array_name == array.name

            for ib, band in enumerate(array.bands):
                band_mask = self.instrument.dets.band_name == band.name
                mask = array_mask & band_mask

                collection = EllipseCollection(
                        widths=getattr(fwhms, offsets.units)[mask],
                        heights=getattr(fwhms, offsets.units)[mask],
                        angles=0,
                        units="xy",
                        edgecolors="k",
                        lw=1e-1,
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
                    s=2.0*60 if self.config == 'mustang' else 2.0,
                    c=col,
                    cmap=cmb_cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                i += 1

        fig.colorbar(scatter)
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