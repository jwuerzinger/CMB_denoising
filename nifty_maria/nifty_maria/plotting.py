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

    def plot_callback(self, samples: jft.evi.Samples, opt_state: OptimizeVIState) -> None:
        """
        Callback function to be used for plotting fit status during optimisation.
        
        Args:
            samples (jft.evi.Samples): Samples to perform plots for.
            opt_state (OptimizeVIState): Optimisation state to plot.
        """
        self.plot_tod_agreement(samples, opt_state)
        self.plot_map_agreement(samples, opt_state)
        self.plot_atmos_simplified(samples, opt_state)
        return

    def plot_results(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Outputs several plots.

        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        self.plot_tod_agreement(samples, opt_state)
        self.plot_map_agreement(samples, opt_state)
        self.plot_tod_samples(samples, opt_state)
        self.plot_map_samples(samples, opt_state)
        self.plot_map_pullplot(samples, opt_state)
        self.plot_map_comparison(samples, opt_state)
        self.plot_atmos_simplified(samples, opt_state)
        self.plot_power_spectrum(samples, opt_state)
        return

    def plot_tod_agreement(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Plot comparison of the optimised TODs with the truth.

        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        res = jft.mean(tuple(self.signal_response_tod(s) for s in samples))

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        n_tods = 10
        for i in np.linspace(0, self.noised_jax_tod.shape[0]-1, n_tods, dtype=int):
            axes[0].plot(np.arange(0, res.shape[1]), res[i], label=f"tod {i}")
            axes[1].plot(np.arange(0, res.shape[1]), res[i] - self.noised_jax_tod[i], label=f"tod {i}")
            axes[2].plot(self.noised_jax_tod[i], label=f"tod {i}")

        axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        axes[0].title.set_text("tods: mean")
        axes[1].title.set_text("tods: mean - truth")
        axes[2].title.set_text("tods: truth")

        for ax in axes:
            ax.set_xlim(0, res.shape[1])
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        name = f", iter: {opt_state.nit}" if opt_state is not None else ""
        fig.suptitle(f"n_sub = {self.n_sub}{name}")

        if self.plotsdir is None: 
            plt.show()
        else:
            name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_tod_agreement.png")
            plt.close()
        return
    
    def plot_map_agreement(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Plot comparison of the optimised maps with the truth.

        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        if self.fit_map: 
            cmb_cmap = plt.get_cmap("cmb")

            sig_maps = tuple(self.gp_map(s) for s in samples)
            if self.padding_map > 0:
                sig_maps = tuple(s[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in sig_maps)
            sig_mean = jft.mean(sig_maps)

            images = (sig_mean, sig_mean - self.mapdata_truth[0, 0], self.mapdata_truth[0, 0])
            titles = ("map: mean", "map: mean - truth", "map: truth")

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for i in range(3):
                im = axes[i].imshow(images[i], cmap=cmb_cmap)
                axes[i].title.set_text(titles[i])

                div = make_axes_locatable(axes[i])
                cax = div.append_axes("right", size="3%", pad="2%")
                fig.colorbar(im, cax)

            for ax in axes[1:]:
                ax.tick_params(labelleft=False)

            fig.tight_layout()
            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")

            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_map_agreement.png")
                plt.close()
        return 
    
    def plot_tod_samples(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Plots samples of the optimised tods.
        
        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        for k in range(0, 3):
            idxk = np.random.randint(0, self.noised_jax_tod.shape[0])
            todk = []
            for i in range(len(samples)):
                ti = self.signal_response_tod(samples[i])
                axes[k].plot(np.arange(0, ti.shape[1]), ti[idxk], label=f"tod {idxk} - sample {i}", c="gray")
                todk.append(ti)

            tmean = jft.mean(todk)
            axes[k].plot(np.arange(0, tmean.shape[1]), tmean[idxk], label=f"tod {idxk} - mean", c="red")

            axes[k].title.set_text(f"tod {idxk}: samples and mean")
            axes[k].legend(bbox_to_anchor=(1.01, 1), loc="upper left")

        for ax in axes:
            ax.set_xlim(0, ti.shape[1])
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        name = f", iter: {opt_state.nit}" if opt_state is not None else ""
        fig.suptitle(f"n_sub = {self.n_sub}{name}")

        if self.plotsdir is None: 
            plt.show()
        else:
            name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_tod_samples.png")
            plt.close()
        return

    def plot_map_samples(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Plots samples of the optimised map.

        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        if self.fit_map: 
            cmb_cmap = plt.get_cmap("cmb")

            fig, axes = plt.subplots(1, len(samples), figsize=(len(samples)*5, 5))

            sig_maps = tuple(self.gp_map(s) for s in samples)
            if self.padding_map > 0:
                sig_maps = tuple(s[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in sig_maps)
            
            vmin = min(s.min() for s in sig_maps)
            vmax = max(s.max() for s in sig_maps)

            for i,s in enumerate(sig_maps):
                im = axes[i].imshow(s, cmap=cmb_cmap, vmin=vmin, vmax=vmax)
                axes[i].title.set_text(f"map: sample {i}")

                div = make_axes_locatable(axes[i])
                cax = div.append_axes("right", size="3%", pad="2%")
                fig.colorbar(im, cax)

            for ax in axes[1:]:
                ax.tick_params(labelleft=False)

            fig.tight_layout()
            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")

            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_map_samples.png")
                plt.close()
        return 
    
    def plot_map_pullplot(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Plots pull plot of the map `= (mean - truth) / std`.
        
        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        if self.fit_map: 
            cmb_cmap = plt.get_cmap("cmb")

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            sig_maps = tuple(self.gp_map(s) for s in samples)
            if self.padding_map > 0:
                sig_maps = tuple(s[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in sig_maps)
            sig_mean, sig_std = jft.mean_and_std(sig_maps)

            sig_pull = (sig_mean - self.mapdata_truth[0, 0]) / sig_std

            im = ax.imshow(sig_pull, cmap=cmb_cmap, vmin=-10, vmax=10)
            ax.title.set_text(f"map: pull plot")

            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="3%", pad="2%")
            fig.colorbar(im, cax)

            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")

            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_map_pullplot.png")
                plt.close()
        return
    
    def plot_map_comparison(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Plot comparison of the optimised maps with maria fit and truth.

        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        if self.fit_map:
            from skimage.transform import resize
            cmb_cmap = plt.get_cmap("cmb")

            sig_maps = tuple(self.gp_map(s) for s in samples)
            if self.padding_map > 0:
                sig_maps = tuple(s[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in sig_maps)
            sig_mean = jft.mean(sig_maps)

            truth_rescaled = resize(self.mapdata_truth[0,0], self.output_map.data[0, 0].shape, anti_aliasing=True)

            images = (
                self.mapdata_truth[0,0], self.output_truthmap.data[0,0],
                self.output_map.data[0, 0], self.output_map.data[0, 0] - truth_rescaled,
                sig_mean, sig_mean - self.mapdata_truth[0, 0],
            )
            titles = (
                "truth", "noisy image (mapper output)",
                "maria mapper", "maria - truth",
                "nifty mean", "nifty mean - truth",
            )
            
            fig = plt.figure(figsize=(10, 15))

            axes = []
            for i in range(6):
                axes.append(fig.add_subplot(3, 2, i+1))

                im = axes[-1].imshow(images[i], cmap=cmb_cmap)
                axes[-1].title.set_text(titles[i])

                div = make_axes_locatable(axes[-1])
                cax = div.append_axes("right", size="3%", pad="2%")
                fig.colorbar(im, cax)

                if i % 2 != 0:
                    axes[-1].tick_params(labelleft=False)
                if i < 4:
                    axes[-1].tick_params(labelbottom=False)

            fig.tight_layout()
            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")

            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_map_comparison.png")
                plt.close()
        return
    
    def plot_atmos_simplified(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Plots simplified atmosphere predictions made by optimised GP and compares with truth.

        Args:
            samples (jft.evi.Samples): Samples to plot fit results for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        if self.fit_atmos:
            fig, axes = plt.subplots(2, 1, figsize=(16, 6))

            preds = []
            for x in samples:
                x_tod = {k: x[k] for k in x if "comb" in k}
                res_tods = self.gp_tod(x_tod)

                preds += [res_tods[:, self.padding_atmos//2:-self.padding_atmos//2], ]

            mean_atmos, std = jft.mean_and_std(tuple(preds))

            n_tods = 10
            for i in np.linspace(0, self.noised_jax_tod.shape[0]-1, n_tods, dtype=int):
                axes[0].plot(np.arange(0, mean_atmos.shape[1]*self.downsampling_factor), jnp.repeat(mean_atmos[i], self.downsampling_factor), label=f"tod {i}")
                axes[0].plot(self.atmos_tod_simplified[i], label=f"truth{i}")
                axes[1].plot(np.arange(0, mean_atmos.shape[1]*self.downsampling_factor), jnp.repeat(mean_atmos[i], self.downsampling_factor) - jnp.array(self.atmos_tod_simplified[i]), label=f"tod {i}")

            axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
            axes[0].title.set_text("simplified atmos: mean & truth (no noise)")
            axes[1].title.set_text("simplified atmos: mean - truth (no noise)")

            for ax in axes:
                ax.set_xlim(0, np.arange(0, mean_atmos.shape[1]*self.downsampling_factor).size)
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)

            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")
            
            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_atmos_simplified.png")
                plt.close()
        return
    
    def plot_power_spectrum(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
        """
        Plots power spectrum of predictions made by optimised GP and compares with truth.
        
        Args:
            samples (jft.evi.Samples): Samples to plot power spectrum for.
            opt_state (OptimizeVIState, optional): Optimisation state to plot. Defaults to None.
        """
        import scipy as sp
        from itertools import cycle

        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        components = [jft.mean(tuple(self.signal_response_tod(s) for s in samples))]
        labels = ["pred. total"]
        linestyles = ["-"]

        if self.fit_map:
            if self.padding_map > 0:
                gp_map_nopad = jnp.broadcast_to(jft.mean(tuple(self.gp_map(s) for s in samples)), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
            else:
                gp_map_nopad = jnp.broadcast_to(jft.mean(tuple(self.gp_map(s) for s in samples)), (1, 1, self.dims_map[0], self.dims_map[1]))

            # res_map = sample_maps(gp_map_nopad, self.dx, self.dy, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side)
            res_map = sample_maps(gp_map_nopad, self.instrument, self.offsets, self.sim_truthmap.map.resolution, self.sim_truthmap.map.x_side, self.sim_truthmap.map.y_side, self.pW_per_K_RJ)
            
            components += [res_map, self.tod_truthmap.data["map"]]
            labels += ["pred. map", "true map"]
            linestyles += ["-", "--"]

        if self.fit_atmos:
            
            res_tods = ()
            for s in samples:
                x_tod = {k: s[k] for k in s if "comb" in k}
                res_tod = self.gp_tod(x_tod)[:, self.padding_atmos//2:-self.padding_atmos//2]
                res_tod = jnp.repeat(res_tod, self.downsampling_factor)[None, :]
                res_tods += (res_tod,)
                
            res_tods = jft.mean(res_tods)
            
            components += [res_tods, self.tod_truthmap.data["atmosphere"]]
            labels += ["pred. atmos", "true atmos"]
            linestyles += ["-", "--"]

        fig, axes = plt.subplots(1, 1, figsize=(16, 6))
        for i in range(len(components)):
            
            f, ps = sp.signal.periodogram(components[i], fs=self.tod_truthmap.fs, window="tukey")

            f_bins = np.geomspace(f[1], f[-1], 256)
            f_mids = np.sqrt(f_bins[1:] * f_bins[:-1])

            binned_ps = sp.stats.binned_statistic(
                f, ps.mean(axis=0), bins=f_bins, statistic="mean"
            )[0]

            use = binned_ps > 0

            if linestyles[i] == "-": color = next(colors)
            axes.plot(
                f_mids[use],
                binned_ps[use],
                lw=1.4,
                color=color,
                label=labels[i],
                linestyle=linestyles[i]
            )
            
        axes.set_xlabel("Frequency [Hz]")
        axes.set_ylabel(f"[{self.tod_truthmap.units}$^2$/Hz]")
        axes.set_xlim(f_mids.min(), f_mids.max())
        axes.loglog()
        axes.legend()

        name = f", iter: {opt_state.nit}" if opt_state is not None else ""
        fig.suptitle(f"n_sub = {self.n_sub}{name}")
        
        if self.plotsdir is None: 
            plt.show()
        else:
            name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_power_spectrum.png")
            plt.close()
        return
        
    def make_atmosphere_det_gif(self, samples: jft.evi.Samples, figname: str = "atmosphere_comp.gif", tmax: int = -1, num_frames: int = 100) -> None:
        """
        Makes gif of simplified atmosphere prediction and truth in 2D detector layout. Does nothing if self.fit_atmos == False.
        
        Args:
            samples (jft.evi.Samples): Samples to make atmosphere prediction plot for.
            figname (str, optional): Location to save gif in. Defaults to "atmosphere_comp.gif"
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
            plt.savefig(buf, format="png")
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

        cmb_cmap = plt.get_cmap("cmb")
        
        best_fit_atmos = ()
        for s in samples:
            x_tod = {k: s[k] for k in s if "comb" in k}
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

        cmb_cmap = plt.get_cmap("cmb")

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