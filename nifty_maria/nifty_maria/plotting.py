"""
Module to colloct plotting functions for nifty-maria fits.
"""

import numpy as np
import scipy.ndimage
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nifty8.re as jft
from nifty8.re.optimize_kl import OptimizeVIState
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.io import fits

from nifty_maria.mapsampling_jax import sample_maps
from nifty_maria.mapsampling_jax import gaussian_filter

class Plotter:
    """Subclass containing plotting functionalities."""

    def smooth_img(self, img):

        sigma_rad = self.instrument.dets.fwhm[0]/ np.sqrt(8 * np.log(2))
        sigma_pixels = float(sigma_rad/self.sim_truthmap.map.resolution)

        img_smoothed = gaussian_filter(img, sigma_pixels)
    
        return img_smoothed

    def plot_callback(self, samples: jft.evi.Samples, opt_state: OptimizeVIState) -> None:
        """
        Callback function to be used for plotting fit status during optimisation.
        
        Args:
            samples (jft.evi.Samples): Samples to perform plots for.
            opt_state (OptimizeVIState): Optimisation state to plot.
        """
        
        iter = opt_state[0]
        self.printfitresults(samples, iter)
        
        if len(samples) == 0:
            samples = (samples.pos,)
        
        if iter % self.printevery != 0: return
        
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
        
        if len(samples) == 0:
            samples = (samples.pos,)
        
        self.plot_tod_agreement(samples, opt_state) # done
        self.plot_map_agreement(samples, opt_state) # done
        self.plot_tod_samples(samples, opt_state)
        self.plot_map_samples(samples, opt_state)
        self.plot_map_pullplot(samples, opt_state)
        self.plot_map_stdev(samples, opt_state)
        self.plot_map_comparison(samples, opt_state)
        self.plot_map_comparison_maxLH(samples, opt_state)
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

        # Plot without noise:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        n_tods = 10
        for i in np.linspace(0, self.noised_jax_tod.shape[0]-1, n_tods, dtype=int):
            axes[0].plot(np.arange(0, res.shape[1]), res[i], label=f"tod {i}")
            axes[1].plot(np.arange(0, res.shape[1]), res[i] - self.denoised_jax_tod[i], label=f"tod {i}")
            axes[2].plot(self.denoised_jax_tod[i], label=f"tod {i}")

        axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        axes[0].title.set_text("tods: mean")
        axes[0].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)
        axes[1].title.set_text("tods: mean - truth (no noise)")
        axes[1].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)
        axes[2].title.set_text("tods: truth (no noise)")
        axes[2].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)

        custom_ticks = np.linspace(0., self.maria_params['duration']*self.maria_params['sample_rate'], min((self.maria_params['duration']+1), 16))
        custom_labels = [f"{t/self.maria_params['sample_rate']:.2f}" for t in custom_ticks]
        for ax in axes: ax.set_xticks(custom_ticks)
        axes[2].set_xticklabels(custom_labels)
        axes[2].set_xlabel('time [s]', fontsize=12)

        for ax in axes:
            ax.set_xlim(0, res.shape[1])
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        name = f", iter: {opt_state.nit}" if opt_state is not None else ""
        fig.suptitle(f"n_sub = {self.n_sub}{name}")
        fig.align_ylabels(axes)

        if self.plotsdir is None: 
            plt.show()
        else:
            name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
            plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_tod_agreement_denoised.png")
            plt.close()
          
        # Plot with noise  
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        n_tods = 10
        for i in np.linspace(0, self.noised_jax_tod.shape[0]-1, n_tods, dtype=int):
            axes[0].plot(np.arange(0, res.shape[1]), res[i], label=f"tod {i}")
            axes[1].plot(np.arange(0, res.shape[1]), res[i] - self.noised_jax_tod[i], label=f"tod {i}")
            axes[2].plot(self.noised_jax_tod[i], label=f"tod {i}")

        axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        axes[0].title.set_text("tods: mean")
        axes[0].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)
        axes[1].title.set_text("tods: mean - truth (with noise)")
        axes[1].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)
        axes[2].title.set_text("tods: truth (with noise)")
        axes[2].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)

        custom_ticks = np.linspace(0., self.maria_params['duration']*self.maria_params['sample_rate'], min((self.maria_params['duration']+1), 16))
        custom_labels = [f"{t/self.maria_params['sample_rate']:.2f}" for t in custom_ticks]
        for ax in axes: ax.set_xticks(custom_ticks)
        axes[2].set_xticklabels(custom_labels)
        axes[2].set_xlabel('time [s]', fontsize=12)

        for ax in axes:
            ax.set_xlim(0, res.shape[1])
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        name = f", iter: {opt_state.nit}" if opt_state is not None else ""
        fig.suptitle(f"n_sub = {self.n_sub}{name}")
        fig.align_ylabels(axes)

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

            images = (self.smooth_img(self.mapdata_truth), sig_mean, sig_mean - self.smooth_img(self.mapdata_truth))
            titles = ("map: truth (smoothed)", "map: mean", "map: mean - truth (smoothed)")

            vmin = np.min(self.smooth_img(self.mapdata_truth))
            vmax = np.max(self.smooth_img(self.mapdata_truth))
            vmins = (vmin, vmin, None)
            vmaxs = (vmax, vmax, None)

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            plt.subplots_adjust(wspace=0.3, left=0.01, right=0.93, top=0.95, bottom=0.01)

            for i in range(3):
                im = axes[i].imshow(images[i], cmap=cmb_cmap, vmin=vmins[i], vmax=vmaxs[i])
                axes[i].title.set_text(titles[i])
                axes[i].tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

                div = make_axes_locatable(axes[i])
                cax = div.append_axes("right", size="3%", pad="2%")
                cb = fig.colorbar(im, cax)
                cb.set_label(r"Intensity [$K_{RJ}$]", fontsize=12)

            for ax in axes[1:]:
                ax.tick_params(labelleft=False)

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
        if len(samples) < 2: return
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        plt.subplots_adjust(wspace=0.1, left=0.1, right=0.9, top=0.9, bottom=0.1)

        for k in range(0, 3):
            idxk = np.random.randint(0, self.noised_jax_tod.shape[0])
            todk = []
            for i in range(len(samples)):
                ti = self.signal_response_tod(samples[i])
                axes[k].plot(np.arange(0, ti.shape[1]), ti[idxk], label=f"sample {i}", c="gray")
                todk.append(ti)

            tmean = jft.mean(todk)
            axes[k].plot(np.arange(0, tmean.shape[1]), tmean[idxk], label=f"mean", c="red")

            axes[k].title.set_text(f"tod {idxk}: samples and mean")
            axes[k].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
            axes[k].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)

        custom_ticks = np.linspace(0., self.maria_params['duration']*self.maria_params['sample_rate'], min((self.maria_params['duration']+1), 16))
        custom_labels = [f"{t/self.maria_params['sample_rate']:.2f}" for t in custom_ticks]
        for ax in axes: ax.set_xticks(custom_ticks)
        axes[2].set_xticklabels(custom_labels)
        axes[2].set_xlabel('time [s]', fontsize=12)

        for ax in axes:
            ax.set_xlim(0, ti.shape[1])
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        name = f", iter: {opt_state.nit}" if opt_state is not None else ""
        fig.suptitle(f"n_sub = {self.n_sub}{name}")
        fig.align_ylabels(axes)

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
        if len(samples) < 2: return
        
        if self.fit_map: 
            cmb_cmap = plt.get_cmap("cmb")

            fig, axes = plt.subplots(1, len(samples), figsize=(len(samples)*5, 5))
            plt.subplots_adjust(wspace=0.3, left=0.01, right=0.93, top=0.95, bottom=0.01)

            sig_maps = tuple(self.gp_map(s) for s in samples)
            if self.padding_map > 0:
                sig_maps = tuple(s[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2] for s in sig_maps)
            
            vmin = min(s.min() for s in sig_maps)
            vmax = max(s.max() for s in sig_maps)

            for i,s in enumerate(sig_maps):
                im = axes[i].imshow(s, cmap=cmb_cmap, vmin=vmin, vmax=vmax)
                axes[i].title.set_text(f"map: sample {i}")
                axes[i].tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

                div = make_axes_locatable(axes[i])
                cax = div.append_axes("right", size="3%", pad="2%")
                cb = fig.colorbar(im, cax)
                cb.set_label(r"Intensity [$K_{RJ}$]", fontsize=12)

            for ax in axes[1:]:
                ax.tick_params(labelleft=False)

            # fig.tight_layout()
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
            sig_mean, sig_std = jft.mean_and_std(sig_maps) if len(sig_maps) > 1 else (jft.mean(sig_maps), 1)
            sig_pull = (sig_mean - self.smooth_img(self.mapdata_truth)) / sig_std

            im = ax.imshow(sig_pull, cmap=cmb_cmap, vmin=-10, vmax=10)
            ax.title.set_text(f"map: pull plot")
            ax.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="3%", pad="2%")
            cb = fig.colorbar(im, cax)
            cb.set_label(r"Pull $(\bar{x}-x_{\mathrm{true}})/\sigma$", fontsize=12)

            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")

            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_map_pullplot.png")
                plt.close()
        return
    
    def plot_map_stdev(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
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
            sig_mean, sig_std = jft.mean_and_std(sig_maps) if len(sig_maps) > 1 else (jft.mean(sig_maps), 1)
            # sig_pull = (sig_mean - self.smooth_img(self.mapdata_truth)) / sig_std

            im = ax.imshow(sig_std, cmap=cmb_cmap)
            ax.title.set_text(f"map: Std. Dev.")
            ax.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="3%", pad="2%")
            cb = fig.colorbar(im, cax)
            cb.set_label(r"$\sigma$", fontsize=12)

            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")

            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_map_stdev.png")
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


            output_map = self.output_map.to(units="K_RJ").data[(0,) * (self.output_map.data.ndim - 2) + (...,)].compute()
            truth_rescaled = resize(self.smooth_img(self.mapdata_truth), output_map.shape, anti_aliasing=True)

            images = (
                output_map+truth_rescaled.mean(), output_map - (truth_rescaled-truth_rescaled.mean()), self.smooth_img(self.mapdata_truth),
                sig_mean, sig_mean - self.smooth_img(self.mapdata_truth),
            )
            titles = (
                "maria mapper", "maria - truth (smoothed)", "truth (smoothed)",
                "nifty mean", "nifty mean - truth (smoothed)",
            )
            
            vmin = jnp.min(self.smooth_img(self.mapdata_truth))
            vmin_comp = jnp.min(jnp.array([jnp.min(output_map - (truth_rescaled-truth_rescaled.mean())), jnp.min(sig_mean - self.smooth_img(self.mapdata_truth))]))
            vmax = jnp.max(self.smooth_img(self.mapdata_truth))
            vmax_comp = jnp.max(jnp.array([jnp.max(output_map - (truth_rescaled-truth_rescaled.mean())), jnp.max(sig_mean - self.smooth_img(self.mapdata_truth))]))
            vmins = (
                vmin, vmin_comp, vmin,
                vmin, vmin_comp
            )
            vmaxs = (
                vmax, vmax_comp, vmax,
                vmax, vmax_comp
            )
            
            fig = plt.figure(figsize=(15, 10))
            plt.subplots_adjust(wspace=0.3, left=0.01, right=0.93, top=0.95, bottom=0.01)

            axes = []
            for i in range(5):
                axes.append(fig.add_subplot(2, 3, i+1))

                im = axes[-1].imshow(images[i], cmap=cmb_cmap, vmin=vmins[i], vmax=vmaxs[i])
                axes[-1].title.set_text(titles[i])
                axes[i].tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

                div = make_axes_locatable(axes[i])
                cax = div.append_axes("right", size="3%", pad="2%")
                cb = fig.colorbar(im, cax)
                cb.set_label(r"Intensity [$K_{RJ}$]", fontsize=12)

                if i % 2 != 0:
                    axes[-1].tick_params(labelleft=False)
                if i < 4:
                    axes[-1].tick_params(labelbottom=False)

            # fig.tight_layout()
            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")

            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_map_comparison.png")
                plt.close()
        return
    
    def plot_map_comparison_maxLH(self, samples: jft.evi.Samples, opt_state: OptimizeVIState = None) -> None:
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

            map_filename = "../simulated/mustang-2_20.fits"
            hdu = fits.open(map_filename)
            smoothed_data = scipy.ndimage.gaussian_filter(hdu[0].data, sigma=1)
            smoothed_data = np.flip(smoothed_data, axis=1)
            # resize to match nifty:
            smoothed_data = smoothed_data[31:-31, 31:-31]
            truth_rescaled = resize(self.smooth_img(self.mapdata_truth), smoothed_data.shape, anti_aliasing=True)

            images = (
                smoothed_data, smoothed_data - truth_rescaled, self.smooth_img(self.mapdata_truth),
                sig_mean, sig_mean - self.smooth_img(self.mapdata_truth),
            )
            titles = (
                "max. Likelihood map", "max. Likelihood - truth (smoothed)", "truth (smoothed)",
                "nifty mean", "nifty mean - truth (smoothed)",
            )
            
            vmin = jnp.min(self.smooth_img(self.mapdata_truth))
            vmin_comp = jnp.min(jnp.array([jnp.min(smoothed_data - truth_rescaled), jnp.min(sig_mean - self.smooth_img(self.mapdata_truth))]))
            vmax = jnp.max(self.smooth_img(self.mapdata_truth))
            vmax_comp = jnp.max(jnp.array([jnp.max(smoothed_data - truth_rescaled), jnp.max(sig_mean - self.smooth_img(self.mapdata_truth))]))
            vmins = (
                vmin, vmin_comp, vmin,
                vmin, vmin_comp
            )
            vmaxs = (
                vmax, vmax_comp, vmax,
                vmax, vmax_comp
            )
            
            fig = plt.figure(figsize=(15, 10))
            plt.subplots_adjust(wspace=0.3, left=0.01, right=0.93, top=0.95, bottom=0.01)

            axes = []
            for i in range(5):
                axes.append(fig.add_subplot(2, 3, i+1))

                im = axes[-1].imshow(images[i], cmap=cmb_cmap, vmin=vmins[i], vmax=vmaxs[i])
                axes[-1].title.set_text(titles[i])
                axes[i].tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

                div = make_axes_locatable(axes[i])
                cax = div.append_axes("right", size="3%", pad="2%")
                cb = fig.colorbar(im, cax)
                cb.set_label(r"Intensity [$K_{RJ}$]", fontsize=12)

                if i % 2 != 0:
                    axes[-1].tick_params(labelleft=False)
                if i < 4:
                    axes[-1].tick_params(labelbottom=False)

            # fig.tight_layout()
            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")

            if self.plotsdir is None: 
                plt.show()
            else:
                name = f"nit_{opt_state.nit}" if opt_state is not None else "final"
                plt.savefig(f"{self.plotsdir}/nsub_{self.n_sub}_{name}_map_comparison_maxLH.png")
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

            preds = ()
            for x in samples:
                x_tod = {k: x[k] for k in x if "comb" in k}
                res_tods = self.gp_tod(x_tod)

                preds += (res_tods[:, self.padding_atmos//2:-self.padding_atmos//2], )

            mean_atmos = jft.mean(preds)

            n_tods = 10
            for i in np.linspace(0, self.noised_jax_tod.shape[0]-1, n_tods, dtype=int):
                axes[0].plot(np.arange(0, mean_atmos.shape[1]*self.downsampling_factor), jnp.repeat(mean_atmos[i], self.downsampling_factor), label=f"tod {i}")
                axes[0].plot(self.atmos_tod_simplified[i], label=f"truth{i}")
                axes[1].plot(np.arange(0, mean_atmos.shape[1]*self.downsampling_factor), jnp.repeat(mean_atmos[i], self.downsampling_factor) - jnp.array(self.atmos_tod_simplified[i]), label=f"tod {i}")

            axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
            axes[0].title.set_text("simplified atmos: mean & truth (no noise)")
            axes[0].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)
            axes[1].title.set_text("simplified atmos: mean - truth (no noise)")
            axes[1].set_ylabel(r"$I_{\nu}$ [$\mathrm{pW}$]", fontsize=12)

            custom_ticks = np.linspace(0., self.maria_params['duration']*self.maria_params['sample_rate'], min((self.maria_params['duration']+1), 16))
            custom_labels = [f"{t/self.maria_params['sample_rate']:.2f}" for t in custom_ticks]
            for ax in axes: ax.set_xticks(custom_ticks)
            axes[1].set_xticklabels(custom_labels)
            axes[1].set_xlabel('time [s]', fontsize=12)

            for ax in axes:
                ax.set_xlim(0, np.arange(0, mean_atmos.shape[1]*self.downsampling_factor).size)
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)

            name = f", iter: {opt_state.nit}" if opt_state is not None else ""
            fig.suptitle(f"n_sub = {self.n_sub}{name}")
            fig.align_ylabels(axes)
            
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

        components = []
        labels = []
        linestyles = []

        if self.fit_map:
            if self.padding_map > 0:
                # gp_map_nopad = jnp.broadcast_to(jft.mean(tuple(self.gp_map(s) for s in samples)), (1, 1, self.dims_map[0], self.dims_map[1]))[:, :, self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
                gp_map_nopad = jft.mean(tuple(self.gp_map(s) for s in samples))[self.padding_map//2:-self.padding_map//2, self.padding_map//2:-self.padding_map//2]
            else:
                # gp_map_nopad = jnp.broadcast_to(jft.mean(tuple(self.gp_map(s) for s in samples)), (1, 1, self.dims_map[0], self.dims_map[1]))
                gp_map_nopad = jft.mean(tuple(self.gp_map(s) for s in samples))

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

                
        # Add noise:
        components += [self.tod_truthmap.data["noise"]]
        labels += ["true noise"]
        linestyles += ["--"]
        # Add truth:
        components += [jft.mean(tuple(self.signal_response_tod(s) for s in samples))]
        labels += ["pred. total"]
        linestyles += ["-"]

        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        for i in range(len(components)):
            
            f, ps = sp.signal.periodogram(components[i], fs=self.tod_truthmap.fs, window="tukey")

            f_bins = np.geomspace(f[1], f[-1], 256)
            f_mids = np.sqrt(f_bins[1:] * f_bins[:-1])

            binned_ps = sp.stats.binned_statistic(
                f, ps.mean(axis=0), bins=f_bins, statistic="mean"
            )[0]

            use = binned_ps > 0

            if linestyles[i] == "-": color = next(colors)
            elif labels[i] == "true noise": color = next(colors)
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
        
        from maria.units import Quantity

        cmb_cmap = plt.get_cmap("cmb")
        
        best_fit_atmos = ()
        for s in samples:
            x_tod = {k: s[k] for k in s if "comb" in k}
            best_fit_atmos += (self.gp_tod(x_tod)[:, self.padding_atmos//2:-self.padding_atmos//2], )
            
        best_fit_atmos = jft.mean(best_fit_atmos)

        test = Quantity(self.instrument.dets.offsets, "rad")
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
        from maria.units import Quantity

        instrument = self.instrument

        cmb_cmap = plt.get_cmap("cmb")

        test = Quantity(instrument.dets.offsets, "rad")
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
        from maria.units import Quantity

        # fwhms = Angle(self.instrument.dets.angular_fwhm(z=z))
        # offsets = Angle(self.instrument.dets.offsets)
        fwhms = Quantity(self.instrument.dets.angular_fwhm(z=z), "rad")
        offsets = Quantity(self.instrument.dets.offsets, "rad")

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