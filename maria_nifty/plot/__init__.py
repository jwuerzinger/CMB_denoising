"""Public plotting API exported by maria-nifty."""

from .animation import scatter_animation
from .extra import detector_plot, masked_tod_plot, tod_plot, truth_plot
from .grid import grid_plot
from .histogram import coefficient_histograms
from .image import image_plot
from .line import line_plot
from .map import map_plot
from .residual import detector_residual_plots
from .scatter import scatter_plot
from .spectrum import tod_spectrum_plot
from .tod_overview import tod_overview_plot
from .util import signal_cbar_label
