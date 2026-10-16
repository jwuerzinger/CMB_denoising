"""Public package exports for maria-nifty.

Importing this package also applies project-wide runtime defaults such as the
Astropy warning filter and JAX float64 enabling.
"""

from . import astropy_warning, enforce_float64
from .cli import get_stage_dirs, load_config, load_noise_masks
from .dejump import DEJUMP_DEFAULTS, DESPIKE_DEFAULTS, dejump, find_spike_mask
from .detector import Detector, hex_ring_clockwise, hex_ring_linear, hex_ring_mapping
from .fit import build_model, draw_init_position, load_samples, run_kl_fit, seed_joint_from_atmosphere
from .fits import from_fits_data, map2fits, map_component_names, map_samples2fits, tods2fits
from .fuse import (
    build_fused_arrays,
    build_fused_plot_models,
    setup_fused_forward_model,
)
from .mapper import run_bin_mapper, run_minkasi_mapper
from .minimize import check_dict, parse_minimization
from .model import (
    AtmosphereModel,
    DiffuseModel,
    GridSpace,
    MapModel,
    MapResponse,
    PointModel,
    PointSpace,
    SpaceCorrelationModel,
    SpacePolynomialModel,
    SpaceTimeCorrelationModel,
    SumResponse,
    TimeCorrelationModel,
    TimePolynomialModel,
    TimeSplineModel,
    apply_pointing_matrix,
    build_pointing_matrix,
    correlated_field_prior,
    gaussian_filter,
    gaussian_prior,
    map2space,
    map2tods,
    normal_prior,
    offsets2coords,
    project_map,
    upsample,
)
from .plot import (
    coefficient_histograms,
    detector_plot,
    detector_residual_plots,
    grid_plot,
    image_plot,
    line_plot,
    map_plot,
    masked_tod_plot,
    scatter_animation,
    scatter_plot,
    signal_cbar_label,
    tod_overview_plot,
    tod_plot,
    tod_spectrum_plot,
    truth_plot,
)
from .util import (
    build_detector,
    fits_label,
    input_indices,
    load_tod_datasets,
    load_tod_file,
    per_detector_sigma,
    raw_indices_of_valid,
    scatter_to_raw,
    setup_forward_model,
    setup_sample_distribution,
    write_boresight_elevation,
)
from .yml import yaml_load, yaml_save
