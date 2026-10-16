import warnings

warnings.filterwarnings(
    "ignore",
    message=".*polar motions for times before IERS data is valid.*",
    module="astropy.coordinates.builtin_frames.utils",
)
