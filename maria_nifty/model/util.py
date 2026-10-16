"""Low-level utilities shared across the model and plotting code."""

from collections.abc import Iterable

import numpy as np

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
SPEEDOFLIGHT = 299792458.0


def str2rad(s):
    """Convert a numeric string with angular units into radians.

    Supported suffixes are `muas`, `mas`, `as`, `amin`, `deg`, and `rad`.
    If no unit suffix is present, the string is interpreted as a raw float.
    """
    c = {
        "muas": AS2RAD * 1e-6,
        "mas": AS2RAD * 1e-3,
        "as": AS2RAD,
        "amin": ARCMIN2RAD,
        "deg": DEG2RAD,
        "rad": 1,
    }
    for k in c:
        if s.endswith(k):
            return float(s[: -len(k)]) * c[k]
    try:
        return float(s)
    except ValueError as err:
        raise RuntimeError("Unit not understood") from err


def check_type(value, *types, uppers=()):
    """Validate a value against nested type specifications.

    The first element of ``types`` is matched against ``value``; subsequent
    types are used recursively to validate elements of iterables.

    Parameters
    ----------
    value : any
        Value to validate.
    *types : type or tuple of type
        Expected type at each nesting level.
    uppers : tuple, optional
        Parent containers accumulated during recursion (used for error messages).

    Raises
    ------
    TypeError
        If ``value`` does not match the expected type at any level.
    """
    if not isinstance(value, types[0]):
        err = f"`{value}`"
        for up in uppers[::-1]:
            err += f" in `{up}`"
        raise TypeError(f"{err} has to be of type `{types[0]}`")
    if isinstance(value, Iterable) and len(types) > 1:
        for v in value:
            check_type(v, *types[1:], uppers=uppers + (value,))


def flatten_list(lst):
    """Flatten nested iterables into a single list, excluding strings.

    Parameters
    ----------
    lst : iterable
        Potentially nested sequence to flatten.

    Returns
    -------
    list
        Flat list containing all scalar (non-iterable) elements.
    """
    new_lst = []
    for val in lst:
        if isinstance(val, Iterable) and not isinstance(val, str):
            new_lst += flatten_list(val)
        else:
            new_lst += [
                val,
            ]
    return new_lst


def to_shape(array, shape, dtype="float64"):
    """Convert a scalar, string, or sequence into a NumPy array of the given shape.

    String elements are parsed as angular values by :func:`str2rad` before
    conversion.  A single-element input is broadcast to ``shape``; a
    multi-element input is reshaped if the total size matches.

    Parameters
    ----------
    array : scalar, str, or iterable
        Input value(s) to convert.
    shape : tuple of int
        Target shape of the output array.
    dtype : str or numpy dtype, optional
        Output data type (default ``"float64"``).

    Returns
    -------
    numpy.ndarray
        Array of the requested shape and dtype.
    """

    lst = (
        array
        if isinstance(array, Iterable) and not isinstance(array, str)
        else [
            array,
        ]
    )
    lst = flatten_list(lst)
    lst = [str2rad(li) if isinstance(li, str) else li for li in lst]
    array = np.array(lst)

    res = np.reshape(array, shape) if array.size == np.prod(shape) else np.broadcast_to(array, shape)

    return res.astype(dtype)


def is_val(array):
    """Return ``True`` when the array contains any non-zero, non-NaN values.

    Parameters
    ----------
    array : array-like
        Array to test.

    Returns
    -------
    bool
    """
    return bool(np.any(array[~np.isnan(array)] != 0))
