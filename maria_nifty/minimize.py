"""Helpers for expanding and normalizing minimization configurations.

This module turns compact YAML-style optimization descriptions into callables
and iteration-aware structures that the reconstruction workflow can consume.
"""

import ast
import re
from collections.abc import Iterable

OPT_NEEDED = {"n_total_iterations", "n_samples"}
OPT_OPTION = {"draw_linear_kwargs", "nonlinearly_update_kwargs", "kl_kwargs", "sample_mode"}


def parse_minimization(opt_dct: dict) -> dict:
    """Expand a minimization config into iteration-aware callables.

    Parameters
    ----------
    opt_dct : dict
        Raw minimization configuration loaded from YAML.

    Returns
    -------
    dict
        Normalized configuration where list-valued entries are callable by
        iteration index.
    """

    opt_dct = check_dict(opt_dct, OPT_NEEDED, OPT_OPTION)

    opt_dct = get_reps(opt_dct, opt_dct["n_total_iterations"])
    opt_dct = clean_reps(opt_dct)
    for ok, ov in opt_dct.items():
        opt_dct[ok] = make_callable(ov)

    return opt_dct


def make_callable(val):
    """Wrap a value or list of values so it can be queried by iteration.

    If ``val`` contains any lists (i.e. it varies per iteration), returns a
    ``fun(it)`` closure.  Otherwise evaluates at iteration 0 and returns the
    plain value directly, so callers that don't need per-iteration access get
    a scalar rather than a callable.
    """

    def fun(it):
        return get_it(val, it)

    if is_or_contains_type(val, list):
        return fun
    else:
        return fun(0)


def check_dict(dictionary: dict, needed_keys, optional_keys: list = None) -> dict:
    """Keep only allowed keys and verify required keys are present.

    Parameters
    ----------
    dictionary : dict
        Input dictionary to filter.
    needed_keys : iterable
        Keys that must be present; raises ``ValueError`` if any are missing.
    optional_keys : iterable, optional
        Additional keys that are retained but not required.

    Returns
    -------
    dict
        A new dict containing only the allowed keys.
    """
    if optional_keys is None:
        optional_keys = []
    allowed_keys = set(needed_keys) | set(optional_keys)

    if isinstance(dictionary, dict):
        allowed_dict = {key: val for key, val in dictionary.items() if key in allowed_keys}
        for key in needed_keys:
            if key not in allowed_dict:
                raise ValueError(f"key `{key}` is missing in dictionary")
    return allowed_dict


def get_reps(sub: dict, total_it: int, minor_it: list | None = None) -> dict:
    """Recursively expand repetition sections to the desired length.

    Parameters
    ----------
    sub : dict
        Minimization sub-configuration to expand in-place.
    total_it : int
        Total number of major iterations; sets the target sequence length.
    minor_it : list of int or None, optional
        Per-major-iteration minor-iteration counts.  When provided, each
        element of a sequence is itself expanded to the corresponding minor
        length (used for nested kwargs such as ``kl_kwargs``).

    Returns
    -------
    dict
        The expanded ``sub`` dict (mutated in-place and returned).
    """
    for key, val in sub.items():
        if isinstance(val, dict):
            sub[key] = get_reps(val, total_it, minor_it)
            continue

        elif key in ["n_total_iterations", "n_major_iterations"]:
            if not isinstance(val, int) or val < 1:
                raise TypeError(f"`{key}` has to be of type `int` and larger than 0")
            sub[key] = val
            continue

        elif isinstance(val, str):
            # Evaluate compact string expressions such as "2*[kl] + [linear]" to lists.
            val = eval_string(val)

        # Wrap scalars in a list so every value is uniformly sequence-shaped.
        if not isinstance(val, list) or val == []:
            val = [val]

        if isinstance(val, list):
            val = eval_list(val)
            # 'constants', 'point_estimates', and 'transitions' are sets of
            # parameter names that become inactive after their last entry;
            # pad with None instead of repeating the last value.
            if key in ["constants", "point_estimates", "transitions"]:
                val = extend_reps(val, total_it, None)
            else:
                val = extend_reps(val, total_it)

            if minor_it and key not in ["n_minor_iterations", "likelihood", "transitions"]:
                # Expand each major-iteration entry to its per-minor-iteration schedule.
                for i, mi in enumerate(minor_it):
                    vi = val[i]
                    if not isinstance(vi, list) or vi == []:
                        vi = [vi]
                    val[i] = extend_reps(vi, mi)

            sub[key] = val

    return sub


def is_or_contains_type(dct, typ):
    """Check whether a value or nested list/dict contains a given type."""
    if isinstance(dct, typ):
        return True
    elif isinstance(dct, dict):
        for val in dct.values():
            if isinstance(val, list):
                return True
            elif isinstance(val, dict):
                if is_or_contains_type(val, typ):
                    return True
    return False


def get_it(dct, it):
    """Resolve the value of a nested structure for a given iteration."""

    def _flatten_list(lst):
        lst = [_flatten_list(val) if isinstance(val, list) else val for val in lst]
        return [v for val in lst for v in (val if isinstance(val, list) else [val])]

    dct_it = {}
    match dct:
        case dict():
            for key, val in dct.items():
                dct_it[key] = get_it(val, it)
        case list():
            dct_it = _flatten_list(dct)[it]
        case _:
            dct_it = dct

    return dct_it


def extend_reps(val, total_it: int, add_val=-1) -> list:
    """Extend or trim a sequence so it matches the target iteration count.

    Parameters
    ----------
    val : iterable or scalar
        The sequence (or single value) to adjust.
    total_it : int
        Desired output length.
    add_val : any, optional
        Value used to pad the sequence when it is too short.  The sentinel
        ``-1`` (default) means "repeat the last element of ``val``".

    Returns
    -------
    list
        A list of exactly ``total_it`` elements.
    """
    val = (
        list(val)
        if isinstance(val, Iterable)
        else [
            val,
        ]
    )
    add_val = val[-1] if add_val == -1 else add_val
    dif = total_it - len(val)
    if dif < 0:
        val = val[:total_it]
    else:
        val += dif * [add_val]
    return val


def clean_reps(dct, simplify=True):
    """Compress repeated list structures in a minimization dictionary.

    This is the display-side inverse of :func:`get_reps`: it converts
    fully-expanded per-iteration lists back into compact run-length strings
    (e.g. ``['kl', 'kl', 'linear']`` → ``"2*[kl] + [linear]"``), producing
    a human-readable YAML-compatible representation for logging.  Sequences
    where every element is identical are collapsed to the scalar value.

    Parameters
    ----------
    dct : dict
        Expanded minimization dictionary to compress.
    simplify : bool, optional
        If ``True``, attempt run-length encoding of non-nested lists.

    Returns
    -------
    dict
        The compressed dict (mutated in-place and returned).
    """

    def _simplify_list(lst):
        # Fallback: plain Python repr of the list with quotes stripped.
        expr = re.sub(r"'([^']*)'", r"\1", str(lst))
        if not any(isinstance(li, list) for li in lst):
            # Build a run-length encoded string, e.g. [kl, kl, linear] → "2*[kl] + [linear]".
            val = []
            l0 = None
            f = 0
            for li in lst + [""]:  # "" is a sentinel to flush the final run
                if f > 0 and type(li) is type(l0) and li == l0:
                    f += 1  # f counts the current run length
                else:
                    if f > 0:
                        val += [f"{f}*[{l0}]"] if f > 1 else [f"[{l0}]"]
                    f = 1
                    l0 = li
            val = " + ".join(val)
            val = re.sub(r"'([^']*)'", r"\1", str(val))
            if len(val) < len(expr):
                return val
        return expr

    def _clean_list(lst):
        lst = [_clean_list(li) if isinstance(li, list) else li for li in lst]
        if not any(isinstance(li, list) for li in lst) and len(set(lst)) == 1:
            return lst[0]
        elif simplify:
            return _simplify_list(lst)
        return lst

    for key, val in dct.items():
        match val:
            case dict():
                dct[key] = clean_reps(val, simplify)
            case list():
                newval = _clean_list(val)
                match newval:
                    case str() as nv if nv.startswith("[") and {"+", "*"} & set(nv):
                        newval = "1*" + newval
                    case str() as nv if {"[", ",", "]"} & set(nv):
                        newval = val
                if not isinstance(newval, list):
                    dct[key] = newval

    return dct


def eval_string(expr: str):
    """Safely evaluate a compact list/tuple expression from a string.

    Supports the YAML shorthand ``"n*[val]"`` and concatenation forms,
    e.g. ``"2*[kl] + [linear]"`` → ``['kl', 'kl', 'linear']``.

    Parameters
    ----------
    expr : str
        A string expression containing only list/tuple literals, integer and
        float constants, bare identifier names, addition, and multiplication.

    Returns
    -------
    list or tuple or scalar
        The evaluated Python object.
    """
    expr = expr.replace(" ", "")
    # Surround bare identifiers with quotes so ast.parse treats them as string
    # constants rather than Name nodes (e.g. kl → "kl", SL → "SL").
    expr = re.sub(r"(?<!\d)([a-zA-Z=_~][a-zA-Z0-9.=_]*)(?!\d)", r'"\1"', expr)
    # Restore None/null/~ to Python None — these were wrongly quoted above.
    expr = re.sub(r'"None"|"null"|"~"', "None", expr)

    def _eval(node):
        match node:
            case ast.List(elts=elts):
                return [_eval(el) for el in elts]
            case ast.Tuple(elts=elts):
                return tuple(_eval(el) for el in elts)
            case ast.BinOp(left=left, op=ast.Add(), right=right):
                left, right = _eval(left), _eval(right)
                if isinstance(left, list | tuple) and isinstance(right, list | tuple):
                    return left + right
                raise ValueError("Invalid addition")
            case ast.BinOp(left=left, op=ast.Mult(), right=right):
                left, right = _eval(left), _eval(right)
                if isinstance(left, list | tuple) and isinstance(right, int):
                    return left * right
                if isinstance(left, int) and isinstance(right, list | tuple):
                    return right * left
                raise ValueError("Invalid multiplication")
            case ast.Constant(value=value):
                return value
            case _:
                raise ValueError("Unsupported expression")

    return _eval(ast.parse(expr, mode="eval").body)


def eval_list(expr):
    """Recursively evaluate string expressions inside a list."""
    if isinstance(expr, list):
        return [eval_list(el) for el in expr]
    elif isinstance(expr, str):
        return eval_string(expr)
    return expr
