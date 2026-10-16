"""YAML import/export helpers used by the steering scripts and package API.

The functions in this module normalize dictionary keys, load one or more YAML
streams into a single mapping, and write reproducible YAML output with a custom
style that keeps top-level sections readable.
"""

import os

import yaml


def yaml_load(fname: str | list[str]) -> dict:
    """Load one or more YAML files into a single Python dictionary.

    Parameters
    ----------
    fname : str or list of str
        File name or list of file names to read.

    Returns
    -------
    dct : dict
        Merged dictionary containing all YAML documents from the input files.
    """
    if isinstance(fname, str):
        fname = [fname]

    dct = {}
    for fn in fname:
        if not os.path.isfile(fn):
            raise RuntimeError(f"`{fn}` not found")
        with open(fn) as f:
            yml_list = list(yaml.safe_load_all(f))
            for ll in yml_list:
                if ll is not None:
                    dct |= get_vals(ll)
    return dct


def yaml_save(dct: dict | list[dict], fname: str) -> None:
    """Write a dictionary or list of dictionaries to a YAML file.

    Parameters
    ----------
    dct : dict or list of dict
        Mapping or list of mappings to write.
    fname : str
        Output file path.
    """
    dct_lst = (
        dct
        if isinstance(dct, list)
        else [
            dct,
        ]
    )
    if not all(isinstance(di, dict) for di in dct_lst):
        raise TypeError
    dumper = MyDumper
    dumper.add_representer(list, flow_list_rep)

    with open(fname, "w") as f:
        yaml.dump_all(dct_lst, f, Dumper=MyDumper, sort_keys=False)


def get_vals(dct):
    """Recursively replace spaces in dictionary keys with underscores."""
    new_dct = {}
    for key, val in dct.items():
        if isinstance(val, dict):
            val = get_vals(val)
        new_dct[key.replace(" ", "_")] = val
    return new_dct


class MyDumper(yaml.SafeDumper):
    """YAML dumper that adds extra spacing between top-level keys."""

    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) <= 1:
            super().write_line_break()


def flow_list_rep(dumper, data):
    """Represent Python lists in flow style when dumping YAML."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
