"""Tests for YAML load/save round-trip."""

import os
import tempfile

import pytest

from maria_nifty.yml import yaml_load, yaml_save


def test_round_trip_simple():
    dct = {"key": "value", "number": 42, "flag": True}
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
        fname = f.name
    try:
        yaml_save(dct, fname)
        loaded = yaml_load(fname)
        assert loaded == dct
    finally:
        os.unlink(fname)


def test_round_trip_nested():
    dct = {"outer": {"inner": [1, 2, 3]}, "top": "level"}
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
        fname = f.name
    try:
        yaml_save(dct, fname)
        loaded = yaml_load(fname)
        assert loaded == dct
    finally:
        os.unlink(fname)


def test_key_spaces_normalised():
    # Keys with spaces should be converted to underscores on load.
    content = "some key: value\n"
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
        f.write(content)
        fname = f.name
    try:
        loaded = yaml_load(fname)
        assert "some_key" in loaded
        assert "some key" not in loaded
    finally:
        os.unlink(fname)


def test_load_missing_file_raises():
    with pytest.raises(RuntimeError):
        yaml_load("/nonexistent/path/file.yml")


def test_save_non_dict_raises():
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
        fname = f.name
    try:
        with pytest.raises(TypeError):
            yaml_save("not a dict", fname)
    finally:
        os.unlink(fname)


def test_load_list_of_files():
    dct1 = {"a": 1}
    dct2 = {"b": 2}
    with (
        tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f1,
        tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f2,
    ):
        fname1, fname2 = f1.name, f2.name
    try:
        yaml_save(dct1, fname1)
        yaml_save(dct2, fname2)
        loaded = yaml_load([fname1, fname2])
        assert loaded == {"a": 1, "b": 2}
    finally:
        os.unlink(fname1)
        os.unlink(fname2)
