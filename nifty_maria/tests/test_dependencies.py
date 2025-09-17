import pytest
import sys
import importlib
import tomllib

import tomllib
import importlib
import pytest
import pathlib

base_dir = pathlib.Path(__file__).resolve().parents[2]
pixi_file = base_dir / "pixi.toml"

with pixi_file.open("rb") as f:
    pixi_config = tomllib.load(f)

deps = pixi_config.get("dependencies", {}).copy()
cpu_deps = pixi_config.get("feature.cpu.dependencies", {})
deps.update(cpu_deps)
gpu_deps = pixi_config.get("feature.gpu.dependencies", {})
deps.update(gpu_deps)
gpu_sys_req = pixi_config-get("feature.gpu.system-requirements", {})

import_name_map = {
    "scikit-image": "skimage"
}

dependencies = {import_name_map.get(k, k): v for k, v in deps.items()}

print(dependencies)

def version_tuple(version_str):
    return tuple(int(x) for x in version_str.split(".") if x.isdigit())

def satisfies(version, spec):
    import re
    match = re.match(r"(>=|<=|==|>|<)(.+)", spec)
    if not match:
        raise ValueError(f"Invalid spec: {spec}")
    op, ver = match.groups()
    version = version_tuple(version)
    ver = version_tuple(ver)
    if op == "==":
        return version == ver
    elif op == ">=":
        return version >= ver
    elif op == "<=":
        return version <= ver
    elif op == ">":
        return version > ver
    elif op == "<":
        return version < ver
    else:
        return False

@pytest.mark.parametrize("pkg_name,spec", [(k, v) for k, v_range in dependencies.items() for v in v_range.split(",")])
def test_dependency_versions(pkg_name, spec):
    pkg = importlib.import_module(pkg_name)
    version_str = getattr(pkg, "__version__", None)
    assert version_str is not None, f"{pkg_name} has no __version__ attribute"
    assert satisfies(version_str, spec), f"{pkg_name} version {version_str} does not satisfy {spec}"
