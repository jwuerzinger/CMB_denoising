import pytest
import sys
import importlib
import tomllib
import re
import pathlib

maria_file = base_dir / "pyproject.toml"
with maria_file.open("rb") as f:
    maria_config = tomllib.load(f)

maria_deps = {}
for dep in maria_config["project"]["dependencies"]:
    match = re.match(r"([A-Za-z0-9_\-]+)(.*)", dep)
    if match:
        name, spec = match.groups()
        name = name.strip()
        maria_deps[name] = spec.strip() if spec else ""

base_dir = pathlib.Path(__file__).resolve().parents[2]
pixi_file = base_dir / "pixi.toml"

with pixi_file.open("rb") as f:
    pixi_config = tomllib.load(f)

cmb_deps = pixi_config.get("dependencies", {}).copy()
cmb_deps.update(pixi_config.get("feature.cpu.dependencies", {}))
cmb_deps.update(pixi_config.get("feature.gpu.dependencies", {}))
cmb_deps.update(pixi_config.get("feature.gpu.system-requirements", {}))

import_name_map = {
    "scikit-image": "skimage"
}

cmb_deps = {import_name_map.get(k, k): v for k, v in cmb_deps.items()}
maria_deps = {import_name_map.get(k, k): v for k, v in maria_deps.items()}

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

def test_version_specs_consistent():
    """Fail if maria and CMB_denoising disagree on version specs."""
    common = set(cmb_deps.keys()) & set(maria_deps.keys())
    for pkg in sorted(common):
        spec1, spec2 = cmb_deps[pkg], maria_deps[pkg]
        if spec1 and spec2 and spec1 != spec2:
            pytest.fail(
                f"Version mismatch for {pkg}: "
                f"CMB_denoising requires '{spec1}', "
                f"but maria requires '{spec2}'"
            )

@pytest.mark.parametrize("pkg_name,spec", [(k, v) for k, v in all_deps.items() if v])
def test_dependency_versions(pkg_name, spec):
    """Fail if installed version does not satisfy declared specs."""
    pkg = importlib.import_module(pkg_name)
    version_str = getattr(pkg, "__version__", None)
    assert version_str is not None, f"{pkg_name} has no __version__ attribute"
    assert satisfies(version_str, spec), (
        f"{pkg_name} version {version_str} does not satisfy {spec}"
    )
