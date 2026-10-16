"""Tests for maria_nifty/util.py — pure Python (dict operations only)."""

from maria_nifty.util import _get_atmosphere_downsampling, _normalize_atmosphere_config


class TestNormalizeAtmosphereConfig:
    def test_empty_dict_unchanged(self):
        assert _normalize_atmosphere_config({}) == {}

    def test_spline_key_renamed(self):
        result = _normalize_atmosphere_config({"spline": {"order": 3}})
        assert "spline_params" in result
        assert "spline" not in result
        assert result["spline_params"] == {"order": 3}

    def test_gp_key_renamed(self):
        result = _normalize_atmosphere_config({"gp": {"downsampling": 4}})
        assert "gp_params" in result
        assert "gp" not in result

    def test_poly_key_renamed(self):
        result = _normalize_atmosphere_config({"poly": {"order": 2}})
        assert "poly_params" in result
        assert "poly" not in result

    def test_existing_spline_params_not_overwritten(self):
        cfg = {"spline": {"order": 1}, "spline_params": {"order": 99}}
        result = _normalize_atmosphere_config(cfg)
        assert result["spline_params"] == {"order": 99}
        assert "spline" in result

    def test_mode_gp_1d_normalized(self):
        result = _normalize_atmosphere_config({"mode": "gp 1d"})
        assert result["mode"] == "gp-1d"

    def test_mode_gp_1d_already_normalized(self):
        result = _normalize_atmosphere_config({"mode": "gp-1d"})
        assert result["mode"] == "gp-1d"

    def test_mode_gp_3d_normalized(self):
        result = _normalize_atmosphere_config({"mode": "GP-3D"})
        assert result["mode"] == "gp-3d"

    def test_mode_spline_unchanged(self):
        result = _normalize_atmosphere_config({"mode": "spline"})
        assert result["mode"] == "spline"

    def test_non_string_mode_unchanged(self):
        result = _normalize_atmosphere_config({"mode": None})
        assert result["mode"] is None

    def test_original_dict_not_mutated(self):
        original = {"gp": {"downsampling": 4}, "mode": "gp-1d"}
        _normalize_atmosphere_config(original)
        assert "gp" in original
        assert "gp_params" not in original

    def test_unrelated_keys_preserved(self):
        result = _normalize_atmosphere_config({"mode": "spline", "prefix": "atm"})
        assert result["prefix"] == "atm"


class TestGetAtmosphereDownsampling:
    def test_no_atmosphere_section_returns_1(self):
        assert _get_atmosphere_downsampling({}) == 1

    def test_empty_atmosphere_section_returns_1(self):
        assert _get_atmosphere_downsampling({"atmosphere": {}}) == 1

    def test_gp_1d_reads_downsampling(self):
        config = {"atmosphere": {"mode": "gp-1d", "gp_params": {"downsampling": 4}}}
        assert _get_atmosphere_downsampling(config) == 4

    def test_gp_3d_reads_downsampling(self):
        config = {"atmosphere": {"mode": "gp-3d", "gp_params": {"downsampling": 8}}}
        assert _get_atmosphere_downsampling(config) == 8

    def test_gp_1d_missing_downsampling_key_defaults_to_1(self):
        config = {"atmosphere": {"mode": "gp-1d", "gp_params": {}}}
        assert _get_atmosphere_downsampling(config) == 1

    def test_gp_1d_missing_gp_params_entirely_defaults_to_1(self):
        config = {"atmosphere": {"mode": "gp-1d"}}
        assert _get_atmosphere_downsampling(config) == 1

    def test_spline_mode_returns_1(self):
        config = {"atmosphere": {"mode": "spline", "spline_params": {"order": 3}}}
        assert _get_atmosphere_downsampling(config) == 1

    def test_no_mode_key_returns_1(self):
        config = {"atmosphere": {"gp_params": {"downsampling": 4}}}
        assert _get_atmosphere_downsampling(config) == 1

    def test_old_gp_key_normalized_before_lookup(self):
        config = {"atmosphere": {"mode": "gp-1d", "gp": {"downsampling": 2}}}
        assert _get_atmosphere_downsampling(config) == 2
