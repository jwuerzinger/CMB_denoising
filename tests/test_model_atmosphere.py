"""Tests for maria_nifty/model/atmosphere.py — no JAX required."""

import pytest

from maria_nifty.model.atmosphere import AtmosphereModel

_interpret_mode = AtmosphereModel._interpret_mode


# ---------------------------------------------------------------------------
# Valid single-base modes
# ---------------------------------------------------------------------------


class TestInterpretModeValidSingle:
    def test_spline(self):
        assert _interpret_mode("spline") == "spline"

    def test_gp_1d(self):
        assert _interpret_mode("gp-1d") == "gp-1d"

    def test_gp_3d(self):
        assert _interpret_mode("gp-3d") == "gp-3d"

    def test_case_insensitive(self):
        assert _interpret_mode("GP-1D") == "gp-1d"
        assert _interpret_mode("Spline") == "spline"
        assert _interpret_mode("GP-3D") == "gp-3d"

    def test_leading_trailing_whitespace(self):
        assert _interpret_mode("  gp-1d  ") == "gp-1d"


# ---------------------------------------------------------------------------
# Valid composite modes (base + polynomial add-on)
# ---------------------------------------------------------------------------


class TestInterpretModeValidComposite:
    def test_gp1d_plus_poly1d(self):
        assert _interpret_mode("gp-1d+poly-1d") == "gp-1d+poly-1d"

    def test_gp1d_plus_poly2d(self):
        assert _interpret_mode("gp-1d+poly-2d") == "gp-1d+poly-2d"

    def test_gp3d_plus_poly1d(self):
        assert _interpret_mode("gp-3d+poly-1d") == "gp-3d+poly-1d"

    def test_spline_plus_poly2d(self):
        assert _interpret_mode("spline+poly-2d") == "spline+poly-2d"

    def test_comma_separator(self):
        assert _interpret_mode("gp-1d,poly-1d") == "gp-1d+poly-1d"

    def test_space_separator(self):
        assert _interpret_mode("gp-1d poly-2d") == "gp-1d+poly-2d"

    def test_mixed_separators(self):
        assert _interpret_mode("gp-1d, poly-1d") == "gp-1d+poly-1d"

    def test_duplicate_base_deduplicates(self):
        # Repeated base token is deduplicated; result is just the single base.
        assert _interpret_mode("gp-1d+gp-1d") == "gp-1d"

    def test_poly_addon_order_irrelevant(self):
        # poly-1d listed before base — output always puts base first.
        assert _interpret_mode("poly-1d+gp-1d") == "gp-1d+poly-1d"


# ---------------------------------------------------------------------------
# Invalid modes — should raise ValueError
# ---------------------------------------------------------------------------


class TestInterpretModeInvalid:
    def test_none_raises(self):
        with pytest.raises(ValueError, match="required"):
            _interpret_mode(None)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _interpret_mode("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _interpret_mode("   ")

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            _interpret_mode("fourier")

    def test_partial_match_is_unknown(self):
        # "gp" alone is not a valid token; only "gp-1d" and "gp-3d" are.
        with pytest.raises(ValueError, match="Unknown"):
            _interpret_mode("gp")

    def test_two_base_modes_raises(self):
        with pytest.raises(ValueError, match="exactly one base"):
            _interpret_mode("gp-1d+gp-3d")

    def test_spline_and_gp_raises(self):
        with pytest.raises(ValueError, match="exactly one base"):
            _interpret_mode("spline+gp-1d")

    def test_poly_without_base_raises(self):
        # poly-1d alone has no base mode.
        with pytest.raises(ValueError, match="exactly one base"):
            _interpret_mode("poly-1d")

    def test_both_poly_types_raises(self):
        with pytest.raises(ValueError, match="either"):
            _interpret_mode("gp-1d+poly-1d+poly-2d")
