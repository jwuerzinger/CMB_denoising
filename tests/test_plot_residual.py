"""Tests for maria_nifty/plot/residual.py — pure Python, NumPy, and NIFTy.re."""

import jax.numpy as jnp
import nifty.re as jft
import numpy as np

from maria_nifty.plot.residual import _cluster_plot_data, _compute_plot_data


class TestComputePlotData:
    def test_no_model_returns_raw_tod(self):
        tod = np.ones((3, 10))
        data, label, suffix = _compute_plot_data(tod, None, None)
        np.testing.assert_array_equal(data, tod)

    def test_no_model_returns_same_object(self):
        tod = np.ones((2, 5))
        data, _, _ = _compute_plot_data(tod, None, None)
        assert data is tod

    def test_no_model_label_is_tods(self):
        _, label, suffix = _compute_plot_data(np.zeros((1, 4)), None, None)
        assert label == "TODs"
        assert suffix == "tods"

    def test_no_model_preserves_values(self):
        rng = np.random.default_rng(0)
        tod = rng.standard_normal((4, 20))
        data, _, _ = _compute_plot_data(tod, None, None)
        np.testing.assert_array_equal(data, tod)


class TestComputePlotDataNifty:
    def test_vector_samples_subtracts_model_output(self):
        tod = np.full((3, 10), 5.0)
        model = lambda s: jnp.full((3, 10), 2.0)  # noqa: E731
        samples = jft.Vector({"param": jnp.zeros(1)})
        data, label, suffix = _compute_plot_data(tod, model, samples)
        np.testing.assert_allclose(np.array(data), 3.0, atol=1e-6)
        assert label == "Residuals"
        assert suffix == "residuals"

    def test_vector_samples_output_shape(self):
        tod = np.zeros((4, 20))
        model = lambda s: jnp.ones((4, 20))  # noqa: E731
        samples = jft.Vector(jnp.zeros(1))
        data, _, _ = _compute_plot_data(tod, model, samples)
        assert data.shape == (4, 20)

    def test_jft_samples_mean_subtracted(self):
        # 2 samples; model returns the same value for all → mean = 2.0
        tod = np.full((3, 10), 5.0)
        posterior = jft.Samples(pos=None, samples=jnp.zeros((2, 1)))
        model = lambda s: jnp.full((3, 10), 2.0)  # noqa: E731
        data, label, suffix = _compute_plot_data(tod, model, posterior)
        np.testing.assert_allclose(np.array(data), 3.0, atol=1e-5)
        assert label == "Residuals"
        assert suffix == "residuals"

    def test_jft_samples_averages_over_samples(self):
        # Two samples return 1.0 and 3.0 respectively; mean = 2.0 → tod(10) - 2 = 8
        tod = np.full((1, 5), 10.0)
        outputs = iter([jnp.full((1, 5), 1.0), jnp.full((1, 5), 3.0)])
        model = lambda s: next(outputs)  # noqa: E731
        posterior = jft.Samples(pos=None, samples=jnp.zeros((2, 1)))
        data, _, _ = _compute_plot_data(tod, model, posterior)
        np.testing.assert_allclose(np.array(data), 8.0, atol=1e-5)

    def test_iterable_samples_else_branch(self):
        # Plain list → falls to else branch; jft.mean averages JAX arrays
        tod = np.full((2, 6), 4.0)
        model = lambda s: jnp.full((2, 6), 1.5)  # noqa: E731
        data, label, suffix = _compute_plot_data(tod, model, [None, None])
        np.testing.assert_allclose(np.array(data), 2.5, atol=1e-5)
        assert label == "Residuals"
        assert suffix == "residuals"


class TestClusterPlotData:
    def test_single_detector_all_in_one_cluster(self):
        data = np.ones((1, 10))
        indices, n_eff, colors = _cluster_plot_data(data, n_clusters=5)
        assert indices.shape == (1,)
        assert indices[0] == 1
        assert n_eff == 1

    def test_cluster_indices_are_1based(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 20))
        indices, n_eff, _ = _cluster_plot_data(data, n_clusters=3)
        assert np.all(indices >= 1)
        assert np.all(indices <= n_eff)

    def test_n_clusters_capped_to_n_detectors(self):
        data = np.array([[1.0] * 10, [2.0] * 10])
        _, n_eff, colors = _cluster_plot_data(data, n_clusters=100)
        assert n_eff == 2
        assert len(colors) == 2

    def test_n_clusters_1_all_same_cluster(self):
        data = np.ones((4, 5))
        indices, n_eff, _ = _cluster_plot_data(data, n_clusters=1)
        assert n_eff == 1
        np.testing.assert_array_equal(indices, np.ones(4, dtype=int))

    def test_group_colors_keys_are_0based(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 10))
        _, n_eff, colors = _cluster_plot_data(data, n_clusters=3)
        assert set(colors.keys()) == set(range(n_eff))

    def test_output_index_shape_matches_n_detectors(self):
        rng = np.random.default_rng(7)
        data = rng.standard_normal((6, 15))
        indices, n_eff, colors = _cluster_plot_data(data, n_clusters=2)
        assert indices.shape == (6,)
        assert n_eff == 2
        assert len(colors) == 2

    def test_group_colors_values_are_tuples_of_length_4(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((4, 8))
        _, _, colors = _cluster_plot_data(data, n_clusters=2)
        for v in colors.values():
            assert len(v) == 4  # RGBA
