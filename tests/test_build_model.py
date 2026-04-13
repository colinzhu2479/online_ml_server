import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transfer_server.model_init import *
# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_x():
    """Small 2D feature matrix: 10 samples, 3 features."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 3))

@pytest.fixture
def small_y():
    rng = np.random.default_rng(1)
    return rng.standard_normal((10,))

@pytest.fixture
def x_1d():
    """1D array for global_normalize."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# ═════════════════════════════════════════════════════════════════════════════
# act_gaussian
# ═════════════════════════════════════════════════════════════════════════════

class TestActGaussian:

    def test_peak_at_zero(self):
        """Gaussian peaks at x=0 → output should be 1.0."""
        fn = act_gaussian(std=5)
        result = fn(tf.constant(0.0, dtype=tf.float64)).numpy()
        assert result == pytest.approx(1.0)

    def test_symmetric(self):
        """Gaussian is symmetric: f(x) == f(-x)."""
        fn = act_gaussian(std=5)
        x = tf.constant(2.0, dtype=tf.float64)
        assert fn(x).numpy() == pytest.approx(fn(-x).numpy())

    def test_decays_away_from_zero(self):
        """Output should decrease as |x| increases."""
        fn = act_gaussian(std=5)
        v0 = fn(tf.constant(0.0, dtype=tf.float64)).numpy()
        v1 = fn(tf.constant(1.0, dtype=tf.float64)).numpy()
        v2 = fn(tf.constant(3.0, dtype=tf.float64)).numpy()
        assert v0 > v1 > v2

    def test_output_always_positive(self):
        """exp(...) is always positive."""
        fn = act_gaussian(std=5)
        for val in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            assert fn(tf.constant(val, dtype=tf.float64)).numpy() > 0

    def test_std_controls_width(self):
        """Larger std → slower decay → higher value at same x."""
        fn_narrow = act_gaussian(std=1)
        fn_wide   = act_gaussian(std=10)
        x = tf.constant(2.0, dtype=tf.float64)
        assert fn_wide(x).numpy() > fn_narrow(x).numpy()


# ═════════════════════════════════════════════════════════════════════════════
# individual_normalize
# ═════════════════════════════════════════════════════════════════════════════

class TestIndividualNormalize:

    def test_output_range_per_feature(self, small_x):
        """Each feature column should be in [0, 1] after normalization."""
        x_norm, _, _ = individual_normalize(small_x)
        assert np.all(x_norm >= 0.0 - 1e-12)
        assert np.all(x_norm <= 1.0 + 1e-12)

    def test_min_is_zero_per_feature(self, small_x):
        """Min of each column should be 0."""
        x_norm, _, _ = individual_normalize(small_x)
        np.testing.assert_allclose(np.min(x_norm, axis=0), 0.0, atol=1e-12)

    def test_max_is_one_per_feature(self, small_x):
        """Max of each column should be 1."""
        x_norm, _, _ = individual_normalize(small_x)
        np.testing.assert_allclose(np.max(x_norm, axis=0), 1.0, atol=1e-12)

    def test_xmin_matches_original(self, small_x):
        _, xmin, _ = individual_normalize(small_x)
        np.testing.assert_allclose(xmin, np.min(small_x, axis=0), atol=1e-12)

    def test_xrange_matches_original(self, small_x):
        _, _, xrange = individual_normalize(small_x)
        np.testing.assert_allclose(
            xrange, np.max(small_x, axis=0) - np.min(small_x, axis=0), atol=1e-12)

    def test_shape_preserved(self, small_x):
        x_norm, _, _ = individual_normalize(small_x)
        assert x_norm.shape == small_x.shape

    def test_invertible(self, small_x):
        """x_norm * xrange + xmin should recover original x."""
        x_norm, xmin, xrange = individual_normalize(small_x)
        recovered = x_norm * xrange + xmin
        np.testing.assert_allclose(recovered, small_x, atol=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# global_normalize
# ═════════════════════════════════════════════════════════════════════════════

class TestGlobalNormalize:

    def test_output_in_unit_range(self, x_1d):
        x_norm, _, _ = global_normalize(x_1d)
        assert np.min(x_norm) >= 0.0 - 1e-12
        assert np.max(x_norm) <= 1.0 + 1e-12

    def test_min_zero_max_one(self, x_1d):
        x_norm, _, _ = global_normalize(x_1d)
        np.testing.assert_allclose(np.min(x_norm), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.max(x_norm), 1.0, atol=1e-12)

    def test_xmin_and_xrange_correct(self, x_1d):
        _, xmin, xrange = global_normalize(x_1d)
        assert xmin == pytest.approx(np.min(x_1d))
        assert xrange == pytest.approx(np.max(x_1d) - np.min(x_1d))

    def test_invertible(self, x_1d):
        x_norm, xmin, xrange = global_normalize(x_1d)
        recovered = x_norm * xrange + xmin
        np.testing.assert_allclose(recovered, x_1d, atol=1e-10)

    def test_shift_moves_xmin(self, x_1d):
        """shift > 0 should lower xmin, expanding the output range below 0."""
        _, xmin_no_shift, _ = global_normalize(x_1d, shift=0)
        _, xmin_shifted, _  = global_normalize(x_1d, shift=0.1)
        assert xmin_shifted < xmin_no_shift

    def test_x_has_min_false_sets_xmin_zero(self, x_1d):
        """x_has_min=False forces xmin=0 regardless of data."""
        _, xmin, _ = global_normalize(x_1d, x_has_min=False)
        assert xmin == 0

    def test_constant_array_exits(self):
        """All-same values → xrange=0 → should call exit()."""
        x = np.array([3.0, 3.0, 3.0])
        with pytest.raises(SystemExit):
            global_normalize(x)


# ═════════════════════════════════════════════════════════════════════════════
# build_model
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildModel:

    @pytest.fixture
    def x_0(self):
        return np.zeros(6)  # 6 features

    def test_returns_keras_model(self, x_0):
        model = build_model(x_0, 'energy', num_atom=3, num_node_ratio=2, num_layer=2)
        assert isinstance(model, keras.Sequential)

    def test_energy_output_dim(self, x_0):
        """Energy model should output a single scalar per sample."""
        model = build_model(x_0, 'energy', num_atom=3, num_node_ratio=2, num_layer=2)
        assert model.output_shape == (None, 1)

    def test_force_output_dim(self, x_0):
        """Force model output dim should equal num_atom."""
        num_atom = 3
        model = build_model(x_0, 'force', num_atom=num_atom, num_node_ratio=2, num_layer=2)
        assert model.output_shape == (None, num_atom)

    def test_num_layers_energy(self, x_0):
        """num_layer=2 → 2 hidden + 1 output = 3 Dense layers total."""
        model = build_model(x_0, 'energy', num_atom=3, num_node_ratio=2, num_layer=2)
        assert len(model.layers) == 3

    def test_num_layers_scales(self, x_0):
        """Adding layers should increase layer count by 1 each time."""
        m2 = build_model(x_0, 'energy', num_atom=3, num_node_ratio=2, num_layer=2)
        m3 = build_model(x_0, 'energy', num_atom=3, num_node_ratio=2, num_layer=3)
        assert len(m3.layers) == len(m2.layers) + 1

    def test_node_count_scales_with_ratio(self, x_0):
        """Hidden layer width = num_node_ratio * num_feature."""
        num_feature = len(x_0)
        for ratio in [1, 2, 4]:
            model = build_model(x_0, 'energy', num_atom=3,
                                num_node_ratio=ratio, num_layer=2)
            assert model.layers[0].units == ratio * num_feature

    def test_input_dim_matches_features(self, x_0):
        model = build_model(x_0, 'energy', num_atom=3, num_node_ratio=2, num_layer=2)
        assert model.input_shape == (None, len(x_0))

    def test_model_compiled_with_mse(self, x_0):
        model = build_model(x_0, 'energy', num_atom=3, num_node_ratio=2, num_layer=2)
        assert model.loss == 'mse'

    def test_unknown_target_type_exits(self, x_0):
        with pytest.raises(SystemExit):
            build_model(x_0, 'unknown', num_atom=3, num_node_ratio=2, num_layer=2)

    def test_model_can_predict(self, x_0):
        """Model should be able to run a forward pass without error."""
        model = build_model(x_0, 'energy', num_atom=3, num_node_ratio=2, num_layer=2)
        dummy_input = np.zeros((1, len(x_0)))
        out = model.predict(dummy_input, verbose=0)
        assert out.shape == (1, 1)


# ═════════════════════════════════════════════════════════════════════════════
# def_normalization
# ═════════════════════════════════════════════════════════════════════════════

class TestDefNormalization:

    def test_force_returns_fixed_y_range(self, small_x, small_y):
        """Force branch uses hardcoded y_range=1/600."""
        _, _, y_range, _, _, _ = def_normalization(small_x, small_y, 'force')
        assert y_range == pytest.approx(1 / 600)

    def test_force_returns_fixed_e_range(self, small_x, small_y):
        _, _, _, _, e_range, _ = def_normalization(small_x, small_y, 'force')
        assert e_range == pytest.approx(1 / 600)

    def test_force_y_min_is_zero(self, small_x, small_y):
        _, _, _, y_min, _, _ = def_normalization(small_x, small_y, 'force')
        assert y_min == pytest.approx(0.0)

    def test_force_x_range_per_feature(self, small_x, small_y):
        """Force branch uses individual_normalize — x_range should be per-feature."""
        x_range, _, _, _, _, _ = def_normalization(small_x, small_y, 'force')
        expected = np.max(small_x, axis=0) - np.min(small_x, axis=0)
        np.testing.assert_allclose(x_range, expected, atol=1e-12)

    def test_energy_x_range_is_scalar(self, small_x, small_y):
        """Energy branch uses global_normalize — x_range should be a scalar."""
        x_range, _, _, _, _, _ = def_normalization(small_x, small_y, 'energy')
        assert np.isscalar(x_range) or np.array(x_range).ndim == 0

    def test_energy_e_range_fixed(self, small_x, small_y):
        _, _, _, _, e_range, _ = def_normalization(small_x, small_y, 'energy')
        assert e_range == pytest.approx(0.001)

    def test_energy_e_min_fixed(self, small_x, small_y):
        _, _, _, _, _, e_min = def_normalization(small_x, small_y, 'energy')
        assert e_min == pytest.approx(-0.004)

    def test_unknown_target_type_exits(self, small_x, small_y):
        with pytest.raises(SystemExit):
            def_normalization(small_x, small_y, 'unknown')

    def test_returns_six_values(self, small_x, small_y):
        result = def_normalization(small_x, small_y, 'energy')
        assert len(result) == 6
