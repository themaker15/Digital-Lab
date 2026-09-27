"""
Test Suite: Layer A - Pure Math & Algorithm Engine Verification
Water Contamination FDD Simulator

Covers Tiers 1-4 for scikit-learn StandardScaler, PCA(2), UCLs,
Hotelling's T², SPE/Q, PCA Biplot Confidence Ellipse, and Root Cause Isolation.
Runs independently of Streamlit UI with zero overhead.
"""

import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────────────
# Reference Ground Truths (Seed 42, 500 samples, 4 sensors)
# ─────────────────────────────────────────────────────────────────────────────
REF_PHYSICAL_MEANS = np.array([25.0, 150.0, 5.0, 7.0])
REF_PHYSICAL_STDS  = np.array([0.25,   4.0, 0.4, 0.05])
REF_LABELS = ['Temp', 'TDS', 'Turb', 'pH']
REF_UNITS  = ['°C', 'ppm', 'NTU', '']

REF_FAULTS = {
    'Normal (Clean Water)':       np.array([0.0,   0.0,  0.0,  0.0]),
    'NaCl (Salt Solution)':       np.array([0.0, 140.0,  0.0,  0.1]),
    'Acetic Acid (Vinegar)':      np.array([0.0,  30.0,  1.5, -2.8]),
    'Milk Protein':               np.array([0.0,  20.0, 28.0, -0.4]),
    'Sensor Drift (Hardware)':    np.array([5.5,   0.0,  0.0,  0.0]),
}

# Empirical ground truths derived with seed 42
GROUND_TRUTH_UCL_T2 = 8.953975470948391
GROUND_TRUTH_UCL_Q  = 8.946037542258892
GROUND_TRUTH_ELLIPSE_A = 3.193302
GROUND_TRUTH_ELLIPSE_B = 3.018862


def simulate_training_pipeline(seed=42, n_samples=500):
    """Executes the exact Phase 1 training pipeline from app.py."""
    rng = np.random.default_rng(seed=seed)
    raw_train = rng.standard_normal((n_samples, 4)) * REF_PHYSICAL_STDS + REF_PHYSICAL_MEANS
    scaler = StandardScaler().fit(raw_train)
    Z_train = scaler.transform(raw_train)
    pca = PCA(n_components=2).fit(Z_train)
    scores_train = pca.transform(Z_train)
    
    T2_train = np.sum((scores_train / np.sqrt(pca.explained_variance_)) ** 2, axis=1)
    ucl_t2 = float(np.percentile(T2_train, 99))
    
    Z_hat = pca.inverse_transform(scores_train)
    Q_train = np.sum((Z_train - Z_hat) ** 2, axis=1)
    ucl_q = float(np.percentile(Q_train, 99))
    
    return raw_train, scaler, pca, ucl_t2, ucl_q


def evaluate_inference(raw, scaler, pca):
    """Executes the exact process_sample() inference math."""
    z = scaler.transform(raw.reshape(1, -1))[0]
    scores = pca.transform(z.reshape(1, -1))[0]
    ev = pca.explained_variance_
    T2 = float(np.sum((scores / np.sqrt(ev)) ** 2))
    z_hat = pca.inverse_transform(scores.reshape(1, -1))[0]
    Q = float(np.sum((z - z_hat) ** 2))
    contrib = np.clip((z ** 2) / max(T2, 1e-9), 0, 1)
    contrib = contrib / max(contrib.sum(), 1e-9)
    return z, scores, T2, Q, contrib


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Feature Coverage (Pure Math & Algorithm Engine)
# ─────────────────────────────────────────────────────────────────────────────
class TestTier1_FeatureCoverage_Math(unittest.TestCase):
    """Tier 1: Comprehensive feature coverage for PCA & FDD math contracts."""

    @classmethod
    def setUpClass(cls):
        cls.raw_train, cls.scaler, cls.pca, cls.ucl_t2, cls.ucl_q = simulate_training_pipeline()

    def test_tc_t1_r2_01_training_dataset_generation(self):
        """Verify NOC dataset has exact shape (500, 4) and close to physical means."""
        self.assertEqual(self.raw_train.shape, (500, 4))
        emp_means = np.mean(self.raw_train, axis=0)
        np.testing.assert_allclose(emp_means, REF_PHYSICAL_MEANS, atol=0.25)

    def test_tc_t1_r2_02_scaler_parameters_invariance(self):
        """Verify StandardScaler mean_ and scale_ match benchmark within 1e-4."""
        expected_means = np.array([24.983339, 149.828953, 4.969540, 6.998251])
        expected_stds  = np.array([0.239391, 4.184791, 0.409194, 0.048871])
        np.testing.assert_allclose(self.scaler.mean_, expected_means, atol=1e-4)
        np.testing.assert_allclose(self.scaler.scale_, expected_stds, atol=1e-4)

    def test_tc_t1_r2_03_pca_explained_variance_ratio(self):
        """Verify PCA(2) preserves 2 components and captures ~53.81% total variance."""
        self.assertEqual(self.pca.n_components, 2)
        vr = self.pca.explained_variance_ratio_
        self.assertEqual(len(vr), 2)
        self.assertAlmostEqual(vr[0], 0.284141, places=4)
        self.assertAlmostEqual(vr[1], 0.253946, places=4)
        self.assertAlmostEqual(sum(vr), 0.538087, places=4)

    def test_tc_t1_r2_04_ucl_percentiles_invariance(self):
        """Verify exact 99th percentile UCL values for T2 and Q."""
        self.assertAlmostEqual(self.ucl_t2, GROUND_TRUTH_UCL_T2, places=4)
        self.assertAlmostEqual(self.ucl_q, GROUND_TRUTH_UCL_Q, places=4)

    def test_tc_t1_r2_05_single_sample_inference(self):
        """Verify process_sample inference produces valid shapes and metrics."""
        raw = REF_PHYSICAL_MEANS + np.array([0.0, 140.0, 0.0, 0.1])
        z, scores, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertEqual(len(z), 4)
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(T2, float)
        self.assertIsInstance(Q, float)
        self.assertEqual(len(contrib), 4)
        self.assertAlmostEqual(float(np.sum(contrib)), 1.0, places=5)

    def test_tc_t1_r3_01_biplot_ellipse_semi_axes(self):
        """Verify 99% Hotelling T2 confidence ellipse semi-axes a and b."""
        ev = self.pca.explained_variance_
        a = float(np.sqrt(ev[0] * self.ucl_t2))
        b = float(np.sqrt(ev[1] * self.ucl_t2))
        self.assertAlmostEqual(a, GROUND_TRUTH_ELLIPSE_A, places=4)
        self.assertAlmostEqual(b, GROUND_TRUTH_ELLIPSE_B, places=4)

    def test_tc_t1_r3_02_biplot_parametric_ellipse(self):
        """Verify parametric ellipse equation (x/a)^2 + (y/b)^2 == 1 for all theta."""
        a = float(np.sqrt(self.pca.explained_variance_[0] * self.ucl_t2))
        b = float(np.sqrt(self.pca.explained_variance_[1] * self.ucl_t2))
        thetas = np.linspace(0, 2 * np.pi, 50)
        x = a * np.cos(thetas)
        y = b * np.sin(thetas)
        ratios = (x / a) ** 2 + (y / b) ** 2
        np.testing.assert_allclose(ratios, 1.0, atol=1e-7)

    def test_tc_t1_r3_03_biplot_loading_vectors(self):
        """Verify PCA loadings components_ matrix shape (2, 4) and non-zero vectors."""
        loadings = self.pca.components_
        self.assertEqual(loadings.shape, (2, 4))
        # Norm of each loading vector across components
        for j in range(4):
            vec_len = np.linalg.norm(loadings[:, j])
            self.assertGreater(vec_len, 0.1)

    def test_tc_t1_r4_01_scree_plot_variance_distribution(self):
        """Verify Scree plot data represents explained variance ratios in descending order."""
        vr = self.pca.explained_variance_ratio_
        self.assertGreaterEqual(vr[0], vr[1])
        self.assertGreater(vr[0] * 100, 25.0)

    def test_tc_t1_r4_02_baseline_table_values(self):
        """Verify baseline values match expected physical ranges."""
        self.assertAlmostEqual(self.scaler.mean_[0], 25.0, delta=0.5)
        self.assertAlmostEqual(self.scaler.mean_[1], 150.0, delta=5.0)
        self.assertAlmostEqual(self.scaler.mean_[2], 5.0, delta=0.5)
        self.assertAlmostEqual(self.scaler.mean_[3], 7.0, delta=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Boundary & Corner Cases (Pure Math & Algorithm Engine)
# ─────────────────────────────────────────────────────────────────────────────
class TestTier2_BoundaryCornerCases_Math(unittest.TestCase):
    """Tier 2: Boundary conditions, numerical stability, and edge inputs."""

    @classmethod
    def setUpClass(cls):
        _, cls.scaler, cls.pca, cls.ucl_t2, cls.ucl_q = simulate_training_pipeline()

    def test_tc_t2_r2_01_single_sample_reshaping(self):
        """Verify scaler & pca handle (1, 4) shaped arrays properly."""
        raw = np.array([[25.0, 150.0, 5.0, 7.0]])
        z = self.scaler.transform(raw)
        self.assertEqual(z.shape, (1, 4))
        scores = self.pca.transform(z)
        self.assertEqual(scores.shape, (1, 2))

    def test_tc_t2_r2_02_zero_division_guard_t2_zero(self):
        """Verify contribution calculation when T2 approaches zero."""
        z = np.zeros(4)
        T2 = 0.0
        contrib = np.clip((z ** 2) / max(T2, 1e-9), 0, 1)
        contrib = contrib / max(contrib.sum(), 1e-9)
        self.assertFalse(np.isnan(contrib).any())
        self.assertFalse(np.isinf(contrib).any())

    def test_tc_t2_r2_03_zero_sum_normalization_guard(self):
        """Verify zero-sum normalization guard prevents division by zero."""
        contrib = np.zeros(4)
        norm_contrib = contrib / max(contrib.sum(), 1e-9)
        np.testing.assert_allclose(norm_contrib, np.zeros(4))

    def test_tc_t2_r2_04_extreme_sensor_shock_100sigma(self):
        """Verify 100-sigma sensor shock does not produce NaN or Inf in T2/Q."""
        extreme_raw = REF_PHYSICAL_MEANS + 100 * REF_PHYSICAL_STDS
        _, _, T2, Q, contrib = evaluate_inference(extreme_raw, self.scaler, self.pca)
        self.assertTrue(np.isfinite(T2))
        self.assertTrue(np.isfinite(Q))
        self.assertTrue(np.all(np.isfinite(contrib)))
        self.assertGreater(T2, 1000.0)

    def test_tc_t2_r2_05_physical_sensor_lower_bounds(self):
        """Verify Turbidity=0 and TDS=0 (physical lower bounds) evaluate cleanly."""
        raw_zero_turb = np.array([25.0, 150.0, 0.0, 7.0])
        _, _, T2, Q, _ = evaluate_inference(raw_zero_turb, self.scaler, self.pca)
        self.assertTrue(np.isfinite(T2))
        self.assertTrue(np.isfinite(Q))

    def test_tc_t2_r3_01_boundary_score_exact_ucl(self):
        """Verify score on the ellipse boundary has T2 exactly equal to UCL_T2."""
        ev = self.pca.explained_variance_
        a = np.sqrt(ev[0] * self.ucl_t2)
        # Point on major vertex (a, 0)
        t = np.array([a, 0.0])
        T2 = float(np.sum((t / np.sqrt(ev)) ** 2))
        self.assertAlmostEqual(T2, self.ucl_t2, places=5)

    def test_tc_t2_r3_02_interior_score_inside_ellipse(self):
        """Verify points inside ellipse have T2 < UCL_T2."""
        ev = self.pca.explained_variance_
        a = np.sqrt(ev[0] * self.ucl_t2)
        b = np.sqrt(ev[1] * self.ucl_t2)
        interior_point = np.array([0.5 * a, 0.5 * b])
        T2 = float(np.sum((interior_point / np.sqrt(ev)) ** 2))
        self.assertLess(T2, self.ucl_t2)

    def test_tc_t2_r3_03_exterior_score_outside_ellipse(self):
        """Verify points outside ellipse have T2 > UCL_T2."""
        ev = self.pca.explained_variance_
        a = np.sqrt(ev[0] * self.ucl_t2)
        b = np.sqrt(ev[1] * self.ucl_t2)
        exterior_point = np.array([1.5 * a, 1.2 * b])
        T2 = float(np.sum((exterior_point / np.sqrt(ev)) ** 2))
        self.assertGreater(T2, self.ucl_t2)

    def test_tc_t2_r3_04_degenerate_variance_protection(self):
        """Verify safety against zero or negative eigenvalues."""
        zero_ev = np.array([0.0, 1.0])
        safe_ev = np.maximum(zero_ev, 1e-6)
        a = np.sqrt(safe_ev[0] * self.ucl_t2)
        self.assertTrue(np.isfinite(a))
        self.assertGreaterEqual(a, 0.0)

    def test_tc_t2_r4_01_high_noise_training_stability(self):
        """Verify training algorithm stability under high noise conditions."""
        rng = np.random.default_rng(seed=999)
        noisy_train = rng.standard_normal((500, 4)) * (REF_PHYSICAL_STDS * 10) + REF_PHYSICAL_MEANS
        scaler = StandardScaler().fit(noisy_train)
        pca = PCA(n_components=2).fit(scaler.transform(noisy_train))
        self.assertEqual(pca.n_components_, 2)
        self.assertTrue(np.all(np.isfinite(pca.explained_variance_)))


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: Cross-Feature Interactions & Transitions (Pure Math)
# ─────────────────────────────────────────────────────────────────────────────
class TestTier3_Interactions_Math(unittest.TestCase):
    """Tier 3: Multi-sensor dynamics, fault switching, and retraining stability."""

    @classmethod
    def setUpClass(cls):
        _, cls.scaler, cls.pca, cls.ucl_t2, cls.ucl_q = simulate_training_pipeline()

    def test_tc_t3_int_01_clean_water_nominal_behavior(self):
        """Verify clean water baseline operates with nominal T2 and Q below UCL."""
        raw = REF_PHYSICAL_MEANS.copy()
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertLess(T2, self.ucl_t2)
        self.assertLess(Q, self.ucl_q)
        # In clean water, no single sensor should command > 50% contribution
        self.assertLess(float(np.max(contrib)), 0.50)

    def test_tc_t3_int_02_nacl_saline_isolation(self):
        """Verify NaCl injection causes T2 alarm and isolates TDS sensor (>95%)."""
        raw = REF_PHYSICAL_MEANS + REF_FAULTS['NaCl (Salt Solution)']
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertGreater(T2, self.ucl_t2)
        self.assertEqual(int(np.argmax(contrib)), 1)  # TDS index 1
        self.assertGreater(contrib[1], 0.95)

    def test_tc_t3_int_03_acetic_acid_isolation(self):
        """Verify Acetic Acid causes Q/T2 alarm and isolates pH sensor (>90%)."""
        raw = REF_PHYSICAL_MEANS + REF_FAULTS['Acetic Acid (Vinegar)']
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertGreater(Q, self.ucl_q)
        self.assertEqual(int(np.argmax(contrib)), 3)  # pH index 3
        self.assertGreater(contrib[3], 0.90)

    def test_tc_t3_int_04_milk_protein_isolation(self):
        """Verify Milk Protein causes Q/T2 alarm and isolates Turbidity sensor (>90%)."""
        raw = REF_PHYSICAL_MEANS + REF_FAULTS['Milk Protein']
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertGreater(T2, self.ucl_t2)
        self.assertGreater(Q, self.ucl_q)
        self.assertEqual(int(np.argmax(contrib)), 2)  # Turb index 2
        self.assertGreater(contrib[2], 0.90)

    def test_tc_t3_int_05_sensor_drift_isolation(self):
        """Verify Hardware Sensor Drift isolates Temp sensor (>95%)."""
        raw = REF_PHYSICAL_MEANS + REF_FAULTS['Sensor Drift (Hardware)']
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertGreater(T2, self.ucl_t2)
        self.assertEqual(int(np.argmax(contrib)), 0)  # Temp index 0
        self.assertGreater(contrib[0], 0.95)

    def test_tc_t3_int_06_sequential_fault_swapping(self):
        """Verify sequential fault transitions correctly update dominant sensor."""
        fault_order = [
            ('Normal (Clean Water)', None),
            ('NaCl (Salt Solution)', 1),
            ('Acetic Acid (Vinegar)', 3),
            ('Milk Protein', 2),
            ('Sensor Drift (Hardware)', 0),
            ('Normal (Clean Water)', None),
        ]
        for fault_name, expected_dom_idx in fault_order:
            raw = REF_PHYSICAL_MEANS + REF_FAULTS[fault_name]
            _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
            if expected_dom_idx is not None:
                self.assertEqual(int(np.argmax(contrib)), expected_dom_idx)
                self.assertTrue(T2 > self.ucl_t2 or Q > self.ucl_q)

    def test_tc_t3_int_07_retrain_reproducibility(self):
        """Verify running training twice with seed 42 produces identical parameters."""
        _, s1, p1, ucl_t2_1, ucl_q_1 = simulate_training_pipeline(seed=42)
        _, s2, p2, ucl_t2_2, ucl_q_2 = simulate_training_pipeline(seed=42)
        self.assertAlmostEqual(ucl_t2_1, ucl_t2_2, places=8)
        self.assertAlmostEqual(ucl_q_1, ucl_q_2, places=8)
        np.testing.assert_allclose(s1.mean_, s2.mean_)
        np.testing.assert_allclose(p1.components_, p2.components_)

    def test_tc_t3_int_08_ellipse_rescaling_with_ucl(self):
        """Verify ellipse axes scale proportionally with sqrt(UCL)."""
        ev = self.pca.explained_variance_
        a_orig = np.sqrt(ev[0] * self.ucl_t2)
        # If UCL is scaled by 4x, semi-major axis must scale by 2x
        scaled_ucl = self.ucl_t2 * 4.0
        a_scaled = np.sqrt(ev[0] * scaled_ucl)
        self.assertAlmostEqual(a_scaled, a_orig * 2.0, places=6)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4: Academic & Student Lab Scenarios (Pure Math)
# ─────────────────────────────────────────────────────────────────────────────
class TestTier4_LabScenarios_Math(unittest.TestCase):
    """Tier 4: Student lab workflows, Monte Carlo false alarm tests, and domain physics."""

    @classmethod
    def setUpClass(cls):
        cls.raw_train, cls.scaler, cls.pca, cls.ucl_t2, cls.ucl_q = simulate_training_pipeline()

    def test_tc_t4_scn_01_calibration_tolerance(self):
        """Scenario 1: Lab baseline calibration tolerances match instrument specs."""
        # Temperature ±0.5°C, TDS ±5 ppm, Turbidity ±0.5 NTU, pH ±0.1
        self.assertTrue(24.5 <= self.scaler.mean_[0] <= 25.5)
        self.assertTrue(145.0 <= self.scaler.mean_[1] <= 155.0)
        self.assertTrue(4.5 <= self.scaler.mean_[2] <= 5.5)
        self.assertTrue(6.9 <= self.scaler.mean_[3] <= 7.1)

    def test_tc_t4_scn_02_noc_steady_state_50_ticks(self):
        """Scenario 2: Steady state clean water operation over 50 samples."""
        rng = np.random.default_rng(seed=123)
        alarms = 0
        for _ in range(50):
            noise = rng.standard_normal(4) * REF_PHYSICAL_STDS
            raw = REF_PHYSICAL_MEANS + noise
            _, _, T2, Q, _ = evaluate_inference(raw, self.scaler, self.pca)
            if T2 > self.ucl_t2 or Q > self.ucl_q:
                alarms += 1
        # False alarm rate should be small (<= 3 out of 50)
        self.assertLessEqual(alarms, 3)

    def test_tc_t4_scn_03_nacl_saline_lab_experiment(self):
        """Scenario 3: Saline experiment student validation."""
        raw = REF_PHYSICAL_MEANS + REF_FAULTS['NaCl (Salt Solution)']
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertGreater(T2, 600.0)  # Strong PCS shift
        self.assertGreater(contrib[1], 0.95)  # TDS dominant

    def test_tc_t4_scn_04_acetic_acid_lab_experiment(self):
        """Scenario 4: Acid spill experiment student validation."""
        raw = REF_PHYSICAL_MEANS + REF_FAULTS['Acetic Acid (Vinegar)']
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertGreater(Q, 1000.0)  # Strong RS correlation breakdown
        self.assertGreater(contrib[3], 0.90)  # pH dominant

    def test_tc_t4_scn_05_milk_protein_lab_experiment(self):
        """Scenario 5: Biological organic turbidity experiment student validation."""
        raw = REF_PHYSICAL_MEANS + REF_FAULTS['Milk Protein']
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertGreater(T2, 1000.0)
        self.assertGreater(Q, 1000.0)
        self.assertGreater(contrib[2], 0.90)  # Turb dominant

    def test_tc_t4_scn_06_hardware_sensor_drift_lab_experiment(self):
        """Scenario 6: Hardware drift vs real water contamination experiment."""
        raw = REF_PHYSICAL_MEANS + REF_FAULTS['Sensor Drift (Hardware)']
        _, _, T2, Q, contrib = evaluate_inference(raw, self.scaler, self.pca)
        self.assertGreater(T2, 100.0)
        self.assertEqual(int(np.argmax(contrib)), 0)  # Temp dominant
        self.assertGreater(contrib[0], 0.95)

    def test_tc_t4_scn_07_monte_carlo_false_alarm_rate(self):
        """Scenario 7: Monte Carlo validation of 99% UCL false alarm rate over 200 samples."""
        rng = np.random.default_rng(seed=2026)
        n_samples = 200
        false_alarms = 0
        for _ in range(n_samples):
            noise = rng.standard_normal(4) * REF_PHYSICAL_STDS
            raw = REF_PHYSICAL_MEANS + noise
            _, _, T2, Q, _ = evaluate_inference(raw, self.scaler, self.pca)
            if T2 > self.ucl_t2 or Q > self.ucl_q:
                false_alarms += 1
        emp_rate = false_alarms / n_samples
        # Empirical false alarm rate must be <= 3.0% (expected ~1-2%)
        self.assertLessEqual(emp_rate, 0.03, f"False alarm rate {emp_rate*100:.1f}% exceeds 3.0%")


if __name__ == '__main__':
    unittest.main()
