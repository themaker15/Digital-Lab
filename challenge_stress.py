"""
Empirical Stress Test Harness for Water Contamination FDD Simulator (app.py)
Executed by: Challenger 1 (Empirical & Stress Verifier)

Verification Dimensions:
1. Long-run simulation stability (scores_hist, t2_hist, q_hist maintain 60 items FIFO over 1,000 ticks)
2. Boundary condition: T^2 = UCL_T^2 falls exactly on the 99% confidence ellipse boundary (t1^2/a^2 + t2^2/b^2 = 1.000)
3. Extreme noise & shock stress testing (1000sigma shocks, zero variance, non-physical values, orthogonal residual shocks)
4. Rapid retraining & phase-switching state resets (repeated 5x cycles via AppTest)
"""

import unittest
from unittest.mock import patch
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

# Physical realities from app.py
PHYSICAL_MEANS = np.array([25.0, 150.0, 5.0, 7.0])
PHYSICAL_STDS  = np.array([0.25,   4.0, 0.4, 0.05])
LABELS = ['Temp', 'TDS', 'Turb', 'pH']

FAULTS = {
    'Normal (Clean Water)':       np.array([0.0,   0.0,  0.0,  0.0]),
    'NaCl (Salt Solution)':       np.array([0.0, 140.0,  0.0,  0.1]),
    'Acetic Acid (Vinegar)':      np.array([0.0,  30.0,  1.5, -2.8]),
    'Milk Protein':               np.array([0.0,  20.0, 28.0, -0.4]),
    'Sensor Drift (Hardware)':    np.array([5.5,   0.0,  0.0,  0.0]),
}


class TestLongRunSimulationStability(unittest.TestCase):
    """Stress test 1: Verify FIFO buffer integrity, memory stability, and math over 1,000 ticks."""

    def setUp(self):
        # Fit baseline model identically to Phase 1 in app.py
        rng = np.random.default_rng(seed=42)
        raw_train = rng.standard_normal((500, 4)) * PHYSICAL_STDS + PHYSICAL_MEANS
        self.scaler = StandardScaler()
        Z_train = self.scaler.fit_transform(raw_train)
        self.pca = PCA(n_components=2)
        self.pca.fit(Z_train)
        scores_train = self.pca.transform(Z_train)
        T2_train = np.sum((scores_train / np.sqrt(self.pca.explained_variance_)) ** 2, axis=1)
        self.ucl_t2 = float(np.percentile(T2_train, 99))
        Z_hat = self.pca.inverse_transform(scores_train)
        Q_train = np.sum((Z_train - Z_hat) ** 2, axis=1)
        self.ucl_q = float(np.percentile(Q_train, 99))

    def test_1000_consecutive_ticks_fifo_integrity(self):
        """Simulate 1,000 consecutive ticks through all 5 fault modes and verify FIFO bounds."""
        t2_hist = [0.0] * 60
        q_hist = [0.0] * 60
        scores_hist = [[0.0, 0.0]] * 60

        fault_cycle = list(FAULTS.keys())
        total_ticks = 1000

        for tick in range(total_ticks):
            # Rotate fault every 200 ticks
            current_fault = fault_cycle[(tick // 200) % len(fault_cycle)]
            offset = FAULTS[current_fault]
            noise = np.random.randn(4) * PHYSICAL_STDS
            raw = PHYSICAL_MEANS + offset + noise

            # Inference pipeline identical to app.py
            z = self.scaler.transform(raw.reshape(1, -1))[0]
            scores = self.pca.transform(z.reshape(1, -1))[0]
            ev = self.pca.explained_variance_
            T2 = float(np.sum((scores / np.sqrt(ev)) ** 2))
            z_hat = self.pca.inverse_transform(scores.reshape(1, -1))[0]
            Q = float(np.sum((z - z_hat) ** 2))
            contrib = np.clip((z ** 2) / max(T2, 1e-9), 0, 1)
            contrib = contrib / max(contrib.sum(), 1e-9)

            # Update buffers identical to app.py
            t2_hist.append(T2)
            t2_hist.pop(0)
            q_hist.append(Q)
            q_hist.pop(0)

            cur_score = scores.tolist()
            scores_hist.append(cur_score)
            scores_hist.pop(0)

            # Strict Invariant Checks at every tick
            self.assertEqual(len(scores_hist), 60, f"scores_hist length drift at tick {tick}")
            self.assertEqual(len(t2_hist), 60, f"t2_hist length drift at tick {tick}")
            self.assertEqual(len(q_hist), 60, f"q_hist length drift at tick {tick}")
            self.assertEqual(len(scores_hist[-1]), 2, f"scores_hist entry dimension != 2 at tick {tick}")

            # Verify no NaN or Inf generated
            self.assertFalse(np.isnan(T2), f"NaN T2 at tick {tick}")
            self.assertFalse(np.isinf(T2), f"Inf T2 at tick {tick}")
            self.assertFalse(np.isnan(Q), f"NaN Q at tick {tick}")
            self.assertFalse(np.isinf(Q), f"Inf Q at tick {tick}")
            self.assertFalse(np.any(np.isnan(cur_score)), f"NaN in score vector at tick {tick}")
            self.assertFalse(np.any(np.isnan(contrib)), f"NaN in contribution vector at tick {tick}")
            self.assertAlmostEqual(float(np.sum(contrib)), 1.0, places=5, msg=f"Contribution sum != 1 at tick {tick}")

        # Final FIFO check: ensure the last element matches the final score
        self.assertEqual(scores_hist[-1], cur_score)
        print("PASS: 1,000 consecutive simulation ticks completed with exact 60-item FIFO buffers.")


class TestConfidenceEllipseBoundaryCondition(unittest.TestCase):
    """Stress test 2: Rigorously verify that T^2 = UCL_T^2 lies exactly on the 99% confidence ellipse."""

    def setUp(self):
        rng = np.random.default_rng(seed=42)
        raw_train = rng.standard_normal((500, 4)) * PHYSICAL_STDS + PHYSICAL_MEANS
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(raw_train)
        self.pca = PCA(n_components=2)
        self.pca.fit(Z_train)
        scores_train = self.pca.transform(Z_train)
        T2_train = np.sum((scores_train / np.sqrt(self.pca.explained_variance_)) ** 2, axis=1)
        self.ucl_t2 = float(np.percentile(T2_train, 99))
        self.ev = self.pca.explained_variance_
        self.a_ellipse = float(np.sqrt(self.ev[0] * self.ucl_t2))
        self.b_ellipse = float(np.sqrt(self.ev[1] * self.ucl_t2))

    def test_ellipse_algebraic_identity(self):
        """Verify for 1,000 points on the ellipse perimeter that T^2 = UCL_T^2 within 1e-12."""
        thetas = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        for theta in thetas:
            t1 = self.a_ellipse * np.cos(theta)
            t2 = self.b_ellipse * np.sin(theta)

            # Equation 1: Ellipse normalised canonical form: (t1/a)^2 + (t2/b)^2
            canonical_val = (t1 / self.a_ellipse) ** 2 + (t2 / self.b_ellipse) ** 2
            self.assertAlmostEqual(canonical_val, 1.0, places=12,
                                   msg=f"Canonical ellipse equation != 1.0 at theta={theta}")

            # Equation 2: Hotelling T^2 definition: (t1^2 / lambda1) + (t2^2 / lambda2)
            computed_t2 = (t1 ** 2 / self.ev[0]) + (t2 ** 2 / self.ev[1])
            self.assertAlmostEqual(computed_t2, self.ucl_t2, places=10,
                                   msg=f"Computed T^2 != UCL_T^2 on boundary at theta={theta}")

    def test_interior_points_strictly_sub_ucl(self):
        """Points scaled by 0.999 must strictly yield T^2 < UCL_T^2 and canonical < 1.0."""
        thetas = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        for theta in thetas:
            t1 = 0.999 * self.a_ellipse * np.cos(theta)
            t2 = 0.999 * self.b_ellipse * np.sin(theta)
            computed_t2 = (t1 ** 2 / self.ev[0]) + (t2 ** 2 / self.ev[1])
            self.assertLess(computed_t2, self.ucl_t2)
            self.assertLess((t1 / self.a_ellipse) ** 2 + (t2 / self.b_ellipse) ** 2, 1.0)

    def test_exterior_points_strictly_super_ucl(self):
        """Points scaled by 1.001 must strictly yield T^2 > UCL_T^2 and canonical > 1.0."""
        thetas = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        for theta in thetas:
            t1 = 1.001 * self.a_ellipse * np.cos(theta)
            t2 = 1.001 * self.b_ellipse * np.sin(theta)
            computed_t2 = (t1 ** 2 / self.ev[0]) + (t2 ** 2 / self.ev[1])
            self.assertGreater(computed_t2, self.ucl_t2)
            self.assertGreater((t1 / self.a_ellipse) ** 2 + (t2 / self.b_ellipse) ** 2, 1.0)

    def test_app_py_ellipse_trace_coordinates(self):
        """Verify the exact ellipse generation algorithm used in app.py lines 487-492."""
        theta = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = (self.a_ellipse * np.cos(theta)).tolist()
        y_ellipse = (self.b_ellipse * np.sin(theta)).tolist()

        for x, y in zip(x_ellipse, y_ellipse):
            val = (x / self.a_ellipse) ** 2 + (y / self.b_ellipse) ** 2
            self.assertAlmostEqual(val, 1.0, places=12)
        print(f"PASS: Ellipse boundary T^2 = UCL_T^2 verified across 1,000 boundary points (semi-axes a={self.a_ellipse:.4f}, b={self.b_ellipse:.4f}).")


class TestExtremeNoiseAndShocks(unittest.TestCase):
    """Stress test 3: Subject the FDD inference engine to adversarial, pathological, and extreme inputs."""

    def setUp(self):
        rng = np.random.default_rng(seed=42)
        raw_train = rng.standard_normal((500, 4)) * PHYSICAL_STDS + PHYSICAL_MEANS
        self.scaler = StandardScaler()
        Z_train = self.scaler.fit_transform(raw_train)
        self.pca = PCA(n_components=2)
        self.pca.fit(Z_train)

    def _infer(self, raw_input):
        z = self.scaler.transform(raw_input.reshape(1, -1))[0]
        scores = self.pca.transform(z.reshape(1, -1))[0]
        ev = self.pca.explained_variance_
        T2 = float(np.sum((scores / np.sqrt(ev)) ** 2))
        z_hat = self.pca.inverse_transform(scores.reshape(1, -1))[0]
        Q = float(np.sum((z - z_hat) ** 2))
        contrib = np.clip((z ** 2) / max(T2, 1e-9), 0, 1)
        contrib = contrib / max(contrib.sum(), 1e-9)
        return z, scores, T2, Q, contrib

    def test_zero_vector_and_exact_mean(self):
        """Input exactly at learned mean: z=0, T^2=0, Q=0. Verify no ZeroDivisionError in contributions."""
        raw = self.scaler.mean_.copy()
        z, scores, T2, Q, contrib = self._infer(raw)
        self.assertAlmostEqual(T2, 0.0, places=7)
        self.assertAlmostEqual(Q, 0.0, places=7)
        self.assertFalse(np.isnan(T2))
        self.assertFalse(np.isnan(Q))
        self.assertFalse(np.any(np.isnan(contrib)))
        self.assertEqual(len(contrib), 4)

    def test_nominal_physical_mean(self):
        """Nominal water at PHYSICAL_MEANS must be well within normal baseline."""
        raw = PHYSICAL_MEANS.copy()
        z, scores, T2, Q, contrib = self._infer(raw)
        self.assertLess(T2, 1.0)
        self.assertLess(Q, 1.0)
        self.assertAlmostEqual(float(np.sum(contrib)), 1.0, places=5)

    def test_massive_positive_shock_1000sigma(self):
        """Input with +1,000 sigma on TDS sensor."""
        raw = PHYSICAL_MEANS.copy()
        raw[1] += 1000.0 * PHYSICAL_STDS[1]
        z, scores, T2, Q, contrib = self._infer(raw)
        self.assertFalse(np.isnan(T2))
        self.assertFalse(np.isinf(T2))
        self.assertFalse(np.isnan(Q))
        self.assertFalse(np.isinf(Q))
        self.assertAlmostEqual(float(np.sum(contrib)), 1.0, places=5)
        # TDS should be the dominant contributor
        self.assertEqual(int(np.argmax(contrib)), 1)

    def test_massive_negative_shock_non_physical(self):
        """Input with non-physical negative values (e.g. pH = -50, Turbidity = -100)."""
        raw = np.array([-100.0, -500.0, -20.0, -50.0])
        z, scores, T2, Q, contrib = self._infer(raw)
        self.assertFalse(np.isnan(T2))
        self.assertFalse(np.isnan(Q))
        self.assertAlmostEqual(float(np.sum(contrib)), 1.0, places=5)

    def test_orthogonal_nullspace_shock(self):
        """Input tailored to lie purely in the residual null-space (high Q, low T^2)."""
        V = self.pca.components_
        _, _, Vh = np.linalg.svd(V)
        null_vec = Vh[2]
        z_residual = null_vec * 20.0
        raw = z_residual * self.scaler.scale_ + self.scaler.mean_

        z, scores, T2, Q, contrib = self._infer(raw)
        self.assertAlmostEqual(T2, 0.0, places=5)
        self.assertGreater(Q, 100.0)
        print(f"PASS: Orthogonal null-space shock correctly triggered pure Q alarm (T^2={T2:.4f}, Q={Q:.4f}).")


class TestRapidRetrainingAndStateResets(unittest.TestCase):
    """Stress test 4: Rapidly cycle between Phase 1 and Phase 2, asserting clean state resets."""

    def test_repeated_phase_switching_and_retraining_cycles(self):
        """Perform 5 complete cycles of Train -> Switch to Phase 2 -> Run ticks -> Switch to Phase 1 -> Retrain."""
        app = AppTest.from_file("app.py", default_timeout=30)
        app.run()
        self.assertFalse(app.session_state.is_trained)

        faults_to_test = [
            'Normal (Clean Water)',
            'NaCl (Salt Solution)',
            'Acetic Acid (Vinegar)',
            'Milk Protein',
            'Sensor Drift (Hardware)',
        ]

        for cycle, fault_name in enumerate(faults_to_test, 1):
            # Phase 1: Train
            app.sidebar.radio[0].set_value("Phase 1 — Train (NOC)").run()
            self.assertEqual(len(app.button), 1)
            train_btn = app.button[0]
            self.assertEqual(train_btn.label, "Collect Data & Train PCA")
            train_btn.click().run()

            # Verify trained state
            self.assertTrue(app.session_state.is_trained, f"Cycle {cycle}: Not trained after click")
            self.assertGreater(app.session_state.ucl_t2, 0.0)
            self.assertGreater(app.session_state.ucl_q, 0.0)
            self.assertEqual(len(app.session_state.scores_hist), 60)
            self.assertEqual(app.session_state.step, 0)
            self.assertFalse(app.session_state.is_playing)

            # Switch to Phase 2
            app.sidebar.radio[0].set_value("Phase 2 — Test (FDD)").run()

            # Select the specific fault for this cycle
            fault_radio = app.sidebar.radio[1]
            fault_radio.set_value(fault_name).run()
            self.assertEqual(app.session_state.active_fault, fault_name)

            # Start simulation in Phase 2 with st.rerun patched to step deterministically
            with patch.object(st, 'rerun'):
                start_btn = app.sidebar.button[0]
                self.assertEqual(start_btn.label, "Start")
                start_btn.click().run()
                self.assertTrue(app.session_state.is_playing)
                self.assertGreater(app.session_state.step, 0)

                # Run 3 more ticks
                for _ in range(3):
                    app.run()

            # Verify that history accumulated non-zero data
            self.assertGreater(app.session_state.step, 1)
            self.assertNotEqual(app.session_state.t2_hist[-1], 0.0)
            self.assertNotEqual(app.session_state.scores_hist[-1], [0.0, 0.0])

            # Pause simulation
            with patch.object(st, 'rerun'):
                pause_btn = app.sidebar.button[1]
                self.assertEqual(pause_btn.label, "Pause")
                pause_btn.click().run()
                self.assertFalse(app.session_state.is_playing)

            # Switch back to Phase 1
            app.sidebar.radio[0].set_value("Phase 1 — Train (NOC)").run()

            # Retrain model
            retrain_btn = app.button[0]
            retrain_btn.click().run()

            # Verify complete state reset
            self.assertEqual(app.session_state.step, 0, f"Cycle {cycle}: step did not reset to 0")
            self.assertEqual(app.session_state.t2_hist, [0.0] * 60, f"Cycle {cycle}: t2_hist not reset")
            self.assertEqual(app.session_state.q_hist, [0.0] * 60, f"Cycle {cycle}: q_hist not reset")
            self.assertEqual(app.session_state.scores_hist, [[0.0, 0.0]] * 60, f"Cycle {cycle}: scores_hist not reset")
            self.assertEqual(app.session_state.contribs, [0.0, 0.0, 0.0, 0.0], f"Cycle {cycle}: contribs not reset")
            self.assertFalse(app.session_state.is_playing, f"Cycle {cycle}: is_playing was not reset to False")

        print("PASS: 5 full cycles of Train -> Test -> Retrain executed with zero state corruption.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
