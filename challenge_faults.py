"""
Adversarial Fault Injection & Telemetry Empirical Stress Testing Harness
Water Contamination FDD Simulator (app.py)

Challenger 2 Verification Suite:
- Empirical Monte Carlo Fault Injection (100 iterations per mode under realistic & amplified noise)
- Root Cause Isolation & Contribution Thresholds (>80%)
- SVG Digital Twin Water Color Transitions & CSS Animation Timings (<=2s)
- Telemetry Metric Deltas Accuracy (cur - prev)
- Boundary & Stress Transitions

Rule Compliance:
- Strictly NO unicode bullet dots (•), ONLY standard hyphens (-)
- Layout compliance: placed in project root, not in .agents/
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import patch

# UTF-8 terminal output safety
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import streamlit as st
from streamlit.testing.v1 import AppTest

# Import physics constants and helper from test_math_fdd
from test_math_fdd import (
    simulate_training_pipeline,
    evaluate_inference,
    REF_PHYSICAL_MEANS,
    REF_PHYSICAL_STDS,
    REF_LABELS,
    REF_FAULTS,
)
from test_streamlit_app import (
    APP_PATH,
    create_fresh_app,
    seed_trained_state,
    create_phase2_app,
    extract_svg_html,
)


EXPECTED_WATER_COLORS = {
    'Normal (Clean Water)':    'rgba(59,130,246,0.28)',
    'NaCl (Salt Solution)':    'rgba(245,158,11,0.40)',
    'Acetic Acid (Vinegar)':   'rgba(239,68,68,0.32)',
    'Milk Protein':            'rgba(220,224,232,0.85)',
    'Sensor Drift (Hardware)': 'rgba(59,130,246,0.28)',
}

TARGET_ROOT_CAUSES = {
    'NaCl (Salt Solution)':       ('TDS', 1),
    'Acetic Acid (Vinegar)':      ('pH', 3),
    'Milk Protein':               ('Turb', 2),
    'Sensor Drift (Hardware)':    ('Temp', 0),
}


def safe_step_simulation(at, steps=1):
    """Steps simulation while mocking time.sleep and st.rerun to prevent hangs."""
    at.session_state['is_playing'] = True
    with patch('time.sleep', return_value=None), patch.object(st, 'rerun'):
        for _ in range(steps):
            at.run()
    return at


# =============================================================================
# Suite 1: Monte Carlo Fault Injection & Root Cause Isolation
# =============================================================================
class TestAdversarialFaultInjection(unittest.TestCase):
    """Adversarial Monte Carlo evaluation of detection rates and contribution isolation."""

    @classmethod
    def setUpClass(cls):
        cls.raw_train, cls.scaler, cls.pca, cls.ucl_t2, cls.ucl_q = simulate_training_pipeline(seed=42)

    def test_01_normal_clean_water_monte_carlo(self):
        """Normal condition under 100 noisy iterations: false alarm rate < 5%."""
        rng = np.random.default_rng(seed=1001)
        n_iters = 100
        alarms = 0
        for _ in range(n_iters):
            noise = rng.standard_normal(4) * REF_PHYSICAL_STDS
            raw = REF_PHYSICAL_MEANS + noise
            _, _, t2, q, _ = evaluate_inference(raw, self.scaler, self.pca)
            if t2 > self.ucl_t2 or q > self.ucl_q:
                alarms += 1
        false_alarm_rate = alarms / n_iters
        print(f" [Normal] False Alarm Rate: {false_alarm_rate*100:.1f}% ({alarms}/{n_iters})")
        self.assertLess(false_alarm_rate, 0.05, f"False alarm rate {false_alarm_rate*100:.1f}% >= 5.0%")

    def test_02_nacl_fault_monte_carlo(self):
        """NaCl fault under 100 noisy iterations: detection rate > 99%, Root Cause TDS > 80%."""
        rng = np.random.default_rng(seed=1002)
        n_iters = 100
        detections = 0
        correct_root_causes = 0
        contrib_values = []
        offset = REF_FAULTS['NaCl (Salt Solution)']

        for _ in range(n_iters):
            noise = rng.standard_normal(4) * REF_PHYSICAL_STDS
            raw = REF_PHYSICAL_MEANS + offset + noise
            _, _, t2, q, contrib = evaluate_inference(raw, self.scaler, self.pca)
            is_detected = (t2 > self.ucl_t2 or q > self.ucl_q)
            if is_detected:
                detections += 1
            max_sensor_idx = int(np.argmax(contrib))
            if max_sensor_idx == 1:
                correct_root_causes += 1
            contrib_values.append(contrib[1])

        det_rate = detections / n_iters
        rc_rate = correct_root_causes / n_iters
        mean_contrib = float(np.mean(contrib_values))
        min_contrib = float(np.min(contrib_values))
        print(f" [NaCl] Detection: {det_rate*100:.1f}%, Root Cause TDS: {rc_rate*100:.1f}%, Min Contrib: {min_contrib*100:.1f}%, Mean Contrib: {mean_contrib*100:.1f}%")

        self.assertGreater(det_rate, 0.99, f"NaCl detection rate {det_rate*100:.1f}% <= 99%")
        self.assertEqual(rc_rate, 1.0, f"NaCl root cause isolation rate {rc_rate*100:.1f}% < 100%")
        self.assertGreater(min_contrib, 0.80, f"NaCl minimum TDS contribution {min_contrib*100:.2f}% <= 80%")

    def test_03_acetic_acid_fault_monte_carlo(self):
        """Acetic Acid under 100 noisy iterations: detection rate > 99%, Root Cause pH > 80%."""
        rng = np.random.default_rng(seed=1003)
        n_iters = 100
        detections = 0
        correct_root_causes = 0
        contrib_values = []
        offset = REF_FAULTS['Acetic Acid (Vinegar)']

        for _ in range(n_iters):
            noise = rng.standard_normal(4) * REF_PHYSICAL_STDS
            raw = REF_PHYSICAL_MEANS + offset + noise
            _, _, t2, q, contrib = evaluate_inference(raw, self.scaler, self.pca)
            is_detected = (t2 > self.ucl_t2 or q > self.ucl_q)
            if is_detected:
                detections += 1
            max_sensor_idx = int(np.argmax(contrib))
            if max_sensor_idx == 3:
                correct_root_causes += 1
            contrib_values.append(contrib[3])

        det_rate = detections / n_iters
        rc_rate = correct_root_causes / n_iters
        mean_contrib = float(np.mean(contrib_values))
        min_contrib = float(np.min(contrib_values))
        print(f" [Acetic Acid] Detection: {det_rate*100:.1f}%, Root Cause pH: {rc_rate*100:.1f}%, Min Contrib: {min_contrib*100:.1f}%, Mean Contrib: {mean_contrib*100:.1f}%")

        self.assertGreater(det_rate, 0.99, f"Acetic Acid detection rate {det_rate*100:.1f}% <= 99%")
        self.assertEqual(rc_rate, 1.0, f"Acetic Acid root cause isolation rate {rc_rate*100:.1f}% < 100%")
        self.assertGreater(min_contrib, 0.80, f"Acetic Acid minimum pH contribution {min_contrib*100:.2f}% <= 80%")

    def test_04_milk_protein_fault_monte_carlo(self):
        """Milk Protein under 100 noisy iterations: detection rate > 99%, Root Cause Turbidity > 80%."""
        rng = np.random.default_rng(seed=1004)
        n_iters = 100
        detections = 0
        correct_root_causes = 0
        contrib_values = []
        offset = REF_FAULTS['Milk Protein']

        for _ in range(n_iters):
            noise = rng.standard_normal(4) * REF_PHYSICAL_STDS
            raw = REF_PHYSICAL_MEANS + offset + noise
            _, _, t2, q, contrib = evaluate_inference(raw, self.scaler, self.pca)
            is_detected = (t2 > self.ucl_t2 or q > self.ucl_q)
            if is_detected:
                detections += 1
            max_sensor_idx = int(np.argmax(contrib))
            if max_sensor_idx == 2:
                correct_root_causes += 1
            contrib_values.append(contrib[2])

        det_rate = detections / n_iters
        rc_rate = correct_root_causes / n_iters
        mean_contrib = float(np.mean(contrib_values))
        min_contrib = float(np.min(contrib_values))
        print(f" [Milk Protein] Detection: {det_rate*100:.1f}%, Root Cause Turb: {rc_rate*100:.1f}%, Min Contrib: {min_contrib*100:.1f}%, Mean Contrib: {mean_contrib*100:.1f}%")

        self.assertGreater(det_rate, 0.99, f"Milk Protein detection rate {det_rate*100:.1f}% <= 99%")
        self.assertEqual(rc_rate, 1.0, f"Milk Protein root cause isolation rate {rc_rate*100:.1f}% < 100%")
        self.assertGreater(min_contrib, 0.80, f"Milk Protein minimum Turb contribution {min_contrib*100:.2f}% <= 80%")

    def test_05_sensor_drift_fault_monte_carlo(self):
        """Sensor Drift under 100 noisy iterations: detection rate > 99%, Root Cause Temp > 80%."""
        rng = np.random.default_rng(seed=1005)
        n_iters = 100
        detections = 0
        correct_root_causes = 0
        contrib_values = []
        offset = REF_FAULTS['Sensor Drift (Hardware)']

        for _ in range(n_iters):
            noise = rng.standard_normal(4) * REF_PHYSICAL_STDS
            raw = REF_PHYSICAL_MEANS + offset + noise
            _, _, t2, q, contrib = evaluate_inference(raw, self.scaler, self.pca)
            is_detected = (t2 > self.ucl_t2 or q > self.ucl_q)
            if is_detected:
                detections += 1
            max_sensor_idx = int(np.argmax(contrib))
            if max_sensor_idx == 0:
                correct_root_causes += 1
            contrib_values.append(contrib[0])

        det_rate = detections / n_iters
        rc_rate = correct_root_causes / n_iters
        mean_contrib = float(np.mean(contrib_values))
        min_contrib = float(np.min(contrib_values))
        print(f" [Sensor Drift] Detection: {det_rate*100:.1f}%, Root Cause Temp: {rc_rate*100:.1f}%, Min Contrib: {min_contrib*100:.1f}%, Mean Contrib: {mean_contrib*100:.1f}%")

        self.assertGreater(det_rate, 0.99, f"Sensor Drift detection rate {det_rate*100:.1f}% <= 99%")
        self.assertEqual(rc_rate, 1.0, f"Sensor Drift root cause isolation rate {rc_rate*100:.1f}% < 100%")
        self.assertGreater(min_contrib, 0.80, f"Sensor Drift minimum Temp contribution {min_contrib*100:.2f}% <= 80%")

    def test_06_amplified_noise_stress_test(self):
        """Adversarial stress: 2.5x standard deviation noise on all 4 fault conditions."""
        rng = np.random.default_rng(seed=2026)
        n_iters = 100
        print(" [Stress Test] 2.5x Noise Amplification Stress Test:")
        for fault_name, (target_sensor, target_idx) in TARGET_ROOT_CAUSES.items():
            offset = REF_FAULTS[fault_name]
            detections = 0
            correct_rc = 0
            for _ in range(n_iters):
                noise = rng.standard_normal(4) * REF_PHYSICAL_STDS * 2.5
                raw = REF_PHYSICAL_MEANS + offset + noise
                _, _, t2, q, contrib = evaluate_inference(raw, self.scaler, self.pca)
                if t2 > self.ucl_t2 or q > self.ucl_q:
                    detections += 1
                if int(np.argmax(contrib)) == target_idx:
                    correct_rc += 1
            print(f"  - {fault_name}: Detection {detections}/{n_iters}, Root Cause {target_sensor} {correct_rc}/{n_iters}")
            self.assertGreaterEqual(detections, 95, f"Detection under 2.5x noise failed for {fault_name}")
            self.assertGreaterEqual(correct_rc, 95, f"Root cause under 2.5x noise failed for {fault_name}")


# =============================================================================
# Suite 2: SVG Water Color Transitions & Timing Verification
# =============================================================================
class TestSvgWaterColorTransitions(unittest.TestCase):
    """Verify SVG digital twin water color rendering, CSS transition timing, and fault drops."""

    def test_01_all_water_colors_rendered_correctly(self):
        """Verify each fault mode injects its exact designated color into the SVG body."""
        at = create_phase2_app()
        for fault_name, expected_color in EXPECTED_WATER_COLORS.items():
            at.session_state['active_fault'] = fault_name
            at.run()
            svg_html = extract_svg_html(at)
            self.assertIn(expected_color, svg_html, f"Expected color {expected_color} missing for {fault_name}")
            # Verify transition duration in CSS is <= 2s (here style='transition:fill 1s ease;')
            self.assertIn("transition:fill 1s ease", svg_html, "CSS fill transition rule missing or altered")

    def test_02_contamination_drop_indicator_conditional_display(self):
        """Verify contamination drop circle is present only when fault != Normal."""
        at = create_phase2_app()
        
        # 1. Normal clean water -> No drop
        at.session_state['active_fault'] = 'Normal (Clean Water)'
        at.run()
        svg_normal = extract_svg_html(at)
        self.assertNotIn('<circle cx="225" cy="115" r="5"', svg_normal)

        # 2. Contaminants -> Drop present with corresponding color
        for fault_name in ['NaCl (Salt Solution)', 'Acetic Acid (Vinegar)', 'Milk Protein', 'Sensor Drift (Hardware)']:
            at.session_state['active_fault'] = fault_name
            at.run()
            svg_fault = extract_svg_html(at)
            expected_color = EXPECTED_WATER_COLORS[fault_name]
            drop_tag = f'<circle cx="225" cy="115" r="5" fill="{expected_color}" opacity="0.9"/>'
            self.assertIn(drop_tag, svg_fault, f"Drop circle missing or incorrect for {fault_name}")

    def test_03_status_led_indicator_and_text(self):
        """Verify ESP32 status LED shows ALARM (red) vs ONLINE (green)."""
        at = create_phase2_app()

        # Normal clean state (online, green)
        at.session_state['active_fault'] = 'Normal (Clean Water)'
        at.session_state['t2_hist'] = [0.0] * 60
        at.session_state['q_hist'] = [0.0] * 60
        at.run()
        svg_nominal = extract_svg_html(at)
        self.assertIn('ONLINE', svg_nominal)
        self.assertIn('#10B981', svg_nominal)

        # Alarm state (alarm, red)
        at.session_state['t2_hist'] = [500.0] * 60
        at.run()
        svg_alarm = extract_svg_html(at)
        self.assertIn('ALARM', svg_alarm)
        self.assertIn('#DC2626', svg_alarm)


# =============================================================================
# Suite 3: Telemetry Metric Deltas Verification
# =============================================================================
class TestTelemetryMetricDeltas(unittest.TestCase):
    """Verify telemetry metric deltas (delta = round(cur - prev, 3)) across simulation steps."""

    def test_01_step_zero_has_empty_or_none_delta(self):
        """At step 0 (initial state), metric deltas must be None/empty."""
        at = create_phase2_app(is_playing=False)
        self.assertEqual(at.session_state['step'], 0)
        metrics = at.metric
        self.assertGreaterEqual(len(metrics), 4)
        for m in metrics[:4]:
            self.assertIn(m.delta, (None, ''), f"Expected empty delta at step 0 for {m.label}, got '{m.delta}'")

    def test_02_step_one_computes_exact_deltas(self):
        """At step 1, metric deltas must strictly equal round(cur - prev, 3)."""
        at = create_phase2_app(is_playing=False)
        safe_step_simulation(at, steps=1)
        self.assertEqual(at.session_state['step'], 1)

        cur_vals = at.session_state['raw_vals']
        prev_vals = at.session_state['prev_raw']
        metrics = at.metric[:4]

        for i, m in enumerate(metrics):
            expected_delta = round(cur_vals[i] - prev_vals[i], 3)
            actual_delta = float(m.delta)
            self.assertAlmostEqual(
                actual_delta,
                expected_delta,
                places=3,
                msg=f"Delta mismatch for sensor {REF_LABELS[i]}: got {actual_delta}, expected {expected_delta}"
            )
            expected_val_str = f"{cur_vals[i]:.2f}"
            self.assertEqual(m.value, expected_val_str)

    def test_03_multi_step_delta_continuity(self):
        """Verify delta computation across 5 consecutive ticks under active sensor loop."""
        at = create_phase2_app(is_playing=False)
        for s in range(1, 6):
            safe_step_simulation(at, steps=1)
            self.assertEqual(at.session_state['step'], s)
            cur = at.session_state['raw_vals']
            prev = at.session_state['prev_raw']
            for i, m in enumerate(at.metric[:4]):
                expected_delta = round(cur[i] - prev[i], 3)
                self.assertAlmostEqual(
                    float(m.delta),
                    expected_delta,
                    places=3,
                    msg=f"Tick {s}: delta mismatch for {REF_LABELS[i]}"
                )


def run_all_challenge_tests():
    """Runs all Challenger 2 test suites."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialFaultInjection))
    suite.addTests(loader.loadTestsFromTestCase(TestSvgWaterColorTransitions))
    suite.addTests(loader.loadTestsFromTestCase(TestTelemetryMetricDeltas))

    print("=" * 80)
    print(" CHALLENGER 2: ADVERSARIAL FAULT INJECTION & TELEMETRY VERIFICATION")
    print("=" * 80)

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("-" * 80)
    print(f" Summary: {result.testsRun} tests run, {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_challenge_tests()
    sys.exit(0 if success else 1)
