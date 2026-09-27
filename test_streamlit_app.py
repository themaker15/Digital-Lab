"""
Test Suite: Layer B - Headless Streamlit Opaque-Box UI Harness
Water Contamination FDD Simulator

Covers Tiers 1-4 for Streamlit UI, widgets, state management,
Phase 1 & Phase 2 transitions, live simulation loop stepping (with mock st.rerun),
SVG digital twin, Plotly charts, and student lab workflows.
"""

import os
import re
import json
import unittest
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

from test_math_fdd import simulate_training_pipeline, REF_PHYSICAL_MEANS, REF_FAULTS


APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app.py'))
CONFIG_TOML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.streamlit', 'config.toml'))


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures & Helper Utilities
# ─────────────────────────────────────────────────────────────────────────────
def create_fresh_app():
    """Initialises a fresh AppTest instance from app.py."""
    return AppTest.from_file(APP_PATH, default_timeout=15)


def seed_trained_state(at):
    """Pre-seeds trained scaler, PCA model, and UCL parameters into session_state."""
    _, scaler, pca, ucl_t2, ucl_q = simulate_training_pipeline(seed=42)
    at.session_state['is_trained'] = True
    at.session_state['scaler'] = scaler
    at.session_state['pca_model'] = pca
    at.session_state['ucl_t2'] = ucl_t2
    at.session_state['ucl_q'] = ucl_q
    at.session_state['var_ratio'] = pca.explained_variance_ratio_.tolist()
    at.session_state['pca_means'] = scaler.mean_.tolist()
    at.session_state['pca_stds'] = scaler.scale_.tolist()
    return at


def create_phase2_app(is_playing=False, active_fault='Normal (Clean Water)'):
    """Initialises an AppTest instance ready in Phase 2 — Test Mode."""
    at = create_fresh_app()
    seed_trained_state(at)
    at.session_state['active_fault'] = active_fault
    at.session_state['is_playing'] = is_playing
    at.run()
    at.sidebar.radio[0].set_value('Phase 2 — Test (FDD)').run()
    return at


def step_simulation(at, steps=1):
    """Steps simulation using mock patch on st.rerun to prevent infinite loop."""
    at.session_state['is_playing'] = True
    with patch.object(st, 'rerun'):
        for _ in range(steps):
            at.run()
    return at


def extract_plotly_chart_specs(at):
    """Extracts parsed JSON specifications of all Plotly charts rendered on page."""
    specs = []
    for node in at.main:
        if getattr(node, 'type', None) == 'plotly_chart':
            if hasattr(node.proto, 'spec'):
                specs.append(json.loads(node.proto.spec))
    return specs


def extract_svg_html(at):
    """Extracts raw HTML/SVG content from the digital twin iframe component."""
    for node in at.main:
        if getattr(node, 'type', None) == 'iframe':
            if hasattr(node.proto, 'srcdoc'):
                return node.proto.srcdoc
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Feature Coverage (Headless Streamlit UI Harness)
# ─────────────────────────────────────────────────────────────────────────────
class TestTier1_FeatureCoverage_UI(unittest.TestCase):
    """Tier 1: Comprehensive feature coverage for R1, R2, R3, R4 UI contracts."""

    def test_tc_t1_r1_01_theme_configuration(self):
        """Verify .streamlit/config.toml enforces enterprise light theme."""
        self.assertTrue(os.path.exists(CONFIG_TOML_PATH), "config.toml not found")
        with open(CONFIG_TOML_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('base="light"', content.replace(' ', ''))
        self.assertIn('#F8FAFC', content.upper())

    def test_tc_t1_r1_02_no_functional_emojis_in_buttons(self):
        """Verify no button labels use unicode emojis as functional icons."""
        at = create_fresh_app().run()
        buttons = list(at.button) + list(at.sidebar.button)
        at.sidebar.radio[0].set_value('Phase 2 — Test (FDD)').run()
        buttons += list(at.button) + list(at.sidebar.button)

        emoji_pattern = re.compile(r'[\u25b6\u23f8\U0001F000-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]')
        violating_buttons = []
        for b in buttons:
            if emoji_pattern.search(b.label):
                violating_buttons.append(b.label)
        self.assertEqual(violating_buttons, [], f"Buttons containing emojis found: {violating_buttons}")

    def test_tc_t1_r1_03_typography_minimum_font_size(self):
        """Verify no font sizes smaller than 11px exist in SVG or Plotly charts."""
        at = create_phase2_app()
        svg_html = extract_svg_html(at)
        chart_specs = extract_plotly_chart_specs(at)

        # Inspect SVG font sizes
        svg_font_sizes = re.findall(r'font-size[:=]["\']?([0-9.]+)(?:px)?["\']?', svg_html)
        small_svg_fonts = [float(s) for s in svg_font_sizes if float(s) < 11.0]

        # Inspect Plotly font sizes
        small_plotly_fonts = []
        def check_fonts(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ('size', 'font_size') and isinstance(v, (int, float)) and v < 11.0:
                        small_plotly_fonts.append(v)
                    else:
                        check_fonts(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_fonts(item)
        for spec in chart_specs:
            check_fonts(spec)

        self.assertEqual(small_svg_fonts, [], f"SVG fonts < 11px found: {small_svg_fonts}")
        self.assertEqual(small_plotly_fonts, [], f"Plotly fonts < 11px found: {small_plotly_fonts}")

    def test_tc_t1_r1_04_svg_rendered_via_components_html(self):
        """Verify digital twin tank SVG is rendered via components.html iframe."""
        at = create_phase2_app()
        svg_html = extract_svg_html(at)
        self.assertIn('<svg', svg_html)
        self.assertIn('viewBox="0 0 500 320"', svg_html)

    def test_tc_t1_r1_05_sidebar_navigation_two_phases(self):
        """Verify sidebar navigation contains both phases and defaults to Phase 1."""
        at = create_fresh_app().run()
        self.assertEqual(len(at.sidebar.radio), 1)
        phase_radio = at.sidebar.radio[0]
        self.assertIn("Phase 1 — Train (NOC)", phase_radio.options)
        self.assertIn("Phase 2 — Test (FDD)", phase_radio.options)
        self.assertEqual(phase_radio.value, "Phase 1 — Train (NOC)")

    def test_tc_t1_r2_01_phase1_default_session_state(self):
        """Verify cold start initialises all 17 default session state keys."""
        at = create_fresh_app().run()
        required_keys = [
            'is_trained', 'scaler', 'pca_model', 'ucl_t2', 'ucl_q',
            'var_ratio', 'pca_means', 'pca_stds', 'step', 't2_hist',
            'q_hist', 'contribs', 'active_fault', 'is_playing',
            'raw_vals', 'prev_raw', 'train_done_flag'
        ]
        for key in required_keys:
            self.assertIn(key, at.session_state, f"Missing key: {key}")
        self.assertFalse(at.session_state.is_trained)
        self.assertEqual(at.session_state.step, 0)

    def test_tc_t1_r2_02_phase1_train_button_execution(self):
        """Verify Train button executes pipeline and sets is_trained to True."""
        at = create_fresh_app().run()
        self.assertEqual(len(at.button), 1)
        at.button[0].click().run()
        self.assertTrue(at.session_state.is_trained)
        self.assertAlmostEqual(at.session_state.ucl_t2, 8.953975, places=4)
        self.assertAlmostEqual(at.session_state.ucl_q, 8.946038, places=4)

    def test_tc_t1_r2_03_phase1_learned_parameters_display(self):
        """Verify Phase 1 displays learned metrics after training."""
        at = create_fresh_app()
        seed_trained_state(at)
        at.run()
        metrics = at.metric
        self.assertGreaterEqual(len(metrics), 4)
        metric_labels = [m.label for m in metrics]
        self.assertTrue(any('Variance explained' in lbl for lbl in metric_labels))
        self.assertTrue(any('T² UCL' in lbl for lbl in metric_labels))
        self.assertTrue(any('Q UCL' in lbl for lbl in metric_labels))

    def test_tc_t1_r2_04_phase2_sidebar_status_chip(self):
        """Verify sidebar displays trained status chip in Phase 2."""
        at = create_phase2_app()
        self.assertTrue(at.session_state.is_trained)
        self.assertTrue(any('Trained' in s.value for s in at.sidebar.success))

    def test_tc_t1_r2_05_phase2_simulation_single_step(self):
        """Verify simulation loop advances step counter and updates histories on 1 tick."""
        at = create_phase2_app()
        self.assertEqual(at.session_state.step, 0)
        step_simulation(at, steps=1)
        self.assertEqual(at.session_state.step, 1)
        self.assertNotEqual(at.session_state.t2_hist[-1], 0.0)

    def test_tc_t1_r3_01_phase2_biplot_rendered(self):
        """Verify PCA Score Biplot is rendered in Phase 2."""
        at = create_phase2_app()
        specs = extract_plotly_chart_specs(at)
        # Biplot title should be present in one of the charts
        biplot_found = False
        for spec in specs:
            title = spec.get('layout', {}).get('title', {}).get('text', '')
            if 'Biplot' in title or 'PC1' in title or 'PC2' in title:
                biplot_found = True
                break
        self.assertTrue(biplot_found, "PCA Biplot chart not found in Phase 2")

    def test_tc_t1_r3_02_phase2_plotly_showlegend_false(self):
        """Verify all Plotly charts in Phase 2 use showlegend=False."""
        at = create_phase2_app()
        specs = extract_plotly_chart_specs(at)
        self.assertGreaterEqual(len(specs), 3)
        for i, spec in enumerate(specs):
            showlegend = spec.get('layout', {}).get('showlegend')
            self.assertFalse(showlegend, f"Chart {i} does not set showlegend=False")

    def test_tc_t1_r3_03_phase2_biplot_session_state_scores_hist(self):
        """Verify scores_hist exists in session_state for rolling score buffer."""
        at = create_phase2_app()
        step_simulation(at, steps=1)
        self.assertIn('scores_hist', at.session_state, "scores_hist buffer missing in session_state")

    def test_tc_t1_r4_01_phase1_scree_plot_rendered(self):
        """Verify Scree plot is rendered with explained variance per PC after training."""
        at = create_fresh_app()
        seed_trained_state(at)
        at.run()
        specs = extract_plotly_chart_specs(at)
        self.assertGreaterEqual(len(specs), 1, "Scree plot not found")
        scree_spec = specs[0]
        data = scree_spec.get('data', [])
        self.assertGreaterEqual(len(data), 1)

    def test_tc_t1_r4_02_phase1_baseline_table_rendered(self):
        """Verify styled dataframe table of learned baselines is rendered."""
        at = create_fresh_app()
        seed_trained_state(at)
        at.run()
        self.assertGreaterEqual(len(at.dataframe), 1, "Baseline dataframe table not rendered")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Boundary & Corner Cases (Headless Streamlit UI Harness)
# ─────────────────────────────────────────────────────────────────────────────
class TestTier2_BoundaryCornerCases_UI(unittest.TestCase):
    """Tier 2: Cold starts, uninitialized state guards, and boundary limits."""

    def test_tc_t2_r1_01_cold_start_clean_load(self):
        """Verify cold start loads with zero exceptions and zero errors."""
        at = create_fresh_app().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

    def test_tc_t2_r1_02_unauthorized_phase2_guard(self):
        """Verify entering Phase 2 before training displays warning and halts cleanly."""
        at = create_fresh_app().run()
        at.sidebar.radio[0].set_value('Phase 2 — Test (FDD)').run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any('not trained' in w.value.lower() for w in at.warning))

    def test_tc_t2_r1_03_svg_clean_water_appearance(self):
        """Verify clean water SVG uses default blue and has no contaminant drop."""
        at = create_phase2_app(active_fault='Normal (Clean Water)')
        svg = extract_svg_html(at)
        self.assertIn('rgba(59,130,246,0.28)', svg.replace(' ', ''))
        # No drop rendered when clean
        self.assertNotIn('<circle cx="225" cy="115" r="5"', svg)

    def test_tc_t2_r1_04_svg_fault_water_appearance(self):
        """Verify NaCl selection changes SVG water color and renders drop."""
        at = create_phase2_app(active_fault='NaCl (Salt Solution)')
        svg = extract_svg_html(at)
        self.assertIn('rgba(245,158,11,0.40)', svg.replace(' ', ''))
        self.assertIn('<circle cx="225" cy="115" r="5"', svg)

    def test_tc_t2_r2_01_step_zero_metric_delta(self):
        """Verify step 0 does not crash metric rendering with None delta."""
        at = create_phase2_app()
        metrics = at.metric
        self.assertGreaterEqual(len(metrics), 4)
        for m in metrics:
            self.assertIsNotNone(m.value)

    def test_tc_t2_r2_02_history_buffers_fifo_length(self):
        """Verify history buffers maintain fixed length of 60 after 10 ticks."""
        at = create_phase2_app()
        step_simulation(at, steps=10)
        self.assertEqual(len(at.session_state.t2_hist), 60)
        self.assertEqual(len(at.session_state.q_hist), 60)
        self.assertEqual(at.session_state.step, 10)

    def test_tc_t2_r4_01_pre_train_no_parameter_metrics(self):
        """Verify pre-trained Phase 1 does not display parameter metrics."""
        at = create_fresh_app().run()
        self.assertEqual(len(at.metric), 0)

    def test_tc_t2_r4_02_progress_bar_cleanup(self):
        """Verify progress bar is cleared after Phase 1 training completes."""
        at = create_fresh_app().run()
        at.button[0].click().run()
        # Progress element should be empty
        self.assertEqual(len(at.get('progress')), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: Cross-Feature Interactions (Headless Streamlit UI Harness)
# ─────────────────────────────────────────────────────────────────────────────
class TestTier3_Interactions_UI(unittest.TestCase):
    """Tier 3: Full UI state transitions, fault swapping, and retrain resets."""

    def test_tc_t3_int_01_full_lifecycle_navigation(self):
        """Verify complete user navigation cycle across Phase 1 and Phase 2."""
        at = create_fresh_app().run()
        # 1. Train in Phase 1
        at.button[0].click().run()
        self.assertTrue(at.session_state.is_trained)
        # 2. Switch to Phase 2
        at.sidebar.radio[0].set_value('Phase 2 — Test (FDD)').run()
        self.assertIn('Phase 2', at.title[0].value)
        # 3. Step 1 tick
        step_simulation(at, steps=1)
        self.assertEqual(at.session_state.step, 1)
        # 4. Switch back to Phase 1
        at.sidebar.radio[0].set_value('Phase 1 — Train (NOC)').run()
        self.assertIn('Phase 1', at.title[0].value)

    def test_tc_t3_int_02_pause_preserves_step_and_histories(self):
        """Verify pausing stops step increment and preserves history values."""
        at = create_phase2_app()
        step_simulation(at, steps=3)
        self.assertEqual(at.session_state.step, 3)
        # Pause
        at.session_state['is_playing'] = False
        at.run()
        self.assertEqual(at.session_state.step, 3)
        self.assertFalse(at.session_state.is_playing)

    def test_tc_t3_int_03_nacl_injection_triggers_alarm_banner(self):
        """Verify injecting NaCl produces ALARM banner isolating TDS within 1 tick."""
        at = create_phase2_app()
        # Select NaCl
        at.session_state['active_fault'] = 'NaCl (Salt Solution)'
        step_simulation(at, steps=1)
        self.assertEqual(len(at.error), 1)
        self.assertIn("ALARM", at.error[0].value)
        self.assertIn("TDS", at.error[0].value)

    def test_tc_t3_int_04_sequential_fault_switching_ui(self):
        """Verify switching faults updates UI alarm text dynamically."""
        at = create_phase2_app()
        # Acetic Acid -> pH alarm
        at.session_state['active_fault'] = 'Acetic Acid (Vinegar)'
        step_simulation(at, steps=1)
        self.assertIn("pH", at.error[0].value)

        # Milk Protein -> Turb alarm
        at.session_state['active_fault'] = 'Milk Protein'
        step_simulation(at, steps=1)
        self.assertIn("Turb", at.error[0].value)

        # Normal -> NOMINAL success
        at.session_state['active_fault'] = 'Normal (Clean Water)'
        step_simulation(at, steps=1)
        self.assertEqual(len(at.error), 0)
        self.assertIn("NOMINAL", at.success[0].value)

    def test_tc_t3_int_05_retrain_resets_phase2_state(self):
        """Verify re-training in Phase 1 resets test histories and step counter."""
        at = create_phase2_app()
        step_simulation(at, steps=5)
        self.assertEqual(at.session_state.step, 5)

        # Navigate Phase 1 and Retrain
        at.sidebar.radio[0].set_value('Phase 1 — Train (NOC)').run()
        at.button[0].click().run()
        self.assertEqual(at.session_state.t2_hist, [0.0] * 60)
        self.assertEqual(at.session_state.q_hist, [0.0] * 60)
        self.assertFalse(at.session_state.is_playing)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4: Academic & Student Lab Scenarios (Headless Streamlit UI Harness)
# ─────────────────────────────────────────────────────────────────────────────
class TestTier4_LabScenarios_UI(unittest.TestCase):
    """Tier 4: End-to-end student lab scenarios executed through UI harness."""

    def test_tc_t4_scn_01_student_phase1_training_flow(self):
        """Scenario 1: Student opens dashboard, reviews steps, and completes model training."""
        at = create_fresh_app().run()
        self.assertIn("Phase 1: Model Training", at.title[0].value)
        # Click collect & train
        at.button[0].click().run()
        self.assertTrue(at.session_state.is_trained)
        self.assertAlmostEqual(at.session_state.ucl_t2, 8.954, places=2)
        self.assertAlmostEqual(at.session_state.ucl_q, 8.946, places=2)

    def test_tc_t4_scn_02_student_clean_water_steady_state(self):
        """Scenario 2: Student enters Phase 2, starts loop, verifies NOMINAL operation."""
        at = create_phase2_app()
        step_simulation(at, steps=2)
        self.assertEqual(len(at.error), 0)
        self.assertIn("NOMINAL", at.success[0].value)

    def test_tc_t4_scn_03_student_saline_investigation(self):
        """Scenario 3: Student injects NaCl, verifies alarm banner and TDS isolation."""
        at = create_phase2_app()
        at.session_state['active_fault'] = 'NaCl (Salt Solution)'
        step_simulation(at, steps=1)
        self.assertEqual(len(at.error), 1)
        self.assertIn("ALARM", at.error[0].value)
        self.assertIn("TDS", at.error[0].value)

    def test_tc_t4_scn_04_student_acid_spill_investigation(self):
        """Scenario 4: Student simulates acid spill, verifies alarm banner and pH isolation."""
        at = create_phase2_app()
        at.session_state['active_fault'] = 'Acetic Acid (Vinegar)'
        step_simulation(at, steps=1)
        self.assertEqual(len(at.error), 1)
        self.assertIn("ALARM", at.error[0].value)
        self.assertIn("pH", at.error[0].value)

    def test_tc_t4_scn_05_student_milk_protein_investigation(self):
        """Scenario 5: Student simulates organic milk spill, verifies Turbidity isolation."""
        at = create_phase2_app()
        at.session_state['active_fault'] = 'Milk Protein'
        step_simulation(at, steps=1)
        self.assertEqual(len(at.error), 1)
        self.assertIn("ALARM", at.error[0].value)
        self.assertIn("Turb", at.error[0].value)

    def test_tc_t4_scn_06_student_hardware_drift_investigation(self):
        """Scenario 6: Student simulates hardware drift, verifies Temp isolation."""
        at = create_phase2_app()
        at.session_state['active_fault'] = 'Sensor Drift (Hardware)'
        step_simulation(at, steps=1)
        self.assertEqual(len(at.error), 1)
        self.assertIn("ALARM", at.error[0].value)
        self.assertIn("Temp", at.error[0].value)


if __name__ == '__main__':
    unittest.main()
