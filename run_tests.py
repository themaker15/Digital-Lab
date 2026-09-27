"""
Master Test Runner for Water Contamination FDD Simulator
Executes Layer A (Math Engine) and Layer B (Headless Streamlit UI) Test Suites.

Features:
- Windows CP1252 / UTF-8 safe stdout & stderr reconfiguring
- Tier-by-tier execution reporting (Tiers 1-4)
- Summary statistics with test counts, durations, and TDD status
"""

import sys
import os
import time
import argparse
import unittest

# Ensure Windows terminal handles UTF-8 safely
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def build_suite(include_math=True, include_ui=True, tier=None):
    """Assembles unittest TestSuite based on filters."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    if include_math:
        import test_math_fdd
        if tier is None or tier == 1:
            suite.addTests(loader.loadTestsFromTestCase(test_math_fdd.TestTier1_FeatureCoverage_Math))
        if tier is None or tier == 2:
            suite.addTests(loader.loadTestsFromTestCase(test_math_fdd.TestTier2_BoundaryCornerCases_Math))
        if tier is None or tier == 3:
            suite.addTests(loader.loadTestsFromTestCase(test_math_fdd.TestTier3_Interactions_Math))
        if tier is None or tier == 4:
            suite.addTests(loader.loadTestsFromTestCase(test_math_fdd.TestTier4_LabScenarios_Math))

    if include_ui:
        import test_streamlit_app
        if tier is None or tier == 1:
            suite.addTests(loader.loadTestsFromTestCase(test_streamlit_app.TestTier1_FeatureCoverage_UI))
        if tier is None or tier == 2:
            suite.addTests(loader.loadTestsFromTestCase(test_streamlit_app.TestTier2_BoundaryCornerCases_UI))
        if tier is None or tier == 3:
            suite.addTests(loader.loadTestsFromTestCase(test_streamlit_app.TestTier3_Interactions_UI))
        if tier is None or tier == 4:
            suite.addTests(loader.loadTestsFromTestCase(test_streamlit_app.TestTier4_LabScenarios_UI))

    return suite


def main():
    parser = argparse.ArgumentParser(description="Water Contamination FDD Simulator - Test Runner")
    parser.add_argument('--math', action='store_true', help="Run only Layer A (Math Engine) tests")
    parser.add_argument('--ui', action='store_true', help="Run only Layer B (Streamlit UI) tests")
    parser.add_argument('--tier', type=int, choices=[1, 2, 3, 4], default=None, help="Run specific Tier (1-4)")
    parser.add_argument('--tdd', action='store_true', help="TDD mode: exit 0 if math passes and only expected M2 features fail")
    args = parser.parse_args()

    include_math = not args.ui
    include_ui = not args.math

    print("=" * 80)
    print(" Water Contamination FDD Simulator - 4-Tier Test Runner")
    print(f" Python Executable: {sys.executable}")
    print(f" Mode: Math={include_math}, UI={include_ui}, Tier={args.tier or 'All (1-4)'}")
    print("=" * 80)

    suite = build_suite(include_math=include_math, include_ui=include_ui, tier=args.tier)
    total_tests = suite.countTestCases()
    print(f"Discovered {total_tests} test cases across requested tiers.")
    print("-" * 80)

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    duration = time.time() - start_time

    print("-" * 80)
    print(" TEST EXECUTION SUMMARY")
    print(f" - Total Tests Run: {result.testsRun}")
    print(f" - Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f" - Failures: {len(result.failures)}")
    print(f" - Errors: {len(result.errors)}")
    print(f" - Elapsed Time: {duration:.2f} seconds")

    # Analyze failure details
    known_m2_failures = [
        'test_tc_t1_r1_02_no_functional_emojis_in_buttons',
        'test_tc_t1_r1_03_typography_minimum_font_size',
        'test_tc_t1_r3_01_phase2_biplot_rendered',
        'test_tc_t1_r3_03_phase2_biplot_session_state_scores_hist',
    ]

    actual_failure_names = [test.id().split('.')[-1] for test, _ in result.failures]
    unexpected_failures = [name for name in actual_failure_names if name not in known_m2_failures]

    print("-" * 80)
    if result.wasSuccessful():
        print(" STATUS: ALL TESTS PASSED (100% GREEN)")
        sys.exit(0)
    elif args.tdd and len(unexpected_failures) == 0 and len(result.errors) == 0:
        print(" STATUS: TDD BASELINE VERIFIED (M1 Test Suite Ready, M2 Tasks Isolated)")
        print(" Known M2 Pending Tasks (Expected Red in Current app.py):")
        for f in actual_failure_names:
            print(f"   - {f}")
        sys.exit(0)
    else:
        print(f" STATUS: {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
        if unexpected_failures:
            print(" Unexpected Failures:")
            for f in unexpected_failures:
                print(f"   - {f}")
        sys.exit(1)


if __name__ == '__main__':
    main()
