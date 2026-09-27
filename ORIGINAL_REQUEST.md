# Original User Request

## Initial Request — 2026-09-24T19:46:14Z

Redesign the entire Streamlit UI of an educational IoT + PCA Fault Detection Dashboard
(Water Contamination FDD Simulator) from scratch, targeting demo-ready quality for
academic presentation and lab report submission.

Working directory: C:\Users\mekan\Documents\DIGITAL LAB\FDD_Simulator
Integrity mode: benchmark

## Context

The existing app is `app.py` in the working directory. It has two phases:

- **Phase 1 — Train (NOC):** Collects 500 simulated clean-water samples, fits a
  `StandardScaler` and `PCA(n_components=2)` via scikit-learn, computes UCL_T²
  and UCL_Q at 99th percentile. Stores all learned parameters in `st.session_state`.

- **Phase 2 — Test (FDD):** Reads live simulated sensor readings every 1 second,
  applies the trained scaler + PCA model, computes Hotelling T² and SPE/Q, and
  displays a Contribution Plot for root-cause isolation.

The existing logic and math are correct and must be preserved exactly. Only the
visual presentation layer should be redesigned.

## Critical Implementation Notes

**SVG Rendering:** Streamlit's `st.markdown(..., unsafe_allow_html=True)` silently
strips `<svg>`, `<pattern>`, `<defs>`, and most SVG child elements due to its HTML
sanitizer — the SVG will render as raw escaped text. You MUST use
`st.components.v1.html(svg_string, height=N, scrolling=False)` for every SVG block.
Never use `st.markdown()` for SVG content.

**Plotly legend:** `legend=dict(display=False)` is not a valid Plotly property and
crashes the app. Use `showlegend=False` at the layout level instead.

**f-string + SVG attribute quotes:** When building SVG strings inside Python f-strings,
avoid embedding Python expressions that produce strings containing single quotes
inside HTML attributes delimited by single quotes. Pre-compute any dynamic SVG
fragments into separate variables before injecting them into the f-string.

## Requirements

### R1. Full UI Redesign — Light Theme, Enterprise Grade

Redesign the entire visual presentation of the app using only:
- Streamlit's native widget API
- `st.components.v1.html()` for SVG or animated physical-hardware visuals
- Plotly for all charts

The app must use a strict **Light Theme** (no dark mode, no dark panels).
Design quality must be demo-ready for academic presentation: clean typography,
consistent spacing, professional color use, no emoji-as-icons, no garish gradients.
The `.streamlit/config.toml` already enforces light theme.

### R2. Preserve All Existing PCA Logic

The scikit-learn training pipeline (StandardScaler → PCA → UCL computation),
the live inference loop (`process_sample()`), and all `st.session_state` keys
must be preserved exactly as-is. Do not modify the mathematics or the
session state schema.

### R3. PCA Biplot — New Feature

Add a PCA Score Biplot (PC1 vs PC2 scatter plot, rendered with Plotly) to
Phase 2 — Test Mode. This plot must:
- Show a rolling window of the last 60 score vectors in (PC1, PC2) space
- Color-code points by time (oldest = faded, newest = vivid) OR by alarm state
  (nominal = blue, alarm = red)
- Overlay the 99% confidence ellipse derived from the training data (Hotelling
  T² boundary in PC space)
- Update every tick alongside the existing T² and Q charts

### R4. Phase 1 Visualisation Improvements

In the Train phase results panel, display:
- A Scree plot (explained variance per PC, bar chart, Plotly)
- The learned baseline table (sensor, μ, σ) as a styled dataframe
- Clear instructional copy that explains what each parameter means to a student

## Acceptance Criteria

### Layout and Theme
- [ ] App renders entirely in light mode with no dark panels visible
- [ ] No emoji used as functional icons anywhere in the app
- [ ] Typography is consistent and readable at 1080p; no text smaller than 11px
- [ ] Both phases load without errors or warnings in the Streamlit console

### Phase 1 — Train Mode
- [ ] "Collect & Train" button triggers a visible progress indicator
- [ ] After training, Scree plot renders correctly with labelled axes
- [ ] Baseline table is visible and contains correct μ/σ values
- [ ] UCL values are displayed numerically after training

### Phase 2 — Test Mode
- [ ] T² and Q time-series charts update every ~1 second when the sensor loop is running
- [ ] UCL reference lines appear on both T² and Q charts at the correct trained values
- [ ] Contribution Plot bar chart highlights the dominant sensor in red when an alarm is active
- [ ] PCA Biplot renders in PC1 vs PC2 space and updates every tick
- [ ] Confidence ellipse is visible on the Biplot
- [ ] Fault injection (contaminant radio button) changes the SVG water colour within 2 seconds
- [ ] Sensor metric deltas are shown (change from previous tick)

### Robustness
- [ ] Switching from Phase 2 back to Phase 1 and re-training resets all histories cleanly
- [ ] Pausing and resuming the sensor loop does not crash or produce errors
