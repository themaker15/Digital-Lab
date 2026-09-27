import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Educational FDD Simulator", layout="wide", initial_sidebar_state="expanded")

# --- Constants & Math Setup ---
MEANS = np.array([25.0, 150.0, 5.0, 7.0])
STDS = np.array([0.25, 4.0, 0.4, 0.05])
UCL_T2 = 12.5
UCL_Q = 4.0
LABELS = ['Temp', 'TDS', 'Turb', 'pH']

FAULTS = {
    'Normal (Clean Water)': np.array([0, 0, 0, 0]),
    'NaCl (Salt Solution)': np.array([0, 140, 0, 0.1]),
    'Acetic Acid (Vinegar)': np.array([0, 30, 1.5, -2.8]),
    'Milk Protein': np.array([0, 20, 28, -0.4]),
    'Sensor Drift (Hardware Error)': np.array([5.5, 0, 0, 0])
}

# --- State Initialization ---
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.t2_hist = [1.5] * 50
    st.session_state.q_hist = [0.5] * 50
    st.session_state.contribs = [0.0, 0.0, 0.0, 0.0]
    st.session_state.active_fault = 'Normal (Clean Water)'
    st.session_state.is_playing = False
    st.session_state.raw_vals = MEANS.copy()

# --- Simulation Logic ---
def generate_data():
    offset = FAULTS[st.session_state.active_fault]
    noise = np.random.randn(4) * STDS
    raw_vals = MEANS + offset + noise
    
    # Standardize
    z = (raw_vals - MEANS) / STDS
    
    # Mock PCA Mathematics for Simulator
    T2 = np.sum(z**2)
    
    # Q Simulation based on broken correlations
    q_err = 0
    if 'Vinegar' in st.session_state.active_fault: q_err = abs(z[3] * 2.0)
    elif 'Milk' in st.session_state.active_fault: q_err = abs(z[2] * 2.5)
    elif 'NaCl' in st.session_state.active_fault: q_err = abs(z[1] * 0.8)
    elif 'Drift' in st.session_state.active_fault: q_err = abs(z[0] * 1.5)
    
    Q = (0.3 + np.random.rand() * 0.4) + q_err

    # Calculate contributions (for the root cause plot)
    total = max(T2, 0.01)
    contribs = (z**2) / total
    
    return raw_vals, T2, Q, contribs

if st.session_state.is_playing:
    raw, t2, q, contrib = generate_data()
    st.session_state.raw_vals = raw
    st.session_state.t2_hist.append(t2)
    st.session_state.t2_hist.pop(0)
    st.session_state.q_hist.append(q)
    st.session_state.q_hist.pop(0)
    st.session_state.contribs = contrib
    st.session_state.step += 1

is_alarm = st.session_state.t2_hist[-1] > UCL_T2 or st.session_state.q_hist[-1] > UCL_Q

# --- UI: Sidebar ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    col1, col2 = st.columns(2)
    if col1.button("▶ Start", use_container_width=True, type="primary"):
        st.session_state.is_playing = True
        st.rerun()
    if col2.button("⏸ Pause", use_container_width=True):
        st.session_state.is_playing = False
        st.rerun()
        
    st.divider()
    st.subheader("🧪 Fault Injection")
    st.caption("Select a contaminant to disrupt the system.")
    
    new_fault = st.radio("Contaminant Type", list(FAULTS.keys()), 
                         index=list(FAULTS.keys()).index(st.session_state.active_fault))
    if new_fault != st.session_state.active_fault:
        st.session_state.active_fault = new_fault
        st.rerun()

# --- UI: Main Dashboard ---
st.title("DIGITAL LAB: Water Contamination FDD")
st.markdown("**(Educational Simulator)** Software-in-the-Loop PCA Anomaly Detection")

# Status Banner
if is_alarm:
    st.error(f"**🚨 ANOMALY DETECTED:** Root cause isolation active. Check Contribution Plot. (Current State: {st.session_state.active_fault})")
else:
    st.success("**✅ SYSTEM NOMINAL:** T² and Q-Statistics are within statistical control limits.")

st.markdown("---")

# Layout
col_vis, col_charts = st.columns([1, 1.3], gap="large")

with col_vis:
    st.subheader("Physical Lab Setup (Digital Twin)")
    st.caption("Visual representation of the CSTR and IoT telemetry flow.")
    
    # Dynamically change water color based on fault
    water_colors = {
        'Normal (Clean Water)': 'rgba(59, 130, 246, 0.3)',
        'NaCl (Salt Solution)': 'rgba(245, 158, 11, 0.4)',
        'Acetic Acid (Vinegar)': 'rgba(239, 68, 68, 0.3)',
        'Milk Protein': 'rgba(203, 213, 225, 0.8)',
        'Sensor Drift (Hardware Error)': 'rgba(59, 130, 246, 0.3)'
    }
    w_color = water_colors[st.session_state.active_fault]
    
    svg_html = f"""
    <div style="background-color: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 20px;">
    <svg viewBox="0 0 500 350" width="100%">
        <!-- Grid -->
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#F1F5F9" stroke-width="1"/>
        </pattern>
        <rect width="100%" height="100%" fill="url(#grid)" />

        <!-- Pipes -->
        <path d="M -10 200 L 100 200" stroke="#CBD5E1" stroke-width="16" fill="none" stroke-linecap="round"/>
        <path d="M 0 200 L 90 200" stroke="#3B82F6" stroke-width="4" stroke-dasharray="8 4" fill="none"/>
        <text x="10" y="185" fill="#64748B" font-size="12" font-family="sans-serif">Inlet</text>

        <path d="M 350 200 L 510 200" stroke="#CBD5E1" stroke-width="16" fill="none" stroke-linecap="round"/>
        <path d="M 360 200 L 500 200" stroke="#3B82F6" stroke-width="4" stroke-dasharray="8 4" fill="none"/>
        <text x="400" y="185" fill="#64748B" font-size="12" font-family="sans-serif">Outlet</text>

        <!-- Tank -->
        <rect x="100" y="80" width="250" height="200" rx="12" stroke="#94A3B8" stroke-width="3" fill="#F8FAFC" />
        <rect x="102" y="130" width="246" height="148" rx="8" fill="{w_color}" style="transition: fill 0.8s;" />
        <line x1="102" y1="130" x2="348" y2="130" stroke="rgba(59, 130, 246, 0.4)" stroke-width="2"/>

        <!-- Sensors -->
        <!-- Temp -->
        <g transform="translate(130, 40)">
            <line x1="0" y1="0" x2="0" y2="150" stroke="#94A3B8" stroke-width="2"/>
            <rect x="-18" y="-15" width="36" height="18" rx="4" fill="#FFFFFF" stroke="#3B82F6" stroke-width="2"/>
            <text x="0" y="-3" fill="#3B82F6" font-size="10" text-anchor="middle" font-family="sans-serif" font-weight="bold">TEMP</text>
        </g>
        <!-- TDS -->
        <g transform="translate(180, 40)">
            <line x1="0" y1="0" x2="0" y2="130" stroke="#94A3B8" stroke-width="2"/>
            <rect x="-18" y="-15" width="36" height="18" rx="4" fill="#FFFFFF" stroke="#F59E0B" stroke-width="2"/>
            <text x="0" y="-3" fill="#F59E0B" font-size="10" text-anchor="middle" font-family="sans-serif" font-weight="bold">TDS</text>
        </g>
        <!-- Turbidity -->
        <g transform="translate(270, 40)">
            <line x1="0" y1="0" x2="0" y2="160" stroke="#94A3B8" stroke-width="2"/>
            <rect x="-18" y="-15" width="36" height="18" rx="4" fill="#FFFFFF" stroke="#64748B" stroke-width="2"/>
            <text x="0" y="-3" fill="#64748B" font-size="10" text-anchor="middle" font-family="sans-serif" font-weight="bold">TURB</text>
        </g>
        <!-- pH -->
        <g transform="translate(320, 40)">
            <line x1="0" y1="0" x2="0" y2="140" stroke="#94A3B8" stroke-width="2"/>
            <rect x="-18" y="-15" width="36" height="18" rx="4" fill="#FFFFFF" stroke="#10B981" stroke-width="2"/>
            <text x="0" y="-3" fill="#10B981" font-size="10" text-anchor="middle" font-family="sans-serif" font-weight="bold">pH</text>
        </g>

        <!-- Stirrer -->
        <line x1="225" y1="60" x2="225" y2="240" stroke="#CBD5E1" stroke-width="3"/>
        <ellipse cx="225" cy="240" rx="30" ry="6" fill="#94A3B8"/>
        <rect x="215" y="50" width="20" height="15" rx="2" fill="#64748B"/>
        
        <!-- ESP32 & MQTT -->
        <rect x="370" y="50" width="70" height="40" rx="4" fill="#FFFFFF" stroke="#0F172A" stroke-width="2"/>
        <text x="405" y="65" fill="#0F172A" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle">ESP32</text>
        <text x="405" y="80" fill="#64748B" font-size="9" font-family="sans-serif" text-anchor="middle">MQTT Data</text>
        
        <!-- Connection Lines -->
        <path d="M 130 25 L 130 10 L 405 10 L 405 50" stroke="#3B82F6" stroke-width="2" fill="none" opacity="0.4"/>
        <path d="M 180 25 L 180 10" stroke="#F59E0B" stroke-width="2" fill="none" opacity="0.4"/>
        <path d="M 270 25 L 270 10" stroke="#64748B" stroke-width="2" fill="none" opacity="0.4"/>
        <path d="M 320 25 L 320 10" stroke="#10B981" stroke-width="2" fill="none" opacity="0.4"/>
    </svg>
    </div>
    """
    st.markdown(svg_html, unsafe_allow_html=True)
    
    st.subheader("Live Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temp (°C)", f"{st.session_state.raw_vals[0]:.2f}")
    m2.metric("TDS (ppm)", f"{st.session_state.raw_vals[1]:.1f}")
    m3.metric("Turbidity", f"{st.session_state.raw_vals[2]:.2f}")
    m4.metric("pH", f"{st.session_state.raw_vals[3]:.2f}")

with col_charts:
    st.subheader("PCA Analytics Engine")
    st.caption("AI detects correlations and variations in real-time.")
    
    # Chart Styling
    chart_layout = dict(
        height=200, margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#F1F5F9')
    )

    # T2 Chart
    fig_t2 = go.Figure()
    fig_t2.add_trace(go.Scatter(y=st.session_state.t2_hist, mode='lines', name="T²", 
                                line=dict(color='#3B82F6', width=2), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig_t2.add_hline(y=UCL_T2, line_dash="dash", line_color="#EF4444", annotation_text="UCL (12.5)")
    fig_t2.update_layout(title="Hotelling's T² (Within-Model Distance)", yaxis_range=[0, 25], **chart_layout)
    st.plotly_chart(fig_t2, use_container_width=True)

    # Q Chart
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(y=st.session_state.q_hist, mode='lines', name="Q", 
                               line=dict(color='#8B5CF6', width=2), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'))
    fig_q.add_hline(y=UCL_Q, line_dash="dash", line_color="#EF4444", annotation_text="UCL (4.0)")
    fig_q.update_layout(title="SPE / Q-Statistic (Correlation Error)", yaxis_range=[0, 10], **chart_layout)
    st.plotly_chart(fig_q, use_container_width=True)

    # Contribution Plot
    colors = ['#EF4444' if (is_alarm and c > 0.35) else '#94A3B8' for c in st.session_state.contribs]
    fig_c = go.Figure(data=[go.Bar(x=LABELS, y=st.session_state.contribs, marker_color=colors)])
    fig_c.update_layout(title="Fault Isolation (Contribution Plot)", yaxis_range=[0, 1], **chart_layout)
    st.plotly_chart(fig_c, use_container_width=True)

# Loop trigger
if st.session_state.is_playing:
    time.sleep(1)
    st.rerun()
