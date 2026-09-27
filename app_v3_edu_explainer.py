import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Educational FDD Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Constants — Physical reality of the clean tank
# ─────────────────────────────────────────────
PHYSICAL_MEANS = np.array([25.0, 150.0, 5.0, 7.0])
PHYSICAL_STDS  = np.array([0.25,   4.0, 0.4, 0.05])
LABELS = ['Temp', 'TDS', 'Turb', 'pH']
UNITS  = ['°C', 'ppm', 'NTU', '']

FAULTS = {
    'Normal (Clean Water)':       np.array([0.0,   0.0,  0.0,  0.0]),
    'NaCl (Salt Solution)':       np.array([0.0, 140.0,  0.0,  0.1]),
    'Acetic Acid (Vinegar)':      np.array([0.0,  30.0,  1.5, -2.8]),
    'Milk Protein':               np.array([0.0,  20.0, 28.0, -0.4]),
    'Sensor Drift (Hardware)':    np.array([5.5,   0.0,  0.0,  0.0]),
}

WATER_COLORS = {
    'Normal (Clean Water)':    'rgba(59,130,246,0.28)',
    'NaCl (Salt Solution)':    'rgba(245,158,11,0.40)',
    'Acetic Acid (Vinegar)':   'rgba(239,68,68,0.32)',
    'Milk Protein':            'rgba(220,224,232,0.85)',
    'Sensor Drift (Hardware)': 'rgba(59,130,246,0.28)',
}

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
_DEFAULTS = {
    'is_trained':   False,
    'scaler':       None,
    'pca_model':    None,
    'ucl_t2':       0.0,
    'ucl_q':        0.0,
    'var_ratio':    [],
    'pca_means':    None,  # scaler mean_ stored for display
    'pca_stds':     None,  # scaler scale_ stored for display
    'step':         0,
    't2_hist':      [0.0] * 60,
    'q_hist':       [0.0] * 60,
    'contribs':     [0.0, 0.0, 0.0, 0.0],
    'active_fault': 'Normal (Clean Water)',
    'is_playing':   False,
    'raw_vals':     PHYSICAL_MEANS.copy().tolist(),
    'prev_raw':     PHYSICAL_MEANS.copy().tolist(),
    'train_done_flag': False,   # used to show success banner persistently
    'scores_hist':  [[0.0, 0.0]] * 60,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("FDD Simulator")
    mode = st.radio(
        "System Phase",
        ["Phase 0 — Project Guide & Setup", "Phase 1 — Train (NOC)", "Phase 2 — Test (FDD)"],
        index=0
    )
    st.divider()

    # Always show model status chip
    if st.session_state.is_trained:
        st.success("Model: **Trained**")
        st.caption(
            f"T² UCL = {st.session_state.ucl_t2:.3f}  \n"
            f"Q UCL = {st.session_state.ucl_q:.3f}  \n"
            f"Variance explained = {sum(st.session_state.var_ratio)*100:.1f}%"
        )
    else:
        st.warning("Model: **Not Trained**")

    st.divider()
    if st.button("Reset Entire System / รีเซ็ตระบบทั้งหมด", use_container_width=True, type="secondary"):
        for k, v in _DEFAULTS.items():
            if k in ('t2_hist', 'q_hist'):
                st.session_state[k] = [0.0] * 60
            elif k == 'scores_hist':
                st.session_state[k] = [[0.0, 0.0]] * 60
            elif k in ('raw_vals', 'prev_raw'):
                st.session_state[k] = PHYSICAL_MEANS.copy().tolist()
            elif k == 'contribs':
                st.session_state[k] = [0.0, 0.0, 0.0, 0.0]
            else:
                st.session_state[k] = v
        st.session_state.is_playing = False
        st.rerun()


# ══════════════════════════════════════════════
# PHASE 0 — PROJECT GUIDE & SETUP
# ══════════════════════════════════════════════
if mode == "Phase 0 — Project Guide & Setup":

    st.title("Project Architecture & Hardware Setup Guide")
    st.markdown("คู่มือสถาปัตยกรรมและรายละเอียดทางวิศวกรรมสำหรับทีมงาน: **DIY Water Contamination FDD System using PCA**")
    st.divider()

    col_img, col_txt = st.columns([1.2, 1], gap="large")
    with col_img:
        st.subheader("Physical Hardware Setup (Lab Prototype)")
        import os
        img_path = "water_fdd_lab_setup.jpg"
        if os.path.exists(img_path):
            st.image(img_path, caption="DIY Clear Acrylic Box Tank (CSTR) & ESP32 Node Setup", use_container_width=True)
        else:
            st.info("Photo not found in directory.")

    with col_txt:
        st.subheader("Project Overview & Objectives")
        st.markdown("""
ระบบ **Fault Detection & Diagnosis (FDD)** ตรวจจับสารปนเปื้อนในน้ำและอุปกรณ์ชำรุดแบบ Real-time ด้วยสถิติหลายตัวแปร (**Multivariate Statistical Process Control: MSPC**) ผสานเทคโนโลยี **IoT Telemetry**

**เป้าหมายทางวิศวกรรม:**
 - Early Anomaly Detection: ตรวจจับสิ่งปนเปื้อนความเข้มข้นต่ำก่อนกระจายตัวสู่กระบวนการปลายน้ำ
 - Root Cause Isolation: ชี้ชัดเซนเซอร์ต้นเหตุผ่าน Contribution Plot ให้ Operator เข้าแก้ไขทันที
 - Alarm Rationalization: ลด False Alarm ที่เกิดจาก Process Noise และการแกว่งตัวตามธรรมชาติ
 - Unsupervised Learning: พัฒนาโมเดลจาก Normal Operating Condition จึงไม่ต้องเสี่ยงฉีดสารพิษล่วงหน้า
""")

    st.divider()
    st.subheader("1. เปรียบเทียบ: ทำไม Multivariate PCA ถึงดีกว่า Univariate Threshold?")
    comp_df = pd.DataFrame({
        "มิติการเปรียบเทียบ": [
            "การตรวจจับความสัมพันธ์ (Correlation)",
            "สารปนเปื้อนความเข้มข้นต่ำ (Trace Level)",
            "อัตราการเกิด False Alarm",
            "การระบุ Root Cause",
            "การเตรียมข้อมูล (Training Requirement)"
        ],
        "ระบบดั้งเดิม (Univariate Threshold)": [
            "พิจารณาตัวแปรแยกเดี่ยว ไม่เห็นความสัมพันธ์ข้ามตัวแปร",
            "หลุดรอดได้ง่าย หากค่าเดี่ยวยังไม่แตะเส้น Alarm",
            "สูงมาก หาก Sensor ตัวใดตัวหนึ่งมี Noise ชั่วขณะ",
            "ผู้ควบคุมต้องไล่เปิดเช็คทีละจุดด้วยตัวเอง",
            "ต้องมีประวัติค่าผิดปกติและเงื่อนไข Alarm ล่วงหน้า"
        ],
        "ระบบใหม่ของเรา (Multivariate PCA FDD)": [
            "ดักจับ Cross-Correlation ระหว่าง 4 เซนเซอร์พร้อมกัน",
            "ตรวจพบได้ทันที เนื่องจากความสัมพันธ์ Baseline เสียรูปทรง",
            "ต่ำมาก ประเมินจากสถิติภาพรวม Hotelling's T² และ Q",
            "คำนวณสัดส่วน Contribution ชี้เป้า Sensor ต้นเหตุอัตโนมัติ",
            "ใช้ Unsupervised Learning ฝึกเฉพาะตอนน้ำสะอาด (NOC)"
        ]
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown("""
**สรุปจุดเด่นทางวิศวกรรม:**
 - Multivariate Correlation: ดักจับค่าผิดปกติซ่อนเร้น กรณีที่ค่า Variable เดี่ยวยังไม่เกิน Threshold
 - Fault Isolation & Diagnosis: ระบุ Root Cause ด้วย Contribution Plot ได้อย่างแม่นยำ
 - Alarm Rationalization: ประเมินจากสถิติ T² และ Q เพื่อลด False Alarm
 - Baseline Reference: อิงสถิติจาก Normal Operating Condition ทำให้ Deploy ใช้งานได้จริง
""")

    st.divider()
    col_setup, col_algo = st.columns(2, gap="large")

    with col_setup:
        st.subheader("2. ขอบเขตและโครงสร้างฮาร์ดแวร์ (Hardware Setup)")
        st.markdown("""
**ขอบเขตและข้อกำหนดโปรเจกต์:**
 - Course Context: วิชา IoT สำหรับวิศวกรรมเคมี (ผู้สอน: ดร. รัชชนนท์ เปี่ยมใจสว่าง)
 - Budget Limit: งบประมาณรวมทั้งโปรเจกต์ไม่เกิน 2,500 บาท
 - Project Timeline: กำหนดส่งมอบและทดสอบระบบทั้งหมด 45 วัน
 - Sensing Dimension: เซนเซอร์ 4 ชนิด (Temp, TDS, Turbidity, pH)

**อุปกรณ์ที่ติดตั้งในถัง CSTR:**
 - DIY Acrylic Tank: กล่องอะคริลิกใสทรงสี่เหลี่ยมเชื่อมขอบกาว พร้อมเจาะรูท่อน้ำเข้าและท่อน้ำล้น
 - Overhead Motorized Stirrer: มอเตอร์กวนใบพัดสแตนเลสให้สารผสมเป็น Homogeneous
 - Temperature Sensor: โพรบกันน้ำ DS18B20 ตรวจจับอุณหภูมิของเหลว
 - TDS Sensor: โพรบวัดสารละลายนำไฟฟ้าและไอออนปนเปื้อน
 - Turbidity Sensor: โพรบวัดความขุ่นแบบแสงดักจับอนุภาคแขวนลอย
 - pH Sensor: โพรบแก้ว BNC สำหรับวัดสมดุลกรด-ด่าง
 - Microcontroller: บอร์ด ESP32 อ่านค่า Sampling ทุก 1 วินาทีและส่งผ่าน MQTT
""")

    with col_algo:
        st.subheader("3. การทำงานของอัลกอริทึม PCA (Algorithm Mechanics)")
        st.markdown("""
**สเต็ปการประมวลผลทางคณิตศาสตร์:**
 - Data Standardization: ปรับสเกลข้อมูล Z = (X - μ) / σ ป้องกันตัวแปรค่าสูงกลบตัวแปรค่าต่ำ
 - Dimension Reduction: ดึง 2 Principal Components (PC1, PC2) ครอบคลุมความแปรปรวนรวมส่วนใหญ่
 - Hotelling's T² Metric: วัดระยะห่างจุดข้อมูลในระนาบโมเดลเพื่อตรวจ Systematic Shift
 - SPE / Q-Statistic Metric: วัด Residual Variance ที่หลุดออกจากระนาบโมเดลเพื่อตรวจ Correlation Breakdown
 - Contribution Plot: คำนวณ Relative Reconstruction Error ชี้เป้าเซนเซอร์ที่เป็น Root Cause

**พฤติกรรมของสารปนเปื้อนในระบบ:**
 - NaCl (Salt): ค่า TDS พุ่งสูงชัดเจน ดักจับได้ด้วย T² และชี้เป้า TDS
 - Acetic Acid (Vinegar): ค่า pH ลดลงฮวบฮาบ ทำลาย Correlation เดิม ดีดค่า Q ทันที
 - Milk Protein: ค่า Turbidity สูงลิ่ว ดีดทั้ง T² และ Q พร้อมระบุ Turbidity
 - Sensor Drift: ค่าอุณหภูมิไต่ระดับผิดธรรมชาติจาก Hardware Failure ทริกเกอร์เตือนทันที
""")

    st.divider()
    st.info("💡 **ขั้นตอนต่อไป:** เลือก **Phase 1 — Train (NOC)** ที่แถบด้านซ้าย เพื่อเริ่มเก็บข้อมูลน้ำสะอาดและฝึกสอนโมเดล PCA")


# ══════════════════════════════════════════════
# PHASE 1 — TRAIN MODE
# ══════════════════════════════════════════════
elif mode == "Phase 1 — Train (NOC)":

    st.title("Phase 1: Model Training (NOC Baseline)")
    st.markdown("การฝึกสอนโมเดล AI ด้วยข้อมูลน้ำสะอาดบริสุทธิ์ (**Normal Operating Condition: NOC**) เพื่อสร้าง Baseline อ้างอิงสถิติสำหรับตรวจจับสิ่งปนเปื้อน")
    st.divider()

    col_explain, col_btn = st.columns([2, 1])

    with col_explain:
        st.markdown("""
**ขั้นตอนการทำงานของระบบแบบ Step-by-Step:**
 - Step 1 Data Acquisition: เก็บตัวอย่างข้อมูลเซนเซอร์น้ำสะอาดในถัง CSTR จำนวน 500 Samples เพื่อสร้าง Baseline ตัวแทนสภาวะปกติ
 - Step 2 Feature Standardization: ปรับสเกลข้อมูลให้เป็น Z-Score ด้วย StandardScaler (μ, σ) ป้องกันตัวแปรหน่วยใหญ่กลบหน่วยเล็ก
 - Step 3 Dimension Reduction: ฟิตโมเดล PCA สกัด 2 แกนหลัก (PC1, PC2) ที่รวบรวม Variance รวมได้สูงสุดจากเซนเซอร์ทั้ง 4 ตัว
 - Step 4 Threshold Calculation: คำนวณเส้นขีดจำกัดบน 99% UCL สำหรับ Hotelling's T² และ SPE / Q-Statistic เพื่อใช้ตรวจจับ Anomaly
""")

    with col_btn:
        st.markdown("&nbsp;")  # vertical spacer
        train_clicked = st.button(
            "Collect Data & Train PCA",
            type="primary",
            use_container_width=True
        )

    if train_clicked:
        progress = st.progress(0, text="Initialising sensors...")
        time.sleep(0.3)

        # Step 1 — Collect
        progress.progress(20, text="Step 1/4 — Collecting 500 NOC samples...")
        rng = np.random.default_rng(seed=42)
        raw_train = rng.standard_normal((500, 4)) * PHYSICAL_STDS + PHYSICAL_MEANS
        time.sleep(0.4)

        # Step 2+3 — Scale
        progress.progress(45, text="Step 2/4 — Fitting StandardScaler...")
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(raw_train)
        time.sleep(0.3)

        # Step 4 — PCA
        progress.progress(65, text="Step 3/4 — Fitting PCA (2 components)...")
        pca = PCA(n_components=2)
        pca.fit(Z_train)
        scores_train = pca.transform(Z_train)
        time.sleep(0.3)

        # Step 5 — UCLs
        progress.progress(85, text="Step 4/4 — Computing control limits (UCL 99%)...")
        T2_train = np.sum(
            (scores_train / np.sqrt(pca.explained_variance_)) ** 2, axis=1
        )
        ucl_t2 = float(np.percentile(T2_train, 99))

        Z_hat = pca.inverse_transform(scores_train)
        Q_train = np.sum((Z_train - Z_hat) ** 2, axis=1)
        ucl_q = float(np.percentile(Q_train, 99))

        # Save to session
        st.session_state.scaler          = scaler
        st.session_state.pca_model       = pca
        st.session_state.ucl_t2          = ucl_t2
        st.session_state.ucl_q           = ucl_q
        st.session_state.var_ratio       = pca.explained_variance_ratio_.tolist()
        st.session_state.pca_means       = scaler.mean_.tolist()
        st.session_state.pca_stds        = scaler.scale_.tolist()
        st.session_state.is_trained      = True
        st.session_state.train_done_flag = True

        # Reset test history so old values don't carry over
        st.session_state.step        = 0
        st.session_state.t2_hist     = [0.0] * 60
        st.session_state.q_hist      = [0.0] * 60
        st.session_state.scores_hist = [[0.0, 0.0]] * 60
        st.session_state.contribs    = [0.0, 0.0, 0.0, 0.0]
        st.session_state.is_playing  = False

        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()

    # ── Results panel (stays visible between reruns) ──────────────────────
    if st.session_state.is_trained:
        if st.session_state.train_done_flag:
            st.success("Training complete! Switch to **Phase 2 — Test** in the sidebar.")
            st.session_state.train_done_flag = False

        st.divider()
        st.subheader("Learned Model Parameters")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Principal Components kept", "2 / 4")
        c2.metric("Variance explained",
                  f"{sum(st.session_state.var_ratio) * 100:.1f}%")
        c3.metric("T² UCL (99%)", f"{st.session_state.ucl_t2:.4f}")
        c4.metric("Q UCL (99%)",  f"{st.session_state.ucl_q:.4f}")

        st.markdown("**Explained Variance per Principal Component (Scree Plot)**")
        vr = st.session_state.var_ratio
        cum_vr = np.cumsum(vr) * 100
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(vr))],
            y=[v * 100 for v in vr],
            name="Individual Variance",
            marker_color=['#2563EB', '#0D9488'],
            text=[f"{v*100:.1f}%" for v in vr],
            textposition='outside',
            textfont=dict(size=11, color='#0F172A')
        ))
        fig_var.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(vr))],
            y=cum_vr.tolist(),
            name="Cumulative Variance",
            mode='lines+markers+text',
            line=dict(color='#0F172A', width=2),
            marker=dict(size=11, color='#0F172A'),
            text=[f"{c:.1f}%" for c in cum_vr],
            textposition='top center',
            textfont=dict(size=11, color='#0F172A')
        ))
        fig_var.update_layout(
            height=230, margin=dict(l=25, r=25, t=30, b=25),
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False,
            font=dict(size=11, color='#0F172A'),
            yaxis=dict(
                title=dict(text="% Variance Explained", font=dict(size=11)),
                gridcolor='#F1F5F9', range=[0, 115],
                tickfont=dict(size=11)
            ),
            xaxis=dict(showgrid=False, tickfont=dict(size=11))
        )
        st.plotly_chart(fig_var, use_container_width=True)

        st.markdown("**Learned Sensor Baselines (Normal Operating Condition)**")
        means = st.session_state.pca_means
        stds = st.session_state.pca_stds
        symbols = ['T', 'TDS', 'Turb', 'pH']
        roles = [
            'Process temperature & thermodynamic stability',
            'Dissolved ionic salts & mineral concentration',
            'Suspended solids & optical clarity',
            'Acid-base equilibrium & chemical balance'
        ]
        normal_ranges = [
            f"[{m - 3*s:.2f}, {m + 3*s:.2f}]" for m, s in zip(means, stds)
        ]
        df = pd.DataFrame({
            'Sensor':                    LABELS,
            'Symbol':                    symbols,
            'Unit':                      UNITS,
            'Learned Mean (μ)':          [f"{v:.4f}" for v in means],
            'Learned Std (σ)':           [f"{v:.4f}" for v in stds],
            'NOC Normal Range (μ ± 3σ)': normal_ranges,
            'Physical Role':             roles,
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown("""
**หลักการทางวิศวกรรมและการคำนวณโมเดล (Phase 1 Engineering Rationale):**
 - Normal Operating Condition: พัฒนาโมเดลจากสภาวะน้ำสะอาดบริสุทธิ์เพื่อสร้าง Statistical Baseline สำหรับอ้างอิง
 - Feature Standardization: ปรับสเกลข้อมูลด้วย StandardScaler (μ, σ) ป้องกันตัวแปรค่าสูงกลบตัวแปรค่าต่ำ
 - Covariance Matrix Decomposition: สกัดแกนหลัก PC1 และ PC2 เพื่อเก็บ Total Variance สูงสุดของระบบ CSTR
 - Scree Plot Variance Analysis: ยืนยันความสมบูรณ์ของโมเดลด้วย Cumulative Variance เกิน 80% ตามเกณฑ์วิศวกรรม
 - Hotelling's T² Threshold: กำหนดขีดจำกัดบน 99% UCL สำหรับดักจับ Systematic Variation ในระนาบโมเดล
 - SPE / Q-Statistic Threshold: กำหนดขีดจำกัดบน 99% UCL สำหรับดักจับ Correlation Breakdown นอกระนาบโมเดล
""")


# ══════════════════════════════════════════════
# PHASE 2 — TEST MODE (FDD)
# ══════════════════════════════════════════════
elif mode == "Phase 2 — Test (FDD)":

    if not st.session_state.is_trained:
        st.warning(
            "Model not trained yet. Go to **Phase 1 — Train** and press the train button first."
        )
        st.stop()

    if 'scores_hist' not in st.session_state:
        st.session_state.scores_hist = [[0.0, 0.0]] * 60

    # ── Sidebar controls (only rendered in Test mode) ──
    with st.sidebar:
        st.subheader("Sensor Loop")
        c1, c2 = st.columns(2)
        if c1.button("Start", use_container_width=True, type="primary"):
            st.session_state.is_playing = True
            st.rerun()
        if c2.button("Pause", use_container_width=True):
            st.session_state.is_playing = False
            st.rerun()

        if st.button("Reset Telemetry / ล้างกราฟเริ่มใหม่", use_container_width=True):
            st.session_state.step = 0
            st.session_state.t2_hist = [0.0] * 60
            st.session_state.q_hist = [0.0] * 60
            st.session_state.scores_hist = [[0.0, 0.0]] * 60
            st.session_state.contribs = [0.0, 0.0, 0.0, 0.0]
            st.session_state.active_fault = 'Normal (Clean Water)'
            st.session_state.is_playing = False
            st.session_state.raw_vals = PHYSICAL_MEANS.copy().tolist()
            st.session_state.prev_raw = PHYSICAL_MEANS.copy().tolist()
            st.rerun()

        st.divider()
        st.subheader("Inject Contaminant")
        fault_keys = list(FAULTS.keys())
        new_fault = st.radio(
            "Select substance",
            fault_keys,
            index=fault_keys.index(st.session_state.active_fault)
        )
        if new_fault != st.session_state.active_fault:
            st.session_state.active_fault = new_fault
            st.rerun()

    # ── PCA Inference function ──────────────────────────────────────────
    def process_sample():
        offset = FAULTS[st.session_state.active_fault]
        noise  = np.random.randn(4) * PHYSICAL_STDS
        raw    = PHYSICAL_MEANS + offset + noise

        # Standardize using TRAINED scaler
        z = st.session_state.scaler.transform(raw.reshape(1, -1))[0]

        # PCA scores
        scores = st.session_state.pca_model.transform(z.reshape(1, -1))[0]

        # Hotelling T²
        ev   = st.session_state.pca_model.explained_variance_
        T2   = float(np.sum((scores / np.sqrt(ev)) ** 2))

        # SPE / Q
        z_hat = st.session_state.pca_model.inverse_transform(scores.reshape(1, -1))[0]
        Q     = float(np.sum((z - z_hat) ** 2))

        # Contribution (clipped to [0,1] and normalised)
        contrib = np.clip((z ** 2) / max(T2, 1e-9), 0, 1)
        contrib = contrib / max(contrib.sum(), 1e-9)  # normalise so bars sum to 1

        return raw.tolist(), T2, Q, contrib.tolist()

    # ── Advance one simulation step if playing ──────────────────────────
    if st.session_state.is_playing:
        raw, t2, q, contrib = process_sample()
        st.session_state.prev_raw  = st.session_state.raw_vals[:]
        st.session_state.raw_vals  = raw
        st.session_state.t2_hist.append(t2);  st.session_state.t2_hist.pop(0)
        st.session_state.q_hist.append(q);    st.session_state.q_hist.pop(0)
        st.session_state.contribs  = contrib
        st.session_state.step     += 1

        # Update scores_hist buffer
        z = st.session_state.scaler.transform(np.array(raw).reshape(1, -1))[0]
        cur_score = st.session_state.pca_model.transform(z.reshape(1, -1))[0].tolist()
        st.session_state.scores_hist.append(cur_score)
        st.session_state.scores_hist.pop(0)

    is_alarm = (
        st.session_state.t2_hist[-1] > st.session_state.ucl_t2 or
        st.session_state.q_hist[-1]  > st.session_state.ucl_q
    )

    # ── Page header ──────────────────────────────────────────────────────
    st.title("Phase 2: Live Fault Detection (FDD)")
    st.markdown("การตรวจจับสารปนเปื้อนและวินิจฉัย Root Cause แบบ Real-time ด้วยโมเดล PCA ที่ผ่านการเทรนจาก Phase 1")

    with st.expander("คู่มือขั้นตอนการทดสอบแบบ Step-by-Step (Testing Walkthrough Guide)", expanded=not st.session_state.is_playing):
        st.markdown("""
 - Step 1 เริ่มระบบสตรีมข้อมูล: กดปุ่ม **Start** ที่แถบเมนูด้านซ้ายเพื่อเริ่มลูปอ่านค่าเซนเซอร์ทุก 1 วินาที
 - Step 2 สังเกตสภาวะปกติ (NOMINAL): ค่า T² และ Q จะอยู่ต่ำกว่าเส้นประสีแดง UCL และพิกัด Biplot จะเกาะกลุ่มในวงรีสีแดง
 - Step 3 ฉีดสารเคมีปนเปื้อน (Fault Injection): เลือกสารเคมีที่เมนูด้านซ้าย (NaCl, Acetic Acid, Milk Protein หรือ Sensor Drift)
 - Step 4 สังเกตการแจ้งเตือน (Anomaly Detection): ค่าสถิติ T² หรือ Q จะพุ่งทะลุเส้น UCL และระบบจะขึ้นแถบแดงเตือน **ALARM**
 - Step 5 วินิจฉัยหาสาเหตุ (Root Cause Isolation): ดูที่กราฟ **Contribution Plot** ระบบจะไฮไลต์แท่งสีแดงชี้เป้าเซนเซอร์ที่เป็นต้นเหตุทันที
""")

    if not st.session_state.is_playing:
        st.info("Press **Start** in the sidebar to begin the sensor loop.")
    elif is_alarm:
        worst_contrib_idx = int(np.argmax(st.session_state.contribs))
        st.error(
            f"**ALARM:** Fault detected — root cause isolated to "
            f"**{LABELS[worst_contrib_idx]}** sensor.  "
            f"(Injected: {st.session_state.active_fault})"
        )
    else:
        st.success("**NOMINAL:** All metrics within trained UCL bounds.")

    st.divider()

    col_vis, col_charts = st.columns([1, 1.3], gap="medium")

    # ── Left: Digital-Twin SVG & Metrics ─────────────────────────────────
    with col_vis:
        st.subheader("Physical Setup")

        w_color  = WATER_COLORS[st.session_state.active_fault]
        is_fault = st.session_state.active_fault != 'Normal (Clean Water)'
        status_led_color = '#DC2626' if is_alarm else '#10B981'
        status_led_text = 'ALARM' if is_alarm else 'ONLINE'

        # Build contamination drop indicator safely (no f-string quote clash)
        drop_svg = (
            f'<circle cx="225" cy="115" r="5" fill="{w_color}" opacity="0.9"/>'
            if is_fault else ''
        )

        svg = f"""<div style="background:#ffffff;border:1px solid #E2E8F0;border-radius:8px;
                       padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:12px;">
        <svg viewBox="0 0 500 320" width="100%" xmlns="http://www.w3.org/2000/svg">

          <!-- Background grid -->
          <defs>
            <pattern id="g" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M40 0L0 0 0 40" fill="none" stroke="#F1F5F9" stroke-width="1"/>
            </pattern>
          </defs>
          <rect width="500" height="320" fill="url(#g)"/>

          <!-- Inlet pipe -->
          <path d="M0 200 L100 200" stroke="#CBD5E1" stroke-width="16" fill="none" stroke-linecap="round"/>
          <path d="M0 200 L90 200"  stroke="#3B82F6" stroke-width="4"  fill="none" stroke-dasharray="8 4"/>
          <text x="12" y="188" fill="#64748B" font-size="11" font-family="sans-serif">Water In</text>

          <!-- Outlet pipe -->
          <path d="M350 200 L500 200" stroke="#CBD5E1" stroke-width="16" fill="none" stroke-linecap="round"/>
          <path d="M360 200 L500 200" stroke="#3B82F6" stroke-width="4"  fill="none" stroke-dasharray="8 4"/>
          <text x="390" y="188" fill="#64748B" font-size="11" font-family="sans-serif">Overflow</text>

          <!-- Tank shell -->
          <rect x="100" y="75"  width="250" height="205" rx="10" stroke="#94A3B8" stroke-width="3" fill="#F8FAFC"/>
          <!-- Water body -->
          <rect x="103" y="120" width="244" height="158" rx="7"
                fill="{w_color}" style="transition:fill 1s ease;"/>
          <!-- Water surface line -->
          <line x1="103" y1="120" x2="347" y2="120" stroke="rgba(59,130,246,0.35)" stroke-width="1.5"/>

          <!-- Stirrer shaft + blade -->
          <line x1="225" y1="55" x2="225" y2="240" stroke="#CBD5E1" stroke-width="3"/>
          <rect x="216" y="46" width="18" height="14" rx="2" fill="#64748B"/>  <!-- motor box -->
          <ellipse cx="225" cy="240" rx="28" ry="6" fill="#94A3B8" opacity="0.8"/>

          <!-- Sensor: TEMP -->
          <line x1="140" y1="30" x2="140" y2="170" stroke="#94A3B8" stroke-width="2"/>
          <rect x="120" y="14" width="40" height="18" rx="4" fill="#fff" stroke="#3B82F6" stroke-width="1.8"/>
          <text x="140" y="27" fill="#3B82F6" font-size="11" text-anchor="middle"
                font-family="sans-serif" font-weight="700">TEMP</text>

          <!-- Sensor: TDS -->
          <line x1="185" y1="30" x2="185" y2="155" stroke="#94A3B8" stroke-width="2"/>
          <rect x="165" y="14" width="40" height="18" rx="4" fill="#fff" stroke="#F59E0B" stroke-width="1.8"/>
          <text x="185" y="27" fill="#F59E0B" font-size="11" text-anchor="middle"
                font-family="sans-serif" font-weight="700">TDS</text>

          <!-- Sensor: TURB -->
          <line x1="265" y1="30" x2="265" y2="180" stroke="#94A3B8" stroke-width="2"/>
          <rect x="245" y="14" width="40" height="18" rx="4" fill="#fff" stroke="#64748B" stroke-width="1.8"/>
          <text x="265" y="27" fill="#64748B" font-size="11" text-anchor="middle"
                font-family="sans-serif" font-weight="700">TURB</text>

          <!-- Sensor: pH -->
          <line x1="310" y1="30" x2="310" y2="165" stroke="#94A3B8" stroke-width="2"/>
          <rect x="290" y="14" width="40" height="18" rx="4" fill="#fff" stroke="#10B981" stroke-width="1.8"/>
          <text x="310" y="27" fill="#10B981" font-size="11" text-anchor="middle"
                font-family="sans-serif" font-weight="700">pH</text>

          <!-- ESP32 -->
          <rect x="375" y="44" width="85" height="42" rx="5"
                fill="#fff" stroke="#0F172A" stroke-width="2"/>
          <text x="417" y="60" fill="#0F172A" font-size="11" font-family="sans-serif"
                font-weight="700" text-anchor="middle">ESP32 Node</text>
          <text x="417" y="75" fill="#64748B" font-size="11" font-family="sans-serif"
                text-anchor="middle">{status_led_text}</text>
          <!-- LED status dot -->
          <circle cx="388" cy="52" r="2.5" fill="{status_led_color}" opacity="0.95"/>

          <!-- Data lines up to ESP32 bus -->
          <path d="M140 14 L140 5 L417 5 L417 44"
                stroke="#3B82F6" stroke-width="1.5" fill="none" opacity="0.35"/>
          <path d="M185 14 L185 5" stroke="#F59E0B" stroke-width="1.5" fill="none" opacity="0.35"/>
          <path d="M265 14 L265 5" stroke="#64748B" stroke-width="1.5" fill="none" opacity="0.35"/>
          <path d="M310 14 L310 5" stroke="#10B981" stroke-width="1.5" fill="none" opacity="0.35"/>

          <!-- Contamination drop (only when fault active) -->
          {drop_svg}

        </svg></div>"""

        components.html(svg, height=325, scrolling=False)

        # Live readouts with delta (change from previous tick)
        cur  = st.session_state.raw_vals
        prev = st.session_state.prev_raw
        mc = st.columns(4)
        for i, (lbl, unit) in enumerate(zip(LABELS, UNITS)):
            delta = round(cur[i] - prev[i], 3) if st.session_state.step > 0 else None
            lbl_text = f"{lbl} {unit}" if unit else lbl
            mc[i].metric(lbl_text, f"{cur[i]:.2f}", delta=delta)

    # ── Right: Analytics charts ──────────────────────────────────────────
    with col_charts:
        st.subheader("Multivariate Analytics Engine")

        ucl_t2 = st.session_state.ucl_t2
        ucl_q  = st.session_state.ucl_q
        ev     = st.session_state.pca_model.explained_variance_

        # ── Biplot ──
        a_ellipse = float(np.sqrt(ev[0] * ucl_t2))
        b_ellipse = float(np.sqrt(ev[1] * ucl_t2))
        theta = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = (a_ellipse * np.cos(theta)).tolist()
        y_ellipse = (b_ellipse * np.sin(theta)).tolist()

        fig_biplot = go.Figure()
        fig_biplot.add_trace(go.Scatter(
            x=x_ellipse, y=y_ellipse,
            mode='lines',
            line=dict(color='#DC2626', width=1.8, dash='dash'),
            name='99% UCL Boundary',
            hoverinfo='name'
        ))

        scores_arr = np.array(st.session_state.scores_hist)
        n_pts = len(scores_arr)
        alphas = np.linspace(0.20, 1.0, n_pts)
        pt_colors = [
            f'rgba(220, 38, 38, {alpha:.2f})' if is_alarm else f'rgba(37, 99, 235, {alpha:.2f})'
            for alpha in alphas
        ]
        pt_sizes = [11] * (n_pts - 1) + [14]

        fig_biplot.add_trace(go.Scatter(
            x=scores_arr[:, 0].tolist(),
            y=scores_arr[:, 1].tolist(),
            mode='markers',
            marker=dict(
                size=pt_sizes,
                color=pt_colors,
                line=dict(width=1, color='#1E293B')
            ),
            name='Scores (PC1 vs PC2)',
            hoverinfo='text',
            hovertext=[f"Tick {i+1}: PC1={s[0]:.2f}, PC2={s[1]:.2f}" for i, s in enumerate(scores_arr)]
        ))

        loadings = st.session_state.pca_model.components_
        scale_arrow = min(a_ellipse, b_ellipse) * 0.75
        for j, label in enumerate(LABELS):
            lx = float(loadings[0, j] * scale_arrow)
            ly = float(loadings[1, j] * scale_arrow)
            fig_biplot.add_trace(go.Scatter(
                x=[0, lx], y=[0, ly],
                mode='lines+text',
                line=dict(color='#64748B', width=1.5),
                text=['', label],
                textposition='top center',
                textfont=dict(size=11, color='#0F172A'),
                hoverinfo='text',
                hovertext=f"Loading: {label}"
            ))

        max_rx = max(a_ellipse * 1.35, float(np.max(np.abs(scores_arr[:, 0]))) * 1.15, 4.5)
        max_ry = max(b_ellipse * 1.35, float(np.max(np.abs(scores_arr[:, 1]))) * 1.15, 4.5)

        fig_biplot.update_layout(
            title=dict(text="PCA Score Biplot (PC1 vs PC2)", font=dict(size=12, color='#0F172A')),
            xaxis=dict(
                title=dict(text="PC1 Score", font=dict(size=11)),
                range=[-max_rx, max_rx],
                zeroline=True, zerolinecolor="#E2E8F0", zerolinewidth=1,
                gridcolor="#F1F5F9", tickfont=dict(size=11)
            ),
            yaxis=dict(
                title=dict(text="PC2 Score", font=dict(size=11)),
                range=[-max_ry, max_ry],
                zeroline=True, zerolinecolor="#E2E8F0", zerolinewidth=1,
                gridcolor="#F1F5F9", tickfont=dict(size=11)
            ),
            showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            height=230, margin=dict(l=35, r=15, t=35, b=25),
            font=dict(size=11, color='#0F172A')
        )

        # ── Contribution / Root Cause ──
        contribs = st.session_state.contribs
        max_idx  = int(np.argmax(contribs))
        bar_colors = [
            '#DC2626' if (is_alarm and i == max_idx) else '#94A3B8'
            for i in range(4)
        ]
        fig_c = go.Figure(go.Bar(
            x=LABELS, y=contribs,
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in contribs],
            textposition='outside',
            textfont=dict(size=11, color='#0F172A')
        ))
        fig_c.update_layout(
            title=dict(text="Contribution Plot — Root Cause Isolation", font=dict(size=12, color='#0F172A')),
            xaxis=dict(showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(range=[0, 1.18], title=dict(text="Normalised Contribution", font=dict(size=11)),
                       gridcolor='#F1F5F9', tickfont=dict(size=11)),
            showlegend=False,
            plot_bgcolor='white', paper_bgcolor='white',
            height=230, margin=dict(l=35, r=15, t=35, b=25),
            font=dict(size=11, color='#0F172A')
        )

        # ── T² Chart ──
        fig_t2 = go.Figure()
        line_color_t2 = '#DC2626' if is_alarm else '#2563EB'
        fill_color_t2 = 'rgba(220,38,38,0.12)' if is_alarm else 'rgba(37,99,235,0.10)'
        fig_t2.add_trace(go.Scatter(
            y=st.session_state.t2_hist, mode='lines', name='T²',
            line=dict(color=line_color_t2, width=1.8),
            fill='tozeroy', fillcolor=fill_color_t2
        ))
        fig_t2.add_hline(
            y=ucl_t2, line_dash='dash', line_color='#DC2626',
            annotation_text=f"UCL={ucl_t2:.3f}", annotation_font_size=11,
            annotation_position="top right"
        )
        fig_t2.update_layout(
            title=dict(text="Hotelling's T² (Within-Model Distance)", font=dict(size=12, color='#0F172A')),
            xaxis=dict(showgrid=False, title=dict(text="Time (ticks)", font=dict(size=11)), tickfont=dict(size=11)),
            yaxis=dict(range=[0, max(ucl_t2 * 2.2, float(np.max(st.session_state.t2_hist)) * 1.15, 0.5)],
                       title=dict(text="T² Value", font=dict(size=11)),
                       gridcolor='#F1F5F9', tickfont=dict(size=11)),
            showlegend=False,
            plot_bgcolor='white', paper_bgcolor='white',
            height=230, margin=dict(l=35, r=15, t=35, b=25),
            font=dict(size=11, color='#0F172A')
        )

        # ── Q / SPE Chart ──
        fig_q = go.Figure()
        line_color_q = '#DC2626' if is_alarm else '#7C3AED'
        fill_color_q = 'rgba(220,38,38,0.12)' if is_alarm else 'rgba(124,58,237,0.10)'
        fig_q.add_trace(go.Scatter(
            y=st.session_state.q_hist, mode='lines', name='Q',
            line=dict(color=line_color_q, width=1.8),
            fill='tozeroy', fillcolor=fill_color_q
        ))
        fig_q.add_hline(
            y=ucl_q, line_dash='dash', line_color='#DC2626',
            annotation_text=f"UCL={ucl_q:.3f}", annotation_font_size=11,
            annotation_position="top right"
        )
        fig_q.update_layout(
            title=dict(text="SPE / Q-Statistic (Correlation Breakdown)", font=dict(size=12, color='#0F172A')),
            xaxis=dict(showgrid=False, title=dict(text="Time (ticks)", font=dict(size=11)), tickfont=dict(size=11)),
            yaxis=dict(range=[0, max(ucl_q * 2.2, float(np.max(st.session_state.q_hist)) * 1.15, 0.3)],
                       title=dict(text="Q Value", font=dict(size=11)),
                       gridcolor='#F1F5F9', tickfont=dict(size=11)),
            showlegend=False,
            plot_bgcolor='white', paper_bgcolor='white',
            height=230, margin=dict(l=35, r=15, t=35, b=25),
            font=dict(size=11, color='#0F172A')
        )

        # Render 2x2 grid
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.plotly_chart(fig_biplot, use_container_width=True)
        with r1_c2:
            st.plotly_chart(fig_c, use_container_width=True)

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            st.plotly_chart(fig_t2, use_container_width=True)
        with r2_c2:
            st.plotly_chart(fig_q, use_container_width=True)

    # ── Engineering Interpretation & Diagnostics (Phase 2 Explainer) ─────
    st.divider()
    st.subheader("หลักการวิศวกรรมและการวินิจฉัยความผิดปกติ (Phase 2 Engineering Diagnostics)")

    col_diag1, col_diag2 = st.columns(2, gap="large")
    with col_diag1:
        st.markdown("""
**การอ่านค่าและวิเคราะห์ผล 4 มิติกราฟ (Multivariate Chart Interpretation):**
 - PCA Score Biplot: สังเกตพิกัด (PC1, PC2) เทียบกับเส้น 99% Confidence Ellipse หากหลุดวงรีแสดงว่าเกิด Anomaly ทันที
 - Loading Vector Overlay: เวกเตอร์ลูกศรชี้ทิศทางของแต่ละ Sensor ช่วยให้ทราบว่าพิกัดที่เบี่ยงเบนถูกดึงด้วยตัวแปรใด
 - Hotelling's T² Metric: ตรวจจับ Systematic Shift ภายในระนาบโมเดล เหมาะกับการเพิ่มขึ้นอย่างพร้อมเพรียงของตัวแปร
 - SPE / Q-Statistic Metric: ตรวจจับ Residual Variance นอกระนาบโมเดล ชี้วัดว่า Correlation ระหว่างตัวแปรถูกทำลาย
 - Contribution Plot: ทำ Fault Isolation คำนวณสัดส่วน Reconstruction Error ชี้เป้า Sensor ที่เป็น Root Cause อัตโนมัติ
""")

    with col_diag2:
        st.markdown("""
**พฤติกรรมทางเคมีและการตอบสนองของเซนเซอร์ (Contaminant Chemical Fingerprints):**
 - NaCl (Saline Solution): เพิ่มประจุไอออนในสารละลาย ดันค่า TDS พุ่งสูง ดีดค่า T² ทะลุ UCL และชี้เป้า TDS
 - Acetic Acid (Vinegar): ปล่อยประจุ H+ ลดค่า pH อย่างรวดเร็ว ทำลาย Correlation ปกติ ดีดค่า Q ทะลุ UCL ทันที
 - Milk Protein: อนุภาคแขวนลอยกระเจิงแสงในน้ำ ดันค่า Turbidity สูงลิ่ว ดีดทั้ง T² และ Q พร้อมระบุ Turbidity
 - Sensor Drift: สัญญาณอุณหภูมิไต่ระดับผิดธรรมชาติจาก Hardware Degradation ทริกเกอร์เตือนอุปกรณ์ชำรุดทันที
 - Alarm Rationalization: การประเมินคู่ทั้ง T² และ Q ช่วยกรอง False Alarm ที่เกิดจาก Sensor Noise ชั่วขณะได้อย่างสมบูรณ์
""")

    # ── Auto-rerun loop ──────────────────────────────────────────────────
    if st.session_state.is_playing:
        time.sleep(1)
        st.rerun()
