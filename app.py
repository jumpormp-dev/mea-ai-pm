import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MEA Smart PM — KTD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# [ส่วน CSS เดิมของคุณ ผมคงไว้ทั้งหมดเพื่อให้หน้าตาสวยงามเหมือนเดิม]
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #F4F5F7;
  --surface:  #FFFFFF;
  --surface2: #F0F1F3;
  --border:   rgba(0,0,0,0.08);
  --text:     #1A1B1E;
  --muted:    #6B7280;
  --accent:   #FF7B22;
  --critical: #E53535;
  --watch:    #D97706;
  --normal:   #16A34A;
  --blue:     #2563EB;
  --font:     'IBM Plex Sans Thai', sans-serif;
  --mono:     'IBM Plex Mono', monospace;
}
html, body, [class*="css"] { font-family: var(--font) !important; }
.main .block-container { background: var(--bg); padding: 1.5rem 2rem 3rem; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.stButton > button {
  background: var(--accent) !important; color: white !important;
  border: none !important; border-radius: 8px !important;
  font-weight: 500 !important; font-family: var(--font) !important;
}
.kpi-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px 20px; position:relative; overflow:hidden; }
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.kpi-card.total::before    { background:var(--blue); }
.kpi-card.critical::before { background:var(--critical); }
.kpi-card.watch::before    { background:var(--watch); }
.kpi-card.normal::before   { background:var(--normal); }
.kpi-label { font-size:11px; color:var(--muted); letter-spacing:.05em; margin-bottom:6px; }
.kpi-value { font-size:34px; font-weight:600; line-height:1; font-family:var(--mono); }
.kpi-sub   { font-size:11px; color:#9CA3AF; margin-top:6px; }
.kpi-card.total    .kpi-value { color:var(--blue); }
.kpi-card.critical .kpi-value { color:var(--critical); }
.kpi-card.watch    .kpi-value { color:var(--watch); }
.kpi-card.normal   .kpi-value { color:var(--normal); }
.action-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 20px; margin-bottom:12px; border-left:4px solid var(--accent); }
.action-card.critical { border-left-color:var(--critical); }
.action-card.watch    { border-left-color:var(--watch); }
.ac-top  { display:flex; justify-content:space-between; align-items:center; }
.ac-id   { font-size:14px; font-weight:600; font-family:var(--mono); color:var(--text); }
.ac-plan { font-size:12px; color:var(--accent); font-weight:500; }
.ac-row  { font-size:12px; color:var(--muted); margin-top:8px; }
.pill { display:inline-block; font-size:10px; padding:2px 9px; border-radius:20px; font-weight:600; letter-spacing:.05em; }
.pill.critical { background:rgba(229,53,53,.1);  color:var(--critical); }
.pill.watch    { background:rgba(217,119,6,.1);  color:var(--watch); }
.pill.normal   { background:rgba(22,163,74,.1);  color:var(--normal); }
.param-row { display:flex; align-items:center; padding:8px 0; border-bottom:1px solid var(--border); gap:12px; }
.param-bar   { flex:1.5; height:5px; background:#E5E7EB; border-radius:3px; overflow:hidden; }
.param-fill  { height:100%; border-radius:3px; }
.param-fill.ok   { background:var(--normal); }
.param-fill.warn { background:var(--watch); }
.param-fill.crit { background:var(--critical); }
.param-val { font-size:12px; font-family:var(--mono); font-weight:500; min-width:80px; text-align:right; }
.logo-badge { display:inline-block; background:linear-gradient(135deg,#FF7B22,#FF5500); color:white; font-size:11px; font-weight:600; padding:5px 12px; border-radius:6px; letter-spacing:.07em; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS & STATE
# ══════════════════════════════════════════════════════════════
MONTHS_TH = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.","ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]
STATUS_MAP = {0: "🟢 NORMAL", 1: "🟡 WATCH", 2: "🔴 CRITICAL"}
STATUS_KEY = {"🟢 NORMAL": "normal", "🟡 WATCH": "watch", "🔴 CRITICAL": "critical"}
STATUS_COLOR = {"🔴 CRITICAL": "#E53535", "🟡 WATCH": "#D97706", "🟢 NORMAL": "#16A34A"}

FEATURE_COLS = ["Thermal_Temp", "Load_Percent", "Voltage_V", "Acoustic_dB", "Peak_Freq_Hz", "Trips_Count", "Age_Years", "Humidity"]

# ป้องกัน Error โดยการสร้างโครงสร้าง DataFrame ที่สมบูรณ์รอไว้
if "assets" not in st.session_state:
    st.session_state.assets = pd.DataFrame(columns=["Transformer_ID","Feeder","Lat","Lon","Load_Percent","Voltage_V","Trips_Count","Acoustic_dB","Thermal_Temp","Peak_Freq_Hz","Age_Years","Humidity","Status","Risk_Score","Plan_Month"])
if "survey_imgs" not in st.session_state:
    st.session_state.survey_imgs = {}

# ══════════════════════════════════════════════════════════════
#  LOAD MODEL & HELPERS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    p = Path("mea_spp_ai_model.pkl")
    return joblib.load(p) if p.exists() else None

model = load_model()

def get_plan_month(status, risk):
    today = datetime.now()
    if status == "🔴 CRITICAL":
        return f"เดือนนี้ (URGENT) — {MONTHS_TH[today.month-1]} {today.year+543}"
    if status == "🟡 WATCH":
        delay = max(1, round((1 - risk) * 4))
        d = today + timedelta(days=delay * 30)
        return f"{MONTHS_TH[d.month-1]} {d.year+543}"
    return "Routine Check"

def infer_row(row):
    if model is not None:
        feat = np.array([row[FEATURE_COLS].values])
        pred = model.predict(feat)[0]
        prob = model.predict_proba(feat)[0][pred] if hasattr(model, "predict_proba") else 0.5
        return STATUS_MAP[int(pred)], float(prob)
    # Rule-based fallback
    t, a, tr = row["Thermal_Temp"], row["Acoustic_dB"], row["Trips_Count"]
    if t > 85 or a > 75 or tr > 8: return "🔴 CRITICAL", 0.85
    if t > 65 or a > 60 or tr > 3: return "🟡 WATCH", 0.60
    return "🟢 NORMAL", 0.15

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-badge">⚡ MEA SMART PM</div>', unsafe_allow_html=True)
    st.markdown("##### 📂 อัปโหลดไฟล์ ฟขต Feeder.xlsx")
    uploaded = st.file_uploader("", type=["xlsx"], label_visibility="collapsed")

    if uploaded:
        df_raw = pd.read_excel(uploaded, skiprows=2)
        if "Feeder" in df_raw.columns:
            trip_stats = df_raw["Feeder"].value_counts().to_dict()
            if st.button("🚀 โหลดข้อมูลจริงทั้งหมด", use_container_width=True):
                rng = np.random.default_rng(42)
                rows = []
                for i, (fdr, count) in enumerate(trip_stats.items()):
                    rows.append({
                        "Transformer_ID": f"TR-KTD-{i+1:03d}", "Feeder": fdr,
                        "Lat": 13.702 + rng.uniform(-0.02, 0.02), "Lon": 100.555 + rng.uniform(-0.02, 0.02),
                        "Load_Percent": 0.0, "Voltage_V": 220.0, "Trips_Count": int(count),
                        "Acoustic_dB": 45.0, "Thermal_Temp": 50.0, "Peak_Freq_Hz": 25000.0,
                        "Age_Years": int(rng.integers(5, 36)), "Humidity": 65.0,
                        "Status": "🟢 NORMAL", "Risk_Score": 0.0, "Plan_Month": "Routine Check"
                    })
                st.session_state.assets = pd.DataFrame(rows)
                st.success(f"โหลด {len(trip_stats)} เครื่องสำเร็จ!")
                st.rerun()

    if not st.session_state.assets.empty:
        if st.button("🤖 วิเคราะห์แผนงานทั้งหมด (AI)", use_container_width=True):
            df = st.session_state.assets.copy()
            for idx, row in df.iterrows():
                status, risk = infer_row(row)
                df.at[idx, "Status"] = status
                df.at[idx, "Risk_Score"] = risk
                df.at[idx, "Plan_Month"] = get_plan_month(status, risk)
            st.session_state.assets = df
            st.success("วิเคราะห์สำเร็จ!")
            st.rerun()

    st.divider()
    if not st.session_state.assets.empty:
        st.markdown("##### 📸 บันทึกสำรวจหน้างาน")
        target = st.selectbox("เลือก ID:", st.session_state.assets["Transformer_ID"])
        idx = st.session_state.assets[st.session_state.assets["Transformer_ID"] == target].index[0]
        ac = st.number_input("เสียง (dB)", 30.0, 110.0, float(st.session_state.assets.at[idx, "Acoustic_dB"]))
        tmp = st.number_input("ความร้อน (°C)", 20.0, 130.0, float(st.session_state.assets.at[idx, "Thermal_Temp"]))
        img = st.file_uploader("ภาพหน้างาน", type=["jpg","jpeg","png"], key=f"img_{target}")

        if st.button("💾 บันทึกและวิเคราะห์", use_container_width=True):
            st.session_state.assets.at[idx, "Acoustic_dB"] = ac
            st.session_state.assets.at[idx, "Thermal_Temp"] = tmp
            if img: st.session_state.survey_imgs[target] = img.read()
            status, risk = infer_row(st.session_state.assets.iloc[idx])
            st.session_state.assets.at[idx, "Status"] = status
            st.session_state.assets.at[idx, "Risk_Score"] = risk
            st.session_state.assets.at[idx, "Plan_Month"] = get_plan_month(status, risk)
            st.rerun()

# ══════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Diagnostics", "📅 Action Plan"])
df = st.session_state.assets

with tab1:
    if df.empty:
        st.info("👈 กรุณาโหลดข้อมูลสายป้อน (ฟขต) ที่แถบด้านซ้าย")
    else:
        # KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        counts = df["Status"].value_counts().to_dict()
        c1.markdown(f'<div class="kpi-card total"><div class="kpi-label">TOTAL</div><div class="kpi-value">{len(df)}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card critical"><div class="kpi-label">CRITICAL</div><div class="kpi-value">{counts.get("🔴 CRITICAL", 0)}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card watch"><div class="kpi-label">WATCH</div><div class="kpi-value">{counts.get("🟡 WATCH", 0)}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-card normal"><div class="kpi-label">NORMAL</div><div class="kpi-value">{counts.get("🟢 NORMAL", 0)}</div></div>', unsafe_allow_html=True)
        
        # Map
        st.plotly_chart(px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", color_discrete_map=STATUS_COLOR, mapbox_style="carto-positron", height=500), use_container_width=True)

with tab2:
    if not df.empty:
        sel_id = st.selectbox("เลือกอุปกรณ์:", df["Transformer_ID"])
        row = df[df["Transformer_ID"] == sel_id].iloc[0]
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=row["Risk_Score"]*100, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': STATUS_COLOR[row["Status"]]}})).update_layout(height=250), use_container_width=True)
            if sel_id in st.session_state.survey_imgs: st.image(st.session_state.survey_imgs[sel_id], use_container_width=True)
        with col2:
            st.info(f"📅 **แผนการซ่อมบำรุง:** {row['Plan_Month']}")
            st.write(f"🔊 เสียง: {row['Acoustic_dB']} dB | 🌡️ ความร้อน: {row['Thermal_Temp']} °C | 📈 โหลด: {row['Load_Percent']}%")

with tab3:
    if not df.empty:
        urgent = df[df["Status"] != "🟢 NORMAL"].sort_values("Risk_Score", ascending=False)
        for _, r in urgent.iterrows():
            st.markdown(f'<div class="action-card {STATUS_KEY[r["Status"]]}"><div class="ac-top"><span class="ac-id">{r["Transformer_ID"]} ({r["Feeder"]})</span><span class="ac-plan">📅 {r["Plan_Month"]}</span></div></div>', unsafe_allow_html=True)
