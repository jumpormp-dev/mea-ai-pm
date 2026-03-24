import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & MODEL LOAD ---
st.set_page_config(page_title="SPP-AI: KTD Smart City", layout="wide")

@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_spp_ai_model.pkl' กรุณาอัปโหลดบน GitHub")

# --- 2. INITIAL DATABASE (KTD 20 UNITS) ---
if 'ktd_assets' not in st.session_state:
    ktd_feeders = ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'SAM-13', 'RPR-423'] * 3
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': ktd_feeders[:20],
        'Lat': np.random.uniform(13.70, 13.72, 20),
        'Lon': np.random.uniform(100.55, 100.58, 20),
        'Temp_Meter': np.random.uniform(50, 70, 20),
        'Load_Meter': np.random.uniform(40, 80, 20),
        'Trips_KTD': [0]*20,
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Age': np.random.randint(5, 30, 20),
        'Acoustic_dB': [45.0]*20,
        'Peak_Hz': [20000]*20,
        'Last_PM': [(datetime.now() - timedelta(days=np.random.randint(30, 300))).strftime('%d/%m/%Y') for _ in range(20)]
    })

# --- 3. UI TABS NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Dashboard", 
    "🔍 Transformer Detail & Diagnostics", 
    "📅 Predictive Action Plan", 
    "⚙️ System & AI Settings"
])

# ---------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# ---------------------------------------------------------
with tab1:
    st.header("🏙️ KTD Smart City Executive Overview")
    
    # Overview Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    total = len(st.session_state.ktd_assets)
    crit = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🔴 CRITICAL"])
    watch = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🟡 WATCH"])
    
    c1.metric("หม้อแปลงทั้งหมด", f"{total} เครื่อง")
    c2.metric("🔴 CRITICAL", f"{crit} เครื่อง")
    c3.metric("🟡 WATCH", f"{watch} เครื่อง")
    c4.metric("🟢 NORMAL", f"{total-crit-watch} เครื่อง")

    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        st.write("### 📍 GIS Risk Map (เขตคลองเตย)")
        color_map = {'🔴 CRITICAL': 'red', '🟡 WATCH': 'orange', '🟢 NORMAL': 'green'}
        fig_map = px.scatter_mapbox(st.session_state.ktd_assets, lat="Lat", lon="Lon", 
                                    color="Status", size="Risk_Score",
                                    color_discrete_map=color_map,
                                    zoom=13, height=500, mapbox_style="carto-positron",
                                    hover_name="Transformer_ID")
        st.plotly_chart(fig_map, use_container_width=True)

    with col_list:
        st.write("### 🚨 Top Urgent List (ภายใน 7 วัน)")
        urgent_df = st.session_state.ktd_assets.sort_values('Risk_Score', ascending=False).head(5)
        for _, row in urgent_df.iterrows():
            days = int(max(2, (1 - row['Risk_Score']) * 90))
            st.warning(f"**{row['Transformer_ID']}** (Risk: {row['Risk_Score']*100:.1f}%) \n\n PM ภายใน: {days} วัน")

# ---------------------------------------------------------
# TAB 2: TRANSFORMER DETAIL & DIAGNOSTICS (Explainability)
# ---------------------------------------------------------
with tab2:
    st.header("🔍 AI Diagnostic Analysis")
    col_sel, col_sync = st.columns([2, 1])
    with col_sel:
        target_id = st.selectbox("เลือกหม้อแปลงที่ต้องการตรวจสอบ:", st.session_state.ktd_assets['Transformer_ID'])
    with col_sync:
        if st.button("📡 Sync Real-time Data"):
            st.toast("กำลังดึงข้อมูลจาก IP 172.16.111.184...")

    res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].iloc[0]
    
    # Gauge & Countdown
    d1, d2, d3 = st.columns([1, 1, 1.5])
    with d1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = res['Risk_Score']*100,
            title = {'text': "Risk Score (%)"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "red" if res['Risk_Score'] > 0.7 else "orange"},
                     'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 75], 'color': "yellow"}]}))
        st.plotly_chart(fig_gauge, use_container_width=True)
    with d2:
        days_rem = int(max(2, (1 - res['Risk_Score']) * 90))
        st.markdown(f"<h3 style='text-align: center;'>Remaining Days</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{days_rem} วัน</h1>", unsafe_allow_html=True)
    with d3:
        st.info("#### 🤖 AI Logic Insights")
        if res['Risk_Score'] > 0.4:
            st.write(f"**เหตุผล:** ตรวจพบ Pattern เสียง Acoustic ({res['Acoustic_dB']} dB) ที่ความถี่สูงร่วมกับสถิติไฟดับสะสมในฟีดเดอร์ {res['Feeder']} แม้ Load จะยังไม่เกินเกณฑ์")
        else:
            st.write("**เหตุผล:** พารามิเตอร์ทั้งหมดอยู่ในเกณฑ์มาตรฐานตามโมเดลพยากรณ์")

    # Monitoring Graphs
    st.write("---")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.write("**📡 Smart Meter (Historical Temp/Load)**")
        chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['Temp', 'Load'])
        st.line_chart(chart_data)
    with g2:
        st.write("**🔊 Acoustic Spectrum**")
        if res['Acoustic_dB'] > 70: st.markdown("<p style='color:red;'>⚠️ High Intensity Detected</p>", unsafe_allow_html=True)
        st.bar_chart(np.random.rand(10))
    with g3:
        st.write("**📜 Contextual & Reliability**")
        st.write(f"อายุ: {res['Age']} ปี | ไฟดับ KTD: {res['Trips_KTD']} ครั้ง")
        st.write(f"บำรุงรักษาล่าสุด: {res['Last_PM']}")

# ---------------------------------------------------------
# TAB 3: PREDICTIVE ACTION PLAN
# ---------------------------------------------------------
with tab3:
    st.header("📅 Maintenance Task Management")
    c_f1, c_f2, c_f3 = st.columns(3)
    f_feeder = c_f1.multiselect("Filter Feeder", st.session_state.ktd_assets['Feeder'].unique())
    f_stat = c_f2.multiselect("Filter Status", ["🔴 CRITICAL", "🟡 WATCH", "🟢 NORMAL"])
    
    st.write("### Current Work Orders")
    for idx, row in st.session_state.ktd_assets.iterrows():
        if row['Status'] != '🟢 NORMAL':
            with st.expander(f"{row['Status']} - {row['Transformer_ID']} (Feeder: {row['Feeder']})"):
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                if col_btn1.button("Create Work Order", key=f"wo_{idx}"): st.success("ส่งข้อมูลเข้า SAP/EAM แล้ว")
                if col_btn2.button("Schedule Thermo Scan", key=f"ts_{idx}"): st.info("ลงคิวตรวจสอบทีมบำรุงรักษา")
                if col_btn3.button("Mark as Resolved", key=f"re_{idx}"):
                    st.session_state.ktd_assets.at[idx, 'Status'] = '🟢 NORMAL'
                    st.session_state.ktd_assets.at[idx, 'Risk_Score'] = 0.05
                    st.rerun()

# ---------------------------------------------------------
# TAB 4: SYSTEM & MODEL SETTINGS
# ---------------------------------------------------------
with tab4:
    st.header("⚙️ System Configuration")
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        st.write("#### AI Model Tuning")
        st.slider("Age Penalty (Days to reduce if > 25 yrs)", 0, 30, 15)
        st.slider("Acoustic Weight Importance", 0.1, 1.0, 0.8)
    with col_set2:
        st.write("#### Sensor Connectivity")
        st.success("🟢 Smart Meter API: Connected (172.16.111.184)")
        st.success("🟢 Acoustic Camera Node: Online")
        st.write(f"Last Model Update: {datetime.now().strftime('%Y-%m-%d')}")
