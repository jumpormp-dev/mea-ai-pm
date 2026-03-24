import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. SETTINGS & MODEL LOAD ---
st.set_page_config(page_title="SPP-AI: KTD Smart City", layout="wide")

@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์ 'mea_spp_ai_model.pkl' บน GitHub")

# --- 2. INITIAL DATABASE (KTD 20 UNITS) ---
if 'ktd_assets' not in st.session_state:
    ktd_feeders = ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'SAM-13', 'RPR-423'] * 3
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': ktd_feeders[:20],
        'Location_Lat': np.random.uniform(13.70, 13.72, 20),
        'Location_Lon': np.random.uniform(100.55, 100.58, 20),
        'Temp': np.random.uniform(50, 70, 20),
        'Load': np.random.uniform(40, 80, 20),
        'Trips': [0]*20,
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Age': np.random.randint(5, 30, 20),
        'Acoustic_dB': [45.0]*20
    })

# --- 3. MAIN NAVIGATION (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Dashboard", 
    "🔍 Transformer Detail", 
    "📅 Action Plan", 
    "⚙️ System Settings"
])

# ---------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD (ภาพรวมใน 3 วินาที)
# ---------------------------------------------------------
with tab1:
    st.header("🏙️ KTD Smart City Overview")
    
    # Overview Cards
    c1, c2, c3, c4 = st.columns(4)
    total = len(st.session_state.ktd_assets)
    crit = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🔴 CRITICAL"])
    watch = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🟡 WATCH"])
    
    c1.metric("หม้อแปลงทั้งหมด", f"{total} เครื่อง")
    c2.metric("🔴 CRITICAL", f"{crit} เครื่อง", delta_color="inverse")
    c3.metric("🟡 WATCH", f"{watch} เครื่อง")
    c4.metric("🟢 NORMAL", f"{total-crit-watch} เครื่อง")

    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        st.write("### 📍 GIS Risk Map (พื้นที่เขตคลองเตย)")
        map_data = st.session_state.ktd_assets.copy()
        # กำหนดสีตามสถานะ
        color_map = {'🔴 CRITICAL': 'red', '🟡 WATCH': 'orange', '🟢 NORMAL': 'green'}
        map_data['color'] = map_data['Status'].map(color_map)
        fig_map = px.scatter_mapbox(map_data, lat="Location_Lat", lon="Location_Lon", 
                                    color="Status", size="Risk_Score",
                                    color_discrete_map=color_map,
                                    zoom=13, height=500, mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)

    with col_list:
        st.write("### 🚨 Top Urgent List")
        urgent_df = st.session_state.ktd_assets.sort_values('Risk_Score', ascending=False).head(5)
        for _, row in urgent_df.iterrows():
            with st.expander(f"{row['Status']} | {row['Transformer_ID']}"):
                st.write(f"Feeder: {row['Feeder']} | Risk: {row['Risk_Score']*100:.1f}%")
                st.button("View Detail", key=row['Transformer_ID']+"_btn")

# ---------------------------------------------------------
# TAB 2: TRANSFORMER DETAIL (ทำไม AI ถึงบอกว่าเสี่ยง?)
# ---------------------------------------------------------
with tab2:
    st.header("🔍 Transformer Detail & AI Diagnostics")
    selected_id = st.selectbox("เลือกหม้อแปลงเพื่อดูวิเคราะห์เจาะลึก:", st.session_state.ktd_assets['Transformer_ID'])
    res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == selected_id].iloc[0]

    # ส่วนพยากรณ์: Gauge & Countdown
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
        days_left = int(max(2, (1 - res['Risk_Score']) * 90))
        st.metric("Remaining Days to PM", f"{days_left} วัน", delta="-2 วัน" if res['Risk_Score'] > 0.5 else None)
        st.info(f"📅 กำหนดทำ PM: {(datetime.now() + timedelta(days=days_left)).strftime('%d/%m/%Y')}")

    with d3:
        st.write("#### 🤖 AI Logic Insights")
        st.success(f"**Pattern Detection:** พบความสัมพันธ์ของอุณหภูมิที่สูงผิดปกติร่วมกับเสียง Acoustic {res['Acoustic_dB']}dB ในช่วงฟีดเดอร์ {res['Feeder']} ที่มีประวัติการทริปบ่อย")

    # Data Monitoring 3 ด้าน
    st.write("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.write("**📡 Smart Meter Data**")
        st.line_chart(np.random.randn(20, 2)) # จำลองเทรน Temp/Load
    with m2:
        st.write("**🔊 Acoustic Camera**")
        st.error(f"Acoustic Peak: {res['Acoustic_dB']} dB") # ไฮไลท์แดงตามโจทย์
    with m3:
        st.write("**📜 Contextual History**")
        st.write(f"- อายุอุปกรณ์: {res['Age']} ปี")
        st.write(f"- ประวัติการทริป (KTD): {res['Trips']} ครั้ง")

# ---------------------------------------------------------
# TAB 3: ACTION PLAN (Kanban & สั่งการ)
# ---------------------------------------------------------
with tab3:
    st.header("📅 Predictive Action Plan")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        f_feeder = st.multiselect("กรองตามฟีดเดอร์:", ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442'])
    
    st.write("### Task List")
    task_df = st.session_state.ktd_assets.copy()
    # แสดงตารางพร้อมปุ่มสั่งการ
    for idx, row in task_df.iterrows():
        if row['Status'] != '🟢 NORMAL':
            c_id, c_stat, c_action = st.columns([1, 1, 2])
            c_id.write(row['Transformer_ID'])
            c_stat.write(row['Status'])
            with c_action:
                if st.button("Create Work Order", key=f"wo_{idx}"): st.toast("ใบสั่งงานถูกส่งไปยังทีมช่างแล้ว")
                if st.button("Mark as Resolved", key=f"res_{idx}"):
                    st.session_state.ktd_assets.at[idx, 'Status'] = '🟢 NORMAL'
                    st.session_state.ktd_assets.at[idx, 'Risk_Score'] = 0.05
                    st.rerun()

# ---------------------------------------------------------
# TAB 4: SYSTEM SETTINGS (Admin & AI Tuning)
# ---------------------------------------------------------
with tab4:
    st.header("⚙️ AI Model & Sensor Settings")
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        st.write("### Weight & Penalty Adjustment")
        p_age = st.slider("Penalty สำหรับอายุ > 25 ปี (วัน)", 5, 30, 15)
        w_acoustic = st.slider("น้ำหนักความสำคัญของเสียง Acoustic", 0.1, 1.0, 0.8)
    
    with col_set2:
        st.write("### 🩺 Sensor Health Status")
        st.write("🟢 Smart Meter: **Online** (172.16.111.184)")
        st.write("🟢 Acoustic Camera: **Online**")
        st.write("🟡 Data Completeness: **92%**")
