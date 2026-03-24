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
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        'Temp_Meter': [0.0]*20,
        'Load_Meter': [0.0]*20,
        'Trips_KTD': [0]*20,
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Age': np.random.randint(5, 30, 20),
        'Acoustic_dB': [45.0]*20,
        'Peak_Hz': [20000]*20,
        'Last_Update': ['-']*20
    })

# --- 3. SIDEBAR: DATA UPLOAD CHANNELS (ช่องทางการอัปโหลดข้อมูล) ---
st.sidebar.header("📥 Data Management Center")

# ช่องทางที่ 1: Sync Smart Meter (API จำลอง)
st.sidebar.subheader("1. Smart Meter Sync")
if st.sidebar.button("📡 Sync KTD Meter (172.16.111.184)"):
    st.session_state.ktd_assets['Temp_Meter'] = np.random.uniform(50, 95, 20)
    st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 120, 20)
    st.session_state.ktd_assets['Last_Update'] = datetime.now().strftime('%H:%M:%S')
    st.sidebar.success("✅ เชื่อมต่อมิเตอร์อัจฉริยะสำเร็จ")

# ช่องทางที่ 2: อัปโหลดไฟล์ Reliability (ฟขต. / KTD)
st.sidebar.subheader("2. Reliability Data (Excel)")
feeder_file = st.sidebar.file_uploader("เลือกไฟล์สถิติไฟดับ ฟขต.", type=["xlsx"])
if feeder_file:
    df_f = pd.read_excel(feeder_file, skiprows=2)
    counts = df_f['Feeder'].value_counts().to_dict()
    for fid, c in counts.items():
        st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
    st.session_state.raw_feeder_df = df_f 
    st.sidebar.success("✅ อัปเดตข้อมูลรายฟีดเดอร์แล้ว")

# ช่องทางที่ 3: บันทึกผลสำรวจ Acoustic (Manual Input)
st.sidebar.subheader("3. Field Survey (Acoustic)")
target = st.sidebar.selectbox("รหัสหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
ac_db_in = st.sidebar.number_input("ค่าความดัง (dB)", 30.0, 120.0, 45.0)
ac_hz_in = st.sidebar.number_input("ความถี่ Peak (Hz)", 1000, 100000, 20000)

if st.sidebar.button("🧠 AI Analyze & Update"):
    idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
    row = st.session_state.ktd_assets.iloc[idx]
    
    # Input 8 Features ให้โมเดล
    features = np.array([[row['Temp_Meter'], row['Load_Meter'], 230.0, ac_db_in, ac_hz_in, row['Trips_KTD'], row['Age'], 55.0]])
    prob = model.predict_proba(features)[0][1]
    
    # อัปเดตสถานะ
    new_stat = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
    st.session_state.ktd_assets.at[idx, 'Status'] = new_stat
    st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
    st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_db_in
    st.session_state.ktd_assets.at[idx, 'Peak_Hz'] = ac_hz_in
    st.sidebar.success(f"วิเคราะห์ {target} เรียบร้อย!")

# --- 4. MAIN NAVIGATION (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Dashboard", 
    "🔍 Transformer Detail", 
    "📅 Action Plan", 
    "⚙️ Settings"
])

# 
# ---------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# ---------------------------------------------------------
with tab1:
    st.header("🏙️ SPP-AI: KTD Smart City Dashboard")
    
    # Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    crit = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🔴 CRITICAL"])
    watch = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🟡 WATCH"])
    
    c1.metric("หม้อแปลงทั้งหมด", "20 ตัว")
    c2.metric("🔴 CRITICAL", f"{crit} ตัว", delta="ด่วนที่สุด", delta_color="inverse")
    c3.metric("🟡 WATCH", f"{watch} ตัว")
    c4.metric("🟢 NORMAL", f"{20-crit-watch} ตัว")

    col_map, col_list = st.columns([2, 1])
    with col_map:
        st.write("### 📍 GIS Risk Map")
        color_map = {'🔴 CRITICAL': 'red', '🟡 WATCH': 'orange', '🟢 NORMAL': 'green'}
        fig_map = px.scatter_mapbox(st.session_state.ktd_assets, lat="Lat", lon="Lon", 
                                    color="Status", size="Risk_Score",
                                    color_discrete_map=color_map, zoom=13, height=450,
                                    mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
    with col_list:
        st.write("### 🚨 Top Urgent List")
        st.dataframe(st.session_state.ktd_assets.sort_values('Risk_Score', ascending=False)[['Transformer_ID', 'Status']].head(5), hide_index=True)

# ---------------------------------------------------------
# TAB 2: TRANSFORMER DETAIL (Explainability)
# ---------------------------------------------------------
with tab2:
    st.header("🔍 Deep Diagnostics & AI Insights")
    sel_id = st.selectbox("เลือกหม้อแปลงที่ต้องการตรวจสอบ:", st.session_state.ktd_assets['Transformer_ID'])
    res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == sel_id].iloc[0]

    d1, d2, d3 = st.columns([1, 1, 1.5])
    with d1:
        st.write("**Risk Score Gauge**")
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red" if res['Risk_Score'] > 0.7 else "orange"}}))
        fig_g.update_layout(height=300)
        st.plotly_chart(fig_g, use_container_width=True)
    with d2:
        days = int(max(2, (1 - res['Risk_Score']) * 90))
        st.markdown(f"<h3 style='text-align: center;'>Remaining Days</h3><h1 style='text-align: center; color: red; font-size: 70px;'>{days}</h1>", unsafe_allow_html=True)
    with d3:
        st.info("#### 🤖 AI Logic Insights")
        st.write(f"**Pattern:** พบความเสี่ยงจากเสียง {res['Acoustic_dB']}dB ร่วมกับสถิติไฟดับใน {res['Feeder']} ({res['Trips_KTD']} ครั้ง)")

    st.write("---")
    m1, m2, m3 = st.columns(3)
    m1.write("**📡 Smart Meter (Temp/Load)**")
    m1.line_chart(np.random.randn(10, 2))
    m2.write("**🔊 Acoustic Peak**")
    m2.bar_chart(np.random.rand(10))
    m3.write("**📜 Reliability Data**")
    m3.write(f"Feeder: {res['Feeder']} | อายุ: {res['Age']} ปี")

# ---------------------------------------------------------
# TAB 3: ACTION PLAN
# ---------------------------------------------------------
with tab3:
    st.header("📅 Maintenance Work Orders")
    for idx, row in st.session_state.ktd_assets.iterrows():
        if row['Status'] != '🟢 NORMAL':
            with st.expander(f"{row['Status']} | {row['Transformer_ID']}"):
                b1, b2 = st.columns(2)
                if b1.button("Create Work Order", key=f"b1_{idx}"): st.success("สั่งงานแล้ว")
                if b2.button("✅ Mark as Resolved", key=f"b2_{idx}"):
                    st.session_state.ktd_assets.at[idx, 'Status'] = '🟢 NORMAL'
                    st.session_state.ktd_assets.at[idx, 'Risk_Score'] = 0.05
                    st.rerun()

# ---------------------------------------------------------
# TAB 4: SYSTEM SETTINGS
# ---------------------------------------------------------
with tab4:
    st.header("⚙️ AI Model & Sensor Health")
    s1, s2 = st.columns(2)
    s1.slider("Age Penalty (Days)", 0, 30, 15)
    s2.write("#### 🩺 Sensor Health")
    s2.write("🟢 Smart Meter: Online | 🟢 Acoustic: Online")
