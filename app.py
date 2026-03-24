import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & CUSTOM STYLE ---
st.set_page_config(page_title="SPP-AI: KTD Smart City", layout="wide")

# ปรับแต่ง CSS สำหรับโทนสีส้ม MEA และความทันสมัย
st.markdown("""
    <style>
    .main { background-color: #F4F7F9; }
    .stMetric { background-color: #FFFFFF; padding: 25px; border-radius: 15px; border-left: 8px solid #FF8C00; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background-color: #FFFFFF; border-radius: 10px; border: 1px solid #E0E0E0; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; height: 3em; }
    h1, h2, h3 { color: #333333; font-family: 'Segoe UI', sans-serif; }
    .stTab { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_spp_ai_model.pkl' บน GitHub")

# --- 2. INITIAL DATABASE (KTD 20 UNITS) ---
if 'ktd_assets' not in st.session_state:
    ktd_feeders = ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'SAM-13', 'RPR-423'] * 3
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': ktd_feeders[:20],
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        'Load_Meter': [0.0]*20,
        'Trips_KTD': [0]*20,
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Acoustic_dB': [45.0]*20,
        'Thermal_Temp': [55.0]*20,
        'Age': np.random.randint(5, 30, 20),
        'Last_Survey_Img': [None]*20
    })

# --- 3. SIDEBAR: DATA INPUT CHANNELS ---
with st.sidebar:
    st.header("📥 Data Management")
    
    # 3.1 Smart Meter Sync
    st.subheader("1. Smart Meter (API)")
    if st.button("📡 Sync Real-time Load"):
        st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 110, 20)
        st.success("Load Data Synced")
    st.caption("🔴 Temp Sensor: Offline (No Data)")

    # 3.2 Reliability Upload
    st.subheader("2. Reliability (Excel)")
    feeder_file = st.file_uploader("Upload ฟขต Feeder.xlsx", type=["xlsx"])
    if feeder_file:
        df_f = pd.read_excel(feeder_file, skiprows=2)
        counts = df_f['Feeder'].value_counts().to_dict()
        for fid, c in counts.items():
            st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
        st.success("Reliability Updated")

    st.divider()
    
    # 3.3 Field Survey (Acoustic & Thermal)
    st.subheader("3. Field Survey Data")
    target = st.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        ac_db = st.number_input("เสียง (dB)", 30.0, 120.0, 45.0, help="ข้อมูลจาก Acoustic Camera")
    with col_in2:
        th_temp = st.number_input("ความร้อน (°C)", 20.0, 120.0, 55.0, help="ข้อมูลจาก Thermal Scan")
    
    survey_img = st.file_uploader("📷 อัปโหลดรูปภาพหน้างาน", type=["jpg", "png", "jpeg"])
    if survey_img:
        st.image(survey_img, caption="Preview: หลักฐานหน้างาน", use_column_width=True)

    if st.button("🧠 AI Analyze & Planning"):
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
        row = st.session_state.ktd_assets.iloc[idx]
        
        # รัน AI Model (ใช้อุณหภูมิ Thermal แทน Smart Meter Temp)
        features = np.array([[th_temp, row['Load_Meter'], 230.0, ac_db, 20000, row['Trips_KTD'], row['Age'], 55.0]])
        prob = model.predict_proba(features)[0][1]
        
        # อัปเดตข้อมูล
        new_stat = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
        st.session_state.ktd_assets.at[idx, 'Status'] = new_stat
        st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_db
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_temp
        if survey_img: st.session_state.ktd_assets.at[idx, 'Last_Survey_Img'] = survey_img
        st.success(f"วิเคราะห์ {target} สำเร็จ")

# --- 4. MAIN DASHBOARD ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Dashboard", 
    "🔍 Diagnostics", 
    "📅 Action Plan", 
    "⚙️ Settings"
])

# --- TAB 1: EXECUTIVE DASHBOARD ---
with tab1:
    st.markdown("## 🏙️ SPP-AI: Executive Command Center")
    m1, m2, m3, m4 = st.columns(4)
    crit_count = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🔴 CRITICAL"])
    watch_count = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🟡 WATCH"])
    
    m1.metric("Total Assets", "20 Units")
    m2.metric("🔴 Critical", crit_count)
    m3.metric("🟡 Watch", watch_count)
    m4.metric("Area Status", "Active (KTD)")

    c_map, c_list = st.columns([2, 1])
    with c_map:
        fig_map = px.scatter_mapbox(st.session_state.ktd_assets, lat="Lat", lon="Lon", color="Status", 
                                    size="Risk_Score", zoom=13, height=450,
                                    color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                    mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
    with c_list:
        st.write("### 🚨 Top Urgent List")
        st.dataframe(st.session_state.ktd_assets.sort_values('Risk_Score', ascending=False)[['Transformer_ID', 'Status']].head(8), hide_index=True)

# --- TAB 2: DIAGNOSTICS (ช่วงเดือน + แยกข้อมูลสำรวจ) ---
with tab2:
    st.markdown("## 🔍 Deep Transformer Diagnostics")
    sel_id = st.selectbox("เลือกอุปกรณ์:", st.session_state.ktd_assets['Transformer_ID'])
    res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == sel_id].iloc[0]

    d1, d2, d3 = st.columns([1, 1, 1.5])
    with d1:
        st.write("**Risk Gauge**")
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, 
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"},
                                             'steps': [{'range': [0, 40], 'color': "#F0F0F0"}, {'range': [40, 75], 'color': "#D0D0D0"}]}))
        fig_g.update_layout(height=280)
        st.plotly_chart(fig_g, use_container_width=True)
    
    with d2:
        # พยากรณ์ช่วงเดือน (ไม่ระบุวัน)
        days_rem = int(max(2, (1 - res['Risk_Score']) * 90))
        target_month = (datetime.now() + timedelta(days=days_rem)).strftime('%B %Y')
        st.markdown(f"<div style='text-align: center; background-color: #FFFFFF; padding: 25px; border-radius: 15px; border: 2px solid #FF8C00;'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Maintenance Window</h3><h1 style='color: #FF8C00;'>{target_month}</h1>", unsafe_allow_html=True)
        if res['Status'] == "🔴 CRITICAL": st.markdown("<p style='color: red;'><b>Priority: Immediate Action</b></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d3:
        st.info("#### AI Diagnostic Insight")
        st.write(f"**Feeder Context:** {res['Feeder']}")
        st.write(f"**🔊 Acoustic Signal:** {res['Acoustic_dB']} dB")
        st.write(f"**🔥 Thermal Surface:** {res['Thermal_Temp']} °C")
        if res['Last_Survey_Img']: st.image(res['Last_Survey_Img'], width=200, caption="Evidence Photo")

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.write("**📡 Smart Meter (Load %)**")
    st.line_chart(np.random.randn(10, 1))
    m2.write("**🔊 Acoustic Spectrum**")
    st.bar_chart(np.random.rand(5), height=150)
    m3.write("**📜 Reliability Context**")
    st.write(f"Trips: {res['Trips_KTD']} | Age: {res['Age']} Yrs")

# --- TAB 3: ACTION PLAN ---
with t3:
    st.markdown("## 📅 PM Monthly Schedule")
    for idx, row in st.session_state.ktd_assets.iterrows():
        if row['Status'] != '🟢 NORMAL':
            st.markdown(f"""
                <div style='background-color: white; padding: 18px; border-radius: 10px; margin-bottom: 12px; border-left: 10px solid #FF8C00; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                    <span style='font-size: 1.2em; font-weight: bold;'>{row['Transformer_ID']}</span> | 
                    Status: {row['Status']} | 
                    <b>Schedule Month: {target_month}</b>
                </div>
            """, unsafe_allow_html=True)
