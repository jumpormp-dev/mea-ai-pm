import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & MODEL LOAD ---
st.set_page_config(page_title="SPP-AI: KTD Smart City", layout="wide")

# ปรับโทนสีหลักของแอปผ่าน Custom CSS
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .stMetric { background-color: #FFFFFF; padding: 15px; border-radius: 10px; border-left: 5px solid #FF8C00; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #4A4A4A; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 5px; border: none; }
    .stTab { color: #666666; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_spp_ai_model.pkl'")

# --- 2. INITIAL DATABASE (KTD 20 UNITS) ---
if 'ktd_assets' not in st.session_state:
    ktd_feeders = ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'SAM-13', 'RPR-423'] * 3
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': ktd_feeders[:20],
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        'Temp_Meter': [65.0]*20,
        'Load_Meter': [0.0]*20,
        'Trips_KTD': [0]*20,
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Age': np.random.randint(5, 30, 20),
        'Acoustic_dB': [45.0]*20
    })

# --- 3. SIDEBAR: DATA CHANNELS ---
with st.sidebar:
    st.image("https://www.mea.or.th/assets/images/logo.png", width=100) # โลโก้จำลอง
    st.header("📥 Data Source")
    
    if st.button("📡 Sync Smart Meter (KTD)"):
        st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 110, 20)
        st.success("Sync Load Data Success")

    feeder_file = st.file_uploader("Upload Reliability Data", type=["xlsx"])
    if feeder_file:
        df_f = pd.read_excel(feeder_file, skiprows=2)
        counts = df_f['Feeder'].value_counts().to_dict()
        for fid, c in counts.items():
            st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
        st.success("Reliability Updated")

    st.divider()
    st.subheader("📸 Acoustic Survey")
    target = st.selectbox("หม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    ac_db_in = st.number_input("dB Level", 30.0, 120.0, 45.0)
    if st.button("🧠 Run AI Analysis"):
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
        row = st.session_state.ktd_assets.iloc[idx]
        features = np.array([[row['Temp_Meter'], row['Load_Meter'], 230.0, ac_db_in, 20000, row['Trips_KTD'], row['Age'], 55.0]])
        prob = model.predict_proba(features)[0][1]
        
        new_stat = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
        st.session_state.ktd_assets.at[idx, 'Status'] = new_stat
        st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_db_in

# --- 4. MAIN NAVIGATION (TABS) ---
t1, t2, t3, t4 = st.tabs(["📊 Executive Dashboard", "🔍 Diagnostics", "📅 Action Plan", "⚙️ Settings"])

# --- TAB 1: EXECUTIVE DASHBOARD ---
with t1:
    st.markdown("## 🏙️ SPP-AI: KTD Command Center")
    m1, m2, m3, m4 = st.columns(4)
    crit_count = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🔴 CRITICAL"])
    watch_count = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🟡 WATCH"])
    
    m1.metric("Total Assets", "20 Units")
    m2.metric("Critical", crit_count)
    m3.metric("Watch", watch_count)
    m4.metric("Status", "Online", delta="KTD LAN")

    c_map, c_list = st.columns([2, 1])
    with c_map:
        fig_map = px.scatter_mapbox(st.session_state.ktd_assets, lat="Lat", lon="Lon", color="Status", 
                                    size="Risk_Score", zoom=13, height=450,
                                    color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                    mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
    with c_list:
        st.write("### 🚨 Urgent Attention")
        st.dataframe(st.session_state.ktd_assets.sort_values('Risk_Score', ascending=False)[['Transformer_ID', 'Status']].head(8), hide_index=True)

# --- TAB 2: TRANSFORMER DETAIL (ช่วงเดือนบำรุงรักษา) ---
with t2:
    st.markdown("## 🔍 Transformer Diagnostic Insight")
    sel_id = st.selectbox("เลือกอุปกรณ์:", st.session_state.ktd_assets['Transformer_ID'])
    res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == sel_id].iloc[0]

    d1, d2, d3 = st.columns([1, 1, 1.5])
    with d1:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, 
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"},
                                             'steps': [{'range': [0, 40], 'color': "#E5E7E9"}, {'range': [40, 75], 'color': "#BDC3C7"}]}))
        fig_g.update_layout(height=280, font={'color': "#4A4A4A"})
        st.plotly_chart(fig_g, use_container_width=True)
    
    with d2:
        # คำนวณช่วงเดือน (ไม่แสดงวัน)
        days_rem = int(max(2, (1 - res['Risk_Score']) * 90))
        target_month = (datetime.now() + timedelta(days=days_rem)).strftime('%B %Y')
        
        st.markdown("<div style='text-align: center; background-color: #F2F4F4; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Recommended PM</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color: #FF8C00; font-size: 45px;'>{target_month}</h1>", unsafe_allow_html=True)
        if res['Status'] == "🔴 CRITICAL":
            st.markdown("<p style='color: red;'><b>Urgent: Action required within this week</b></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d3:
        st.info("#### AI Diagnostic Summary")
        st.write(f"วิเคราะห์ความเสี่ยงเชิงพยากรณ์สำหรับ **{sel_id}** โดยอิงจากข้อมูล Smart Meter และประวัติ Reliability ฟีดเดอร์ **{res['Feeder']}**")
        st.write("**Analysis Result:** ตรวจพบความผิดปกติสะสมในระดับที่ส่งผลต่อดัชนีความเชื่อถือได้")

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.write("**📡 Sensor Connectivity**")
    m1.write("Smart Meter: 🟢 Online")
    m1.error("Temp Sensor: 🔴 Offline")
    m2.write("**🔊 Acoustic Peak Data**")
    st.bar_chart(np.random.rand(5), height=150)
    m3.write("**📜 Asset Info**")
    st.write(f"Feeder: {res['Feeder']} | Age: {res['Age']} Yrs")

# --- TAB 3: ACTION PLAN ---
with t3:
    st.markdown("## 📅 Maintenance Schedule")
    for idx, row in st.session_state.ktd_assets.iterrows():
        if row['Status'] != '🟢 NORMAL':
            st.markdown(f"""
                <div style='background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 10px solid #FF8C00;'>
                    <b>{row['Transformer_ID']}</b> | Status: {row['Status']} | <b>Target Month: {target_month}</b>
                </div>
            """, unsafe_allow_html=True)

# --- TAB 4: SETTINGS ---
with t4:
    st.markdown("## ⚙️ System Configuration")
    st.write("API Gateway: `http://172.16.111.184:8501`")
    st.slider("Reliability Penalty Weight", 0, 30, 15)
