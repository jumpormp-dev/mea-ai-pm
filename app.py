import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="SPP-AI: KTD Smart City", layout="wide")

# Custom CSS สำหรับโทนสี ส้ม-เทา-ขาว
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .stMetric { background-color: #FFFFFF; padding: 20px; border-radius: 12px; border-left: 6px solid #FF8C00; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #343A40; font-family: 'Kanit', sans-serif; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; width: 100%; border: none; }
    .stExpander { background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E9ECEF; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_spp_ai_model.pkl'")

# --- 2. INITIAL DATABASE ---
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
        'Thermal_Temp': [55.0]*20, # ค่าจากกล้องเทอร์โม
        'Age': np.random.randint(5, 30, 20)
    })

# --- 3. SIDEBAR: DATA INPUT ---
with st.sidebar:
    st.header("📥 Data Management")
    
    # 3.1 Sync Smart Meter (Load Only)
    if st.button("📡 Sync Smart Meter (Load Only)"):
        st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 110, 20)
        st.success("Load Data Synced")

    # 3.2 Upload Reliability (Excel)
    feeder_file = st.file_uploader("Upload Reliability Data", type=["xlsx"])
    if feeder_file:
        df_f = pd.read_excel(feeder_file, skiprows=2)
        counts = df_f['Feeder'].value_counts().to_dict()
        for fid, c in counts.items():
            st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
        st.success("Reliability Updated")

    st.divider()
    
    # 3.3 Separate Field Survey Inputs
    st.subheader("📸 Field Survey Data")
    target = st.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        ac_db = st.number_input("Acoustic (dB)", 30.0, 120.0, 45.0)
    with col_in2:
        th_temp = st.number_input("Thermal (°C)", 20.0, 120.0, 55.0)
    
    survey_img = st.file_uploader("แนบรูปภาพหน้างาน (Acoustic/Thermo)", type=["jpg", "png", "jpeg"])
    if survey_img:
        st.image(survey_img, caption="Evidence Preview", use_column_width=True)

    if st.button("🧠 Run AI Analysis"):
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
        row = st.session_state.ktd_assets.iloc[idx]
        # ใช้ข้อมูล Thermal_Temp แทน Temp_Meter ในโมเดล
        features = np.array([[th_temp, row['Load_Meter'], 230.0, ac_db, 20000, row['Trips_KTD'], row['Age'], 55.0]])
        prob = model.predict_proba(features)[0][1]
        
        st.session_state.ktd_assets.at[idx, 'Status'] = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
        st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_db
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_temp
        st.success(f"Analysis Complete: {target}")

# --- 4. MAIN UI ---
t1, t2, t3, t4 = st.tabs(["📊 Executive Dashboard", "🔍 Diagnostics", "📅 Action Plan", "⚙️ Settings"])

# TAB 1: EXECUTIVE DASHBOARD
with t1:
    st.markdown("## 🏙️ SPP-AI: Executive Command Center")
    m1, m2, m3, m4 = st.columns(4)
    crit_count = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🔴 CRITICAL"])
    watch_count = len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🟡 WATCH"])
    
    m1.metric("Total Assets", "20 Units")
    m2.metric("Critical", crit_count)
    m3.metric("Watch", watch_count)
    m4.metric("KTD Area Status", "Monitoring")

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

# TAB 2: DIAGNOSTICS (ช่วงเดือน + แยกข้อมูล 2 ส่วน)
with t2:
    st.markdown("## 🔍 Transformer Health Insights")
    sel_id = st.selectbox("เลือกอุปกรณ์เพื่อดูบทวิเคราะห์:", st.session_state.ktd_assets['Transformer_ID'])
    res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == sel_id].iloc[0]

    d1, d2, d3 = st.columns([1, 1, 1.5])
    with d1:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, 
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"},
                                             'steps': [{'range': [0, 40], 'color': "#E5E7E9"}, {'range': [40, 75], 'color': "#BDC3C7"}]}))
        fig_g.update_layout(height=280)
        st.plotly_chart(fig_g, use_container_width=True)
    
    with d2:
        days_rem = int(max(2, (1 - res['Risk_Score']) * 90))
        target_month = (datetime.now() + timedelta(days=days_rem)).strftime('%B %Y')
        st.markdown(f"<div style='text-align: center; background-color: #F8F9FA; padding: 25px; border-radius: 15px; border: 1px solid #FF8C00;'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Recommended PM Month</h3><h1 style='color: #FF8C00;'>{target_month}</h1>", unsafe_allow_html=True)
        if res['Status'] == "🔴 CRITICAL": st.markdown("<p style='color: red;'><b>Urgent Priority</b></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d3:
        st.info("#### AI Diagnostic Logic")
        st.write(f"**Target:** {sel_id} | **Feeder:** {res['Feeder']}")
        st.write(f"**Acoustic Signal:** {res['Acoustic_dB']} dB (Detected by Camera)")
        st.write(f"**Thermal Surface:** {res['Thermal_Temp']} °C (Detected by Thermal Scan)")

    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        st.write("**📡 Smart Meter (Load %)**")
        st.line_chart(np.random.randn(10, 1))
    with m2:
        st.write("**🔊 Acoustic Peak Analysis**")
        st.bar_chart(np.random.rand(5))
    with m3:
        st.write("**📜 Contextual History**")
        st.write(f"Age: {res['Age']} Years | Trips: {res['Trips_KTD']} Counts")

# TAB 3: ACTION PLAN
with t3:
    st.markdown("## 📅 PM Monthly Schedule")
    for idx, row in st.session_state.ktd_assets.iterrows():
        if row['Status'] != '🟢 NORMAL':
            st.markdown(f"""
                <div style='background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 8px solid #FF8C00; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);'>
                    <b>{row['Transformer_ID']}</b> | Status: {row['Status']} | <b>Target: {target_month}</b>
                </div>
            """, unsafe_allow_html=True)
