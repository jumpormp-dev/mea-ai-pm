import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & CUSTOM STYLE ---
st.set_page_config(page_title="SPP-AI: ระบบวิเคราะห์และพยากรณ์การบำรุงรักษา", layout="wide")

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
        'Thermal_Temp': [55.0]*20,
        'Age': np.random.randint(5, 30, 20),
        'Last_Survey_Img': [None]*20
    })

# --- 3. CUSTOM CSS (ปรับสี Metric ตามคะแนนความเสี่ยง) ---
df = st.session_state.ktd_assets
crit_count = len(df[df['Status'] == "🔴 CRITICAL"])
watch_count = len(df[df['Status'] == "🟡 WATCH"])

st.markdown(f"""
    <style>
    .main {{ background-color: #F8F9FA; }}
    /* Metric Cards Customization */
    .metric-card {{ background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 8px solid #FF8C00; }}
    .metric-crit {{ border-top: 8px solid #FF4B4B; }} /* สีแดง */
    .metric-watch {{ border-top: 8px solid #FF8C00; }} /* สีส้ม */
    .metric-normal {{ border-top: 8px solid #28A745; }} /* สีเขียว */
    
    .stButton>button {{ background-color: #FF8C00; color: white; border-radius: 8px; border: none; font-weight: bold; width: 100%; }}
    h1, h2 {{ font-family: 'Kanit', sans-serif; }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.markdown("<h1 style='text-align: center; color: #444444;'>ระบบวิเคราะห์และพยากรณ์การบำรุงรักษา</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666;'>Smart Plan Predictive Maintenance AI</p>", unsafe_allow_html=True)
st.divider()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("📥 การจัดการข้อมูล")
    if st.button("📡 Sync Smart Meter"):
        st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 110, 20)
        st.success("Load Synced")
    
    st.divider()
    st.subheader("📸 ผลสำรวจหน้างาน")
    target = st.selectbox("เลือกหม้อแปลง:", df['Transformer_ID'])
    ac_val = st.number_input("เสียง Acoustic (dB)", 30.0, 120.0, 45.0)
    th_val = st.number_input("ความร้อน Thermal (°C)", 20.0, 120.0, 55.0)
    survey_img = st.file_uploader("แนบรูปภาพหน้างาน", type=["jpg", "png", "jpeg"])

    if st.button("🧠 ประมวลผล AI"):
        idx = df[df['Transformer_ID'] == target].index[0]
        features = np.array([[th_val, df.at[idx, 'Load_Meter'], 230.0, ac_val, 20000, df.at[idx, 'Trips_KTD'], df.at[idx, 'Age'], 55.0]])
        prob = model.predict_proba(features)[0][1]
        
        st.session_state.ktd_assets.at[idx, 'Status'] = "🔴 CRITICAL" if (prob > 0.75 or df.at[idx, 'Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
        st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_val
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_val
        if survey_img: st.session_state.ktd_assets.at[idx, 'Last_Survey_Img'] = survey_img
        st.rerun()

# --- 6. MAIN CONTENT ---
t1, t2, t3 = st.tabs(["📊 สรุปภาพรวม (Executive)", "🔍 วิเคราะห์เจาะลึก (Diagnostics)", "📅 แผนงาน (Action Plan)"])

with t1:
    # Custom Metric Cards with Dynamic Coloring
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><h4>ทั้งหมด</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card metric-crit'><h4>วิกฤต (Crit)</h4><h1>{crit_count}</h1></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card metric-watch'><h4>เฝ้าระวัง (Watch)</h4><h1>{watch_count}</h1></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card metric-normal'><h4>พื้นที่</h4><h1>KTD</h1></div>", unsafe_allow_html=True)

    st.write("")
    fig = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Risk_Score", zoom=13, height=450,
                            color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                            mapbox_style="carto-positron")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.header("🔍 รายละเอียดความเสี่ยงรายตัว")
    sel_id = st.selectbox("เลือกหม้อแปลง:", df['Transformer_ID'])
    res = df[df['Transformer_ID'] == sel_id].iloc[0]

    col_gauge, col_info = st.columns([1, 1.5])
    with col_gauge:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, 
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        st.plotly_chart(fig_g, use_container_width=True)
    with col_info:
        # พยากรณ์ช่วงเดือน
        days = int(max(2, (1 - res['Risk_Score']) * 90))
        target_m = (datetime.now() + timedelta(days=days)).strftime('%B %Y')
        st.markdown(f"<div style='border: 2px solid #FF8C00; padding: 20px; border-radius: 10px; background-color: white;'><h3>แผนบำรุงรักษา: {target_m}</h3>"
                    f"<p><b>ผลวิเคราะห์:</b> AI ตรวจพบความเสี่ยงจากเสียง {res['Acoustic_dB']}dB และความร้อน {res['Thermal_Temp']}°C</p></div>", unsafe_allow_html=True)
        if res['Last_Survey_Img']: st.image(res['Last_Survey_Img'], width=300)

with t3:
    st.header("📅 ตารางแผนงานบำรุงรักษา")
    for _, row in df[df['Status'] != '🟢 NORMAL'].iterrows():
        st.info(f"📍 **{row['Transformer_ID']}** | ฟีดเดอร์: {row['Feeder']} | สถานะ: {row['Status']}")
