import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="SPP-AI: ระบบพยากรณ์การบำรุงรักษาหม้อแปลง", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F4F7F9; }
    .stMetric { background-color: #FFFFFF; padding: 25px; border-radius: 15px; border-left: 8px solid #FF8C00; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; height: 3em; }
    h1, h2, h3 { font-family: 'Kanit', sans-serif; }
    .highlight-box { background-color: #FFF3E0; padding: 20px; border-radius: 10px; border: 1px solid #FFCC80; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_spp_ai_model.pkl' บน GitHub")

# --- 3. INITIAL DATABASE ---
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

# --- 4. HEADER (ชื่อระบบภาษาไทย) ---
st.markdown("<h1 style='text-align: center; color: #FF8C00;'>ระบบวิเคราะห์และพยากรณ์การบำรุงรักษาหม้อแปลงอัจฉริยะ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em;'>Smart Plan Predictive Maintenance AI (SPP-AI) | เขตคลองเตย (KTD)</p>", unsafe_allow_html=True)
st.divider()

# --- 5. SIDEBAR (Data Input) ---
with st.sidebar:
    st.header("📥 การจัดการข้อมูล")
    if st.button("📡 Sync ข้อมูล Smart Meter"):
        st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 110, 20)
        st.success("อัปเดตข้อมูล Load เรียบร้อย")
    
    feeder_file = st.file_uploader("อัปโหลดสถิติไฟดับ ฟขต. (.xlsx)", type=["xlsx"])
    if feeder_file:
        df_f = pd.read_excel(feeder_file, skiprows=2)
        counts = df_f['Feeder'].value_counts().to_dict()
        for fid, c in counts.items():
            st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
        st.success("อัปเดตข้อมูล Reliability แล้ว")

    st.divider()
    st.subheader("📸 บันทึกผลสำรวจหน้างาน")
    target = st.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    ac_val = st.number_input("ค่าเสียง Acoustic Camera (dB)", 30.0, 120.0, 45.0)
    th_val = st.number_input("ค่าความร้อน Thermal Scan (°C)", 20.0, 120.0, 55.0)
    survey_img = st.file_uploader("📷 แนบรูปภาพหลักฐาน", type=["jpg", "png", "jpeg"])

    if st.button("🧠 ประมวลผลด้วย AI"):
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
        row = st.session_state.ktd_assets.iloc[idx]
        features = np.array([[th_val, row['Load_Meter'], 230.0, ac_val, 20000, row['Trips_KTD'], row['Age'], 55.0]])
        prob = model.predict_proba(features)[0][1]
        
        st.session_state.ktd_assets.at[idx, 'Status'] = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
        st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_val
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_val
        if survey_img: st.session_state.ktd_assets.at[idx, 'Last_Survey_Img'] = survey_img
        st.success(f"วิเคราะห์ {target} สำเร็จ!")

# --- 6. MAIN TABS ---
t1, t2, t3 = st.tabs(["📊 สรุปภาพรวม (Executive)", "🔍 วิเคราะห์เจาะลึก (Diagnostics)", "📅 แผนงาน (Action Plan)"])

with t1:
    m1, m2, m3, m4 = st.columns(4)
    df = st.session_state.ktd_assets
    m1.metric("จำนวนเครื่องทั้งหมด", "20 ตัว")
    m2.metric("วิกฤต (Critical)", len(df[df['Status'] == "🔴 CRITICAL"]))
    m3.metric("เฝ้าระวัง (Watch)", len(df[df['Status'] == "🟡 WATCH"]))
    m4.metric("พื้นที่ดูแล", "ฟขต. (KTD)")
    
    fig = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Risk_Score", zoom=13, height=450,
                            color_discrete_map={'🔴 CRITICAL': 'red', '🟡 WATCH': 'orange', '🟢 NORMAL': 'green'},
                            mapbox_style="carto-positron")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.header("🔍 วิเคราะห์สาเหตุและความเสี่ยง")
    sel_id = st.selectbox("เลือกหม้อแปลงเพื่อดูข้อมูล:", df['Transformer_ID'])
    res = df[df['Transformer_ID'] == sel_id].iloc[0]

    c_a, c_b = st.columns([1, 1.5])
    with c_a:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, 
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        st.plotly_chart(fig_g, use_container_width=True)
        
        days = int(max(2, (1 - res['Risk_Score']) * 90))
        target_month = (datetime.now() + timedelta(days=days)).strftime('%B %Y')
        st.markdown(f"<div style='text-align: center; border: 2px solid #FF8C00; padding: 20px; border-radius: 10px;'><h3>ช่วงเดือนที่ควรเข้าบำรุงรักษา</h3><h1 style='color: #FF8C00;'>{target_month}</h1></div>", unsafe_allow_html=True)

    with c_b:
        st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
        st.subheader("⚖️ ปัจจัยที่มีผลต่อการตัดสินใจของ AI")
        # แสดงความน่าเชื่อถือด้วยการแจกแจงปัจจัย
        st.write(f"**1. ด้านเสียง (Acoustic):** ตรวจพบ {res['Acoustic_dB']} dB (ให้น้ำหนักสูงสุดในการตรวจพบความผิดปกติภายใน)")
        st.write(f"**2. ด้านความร้อน (Thermal):** ตรวจพบ {res['Thermal_Temp']} °C (จากการ Scan หน้างาน)")
        st.write(f"**3. ด้านประวัติพื้นที่ (Reliability):** พบสถิติไฟดับในฟีดเดอร์ {res['Feeder']} จำนวน {res['Trips_KTD']} ครั้ง")
        st.write(f"**4. ด้านภาระไฟฟ้า (Smart Meter):** ภาระปัจจุบันอยู่ที่ {res['Load_Meter']:.1f}%")
        
        if res['Last_Survey_Img'] is not None:
            st.image(res['Last_Survey_Img'], caption="ภาพหลักฐานจากการสำรวจหน้างาน", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

with t3:
    st.header("📅 ตารางงานบำรุงรักษาเชิงพยากรณ์")
    for _, row in df[df['Status'] != '🟢 NORMAL'].iterrows():
        d = int(max(2, (1 - row['Risk_Score']) * 90))
        m = (datetime.now() + timedelta(days=d)).strftime('%B %Y')
        st.info(f"📍 **{row['Transformer_ID']}** (ฟีดเดอร์: {row['Feeder']}) | สถานะ: {row['Status']} | แผนงานเดือน: **{m}**")
