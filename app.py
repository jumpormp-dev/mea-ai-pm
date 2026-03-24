import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE (ส้ม-เทา-ขาว) ---
st.set_page_config(page_title="SPP-AI: ระบบวิเคราะห์และพยากรณ์การบำรุงรักษา", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 8px solid #FF8C00; }
    .metric-crit { border-top: 8px solid #FF4B4B; }
    .metric-watch { border-top: 8px solid #FF8C00; }
    .metric-normal { border-top: 8px solid #28A745; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; }
    h1, h2, h3 { font-family: 'Kanit', sans-serif; color: #444444; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL (.pkl) ---
@st.cache_resource
def load_spp_model():
    try:
        return joblib.load('mea_spp_ai_model.pkl')
    except:
        return None

model = load_spp_model()

# --- 3. INITIAL DATABASE (KTD & ฟขต) ---
if 'ktd_assets' not in st.session_state:
    # รายชื่อ Feeder จากไฟล์ ฟขต ที่คุณใช้งาน
    feeders = ['EM-418', 'SAM-13', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'RPR-423']
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': [feeders[i % len(feeders)] for i in range(20)],
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        'Load_Percent': [0.0]*20,
        'Voltage_V': [220.0]*20,
        'Trips_Count': [0]*20,
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Acoustic_dB': [45.0]*20,
        'Thermal_Temp': [50.0]*20,
        'Age_Years': np.random.randint(5, 30, 20),
        'Last_Img': [None]*20
    })

# --- 4. HEADER ---
st.markdown("<h1 style='text-align: center;'>ระบบวิเคราะห์และพยากรณ์การบำรุงรักษา</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666;'>Smart Plan Predictive Maintenance AI (KTD Area)</p>", unsafe_allow_html=True)
st.divider()

if model is None:
    st.error("⚠️ ไม่พบไฟล์ 'mea_spp_ai_model.pkl' กรุณาตรวจสอบว่าวางไฟล์ไว้ใน Folder เดียวกับ app.py")

# --- 5. SIDEBAR: DATA INPUT ---
with st.sidebar:
    st.header("📥 การจัดการข้อมูล")
    
    # ส่วนเชื่อมต่อ Smart Meter (จำลองดึงจากลิ้งค์ 172.16.111.184)
    if st.button("📡 Sync ข้อมูล Smart Meter (กฟน.)"):
        st.session_state.ktd_assets['Load_Percent'] = np.random.uniform(40, 115, 20)
        st.session_state.ktd_assets['Voltage_V'] = np.random.uniform(215, 235, 20)
        st.success("โหลดข้อมูลจาก Smart Meter สำเร็จ")

    # อัปโหลดไฟล์ ฟขต เพื่ออัปเดตสถิติ Trip
    uploaded_xlsx = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    if uploaded_xlsx:
        df_xlsx = pd.read_excel(uploaded_xlsx, skiprows=2)
        trip_map = df_xlsx['Feeder'].value_counts().to_dict()
        for i, row in st.session_state.ktd_assets.iterrows():
            st.session_state.ktd_assets.at[i, 'Trips_Count'] = trip_map.get(row['Feeder'], 0)
        st.success("อัปเดตสถิติไฟดับจากไฟล์ ฟขต เรียบร้อย")

    st.divider()
    st.subheader("📸 บันทึกผลสำรวจหน้างาน")
    target_id = st.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    ac_val = st.number_input("ค่าเสียง Acoustic (dB)", 30.0, 110.0, 45.0)
    th_val = st.number_input("ความร้อน Thermal (°C)", 20.0, 120.0, 50.0)
    img_file = st.file_uploader("แนบรูปหลักฐาน", type=["jpg", "png", "jpeg"])

    if st.button("🧠 รันการประมวลผล AI"):
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].index[0]
        row = st.session_state.ktd_assets.iloc[idx]
        
        # เตรียม 8 Features ให้ตรงกับโมเดลที่เทรนมา
        # ['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']
        features = np.array([[
            th_val, row['Load_Percent'], row['Voltage_V'], ac_val, 
            25000, row['Trips_Count'], row['Age_Years'], 65.0
        ]])
        
        if model:
            res_idx = model.predict(features)[0]
            prob = model.predict_proba(features)[0][res_idx]
            
            status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
            st.session_state.ktd_assets.at[idx, 'Status'] = status_map[res_idx]
            st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
            st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_val
            st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_val
            if img_file: st.session_state.ktd_assets.at[idx, 'Last_Img'] = img_file
            st.success(f"วิเคราะห์ {target_id} สำเร็จ")
            st.rerun()

# --- 6. MAIN CONTENT (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 Executive", "🔍 Diagnostics", "📅 Action Plan"])

with tab1:
    df = st.session_state.ktd_assets
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><h4>รวม</h4><h1>20</h1></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card metric-crit'><h4>วิกฤต</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card metric-watch'><h4>เฝ้าระวัง</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card metric-normal'><h4>พื้นที่</h4><h1>KTD</h1></div>", unsafe_allow_html=True)
    
    fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Risk_Score", zoom=13, height=500,
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    sel_id = st.selectbox("เลือกอุปกรณ์เพื่อดูสาเหตุความเสี่ยง:", df['Transformer_ID'])
    res = df[df['Transformer_ID'] == sel_id].iloc[0]
    
    col_l, col_r = st.columns([1, 1.5])
    with col_l:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100,
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        st.plotly_chart(fig_g, use_container_width=True)
        
        days = int(max(2, (1 - res['Risk_Score']) * 90))
        month_target = (datetime.now() + timedelta(days=days)).strftime('%B %Y')
        st.markdown(f"<div style='text-align: center; border: 2px solid #FF8C00; padding: 20px; border-radius: 12px; background-color: white;'>"
                    f"<h3>แผนบำรุงรักษา: {month_target}</h3></div>", unsafe_allow_html=True)

    with col_r:
        st.info("### AI Explainability (ปัจจัยหลัก)")
        st.write(f"- **เสียง (Acoustic):** {res['Acoustic_dB']} dB")
        st.write(f"- **ความร้อน (Thermal):** {res['Thermal_Temp']} °C")
        st.write(f"- **ภาระไฟฟ้า (Smart Meter):** {res['Load_Percent']:.1f}%")
        st.write(f"- **สถิติไฟดับ (ฟขต. Trip):** {res['Trips_Count']} ครั้ง")
        if res['Last_Img']: st.image(res['Last_Img'], width=300, caption="ภาพถ่ายหลักฐานหน้างาน")

with tab3:
    st.header("📅 ตารางแผนงานบำรุงรักษา (จัดลำดับตามความเสี่ยง)")
    for _, row in df[df['Status'] != '🟢 NORMAL'].sort_values('Risk_Score', ascending=False).iterrows():
        st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 10px; border-left: 10px solid #FF8C00; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);'>
            <b>{row['Transformer_ID']}</b> (ฟีดเดอร์: {row['Feeder']}) | สถานะ: {row['Status']} | <b>แผนงาน: {month_target}</b>
            </div>
        """, unsafe_allow_html=True)
