import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# 1. โหลดโมเดล AI (8 ตัวแปร)
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl')

model = load_my_model()

st.set_page_config(page_title="MEA Smart PM Planner", layout="wide")
st.title("⚡ MEA Smart Maintenance Planner (Integrated AI)")
st.write("ระบบวางแผน PM อัจฉริยะ: รวมฐานข้อมูลทรัพย์สิน สถิติการขัดข้อง และผลสำรวจ Acoustic")

# --- ส่วนที่ 1: ฐานข้อมูลทรัพย์สิน (Asset Database) ---
# ในสถานการณ์จริง ส่วนนี้จะดึงจาก GIS หรือ SAP ของ MEA
if 'asset_db' not in st.session_state:
    st.session_state.asset_db = pd.DataFrame({
        'Transformer_ID': ['TR-BK-001', 'TR-BK-002', 'TR-BK-003', 'TR-BK-004', 'TR-BK-005'],
        'Area': ['Bangkok', 'Bangkok', 'Nonthaburi', 'Samut Prakan', 'Bangkok'],
        'Age (Years)': [5, 18, 12, 22, 8],
        'Last_PM': ['2025-01-10', '2024-05-20', '2024-11-15', '2023-08-10', '2025-02-01']
    })

# --- ส่วนที่ 2: การนำเข้าข้อมูลสถิติ และ ผลสำรวจ (Data Input) ---
st.sidebar.header("📥 Data Integration")
selected_id = st.sidebar.selectbox("เลือกหม้อแปลงที่ต้องการวิเคราะห์:", st.session_state.asset_db['Transformer_ID'])

# ดึงข้อมูลพื้นฐานจาก DB มาโชว์
asset_info = st.session_state.asset_db[st.session_state.asset_db['Transformer_ID'] == selected_id].iloc[0]

st.sidebar.divider()
st.sidebar.subheader("🌐 สถิติจากหน่วยงาน (Web Stats)")
stat_trips = st.sidebar.number_input("จำนวนครั้งที่สวิตช์ตก/ไฟดับ (ปีล่าสุด)", 0, 20, 2)
humidity = st.sidebar.slider("ความชื้นสะสมในพื้นที่ (%)", 30, 90, 55)

st.sidebar.subheader("🔍 ข้อมูลสำรวจหน้างาน (Survey)")
temp = st.sidebar.number_input("อุณหภูมิที่วัดได้ (°C)", 30.0, 110.0, 55.0)
load_pct = st.sidebar.slider("โหลดขณะตรวจวัด (%)", 0, 120, 70)
oil_level = st.sidebar.slider("ระดับน้ำมัน (%)", 0, 100, 95)
vib = st.sidebar.number_input("ความสั่นสะเทือน (mm/s)", 0.0, 10.0, 1.2)

st.sidebar.subheader("🔊 Acoustic Camera Results")
ac_db = st.sidebar.number_input("ค่าความดังเสียง (dB)", 30.0, 110.0, 45.0)
ac_hz = st.sidebar.number_input("ความถี่ Peak (Hz)", 1000, 100000, 20000)

# --- ส่วนที่ 3: การวิเคราะห์และแสดงผล (AI Analysis) ---
if st.sidebar.button("🤖 ประมวลผลแผน PM"):
    # เตรียม Features 8 ตัวแปรตามลำดับโมเดล
    # [Temp, Load, Oil, Vib, Age, Humidity, Acoustic_dB, Peak_Freq]
    features = np.array([[temp, load_pct, oil_level, vib, asset_info['Age (Years)'], humidity, ac_db, ac_hz]])
    
    prob = model.predict_proba(features)[0][1]
    risk_level = "🔴 CRITICAL" if prob > 0.8 else "🟡 WARNING" if prob > 0.4 else "🟢 NORMAL"

    # แสดงผล Dashboard
    st.subheader(f"📊 รายงานการวิเคราะห์: {selected_id} ({asset_info['Area']})")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("คะแนนความเสี่ยง (AI)", f"{prob*100:.1f}%")
    col2.metric("สถานะแนะนำ", risk_level)
    col3.metric("อายุการใช้งาน", f"{asset_info['Age (Years)']} ปี")

    # กราฟ Radar Chart แสดงจุดที่ต้องเฝ้าระวัง
    st.write("### วิเคราะห์ปัจจัยรายด้าน (Multi-Factor Analysis)")
    radar_data = pd.DataFrame({
        'Factor': ['ความร้อน', 'ภาระไฟฟ้า', 'ความเก่า', 'สถิติไฟดับ', 'เสียง Acoustic'],
        'Score': [temp/100, load_pct/100, asset_info['Age (Years)']/25, stat_trips/10, ac_db/100]
    })
    fig_radar = px.line_polar(radar_data, r='Score', theta='Factor', line_close=True, range_r=[0,1])
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)

    # สรุปแผน PM (Decision Support)
    st.divider()
    st.subheader("📝 แผนงาน PM ที่แนะนำ (Action Plan)")
    
    if risk_level == "🔴 CRITICAL":
        st.error(f"**เหตุผล:** สถิติการขัดข้องสูง ({stat_trips} ครั้ง) ร่วมกับค่า Acoustic ({ac_db} dB) บ่งบอกถึงความเสียหายภายใน")
        st.markdown("- **Action:** เลื่อนกำหนด PM มาเป็นด่วนที่สุด (ภายใน 48 ชม.)")
        st.markdown("- **Resource:** ต้องใช้รถ Mobile Unit และทีมช่างเทคนิคพิเศษ")
    elif risk_level == "🟡 WARNING":
        st.warning(f"**เหตุผล:** อุปกรณ์เริ่มมีอายุ ({asset_info['Age (Years)']} ปี) และเริ่มตรวจพบเสียงผิดปกติ")
        st.markdown("- **Action:** บรรจุเข้าแผน PM รายไตรมาสถัดไป")
        st.markdown("- **Resource:** ตรวจสอบซ้ำด้วยการเก็บตัวอย่างน้ำมัน (DGA)")
    else:
        st.success("**เหตุผล:** ปัจจัยทุกอย่างยังอยู่ในเกณฑ์มาตรฐาน")
        st.markdown("- **Action:** ดำเนินการตามรอบ PM ปกติ (Last PM: " + asset_info['Last_PM'] + ")")

else:
    # หน้าแรกตอนยังไม่ได้กดวิเคราะห์
    st.info("👈 กรุณาระบุรหัสหม้อแปลงและกรอกข้อมูลสถิติ/ผลสำรวจหน้างานจากแถบเมนูด้านซ้าย")
    st.write("### รายการหม้อแปลงในฐานข้อมูลเขตพื้นที่")
    st.table(st.session_state.asset_db)
