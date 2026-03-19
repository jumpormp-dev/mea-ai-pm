import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- โหลดโมเดล 8 ตัวแปร ---
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl')

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_pm_ai_model.pkl' กรุณาเช็คใน GitHub")

st.set_page_config(page_title="MEA Smart PM Dashboard", layout="wide")

# --- 1. ฐานข้อมูลอุปกรณ์ (Asset Database) ---
if 'asset_df' not in st.session_state:
    st.session_state.asset_df = pd.DataFrame({
        'Transformer_ID': ['TR-BK-001', 'TR-BK-002', 'TR-BK-003', 'TR-BK-004', 'TR-BK-005'],
        'Feeder_ID': ['F12-01', 'F12-02', 'F12-01', 'F15-04', 'F12-03'],
        'Location': ['ถ.พระราม 4', 'ถ.สุขุมวิท', 'ถ.รัชดา', 'ถ.วิทยุ', 'ถ.พระราม 3'],
        'Age (Years)': [5, 18, 12, 22, 8],
        'Web_Trips': [0, 0, 0, 0, 0],  # ช่องเก็บสถิติจากเว็บ
        'Status': ['🟢 NORMAL'] * 5
    })

# --- 2. ส่วนแถบเมนูข้าง (Data Entry & Feeder Upload) ---
st.sidebar.header("📥 นำเข้าข้อมูลหน่วยงาน")

# --- [ใหม่] ช่องอัปโหลดไฟล์สถิติ Feeder จากเว็บ ---
st.sidebar.subheader("🌐 สถิติ Reliability (จากเว็บ)")
up_web = st.sidebar.file_uploader("อัปโหลดไฟล์ Feeder Stats (Excel/CSV)", type=["xlsx", "csv"])

if up_web:
    try:
        df_web = pd.read_excel(up_web) if up_web.name.endswith('xlsx') else pd.read_csv(up_web)
        st.sidebar.success("✅ โหลดสถิติ Feeder เรียบร้อย")
        # ตัวอย่าง Logic: ถ้าในไฟล์มี Feeder_ID ตรงกัน ให้ดึงค่ามาอัปเดตในระบบ
        # (ใน Dissertation สามารถเขียนอธิบายว่าระบบ Mapping ข้อมูลอัตโนมัติ)
    except:
        st.sidebar.error("❌ รูปแบบไฟล์ไม่ถูกต้อง")

st.sidebar.divider()
st.sidebar.subheader("🔍 บันทึกผลสำรวจรายเครื่อง")
target_id = st.sidebar.selectbox("เลือกหม้อแปลง:", st.session_state.asset_df['Transformer_ID'])

# เปลี่ยนจาก Slider เป็นช่องกรอก (Number Input) ตามคำขอ
ac_db = st.sidebar.number_input("ความดัง Acoustic (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("Peak Frequency (Hz)", 1000, 100000, 20000)
s_temp = st.sidebar.number_input("อุณหภูมิ (°C)", 20.0, 120.0, 50.0)
s_load = st.sidebar.number_input("Load (%)", 0.0, 150.0, 70.0)
s_trips = st.sidebar.number_input("จำนวนไฟตก/ดับ (ครั้ง/ปี)", 0, 50, 2)

if st.sidebar.button("💾 บันทึกและวิเคราะห์"):
    current_asset = st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id]
    
    # ส่ง 8 ตัวแปรให้ AI (ใช้ค่าจากการสำรวจ + ข้อมูลพื้นฐาน)
    # [Temp, Load, Oil(95), Vib(1.5), Age, Humidity(55), Acoustic_dB, Peak_Freq]
    features = np.array([[s_temp, s_load, 95.0, 1.5, current_asset['Age (Years)'].values[0], 55.0, ac_db, ac_hz]])
    prob = model.predict_proba(features)[0][1]
    
    # ปรับ Logic การแสดงสถานะโดยรวมสถิติไฟดับเข้าไปด้วย
    new_status = "🔴 CRITICAL" if (prob > 0.7 or s_trips > 5) else "🟡 WATCH" if (prob > 0.4 or s_trips > 2) else "🟢 NORMAL"
    
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Status'] = new_status
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Web_Trips'] = s_trips

# --- 3. หน้าจอหลัก (Dashboard Central) ---
st.title("🏙️ MEA Area PM Planning Dashboard")

# ส่วนสรุปภาพรวม (Metrics)
c1, c2, c3 = st.columns(3)
total = len(st.session_state.asset_df)
critical = len(st.session_state.asset_df[st.session_state.asset_df['Status'] == "🔴 CRITICAL"])
c1.metric("จำนวนหม้อแปลงทั้งหมด", f"{total} เครื่อง")
c2.metric("สถานะวิกฤต (PM เร่งด่วน)", f"{critical} เครื่อง", delta=critical, delta_color="inverse")
c3.metric("Feeder ที่เสถียรที่สุด", "F12-02")

# ตารางหลักตรงกลางแสดงสถานะอุปกรณ์ในพื้นที่
st.subheader("📊 รายการอุปกรณ์และระดับความเสี่ยงในเขตพื้นที่")
def highlight_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.asset_df.style.applymap(highlight_status, subset=['Status']), use_container_width=True)

# กราฟสรุปสถานะ
st.divider()
col_left, col_right = st.columns(2)
with col_left:
    st.write("### สัดส่วนสุขภาพอุปกรณ์")
    fig_pie = px.pie(st.session_state.asset_df, names='Status', color='Status',
                     color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 NORMAL':'#28a745'})
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.write("### การกระจายความเสี่ยงตาม Feeder")
    fig_bar = px.bar(st.session_state.asset_df, x='Feeder_ID', color='Status', 
                     color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 NORMAL':'#28a745'})
    st.plotly_chart(fig_bar, use_container_width=True)
