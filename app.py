import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- โหลดโมเดล AI (8 ตัวแปร) ---
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl')

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_pm_ai_model.pkl' กรุณาตรวจสอบบน GitHub")

st.set_page_config(page_title="MEA Smart PM - ฟขจ. Dashboard", layout="wide")

# --- 1. ฐานข้อมูลหม้อแปลงจำลองในเขตนวลจันทร์ (ฟขจ.) ---
if 'asset_df' not in st.session_state:
    st.session_state.asset_df = pd.DataFrame({
        'Transformer_ID': ['TR-NJ-001', 'TR-NJ-002', 'TR-NJ-003', 'TR-NJ-004', 'TR-NJ-005'],
        'Feeder': ['GK-424', 'GK-422', 'LK-422', 'KJ-433', 'GK-424'],
        'Location': ['ถ.นวลจันทร์', 'ถ.รามอินทรา', 'ซ.สุคนธสวัสดิ์', 'ถ.ประดิษฐ์มนูธรรม', 'ซ.นวลจันทร์ 36'],
        'Age (Years)': [12, 22, 5, 18, 10],
        'Trip_History': [0, 0, 0, 0, 0], 
        'Status': ['🟢 NORMAL'] * 5
    })

# --- 2. แถบเมนูข้าง (Data Control Center) ---
st.sidebar.header("📥 ศูนย์นำเข้าข้อมูล ฟขจ.")

# 2.1 อัปโหลดไฟล์สถิติฟีดเดอร์ (รองรับ Excel .xlsx)
st.sidebar.subheader("🌐 สถิติ Reliability (จากเว็บ)")
feeder_file = st.sidebar.file_uploader("อัปโหลดไฟล์ Feeder Indices", type=["xlsx", "xls"])

if feeder_file:
    try:
        # อ่านไฟล์ Excel และข้ามหัว 2 บรรทัดแรกตามโครงสร้างไฟล์ MEA
        df_feeder = pd.read_excel(feeder_file, skiprows=2)
        
        # ตรวจสอบว่ามีคอลัมน์ชื่อ 'Feeder' หรือไม่
        if 'Feeder' in df_feeder.columns:
            trip_counts = df_feeder['Feeder'].value_counts().to_dict()
            for fid, count in trip_counts.items():
                st.session_state.asset_df.loc[st.session_state.asset_df['Feeder'] == fid, 'Trip_History'] = count
            st.sidebar.success("✅ อัปเดตสถิติจาก Excel เรียบร้อย")
        else:
            st.sidebar.error("❌ ไม่พบคอลัมน์ 'Feeder' ในไฟล์")
    except Exception as e:
        st.sidebar.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

st.sidebar.divider()

# 2.2 บันทึกผลสำรวจหน้างาน & อัปโหลดรูปกล้อง Acoustic
st.sidebar.subheader("📸 ผลสำรวจ & Acoustic Image")
target_id = st.sidebar.selectbox("เลือกหม้อแปลงที่สำรวจ:", st.session_state.asset_df['Transformer_ID'])

# ช่องอัปโหลดรูปภาพ
acoustic_img = st.sidebar.file_uploader("อัปโหลดรูปจากกล้อง Acoustic", type=["jpg", "png", "jpeg"])
if acoustic_img:
    st.sidebar.image(acoustic_img, caption="ภาพถ่ายหน้างาน", use_column_width=True)

# ช่องกรอกข้อมูล (Number Input)
ac_db = st.sidebar.number_input("ค่าเสียง Acoustic (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("Peak Frequency (Hz)", 1000, 100000, 20000)
s_temp = st.sidebar.number_input("อุณหภูมิ (°C)", 20.0, 120.0, 55.0)
s_load = st.sidebar.number_input("ภาระไฟฟ้า Load (%)", 0.0, 150.0, 75.0)

if st.sidebar.button("🤖 วิเคราะห์และบันทึกข้อมูล"):
    asset_data = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == target_id].iloc[0]
    
    # ส่ง 8 ตัวแปรให้ AI [Temp, Load, Oil(95), Vib(1.5), Age, Hum(55), dB, Hz]
    features = np.array([[s_temp, s_load, 95.0, 1.5, asset_data['Age (Years)'], 55.0, ac_db, ac_hz]])
    prob = model.predict_proba(features)[0][1]
    
    # คำนวณสถานะร่วมกับสถิติไฟดับ
    trips = asset_data['Trip_History']
    if prob > 0.75 or trips >= 5:
        new_status = "🔴 CRITICAL"
    elif prob > 0.4 or trips >= 2:
        new_status = "🟡 WATCH"
    else:
        new_status = "🟢 NORMAL"
    
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Status'] = new_status
    st.sidebar.success(f"อัปเดต {target_id} สำเร็จ!")

# --- 3. หน้าจอหลัก (Dashboard Central) ---
st.title("🏙️ MEA Asset Intelligence - เขตนวลจันทร์ (ฟขจ.)")
st.write(f"วิเคราะห์ข้อมูลแผนงานบำรุงรักษา ประจำวันที่ {datetime.now().strftime('%d/%m/%Y')}")

# ตารางแดชบอร์ดศูนย์กลาง
st.subheader("📊 ตารางติดตามสถานะสุขภาพอุปกรณ์รายพื้นที่")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.asset_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

# กราฟวิเคราะห์
st.divider()
col_l, col_r = st.columns(2)
with col_l:
    st.write("### สัดส่วนสถานะสุขภาพหม้อแปลง")
    fig_pie = px.pie(st.session_state.asset_df, names='Status', color='Status',
                     color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 NORMAL':'#28a745'})
    st.plotly_chart(fig_pie, use_column_width=True)

with col_r:
    st.write("### สถิติไฟดับสะสมแยกตามฟีดเดอร์ (ฟขจ.)")
    fig_bar = px.bar(st.session_state.asset_df, x='Feeder', y='Trip_History', color='Status',
                     color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 NORMAL':'#28a745'})
    st.plotly_chart(fig_bar, use_column_width=True)
