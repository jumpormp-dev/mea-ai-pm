import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. โหลดโมเดล AI (8 ตัวแปร) ---
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl')

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดลบน GitHub (กรุณาอัปโหลด mea_pm_ai_model.pkl)")

st.set_page_config(page_title="MEA Smart PM - นวลจันทร์", layout="wide")

# --- 2. ฐานข้อมูลเขตนวลจันทร์ (ข้อมูลจากไฟล์ที่คุณอัปโหลด) ---
if 'asset_df' not in st.session_state:
    st.session_state.asset_df = pd.DataFrame({
        'Transformer_ID': ['TR-NJ-001', 'TR-NJ-002', 'TR-NJ-003', 'TR-NJ-004', 'TR-NJ-005'],
        'Feeder': ['GK-424', 'GK-422', 'LK-422', 'KJ-433', 'KJ-432'],
        'Location': ['ถ.นวลจันทร์', 'ถ.รามอินทรา', 'ซ.สุคนธสวัสดิ์', 'ถ.ประดิษฐ์มนูธรรม', 'ซ.นวลจันทร์ 36'],
        'Age (Years)': [12, 22, 5, 18, 10],
        'Trip_Count': [0, 0, 0, 0, 0], 
        'Risk_Score': [0.0] * 5,
        'Status': ['🟢 NORMAL'] * 5,
        'Action_Plan': ['ตรวจสอบตามรอบปกติ'] * 5
    })

# --- 3. Sidebar: ศูนย์รับข้อมูล (Input Center) ---
st.sidebar.header("📥 จัดการข้อมูล ฟขจ.")

# ส่วนอัปโหลดสถิติ Feeder (Excel)
feeder_file = st.sidebar.file_uploader("อัปโหลดสถิติไฟดับ (.xlsx)", type=["xlsx"])
if feeder_file:
    try:
        df_web = pd.read_excel(feeder_file, skiprows=2)
        trip_stats = df_web['Feeder'].value_counts().to_dict()
        for fid, count in trip_stats.items():
            st.session_state.asset_df.loc[st.session_state.asset_df['Feeder'] == fid, 'Trip_Count'] = count
        st.sidebar.success("✅ อัปเดตสถิติไฟดับสำเร็จ")
    except:
        st.sidebar.error("❌ รูปแบบไฟล์ไม่ถูกต้อง")

st.sidebar.divider()
st.sidebar.subheader("📸 บันทึกผลสำรวจรายเครื่อง")
target_id = st.sidebar.selectbox("เลือกหม้อแปลง:", st.session_state.asset_df['Transformer_ID'])

# อัปโหลดรูปภาพ Acoustic
acoustic_img = st.sidebar.file_uploader("อัปโหลดรูปภาพกล้อง Acoustic", type=["jpg", "png", "jpeg"])
if acoustic_img:
    st.sidebar.image(acoustic_img, caption="หลักฐานหน้างาน", use_column_width=True)

# กรอกข้อมูล (Number Input)
ac_db = st.sidebar.number_input("ความดัง Acoustic (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("Peak Frequency (Hz)", 1000, 100000, 20000)
s_temp = st.sidebar.number_input("อุณหภูมิ (°C)", 20.0, 120.0, 50.0)
s_load = st.sidebar.number_input("ภาระไฟฟ้า Load (%)", 0.0, 150.0, 70.0)

if st.sidebar.button("🤖 AI วิเคราะห์และวางแผน"):
    asset = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == target_id].iloc[0]
    # Input 8 ตัวแปรให้ AI
    features = np.array([[s_temp, s_load, 95.0, 1.5, asset['Age (Years)'], 55.0, ac_db, ac_hz]])
    prob = model.predict_proba(features)[0][1]
    
    # Logic วิเคราะห์สถานะ
    trips = asset['Trip_Count']
    if prob > 0.75 or trips >= 5:
        st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Status'] = "🔴 CRITICAL"
        st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Action_Plan'] = "PM ด่วนที่สุด"
    elif prob > 0.4 or trips >= 2:
        st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Status'] = "🟡 WATCH"
        st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Action_Plan'] = "เข้าแผน PM รายเดือน"
    else:
        st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Status'] = "🟢 NORMAL"
        st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Action_Plan'] = "ตรวจสอบตามรอบปกติ"
    
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Risk_Score'] = prob
    st.sidebar.success(f"วิเคราะห์ {target_id} สำเร็จ")

# --- 4. Dashboard กลาง ---
st.title("🏙️ MEA Asset Management - เขตนวลจันทร์ (ฟขจ.)")
st.write(f"สรุปการวิเคราะห์ข้อมูลประจำวันที่ {datetime.now().strftime('%d/%m/%Y')}")

# ตารางหลัก
st.subheader("📊 ตารางติดตามสถานะและแผนงาน")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.asset_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

# --- ส่วนการวิเคราะห์เหตุผลและความเสี่ยง (AI Diagnostic) ---
st.divider()
col_l, col_r = st.columns([1.5, 1])

with col_l:
    st.subheader("🔍 บทวิเคราะห์รายอุปกรณ์")
    sel_id = st.selectbox("เลือกหม้อแปลงเพื่อดูเหตุผลความเสี่ยง:", st.session_state.asset_df['Transformer_ID'])
    row = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == sel_id].iloc[0]
    
    # วิเคราะห์เหตุผล
    st.write(f"**ทำไมถึงเสี่ยง:**")
    if row['Risk_Score'] > 0.6: st.write("- AI ตรวจพบรูปแบบค่า Acoustic และความร้อนที่ผิดปกติ")
    if row['Trip_Count'] >= 3: st.write(f"- สถิติไฟดับในฟีดเดอร์ {row['Feeder']} มีแนวโน้มสูงผิดปกติ ({row['Trip_Count']} ครั้ง)")
    if row['Age (Years)'] > 20: st.write(f"- อุปกรณ์มีอายุการใช้งานสูง ({row['Age (Years)']} ปี)")
    if row['Status'] == "🟢 NORMAL": st.success("ทุกปัจจัยยังอยู่ในเกณฑ์มาตรฐาน")

    st.write(f"**แนวโน้ม (Trend):**")
    if row['Status'] == "🔴 CRITICAL": st.error("เสี่ยงต่อการเกิด breakdown ทันที หากภาระไฟฟ้าเพิ่มสูงขึ้น")
    else: st.info("สถานะคงที่ สามารถใช้งานได้ตามรอบบำรุงรักษา")

with col_r:
    st.subheader("📅 แผนการเข้าทำ PM")
    days = int(max(2, (1 - row['Risk_Score']) * 90))
    pm_date = (datetime.now() + timedelta(days=days)).strftime('%d/%m/%Y')
    
    st.metric("วันที่แนะนำให้เข้าดำเนินการ", pm_date)
    st.write(f"**ความเร่งด่วน:** {'สูงมาก' if row['Risk_Score'] > 0.7 else 'ปานกลาง' if row['Risk_Score'] > 0.4 else 'ปกติ'}")
    st.write(f"**งานที่ต้องทำ:** {row['Action_Plan']}")
