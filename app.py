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
    st.error("❌ ไม่พบไฟล์โมเดลบน GitHub")

st.set_page_config(page_title="MEA Smart PM - ฟขจ.", layout="wide")

# --- 2. ฐานข้อมูลหม้อแปลงเขตนวลจันทร์ (ข้อมูลเริ่มต้น) ---
if 'asset_df' not in st.session_state:
    st.session_state.asset_df = pd.DataFrame({
        'Transformer_ID': ['TR-NJ-001', 'TR-NJ-002', 'TR-NJ-003', 'TR-NJ-004', 'TR-NJ-005'],
        'Feeder': ['GK-424', 'GK-422', 'LK-422', 'KJ-433', 'GK-424'],
        'Age (Years)': [12, 22, 5, 18, 10],
        'Trip_History': [0, 0, 0, 0, 0], 
        'Risk_Score': [0.0] * 5,
        'Status': ['🟢 NORMAL'] * 5,
        'Recommendation': ['ตรวจสอบตามรอบปกติ'] * 5
    })

# --- 3. แถบเมนูข้าง (Data & Survey Input) ---
st.sidebar.header("📥 ศูนย์นำเข้าข้อมูล ฟขจ.")

# อัปโหลดไฟล์สถิติจากหน่วยงาน (Excel)
feeder_file = st.sidebar.file_uploader("อัปโหลดไฟล์ Feeder Indices (.xlsx)", type=["xlsx"])
if feeder_file:
    try:
        df_feeder = pd.read_excel(feeder_file, skiprows=2)
        trip_counts = df_feeder['Feeder'].value_counts().to_dict()
        for fid, count in trip_counts.items():
            st.session_state.asset_df.loc[st.session_state.asset_df['Feeder'] == fid, 'Trip_History'] = count
        st.sidebar.success("✅ อัปเดตสถิติไฟดับเรียบร้อย")
    except:
        st.sidebar.error("❌ ไฟล์ไม่ถูกต้อง")

st.sidebar.divider()
st.sidebar.subheader("📸 บันทึกผลสำรวจรายเครื่อง")
target_id = st.sidebar.selectbox("เลือกหม้อแปลง:", st.session_state.asset_df['Transformer_ID'])

# อัปโหลดรูปภาพ Acoustic
acoustic_img = st.sidebar.file_uploader("อัปโหลดรูป Acoustic Camera", type=["jpg", "png", "jpeg"])
if acoustic_img:
    st.sidebar.image(acoustic_img, caption="หลักฐานหน้างาน", use_column_width=True)

# ช่องกรอกข้อมูล (Number Input)
ac_db = st.sidebar.number_input("ความดัง Acoustic (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("Peak Frequency (Hz)", 1000, 100000, 20000)
s_temp = st.sidebar.number_input("อุณหภูมิ (°C)", 20.0, 120.0, 50.0)
s_load = st.sidebar.number_input("ภาระไฟฟ้า Load (%)", 0.0, 150.0, 70.0)

if st.sidebar.button("🤖 วิเคราะห์เหตุผลและวางแผน PM"):
    asset_data = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == target_id].iloc[0]
    
    # 8 Features: [Temp, Load, Oil, Vib, Age, Hum, dB, Hz]
    input_data = np.array([[s_temp, s_load, 95.0, 1.5, asset_data['Age (Years)'], 55.0, ac_db, ac_hz]])
    prob = model.predict_proba(input_data)[0][1]
    
    # คำนวณสถานะและแผนงาน
    trips = asset_data['Trip_History']
    if prob > 0.75 or trips >= 5:
        status = "🔴 CRITICAL"
        rectext = "ดำเนินการบำรุงรักษาด่วนที่สุด"
    elif prob > 0.4 or trips >= 2:
        status = "🟡 WATCH"
        rectext = "จัดเข้าแผนบำรุงรักษารายเดือน"
    else:
        status = "🟢 NORMAL"
        rectext = "ตรวจสอบตามรอบปกติ"
    
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Status'] = status
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Risk_Score'] = prob
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Recommendation'] = rectext
    
    st.sidebar.success(f"วิเคราะห์ {target_id} สำเร็จ")

# --- 4. หน้าจอหลัก (Dashboard & Analysis) ---
st.title("🏙️ MEA Asset Intelligence - เขตนวลจันทร์ (ฟขจ.)")

# ตาราง Monitor ตรงกลาง
st.subheader("📊 ตารางติดตามสถานะและแผน PM")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.asset_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

# --- ส่วนการวิเคราะห์เจาะลึก (Deep Analysis) ---
st.divider()
selected_asset = st.selectbox("เลือกอุปกรณ์เพื่อดูบทวิเคราะห์เหตุผลและความเสี่ยง:", st.session_state.asset_df['Transformer_ID'])
data_row = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == selected_asset].iloc[0]

col_a, col_b = st.columns([1.5, 1])

with col_a:
    st.write("### 🔍 ทำไมถึงเสี่ยง? (Diagnostic Insights)")
    
    # วิเคราะห์ปัจจัยเสี่ยงหลัก (Logic-based Explanation)
    reasons = []
    if data_row['Trip_History'] >= 5: reasons.append(f"• สถิติไฟดับในฟีดเดอร์ {data_row['Feeder']} สูงผิดปกติ ({data_row['Trip_History']} ครั้ง)")
    if data_row['Risk_Score'] > 0.6: reasons.append("• ค่า Acoustic และความร้อนสอดคล้องกับรูปแบบการชำรุดในอดีต")
    if data_row['Age (Years)'] > 20: reasons.append(f"• อุปกรณ์มีอายุการใช้งานสูง ({data_row['Age (Years)']} ปี) เสี่ยงต่อความเสื่อมสภาพตามกาลเวลา")
    
    if not reasons:
        st.success("✅ ปัจจัยทางเทคนิคและบริบทพื้นทียังอยู่ในเกณฑ์ปกติ")
    else:
        for r in reasons: st.write(r)

    # แนวโน้ม (Trend Prediction)
    st.write("### 📈 แนวโน้มในอนาคต (Predictive Trend)")
    if data_row['Status'] == "🔴 CRITICAL":
        st.error("⚠️ **แนวโน้ม:** มีความเสี่ยงสูงที่จะเกิดการขัดข้อง (Breakdown) ภายใน 1-2 สัปดาห์ หากภาระไฟฟ้าเพิ่มสูงขึ้น")
    elif data_row['Status'] == "🟡 WATCH":
        st.warning("⚠️ **แนวโน้ม:** อุปกรณ์เริ่มเสื่อมสภาพ หากไม่เข้าดำเนินการภายใน 3 เดือน ความเสี่ยงจะขยับขึ้นสู่ระดับวิกฤต")
    else:
        st.info("ℹ️ **แนวโน้ม:** สถานะคงที่ สามารถใช้งานได้ตามปกติจนถึงรอบการบำรุงรักษาถัดไป")

with col_b:
    st.write("### 📅 แผนการเข้าดำเนินการ (PM Scheduler)")
    prob_val = data_row['Risk_Score']
    
    # คำนวณวันที่ควรทำ PM
    days_to_pm = int(max(2, (1 - prob_val) * 90)) # เสี่ยงมาก วันยิ่งน้อย
    pm_date = (datetime.now() + timedelta(days=days_to_pm)).strftime('%d/%m/%Y')
    
    st.metric("กำหนดการ PM ที่แนะนำ", pm_date)
    st.write(f"**ความเร่งด่วน:** {'สูงมาก (Immediate)' if prob_val > 0.7 else 'ปานกลาง (Planned)' if prob_val > 0.3 else 'ปกติ (Routine)'}")
    
    # สรุปขั้นตอนงาน
    st.write("**ขั้นตอนที่แนะนำ:**")
    if data_row['Status'] == "🔴 CRITICAL":
        st.write("1. ส่งทีมแก้ไฟเข้าตรวจสอบจุดร้อนหน้างาน")
        st.write("2. เตรียมแผนย้ายโหลด (Load Transfer)")
        st.write("3. ดำเนินการเปลี่ยนหรือซ่อมบำรุงใหญ่")
    else:
        st.write("1. บันทึกค่าเก็บในฐานข้อมูล")
        st.write("2. ตรวจสอบซ้ำในรอบ 6 เดือน")
