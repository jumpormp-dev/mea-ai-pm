import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. โหลดโมเดล AI (8 ตัวแปร) ---
@st.cache_resource
def load_my_model():
    # ชื่อไฟล์โมเดลที่คุณเทรนจาก Colab
    return joblib.load('mea_pm_ai_model.pkl')

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_pm_ai_model.pkl' กรุณาตรวจสอบบน GitHub")

# ตั้งชื่อแอปในแท็บ Browser
st.set_page_config(page_title="SPP-AI: Smart Predictive Planning", layout="wide")

# --- 2. ฐานข้อมูลเริ่มต้นของเขตนวลจันทร์ (ฟขจ.) ---
if 'asset_df' not in st.session_state:
    st.session_state.asset_df = pd.DataFrame({
        'Transformer_ID': ['TR-NJ-001', 'TR-NJ-002', 'TR-NJ-003', 'TR-NJ-004', 'TR-NJ-005'],
        'Feeder': ['GK-424', 'GK-422', 'LK-422', 'KJ-433', 'KJ-432'],
        'Location': ['ถ.นวลจันทร์', 'ถ.รามอินทรา', 'ซ.สุคนธสวัสดิ์', 'ถ.ประดิษฐ์มนูธรรม', 'ซ.นวลจันทร์ 36'],
        'Age (Years)': [12, 22, 5, 18, 10],
        'Trip_Count': [0] * 5,
        'Risk_Score': [0.1] * 5,
        'Status': ['🟢 NORMAL'] * 5,
        'Last_Update': ['-'] * 5
    })

# --- 3. Sidebar: ส่วนนำเข้าข้อมูลและอัปโหลดรูป ---
st.sidebar.header("📥 SPP Data Management")

# อัปโหลดสถิติ Feeder จาก Excel
feeder_file = st.sidebar.file_uploader("อัปโหลดสถิติไฟดับ ฟขจ. (.xlsx)", type=["xlsx"])
if feeder_file:
    try:
        df_web = pd.read_excel(feeder_file, skiprows=2)
        trip_stats = df_web['Feeder'].value_counts().to_dict()
        for fid, count in trip_stats.items():
            st.session_state.asset_df.loc[st.session_state.asset_df['Feeder'] == fid, 'Trip_Count'] = count
        st.sidebar.success("✅ อัปเดตสถิติฟีดเดอร์เรียบร้อย")
    except:
        st.sidebar.error("❌ รูปแบบไฟล์ Excel ไม่ถูกต้อง")

st.sidebar.divider()
st.sidebar.subheader("📸 บันทึกผลสำรวจ (Field Survey)")
target_id = st.sidebar.selectbox("เลือกอุปกรณ์ที่จะบันทึก:", st.session_state.asset_df['Transformer_ID'])

# อัปโหลดรูปภาพ Acoustic Camera
acoustic_img = st.sidebar.file_uploader("อัปโหลดรูป Acoustic Camera", type=["jpg", "png", "jpeg"])
if acoustic_img:
    st.sidebar.image(acoustic_img, caption=f"ภาพหน้างาน: {target_id}", use_column_width=True)

# ช่องกรอกข้อมูลเทคนิค (เลิกใช้ Slider)
ac_db = st.sidebar.number_input("ความดังเสียง (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("Peak Frequency (Hz)", 1000, 100000, 20000)
s_temp = st.sidebar.number_input("อุณหภูมิ (°C)", 20.0, 120.0, 50.0)
s_load = st.sidebar.number_input("ภาระไฟฟ้า Load (%)", 0.0, 150.0, 75.0)

if st.sidebar.button("🤖 วิเคราะห์และอัปเดตสถานะ"):
    idx = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == target_id].index[0]
    asset = st.session_state.asset_df.iloc[idx]
    
    # ส่ง 8 ตัวแปรให้สมอง AI (Input Data)
    input_features = np.array([[s_temp, s_load, 95.0, 1.5, asset['Age (Years)'], 55.0, ac_db, ac_hz]])
    prob = model.predict_proba(input_features)[0][1]
    
    # คำนวณสถานะ (AI Risk + Trip Context)
    trips = asset['Trip_Count']
    if prob > 0.7 or trips >= 5:
        new_stat = "🔴 CRITICAL"
    elif prob > 0.4 or trips >= 2:
        new_stat = "🟡 WATCH"
    else:
        new_stat = "🟢 NORMAL"
    
    st.session_state.asset_df.at[idx, 'Status'] = new_stat
    st.session_state.asset_df.at[idx, 'Risk_Score'] = prob
    st.session_state.asset_df.at[idx, 'Last_Update'] = datetime.now().strftime('%d/%m/%Y %H:%M')
    st.sidebar.success(f"อัปเดตข้อมูล {target_id} สำเร็จ!")

# --- 4. Dashboard กลาง (Main Table & Selected Analysis) ---
st.title("🏙️ SPP-AI: Smart Predictive Planning Dashboard")
st.subheader("ระบบบริหารจัดการเขตพื้นที่: นวลจันทร์ (ฟขจ.)")

# ส่วนที่ 1: ตารางแสดงสถานะอุปกรณ์ทุกเครื่องในพื้นที่
st.write("### 📋 ตารางติดตามสถานะสุขภาพอุปกรณ์ (Area Monitor)")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.asset_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

st.divider()

# ส่วนที่ 2: Interactive Asset Explorer (คลิกเลือกดูรายตัว)
st.write("### 🔍 เจาะลึกบทวิเคราะห์และแผนงาน (Asset Insight)")
selected_id = st.selectbox("คลิกเพื่อเลือกหม้อแปลงที่ต้องการดูรายละเอียด:", st.session_state.asset_df['Transformer_ID'])
res = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == selected_id].iloc[0]

col_diag, col_plan = st.columns([1.5, 1])

with col_diag:
    st.write(f"#### ⚙️ ข้อมูลเทคนิค: {selected_id}")
    st.write(f"**Feeder:** {res['Feeder']} | **สถานที่:** {res['Location']}")
    st.write(f"**คะแนนความเสี่ยงจาก AI:** {res['Risk_Score']*100:.1f}%")
    
    st.write("---")
    st.write("#### 🤖 บทวิเคราะห์เหตุผล (Why this status?)")
    if res['Status'] == "🟢 NORMAL":
        st.success("✅ อุปกรณ์ทำงานปกติ: ปัจจัยความร้อนและเสียง Acoustic ยังอยู่ในเกณฑ์มาตรฐาน")
    else:
        if res['Risk_Score'] > 0.5:
            st.warning("⚠️ **ปัจจัยภายใน:** AI ตรวจพบความสัมพันธ์ของอุณหภูมิและเสียงที่ผิดปกติ")
        if res['Trip_Count'] >= 2:
            st.warning(f"⚠️ **ปัจจัยภายนอก:** มีสถิติไฟดับใน Feeder {res['Feeder']} สูง ({res['Trip_Count']} ครั้ง) กระทบต่อฉนวน")

with col_plan:
    st.write("#### 📅 การวางแผนบำรุงรักษา (PM Schedule)")
    # สมการพยากรณ์วัน PM
    days_left = int(max(2, (1 - res['Risk_Score']) * 90))
    pm_date = (datetime.now() + timedelta(days=days_left)).strftime('%d/%m/%Y')
    
    st.metric("กำหนดการ PM ที่แนะนำ", pm_date)
    st.write(f"**ความเร่งด่วน:** {'สูงมาก (Immediate)' if res['Risk_Score'] > 0.7 else 'ปานกลาง (Planned)' if res['Risk_Score'] > 0.3 else 'ปกติ (Routine)'}")
    
    # คำแนะนำการทำงาน
    if res['Status'] == "🔴 CRITICAL":
        st.error("🚨 **ข้อแนะนำ:** จัดทีมเข้าตรวจสอบจุดร้อนและเปลี่ยนอุปกรณ์ทันที")
    elif res['Status'] == "🟡 WATCH":
        st.warning("⏳ **ข้อแนะนำ:** บรรจุเข้าแผน PM ประจำเดือน และตรวจสอบซ้ำด้วยเครื่องวัดระดับเสียง")
    else:
        st.info("📅 **ข้อแนะนำ:** ดำเนินการบำรุงรักษาตามรอบปกติ")
