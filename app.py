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
    st.error("❌ ไม่พบไฟล์ 'mea_pm_ai_model.pkl' กรุณาอัปโหลดขึ้น GitHub")

st.set_page_config(page_title="MEA Smart Area Monitor - ฟขจ.", layout="wide")

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
        'Last_Inspect': ['-'] * 5
    })

# --- 3. Sidebar: ส่วนนำเข้าข้อมูล ---
st.sidebar.header("📥 Data Management")

# อัปโหลดไฟล์สถิติ Feeder (Excel)
feeder_file = st.sidebar.file_uploader("อัปโหลดสถิติไฟดับ ฟขจ. (.xlsx)", type=["xlsx"])
if feeder_file:
    try:
        df_web = pd.read_excel(feeder_file, skiprows=2)
        trip_stats = df_web['Feeder'].value_counts().to_dict()
        for fid, count in trip_stats.items():
            st.session_state.asset_df.loc[st.session_state.asset_df['Feeder'] == fid, 'Trip_Count'] = count
        st.sidebar.success("✅ อัปเดตสถิติฟีดเดอร์แล้ว")
    except:
        st.sidebar.error("❌ รูปแบบไฟล์ไม่ถูกต้อง")

st.sidebar.divider()
st.sidebar.subheader("📸 บันทึกผลสำรวจใหม่")
target_id = st.sidebar.selectbox("เลือกอุปกรณ์ที่จะบันทึก:", st.session_state.asset_df['Transformer_ID'])
acoustic_img = st.sidebar.file_uploader("อัปโหลดรูป Acoustic", type=["jpg", "png", "jpeg"])

# ช่องกรอกข้อมูล
ac_db = st.sidebar.number_input("ความดัง (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("ความถี่ (Hz)", 1000, 100000, 20000)
s_temp = st.sidebar.number_input("อุณหภูมิ (°C)", 20.0, 120.0, 50.0)
s_load = st.sidebar.number_input("Load (%)", 0.0, 150.0, 70.0)

if st.sidebar.button("🤖 ประมวลผลและอัปเดตสถานะ"):
    idx = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == target_id].index[0]
    asset = st.session_state.asset_df.iloc[idx]
    
    # AI Prediction (8 Features)
    features = np.array([[s_temp, s_load, 95.0, 1.5, asset['Age (Years)'], 55.0, ac_db, ac_hz]])
    prob = model.predict_proba(features)[0][1]
    
    # Update Status Logic
    trips = asset['Trip_Count']
    new_stat = "🔴 CRITICAL" if (prob > 0.7 or trips >= 5) else "🟡 WATCH" if (prob > 0.4 or trips >= 2) else "🟢 NORMAL"
    
    st.session_state.asset_df.at[idx, 'Status'] = new_stat
    st.session_state.asset_df.at[idx, 'Risk_Score'] = prob
    st.session_state.asset_df.at[idx, 'Last_Inspect'] = datetime.now().strftime('%d/%m/%Y %H:%M')
    st.sidebar.success(f"อัปเดต {target_id} เรียบร้อย!")

# --- 4. หน้าจอหลัก (Dashboard & Selector) ---
st.title("🏙️ MEA Asset Intelligence - เขตนวลจันทร์ (ฟขจ.)")

# ส่วนที่ 1: ตารางสถานะทั้งหมด (คลิกเลือกแถวเพื่อดูรายละเอียดได้ในส่วนถัดไป)
st.subheader("📋 สถานะอุปกรณ์ทั้งหมดในพื้นที่")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.asset_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

st.divider()

# ส่วนที่ 2: เจาะลึกรายเครื่อง (Interactive Explorer)
st.subheader("🔍 คลิกเลือกอุปกรณ์เพื่อดูบทวิเคราะห์รายตัว")
selected_id = st.selectbox("เลือกหม้อแปลงที่ต้องการตรวจสอบรายละเอียด:", st.session_state.asset_df['Transformer_ID'])
res = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == selected_id].iloc[0]

col_detail, col_plan = st.columns([1.5, 1])

with col_detail:
    st.write(f"### ⚙️ ข้อมูลทางเทคนิค: {selected_id}")
    st.write(f"**ฟีดเดอร์:** {res['Feeder']} | **สถานที่:** {res['Location']}")
    st.write(f"**อายุอุปกรณ์:** {res['Age (Years)']} ปี | **ตรวจสอบล่าสุด:** {res['Last_Inspect']}")
    
    # บทวิเคราะห์จาก AI และสถิติ
    st.write("---")
    st.write("#### 🤖 บทวิเคราะห์เหตุผล (Diagnostic)")
    if res['Status'] == "🟢 NORMAL":
        st.success("✅ อุปกรณ์ทำงานปกติ: ไม่พบเสียงผิดปกติและความร้อนอยู่ในเกณฑ์มาตรฐาน")
    else:
        if res['Risk_Score'] > 0.5:
            st.warning("⚠️ **ปัจจัยภายใน:** ตรวจพบรูปแบบคลื่นเสียง Acoustic และอุณหภูมิที่บ่งบอกถึงการเสื่อมสภาพ")
        if res['Trip_Count'] >= 2:
            st.warning(f"⚠️ **ปัจจัยภายนอก:** สถิติไฟดับในฟีดเดอร์ {res['Feeder']} สูง ({res['Trip_Count']} ครั้ง) ส่งผลกระทบต่ออายุการใช้งาน")

with col_plan:
    st.write("### 📅 แผนการซ่อมบำรุง (PM Plan)")
    # คำนวณวัน PM
    days = int(max(2, (1 - res['Risk_Score']) * 90))
    pm_date = (datetime.now() + timedelta(days=days)).strftime('%d/%m/%Y')
    
    st.metric("ระดับความเสี่ยง", f"{res['Risk_Score']*100:.1f}%")
    st.metric("กำหนดเข้าทำ PM ที่แนะนำ", pm_date)
    
    st.write("**แนวโน้มและความเร่งด่วน:**")
    if res['Status'] == "🔴 CRITICAL":
        st.error("🚨 **ด่วนที่สุด:** ต้องเข้าดำเนินการภายใน 7 วัน เพื่อป้องกัน Breakdown")
    elif res['Status'] == "🟡 WATCH":
        st.warning("⏳ **ปานกลาง:** บรรจุเข้าแผน PM ประจำเดือน และเฝ้าระวังจุดร้อน")
    else:
        st.info("📅 **ปกติ:** ดำเนินการตรวจสอบตามรอบประจำปี")
