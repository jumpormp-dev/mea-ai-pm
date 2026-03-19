import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from PIL import Image

# --- โหลดโมเดล 8 ตัวแปร ---
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl')

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_pm_ai_model.pkl' กรุณาตรวจสอบบน GitHub")

st.set_page_config(page_title="MEA Smart PM Dashboard", layout="wide")

# --- 1. ฐานข้อมูลอุปกรณ์ในพื้นที่ (Asset Database) ---
if 'asset_df' not in st.session_state:
    st.session_state.asset_df = pd.DataFrame({
        'Transformer_ID': ['TR-BK-001', 'TR-BK-002', 'TR-BK-003', 'TR-BK-004', 'TR-BK-005'],
        'Feeder': ['GK-424', 'GK-422', 'LK-422', 'KJ-433', 'GK-424'],
        'Location': ['เขตนวลจันทร์', 'เขตนวลจันทร์', 'เขตนวลจันทร์', 'เขตนวลจันทร์', 'เขตนวลจันทร์'],
        'Age (Years)': [5, 18, 12, 22, 8],
        'Trip_Count': [0, 0, 0, 0, 0],
        'Status': ['🟢 NORMAL'] * 5
    })

# --- 2. ส่วนแถบเมนูข้าง (Sidebar Control) ---
st.sidebar.header("📥 ศูนย์นำเข้าข้อมูล (Data Hub)")

# 2.1 อัปโหลดไฟล์สถิติฟีดเดอร์ (จากเว็บ 10.99.1.36)
st.sidebar.subheader("🌐 สถิติจากระบบ Reliability")
feeder_file = st.sidebar.file_uploader("อัปโหลดไฟล์สถิติฟีดเดอร์ (CSV/Excel)", type=["csv", "xlsx"])

if feeder_file:
    try:
        # อ่านไฟล์สถิติที่อัปโหลด (ข้ามหัว 2 บรรทัดแรกตามโครงสร้างไฟล์คุณ)
        df_feeder = pd.read_csv(feeder_file, skiprows=2)
        # นับจำนวนครั้งที่เกิด Outage แยกตาม Feeder
        trip_stats = df_feeder['Feeder'].value_counts()
        
        # อัปเดตลงในฐานข้อมูลจำลอง
        for fid in trip_stats.index:
            st.session_state.asset_df.loc[st.session_state.asset_df['Feeder'] == fid, 'Trip_Count'] = trip_stats[fid]
        st.sidebar.success("✅ อัปเดตสถิติฟีดเดอร์เรียบร้อย")
    except Exception as e:
        st.sidebar.error(f"❌ รูปแบบไฟล์ไม่ถูกต้อง: {e}")

st.sidebar.divider()

# 2.2 การสำรวจหน้างานและการอัปโหลดรูป
st.sidebar.subheader("📸 บันทึกผลสำรวจ & Acoustic Image")
target_id = st.sidebar.selectbox("เลือกหม้อแปลงที่สำรวจ:", st.session_state.asset_df['Transformer_ID'])

# ช่องอัปโหลดรูปภาพจากกล้อง Acoustic
acoustic_img = st.sidebar.file_uploader("อัปโหลดรูปจาก Acoustic Camera", type=["jpg", "png", "jpeg"])
if acoustic_img:
    st.sidebar.image(acoustic_img, caption="ตัวอย่างภาพที่อัปโหลด", use_column_width=True)

# เปลี่ยนจาก Slider เป็นช่องกรอกตามที่ขอ
ac_db = st.sidebar.number_input("ค่าความดังเสียง (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("Peak Frequency (Hz)", 1000, 100000, 20000)
s_temp = st.sidebar.number_input("อุณหภูมิที่วัดได้ (°C)", 20.0, 120.0, 50.0)
s_load = st.sidebar.number_input("ภาระไฟฟ้า Load (%)", 0.0, 150.0, 70.0)

# ปุ่มประมวลผล
if st.sidebar.button("🤖 วิเคราะห์สถานะและบันทึก"):
    asset_data = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == target_id].iloc[0]
    
    # ส่ง 8 ตัวแปรให้ AI [Temp, Load, Oil, Vib, Age, Hum, dB, Hz]
    features = np.array([[s_temp, s_load, 95.0, 1.5, asset_data['Age (Years)'], 55.0, ac_db, ac_hz]])
    prob = model.predict_proba(features)[0][1]
    
    # สรุปสถานะ (รวมสถิติ Trip จากไฟล์ Feeder เข้าไปด้วย)
    trips = asset_data['Trip_Count']
    if prob > 0.75 or trips >= 5:
        res_status = "🔴 CRITICAL"
    elif prob > 0.4 or trips >= 2:
        res_status = "🟡 WATCH"
    else:
        res_status = "🟢 NORMAL"
    
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Status'] = res_status
    st.sidebar.success(f"บันทึกข้อมูล {target_id} สำเร็จ!")

# --- 3. ส่วนการแสดงผลหลัก (Dashboard) ---
st.title("🏙️ MEA Smart Maintenance Dashboard")
st.write(f"วิเคราะห์พื้นที่: เขตนวลจันทร์ | ข้อมูลล่าสุด: {datetime.now().strftime('%d/%m/%Y')}")

# สรุปภาพรวม Metrics
m1, m2, m3 = st.columns(3)
m1.metric("จำนวนหม้อแปลง", len(st.session_state.asset_df))
m2.metric("สถานะวิกฤต (PM ด่วน)", len(st.session_state.asset_df[st.session_state.asset_df['Status'] == "🔴 CRITICAL"]))
m3.metric("ฟีดเดอร์ที่ขัดข้องสูงสุด", st.session_state.asset_df.loc[st.session_state.asset_df['Trip_Count'].idxmax(), 'Feeder'])

# ตารางหลักตรงกลาง (Status Monitor)
st.subheader("📊 ตารางติดตามสถานะอุปกรณ์ในพื้นที่")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.asset_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

# กราฟแสดงสถิติ
st.divider()
c_left, c_right = st.columns(2)
with c_left:
    st.write("### สัดส่วนสุขภาพอุปกรณ์")
    fig_pie = px.pie(st.session_state.asset_df, names='Status', color='Status',
                     color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 NORMAL':'#28a745'})
    st.plotly_chart(fig_pie, use_column_width=True)

with c_right:
    st.write("### สถิติการขัดข้องแยกตาม Feeder (จากไฟล์ที่อัปโหลด)")
    fig_bar = px.bar(st.session_state.asset_df, x='Feeder', y='Trip_Count', color='Status',
                     color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 NORMAL':'#28a745'})
    st.plotly_chart(fig_bar, use_column_width=True)
