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
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_pm_ai_model.pkl' บน GitHub")

st.set_page_config(page_title="MEA Smart Area PM Planner", layout="wide")

# --- 1. ฐานข้อมูลหม้อแปลงจำลองในเขตพื้นที่ ---
if 'asset_df' not in st.session_state:
    st.session_state.asset_df = pd.DataFrame({
        'Transformer_ID': ['TR-BK-001', 'TR-BK-002', 'TR-BK-003', 'TR-BK-004', 'TR-BK-005'],
        'Feeder': ['GK-424', 'GK-422', 'LK-422', 'KJ-433', 'GK-424'],
        'Location': ['เขตนวลจันทร์', 'เขตนวลจันทร์', 'เขตนวลจันทร์', 'เขตนวลจันทร์', 'เขตนวลจันทร์'],
        'Age (Years)': [5, 18, 12, 22, 8],
        'Web_Trip_Count': [0, 0, 0, 0, 0], # สถิติจากไฟล์ Feeder
        'Status': ['🟢 NORMAL'] * 5
    })

# --- 2. ส่วนแถบเมนูข้าง (Data Integration) ---
st.sidebar.header("📥 ศูนย์นำเข้าข้อมูล (Data Hub)")

# 2.1 อัปโหลดไฟล์สถิติ Feeder (จากเว็บหน่วยงาน)
st.sidebar.subheader("🌐 สถิติ Reliability (จากเว็บ)")
feeder_file = st.sidebar.file_uploader("อัปโหลดไฟล์ Feeder Indices (CSV)", type=["csv"])

if feeder_file:
    try:
        # อ่านไฟล์โดยข้ามหัว 2 บรรทัดแรกตามโครงสร้างไฟล์ MEA ที่คุณส่งมา
        df_feeder = pd.read_csv(feeder_file, skiprows=2)
        # นับจำนวนครั้งที่ Feeder เกิดปัญหา (Trip)
        trip_counts = df_feeder['Feeder'].value_counts().to_dict()
        
        # อัปเดตข้อมูลในระบบอัตโนมัติ
        for fid, count in trip_counts.items():
            st.session_state.asset_df.loc[st.session_state.asset_df['Feeder'] == fid, 'Web_Trip_Count'] = count
        st.sidebar.success("✅ อัปเดตสถิติไฟดับสำเร็จ")
    except:
        st.sidebar.error("❌ ไฟล์ไม่ตรงตามรูปแบบหน่วยงาน")

st.sidebar.divider()

# 2.2 บันทึกข้อมูลสำรวจ & รูปภาพ Acoustic
st.sidebar.subheader("📸 บันทึกผลสำรวจ & ภาพ Acoustic")
target_id = st.sidebar.selectbox("เลือกหม้อแปลงที่จะอัปเดต:", st.session_state.asset_df['Transformer_ID'])

# ช่องอัปโหลดรูปภาพ
acoustic_img = st.sidebar.file_uploader("อัปโหลดรูปจาก Acoustic Camera", type=["jpg", "png", "jpeg"])
if acoustic_img:
    st.sidebar.image(acoustic_img, caption="Preview ภาพหน้างาน", use_column_width=True)

# ช่องกรอกข้อมูล (เลิกใช้ Slider ตามคำขอ)
ac_db = st.sidebar.number_input("ความดัง Acoustic (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("Peak Frequency (Hz)", 1000, 100000, 20000)
s_temp = st.sidebar.number_input("อุณหภูมิ (°C)", 20.0, 120.0, 50.0)
s_load = st.sidebar.number_input("ภาระไฟฟ้า Load (%)", 0.0, 150.0, 70.0)

if st.sidebar.button("💾 วิเคราะห์ AI และบันทึกผล"):
    asset_info = st.session_state.asset_df[st.session_state.asset_df['Transformer_ID'] == target_id].iloc[0]
    
    # ส่ง 8 ตัวแปรให้ AI [Temp, Load, Oil, Vib, Age, Hum, dB, Hz]
    features = np.array([[s_temp, s_load, 95.0, 1.5, asset_info['Age (Years)'], 55.0, ac_db, ac_hz]])
    prob = model.predict_proba(features)[0][1]
    
    # คำนวณสถานะ (รวมสถิติ Trip เข้าไปใน Logic)
    trips = asset_info['Web_Trip_Count']
    if prob > 0.75 or trips >= 5:
        new_status = "🔴 CRITICAL"
    elif prob > 0.4 or trips >= 2:
        new_status = "🟡 WATCH"
    else:
        new_status = "🟢 NORMAL"
    
    st.session_state.asset_df.loc[st.session_state.asset_df['Transformer_ID'] == target_id, 'Status'] = new_status
    st.sidebar.success(f"บันทึกข้อมูล {target_id} เรียบร้อย")

# --- 3. หน้าจอหลัก (Dashboard Central) ---
st.title("🏙️ MEA Asset Intelligence Dashboard")
st.write(f"ข้อมูลเขตพื้นที่นวลจันทร์ ณ วันที่ {datetime.now().strftime('%d/%m/%Y')}")

# สรุปภาพรวมด้วย Metrics
c1, c2, c3 = st.columns(3)
c1.metric("หม้อแปลงทั้งหมด", len(st.session_state.asset_df))
c2.metric("สถานะวิกฤต (PM ด่วน)", len(st.session_state.asset_df[st.session_state.asset_df['Status'] == "🔴 CRITICAL"]))
c3.metric("Feeder ที่เสถียรที่สุด", "F12-02")

# ตารางหลัก (สถานะอุปกรณ์รายพื้นที่)
st.subheader("📊 ตารางติดตามสถานะและแผน PM รายอุปกรณ์")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.asset_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

# กราฟวิเคราะห์
st.divider()
col_left, col_right = st.columns(2)
with col_left:
    st.write("### สัดส่วนความเสี่ยงในพื้นที่")
    fig_pie = px.pie(st.session_state.asset_df, names='Status', color='Status',
                     color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 NORMAL':'#28a745'})
    st.plotly_chart(fig_pie, use_column_width=True)

with col_right:
    st.write("### สถิติไฟดับเทียบกับสถานะสุขภาพ (แยกตาม Feeder)")
    fig_bar = px.bar(st.session_state.asset_df, x='Feeder', y='Web_Trip_Count', color='Status',
                     color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 NORMAL':'#28a745'})
    st.plotly_chart(fig_bar, use_column_width=True)
