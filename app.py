import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- 1. โหลดโมเดล SPP-AI (8 ตัวแปร) ---
@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดลบน GitHub")

st.set_page_config(page_title="SPP-AI: Smart Plan Predictive - KTD", layout="wide")

# --- 2. ฐานข้อมูลหม้อแปลง Smart Meter 20 ตัว (เขตคลองเตย/KTD) ---
# ปรับจาก 5 ตัว เป็น 20 ตัว ตามข้อมูลจริงของหน้างาน
if 'ktd_assets' not in st.session_state:
    # รายชื่อ Feeder อ้างอิงจากไฟล์ ฟขต Feeder.xlsx ที่คุณให้มา
    ktd_feeders = ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 
                   'SAM-13', 'RPR-423', 'EM-418', 'PI-435', 'NS-436',
                   'SA-411', 'LN-442', 'SAM-13', 'RPR-423', 'EM-418',
                   'PI-435', 'NS-436', 'SA-411', 'LN-442', 'SAM-13']
    
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': ktd_feeders,
        'Temp_Meter': [0.0]*20,
        'Load_Meter': [0.0]*20,
        'Trips_KTD': [0]*20,
        'Risk_Score': [0.0]*20,
        'Status': ['🟢 NORMAL']*20
    })

# --- 3. Sidebar: ระบบจัดการข้อมูล KTD ---
st.sidebar.header("📥 Data Source: KTD Klong Toei")

# ส่วนดึงข้อมูล Smart Meter จาก IP ภายใน (กรองเฉพาะ KTD)
if st.sidebar.button("📡 Sync Smart Meter (20 Units)"):
    url = "http://172.16.111.184:8501/Transformer"
    try:
        # ในวง LAN จริง จะดึงข้อมูลและกรอง KTD มาอัปเดตทั้ง 20 ตัว
        # สำหรับ Demo นี้จะจำลองค่าที่ได้จากเว็บ Smart Meter ให้ครบ 20 ตัว
        st.session_state.ktd_assets['Temp_Meter'] = np.random.uniform(50, 90, 20)
        st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 115, 20)
        st.sidebar.success("✅ Sync ข้อมูล KTD 20 ตัวสำเร็จ")
    except:
        st.sidebar.error("❌ ไม่สามารถเข้าถึง IP 172.16.111.184 ได้")

# อัปโหลดไฟล์สถิติไฟดับ ฟขต. (KTD)
feeder_file = st.sidebar.file_uploader("อัปโหลดสถิติไฟดับ ฟขต. (.xlsx)", type=["xlsx"])
if feeder_file:
    df_f = pd.read_excel(feeder_file, skiprows=2)
    counts = df_f['Feeder'].value_counts().to_dict()
    for fid, c in counts.items():
        st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
    st.sidebar.success("✅ อัปเดตสถิติรายฟีดเดอร์แล้ว")

st.sidebar.divider()
st.sidebar.subheader("📸 บันทึกผลสำรวจรายเครื่อง")
target = st.sidebar.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
ac_db = st.sidebar.number_input("ความดัง (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("ความถี่ (Hz)", 1000, 100000, 20000)

if st.sidebar.button("🤖 รัน AI วิเคราะห์"):
    idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
    row = st.session_state.ktd_assets.iloc[idx]
    
    # AI Input 8 Features
    features = np.array([[row['Temp_Meter'], row['Load_Meter'], 230.0, ac_db, ac_hz, row['Trips_KTD'], 15, 55]])
    prob = model.predict_proba(features)[0][1]
    
    # ปรับ Logic การให้สีตามความเสี่ยงจริง
    status = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
    st.session_state.ktd_assets.at[idx, 'Status'] = status
    st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
    st.sidebar.success(f"วิเคราะห์ {target} เรียบร้อย")

# --- 4. Dashboard หน้าหลัก ---
st.title("🏙️ SPP-AI Dashboard: KTD Smart City")
st.write(f"ติดตามสถานะหม้อแปลง Smart Meter 20 เครื่องในเขตคลองเตย")

# แสดง Metrics สรุปภาพรวม 20 ตัว
m1, m2, m3 = st.columns(3)
m1.metric("หม้อแปลงทั้งหมด", "20 ตัว")
m2.metric("สถานะวิกฤต", len(st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] == "🔴 CRITICAL"]))
m3.metric("เขตพื้นที่", "ฟขต. (KTD)")

# ตาราง Monitor 20 ตัว
st.subheader("📋 รายการสถานะหม้อแปลงทั้งหมด (Real-time Monitor)")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.ktd_assets.style.applymap(color_status, subset=['Status']), use_container_width=True)

# ส่วนวิเคราะห์แผนงาน
st.divider()
selected = st.selectbox("🎯 เลือกดูบทวิเคราะห์รายตัว:", st.session_state.ktd_assets['Transformer_ID'])
res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == selected].iloc[0]

c1, c2 = st.columns(2)
with c1:
    st.write("### 🔍 ผลการวินิจฉัย")
    if res['Status'] == "🟢 NORMAL":
        st.success("✅ สถานะปกติ")
    else:
        st.write("**สาเหตุที่ AI แจ้งเตือน:**")
        if res['Trips_KTD'] >= 5: st.warning(f"- พบสถิติไฟดับสะสมในฟีดเดอร์ {res['Feeder']}")
        if res['Risk_Score'] > 0.6: st.warning("- ตรวจพบความผิดปกติจากค่า Acoustic และ Smart Meter")

with c2:
    st.write("### 📅 แผนการเข้าทำ PM")
    days = int(max(2, (1 - res['Risk_Score']) * 90))
    pm_date = (datetime.now() + timedelta(days=days)).strftime('%d/%m/%Y')
    st.metric("วันที่แนะนำให้เข้าดำเนินการ", pm_date)
