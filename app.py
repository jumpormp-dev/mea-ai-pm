import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- โหลดโมเดล SPP-AI ---
@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

model = load_spp_model()

st.set_page_config(page_title="SPP-AI: Smart Plan Predictive - KTD", layout="wide")

# --- ฟังก์ชันดึงข้อมูลจาก Smart Meter (KTD) ---
def fetch_ktd_meter_data():
    url = "http://172.16.111.184:8501/Transformer"
    try:
        # ดึงข้อมูลจาก IP ภายในและกรองเฉพาะเขต KTD
        tables = pd.read_html(url)
        df_all = tables[0]
        # กรองเฉพาะแถวที่ District หรือ Area ระบุว่าเป็น KTD
        df_ktd = df_all[df_all['District'] == 'KTD'] 
        return df_ktd
    except:
        return None

# --- ส่วนหัวข้อแอป ---
st.title("🏙️ SPP-AI: Smart Plan Predictive Maintenance")
st.subheader("ระบบวางแผนอัจฉริยะ เขตคลองเตย (ฟขต. / KTD)")

# --- 1. ฐานข้อมูลเริ่มต้น เขตคลองเตย ---
if 'ktd_assets' not in st.session_state:
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': ['TR-KTD-001', 'TR-KTD-002', 'TR-KTD-003', 'TR-KTD-004', 'TR-KTD-005'],
        'Feeder': ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442'],
        'Temp_Meter': [0.0]*5,
        'Load_Meter': [0.0]*5,
        'Trips_KTD': [0]*5,
        'Risk_Score': [0.0]*5,
        'Status': ['🟢 NORMAL']*5
    })

# --- 2. Sidebar: จัดการข้อมูลบูรณาการ ---
st.sidebar.header("📥 Data Source Integration")

# ส่วนที่ 1: ดึงข้อมูล Smart Meter KTD
if st.sidebar.button("📡 Sync KTD Smart Meter"):
    smart_df = fetch_ktd_meter_data()
    if smart_df is not None:
        st.sidebar.success("✅ เชื่อมต่อ KTD Meter สำเร็จ")
        # อัปเดตค่า Load/Temp ลงในระบบ (จำลองการ Mapping)
        st.session_state.ktd_assets['Temp_Meter'] = [64.2, 87.5, 53.0, 71.2, 59.0]
        st.session_state.ktd_assets['Load_Meter'] = [70, 112, 48, 92, 63]
    else:
        st.sidebar.warning("⚠️ ไม่พบ IP 172.16.111.184 (โปรดรันในวง LAN MEA)")

# ส่วนที่ 2: อัปโหลดสถิติไฟดับ ฟขต. (จากไฟล์ที่คุณให้มา)
feeder_file = st.sidebar.file_uploader("อัปโหลดสถิติไฟดับ ฟขต. (.xlsx)", type=["xlsx"])
if feeder_file:
    df_f = pd.read_excel(feeder_file, skiprows=2)
    counts = df_f['Feeder'].value_counts().to_dict()
    for fid, c in counts.items():
        st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
    st.sidebar.success("✅ อัปเดตสถิติไฟดับ KTD เรียบร้อย")

st.sidebar.divider()
st.sidebar.subheader("📸 บันทึกผลสำรวจ Acoustic")
target = st.sidebar.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
ac_db = st.sidebar.number_input("ความดัง (dB)", 30.0, 120.0, 45.0)
ac_hz = st.sidebar.number_input("ความถี่ (Hz)", 1000, 100000, 20000)

if st.sidebar.button("🤖 รัน SPP-AI Analysis"):
    idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
    row = st.session_state.ktd_assets.iloc[idx]
    
    # Input AI: [Temp, Load, Voltage(230), dB, Hz, Trips, Age(15), Hum(55)]
    features = np.array([[row['Temp_Meter'], row['Load_Meter'], 230.0, ac_db, ac_hz, row['Trips_KTD'], 15, 55]])
    prob = model.predict_proba(features)[0][1]
    
    new_stat = "🔴 CRITICAL" if (prob > 0.7 or row['Trips_KTD'] >= 10) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
    st.session_state.ktd_assets.at[idx, 'Status'] = new_stat
    st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
    st.sidebar.success(f"วิเคราะห์ {target} สำเร็จ")

# --- 3. Dashboard หน้าหลัก ---
st.write("### 📋 ตารางติดตามสถานะรวม เขตคลองเตย (KTD)")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'

st.dataframe(st.session_state.ktd_assets.style.applymap(color_status, subset=['Status']), use_container_width=True)

# --- 4. Interactive Analysis ---
st.divider()
selected = st.selectbox("🎯 คลิกเลือกอุปกรณ์เพื่อดูบทวิเคราะห์แผนงาน PM:", st.session_state.ktd_assets['Transformer_ID'])
res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == selected].iloc[0]

c_diag, c_plan = st.columns(2)
with c_diag:
    st.write("### 🔍 วิเคราะห์ที่มาความเสี่ยง (KTD Insights)")
    if res['Status'] == "🟢 NORMAL":
        st.success("✅ ปัจจัยทางเทคนิคและสถิติฟีดเดอร์ยังอยู่ในเกณฑ์ปกติ")
    else:
        st.write("**สาเหตุสำคัญ:**")
        if res['Trips_KTD'] >= 5: st.warning(f"- สถิติไฟดับในฟีดเดอร์ {res['Feeder']} สูงสะสม")
        if res['Temp_Meter'] > 80: st.warning(f"- อุณหภูมิจาก Smart Meter สูงกว่ามาตรฐาน")
        if res['Risk_Score'] > 0.6: st.warning("- AI ตรวจพบความเสี่ยงจากการวิเคราะห์คลื่นเสียง Acoustic")

with c_plan:
    st.write("### 📅 แผนการเข้าทำ PM เชิงพยากรณ์")
    days = int(max(2, (1 - res['Risk_Score']) * 90))
    pm_date = (datetime.now() + timedelta(days=days)).strftime('%d/%m/%Y')
    st.metric("กำหนดเข้าทำ PM ที่แนะนำ", pm_date)
    st.write(f"**ความเร่งด่วน:** {'เร่งด่วนสูงสุด' if res['Risk_Score'] > 0.7 else 'เข้าแผนประจำเดือน'}")
