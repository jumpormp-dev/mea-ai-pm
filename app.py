import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. โหลดโมเดล SPP-AI ---
@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดลบน GitHub")

st.set_page_config(page_title="SPP-AI: Advanced Analysis - KTD", layout="wide")

# --- 2. ฐานข้อมูลเริ่มต้น (20 Units KTD) ---
if 'ktd_assets' not in st.session_state:
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
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Acoustic_dB': [45.0]*20,
        'Peak_Hz': [20000]*20
    })

# --- 3. Sidebar (ข้อมูลนำเข้า) ---
st.sidebar.header("📥 Data Source: KTD Klong Toei")

if st.sidebar.button("📡 Sync Smart Meter (20 Units)"):
    st.session_state.ktd_assets['Temp_Meter'] = np.random.uniform(50, 95, 20)
    st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 120, 20)
    st.sidebar.success("✅ Sync ข้อมูล KTD สำเร็จ")

feeder_file = st.sidebar.file_uploader("อัปโหลดสถิติไฟดับ ฟขต. (.xlsx)", type=["xlsx"])
if feeder_file:
    df_f = pd.read_excel(feeder_file, skiprows=2)
    counts = df_f['Feeder'].value_counts().to_dict()
    for fid, c in counts.items():
        st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
    st.session_state.last_feeder_data = df_f # เก็บข้อมูลไว้ทำวิเคราะห์สาเหตุ
    st.sidebar.success("✅ อัปเดตสถิติรายฟีดเดอร์แล้ว")

st.sidebar.divider()
st.sidebar.subheader("📸 บันทึกผลสำรวจรายเครื่อง")
target = st.sidebar.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
ac_db_in = st.sidebar.number_input("ความดัง (dB)", 30.0, 120.0, 45.0)
ac_hz_in = st.sidebar.number_input("ความถี่ (Hz)", 1000, 100000, 20000)

if st.sidebar.button("🤖 รัน AI วิเคราะห์"):
    idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
    row = st.session_state.ktd_assets.iloc[idx]
    
    features = np.array([[row['Temp_Meter'], row['Load_Meter'], 230.0, ac_db_in, ac_hz_in, row['Trips_KTD'], 15, 55]])
    prob = model.predict_proba(features)[0][1]
    
    status = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
    st.session_state.ktd_assets.at[idx, 'Status'] = status
    st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
    st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_db_in
    st.session_state.ktd_assets.at[idx, 'Peak_Hz'] = ac_hz_in
    st.sidebar.success(f"วิเคราะห์ {target} เรียบร้อย")

# --- 4. Dashboard หน้าหลัก ---
st.title("🏙️ SPP-AI Dashboard: KTD Smart City")
st.write("ระบบวิเคราะห์สุขภาพหม้อแปลงเชิงลึก เขตคลองเตย")

# ตาราง Monitor
st.subheader("📋 สถานะหม้อแปลง KTD (Real-time Monitor)")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'
st.dataframe(st.session_state.ktd_assets.style.applymap(color_status, subset=['Status']), use_container_width=True)

st.divider()

# --- 5. ส่วนวิเคราะห์เจาะลึก (Deep Diagnostic) ---
st.subheader("🔍 บทวิเคราะห์ความเสี่ยงรายอุปกรณ์ (Diagnostic Insight)")
selected = st.selectbox("🎯 เลือกหม้อแปลงเพื่อดูรายงานฉบับเต็ม:", st.session_state.ktd_assets['Transformer_ID'])
res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == selected].iloc[0]

diag1, diag2, diag3 = st.columns([1, 1, 1])

with diag1:
    st.write("#### 📡 สัดส่วนปัจจัยเสี่ยง (Risk Radar)")
    # สร้าง Radar Chart เปรียบเทียบปัจจัยต่างๆ
    categories = ['Temp', 'Load', 'Acoustic', 'Trips', 'Age']
    # Normalize ค่าให้เป็น 0-1 สำหรับกราฟ
    values = [res['Temp_Meter']/100, res['Load_Meter']/120, res['Acoustic_dB']/100, res['Trips_KTD']/10, 0.5]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Risk Profile', line_color='red'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with diag2:
    st.write("#### 📑 รายละเอียดการวิเคราะห์")
    if res['Status'] == "🟢 NORMAL":
        st.success("✅ อุปกรณ์ทำงานปกติ")
    else:
        st.write("**⚠️ ปัจจัยที่ส่งผลกระทบสูง:**")
        if res['Temp_Meter'] > 85: st.error(f"- ความร้อนสูงเกินเกณฑ์ ({res['Temp_Meter']}°C)")
        if res['Acoustic_dB'] > 70: st.error(f"- ตรวจพบสัญญาณคลื่นเสียงผิดปกติ ({res['Acoustic_dB']} dB)")
        if res['Trips_KTD'] >= 5: st.warning(f"- ประวัติการทริปในฟีดเดอร์ {res['Feeder']} สูงสะสม")

    # วิเคราะห์สาเหตุหลักจากไฟล์ Excel
    if 'last_feeder_data' in st.session_state:
        df_f = st.session_state.last_feeder_data
        feeder_causes = df_f[df_f['Feeder'] == res['Feeder']]['Main Causes'].value_counts()
        if not feeder_causes.empty:
            st.write(f"**ประวัติสาเหตุในฟีดเดอร์ {res['Feeder']}:**")
            st.caption(f"ส่วนใหญ่เกิดจาก: {feeder_causes.idxmax()}")

with diag3:
    st.write("#### 📅 แผนการเข้าดำเนินการ")
    days = int(max(2, (1 - res['Risk_Score']) * 90))
    pm_date = (datetime.now() + timedelta(days=days)).strftime('%d/%m/%Y')
    
    st.metric("Risk Probability", f"{res['Risk_Score']*100:.1f}%")
    st.metric("Recommended PM Date", pm_date)
    
    st.write("**คำแนะนำทางเทคนิค:**")
    if res['Risk_Score'] > 0.75:
        st.write("🚨 **ด่วน:** ตรวจสอบจุดต่อด้วย Thermo Scan และทำความสะอาด Bushing ทันที")
    elif res['Risk_Score'] > 0.4:
        st.write("⏳ **เฝ้าระวัง:** บรรจุในแผน PM ประจำเดือน ตรวจวัดระดับน้ำมันและค่าความเป็นฉนวน")
    else:
        st.write("📅 **ปกติ:** ดำเนินการตามแผนบำรุงรักษาประจำปี")
