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
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_spp_ai_model.pkl' กรุณาอัปโหลดบน GitHub")

st.set_page_config(page_title="SPP-AI: Smart Plan Predictive - KTD", layout="wide")

# --- 2. ฐานข้อมูลหม้อแปลง 20 ตัว (ฟขต. / KTD) ---
if 'ktd_assets' not in st.session_state:
    ktd_feeders = ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 
                   'SAM-13', 'RPR-423', 'EM-418', 'PI-435', 'NS-436',
                   'SA-411', 'LN-442', 'SAM-13', 'RPR-423', 'EM-418',
                   'PI-435', 'NS-436', 'SA-411', 'LN-442', 'SAM-13']
    
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': ktd_feeders,
        'Location': ['คลองเตย'] * 20,
        'Temp_Meter': [0.0]*20,
        'Load_Meter': [0.0]*20,
        'Trips_KTD': [0]*20,
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Acoustic_dB': [45.0]*20,
        'Peak_Hz': [20000]*20
    })

# --- 3. Sidebar: จัดการข้อมูล ---
st.sidebar.header("📥 Data Source: KTD Dashboard")

if st.sidebar.button("📡 Sync Smart Meter (20 Units)"):
    # จำลองการดึงข้อมูลจาก IP 172.16.111.184
    st.session_state.ktd_assets['Temp_Meter'] = np.random.uniform(50, 95, 20)
    st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 120, 20)
    st.sidebar.success("✅ Sync ข้อมูล KTD สำเร็จ")

feeder_file = st.sidebar.file_uploader("อัปโหลดสถิติไฟดับ ฟขต. (.xlsx)", type=["xlsx"])
if feeder_file:
    df_f = pd.read_excel(feeder_file, skiprows=2)
    counts = df_f['Feeder'].value_counts().to_dict()
    for fid, c in counts.items():
        st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
    st.session_state.raw_feeder_df = df_f # เก็บไว้เพื่อวิเคราะห์สาเหตุ (Main Causes)
    st.sidebar.success("✅ อัปเดตสถิติรายฟีดเดอร์แล้ว")

st.sidebar.divider()
st.sidebar.subheader("📸 บันทึกผลสำรวจรายเครื่อง")
target = st.sidebar.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
ac_db_val = st.sidebar.number_input("ความดัง (dB)", 30.0, 120.0, 45.0)
ac_hz_val = st.sidebar.number_input("ความถี่ (Hz)", 1000, 100000, 20000)

if st.sidebar.button("🤖 รัน AI วิเคราะห์เชิงลึก"):
    idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
    row = st.session_state.ktd_assets.iloc[idx]
    
    # AI Input (8 Features)
    features = np.array([[row['Temp_Meter'], row['Load_Meter'], 230.0, ac_db_val, ac_hz_val, row['Trips_KTD'], 15, 55]])
    prob = model.predict_proba(features)[0][1]
    
    # Logic วิเคราะห์สถานะ
    status = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
    st.session_state.ktd_assets.at[idx, 'Status'] = status
    st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
    st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_db_val
    st.session_state.ktd_assets.at[idx, 'Peak_Hz'] = ac_hz_val
    st.sidebar.success(f"วิเคราะห์ {target} สำเร็จ!")

# --- 4. หน้าจอหลัก Dashboard ---
st.title("🏙️ SPP-AI: Smart Plan Predictive")
st.write(f"ศูนย์บริหารจัดการหม้อแปลง Smart Meter เขตคลองเตย (KTD) - {datetime.now().strftime('%d/%m/%Y')}")

# ตารางติดตามสถานะรวม
st.subheader("📋 สถานะสุขภาพหม้อแปลงรายฟีดเดอร์ (KTD Assets)")
def color_status(val):
    color = '#ff4b4b' if 'CRITICAL' in val else '#ffa500' if 'WATCH' in val else '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold;'
st.dataframe(st.session_state.ktd_assets.style.applymap(color_status, subset=['Status']), use_container_width=True)

st.divider()

# --- 5. ระบบวิเคราะห์ความเสี่ยงเชิงลึก (Interactive Diagnostic) ---
st.subheader("🔍 บทวิเคราะห์และพยากรณ์รายอุปกรณ์ (Diagnostic Insight)")
selected = st.selectbox("เลือกหม้อแปลงที่ต้องการตรวจสอบรายละเอียด:", st.session_state.ktd_assets['Transformer_ID'])
res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == selected].iloc[0]

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.write("#### 📡 องค์ประกอบความเสี่ยง (Risk Radar)")
    # กราฟเรดาร์แสดงปัจจัยต่างๆ
    categories = ['Temp', 'Load', 'Acoustic', 'Trips', 'Age']
    vals = [res['Temp_Meter']/100, res['Load_Meter']/120, res['Acoustic_dB']/100, res['Trips_KTD']/10, 0.5]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself', name='Risk Profile', line_color='red'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.write("#### 📑 สรุปการวินิจฉัย (AI Diagnostic)")
    if res['Status'] == "🟢 NORMAL":
        st.success("✅ อุปกรณ์ทำงานปกติ: ไม่พบสิ่งบ่งชี้อันตราย")
    else:
        st.write("**⚠️ ปัจจัยกระตุ้นความเสี่ยง:**")
        if res['Risk_Score'] > 0.6: st.warning("- AI พบรูปแบบคลื่นเสียงและอุณหภูมิที่สัมพันธ์กับการชำรุด")
        if res['Trips_KTD'] >= 5: st.warning(f"- ฟีดเดอร์ {res['Feeder']} มีประวัติการขัดข้องสูงสะสม")
        
        # ดึงสาเหตุหลักจากไฟล์ ฟขต.
        if 'raw_feeder_df' in st.session_state:
            df_f = st.session_state.raw_feeder_df
            causes = df_f[df_f['Feeder'] == res['Feeder']]['Main Causes'].value_counts()
            if not causes.empty:
                st.info(f"💡 **ข้อมูลพื้นที่:** ฟีดเดอร์นี้มักมีปัญหาจาก '{causes.idxmax()}'")

with c3:
    st.write("#### 📅 แผนงานบำรุงรักษา (PM Schedule)")
    days = int(max(2, (1 - res['Risk_Score']) * 90))
    pm_date = (datetime.now() + timedelta(days=days)).strftime('%d/%m/%Y')
    
    st.metric("ดัชนีความเสี่ยง (AI)", f"{res['Risk_Score']*100:.1f}%")
    st.metric("กำหนด PM ล่วงหน้า", pm_date)
    
    st.write("**คำแนะนำงาน:**")
    if res['Status'] == "🔴 CRITICAL":
        st.error("🚨 จัดส่งทีมช่างตรวจสอบจุดร้อนและเสียงทันทีภายใน 7 วัน")
    elif res['Status'] == "🟡 WATCH":
        st.warning("⏳ บรรจุลงแผน PM ประจำเดือน และเฝ้าระวังภาระไฟฟ้า")
    else:
        st.write("📅 ตรวจสอบตามรอบบำรุงรักษาประจำปี")
