import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- ส่วนโหลดโมเดล ---
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl') # ต้องเป็นโมเดล 8 ตัวแปร

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_pm_ai_model.pkl' บน GitHub")

st.set_page_config(page_title="MEA Maintenance Hub", layout="wide")
st.title("⚡ MEA Maintenance & PM Planning Center")

# --- การจัดการข้อมูล (Sidebar) ---
st.sidebar.header("📥 Data Integration Center")

# ส่วนที่ 1: อัปโหลดข้อมูลเดิม (จากกล้อง Acoustic หรือประวัติ Excel)
st.sidebar.subheader("1. อัปโหลดข้อมูลเดิม")
uploaded_acoustic = st.sidebar.file_uploader("อัปโหลดไฟล์จากกล้อง Acoustic (.xlsx)", type=["xlsx"])

# ส่วนที่ 2: ข้อมูลจากการสำรวจหน้างาน (Manual Survey)
st.sidebar.divider()
st.sidebar.subheader("2. ข้อมูลสำรวจหน้างานใหม่")
s_id = st.sidebar.text_input("รหัสหม้อแปลง", "TR-BK-001")
f1 = st.sidebar.slider("1. Temp (°C)", 30, 110, 60)
f2 = st.sidebar.slider("2. Load (%)", 0, 150, 80)
f3 = st.sidebar.slider("3. Oil Level (%)", 0, 100, 95)
f4 = st.sidebar.number_input("4. Vibration (mm/s)", 0.0, 10.0, 1.5)
f5 = st.sidebar.number_input("5. Age (Years)", 1, 40, 15)
f6 = st.sidebar.slider("6. Humidity (%)", 0, 100, 55)

# ส่วนที่ 3: รับค่าจากกล้อง Acoustic (ถ้าไม่มีการอัปโหลด)
if uploaded_acoustic is None:
    st.sidebar.subheader("🔊 ผลจากกล้อง Acoustic (กรอกเอง)")
    f7 = st.sidebar.number_input("7. Acoustic (dB)", 30.0, 110.0, 45.0)
    f8 = st.sidebar.number_input("8. Peak Freq (Hz)", 1000, 100000, 20000)
else:
    # ถ้ามีการอัปโหลด ให้ AI ลองอ่านค่าdB/Hz จากไฟล์ (Simplified สำหรับโปรโตไทป์)
    df_ac = pd.read_excel(uploaded_acoustic)
    st.sidebar.success("✅ อัปโหลดไฟล์จากกล้องสำเร็จ")
    # สมมติว่าไฟล์มีคอลัมน์ชื่อ 'dB_Level' และ 'Frequency'
    try:
        f7 = df_ac['dB_Level'].mean()
        f8 = df_ac['Frequency'].mean()
        st.sidebar.info(f"AI อ่านค่า dB เฉลี่ย: {f7:.1f}, Freq: {f8:.1f} Hz")
    except:
        st.sidebar.warning("⚠️ ไฟล์ไม่มีคอลัมน์ dB_Level/Frequency")
        f7 = 45.0
        f8 = 20000

# --- ส่วนประมวลผล ---
if st.sidebar.button("🤖 AI วิเคราะห์และวางแผน PM"):
    input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])
    prob = model.predict_proba(input_data)[0][1]
    
    st.header(f"📊 ผลการประเมินสุขภาพอุปกรณ์: {s_id}")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.metric("Risk Score (AI)", f"{prob*100:.1f}%")
        status = "🔴 CRITICAL" if prob > 0.75 else "🟡 WATCH" if prob > 0.4 else "🟢 NORMAL"
        st.subheader(f"สถานะ: {status}")
        
    with c2:
        # กราฟ Radar โชว์ที่มาของความเสี่ยง
        radar_df = pd.DataFrame({
            'ด้าน': ['ความร้อน', 'โหลด', 'อายุอุปกรณ์', 'เสียง Acoustic', 'ความสั่นสะเทือน'],
            'คะแนน': [f1/110, f2/150, f5/40, f7/110, f4/10]
        })
        fig = px.line_polar(radar_df, r='คะแนน', theta='ด้าน', line_close=True)
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

    # --- ส่วนที่ 4: แสดงประวัติจากไฟล์ (ถ้ามีการอัปโหลด) ---
    if uploaded_acoustic:
        st.divider()
        st.subheader("📜 ประวัติข้อมูลเดิมที่อัปโหลด")
        st.dataframe(df_ac.head(5), use_container_width=True)
