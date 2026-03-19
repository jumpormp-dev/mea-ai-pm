import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- ส่วนโหลดโมเดล ---
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl')

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_pm_ai_model.pkl' บน GitHub")

st.set_page_config(page_title="MEA Smart PM Planner", layout="wide")
st.title("⚡ MEA Smart Maintenance & PM Planner")
st.write("ระบบวางแผนบำรุงรักษาอัจฉริยะ (Asset Database + Web Stats + Acoustic Survey)")

# --- 1. การดึงข้อมูลจากเว็บภายนอก (Simulation 10.99.1.36) ---
st.sidebar.header("📂 Data Integration")
if st.sidebar.button("🔗 ดึงข้อมูลสถิติจาก 10.99.1.36"):
    # ใน LAN จริงจะใช้ pd.read_html("http://10.99.1.36/...")
    st.session_state.web_trips = 4.5 # สมมติค่าจากเว็บ
    st.sidebar.success(f"ดึงข้อมูลสำเร็จ: {st.session_state.web_trips} ครั้ง/ปี")

# --- 2. ข้อมูลจากการสำรวจหน้างาน (Survey) ---
st.sidebar.subheader("🔍 Field Survey Input")
s_id = st.sidebar.text_input("รหัสหม้อแปลง", "TR-MEA-001")
f1 = st.sidebar.slider("1. Temp (°C)", 30, 110, 60)
f2 = st.sidebar.slider("2. Load (%)", 0, 150, 80)
f3 = st.sidebar.slider("3. Oil Level (%)", 0, 100, 95)
f4 = st.sidebar.number_input("4. Vibration (mm/s)", 0.0, 10.0, 1.5)
f5 = st.sidebar.number_input("5. Age (Years)", 1, 40, 15)
f6 = st.sidebar.slider("6. Humidity (%)", 0, 100, 55)
f7 = st.sidebar.number_input("7. Acoustic (dB)", 30.0, 110.0, 45.0)
f8 = st.sidebar.number_input("8. Peak Freq (Hz)", 1000, 100000, 20000)

# --- 3. ประมวลผลและวางแผน ---
if st.sidebar.button("🤖 AI วิเคราะห์แผน PM"):
    input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])
    prob = model.predict_proba(input_data)[0][1]
    
    st.header(f"📊 ผลการประเมินสุขภาพ: {s_id}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score (AI)", f"{prob*100:.1f}%")
    c2.metric("สถิติไฟดับในพื้นที่", f"{st.session_state.get('web_trips', 'N/A')} ครั้ง/ปี")
    status = "🔴 CRITICAL" if prob > 0.7 else "🟡 WATCH" if prob > 0.4 else "🟢 STABLE"
    c3.subheader(status)

    # กราฟ Radar
    radar_df = pd.DataFrame({
        'Factor': ['ความร้อน', 'โหลด', 'อายุอุปกรณ์', 'สถิติไฟดับ', 'เสียง Acoustic'],
        'Score': [f1/110, f2/150, f5/40, st.session_state.get('web_trips', 2)/10, f7/110]
    })
    fig = px.line_polar(radar_df, r='Score', theta='Factor', line_close=True)
    fig.update_traces(fill='toself', fillcolor='rgba(255, 0, 0, 0.2)')
    st.plotly_chart(fig)

    # สรุปแผนงาน PM
    st.divider()
    st.subheader("📝 แผนการบำรุงรักษาที่แนะนำ (Action Plan)")
    if status == "🔴 CRITICAL":
        st.error(f"**เหตุผล:** พบสัญญาณเสียง Acoustic ผิดปกติ ({f7} dB) ร่วมกับความร้อนสูง")
        st.markdown("- **Action:** ดำเนินการ PM ด่วนที่สุดภายใน 72 ชั่วโมง")
    else:
        st.success("**Action:** ดำเนินการตามรอบปกติ (Annual Maintenance)")

else:
    st.info("💡 กรุณากรอกข้อมูลสำรวจและกดปุ่มวิเคราะห์ด้านซ้ายเพื่อเริ่มงาน")
