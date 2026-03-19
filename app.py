import streamlit as st
import joblib
import numpy as np
import time

# 1. โหลดโมเดล
@st.cache_resource
def load_my_model():
    return joblib.load('pdm_model.pkl')

model = load_my_model()

st.set_page_config(page_title="MEA Real-time AI Monitoring", layout="wide")
st.title("⚡ MEA Smart Incident AI (MSIA) - Real-time Monitoring")

# 2. ส่วนจำลองสถานะ (Auto-running)
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("▶️ Start Monitoring"):
        st.session_state.running = True
    if st.button("⏹️ Stop"):
        st.session_state.running = False

# 3. พื้นที่แสดงผล Dashboard
placeholder = st.empty()

while st.session_state.running:
    with placeholder.container():
        # จำลองค่า Sensor ที่ไหลเข้ามา (สุ่มค่าในช่วงปกติและเสี่ยง)
        temp = np.random.uniform(40, 95)
        load = np.random.uniform(50, 110)
        oil = np.random.uniform(60, 100)
        vibration = np.random.uniform(1.0, 5.0)

        # AI ประมวลผลทันที
        features = np.array([[temp, load, oil, vibration]])
        prob = model.predict_proba(features)[0][1]
        
        # แสดงผล Dashboard
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temp", f"{temp:.1f} °C", delta=f"{temp-60:.1f}", delta_color="inverse")
        c2.metric("Load", f"{load:.1f} %")
        c3.metric("Oil", f"{oil:.1f} %")
        c4.metric("Vibration", f"{vibration:.2f} mm/s")

        # 4. ระบบ Warning อัตโนมัติ
        if prob > 0.8:
            st.error(f"🚨 CRITICAL WARNING: ตรวจพบความเสี่ยงสูง {prob*100:.1f}%")
            st.toast("ส่งสัญญาณแจ้งเตือนไปยังทีมช่างหน้างานแล้ว!", icon="📢")
            # ในงานจริงตรงนี้จะใส่คำสั่งส่ง LINE Notify หรือ Email
        elif prob > 0.5:
            st.warning(f"⚠️ ATTENTION: เฝ้าระวังเป็นพิเศษ ({prob*100:.1f}%)")
        else:
            st.success("✅ System Status: Normal")

        st.progress(prob)
        time.sleep(3) # รอ 3 วินาทีแล้วดึงค่าใหม่ (Simulate Real-time)
