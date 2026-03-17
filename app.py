import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. โหลดโมเดล (ต้องมั่นใจว่าไฟล์ pkl อยู่ในโฟลเดอร์เดียวกันบน GitHub)
@st.cache_resource # ช่วยให้โหลดโมเดลเร็วขึ้น
def load_my_model():
    return joblib.load('pdm_model.pkl')

model = load_my_model()

# 2. ตั้งค่าหน้าตาเว็บ
st.set_page_config(page_title="AI PM Prototype", page_icon="⚡")
st.title("⚡ AI Predictive Maintenance Prototype")
st.write("ระบบพยากรณ์ความเสี่ยงอุปกรณ์ไฟฟ้า (MEA Case Study)")

# 3. ส่วนรับข้อมูล (Sidebar)
st.sidebar.header("Input Sensor Data")
temp = st.sidebar.slider("อุณหภูมิ (°C)", 30.0, 120.0, 50.0)
load = st.sidebar.slider("โหลด (%)", 0, 120, 60)
oil = st.sidebar.slider("ระดับน้ำมัน (%)", 0, 100, 90)
vibration = st.sidebar.slider("ความสั่นสะเทือน (mm/s)", 0.0, 10.0, 1.5)

# 4. ปุ่มคำนวณ
if st.button("วิเคราะห์ความเสี่ยง"):
    # ทำข้อมูลให้เป็นตารางเหมือนตอนเทรน
    input_df = pd.DataFrame([[temp, load, oil, vibration]], 
                            columns=['temp', 'load_pct', 'oil_level', 'vibration'])
    
    # พยากรณ์
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # 5. แสดงผลลัพธ์
    st.subheader("ผลการวิเคราะห์:")
    if prediction == 1:
        st.error(f"⚠️ สถานะ: อันตราย! พบความเสี่ยงสูง (โอกาสเสีย {prob*100:.2f}%)")
        st.write("คำแนะนำ: ควรส่งช่างตรวจเช็คอุปกรณ์ทันที")
    else:
        st.success(f"✅ สถานะ: ปกติ (โอกาสเสีย {prob*100:.2f}%)")
        st.write("คำแนะนำ: บำรุงรักษาตามรอบปกติ")

    # โชว์กราฟ Gauge หรือความสำคัญ (Optional)
    st.progress(prob)