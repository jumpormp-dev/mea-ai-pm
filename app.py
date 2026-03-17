import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. โหลดโมเดล
@st.cache_resource
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
    # เตรียมข้อมูลเป็น Array (ตัดปัญหาเรื่องชื่อคอลัมน์ไม่ตรง)
    features = np.array([[temp, load, oil, vibration]])
    
    # พยากรณ์
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    # 5. แสดงผลลัพธ์
    st.subheader("ผลการวิเคราะห์:")
    if prediction == 1:
        st.error(f"⚠️ สถานะ: อันตราย! พบความเสี่ยงสูง (โอกาสเสีย {prob*100:.2f}%)")
        st.markdown("### **คำแนะนำ:** ควรส่งทีมช่างเข้าตรวจสอบอุปกรณ์ทันที")
    else:
        st.success(f"✅ สถานะ: ปกติ (โอกาสเสีย {prob*100:.2f}%)")
        st.markdown("### **คำแนะนำ:** บำรุงรักษาตามรอบปกติ")

    # แสดงแถบความเสี่ยง
    st.write("ระดับความเสี่ยงพิจารณาจาก AI:")
    st.progress(prob)
