import streamlit as st
import joblib
import pandas as pd
import numpy as np
import cv2 # สำหรับประมวลผลภาพ
import pytesseract # สำหรับอ่านตัวเลขจากภาพ (OCR)
from datetime import datetime

# 1. โหลดโมเดล AI (mea_pm_ai_model.pkl ที่เทรนมาใหม่ด้วยค่า Acoustic)
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl')

# model = load_my_model() # เปิดใช้งานเมื่อมีไฟล์โมเดลจริง

st.set_page_config(page_title="MEA Smart Diagnostic", layout="wide")
st.title("⚡ MEA Smart maintenance AI (MSIA)")
st.subheader("ระบบวิเคราะห์แผน PM อุปกรณ์ไฟฟ้าด้วยสถิติและภาพถ่าย Acoustic")

# 2. ส่วนแถบเมนูข้าง (Input)
st.sidebar.header("📥 นำเข้าข้อมูลเพื่อวิเคราะห์")
transformer_id = st.sidebar.text_input("รหัสหม้อแปลง (Transformer ID)", "TR-MEA-XXXX")

# 2.1 ส่วนข้อมูลสถิติ (ดึงจากฐานข้อมูล MEA)
st.sidebar.subheader("📊 ข้อมูลสถิติ (MEA Database)")
hist_failure = st.sidebar.slider("สถิติการขัดข้องในอดีต (ครั้ง/ปี)", 0, 10, 2)
age = st.sidebar.slider("อายุการใช้งาน (ปี)", 0, 30, 10)

# 2.2 ส่วนข้อมูล Acoustic Camera (ไฮไลท์)
st.sidebar.subheader("🔊 ข้อมูล Acoustic Camera")
uploaded_image = st.sidebar.file_uploader("อัปโหลดภาพถ่ายหน้าจอกล้อง Acoustic (dB/Hz)", type=["jpg", "png", "jpeg"])

# ฟังก์ชัน AI อ่านตัวเลขจากภาพ (OCR)
def read_acoustic_data(image_file):
    # เปลี่ยนไฟล์อัปโหลดเป็นภาพ OpenCV
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # ประมวลผลภาพให้เป็นขาวดำเพื่อให้ OCR อ่านง่ายขึ้น
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # ใช้ Tesseract OCR อ่านข้อความ
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    
    # (Simplified) หาตัวเลข dB และ Hz จากข้อความ
    # ในงานจริงต้องใช้ Regex เพื่อดึงเฉพาะตัวเลขที่ต้องการ
    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)
    st.write("🔍 **AI กำลังวิเคราะห์ภาพ...**")
    st.text(f"ข้อความที่ AI อ่านได้ (Raw Text):\n{text}") # โชว์เพื่อตรวจสอบ
    
    # จำลองค่าที่ได้จากการอ่านจริง (ใน PoC นี้)
    decibel = 85.5 
    frequency = 45000 
    return decibel, frequency

# ตรวจสอบการอัปโหลดภาพ
acoustic_db = 0
peak_freq = 0

if uploaded_image is not None:
    # เรียกใช้ AI อ่านภาพ
    # acoustic_db, peak_freq = read_acoustic_data(uploaded_image) # เปิดใช้งานเมื่อติดตั้ง Tesseract เรียบร้อย
    # จำลองค่าเพื่อโชว์ UI
    acoustic_db = 88.2
    peak_freq = 42500
    st.sidebar.success(f"✅ AI อ่านค่า dB: {acoustic_db}, Freq: {peak_freq} Hz")
else:
    # ถ้าไม่อัปโหลดภาพ ให้กรอกเองได้
    acoustic_db = st.sidebar.number_input("หรือกรอกค่า dB เอง", 30.0, 120.0, 60.0)
    peak_freq = st.sidebar.number_input("หรือกรอกค่า Freq (Hz) เอง", 1000, 100000, 20000)

# 3. ปุ่มวิเคราะห์
analyze_btn = st.sidebar.button("🤖 AI วิเคราะห์แผน PM")

# 4. ส่วนแสดงผลหลัก (Dashboard)
if analyze_btn:
    # เตรียมข้อมูลส่งให้ AI
    input_data = [[acoustic_db, peak_freq, hist_failure, age, 0]]
    
    # AI พยากรณ์ (จำลองค่า PoC)
    # priority = model.predict(input_data)[0]
    # prob = model.predict_proba(input_data)[0]
    priority = 2 # 0=Stable, 1=Watch, 2=Critical
    prob = [0.05, 0.15, 0.80]

    # แสดงผลระดับความสำคัญ (Integrated Health Score)
    st.header("📊 ผลการวิเคราะห์และแนะนำแผน PM")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    
    with c1:
        st.write("### ระดับความสำคัญในการทำ PM")
        if priority == 2:
            st.error("🆘 **URGENT: ต้องทำ PM ทันที (ภายใน 7 วัน)**")
            st.toast("ส่งสัญญาณแจ้งเตือนไปยังทีมช่างหน้างานแล้ว!", icon="🚨")
        elif priority == 1:
            st.warning("⚠️ **WATCH: เฝ้าระวังและวางแผน PM (ภายใน 1 เดือน)**")
        else:
            st.success("✅ **STABLE: สภาพปกติ (PM ตามรอบปกติ)**")
            
        st.write(f"**โอกาสเกิดเหตุขัดข้อง (AI Confidence):** {prob[priority]*100:.1f}%")
        st.progress(prob[priority])

    with c2:
        st.write("### ปัจจัยจาก Acoustic Camera")
        st.metric("Sound Intensity", f"{acoustic_db} dB", delta=f"{acoustic_db-60:.1f}", delta_color="inverse")
        st.metric("Peak Frequency", f"{peak_freq/1000:.1f} kHz")

    with c3:
        st.write("### ปัจจัยจากสถิติ MEA")
        st.metric("ประวัติการเสีย", f"{hist_failure} ครั้ง/ปี")
        st.metric("อายุอุปกรณ์", f"{age} ปี")

    # ส่วนคำแนะนำ Decision Support (XAI)
    st.divider()
    st.subheader("💡 คำแนะนำในการตัดสินใจ (AI Decision Support)")
    st.write(f"**รหัสทรัพย์สิน:** {transformer_id} | **วันที่วิเคราะห์:** {datetime.now().strftime('%d/%m/%Y')}")
    st.markdown(f"""
    **เหตุผลการวิเคราะห์ของ AI:**
    - ตรวจพบค่าความดังของเสียง **({acoustic_db} dB)** และความถี่ **({peak_freq/1000:.1f} kHz)** สูงผิดปกติ ซึ่งบ่งบอกถึงการเกิด Partial Discharge หรือ Arcing ภายใน
    - เมื่อรวมกับข้อมูลสถิติประวัติการเสีย **({hist_failure} ครั้ง/ปี)** ทำให้มีความเสี่ยงสูงที่จะเกิดเหตุขัดข้อง
    
    **ข้อแนะนำสำหรับพนักงาน MEA:**
    1. **ส่งทีมช่างหน้างาน:** เข้าตรวจสอบหม้อแปลงตัวนี้เป็นการเร่งด่วน
    2. **เตรียมอุปกรณ์:** เตรียมชุดตรวจ Partial Discharge หรือ DGA (Dissolved Gas Analysis) ไปหน้างาน
    3. **ปรับแผน PM:** เลื่อนกำหนดการทำ PM ของอุปกรณ์ตัวนี้ขึ้นมาให้เร็วที่สุด (Uptake)
    """)

else:
    st.info("💡 กรุณากรอกข้อมูลสถิติหรืออัปโหลดภาพจาก Acoustic Camera ทางแถบด้านซ้ายเพื่อเริ่มการวิเคราะห์")
    st.image("https://via.placeholder.com/1000x300.png?text=MEA+Acoustic+AI+Diagnostic+Dashboard", use_column_width=True)
