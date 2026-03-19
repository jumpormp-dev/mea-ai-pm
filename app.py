import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px # ต้องเพิ่ม library นี้ใน requirements.txt

# 1. โหลดโมเดล
@st.cache_resource
def load_my_model():
    return joblib.load('pdm_model.pkl')

model = load_my_model()

st.set_page_config(page_title="MEA AI-Powered Predictive Maintenance", layout="wide")
st.title("⚡ MEA Smart maintenance AI (MSIA)")
st.write(f"ข้อมูล ณ วันที่: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# 2. จำลองรายการอุปกรณ์ (Asset Inventory)
assets = [f"Transformer-{i:03d}" for i in range(1, 11)] # จำลอง 10 เครื่อง

# 3. สร้างข้อมูลจำลองสำหรับแต่ละอุปกรณ์
data_list = []
for asset in assets:
    temp = np.random.uniform(45, 95)
    load = np.random.uniform(50, 110)
    oil = np.random.uniform(65, 100)
    vibration = np.random.uniform(1.2, 5.0)
    
    # ส่งให้ AI วิเคราะห์
    features = np.array([[temp, load, oil, vibration]])
    prob = model.predict_proba(features)[0][1]
    
    # วิเคราะห์ปัจจัยหลัก (Root Cause Analysis - Simple Logic)
    factors = []
    if temp > 80: factors.append("High Temp")
    if load > 100: factors.append("Overload")
    if oil < 75: factors.append("Low Oil")
    if vibration > 4.0: factors.append("Vibration")
    factor_str = ", ".join(factors) if factors else "Normal"

    # พยากรณ์ช่วงเวลาชำรุด (Estimate Time to Failure)
    days_to_fail = int(max(0, (1 - prob) * 30)) # ยิ่งเสี่ยงสูง จำนวนวันยิ่งน้อย
    fail_date = (datetime.now() + timedelta(days=days_to_fail)).strftime('%d/%m/%Y')

    data_list.append({
        "Asset ID": asset,
        "Status": "🔴 Critical" if prob > 0.8 else "🟡 Warning" if prob > 0.5 else "🟢 Normal",
        "Risk Score": f"{prob*100:.1f}%",
        "Predict Fail Date": fail_date if prob > 0.5 else "-",
        "Main Factors": factor_str,
        "Temp": f"{temp:.1f}",
        "Load": f"{load:.1f}",
        "Oil": f"{oil:.1f}",
        "Vib": f"{vibration:.2f}"
    })

df_display = pd.DataFrame(data_list)

# 4. แสดงผลลัพธ์เป็นตาราง
def color_status(val):
    color = 'red' if 'Critical' in val else 'orange' if 'Warning' in val else 'green'
    return f'color: {color}; font-weight: bold'

st.subheader("📋 รายการตรวจสอบสุขภาพหม้อแปลงไฟฟ้า (MEA Area Overview)")
st.table(df_display.style.applymap(color_status, subset=['Status']))

# 5. ฟีเจอร์ AI Insights (ส่วนที่เพิ่มใหม่)
st.divider()
st.subheader("💡 AI Insights & Decision Support")

c1, c2 = st.columns(2)

with c1:
    # แสดง Feature Importance
    st.write("📊 ตัวแปรที่มีผลต่อการตัดสินใจของ AI (MEA Transformer Model)")
    # ดึงค่าความสำคัญจากโมเดลจริง
    importances = pd.Series(model.feature_importances_, index=['Temp', 'Load', 'Oil', 'Vibration'])
    fig = px.bar(importances, orientation='h', labels={'value': 'Importance score'})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # แสดงโอกาสเสียรายเดือน
    st.write("📅 พยากรณ์โอกาสการเกิดเหตุขัดข้อง (Future Probability)")
    # จำลองโอกาสเสียรายเดือน (ยิ่งเครื่อง Critical โอกาสเสียยิ่งเพิ่มเร็ว)
    critical_assets = df_display[df_display['Status'] == "🔴 Critical"]
    if not critical_assets.empty:
        selected_asset = st.selectbox("เลือกอุปกรณ์ที่เสี่ยงสูง:", critical_assets['Asset ID'].tolist())
        months = ["Month-1", "Month-2", "Month-3", "Month-4", "Month-5", "Month-6"]
        prob_values = np.linspace(0.85, 0.99, 6) # สมมติว่าโอกาสเสียเพิ่มขึ้นเรื่อยๆ
        fig = px.line(x=months, y=prob_values, labels={'x': 'Month', 'y': 'Fail Probability'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("✅ ไม่พบอุปกรณ์เสี่ยงสูงในขณะนี้")

# 6. สรุปภาพรวม (Summary Cards)
st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("จำนวนอุปกรณ์ทั้งหมด", len(assets))
c2.metric("เสี่ยงสูง (Critical)", len(df_display[df_display['Status'] == "🔴 Critical"]))
c3.metric("ต้องเฝ้าระวัง (Warning)", len(df_display[df_display['Status'] == "🟡 Warning"]))

if st.button("🔄 Refresh Data"):
    st.rerun()
