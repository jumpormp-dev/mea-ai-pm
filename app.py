import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# 1. โหลดโมเดล
@st.cache_resource
def load_my_model():
    return joblib.load('pdm_model.pkl')

model = load_my_model()

st.set_page_config(page_title="MEA AI Asset Intelligence", layout="wide")
st.title("⚡ MEA Smart Maintenance AI (MSIA)")
st.subheader("ระบบบริหารจัดการสุขภาพทรัพย์สินด้วยปัญญาประดิษฐ์")

# 2. จำลองรายการอุปกรณ์
assets = [f"Transformer-{i:03d}" for i in range(1, 11)]

data_list = []
for asset in assets:
    # จำลองค่า Sensor แบบสุ่ม
    temp = np.random.uniform(45, 95)
    load = np.random.uniform(50, 115)
    oil = np.random.uniform(60, 100)
    vibration = np.random.uniform(1.2, 5.2)
    
    # AI พยากรณ์ (ใช้แบบ Array เพื่อความชัวร์เรื่องชื่อคอลัมน์)
    features = np.array([[temp, load, oil, vibration]])
    prob = model.predict_proba(features)[0][1]
    
    # ตรวจสอบสาเหตุที่ AI กังวล
    factors = []
    if temp > 82: factors.append("Critical Heat")
    if load > 100: factors.append("Overload")
    if oil < 70: factors.append("Oil Deterioration")
    if vibration > 4.2: factors.append("Mechanical Stress")
    factor_str = ", ".join(factors) if factors else "Healthy"

    # พยากรณ์วันชำรุดล่วงหน้า
    days_to_fail = int(max(0, (1 - prob) * 45)) 
    fail_date = (datetime.now() + timedelta(days=days_to_fail)).strftime('%d/%m/%Y')

    data_list.append({
        "Asset ID": asset,
        "Health Status": "🔴 CRITICAL" if prob > 0.8 else "🟡 WATCH" if prob > 0.5 else "🟢 STABLE",
        "Risk Score": f"{prob*100:.1f}%",
        "Est. Fail Date": fail_date if prob > 0.5 else "N/A",
        "Primary Risks": factor_str,
        "Temp": temp, "Load": load, "Oil": oil, "Vib": vibration
    })

df = pd.DataFrame(data_list)

# 3. ส่วนการแสดงผลตาราง
st.subheader("📊 Asset Health Inventory")
def color_status(val):
    if "CRITICAL" in val: color = '#ff4b4b'
    elif "WATCH" in val: color = '#ffa500'
    else: color = '#28a745'
    return f'background-color: {color}; color: white; font-weight: bold; border-radius: 5px;'

st.dataframe(df.style.applymap(color_status, subset=['Health Status']), use_container_width=True)

# 4. ส่วน AI Analytic Insights
st.divider()
st.subheader("🧠 AI Decision Analytics")
c1, c2 = st.columns(2)

with c1:
    st.write("**Feature Importance (Global Logic)**")
    # ดึงค่าจาก Model จริง
    importances = pd.Series(model.feature_importances_, index=['Temp', 'Load', 'Oil', 'Vib'])
    fig = px.bar(importances, orientation='h', color=importances, color_continuous_scale='RdYlGn_r')
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("กราฟแสดงน้ำหนักที่ AI ใช้ในการตัดสินใจพยากรณ์ความเสี่ยง")

with c2:
    st.write("**Risk Trend Projection**")
    # เลือกเครื่องที่เสี่ยงที่สุดมาโชว์กราฟพยากรณ์
    critical_asset = df.iloc[df['Risk Score'].str.replace('%','').astype(float).idxmax()]
    months = ["Current", "Next 1M", "Next 2M", "Next 3M"]
    current_risk = float(critical_asset['Risk Score'].replace('%','')) / 100
    risk_projection = [current_risk, min(0.99, current_risk*1.05), min(0.99, current_risk*1.15), min(0.99, current_risk*1.3)]
    
    fig_line = px.line(x=months, y=risk_projection, markers=True)
    fig_line.update_traces(line_color='#ff4b4b')
    fig_line.update_layout(yaxis_range=[0, 1], height=300)
    st.plotly_chart(fig_line, use_container_width=True)
    st.caption(f"การพยากรณ์แนวโน้มความเสี่ยงของ {critical_asset['Asset ID']} ใน 3 เดือนข้างหน้า")

if st.button("🔄 ดึงข้อมูล Real-time จาก Sensor"):
    st.rerun()
