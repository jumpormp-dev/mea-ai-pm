import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# 1. โหลดโมเดล
@st.cache_resource
def load_my_model():
    return joblib.load('pdm_model.pkl')

model = load_my_model()

st.set_page_config(page_title="MEA AI Asset Intelligence", layout="wide")
st.title("⚡ MEA Smart Maintenance AI (MSIA)")
st.write("ระบบวิเคราะห์และพยากรณ์ความเสี่ยงอุปกรณ์ไฟฟ้าด้วยปัญญาประดิษฐ์")

# 2. ส่วนอัปโหลดไฟล์
st.sidebar.header("📥 นำเข้าข้อมูลจากหน้างาน")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # อ่านข้อมูลจาก Excel
    df_input = pd.read_excel(uploaded_file)
    
    # ตรวจสอบว่ามีคอลัมน์ครบตามที่ AI ต้องการไหม
    required_cols = ['Temp (°C)', 'Load (%)', 'Oil Level (%)', 'Vibration (mm/s)']
    
    if all(col in df_input.columns for col in required_cols):
        st.success("✅ โหลดข้อมูลสำเร็จ! กำลังประมวลผลด้วย AI...")
        
        results = []
        for index, row in df_input.iterrows():
            # ดึงค่ามาวิเคราะห์
            features = np.array([[row['Temp (°C)'], row['Load (%)'], row['Oil Level (%)'], row['Vibration (mm/s)']]])
            prob = model.predict_proba(features)[0][1]
            
            # พยากรณ์วันชำรุด
            days_to_fail = int(max(0, (1 - prob) * 45))
            fail_date = (datetime.now() + timedelta(days=days_to_fail)).strftime('%d/%m/%Y')
            
            results.append({
                "Asset ID": row.get('Asset ID', f"Asset-{index+1}"),
                "Health Status": "🔴 CRITICAL" if prob > 0.8 else "🟡 WATCH" if prob > 0.5 else "🟢 STABLE",
                "Risk Score": f"{prob*100:.1f}%",
                "Est. Fail Date": fail_date if prob > 0.5 else "N/A",
                "Temp": row['Temp (°C)'],
                "Load": row['Load (%)']
            })
        
        df_result = pd.DataFrame(results)

        # 3. แสดงผลตาราง
        st.subheader("📊 ผลการวิเคราะห์สุขภาพอุปกรณ์ (จากไฟล์ที่อัปโหลด)")
        def color_status(val):
            if "CRITICAL" in val: color = '#ff4b4b'
            elif "WATCH" in val: color = '#ffa500'
            else: color = '#28a745'
            return f'background-color: {color}; color: white; font-weight: bold;'
        
        st.dataframe(df_result.style.applymap(color_status, subset=['Health Status']), use_container_width=True)

        # 4. AI Insights กราฟวิเคราะห์
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Feature Importance (น้ำหนักปัจจัยที่ AI ใช้)**")
            importances = pd.Series(model.feature_importances_, index=['Temp', 'Load', 'Oil', 'Vib'])
            fig = px.bar(importances, orientation='h', color=importances, color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.write("**Risk Distribution (สัดส่วนความเสี่ยง)**")
            fig_pie = px.pie(df_result, names='Health Status', color='Health Status',
                             color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 STABLE':'#28a745'})
            st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.error(f"❌ รูปแบบไฟล์ไม่ถูกต้อง! ต้องมีคอลัมน์: {', '.join(required_cols)}")
else:
    st.info("💡 กรุณาอัปโหลดไฟล์ Excel ทางแถบด้านซ้ายเพื่อเริ่มการวิเคราะห์")
    st.image("https://via.placeholder.com/800x200.png?text=Waiting+for+Data+Upload...", use_column_width=True)
