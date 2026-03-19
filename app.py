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
st.write("ระบบวิเคราะห์และพยากรณ์ความเสี่ยงอุปกรณ์ไฟฟ้า (Version: Research Prototype)")

# 2. ส่วนอัปโหลดไฟล์
st.sidebar.header("📥 นำเข้าข้อมูลจากหน้างาน")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df_input = pd.read_excel(uploaded_file)
    
    # กำหนดคอลัมน์ที่ต้องใช้ในการพยากรณ์ (Input Features)
    # หมายเหตุ: ลำดับต้องตรงกับตอนที่เทรนใน Colab 
    # หากตอนเทรนมีแค่ 4 ตัวแปร ให้เลือกเฉพาะ 4 ตัวแปรหลัก
    predict_cols = ['Temp (°C)', 'Load (%)', 'Oil_Level (%)', 'Vibration (mm/s)']
    
    if all(col in df_input.columns for col in predict_cols):
        st.success("✅ โหลดข้อมูลสำเร็จ! กำลังประมวลผลด้วย AI...")
        
        results = []
        for index, row in df_input.iterrows():
            # ดึงค่าตามลำดับที่ AI ต้องการ
            features = np.array([[row['Temp (°C)'], row['Load (%)'], row['Oil_Level (%)'], row['Vibration (mm/s)']]])
            prob = model.predict_proba(features)[0][1]
            
            # คำนวณวันที่จะพัง (Logic: ยิ่งความเสี่ยงสูง วันยิ่งน้อย)
            days_to_fail = int(max(0, (1 - prob) * 45))
            fail_date = (datetime.now() + timedelta(days=days_to_fail)).strftime('%d/%m/%Y')
            
            results.append({
                "Transformer ID": row.get('Transformer_ID', f"TR-{index+1}"),
                "Health Status": "🔴 CRITICAL" if prob > 0.8 else "🟡 WATCH" if prob > 0.5 else "🟢 STABLE",
                "Risk Score": f"{prob*100:.1f}%",
                "Est. Fail Date": fail_date if prob > 0.5 else "N/A",
                "Age (Years)": row.get('Age (Years)', '-'),
                "Main Factors": "Temp/Load" if prob > 0.5 else "None"
            })
        
        df_result = pd.DataFrame(results)

        # 3. แสดงผลตาราง
        st.subheader("📊 ผลการวิเคราะห์รายอุปกรณ์")
        def color_status(val):
            if "CRITICAL" in val: color = '#ff4b4b'
            elif "WATCH" in val: color = '#ffa500'
            else: color = '#28a745'
            return f'background-color: {color}; color: white; font-weight: bold;'
        
        st.dataframe(df_result.style.applymap(color_status, subset=['Health Status']), use_container_width=True)

        # 4. กราฟวิเคราะห์ภาพรวม (AI Visuals)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ความสำคัญของปัจจัย (Feature Importance)**")
            importances = pd.Series(model.feature_importances_, index=['Temp', 'Load', 'Oil', 'Vibration'])
            fig = px.bar(importances, orientation='h', color=importances, color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("**สัดส่วนสถานะสุขภาพหม้อแปลงในเขตพื้นที่**")
            fig_pie = px.pie(df_result, names='Health Status', color='Health Status',
                             color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟡 WATCH':'#ffa500', '🟢 STABLE':'#28a745'})
            st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.error(f"❌ หัวตารางไม่ถูกต้อง! โปรดตรวจสอบว่ามีคอลัมน์: {', '.join(predict_cols)}")
else:
    st.info("💡 กรุณาอัปโหลดไฟล์ Excel ที่มีข้อมูล Sensor จากแถบเมนูด้านซ้าย")
