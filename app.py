import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# 1. โหลดโมเดล AI ตัวใหม่ (ต้องอัปโหลดไฟล์ mea_pm_ai_model.pkl ที่มี 8 ตัวแปรขึ้น GitHub ก่อน)
@st.cache_resource
def load_my_model():
    return joblib.load('mea_pm_ai_model.pkl')

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_pm_ai_model.pkl' กรุณาเทรนและอัปโหลดไฟล์ขึ้น GitHub ก่อนครับ")

st.set_page_config(page_title="MEA Smart Diagnostic AI", layout="wide")
st.title("⚡ MEA Smart Maintenance AI (MSIA)")
st.subheader("ระบบวิเคราะห์และวางแผน PM อุปกรณ์ไฟฟ้าด้วยสถิติและ Acoustic Camera")

# 2. ส่วนรับข้อมูล (Sidebar)
st.sidebar.header("📥 นำเข้าข้อมูลเพื่อวิเคราะห์")
input_mode = st.sidebar.radio("เลือกวิธีนำเข้าข้อมูล:", ["อัปโหลดไฟล์ Excel", "กรอกข้อมูลเองหน้างาน"])

# รายชื่อตัวแปรทั้ง 8 ที่ AI ใช้ (ต้องเรียงลำดับให้ตรงกับตอนเทรน)
# 1.Temp, 2.Load, 3.Oil, 4.Vib, 5.Age, 6.Humidity, 7.Acoustic_dB, 8.Peak_Freq

if input_mode == "อัปโหลดไฟล์ Excel":
    uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx"])
    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file)
        # ตรวจสอบหัวตาราง
        required = ['Temp (°C)', 'Load (%)', 'Oil_Level (%)', 'Vibration (mm/s)', 'Age (Years)', 'Humidity (%)', 'Acoustic_dB', 'Peak_Freq_Hz']
        if all(col in df_raw.columns for col in required):
            st.sidebar.success("✅ หัวตารางถูกต้อง")
            df_to_predict = df_raw[required]
        else:
            st.sidebar.error(f"❌ หัวตารางไม่ครบ ต้องมี: {required}")
            df_to_predict = None
    else:
        df_to_predict = None
else:
    # กรณีพนักงานกรอกเองหน้างาน
    st.sidebar.subheader("✍️ กรอกค่าจากเครื่องวัดและระบบสถิติ")
    t_id = st.sidebar.text_input("รหัสหม้อแปลง", "TR-MEA-001")
    f1 = st.sidebar.number_input("1. Temp (°C)", 30.0, 120.0, 60.0)
    f2 = st.sidebar.number_input("2. Load (%)", 0.0, 150.0, 70.0)
    f3 = st.sidebar.number_input("3. Oil Level (%)", 0.0, 100.0, 90.0)
    f4 = st.sidebar.number_input("4. Vibration (mm/s)", 0.0, 10.0, 1.5)
    f5 = st.sidebar.number_input("5. Age (Years)", 0, 40, 10)
    f6 = st.sidebar.number_input("6. Humidity (%)", 0.0, 100.0, 50.0)
    f7 = st.sidebar.number_input("7. Acoustic (dB)", 30.0, 120.0, 45.0)
    f8 = st.sidebar.number_input("8. Peak Freq (Hz)", 1000, 100000, 20000)
    
    # รวมเป็น DataFrame แถวเดียว
    df_to_predict = pd.DataFrame([[f1, f2, f3, f4, f5, f6, f7, f8]], columns=['Temp (°C)', 'Load (%)', 'Oil_Level (%)', 'Vibration (mm/s)', 'Age (Years)', 'Humidity (%)', 'Acoustic_dB', 'Peak_Freq_Hz'])
    df_to_predict['Transformer_ID'] = t_id

# 3. ส่วนประมวลผลและแสดงผล
if df_to_predict is not None:
    # AI ทำนายผล
    # เตรียมข้อมูล Features (8 ตัวแปร)
    features_only = df_to_predict[['Temp (°C)', 'Load (%)', 'Oil_Level (%)', 'Vibration (mm/s)', 'Age (Years)', 'Humidity (%)', 'Acoustic_dB', 'Peak_Freq_Hz']]
    
    predictions = model.predict(features_only)
    probabilities = model.predict_proba(features_only)

    # รวมผลลัพธ์กลับเข้า DataFrame
    display_df = df_to_predict.copy()
    display_df['Risk_Score'] = [f"{p[1]*100:.1f}%" for p in probabilities]
    display_df['AI_Priority'] = ["🔴 CRITICAL" if p == 1 else "🟢 NORMAL" for p in predictions]
    
    # วันพยากรณ์ชำรุดล่วงหน้า (Logic: Risk สูง วันยิ่งสั้น)
    def est_date(prob):
        days = int(max(0, (1 - prob) * 60))
        return (datetime.now() + timedelta(days=days)).strftime('%d/%m/%Y')
    
    display_df['Est. Failure Date'] = [est_date(p[1]) for p in probabilities]

    # แสดงตารางผลลัพธ์
    st.subheader("📋 ตารางสรุปการวิเคราะห์สุขภาพอุปกรณ์และแผน PM")
    def color_priority(val):
        color = '#ff4b4b' if 'CRITICAL' in val else '#28a745'
        return f'background-color: {color}; color: white; font-weight: bold;'
    
    st.dataframe(display_df.style.applymap(color_priority, subset=['AI_Priority']), use_container_width=True)

    # 4. ส่วน Dashboard กราฟ
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**ความสำคัญของปัจจัยที่ส่งผลต่อความเสี่ยง (AI Feature Importance)**")
        importances = pd.Series(model.feature_importances_, index=features_only.columns)
        fig = px.bar(importances, orientation='h', color=importances, color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("กราฟนี้บอกพนักงานว่า AI ให้น้ำหนักกับตัวแปรไหนมากที่สุดในการวางแผน PM")

    with col2:
        st.write("**การกระจายความเสี่ยงในพื้นที่ (Area Risk Distribution)**")
        fig_pie = px.pie(display_df, names='AI_Priority', color='AI_Priority', 
                         color_discrete_map={'🔴 CRITICAL':'#ff4b4b', '🟢 NORMAL':'#28a745'})
        st.plotly_chart(fig_pie, use_container_width=True)

    # 5. คำแนะนำ Decision Support สำหรับเครื่องที่วิกฤตที่สุด
    critical_cases = display_df[display_df['AI_Priority'] == "🔴 CRITICAL"]
    if not critical_cases.empty:
        st.error(f"🚨 ตรวจพบหม้อแปลงที่ต้องทำ PM ทันทีจำนวน {len(critical_cases)} เครื่อง")
        for _, row in critical_cases.iterrows():
            with st.expander(f"🔍 รายละเอียดแผนงานสำหรับ {row.get('Transformer_ID', 'N/A')}"):
                st.write(f"**เหตุผลที่ AI แจ้งเตือน:** พบค่า Acoustic ({row['Acoustic_dB']} dB) และ Temp ({row['Temp (°C)']} °C) สอดคล้องกับรูปแบบการชำรุดในสถิติ")
                st.write("**ข้อแนะนำ:** 1. เตรียมทีมช่างเข้าตรวจสอบหน้างานภายใน 48 ชม. | 2. ตรวจสอบสภาพขั้วต่อและระดับน้ำมันเพิ่มเติม")
else:
    st.info("💡 กรุณาเลือกวิธีนำเข้าข้อมูลทางด้านซ้าย (อัปโหลดไฟล์ หรือ กรอกค่าหน้างาน) เพื่อให้ AI เริ่มการวิเคราะห์")
