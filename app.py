import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="SPP-AI Dashboard (KTD)", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 5px solid #FF8C00; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_mea_model():
    try:
        return joblib.load('mea_spp_ai_model.pkl')
    except:
        return None

model = load_mea_model()

# --- 3. HELPER FUNCTIONS ---
def run_ai_analysis(df):
    if model is not None and not df.empty:
        # 8 Features: [Thermal, Load, Voltage, Acoustic, PeakFreq, Trips, Age, Humidity]
        X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 
                'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
        preds = model.predict(X)
        probs = model.predict_proba(X)
        status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
        df['Status'] = [status_map[p] for p in preds]
        df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
    return df

# --- 4. DATABASE INITIALIZATION ---
if 'ktd_assets' not in st.session_state:
    st.session_state.ktd_assets = pd.DataFrame(columns=[
        'Transformer_ID', 'Feeder', 'Lat', 'Lon', 'Thermal_Temp', 'Load_Percent', 
        'Voltage_V', 'Acoustic_dB', 'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity', 'Status', 'Risk_Score'
    ])

# --- 5. SIDEBAR: DATA INPUT & FILE UPLOAD ---
with st.sidebar:
    st.header("📂 นำเข้าข้อมูล ฟขต.")
    
    # ช่องสำหรับอัปโหลดไฟล์ ฟขต Feeder.xlsx
    uploaded_file = st.file_uploader("เลือกไฟล์ ฟขต Feeder.xlsx", type=["xlsx", "csv"])
    
    if uploaded_file:
        try:
            # อ่านไฟล์ (รองรับทั้ง csv และ xlsx)
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file, skiprows=2) # ข้ามหัวกระดาษตามไฟล์จริง
            
            if 'Feeder' in raw_df.columns:
                # สรุปสถิติ Trip รายฟีดเดอร์
                trip_stats = raw_df['Feeder'].value_counts().to_dict()
                unique_feeders = list(trip_stats.keys())
                
                if st.button("🔄 อัปเดตรายชื่อฟีดเดอร์เข้าสู่ระบบ"):
                    new_data = []
                    for i, fdr in enumerate(unique_feeders[:30]): # สมมติหม้อแปลง 30 ตัวจากฟีดเดอร์ต่างๆ
                        new_data.append({
                            'Transformer_ID': f'TR-KTD-{i+1:03d}',
                            'Feeder': fdr,
                            'Lat': 13.702 + (i * 0.001),
                            'Lon': 100.555 + (i * 0.001),
                            'Thermal_Temp': np.random.uniform(40, 95),
                            'Load_Percent': np.random.uniform(40, 110),
                            'Voltage_V': 220.0,
                            'Acoustic_dB': np.random.uniform(40, 90),
                            'Peak_Freq_Hz': 25000.0,
                            'Trips_Count': trip_stats.get(fdr, 0),
                            'Age_Years': np.random.randint(5, 30),
                            'Humidity': 65.0
                        })
                    st.session_state.ktd_assets = pd.DataFrame(new_data)
                    st.session_state.ktd_assets = run_ai_analysis(st.session_state.ktd_assets)
                    st.success(f"โหลดข้อมูล {len(unique_feeders)} ฟีดเดอร์สำเร็จ")
            else:
                st.error("ไม่พบคอลัมน์ 'Feeder' ในไฟล์")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

    st.divider()
    if not st.session_state.ktd_assets.empty:
        st.subheader("🛠 สำรวจหน้างานรายเครื่อง")
        target_id = st.selectbox("เลือก ID หม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].index[0]
        
        new_th = st.slider("ความร้อน (°C)", 30, 120, int(st.session_state.ktd_assets.at[idx, 'Thermal_Temp']))
        new_ac = st.slider("เสียง (dB)", 30, 110, int(st.session_state.ktd_assets.at[idx, 'Acoustic_dB']))
        
        if st.button("💾 บันทึกและวิเคราะห์"):
            st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = new_th
            st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = new_ac
            st.session_state.ktd_assets = run_ai_analysis(st.session_state.ktd_assets)
            st.rerun()

# --- 6. MAIN DISPLAY ---
st.title("⚡ SPP-AI: ระบบวิเคราะห์ความเสี่ยงหม้อแปลง (KTD)")

if st.session_state.ktd_assets.empty:
    st.info("👈 กรุณาอัปโหลดไฟล์ 'ฟขต Feeder.xlsx' ที่แถบด้านซ้ายเพื่อเริ่มต้นระบบ")
else:
    df = st.session_state.ktd_assets
    # Dashboard Metrics
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card' style='border-top:5px solid #FF4B4B;'><h4>วิกฤต (Critical)</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card' style='border-top:5px solid #FF8C00;'><h4>เฝ้าระวัง (Watch)</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card' style='border-top:5px solid #28A745;'><h4>ปกติ (Normal)</h4><h1>{len(df[df['Status'] == '🟢 NORMAL'])}</h1></div>", unsafe_allow_html=True)

    # Map & Table
    tab1, tab2 = st.tabs(["📍 แผนที่ความเสี่ยง", "📋 ตารางข้อมูล"])
    with tab1:
        fig = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent",
                                hover_name="Transformer_ID", hover_data=["Feeder", "Trips_Count"],
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                zoom=13, height=500)
        fig.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.dataframe(df[['Transformer_ID', 'Feeder', 'Status', 'Thermal_Temp', 'Acoustic_dB', 'Trips_Count', 'Load_Percent']])
