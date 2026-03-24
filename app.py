import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE (UI โทน กฟน. ส้ม-เทา-ขาว) ---
st.set_page_config(page_title="SPP-AI: KTD Smart Maintenance", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 5px solid #FF8C00; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; height: 3em; border: none; }
    .status-panel { padding: 10px; border-radius: 8px; color: white; font-weight: bold; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD AI MODEL (.pkl) ---
@st.cache_resource
def load_mea_model():
    try:
        return joblib.load('mea_spp_ai_model.pkl')
    except:
        return None

model = load_mea_model()

# --- 3. DATABASE & AI PROCESSING ---
def run_ai_analysis(df):
    """ฟังก์ชันหลักสำหรับส่งข้อมูล 8 Features เข้าโมเดล AI"""
    if model and not df.empty:
        # เรียงลำดับ Features ตามที่เทรนใน Colab: 
        # [Thermal, Load, Voltage, Acoustic, PeakFreq, Trips, Age, Humidity]
        X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 
                'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
        
        preds = model.predict(X)
        probs = model.predict_proba(X)
        
        status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
        df['Status'] = [status_map[p] for p in preds]
        df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
    return df

if 'ktd_assets' not in st.session_state:
    np.random.seed(42)
    feeders = ['EM-418', 'SAM-13', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'RPR-423']
    
    # สร้างข้อมูลเริ่มต้นที่มีความหลากหลาย (เพื่อให้ AI เจอตัวแดง/ส้ม)
    df_init = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i+1:03d}' for i in range(20)],
        'Feeder': [np.random.choice(feeders) for _ in range(20)],
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        'Thermal_Temp': np.random.uniform(40, 110, 20),
        'Load_Percent': np.random.uniform(30, 115, 20),
        'Voltage_V': np.random.uniform(215, 235, 20),
        'Acoustic_dB': np.random.uniform(35, 105, 20),
        'Peak_Freq_Hz': [25000.0] * 20,
        'Trips_Count': np.random.randint(0, 15, 20),
        'Age_Years': np.random.randint(1, 35, 20),
        'Humidity': [65.0] * 20
    })
    
    # วิเคราะห์ทันทีที่สร้างข้อมูลเสร็จ
    st.session_state.ktd_assets = run_ai_analysis(df_init)

# --- 4. HEADER ---
st.markdown("<h1 style='text-align: center; color: #FF8C00;'>SPP-AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>ระบบวิเคราะห์และพยากรณ์ความเสี่ยงหม้อแปลง เขตคลองเตย (KTD)</p>", unsafe_allow_html=True)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ การจัดการข้อมูล")
    
    if st.button("📡 Sync Smart Meter (172.16.111.184)"):
        st.session_state.ktd_assets['Load_Percent'] = np.random.uniform(40, 120, 20)
        st.session_state.ktd_assets = run_ai_analysis(st.session_state.ktd_assets)
        st.success("อัปเดตข้อมูล Smart Meter และรัน AI ใหม่แล้ว")

    st.divider()
    st.subheader("🔍 ตรวจสอบรายเครื่อง")
    target_id = st.selectbox("เลือก ID หม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].index[0]
    
    new_th = st.slider("ความร้อน Thermal (°C)", 30, 120, int(st.session_state.ktd_assets.at[idx, 'Thermal_Temp']))
    new_ac = st.slider("ระดับเสียง Acoustic (dB)", 30, 110, int(st.session_state.ktd_assets.at[idx, 'Acoustic_dB']))
    
    if st.button("💾 บันทึกและวิเคราะห์ใหม่"):
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = new_th
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = new_ac
        st.session_state.ktd_assets = run_ai_analysis(st.session_state.ktd_assets)
        st.rerun()

# --- 6. MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "📍 Risk Map", "📋 Asset Diagnostics"])

with tab1:
    df = st.session_state.ktd_assets
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><h4>ทั้งหมด</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card' style='border-top:5px solid #FF4B4B;'><h4>วิกฤต (Red)</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card' style='border-top:5px solid #FF8C00;'><h4>เฝ้าระวัง (Orange)</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card' style='border-top:5px solid #28A745;'><h4>ปกติ (Green)</h4><h1>{len(df[df['Status'] == '🟢 NORMAL'])}</h1></div>", unsafe_allow_html=True)

    st.subheader("📊 สัดส่วนความเสี่ยงแยกตามฟีดเดอร์ (Feeder)")
    fig_bar = px.bar(df, x="Feeder", color="Status", barmode="group",
                     color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'})
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent",
                                hover_name="Transformer_ID", hover_data=["Thermal_Temp", "Acoustic_dB", "Trips_Count"],
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                zoom=14, height=600)
    fig_map.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("🧐 ข้อมูลเชิงลึกจาก AI")
    sel_id = st.selectbox("ดูรายละเอียดอุปกรณ์:", df['Transformer_ID'], key="diag_sel")
    row = df[df['Transformer_ID'] == sel_id].iloc[0]
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk Confidence Score", f"{row['Risk_Score']*100:.1f}%")
        status_color = "#FF4B4B" if "🔴" in row['Status'] else "#FF8C00" if "🟡" in row['Status'] else "#28A745"
        st.markdown(f"<div class='status-panel' style='background-color:{status_color};'>{row['Status']}</div>", unsafe_allow_html=True)
    
    with c2:
        st.info(f"**เหตุผลการวิเคราะห์:** หม้อแปลง {sel_id} มีระดับความร้อน {row['Thermal_Temp']}°C และระดับเสียง {row['Acoustic_dB']}dB พร้อมสถิติการขัดข้อง {row['Trips_Count']} ครั้ง")
        st.table(df[df['Transformer_ID'] == sel_id][['Feeder', 'Load_Percent', 'Voltage_V', 'Age_Years']])
