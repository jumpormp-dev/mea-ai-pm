import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIG & STYLE (แบบเดิมที่คุยกัน) ---
st.set_page_config(page_title="SPP-AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 5px solid #FF8C00; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; height: 3em; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_mea_model():
    try:
        return joblib.load('mea_spp_ai_model.pkl')
    except:
        return None

model = load_mea_model()

# --- 3. AI INFERENCE FUNCTION ---
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

# --- 4. DATA STORAGE ---
if 'ktd_assets' not in st.session_state:
    st.session_state.ktd_assets = pd.DataFrame()

# --- 5. SIDEBAR: DATA INPUT ---
with st.sidebar:
    st.header("⚙️ จัดการข้อมูล")
    
    # ช่องโหลดไฟล์ ฟขต (ข้อมูลจริง)
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    
    if uploaded_file:
        # อ่านไฟล์ข้ามหัว 2 บรรทัดตามไฟล์จริง
        raw_df = pd.read_excel(uploaded_file, skiprows=2)
        if 'Feeder' in raw_df.columns:
            trip_stats = raw_df['Feeder'].value_counts().to_dict()
            unique_feeders = list(trip_stats.keys())
            
            if st.button("🔄 อัปเดตข้อมูลและประมวลผล AI"):
                new_rows = []
                for i, fdr in enumerate(unique_feeders[:20]):
                    new_rows.append({
                        'Transformer_ID': f'TR-KTD-{i+1:03d}',
                        'Feeder': fdr,
                        'Lat': 13.702 + np.random.uniform(-0.005, 0.005),
                        'Lon': 100.560 + np.random.uniform(-0.005, 0.005),
                        'Thermal_Temp': np.random.uniform(40, 110),
                        'Load_Percent': np.random.uniform(30, 120),
                        'Voltage_V': 220.0,
                        'Acoustic_dB': np.random.uniform(35, 105),
                        'Peak_Freq_Hz': 25000.0,
                        'Trips_Count': trip_stats.get(fdr, 0),
                        'Age_Years': np.random.randint(5, 30),
                        'Humidity': 65.0
                    })
                st.session_state.ktd_assets = run_ai_analysis(pd.DataFrame(new_rows))
                st.success("อัปเดตข้อมูลจาก ฟขต. สำเร็จ")

    st.divider()
    if not st.session_state.ktd_assets.empty:
        st.subheader("🔍 สำรวจรายเครื่อง")
        target_id = st.selectbox("เลือก ID หม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].index[0]
        
        new_th = st.slider("ความร้อน (°C)", 30, 120, int(st.session_state.ktd_assets.at[idx, 'Thermal_Temp']))
        new_ac = st.slider("เสียง (dB)", 30, 110, int(st.session_state.ktd_assets.at[idx, 'Acoustic_dB']))
        
        if st.button("💾 บันทึกและวิเคราะห์"):
            st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = new_th
            st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = new_ac
            st.session_state.ktd_assets = run_ai_analysis(st.session_state.ktd_assets)
            st.rerun()

# --- 6. MAIN CONTENT ---
st.title("⚡ SPP-AI Dashboard")
st.caption("ระบบวิเคราะห์และพยากรณ์ความเสี่ยงหม้อแปลง เขตคลองเตย (KTD)")

if st.session_state.ktd_assets.empty:
    st.warning("👈 กรุณาอัปโหลดไฟล์ 'ฟขต Feeder.xlsx' เพื่อเริ่มต้น")
else:
    df = st.session_state.ktd_assets
    
    # ROW 1: METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h4>รวม</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card' style='border-top-color:#FF4B4B'><h4>วิกฤต</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card' style='border-top-color:#FF8C00'><h4>เฝ้าระวัง</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card' style='border-top-color:#28A745'><h4>ปกติ</h4><h1>{len(df[df['Status'] == '🟢 NORMAL'])}</h1></div>", unsafe_allow_html=True)

    # ROW 2: TABS
    tab1, tab2 = st.tabs(["📍 Risk Map", "📋 Asset Table"])
    
    with tab1:
        fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent",
                                    hover_name="Transformer_ID", hover_data=["Feeder", "Trips_Count"],
                                    color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                    zoom=13, height=600)
        fig_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
        
    with tab2:
        st.dataframe(df[['Transformer_ID', 'Feeder', 'Status', 'Risk_Score', 'Thermal_Temp', 'Acoustic_dB', 'Load_Percent', 'Trips_Count']], use_container_width=True)
