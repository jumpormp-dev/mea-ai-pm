import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="SPP-AI: Dashboard (KTD)", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 8px solid #FF8C00; }
    .metric-crit { border-top: 8px solid #FF4B4B; }
    .metric-watch { border-top: 8px solid #FF8C00; }
    .metric-normal { border-top: 8px solid #28A745; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; height: 3em; }
    h1, h2, h3 { font-family: 'Kanit', sans-serif; color: #444444; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_spp_model():
    try:
        return joblib.load('mea_spp_ai_model.pkl')
    except:
        return None

model = load_spp_model()

# --- 3. INITIAL DATABASE (KTD & ฟขต) ---
if 'ktd_assets' not in st.session_state:
    np.random.seed(42)
    feeders = ['EM-418', 'SAM-13', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'RPR-423']
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': [np.random.choice(feeders) for _ in range(20)],
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        'Load_Percent': np.random.uniform(30, 110, 20),
        'Voltage_V': np.random.uniform(215, 235, 20),
        'Trips_Count': np.random.randint(0, 10, 20),
        'Acoustic_dB': np.random.uniform(40, 90, 20),
        'Thermal_Temp': np.random.uniform(45, 95, 20),
        'Peak_Freq_Hz': [25000.0] * 20,
        'Age_Years': np.random.randint(5, 30, 20),
        'Humidity': [65.0] * 20,
        'Risk_Score': [0.0] * 20,
        'Status': ['🟢 NORMAL'] * 20
    })

# --- 4. HEADER ---
st.title("⚡ SPP-AI: ระบบวิเคราะห์ความเสี่ยงหม้อแปลง")
st.caption("Smart Plan Predictive Maintenance AI (KTD Area)")
st.divider()

if model is None:
    st.error("⚠️ ไม่พบไฟล์ 'mea_spp_ai_model.pkl' กรุณาตรวจสอบไฟล์ในโฟลเดอร์")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ การจัดการข้อมูล")
    
    # ปุ่มวิเคราะห์ทั้งหมด (Bulk Analysis)
    if st.button("🚀 วิเคราะห์ความเสี่ยงทั้งหมด (Bulk)"):
        if model:
            df = st.session_state.ktd_assets
            # เตรียม 8 Features ให้ตรงลำดับ
            X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 
                    'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
            
            preds = model.predict(X)
            # เช็คว่าโมเดลมี predict_proba หรือไม่
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
            else:
                df['Risk_Score'] = 0.5 # ค่า default หากโมเดลไม่มี proba
                
            status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
            df['Status'] = [status_map[p] for p in preds]
            st.session_state.ktd_assets = df
            st.success("ประมวลผล AI สำเร็จ")
            st.rerun()

    uploaded_xlsx = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    if uploaded_xlsx:
        df_xlsx = pd.read_excel(uploaded_xlsx, skiprows=2)
        trip_map = df_xlsx['Feeder'].value_counts().to_dict()
        for i, row in st.session_state.ktd_assets.iterrows():
            st.session_state.ktd_assets.at[i, 'Trips_Count'] = trip_map.get(row['Feeder'], 0)
        st.success("อัปเดตสถิติไฟดับแล้ว")

    st.divider()
    target_id = st.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].index[0]
    ac_in = st.number_input("เสียง (dB)", 30.0, 110.0, float(st.session_state.ktd_assets.at[idx, 'Acoustic_dB']))
    th_in = st.number_input("ความร้อน (°C)", 20.0, 120.0, float(st.session_state.ktd_assets.at[idx, 'Thermal_Temp']))

    if st.button("💾 บันทึกและวิเคราะห์เฉพาะตัว"):
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_in
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_in
        # รัน AI รายเครื่อง
        row = st.session_state.ktd_assets.iloc[idx]
        feat = np.array([[row['Thermal_Temp'], row['Load_Percent'], row['Voltage_V'], ac_in, 25000, row['Trips_Count'], row['Age_Years'], 65.0]])
        res = model.predict(feat)[0]
        st.session_state.ktd_assets.at[idx, 'Status'] = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}[res]
        st.success(f"บันทึก {target_id} สำเร็จ")
        st.rerun()

# --- 6. MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["📊 Executive", "🔍 Diagnostics", "📅 Action Plan"])

with tab1:
    df = st.session_state.ktd_assets
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h4>รวม</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card metric-crit'><h4>วิกฤต</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card metric-watch'><h4>เฝ้าระวัง</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card metric-normal'><h4>พื้นที่</h4><h1>KTD</h1></div>", unsafe_allow_html=True)
    
    fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", zoom=13, height=500,
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    sel_id = st.selectbox("ตรวจสอบข้อมูลเชิงลึก:", df['Transformer_ID'], key="diag_sel")
    res = df[df['Transformer_ID'] == sel_id].iloc[0]
    col_l, col_r = st.columns([1, 1.5])
    with col_l:
        # Gauge แสดงคะแนนความเสี่ยง
        val = res['Risk_Score'] * 100 if res['Risk_Score'] > 0 else 50.0
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': "Risk Score (%)"},
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        st.plotly_chart(fig_g, use_container_width=True)
    with col_r:
        st.info("### ปัจจัยที่ AI ใช้ตัดสินใจ")
        st.write(f"- 🔊 เสียง: {res['Acoustic_dB']} dB")
        st.write(f"- 🌡️ ความร้อน: {res['Thermal_Temp']} °C")
        st.write(f"- ⚡ โหลด: {res['Load_Percent']:.1f}%")
        st.write(f"- 📉 สถิติไฟดับ: {res['Trips_Count']} ครั้ง")

with tab3:
    st.subheader("📋 แผนงานบำรุงรักษา (Priority)")
    urgent = df[df['Status'] != '🟢 NORMAL'].sort_values('Status', ascending=False)
    st.table(urgent[['Transformer_ID', 'Feeder', 'Status', 'Thermal_Temp', 'Acoustic_dB', 'Load_Percent']])
