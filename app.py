import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="SPP-AI Dashboard", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    .main { background-color: #F8F9FA; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 8px solid #FF8C00; }
    .action-card { background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 10px solid #FF8C00; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .crit-border { border-left-color: #FF4B4B; }
    .watch-border { border-left-color: #FF8C00; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; height: 3em; }
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

# --- 3. MAINTENANCE LOGIC ---
def calculate_pm_plan(status, risk_score):
    """คำนวณเดือนที่ควรเข้าบำรุงรักษา (PM)"""
    today = datetime.now()
    if status == '🔴 CRITICAL':
        return today.strftime("%B %Y")
    elif status == '🟡 WATCH':
        # กระจายแผนงาน 1-3 เดือนตามความรุนแรง
        months_ahead = int(max(1, (1 - risk_score) * 4))
        target_date = today + timedelta(days=months_ahead * 30)
        return target_date.strftime("%B %Y")
    return "Next Year (Routine)"

# --- 4. INITIAL DATABASE ---
if 'ktd_assets' not in st.session_state:
    np.random.seed(42)
    feeders = ['EM-418', 'SAM-13', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'RPR-423']
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': [np.random.choice(feeders) for _ in range(20)],
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        'Load_Percent': [0.0] * 20,
        'Voltage_V': [220.0] * 20,
        'Trips_Count': [0] * 20,
        'Acoustic_dB': np.random.uniform(40, 90, 20),
        'Thermal_Temp': np.random.uniform(45, 95, 20),
        'Peak_Freq_Hz': [25000.0] * 20,
        'Age_Years': np.random.randint(5, 35, 20),
        'Humidity': [65.0] * 20,
        'Risk_Score': [0.1] * 20,
        'Status': ['🟢 NORMAL'] * 20,
        'PM_Plan': ['-'] * 20
    })

# --- 5. HEADER ---
st.title("⚡ ระบบวิเคราะห์และวางแผนการบำรุงรักษา")
st.caption("Smart Plan Predictive Maintenance AI (KTD Area)")
st.divider()

# --- 6. SIDEBAR: DATA SYNC ---
with st.sidebar:
    st.header("⚙️ การจัดการข้อมูล")
    
    # ส่วนที่ 1: ดึงข้อมูลจากเว็บ (Smart Meter)
    if st.button("📡 Sync ข้อมูล Smart Meter (172.16.111.184)"):
        with st.spinner('กำลังเชื่อมต่อฐานข้อมูล กฟน...'):
            # ในสภาวะจริงจะใช้ requests.get('http://172.16.111.184:8501/data')
            # จำลองการดึงข้อมูล Load สดๆ
            st.session_state.ktd_assets['Load_Percent'] = np.random.uniform(40, 115, 20)
            st.session_state.ktd_assets['Voltage_V'] = np.random.uniform(215, 235, 20)
            st.success("ดึงข้อมูลจาก Smart Meter สำเร็จ")

    # ส่วนที่ 2: วิเคราะห์ภาพรวม (สร้างแผน PM)
    if st.button("🚀 วิเคราะห์ความเสี่ยงและสร้างแผน PM"):
        if model:
            df = st.session_state.ktd_assets
            X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
            preds = model.predict(X)
            probs = model.predict_proba(X) if hasattr(model, "predict_proba") else [[0.5]*3]*len(preds)
            
            status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
            df['Status'] = [status_map[p] for p in preds]
            df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
            
            # สร้างแผน PM รายเดือน
            df['PM_Plan'] = df.apply(lambda r: calculate_pm_plan(r['Status'], r['Risk_Score']), axis=1)
            
            st.session_state.ktd_assets = df
            st.success("วิเคราะห์ความเสี่ยงและจัดทำแผน PM สำเร็จ!")
            st.rerun()

    uploaded_xlsx = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    if uploaded_xlsx:
        df_xlsx = pd.read_excel(uploaded_xlsx, skiprows=2)
        trip_map = df_xlsx['Feeder'].value_counts().to_dict()
        for i, row in st.session_state.ktd_assets.iterrows():
            st.session_state.ktd_assets.at[i, 'Trips_Count'] = trip_map.get(row['Feeder'], 0)
        st.success("อัปเดตสถิติไฟดับเรียบร้อย")

# --- 7. MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "🔍 Diagnostics", "📅 แผนงานบำรุงรักษา (PM)"])

with tab1:
    df = st.session_state.ktd_assets
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.markdown(f"<div class='metric-card'><h4>ทั้งหมด</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
    col_m2.markdown(f"<div class='metric-card' style='border-top-color:#FF4B4B'><h4>วิกฤต</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    col_m3.markdown(f"<div class='metric-card' style='border-top-color:#FF8C00'><h4>เฝ้าระวัง</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    col_m4.markdown(f"<div class='metric-card' style='border-top-color:#28A745'><h4>พื้นที่</h4><h1>KTD</h1></div>", unsafe_allow_html=True)
    
    fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", zoom=13, height=500,
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    sel_id = st.selectbox("เลือก ID เพื่อดูข้อมูลเชิงลึก:", df['Transformer_ID'])
    res = df[df['Transformer_ID'] == sel_id].iloc[0]
    c_l, c_r = st.columns([1, 1.5])
    with c_l:
        val = res['Risk_Score'] * 100
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': "Risk Score (%)"},
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        st.plotly_chart(fig_g, use_container_width=True)
    with c_r:
        st.info(f"### 📋 สรุปผลสำหรับ {sel_id}")
        st.write(f"- **แผนบำรุงรักษา:** {res['PM_Plan']}")
        st.write(f"- **สถานะความเสี่ยง:** {res['Status']}")
        st.write(f"- **ปัจจัยหลัก:** ความร้อน {res['Thermal_Temp']}°C, เสียง {res['Acoustic_dB']}dB")

with tab3:
    st.header("📅 แผนการบำรุงรักษาเชิงป้องกันรายเดือน")
    urgent = df[df['Status'] != '🟢 NORMAL'].sort_values(by=['Status', 'Risk_Score'], ascending=[False, False])
    
    if urgent.empty:
        st.info("✅ ยังไม่มีรายการที่ต้องวางแผนซ่อมเร่งด่วน กรุณากดปุ่ม Sync ข้อมูลและวิเคราะห์ความเสี่ยงที่ Sidebar")
    else:
        for _, row in urgent.iterrows():
            card_class = "crit-border" if row['Status'] == '🔴 CRITICAL' else "watch-border"
            st.markdown(f"""
                <div class="action-card {card_class}">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-size: 1.25em; font-weight: bold;">{row['Transformer_ID']} ({row['Feeder']})</span>
                        <span style="color: #FF8C00; font-weight: bold; font-size: 1.1em;">ช่วงเดือนที่ต้องเข้า: {row['PM_Plan']}</span>
                    </div>
                    <div style="margin-top: 10px; color: #555;">
                        ระดับความรุนแรง: <b>{row['Status']}</b> | ความแม่นยำ AI: <b>{row['Risk_Score']*100:.1f}%</b>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.9em; color: #777;">
                        ความร้อน: {row['Thermal_Temp']}°C | เสียง: {row['Acoustic_dB']}dB | ภาระไฟฟ้า: {row['Load_Percent']:.1f}% | Trip: {row['Trips_Count']} ครั้ง
                    </div>
                </div>
            """, unsafe_allow_html=True)
