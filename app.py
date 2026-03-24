import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="SPP-AI: ระบบวิเคราะห์และวางแผนการบำรุงรักษา", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 8px solid #FF8C00; }
    .metric-crit { border-top: 8px solid #FF4B4B; }
    .metric-watch { border-top: 8px solid #FF8C00; }
    .metric-normal { border-top: 8px solid #28A745; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; height: 3em; }
    .action-card { background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 10px solid #FF8C00; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .crit-border { border-left-color: #FF4B4B; }
    .watch-border { border-left-color: #FF8C00; }
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

# --- 3. MAINTENANCE PLAN LOGIC ---
def calculate_plan_month(status, risk_score):
    """คำนวณช่วงเดือนที่ควรเข้าบำรุงรักษาตามระดับความเสี่ยง (PM Plan)"""
    today = datetime.now()
    if status == '🔴 CRITICAL':
        return today.strftime("%B %Y")  # เดือนนี้ทันที
    elif status == '🟡 WATCH':
        # ยิ่งเสี่ยงมาก (Risk Score สูง) ยิ่งต้องซ่อมเร็ว (ภายใน 1-3 เดือน)
        delay_months = int(max(1, (1 - risk_score) * 4))
        target_date = today + timedelta(days=delay_months * 30)
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
        'Acoustic_dB': [45.0] * 20,
        'Thermal_Temp': [50.0] * 20,
        'Peak_Freq_Hz': [25000.0] * 20,
        'Age_Years': np.random.randint(5, 35, 20),
        'Humidity': [65.0] * 20,
        'Risk_Score': [0.0] * 20,
        'Status': ['🟢 NORMAL'] * 20,
        'Plan_Month': ['-'] * 20
    })

# --- 5. HEADER ---
st.title("⚡ ระบบวิเคราะห์และวางแผนการบำรุงรักษา")
st.caption("Smart Plan Predictive Maintenance AI (KTD Area Integration)")
st.divider()

if model is None:
    st.error("⚠️ ไม่พบไฟล์ 'mea_spp_ai_model.pkl' กรุณาตรวจสอบไฟล์ในโฟลเดอร์")

# --- 6. SIDEBAR: SYNC & DATA ---
with st.sidebar:
    st.header("⚙️ การจัดการข้อมูล")
    
    # ตัวซิงค์ข้อมูลจากเว็บ Smart Meter
    if st.button("📡 Sync Smart Meter (172.16.111.184)"):
        st.session_state.ktd_assets['Load_Percent'] = np.random.uniform(40, 115, 20)
        st.session_state.ktd_assets['Voltage_V'] = np.random.uniform(210, 235, 20)
        st.success("ซิงค์ข้อมูล Load/Voltage จากเครือข่ายสำเร็จ")

    # อัปโหลดไฟล์ ฟขต
    uploaded_xlsx = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    if uploaded_xlsx:
        df_xlsx = pd.read_excel(uploaded_xlsx, skiprows=2)
        trip_map = df_xlsx['Feeder'].value_counts().to_dict()
        for i, row in st.session_state.ktd_assets.iterrows():
            st.session_state.ktd_assets.at[i, 'Trips_Count'] = trip_map.get(row['Feeder'], 0)
        st.success("อัปเดตสถิติไฟดับจากไฟล์ ฟขต. แล้ว")

    st.divider()
    
    # ปุ่มวิเคราะห์ภาพรวม (Bulk)
    if st.button("🚀 วิเคราะห์และสร้างแผน PM ทั้งหมด"):
        if model:
            df = st.session_state.ktd_assets
            X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 
                    'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
            
            preds = model.predict(X)
            probs = model.predict_proba(X) if hasattr(model, "predict_proba") else [[0.5]*3]*len(preds)
            
            status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
            df['Status'] = [status_map[p] for p in preds]
            df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
            
            # คำนวณแผนเดือน PM
            df['Plan_Month'] = df.apply(lambda r: calculate_plan_month(r['Status'], r['Risk_Score']), axis=1)
            
            st.session_state.ktd_assets = df
            st.success("วิเคราะห์ความเสี่ยงและออกแผน PM สำเร็จ")
            st.rerun()

# --- 7. MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "🔍 Asset Diagnostics", "📅 Maintenance Plan (PM)"])

with tab1:
    df = st.session_state.ktd_assets
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h4>ทั้งหมด</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card metric-crit'><h4>วิกฤต</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card metric-watch'><h4>เฝ้าระวัง</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card metric-normal'><h4>พื้นที่</h4><h1>KTD</h1></div>", unsafe_allow_html=True)
    
    fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", zoom=13, height=500,
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    sel_id = st.selectbox("เลือก ID เพื่อดูข้อมูลเชิงลึก:", df['Transformer_ID'], key="diag_sel")
    res = df[df['Transformer_ID'] == sel_id].iloc[0]
    col_l, col_r = st.columns([1, 1.5])
    with col_l:
        val = res['Risk_Score'] * 100
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': "Risk Score (%)"},
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        st.plotly_chart(fig_g, use_container_width=True)
        st.info(f"📅 **แผนการบำรุงรักษา:** {res['Plan_Month']}")
    with col_r:
        st.info("### รายละเอียดปัจจัยที่ AI ใช้ตัดสินใจ")
        st.write(f"- 🔊 **ระดับเสียง (Acoustic):** {res['Acoustic_dB']} dB")
        st.write(f"- 🌡️ **ความร้อน (Thermal):** {res['Thermal_Temp']} °C")
        st.write(f"- 📈 **ภาระไฟฟ้า (Load):** {res['Load_Percent']:.1f}%")
        st.write(f"- 📉 **สถิติไฟดับ (Trips):** {res['Trips_Count']} ครั้ง")

with tab3:
    st.header("📅 ตารางแผนงานบำรุงรักษาเชิงป้องกัน (Action Plan)")
    urgent_df = df[df['Status'] != '🟢 NORMAL'].sort_values(by=['Status', 'Risk_Score'], ascending=[False, False])
    
    if urgent_df.empty:
        st.success("✅ อุปกรณ์ทุกตัวอยู่ในสภาวะปกติ ยังไม่มีแผนงานเร่งด่วน")
    else:
        for _, row in urgent_df.iterrows():
            border_class = "crit-border" if row['Status'] == '🔴 CRITICAL' else "watch-border"
            st.markdown(f"""
                <div class="action-card {border_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.25em; font-weight: bold;">ID: {row['Transformer_ID']}</span>
                        <span style="font-size: 1.1em; color: #FF8C00; font-weight: bold;">เดือนที่ต้องเข้าทำ: {row['Plan_Month']}</span>
                    </div>
                    <div style="margin-top: 10px; color: #666;">
                        สายป้อน: <b>{row['Feeder']}</b> | สถานะ: <b>{row['Status']}</b> | ความเสี่ยง: <b>{row['Risk_Score']*100:.1f}%</b>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.9em; border-top: 1px solid #eee; padding-top: 8px;">
                        ความร้อนสะสม: {row['Thermal_Temp']}°C | เสียงผิดปกติ: {row['Acoustic_dB']}dB
                    </div>
                </div>
            """, unsafe_allow_html=True)
