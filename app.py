import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="SPP-AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 8px solid #FF8C00; }
    .action-card { background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 10px solid #FF8C00; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .crit-border { border-left-color: #FF4B4B; }
    .watch-border { border-left-color: #FF8C00; }
    .status-badge { padding: 4px 12px; border-radius: 20px; color: white; font-weight: bold; font-size: 0.85em; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_spp_model():
    try: return joblib.load('mea_spp_ai_model.pkl')
    except: return None

model = load_spp_model()

# --- 3. MAINTENANCE LOGIC ---
def calculate_plan_month(status, risk_score):
    """คำนวณเดือนที่ควรซ่อมตามระดับความเสี่ยง"""
    today = datetime.now()
    if status == '🔴 CRITICAL':
        return today.strftime("%B %Y")
    elif status == '🟡 WATCH':
        # ยิ่ง Risk Score สูง (เข้าใกล้ 1) ยิ่งต้องซ่อมเร็ว
        delay_months = int(max(1, (1 - risk_score) * 4))
        target_date = today + timedelta(days=delay_months * 30)
        return target_date.strftime("%B %Y")
    return "Routine Checkup"

# --- 4. DATA INITIALIZATION ---
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
        'Age_Years': np.random.randint(5, 35, 20),
        'Humidity': [65.0] * 20,
        'Risk_Score': [0.1] * 20,
        'Status': ['🟢 NORMAL'] * 20,
        'Plan_Month': ['-'] * 20
    })

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ ระบบจัดการข้อมูล")
    if st.button("🚀 วิเคราะห์ความเสี่ยงทั้งหมด (Bulk)"):
        if model:
            df = st.session_state.ktd_assets
            X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
            preds = model.predict(X)
            # เช็ค predict_proba เพื่อความแม่นยำของเดือน
            probs = model.predict_proba(X) if hasattr(model, "predict_proba") else [[0.5, 0.5, 0.5]] * len(preds)
            
            status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
            df['Status'] = [status_map[p] for p in preds]
            df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
            df['Plan_Month'] = df.apply(lambda r: calculate_plan_month(r['Status'], r['Risk_Score']), axis=1)
            
            st.session_state.ktd_assets = df
            st.success("ประมวลผลสำเร็จ!")
            st.rerun()

    uploaded_xlsx = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    if uploaded_xlsx:
        df_xlsx = pd.read_excel(uploaded_xlsx, skiprows=2)
        trip_map = df_xlsx['Feeder'].value_counts().to_dict()
        for i, row in st.session_state.ktd_assets.iterrows():
            st.session_state.ktd_assets.at[i, 'Trips_Count'] = trip_map.get(row['Feeder'], 0)
        st.success("อัปเดตสถิติไฟดับแล้ว")

# --- 6. MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "🔍 Diagnostics", "📅 Action Plan"])

with tab1:
    df = st.session_state.ktd_assets
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown(f"<div class='metric-card' style='border-top-color:#FF4B4B'><h4>วิกฤต (Critical)</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    col_m2.markdown(f"<div class='metric-card' style='border-top-color:#FF8C00'><h4>เฝ้าระวัง (Watch)</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    col_m3.markdown(f"<div class='metric-card' style='border-top-color:#28A745'><h4>ปกติ (Normal)</h4><h1>{len(df[df['Status'] == '🟢 NORMAL'])}</h1></div>", unsafe_allow_html=True)
    
    fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", zoom=13, height=500,
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    sel_id = st.selectbox("เลือก ID อุปกรณ์:", df['Transformer_ID'], key="diag_sel")
    res = df[df['Transformer_ID'] == sel_id].iloc[0]
    cl, cr = st.columns([1, 1.5])
    with cl:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, title={'text': "Risk Score (%)"},
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        st.plotly_chart(fig_g, use_container_width=True)
        st.info(f"📅 **แผนงานแนะนำ:** {res['Plan_Month']}")
    with cr:
        st.info("### รายละเอียดปัจจัยเสี่ยง")
        st.write(f"- **🌡️ ความร้อน:** {res['Thermal_Temp']} °C")
        st.write(f"- **🔊 เสียง:** {res['Acoustic_dB']} dB")
        st.write(f"- **📈 โหลดไฟฟ้า:** {res['Load_Percent']:.1f}%")
        st.write(f"- **📉 ประวัติไฟดับ:** {res['Trips_Count']} ครั้ง")

with tab3:
    st.header("📅 รายการแผนบำรุงรักษาเชิงป้องกัน (KTD Area)")
    urgent = df[df['Status'] != '🟢 NORMAL'].sort_values(by=['Status', 'Risk_Score'], ascending=[False, False])
    
    if urgent.empty:
        st.success("✅ อุปกรณ์ทุกตัวอยู่ในสถานะปกติ")
    else:
        for _, row in urgent.iterrows():
            card_style = "crit-border" if row['Status'] == '🔴 CRITICAL' else "watch-border"
            badge_color = "#FF4B4B" if row['Status'] == '🔴 CRITICAL' else "#FF8C00"
            
            st.markdown(f"""
                <div class="action-card {card_style}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.25em; font-weight: bold; color: #333;">ID: {row['Transformer_ID']}</span>
                        <span style="background-color: {badge_color};" class="status-badge">{row['Status']}</span>
                    </div>
                    <div style="margin-top: 10px; display: flex; justify-content: space-between; color: #666;">
                        <span><b>สายป้อน (Feeder):</b> {row['Feeder']}</span>
                        <span style="color: #FF8C00; font-weight: bold;">แผนงาน: {row['Plan_Month']}</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.9em; border-top: 1px solid #eee; padding-top: 8px;">
                        ความร้อน: <b>{row['Thermal_Temp']}°C</b> | เสียง: <b>{row['Acoustic_dB']}dB</b> | 
                        โหลด: <b>{row['Load_Percent']:.1f}%</b> | ความเสี่ยง: <b>{row['Risk_Score']*100:.1f}%</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
