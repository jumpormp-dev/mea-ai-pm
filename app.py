import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. CONFIG & STITCH UI STYLE ---
st.set_page_config(page_title="MEA Smart Plan - KTD Area", layout="wide")

# นำ CSS จาก Stitch มาปรับใช้ให้เข้ากับ Streamlit
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600&display=swap');
    
    html, body, [data-testid="stSidebar"], .main {
        font-family: 'Kanit', sans-serif;
        background-color: #F8F9FA;
    }

    /* สไตล์ Header แบบ Stitch */
    .stitch-header {
        background: linear-gradient(90deg, #904d00 0%, #ff8c00 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(144, 77, 0, 0.2);
    }

    /* Metric Card สไตล์ Bento Grid */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    .m-card {
        flex: 1;
        background: white;
        padding: 20px;
        border-radius: 16px;
        border-top: 5px solid #ff8c00;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
    }
    .m-card.crit { border-top-color: #ba1a1a; }
    .m-card h4 { color: #666; font-size: 0.9em; margin: 0; }
    .m-card h2 { color: #191c1d; font-size: 2em; margin: 5px 0; }

    /* Action Card สไตล์ Maintenance Plan ใน Stitch */
    .action-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 8px solid #ff8c00;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .crit-border { border-left-color: #ba1a1a; }
    
    /* ซ่อน Streamlit Elements บางส่วนเพื่อให้ดูเหมือนเว็บแอป */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL & LOGIC (เหมือนเดิม) ---
@st.cache_resource
def load_spp_model():
    try: return joblib.load('mea_spp_ai_model.pkl')
    except: return None

model = load_spp_model()

def calculate_plan_month(status, risk_score):
    today = datetime.now()
    if status == '🔴 CRITICAL': return today.strftime("%B %Y")
    elif status == '🟡 WATCH':
        delay = int(max(1, (1 - risk_score) * 4))
        return (today + timedelta(days=delay * 30)).strftime("%B %Y")
    return "Routine Check"

# Initial Data
if 'ktd_assets' not in st.session_state:
    st.session_state.ktd_assets = pd.DataFrame(columns=[
        'Transformer_ID', 'Feeder', 'Lat', 'Lon', 'Load_Percent', 'Voltage_V', 
        'Trips_Count', 'Acoustic_dB', 'Thermal_Temp', 'Peak_Freq_Hz', 'Age_Years', 
        'Humidity', 'Status', 'Risk_Score', 'Plan_Month', 'Survey_Img'
    ])

# --- 3. HEADER UI (Stitch Style) ---
st.markdown("""
    <div class="stitch-header">
        <h1 style='margin:0; font-size: 24px;'>⚡ MEA Smart Plan Predictive Maintenance</h1>
        <p style='margin:0; opacity: 0.8;'>KTD Area Intelligence - ระบบวิเคราะห์และวางแผนการบำรุงรักษา</p>
    </div>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR (ใส่ปุ่ม Record Field Survey สีส้มแบบ Stitch) ---
with st.sidebar:
    st.image("https://www.mea.or.th/assets/images/logo.png", width=100) # ตัวอย่าง Logo
    st.header("⚙️ Data Management")
    
    uploaded_xlsx = st.file_uploader("Upload XLSX (ฟขต Feeder)", type=["xlsx"])
    if uploaded_xlsx and st.button("🚀 Load Data"):
        # ... (Logic โหลดข้อมูลจากโค้ดเดิมของคุณ)
        pass

    st.divider()
    st.subheader("📸 Record Field Survey")
    # ส่วนนี้คือ Form สำหรับคีย์ข้อมูลหน้างานตาม UI Stitch
    with st.form("survey_form"):
        target_id = st.text_input("Asset ID (เช่น TR-KTD-042)")
        ac_val = st.number_input("Sound (dB)", value=45.0)
        th_val = st.number_input("Temp (°C)", value=50.0)
        submitted = st.form_submit_button("บันทึกข้อมูลและวิเคราะห์")
        if submitted:
            st.success("บันทึกข้อมูลสำเร็จ ระบบกำลังประมวลผล...")

# --- 5. MAIN CONTENT (Tabs) ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🔍 Asset Diagnostics", "📅 Maintenance Plan"])

with tab1:
    df = st.session_state.ktd_assets
    if not df.empty:
        # Bento Grid Metrics
        st.markdown(f"""
            <div class="metric-container">
                <div class="m-card"><h4>Total Assets</h4><h2>{len(df)}</h2></div>
                <div class="m-card crit"><h4>Critical</h4><h2>{len(df[df['Status'] == '🔴 CRITICAL'])}</h2></div>
                <div class="m-card"><h4>Watch List</h4><h2>{len(df[df['Status'] == '🟡 WATCH'])}</h2></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Real Map (Plotly)
        fig = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", zoom=12,
                                color_discrete_map={'🔴 CRITICAL': '#ba1a1a', '🟡 WATCH': '#ff8c00', '🟢 NORMAL': '#006e25'},
                                mapbox_style="carto-positron", height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📅 แผนบำรุงรักษาเชิงป้องกัน (KTD Action Plan)")
    # ดึงข้อมูลจาก Session State มาแสดงใน Card สไตล์ Stitch
    urgent_assets = st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] != '🟢 NORMAL']
    
    if urgent_assets.empty:
        st.write("ยังไม่มีข้อมูลอุปกรณ์ที่ต้องเฝ้าระวัง")
    else:
        for _, row in urgent_assets.iterrows():
            is_crit = "crit-border" if row['Status'] == '🔴 CRITICAL' else ""
            st.markdown(f"""
                <div class="action-card {is_crit}">
                    <div>
                        <div style="font-weight: bold; font-size: 1.2em;">{row['Transformer_ID']} <span style="font-size: 0.7em; font-weight: normal; color: #666;">({row['Feeder']})</span></div>
                        <div style="font-size: 0.9em; color: #444;">Temp: {row['Thermal_Temp']}°C | Sound: {row['Acoustic_dB']}dB</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #ff8c00; font-weight: bold;">แผนงาน: {row['Plan_Month']}</div>
                        <div style="font-size: 0.8em; color: #888;">สถานะ: {row['Status']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
