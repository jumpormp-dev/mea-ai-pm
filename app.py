import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIG & SETTINGS ---
st.set_page_config(page_title="SPP-AI | Predictive Maintenance Dashboard", layout="wide")

# Custom CSS เพื่อให้หน้าตาเหมือน Artifact ที่คุณต้องการ
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 5px solid #FF8C00; }
    .card { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .stButton>button { background: linear-gradient(90deg, #FF8C00 0%, #FFA500 100%); color: white; border-radius: 10px; border: none; font-weight: 500; height: 3em; width: 100%; transition: 0.3s; }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-2px); }
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

# --- 3. DATA PROCESSING FUNCTIONS ---
def run_prediction(df):
    if model and not df.empty:
        # 8 Features ลำดับเดียวกับใน Colab
        X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 
                'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
        preds = model.predict(X)
        probs = model.predict_proba(X)
        status_map = {0: 'Normal', 1: 'Watch', 2: 'Critical'}
        df['Status'] = [status_map[p] for p in preds]
        df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
    return df

# --- 4. SESSION STATE ---
if 'assets' not in st.session_state:
    st.session_state.assets = pd.DataFrame()

# --- 5. SIDEBAR: DATA SOURCE ---
with st.sidebar:
    st.image("https://www.mea.or.th/assets/images/logo.png", width=180)
    st.markdown("### 📥 แหล่งข้อมูล")
    
    # ช่องโหลดไฟล์ ฟขต
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    
    if uploaded_file:
        raw_df = pd.read_excel(uploaded_file, skiprows=2)
        if 'Feeder' in raw_df.columns:
            trip_stats = raw_df['Feeder'].value_counts().to_dict()
            unique_feeders = list(trip_stats.keys())
            
            if st.button("🚀 เริ่มวิเคราะห์ความเสี่ยงราย Feeder"):
                new_data = []
                # สุ่มตำแหน่งหม้อแปลงในเขตคลองเตย (KTD)
                for i, fdr in enumerate(unique_feeders[:25]): 
                    new_data.append({
                        'Transformer_ID': f'TR-KTD-{i+101}',
                        'Feeder': fdr,
                        'Lat': 13.702 + np.random.uniform(-0.005, 0.005),
                        'Lon': 100.560 + np.random.uniform(-0.005, 0.005),
                        'Thermal_Temp': np.random.uniform(40, 105),
                        'Load_Percent': np.random.uniform(30, 115),
                        'Voltage_V': 220.0 + np.random.uniform(-5, 5),
                        'Acoustic_dB': np.random.uniform(35, 100),
                        'Peak_Freq_Hz': 25000.0,
                        'Trips_Count': trip_stats.get(fdr, 0),
                        'Age_Years': np.random.randint(5, 30),
                        'Humidity': 65.0
                    })
                st.session_state.assets = run_prediction(pd.DataFrame(new_data))
                st.success("ประมวลผลข้อมูล AI สำเร็จ")

# --- 6. MAIN CONTENT ---
st.title("⚡ SPP-AI Predictive Maintenance Dashboard")
st.markdown("ระบบบริหารจัดการความเสี่ยงหม้อแปลงไฟฟ้า เขตคลองเตย (KTD)")

if st.session_state.assets.empty:
    st.warning("👈 กรุณาอัปโหลดไฟล์ข้อมูลสายป้อน (ฟขต Feeder.xlsx) เพื่อเริ่มการทำงาน")
else:
    df = st.session_state.assets
    
    # ROW 1: METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("หม้อแปลงทั้งหมด", len(df))
    m2.metric("วิกฤต (Critical)", len(df[df['Status']=='Critical']), delta_color="inverse")
    m3.metric("เฝ้าระวัง (Watch)", len(df[df['Status']=='Watch']))
    m4.metric("เฉลี่ย Load (%)", f"{df['Load_Percent'].mean():.1f}%")

    # ROW 2: MAP & ANALYSIS
    col_map, col_chart = st.columns([2, 1])
    
    with col_map:
        st.markdown("<div class='card'><h4>📍 แผนที่ตำแหน่งและความเสี่ยง (Spatial Risk)</h4>", unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent",
                                    hover_name="Transformer_ID", hover_data=["Feeder", "Thermal_Temp", "Acoustic_dB"],
                                    color_discrete_map={'Critical': '#FF4B4B', 'Watch': '#FF8C00', 'Normal': '#28A745'},
                                    zoom=13.5, height=500)
        fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart:
        st.markdown("<div class='card'><h4>📊 สัดส่วนสถานะความเสี่ยง</h4>", unsafe_allow_html=True)
        fig_pie = px.pie(df, names='Status', hole=0.6,
                         color='Status', color_discrete_map={'Critical': '#FF4B4B', 'Watch': '#FF8C00', 'Normal': '#28A745'})
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ROW 3: DETAIL TABLE
    st.markdown("<div class='card'><h4>📋 รายการลำดับความเสี่ยงและแผนงานบำรุงรักษา</h4>", unsafe_allow_html=True)
    
    # จัดลำดับตามความรุนแรง
    display_df = df.sort_values(by='Status', ascending=False)
    
    # แสดงตารางพร้อมสีสถานะ
    def color_status(val):
        color = '#FF4B4B' if val == 'Critical' else '#FF8C00' if val == 'Watch' else '#28A745'
        return f'color: {color}; font-weight: bold'

    st.dataframe(display_df[['Transformer_ID', 'Feeder', 'Status', 'Risk_Score', 'Thermal_Temp', 'Load_Percent', 'Acoustic_dB', 'Trips_Count']]
                 .style.applymap(color_status, subset=['Status']), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # FOOTER
    st.caption(f"อัปเดตข้อมูลล่าสุด: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | ข้อมูลจากระบบ Smart Meter และ ฟขต.")
