import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="SPP-AI: KTD Smart City", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F4F7F9; }
    .stMetric { background-color: #FFFFFF; padding: 25px; border-radius: 15px; border-left: 8px solid #FF8C00; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background-color: #FFFFFF; border-radius: 10px; border: 1px solid #E0E0E0; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; height: 3em; }
    h1, h2, h3 { color: #333333; font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_spp_model():
    return joblib.load('mea_spp_ai_model.pkl')

try:
    model = load_spp_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'mea_spp_ai_model.pkl' บน GitHub")

# --- 3. INITIAL DATABASE (ป้องกัน KeyError) ---
if 'ktd_assets' not in st.session_state:
    ktd_feeders = ['EM-418', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'SAM-13', 'RPR-423'] * 3
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i:03d}' for i in range(1, 21)],
        'Feeder': ktd_feeders[:20],
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        'Load_Meter': [0.0]*20,
        'Trips_KTD': [0]*20,
        'Risk_Score': [0.1]*20,
        'Status': ['🟢 NORMAL']*20,
        'Acoustic_dB': [45.0]*20,
        'Thermal_Temp': [55.0]*20,  # เพิ่ม Column รองรับข้อมูลเทอร์โม
        'Age': np.random.randint(5, 30, 20),
        'Last_Survey_Img': [None]*20
    })

# --- 4. SIDEBAR (Data Input) ---
with st.sidebar:
    st.header("📥 Data Management")
    
    # 4.1 Sync Data
    if st.button("📡 Sync Real-time Load"):
        st.session_state.ktd_assets['Load_Meter'] = np.random.uniform(40, 110, 20)
        st.success("Load Data Synced")
    
    feeder_file = st.file_uploader("Upload Reliability Data", type=["xlsx"])
    if feeder_file:
        df_f = pd.read_excel(feeder_file, skiprows=2)
        counts = df_f['Feeder'].value_counts().to_dict()
        for fid, c in counts.items():
            st.session_state.ktd_assets.loc[st.session_state.ktd_assets['Feeder'] == fid, 'Trips_KTD'] = c
        st.success("Reliability Updated")

    st.divider()
    
    # 4.2 Field Survey (Acoustic Camera & Thermal Scan)
    st.subheader("📸 Field Survey Data")
    target = st.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    
    c_in1, c_in2 = st.columns(2)
    with c_in1:
        ac_val = st.number_input("เสียง (dB)", 30.0, 120.0, 45.0)
    with c_in2:
        th_val = st.number_input("ความร้อน (°C)", 20.0, 120.0, 55.0)
    
    survey_img = st.file_uploader("📷 แนบรูปภาพหน้างาน", type=["jpg", "png", "jpeg"])
    if survey_img:
        st.image(survey_img, caption="Preview ภาพหน้างาน", use_container_width=True)

    if st.button("🧠 รันการวิเคราะห์ AI"):
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target].index[0]
        row = st.session_state.ktd_assets.iloc[idx]
        
        # รัน AI Model (ใช้อุณหภูมิ Thermal แทน)
        features = np.array([[th_val, row['Load_Meter'], 230.0, ac_val, 20000, row['Trips_KTD'], row['Age'], 55.0]])
        prob = model.predict_proba(features)[0][1]
        
        # Update session state
        st.session_state.ktd_assets.at[idx, 'Status'] = "🔴 CRITICAL" if (prob > 0.75 or row['Trips_KTD'] >= 8) else "🟡 WATCH" if (prob > 0.4) else "🟢 NORMAL"
        st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_val
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_val
        if survey_img:
            st.session_state.ktd_assets.at[idx, 'Last_Survey_Img'] = survey_img
        st.success(f"วิเคราะห์ {target} สำเร็จ!")

# --- 5. MAIN DASHBOARD ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive", "🔍 Diagnostics", "📅 Action Plan", "⚙️ Settings"])

with tab1:
    st.markdown("## 🏙️ SPP-AI: Executive Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    df = st.session_state.ktd_assets
    m1.metric("Total Assets", "20 Units")
    m2.metric("Critical", len(df[df['Status'] == "🔴 CRITICAL"]))
    m3.metric("Watch", len(df[df['Status'] == "🟡 WATCH"]))
    m4.metric("KTD Status", "Online")

    c_map, c_list = st.columns([2, 1])
    with c_map:
        fig = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Risk_Score", zoom=13, height=450,
                                color_discrete_map={'🔴 CRITICAL': 'red', '🟡 WATCH': 'orange', '🟢 NORMAL': 'green'},
                                mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)
    with c_list:
        st.write("### 🚨 Urgent Attention")
        st.dataframe(df.sort_values('Risk_Score', ascending=False)[['Transformer_ID', 'Status']].head(8), hide_index=True)

with tab2:
    st.markdown("## 🔍 Deep Diagnostics")
    sel_id = st.selectbox("เลือกหม้อแปลงเพื่อดูข้อมูล:", df['Transformer_ID'])
    res = df[df['Transformer_ID'] == sel_id].iloc[0]

    col_a, col_b, col_c = st.columns([1, 1, 1.5])
    with col_a:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, 
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        fig_g.update_layout(height=280)
        st.plotly_chart(fig_g, use_container_width=True)
    with col_b:
        days = int(max(2, (1 - res['Risk_Score']) * 90))
        target_month = (datetime.now() + timedelta(days=days)).strftime('%B %Y')
        st.markdown(f"<div style='text-align: center; border: 2px solid #FF8C00; padding: 20px; border-radius: 10px;'><h3>Target Month</h3><h1 style='color: #FF8C00;'>{target_month}</h1></div>", unsafe_allow_html=True)
    with col_c:
        st.info("#### AI Insight")
        st.write(f"**🔊 Acoustic:** {res['Acoustic_dB']} dB | **🔥 Thermal:** {res['Thermal_Temp']} °C")
        if res['Last_Survey_Img'] is not None:
            st.image(res['Last_Survey_Img'], caption="รูปหลักฐานหน้างานล่าสุด", width=250)

with tab3:
    st.markdown("## 📅 PM Action Plan")
    for _, row in df[df['Status'] != '🟢 NORMAL'].iterrows():
        # คำนวณเดือนรายตัวเพื่อให้ตรงตามความเสี่ยง
        d = int(max(2, (1 - row['Risk_Score']) * 90))
        m = (datetime.now() + timedelta(days=d)).strftime('%B %Y')
        st.markdown(f"""<div style='background-color: white; padding: 15px; border-radius: 10px; border-left: 10px solid #FF8C00; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);'>
            <b>{row['Transformer_ID']}</b> | Status: {row['Status']} | <b>แผนงานเดือน: {m}</b></div>""", unsafe_allow_html=True)
