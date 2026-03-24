import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. SETTINGS & STYLE ---
st.set_page_config(page_title="SPP-AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F0F2F6; }
    .stButton>button { background-color: #FF8C00; color: white; border-radius: 8px; height: 3em; width: 100%; border: none; font-weight: bold; }
    .metric-card { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD AI MODEL (.pkl) ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('mea_spp_ai_model.pkl')
    except:
        return None

model = load_model()

# --- 3. DATABASE INITIALIZATION (KTD Area) ---
if 'ktd_assets' not in st.session_state:
    # สร้างข้อมูลตั้งต้น 20 เครื่องที่มีค่าสุ่มหลากหลายเพื่อให้ AI วิเคราะห์เห็นผลต่าง
    np.random.seed(42)
    feeders = ['EM-418', 'SAM-13', 'PI-435', 'NS-436', 'SA-411', 'LN-442', 'RPR-423']
    
    st.session_state.ktd_assets = pd.DataFrame({
        'Transformer_ID': [f'TR-KTD-{i+1:03d}' for i in range(20)],
        'Feeder': [np.random.choice(feeders) for _ in range(20)],
        'Lat': np.random.uniform(13.702, 13.715, 20),
        'Lon': np.random.uniform(100.555, 100.575, 20),
        # 8 Features หลัก (ต้องตรงกับตอนเทรน)
        'Thermal_Temp': np.random.uniform(40, 100, 20),
        'Load_Percent': np.random.uniform(30, 110, 20),
        'Voltage_V': np.random.uniform(215, 235, 20),
        'Acoustic_dB': np.random.uniform(35, 95, 20),
        'Peak_Freq_Hz': [25000.0] * 20,
        'Trips_Count': np.random.randint(0, 12, 20),
        'Age_Years': np.random.randint(1, 35, 20),
        'Humidity': [65.0] * 20,
        # สถานะการแสดงผล
        'Risk_Score': [0.0] * 20,
        'Status': ['🟢 NORMAL'] * 20
    })

# --- 4. SIDEBAR & CONTROLS ---
with st.sidebar:
    st.image("https://www.mea.or.th/assets/images/logo.png", width=150) # หรือใส่โลโก้โครงการ
    st.header("⚙️ ระบบจัดการข้อมูล")
    
    # ปุ่มวิเคราะห์ภาพรวม (หัวใจสำคัญ)
    if st.button("🚀 วิเคราะห์ความเสี่ยงทั้งหมด (AI Bulk)"):
        if model:
            df = st.session_state.ktd_assets
            # เตรียม Features 8 ตัวตามลำดับ Colab
            X_input = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 
                          'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
            
            predictions = model.predict(X_input)
            probs = model.predict_proba(X_input)
            
            status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
            
            for idx in range(len(df)):
                res_idx = predictions[idx]
                st.session_state.ktd_assets.at[idx, 'Status'] = status_map[res_idx]
                st.session_state.ktd_assets.at[idx, 'Risk_Score'] = probs[idx][res_idx]
                
            st.success("AI ประมวลผลสำเร็จ!")
            st.rerun()
        else:
            st.error("ไม่พบไฟล์โมเดล .pkl")

    st.divider()
    st.subheader("📝 บันทึกผลสำรวจรายเครื่อง")
    target_id = st.selectbox("เลือกหม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    new_ac = st.slider("Acoustic (dB)", 30, 110, 50)
    new_th = st.slider("Thermal (°C)", 30, 120, 55)
    
    if st.button("💾 อัปเดตและวิเคราะห์รายเครื่อง"):
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].index[0]
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = new_ac
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = new_th
        # รัน AI เฉพาะตัวที่เลือก
        row = st.session_state.ktd_assets.iloc[idx]
        feat = np.array([[row['Thermal_Temp'], row['Load_Percent'], row['Voltage_V'], 
                          row['Acoustic_dB'], row['Peak_Freq_Hz'], row['Trips_Count'], 
                          row['Age_Years'], row['Humidity']]])
        res = model.predict(feat)[0]
        st.session_state.ktd_assets.at[idx, 'Status'] = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}[res]
        st.success(f"อัปเดต {target_id} เรียบร้อย")
        st.rerun()

# --- 5. MAIN DASHBOARD ---
st.title("⚡ SPP-AI: Smart Predictive Maintenance")
st.caption("ระบบวิเคราะห์และพยากรณ์ความเสี่ยงหม้อแปลงไฟฟ้า เขตคลองเตย (KTD)")

tab1, tab2, tab3 = st.tabs(["📊 สรุปภาพรวม", "📍 แผนที่ความเสี่ยง", "📅 แผนงานบำรุงรักษา"])

with tab1:
    df = st.session_state.ktd_assets
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("จำนวนทั้งหมด", len(df))
    col2.metric("วิกฤต (Critical)", len(df[df['Status'] == '🔴 CRITICAL']), delta_color="inverse")
    col3.metric("เฝ้าระวัง (Watch)", len(df[df['Status'] == '🟡 WATCH']))
    col4.metric("ปกติ (Normal)", len(df[df['Status'] == '🟢 NORMAL']))

    # กราฟแท่งแสดงจำนวนสถานะ
    fig_status = px.bar(df['Status'].value_counts().reset_index(), x='Status', y='count', 
                        color='Status', color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'})
    st.plotly_chart(fig_status, use_container_width=True)

with tab2:
    fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent",
                                hover_name="Transformer_ID", hover_data=["Feeder", "Thermal_Temp", "Acoustic_dB"],
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                zoom=14, height=600)
    fig_map.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("📋 รายการที่ต้องดำเนินการเร่งด่วน")
    urgent_df = df[df['Status'] != '🟢 NORMAL'].sort_values(by='Status', ascending=False)
    if not urgent_df.empty:
        st.table(urgent_df[['Transformer_ID', 'Feeder', 'Status', 'Thermal_Temp', 'Acoustic_dB', 'Load_Percent']])
    else:
        st.write("ยังไม่มีรายการวิกฤตในขณะนี้")
