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
    today = datetime.now()
    if status == '🔴 CRITICAL':
        return today.strftime("%B %Y")
    elif status == '🟡 WATCH':
        delay_months = int(max(1, (1 - risk_score) * 4))
        target_date = today + timedelta(days=delay_months * 30)
        return target_date.strftime("%B %Y")
    return "Routine (6-12 Months)"

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
st.caption("Smart Plan Predictive Maintenance AI (KTD Area)")
st.divider()

if model is None:
    st.error("⚠️ ไม่พบไฟล์ 'mea_spp_ai_model.pkl' กรุณาตรวจสอบไฟล์ในโฟลเดอร์")

# --- 6. SIDEBAR: DATA & SURVEY ---
with st.sidebar:
    st.header("⚙️ การจัดการข้อมูล")
    
    # 1. ซิงค์เว็บ
    if st.button("📡 Sync Smart Meter (172.16.111.184)"):
        st.session_state.ktd_assets['Load_Percent'] = np.random.uniform(40, 115, 20)
        st.session_state.ktd_assets['Voltage_V'] = np.random.uniform(210, 235, 20)
        st.success("ซิงค์ข้อมูล Load/Voltage สำเร็จ")

    # 2. อัปโหลดไฟล์ ฟขต
    uploaded_xlsx = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    if uploaded_xlsx:
        df_xlsx = pd.read_excel(uploaded_xlsx, skiprows=2)
        trip_map = df_xlsx['Feeder'].value_counts().to_dict()
        for i, row in st.session_state.ktd_assets.iterrows():
            st.session_state.ktd_assets.at[i, 'Trips_Count'] = trip_map.get(row['Feeder'], 0)
        st.success("อัปเดตสถิติไฟดับสำเร็จ")

    st.divider()
    
    # 3. ช่องกรอกข้อมูลสำรวจหน้างาน (Acoustic & Thermal)
    st.subheader("📸 บันทึกข้อมูลสำรวจหน้างาน")
    target_id = st.selectbox("เลือก ID หม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
    idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].index[0]
    
    # ช่องใส่ค่าจาก Acoustic Camera และ Thermal Scan
    ac_val = st.number_input("ค่าเสียง Acoustic Camera (dB)", 30.0, 110.0, float(st.session_state.ktd_assets.at[idx, 'Acoustic_dB']))
    th_val = st.number_input("ความร้อน Thermal Scan (°C)", 20.0, 120.0, float(st.session_state.ktd_assets.at[idx, 'Thermal_Temp']))

    if st.button("💾 บันทึกและวิเคราะห์เฉพาะเครื่อง"):
        st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_val
        st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_val
        
        # รัน AI รายตัว
        row = st.session_state.ktd_assets.iloc[idx]
        feat = np.array([[th_val, row['Load_Percent'], row['Voltage_V'], ac_val, 25000, row['Trips_Count'], row['Age_Years'], 65.0]])
        res = model.predict(feat)[0]
        prob = model.predict_proba(feat)[0][res] if hasattr(model, "predict_proba") else 0.5
        
        status_map = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}
        st.session_state.ktd_assets.at[idx, 'Status'] = status_map[res]
        st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
        st.session_state.ktd_assets.at[idx, 'Plan_Month'] = calculate_plan_month(status_map[res], prob)
        st.success(f"อัปเดตข้อมูล {target_id} และคำนวณแผน PM สำเร็จ")
        st.rerun()

    st.divider()
    # 4. ปุ่มวิเคราะห์ Bulk
    if st.button("🚀 วิเคราะห์และสร้างแผน PM ทั้งหมด"):
        if model:
            df = st.session_state.ktd_assets
            X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
            preds = model.predict(X)
            probs = model.predict_proba(X) if hasattr(model, "predict_proba") else [[0.5]*3]*len(preds)
            
            df['Status'] = [{0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}[p] for p in preds]
            df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
            df['Plan_Month'] = df.apply(lambda r: calculate_plan_month(r['Status'], r['Risk_Score']), axis=1)
            st.session_state.ktd_assets = df
            st.success("สร้างแผนงานบำรุงรักษาภาพรวมสำเร็จ")
            st.rerun()

# --- 7. MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🔍 Asset Diagnostics", "📅 Maintenance Plan (PM)"])

with tab1:
    df = st.session_state.ktd_assets
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h4>ทั้งหมด</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card metric-crit'><h4>วิกฤต</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card metric-watch'><h4>เฝ้าระวัง</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card metric-normal'><h4>ปกติ</h4><h1>{len(df[df['Status'] == '🟢 NORMAL'])}</h1></div>", unsafe_allow_html=True)
    
    fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", zoom=13, height=500,
                                color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    sel_id = st.selectbox("เลือก ID อุปกรณ์:", df['Transformer_ID'], key="diag_sel")
    res = df[df['Transformer_ID'] == sel_id].iloc[0]
    cl, cr = st.columns([1, 1.5])
    with cl:
        val = res['Risk_Score'] * 100
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': "Risk Score (%)"},
                                      gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
        st.plotly_chart(fig_g, use_container_width=True)
        st.info(f"📅 **แผนงาน PM:** {res['Plan_Month']}")
    with cr:
        st.info("### รายละเอียดปัจจัยที่ AI ใช้ประมวลผล")
        st.write(f"- 🌡️ **ความร้อน (Thermal):** {res['Thermal_Temp']} °C")
        st.write(f"- 🔊 **เสียง (Acoustic):** {res['Acoustic_dB']} dB")
        st.write(f"- 📈 **ภาระไฟฟ้า (Load):** {res['Load_Percent']:.1f}%")
        st.write(f"- 📉 **สถิติไฟดับ (Trips):** {res['Trips_Count']} ครั้ง")

with tab3:
    st.header("📅 รายการแผนงานบำรุงรักษาเชิงป้องกัน (Action Plan)")
    urgent = df[df['Status'] != '🟢 NORMAL'].sort_values(by=['Status', 'Risk_Score'], ascending=[False, False])
    
    if urgent.empty:
        st.success("✅ อุปกรณ์ทุกตัวปกติ ยังไม่มีรายการที่ต้องเข้าบำรุงรักษาเร่งด่วน")
    else:
        for _, row in urgent.iterrows():
            border_class = "crit-border" if row['Status'] == '🔴 CRITICAL' else "watch-border"
            st.markdown(f"""
                <div class="action-card {border_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.25em; font-weight: bold;">ID: {row['Transformer_ID']}</span>
                        <span style="font-size: 1.1em; color: #FF8C00; font-weight: bold;">แผนงาน: {row['Plan_Month']}</span>
                    </div>
                    <div style="margin-top: 10px; color: #666;">
                        สายป้อน: <b>{row['Feeder']}</b> | สถานะ: <b>{row['Status']}</b> | ความเสี่ยง: <b>{row['Risk_Score']*100:.1f}%</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
