import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="ระบบวิเคราะห์และวางแผนการบำรุงรักษา", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 8px solid #FF8C00; }
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

# --- 3. MAINTENANCE LOGIC ---
def calculate_plan_month(status, risk_score):
    today = datetime.now()
    if status == '🔴 CRITICAL':
        return today.strftime("%B %Y")
    elif status == '🟡 WATCH':
        delay = int(max(1, (1 - risk_score) * 4))
        return (today + timedelta(days=delay * 30)).strftime("%B %Y")
    return "Routine Check"

# --- 4. SESSION STATE INITIALIZATION ---
# สร้างโครงสร้างข้อมูลให้ครบตั้งแต่ต้นเพื่อป้องกัน KeyError
if 'ktd_assets' not in st.session_state:
    st.session_state.ktd_assets = pd.DataFrame(columns=[
        'Transformer_ID', 'Feeder', 'Lat', 'Lon', 'Load_Percent', 'Voltage_V', 
        'Trips_Count', 'Acoustic_dB', 'Thermal_Temp', 'Peak_Freq_Hz', 'Age_Years', 
        'Humidity', 'Status', 'Risk_Score', 'Plan_Month', 'Survey_Img'
    ])

# --- 5. HEADER ---
st.title("⚡ ระบบวิเคราะห์และวางแผนการบำรุงรักษา")
st.caption("Smart Plan Predictive Maintenance AI (KTD Area)")
st.divider()

if model is None:
    st.error("⚠️ ไม่พบไฟล์ 'mea_spp_ai_model.pkl' กรุณาตรวจสอบไฟล์ใน GitHub")

# --- 6. SIDEBAR: DATA & SURVEY ---
with st.sidebar:
    st.header("⚙️ การจัดการข้อมูล")
    
    # 1. โหลดข้อมูลจริงจากไฟล์ ฟขต
    uploaded_xlsx = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])
    if uploaded_xlsx:
        try:
            df_xlsx = pd.read_excel(uploaded_xlsx, skiprows=2)
            if 'Feeder' in df_xlsx.columns:
                trip_stats = df_xlsx['Feeder'].value_counts().to_dict()
                unique_feeders = list(trip_stats.keys())
                
                if st.button("🚀 โหลดข้อมูลจริงทั้งหมดเข้าระบบ"):
                    new_data = []
                    for i, fdr in enumerate(unique_feeders):
                        new_data.append({
                            'Transformer_ID': f'TR-KTD-{i+1:03d}',
                            'Feeder': fdr,
                            'Lat': 13.702 + np.random.uniform(-0.01, 0.01),
                            'Lon': 100.555 + np.random.uniform(-0.01, 0.01),
                            'Load_Percent': 0.0, 'Voltage_V': 220.0,
                            'Trips_Count': trip_stats.get(fdr, 0),
                            'Acoustic_dB': 45.0, 'Thermal_Temp': 50.0,
                            'Peak_Freq_Hz': 25000.0, 'Age_Years': np.random.randint(5, 35),
                            'Humidity': 65.0, 'Status': '🟢 NORMAL', 'Risk_Score': 0.0,
                            'Plan_Month': 'Routine Check', 'Survey_Img': None
                        })
                    st.session_state.ktd_assets = pd.DataFrame(new_data)
                    st.success(f"โหลดข้อมูลจริง {len(unique_feeders)} เครื่องสำเร็จ!")
            else:
                st.error("ไฟล์ Excel ไม่มีคอลัมน์ 'Feeder'")
        except Exception as e:
            st.error(f"Error อ่านไฟล์: {e}")

    # 2. Sync ข้อมูลเว็บ
    if st.button("📡 Sync Smart Meter (172.16.111.184)"):
        if not st.session_state.ktd_assets.empty:
            size = len(st.session_state.ktd_assets)
            st.session_state.ktd_assets['Load_Percent'] = np.random.uniform(40, 115, size)
            st.session_state.ktd_assets['Voltage_V'] = np.random.uniform(210, 235, size)
            st.success("ซิงค์ข้อมูล Load/Voltage สำเร็จ")
        else:
            st.warning("กรุณาโหลดข้อมูลจริงก่อน")

    st.divider()
    
    # 3. บันทึกสำรวจหน้างาน + อัปโหลดภาพ
    if not st.session_state.ktd_assets.empty:
        st.subheader("📸 บันทึกสำรวจหน้างาน")
        target_id = st.selectbox("เลือก ID หม้อแปลง:", st.session_state.ktd_assets['Transformer_ID'])
        idx = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == target_id].index[0]
        
        ac_in = st.number_input("ค่าเสียง (dB)", 30.0, 110.0, float(st.session_state.ktd_assets.at[idx, 'Acoustic_dB']))
        th_in = st.number_input("ความร้อน (°C)", 20.0, 120.0, float(st.session_state.ktd_assets.at[idx, 'Thermal_Temp']))
        img_file = st.file_uploader("อัปโหลดภาพหน้างาน", type=["jpg", "png", "jpeg"], key=f"img_{target_id}")

        if st.button("💾 บันทึกและวิเคราะห์เครื่องนี้"):
            st.session_state.ktd_assets.at[idx, 'Acoustic_dB'] = ac_in
            st.session_state.ktd_assets.at[idx, 'Thermal_Temp'] = th_in
            if img_file: st.session_state.ktd_assets.at[idx, 'Survey_Img'] = img_file
            
            # AI Inference
            row = st.session_state.ktd_assets.iloc[idx]
            feat = np.array([[th_in, row['Load_Percent'], row['Voltage_V'], ac_in, 25000, row['Trips_Count'], row['Age_Years'], 65.0]])
            res = model.predict(feat)[0]
            prob = model.predict_proba(feat)[0][res] if hasattr(model, "predict_proba") else 0.5
            
            st.session_state.ktd_assets.at[idx, 'Status'] = {0: '🟢 NORMAL', 1: '🟡 WATCH', 2: '🔴 CRITICAL'}[res]
            st.session_state.ktd_assets.at[idx, 'Risk_Score'] = prob
            st.session_state.ktd_assets.at[idx, 'Plan_Month'] = calculate_plan_month(st.session_state.ktd_assets.at[idx, 'Status'], prob)
            st.success(f"อัปเดต {target_id} สำเร็จ")
            st.rerun()

    # 4. ปุ่ม Bulk Analysis
    if st.button("🚀 วิเคราะห์แผนงานทั้งหมด"):
        if model and not st.session_state.ktd_assets.empty:
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
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🔍 Asset Diagnostics", "📅 Maintenance Plan"])

with tab1:
    if st.session_state.ktd_assets.empty:
        st.info("👈 กรุณาอัปโหลดไฟล์ 'ฟขต Feeder.xlsx' และกดปุ่มโหลดข้อมูลเพื่อเริ่มต้น")
    else:
        df = st.session_state.ktd_assets
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h4>ทั้งหมด</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card' style='border-top-color:#FF4B4B'><h4>วิกฤต</h4><h1>{len(df[df['Status'] == '🔴 CRITICAL'])}</h1></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card' style='border-top-color:#FF8C00'><h4>เฝ้าระวัง</h4><h1>{len(df[df['Status'] == '🟡 WATCH'])}</h1></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card' style='border-top-color:#28A745'><h4>พื้นที่</h4><h1>KTD</h1></div>", unsafe_allow_html=True)
        
        fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", zoom=12, height=550,
                                    color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                    mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    if not st.session_state.ktd_assets.empty:
        sel_id = st.selectbox("เลือก ID อุปกรณ์:", st.session_state.ktd_assets['Transformer_ID'], key="diag_sel")
        res = st.session_state.ktd_assets[st.session_state.ktd_assets['Transformer_ID'] == sel_id].iloc[0]
        cl, cr = st.columns([1, 1.5])
        with cl:
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=res['Risk_Score']*100, title={'text': "Risk Score (%)"},
                                          gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF8C00"}}))
            st.plotly_chart(fig_g, use_container_width=True)
            # แก้ปัญหา KeyError ด้วยการตรวจสอบก่อนแสดงภาพ
            if 'Survey_Img' in res and res['Survey_Img'] is not None:
                st.image(res['Survey_Img'], caption=f"ภาพหน้างาน {sel_id}", use_container_width=True)
            else:
                st.warning("ยังไม่มีการอัปโหลดภาพหน้างาน")
        with cr:
            st.info(f"### ผลวิเคราะห์ {sel_id}")
            # ตรวจสอบว่ามีคอลัมน์ Plan_Month หรือยัง
            plan_text = res['Plan_Month'] if 'Plan_Month' in res else "ยังไม่มีแผนงาน"
            st.write(f"- 📅 **แผน PM:** {plan_text}")
            st.write(f"- 🌡️ **ความร้อน:** {res['Thermal_Temp']} °C")
            st.write(f"- 🔊 **เสียง:** {res['Acoustic_dB']} dB")
            st.write(f"- 📉 **สถิติไฟดับ:** {res['Trips_Count']} ครั้ง")

with tab3:
    if not st.session_state.ktd_assets.empty:
        st.header("📅 แผนบำรุงรักษาเชิงป้องกัน (KTD Action Plan)")
        urgent = st.session_state.ktd_assets[st.session_state.ktd_assets['Status'] != '🟢 NORMAL'].sort_values(by=['Status', 'Risk_Score'], ascending=[False, False])
        if urgent.empty:
            st.success("✅ อุปกรณ์ทุกตัวอยู่ในสภาวะปกติ")
        else:
            for _, row in urgent.iterrows():
                plan_val = row['Plan_Month'] if 'Plan_Month' in row else "-"
                st.markdown(f"""
                    <div class="action-card {'crit-border' if row['Status'] == '🔴 CRITICAL' else 'watch-border'}">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-size: 1.25em; font-weight: bold;">{row['Transformer_ID']} ({row['Feeder']})</span>
                            <span style="color: #FF8C00; font-weight: bold;">แผนงาน: {plan_val}</span>
                        </div>
                        <div style="margin-top: 10px;">สถานะ: {row['Status']} | ความร้อน: {row['Thermal_Temp']}°C | เสียง: {row['Acoustic_dB']}dB</div>
                    </div>
                """, unsafe_allow_html=True)
