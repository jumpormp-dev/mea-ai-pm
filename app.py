import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIG & THEME (Dark Mode - MEA Style) ---
st.set_page_config(page_title="MEA Smart PM Dashboard", layout="wide")

# CSS สำหรับปรับหน้าตาให้เหมือน Google Stitch (ส้ม-ดำ-เทา)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500&display=swap');
    
    /* พื้นหลังและฟอนต์ */
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; background-color: #121212; color: #E0E0E0; }
    .stApp { background-color: #121212; }
    
    /* Navigation Tabs ด้านบน */
    .stTabs [data-baseweb="tab-list"] { background-color: #1E1E1E; padding: 10px; border-radius: 10px; gap: 15px; }
    .stTabs [data-baseweb="tab"] { color: #9E9E9E; border: none; font-size: 16px; }
    .stTabs [aria-selected="true"] { color: #FF8C00 !important; border-bottom: 3px solid #FF8C00 !important; font-weight: bold; }

    /* KPI Cards */
    .metric-card { background-color: #1E1E1E; padding: 25px; border-radius: 15px; border-left: 6px solid #FF8C00; box-shadow: 0 4px 15px rgba(0,0,0,0.3); text-align: center; }
    .val-text { font-size: 32px; font-weight: bold; color: #FFFFFF; }
    .label-text { color: #FF8C00; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }

    /* Action Cards ในหน้าแผนงาน */
    .action-card { background-color: #252525; padding: 20px; border-radius: 12px; margin-bottom: 15px; border-right: 8px solid #FF8C00; }
    .crit-card { border-right-color: #FF4B4B; }
    
    /* ปุ่มกด */
    .stButton>button { background: linear-gradient(90deg, #FF8C00 0%, #FF5500 100%); color: white; border-radius: 10px; border: none; font-weight: bold; height: 3.5em; width: 100%; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1E1E1E !important; border-right: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_spp_model():
    try: return joblib.load('mea_spp_ai_model.pkl')
    except: return None
model = load_spp_model()

# --- 3. LOGIC ---
def get_pm_month(status, risk):
    today = datetime.now()
    if status == '🔴 CRITICAL': return today.strftime("%B %Y")
    elif status == '🟡 WATCH':
        delay = int(max(1, (1 - risk) * 4))
        return (today + timedelta(days=delay * 30)).strftime("%B %Y")
    return "Routine (6M)"

# --- 4. DATA INITIALIZATION ---
if 'assets' not in st.session_state:
    st.session_state.assets = pd.DataFrame()
if 'survey_imgs' not in st.session_state:
    st.session_state.survey_imgs = {}

# --- 5. SIDEBAR: DATA & SURVEY ---
with st.sidebar:
    st.image("https://www.mea.or.th/assets/images/logo.png", width=180)
    st.markdown("<h3 style='color: #FF8C00;'>DATA MANAGEMENT</h3>", unsafe_allow_html=True)
    
    # โหลดไฟล์ ฟขต
    uploaded = st.file_uploader("ฟขต Feeder.xlsx", type=["xlsx"])
    if uploaded:
        df_raw = pd.read_excel(uploaded, skiprows=2)
        if st.button("🚀 IMPORT REAL DATA"):
            trip_map = df_raw['Feeder'].value_counts().to_dict()
            rows = []
            for i, (fdr, count) in enumerate(trip_map.items()):
                rows.append({
                    'Transformer_ID': f'TR-KTD-{i+101}', 'Feeder': fdr,
                    'Lat': 13.702 + np.random.uniform(-0.01, 0.01), 'Lon': 100.560 + np.random.uniform(-0.01, 0.01),
                    'Load_Percent': 0.0, 'Voltage_V': 220.0, 'Trips_Count': count,
                    'Acoustic_dB': 45.0, 'Thermal_Temp': 50.0, 'Peak_Freq_Hz': 25000.0,
                    'Age_Years': np.random.randint(5, 30), 'Humidity': 65.0,
                    'Status': '🟢 NORMAL', 'Risk_Score': 0.0, 'Plan_Month': '-'
                })
            st.session_state.assets = pd.DataFrame(rows)
            st.rerun()

    # Sync เว็บ
    if not st.session_state.assets.empty:
        if st.button("📡 SYNC SMART METER"):
            st.session_state.assets['Load_Percent'] = np.random.uniform(40, 115, len(st.session_state.assets))
            st.success("Synced 172.16.111.184")

    st.divider()
    # สำรวจหน้างาน
    if not st.session_state.assets.empty:
        st.subheader("FIELD SURVEY")
        target = st.selectbox("ID:", st.session_state.assets['Transformer_ID'])
        idx = st.session_state.assets[st.session_state.assets['Transformer_ID'] == target].index[0]
        ac = st.number_input("Acoustic (dB)", 30.0, 110.0, 45.0)
        th = st.number_input("Thermal (°C)", 20.0, 120.0, 50.0)
        img = st.file_uploader("Upload Image", type=["jpg", "png"])
        if st.button("💾 SAVE & ANALYZE"):
            st.session_state.assets.at[idx, 'Acoustic_dB'] = ac
            st.session_state.assets.at[idx, 'Thermal_Temp'] = th
            if img: st.session_state.survey_imgs[target] = img.read()
            st.success("Saved!")

# --- 6. MAIN UI TABS ---
st.markdown("<h2 style='color: #FF8C00;'>⚡ SMART PREDICTIVE MAINTENANCE</h2>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 OVERVIEW", "🔍 DIAGNOSTICS", "📅 PM PLAN"])

if st.session_state.assets.empty:
    st.info("👈 Please upload 'ฟขต Feeder.xlsx' to start.")
else:
    df = st.session_state.assets
    
    with tab1:
        # KPI ROW
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='label-text'>Total</div><div class='val-text'>{len(df)}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card' style='border-left-color:#FF4B4B'><div class='label-text'>Critical</div><div class='val-text'>{len(df[df['Status']=='🔴 CRITICAL'])}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='label-text'>Watch</div><div class='val-text'>{len(df[df['Status']=='🟡 WATCH'])}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card' style='border-left-color:#28A745'><div class='label-text'>Normal</div><div class='val-text'>{len(df[df['Status']=='🟢 NORMAL'])}</div></div>", unsafe_allow_html=True)
        
        # MAP
        fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent",
                                    color_discrete_map={'🔴 CRITICAL': '#FF4B4B', '🟡 WATCH': '#FF8C00', '🟢 NORMAL': '#28A745'},
                                    zoom=13, height=500, mapbox_style="carto-darkmatter")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#121212", plot_bgcolor="#121212")
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        sel_id = st.selectbox("Select Asset:", df['Transformer_ID'])
        row = df[df['Transformer_ID'] == sel_id].iloc[0]
        col_l, col_r = st.columns([1, 1.5])
        with col_l:
            if sel_id in st.session_state.survey_imgs:
                st.image(st.session_state.survey_imgs[sel_id], caption=f"Site Image: {sel_id}")
            else: st.warning("No site image uploaded.")
        with col_r:
            st.markdown(f"### Diagnostics: {sel_id}")
            st.write(f"**PM Schedule:** {row['Plan_Month']}")
            st.progress(row['Risk_Score'])
            st.write(f"Load: {row['Load_Percent']:.1f}% | Thermal: {row['Thermal_Temp']}°C | Acoustic: {row['Acoustic_dB']}dB")

    with tab3:
        st.markdown("### Monthly Maintenance Schedule")
        if st.button("🚀 GENERATE PM PLAN WITH AI"):
            X = df[['Thermal_Temp', 'Load_Percent', 'Voltage_V', 'Acoustic_dB', 'Peak_Freq_Hz', 'Trips_Count', 'Age_Years', 'Humidity']].values
            if model:
                preds = model.predict(X)
                probs = model.predict_proba(X) if hasattr(model, "predict_proba") else [[0.5]*3]*len(preds)
                df['Status'] = [{0:'🟢 NORMAL', 1:'🟡 WATCH', 2:'🔴 CRITICAL'}[p] for p in preds]
                df['Risk_Score'] = [probs[i][preds[i]] for i in range(len(preds))]
                df['Plan_Month'] = df.apply(lambda r: get_pm_month(r['Status'], r['Risk_Score']), axis=1)
                st.session_state.assets = df
                st.rerun()

        urgent = df[df['Status'] != '🟢 NORMAL'].sort_values('Risk_Score', ascending=False)
        for _, r in urgent.iterrows():
            cls = "crit-card" if r['Status'] == '🔴 CRITICAL' else ""
            st.markdown(f"""
                <div class='action-card {cls}'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='font-size:1.2em; font-weight:bold;'>{r['Transformer_ID']}</span>
                        <span style='color:#FF8C00;'>PLAN: {r['Plan_Month']}</span>
                    </div>
                    <div style='color:#AAA; font-size:0.9em;'>Feeder: {r['Feeder']} | Status: {r['Status']}</div>
                </div>
                """, unsafe_allow_html=True)
