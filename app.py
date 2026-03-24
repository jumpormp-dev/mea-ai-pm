import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. SET PAGE CONFIG (Wide & Modern) ---
st.set_page_config(page_title="MEA Smart Plan | AI Diagnostics", layout="wide", initial_sidebar_state="expanded")

# --- 2. THE PERFECT STITCH UI ENGINE (CSS & TAILWIND) ---
# เราจะฉีด Tailwind และ Custom CSS เพื่อลบความเป็น Streamlit ออกให้มากที่สุด
st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8F9FA !important;
        font-family: 'Inter', 'Kanit', sans-serif !important;
    }
    
    /* Hide Streamlit Garbage */
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E5E7EB !important; }
    .block-container { padding: 1.5rem 3rem !important; max-width: 100% !important; }

    /* Stitch Card Components */
    .bento-card {
        background: white;
        border-radius: 1.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0,0,0,0.03);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .bento-card:hover { transform: translateY(-4px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08); }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: transparent !important;
        border: none !important; font-weight: 600 !important; color: #6B7280 !important;
    }
    .stTabs [aria-selected="true"] { color: #904D00 !important; border-bottom: 3px solid #FF8C00 !important; }

    /* Custom Buttons */
    .btn-primary {
        background: linear-gradient(135deg, #904D00 0%, #FF8C00 100%);
        color: white; padding: 12px 24px; border-radius: 12px;
        font-weight: 700; text-align: center; cursor: pointer;
        box-shadow: 0 10px 15px -3px rgba(144, 77, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE AI LOGIC (Keep it simple & stable) ---
@st.cache_resource
def load_model():
    try: return joblib.load('mea_spp_ai_model.pkl')
    except: return None

model = load_model()

# Session State
if 'ktd_assets' not in st.session_state:
    # สร้างข้อมูล Dummy ให้เห็น UI ก่อน ถ้ายังไม่ได้โหลดไฟล์จริง
    data = []
    for i in range(10):
        status = np.random.choice(['🔴 CRITICAL', '🟡 WATCH', '🟢 NORMAL'], p=[0.1, 0.2, 0.7])
        data.append({
            'Transformer_ID': f'TR-KTD-{i+1:03d}', 'Feeder': f'FDR-{np.random.randint(100,999)}',
            'Status': status, 'Risk_Score': np.random.uniform(0.1, 0.9), 'Plan_Month': 'July 2026',
            'Thermal_Temp': np.random.randint(40, 95), 'Acoustic_dB': np.random.randint(40, 100),
            'Lat': 13.75 + np.random.uniform(-0.02, 0.02), 'Lon': 100.5 + np.random.uniform(-0.02, 0.02),
            'Load_Percent': np.random.randint(30, 120)
        })
    st.session_state.ktd_assets = pd.DataFrame(data)

# --- 4. TOP NAVIGATION BAR (Fixed-style Header) ---
st.markdown("""
    <div class="flex justify-between items-center mb-10">
        <div>
            <h1 class="text-3xl font-extrabold tracking-tight text-slate-900 font-['Kanit']">⚡ MEA Smart Plan</h1>
            <p class="text-slate-500 font-medium">Predictive Maintenance AI Intelligence Suite</p>
        </div>
        <div class="flex gap-4">
            <div class="bg-white px-6 py-2 rounded-2xl shadow-sm border border-gray-100 flex items-center gap-3">
                <span class="w-3 h-3 rounded-full bg-orange-500 animate-pulse"></span>
                <span class="text-sm font-bold text-slate-700">KTD Area Hub</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. SIDEBAR (Clean & Modern) ---
with st.sidebar:
    st.markdown("""<div class="p-4 bg-orange-50 rounded-2xl mb-6">
        <p class="text-xs font-bold text-orange-600 uppercase">Engineer Console</p>
        <p class="text-sm font-semibold text-orange-900 mt-1">Logged in as KTD_ADMIN</p>
    </div>""", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📥 Upload Feeder XLSX", type=["xlsx"])
    st.divider()
    
    st.subheader("📸 Record Survey")
    tr_id = st.selectbox("Select Asset", st.session_state.ktd_assets['Transformer_ID'])
    temp_in = st.slider("Thermal Temp (°C)", 20, 120, 50)
    sound_in = st.slider("Acoustic (dB)", 30, 110, 50)
    
    if st.button("🚀 Analyze Now", use_container_width=True):
        st.toast("AI is analyzing data...", icon="🤖")

# --- 6. MAIN DASHBOARD (The Perfect Bento Layout) ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🔍 Asset Diagnostics", "📅 Maintenance Plan"])

with tab1:
    df = st.session_state.ktd_assets
    # Bento Metrics Row
    st.markdown(f"""
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bento-card border-t-4 border-orange-500">
                <p class="text-xs font-bold text-gray-400 uppercase tracking-widest">Total Units</p>
                <h2 class="text-4xl font-extrabold mt-2">{len(df)}</h2>
                <p class="text-[10px] text-green-600 font-bold mt-2">● Online & Syncing</p>
            </div>
            <div class="bento-card border-t-4 border-red-600">
                <p class="text-xs font-bold text-gray-400 uppercase tracking-widest text-red-600">Critical</p>
                <h2 class="text-4xl font-extrabold mt-2 text-red-600">{len(df[df['Status'] == '🔴 CRITICAL'])}</h2>
                <p class="text-[10px] text-red-400 font-bold mt-2">Action Required Immediately</p>
            </div>
            <div class="bento-card border-t-4 border-orange-400">
                <p class="text-xs font-bold text-gray-400 uppercase tracking-widest text-orange-600">Watch List</p>
                <h2 class="text-4xl font-extrabold mt-2 text-orange-600">{len(df[df['Status'] == '🟡 WATCH'])}</h2>
                <p class="text-[10px] text-orange-400 font-bold mt-2">Monitoring Active</p>
            </div>
            <div class="bento-card border-t-4 border-green-600">
                <p class="text-xs font-bold text-gray-400 uppercase tracking-widest text-green-700">Health Index</p>
                <h2 class="text-4xl font-extrabold mt-2 text-green-700">94.2%</h2>
                <div class="w-full bg-gray-100 h-1.5 rounded-full mt-4"><div class="bg-green-500 h-1.5 rounded-full" style="width: 94%"></div></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Big Map Section
    st.markdown('<div class="bento-card mb-8">', unsafe_allow_html=True)
    st.markdown('<h3 class="text-lg font-bold mb-4 flex items-center gap-2">📍 Geographic Distribution <span class="text-xs font-normal text-gray-400">(Real-time Map)</span></h3>', unsafe_allow_html=True)
    fig = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Status", size="Load_Percent", zoom=11.5,
                            color_discrete_map={'🔴 CRITICAL': '#E11D48', '🟡 WATCH': '#FB923C', '🟢 NORMAL': '#16A34A'},
                            mapbox_style="carto-positron", height=500)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="flex justify-between items-center mb-6"><div><h2 class="text-2xl font-bold font-["Kanit"]">📅 แผนงานบำรุงรักษา</h2><p class="text-sm text-gray-500">Sorted by AI Risk Priority Level</p></div></div>', unsafe_allow_html=True)
    
    urgent = df[df['Status'] != '🟢 NORMAL'].sort_values(by='Risk_Score', ascending=False)
    
    if urgent.empty:
        st.markdown('<div class="p-10 text-center bento-card text-gray-400">✅ No maintenance required at this moment</div>', unsafe_allow_html=True)
    else:
        for _, row in urgent.iterrows():
            status_color = "red-600" if row['Status'] == '🔴 CRITICAL' else "orange-500"
            border_color = "red-100" if row['Status'] == '🔴 CRITICAL' else "orange-100"
            st.markdown(f"""
                <div class="bento-card mb-4 border-l-8 border-{status_color} flex items-center justify-between">
                    <div class="flex items-center gap-6">
                        <div class="bg-{border_color} p-4 rounded-2xl">
                            <span class="text-2xl">⚡</span>
                        </div>
                        <div>
                            <h4 class="text-xl font-bold text-slate-800">{row['Transformer_ID']}</h4>
                            <p class="text-sm font-medium text-slate-500 italic">{row['Feeder']} | Load: {row['Load_Percent']}%</p>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-8 px-10 border-x border-gray-100">
                        <div><p class="text-[10px] font-bold text-gray-400 uppercase">Temperature</p><p class="text-lg font-bold text-slate-700">{row['Thermal_Temp']}°C</p></div>
                        <div><p class="text-[10px] font-bold text-gray-400 uppercase">Acoustic</p><p class="text-lg font-bold text-slate-700">{row['Acoustic_dB']}dB</p></div>
                    </div>
                    <div class="text-right min-w-[150px]">
                        <p class="text-[10px] font-bold text-{status_color} uppercase tracking-widest">Planned Month</p>
                        <p class="text-xl font-extrabold text-[#904D00] font-['Kanit']">{row['Plan_Month']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- 7. FOOTER ACTION ---
st.markdown("""
    <div class="mt-10 text-center text-gray-400 text-xs">
        Powered by Smart Plan AI Engine v2.0 | MEA Digital Service Ecosystem
    </div>
    """, unsafe_allow_html=True)
