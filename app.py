import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SPP-AI Maintenance Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Kanit', sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #0B1220 0%, #101A2E 35%, #F5F7FB 35%, #F5F7FB 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #0F172A 0%, #13213D 45%, #1D4ED8 100%);
            border-radius: 24px;
            padding: 28px 30px;
            color: white;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.28);
            margin-bottom: 18px;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .hero-sub {
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
            margin-bottom: 0;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
        }

        .glass-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(226,232,240,0.9);
            border-radius: 22px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(10px);
        }

        .metric-card {
            border-radius: 20px;
            padding: 18px 18px 14px 18px;
            background: white;
            border: 1px solid #E5E7EB;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .metric-label {
            font-size: 0.88rem;
            color: #64748B;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1;
        }

        .metric-foot {
            margin-top: 8px;
            font-size: 0.85rem;
            color: #94A3B8;
        }

        .accent-blue { border-top: 6px solid #2563EB; }
        .accent-red { border-top: 6px solid #EF4444; }
        .accent-amber { border-top: 6px solid #F59E0B; }
        .accent-green { border-top: 6px solid #22C55E; }

        .panel-card {
            background: white;
            border-radius: 22px;
            padding: 18px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        .plan-card {
            background: white;
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            border-left: 8px solid #CBD5E1;
            border: 1px solid #E2E8F0;
        }

        .plan-critical { border-left-color: #EF4444; }
        .plan-watch { border-left-color: #F59E0B; }

        .status-pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .pill-red { background: #FEE2E2; color: #B91C1C; }
        .pill-amber { background: #FEF3C7; color: #B45309; }
        .pill-green { background: #DCFCE7; color: #166534; }

        .small-muted {
            color: #64748B;
            font-size: 0.92rem;
        }

        .sidebar-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 0.3rem;
            margin-bottom: 0.5rem;
        }

        .stButton > button {
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: white;
            font-weight: 600;
            height: 2.9rem;
            box-shadow: 0 10px 20px rgba(37,99,235,0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
            color: white;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
        }

        button[data-baseweb="tab"] {
            background: rgba(255,255,255,0.88);
            border-radius: 12px;
            padding: 10px 16px;
            border: 1px solid #E5E7EB;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: #0F172A;
            color: white;
            border-color: #0F172A;
        }

        .upload-hint {
            background: #EFF6FF;
            color: #1D4ED8;
            border: 1px dashed #93C5FD;
            padding: 12px 14px;
            border-radius: 14px;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
@st.cache_resource

def load_spp_model():
    try:
        return joblib.load("mea_spp_ai_model.pkl")
    except Exception:
        return None


model = load_spp_model()


def calculate_plan_month(status: str, risk_score: float) -> str:
    today = datetime.now()
    if status == "🔴 CRITICAL":
        return today.strftime("%B %Y")
    if status == "🟡 WATCH":
        delay = int(max(1, (1 - risk_score) * 4))
        return (today + timedelta(days=delay * 30)).strftime("%B %Y")
    return "Routine Check"


def status_class(status: str) -> str:
    if status == "🔴 CRITICAL":
        return "pill-red"
    if status == "🟡 WATCH":
        return "pill-amber"
    return "pill-green"


def seed_demo_data_from_feeders(feeders: list[str], trips: dict) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for i, feeder in enumerate(feeders):
        rows.append(
            {
                "Transformer_ID": f"TR-KTD-{i+1:03d}",
                "Feeder": feeder,
                "Lat": 13.702 + np.random.uniform(-0.01, 0.01),
                "Lon": 100.555 + np.random.uniform(-0.01, 0.01),
                "Load_Percent": 0.0,
                "Voltage_V": 220.0,
                "Trips_Count": trips.get(feeder, 0),
                "Acoustic_dB": 45.0,
                "Thermal_Temp": 50.0,
                "Peak_Freq_Hz": 25000.0,
                "Age_Years": int(np.random.randint(5, 35)),
                "Humidity": 65.0,
                "Status": "🟢 NORMAL",
                "Risk_Score": 0.0,
                "Plan_Month": "Routine Check",
                "Survey_Img": None,
            }
        )
    return pd.DataFrame(rows)


def infer_single_asset(df: pd.DataFrame, idx: int):
    row = df.iloc[idx]
    feat = np.array(
        [[
            row["Thermal_Temp"],
            row["Load_Percent"],
            row["Voltage_V"],
            row["Acoustic_dB"],
            row["Peak_Freq_Hz"],
            row["Trips_Count"],
            row["Age_Years"],
            row["Humidity"],
        ]]
    )
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0][pred] if hasattr(model, "predict_proba") else 0.5
    status = {0: "🟢 NORMAL", 1: "🟡 WATCH", 2: "🔴 CRITICAL"}[pred]
    df.at[idx, "Status"] = status
    df.at[idx, "Risk_Score"] = float(prob)
    df.at[idx, "Plan_Month"] = calculate_plan_month(status, float(prob))
    return df


def infer_all_assets(df: pd.DataFrame):
    X = df[[
        "Thermal_Temp", "Load_Percent", "Voltage_V", "Acoustic_dB",
        "Peak_Freq_Hz", "Trips_Count", "Age_Years", "Humidity"
    ]].values
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else [[0.5] * 3] * len(preds)
    df["Status"] = [{0: "🟢 NORMAL", 1: "🟡 WATCH", 2: "🔴 CRITICAL"}[p] for p in preds]
    df["Risk_Score"] = [float(probs[i][preds[i]]) for i in range(len(preds))]
    df["Plan_Month"] = df.apply(lambda r: calculate_plan_month(r["Status"], r["Risk_Score"]), axis=1)
    return df


# =========================
# SESSION STATE
# =========================
if "ktd_assets" not in st.session_state:
    st.session_state.ktd_assets = pd.DataFrame(columns=[
        "Transformer_ID", "Feeder", "Lat", "Lon", "Load_Percent", "Voltage_V",
        "Trips_Count", "Acoustic_dB", "Thermal_Temp", "Peak_Freq_Hz", "Age_Years",
        "Humidity", "Status", "Risk_Score", "Plan_Month", "Survey_Img"
    ])

# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">⚡ SPP-AI Maintenance Command Center</div>
        <p class="hero-sub">ระบบวิเคราะห์ความเสี่ยงและวางแผนบำรุงรักษาหม้อแปลงไฟฟ้า สำหรับทีมปฏิบัติการภาคสนาม พร้อมมุมมองเชิงบริหารและการวินิจฉัยรายอุปกรณ์</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error("⚠️ ไม่พบไฟล์ mea_spp_ai_model.pkl กรุณาใส่ไฟล์โมเดลในโฟลเดอร์เดียวกับแอป")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">ศูนย์ควบคุมข้อมูล</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">เริ่มจากอัปโหลดไฟล์ Feeder เพื่อสร้างฐานข้อมูลอุปกรณ์ จากนั้นจึง Sync ค่าจาก Smart Meter และบันทึกข้อมูลสำรวจหน้างาน</div>', unsafe_allow_html=True)
    st.write("")

    uploaded_xlsx = st.file_uploader("อัปโหลดไฟล์ ฟขต Feeder.xlsx", type=["xlsx"])

    if uploaded_xlsx:
        try:
            feeder_df = pd.read_excel(uploaded_xlsx, skiprows=2)
            if "Feeder" in feeder_df.columns:
                trip_stats = feeder_df["Feeder"].value_counts().to_dict()
                feeder_list = list(trip_stats.keys())
                st.success(f"พบข้อมูล Feeder {len(feeder_list)} รายการ")
                if st.button("โหลดข้อมูลเข้าระบบ"):
                    st.session_state.ktd_assets = seed_demo_data_from_feeders(feeder_list, trip_stats)
                    st.success("โหลดข้อมูลอุปกรณ์สำเร็จ")
                    st.rerun()
            else:
                st.error("ไฟล์ไม่มีคอลัมน์ Feeder")
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")

    st.write("")
    if st.button("📡 Sync Smart Meter"):
        if not st.session_state.ktd_assets.empty:
            size = len(st.session_state.ktd_assets)
            st.session_state.ktd_assets["Load_Percent"] = np.random.uniform(40, 115, size)
            st.session_state.ktd_assets["Voltage_V"] = np.random.uniform(210, 235, size)
            st.success("ซิงค์ข้อมูล Smart Meter สำเร็จ")
            st.rerun()
        else:
            st.warning("กรุณาโหลดข้อมูล Feeder ก่อน")

    st.divider()

    if not st.session_state.ktd_assets.empty:
        st.markdown('<div class="sidebar-title">บันทึกสำรวจหน้างาน</div>', unsafe_allow_html=True)

        target_id = st.selectbox("เลือกหม้อแปลง", st.session_state.ktd_assets["Transformer_ID"])
        idx = st.session_state.ktd_assets[
            st.session_state.ktd_assets["Transformer_ID"] == target_id
        ].index[0]

        col_a, col_b = st.columns(2)
        with col_a:
            acoustic_val = st.number_input(
                "Acoustic (dB)",
                min_value=30.0,
                max_value=110.0,
                value=float(st.session_state.ktd_assets.at[idx, "Acoustic_dB"]),
            )
        with col_b:
            thermal_val = st.number_input(
                "Thermal (°C)",
                min_value=20.0,
                max_value=120.0,
                value=float(st.session_state.ktd_assets.at[idx, "Thermal_Temp"]),
            )

        survey_img = st.file_uploader(
            "อัปโหลดภาพหน้างาน",
            type=["jpg", "jpeg", "png"],
            key=f"survey_{target_id}",
        )

        if st.button("💾 บันทึกและวิเคราะห์เครื่องนี้"):
            st.session_state.ktd_assets.at[idx, "Acoustic_dB"] = acoustic_val
            st.session_state.ktd_assets.at[idx, "Thermal_Temp"] = thermal_val
            if survey_img is not None:
                st.session_state.ktd_assets.at[idx, "Survey_Img"] = survey_img

            if model is not None:
                st.session_state.ktd_assets = infer_single_asset(st.session_state.ktd_assets, idx)
            st.success(f"อัปเดตข้อมูล {target_id} สำเร็จ")
            st.rerun()

        st.write("")
        if st.button("🚀 วิเคราะห์แผนงานทั้งหมด"):
            if model is not None:
                st.session_state.ktd_assets = infer_all_assets(st.session_state.ktd_assets)
                st.success("วิเคราะห์ภาพรวมสำเร็จ")
                st.rerun()

# =========================
# MAIN TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Diagnostics", "Maintenance Plan", "Asset Table"
])

with tab1:
    if st.session_state.ktd_assets.empty:
        st.info("กรุณาอัปโหลดไฟล์ Feeder และโหลดข้อมูลเข้าระบบจากแถบด้านซ้าย")
    else:
        df = st.session_state.ktd_assets.copy()
        total_assets = len(df)
        critical_count = len(df[df["Status"] == "🔴 CRITICAL"])
        watch_count = len(df[df["Status"] == "🟡 WATCH"])
        normal_count = len(df[df["Status"] == "🟢 NORMAL"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card accent-blue">
                    <div class="metric-label">อุปกรณ์ทั้งหมด</div>
                    <div class="metric-value">{total_assets}</div>
                    <div class="metric-foot">Transformers in monitoring scope</div>
                </div>
                """,
                unsafe_allo
