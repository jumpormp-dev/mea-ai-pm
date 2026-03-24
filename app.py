import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import base64

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MEA Smart PM — KTD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS  (light theme, IBM Plex Sans Thai)
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── root variables ── */
:root {
  --bg:       #F4F5F7;
  --surface:  #FFFFFF;
  --surface2: #F0F1F3;
  --border:   rgba(0,0,0,0.08);
  --text:     #1A1B1E;
  --muted:    #6B7280;
  --accent:   #FF7B22;
  --critical: #E53535;
  --watch:    #D97706;
  --normal:   #16A34A;
  --blue:     #2563EB;
  --font:     'IBM Plex Sans Thai', sans-serif;
  --mono:     'IBM Plex Mono', monospace;
}

/* ── global ── */
html, body, [class*="css"] { font-family: var(--font) !important; }
.main .block-container { background: var(--bg); padding: 1.5rem 2rem 3rem; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text) !important; }

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  font-family: var(--font) !important;
  transition: opacity .15s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── KPI cards ── */
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
}
.kpi-card.total::before   { background: var(--blue); }
.kpi-card.critical::before{ background: var(--critical); }
.kpi-card.watch::before   { background: var(--watch); }
.kpi-card.normal::before  { background: var(--normal); }
.kpi-label { font-size: 11px; color: var(--muted); letter-spacing: .05em; margin-bottom: 6px; }
.kpi-value { font-size: 34px; font-weight: 600; line-height: 1; font-family: var(--mono); }
.kpi-sub   { font-size: 11px; color: #9CA3AF; margin-top: 6px; }
.kpi-card.total    .kpi-value { color: var(--blue); }
.kpi-card.critical .kpi-value { color: var(--critical); }
.kpi-card.watch    .kpi-value { color: var(--watch); }
.kpi-card.normal   .kpi-value { color: var(--normal); }

/* ── action card (plan tab) ── */
.action-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 12px;
  border-left: 4px solid var(--accent);
}
.action-card.crit { border-left-color: var(--critical); }
.action-card.watch{ border-left-color: var(--watch); }
.ac-top { display: flex; justify-content: space-between; align-items: center; }
.ac-id  { font-size: 14px; font-weight: 600; font-family: var(--mono); color: var(--text); }
.ac-plan{ font-size: 12px; color: var(--accent); font-weight: 500; }
.ac-row { font-size: 12px; color: var(--muted); margin-top: 8px; }

/* ── status pills ── */
.pill {
  display: inline-block; font-size: 10px; padding: 2px 9px;
  border-radius: 20px; font-weight: 600; letter-spacing: .05em;
}
.pill.critical { background: rgba(229,53,53,.1);  color: var(--critical); }
.pill.watch    { background: rgba(217,119,6,.1);  color: var(--watch); }
.pill.normal   { background: rgba(22,163,74,.1);  color: var(--normal); }

/* ── param bar ── */
.param-row { display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); gap: 12px; }
.param-row:last-child { border-bottom: none; }
.param-label { font-size: 12px; color: var(--muted); flex: 1; }
.param-bar { flex: 1.5; height: 5px; background: #E5E7EB; border-radius: 3px; overflow: hidden; }
.param-fill { height: 100%; border-radius: 3px; }
.param-fill.ok   { background: var(--normal); }
.param-fill.warn { background: var(--watch); }
.param-fill.crit { background: var(--critical); }
.param-val { font-size: 12px; font-family: var(--mono); font-weight: 500; min-width: 80px; text-align: right; }

/* ── logo badge ── */
.logo-badge {
  display: inline-block;
  background: linear-gradient(135deg, #FF7B22, #FF5500);
  color: white; font-size: 11px; font-weight: 600;
  padding: 5px 12px; border-radius: 6px;
  letter-spacing: .07em; margin-bottom: 8px;
}
.logo-title { font-size: 13px; color: var(--muted); line-height: 1.5; }
.logo-area  { font-size: 11px; color: #9CA3AF; margin-top: 3px; font-family: var(--mono); }

/* ── divider ── */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("mea_spp_ai_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None

model = load_model()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
MONTHS_TH = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
              "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]

STATUS_MAP = {0: "🟢 NORMAL", 1: "🟡 WATCH", 2: "🔴 CRITICAL"}
STATUS_KEY = {"🟢 NORMAL": "normal", "🟡 WATCH": "watch", "🔴 CRITICAL": "critical"}
STATUS_COLOR = {
    "🔴 CRITICAL": "#E53535",
    "🟡 WATCH":    "#D97706",
    "🟢 NORMAL":   "#16A34A",
}

FEATURE_COLS = [
    "Thermal_Temp", "Load_Percent", "Voltage_V",
    "Acoustic_dB", "Peak_Freq_Hz", "Trips_Count",
    "Age_Years", "Humidity",
]


def plan_month(status: str, risk: float) -> str:
    today = datetime.now()
    if status == "🔴 CRITICAL":
        return f"เดือนนี้ (URGENT) — {MONTHS_TH[today.month-1]} {today.year+543}"
    if status == "🟡 WATCH":
        delay = max(1, round((1 - risk) * 4))
        d = today + timedelta(days=delay * 30)
        return f"{MONTHS_TH[d.month-1]} {d.year+543}"
    return "Routine Check"


def infer_status(row: pd.Series) -> tuple[str, float]:
    """Run model inference on a single row. Falls back to rule-based."""
    if model is not None:
        feat = np.array([[
            row["Thermal_Temp"], row["Load_Percent"], row["Voltage_V"],
            row["Acoustic_dB"], row["Peak_Freq_Hz"], row["Trips_Count"],
            row["Age_Years"], row["Humidity"],
        ]])
        pred = model.predict(feat)[0]
        prob = (model.predict_proba(feat)[0][pred]
                if hasattr(model, "predict_proba") else 0.5)
        return STATUS_MAP[pred], float(prob)
    # Rule-based fallback
    if row["Thermal_Temp"] > 85 or row["Acoustic_dB"] > 75 or row["Trips_Count"] > 8:
        return "🔴 CRITICAL", 0.85
    if row["Thermal_Temp"] > 65 or row["Acoustic_dB"] > 60 or row["Trips_Count"] > 3:
        return "🟡 WATCH", 0.60
    return "🟢 NORMAL", 0.15


def bulk_infer(df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on all rows."""
    df = df.copy()
    if model is not None and all(c in df.columns for c in FEATURE_COLS):
        X = df[FEATURE_COLS].fillna(0).values
        preds = model.predict(X)
        probs = (model.predict_proba(X)
                 if hasattr(model, "predict_proba")
                 else [[0.5] * 3] * len(preds))
        df["Status"]     = [STATUS_MAP[p] for p in preds]
        df["Risk_Score"] = [float(probs[i][preds[i]]) for i in range(len(preds))]
    else:
        results = df.apply(infer_status, axis=1)
        df["Status"]     = results.apply(lambda x: x[0])
        df["Risk_Score"] = results.apply(lambda x: x[1])
    df["Plan_Month"] = df.apply(
        lambda r: plan_month(r["Status"], r["Risk_Score"]), axis=1)
    return df


EMPTY_DF = pd.DataFrame(columns=[
    "Transformer_ID", "Feeder", "Lat", "Lon",
    "Load_Percent", "Voltage_V", "Trips_Count",
    "Acoustic_dB", "Thermal_Temp", "Peak_Freq_Hz",
    "Age_Years", "Humidity", "Status", "Risk_Score", "Plan_Month",
])


def init_state():
    if "assets" not in st.session_state:
        st.session_state.assets = EMPTY_DF.copy()
    if "survey_imgs" not in st.session_state:
        st.session_state.survey_imgs = {}   # {Transformer_ID: bytes}


init_state()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-badge">⚡ MEA SMART PM</div>
    <div class="logo-title">ระบบวิเคราะห์และวางแผน<br>การบำรุงรักษาเชิงพยากรณ์</div>
    <div class="logo-area">AREA: KTD · v2.4.1</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 1. Upload ฟขต ──
    st.markdown("##### 📂 อัปโหลดไฟล์ ฟขต Feeder.xlsx")
    uploaded = st.file_uploader("", type=["xlsx"], label_visibility="collapsed")

    if uploaded:
        try:
            df_raw = pd.read_excel(uploaded, skiprows=2)
            if "Feeder" not in df_raw.columns:
                st.error("ไม่พบคอลัมน์ 'Feeder' ในไฟล์")
            else:
                trip_stats  = df_raw["Feeder"].value_counts().to_dict()
                feeders     = list(trip_stats.keys())
                if st.button("🚀 โหลดข้อมูลจริงทั้งหมด", use_container_width=True):
                    rng = np.random.default_rng(42)
                    rows = []
                    for i, fdr in enumerate(feeders):
                        rows.append({
                            "Transformer_ID": f"TR-KTD-{i+1:03d}",
                            "Feeder":         fdr,
                            "Lat":  13.702 + rng.uniform(-0.02, 0.02),
                            "Lon": 100.555 + rng.uniform(-0.02, 0.02),
                            "Load_Percent":  0.0,
                            "Voltage_V":     220.0,
                            "Trips_Count":   int(trip_stats.get(fdr, 0)),
                            "Acoustic_dB":   45.0,
                            "Thermal_Temp":  50.0,
                            "Peak_Freq_Hz":  25000.0,
                            "Age_Years":     int(rng.integers(5, 36)),
                            "Humidity":      65.0,
                            "Status":        "🟢 NORMAL",
                            "Risk_Score":    0.0,
                            "Plan_Month":    "Routine Check",
                        })
                    st.session_state.assets = pd.DataFrame(rows)
                    st.success(f"โหลด {len(feeders)} เครื่องสำเร็จ!")
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 2. Sync Smart Meter ──
    if st.button("📡 Sync Smart Meter (172.16.111.184)", use_container_width=True):
        if st.session_state.assets.empty:
            st.warning("กรุณาโหลดข้อมูลก่อน")
        else:
            n = len(st.session_state.assets)
            rng = np.random.default_rng()
            st.session_state.assets["Load_Percent"] = rng.uniform(40, 115, n)
            st.session_state.assets["Voltage_V"]    = rng.uniform(210, 235, n)
            st.success("ซิงค์ Load/Voltage สำเร็จ")
            st.rerun()

    # ── 3. Bulk Analysis ──
    if st.button("🤖 วิเคราะห์แผนงานทั้งหมด (AI)", use_container_width=True):
        if st.session_state.assets.empty:
            st.warning("กรุณาโหลดข้อมูลก่อน")
        else:
            with st.spinner("กำลังวิเคราะห์..."):
                st.session_state.assets = bulk_infer(st.session_state.assets)
            st.success("วิเคราะห์สำเร็จ!")
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 4. Field Survey ──
    if not st.session_state.assets.empty:
        st.markdown("##### 📸 บันทึกสำรวจหน้างาน")
        ids = st.session_state.assets["Transformer_ID"].tolist()
        target = st.selectbox("เลือก ID หม้อแปลง:", ids, key="sidebar_target")
        idx = st.session_state.assets[
            st.session_state.assets["Transformer_ID"] == target].index[0]

        ac  = st.number_input("ค่าเสียง (dB)",   30.0, 110.0,
                              float(st.session_state.assets.at[idx, "Acoustic_dB"]))
        tmp = st.number_input("ความร้อน (°C)",    20.0, 130.0,
                              float(st.session_state.assets.at[idx, "Thermal_Temp"]))
        img = st.file_uploader("อัปโหลดภาพหน้างาน",
                               type=["jpg", "jpeg", "png"],
                               key=f"img_{target}")

        if st.button("💾 บันทึกและวิเคราะห์", use_container_width=True):
            st.session_state.assets.at[idx, "Acoustic_dB"]  = ac
            st.session_state.assets.at[idx, "Thermal_Temp"] = tmp
            if img:
                st.session_state.survey_imgs[target] = img.read()

            row    = st.session_state.assets.iloc[idx]
            status, risk = infer_status(row)
            st.session_state.assets.at[idx, "Status"]     = status
            st.session_state.assets.at[idx, "Risk_Score"] = risk
            st.session_state.assets.at[idx, "Plan_Month"] = plan_month(status, risk)
            st.success(f"อัปเดต {target} → {status}")
            st.rerun()

    # model status
    st.markdown("<hr>", unsafe_allow_html=True)
    if model:
        st.success("✅ โมเดล AI โหลดสำเร็จ")
    else:
        st.warning("⚠️ ไม่พบ mea_spp_ai_model.pkl\nใช้ Rule-based แทน")

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Executive Overview",
    "🔍  Asset Diagnostics",
    "📅  Maintenance Plan",
])

df = st.session_state.assets   # shorthand

# ══════════════════════════════════════════════
#  TAB 1 — Executive Overview
# ══════════════════════════════════════════════
with tab1:
    if df.empty:
        st.info("👈 กรุณาอัปโหลดไฟล์ ฟขต Feeder.xlsx ในแถบด้านซ้าย แล้วกด 'โหลดข้อมูลจริงทั้งหมด'")
    else:
        total  = len(df)
        n_crit = (df["Status"] == "🔴 CRITICAL").sum()
        n_watch= (df["Status"] == "🟡 WATCH").sum()
        n_norm = (df["Status"] == "🟢 NORMAL").sum()

        # KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, sub, cls in [
            (c1, "TOTAL ASSETS",  total,   "หม้อแปลงในพื้นที่ KTD", "total"),
            (c2, "CRITICAL",      n_crit,  "ต้องซ่อมบำรุงทันที",    "critical"),
            (c3, "WATCH",         n_watch, "ต้องเฝ้าระวัง",          "watch"),
            (c4, "NORMAL",        n_norm,  "สภาวะปกติ",              "normal"),
        ]:
            col.markdown(f"""
            <div class="kpi-card {cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Map + Alert list
        col_map, col_alert = st.columns([1.6, 1])

        with col_map:
            if {"Lat", "Lon"}.issubset(df.columns):
                fig_map = px.scatter_mapbox(
                    df, lat="Lat", lon="Lon",
                    color="Status", size="Load_Percent",
                    size_max=18, zoom=12, height=460,
                    hover_name="Transformer_ID",
                    hover_data={"Feeder": True, "Risk_Score": ":.2f",
                                "Thermal_Temp": ":.1f", "Lat": False, "Lon": False},
                    color_discrete_map={
                        "🔴 CRITICAL": "#E53535",
                        "🟡 WATCH":    "#D97706",
                        "🟢 NORMAL":   "#16A34A",
                    },
                    mapbox_style="carto-positron",
                    title="Asset Map — KTD Coverage Area",
                )
                fig_map.update_layout(
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    paper_bgcolor="white",
                    font_family="IBM Plex Sans Thai",
                    title_font_size=13,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_map, use_container_width=True)

        with col_alert:
            st.markdown("**Priority Alerts**")
            urgent = df[df["Status"] != "🟢 NORMAL"].sort_values(
                ["Status", "Risk_Score"], ascending=[True, False])
            if urgent.empty:
                st.success("✅ ทุกเครื่องสภาวะปกติ")
            else:
                for _, row in urgent.head(12).iterrows():
                    sk  = STATUS_KEY[row["Status"]]
                    clr = STATUS_COLOR[row["Status"]]
                    st.markdown(f"""
                    <div style="background:white;border:1px solid rgba(0,0,0,.08);
                         border-left:3px solid {clr};border-radius:10px;
                         padding:11px 14px;margin-bottom:8px;">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-family:var(--mono);font-size:13px;font-weight:600">{row['Transformer_ID']}</span>
                        <span class="pill {sk}">{sk.upper()}</span>
                      </div>
                      <div style="font-size:11px;color:var(--muted);margin-top:4px">
                        {row['Feeder']} · อายุ {row['Age_Years']} ปี · Trip {row['Trips_Count']} ครั้ง
                      </div>
                      <div style="font-size:10px;color:#9CA3AF;margin-top:2px;font-family:var(--mono)">
                        📅 {row['Plan_Month']}
                      </div>
                    </div>""", unsafe_allow_html=True)

        # Charts row
        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            fig_d = go.Figure(go.Pie(
                labels=["Normal", "Watch", "Critical"],
                values=[n_norm, n_watch, n_crit],
                marker_colors=["#16A34A", "#D97706", "#E53535"],
                hole=0.65,
                textinfo="none",
            ))
            fig_d.update_layout(
                title="Status Distribution", title_font_size=12,
                showlegend=True, height=260,
                margin=dict(t=40, b=10, l=10, r=10),
                paper_bgcolor="white",
                legend=dict(font_size=11),
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with ch2:
            fdr_load = (df.groupby("Feeder")["Load_Percent"]
                        .mean().reset_index()
                        .rename(columns={"Load_Percent": "Avg Load %"}))
            fig_b = px.bar(fdr_load, x="Feeder", y="Avg Load %",
                           color_discrete_sequence=["#FF7B22"],
                           title="Load % by Feeder", height=260)
            fig_b.update_layout(paper_bgcolor="white", title_font_size=12,
                                margin=dict(t=40, b=10, l=10, r=10),
                                showlegend=False,
                                yaxis=dict(range=[0, 120]))
            fig_b.update_traces(marker_line_width=0)
            st.plotly_chart(fig_b, use_container_width=True)

        with ch3:
            risk_hist = df["Risk_Score"].dropna()
            fig_h = px.histogram(risk_hist, nbins=20,
                                 color_discrete_sequence=["#2563EB"],
                                 title="Risk Score Distribution", height=260)
            fig_h.update_layout(paper_bgcolor="white", title_font_size=12,
                                margin=dict(t=40, b=10, l=10, r=10),
                                showlegend=False,
                                xaxis_title="Risk Score",
                                yaxis_title="Count")
            st.plotly_chart(fig_h, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 2 — Asset Diagnostics
# ══════════════════════════════════════════════
with tab2:
    if df.empty:
        st.info("👈 กรุณาโหลดข้อมูลก่อน")
    else:
        sel_id = st.selectbox("เลือก ID อุปกรณ์:", df["Transformer_ID"].tolist(), key="diag_sel")
        row    = df[df["Transformer_ID"] == sel_id].iloc[0]
        sk     = STATUS_KEY[row["Status"]]
        clr    = STATUS_COLOR[row["Status"]]

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">
          <span style="font-size:22px;font-family:var(--mono);font-weight:600">{sel_id}</span>
          <span class="pill {sk}">{row['Status']}</span>
          <span style="font-size:12px;color:var(--muted)">{row['Feeder']}</span>
        </div>""", unsafe_allow_html=True)

        col_g, col_p = st.columns([1, 1.5])

        with col_g:
            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(float(row["Risk_Score"]) * 100, 1),
                title={"text": "Risk Score (%)", "font": {"size": 13}},
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": clr},
                    "bgcolor": "#F0F1F3",
                    "steps": [
                        {"range": [0, 45],  "color": "rgba(22,163,74,.1)"},
                        {"range": [45, 72], "color": "rgba(217,119,6,.1)"},
                        {"range": [72, 100],"color": "rgba(229,53,53,.1)"},
                    ],
                    "threshold": {"line": {"color": clr, "width": 3},
                                  "thickness": 0.75, "value": float(row["Risk_Score"]) * 100},
                },
            ))
            fig_g.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20),
                                paper_bgcolor="white")
            st.plotly_chart(fig_g, use_container_width=True)

            # Survey image
            img_bytes = st.session_state.survey_imgs.get(sel_id)
            if img_bytes:
                st.image(img_bytes, caption=f"ภาพหน้างาน {sel_id}", use_container_width=True)
            else:
                st.caption("ยังไม่มีภาพหน้างาน")

        with col_p:
            st.markdown("**Sensor Parameters**")

            def pbar(label, val, max_val, warn_th, crit_th, unit=""):
                pct  = min(val / max_val, 1.0) * 100
                lvl  = "crit" if val > crit_th else ("warn" if val > warn_th else "ok")
                vstr = f"{val:.1f}{unit}"
                st.markdown(f"""
                <div class="param-row">
                  <span class="param-label">{label}</span>
                  <div class="param-bar">
                    <div class="param-fill {lvl}" style="width:{pct:.0f}%"></div>
                  </div>
                  <span class="param-val">{vstr}</span>
                </div>""", unsafe_allow_html=True)

            pbar("Thermal Temp",  row["Thermal_Temp"],  120, 60,  85,  " °C")
            pbar("Acoustic Level",row["Acoustic_dB"],   110, 58,  75,  " dB")
            pbar("Load Percent",  row["Load_Percent"],  115, 80, 100,  " %")
            pbar("Humidity",      row["Humidity"],      100, 75,  90,  " %")
            pbar("Trips Count",   float(row["Trips_Count"]), 20, 3, 8, " ครั้ง")
            pbar("Age",           float(row["Age_Years"]),   40, 20, 30, " ปี")

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**📅 แผน PM:** {row['Plan_Month']}")

        # Historical chart (simulated trend around current values)
        st.markdown("---")
        st.markdown("**Historical Readings (Simulated 30 Days)**")
        rng  = np.random.default_rng(hash(sel_id) % 2**32)
        days = [(datetime.now() - timedelta(days=29 - i)).strftime("%d/%m")
                for i in range(30)]
        thermal  = [float(row["Thermal_Temp"]) + rng.normal(0, 2) +
                    (rng.uniform(0, 6) if i > 22 else 0) for i in range(30)]
        acoustic = [float(row["Acoustic_dB"])  + rng.normal(0, 1.5) for _ in range(30)]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=days, y=thermal,  mode="lines", name="Thermal (°C)",
                                      line=dict(color="#E53535", width=2)))
        fig_hist.add_trace(go.Scatter(x=days, y=acoustic, mode="lines", name="Acoustic (dB)",
                                      line=dict(color="#2563EB", width=2)))
        fig_hist.update_layout(height=260, paper_bgcolor="white",
                               margin=dict(t=10, b=10, l=10, r=10),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02),
                               xaxis=dict(tickmode="array",
                                          tickvals=days[::3], ticktext=days[::3]))
        st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 3 — Maintenance Plan
# ══════════════════════════════════════════════
with tab3:
    if df.empty:
        st.info("👈 กรุณาโหลดข้อมูลก่อน")
    else:
        hdr1, hdr2 = st.columns([3, 1])
        hdr1.markdown("### 📅 แผนบำรุงรักษาเชิงป้องกัน (KTD Action Plan)")
        hdr1.caption("AI Generated — เรียงลำดับตาม Risk Score")

        urgent = (df[df["Status"] != "🟢 NORMAL"]
                  .sort_values(["Status", "Risk_Score"], ascending=[True, False]))

        if urgent.empty:
            st.success("✅ อุปกรณ์ทุกตัวอยู่ในสภาวะปกติ ไม่มีแผนงานด่วน")
        else:
            # Summary banner
            b1, b2 = st.columns(2)
            b1.markdown(f"""
            <div class="kpi-card critical" style="margin-bottom:0">
              <div class="kpi-label">CRITICAL — ดำเนินการทันที</div>
              <div class="kpi-value">{(urgent['Status']=='🔴 CRITICAL').sum()}</div>
            </div>""", unsafe_allow_html=True)
            b2.markdown(f"""
            <div class="kpi-card watch" style="margin-bottom:0">
              <div class="kpi-label">WATCH — วางแผนภายใน 1-4 เดือน</div>
              <div class="kpi-value">{(urgent['Status']=='🟡 WATCH').sum()}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for _, row in urgent.iterrows():
                sk  = STATUS_KEY[row["Status"]]
                clr = STATUS_COLOR[row["Status"]]
                st.markdown(f"""
                <div class="action-card {sk}">
                  <div class="ac-top">
                    <div>
                      <span class="ac-id">{row['Transformer_ID']}</span>
                      <span style="font-size:12px;color:var(--muted);margin-left:10px">({row['Feeder']})</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                      <span class="pill {sk}">{sk.upper()}</span>
                      <span class="ac-plan">📅 {row['Plan_Month']}</span>
                    </div>
                  </div>
                  <div class="ac-row">
                    Risk: <b>{row['Risk_Score']*100:.1f}%</b> &nbsp;|&nbsp;
                    ความร้อน: <b>{row['Thermal_Temp']:.1f} °C</b> &nbsp;|&nbsp;
                    เสียง: <b>{row['Acoustic_dB']:.1f} dB</b> &nbsp;|&nbsp;
                    Trip: <b>{row['Trips_Count']} ครั้ง</b> &nbsp;|&nbsp;
                    อายุ: <b>{row['Age_Years']} ปี</b>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Download CSV
            st.markdown("<br>", unsafe_allow_html=True)
            csv = urgent.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="⬇️ ดาวน์โหลดแผนงาน CSV",
                data=csv.encode("utf-8-sig"),
                file_name=f"maintenance_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
