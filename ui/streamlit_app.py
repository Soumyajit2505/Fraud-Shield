"""
FraudShield — Fraud Detection Dashboard
Production-grade Streamlit app with corrected Plotly + premium UI.

Requirements: streamlit, plotly, pandas, numpy, joblib, scikit-learn (optional)
"""

from __future__ import annotations

import io
import pickle
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
DATA_DIR     = Path("data")
HISTORY_FILE = DATA_DIR / "history.csv"

HISTORY_COLUMNS = [
    "Timestamp",
    "Source",
    "Amount",
    "Fraud_Prob_%",
    "Risk",
    "Prediction",
]

ALL_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ═════════════════════════════════════════════════════════════════════════════
# PREMIUM DARK THEME CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:         #0F0F1E;
    --surface:    #1A1A2E;
    --surface2:   #16213E;
    --surface3:   #0F3460;
    --surface4:   #1E5A96;

    --accent-1:   #00D9FF;
    --accent-2:   #7C3AED;
    --danger:     #FF006E;
    --warning:    #FFB703;
    --ok:         #06D6A0;

    --text:       #FFFFFF;
    --text-2:     #B0B0C0;
    --muted:      #757580;

    --border:     rgba(255,255,255,0.08);
    --border-hi:  rgba(0,217,255,0.35);

    --r:          14px;
    --r-sm:       8px;
    --r-lg:       20px;

    --font-display: 'Space Grotesk', sans-serif;
    --font-body:    'Inter', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

html, body, [class*="css"], .stApp, .stMarkdown, p, span, div {
    font-family: var(--font-body);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text) !important;
}

.stApp {
    background: linear-gradient(135deg, #0F0F1E 0%, #1A1A2E 50%, #16213E 100%);
    background-attachment: fixed;
}

/* ═══════════════════════════════════════ */
/* SIDEBAR                                */
/* ═══════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #1A1A2E;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.2rem 1.2rem;
}

/* ── NAV BUTTONS: base state ── */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    text-align: left !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-2) !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    margin-bottom: 6px !important;
    font-family: 'Inter', 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}

/* ── NAV BUTTONS: hover state ── */
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    color: var(--text) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── NAV BUTTONS: focus (active page) ── */
[data-testid="stSidebar"] .stButton > button:focus:not(:active) {
    background: rgba(124, 92, 255, 0.15) !important;
    border-color: rgba(124, 92, 255, 0.4) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}

/*
   Streamlit injects an inner <p> tag inside every button.
   Without targeting it, flex layout breaks and text wraps.
*/
[data-testid="stSidebar"] .stButton > button p {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: 'Inter', 'DM Sans', sans-serif !important;
    line-height: 1.4 !important;
    width: 100% !important;
}

/* ── ACTIVE NAV ITEM: inject via helper class on wrapping div ── */
[data-testid="stSidebar"] .nav-active .stButton > button {
    background: rgba(124, 92, 255, 0.15) !important;
    border-color: rgba(124, 92, 255, 0.4) !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] .nav-active .stButton > button p {
    color: #ffffff !important;
}

/* ═══════════════════════════════════════ */
/* BUTTONS (non-nav)                      */
/* ═══════════════════════════════════════ */
.stButton > button {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    border-radius: var(--r-sm) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important;
    color: var(--text) !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    border-color: var(--border-hi) !important;
    box-shadow: 0 8px 24px rgba(0,217,255,0.2) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00D9FF 0%, #7C3AED 100%) !important;
    color: #0F0F1E !important;
    border: none !important;
    font-weight: 700 !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 32px rgba(0,217,255,0.3) !important;
    transform: translateY(-3px) !important;
}

/* ═══════════════════════════════════════ */
/* CARDS                                  */
/* ═══════════════════════════════════════ */
.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.6rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}

.card:hover {
    border-color: var(--border-hi);
    box-shadow: 0 16px 48px rgba(0,217,255,0.15);
    transform: translateY(-2px);
}

/* ═══════════════════════════════════════ */
/* METRIC TILES                           */
/* ═══════════════════════════════════════ */
.metric-tile {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.5rem 1.8rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--tile-accent, var(--accent-1));
    box-shadow: 0 0 12px var(--tile-accent, var(--accent-1));
}

.metric-tile:hover {
    border-color: var(--border-hi);
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,217,255,0.15);
}

.metric-value {
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--tile-color, var(--text));
    letter-spacing: -0.02em;
}

.metric-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.6rem;
}

/* ═══════════════════════════════════════ */
/* PAGE HEADER                            */
/* ═══════════════════════════════════════ */
.page-header {
    margin-bottom: 2.2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

.page-title {
    font-family: var(--font-display);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    color: var(--text);
}

.page-sub {
    font-size: 0.95rem;
    color: var(--text-2);
    margin: 0.4rem 0 0;
}

/* ═══════════════════════════════════════ */
/* WELCOME HERO                           */
/* ═══════════════════════════════════════ */
.welcome-hero {
    background: linear-gradient(135deg, rgba(0,217,255,0.12) 0%, rgba(124,58,237,0.08) 100%);
    border: 1px solid rgba(0,217,255,0.25);
    border-radius: var(--r-lg);
    padding: 2.4rem 2.6rem;
    margin-bottom: 2.4rem;
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    box-shadow: 0 8px 32px rgba(0,217,255,0.08);
}

.username-highlight {
    background: linear-gradient(135deg, #00D9FF 0%, #7C3AED 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* ═══════════════════════════════════════ */
/* BADGES                                 */
/* ═══════════════════════════════════════ */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    border-radius: 99px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-ok {
    background: rgba(6,214,160,0.15);
    color: var(--ok);
    border: 1px solid rgba(6,214,160,0.3);
    box-shadow: 0 0 10px rgba(6,214,160,0.2);
}

.badge-warn {
    background: rgba(255,183,3,0.15);
    color: var(--warning);
    border: 1px solid rgba(255,183,3,0.3);
}

.badge-danger {
    background: rgba(255,0,110,0.15);
    color: var(--danger);
    border: 1px solid rgba(255,0,110,0.3);
}

.badge-info {
    background: rgba(0,217,255,0.15);
    color: var(--accent-1);
    border: 1px solid rgba(0,217,255,0.3);
}

/* ═══════════════════════════════════════ */
/* LABELS & HELPERS                       */
/* ═══════════════════════════════════════ */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 1.4rem 0 0.7rem;
}

.helper {
    background: rgba(0,217,255,0.08);
    padding: 0.9rem 1.1rem;
    border-radius: var(--r-sm);
    border-left: 3px solid var(--accent-1);
    font-size: 0.87rem;
    color: var(--text-2);
    margin: 0 0 1rem;
}

.helper code {
    background: var(--surface3);
    color: var(--accent-1);
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-size: 0.85em;
}

/* ═══════════════════════════════════════ */
/* FORMS & INPUTS                         */
/* ═══════════════════════════════════════ */
div[data-testid="stNumberInput"] input,
.stTextInput input,
.stSelectbox select {
    background: var(--surface3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    padding: 0.65rem 0.9rem !important;
}

div[data-testid="stNumberInput"] input:focus,
.stTextInput input:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px rgba(0,217,255,0.2) !important;
}

/* ═══════════════════════════════════════ */
/* FILE UPLOADER                          */
/* ═══════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--surface3) !important;
    border: 2px dashed var(--border-hi) !important;
    border-radius: var(--r) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-1) !important;
    background: rgba(0,217,255,0.08) !important;
}

/* ═══════════════════════════════════════ */
/* SIDEBAR BRAND                          */
/* ═══════════════════════════════════════ */
.sidebar-brand {
    font-family: var(--font-display);
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00D9FF 0%, #7C3AED 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ═══════════════════════════════════════ */
/* EMPTY STATE                            */
/* ═══════════════════════════════════════ */
.empty-state {
    text-align: center;
    padding: 4rem 2.5rem;
    background: var(--surface2);
    border: 1px dashed var(--border-hi);
    border-radius: var(--r);
}

.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.7;
}

.empty-title {
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
}

.empty-desc {
    font-size: 0.9rem;
    color: var(--text-2);
    margin-top: 0.6rem;
    max-width: 480px;
    margin-left: auto;
    margin-right: auto;
}

/* ═══════════════════════════════════════ */
/* DIVIDERS                               */
/* ═══════════════════════════════════════ */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 1rem 0 !important;
}

/* ═══════════════════════════════════════ */
/* SCROLLBAR                              */
/* ═══════════════════════════════════════ */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(0,217,255,0.3);
    border-radius: 99px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0,217,255,0.6);
}

/* ═══════════════════════════════════════ */
/* ALERTS                                 */
/* ═══════════════════════════════════════ */
.stAlert {
    border-radius: var(--r-sm) !important;
    border-left: 3px solid !important;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENCE LAYER
# ═════════════════════════════════════════════════════════════════════════════
def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def load_history() -> list[dict]:
    ensure_data_dir()
    if not HISTORY_FILE.exists():
        return []
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        return []
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[HISTORY_COLUMNS].to_dict("records")


def save_history(records: list[dict]) -> None:
    ensure_data_dir()
    df = pd.DataFrame(records, columns=HISTORY_COLUMNS) if records else _empty_history_df()
    df.to_csv(HISTORY_FILE, index=False)


def append_history_record(record: dict) -> None:
    ensure_data_dir()
    df_new = pd.DataFrame([record], columns=HISTORY_COLUMNS)
    header = not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0
    df_new.to_csv(HISTORY_FILE, mode="a", header=header, index=False)


def append_history_batch(records: list[dict]) -> None:
    if not records:
        return
    ensure_data_dir()
    df_new = pd.DataFrame(records, columns=HISTORY_COLUMNS)
    header = not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0
    df_new.to_csv(HISTORY_FILE, mode="a", header=header, index=False)


def clear_history_store() -> None:
    ensure_data_dir()
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
    st.session_state.transaction_history = []


def refresh_session_history() -> None:
    st.session_state.transaction_history = load_history()


def build_history_record(
    *,
    amount: float,
    probability: float,
    risk: str,
    is_fraud: bool,
    source: str,
) -> dict:
    return {
        "Timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source":       source,
        "Amount":       f"${float(amount):.2f}",
        "Fraud_Prob_%": round(float(probability) * 100, 2),
        "Risk":         risk,
        "Prediction":   "Fraud" if is_fraud else "Legit",
    }


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
def _init_state() -> None:
    defaults = {
        "custom_model":        None,
        "model_name":          None,
        "nav":                 "Overview",
        "username":            "",
        "transaction_history": None,
        "_batch_result":       None,
        "_batch_key":          None,
        "_batch_saved":        False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.transaction_history is None:
        st.session_state.transaction_history = load_history()


_init_state()


# ═════════════════════════════════════════════════════════════════════════════
# FALLBACK MODEL
# ═════════════════════════════════════════════════════════════════════════════
class StatisticalFallback:
    _COEF = {
        "V1":  -0.15, "V2":   0.10, "V3":  -0.18, "V4":   0.12,
        "V5":  -0.08, "V6":  -0.06, "V7":  -0.14, "V8":   0.04,
        "V9":  -0.10, "V10": -0.16, "V11":  0.09, "V12": -0.17,
        "V13": -0.04, "V14": -0.22, "V15": -0.03, "V16": -0.11,
        "V17": -0.19, "V18": -0.08, "V19":  0.02, "V20":  0.03,
        "V21":  0.06, "V22":  0.02, "V23": -0.02, "V24": -0.01,
        "V25": -0.01, "V26": -0.02, "V27":  0.04, "V28":  0.03,
        "Amount": 0.00005,
    }
    _INTERCEPT = -3.8

    def _score(self, row: dict) -> float:
        z = self._INTERCEPT + sum(
            self._COEF.get(k, 0) * row.get(k, 0) for k in self._COEF
        )
        return float(1.0 / (1.0 + np.exp(-z)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = [self._score(r) for r in X.to_dict("records")]
        return np.column_stack([1 - np.array(probs), probs])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@st.cache_resource
def _fallback_model() -> StatisticalFallback:
    return StatisticalFallback()


def get_active_model():
    return st.session_state.custom_model or _fallback_model()


def is_using_fallback() -> bool:
    return st.session_state.custom_model is None


def load_model_from_bytes(raw: bytes):
    errors = []
    for name, fn in [
        ("joblib", lambda b: joblib.load(io.BytesIO(b))),
        ("pickle", lambda b: pickle.loads(b)),
    ]:
        try:
            model = fn(raw)
            if not (hasattr(model, "predict") and hasattr(model, "predict_proba")):
                return None, "Object missing predict()/predict_proba() — not a valid classifier."
            return model, None
        except Exception as e:
            errors.append(f"{name}: {e}")
    return None, "Failed to deserialize. Tried: " + " | ".join(errors)


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def risk_label(p: float) -> str:
    return "High" if p >= 0.70 else ("Medium" if p >= 0.35 else "Low")


def risk_color(risk: str) -> str:
    return {"Low": "#06D6A0", "Medium": "#FFB703", "High": "#FF006E"}.get(risk, "#FFFFFF")


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    model   = get_active_model()
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X     = df[ALL_FEATURES].fillna(0.0)
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    out = df.copy()
    out["Fraud_Probability"] = np.round(proba, 4)
    out["Prediction"]        = np.where(preds == 1, "Fraud", "Legit")
    out["Risk_Level"]        = [risk_label(p) for p in proba]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# DATAFRAME STYLING
# ═════════════════════════════════════════════════════════════════════════════
def _style_pred_col(col: pd.Series) -> list[str]:
    return [
        "color:#FF006E;font-weight:600" if v == "Fraud" else
        ("color:#06D6A0;font-weight:600" if v == "Legit" else "")
        for v in col
    ]


def _style_risk_col(col: pd.Series) -> list[str]:
    mapping = {
        "High":   "color:#FF006E;font-weight:600",
        "Medium": "color:#FFB703;font-weight:600",
        "Low":    "color:#06D6A0;font-weight:600",
    }
    return [mapping.get(v, "") for v in col]


def apply_df_styles(df: pd.DataFrame):
    s = df.style
    if "Prediction" in df.columns:
        s = s.apply(_style_pred_col, subset=["Prediction"])
    for col in ("Risk_Level", "Risk"):
        if col in df.columns:
            s = s.apply(_style_risk_col, subset=[col])
            break
    return s


# ═════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def page_header(title: str, subtitle: str):
    st.markdown(
        f"<div class='page-header'>"
        f"<p class='page-title'>{title}</p>"
        f"<p class='page-sub'>{subtitle}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def metric_tile(col, value: str, label: str, color: str):
    col.markdown(
        f"<div class='metric-tile' style='--tile-color:{color};--tile-accent:{color}'>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-label'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def welcome_banner(username: str):
    display_name = username.strip() if username else "User"
    st.markdown(
        f"<div class='welcome-hero'>Welcome, <span class='username-highlight'>{display_name}</span></div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'>🛡️ FraudShield</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("<div class='section-label'>User</div>", unsafe_allow_html=True)
    st.session_state.username = st.text_input(
        "Username",
        value=st.session_state.username,
        placeholder="Enter your name",
        label_visibility="collapsed",
    )

    st.divider()

    # ── NAV ITEMS ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Navigate</div>", unsafe_allow_html=True)

    NAV_ITEMS = [
        ("◈", "Overview"),
        ("⊞", "Transaction Risk Analysis"),
        ("⬡", "Model Insights"),
    ]

    for icon, label in NAV_ITEMS:
        is_active = st.session_state.nav == label

        # Wrap each button in a div so we can apply active CSS via a class
        if is_active:
            st.markdown("<div class='nav-active'>", unsafe_allow_html=True)

        if st.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
            help=label,          # tooltip shows full label even when ellipsed
        ):
            st.session_state.nav = label
            st.rerun()

        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)
    # ── END NAV ITEMS ─────────────────────────────────────────────────────────

    st.divider()

    st.markdown("<div class='section-label'>Model</div>", unsafe_allow_html=True)
    if st.session_state.custom_model is not None:
        st.markdown(
            f"<span class='badge badge-ok'>✓ {st.session_state.model_name}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("↩  Remove model", use_container_width=True):
            st.session_state.custom_model = None
            st.session_state.model_name   = None
            st.rerun()
    else:
        st.markdown(
            "<span class='badge badge-warn'>⚠ Statistical Fallback Active</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    uploaded_model = st.file_uploader(
        "Upload .pkl or .joblib",
        type=["pkl", "joblib"],
        label_visibility="collapsed",
        key="model_uploader",
    )
    if uploaded_model is not None:
        with st.spinner("Loading model…"):
            mdl, err = load_model_from_bytes(uploaded_model.read())
        if err:
            st.error(f"❌ {err}")
        else:
            st.session_state.custom_model = mdl
            st.session_state.model_name   = uploaded_model.name
            st.success("✅ Loaded successfully")
            st.rerun()

    st.divider()

    st.markdown("<div class='section-label'>Session</div>", unsafe_allow_html=True)
    hist    = st.session_state.transaction_history
    total   = len(hist)
    flagged = sum(1 for t in hist if t.get("Prediction") == "Fraud")
    ca, cb  = st.columns(2)
    ca.metric("Analyzed", total)
    cb.metric("Flagged",  flagged)

    st.markdown(
        f"<div style='font-size:0.72rem;color:var(--muted);margin-top:0.5rem;font-family:var(--font-mono)'>"
        f"💾 {HISTORY_FILE}</div>",
        unsafe_allow_html=True,
    )

    if hist:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        if st.button("🗑  Clear history", use_container_width=True):
            clear_history_store()
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# ROUTING
# ═════════════════════════════════════════════════════════════════════════════
page = st.session_state.nav


# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW PAGE
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    page_header("Overview", "Session activity and persistent transaction history")
    welcome_banner(st.session_state.username)

    hist   = st.session_state.transaction_history
    total  = len(hist)
    frauds = sum(1 for t in hist if t.get("Prediction") == "Fraud")
    legits = total - frauds
    rate   = (frauds / total * 100) if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    metric_tile(c1, str(total),     "Total Analyzed", "#B0B0C0")
    metric_tile(c2, str(frauds),    "Fraud Detected", "#FF006E")
    metric_tile(c3, str(legits),    "Legitimate",     "#06D6A0")
    metric_tile(c4, f"{rate:.1f}%", "Fraud Rate",     "#FFB703")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    if not hist:
        st.markdown(
            "<div class='empty-state'>"
            "<div class='empty-icon'>◈</div>"
            "<div class='empty-title'>No transaction records available</div>"
            "<div class='empty-desc'>Upload a transaction CSV file in <strong>Transaction Risk Analysis</strong> "
            "to begin risk analysis. Everything you analyze is saved to disk automatically.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("#### Recent Activity")
        df_hist = pd.DataFrame(hist[::-1])
        st.dataframe(apply_df_styles(df_hist), use_container_width=True, hide_index=True)

        if total >= 3:
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            st.markdown("#### Score Distribution")
            probs = [t["Fraud_Prob_%"] for t in hist]

            fig = go.Figure(go.Histogram(
                x=probs, nbinsx=20,
                marker=dict(
                    color=probs,
                    colorscale=[
                        [0,    "#06D6A0"],
                        [0.35, "#FFB703"],
                        [0.7,  "#FF006E"],
                    ],
                    line=dict(color="rgba(0,217,255,0.2)", width=0.5),
                    opacity=0.85,
                ),
                hovertemplate="Probability: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ))

            fig.update_layout(
                title_text="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#B0B0C0", size=11),
                height=300,
                margin=dict(t=20, b=60, l=60, r=20),
                bargap=0.1,
                hovermode="x unified",
                showlegend=False,
                hoverlabel=dict(
                    bgcolor="#131636",
                    bordercolor="#00D9FF",
                    font=dict(family="Inter", color="#F1F3FF", size=12),
                ),
                xaxis=dict(
                    title=dict(
                        text="Fraud Probability (%)",
                        font=dict(family="Inter", size=12, color="#00D9FF"),
                    ),
                    tickfont=dict(family="Inter", size=10, color="#B0B0C0"),
                    gridcolor="rgba(0,217,255,0.1)",
                    showgrid=True,
                    zeroline=False,
                    showline=True,
                    linewidth=1,
                    linecolor="rgba(0,217,255,0.1)",
                ),
                yaxis=dict(
                    title=dict(
                        text="Transaction Count",
                        font=dict(family="Inter", size=12, color="#00D9FF"),
                    ),
                    tickfont=dict(family="Inter", size=10, color="#B0B0C0"),
                    gridcolor="rgba(0,217,255,0.1)",
                    zeroline=False,
                    showline=True,
                    linewidth=1,
                    linecolor="rgba(0,217,255,0.1)",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TRANSACTION RISK ANALYSIS PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Transaction Risk Analysis":
    page_header("Transaction Risk Analysis", "Score multiple transactions from a CSV file")
    welcome_banner(st.session_state.username)

    if is_using_fallback():
        st.info("ℹ️ No custom model uploaded. Statistical fallback is currently active.")

    st.markdown(
        "<div class='card' style='margin-bottom:1.1rem'>"
        "<span style='font-size:0.88rem;color:var(--text-2)'>"
        "<strong style='color:var(--text)'>Expected columns:</strong> "
        "<code style='font-family:var(--font-mono);color:var(--accent-1);background:var(--surface3);"
        "padding:0.1rem 0.35rem;border-radius:4px'>Time</code>, "
        "<code style='font-family:var(--font-mono);color:var(--accent-1);background:var(--surface3);"
        "padding:0.1rem 0.35rem;border-radius:4px'>V1</code>–"
        "<code style='font-family:var(--font-mono);color:var(--accent-1);background:var(--surface3);"
        "padding:0.1rem 0.35rem;border-radius:4px'>V28</code>, "
        "<code style='font-family:var(--font-mono);color:var(--accent-1);background:var(--surface3);"
        "padding:0.1rem 0.35rem;border-radius:4px'>Amount</code>. "
        "Extra columns are preserved in the output."
        "</span></div>",
        unsafe_allow_html=True,
    )

    uploaded_csv = st.file_uploader(
        "Upload a transaction CSV file to begin risk analysis",
        type=["csv"],
        label_visibility="collapsed",
        key="batch_csv",
    )

    if uploaded_csv:
        try:
            df_raw = pd.read_csv(uploaded_csv)
            st.markdown(f"**{len(df_raw):,} rows loaded** · Preview:")
            st.dataframe(df_raw.head(5), use_container_width=True, hide_index=True)

            upload_key = f"{uploaded_csv.name}:{len(df_raw)}"

            if st.button("⚡  Analyze Transactions", type="primary"):
                with st.spinner(f"Scoring {len(df_raw):,} transactions…"):
                    df_result = predict_batch(df_raw)
                st.session_state._batch_result = df_result
                st.session_state._batch_key    = upload_key
                st.session_state._batch_saved  = False

            df_result = st.session_state._batch_result
            if df_result is not None and st.session_state._batch_key == upload_key:
                n_fraud = int((df_result["Prediction"] == "Fraud").sum())
                r1, r2, r3 = st.columns(3)
                r1.metric("Total",  f"{len(df_result):,}")
                r2.metric("Fraud",  f"{n_fraud:,}",
                          delta=f"{n_fraud/len(df_result)*100:.1f}% rate",
                          delta_color="inverse")
                r3.metric("Legit",  f"{len(df_result)-n_fraud:,}")

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.markdown("#### Results")
                st.dataframe(apply_df_styles(df_result), use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇  Download Results CSV",
                    data=df_result.to_csv(index=False).encode(),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

                if not st.session_state._batch_saved:
                    records = [
                        build_history_record(
                            amount=float(row.get("Amount", 0) or 0),
                            probability=float(row["Fraud_Probability"]),
                            risk=row["Risk_Level"],
                            is_fraud=(row["Prediction"] == "Fraud"),
                            source="Batch",
                        )
                        for _, row in df_result.iterrows()
                    ]
                    append_history_batch(records)
                    refresh_session_history()
                    st.session_state._batch_saved = True
                    st.success(f"✅ {len(df_result):,} transactions saved to persistent history.")
                else:
                    st.info("ℹ️ This analysis has already been saved to history. Upload a new file to add more records.")

        except ValueError as e:
            st.error(f"❌ Column mismatch: {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        sample_df = pd.DataFrame([[0.0] + [0.0] * 28 + [50.0]], columns=ALL_FEATURES)
        st.download_button(
            "⬇  Download CSV template",
            data=sample_df.to_csv(index=False).encode(),
            file_name="sample_transactions.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODEL INSIGHTS PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Insights":
    page_header("Model Insights", "Feature weights, risk thresholds, and model details")
    welcome_banner(st.session_state.username)

    model    = get_active_model()
    is_fb    = is_using_fallback()
    mdl_name = st.session_state.model_name or "Statistical Fallback"
    mdl_type = type(model).__name__

    status_badge = (
        "<span class='badge badge-warn'>Statistical Fallback Active</span>"
        if is_fb else
        "<span class='badge badge-ok'>Custom Model</span>"
    )

    st.markdown(
        f"<div class='card' style='display:flex;align-items:center;gap:1.1rem;margin-bottom:1.4rem'>"
        f"<div style='font-size:2rem'>⬡</div>"
        f"<div><div style='font-family:var(--font-display);font-weight:700;font-size:1.05rem;"
        f"letter-spacing:-0.025em'>{mdl_name}</div>"
        f"<div style='color:var(--text-2);font-size:0.85rem;margin-top:0.2rem'>"
        f"{mdl_type} &nbsp;·&nbsp; {status_badge}</div></div></div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Feature Weights")

        if not is_fb and hasattr(model, "coef_") and model.coef_ is not None:
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            imp  = pd.Series(np.abs(coef), index=ALL_FEATURES[:len(coef)]).nlargest(15)
        elif not is_fb and hasattr(model, "feature_importances_"):
            fi  = model.feature_importances_
            imp = pd.Series(fi, index=ALL_FEATURES[:len(fi)]).nlargest(15)
        else:
            imp_dict = {k: abs(v) for k, v in StatisticalFallback._COEF.items() if v != 0}
            imp = pd.Series(imp_dict).sort_values(ascending=False).head(15)

        fig_imp = go.Figure(go.Bar(
            x=imp.values,
            y=imp.index,
            orientation="h",
            marker=dict(
                color=imp.values,
                colorscale=[
                    [0,   "rgba(15,52,96,0.8)"],
                    [0.5, "rgba(0,217,255,0.6)"],
                    [1,   "#00D9FF"],
                ],
                line=dict(color="rgba(0,217,255,0.2)", width=0.5),
                opacity=0.9,
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))

        fig_imp.update_layout(
            title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#B0B0C0"),
            margin=dict(t=20, b=60, l=150, r=20),
            height=420,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(
                title=dict(
                    text="Importance Score",
                    font=dict(family="Inter", size=12, color="#00D9FF"),
                ),
                tickfont=dict(family="Inter", size=9, color="#B0B0C0"),
                gridcolor="rgba(0,217,255,0.1)",
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor="rgba(0,217,255,0.1)",
            ),
            yaxis=dict(
                tickfont=dict(family="JetBrains Mono", size=10, color="#B0B0C0"),
                autorange="reversed",
                showline=True,
                linewidth=1,
                linecolor="rgba(0,217,255,0.1)",
            ),
        )

        st.plotly_chart(fig_imp, use_container_width=True)

    with right_col:
        st.markdown("#### Risk Bands")
        for lbl, lo, hi, color in [
            ("Low",    0,  35, "#06D6A0"),
            ("Medium", 35, 70, "#FFB703"),
            ("High",   70,100, "#FF006E"),
        ]:
            st.markdown(f"""
            <div style='margin-bottom:1rem'>
                <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem'>
                    <span style='font-size:0.85rem;color:var(--text-2);font-weight:500'>{lbl} Risk</span>
                    <span style='font-family:var(--font-mono);font-size:0.8rem;font-weight:500;color:{color}'>
                        {lo}%–{hi}%
                    </span>
                </div>
                <div style='background:var(--surface3);border-radius:99px;height:5px'>
                    <div style='margin-left:{lo}%;width:{hi-lo}%;background:{color};
                                height:5px;border-radius:99px;
                                box-shadow:0 0 10px {color}70'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown("#### Classification Thresholds")
        st.markdown("""
        <div class='card'>
            <div style='display:flex;flex-direction:column;gap:0.65rem;font-size:0.88rem'>
                <div style='display:flex;justify-content:space-between'>
                    <span style='color:var(--text-2)'>Fraud classification at</span>
                    <span style='font-family:var(--font-mono);color:#FF006E;font-weight:500'>≥ 50%</span>
                </div>
                <div style='display:flex;justify-content:space-between'>
                    <span style='color:var(--text-2)'>Low → Medium risk at</span>
                    <span style='font-family:var(--font-mono);color:#FFB703;font-weight:500'>≥ 35%</span>
                </div>
                <div style='display:flex;justify-content:space-between'>
                    <span style='color:var(--text-2)'>Medium → High risk at</span>
                    <span style='font-family:var(--font-mono);color:#FF006E;font-weight:500'>≥ 70%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not is_fb:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown("#### Model Parameters")
        ATTRS = ("n_estimators", "max_depth", "C", "kernel", "n_features_in_", "classes_")
        found = {a: getattr(model, a) for a in ATTRS if hasattr(model, a)}
        if found:
            acols = st.columns(min(len(found), 4))
            for i, (k, v) in enumerate(found.items()):
                acols[i % 4].metric(k, str(v))
        else:
            st.info("No standard sklearn attributes found on this model.")