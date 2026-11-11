"""
ui_theme.py — Orange & Black Fitness-Tech Design System for Streamlit.

Visual Identity:
- Deep black background:   #101010
- Secondary black / bars:  #161616
- Card surfaces:           #202020
- High-contrast surface:   #262626
- Subtle borders:          #303030
- Primary vibrant orange:  #FF6B00
- Secondary warm orange:   #FF8A24
- Text primary on dark:    #FFFFFF
- Text secondary / muted:  #9E9E9E
- Positive / success:      #22C55E
- Alert / warning:         #EF4444
"""

from typing import Optional, Tuple, Any

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    plt = None

THEME_CSS = """
<style>
/* ==========================================================================
   ORANGE & BLACK DESIGN SYSTEM — GLOBAL OVERRIDES
   ========================================================================== */

/* Main canvas background */
.stApp {
    background-color: #101010 !important;
    color: #FFFFFF !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #141414 !important;
    border-right: 1px solid #262626 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #262626 !important;
    margin: 12px 0 !important;
}

/* Header container */
header[data-testid="stHeader"] {
    background-color: rgba(16, 16, 16, 0.85) !important;
    backdrop-filter: blur(8px);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

p, span, label {
    color: #E0E0E0;
}

/* Streamlit Buttons — Primary Orange */
.stButton > button {
    background: linear-gradient(135deg, #FF6B00 0%, #FF8A24 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(255, 107, 0, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #FF8A24 0%, #FFA04D 100%) !important;
    box-shadow: 0 4px 14px rgba(255, 107, 0, 0.4) !important;
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Secondary Button variant */
.stButton > button[kind="secondary"] {
    background: #242424 !important;
    color: #E0E0E0 !important;
    border: 1px solid #363636 !important;
    box-shadow: none !important;
}

.stButton > button[kind="secondary"]:hover {
    background: #2A2A2A !important;
    border-color: #FF6B00 !important;
    color: #FFFFFF !important;
}

/* Form Inputs, Selectboxes, Text Areas */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border: 1px solid #303030 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #FF6B00 !important;
    box-shadow: 0 0 0 2px rgba(255, 107, 0, 0.2) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background-color: #FF6B00 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #141414;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #262626;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: #9E9E9E !important;
    font-weight: 500;
    padding: 8px 16px;
    background-color: transparent;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background-color: #242424 !important;
    color: #FF6B00 !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
}

/* Radio button pills in sidebar */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    transition: all 0.15s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(255, 107, 0, 0.08);
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    background-color: #1A1A1A;
    border: 1px solid #2D2D2D;
    border-radius: 8px;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

[data-testid="stMetricLabel"] {
    color: #9E9E9E !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ==========================================================================
   CUSTOM CARD & VISUAL COMPONENT CLASSES
   ========================================================================== */

/* Brand Header */
.k-brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 16px 0;
}

.k-brand-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, #FF6B00 0%, #FF8A24 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-weight: 900;
    font-size: 1.1rem;
    box-shadow: 0 2px 8px rgba(255, 107, 0, 0.35);
}

.k-brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.1;
}

.k-brand-sub {
    font-size: 0.72rem;
    color: #9E9E9E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 2px 0 0 0;
}

/* Top App Bar */
.k-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #171717;
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 22px;
}

.k-topbar-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.k-topbar-patient {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #202020;
    border: 1px solid #333333;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
}

.k-patient-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #FF6B00;
    box-shadow: 0 0 6px #FF6B00;
}

/* Standard Content Card */
.k-card {
    background-color: #1A1A1A;
    border: 1px solid #282828;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: border-color 0.2s ease;
}

.k-card:hover {
    border-color: #383838;
}

.k-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #262626;
}

.k-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.k-card-subtitle {
    font-size: 0.8rem;
    color: #8C8C8C;
    margin: 0;
}

/* Metric KPI Card (Orange Accents) */
.k-kpi-card {
    background: linear-gradient(180deg, #1E1E1E 0%, #161616 100%);
    border: 1px solid #2E2E2E;
    border-radius: 14px;
    padding: 18px;
    height: 100%;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25);
    position: relative;
    overflow: hidden;
}

.k-kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FF6B00 0%, #FF8A24 100%);
}

.k-kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9E9E9E;
    font-weight: 600;
    margin-bottom: 6px;
}

.k-kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
    margin-bottom: 6px;
    letter-spacing: -0.03em;
}

.k-kpi-unit {
    font-size: 1.0rem;
    font-weight: 500;
    color: #FF8A24;
    margin-left: 2px;
}

.k-kpi-sub {
    font-size: 0.8rem;
    color: #7A7A7A;
    display: flex;
    align-items: center;
    gap: 6px;
}

.k-kpi-bar {
    width: 100%;
    height: 6px;
    background-color: #282828;
    border-radius: 4px;
    margin-top: 10px;
    overflow: hidden;
}

.k-kpi-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #FF6B00 0%, #FF8A24 100%);
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* Status Pill Badges */
.k-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.k-pill-orange {
    background: rgba(255, 107, 0, 0.15);
    color: #FF8A24;
    border: 1px solid rgba(255, 107, 0, 0.35);
}

.k-pill-green {
    background: rgba(34, 197, 94, 0.14);
    color: #4ADE80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.k-pill-red {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.k-pill-neutral {
    background: #242424;
    color: #A3A3A3;
    border: 1px solid #333333;
}

/* Clinical Medical Disclaimer Banner */
.k-disclaimer {
    background: #1A1612;
    border-left: 4px solid #FF6B00;
    border-top: 1px solid #33261C;
    border-right: 1px solid #33261C;
    border-bottom: 1px solid #33261C;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.82rem;
    color: #D1C5B8;
    line-height: 1.45;
}

.k-disclaimer strong {
    color: #FF8A24;
}

/* Anomaly / Alert Box */
.k-alert-monitor {
    background: #221414;
    border: 1px solid #4D1F1F;
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.k-alert-baseline {
    background: #131E14;
    border: 1px solid #1C3820;
    border-left: 4px solid #22C55E;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

/* Sensor Visual Meter */
.k-sensor-box {
    background: #181818;
    border: 1px solid #292929;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}

.k-sensor-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 600;
    color: #E0E0E0;
    margin-bottom: 6px;
}

.k-sensor-val {
    color: #FF8A24;
    font-weight: 700;
}

.k-sensor-track {
    width: 100%;
    height: 8px;
    background: #262626;
    border-radius: 4px;
    overflow: hidden;
}

.k-sensor-fill {
    height: 100%;
    background: linear-gradient(90deg, #FF6B00 0%, #FF8A24 100%);
    border-radius: 4px;
}
</style>
"""


def apply_theme():
    """Inject the orange & black theme stylesheet into the active Streamlit app."""
    import streamlit as st
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def create_dark_figure(figsize: Tuple[float, float] = (8, 3.5), nrows: int = 1, ncols: int = 1):
    """
    Create a Matplotlib figure and axes pre-styled with the dark & orange theme.
    Returns (fig, axes) or (None, None) if Matplotlib is unavailable.
    """
    if not _MATPLOTLIB_AVAILABLE or plt is None:
        return None, None

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor="#161616")
    ax_list = [axes] if not hasattr(axes, "__len__") else (axes.flat if hasattr(axes, "flat") else axes)

    for ax in ax_list:
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="#9E9E9E", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#2E2E2E")
        ax.grid(True, color="#262626", linestyle="--", linewidth=0.7, alpha=0.8)

    return fig, axes

