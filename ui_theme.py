
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"

def apply_theme():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --bg:#061a33; --panel:#0b2545; --panel2:#0d2d52; --line:#174775; --blue:#1685f8; --cyan:#35c5f4; --muted:#9bb6d2; --text:#eaf4ff; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: radial-gradient(circle at 75% 0%, #10365f 0, #061a33 48%, #04142a 100%); color:var(--text); }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1500px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#04162d,#061d38); border-right:1px solid #17416a; }
[data-testid="stSidebar"] * { color:#dcecff; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#0e3158,#092443); border:1px solid #1b5488; border-radius:12px; padding:14px 16px; box-shadow:0 8px 22px #00102055; }
[data-testid="stMetricLabel"] { color:#a9c5e2 !important; font-size:.82rem; }
[data-testid="stMetricValue"] { color:#f3f8ff !important; font-weight:750; }
h1,h2,h3 { color:#f0f7ff !important; letter-spacing:-.025em; }
h1 { font-size:1.8rem !important; }
h2 { font-size:1.25rem !important; }
h3 { font-size:1.02rem !important; }
p, label, .stCaption { color:#b6cce2; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #1a4a78; border-radius:10px; overflow:hidden; }
.stSelectbox > div > div, .stDateInput > div > div, .stTextInput > div > div { background:#0b294a; border-color:#1b5488; color:#eaf4ff; }
.stButton > button, .stDownloadButton > button { background:linear-gradient(90deg,#0877e8,#0c9be8); color:white; border:0; border-radius:8px; font-weight:650; }
.stButton > button:hover, .stDownloadButton > button:hover { border:1px solid #5dc9ff; color:white; }
div[data-testid="stAlert"] { border-radius:10px; border:1px solid #285783; background:#0c2a49; }
hr { border-color:#1a4268; }
.gg-brand { font-size:1.15rem; font-weight:800; color:#f2f8ff; letter-spacing:-.03em; }
.gg-sub { color:#82b9e8; font-size:.72rem; margin-top:-5px; }
.gg-eyebrow { color:#52baff; font-size:.72rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; }
.gg-panel { background:linear-gradient(145deg,#0d2c4e,#09213d); border:1px solid #174775; border-radius:12px; padding:16px 18px; margin:8px 0 14px; }
.gg-muted { color:#9bb6d2; font-size:.82rem; }
.gg-pill { display:inline-block; padding:4px 9px; border-radius:20px; background:#123d64; border:1px solid #226394; color:#8fd5ff; font-size:.72rem; font-weight:700; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown('<div class="gg-brand">⚡ GridGuard AI</div><div class="gg-sub">WEST BENGAL · POWER DEMAND INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.caption("PROJECT STATUS")
        st.markdown('<span class="gg-pill">Historical data mode</span>', unsafe_allow_html=True)
        st.markdown("")
        st.info("Electricity observations are available through 28 April 2024. This is not a live grid-monitoring system.")
        st.markdown("---")
        st.caption("Use the page navigation to explore demand, risk and forecast analysis.")
        st.markdown("---")
        st.caption("GRIDGUARD AI · RESEARCH PROTOTYPE")
