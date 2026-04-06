
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm, multivariate_normal, kendalltau, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  MOUNTAIN PATH BRAND
# ══════════════════════════════════════════════════════════════
BLUE   = "#003366"
GOLD   = "#FFD700"
LB     = "#ADD8E6"
CARD   = "#112240"
TXT    = "#e6f1ff"
MUTED  = "#8892b0"
GRN    = "#28a745"
RED    = "#dc3545"
MID    = "#004d80"
ORANGE = "#fd7e14"
PURPLE = "#6f42c1"
BG_GRAD = "linear-gradient(135deg,#0a1628,#112240,#1a2f4a)"

st.set_page_config(
    page_title="Correlation & Dependency Analysis | Mountain Path",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Background ── */
  .stApp {{ background: {BG_GRAD}; }}
  section[data-testid="stSidebar"] {{ background: #07101f !important; border-right: 1px solid {GOLD}33; }}
  .stApp > header {{ background: transparent !important; }}

  /* ── Typography ── */
  html, body, [class*="css"], [class*="st-"] {{ font-family: 'Inter', sans-serif; color: {TXT}; }}
  * {{ -webkit-font-smoothing: antialiased; }}
  .stMarkdown, .stMarkdown p, .stMarkdown span {{ color:{TXT} !important; }}
  div[data-testid="stVerticalBlock"] p {{ color:{TXT} !important; }}
  div[data-testid="column"] p {{ color:{TXT} !important; }}

  /* ── Sidebar ── */
  .sidebar-logo {{ text-align:center; padding:18px 0 8px; }}
  .sidebar-logo h1 {{ font-family:'Playfair Display',serif; font-size:1.15rem; color:{GOLD}; margin:0; letter-spacing:1px; }}
  .sidebar-logo p  {{ font-size:0.65rem; color:{MUTED}; margin:0; }}
  .nav-label {{ font-size:0.7rem; font-weight:600; color:{GOLD}; text-transform:uppercase; letter-spacing:2px; padding:16px 0 4px; }}

  /* ── Cards ── */
  .mp-card {{
    background: linear-gradient(135deg,{CARD},rgba(0,77,128,0.13));
    border: 1px solid rgba(255,215,0,0.2); border-radius:12px;
    padding:22px; margin:8px 0;
  }}
  .mp-card-accent {{ border-left:4px solid {GOLD}; }}
  .mp-metric {{
    background: linear-gradient(135deg,{CARD},{BLUE}88);
    border:1px solid {GOLD}44; border-radius:10px;
    padding:16px; text-align:center;
  }}
  .mp-metric .val {{ font-size:1.8rem; font-weight:700; color:{GOLD}; font-family:'Playfair Display',serif; }}
  .mp-metric .lbl {{ font-size:0.72rem; color:{MUTED}; margin-top:2px; text-transform:uppercase; letter-spacing:1px; }}

  /* ── Headers ── */
  .page-title {{
    font-family:'Playfair Display',serif;
    font-size:2.4rem; font-weight:900; color:{GOLD};
    border-bottom:2px solid {GOLD}44; padding-bottom:10px; margin-bottom:6px;
  }}
  .page-sub {{ font-size:0.92rem; color:{MUTED}; margin-bottom:24px; }}
  .section-hdr {{
    font-family:'Playfair Display',serif;
    font-size:1.4rem; font-weight:700; color:{GOLD};
    margin: 24px 0 12px; border-left:4px solid {GOLD};
    padding-left:12px;
  }}

  /* ── Tags / badges ── */
  .badge {{ display:inline-block; background:{GOLD}22; color:{GOLD}; border:1px solid {GOLD}55;
            border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:600; margin:2px; }}
  .badge-red {{ background:{RED}22; color:#ff8a8a; border-color:{RED}55; }}
  .badge-green {{ background:{GRN}22; color:#7aff8a; border-color:{GRN}55; }}

  /* ── Streamlit overrides ── */
  div[data-testid="stSelectbox"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stNumberInput"] label {{ color:{TXT} !important; font-size:0.85rem; }}
  .stButton>button {{
    background:linear-gradient(135deg,{BLUE},{MID}); color:{GOLD};
    border:1px solid {GOLD}55; border-radius:8px; font-weight:600;
    padding:8px 20px; transition:all .2s;
  }}
  .stButton>button:hover {{ background:{GOLD}; color:{BLUE}; border-color:{GOLD}; }}
  div[data-testid="stTabs"] button {{ color:{MUTED}; font-size:0.85rem; }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color:{GOLD} !important; border-bottom:2px solid {GOLD} !important;
  }}
  /* Radio button contrast fix */
  div[data-testid="stRadio"] label span {{ color:{TXT} !important; -webkit-text-fill-color:{TXT} !important; }}
  div[data-testid="stRadio"] label {{ color:{TXT} !important; }}
  div[data-testid="stSidebarContent"] div[data-testid="stRadio"] label span {{
    color:{TXT} !important; -webkit-text-fill-color:{TXT} !important;
  }}
  section[data-testid="stSidebar"] label {{
    color:{TXT} !important; -webkit-text-fill-color:{TXT} !important;
  }}
  section[data-testid="stSidebar"] p {{
    color:{TXT} !important; -webkit-text-fill-color:{TXT} !important;
  }}
  div[data-testid="metric-container"] {{
    background:{CARD} !important; border:1px solid rgba(255,215,0,0.2) !important;
    border-radius:8px !important; padding:12px !important;
  }}
  div[data-testid="metric-container"] label,
  div[data-testid="metric-container"] [data-testid="stMetricLabel"],
  div[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
  div[data-testid="metric-container"] [data-testid="stMetricLabel"] div {{
    color:{MUTED} !important; -webkit-text-fill-color:{MUTED} !important; font-size:0.78rem !important;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"],
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] span,
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] div {{
    color:{GOLD} !important; -webkit-text-fill-color:{GOLD} !important;
    font-family:'Playfair Display',serif !important; font-size:1.6rem !important;
  }}
  div[data-testid="stMetricDelta"] {{
    color:{TXT} !important; -webkit-text-fill-color:{TXT} !important;
  }}
  .stExpander {{ background:{CARD}; border:1px solid {GOLD}22; border-radius:8px; }}
  .stExpander summary {{ color:{GOLD}; }}
  .stAlert {{ border-radius:8px; }}
  hr {{ border-color:{GOLD}33; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PLOTLY TEMPLATE
# ══════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,16,31,0.88)",
    font=dict(family="Inter", color=TXT, size=11),
    title_font=dict(family="Playfair Display", color=GOLD, size=13),
    legend=dict(
        bgcolor="rgba(10,22,40,0.92)",
        bordercolor="rgba(255,215,0,0.55)",
        borderwidth=1,
        font=dict(color=TXT, size=10),
        itemsizing="constant",
    ),
    margin=dict(l=55, r=30, t=55, b=50),
    colorway=["#FFD700","#dc3545","#28a745","#ADD8E6","#fd7e14","#6f42c1","#004d80"],
)

AXIS_STYLE = dict(
    gridcolor="rgba(0,51,102,0.5)",
    linecolor="rgba(255,215,0,0.3)",
    zerolinecolor="rgba(255,215,0,0.15)",
    tickfont=dict(color=TXT, size=10),
    title_font=dict(color=MUTED, size=11),
    showgrid=True,
    zeroline=True,
)

def mp_layout(**kwargs):
    """Build layout dict — never include xaxis/yaxis here; apply via update_axes() instead."""
    d = PLOT_LAYOUT.copy()
    # Strip out any axis keys passed by caller to avoid Plotly 5.x validation errors
    axis_overrides = {}
    clean = {}
    for k, v in kwargs.items():
        if k.startswith("xaxis") or k.startswith("yaxis"):
            axis_overrides[k] = v
        else:
            clean[k] = v
    d.update(clean)
    d["_axis_overrides"] = axis_overrides  # carry them separately
    return d

def apply_layout(fig, layout_dict, rows=1, cols=1):
    """Apply layout to fig, then style all axes consistently."""
    ax_overrides = layout_dict.pop("_axis_overrides", {})

    # Extract axis title overrides (xaxis_title -> title for xaxis dict)
    x_title = ax_overrides.pop("xaxis_title", None)
    y_title = ax_overrides.pop("yaxis_title", None)

    fig.update_layout(**layout_dict)

    # Apply default axis style + any per-axis overrides to every subplot axis
    for r in range(1, rows+1):
        for c in range(1, cols+1):
            suffix = "" if (r==1 and c==1) else str((r-1)*cols + c)
            for prefix in ("xaxis", "yaxis"):
                key = f"{prefix}{suffix}"
                style = {**AXIS_STYLE, **(ax_overrides.get(key) or {})}
                fig.update_layout(**{key: style})

    # Apply axis titles to primary axes only
    if x_title:
        try:
            fig.update_xaxes(title_text=x_title, title_font=dict(color=MUTED, size=11), row=1, col=1)
        except Exception:
            fig.update_layout(xaxis=dict(title_text=x_title, title_font=dict(color=MUTED, size=11)))
    if y_title:
        try:
            fig.update_yaxes(title_text=y_title, title_font=dict(color=MUTED, size=11), row=1, col=1)
        except Exception:
            fig.update_layout(yaxis=dict(title_text=y_title, title_font=dict(color=MUTED, size=11)))

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def gaussian_copula_sample(rho, n=1000, seed=42):
    np.random.seed(seed)
    cov = [[1, rho], [rho, 1]]
    Z = np.random.multivariate_normal([0, 0], cov, n)
    return norm.cdf(Z)

def t_copula_sample(rho, nu, n=1000, seed=42):
    np.random.seed(seed)
    cov = [[1, rho], [rho, 1]]
    Z = np.random.multivariate_normal([0, 0], cov, n)
    W = np.random.chisquare(nu, n)
    T = Z / np.sqrt(W[:, None] / nu)
    return stats.t.cdf(T, df=nu)

def clayton_sample(theta, n=1000, seed=42):
    np.random.seed(seed)
    u = np.random.uniform(0, 1, n)
    w = np.random.exponential(1, n)
    v_raw = np.random.uniform(0, 1, n)
    v = (1 + theta * (u**(-theta) - 1) * (-np.log(v_raw)) / w)**(-1/theta)
    v = np.clip(v, 1e-6, 1-1e-6)
    return np.column_stack([u, v])

def gumbel_sample(theta, n=1000, seed=42):
    np.random.seed(seed)
    # Marshall-Olkin algorithm
    u = np.random.uniform(0, 1, n)
    v = np.random.uniform(0, 1, n)
    gamma_rv = np.random.gamma(1/theta, 1, n)
    u_out = np.exp(-(-np.log(u) / gamma_rv)**(1/theta))
    v_out = np.exp(-(-np.log(v) / gamma_rv)**(1/theta))
    return np.column_stack([np.clip(u_out, 1e-6, 1-1e-6),
                            np.clip(v_out, 1e-6, 1-1e-6)])

def wcdr(PD, rho, alpha=0.999):
    return norm.cdf((norm.ppf(PD) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho))

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
st.html("""
<div class="sidebar-logo">
  <h1>⛰️ THE MOUNTAIN PATH</h1>
  <p>WORLD OF FINANCE</p>
  <p style="font-size:0.6rem;color:rgba(255,215,0,0.53);margin-top:4px;">themountainpathacademy.com</p>
</div>
""")

st.sidebar.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

pages = {
    "🏠 Overview": "overview",
    "📖 Introduction & History": "intro",
    "📐 Correlation Explorer": "correlation",
    "🔗 Copula Lab — 2 Assets": "copula2",
    "🔺 Copula Lab — 3 Assets": "copula3",
    "📚 Case Studies": "cases",
    "⚙️ Applications": "applications",
    "🎓 WCDR & Basel II": "wcdr",
}
page_key = st.sidebar.radio("", list(pages.keys()), label_visibility="collapsed")
PAGE = pages[page_key]

st.sidebar.markdown("---")
st.sidebar.html(f"""
<div style="font-size:0.68rem;color:{MUTED};text-align:center;line-height:1.6;">
  <b style="color:{GOLD}">Prof. V. Ravichandran</b><br>
  Professor of Practice &amp; Visiting Faculty<br>
  @ Business Schools India<br><br>
  <a href="https://themountainpathacademy.com" style="color:{GOLD}">
    themountainpathacademy.com
  </a><br>
  <a href="https://linkedin.com/in/trichyravis" style="color:{LB}">LinkedIn</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/trichyravis" style="color:{LB}">GitHub</a>
</div>
""")

# ══════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if PAGE == "overview":
    st.html(f"""
    <div class="page-title">Correlation & Dependency Analysis</div>
    <div class="page-sub">Advanced Financial Risk Management · Mountain Path Academy</div>
    """)

    # Hero metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["4", "5", "3+", "∞"],
        ["Copula Families", "Case Studies", "Asset Classes", "Dependency Structures"]
    ):
        col.html(f"""
        <div class="mp-metric">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""")

    st.markdown("---")
    st.html('<div class="section-hdr">What Is This App?</div>')

    col1, col2 = st.columns([3, 2])
    with col1:
        st.html(f"""
        <div class="mp-card mp-card-accent">
          <p style="color:{TXT};line-height:1.8;font-size:0.92rem;">
          This interactive platform teaches <b style="color:{GOLD}">correlation structures and tail
          dependency</b> — two of the most critical and misunderstood topics in financial risk
          management. From the limitations of Pearson correlation to the power of copula models,
          every concept is backed by live simulation and real-world case studies.
          </p>
          <p style="color:{TXT};line-height:1.8;font-size:0.92rem;">
          Built around the <b style="color:{GOLD}">Gaussian Copula framework</b> used in Basel II IRB
          capital models, with extensions to Student-t, Clayton, and Gumbel families. Each section
          provides interactive sliders, visualisations, and worked numerical examples.
          </p>
        </div>
        """)
    with col2:
        st.html(f"""
        <div class="mp-card">
          <p style="color:{GOLD};font-weight:600;margin-bottom:8px;">📋 Module Contents</p>
          <ul style="color:{TXT};font-size:0.85rem;line-height:2;list-style:none;padding:0;">
            <li>📐 Pearson · Spearman · Kendall</li>
            <li>🔗 2-Asset Gaussian Copula</li>
            <li>🔺 3-Asset Cholesky Extension</li>
            <li>📚 5 Real Market Case Studies</li>
            <li>⚙️ VaR · CVaR · Credit Risk</li>
            <li>🎓 WCDR · Basel II Capital</li>
          </ul>
        </div>
        """)

    st.html('<div class="section-hdr">The Core Challenge</div>')

    # Animated demo: correlation is not enough
    np.random.seed(42)
    fig = make_subplots(1, 3, subplot_titles=[
        "Linear (Pearson works)", "Nonlinear Monotone", "Tail Asymmetric"
    ])
    x = np.random.randn(300)
    datasets = [
        (0.8*x + 0.2*np.random.randn(300), f"ρ={np.corrcoef(x,0.8*x+0.2*np.random.randn(300))[0,1]:.2f}"),
        (np.exp(0.6*x) + 0.1*np.random.randn(300), "Same ρ — different shape"),
        (None, "Copula captures this"),
    ]
    U_cla = clayton_sample(3, 300, seed=7)
    r1_c = norm.ppf(U_cla[:,0], 0, 1); r2_c = norm.ppf(U_cla[:,1], 0, 1)

    for i, (yi, lbl) in enumerate(datasets):
        if i < 2:
            xi_use, yi_use = x, yi
        else:
            xi_use, yi_use = r1_c, r2_c
        fig.add_trace(go.Scatter(
            x=xi_use, y=yi_use, mode='markers',
            marker=dict(size=5, color=[GOLD, GRN, LB][i], opacity=0.75),
            name=lbl, showlegend=True
        ), row=1, col=i+1)

    lo = mp_layout(title="Why Pearson Correlation Alone Is Insufficient",
                   height=320, showlegend=True,
                   legend=dict(orientation="h", y=-0.15))
    for ax in ['xaxis','xaxis2','xaxis3','yaxis','yaxis2','yaxis3']:
        lo[ax] = dict(gridcolor="rgba(0,51,102,0.33)", linecolor="rgba(255,215,0,0.2)",
                      tickfont=dict(color=MUTED), zerolinecolor="rgba(255,215,0,0.13)")
    apply_layout(fig, lo)
    for ann in fig.layout.annotations:
        ann.font.color = GOLD
        ann.font.family = "Playfair Display"
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Key insight:** Two datasets can have identical Pearson correlation but completely different joint behaviour — especially in the tails. Copulas separate the marginal distributions from the dependence structure.")


# ══════════════════════════════════════════════════════════════
#  PAGE: INTRODUCTION & HISTORY
# ══════════════════════════════════════════════════════════════
elif PAGE == "intro":
    st.html(f'''<div class="page-title">Introduction &amp; History of Copulas</div>''')
    st.html(f'''<div class="page-sub">From Markowitz (1952) to Basel III &mdash; How Crises Built the Copula Framework</div>''')

    tab_int, tab_pre, tab_warn, tab_li, tab_cdo, tab_aft, tab_skl, tab_found = st.tabs([
        "📋 Why This Matters", "📊 Pre-Copula Era", "⚡ Warning Shots",
        "📄 Li (2000) Paper", "🏦 CDO Boom & Flaw", "🔥 2008 & Aftermath",
        "🧮 Sklar Theorem", "🔬 Copula Foundations",
    ])

    with tab_int:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.html(f"""
            <div class="mp-card mp-card-accent">
              <p style="color:{GOLD};font-family:Playfair Display,serif;font-size:1.15rem;font-weight:700;margin-bottom:10px;">The Central Question of Financial Risk</p>
              <p style="color:{TXT};line-height:1.9;font-size:0.92rem;">
              Finance has always rested on one deceptively simple question:
              <b style="color:{GOLD}">what happens to my portfolio when everything goes wrong at once?</b>
              For most of the twentieth century, that question was answered with a single number &mdash; correlation &mdash;
              embedded inside a Gaussian distribution. That answer, as 1987, 1998, and 2008 demonstrated,
              was <b style="color:{RED}">catastrophically incomplete.</b>
              </p>
              <p style="color:{TXT};line-height:1.9;font-size:0.92rem;margin-top:10px;">
              Copulas emerged as the mathematical response. They separate two questions that standard
              correlation silently conflates: <i>how does each asset behave on its own?</i> and
              <i>how do the assets move in relation to each other?</i>
              </p>
            </div>
            <div class="mp-card" style="margin-top:12px;">
              <p style="color:{GOLD};font-weight:700;margin-bottom:8px;">Module Contents</p>
              <ul style="color:{TXT};font-size:0.88rem;line-height:2.1;list-style:none;padding:0;">
                <li>📊 <b style="color:{GOLD}">Pre-Copula Era (1952-1994)</b> - Markowitz, Black-Scholes, RiskMetrics and the Gaussian illusion</li>
                <li>⚡ <b style="color:{GOLD}">Warning Shots</b> - Black Monday (1987), Asian Crisis (1997), LTCM collapse (1998)</li>
                <li>📄 <b style="color:{GOLD}">The Li Paper (2000)</b> - How one equation changed structured finance forever</li>
                <li>🏦 <b style="color:{GOLD}">CDO Boom & Fatal Flaw (2000-2008)</b> - Zero tail dependence and the rating agency failure</li>
                <li>🔥 <b style="color:{GOLD}">2008 Crisis & Aftermath</b> - Basel III, FRTB, and the new paradigm</li>
                <li>🧮 <b style="color:{GOLD}">Sklar Theorem (1959)</b> - The mathematics that predated the finance by 40 years</li>
                <li>🔬 <b style="color:{GOLD}">Copula Foundations</b> - What a copula is and why correlation is insufficient</li>
              </ul>
            </div>
            """)
        with c2:
            milestones = [
                ("1952","Markowitz","Mean-variance framework born",BLUE),
                ("1973","Black-Scholes","Options pricing; Gaussian world peaks",MID),
                ("1987","Black Monday","Correlations surge to 0.95+",RED),
                ("1994","RiskMetrics","VaR becomes industry standard",MID),
                ("1998","LTCM","Correlation diverges, not converges",RED),
                ("2000","Li Copula","Gaussian copula adopted for CDOs",GOLD),
                ("2008","GFC","Copula blamed; trillions lost",RED),
                ("2010","Basel III","Stressed correlations mandated",GRN),
                ("2016","FRTB","CVaR replaces VaR; copulas central",GRN),
            ]
            for year, event, desc, col in milestones:
                txt_col = BLUE if col == GOLD else TXT
                st.html(f"""<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;">
                  <div style="min-width:42px;background:{col};color:{txt_col};border-radius:6px;
                    padding:3px 6px;font-size:0.72rem;font-weight:700;text-align:center;">{year}</div>
                  <div><div style="color:{GOLD};font-size:0.82rem;font-weight:600;">{event}</div>
                  <div style="color:{MUTED};font-size:0.75rem;">{desc}</div></div></div>""")

    with tab_pre:
        st.html(f'''<div class="section-hdr">The Pre-Copula Era: Correlation and Its Comfortable Illusion</div>''')
        c1, c2 = st.columns(2)
        with c1:
            st.html(f"""
            <div class="mp-card mp-card-accent">
              <p style="color:{GOLD};font-weight:700;font-size:1rem;">Markowitz & MPT (1952)</p>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              The covariance matrix Σ captured how assets moved together. The entire edifice rested on:
              <b style="color:{GOLD}">returns are jointly normally distributed.</b>
              Under joint normality, the correlation matrix fully specifies the dependence structure
              in the centre of the distribution <i>and in the tails.</i>
              This makes the Gaussian world tractable, and <b style="color:{RED}">dangerously misleading.</b>
              </p>
            </div>
            <div class="mp-card" style="margin-top:12px;">
              <p style="color:{GOLD};font-weight:700;font-size:1rem;">Black-Scholes-Merton (1973)</p>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              BSM embedded the Gaussian assumption even more deeply. Log-returns are normally distributed,
              volatility is constant, no jumps. This powered the entire modern derivatives industry and
              underpinned every major bank risk system by the 1980s.
              </p>
            </div>
            """)
        with c2:
            st.html(f"""
            <div class="mp-card">
              <p style="color:{GOLD};font-weight:700;font-size:1rem;">J.P. Morgan RiskMetrics & VaR (1994)</p>
              <div style="background:#07101f;border-radius:8px;padding:12px;margin:10px 0;text-align:center;">
                <code style="color:{GOLD};font-size:0.95rem;">VaR_alpha = mu_P - z_alpha * sigma_P</code><br>
                <code style="color:{LB};font-size:0.85rem;">sigma_P^2 = w^T * Sigma * w</code>
              </div>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              VaR was <b style="color:{GOLD}">elegant, fast,</b> and
              <b style="color:{RED}">wrong in ways that would take another decade to become apparent.</b>
              </p>
            </div>
            <div class="mp-card" style="margin-top:12px;border-left:4px solid {RED};">
              <p style="color:{RED};font-weight:700;">Hidden Consequence of Joint Normality</p>
              <ul style="color:{TXT};font-size:0.86rem;line-height:1.9;padding-left:16px;margin:6px 0 0;">
                <li><b style="color:{GOLD}">Zero tail dependence:</b> lambda_L = lambda_U = 0. As u to 0,
                  joint extreme probability goes to zero.</li>
                <li><b style="color:{RED}">Reality:</b> Markets disagree, repeatedly and violently.</li>
              </ul>
            </div>
            """)
        st.html(f'''<div class="section-hdr" style="margin-top:16px;">Interactive: Gaussian Tail Dependence = 0 for Any rho</div>''')
        rho_pre = st.slider("Correlation rho", -0.95, 0.95, 0.70, 0.05, key="pre_rho")
        U_pre = gaussian_copula_sample(rho_pre, 2000, seed=42)
        mask_pre = (U_pre[:,0] < 0.05) & (U_pre[:,1] < 0.05)
        fig_pre = go.Figure()
        fig_pre.add_trace(go.Scatter(x=U_pre[:,0], y=U_pre[:,1], mode="markers",
            marker=dict(color=LB, size=4, opacity=0.55), name="Joint sample"))
        fig_pre.add_trace(go.Scatter(x=U_pre[mask_pre,0], y=U_pre[mask_pre,1], mode="markers",
            marker=dict(color=RED, size=9, symbol="x", line=dict(color="white", width=1)),
            name=f"Joint crashes: {mask_pre.sum()} (indep expects {int(0.05**2*2000)})"))
        for v in [0.05, 0.95]:
            fig_pre.add_vline(x=v, line_dash="dot", line_color=GOLD, line_width=1)
            fig_pre.add_hline(y=v, line_dash="dot", line_color=GOLD, line_width=1)
        apply_layout(fig_pre, mp_layout(
            title=f"Gaussian Copula (ρ={rho_pre}) — Tail Dependence λ = 0 Always",
            xaxis_title="U₁ = Φ(Z₁)", yaxis_title="U₂ = Φ(Z₂)", height=380))
        fig_pre.update_xaxes(range=[-0.02, 1.02])
        fig_pre.update_yaxes(range=[-0.02, 1.02])
        st.plotly_chart(fig_pre, use_container_width=True)
        st.info(f"Even at rho = {rho_pre}, the Gaussian copula has zero tail dependence. Joint crashes: {mask_pre.sum()} vs {int(0.05**2*2000)} expected under independence.")

    with tab_warn:
        st.html(f'''<div class="section-hdr">The Warning Shots: Crises That Exposed the Models Limits</div>''')
        crises_data = [
            {
                "title": "Black Monday - October 19, 1987", "col": RED,
                "what": "The Dow fell 22.6% in one day. Equity markets worldwide fell simultaneously. Diversification benefits built on correlations of 0.3-0.5 evaporated in one session.",
                "corr": "Pre-crash avg pairwise correlation between major equity indices: 0.35. During crash: implied correlations exceeded 0.95. A model calibrated on 5 prior years would estimate VaR at one-third of the actual loss.",
                "lesson": "The Gaussian model had no mechanism to express that 'in a crash, everything falls together.' A t-copula or Clayton copula would have captured lower-tail clustering.",
            },
            {
                "title": "Asian Financial Crisis - 1997", "col": ORANGE,
                "what": "Thailand baht devaluation triggered sequential currency crises across Indonesia, South Korea, Malaysia, and the Philippines. The Indonesian Rupiah lost 80% of its value.",
                "corr": "Standard correlation models dramatically understated joint crash risk. Cross-currency correlations in the lower tail were near 1.0 vs normal-period estimates of 0.3-0.5.",
                "lesson": "Clayton copulas calibrated on stressed data would have shown lambda_L > 0.6 for these currency pairs. The Gaussian assumption led to systematic underestimation of joint currency risk.",
            },
            {
                "title": "LTCM Collapse - August 1998", "col": "#ff6b6b",
                "what": "LTCM employed Nobel laureates and held convergence trades betting on correlations returning to historical norms. Russia defaulted; every correlation assumption failed simultaneously. Loss: USD 4.6 billion in four months.",
                "corr": "LTCM assumed the same correlation structure in the left tail as in the centre. Correlations between liquid and illiquid assets jumped from 0.3 to over 0.9.",
                "lesson": "A copula with asymmetric tail dependence would have captured the non-linear change in co-movement under stress. Gaussian copula zero tail dependence made this scenario seem nearly impossible.",
            },
        ]
        for crisis in crises_data:
            c_title  = crisis["title"]
            c_col    = crisis["col"]
            c_what   = crisis["what"]
            c_corr   = crisis["corr"]
            c_lesson = crisis["lesson"]
            st.html(f"""
            <div style="border:1px solid {c_col}55;border-radius:10px;margin-bottom:16px;overflow:hidden;">
              <div style="background:{c_col}22;padding:12px 18px;border-bottom:1px solid {c_col}44;">
                <span style="color:{c_col};font-weight:700;font-size:0.95rem;">{c_title}</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;padding:14px;">
                <div style="background:#07101f;border-radius:8px;padding:14px;">
                  <p style="color:{c_col};font-weight:700;font-size:0.88rem;margin-bottom:8px;">What Happened</p>
                  <p style="color:{TXT};font-size:0.84rem;line-height:1.75;margin:0;">{c_what}</p>
                </div>
                <div style="background:#07101f;border-radius:8px;padding:14px;">
                  <p style="color:{GOLD};font-weight:700;font-size:0.88rem;margin-bottom:8px;">Correlation Evidence</p>
                  <p style="color:{TXT};font-size:0.84rem;line-height:1.75;margin:0;">{c_corr}</p>
                </div>
                <div style="background:#07101f;border-radius:8px;padding:14px;border-left:3px solid {GOLD};">
                  <p style="color:{GOLD};font-weight:700;font-size:0.88rem;margin-bottom:8px;">Copula Lesson</p>
                  <p style="color:{TXT};font-size:0.84rem;line-height:1.75;margin:0;">{c_lesson}</p>
                </div>
              </div>
            </div>
            """)

    with tab_li:
        st.html(f'''<div class="section-hdr">The Copula Paper That Changed Finance: David X. Li (2000)</div>''')
        c1, c2 = st.columns([3, 2])
        with c1:
            st.html(f"""
            <div class="mp-card mp-card-accent">
              <p style="color:{GOLD};font-weight:700;font-size:1rem;margin-bottom:8px;">
                "On Default Correlation: A Copula Function Approach" - Journal of Fixed Income, 2000
              </p>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              Li applied Sklar (1959) copula theorem to model
              <b style="color:{GOLD}">joint probability of default</b> for multiple credit obligors.
              For survival times T_i, T_j:
              </p>
              <div style="background:#07101f;border-radius:8px;padding:14px;margin:12px 0;text-align:center;">
                <code style="color:{GOLD};font-size:1rem;">P(Ti le ti, Tj le tj) = C_rho_Ga(Fi(ti), Fj(tj))</code>
              </div>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              Where F_i is calibrated from CDS spreads, rho from historical equity correlations.
              By 2003: standard CDO pricing model. By 2006: embedded in every major bank.
              </p>
            </div>
            """)
            reasons = [
                ("One number per pair", "Just rho_ij via CORREL(). No complex optimisation."),
                ("Closed-form", "No numerical integration. 100+ name CDOs priced fast."),
                ("Modular", "Any survival distribution + any correlation. Mix and match."),
                ("Excel-friendly", "NORM.S.INV, CORREL, SQRT - implementable in a spreadsheet."),
            ]
            cols_r = st.columns(2)
            for i, (h, d) in enumerate(reasons):
                cols_r[i%2].html(f"""<div style="background:#07101f;border-radius:8px;padding:10px;margin-bottom:8px;">
                  <p style="color:{GOLD};font-size:0.82rem;font-weight:600;margin-bottom:3px;">{h}</p>
                  <p style="color:{TXT};font-size:0.80rem;">{d}</p></div>""")
        with c2:
            st.html(f'''<p style="color:{GOLD};font-weight:700;margin-bottom:10px;">Adoption Timeline</p>''')
            for year, text, col in [
                ("2000","Li publishes Gaussian copula for CDOs",BLUE),
                ("2001","First CDOs priced using the model",MID),
                ("2003","Standard at all structured credit desks",GRN),
                ("2004","Rating agencies adopt for tranche ratings",GRN),
                ("2006","Embedded in every major bank",GOLD),
                ("2007","First cracks - model begins failing",ORANGE),
                ("2008","Collapse - trillions in losses",RED),
                ("2009","Formula That Killed Wall St - Wired",RED),
            ]:
                txt_col = BLUE if col == GOLD else TXT
                st.html(f"""<div style="display:flex;gap:8px;margin-bottom:9px;">
                  <span style="background:{col};color:{txt_col};border-radius:5px;padding:2px 8px;
                    font-size:0.7rem;font-weight:700;min-width:36px;text-align:center;">{year}</span>
                  <span style="color:{TXT};font-size:0.82rem;">{text}</span></div>""")

    with tab_cdo:
        st.html(f'''<div class="section-hdr">The Structured Credit Boom and the Copula Fatal Flaw (2000-2008)</div>''')
        c1, c2 = st.columns(2)
        with c1:
            st.html(f"""
            <div class="mp-card mp-card-accent">
              <p style="color:{GOLD};font-weight:700;font-size:1rem;">How the CDO Machine Worked</p>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              A CDO pools mortgages/bonds and repackages cash flows into tranches.
              Key question: <b style="color:{GOLD}">how many obligors default simultaneously?</b>
              The Gaussian copula: calibrate rho 0.05-0.30, run Monte Carlo, compute expected tranche losses.
              Rating agencies awarded <b style="color:{GRN}">AAA ratings</b> to senior tranches.
              </p>
            </div>
            """)
        with c2:
            flaws = [
                ("1. Parameter Instability", "rho 0.10-0.20 calibrated on 2003-06 bull market. No credit stress in calibration."),
                ("2. Zero Tail Dependence", "lambda_L = lambda_U = 0. A t-copula produces 3-8x more joint defaults in the tail."),
                ("3. Correlation Homogeneity", "One rho for all pairs. Subprime mortgages same city/year had correlations above 0.90."),
            ]
            st.html(f'''<div class="mp-card" style="border-left:4px solid {RED};">
              <p style="color:{RED};font-weight:700;font-size:1rem;">Three Fatal Flaws</p>''' +
                "".join([f'''<div style="background:#07101f;border-radius:6px;padding:10px;margin-bottom:8px;">
                  <p style="color:{GOLD};font-size:0.85rem;font-weight:600;margin-bottom:3px;">{h}</p>
                  <p style="color:{TXT};font-size:0.82rem;">{d}</p></div>''' for h,d in flaws]) +
            '''</div>''')
        st.html(f'''<div class="section-hdr" style="margin-top:16px;">CDO Tranche Loss Probability vs Asset Correlation</div>''')
        rho_cdo = st.slider("Li model assumed rho", 0.05, 0.50, 0.10, 0.01, key="cdo_rho")
        rho_pts_cdo = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        np.random.seed(7)
        eq_l_c, mz_l_c, sr_l_c = [], [], []
        for r in rho_pts_cdo:
            Z_c = np.random.randn(5000)
            eps_c = np.random.randn(5000, 100)
            X_c = r*Z_c[:,None] + np.sqrt(1-r**2)*eps_c
            defs_c = (X_c < norm.ppf(0.03)).sum(axis=1)/100
            eq_l_c.append(float(np.mean(defs_c > 0.03)))
            mz_l_c.append(float(np.mean(defs_c > 0.07)))
            sr_l_c.append(float(np.mean(defs_c > 0.12)))
        fig_cdo = go.Figure()
        fig_cdo.add_trace(go.Scatter(x=rho_pts_cdo, y=eq_l_c, name="Equity (>3% loss)", line=dict(color=RED, width=2.5), mode="lines+markers"))
        fig_cdo.add_trace(go.Scatter(x=rho_pts_cdo, y=mz_l_c, name="Mezzanine (>7%)", line=dict(color=ORANGE, width=2.5), mode="lines+markers"))
        fig_cdo.add_trace(go.Scatter(x=rho_pts_cdo, y=sr_l_c, name="Senior AAA (>12%)", line=dict(color=BLUE, width=2.5), mode="lines+markers"))
        fig_cdo.add_vline(x=rho_cdo, line_dash="dash", line_color=GOLD, annotation_text=f"Li model<br>rho={rho_cdo}", annotation_font_color=GOLD, annotation_font_size=10, annotation_yshift=20)
        fig_cdo.add_vline(x=0.75, line_dash="dash", line_color=RED, annotation_text="Crisis<br>rho=0.75+", annotation_font_color=RED, annotation_font_size=10, annotation_yshift=40)
        apply_layout(fig_cdo, mp_layout(title="CDO Tranche Loss Probability vs Asset Correlation", xaxis_title="Correlation rho", yaxis_title="Tranche Loss Probability", height=380))
        st.plotly_chart(fig_cdo, use_container_width=True)
        closest_i = min(range(len(rho_pts_cdo)), key=lambda i: abs(rho_pts_cdo[i]-rho_cdo))
        st.error(f"At Li model rho = {rho_cdo}: Senior AAA loss prob = {sr_l_c[closest_i]*100:.1f}%. At crisis rho = 0.75: dramatically higher. The entire CDO rating system was built on the left side of this chart.")

    with tab_aft:
        st.html(f'''<div class="section-hdr">2008 Crisis and Its Aftermath: A New Approach to Dependence</div>''')
        c1, c2 = st.columns(2)
        with c1:
            st.html(f"""
            <div class="mp-card" style="border-left:4px solid {RED};">
              <p style="color:{RED};font-weight:700;font-size:1rem;margin-bottom:8px;">September 15, 2008 - Lehman Brothers</p>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              The structured credit market collapsed. CDO tranches rated AAA traded at
              <b style="color:{RED}">30-40 cents on the dollar.</b> Banks that used the Gaussian copula
              to justify thin capital buffers faced insolvency.
              Wired (Feb 2009): <i style="color:{GOLD}">"Recipe for Disaster: The Formula That Killed Wall Street"</i>
              </p>
            </div>
            <div class="mp-card" style="margin-top:12px;">
              <p style="color:{GOLD};font-weight:700;margin-bottom:8px;">Evolution of Dependence Models</p>
              <table style="width:100%;border-collapse:collapse;font-size:0.8rem;">
                <thead><tr style="border-bottom:1px solid rgba(255,215,0,0.3);">
                  <th style="color:{GOLD};padding:6px;text-align:left;">Era</th>
                  <th style="color:{GOLD};padding:6px;text-align:left;">Model</th>
                  <th style="color:{GOLD};padding:6px;text-align:left;">Flaw Exposed</th>
                </tr></thead>
                <tbody>
                  <tr style="border-bottom:1px solid rgba(0,51,102,0.4);"><td style="padding:6px;color:{TXT};">Pre-1987</td><td style="padding:6px;color:{GOLD};">Markowitz covariance</td><td style="padding:6px;color:{MUTED};">Ignores fat tails</td></tr>
                  <tr style="border-bottom:1px solid rgba(0,51,102,0.4);"><td style="padding:6px;color:{TXT};">1987-1994</td><td style="padding:6px;color:{GOLD};">RiskMetrics VaR</td><td style="padding:6px;color:{MUTED};">Constant correlation</td></tr>
                  <tr style="border-bottom:1px solid rgba(0,51,102,0.4);"><td style="padding:6px;color:{TXT};">1994-2008</td><td style="padding:6px;color:{GOLD};">Gaussian copula</td><td style="padding:6px;color:{MUTED};">Zero tail dependence</td></tr>
                  <tr style="border-bottom:1px solid rgba(0,51,102,0.4);"><td style="padding:6px;color:{TXT};">Post-2008</td><td style="padding:6px;color:{GOLD};">t-Copula, Archimedean</td><td style="padding:6px;color:{MUTED};">Symmetric vs asymmetric</td></tr>
                  <tr><td style="padding:6px;color:{TXT};">Post-2013</td><td style="padding:6px;color:{GOLD};">Vine, DCC-GARCH</td><td style="padding:6px;color:{MUTED};">Time-varying, high-dim</td></tr>
                </tbody>
              </table>
            </div>
            """)
        with c2:
            reg_items = [
                ("Basel III (2010)", "Banks must compute VaR under both normal and stressed correlation matrices. No longer optional."),
                ("CVaR replaces VaR (FRTB 2016)", "Expected Shortfall penalises zero-tail-dependence models. Capital must cover the full tail."),
                ("Correlation Bounds", "Standardised approach specifies rho ranges within and between risk classes."),
                ("Model Validation", "Backtesting must confirm joint loss scenarios are not systematically underestimated."),
            ]
            st.html(f'''<div class="mp-card"><p style="color:{GOLD};font-weight:700;font-size:1rem;margin-bottom:8px;">Regulatory Response: Basel III & FRTB</p>''' +
                "".join([f'''<div style="background:#07101f;border-radius:6px;padding:10px;margin-bottom:8px;">
                  <p style="color:{GRN};font-size:0.85rem;font-weight:600;">{h}</p>
                  <p style="color:{TXT};font-size:0.82rem;">{d}</p></div>''' for h,d in reg_items]) +
            '''</div>''')

    with tab_skl:
        st.html(f'''<div class="section-hdr">The Mathematical Origins: Sklar Theorem (1959)</div>''')
        c1, c2 = st.columns([2, 3])
        with c1:
            st.html(f"""
            <div class="mp-card mp-card-accent">
              <p style="color:{GOLD};font-weight:700;font-size:1rem;margin-bottom:8px;">Abe Sklar (1959)</p>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              The mathematics predates the finance by <b style="color:{GOLD}">four decades.</b>
              Sklar published in 1959 in probabilistic metric spaces - entirely without financial applications.
              For thirty years copulas were used only in actuarial science.
              The machinery was <b>complete and waiting.</b>
              It took a sequence of financial crises to drive practitioners to discover it.
              </p>
              <div style="background:#07101f;border-radius:8px;padding:14px;margin-top:12px;">
                <p style="color:{GOLD};font-size:0.9rem;font-weight:700;margin-bottom:6px;">Sklar Theorem</p>
                <p style="color:{TXT};font-size:0.85rem;line-height:1.7;">For any joint CDF H(x1,...,xn) with marginals F1,...,Fn, there exists a unique copula C such that:</p>
                <div style="text-align:center;padding:10px;">
                  <code style="color:{GOLD};font-size:0.92rem;">H(x1,...,xn) = C(F1(x1),...,Fn(xn))</code>
                </div>
                <p style="color:{LB};font-size:0.82rem;">Any joint distribution = copula + individual marginals.</p>
              </div>
            </div>
            """)
        with c2:
            st.html(f'''<p style="color:{GOLD};font-weight:600;font-size:1rem;margin-bottom:10px;">From Pure Mathematics to Market Standard</p>''')
            timeline_items = [
                ("1959","Sklar publishes in probabilistic metric spaces. No financial application intended.",BLUE,"🧮"),
                ("1960s-80s","Copulas used in actuarial science and extreme value theory only.",MID,"📊"),
                ("1992","Genest & Rivest formalise Archimedean copulas for applied statistical work.",GRN,"📐"),
                ("1999","Embrechts, McNeil & Straumann connect copulas explicitly to financial risk.",GRN,"💡"),
                ("2000","David X. Li publishes Gaussian copula for default correlation. Finance discovers copulas.",GOLD,"📄"),
                ("2001-06","Rapid Wall Street adoption. CDO pricing standardises on the Gaussian copula.",ORANGE,"🏦"),
                ("2008","Gaussian copula blamed in media and regulatory reports for CDO mispricing.",RED,"🔥"),
                ("2009-13","t-Copula, Archimedean families, and Vine copulas adopted as replacements.",GRN,"✅"),
                ("2016","FRTB mandates stress-tested dependence modelling. Copulas become regulatory.",GRN,"🏛️"),
                ("2020+","Dynamic copulas (DCC-GARCH + copula) become standard in bank IMM models.",GRN,"🚀"),
            ]
            for year, desc, col, icon in timeline_items:
                txt_col = BLUE if col == GOLD else TXT
                st.html(f"""<div style="display:flex;gap:10px;margin-bottom:9px;align-items:flex-start;">
                  <div style="display:flex;flex-direction:column;align-items:center;min-width:70px;">
                    <div style="background:{col};color:{txt_col};border-radius:6px;padding:3px 8px;font-size:0.72rem;font-weight:700;text-align:center;">{year}</div>
                    <div style="font-size:1rem;margin-top:3px;">{icon}</div>
                  </div>
                  <div style="background:#07101f;border-radius:6px;padding:8px 12px;flex:1;">
                    <p style="color:{TXT};font-size:0.82rem;margin:0;line-height:1.6;">{desc}</p>
                  </div></div>""")

    with tab_found:
        st.html(f'''<div class="section-hdr">What Is a Copula? - Core Foundations</div>''')
        c1, c2 = st.columns(2)
        with c1:
            st.html(f"""
            <div class="mp-card mp-card-accent">
              <p style="color:{GOLD};font-weight:700;font-size:1rem;margin-bottom:8px;">Core Definition</p>
              <p style="color:{TXT};font-size:0.88rem;line-height:1.85;">
              A <b style="color:{GOLD}">copula</b> is a multivariate probability distribution on [0,1]^n
              with uniform marginal distributions. It models <b>only the dependence structure</b>,
              completely separated from individual marginals.
              </p>
              <div style="background:#07101f;border-radius:8px;padding:12px;margin:12px 0;">
                <p style="color:{GOLD};font-size:0.88rem;font-weight:600;margin-bottom:6px;">Three Separate Questions</p>
                <ol style="color:{TXT};font-size:0.85rem;line-height:1.9;padding-left:16px;margin:0;">
                  <li><b style="color:{LB}">What does Asset X look like alone?</b> Marginal F_X</li>
                  <li><b style="color:{LB}">What does Asset Y look like alone?</b> Marginal F_Y</li>
                  <li><b style="color:{GOLD}">How do X and Y move together?</b> Copula C</li>
                </ol>
              </div>
              <p style="color:{GOLD};font-weight:700;margin-bottom:8px;">Why Not Just Use Correlation? - 4 Key Reasons</p>
              <ul style="color:{TXT};font-size:0.85rem;line-height:2;list-style:none;padding:0;">
                <li><span style="color:{RED};">1</span> <b>Linear only</b> - measures only linear co-movement</li>
                <li><span style="color:{RED};">2</span> <b>Not margin-free</b> - mixes marginals into dependence</li>
                <li><span style="color:{RED};">3</span> <b>Tail blindness</b> - silent on joint extreme events</li>
                <li><span style="color:{RED};">4</span> <b>Not invariant</b> - Corr(X,Y) not equal Corr(f(X),g(Y))</li>
              </ul>
            </div>
            """)
        with c2:
            st.html(f"""
            <div class="mp-card">
              <p style="color:{GOLD};font-weight:700;font-size:1rem;margin-bottom:8px;">Copula Family Summary</p>
              <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
                <thead><tr style="border-bottom:1px solid rgba(255,215,0,0.3);">
                  <th style="color:{GOLD};padding:7px;text-align:left;">Copula</th>
                  <th style="color:{GOLD};padding:7px;text-align:center;">lambda_L</th>
                  <th style="color:{GOLD};padding:7px;text-align:center;">lambda_U</th>
                  <th style="color:{GOLD};padding:7px;text-align:left;">Best For</th>
                </tr></thead>
                <tbody>
                  <tr style="border-bottom:1px solid rgba(0,51,102,0.4);"><td style="padding:7px;color:{LB};">Gaussian</td><td style="padding:7px;text-align:center;color:{MUTED};">0</td><td style="padding:7px;text-align:center;color:{MUTED};">0</td><td style="padding:7px;color:{TXT};">Baseline; large portfolios</td></tr>
                  <tr style="border-bottom:1px solid rgba(0,51,102,0.4);"><td style="padding:7px;color:{RED};">Student-t</td><td style="padding:7px;text-align:center;color:{GRN};">+</td><td style="padding:7px;text-align:center;color:{GRN};">+</td><td style="padding:7px;color:{TXT};">Equity; symmetric tails</td></tr>
                  <tr style="border-bottom:1px solid rgba(0,51,102,0.4);"><td style="padding:7px;color:{GRN};">Clayton</td><td style="padding:7px;text-align:center;color:{GRN};">+</td><td style="padding:7px;text-align:center;color:{MUTED};">0</td><td style="padding:7px;color:{TXT};">Credit defaults; crashes</td></tr>
                  <tr style="border-bottom:1px solid rgba(0,51,102,0.4);"><td style="padding:7px;color:{ORANGE};">Gumbel</td><td style="padding:7px;text-align:center;color:{MUTED};">0</td><td style="padding:7px;text-align:center;color:{GRN};">+</td><td style="padding:7px;color:{TXT};">Commodity booms; rallies</td></tr>
                  <tr><td style="padding:7px;color:{MUTED};">Frank</td><td style="padding:7px;text-align:center;color:{MUTED};">0</td><td style="padding:7px;text-align:center;color:{MUTED};">0</td><td style="padding:7px;color:{TXT};">Mild symmetric; neg rho</td></tr>
                </tbody>
              </table>
            </div>
            """)
            st.html(f'''<p style="color:{GOLD};font-weight:600;margin-top:14px;">Interactive Sklar Demo</p>''')
            rho_found = st.slider("Copula rho (Gaussian)", -0.9, 0.9, 0.65, 0.05, key="found_rho")
            np.random.seed(42)
            U_fd = gaussian_copula_sample(rho_found, 1200, seed=42)
            X_fd1 = stats.skewnorm.ppf(U_fd[:,0], a=4, loc=0.001, scale=0.012)
            X_fd2 = stats.t.ppf(U_fd[:,1], df=4, loc=0.0005, scale=0.009)
            fig_fd = make_subplots(1, 2, subplot_titles=["Copula Space (uniform)", "Output (mixed marginals)"])
            fig_fd.add_trace(go.Scatter(x=U_fd[:,0], y=U_fd[:,1], mode="markers", marker=dict(color=GOLD, size=4, opacity=0.65), name="Copula"), 1, 1)
            fig_fd.add_trace(go.Scatter(x=X_fd1, y=X_fd2, mode="markers", marker=dict(color=LB, size=4, opacity=0.65), name="Skewed-t margins"), 1, 2)
            apply_layout(fig_fd, mp_layout(title=f"Same Copula (rho={rho_found}), Different Marginals", height=300), rows=1, cols=2)
            for ann in fig_fd.layout.annotations: ann.font.color = GOLD; ann.font.size = 11
            fig_fd.update_xaxes(range=[-0.02, 1.02], col=1)
            fig_fd.update_yaxes(range=[-0.02, 1.02], col=1)
            st.plotly_chart(fig_fd, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: CORRELATION EXPLORER
# ══════════════════════════════════════════════════════════════
elif PAGE == "correlation":
    st.html('<div class="page-title">Correlation Explorer</div>')
    st.html(f'<div class="page-sub">Compare Pearson, Spearman, and Kendall across relationship types</div>')

    tab1, tab2, tab3 = st.tabs(["📊 Measures Comparison", "🌡️ Correlation Matrix", "⏱️ Rolling Correlation"])

    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            rel_type = st.selectbox("Relationship Type", [
                "Linear (ρ=0.8)", "Quadratic (non-monotone)", "Exponential (monotone)",
                "Sine wave", "Independent", "Negative linear"
            ], key="corr_rel_type")
            n_obs = st.slider("Observations", 100, 1000, 400, 50, key="corr_obs")
            noise = st.slider("Noise level", 0.0, 2.0, 0.3, 0.1, key="corr_noise")

        np.random.seed(42)
        x = np.random.randn(n_obs)
        eps = noise * np.random.randn(n_obs)
        if rel_type == "Linear (ρ=0.8)":    y = 0.8*x + eps
        elif rel_type == "Quadratic (non-monotone)": y = x**2 + eps
        elif rel_type == "Exponential (monotone)": y = np.exp(0.5*x) + eps
        elif rel_type == "Sine wave": y = np.sin(2*x) + 0.3*eps
        elif rel_type == "Independent": y = np.random.randn(n_obs)
        else: y = -0.8*x + eps

        pr = np.corrcoef(x, y)[0,1]
        sr, _ = spearmanr(x, y)
        kr, _ = kendalltau(x, y)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(color=GOLD, size=5, opacity=0.80,
                            line=dict(color=BLUE, width=0.3)),
                name="Data"
            ))
            apply_layout(fig, mp_layout(
                title=f"Scatter: {rel_type}",
                height=380, xaxis_title="X", yaxis_title="Y"
            ))
            st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Pearson ρ", f"{pr:.4f}", help="Linear correlation")
        m2.metric("Spearman ρₛ", f"{sr:.4f}", help="Rank-based monotone")
        m3.metric("Kendall τ", f"{kr:.4f}", help="Concordance-based")

        with st.expander("📖 Interpretation Guide"):
            st.html(f"""
            <div class="mp-card">
            <ul style="color:{TXT};font-size:0.87rem;line-height:2;">
              <li><b style="color:{GOLD}">Pearson ρ</b>: Measures <i>linear</i> co-movement. Fails for nonlinear relationships. Sensitive to outliers.</li>
              <li><b style="color:{GOLD}">Spearman ρₛ</b>: Pearson on ranks. Captures any <i>monotone</i> relationship. Robust to outliers.</li>
              <li><b style="color:{GOLD}">Kendall τ</b>: Net concordance of pairs. Has a direct copula interpretation: τ = 4∫C dC − 1.</li>
              <li><b style="color:{RED}">Warning</b>: ρ = 0 does NOT imply independence (only for jointly normal data).</li>
            </ul>
            </div>
            """)

    with tab2:
        st.html(f'<div class="section-hdr">Multi-Asset Correlation Matrix</div>')
        assets = ["Nifty 50", "Bank Nifty", "Nifty IT", "Gold", "USD/INR", "10Y G-Bond", "Oil", "REIT"]
        C = np.array([
            [1.00, 0.88, 0.75, -0.25,  0.15, -0.30,  0.40,  0.35],
            [0.88, 1.00, 0.65, -0.18,  0.12, -0.28,  0.38,  0.40],
            [0.75, 0.65, 1.00, -0.20,  0.10, -0.35,  0.30,  0.30],
            [-0.25,-0.18,-0.20, 1.00, -0.40,  0.20, -0.10,  0.05],
            [0.15, 0.12, 0.10,-0.40,  1.00, -0.15,  0.30, -0.05],
            [-0.30,-0.28,-0.35, 0.20, -0.15,  1.00, -0.20,  0.10],
            [0.40, 0.38, 0.30,-0.10,  0.30, -0.20,  1.00,  0.20],
            [0.35, 0.40, 0.30, 0.05, -0.05,  0.10,  0.20,  1.00],
        ])
        mode = st.radio("Mode", ["Normal Period", "Crisis Period (+40% equity corr)"], horizontal=True, key="corr_mode")
        if mode == "Crisis Period (+40% equity corr)":
            C_show = C.copy()
            eq_idx = [0,1,2]
            for i in eq_idx:
                for j in eq_idx:
                    if i != j: C_show[i,j] = min(C[i,j]*1.4, 0.99)
        else:
            C_show = C.copy()

        fig = go.Figure(go.Heatmap(
            z=C_show, x=assets, y=assets,
            colorscale=[[0,'#8B0000'],[0.5,'#112240'],[1,'#FFD700']],
            zmin=-1, zmax=1,
            text=[[f"{C_show[i,j]:.2f}" for j in range(8)] for i in range(8)],
            texttemplate="%{text}", textfont=dict(size=10, color="white"),
            hoverongaps=False,
        ))
        apply_layout(fig, mp_layout(
            title=f"Indian Market Correlation Matrix — {mode}",
            height=480, xaxis_title="", yaxis_title="",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.info("🔍 **Correlation Surge**: Bank Nifty ↔ Nifty 50 rises from 0.88 → 0.99 in crisis. Gold maintains negative correlation — the only true diversifier.")

    with tab3:
        st.html(f'<div class="section-hdr">Rolling Correlation Simulation</div>')
        col1, col2 = st.columns([1,3])
        with col1:
            rho_normal_r = st.slider("Normal ρ", -0.9, 0.9, 0.35, 0.05, key="roll_rho_n")
            rho_crisis_r  = st.slider("Crisis ρ", 0.0, 0.99, 0.88, 0.01, key="roll_rho_c")
            crisis_start  = st.slider("Crisis starts (day)", 100, 350, 200, key="roll_crisis")
            window        = st.slider("Rolling window", 20, 90, 30, 5, key="roll_window")

        np.random.seed(99)
        n_days = 500
        rho_series = [rho_normal_r]*crisis_start + [rho_crisis_r]*(n_days-crisis_start)
        r1_all, r2_all = [], []
        for rho_d in rho_series:
            cov_d = [[1,rho_d],[rho_d,1]]
            draw = np.random.multivariate_normal([0,0], cov_d)
            r1_all.append(draw[0]*0.012); r2_all.append(draw[1]*0.012)
        r1_a = np.array(r1_all); r2_a = np.array(r2_all)
        roll_corr = [np.corrcoef(r1_a[i:i+window], r2_a[i:i+window])[0,1]
                     for i in range(len(r1_a)-window)]

        fig = make_subplots(2,1,shared_xaxes=True,
                            subplot_titles=["Asset Returns","Rolling Correlation"],
                            vertical_spacing=0.08)
        days = list(range(n_days))
        fig.add_trace(go.Scatter(x=days, y=r1_a*100, name="Asset 1", line=dict(color=GOLD,width=1)),1,1)
        fig.add_trace(go.Scatter(x=days, y=r2_a*100, name="Asset 2", line=dict(color=LB,width=1)),1,1)
        fig.add_trace(go.Scatter(x=list(range(window,n_days)), y=roll_corr,
                                 fill='tozeroy', name="Rolling ρ",
                                 line=dict(color=RED,width=2.5),
                                 fillcolor="rgba(220,53,69,0.3)"),2,1)
        fig.add_vline(x=crisis_start, line_dash="dash", line_color=GOLD,
                      annotation_text="Crisis Start", annotation_font_color=GOLD)
        lo = mp_layout(title="Dynamic Correlation — Regime Change Simulation",
                       height=500, showlegend=True)
        for ax in ['xaxis','xaxis2','yaxis','yaxis2']:
            lo[ax] = dict(gridcolor="rgba(0,51,102,0.33)",linecolor="rgba(255,215,0,0.2)",
                          tickfont=dict(color=MUTED),zerolinecolor="rgba(255,215,0,0.13)")
        apply_layout(fig, lo)
        for ann in fig.layout.annotations:
            ann.font.color = GOLD; ann.font.size = 12
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: COPULA 2 ASSETS
# ══════════════════════════════════════════════════════════════
elif PAGE == "copula2":
    st.html('<div class="page-title">Copula Laboratory — 2 Assets</div>')
    st.html(f'<div class="page-sub">Interactive Gaussian Copula simulation with Sklar\'s Theorem step-by-step</div>')

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Copula Families", "📐 Sklar Pipeline", "📊 Tail Dependence", "🎲 Risk Simulation"
    ])

    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            family = st.selectbox("Copula Family", [
                "Gaussian", "Student-t", "Clayton", "Gumbel"
            ], key="cop2_family")
            rho_c = st.slider("Correlation ρ", -0.95, 0.95, 0.65, 0.05, key="cop2_rho",
                              help="Applies to Gaussian and t-copula")
            nu_c  = st.slider("Degrees of freedom ν", 2, 30, 4, 1, key="cop2_nu",
                              help="t-copula only; lower = fatter tails")
            theta_c = st.slider("Archimedean θ", 0.1, 8.0, 2.5, 0.1, key="cop2_theta",
                                help="Clayton / Gumbel parameter")
            n_sim_c = st.select_slider("Simulations", [500,1000,2000,5000], 2000, key="cop2_nsim")

        if family == "Gaussian":   U = gaussian_copula_sample(rho_c, n_sim_c)
        elif family == "Student-t": U = t_copula_sample(rho_c, nu_c, n_sim_c)
        elif family == "Clayton":   U = clayton_sample(theta_c, n_sim_c)
        else:                       U = gumbel_sample(max(theta_c, 1.01), n_sim_c)

        with col2:
            thresh = 0.05
            mask_ll = (U[:,0]<thresh) & (U[:,1]<thresh)
            mask_uu = (U[:,0]>1-thresh) & (U[:,1]>1-thresh)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=U[~mask_ll & ~mask_uu, 0], y=U[~mask_ll & ~mask_uu, 1],
                mode='markers', name="Joint sample",
                marker=dict(color=LB, size=5, opacity=0.60)
            ))
            fig.add_trace(go.Scatter(
                x=U[mask_ll,0], y=U[mask_ll,1], mode='markers',
                name=f"Joint lower tail: {mask_ll.sum()}",
                marker=dict(color=RED, size=8, symbol='diamond',
                            line=dict(color='white',width=1))
            ))
            fig.add_trace(go.Scatter(
                x=U[mask_uu,0], y=U[mask_uu,1], mode='markers',
                name=f"Joint upper tail: {mask_uu.sum()}",
                marker=dict(color=GRN, size=8, symbol='star',
                            line=dict(color='white',width=1))
            ))
            for v in [thresh, 1-thresh]:
                fig.add_vline(x=v, line_dash="dot", line_color=GOLD, line_width=1)
                fig.add_hline(y=v, line_dash="dot", line_color=GOLD, line_width=1)
            apply_layout(fig, mp_layout(
                title=f"{family} Copula — Uniform Space [0,1]²",
                xaxis_title="U₁", yaxis_title="U₂", height=460,
                legend=dict(bgcolor="rgba(10,22,40,0.9)", bordercolor="rgba(255,215,0,0.5)",
                            borderwidth=1, font=dict(color=TXT, size=10),
                            x=0.55, y=0.98, xanchor="left", yanchor="top")))
            fig.update_xaxes(range=[-0.02, 1.02])
            fig.update_yaxes(range=[-0.02, 1.02])
            st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        exp_ind = thresh**2 * n_sim_c
        m1.metric("Joint Crashes", int(mask_ll.sum()), f"vs {exp_ind:.0f} (indep)")
        m2.metric("Joint Rallies", int(mask_uu.sum()), f"vs {exp_ind:.0f} (indep)")
        m3.metric("Crash Multiplier", f"{mask_ll.sum()/max(exp_ind,1):.1f}x")
        m4.metric("Empirical ρ", f"{np.corrcoef(U[:,0],U[:,1])[0,1]:.3f}")

        # Tail dep coefficients
        if family == "Gaussian":
            lam_l = lam_u = 0.0
        elif family == "Student-t":
            lam_l = lam_u = 2*stats.t.sf(np.sqrt((nu_c+1)*(1-rho_c)/(1+rho_c+1e-9)), df=nu_c+1)
        elif family == "Clayton":
            lam_l = 2**(-1/theta_c); lam_u = 0.0
        else:
            lam_l = 0.0; lam_u = 2 - 2**(1/max(theta_c,1.01))
        st.html(f"""
        <div class="mp-card" style="margin-top:8px;">
          <span style="color:{GOLD};font-weight:600">Tail Dependence: </span>
          <span class="badge">λ_L = {lam_l:.3f}</span>
          <span class="badge">λ_U = {lam_u:.3f}</span>
          &nbsp;·&nbsp;
          <span style="color:{MUTED};font-size:0.82rem">
          λ = 0 → tail independent (Gaussian, Frank) &nbsp;|&nbsp;
          λ > 0 → joint extremes cluster
          </span>
        </div>
        """)

    with tab2:
        st.html('<div class="section-hdr">Sklar\'s Theorem — Step-by-Step Pipeline</div>')
        rho_sk = st.slider("Correlation ρ (Gaussian)", -0.9, 0.9, 0.65, 0.05, key="sk_rho")
        marg1  = st.selectbox("Asset 1 Marginal", ["Normal", "Lognormal", "t(5)"], key="skl_marg1")
        marg2  = st.selectbox("Asset 2 Marginal", ["Lognormal", "Normal", "t(5)"], index=0, key="skl_marg2")

        np.random.seed(42)
        U_sk = gaussian_copula_sample(rho_sk, 1500, seed=42)

        def apply_marginal(u, name, mu=0.001, sig=0.015):
            if name == "Normal":    return norm.ppf(u, mu, sig)
            elif name == "Lognormal": return stats.lognorm.ppf(u, s=sig, scale=np.exp(mu))
            else:                   return stats.t.ppf(u, df=5, loc=mu, scale=sig)

        X1 = apply_marginal(U_sk[:,0], marg1)
        X2 = apply_marginal(U_sk[:,1], marg2)

        fig = make_subplots(1, 3,
            subplot_titles=["Step 1: Z ~ N(0,Σ)", "Step 2: Copula U=Φ(Z)", "Step 3: Custom Marginals"])
        np.random.seed(42)
        cov_sk = [[1,rho_sk],[rho_sk,1]]
        Z_sk = np.random.multivariate_normal([0,0], cov_sk, 1500)

        for i,(xi,yi,col,nm) in enumerate([
            (Z_sk[:,0],Z_sk[:,1],GOLD,"Bivariate Normal"),
            (U_sk[:,0],U_sk[:,1],GRN,"Gaussian Copula"),
            (X1,X2,LB,"Custom Marginals"),
        ]):
            fig.add_trace(go.Scatter(
                x=xi, y=yi, mode='markers',
                marker=dict(color=col,size=3,opacity=0.4), name=nm
            ), 1, i+1)

        lo = mp_layout(title=f"Sklar's Decomposition — Gaussian Copula (ρ={rho_sk})", height=420)
        for ax in ['xaxis','xaxis2','xaxis3','yaxis','yaxis2','yaxis3']:
            lo[ax] = dict(gridcolor="rgba(0,51,102,0.33)",linecolor="rgba(255,215,0,0.2)",
                          tickfont=dict(color=MUTED),zerolinecolor="rgba(255,215,0,0.13)")
        apply_layout(fig, lo)
        for ann in fig.layout.annotations:
            ann.font.color = GOLD; ann.font.size = 11
                # Fix copula panel range [0,1]
        fig.update_xaxes(range=[-0.02, 1.02], col=2)
        fig.update_yaxes(range=[-0.02, 1.02], col=2)
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ Empirical correlation in copula space: **{np.corrcoef(U_sk[:,0],U_sk[:,1])[0,1]:.3f}** | Final output: **{np.corrcoef(X1,X2)[0,1]:.3f}**")

    with tab3:
        st.html('<div class="section-hdr">Tail Dependence Coefficient — Visual</div>')
        rho_range = np.linspace(-0.99, 0.99, 200)
        nu_td = st.slider("t-Copula ν", 2, 20, 4, 1, key="td_nu")

        td_t   = [2*stats.t.sf(np.sqrt((nu_td+1)*(1-r)/(1+r+1e-9)), df=nu_td+1) for r in rho_range]
        theta_r= np.linspace(0.1, 8, 200)
        theta_gum = np.linspace(1.01, 8, 200)   # Gumbel requires theta >= 1
        td_cla = 2**(-1/theta_r)
        td_gum = 2 - 2**(1/theta_gum)

        fig = make_subplots(1,2,subplot_titles=[
            "t-Copula vs Gaussian Tail Dep","Clayton vs Gumbel Tail Dep"])
        fig.add_trace(go.Scatter(x=rho_range,y=td_t,name=f"t-copula ν={nu_td}",
                                  line=dict(color=RED,width=2.5)),1,1)
        fig.add_trace(go.Scatter(x=rho_range,y=[0]*200,name="Gaussian (λ=0)",
                                  line=dict(color=GOLD,width=1.5,dash="dash")),1,1)
        fig.add_trace(go.Scatter(x=theta_r,y=td_cla,name="Clayton λ_L",
                                  line=dict(color=GRN,width=2.5)),1,2)
        fig.add_trace(go.Scatter(x=theta_gum,y=td_gum,name="Gumbel λ_U",
                                  line=dict(color=ORANGE,width=2.5)),1,2)
        lo = mp_layout(title="Tail Dependence Coefficients", height=380)
        for ax in ['xaxis','xaxis2','yaxis','yaxis2']:
            lo[ax] = dict(gridcolor="rgba(0,51,102,0.33)",linecolor="rgba(255,215,0,0.2)",
                          tickfont=dict(color=MUTED),zerolinecolor="rgba(255,215,0,0.13)")
        apply_layout(fig, lo, rows=1, cols=2)
        fig.update_yaxes(range=[-0.05, 1.05])
        fig.update_xaxes(range=[-1.02, 1.02], col=1)  # rho range for t-copula panel
        fig.update_xaxes(range=[0, 8.2], col=2)        # theta range for Archimedean panel
        for ann in fig.layout.annotations:
            ann.font.color=GOLD; ann.font.size = 11
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.html('<div class="section-hdr">Portfolio Risk Simulation via Copula</div>')
        col1, col2 = st.columns([1,3])
        with col1:
            w1_s = st.slider("Weight Asset 1 (%)", 10, 90, 60, 5, key="risk_w1") / 100
            w2_s = 1 - w1_s
            mu1_s  = st.number_input("μ₁ (daily %)", value=0.08, step=0.01, key="risk_mu1") / 100
            sig1_s = st.number_input("σ₁ (daily %)", value=1.50, step=0.10, key="risk_sig1") / 100
            mu2_s  = st.number_input("μ₂ (daily %)", value=0.05, step=0.01, key="risk_mu2") / 100
            sig2_s = st.number_input("σ₂ (daily %)", value=1.00, step=0.10, key="risk_sig2") / 100
            rho_s  = st.slider("Copula ρ", -0.9, 0.9, 0.55, 0.05, key="risk_rho")
            fam_s  = st.selectbox("Copula", ["Gaussian","Student-t","Clayton"], key="risk_fam")
            nu_s   = st.slider("ν (t-copula)", 2, 20, 4, 1, key="risk_nu")
            n_s    = 50000

        np.random.seed(42)
        if fam_s == "Gaussian":   Us = gaussian_copula_sample(rho_s, n_s)
        elif fam_s == "Student-t": Us = t_copula_sample(rho_s, nu_s, n_s)
        else:                      Us = clayton_sample(max(0.1,2*rho_s/(1-rho_s+1e-9)), n_s)

        R1s = norm.ppf(Us[:,0], mu1_s, sig1_s)
        R2s = norm.ppf(Us[:,1], mu2_s, sig2_s)
        Rps = w1_s*R1s + w2_s*R2s

        var95 = np.percentile(Rps, 5)
        var99 = np.percentile(Rps, 1)
        cvar95 = Rps[Rps<=var95].mean()
        cvar99 = Rps[Rps<=var99].mean()
        mu_p  = Rps.mean()
        sig_p = Rps.std()

        with col2:
            from scipy.stats import gaussian_kde
            kde_vals = gaussian_kde(Rps*100)
            x_range = np.linspace(Rps.min()*100, Rps.max()*100, 400)
            y_kde = kde_vals(x_range)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range,y=y_kde,fill='tozeroy',
                                     name="Portfolio Return Distribution",
                                     line=dict(color=GOLD,width=2),
                                     fillcolor="rgba(255,215,0,0.09)"))
            x_tail = x_range[x_range <= var99*100]
            y_tail = y_kde[x_range <= var99*100]
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_tail,[x_tail[-1],x_tail[0]]]),
                y=np.concatenate([y_tail,[0,0]]),
                fill='toself', fillcolor="rgba(220,53,69,0.27)",
                line=dict(color=RED, width=0), name="1% Tail", showlegend=True
            ))
            for val, col, nm in [
                (var95*100, ORANGE, f"VaR 95%: {abs(var95)*100:.2f}%"),
                (var99*100, RED, f"VaR 99%: {abs(var99)*100:.2f}%"),
                (cvar99*100, "#ff00ff", f"CVaR 99%: {abs(cvar99)*100:.2f}%"),
            ]:
                fig.add_vline(x=val,line_dash="dash",line_color=col,
                              annotation_text=nm,annotation_font_color=col,
                              annotation_font_size=10)
            apply_layout(fig, mp_layout(
                title=f"{fam_s} Copula Portfolio — Return Distribution (n={n_s:,})",
                xaxis_title="Daily Return (%)", yaxis_title="Density",
                height=420
            ))
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("VaR 95%", f"{abs(var95)*100:.3f}%")
        c2.metric("VaR 99%", f"{abs(var99)*100:.3f}%")
        c3.metric("CVaR 99%", f"{abs(cvar99)*100:.3f}%")
        c4.metric("Portfolio σ", f"{sig_p*100:.3f}%")

# ══════════════════════════════════════════════════════════════
#  PAGE: COPULA 3 ASSETS
# ══════════════════════════════════════════════════════════════
elif PAGE == "copula3":
    st.html('<div class="page-title">Copula Laboratory — 3 Assets</div>')
    st.html(f'<div class="page-sub">Cholesky decomposition, 3×3 correlation matrix, and pairwise copula visualisation</div>')

    col1, col2 = st.columns([1, 2])
    with col1:
        st.html(f'<p style="color:{GOLD};font-weight:600">Asset Correlations (Z-score space)</p>')
        rho12 = st.slider("ρ₁₂ (Asset 1 ↔ 2)", -0.95, 0.95, 0.65, 0.05, key="c3_rho12")
        rho13 = st.slider("ρ₁₃ (Asset 1 ↔ 3)", -0.95, 0.95, 0.30, 0.05, key="c3_rho13")
        rho23 = st.slider("ρ₂₃ (Asset 2 ↔ 3)", -0.95, 0.95, -0.20, 0.05, key="c3_rho23")
        n3    = st.select_slider("Simulations", [500,1000,2000,5000], 2000, key="n3")
        asset_names = ["Nifty 50", "Gold", "USD/INR"]
        asset_names[0] = st.text_input("Asset 1 name", "Nifty 50", key="c3_a1")
        asset_names[1] = st.text_input("Asset 2 name", "Gold",     key="c3_a2")
        asset_names[2] = st.text_input("Asset 3 name", "USD/INR",  key="c3_a3")

    P3 = np.array([[1, rho12, rho13],
                   [rho12, 1, rho23],
                   [rho13, rho23, 1]])

    # Cholesky
    try:
        L3 = np.linalg.cholesky(P3)
        psd_ok = True
    except np.linalg.LinAlgError:
        psd_ok = False
        L3 = np.eye(3)

    if not psd_ok:
        st.error("⚠️ Correlation matrix is NOT positive definite. Adjust correlations — the Cholesky factorisation failed.")
    else:
        np.random.seed(42)
        W_ind = np.random.randn(n3, 3)
        Z3 = W_ind @ L3.T
        U3 = norm.cdf(Z3)

        with col2:
            tab_a, tab_b, tab_c = st.tabs(["📐 Matrices", "🔗 Pairwise Copulas", "📊 Verification"])

            with tab_a:
                c1, c2 = st.columns(2)
                with c1:
                    fig_p = go.Figure(go.Heatmap(
                        z=P3, x=asset_names, y=asset_names,
                        colorscale=[[0,'#8B0000'],[0.5,'#112240'],[1,'#FFD700']],
                        zmin=-1, zmax=1,
                        text=[[f"{P3[i,j]:.3f}" for j in range(3)] for i in range(3)],
                        texttemplate="%{text}", textfont=dict(size=13, color="white"),
                    ))
                    apply_layout(fig_p, mp_layout(title="Correlation Matrix P", height=280))
                    st.plotly_chart(fig_p, use_container_width=True)
                with c2:
                    fig_l = go.Figure(go.Heatmap(
                        z=L3, x=["c1","c2","c3"], y=[asset_names[0],asset_names[1],asset_names[2]],
                        colorscale=[[0,'#112240'],[0.5,'#004d80'],[1,'#FFD700']],
                        text=[[f"{L3[i,j]:.4f}" for j in range(3)] for i in range(3)],
                        texttemplate="%{text}", textfont=dict(size=11, color="white"),
                    ))
                    apply_layout(fig_l, mp_layout(title="Cholesky Factor L (P=LLᵀ)", height=280))
                    st.plotly_chart(fig_l, use_container_width=True)

                st.html(f"""
                <div class="mp-card mp-card-accent" style="margin-top:8px;">
                  <p style="color:{GOLD};font-weight:600;margin-bottom:6px;">Cholesky Elements</p>
                  <code style="color:{TXT};font-size:0.82rem;">
                  L₁₁={L3[0,0]:.4f} &nbsp; L₂₁={L3[1,0]:.4f} &nbsp; L₂₂={L3[1,1]:.4f}<br>
                  L₃₁={L3[2,0]:.4f} &nbsp; L₃₂={L3[2,1]:.4f} &nbsp; L₃₃={L3[2,2]:.4f}
                  </code>
                  <p style="color:{MUTED};font-size:0.78rem;margin-top:6px;">
                  W = LZ injects exact correlation structure. L₃₃² = 1−ρ₁₃²−L₃₂² > 0 required.
                  </p>
                </div>
                """)

            with tab_b:
                pairs = [(0,1,rho12,GOLD),(0,2,rho13,GRN),(1,2,rho23,LB)]
                fig = make_subplots(1,3,
                    subplot_titles=[f"{asset_names[i]}↔{asset_names[j]}\nρ={r:.2f}"
                                    for i,j,r,_ in pairs])
                for col_i,(i,j,r,col) in enumerate(pairs):
                    mask_ll = (U3[:,i]<0.05)&(U3[:,j]<0.05)
                    fig.add_trace(go.Scatter(
                        x=U3[:,i],y=U3[:,j],mode='markers',
                        marker=dict(color=col,size=5,opacity=0.75),
                        name=f"{asset_names[i]}↔{asset_names[j]}"
                    ),1,col_i+1)
                    fig.add_trace(go.Scatter(
                        x=U3[mask_ll,i],y=U3[mask_ll,j],mode='markers',
                        marker=dict(color=RED,size=9,symbol='x'),
                        name="Joint tail",showlegend=col_i==0
                    ),1,col_i+1)
                lo3 = mp_layout(title="3-Asset Pairwise Copulas in Uniform Space",height=420,showlegend=False)
                for ax in ['xaxis','xaxis2','xaxis3','yaxis','yaxis2','yaxis3']:
                    lo3[ax]=dict(range=[0,1],gridcolor="rgba(0,51,102,0.33)",linecolor="rgba(255,215,0,0.2)",
                                 tickfont=dict(color=MUTED),zerolinecolor="rgba(255,215,0,0.13)")
                apply_layout(fig, lo3, rows=1, cols=3)
                fig.update_xaxes(range=[-0.02, 1.02])
                fig.update_yaxes(range=[-0.02, 1.02])
                for ann in fig.layout.annotations: ann.font.color=GOLD; ann.font.size = 11
                st.plotly_chart(fig, use_container_width=True)

            with tab_c:
                emp = [[np.corrcoef(U3[:,i],U3[:,j])[0,1] for j in range(3)] for i in range(3)]
                diff = [[abs(P3[i,j]-emp[i][j]) for j in range(3)] for i in range(3)]
                fig_v = make_subplots(1,2,subplot_titles=["Empirical Correlation","Abs Error |P−Emp|"])
                fig_v.add_trace(go.Heatmap(
                    z=emp, x=asset_names, y=asset_names,
                    colorscale=[[0,'#8B0000'],[0.5,'#112240'],[1,'#FFD700']],
                    zmin=-1,zmax=1,
                    text=[[f"{emp[i][j]:.3f}" for j in range(3)] for i in range(3)],
                    texttemplate="%{text}",textfont=dict(size=13, color="white"),
                    colorbar=dict(tickfont=dict(color=TXT,size=9),thickness=10)
                ),1,1)
                fig_v.add_trace(go.Heatmap(
                    z=diff,x=asset_names,y=asset_names,
                    colorscale=[[0,GRN],[0.5,ORANGE],[1,RED]],
                    zmin=0,zmax=0.1,
                    text=[[f"{diff[i][j]:.4f}" for j in range(3)] for i in range(3)],
                    texttemplate="%{text}",textfont=dict(size=13, color="white"),
                    colorbar=dict(tickfont=dict(color=TXT,size=9),thickness=10)
                ),1,2)
                lo_v = mp_layout(title="Verification: Target vs Empirical Correlation",height=300)
                for ax in ['xaxis','xaxis2','yaxis','yaxis2']:
                    lo_v[ax]=dict(gridcolor="rgba(0,51,102,0.33)",linecolor="rgba(255,215,0,0.2)",
                                  tickfont=dict(color=MUTED),zerolinecolor="rgba(255,215,0,0.13)")
                apply_layout(fig_v, lo_v, rows=1, cols=2)
                for ann in fig_v.layout.annotations: ann.font.color=GOLD; ann.font.size = 11
                st.plotly_chart(fig_v, use_container_width=True)
                max_err = max(diff[i][j] for i in range(3) for j in range(3) if i!=j)
                if max_err < 0.03:
                    st.success(f"✅ Max error = {max_err:.4f} — excellent. Increase simulations for tighter convergence.")
                else:
                    st.warning(f"⚠️ Max error = {max_err:.4f} — increase simulations for better convergence.")

# ══════════════════════════════════════════════════════════════
#  PAGE: CASE STUDIES
# ══════════════════════════════════════════════════════════════
elif PAGE == "cases":
    st.html('<div class="page-title">Case Studies</div>')
    st.html(f'<div class="page-sub">Five real-world episodes where correlation and dependency structures determined financial outcomes</div>')

    cases = {
        "📉 2008 Global Financial Crisis": {
            "subtitle": "The Gaussian Copula and CDO Collapse",
            "color": RED,
            "period": "2006–2009",
            "context": "Banks used the Gaussian copula (Li, 2000) to price CDO tranches, assuming asset correlation ρ ≈ 0.10–0.20. Subprime mortgage defaults were treated as nearly independent. The copula had zero tail dependence — joint defaults in the extreme were considered essentially impossible.",
            "what_happened": "When US house prices fell nationwide in 2007–08, subprime mortgages across all geographies defaulted simultaneously. The true correlation in the lower tail was near 1.0 — not 0.10. CDO tranches rated AAA lost 60–80% of value. Global losses exceeded USD 2 trillion.",
            "copula_lesson": "Gaussian copula: λ_L = 0. t-copula (ν=3): λ_L ≈ 0.39 at ρ=0.5. The difference is a joint default probability 5–8× higher in the tail — which is exactly what materialised.",
            "rho_normal": 0.15, "rho_crisis": 0.92,
            "pd": 0.04,
        },
        "🏦 LTCM Collapse 1998": {
            "subtitle": "When Correlation Mean-Reversion Failed",
            "color": ORANGE,
            "period": "1997–1998",
            "context": "Long-Term Capital Management built convergence trades based on historically stable correlations between on-the-run and off-the-run US Treasuries, Russian bonds, and European swap spreads. Their copula was implicitly Gaussian with low tail dependence.",
            "what_happened": "Russia's August 1998 default triggered a global flight to quality. All LTCM positions moved against them simultaneously. Correlations between liquid and illiquid assets jumped from ~0.3 to >0.9. LTCM lost USD 4.6 billion in four months — requiring a Fed-orchestrated USD 3.6 billion rescue.",
            "copula_lesson": "A Clayton copula would have captured the asymmetric lower-tail co-movement (everything falls together in a crisis). The Gaussian model treated extreme co-movement as near-impossible — exactly the wrong assumption for a leveraged convergence strategy.",
            "rho_normal": 0.30, "rho_crisis": 0.91,
            "pd": 0.02,
        },
        "🌏 COVID-19 March 2020": {
            "subtitle": "Simultaneous Global Asset Crash",
            "color": PURPLE,
            "period": "Feb–Apr 2020",
            "context": "Pre-COVID portfolio models used 3–5 year historical correlations. Nifty 50 ↔ Bank Nifty ≈ 0.75; Nifty ↔ Gold ≈ −0.25; equity indices across geographies ≈ 0.4–0.6. Diversified portfolios appeared well-protected.",
            "what_happened": "Between Feb 20 and Mar 23, 2020, Nifty fell 38% in 33 days. All equity indices crashed simultaneously — S&P 500, FTSE, Nifty all hit multi-year lows within days of each other. Global equity correlations spiked to 0.95+. Gold initially fell with equities before recovering.",
            "copula_lesson": "A dynamic copula model (DCC-GARCH + t-copula) calibrated on 2008 data would have raised crisis-period correlations automatically. Static models with long lookback windows were blind to the regime shift.",
            "rho_normal": 0.45, "rho_crisis": 0.95,
            "pd": 0.05,
        },
        "🏘️ IL&FS Crisis 2018": {
            "subtitle": "Indian NBFC Contagion and Credit Correlation",
            "color": GOLD,
            "period": "Sep 2018 – Mar 2019",
            "context": "IL&FS, a large Indian infrastructure finance company, defaulted on short-term debt in September 2018. Credit models for NBFCs and HFCs assumed low cross-sector default correlation — each company was treated as largely idiosyncratic.",
            "what_happened": "The IL&FS default triggered a funding freeze across Indian shadow banking. DHFL, Indiabulls, Yes Bank all saw bond yields spike simultaneously. Mutual fund redemptions accelerated. The RBI had to intervene. CDS spreads on all NBFCs moved in near-perfect correlation.",
            "copula_lesson": "A Clayton copula calibrated on stressed credit data would have captured the lower-tail clustering. The 2018 NBFC crisis showed that Indian credit correlations are highly asymmetric: defaults cluster (lower tail dependence > 0) while recoveries are independent (upper tail dependence ≈ 0).",
            "rho_normal": 0.25, "rho_crisis": 0.87,
            "pd": 0.06,
        },
        "💱 Asian Financial Crisis 1997": {
            "subtitle": "Currency Contagion and Copula Tail Risk",
            "color": GRN,
            "period": "Jul 1997 – Jan 1998",
            "context": "Asian currencies (Thai Baht, Indonesian Rupiah, Korean Won, Malaysian Ringgit) were treated as having moderate co-movement by global portfolio managers. Correlation models showed ρ ≈ 0.3–0.5 in normal conditions.",
            "what_happened": "Thailand's baht devaluation on July 2, 1997 triggered sequential currency crises across Asia. All currencies collapsed within months — the Rupiah lost 80% of its value. Equity markets fell 40–70%. Cross-currency correlations in the lower tail were near 1.0.",
            "copula_lesson": "Gumbel and Clayton copulas calibrated on emerging market stress would have shown λ_L > 0.6 for these currency pairs — consistent with currency crisis contagion. The Gaussian copula assumption led to systematic underestimation of joint currency risk.",
            "rho_normal": 0.35, "rho_crisis": 0.90,
            "pd": 0.07,
        },
    }

    case_names = list(cases.keys())
    sel = st.selectbox("Select Case Study", case_names, key="case_sel")
    cs = cases[sel]

    st.html(f"""
    <div class="mp-card" style="border-left:4px solid {cs['color']};margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <span style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;color:{cs['color']}">{sel}</span>
        <span class="badge" style="background:{cs['color']}22;color:{cs['color']};border-color:{cs['color']}55;">{cs['period']}</span>
      </div>
      <p style="color:{MUTED};font-size:0.92rem;font-style:italic;margin-bottom:12px;">{cs['subtitle']}</p>
    </div>
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.html(f'<p style="color:{GOLD};font-weight:600;font-size:1rem;">📋 Context</p>')
        st.html(f'<div class="mp-card"><p style="color:{TXT};font-size:0.88rem;line-height:1.8;">{cs["context"]}</p></div>')
        st.html(f'<p style="color:{GOLD};font-weight:600;font-size:1rem;margin-top:12px;">⚡ What Happened</p>')
        st.html(f'<div class="mp-card"><p style="color:{TXT};font-size:0.88rem;line-height:1.8;">{cs["what_happened"]}</p></div>')
        st.html(f'<p style="color:{GOLD};font-weight:600;font-size:1rem;margin-top:12px;">🔗 Copula Lesson</p>')
        st.html(f'<div class="mp-card mp-card-accent"><p style="color:{TXT};font-size:0.88rem;line-height:1.8;">{cs["copula_lesson"]}</p></div>')

    with col2:
        # Visualise: normal vs crisis copula scatters
        U_n = gaussian_copula_sample(cs['rho_normal'], 1000, seed=11)
        U_c = gaussian_copula_sample(cs['rho_crisis'], 1000, seed=11)

        fig = make_subplots(1,2,
            subplot_titles=[
                f"Normal Period (ρ={cs['rho_normal']})",
                f"Crisis Period (ρ={cs['rho_crisis']})"
            ])
        for i,(U,col) in enumerate([(U_n, GOLD),(U_c, cs['color'])]):
            mask_ll = (U[:,0]<0.05)&(U[:,1]<0.05)
            fig.add_trace(go.Scatter(
                x=U[:,0],y=U[:,1],mode='markers',
                marker=dict(color=col,size=5,opacity=0.75),
                name=["Normal","Crisis"][i]
            ),1,i+1)
            fig.add_trace(go.Scatter(
                x=U[mask_ll,0],y=U[mask_ll,1],mode='markers',
                marker=dict(color=RED,size=9,symbol='x',line=dict(color='white',width=1)),
                name=f"Joint crashes: {mask_ll.sum()}",showlegend=i==1
            ),1,i+1)
        lo_cs = mp_layout(title="Copula Structure: Normal vs Crisis",height=340,showlegend=True)
        for ax in ['xaxis','xaxis2','yaxis','yaxis2']:
            lo_cs[ax]=dict(range=[-0.02,1.02],gridcolor="rgba(0,51,102,0.33)",linecolor="rgba(255,215,0,0.2)",
                           tickfont=dict(color=MUTED),zerolinecolor="rgba(255,215,0,0.13)")
        apply_layout(fig, lo_cs, rows=1, cols=2)
        for ann in fig.layout.annotations: ann.font.color=GOLD; ann.font.size = 11
        st.plotly_chart(fig, use_container_width=True)

        # WCDR comparison
        rho_range_cs = np.linspace(0.01, 0.99, 200)
        wcdr_n = wcdr(cs['pd'], cs['rho_normal'])
        wcdr_c = wcdr(cs['pd'], cs['rho_crisis'])

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rho_range_cs, y=[wcdr(cs['pd'],r)*100 for r in rho_range_cs],
            name="WCDR at 99.9%", line=dict(color=GOLD,width=2.5)
        ))
        fig2.add_vline(x=cs['rho_normal'],line_dash="dash",line_color=GRN,
                       annotation_text=f"Normal ρ={cs['rho_normal']}<br>WCDR={wcdr_n*100:.1f}%",
                       annotation_font_color=GRN, annotation_font_size=10)
        fig2.add_vline(x=cs['rho_crisis'],line_dash="dash",line_color=RED,
                       annotation_text=f"Crisis ρ={cs['rho_crisis']}<br>WCDR={wcdr_c*100:.1f}%",
                       annotation_font_color=RED, annotation_font_size=10)
        fig2.add_hline(y=cs['pd']*100,line_dash="dot",line_color=MUTED,
                       annotation_text=f"PD={cs['pd']*100:.1f}%",annotation_font_color=MUTED)
        apply_layout(fig2, mp_layout(
            title=f"WCDR vs Correlation (PD={cs['pd']*100:.1f}%)",
            xaxis_title="Asset Correlation ρ",yaxis_title="WCDR (%)",
            height=320
        ))
        st.plotly_chart(fig2, use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("WCDR (Normal ρ)", f"{wcdr_n*100:.1f}%", f"{cs['rho_normal']} correlation")
        m2.metric("WCDR (Crisis ρ)", f"{wcdr_c*100:.1f}%", f"+{(wcdr_c-wcdr_n)*100:.1f}pp shock",
                  delta_color="inverse")

# ══════════════════════════════════════════════════════════════
#  PAGE: APPLICATIONS
# ══════════════════════════════════════════════════════════════
elif PAGE == "applications":
    st.html('<div class="page-title">Applications</div>')
    st.html(f'<div class="page-sub">Copula models in portfolio management, credit risk, and regulatory capital</div>')

    app_tab1, app_tab2, app_tab3, app_tab4 = st.tabs([
        "📊 Portfolio VaR", "🏦 Credit Portfolio", "⚡ Stress Testing", "📐 Copula Selection"
    ])

    with app_tab1:
        st.html('<div class="section-hdr">Portfolio VaR: Copula vs Parametric</div>')
        col1, col2 = st.columns([1,2])
        with col1:
            assets_app = ["Nifty 50","Gold","USD/INR","10Y Bond"]
            weights_app = [st.slider(f"Weight: {a} (%)", 0, 100, w, 5, key=f"app_w_{a.replace(' ','_')}")
                           for a,w in zip(assets_app,[40,20,20,20])]
            tot = sum(weights_app)
            if tot != 100:
                st.warning(f"Weights sum to {tot}% — normalising")
            w_arr = np.array(weights_app)/tot
            sigs_app = np.array([0.012, 0.008, 0.004, 0.003])
            mus_app  = np.array([0.0008, 0.0003, 0.0001, 0.0001])
            rho_app  = st.slider("Uniform copula ρ", 0.0, 0.95, 0.40, 0.05, key="app_rho")
            n_app    = 50000

        np.random.seed(42)
        P_app = rho_app*np.ones((4,4)) + (1-rho_app)*np.eye(4)
        L_app = np.linalg.cholesky(P_app)
        Z_app = np.random.randn(n_app,4) @ L_app.T
        U_app = norm.cdf(Z_app)
        R_app = np.column_stack([norm.ppf(U_app[:,i], mus_app[i], sigs_app[i]) for i in range(4)])
        Rp_app = R_app @ w_arr

        mu_p_app  = Rp_app.mean()*100
        sig_p_app = Rp_app.std()*100
        var95_a = np.percentile(Rp_app,5)*100
        var99_a = np.percentile(Rp_app,1)*100
        cvar99_a = Rp_app[Rp_app<=np.percentile(Rp_app,1)].mean()*100

        # Parametric
        mu_par = (mus_app@w_arr)*100
        sig_par = np.sqrt(w_arr@(np.outer(sigs_app,sigs_app)*P_app)@w_arr)*100
        var99_par = mu_par - 2.326*sig_par
        var99_par_pct = abs(var99_par)

        with col2:
            fig = go.Figure()
            from scipy.stats import gaussian_kde
            kde_app = gaussian_kde(Rp_app*100)
            x_app = np.linspace(Rp_app.min()*100, Rp_app.max()*100, 400)
            fig.add_trace(go.Scatter(x=x_app,y=kde_app(x_app),
                                     fill='tozeroy',name="Copula Simulation",
                                     line=dict(color=GOLD,width=2),fillcolor="rgba(255,215,0,0.09)"))
            fig.add_trace(go.Scatter(x=x_app,
                                     y=norm.pdf(x_app,mu_par,sig_par),
                                     name="Parametric Normal",
                                     line=dict(color=LB,width=2,dash="dash")))
            fig.add_vline(x=var99_a,line_dash="dash",line_color=RED,
                          annotation_text=f"Copula VaR99: {abs(var99_a):.2f}%",
                          annotation_font_color=RED,annotation_font_size=10)
            fig.add_vline(x=var99_par,line_dash="dot",line_color=LB,
                          annotation_text=f"Param VaR99: {var99_par_pct:.2f}%",
                          annotation_font_color=LB,annotation_font_size=10)
            apply_layout(fig, mp_layout(
                title="4-Asset Portfolio — Copula vs Parametric Distribution",
                xaxis_title="Daily Return (%)",yaxis_title="Density",height=380))
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Copula VaR 99%", f"{abs(var99_a):.3f}%")
        c2.metric("Param VaR 99%", f"{var99_par_pct:.3f}%")
        c3.metric("Copula CVaR 99%", f"{abs(cvar99_a):.3f}%")
        c4.metric("Difference", f"{abs(abs(var99_a)-var99_par_pct):.3f}%",
                  "Copula captures tail" if abs(var99_a)>var99_par_pct else "Similar")

    with app_tab2:
        st.html('<div class="section-hdr">Credit Portfolio — Joint Default & WCDR</div>')
        col1, col2 = st.columns([1,2])
        with col1:
            pd_cr  = st.slider("PD (%)", 0.5, 20.0, 3.0, 0.5, key="cred_pd") / 100
            rho_cr = st.slider("Copula ρ", 0.01, 0.99, 0.15, 0.01, key="cred_rho")
            lgd_cr = st.slider("LGD (%)", 20, 80, 45, 5, key="cred_lgd") / 100
            ead_cr = st.number_input("EAD (INR Crore)", value=1000, step=100, key="cred_ead")
            alpha_cr = st.select_slider("Confidence α", [0.95,0.99,0.999,0.9999], 0.999, key="cred_alpha")

        wcdr_val = wcdr(pd_cr, rho_cr, alpha_cr)
        el_cr = pd_cr * lgd_cr * ead_cr
        ul_cr = (wcdr_val - pd_cr) * lgd_cr * ead_cr

        # Basel R
        R_bas = 0.12*(1-np.exp(-50*pd_cr))/(1-np.exp(-50)) + \
                0.24*(1-(1-np.exp(-50*pd_cr))/(1-np.exp(-50)))
        wcdr_bas = wcdr(pd_cr, R_bas, 0.999)
        ul_bas = (wcdr_bas-pd_cr)*lgd_cr*ead_cr

        with col2:
            rho_range_cr = np.linspace(0.01, 0.99, 200)
            wcdr_curve = [wcdr(pd_cr, r, alpha_cr)*100 for r in rho_range_cr]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rho_range_cr,y=wcdr_curve,
                                     name="WCDR",line=dict(color=GOLD,width=2.5)))
            fig.add_hline(y=pd_cr*100,line_dash="dot",line_color=MUTED,
                          annotation_text=f"PD={pd_cr*100:.1f}%",annotation_font_color=MUTED)
            fig.add_vline(x=rho_cr,line_dash="dash",line_color=RED,
                          annotation_text=f"Selected ρ\nWCDR={wcdr_val*100:.1f}%",
                          annotation_font_color=RED,annotation_font_size=10)
            fig.add_vline(x=R_bas,line_dash="dash",line_color=GRN,
                          annotation_text=f"Basel R={R_bas:.3f}\nWCDR={wcdr_bas*100:.1f}%",
                          annotation_font_color=GRN,annotation_font_size=10)
            apply_layout(fig, mp_layout(
                title=f"WCDR vs ρ (PD={pd_cr*100:.1f}%, α={alpha_cr*100:.1f}%)",
                xaxis_title="Asset Correlation ρ",yaxis_title="WCDR (%)",height=350))
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("WCDR", f"{wcdr_val*100:.2f}%")
        c2.metric("Expected Loss (EL)", f"INR {el_cr:.1f} Cr")
        c3.metric("Unexpected Loss (UL)", f"INR {ul_cr:.1f} Cr", "Capital Required")
        c4.metric("Basel UL", f"INR {ul_bas:.1f} Cr", f"R={R_bas:.3f}")

    with app_tab3:
        st.html('<div class="section-hdr">Correlation Stress Testing</div>')
        st.html(f"""
        <div class="mp-card mp-card-accent">
          <p style="color:{TXT};font-size:0.88rem;line-height:1.8;">
          <b style="color:{GOLD}">Basel III / FRTB requirement:</b> Banks must compute risk measures under
          both normal and stressed correlation scenarios. The stressed scenario typically applies
          multipliers of 0.75 and 1.25 to correlation estimates, or uses historical crisis-period data.
          </p>
        </div>
        """)
        base_rho = st.slider("Base Correlation ρ", 0.0, 0.95, 0.40, 0.05, key="stress_rho")
        pd_st = st.slider("PD (%)", 0.5, 20.0, 3.0, 0.5, key="stress_pd") / 100
        lgd_st = 0.45; ead_st = 1000

        scenarios = {
            "Normal (base ρ)": base_rho,
            "Mild stress (ρ×1.25)": min(base_rho*1.25, 0.99),
            "Severe stress (ρ×1.50)": min(base_rho*1.50, 0.99),
            "Historical crisis": min(base_rho + 0.40, 0.99),
            "Extreme (ρ=0.95)": 0.95,
        }
        rows = []
        for name, rho_s in scenarios.items():
            w = wcdr(pd_st, rho_s)
            ul = (w - pd_st)*lgd_st*ead_st
            rows.append({"Scenario": name, "ρ": rho_s,
                         "WCDR (%)": w*100, "UL (INR Cr)": ul,
                         "Capital Increase": (ul-(w-pd_st)*lgd_st*ead_st) if name=="Normal (base ρ)" else None})
        df_stress = pd.DataFrame(rows)
        base_ul = df_stress.iloc[0]["UL (INR Cr)"]
        df_stress["vs Base (%)"] = (df_stress["UL (INR Cr)"] / base_ul - 1)*100

        fig = go.Figure()
        colors_st = [GRN, GOLD, ORANGE, RED, "#8B0000"]
        fig.add_trace(go.Bar(
            x=df_stress["Scenario"],
            y=df_stress["UL (INR Cr)"],
            marker_color=colors_st,
            text=[f"INR {v:.0f} Cr\n({p:+.1f}%)" for v,p in
                  zip(df_stress["UL (INR Cr)"],df_stress["vs Base (%)"])],
            textposition="outside",
            textfont=dict(color=TXT,size=9)
        ))
        apply_layout(fig, mp_layout(
            title="Capital Requirement Across Stress Scenarios",
            xaxis_title="Scenario",yaxis_title="Unexpected Loss / Capital (INR Crore)",
            height=400
        ))
        fig.update_traces(textfont=dict(color=TXT, size=9))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_stress.round(3), use_container_width=True,
                     column_config={"UL (INR Cr)": st.column_config.NumberColumn(format="%.1f"),
                                    "vs Base (%)": st.column_config.NumberColumn(format="%+.1f")})

    with app_tab4:
        st.html('<div class="section-hdr">Copula Family Selection Guide</div>')
        st.html(f"""
        <div class="mp-card">
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem;color:{TXT};">
          <thead>
            <tr style="border-bottom:1px solid {GOLD}44;">
              <th style="color:{GOLD};padding:8px;text-align:left">Copula</th>
              <th style="color:{GOLD};padding:8px;text-align:center">λ_L</th>
              <th style="color:{GOLD};padding:8px;text-align:center">λ_U</th>
              <th style="color:{GOLD};padding:8px;text-align:left">Best for</th>
              <th style="color:{GOLD};padding:8px;text-align:left">Avoid when</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid {BLUE}44;">
              <td style="padding:8px"><b style="color:{LB}">Gaussian</b></td>
              <td style="padding:8px;text-align:center">0</td><td style="padding:8px;text-align:center">0</td>
              <td style="padding:8px">Baseline; symmetric dependence; large portfolios</td>
              <td style="padding:8px">Credit portfolios; crisis calibration</td>
            </tr>
            <tr style="border-bottom:1px solid {BLUE}44;">
              <td style="padding:8px"><b style="color:{RED}">Student-t</b></td>
              <td style="padding:8px;text-align:center">&gt;0</td><td style="padding:8px;text-align:center">&gt;0</td>
              <td style="padding:8px">Equity portfolios; symmetric tail events</td>
              <td style="padding:8px">Strongly asymmetric tail behaviour</td>
            </tr>
            <tr style="border-bottom:1px solid {BLUE}44;">
              <td style="padding:8px"><b style="color:{GRN}">Clayton</b></td>
              <td style="padding:8px;text-align:center">&gt;0</td><td style="padding:8px;text-align:center">0</td>
              <td style="padding:8px">Credit default; assets that crash together</td>
              <td style="padding:8px">Positive upper-tail co-movement needed</td>
            </tr>
            <tr style="border-bottom:1px solid {BLUE}44;">
              <td style="padding:8px"><b style="color:{ORANGE}">Gumbel</b></td>
              <td style="padding:8px;text-align:center">0</td><td style="padding:8px;text-align:center">&gt;0</td>
              <td style="padding:8px">Commodity booms; joint rallies</td>
              <td style="padding:8px">Crash-heavy portfolios</td>
            </tr>
            <tr>
              <td style="padding:8px"><b style="color:{MUTED}">Frank</b></td>
              <td style="padding:8px;text-align:center">0</td><td style="padding:8px;text-align:center">0</td>
              <td style="padding:8px">Mild symmetric dependence; negative ρ</td>
              <td style="padding:8px">Any tail-dependent application</td>
            </tr>
          </tbody>
        </table>
        </div>
        """)

# ══════════════════════════════════════════════════════════════
#  PAGE: WCDR & BASEL II
# ══════════════════════════════════════════════════════════════
elif PAGE == "wcdr":
    st.html('<div class="page-title">WCDR & Basel II Capital</div>')
    st.html(f'<div class="page-sub">Worst Case Default Rate — the Gaussian Copula embedded in banking regulation</div>')

    st.html(f"""
    <div class="mp-card mp-card-accent">
      <p style="color:{GOLD};font-weight:700;font-size:1rem;margin-bottom:6px;">The Basel II Formula</p>
      <p style="color:{TXT};font-size:0.88rem;line-height:1.9;">
      Basel II Internal Ratings-Based capital is built on the <b>Vasicek one-factor Gaussian copula</b>.
      Every corporate, retail, and mortgage loan on a bank's book has its capital requirement determined by:
      </p>
      <div style="text-align:center;padding:12px;background:#07101f;border-radius:8px;margin:8px 0;">
        <code style="color:{GOLD};font-size:1rem;">
          WCDR(α) = Φ[ (Φ⁻¹(PD) + √ρ · Φ⁻¹(α)) / √(1−ρ) ]
        </code>
      </div>
      <p style="color:{MUTED};font-size:0.82rem;">Basel uses α = 99.9% and specifies ρ via the regulatory formula R ∈ [0.12, 0.24]</p>
    </div>
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        pd_w = st.slider("PD (%)", 0.10, 30.0, 3.0, 0.10, key="wcdr_pd") / 100
        use_bas_r = st.checkbox("Use Basel II R formula", value=True, key="wcdr_baselR")
        if use_bas_r:
            rho_w = 0.12*(1-np.exp(-50*pd_w))/(1-np.exp(-50)) + \
                    0.24*(1-(1-np.exp(-50*pd_w))/(1-np.exp(-50)))
            st.info(f"Basel II R = **{rho_w:.4f}**")
        else:
            rho_w = st.slider("Correlation ρ", 0.01, 0.99, 0.15, 0.01, key="wcdr_rho")
        lgd_w = st.slider("LGD (%)", 10, 90, 45, 5, key="wcdr_lgd") / 100
        maturity_w = st.slider("Maturity (years)", 1.0, 5.0, 2.5, 0.5, key="wcdr_mat")
        ead_w = st.number_input("EAD (INR Crore)", value=500, step=50, key="wcdr_ead")
        alpha_w = 0.999

    wcdr_w = wcdr(pd_w, rho_w, alpha_w)
    el_w = pd_w * lgd_w * ead_w
    b_w = (0.11852 - 0.05478*np.log(max(pd_w,1e-6)))**2
    ma_w = (1 + (maturity_w-2.5)*b_w) / (1 - 1.5*b_w)
    K_w = lgd_w*(wcdr_w - pd_w)*ma_w
    cap_w = K_w * ead_w

    with col2:
        # WCDR sensitivity surface
        pd_grid = np.linspace(0.005, 0.25, 50)
        rho_grid_w = np.linspace(0.05, 0.50, 50)
        Z_surf = np.array([[wcdr(p,r,alpha_w)*100 for r in rho_grid_w] for p in pd_grid])

        fig_surf = go.Figure(go.Surface(
            x=rho_grid_w*100, y=pd_grid*100, z=Z_surf,
            colorscale=[[0,BLUE],[0.5,GOLD],[1,RED]],
            opacity=0.85,
            contours_z=dict(show=True,usecolormap=True,highlightcolor=GOLD,project_z=True)
        ))
        fig_surf.add_trace(go.Scatter3d(
            x=[rho_w*100], y=[pd_w*100], z=[wcdr_w*100],
            mode='markers',
            marker=dict(size=10,color=RED,symbol='diamond'),
            name="Current"
        ))
        fig_surf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(title="Correlation ρ (%)",gridcolor="rgba(0,51,102,0.6)",
                           tickfont=dict(color=TXT),title_font=dict(color=GOLD),range=[5,50]),
                yaxis=dict(title="PD (%)",gridcolor="rgba(0,51,102,0.6)",
                           tickfont=dict(color=TXT),title_font=dict(color=GOLD),range=[0.5,25]),
                zaxis=dict(title="WCDR (%)",gridcolor="rgba(0,51,102,0.6)",
                           tickfont=dict(color=TXT),title_font=dict(color=GOLD),range=[0,100]),
                bgcolor="rgba(7,16,31,0.9)",
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
            ),
            title=dict(text="WCDR Surface (α=99.9%)",font=dict(color=GOLD,family="Playfair Display",size=14)),
            font=dict(color=TXT), height=420, margin=dict(l=0,r=0,t=50,b=0)
        )
        st.plotly_chart(fig_surf, use_container_width=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("WCDR", f"{wcdr_w*100:.2f}%")
    c2.metric("Expected Loss", f"INR {el_w:.1f} Cr")
    c3.metric("Capital K", f"{K_w*100:.2f}%")
    c4.metric("Capital (INR Cr)", f"{cap_w:.1f}")
    c5.metric("UL Multiplier", f"{(wcdr_w-pd_w)/pd_w:.1f}×", "WCDR/PD ratio")

    st.markdown("---")
    st.html('<div class="section-hdr">WCDR Sensitivity Table</div>')
    pds_tbl = [0.005, 0.01, 0.02, 0.035, 0.05, 0.10, 0.20]
    rhos_tbl = [0.05, 0.12, 0.15, 0.24, 0.40]
    tbl_data = {}
    for r in rhos_tbl:
        tbl_data[f"ρ={r}"] = [f"{wcdr(p,r)*100:.1f}%" for p in pds_tbl]
    df_tbl = pd.DataFrame(tbl_data, index=[f"{p*100:.1f}%" for p in pds_tbl])
    df_tbl.index.name = "PD"
    st.dataframe(df_tbl, use_container_width=True)

    with st.expander("📖 Excel Formula"):
        st.code("""
# WCDR in Excel (PD in A1, rho in B1, alpha in C1):
=NORM.S.DIST((NORM.S.INV(A1) + SQRT(B1)*NORM.S.INV(C1)) / SQRT(1-B1), TRUE)

# Basel II R formula (PD in A1):
=0.12*(1-EXP(-50*A1))/(1-EXP(-50)) + 0.24*(1-(1-EXP(-50*A1))/(1-EXP(-50)))

# Capital charge K (PD in A1, rho in B1, LGD in C1, M in D1):
=LET(wcdr, NORM.S.DIST((NORM.S.INV(A1)+SQRT(B1)*3.090)/SQRT(1-B1),TRUE),
     b, (0.11852-0.05478*LN(A1))^2,
     ma, (1+(D1-2.5)*b)/(1-1.5*b),
     C1*(wcdr-A1)*ma)
        """, language="")

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.html(f"""
<div style="text-align:center;padding:16px 0 8px;color:{MUTED};font-size:0.75rem;line-height:2;">
  <b style="color:{GOLD}">⛰️ The Mountain Path — World of Finance</b><br>
  <a href="https://themountainpathacademy.com" style="color:{GOLD}">themountainpathacademy.com</a>
  &nbsp;·&nbsp;
  <a href="https://linkedin.com/in/trichyravis" style="color:{LB}">LinkedIn</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/trichyravis" style="color:{LB}">GitHub</a><br>
  Professor of Practice &amp; Visiting Faculty @ Business Schools India<br>
  <span style="color:{MUTED};font-size:0.68rem;">
    Prof. V. Ravichandran · 28+ Years Corporate Finance & Banking · 10+ Years Academia
  </span>
</div>
""")
