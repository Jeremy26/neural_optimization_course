"""Streamlit lead-magnet: Deployment Health Score for PyTorch models.

Run with:

    streamlit run deployment_visualizer/app.py
"""

from __future__ import annotations

import hashlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyzer import analyze
from compatibility import compute_compatibility
from problems import detect_problems
from scenarios import build_scenarios


COURSE_URL = "https://www.thinkautonomous.ai/"


def _compat_card(family) -> None:
    color = _score_color(family.score)
    st.markdown(
        f"""
        <div class="metric-card" style="height:100%;">
          <div class="metric-label">{family.name}</div>
          <div style="display:flex;align-items:baseline;gap:6px;
                       margin:6px 0 8px;">
            <div style="font-size:2rem;font-weight:800;color:{color};
                         line-height:1;">{family.score:.0f}</div>
            <div style="font-size:0.8rem;color:#64748b;">/ 100</div>
          </div>
          <div class="metric-sub" style="font-size:0.88rem;">
            {family.verdict}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Model Deployment Health Score",
    page_icon=None,
    layout="wide",
)


# ---------------------------------------------------------------------------
# Theme / shared CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      /* ============ PAGE CHROME ============
         Light theme by default.  Tab 4 paints itself dark below. */
      .stApp { background: #f8fafc; }
      /* Push content below Streamlit's floating toolbar (Deploy button etc.) */
      .block-container {
        padding-top: 4rem; max-width: 1320px;
      }
      [data-testid="stHeader"] {
        background: rgba(248,250,252,0.85);
        backdrop-filter: blur(6px);
      }

      /* ============ TYPOGRAPHY ============ */
      h1, h2, h3, h4, h5, h6 { color: #0f172a !important; letter-spacing: -0.005em; }
      h1 { font-size: 2.0rem; font-weight: 800; }
      h2 { font-size: 1.5rem; font-weight: 800; }
      h3 {
        font-size: 1.25rem; font-weight: 700;
        position: relative; padding-left: 14px; margin-top: 1.6rem;
      }
      h3::before {
        content: ""; position: absolute; left: 0; top: 0.2em; bottom: 0.2em;
        width: 4px; border-radius: 2px; background: #f59e0b;
      }
      h4 { font-size: 1.05rem; font-weight: 700; margin-top: 1.2rem; }
      .stMarkdown p, .stMarkdown li { color: #334155; }
      [data-testid="stCaptionContainer"] { color: #64748b; }

      /* ============ TABS ============ */
      .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 1px solid #e2e8f0;
      }
      .stTabs [data-baseweb="tab"] {
        height: 48px; padding: 0 22px;
        background: white;
        border: 1px solid #e2e8f0; border-bottom: none;
        border-radius: 8px 8px 0 0;
        color: #64748b !important;
        font-weight: 700; font-size: 0.95rem;
      }
      .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a !important; background: #f1f5f9;
      }
      .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        background: #fff7ed !important;
        border-color: #fdba74 !important;
        box-shadow: inset 0 2px 0 #f59e0b;
      }

      /* ============ CARDS (light) ============ */
      .metric-card {
        border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 18px 20px 16px; background: white;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
      }
      .metric-label {
        font-size: 0.72rem; color: #64748b;
        text-transform: uppercase; letter-spacing: .14em; font-weight: 700;
      }
      .metric-value {
        font-size: 2.0rem; font-weight: 800; line-height: 1.1; color: #0f172a;
      }
      .metric-sub { font-size: 0.92rem; color: #475569; margin-top: 4px; }

      /* ============ BADGES ============ */
      .arch-badge {
        display: inline-block; padding: 5px 12px; border-radius: 6px;
        background: #e0f2fe; border: 1px solid #7dd3fc;
        color: #075985; font-weight: 700;
        font-size: 0.78rem; letter-spacing: .04em;
        margin-right: 6px; margin-bottom: 4px;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
      }

      /* ============ STATUS PILL ============ */
      .status-pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 14px; border-radius: 999px;
        background: #ecfdf5; border: 1px solid #a7f3d0;
        color: #047857; font-size: 0.78rem; font-weight: 700;
        letter-spacing: .12em;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
      }
      .status-pill::before {
        content: ""; width: 8px; height: 8px; border-radius: 50%;
        background: #10b981; box-shadow: 0 0 8px #10b981;
        animation: pulse 1.6s ease-in-out infinite;
      }
      @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .5; } }

      /* ============ BUTTONS ============ */
      .stButton > button[kind="primary"] {
        background: #f59e0b !important; border: 1px solid #b45309;
        color: #0c0a09 !important; font-weight: 800;
        letter-spacing: .04em; text-transform: uppercase;
      }
      .stButton > button[kind="primary"]:hover { background: #fbbf24 !important; }
      .stDownloadButton > button {
        background: #ecfeff; border: 1px solid #67e8f9;
        color: #0e7490 !important; font-weight: 800;
        letter-spacing: .04em; text-transform: uppercase;
      }
      .stDownloadButton > button:hover { background: #cffafe; }

      /* ============ INLINE CODE ============ */
      [data-testid="stMarkdownContainer"] code {
        background: #ecfeff; border: 1px solid #a5f3fc;
        border-radius: 4px; padding: 1px 6px;
        color: #0e7490; font-size: 0.88em;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
      }

      /* =================================================================
         WORKSHOP ZONE -- target the 4th tab panel only.
         Streamlit renders all tab panels in the DOM; nth-of-type(4) is the
         Optimization tab.  Everything inside flips to dark.
         ================================================================= */
      .stTabs [role="tabpanel"]:nth-of-type(4) {
        background:
          radial-gradient(800px 400px at 0% 0%, rgba(245,158,11,0.10), transparent 60%),
          radial-gradient(800px 400px at 100% 0%, rgba(34,211,238,0.08), transparent 60%),
          #0a0f1f;
        background-image:
          linear-gradient(rgba(148,163,184,0.06) 1px, transparent 1px),
          linear-gradient(90deg, rgba(148,163,184,0.06) 1px, transparent 1px),
          linear-gradient(180deg, #0a0f1f 0%, #060914 100%);
        background-size: 36px 36px, 36px 36px, 100% 100%;
        border: 1px solid #1f2937;
        border-top: 2px solid #f59e0b;
        border-radius: 0 0 14px 14px;
        padding: 28px 28px 32px;
        margin-top: -1px;
      }
      .stTabs [role="tabpanel"]:nth-of-type(4) h1,
      .stTabs [role="tabpanel"]:nth-of-type(4) h2,
      .stTabs [role="tabpanel"]:nth-of-type(4) h3,
      .stTabs [role="tabpanel"]:nth-of-type(4) h4 {
        color: #f8fafc !important;
      }
      .stTabs [role="tabpanel"]:nth-of-type(4) h3::before {
        background: #f59e0b;
      }
      .stTabs [role="tabpanel"]:nth-of-type(4) .stMarkdown p,
      .stTabs [role="tabpanel"]:nth-of-type(4) .stMarkdown li,
      .stTabs [role="tabpanel"]:nth-of-type(4) [data-testid="stCaptionContainer"] {
        color: #cbd5e1;
      }
      .stTabs [role="tabpanel"]:nth-of-type(4) .metric-card {
        background: #111827;
        border-color: rgba(148,163,184,0.18);
        box-shadow: none;
      }
      .stTabs [role="tabpanel"]:nth-of-type(4) .metric-label { color: #94a3b8; }
      .stTabs [role="tabpanel"]:nth-of-type(4) .metric-value { color: #f8fafc; }
      .stTabs [role="tabpanel"]:nth-of-type(4) .metric-sub   { color: #cbd5e1; }
      .stTabs [role="tabpanel"]:nth-of-type(4) [data-testid="stMarkdownContainer"] code {
        background: rgba(34,211,238,0.10);
        border: 1px solid rgba(34,211,238,0.30);
        color: #67e8f9;
      }
      .stTabs [role="tabpanel"]:nth-of-type(4) .stAlert {
        background: rgba(34,211,238,0.06);
        border: 1px solid rgba(34,211,238,0.30);
        color: #cbd5e1;
      }
      .stTabs [role="tabpanel"]:nth-of-type(4) [data-testid="stExpander"] {
        background: #111827;
        border: 1px solid rgba(148,163,184,0.18);
      }
      .stTabs [role="tabpanel"]:nth-of-type(4) [data-testid="stExpander"] summary,
      .stTabs [role="tabpanel"]:nth-of-type(4) [data-testid="stExpander"] p {
        color: #e2e8f0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _score_color(score: float) -> str:
    if score >= 80:
        return "#16a34a"
    if score >= 60:
        return "#84cc16"
    if score >= 40:
        return "#f59e0b"
    return "#ef4444"


def _gauge(score: float) -> go.Figure:
    color = _score_color(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 38}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.18)"},
                    {"range": [40, 60], "color": "rgba(245,158,11,0.18)"},
                    {"range": [60, 80], "color": "rgba(132,204,22,0.18)"},
                    {"range": [80, 100], "color": "rgba(22,163,74,0.18)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85, "value": score,
                },
            },
        )
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=20, r=20),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _score_card(
    label: str,
    score: float,
    verdict: str,
    *,
    delta: float | None = None,
) -> None:
    color = _score_color(score)
    score_pct = max(0.0, min(100.0, score))
    pro_min, pro_max = 75, 100
    delta_html = ""
    if delta is not None and abs(delta) >= 1:
        sign = "+" if delta > 0 else ""
        delta_color = "#16a34a" if delta > 0 else "#ef4444"
        delta_html = (
            f'<div style="font-size:0.85rem;color:{delta_color};'
            f'font-weight:700;margin-left:4px;">{sign}{delta:.0f}</div>'
        )
    st.markdown(
        f"""
        <div class="metric-card score-card">
          <div class="metric-label">{label}</div>
          <div style="display:flex;align-items:baseline;gap:6px;
                       margin:8px 0 12px;">
            <div style="font-size:3rem;font-weight:800;color:{color};
                         line-height:1;">{score:.0f}</div>
            <div style="font-size:1rem;color:#64748b;">/ 100</div>
            {delta_html}
          </div>
          <div class="metric-sub" style="min-height:2.6em;">{verdict}</div>
          <div style="position:relative;height:8px;border-radius:4px;
                       background:rgba(148,163,184,0.18);margin-top:14px;">
            <div style="position:absolute;left:{pro_min}%;width:{pro_max-pro_min}%;
                         top:0;bottom:0;background:rgba(22,163,74,0.20);
                         border-radius:4px;"></div>
            <div style="position:absolute;left:0;top:0;height:100%;
                         width:{score_pct}%;background:{color};
                         border-radius:4px;"></div>
            <div style="position:absolute;left:{score_pct}%;top:-3px;
                         width:2px;height:14px;background:{color};
                         transform:translateX(-1px);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
                       font-size:0.7rem;color:#64748b;margin-top:6px;">
            <span>amateur</span><span>pro range</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _problem_callout(p) -> None:
    # accent / bg / border tuned for a light page so the tinted cards stay
    # readable without competing with the surrounding content.
    color_map = {
        "critical": ("#b91c1c", "#fef2f2", "#fca5a5"),
        "warning":  ("#b45309", "#fffbeb", "#fcd34d"),
        "info":     ("#0369a1", "#f0f9ff", "#7dd3fc"),
    }
    accent, bg, border = color_map.get(p.severity, color_map["info"])
    severity_label = {"critical": "CRITICAL",
                      "warning": "WEAKNESS",
                      "info": "WATCH-OUT"}.get(p.severity, "ISSUE")
    st.markdown(
        f"""
        <div style="border:1px solid {border};background:{bg};
                     border-radius:12px;padding:16px 20px;margin-bottom:10px;">
          <div style="font-size:0.7rem;letter-spacing:.14em;font-weight:800;
                       color:{accent};margin-bottom:6px;">{severity_label}</div>
          <div style="font-size:1.1rem;font-weight:700;line-height:1.3;
                       margin-bottom:6px;color:#0f172a;">{p.headline}</div>
          <div style="font-size:0.92rem;color:#334155;line-height:1.5;">
            {p.detail}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_flops(flops: int) -> str:
    for unit, divisor in [("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if flops >= divisor:
            return f"{flops / divisor:.2f} {unit}FLOPs"
    return f"{flops} FLOPs"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="display:flex;justify-content:space-between;align-items:center;
                 flex-wrap:wrap;gap:14px;margin-bottom:18px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:48px;height:48px;border-radius:10px;
                     background:#f59e0b;
                     display:flex;align-items:center;justify-content:center;
                     font-family:ui-monospace,monospace;font-weight:900;
                     font-size:1.5rem;color:#0c0a09;
                     box-shadow:0 6px 14px rgba(245,158,11,0.35);">
          ⌬
        </div>
        <div>
          <div style="font-size:0.72rem;letter-spacing:.18em;
                       color:#64748b;font-weight:700;">
            DEPLOYMENT QUALIFICATION TERMINAL
          </div>
          <div style="font-size:2.0rem;font-weight:800;line-height:1.1;
                       margin-top:2px;color:#0f172a;">
            Deploy/X &nbsp;·&nbsp; Model Health Audit
          </div>
        </div>
      </div>
      <div class="status-pill">SYSTEM ONLINE</div>
    </div>
    <div style="font-size:0.98rem;color:#334155;line-height:1.55;
                 max-width:820px;margin-bottom:8px;">
      Drop a PyTorch checkpoint and we'll tell you what's wrong with it on the
      way to a robot, a self-driving car, or any hardware that has to run it
      for real. Then <b style="color:#b45309;">fix it in the workshop</b>
      and walk out with a deploy-grade <code>.onnx</code> file.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="border:1px solid #7dd3fc;
                     border-radius:14px; padding:18px;
                     background:#f0f9ff;">
          <div style="font-size:0.75rem;letter-spacing:.12em;
                       text-transform:uppercase;color:#0369a1;
                       font-weight:700;margin-bottom:6px;">
            Neural Network Optimization
          </div>
          <div style="font-size:1.15rem;font-weight:700;line-height:1.25;
                       margin-bottom:10px;color:#0f172a;">
            From training engineer to deployment engineer.
          </div>
          <div style="font-size:0.92rem;color:#334155;line-height:1.55;">
            This audit shows you <i>what's wrong</i>. The course turns you
            into the engineer who <b>ships</b> -- on real hardware, in real
            robots and vehicles, end to end.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        "**You'll learn to:**\n"
        "- Cut model size 4x without losing accuracy\n"
        "- Hit 2-3x lower latency on the same hardware\n"
        "- Ship to robots, drones, and edge devices without surprises\n"
        "- Diagnose models that won't survive deployment"
    )
    st.link_button(
        "Enroll in the course",
        "https://www.thinkautonomous.ai/",
        type="primary",
        use_container_width=True,
    )
    st.divider()
    st.caption(
        "Your file never leaves this process -- analysis runs locally in "
        "Streamlit."
    )

# ---------------------------------------------------------------------------
# Upload area
# ---------------------------------------------------------------------------

ss = st.session_state

uploaded = st.file_uploader(
    "Drop your `.pt` / `.pth` here", type=["pt", "pth"], accept_multiple_files=False
)

if uploaded is None:
    st.info("Waiting for a checkpoint -- drop a `.pt` or `.pth` file above.")
    st.stop()

buffer = uploaded.getvalue()
source_label = uploaded.name

st.caption(f"Analyzing: **{source_label}**")

size_mb = len(buffer) / (1024 * 1024)
if size_mb > 2048:
    st.error(
        f"File is {size_mb:.0f} MB. Skip the upload and run the analyzer "
        "locally for files this large."
    )
    st.stop()

file_hash = hashlib.sha1(buffer).hexdigest()

if ss.get("file_hash") != file_hash:
    with st.spinner("Loading & analyzing checkpoint..."):
        try:
            report, load_mode, obj = analyze(buffer)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load this checkpoint: {exc}")
            st.stop()
    # Original = the file as uploaded (never mutated -- Tabs 1-3 always
    # render against this).  Optimized = whatever Tab 4 has produced so far,
    # which stacks across multiple Apply clicks until the user resets.
    ss["file_hash"] = file_hash
    ss["report"] = report
    ss["original_report"] = report
    ss["load_mode"] = load_mode
    ss["obj"] = obj
    ss["original_obj"] = obj
    ss["optimized_obj"] = None          # populated by Tab 4 actions
    ss["optimized_report"] = None
    ss["applied_actions"] = []          # chronological list of technique names
    ss["last_snippet"] = None           # most recent code snippet for CODE Review

report = ss["report"]
load_mode = ss["load_mode"]
original_report = ss["original_report"]


def _reset_optimizations() -> None:
    ss["optimized_obj"] = None
    ss["optimized_report"] = None
    ss["applied_actions"] = []
    ss["last_snippet"] = None
    ss["onnx_result"] = None


if load_mode == "pickle":
    st.warning(
        "This checkpoint required Python's pickle to load, which can execute "
        "arbitrary code. We loaded it because you uploaded it -- only do this "
        "with files you trust."
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

hero_left, hero_right = st.columns([1, 2])

with hero_left:
    st.plotly_chart(_gauge(report.deployment_health_score), use_container_width=True)

with hero_right:
    st.markdown("##### Deployment Health Score")
    st.markdown(f"### {report.overall_verdict}")
    arch = report.architecture
    badges = [
        f"<span class='arch-badge'>{report.parameter_count / 1e6:.2f}M params</span>",
        f"<span class='arch-badge'>{report.file_size_mb:.1f} MB on disk</span>",
    ]
    if arch:
        badges.insert(0, f"<span class='arch-badge'>Looks like: {arch}</span>")
    if report.estimated_flops:
        badges.append(
            f"<span class='arch-badge'>~{_format_flops(report.estimated_flops)}</span>"
        )
    st.markdown(" ".join(badges), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Three-tab structure: 1. The network -> 2. Opportunities -> 3. Deployment
# ---------------------------------------------------------------------------

from device_estimates import network_stats  # noqa: E402

stats = network_stats(ss["obj"])

# Build the model passport once -- used in Tab 1.  Optimization changes
# inside Tab 4 do *not* mutate the passport or stats: those describe the
# uploaded model, period.
from passport import build_passport  # noqa: E402

passport = build_passport(ss["obj"], report.architecture)

tab_about, tab_opportunities, tab_deployment, tab_optimization = st.tabs([
    "1. About",
    "2. Opportunities",
    "3. Deployment",
    "4. Optimization",
])

# =====================================================================
# Tab 1 -- About: who this model is, what it expects, what it does
# =====================================================================
with tab_about:
    st.markdown("### Model passport")
    st.caption(
        "A spec sheet for the artifact you uploaded. Everything here is "
        "inferred from the file itself -- no forward pass needed."
    )

    p1, p2 = st.columns(2)
    with p1:
        arch_label = passport.architecture or "Unknown architecture"
        st.markdown(
            f"""
            <div class="metric-card" style="padding:22px 24px;">
              <div class="metric-label">Architecture</div>
              <div style="font-size:1.6rem;font-weight:800;color:#0369a1;
                           margin:4px 0 14px;line-height:1.1;">
                {arch_label}
              </div>
              <div style="display:grid;grid-template-columns:120px 1fr;
                           gap:6px 14px;font-size:0.92rem;color:#1e293b;">
                <div style="color:#64748b;">Likely task</div>
                <div><b>{passport.likely_task}</b></div>
                <div style="color:#64748b;">Trained on</div>
                <div>{passport.likely_dataset or "Unknown / custom dataset"}</div>
                <div style="color:#64748b;">Parameters</div>
                <div>{stats.parameter_count / 1e6:.2f}M</div>
                <div style="color:#64748b;">Layers</div>
                <div>{stats.leaf_modules}</div>
                <div style="color:#64748b;">On disk</div>
                <div>{report.file_size_mb:.1f} MB</div>
                <div style="color:#64748b;">Compute</div>
                <div>~{_format_flops(report.estimated_flops or 0)} per forward</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        out_dim_str = (
            ", ".join(str(x) for x in passport.output_shape)
            if passport.output_shape else "?"
        )
        in_dim_str = (
            ", ".join(str(x) for x in passport.input_shape)
            if passport.input_shape else "?"
        )
        sample_html = ""
        if passport.sample_classes:
            sample_html = (
                "<div style='font-size:0.85rem;color:#64748b;"
                "margin-top:10px;'>Likely outputs include: "
                + ", ".join(f"<i>{c}</i>" for c in passport.sample_classes)
                + ", ...</div>"
            )
        st.markdown(
            f"""
            <div class="metric-card" style="padding:22px 24px;">
              <div class="metric-label">Data flow</div>
              <div style="display:flex;align-items:center;gap:14px;
                           margin:18px 0;">
                <div style="flex:1;text-align:center;">
                  <div style="font-size:0.72rem;color:#64748b;
                               letter-spacing:.1em;">INPUT</div>
                  <div style="font-size:1.05rem;font-weight:700;
                               color:#047857;margin:4px 0;">
                    {passport.input_description}
                  </div>
                  <div style="font-size:0.78rem;color:#475569;
                               font-family:ui-monospace,monospace;">
                    [{in_dim_str}]
                  </div>
                </div>
                <div style="font-size:1.4rem;color:#0369a1;">→</div>
                <div style="flex:0 0 auto;font-size:0.72rem;
                             color:#0369a1;letter-spacing:.1em;
                             font-weight:800;
                             padding:6px 12px;border-radius:8px;
                             background:#e0f2fe;
                             border:1px solid #7dd3fc;">
                  {arch_label}
                </div>
                <div style="font-size:1.4rem;color:#0369a1;">→</div>
                <div style="flex:1;text-align:center;">
                  <div style="font-size:0.72rem;color:#64748b;
                               letter-spacing:.1em;">OUTPUT</div>
                  <div style="font-size:1.05rem;font-weight:700;
                               color:#b91c1c;margin:4px 0;">
                    {passport.output_description}
                  </div>
                  <div style="font-size:0.78rem;color:#475569;
                               font-family:ui-monospace,monospace;">
                    [{out_dim_str}]
                  </div>
                </div>
              </div>
              {sample_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Network at a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Parameters", f"{stats.parameter_count / 1e6:.2f}M")
    m2.metric("Layers", f"{stats.leaf_modules}")
    m3.metric("Carries today", f"{stats.fp32_mb:.0f} MB")
    slim_mb = stats.int8_mb
    m4.metric(
        "Could carry",
        f"{slim_mb:.0f} MB",
        delta=f"-{stats.fp32_mb - slim_mb:.0f} MB",
        delta_color="inverse",
        help="A deploy-grade version of this model would weigh roughly a quarter of what it does today.",
    )

    glance_left, glance_right = st.columns([3, 2])
    with glance_left:
        slim_pct = max(8, int(slim_mb / max(stats.fp32_mb, 1) * 100))
        st.markdown(
            f"""
            <div class="metric-card" style="height:210px;
                                              display:flex;flex-direction:column;
                                              justify-content:center;">
              <div class="metric-label">Carry weight</div>
              <div style="margin-top:14px;">
                <div style="display:flex;justify-content:space-between;
                             font-size:0.82rem;color:#64748b;margin-bottom:4px;">
                  <span>Today</span><span><b style='color:#b91c1c;'>{stats.fp32_mb:.0f} MB</b></span>
                </div>
                <div style="height:18px;border-radius:6px;
                             background:#fee2e2;
                             border:1px solid #fca5a5;
                             margin-bottom:18px;"></div>
                <div style="display:flex;justify-content:space-between;
                             font-size:0.82rem;color:#64748b;margin-bottom:4px;">
                  <span>Deploy-grade</span><span><b style='color:#15803d;'>{slim_mb:.0f} MB</b></span>
                </div>
                <div style="height:18px;border-radius:6px;
                             background:#f1f5f9;position:relative;">
                  <div style="position:absolute;left:0;top:0;bottom:0;
                               width:{slim_pct}%;border-radius:6px;
                               background:#bbf7d0;
                               border:1px solid #86efac;"></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with glance_right:
        if stats.biggest_layer:
            bn, bc = stats.biggest_layer
            share = bc / max(stats.parameter_count, 1)
            st.markdown(
                f"""
                <div class="metric-card" style="height:210px;display:flex;
                                                  flex-direction:column;
                                                  justify-content:center;">
                  <div class="metric-label">Heaviest single piece</div>
                  <div style="font-size:2rem;font-weight:800;color:#0369a1;
                               line-height:1;margin:8px 0 6px;">
                    {share:.0%}
                  </div>
                  <div class="metric-sub">
                    of the entire model lives in one component
                    ({bc / 1e6:.1f}M parameters).
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =====================================================================
# Tab 2 -- Opportunities: category breakdown + weaknesses + problems
# =====================================================================
with tab_opportunities:
    st.markdown("### Category breakdown")
    st.caption(
        "Four dimensions of deployment-readiness. None of these are "
        "techniques you should run -- they're attributes the model has."
    )
    cols = st.columns(4)
    with cols[0]:
        _score_card("Efficiency", report.precision.score, report.precision.verdict)
    with cols[1]:
        _score_card("Leanness", report.pruning.score, report.pruning.verdict)
    with cols[2]:
        _score_card("Compactness", report.size.score, report.size.verdict)
    with cols[3]:
        _score_card(
            "Exportability", report.exportability.score, report.exportability.verdict
        )

    st.markdown("### Weaknesses & opportunities")
    st.caption("What's wrong with this model, in plain language.")
    for rec in report.recommendations:
        st.markdown(f"- {rec}")

    st.markdown("### Problems detected")
    problems = detect_problems(report)
    for p in problems:
        _problem_callout(p)

# =====================================================================
# Tab 3 -- Deployment: compatibility matrix + scenario cards
# =====================================================================
with tab_deployment:
    st.markdown("### Deployment compatibility")
    st.caption(
        "Each runtime is its own world. Here's how your model lands on each, "
        "given its current state -- not its theoretical ceiling."
    )
    families = compute_compatibility(report)
    fam_cols = st.columns(3)
    for i, fam in enumerate(families):
        with fam_cols[i % 3]:
            _compat_card(fam)

    st.markdown("### If this shipped today...")
    st.caption(
        "What your model would actually do in the field. Verdicts use a 50% "
        "safety margin against the budget."
    )
    scenarios = build_scenarios(report)
    # (accent_text, deep_text, bg, border) per severity -- chosen so the card
    # stays readable on a white page even with a translucent tint.
    verdict_color = {
        "good":     ("#15803d", "#14532d", "#f0fdf4", "#86efac"),
        "warning":  ("#b45309", "#78350f", "#fffbeb", "#fcd34d"),
        "critical": ("#b91c1c", "#7f1d1d", "#fef2f2", "#fca5a5"),
    }
    scen_cols = st.columns(2)
    for i, scen in enumerate(scenarios):
        accent, deep, bg, border = verdict_color.get(
            scen.severity, verdict_color["warning"]
        )
        with scen_cols[i % 2]:
            st.markdown(
                f"""
                <div style="border:1px solid {border};background:{bg};
                             border-radius:14px;padding:18px 22px;
                             margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;
                               align-items:flex-start;margin-bottom:8px;gap:14px;">
                    <div style="display:flex;align-items:center;gap:12px;
                                 flex:1;min-width:0;">
                      <div style="color:{accent};flex-shrink:0;">
                        <svg width="32" height="32" viewBox="0 0 24 24"
                             xmlns="http://www.w3.org/2000/svg">
                          {scen.icon}
                        </svg>
                      </div>
                      <div style="min-width:0;">
                        <div style="font-size:0.72rem;letter-spacing:.14em;
                                     color:{accent};font-weight:800;">
                          {scen.setting.upper()}
                        </div>
                        <div style="font-size:0.85rem;color:#475569;
                                     margin-top:2px;">
                          {scen.target}
                        </div>
                      </div>
                    </div>
                    <div style="font-size:0.72rem;letter-spacing:.12em;
                                 font-weight:800;color:white;
                                 background:{accent};
                                 padding:4px 11px;border-radius:999px;
                                 flex-shrink:0;">
                      {scen.verdict.upper()}
                    </div>
                  </div>
                  <div style="font-size:0.98rem;font-style:italic;color:{deep};
                               margin-bottom:12px;line-height:1.45;">
                    "{scen.hook}"
                  </div>
                  <div style="font-size:0.92rem;color:#1e293b;line-height:1.55;
                               margin-bottom:8px;">
                    <b style="color:#b91c1c;">Today:</b> {scen.today_line}
                  </div>
                  <div style="font-size:0.92rem;color:#1e293b;line-height:1.55;">
                    <b style="color:#15803d;">Optimized:</b> {scen.optimized_line}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =====================================================================
# Tab 4 -- OPTIMIZATION ZONE: visually distinct workshop where techniques
# stack on top of each other.  Tabs 1-3 stay frozen on the original
# upload; only this tab reflects the cumulative transformation.
# =====================================================================
with tab_optimization:
    from optimizers import TECHNIQUES, try_export_onnx  # noqa: E402

    # The "workshop" view reads from optimized_* when present; falls back
    # to the original upload otherwise.  Either way, Tabs 1-3 are
    # untouched.
    opt_obj = ss.get("optimized_obj") or ss["original_obj"]
    opt_report = ss.get("optimized_report") or original_report
    has_applied = bool(ss.get("applied_actions"))

    # Workshop banner: amber accent, distinct from the cyan elsewhere,
    # so this tab visually reads as a different room of the factory.
    chain = " → ".join(ss["applied_actions"]) if has_applied else "Untouched"
    st.markdown(
        f"""
        <div style="border:1px solid rgba(245,158,11,0.45);
                     border-left:5px solid #f59e0b;
                     background:linear-gradient(160deg,
                                                 rgba(245,158,11,0.10),
                                                 rgba(15,23,42,0.0));
                     border-radius:12px;padding:18px 22px;
                     margin-bottom:18px;">
          <div style="display:flex;justify-content:space-between;
                       align-items:center;flex-wrap:wrap;gap:14px;">
            <div>
              <div style="font-size:0.72rem;letter-spacing:.16em;
                           font-weight:800;color:#f59e0b;">
                OPTIMIZATION ZONE
              </div>
              <div style="font-size:1.4rem;font-weight:800;
                           margin-top:4px;line-height:1.2;">
                The workshop
              </div>
              <div style="font-size:0.92rem;color:#cbd5e1;
                           margin-top:4px;">
                Stack techniques on top of each other. The other tabs
                stay frozen on the file you uploaded.
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:0.72rem;color:#94a3b8;
                           letter-spacing:.1em;">CURRENT CHAIN</div>
              <div style="font-family:ui-monospace,monospace;
                           font-size:0.95rem;font-weight:700;
                           color:#fcd34d;margin-top:4px;
                           max-width:480px;overflow-wrap:anywhere;">
                {chain}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Pick a technique to add to the chain")

    technique_keys = [t.key for t in TECHNIQUES]
    technique_labels = {t.key: t.label for t in TECHNIQUES}
    technique_descriptions = {t.key: t.description for t in TECHNIQUES}
    technique_notebooks = {t.key: t.notebook for t in TECHNIQUES}
    technique_apply = {t.key: t.apply for t in TECHNIQUES}

    chosen = st.radio(
        "Technique",
        technique_keys,
        format_func=lambda k: technique_labels[k],
        index=0,
        key="opt_technique",
        label_visibility="collapsed",
    )
    st.info(technique_descriptions[chosen])
    st.caption(f"Course material: `{technique_notebooks[chosen]}` in this repo.")

    btn_apply, btn_reset, btn_dl = st.columns([2, 1, 1])
    with btn_apply:
        if st.button(
            f"Apply on top: {technique_labels[chosen]}",
            type="primary", use_container_width=True, key="opt_apply",
        ):
            import io
            import torch as _torch
            current_obj = ss.get("optimized_obj") or ss["original_obj"]
            try:
                new_obj, snippet = technique_apply[chosen](current_obj)
                bio = io.BytesIO()
                _torch.save(new_obj, bio)
                new_report, _mode, new_obj_loaded = analyze(bio.getvalue())
                ss["optimized_obj"] = new_obj_loaded
                ss["optimized_report"] = new_report
                ss["applied_actions"] = (
                    ss.get("applied_actions", []) + [technique_labels[chosen]]
                )
                ss["last_snippet"] = (snippet, None)
            except Exception as exc:  # noqa: BLE001 -- shown to the user
                # Some chains are illegal (e.g. quantizing already-quantized
                # weights).  Surface a clean error in CODE Review.
                from optimizers import Snippet as _S
                err_snip = _S(
                    title=f"{technique_labels[chosen]} couldn't stack on this chain",
                    summary="Some optimizations don't compose -- try Reset, then a different sequence.",
                    notebook=technique_notebooks[chosen],
                    code=f"# Error: {exc}",
                )
                ss["last_snippet"] = (err_snip, str(exc).splitlines()[0][:200])
            st.rerun()

    with btn_reset:
        if st.button(
            "Reset", use_container_width=True, key="opt_reset",
            help="Roll the chain back to the original upload.",
        ):
            _reset_optimizations()
            st.rerun()

    with btn_dl:
        # ONNX export of the current chain.  On success, the .onnx bytes
        # are stashed in session state and surfaced as a download button
        # immediately below.
        if st.button(
            "Export to ONNX",
            use_container_width=True,
            key="opt_onnx",
            help="Export the current optimized chain to a real .onnx file.",
        ):
            with st.spinner("Tracing the graph..."):
                _, snippet, err, onnx_bytes = try_export_onnx(opt_obj)
            ss["onnx_result"] = (snippet, err, onnx_bytes)
            st.rerun()

    # CODE Review -- right under the buttons.
    last = ss.get("last_snippet")
    if last:
        snippet, err = last
        accent = "#ef4444" if err else "#86efac"
        result_html = (
            f"<span style='color:{accent};font-weight:700;'>FAILED</span> &middot; {err}"
            if err else
            "<span style='color:#86efac;font-weight:700;'>OK</span> &middot; applied to the chain"
        )
        st.markdown(
            f"""
            <div style="border:1px solid rgba(245,158,11,0.30);
                         background:rgba(15,23,42,0.35);
                         border-radius:12px;padding:14px 18px;
                         margin:14px 0 6px;">
              <div style="display:flex;justify-content:space-between;
                           align-items:baseline;margin-bottom:8px;">
                <div style="font-size:0.72rem;letter-spacing:.16em;
                             font-weight:800;color:#f59e0b;">
                  CODE REVIEW
                </div>
                <div style="font-size:0.78rem;color:#94a3b8;">
                  {result_html}
                </div>
              </div>
              <div style="font-size:1.05rem;font-weight:700;
                           margin-bottom:4px;">{snippet.title}</div>
              <div style="font-size:0.9rem;color:#cbd5e1;
                           margin-bottom:6px;">{snippet.summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(snippet.code, language="python")
        st.caption(
            f"Lifted from `{snippet.notebook}` in this repo. "
            "The notebook walks through *why* it works, not just what."
        )

    with st.expander("Why precision matters (the 30-second version)"):
        st.markdown(
            "Every weight in your model is a number stored at some precision. "
            "The fewer bits per number, the less memory it takes, the less "
            "bandwidth your hardware spends moving it around, and the faster "
            "your model runs. Modern deployment chips have dedicated "
            "low-precision math units that sit idle when you ship at full "
            "precision."
        )
        st.markdown(
            """
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                         gap:12px;margin-top:12px;">
              <div class="metric-card" style="padding:14px 16px;">
                <div class="metric-label">FP32</div>
                <div style="font-size:1.4rem;font-weight:800;color:#fca5a5;
                             margin:4px 0;">32 bits</div>
                <div class="metric-sub">Where most models are trained.
                Almost never where they should ship.</div>
              </div>
              <div class="metric-card" style="padding:14px 16px;">
                <div class="metric-label">FP16</div>
                <div style="font-size:1.4rem;font-weight:800;color:#fcd34d;
                             margin:4px 0;">16 bits</div>
                <div class="metric-sub">Half the memory, same accuracy
                for almost any inference task.</div>
              </div>
              <div class="metric-card" style="padding:14px 16px;">
                <div class="metric-label">INT8</div>
                <div style="font-size:1.4rem;font-weight:800;color:#86efac;
                             margin:4px 0;">8 bits</div>
                <div class="metric-sub">A quarter of the carry weight.
                Pays for itself immediately on edge hardware.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Score after the chain")
    cols = st.columns(4)
    pairs = [
        ("Efficiency", opt_report.precision, original_report.precision),
        ("Leanness", opt_report.pruning, original_report.pruning),
        ("Compactness", opt_report.size, original_report.size),
        ("Exportability", opt_report.exportability, original_report.exportability),
    ]
    for col, (label, cur, orig) in zip(cols, pairs):
        with col:
            delta = cur.score - orig.score if has_applied else None
            _score_card(label, cur.score, cur.verdict, delta=delta)

    if has_applied:
        delta_score = (
            opt_report.deployment_health_score
            - original_report.deployment_health_score
        )
        sign = "+" if delta_score >= 0 else ""
        st.success(
            f"Overall score: **{original_report.deployment_health_score:.0f} "
            f"→ {opt_report.deployment_health_score:.0f}** ({sign}{delta_score:.0f}). "
            f"Carry weight: {original_report.file_size_mb:.0f} MB → "
            f"{opt_report.file_size_mb:.0f} MB."
        )

    # ONNX result (if the user has tried Export to ONNX from the button row).
    onnx_result = ss.get("onnx_result")
    if onnx_result:
        snippet, err, onnx_bytes = onnx_result
        st.markdown("#### ONNX export")
        if err:
            st.error(f"Export failed: {err}")
        else:
            mb = (len(onnx_bytes) / (1024 * 1024)) if onnx_bytes else 0
            st.success(
                f"Export succeeded -- {mb:.1f} MB of clean ONNX, ready for "
                "any deployment runtime."
            )
            if onnx_bytes:
                base = source_label.rsplit(".", 1)[0]
                fname = f"{base}_optimized.onnx".replace(" ", "_").replace("/", "_")
                st.download_button(
                    "Download .onnx",
                    data=onnx_bytes,
                    file_name=fname,
                    mime="application/octet-stream",
                    help="The optimized model in deploy-grade ONNX form.",
                )
        with st.expander("The course code that does this", expanded=not err):
            st.markdown(f"_{snippet.summary}_")
            st.code(snippet.code, language="python")
            st.caption(f"From `{snippet.notebook}` in the course repo.")

st.divider()
st.markdown(
    "Want to close every gap above? The Neural Network Optimization course "
    "trains the engineer who ships -- not just the engineer who trains."
)
st.link_button("Enroll in the course", COURSE_URL, type="primary")
