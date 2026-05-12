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


COURSE_URL = "https://www.thinkautonomous.ai/"


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
      /* ============ PALETTE ============
         Black & white industrial palette with electric blue as the single
         accent.  Audit (Tab 1) is white-on-near-black-text; Workshop
         (Tab 2) flips to pure black-on-white-text.  No warm tones. */

      /* ============ PAGE CHROME ============ */
      .stApp { background: #ffffff; }
      .block-container { padding-top: 4rem; max-width: 1320px; }
      [data-testid="stHeader"] {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(6px);
      }

      /* ============ TYPOGRAPHY ============ */
      h1, h2, h3, h4, h5, h6 { color: #202F46 !important; letter-spacing: -0.01em; }
      h1 { font-size: 2.0rem; font-weight: 800; }
      h2 { font-size: 1.5rem; font-weight: 800; }
      h3 {
        font-size: 1.25rem; font-weight: 700;
        position: relative; padding-left: 14px; margin-top: 1.6rem;
      }
      h3::before {
        content: ""; position: absolute; left: 0; top: 0.2em; bottom: 0.2em;
        width: 3px; background: #202F46;
      }
      h4 { font-size: 1.05rem; font-weight: 700; margin-top: 1.2rem; }
      .stMarkdown p, .stMarkdown li { color: #1a1a1a; }
      [data-testid="stCaptionContainer"] { color: #595959; }

      /* ============ TABS ============ */
      .stTabs [data-baseweb="tab-list"] {
        gap: 2px; border-bottom: 1px solid #d4d4d4;
      }
      .stTabs [data-baseweb="tab"] {
        height: 46px; padding: 0 26px;
        background: #fafafa;
        border: 1px solid #d4d4d4; border-bottom: none;
        border-radius: 0;
        color: #525252 !important;
        font-weight: 700; font-size: 0.92rem;
        text-transform: uppercase; letter-spacing: .06em;
      }
      .stTabs [data-baseweb="tab"]:hover {
        color: #202F46 !important; background: #f5f5f5;
      }
      .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: #202F46 !important;
        border-color: #202F46 !important;
        box-shadow: inset 0 -3px 0 #0e80e5;
      }

      /* ============ CARDS ============ */
      .metric-card {
        border: 1px solid #d4d4d4; border-radius: 0;
        padding: 18px 20px 16px; background: white;
        box-shadow: none;
      }
      .metric-label {
        font-size: 0.72rem; color: #595959;
        text-transform: uppercase; letter-spacing: .14em; font-weight: 700;
      }
      .metric-value {
        font-size: 2.0rem; font-weight: 800; line-height: 1.1; color: #202F46;
      }
      .metric-sub { font-size: 0.92rem; color: #3f3f3f; margin-top: 4px; }

      /* ============ BADGES ============ */
      .arch-badge {
        display: inline-block; padding: 4px 12px; border-radius: 0;
        background: #f5f5f5; border: 1px solid #d4d4d4;
        color: #202F46; font-weight: 700;
        font-size: 0.78rem; letter-spacing: .04em;
        margin-right: 6px; margin-bottom: 4px;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
      }

      /* ============ STATUS PILL ============ */
      .status-pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 5px 14px; border-radius: 0;
        background: #202F46; border: 1px solid #202F46;
        color: #ffffff; font-size: 0.74rem; font-weight: 700;
        letter-spacing: .14em;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
      }
      .status-pill::before {
        content: ""; width: 8px; height: 8px; border-radius: 50%;
        background: #0e80e5; box-shadow: 0 0 8px #0e80e5;
        animation: pulse 1.6s ease-in-out infinite;
      }
      @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .4; } }

      /* ============ BUTTONS ============ */
      .stButton > button[kind="primary"] {
        background: #202F46 !important; border: 1px solid #202F46;
        color: #ffffff !important; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
        border-radius: 0;
      }
      .stButton > button[kind="primary"]:hover {
        background: #0e80e5 !important; border-color: #0e80e5;
      }
      .stButton > button[kind="secondary"] {
        background: #ffffff;
        border: 1px solid #202F46;
        color: #202F46 !important; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
        border-radius: 0;
      }
      .stButton > button[kind="secondary"]:hover {
        background: #202F46;
        color: #ffffff !important;
      }
      .stDownloadButton > button {
        background: #0e80e5;
        border: 1px solid #0e80e5;
        color: #ffffff !important; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
        border-radius: 0;
      }
      .stDownloadButton > button:hover {
        background: #0a66b8; border-color: #0a66b8;
      }

      /* ============ INLINE CODE ============ */
      [data-testid="stMarkdownContainer"] code {
        background: #f5f5f5; border: 1px solid #d4d4d4;
        border-radius: 0; padding: 1px 6px;
        color: #202F46; font-size: 0.88em;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
      }

      /* ============ FILE UPLOADER ============ */
      [data-testid="stFileUploaderDropzone"] {
        background: #fafafa;
        border: 2px dashed #a3a3a3;
        border-radius: 0;
      }
      [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #0e80e5; background: #f5f5f5;
      }

      /* ============ RADIO (technique chooser inside workshop) ============ */
      .stRadio > label, .stRadio > div { gap: 8px; }

      /* =================================================================
         WORKSHOP ZONE -- Tab 2 panel.  Pure-black factory floor with
         blue as the only accent.  Crisp white text, no warm tones.
         ================================================================= */
      .stTabs [role="tabpanel"]:nth-of-type(2) {
        background: #000000;
        border: 1px solid #000000;
        border-top: 3px solid #0e80e5;
        border-radius: 0;
        padding: 28px 28px 32px;
        margin-top: -1px;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) h1,
      .stTabs [role="tabpanel"]:nth-of-type(2) h2,
      .stTabs [role="tabpanel"]:nth-of-type(2) h3,
      .stTabs [role="tabpanel"]:nth-of-type(2) h4 {
        color: #ffffff !important;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) h3::before {
        background: #0e80e5;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) .stMarkdown p,
      .stTabs [role="tabpanel"]:nth-of-type(2) .stMarkdown li,
      .stTabs [role="tabpanel"]:nth-of-type(2) [data-testid="stCaptionContainer"] {
        color: #d4d4d4;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) .metric-card {
        background: #0f0f0f;
        border-color: #262626;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) .metric-label { color: #a3a3a3; }
      .stTabs [role="tabpanel"]:nth-of-type(2) .metric-value { color: #ffffff; }
      .stTabs [role="tabpanel"]:nth-of-type(2) .metric-sub   { color: #d4d4d4; }
      .stTabs [role="tabpanel"]:nth-of-type(2) [data-testid="stMarkdownContainer"] code {
        background: #0f0f0f; border: 1px solid #404040;
        color: #7bbef0;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) .stAlert {
        background: #0f0f0f;
        border: 1px solid #262626;
        color: #d4d4d4;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) [data-testid="stExpander"] {
        background: #0f0f0f;
        border: 1px solid #262626;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) [data-testid="stExpander"] summary,
      .stTabs [role="tabpanel"]:nth-of-type(2) [data-testid="stExpander"] p,
      .stTabs [role="tabpanel"]:nth-of-type(2) [data-testid="stExpander"] label {
        color: #ffffff !important;
      }
      /* Radio labels inside workshop need explicit white */
      .stTabs [role="tabpanel"]:nth-of-type(2) .stRadio label,
      .stTabs [role="tabpanel"]:nth-of-type(2) .stRadio p {
        color: #ffffff !important;
      }
      /* Primary "Apply" button stays black-on-white inside the workshop --
         that's the strongest contrast against a black panel. */
      .stTabs [role="tabpanel"]:nth-of-type(2) .stButton > button[kind="primary"] {
        background: #ffffff !important; border-color: #ffffff;
        color: #202F46 !important;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) .stButton > button[kind="primary"]:hover {
        background: #0e80e5 !important; border-color: #0e80e5;
        color: #ffffff !important;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) .stButton > button[kind="secondary"] {
        background: #0f0f0f; border-color: #525252; color: #ffffff !important;
      }
      .stTabs [role="tabpanel"]:nth-of-type(2) .stButton > button[kind="secondary"]:hover {
        background: #0e80e5; border-color: #0e80e5;
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
        <div style="width:48px;height:48px;
                     background:#202F46;
                     display:flex;align-items:center;justify-content:center;
                     font-family:ui-monospace,monospace;font-weight:900;
                     font-size:1.5rem;color:#ffffff;
                     border:2px solid #202F46;">
          ⌬
        </div>
        <div>
          <div style="font-size:0.72rem;letter-spacing:.18em;
                       color:#595959;font-weight:700;">
            DEPLOYMENT QUALIFICATION TERMINAL
          </div>
          <div style="font-size:2.0rem;font-weight:800;line-height:1.1;
                       margin-top:2px;color:#202F46;">
            Deploy/X &nbsp;·&nbsp; Model Health Audit
          </div>
        </div>
      </div>
      <div class="status-pill">SYSTEM ONLINE</div>
    </div>
    <div style="font-size:0.98rem;color:#1a1a1a;line-height:1.55;
                 max-width:820px;margin-bottom:8px;">
      Drop a PyTorch checkpoint and we'll tell you what's wrong with it on the
      way to a robot, a self-driving car, or any hardware that has to run it
      for real. Then <b style="color:#0e80e5;">fix it in the workshop</b>
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
# Two-tab structure: 1. Audit  ·  2. Optimization
# ---------------------------------------------------------------------------

tab_audit, tab_optimization = st.tabs([
    "1. Audit",
    "2. Optimization",
])

# =====================================================================
# Tab 1 -- Audit: the headline score + the four category cards
# =====================================================================
with tab_audit:
    hero_left, hero_right = st.columns([1, 2])
    with hero_left:
        st.plotly_chart(
            _gauge(report.deployment_health_score), use_container_width=True
        )
    with hero_right:
        st.markdown("##### Deployment Health Score")
        st.markdown(f"### {report.overall_verdict}")
        # Architecture badge dropped on purpose -- fingerprint is too unreliable
        # to display as a confident "this is what your model is".
        badges = [
            f"<span class='arch-badge'>{report.parameter_count / 1e6:.2f}M params</span>",
            f"<span class='arch-badge'>{report.file_size_mb:.1f} MB on disk</span>",
        ]
        if report.estimated_flops:
            badges.append(
                f"<span class='arch-badge'>~{_format_flops(report.estimated_flops)}</span>"
            )
        st.markdown(" ".join(badges), unsafe_allow_html=True)

    st.markdown("### Category breakdown")
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

    # Workshop banner: factory-floor look -- black background, blue accent,
    # mono-style CURRENT CHAIN readout.
    chain = " → ".join(ss["applied_actions"]) if has_applied else "Untouched"
    st.markdown(
        f"""
        <div style="border:1px solid #262626;
                     border-left:4px solid #0e80e5;
                     background:#0f0f0f;
                     padding:18px 22px;margin-bottom:22px;">
          <div style="display:flex;justify-content:space-between;
                       align-items:center;flex-wrap:wrap;gap:14px;">
            <div>
              <div style="font-size:0.72rem;letter-spacing:.18em;
                           font-weight:800;color:#0e80e5;">
                THE WORKSHOP
              </div>
              <div style="font-size:1.4rem;font-weight:800;
                           margin-top:4px;line-height:1.2;color:#ffffff;">
                Stack optimizations. Export when ready.
              </div>
              <div style="font-size:0.92rem;color:#d4d4d4;
                           margin-top:4px;">
                Each technique applies on top of the previous one.
                The audit tab stays frozen on your original file.
              </div>
            </div>
            <div style="text-align:right;min-width:180px;">
              <div style="font-size:0.72rem;color:#a3a3a3;
                           letter-spacing:.14em;font-weight:700;">
                CURRENT CHAIN
              </div>
              <div style="font-family:ui-monospace,monospace;
                           font-size:0.92rem;font-weight:700;
                           color:#7bbef0;margin-top:4px;
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
    technique_videos = {t.key: t.video_url for t in TECHNIQUES}

    chosen = st.radio(
        "Technique",
        technique_keys,
        format_func=lambda k: technique_labels[k],
        index=0,
        key="opt_technique",
        label_visibility="collapsed",
    )

    # Two-column layout: description on the left, video walkthrough on the
    # right.  When ``video_url`` isn't set yet, the right column shows a
    # placeholder card so the slot is clearly reserved for a future clip.
    desc_col, vid_col = st.columns([3, 2])
    with desc_col:
        st.info(technique_descriptions[chosen])
        st.caption(
            f"Course material: `{technique_notebooks[chosen]}` in this repo."
        )
    with vid_col:
        video_url = technique_videos.get(chosen)
        if video_url:
            st.video(video_url)
        else:
            st.markdown(
                f"""
                <div style="border:1px dashed #404040;
                             background:#0f0f0f;
                             padding:18px 20px;
                             min-height:140px;display:flex;flex-direction:column;
                             align-items:center;justify-content:center;
                             text-align:center;">
                  <div style="font-size:1.6rem;color:#0e80e5;
                               margin-bottom:6px;">▶</div>
                  <div style="font-size:0.92rem;color:#ffffff;
                               font-weight:700;margin-bottom:2px;">
                    Walkthrough: {technique_labels[chosen]}
                  </div>
                  <div style="font-size:0.78rem;color:#a3a3a3;
                               letter-spacing:.04em;">
                    Video coming soon
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
        if err:
            result_html = (
                f"<span style='color:#fca5a5;font-weight:700;'>FAILED</span>"
                f" &middot; <span style='color:#a3a3a3;'>{err}</span>"
            )
        else:
            result_html = (
                "<span style='color:#7bbef0;font-weight:700;'>OK</span>"
                " &middot; <span style='color:#a3a3a3;'>applied to the chain</span>"
            )
        st.markdown(
            f"""
            <div style="border:1px solid #262626;
                         background:#0f0f0f;
                         padding:14px 18px;margin:14px 0 6px;">
              <div style="display:flex;justify-content:space-between;
                           align-items:baseline;margin-bottom:8px;">
                <div style="font-size:0.72rem;letter-spacing:.18em;
                             font-weight:800;color:#0e80e5;">
                  CODE REVIEW
                </div>
                <div style="font-size:0.78rem;">
                  {result_html}
                </div>
              </div>
              <div style="font-size:1.05rem;font-weight:700;
                           margin-bottom:4px;color:#ffffff;">{snippet.title}</div>
              <div style="font-size:0.9rem;color:#d4d4d4;
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
                <div style="font-size:1.4rem;font-weight:800;color:#7bbef0;
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
