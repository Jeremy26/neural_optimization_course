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
      /* Tighten Streamlit's default vertical rhythm */
      .block-container { padding-top: 2rem; max-width: 1280px; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      .metric-card {
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 14px;
        padding: 18px 18px 14px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
      }
      .metric-label {
        font-size: 0.78rem; color: #94a3b8;
        text-transform: uppercase; letter-spacing: .08em;
      }
      .metric-value { font-size: 2.0rem; font-weight: 700; line-height: 1.1; }
      .metric-sub  { font-size: 0.92rem; color: #cbd5e1; margin-top: 4px; }
      .arch-badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(56,189,248,0.15); color: #38bdf8; font-weight: 600;
        font-size: 0.85rem; letter-spacing: .04em;
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
    color_map = {
        "critical": ("#ef4444", "rgba(239,68,68,0.10)", "rgba(239,68,68,0.35)"),
        "warning":  ("#f59e0b", "rgba(245,158,11,0.10)", "rgba(245,158,11,0.35)"),
        "info":     ("#38bdf8", "rgba(56,189,248,0.10)", "rgba(56,189,248,0.35)"),
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
                       margin-bottom:6px;">{p.headline}</div>
          <div style="font-size:0.92rem;color:#cbd5e1;line-height:1.5;">
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

st.title("Model Deployment Health Score")
st.markdown(
    "Upload a PyTorch checkpoint (`.pt` / `.pth`) for a free deployment audit. "
    "We tell you what's wrong with your model on the way to a robot, a "
    "self-driving car, or any other hardware that has to run it for real."
)

with st.sidebar:
    st.markdown(
        """
        <div style="border:1px solid rgba(56,189,248,0.35);
                     border-radius:14px; padding:18px;
                     background:linear-gradient(160deg, rgba(56,189,248,0.10),
                                                rgba(15,23,42,0.0));">
          <div style="font-size:0.75rem;letter-spacing:.12em;
                       text-transform:uppercase;color:#38bdf8;
                       font-weight:700;margin-bottom:6px;">
            Neural Network Optimization
          </div>
          <div style="font-size:1.15rem;font-weight:700;line-height:1.25;
                       margin-bottom:10px;">
            From training engineer to deployment engineer.
          </div>
          <div style="font-size:0.92rem;color:#cbd5e1;line-height:1.55;">
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

uploaded = st.file_uploader(
    "Drop your `.pt` / `.pth` here", type=["pt", "pth"], accept_multiple_files=False
)

if uploaded is None:
    st.info(
        "Waiting for a checkpoint... Save the **full module** "
        "(`torch.save(model, 'demo.pt')`, not `state_dict()`) to unlock the "
        "live benchmarks."
    )
    st.stop()

buffer = uploaded.getvalue()
size_mb = len(buffer) / (1024 * 1024)
if size_mb > 2048:
    st.error(
        f"File is {size_mb:.0f} MB. Skip the upload and run the analyzer "
        "locally for files this large."
    )
    st.stop()

file_hash = hashlib.sha1(buffer).hexdigest()
ss = st.session_state

if ss.get("file_hash") != file_hash:
    with st.spinner("Loading & analyzing checkpoint..."):
        try:
            report, load_mode, obj = analyze(buffer)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load this checkpoint: {exc}")
            st.stop()
    # Original = the file as uploaded (never mutated).  Working = the model
    # after any optimize-button presses.  We re-render against `working_*`.
    ss["file_hash"] = file_hash
    ss["report"] = report
    ss["original_report"] = report
    ss["load_mode"] = load_mode
    ss["obj"] = obj
    ss["original_obj"] = obj
    ss["applied_actions"] = []          # list of action names applied
    ss["last_snippet"] = None           # most recent code snippet to surface

report = ss["report"]
load_mode = ss["load_mode"]
original_report = ss["original_report"]


def _apply_optimization(action: str) -> None:
    """Run a one-click optimizer, re-analyze the result, persist to session
    state, and trigger a re-render so the cards update in place."""
    import io
    import torch
    from optimizers import make_efficient, make_lean, make_compact, try_export_onnx

    obj = ss["obj"]
    snippet = None
    err = None
    if action == "efficient":
        new_obj, snippet = make_efficient(obj)
    elif action == "lean":
        new_obj, snippet = make_lean(obj)
    elif action == "compact":
        new_obj, snippet = make_compact(obj)
    elif action == "export":
        new_obj, snippet, err = try_export_onnx(obj)
    else:
        return

    if action != "export":
        # Re-serialize + re-analyze so all downstream visuals reflect the
        # transformed model (file size shrinks, dtype mix changes, etc.).
        bio = io.BytesIO()
        torch.save(new_obj, bio)
        new_report, _mode, new_obj_loaded = analyze(bio.getvalue())
        ss["obj"] = new_obj_loaded
        ss["report"] = new_report
        ss["applied_actions"] = ss.get("applied_actions", []) + [action]
    ss["last_snippet"] = (snippet, err)


def _reset_optimizations() -> None:
    ss["obj"] = ss["original_obj"]
    ss["report"] = ss["original_report"]
    ss["applied_actions"] = []
    ss["last_snippet"] = None


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

# A small banner above the tabs when the user has applied optimizations,
# with a Reset button to undo back to the original upload.
if ss.get("applied_actions"):
    applied = ", ".join(ss["applied_actions"])
    delta_score = report.deployment_health_score - original_report.deployment_health_score
    sign = "+" if delta_score >= 0 else ""
    bar_l, bar_r = st.columns([4, 1])
    with bar_l:
        st.success(
            f"Applied: **{applied}**. "
            f"Score moved from {original_report.deployment_health_score:.0f} "
            f"to {report.deployment_health_score:.0f} ({sign}{delta_score:.0f})."
        )
    with bar_r:
        st.button("Reset", on_click=_reset_optimizations, use_container_width=True)

tab_network, tab_opportunities, tab_deployment = st.tabs([
    "1. The network",
    "2. Opportunities",
    "3. Deployment",
])

# =====================================================================
# Tab 1 -- The network: stats, weight visual, category breakdown +
# 1-click Optimize buttons under each category card
# =====================================================================
with tab_network:
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
                             font-size:0.82rem;color:#94a3b8;margin-bottom:4px;">
                  <span>Today</span><span><b style='color:#fca5a5;'>{stats.fp32_mb:.0f} MB</b></span>
                </div>
                <div style="height:18px;border-radius:6px;
                             background:rgba(239,68,68,0.20);
                             border:1px solid rgba(239,68,68,0.45);
                             margin-bottom:18px;"></div>
                <div style="display:flex;justify-content:space-between;
                             font-size:0.82rem;color:#94a3b8;margin-bottom:4px;">
                  <span>Deploy-grade</span><span><b style='color:#86efac;'>{slim_mb:.0f} MB</b></span>
                </div>
                <div style="height:18px;border-radius:6px;
                             background:rgba(148,163,184,0.12);position:relative;">
                  <div style="position:absolute;left:0;top:0;bottom:0;
                               width:{slim_pct}%;border-radius:6px;
                               background:rgba(22,163,74,0.40);
                               border:1px solid rgba(22,163,74,0.55);"></div>
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
                  <div style="font-size:2rem;font-weight:800;color:#38bdf8;
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

    st.markdown("### Category breakdown")
    st.caption(
        "Click **Optimize** under any card to apply the matching action. The "
        "score updates in place and the actual code we ran appears below."
    )

    # Map each category to its underlying score, current report, original report
    # (for delta), and the one-click action that fixes it.
    category_specs = [
        ("Efficiency",   report.precision,    original_report.precision,    "efficient"),
        ("Leanness",     report.pruning,      original_report.pruning,      "lean"),
        ("Compactness",  report.size,         original_report.size,         "compact"),
        ("Exportability", report.exportability, original_report.exportability, "export"),
    ]
    cols = st.columns(4)
    button_labels = {
        "efficient": "Optimize",
        "lean":      "Optimize",
        "compact":   "Optimize",
        "export":    "Try export",
    }
    for col, (label, cur, orig, action) in zip(cols, category_specs):
        with col:
            delta = cur.score - orig.score if ss.get("applied_actions") else None
            _score_card(label, cur.score, cur.verdict, delta=delta)
            st.button(
                button_labels[action],
                key=f"opt_{action}",
                on_click=_apply_optimization,
                args=(action,),
                use_container_width=True,
                help=(
                    "Try a real ONNX export on this model."
                    if action == "export"
                    else "Apply the action and see the score move in place."
                ),
            )

    # Surface the most recent code snippet just below the category cards so
    # it sits in eyeshot of the button that produced it.
    last = ss.get("last_snippet")
    if last:
        snippet, err = last
        with st.expander(f"What just happened: {snippet.title}", expanded=True):
            st.markdown(f"_{snippet.summary}_")
            if err:
                st.error(f"Result: {err}")
            else:
                st.success("Result: applied successfully.")
            st.code(snippet.code, language="python")
            st.caption(
                f"Lifted from `{snippet.notebook}` in the course repo. "
                "The notebook walks through *why* it works, not just what."
            )

# =====================================================================
# Tab 2 -- Opportunities: hero recommendations + quantified callouts
# =====================================================================
with tab_opportunities:
    st.markdown("### Weaknesses & opportunities")
    st.caption(
        "What's wrong with this model -- in plain language. No fixes yet, "
        "just the gap."
    )
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
    verdict_color = {
        "good": ("#16a34a", "rgba(22,163,74,0.10)", "rgba(22,163,74,0.40)"),
        "warning": ("#f59e0b", "rgba(245,158,11,0.10)", "rgba(245,158,11,0.40)"),
        "critical": ("#ef4444", "rgba(239,68,68,0.10)", "rgba(239,68,68,0.40)"),
    }
    scen_cols = st.columns(2)
    for i, scen in enumerate(scenarios):
        accent, bg, border = verdict_color.get(scen.severity, verdict_color["warning"])
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
                        <div style="font-size:0.85rem;color:#94a3b8;
                                     margin-top:2px;">
                          {scen.target}
                        </div>
                      </div>
                    </div>
                    <div style="font-size:0.72rem;letter-spacing:.12em;
                                 font-weight:800;color:{accent};
                                 background:{bg};
                                 border:1px solid {border};
                                 padding:4px 11px;border-radius:999px;
                                 flex-shrink:0;">
                      {scen.verdict.upper()}
                    </div>
                  </div>
                  <div style="font-size:0.98rem;font-style:italic;color:#e2e8f0;
                               margin-bottom:12px;line-height:1.45;">
                    "{scen.hook}"
                  </div>
                  <div style="font-size:0.92rem;color:#cbd5e1;line-height:1.55;
                               margin-bottom:8px;">
                    <b style="color:#fca5a5;">Today:</b> {scen.today_line}
                  </div>
                  <div style="font-size:0.92rem;color:#cbd5e1;line-height:1.55;">
                    <b style="color:#86efac;">Optimized:</b> {scen.optimized_line}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.markdown(
    "Want to close every gap above? The Neural Network Optimization course "
    "trains the engineer who ships -- not just the engineer who trains."
)
st.link_button("Enroll in the course", COURSE_URL, type="primary")
