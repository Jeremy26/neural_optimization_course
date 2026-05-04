"""Streamlit lead-magnet: Deployment Health Score for PyTorch models.

Run with:

    streamlit run deployment_visualizer/app.py
"""

from __future__ import annotations

import hashlib
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyzer import analyze, attach_benchmarks
from problems import detect_problems
from scenarios import build_scenarios
import viz


# Pro features (live benchmarks, what-if simulations, action plans).
# Default is unlocked for admin/dev use; set ``DEPLOYMENT_VIZ_FREE=1`` to
# preview what the free tier looks like before we split into two files.
PRO = os.getenv("DEPLOYMENT_VIZ_FREE", "0") != "1"
COURSE_URL = "https://www.thinkautonomous.ai/"


def _locked_card(title: str, body: str, preview_bullets: list[str]) -> None:
    bullets_html = "".join(
        f"<li style='margin:4px 0;color:#cbd5e1;'>{b}</li>"
        for b in preview_bullets
    )
    st.markdown(
        f"""
        <div style="position:relative;border:1px dashed rgba(56,189,248,0.45);
                     border-radius:14px;padding:22px 22px 18px;
                     background:linear-gradient(160deg, rgba(56,189,248,0.08),
                                                rgba(15,23,42,0.0));
                     margin-bottom:8px;">
          <div style="position:absolute;top:14px;right:18px;
                       background:rgba(56,189,248,0.18);color:#38bdf8;
                       padding:3px 10px;border-radius:999px;
                       font-size:0.72rem;font-weight:700;letter-spacing:.1em;">
            PRO
          </div>
          <div style="font-size:1.15rem;font-weight:700;margin-bottom:4px;">
            {title}
          </div>
          <div style="font-size:0.92rem;color:#cbd5e1;margin-bottom:10px;">
            {body}
          </div>
          <ul style="margin:0 0 14px 18px;padding:0;font-size:0.9rem;">
            {bullets_html}
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button(
        "Unlock with the course",
        COURSE_URL,
        type="primary",
        use_container_width=False,
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


def _score_card(label: str, score: float, verdict: str) -> None:
    color = _score_color(score)
    # Position bar: where the score sits, with a "pro range" band marked.
    score_pct = max(0.0, min(100.0, score))
    pro_min, pro_max = 75, 100  # what well-optimized models score
    st.markdown(
        f"""
        <div class="metric-card score-card">
          <div class="metric-label">{label}</div>
          <div style="display:flex;align-items:baseline;gap:6px;
                       margin:8px 0 12px;">
            <div style="font-size:3rem;font-weight:800;color:{color};
                         line-height:1;">{score:.0f}</div>
            <div style="font-size:1rem;color:#64748b;">/ 100</div>
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
    "We inspect precision, pruning headroom, size, and ONNX / TensorRT "
    "compatibility -- then run **live benchmarks** (latency, FLOPs, real ONNX "
    "export, quantization & pruning simulations) to tell you exactly what to "
    "fix."
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
            Stop shipping FP32 ResNets in 2026.
          </div>
          <div style="font-size:0.92rem;color:#cbd5e1;line-height:1.55;">
            This audit shows you <i>what's wrong</i>. The course shows you
            <b>how to fix it</b> -- quantization, pruning, distillation, and
            ONNX / TensorRT deployment, end to end, on real models.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        "**You'll learn to:**\n"
        "- Cut model size 4x without touching accuracy\n"
        "- Hit 2-3x lower latency on the same hardware\n"
        "- Ship to mobile, edge, and serverless without surprises\n"
        "- Debug failed ONNX / TensorRT exports like a pro"
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
            report, load_mode, obj = analyze(buffer, run_benchmarks=False)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load this checkpoint: {exc}")
            st.stop()
    ss["file_hash"] = file_hash
    ss["report"] = report
    ss["load_mode"] = load_mode
    ss["obj"] = obj
    ss["benchmarks_done"] = False

report = ss["report"]
load_mode = ss["load_mode"]

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
    arch = report.dynamic.architecture_guess if report.dynamic else None
    badges = [
        f"<span class='arch-badge'>{report.parameter_count / 1e6:.2f}M params</span>",
        f"<span class='arch-badge'>{report.file_size_mb:.1f} MB on disk</span>",
    ]
    if arch:
        badges.insert(0, f"<span class='arch-badge'>Looks like: {arch}</span>")
    if report.dynamic and report.dynamic.flops:
        badges.append(
            f"<span class='arch-badge'>{_format_flops(report.dynamic.flops)}</span>"
        )
    st.markdown(" ".join(badges), unsafe_allow_html=True)

    st.markdown("**Weaknesses & Opportunities**")
    for rec in report.recommendations:
        st.markdown(f"- {rec}")
    st.caption(
        "Spotted the gap? The course teaches you how to close it -- see "
        "the panel on the left."
    )

# ---------------------------------------------------------------------------
# Network at a glance -- the by-the-numbers panel
# ---------------------------------------------------------------------------

from device_estimates import network_stats, estimate_latency_ms  # noqa: E402

stats = network_stats(ss["obj"])

st.markdown("### Network at a glance")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Parameters", f"{stats.parameter_count / 1e6:.2f}M")
m2.metric("Weight tensors", f"{stats.weight_tensors}")
m3.metric("Leaf modules", f"{stats.leaf_modules}")
m4.metric("FP32 footprint", f"{stats.fp32_mb:.1f} MB")
m5.metric(
    "→ FP16",
    f"{stats.fp16_mb:.1f} MB",
    delta=f"-{stats.fp32_mb - stats.fp16_mb:.1f} MB",
    delta_color="inverse",
)
m6.metric(
    "→ INT8",
    f"{stats.int8_mb:.1f} MB",
    delta=f"-{stats.fp32_mb - stats.int8_mb:.1f} MB",
    delta_color="inverse",
)

# Layer-kind composition: what fraction of params lives in Conv vs Linear vs Norm?
kind_left, kind_right = st.columns([2, 1])
with kind_left:
    if stats.layer_kind_share:
        kdf = pd.DataFrame(
            [(k, v) for k, v in stats.layer_kind_share.items()],
            columns=["kind", "params"],
        ).sort_values("params", ascending=True)
        kdf["share"] = kdf["params"] / max(kdf["params"].sum(), 1)
        fig = px.bar(
            kdf, y="kind", x="params", orientation="h",
            text=kdf["share"].map(lambda x: f"{x:.0%}"),
            color="kind",
            labels={"params": "Parameters", "kind": ""},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=210, margin=dict(t=10, b=10, l=10, r=40),
            showlegend=False,
            title=dict(text="Where the weight lives by layer kind", font=dict(size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)

with kind_right:
    if stats.biggest_layer:
        bn, bc = stats.biggest_layer
        short = bn if len(bn) <= 36 else "..." + bn[-33:]
        share = bc / max(stats.parameter_count, 1)
        st.markdown(
            f"""
            <div class="metric-card" style="height:210px;display:flex;
                                              flex-direction:column;
                                              justify-content:center;">
              <div class="metric-label">Heaviest single layer</div>
              <div style="font-size:1.05rem;font-weight:700;
                           margin:6px 0 4px;
                           font-family:ui-monospace,monospace;
                           word-break:break-all;">
                {short}
              </div>
              <div class="metric-sub">
                <b>{bc / 1e6:.2f}M params</b> &nbsp;·&nbsp; {share:.1%} of model
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Device latency estimate -- compute statically on FLOPs (when available).
# ---------------------------------------------------------------------------

flops = ss["report"].dynamic.flops if (ss["report"].dynamic and ss["report"].dynamic.flops) else None
if flops:
    st.markdown("#### Estimated single-pass latency on common targets")
    dtype_choice = st.radio(
        "Inference precision",
        options=["fp32", "fp16", "int8"],
        format_func=lambda d: d.upper(),
        index=1,
        horizontal=True,
        key="latency_dtype",
    )
    rows = estimate_latency_ms(flops, dtype=dtype_choice)
    df = pd.DataFrame(rows)
    df["latency"] = df["latency_ms"].map(
        lambda x: f"{x:.2f} ms" if x >= 0.5 else f"{x*1000:.0f} µs"
    )
    df["throughput"] = df["throughput_per_s"].map(
        lambda x: f"{x:,.0f} /s" if x < 1e5 else f"{x/1000:,.0f}k /s"
    )
    df["peak"] = df.apply(
        lambda r: f"{r['peak_tflops']:.0f} {'TOPS' if dtype_choice == 'int8' else 'TFLOPS'}",
        axis=1,
    )
    df["efficiency"] = df["efficiency"].map(lambda x: f"{x:.0%}")
    st.dataframe(
        df[["device", "peak", "efficiency", "latency", "throughput"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "device": "Target",
            "peak": "Peak",
            "efficiency": "Realistic eff.",
            "latency": "Per forward",
            "throughput": "Throughput",
        },
    )
    st.caption(
        f"Theoretical estimates from {flops/1e9:.2f} GFLOPs at "
        f"{dtype_choice.upper()}. Real batch=1 latency is typically 2-5x "
        "slower due to kernel-launch overhead; the course covers how to "
        "close that gap."
    )
else:
    st.info(
        "Run the live benchmarks below to enable device latency estimates "
        "(needs FLOP count from a real forward pass)."
    )

# ---------------------------------------------------------------------------
# If this shipped today -- deployment narratives
# ---------------------------------------------------------------------------

st.markdown("### If this shipped today...")
st.caption(
    "Concrete deployment scenarios for your model, as-is. Numbers are "
    "estimates -- but the order of magnitude is what teams actually live with."
)

scenarios = build_scenarios(report)
verdict_color = {
    "good": ("#16a34a", "rgba(22,163,74,0.10)", "rgba(22,163,74,0.40)"),
    "warning": ("#f59e0b", "rgba(245,158,11,0.10)", "rgba(245,158,11,0.40)"),
    "critical": ("#ef4444", "rgba(239,68,68,0.10)", "rgba(239,68,68,0.40)"),
}

# Two columns of scenario cards.
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
                           align-items:center;margin-bottom:8px;">
                <div style="font-size:0.72rem;letter-spacing:.14em;
                             color:{accent};font-weight:800;">
                  {scen.setting.upper()}
                </div>
                <div style="font-size:0.72rem;letter-spacing:.12em;
                             font-weight:800;color:{accent};
                             background:{bg};
                             border:1px solid {border};
                             padding:3px 10px;border-radius:999px;">
                  {scen.verdict.upper()}
                </div>
              </div>
              <div style="font-size:0.85rem;color:#94a3b8;
                           margin-bottom:10px;">
                {scen.target}
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

if not any("FPS" in s.target or "req" in s.target for s in scenarios):
    st.caption(
        "Run live benchmarks below to unlock the latency-based scenarios "
        "(robot, phone AR, cloud throughput)."
    )

# ---------------------------------------------------------------------------
# Problems detected -- damning quantified callouts
# ---------------------------------------------------------------------------

st.markdown("### Problems detected")
problems = detect_problems(report)
for p in problems:
    _problem_callout(p)

# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------

st.markdown("### Category breakdown")
cols = st.columns(4)
with cols[0]:
    _score_card("Precision", report.precision.score, report.precision.verdict)
with cols[1]:
    _score_card("Pruning", report.pruning.score, report.pruning.verdict)
with cols[2]:
    _score_card("Size", report.size.score, report.size.verdict)
with cols[3]:
    _score_card(
        "Exportability", report.exportability.score, report.exportability.verdict
    )

# ---------------------------------------------------------------------------
# Live benchmarks (opt-in -- they're slow on big models)
# ---------------------------------------------------------------------------

if not PRO:
    st.markdown("### Live benchmarks & what-if simulations")
    _locked_card(
        title="See exactly what optimization will buy you -- before you do it.",
        body=(
            "We run your model under realistic conditions and simulate the "
            "transforms the course teaches, so you know which moves are "
            "worth your time."
        ),
        preview_bullets=[
            "Real CPU latency (mean + p95) and peak activation memory",
            "FLOP count + per-layer cost breakdown",
            "Actual ONNX export attempt -- pass / fail with the real error",
            "INT8 quantization simulation: projected size + output drift",
            "Magnitude pruning at 30 / 50 / 70%: projected savings + drift",
        ],
    )

if PRO and not ss.get("benchmarks_done") and report.dynamic is None:
    st.markdown("### Live benchmarks")
    is_module = report.raw.get("is_module", False)
    if is_module:
        if st.button(
            "Run live benchmarks", type="primary",
            help=(
                "Runs a real forward pass, attempts ONNX export, and simulates "
                "INT8 quantization + magnitude pruning. Takes ~5s on small "
                "models, up to a minute on large ones."
            ),
        ):
            with st.spinner("Benchmarking..."):
                ss["report"] = attach_benchmarks(ss["report"], ss["obj"])
                ss["benchmarks_done"] = True
            st.rerun()
    else:
        # State-dict path: let the user pick a torchvision architecture so we
        # can instantiate it, load the weights in, and benchmark for real.
        from benchmarks import available_architectures, try_load_into_arch

        archs = available_architectures()
        if not archs:
            st.info(
                "Live benchmarks need either a full ``nn.Module`` or "
                "``torchvision`` installed (it isn't). Install torchvision "
                "and re-run, or save with ``torch.save(model, ...)``."
            )
        else:
            # Pre-select from the fingerprint where possible.
            fingerprint = (
                report.dynamic.architecture_guess if report.dynamic else None
            )
            default_idx = 0
            for i, name in enumerate(archs):
                if fingerprint and name.startswith(fingerprint):
                    default_idx = i
                    break
            st.markdown(
                "This file is a state-dict only. Pick the matching "
                "architecture and we'll load the weights into a fresh "
                "torchvision model so we can benchmark it for real:"
            )
            arch_choice = st.selectbox(
                "Architecture", archs, index=default_idx,
                label_visibility="collapsed",
            )
            if st.button("Load & run live benchmarks", type="primary"):
                with st.spinner(f"Loading weights into {arch_choice}..."):
                    module, err = try_load_into_arch(ss["obj"], arch_choice)
                if err:
                    st.error(err)
                else:
                    with st.spinner("Benchmarking..."):
                        ss["obj"] = module
                        ss["report"] = attach_benchmarks(ss["report"], module)
                        ss["benchmarks_done"] = True
                    st.rerun()

dyn = ss["report"].dynamic if PRO else None
report = ss["report"]
if dyn is not None:
    st.markdown("### Live benchmarks")
    for note in dyn.notes:
        st.info(note)
    if dyn.input_shape is not None:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric(
            "Latency (mean)",
            f"{dyn.latency_ms_mean:.2f} ms" if dyn.latency_ms_mean else "n/a",
        )
        b2.metric(
            "Latency (p95)",
            f"{dyn.latency_ms_p95:.2f} ms" if dyn.latency_ms_p95 else "n/a",
        )
        b3.metric(
            "Peak activations",
            f"{dyn.peak_activation_mb:.1f} MB" if dyn.peak_activation_mb else "n/a",
        )
        b4.metric(
            "FLOPs / forward",
            _format_flops(dyn.flops) if dyn.flops else "n/a",
        )
        st.caption(
            f"Run on CPU with dummy input shape {tuple(dyn.input_shape)}. "
            "Latency varies with hardware -- relative comparisons are what matter."
        )

    # ONNX export attempt
    if dyn.onnx_export is not None:
        if dyn.onnx_export["ok"]:
            st.success(
                f"Real ONNX export succeeded -- {dyn.onnx_export['size_mb']:.1f} MB "
                f"at opset {dyn.onnx_export['opset']}."
            )
        else:
            st.error(f"ONNX export failed: {dyn.onnx_export['error']}")

# ---------------------------------------------------------------------------
# What-if simulations
# ---------------------------------------------------------------------------

if dyn is not None and (dyn.quantization_sim or dyn.pruning_sim):
    st.markdown("### What if you optimized this?")
    st.caption(
        "Simulations applied to a copy of your model -- nothing modifies the "
        "uploaded file. Output drift is measured against the unmodified model "
        "on the dummy input."
    )

    sim_left, sim_right = st.columns(2)

    with sim_left:
        st.markdown("**Dynamic INT8 quantization**")
        q = dyn.quantization_sim
        if q:
            qa, qb = st.columns(2)
            qa.metric("New size", f"{q['size_mb']:.1f} MB",
                      delta=f"-{q['size_reduction_pct']:.1f}%")
            qb.metric(
                "Output drift",
                f"{q['output_drift']:.3f}" if q['output_drift'] is not None else "n/a",
                help="0 = identical to original. >0.1 may need calibration.",
            )
            st.caption(
                f"Quantizes {', '.join(q['supported_layers'])}. "
                "Conv layers need static / QAT pipelines for full INT8 savings."
            )
        else:
            st.caption(
                "Dynamic quantization didn't run -- this model has no "
                "Linear/LSTM/GRU layers, or quantization tripped on a custom op."
            )

    with sim_right:
        st.markdown("**Magnitude pruning**")
        if dyn.pruning_sim:
            df = pd.DataFrame(dyn.pruning_sim)
            df["ratio"] = df["ratio"].map(lambda r: f"{int(r*100)}%")
            df = df.rename(columns={
                "ratio": "Prune ratio",
                "theoretical_size_mb": "Size (sparse, MB)",
                "output_drift": "Output drift",
            })
            df["Output drift"] = df["Output drift"].map(
                lambda x: f"{x:.3f}" if x is not None else "n/a"
            )
            df["Size (sparse, MB)"] = df["Size (sparse, MB)"].map(lambda x: f"{x:.1f}")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                "Sizes assume sparse storage or a quant+prune export. Drift "
                "values <0.1 usually recover with a short fine-tune."
            )
        else:
            st.caption("Pruning simulation skipped (no Linear/Conv layers).")

# ---------------------------------------------------------------------------
# Inside your model -- visualizations
# ---------------------------------------------------------------------------

st.markdown("### Where you're leaking performance")
st.caption(
    "Each rectangle is a parameter tensor sized by its share of total weight. "
    "Big boxes are big targets -- the layers you'd hit first if you were "
    "serious about shipping this model."
)

obj = ss.get("obj")

# 1. Architecture treemap -- one glance: which modules eat your parameter budget?
tree = viz.module_tree(obj)
# Plotly renders nothing if the only node is the synthetic root with 0 params.
has_real_nodes = any(r["params"] > 0 and r["id"] != "model" for r in tree)
if tree and has_real_nodes:
    df = pd.DataFrame(tree)
    fig = go.Figure(go.Treemap(
        ids=df["id"],
        labels=df["label"],
        parents=df["parent"],
        values=df["params"],
        # "remainder" lets parents have value 0 while showing children inside,
        # which is what we want -- intermediate paths like "net.layer1" carry
        # no params of their own, only the leaves do.
        branchvalues="remainder",
        hovertemplate="<b>%{label}</b><br>%{value:,} params<extra></extra>",
        marker=dict(
            colors=df["params"],
            colorscale="Tealgrn",
            showscale=False,
            line=dict(width=1, color="rgba(15,23,42,0.4)"),
        ),
        textfont=dict(size=13),
    ))
    fig.update_layout(
        height=460,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "Treemap unavailable -- couldn't find tensor data at the top level "
        "of this checkpoint. If your file is a wrapper like "
        "``{'state_dict': ..., 'optimizer': ...}`` we usually unwrap it "
        "automatically; let us know if your structure is different."
    )

# 2. Two-column visual breakdown
viz_left, viz_right = st.columns(2)

with viz_left:
    st.markdown("#### Where your parameters live")
    layers = viz.per_layer_costs(obj)
    if layers:
        total_params = sum(l["params"] for l in layers)
        top_layers = layers[:15]
        top3_share = sum(l["params"] for l in layers[:3]) / max(total_params, 1)
        # Find the "fat" prefix -- e.g. all of net.layer4.* if that branch
        # dominates the param count.  Group by 3-deep prefix.
        from collections import defaultdict as _dd
        prefix_share = _dd(int)
        for l in layers:
            parts = l["name"].split(".")
            prefix = ".".join(parts[: min(3, len(parts) - 1)]) if len(parts) > 1 else l["name"]
            prefix_share[prefix] += l["params"]
        biggest_prefix, biggest_share = max(
            prefix_share.items(), key=lambda kv: kv[1]
        )
        biggest_pct = biggest_share / max(total_params, 1)
        st.markdown(
            f"**Top 3 layers = {top3_share:.0%} of all parameters.** "
            f"Branch ``{biggest_prefix}.*`` alone holds {biggest_pct:.0%} -- "
            "the highest-impact pruning target."
        )

        df = pd.DataFrame(top_layers)
        df["share"] = df["params"] / max(total_params, 1)
        df["short_name"] = df["name"].apply(
            lambda s: s if len(s) <= 32 else "..." + s[-29:]
        )
        fig = px.bar(
            df,
            x="params", y="short_name", color="type",
            orientation="h",
            text=df["share"].map(lambda x: f"{x:.1%}"),
            hover_data={
                "name": True, "short_name": False,
                "params": ":,", "type": True, "share": ":.1%",
            },
            labels={"params": "Parameters", "short_name": "", "type": "Module"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=460, margin=dict(t=10, b=40, l=10, r=40),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No layer-level parameter data found in this checkpoint.")

with viz_right:
    st.markdown("#### Weight distributions & prunability")
    hists = viz.weight_histograms(obj)
    if hists:
        st.caption(
            "For each layer: the share of weights below 1% of the layer's "
            "max magnitude (the dashed line). That's roughly how much you "
            "can prune with magnitude pruning before retraining."
        )
        for h in hists:
            edges = h["bin_edges"]
            counts = h["counts"]
            centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
            threshold = h["max_abs"] * 0.01
            below = sum(c for c, e in zip(counts, edges[:-1]) if e < threshold)
            below_pct = below / max(sum(counts), 1)
            short = h["name"] if len(h["name"]) <= 40 else "..." + h["name"][-37:]
            fig = go.Figure(go.Bar(
                x=centers, y=counts,
                marker=dict(color="#38bdf8"),
                hovertemplate="|w|=%{x:.4f}<br>count=%{y:,}<extra></extra>",
            ))
            fig.add_vline(
                x=threshold, line_dash="dash", line_color="#f59e0b",
                annotation_text=f"{below_pct:.0%} below",
                annotation_position="top right",
                annotation_font_color="#f59e0b",
            )
            fig.update_layout(
                title=dict(
                    text=(
                        f"{short}  ·  shape={tuple(h['shape'])}  ·  "
                        f"<span style='color:#f59e0b'>"
                        f"{below_pct:.0%} prunable</span>"
                    ),
                    font=dict(size=12),
                ),
                height=200,
                margin=dict(t=36, b=24, l=10, r=10),
                xaxis_title="|weight|",
                yaxis_title=None,
                bargap=0.0,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.04)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 2D+ weight tensors to plot.")

# 3. Compact module-class composition + dtype mix as side-by-side donuts.
mix_left, mix_right = st.columns(2)

with mix_left:
    classes = report.exportability.details.get("module_classes", {})
    if classes:
        st.markdown("#### Module class mix")
        cdf = pd.DataFrame(
            [(k, v) for k, v in classes.items()], columns=["class", "count"]
        )
        fig = px.pie(
            cdf, values="count", names="class", hole=0.55,
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10),
                          showlegend=True,
                          legend=dict(orientation="v", x=1.02, y=0.5))
        st.plotly_chart(fig, use_container_width=True)

with mix_right:
    dist = report.precision.details.get("dtype_distribution", {})
    if dist:
        st.markdown("#### Precision mix")
        ddf = pd.DataFrame(
            [(k, v) for k, v in dist.items()], columns=["dtype", "elements"]
        )
        fig = px.pie(
            ddf, values="elements", names="dtype", hole=0.55,
            color_discrete_sequence=px.colors.sequential.Sunset,
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10),
                          showlegend=True,
                          legend=dict(orientation="v", x=1.02, y=0.5))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### Action plan")
if not PRO:
    _locked_card(
        title="Tailored, copy-paste code recipes for each opportunity above.",
        body=(
            "The course pairs every gap we just spotted with a working "
            "PyTorch / ONNX / TensorRT recipe -- and an explanation of when "
            "to reach for each."
        ),
        preview_bullets=[
            "Quantization recipes: dynamic INT8, static INT8, QAT",
            "Pruning recipes: magnitude, structured, movement -- with "
            "fine-tune schedules",
            "Distillation: building a student that keeps the accuracy",
            "ONNX & TensorRT: opset choice, dynamic axes, plugin authoring",
            "Hardware-aware tuning for CPU, GPU, mobile, edge",
        ],
    )
else:
    # Pro tier: actually show the recipes.
    for rec in report.recommendations:
        st.markdown(f"- {rec}")
    st.code(
        "# Quantize to FP16\n"
        "model.half()\n"
        "torch.save(model, 'model_fp16.pt')\n\n"
        "# Magnitude pruning at 50%\n"
        "import torch.nn.utils.prune as prune\n"
        "for m in model.modules():\n"
        "    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):\n"
        "        prune.l1_unstructured(m, name='weight', amount=0.5)\n"
        "        prune.remove(m, 'weight')\n\n"
        "# ONNX export\n"
        "torch.onnx.export(model, dummy, 'model.onnx', opset_version=17)\n",
        language="python",
    )

# Power-user JSON dump, hidden by default.
with st.expander("Raw analysis JSON"):
    st.json(
        {
            "file_size_mb": report.file_size_mb,
            "parameter_count": report.parameter_count,
            "deployment_health_score": report.deployment_health_score,
            "precision": {
                "score": report.precision.score,
                "verdict": report.precision.verdict,
                "details": report.precision.details,
            },
            "pruning": {
                "score": report.pruning.score,
                "verdict": report.pruning.verdict,
                "details": report.pruning.details,
            },
            "size": {
                "score": report.size.score,
                "verdict": report.size.verdict,
                "details": report.size.details,
            },
            "exportability": {
                "score": report.exportability.score,
                "verdict": report.exportability.verdict,
                "details": report.exportability.details,
            },
            "dynamic": {
                "input_shape": list(report.dynamic.input_shape)
                if (report.dynamic and report.dynamic.input_shape) else None,
                "latency_ms_mean": report.dynamic.latency_ms_mean if report.dynamic else None,
                "latency_ms_p95": report.dynamic.latency_ms_p95 if report.dynamic else None,
                "peak_activation_mb": report.dynamic.peak_activation_mb if report.dynamic else None,
                "flops": report.dynamic.flops if report.dynamic else None,
                "onnx_export": report.dynamic.onnx_export if report.dynamic else None,
                "quantization_sim": report.dynamic.quantization_sim if report.dynamic else None,
                "pruning_sim": report.dynamic.pruning_sim if report.dynamic else [],
                "architecture_guess": report.dynamic.architecture_guess if report.dynamic else None,
            },
        }
    )

st.divider()
st.markdown(
    "**Want to take this further?** The full Neural Network Optimization "
    "course covers quantization, pruning, distillation, and ONNX / TensorRT "
    "deployment end-to-end. The notebooks in this repo are a good starting "
    "point."
)
