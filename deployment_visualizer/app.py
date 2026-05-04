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

from analyzer import analyze, attach_benchmarks
import viz


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
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color};">
            {score:.0f}<span style="font-size:1rem;color:#64748b;"> / 100</span>
          </div>
          <div class="metric-sub">{verdict}</div>
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
    st.header("How the score works")
    st.markdown(
        "- **Precision (30%)** -- FP32 weights drag the score down; FP16 / "
        "INT8 push it up.\n"
        "- **Pruning (20%)** -- Realised sparsity + near-zero headroom.\n"
        "- **Size (20%)** -- Smaller artifacts deploy more easily.\n"
        "- **Exportability (30%)** -- Heuristic, upgraded with a real ONNX "
        "export attempt when possible.\n\n"
        "All analysis runs locally in this Streamlit process."
    )
    st.divider()
    st.caption(
        "Live benchmarks (latency / FLOPs / ONNX / quant & prune sims) run on "
        "demand from the main panel. They take 5-30 seconds depending on "
        "model size."
    )
    st.divider()
    st.caption("Companion tool to the Neural Network Optimization course.")

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

    st.markdown("**What to do next**")
    for rec in report.recommendations:
        st.markdown(f"- {rec}")

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

if not ss.get("benchmarks_done") and report.dynamic is None:
    st.markdown("### Live benchmarks")
    is_module = report.raw.get("is_module", False)
    if is_module:
        button_label = "Run live benchmarks"
        help_text = (
            "Runs a real forward pass, attempts an ONNX export, and simulates "
            "INT8 quantization + magnitude pruning. Takes ~5s on small models, "
            "up to a minute on large ones."
        )
    else:
        button_label = "Run available benchmarks"
        help_text = (
            "Checkpoint is a state-dict only -- we'll fingerprint the "
            "architecture but can't run a forward pass without the module."
        )
    if st.button(button_label, type="primary", help=help_text):
        with st.spinner("Benchmarking (forward pass, ONNX export, quant/prune sims)..."):
            ss["report"] = attach_benchmarks(ss["report"], ss["obj"])
            ss["benchmarks_done"] = True
        st.rerun()

dyn = ss["report"].dynamic
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

st.markdown("### Inside your model")
st.caption(
    "Where the weight (and the cost) actually lives. Hover, zoom, click into "
    "branches."
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
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>%{value:,} params<br>%{percentRoot:.1%} of model<extra></extra>",
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
    st.markdown("#### Per-layer parameter cost")
    layers = viz.per_layer_costs(obj)
    if layers:
        df = pd.DataFrame(layers).head(15)
        df["short_name"] = df["name"].apply(
            lambda s: s if len(s) <= 30 else "..." + s[-27:]
        )
        fig = px.bar(
            df,
            x="params", y="short_name", color="type",
            orientation="h",
            hover_data={"name": True, "short_name": False, "params": ":,", "type": True},
            labels={"params": "Parameters", "short_name": "", "type": "Module"},
        )
        fig.update_layout(
            height=420, margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No layer-level parameter data found in this checkpoint.")

with viz_right:
    st.markdown("#### Weight magnitude distributions")
    hists = viz.weight_histograms(obj)
    if hists:
        for h in hists:
            edges = h["bin_edges"]
            counts = h["counts"]
            centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
            short = h["name"] if len(h["name"]) <= 40 else "..." + h["name"][-37:]
            fig = go.Figure(go.Bar(
                x=centers, y=counts,
                marker=dict(color="#38bdf8"),
                hovertemplate="|w|=%{x:.4f}<br>count=%{y}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(
                    text=f"{short}  ·  shape={tuple(h['shape'])}",
                    font=dict(size=12),
                ),
                height=180,
                margin=dict(t=30, b=20, l=10, r=10),
                xaxis_title="|weight|",
                yaxis_title=None,
                bargap=0.0,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.04)",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Heavy mass near zero = pruning headroom. A long tail to the "
            "right = a few very large weights driving the layer."
        )
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
