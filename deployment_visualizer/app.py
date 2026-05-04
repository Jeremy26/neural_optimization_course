"""Streamlit lead-magnet: Deployment Health Score for PyTorch models.

Run with:

    streamlit run deployment_visualizer/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analyzer import analyze


st.set_page_config(
    page_title="Model Deployment Health Score",
    page_icon=None,
    layout="wide",
)


def _score_color(score: float) -> str:
    if score >= 80:
        return "#16a34a"  # green
    if score >= 60:
        return "#65a30d"  # lime
    if score >= 40:
        return "#d97706"  # amber
    return "#dc2626"  # red


def _score_card(label: str, score: float, verdict: str) -> None:
    color = _score_color(score)
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:16px;">
          <div style="font-size:0.85rem;color:#6b7280;
                       text-transform:uppercase;letter-spacing:.05em;">
            {label}
          </div>
          <div style="font-size:2.2rem;font-weight:700;color:{color};
                       line-height:1.1;margin:6px 0;">
            {score:.0f}<span style="font-size:1rem;color:#9ca3af;"> / 100</span>
          </div>
          <div style="font-size:0.95rem;color:#374151;">{verdict}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hero_score(score: float, verdict: str) -> None:
    color = _score_color(score)
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);
                     padding:32px;border-radius:16px;color:white;
                     display:flex;align-items:center;gap:32px;">
          <div style="font-size:5rem;font-weight:800;color:{color};
                       line-height:1;">
            {score:.0f}
          </div>
          <div>
            <div style="font-size:0.9rem;letter-spacing:.1em;
                         text-transform:uppercase;color:#94a3b8;">
              Deployment Health Score
            </div>
            <div style="font-size:1.5rem;font-weight:600;margin-top:4px;">
              {verdict}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Model Deployment Health Score")
st.markdown(
    "Upload a PyTorch checkpoint (`.pt` or `.pth`) and get a free audit of how "
    "deployment-ready your model is. We check **precision**, **pruning "
    "headroom**, **size**, and **export compatibility** with ONNX / TensorRT, "
    "then roll the results into a single score."
)

with st.sidebar:
    st.header("How the score works")
    st.markdown(
        "- **Precision (30%)** -- FP32 weights drag the score down; FP16 / "
        "INT8 push it up.\n"
        "- **Pruning (20%)** -- Rewards realised sparsity, flags near-zero "
        "headroom.\n"
        "- **Size (20%)** -- Smaller artifacts deploy more easily.\n"
        "- **Exportability (30%)** -- Penalises ops that ONNX / TensorRT "
        "trip on.\n\n"
        "All analysis runs locally in this Streamlit process. Nothing is "
        "uploaded to a remote server."
    )
    st.divider()
    st.caption(
        "Built as a companion tool to the Neural Network Optimization course."
    )

uploaded = st.file_uploader(
    "Drop your `.pt` / `.pth` here", type=["pt", "pth"], accept_multiple_files=False
)

if uploaded is None:
    st.info(
        "Waiting for a checkpoint... You can export one with "
        "`torch.save(model.state_dict(), 'model.pt')`."
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

with st.spinner("Analyzing checkpoint..."):
    try:
        report, load_mode = analyze(buffer)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user
        st.error(f"Could not load this checkpoint: {exc}")
        st.stop()

if load_mode == "pickle":
    st.warning(
        "This checkpoint required Python's pickle to load, which can execute "
        "arbitrary code. We loaded it because you uploaded it -- only do this "
        "with files you trust."
    )

_hero_score(report.deployment_health_score, report.overall_verdict)

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

st.markdown("### What to do next")
for rec in report.recommendations:
    st.markdown(f"- {rec}")

st.markdown("### Drill down")

precision_tab, pruning_tab, export_tab, raw_tab = st.tabs(
    ["Precision", "Pruning", "Exportability", "Raw"]
)

with precision_tab:
    dist = report.precision.details.get("dtype_distribution", {})
    if dist:
        df = pd.DataFrame(
            [(k, v) for k, v in dist.items()], columns=["dtype", "elements"]
        ).sort_values("elements", ascending=False)
        st.bar_chart(df.set_index("dtype"))
        st.dataframe(df, use_container_width=True, hide_index=True)
    bytes_in_mem = report.precision.details.get("bytes_in_memory", 0)
    st.caption(f"Total weight bytes in memory: {bytes_in_mem / 1e6:.2f} MB")

with pruning_tab:
    sparsity = report.pruning.details.get("global_sparsity", 0.0)
    headroom = report.pruning.details.get("near_zero_headroom", 0.0)
    a, b = st.columns(2)
    a.metric("Global sparsity (exact zeros)", f"{sparsity:.2%}")
    b.metric("Near-zero headroom (|w| < 1% of max)", f"{headroom:.2%}")
    layers = report.pruning.details.get("top_layers", [])
    if layers:
        st.markdown("**Layers with the most pruning headroom**")
        df = pd.DataFrame(layers)
        df["sparsity"] = df["sparsity"].map(lambda x: f"{x:.2%}")
        df["near_zero_fraction"] = df["near_zero_fraction"].map(
            lambda x: f"{x:.2%}"
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

with export_tab:
    details = report.exportability.details
    if details.get("is_state_dict"):
        st.info(
            "Checkpoint contains weights only -- we couldn't introspect the "
            "module graph. Re-run after pickling the full ``nn.Module`` for a "
            "more precise verdict."
        )
    a, b = st.columns(2)
    a.metric("ONNX-risky modules", details.get("onnx_risky_modules", 0))
    b.metric("TensorRT-risky modules", details.get("tensorrt_risky_modules", 0))
    classes = details.get("module_classes", {})
    if classes:
        st.markdown("**Module class mix**")
        df = pd.DataFrame(
            [(k, v) for k, v in classes.items()],
            columns=["class", "count"],
        ).sort_values("count", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

with raw_tab:
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
        }
    )

st.divider()
st.markdown(
    "**Want to take this further?** The full Neural Network Optimization "
    "course covers quantization, pruning, distillation, and ONNX / TensorRT "
    "deployment end-to-end. The notebooks in this repo are a good starting "
    "point."
)
