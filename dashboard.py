"""
MemoryLens Dashboard — Streamlit visualisation layer.

Run:  streamlit run dashboard.py
"""

import json
import os
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="MemoryLens", page_icon="🔭", layout="wide")

st.markdown("""
<style>
[data-testid="stMetric"] { background:#1e1e2e; border-radius:10px; padding:12px; }
</style>
""", unsafe_allow_html=True)

COLORS = {"naive": "#f38ba8", "rag": "#89b4fa", "cascading": "#a6e3a1"}
MONTHLY_QUERIES = 100_000
COST_PER_TOKEN_INR = 83 / 1_000_000  # ~$1 per 1M tokens * 83 INR/USD


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/anthropic.svg",
             width=32)
    st.title("MemoryLens")
    st.caption("LLM Memory Decay Evaluation Framework")
    st.divider()

    st.subheader("Benchmark Config")
    total_turns = st.slider("Conversation turns", 25, 200, 100, 25)
    checkpoints = st.multiselect(
        "Eval checkpoints",
        [10, 25, 50, 75, 100, 150, 200],
        default=[10, 25, 50, 75, 100],
    )
    backends = st.multiselect(
        "Memory backends",
        ["naive", "rag", "cascading"],
        default=["naive", "rag", "cascading"],
    )
    st.divider()

    col_run, col_demo = st.columns(2)
    run_btn  = col_run.button("▶ Run Live",  type="primary",   use_container_width=True)
    demo_btn = col_demo.button("📊 Demo",    use_container_width=True)

    st.divider()
    st.caption("Facts tracked (sample):")
    for label, turn in [("name @ T=0", 0), ("city @ T=1→40", 1),
                        ("occupation @ T=2", 2), ("age @ T=3→60", 3)]:
        st.markdown(f"- `{label}`")


# ─── Helpers ────────────────────────────────────────────────────────────────
def load_demo() -> Dict:
    path = os.path.join(os.path.dirname(__file__), "demo_results.json")
    with open(path) as fh:
        return json.load(fh)


def render_results(data: Dict, is_demo: bool = False) -> None:
    cps: List[int] = data["checkpoints"]
    present = [b for b in ["naive", "rag", "cascading"] if b in data]

    if is_demo:
        st.info("📊 Showing pre-computed demo results. Set GROQ_API_KEY and click ▶ Run Live for real evaluation.", icon="ℹ️")

    # ── KPI cards ──────────────────────────────────────────────────────────
    st.subheader("Summary at Final Checkpoint")
    cols = st.columns(len(present))
    for col, name in zip(cols, present):
        d = data[name]
        with col:
            st.markdown(f"#### {name.capitalize()}")
            st.metric("Recall@Final",    f"{d['recall'][-1]*100:.1f}%")
            st.metric("Avg Tokens",      f"{d['tokens'][-1]:,}")
            st.metric("Temporal Drift",  f"{d['drift'][-1]*100:.1f}%")
            st.metric("Precision@K",     f"{d['precision'][-1]*100:.1f}%")

    st.divider()

    # ── Recall decay ───────────────────────────────────────────────────────
    st.subheader("Memory Recall Decay Over Time")
    fig = go.Figure()
    for name in present:
        fig.add_trace(go.Scatter(
            x=cps, y=[v * 100 for v in data[name]["recall"]],
            name=name.capitalize(), mode="lines+markers",
            line=dict(color=COLORS[name], width=3), marker=dict(size=9),
        ))
    fig.update_layout(
        xaxis_title="Conversation Turn", yaxis_title="Recall (%)",
        yaxis=dict(range=[0, 105]), template="plotly_dark",
        height=360, legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Drift + Noise ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Temporal Drift Score")
        fig2 = go.Figure()
        for name in present:
            fig2.add_trace(go.Scatter(
                x=cps, y=[v * 100 for v in data[name]["drift"]],
                name=name.capitalize(), mode="lines+markers",
                fill="tozeroy",
                fillcolor=COLORS[name] + "33",
                line=dict(color=COLORS[name], width=2),
            ))
        fig2.update_layout(
            xaxis_title="Turn", yaxis_title="Drift (%)",
            yaxis=dict(range=[0, 80]), template="plotly_dark",
            height=310, showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Memory Noise Ratio")
        fig3 = go.Figure()
        for name in present:
            fig3.add_trace(go.Bar(
                x=[f"T={c}" for c in cps],
                y=[v * 100 for v in data[name]["noise"]],
                name=name.capitalize(), marker_color=COLORS[name],
            ))
        fig3.update_layout(
            xaxis_title="Turn", yaxis_title="Noise (%)",
            template="plotly_dark", height=310, barmode="group",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Cascade Efficiency ─────────────────────────────────────────────────
    if "cascading" in present and any(v != 1.0 for v in data["cascading"].get("cascade_eff", [1.0])):
        st.subheader("Cascade Efficiency (recall-per-token vs Naive baseline)")
        eff_vals = data["cascading"].get("cascade_eff", [1.0] * len(cps))
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scatter(
            x=cps, y=eff_vals,
            name="Cascade Efficiency",
            mode="lines+markers+text",
            text=[f"{v:.2f}x" for v in eff_vals],
            textposition="top center",
            line=dict(color=COLORS["cascading"], width=3),
            marker=dict(size=10),
        ))
        fig_eff.add_hline(y=1.0, line_dash="dot", line_color="#cdd6f4", annotation_text="Naive baseline")
        fig_eff.update_layout(
            xaxis_title="Turn", yaxis_title="Cascade Efficiency (x)",
            template="plotly_dark", height=300,
        )
        st.plotly_chart(fig_eff, use_container_width=True)
        st.caption("Efficiency > 1.0 means cascading delivers more recall per token spent than the naive baseline.")

    # ── Token cost table ────────────────────────────────────────────────────
    st.subheader("Business Impact — Monthly Token Cost")
    rows = []
    for name in present:
        final_tok = data[name]["tokens"][-1]
        monthly = final_tok * MONTHLY_QUERIES * COST_PER_TOKEN_INR
        rows.append({
            "Backend":           name.capitalize(),
            "Tokens / Query":    f"{final_tok:,}",
            "Monthly Cost (₹)":  f"₹{monthly:,.0f}",
            "Recall @ Final":    f"{data[name]['recall'][-1]*100:.1f}%",
            "Drift @ Final":     f"{data[name]['drift'][-1]*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Savings callout
    if "naive" in data and "cascading" in data:
        n_cost = data["naive"]["tokens"][-1]    * MONTHLY_QUERIES * COST_PER_TOKEN_INR
        c_cost = data["cascading"]["tokens"][-1] * MONTHLY_QUERIES * COST_PER_TOKEN_INR
        pct = (n_cost - c_cost) / n_cost * 100
        recall_delta = (data["cascading"]["recall"][-1] - data["naive"]["recall"][-1]) * 100
        st.success(
            f"**Cascading Temporal Memory saves {pct:.0f}% in token costs** vs Naive "
            f"while delivering {recall_delta:+.1f}pp better recall at 100 K queries/month."
        )

    # Cost bar chart
    fig4 = go.Figure()
    for row in rows:
        cost_val = float(row["Monthly Cost (₹)"].replace("₹", "").replace(",", ""))
        fig4.add_trace(go.Bar(
            x=[row["Backend"]], y=[cost_val],
            name=row["Backend"],
            marker_color=COLORS.get(row["Backend"].lower(), "#cdd6f4"),
            text=f"₹{cost_val:,.0f}", textposition="outside",
        ))
    fig4.update_layout(
        yaxis_title="Monthly cost (₹) @ 100K queries",
        template="plotly_dark", height=360, showlegend=False,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Research export ─────────────────────────────────────────────────────
    st.subheader("Research Export")
    latex = _latex_table(data, cps, present)
    ec1, ec2 = st.columns(2)
    ec1.download_button(
        "⬇ results.json", json.dumps(data, indent=2),
        "memorylens_results.json", "application/json",
    )
    ec2.download_button(
        "⬇ LaTeX table", latex,
        "memorylens_table.tex", "text/plain",
    )


def _latex_table(data: Dict, checkpoints: List[int], present: List[str]) -> str:
    cols = "l" + "c" * len(checkpoints)
    header = "Backend & " + " & ".join(f"T={c}" for c in checkpoints) + r" \\"
    rows = []
    for name in present:
        vals = " & ".join(f"{v*100:.1f}" for v in data[name]["recall"])
        rows.append(f"{name.capitalize()} & {vals}" + r" \\")
    body = "\n".join(rows)
    return (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{MemoryLens: Recall@T by Memory Architecture}" "\n"
        f"\\begin{{tabular}}{{{cols}}}\n"
        r"\hline" "\n"
        + header + "\n"
        r"\hline" "\n"
        + body + "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# ─── Main logic ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.is_demo = False

if demo_btn:
    st.session_state.results = load_demo()
    st.session_state.is_demo = True
    st.rerun()

if run_btn:
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Add it to a `.env` file in the project root.")
    elif not checkpoints:
        st.warning("Select at least one checkpoint.")
    else:
        log_area = st.empty()
        logs: List[str] = []

        def push_log(msg: str) -> None:
            logs.append(msg)
            log_area.text_area("Progress", "\n".join(logs[-12:]), height=200)

        with st.spinner("Running benchmark…"):
            from evaluation.benchmark import run_benchmark, results_to_display_dict
            from evaluation.logger import log_run
            raw = run_benchmark(
                total_turns=total_turns,
                eval_checkpoints=sorted(checkpoints),
                backends=backends,
                progress=push_log,
            )
            display = results_to_display_dict(raw)
            st.session_state.results = display
            st.session_state.is_demo = False
            saved = log_run(display, {"total_turns": total_turns, "backends": backends})
            push_log(f"Results saved → {saved}")

        log_area.empty()
        st.rerun()

if st.session_state.results:
    render_results(st.session_state.results, is_demo=st.session_state.is_demo)
else:
    # ── Landing page ──────────────────────────────────────────────────────
    st.title("🔭 MemoryLens")
    st.markdown("### *An Evaluation Framework for LLM Memory Decay*")
    st.markdown("> **You can't improve what you can't measure. Nobody is measuring memory.**")
    st.divider()

    st.markdown("""
| Layer | What It Does |
|-------|--------------|
| **Memory Injection** | Injects personal facts at T=0 and queries them at T=10, 25, 50, 100 |
| **3 Backends** | Naive (full history), RAG (vector retrieval), Cascading Temporal (tiered decay) |
| **5 Metrics** | Recall@T · Precision@K · Temporal Drift · Memory Noise Ratio · Token Cost |
| **Dashboard** | Decay curves, cost impact, LaTeX-ready research tables |

**Click 📊 Demo** in the sidebar for instant results, or set `GROQ_API_KEY` and click **▶ Run Live**.
""")

    st.markdown("---")
    st.markdown("#### How Cascading Temporal Memory Works")
    st.code("""
┌─────────────────────────────────────────────────────────┐
│                  CASCADING TEMPORAL MEMORY               │
│                                                          │
│  ┌─────────────────┐                                    │
│  │   HOT TIER      │  Last 12 messages — full fidelity  │
│  │  (verbatim)     │  Always in context                 │
│  └────────┬────────┘                                    │
│           │ overflow                                     │
│  ┌────────▼────────┐                                    │
│  │   WARM TIER     │  Older messages — semantic filter  │
│  │  (full text)    │  Top-3 retrieved per query          │
│  └────────┬────────┘                                    │
│           │ overflow                                     │
│  ┌────────▼────────┐                                    │
│  │   COLD TIER     │  Ancient context — compressed      │
│  │  (summaries)    │  Injected as system context        │
│  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
""", language="")
