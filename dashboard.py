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

COLORS = {
    "naive":       "#f38ba8",
    "rag":         "#89b4fa",
    "rag_chunked": "#74c7ec",
    "cascading":   "#a6e3a1",
    "summary":     "#fab387",
    "entity":      "#cba6f7",
    "graph":       "#f9e2af",
    "faiss":       "#94e2d5",
}
ALL_BACKENDS = ["naive", "rag", "rag_chunked", "cascading", "summary", "entity", "graph", "faiss"]
# Illustrative cost assumption, stated in the UI: $1 per 1M input tokens.
MONTHLY_QUERIES = 100_000
COST_PER_TOKEN_USD = 1 / 1_000_000

_PROVIDER_KEYS = {
    "groq":       "GROQ_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama":     None,  # no key needed — just a running server
}


def _detect_available_providers() -> List[str]:
    """Return provider names whose credentials are currently present."""
    available = []
    for name, env_var in _PROVIDER_KEYS.items():
        if env_var is None:
            # Ollama: try a quick ping
            import urllib.request
            try:
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
                available.append(name)
            except Exception:
                pass
        elif os.getenv(env_var):
            available.append(name)
    return available


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
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
        ALL_BACKENDS,
        default=["naive", "rag", "cascading"],
        help='faiss requires the optional dependency: pip install "memorylens-bench[faiss]"',
    )
    st.divider()

    # ── LLM Provider ──────────────────────────────────────────────────────
    st.subheader("LLM Evaluation (optional)")
    available_providers = _detect_available_providers()
    provider_options = ["None (content-only)"] + available_providers
    selected_provider_label = st.selectbox(
        "Provider",
        provider_options,
        help=(
            "Run a real answer+judge pass on top of content-based metrics. "
            "Set the matching API key in your .env file to unlock a provider."
        ),
    )
    selected_provider = (
        None if selected_provider_label == "None (content-only)"
        else selected_provider_label
    )

    if not available_providers:
        st.caption(
            "No provider detected. Add an API key to `.env`:\n"
            "`GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, "
            "`OPENROUTER_API_KEY`, or start Ollama."
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
    present = [b for b in ALL_BACKENDS if b in data]
    has_llm  = data.get("has_llm_eval", False)

    if is_demo:
        st.info(
            "📊 Showing pre-computed demo results. "
            "Set an API key in `.env` and click **▶ Run Live** for real LLM evaluation.",
            icon="ℹ️",
        )

    # ── KPI cards ──────────────────────────────────────────────────────────
    st.subheader("Summary at Final Checkpoint")
    cols = st.columns(len(present))
    for col, name in zip(cols, present):
        d = data[name]
        llm_val = d.get("llm_recall", [None])[-1]
        with col:
            st.markdown(f"#### {name.capitalize()}")
            st.metric("Recall@Final",    f"{d['recall'][-1]*100:.1f}%")
            if llm_val is not None:
                gap = (d["recall"][-1] - llm_val) * 100
                st.metric(
                    "LLM Recall@Final",
                    f"{llm_val*100:.1f}%",
                    delta=f"{gap:+.1f}pp gap",
                    delta_color="inverse",
                )
            st.metric("Avg Tokens",      f"{d['tokens'][-1]:,}")
            st.metric("Temporal Drift",  f"{d['drift'][-1]*100:.1f}%")
            if d.get("contradiction"):
                st.metric("Contradiction", f"{d['contradiction'][-1]*100:.1f}%")
            st.metric("Precision@K",     f"{d['precision'][-1]*100:.1f}%")

    st.divider()

    # ── Recall decay ───────────────────────────────────────────────────────
    st.subheader("Memory Recall Decay Over Time")

    if has_llm:
        tab_content, tab_llm, tab_gap = st.tabs(
            ["Content Recall", "LLM Recall", "Gap (Content − LLM)"]
        )
    else:
        (tab_content,) = st.tabs(["Content Recall"])
        tab_llm = tab_gap = None

    with tab_content:
        fig = go.Figure()
        for name in present:
            color = COLORS.get(name, "#cdd6f4")
            fig.add_trace(go.Scatter(
                x=cps, y=[v * 100 for v in data[name]["recall"]],
                name=name.capitalize(), mode="lines+markers",
                line=dict(color=color, width=3), marker=dict(size=9),
            ))
        fig.update_layout(
            xaxis_title="Conversation Turn", yaxis_title="Recall (%)",
            yaxis=dict(range=[0, 105]), template="plotly_dark",
            height=360, legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Content Recall: substring match on retrieved context chunks — "
            "fast, reproducible, zero API cost."
        )

    if has_llm and tab_llm is not None:
        with tab_llm:
            fig_llm = go.Figure()
            for name in present:
                color = COLORS.get(name, "#cdd6f4")
                llm_vals = data[name].get("llm_recall", [])
                if any(v is not None for v in llm_vals):
                    fig_llm.add_trace(go.Scatter(
                        x=cps,
                        y=[v * 100 if v is not None else None for v in llm_vals],
                        name=name.capitalize(), mode="lines+markers",
                        line=dict(color=color, width=3, dash="dash"),
                        marker=dict(size=9, symbol="diamond"),
                    ))
            fig_llm.update_layout(
                xaxis_title="Conversation Turn", yaxis_title="LLM Recall (%)",
                yaxis=dict(range=[0, 105]), template="plotly_dark",
                height=360, legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_llm, use_container_width=True)
            provider_used = next(
                (data[b].get("provider") for b in present if data[b].get("provider")),
                "unknown",
            )
            st.caption(
                f"LLM Recall: two-stage answer+judge pipeline using **{provider_used}**. "
                "The LLM actually answers each question; a judge call verifies correctness."
            )

    if has_llm and tab_gap is not None:
        with tab_gap:
            fig_gap = go.Figure()
            for name in present:
                color = COLORS.get(name, "#cdd6f4")
                content_vals = data[name]["recall"]
                llm_vals = data[name].get("llm_recall", [None] * len(content_vals))
                gaps = [
                    (c - l) * 100 if l is not None else None
                    for c, l in zip(content_vals, llm_vals)
                ]
                if any(g is not None for g in gaps):
                    fig_gap.add_trace(go.Bar(
                        x=[f"T={c}" for c in cps],
                        y=gaps,
                        name=name.capitalize(),
                        marker_color=color,
                    ))
            fig_gap.add_hline(y=0, line_dash="dot", line_color="#cdd6f4")
            fig_gap.update_layout(
                xaxis_title="Checkpoint",
                yaxis_title="Content Recall − LLM Recall (pp)",
                template="plotly_dark", height=360,
                barmode="group",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_gap, use_container_width=True)
            st.caption(
                "Positive gap means content recall *overestimates* true answer quality. "
                "A large gap signals the backend retrieves the right text but the LLM "
                "still fails to extract the correct answer."
            )

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
    st.subheader("Projected Monthly Token Cost")
    st.caption(
        f"Illustrative projection: {MONTHLY_QUERIES:,} queries/month at "
        "$1 per 1M input tokens. Scale to your own price and volume."
    )
    rows = []
    for name in present:
        final_tok = data[name]["tokens"][-1]
        monthly = final_tok * MONTHLY_QUERIES * COST_PER_TOKEN_USD
        rows.append({
            "Backend":            name.capitalize(),
            "Tokens / Query":     f"{final_tok:,}",
            "Monthly Cost ($)":   f"${monthly:,.0f}",
            "Recall @ Final":     f"{data[name]['recall'][-1]*100:.1f}%",
            "Drift @ Final":      f"{data[name]['drift'][-1]*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Savings callout
    if "naive" in data and "cascading" in data:
        n_tok = data["naive"]["tokens"][-1]
        c_tok = data["cascading"]["tokens"][-1]
        if n_tok > 0:
            pct = (n_tok - c_tok) / n_tok * 100
            recall_delta = (data["cascading"]["recall"][-1] - data["naive"]["recall"][-1]) * 100
            st.success(
                f"**Cascading uses {pct:.0f}% fewer tokens per query** than Naive "
                f"with {recall_delta:+.1f}pp recall difference in this run."
            )

    # Cost bar chart
    fig4 = go.Figure()
    for row in rows:
        cost_val = float(row["Monthly Cost ($)"].replace("$", "").replace(",", ""))
        fig4.add_trace(go.Bar(
            x=[row["Backend"]], y=[cost_val],
            name=row["Backend"],
            marker_color=COLORS.get(row["Backend"].lower(), "#cdd6f4"),
            text=f"${cost_val:,.0f}", textposition="outside",
        ))
    fig4.update_layout(
        yaxis_title=f"Monthly cost ($) @ {MONTHLY_QUERIES:,} queries",
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


def _series_value(v):
    """Multi-seed logs store per-checkpoint stat dicts; single-seed logs store floats."""
    return v.get("mean") if isinstance(v, dict) else v


def render_history() -> None:
    from memorylens.evaluation.logger import list_runs, get_run_results

    runs = list_runs()
    if not runs:
        st.info(
            "No logged runs yet. Run `memorylens --log` or click **▶ Run Live** — "
            "every run is saved to `experiment_logs/`."
        )
        return

    labels = {}
    for r in runs:
        cfg = r["config"]
        label = (
            f"{r['run_id']} · {cfg.get('total_turns', '?')} turns · "
            f"{', '.join(cfg.get('backends', []))}"
            + (f" · {cfg['provider']}" if cfg.get("provider") else "")
        )
        labels[label] = r

    selected = st.multiselect(
        "Runs to compare",
        list(labels),
        default=list(labels)[:2],
        help="Overlays Recall@T curves; solid/dash/dot line style distinguishes runs.",
    )
    if not selected:
        return

    dashes = ["solid", "dash", "dot", "dashdot", "longdash"]
    fig = go.Figure()
    table_rows = []

    for run_idx, label in enumerate(selected):
        run = labels[label]
        results = get_run_results(run["run_id"]) or {}
        cps = results.get("checkpoints", [])
        for name in [b for b in ALL_BACKENDS if b in results]:
            recall = [_series_value(v) for v in results[name].get("recall", [])]
            tokens = [_series_value(v) for v in results[name].get("tokens", [])]
            fig.add_trace(go.Scatter(
                x=cps,
                y=[v * 100 if v is not None else None for v in recall],
                name=f"{run['run_id']} · {name}",
                mode="lines+markers",
                line=dict(color=COLORS.get(name, "#cdd6f4"),
                          dash=dashes[run_idx % len(dashes)], width=2),
            ))
            final_recall = next((v for v in reversed(recall) if v is not None), None)
            final_tokens = next((v for v in reversed(tokens) if v is not None), None)
            table_rows.append({
                "Run":            run["run_id"],
                "Backend":        name,
                "Recall @ Final": f"{final_recall*100:.1f}%" if final_recall is not None else "—",
                "Tokens @ Final": f"{final_tokens:,.0f}"     if final_tokens is not None else "—",
                "Turns":          run["config"].get("total_turns", "—"),
                "Provider":       run["config"].get("provider") or "content-only",
            })

    fig.update_layout(
        xaxis_title="Conversation Turn", yaxis_title="Recall (%)",
        yaxis=dict(range=[0, 105]), template="plotly_dark",
        height=420, legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ─── Main logic ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.is_demo = False

if demo_btn:
    st.session_state.results = load_demo()
    st.session_state.is_demo = True
    st.rerun()

if run_btn:
    if not checkpoints:
        st.warning("Select at least one checkpoint.")
    else:
        log_area = st.empty()
        logs: List[str] = []

        def push_log(msg: str) -> None:
            logs.append(msg)
            log_area.text_area("Progress", "\n".join(logs[-12:]), height=200)

        with st.spinner("Running benchmark…"):
            from memorylens.evaluation.benchmark import run_benchmark, results_to_display_dict
            from memorylens.evaluation.logger import log_run

            # Resolve provider (None = content-only)
            provider_obj = None
            if selected_provider:
                try:
                    from memorylens.utils.providers import get_provider
                    provider_obj = get_provider(selected_provider)
                    push_log(f"LLM provider: {provider_obj.name}")
                except Exception as e:
                    st.error(f"Provider error: {e}")
                    st.stop()

            raw = run_benchmark(
                total_turns=total_turns,
                eval_checkpoints=sorted(checkpoints),
                backends=backends,
                provider=provider_obj,
                progress=push_log,
            )
            display = results_to_display_dict(raw)
            st.session_state.results = display
            st.session_state.is_demo = False
            saved = log_run(display, {
                "total_turns": total_turns,
                "backends":    backends,
                "provider":    provider_obj.name if provider_obj else None,
            })
            push_log(f"Results saved -> {saved}")

        log_area.empty()
        st.rerun()

tab_bench, tab_history = st.tabs(["Benchmark", "Run History"])

with tab_history:
    render_history()

with tab_bench:
    if st.session_state.results:
        render_results(st.session_state.results, is_demo=st.session_state.is_demo)
    else:
        # ── Landing page ──────────────────────────────────────────────────
        st.title("🔭 MemoryLens")
        st.markdown("### *An Evaluation Framework for LLM Memory Decay*")
        st.divider()

        st.markdown(f"""
| Layer | What It Does |
|-------|--------------|
| **Memory Injection** | Injects personal facts at known turns and queries them at checkpoints |
| **{len(ALL_BACKENDS)} Backends** | {' · '.join(ALL_BACKENDS)} |
| **6 Metrics** | Recall@T · Precision@K · Temporal Drift · Contradiction · Noise Ratio · Token Cost |
| **LLM Eval** | Two-stage answer+judge pipeline — 5 providers (Groq, OpenAI, Anthropic, OpenRouter, Ollama) |
| **Dashboard** | Decay curves, content vs LLM recall gap, cost projection, run history, LaTeX export |

**Click 📊 Demo** in the sidebar for instant results, or configure a provider and click **▶ Run Live**.
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
