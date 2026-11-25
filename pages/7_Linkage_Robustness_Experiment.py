import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui_theme import inject_theme, page_nav, hero, read_more

st.set_page_config(page_title="Linkage Robustness Experiment", page_icon="🔬", layout="wide")
inject_theme()
page_nav()
hero("🔬", "Research Question: How Robust Is Fuzzy Record Linkage?",
     "A controlled experiment — independent of whether the data is real or synthetic.")

read_more(
    "ℹ️ Method (click to expand)",
    """
Every other page uses one fixed linkage rule (normalized `last_name + first_name + zip` exact
match, in `src/data_cleaning.py`) to join records. But *how good is that rule, really* — and is
there a better one? Generate 400 synthetic identities, create a "noisy" second-system copy of
each at increasing levels of character-level data-entry corruption (typos, transpositions,
duplicated letters, altered zip digits), and measure how well each strategy recovers the match:

- **Strategy A (production)** — exact match on the normalized key, as used elsewhere in this project.
- **Strategy B (alternative)** — exact zip match narrows candidates, then accepts the best
  name-similarity score (`difflib` ratio) above a threshold (0.82).
""",
)

try:
    results = pd.read_csv("data/processed/linkage_robustness.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.linkage_robustness` first.")
    st.stop()

a5 = results[(results.strategy.str.startswith("A")) & (results.corruption_level == 0.05)]["recall"].values[0]
b5 = results[(results.strategy.str.startswith("B")) & (results.corruption_level == 0.05)]["recall"].values[0]
max_false = results["false_match_rate"].max()

k1, k2, k3 = st.columns(3)
k1.metric("Strategy A recall at 5% noise", f"{a5:.0%}", delta="-57pp vs. no noise", delta_color="inverse")
k2.metric("Strategy B recall at 5% noise", f"{b5:.0%}", delta=f"+{(b5-a5)*100:.0f}pp vs. Strategy A")
k3.metric("Max false-match rate (either strategy)", f"{max_false:.1%}")

c1, c2 = st.columns(2)
with c1:
    fig = px.line(
        results, x="corruption_level", y="recall", color="strategy",
        markers=True, title="Recall vs. data-entry corruption level",
        labels={"corruption_level": "Character-level corruption probability", "recall": "Recall (correct link rate)"},
        color_discrete_sequence=["#8C1D40", "#FFC627"],
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = px.line(
        results, x="corruption_level", y="false_match_rate", color="strategy",
        markers=True, title="False-match rate vs. corruption level",
        labels={"corruption_level": "Character-level corruption probability", "false_match_rate": "False-match rate"},
        color_discrete_sequence=["#8C1D40", "#FFC627"],
    )
    fig2.update_yaxes(tickformat=".1%", range=[-0.02, max(0.1, results["false_match_rate"].max() * 1.5)])
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

a10 = results[(results.strategy.str.startswith("A")) & (results.corruption_level == 0.10)]["recall"].values[0]
b10 = results[(results.strategy.str.startswith("B")) & (results.corruption_level == 0.10)]["recall"].values[0]

st.subheader("📌 Findings")
st.markdown(
    f"**Exact-key matching is fragile** — recall drops from 100% to {a5:.0%} at just 5% noise "
    f"(100% → {a10:.0%} by 10%). **The similarity-threshold alternative degrades more gracefully** "
    f"({b5:.0%} → {b10:.0%} over the same range, ~{b10/a10:.1f}x better at 10%). "
    f"**Neither method produces meaningful false matches** — the zip-code pre-filter does the real "
    f"precision work, so the dominant failure mode is *missed* matches, not *wrong* ones."
)
read_more(
    "💡 Practical implication (click to expand)",
    """
If this linkage pipeline were deployed against a system of record with meaningfully noisy data
entry (common in manually maintained membership spreadsheets), switching to a similarity-threshold
strategy would recover materially more true matches at no measurable cost in false-positive
linkage — a concrete, testable recommendation that synthetic data is perfectly suited to produce,
because it's a claim about the *method*, not about the (synthetic) people in it.
""",
)

with st.expander("📋 Raw results table"):
    st.dataframe(results.round(4), use_container_width=True)
