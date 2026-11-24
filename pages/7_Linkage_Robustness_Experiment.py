import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Linkage Robustness Experiment", page_icon="🔬", layout="wide")
st.title("🔬 Research Question: How Robust Is Fuzzy Record Linkage?")

st.markdown(
    """
Every other page in this project uses one fixed linkage rule (normalized
`last_name + first_name + zip` exact match, in `src/data_cleaning.py`) to
join records across the three source spreadsheets. But *how good is that
rule, really* — and is there a better one? That's an empirical question we
can actually answer with a controlled experiment, independent of whether
the underlying membership data is real or synthetic.

**Method:** generate 400 synthetic identities, create a "noisy" second-
system copy of each at increasing levels of character-level data-entry
corruption (typos, transpositions, duplicated letters, altered zip digits),
and measure how well each linkage strategy recovers the correct match:

- **Strategy A (production)** — exact match on the normalized key, exactly
  as implemented and used elsewhere in this project.
- **Strategy B (alternative)** — exact zip match narrows candidates, then
  accept the best name-similarity score (`difflib` ratio) above a
  threshold (0.82).
"""
)

try:
    results = pd.read_csv("data/processed/linkage_robustness.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.linkage_robustness` first.")
    st.stop()

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

a5 = results[(results.strategy.str.startswith("A")) & (results.corruption_level == 0.05)]["recall"].values[0]
b5 = results[(results.strategy.str.startswith("B")) & (results.corruption_level == 0.05)]["recall"].values[0]
a10 = results[(results.strategy.str.startswith("A")) & (results.corruption_level == 0.10)]["recall"].values[0]
b10 = results[(results.strategy.str.startswith("B")) & (results.corruption_level == 0.10)]["recall"].values[0]
max_false = results["false_match_rate"].max()

st.subheader("📌 Findings")
st.markdown(
    f"""
1. **Exact-key matching is fragile.** At just **5% character-level
   corruption** (roughly one typo every 20 characters — realistic for
   manual data entry), production recall drops from 100% to **{a5:.0%}**.
   By 10% corruption it's down to **{a10:.0%}**.
2. **The similarity-threshold alternative degrades far more gracefully**:
   **{b5:.0%}** recall at 5% corruption and **{b10:.0%}** at 10% —
   roughly {b10/a10:.1f}x better recall than the exact-key method at that
   noise level, for the same records.
3. **Neither method produces meaningful false matches** (max observed rate:
   {max_false:.1%}). The zip-code pre-filter is doing the real precision
   work in both strategies — the dominant failure mode is **missed
   matches, not wrong matches**. In a production setting, this means the
   bigger operational risk from the current linkage rule is silently
   *dropping* real members from the linked dataset, not accidentally
   merging two different people's records.

**Practical implication:** if this linkage pipeline were deployed against
a system of record with meaningfully noisy data entry (common in manually
maintained membership spreadsheets), switching to a similarity-threshold
strategy would recover materially more true matches at no measurable cost
in false-positive linkage — a concrete, testable recommendation that
synthetic data is perfectly suited to produce, because it's a claim about
the *method*, not about the (synthetic) people in it.
"""
)

st.subheader("Raw results")
st.dataframe(results.round(4), use_container_width=True)
