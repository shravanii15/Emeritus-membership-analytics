import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.parking_analysis import estimate_savings, sensitivity_table
from src.ui_theme import inject_theme, page_nav, hero, read_more

st.set_page_config(page_title="Parking Budget Analysis", page_icon="🅿️", layout="wide")
inject_theme()
page_nav()
hero("🅿️", "Parking Budget Impact Analysis",
     "Could the association save money by not issuing campus parking passes to out-of-state members?")

st.warning(
    "⚠️ **Two assumptions drive the dollar figure below, and neither is measured** — the "
    "per-pass cost and the share of out-of-state members who actually hold/use a pass. "
    "Use the sliders in the **Sensitivity analysis** tab before treating any single number here "
    "as a recommendation."
)

try:
    active = pd.read_csv("data/processed/members_clean.csv")
    active = active[active["source_status"] == "active"]
    by_state = pd.read_csv("data/processed/parking_by_state.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

read_more(
    "ℹ️ Problem, constraints, and method (click to expand)",
    """
**Problem:** campus parking passes are issued to all active members by default, but a material
share of the roster lives out of state (retirees who moved away, or never relocated near campus)
and is geographically unlikely to use on-campus parking at all — a recurring cost with no clear
offsetting benefit.

**Constraint:** there is no field in the source data recording whether a member actually uses a
parking pass, only where they live. So this analysis estimates an *upper-bound* opportunity, not
a measured one, and says so explicitly rather than presenting a single confident dollar figure.

**Method:**
1. Identify active members whose mailing state is not Arizona (`is_in_state == 0`).
2. Apply an assumed per-pass annual cost and an assumed "actually holds a pass" rate to that
   out-of-state count.
3. Report the result as a **range across several plausible assumptions**, not a point estimate —
   see the Sensitivity analysis tab.

**Decision this supports:** whether it's worth the membership office's time to audit out-of-state
members' actual parking-pass usage before the next renewal cycle — not a final budget line.
""",
)

base = estimate_savings(active)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Active members", base["total_active_members"])
k2.metric("Out-of-state members", base["n_out_of_state"], help="Mailing state ≠ Arizona")
k3.metric("Share out-of-state", f"{base['pct_out_of_state']:.1%}")
k4.metric("Base-case estimated annual savings", f"${base['estimated_annual_savings']:,.0f}",
          help=f"Assumes ${base['pass_cost']:.0f}/pass, {base['holder_rate']:.0%} of out-of-state "
               f"members hold one")

tab1, tab2, tab3 = st.tabs(["🗺️ Where out-of-state members live", "🧪 Sensitivity analysis", "📋 Case study"])

with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(
            by_state.sort_values("member_count", ascending=True),
            x="member_count", y="state", orientation="h",
            title="Out-of-state active members, by state",
            labels={"member_count": "Members", "state": ""},
            color_discrete_sequence=["#8C1D40"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Top states**")
        st.dataframe(by_state.rename(columns={"state": "State", "member_count": "Members"}),
                     use_container_width=True, hide_index=True, height=360)

with tab2:
    st.markdown(
        "Instead of trusting one guessed cost and one guessed holder rate, this re-runs the "
        "estimate across **3 pass-cost scenarios × 3 holder-rate scenarios** (9 combinations)."
    )
    sc1, sc2 = st.columns(2)
    with sc1:
        pass_cost = st.slider("Assumed annual cost per parking pass ($)", 100, 600, 280, step=10)
    with sc2:
        holder_rate = st.slider("Assumed share of out-of-state members who hold a pass", 0.0, 1.0, 1.0, 0.05)

    live = estimate_savings(active, pass_cost=pass_cost, holder_rate=holder_rate)
    st.metric("Estimated annual savings at these assumptions", f"${live['estimated_annual_savings']:,.0f}")

    sens = sensitivity_table(active)
    pivot = sens.pivot(index="pass_cost", columns="holder_rate", values="estimated_annual_savings")
    pivot.index = [f"${c:,.0f}/pass" for c in pivot.index]
    pivot.columns = [f"{r:.0%} hold a pass" for r in pivot.columns]
    st.markdown("**Estimated annual savings across assumption combinations**")
    st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)

    st.markdown(
        f"""
**Finding:** across every combination tested, the estimated annual savings range from
**${sens['estimated_annual_savings'].min():,.0f} to ${sens['estimated_annual_savings'].max():,.0f}**.
Even under the most conservative assumptions (lower per-pass cost, only half of out-of-state
members actually holding a pass), the figure is large enough to justify a one-time audit of
actual out-of-state pass usage before the next renewal cycle — which is the real, low-risk next
step, rather than cutting passes immediately on an unverified assumption.
"""
    )

with tab3:
    st.markdown("#### Case study: identifying a budget-recovery opportunity in membership data")
    st.markdown(
        """
**Problem.** Years of fragmented membership and dues spreadsheets meant the association had no
reliable, unified view of who was an active member, let alone whether benefits like parking
passes were being issued to people unlikely to use them.

**Constraints.** The underlying data is sensitive (names, addresses, payment records) and the
people who needed the answer — membership-office staff — are not technical users. Any tool built
on it had to protect member privacy and be usable without a data background.

**Decisions.**
- Built a two-tier record-linkage strategy (ID match first, normalized name+zip match second) to
  reconcile three separately maintained spreadsheets into one clean member record — see the
  **Data Quality & Linkage** page.
- Published this entire portfolio version on **synthetic data only**, so the method, the code,
  and this parking analysis can be shared publicly without exposing any real member's information.
- Framed the parking-cost estimate as a **range under stated assumptions**, not a single number,
  because the source data can't actually confirm who uses a pass — an honest, audit-first
  recommendation is more credible than a false-precision dollar figure.

**Results.** *(Add the specific, already-on-your-résumé numbers here from the real engagement —
dollars identified, members reconciled, time saved — this synthetic rebuild intentionally doesn't
invent those.)*

**What I'd do differently.** Add an actual pass-usage field to the intake form going forward, so
future versions of this analysis are measured rather than estimated — the single biggest
limitation of this piece of work.
"""
    )

st.caption(
    "All figures on this page are computed from 100% synthetic data generated for this portfolio "
    "project — see **About & Methodology** for the full privacy and limitations notes."
)
