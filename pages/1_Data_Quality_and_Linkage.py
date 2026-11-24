import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Quality & Linkage", page_icon="🧹", layout="wide")
st.title("🧹 Data Quality & Record Linkage")

st.markdown(
    """
This project starts from **three separate spreadsheets** maintained by
different staff over time — a very common real-world data-engineering
situation. None of them share a single reliable foreign key:

| Source file | What it contains | Join problems |
|---|---|---|
| `Regular_Members_and_Historical_List` | Active / deceased / special-case rosters | ASU ID sometimes the literal string `"Unknown"` |
| `Membership_Dues_-_EV_Subscriptions` | Dues payments + EV magazine subscription status | No shared primary key with the roster file |
| `EV_Subscriptions_-_CURRENT` | Current print-subscription fulfillment | Zip codes mix 5-digit and ZIP+4 formats |

**Linkage strategy:** since ASU ID is unreliable, records are joined on a
composite fuzzy key — normalized `last_name + first_name + 5-digit zip`
(accents stripped, punctuation removed, case-folded). This recovers
cross-file matches that a naive exact-ID join would miss entirely.
"""
)

try:
    clean = pd.read_csv("data/processed/members_clean.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

st.subheader("Linked dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total linked records", f"{len(clean):,}")
c2.metric("Resolved numeric ASU IDs", f"{clean['asu_id'].notna().mean():.0%}")
c3.metric("Matched to a dues record", f"{clean['membership_fee'].notna().mean():.0%}")
c4.metric("Matched to current EV sub", f"{clean['has_current_ev_subscription'].mean():.0%}")

st.subheader("Status breakdown")
status_counts = clean["source_status"].value_counts().reset_index()
status_counts.columns = ["status", "count"]
fig = px.bar(status_counts, x="status", y="count", color="status",
             color_discrete_sequence=px.colors.qualitative.Safe,
             title="Member records by roster status")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Data quality issues found & handled")
st.markdown(
    f"""
- **{(clean['asu_id'].isna()).sum()} records** had an unusable ASU ID
  (`"Unknown"` or blank) and were linked purely on the fuzzy name+zip key.
- **{int(clean['opted_out_of_email'].sum())} active members** had an
  administrative note (e.g. *"Has requested removal from all email lists"*)
  sitting in what should have been a discipline/date field — these were
  detected and separated into a proper `opted_out_of_email` flag rather than
  being left to corrupt the discipline/date columns.
- Zip codes were normalized from a mix of 5-digit and ZIP+4 formats to a
  consistent 5-digit key before matching.
"""
)

st.subheader("Sample of the linked dataset")
st.dataframe(
    clean[["member_id", "first_name", "last_name", "city", "state", "source_status",
           "has_current_ev_subscription", "opted_out_of_email"]].sample(
        min(15, len(clean)), random_state=1
    ),
    use_container_width=True,
)
