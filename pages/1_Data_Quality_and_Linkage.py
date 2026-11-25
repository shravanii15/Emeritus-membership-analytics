import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui_theme import inject_theme, page_nav, hero, read_more

st.set_page_config(page_title="Data Quality & Linkage", page_icon="🧹", layout="wide")
inject_theme()
page_nav()
hero("🧹", "Data Quality & Record Linkage",
     "Three spreadsheets, no shared reliable key — joined on a fuzzy name+zip match.")

try:
    clean = pd.read_csv("data/processed/members_clean.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total linked records", f"{len(clean):,}")
c2.metric("Resolved numeric ASU IDs", f"{clean['asu_id'].notna().mean():.0%}")
c3.metric("Matched to a dues record", f"{clean['membership_fee'].notna().mean():.0%}")
c4.metric("Matched to current EV sub", f"{clean['has_current_ev_subscription'].mean():.0%}")

read_more(
    "📄 The 3 source files & why they don't share a key (click to expand)",
    """
| Source file | What it contains | Join problems |
|---|---|---|
| `Regular_Members_and_Historical_List` | Active / deceased / special-case rosters | ASU ID sometimes the literal string `"Unknown"` |
| `Membership_Dues_-_EV_Subscriptions` | Dues payments + EV magazine subscription status | No shared primary key with the roster file |
| `EV_Subscriptions_-_CURRENT` | Current print-subscription fulfillment | Zip codes mix 5-digit and ZIP+4 formats |

**Linkage strategy:** since ASU ID is unreliable, records are joined on a composite fuzzy key —
normalized `last_name + first_name + 5-digit zip` (accents stripped, punctuation removed,
case-folded). This recovers cross-file matches a naive exact-ID join would miss entirely.
(See the **Linkage Robustness Experiment** page for how well this actually holds up under noise.)
""",
)

st.subheader("Status breakdown")
status_counts = clean["source_status"].value_counts().reset_index()
status_counts.columns = ["status", "count"]
fig = px.bar(status_counts, x="status", y="count", color="status",
             color_discrete_sequence=px.colors.qualitative.Safe,
             title="Member records by roster status")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

read_more(
    f"🔍 Data quality issues found & handled (click to expand)",
    f"""
- **{(clean['asu_id'].isna()).sum()} records** had an unusable ASU ID (`"Unknown"` or blank) and
  were linked purely on the fuzzy name+zip key.
- **{int(clean['opted_out_of_email'].sum())} active members** had an administrative note (e.g.
  *"Has requested removal from all email lists"*) sitting in what should have been a
  discipline/date field — detected and separated into a proper `opted_out_of_email` flag.
- Zip codes were normalized from a mix of 5-digit and ZIP+4 formats to a consistent 5-digit key.
""",
)

st.subheader("Sample of the linked dataset")
st.dataframe(
    clean[["member_id", "first_name", "last_name", "city", "state", "source_status",
           "has_current_ev_subscription", "opted_out_of_email"]].sample(
        min(15, len(clean)), random_state=1
    ),
    use_container_width=True,
)
