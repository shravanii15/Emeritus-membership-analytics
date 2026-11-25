"""
ui_theme.py
============
Shared visual styling + layout helpers for every Streamlit page, so the
dashboard reads as an interactive tool rather than a scrollable document.

Call inject_theme() once near the top of every page (after st.set_page_config).
"""
import streamlit as st


def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

        /* Tighter top padding so content starts higher on the page */
        .block-container { padding-top: 2rem; max-width: 1200px; }

        /* Metric tiles: card look */
        div[data-testid="stMetric"] {
            background: #FAFAFA;
            border: 1px solid #ECECEC;
            border-radius: 10px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }
        div[data-testid="stMetricValue"] { font-weight: 800; color: #8C1D40; }

        /* Headings tighter + bolder */
        h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
        h2, h3 { font-weight: 700 !important; }

        /* Expanders: slightly card-like, so "read more" sections look optional */
        div[data-testid="stExpander"] {
            border: 1px solid #ECECEC;
            border-radius: 10px;
            background: #FDFDFD;
        }

        /* Tabs: bolder active indicator */
        button[data-baseweb="tab"] { font-weight: 600; }

        /* Shrink default vertical gaps between elements slightly */
        div[data-testid="stVerticalBlock"] > div { padding-top: 0.15rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_nav():
    """A compact, always-visible row of quick links to every page, so moving
    between stages of the pipeline doesn't require hunting in the sidebar."""
    pages = [
        ("app.py", "🏠 Home"),
        ("pages/1_Data_Quality_and_Linkage.py", "🧹 Data"),
        ("pages/2_Exploratory_Data_Analysis.py", "📊 EDA"),
        ("pages/3_Renewal_Prediction_Model.py", "🤖 Model"),
        ("pages/4_Member_Segmentation.py", "🧩 Segments"),
        ("pages/5_Outreach_Budget_Optimizer.py", "📞 Optimizer"),
        ("pages/8_Parking_Budget_Analysis.py", "🅿️ Parking"),
        ("pages/7_Linkage_Robustness_Experiment.py", "🔬 Experiment"),
        ("pages/6_About_and_Methodology.py", "📖 About"),
    ]
    cols = st.columns(len(pages))
    for col, (target, label) in zip(cols, pages):
        with col:
            st.page_link(target, label=label)
    st.divider()


def hero(icon: str, title: str, one_liner: str):
    """A compact title block: icon + title + one plain sentence. Replaces
    multi-paragraph intros — anything longer belongs in an expander below."""
    st.markdown(f"## {icon} {title}")
    st.caption(one_liner)


def read_more(label: str, body_markdown: str, expanded: bool = False):
    """Puts longer explanatory text behind a collapsed expander so the page
    leads with numbers/charts/controls instead of paragraphs."""
    with st.expander(label, expanded=expanded):
        st.markdown(body_markdown)
