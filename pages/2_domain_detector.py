"""
pages/2_Domain_Detector.py
"""

import streamlit as st
from domain_detector import DomainDetector
from auth import render_login_page, check_authentication, render_logout

# ── set_page_config MUST be the very first st call ────────────────────────────
st.set_page_config(page_title="Domain Detector", page_icon="🧭", layout="wide")

# ── Auth ──────────────────────────────────────────────────────────────────────
authenticator, config = render_login_page()
is_authenticated, name, username = check_authentication(authenticator, config)
if not is_authenticated:
    st.stop()

render_logout(authenticator, name)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0a0f; color: #e2e8f0; }
h1 {
    background: linear-gradient(90deg, #00f5ff, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important; font-weight: 800 !important;
}
h2, h3 { color: #00f5ff !important; }
.stButton > button {
    background: linear-gradient(135deg, #00f5ff22, #7c3aed22) !important;
    border: 1px solid #00f5ff !important; color: #00f5ff !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
[data-testid="baseButton-primary"] > button {
    background: linear-gradient(135deg,#00f5ff,#7c3aed) !important;
    color: #000 !important; border: none !important; font-weight: 700 !important;
}
hr { border-color: #00f5ff22 !important; }
div[data-testid="stExpander"] { border: 1px solid #334155; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

DOMAIN_ICONS = {
    "Healthcare":                "🏥",
    "Retail":                    "🛍️",
    "Banking & Finance":         "🏦",
    "Insurance":                 "🛡️",
    "Human Resources":           "🧑‍💼",
    "Manufacturing":             "🏭",
    "Education":                 "🎓",
    "Logistics":                 "🚚",
    "Telecom":                   "📡",
    "E-commerce":                "🛒",
    "Real Estate":               "🏠",
    "Agriculture":               "🌾",
    "Energy & Utilities":        "⚡",
    "Government / Public Sector":"🏛️",
    "Other":                     "📊",
}

CONFIDENCE_COLOR = {
    "High":   "#16a34a",
    "Medium": "#d97706",
    "Low":    "#dc2626",
}


def render_header():
    st.markdown(
        """
        <div style="padding: 1.5rem 0 0.5rem 0;">
            <h1 style="margin-bottom:0;">🧭 Domain Detector</h1>
            <p style="color:#94a3b8; font-size:1.05rem; margin-top:0.25rem;">
                Identifying the business context behind your data before any analysis begins.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_badge(confidence):
    color = CONFIDENCE_COLOR.get(confidence, "#64748b")
    st.markdown(
        f"""
        <span style="
            background-color:{color}20;
            color:{color};
            border:1px solid {color}55;
            padding: 4px 14px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
        ">
            {confidence} Confidence
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(result):
    icon = DOMAIN_ICONS.get(result["domain"], "📊")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid #334155;
                border-radius: 16px;
                padding: 28px 32px;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 4px;">{icon}</div>
                <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em;">
                    Detected Domain
                </div>
                <div style="color:white; font-size:2rem; font-weight:700; margin-top:4px;">
                    {result['domain']}
                </div>
                <div style="color:#cbd5e1; font-size:1rem; margin-top:8px;">
                    Primary Entity: <strong>{result['primary_entity']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.write("")
        render_confidence_badge(result["confidence"])
        if result.get("alternative_domain"):
            st.caption(f"Second guess: {result['alternative_domain']}")

    st.write("")

    with st.expander("🔍 Why the agent thinks this", expanded=True):
        if result.get("reasoning_summary"):
            st.markdown(f"**Summary:** {result['reasoning_summary']}")
            st.write("")
        if result.get("evidence"):
            for point in result["evidence"]:
                st.markdown(f"- {point}")
        else:
            st.caption("No specific evidence points were returned.")


def main():
    render_header()
    st.divider()

    if "cleaned_df" not in st.session_state:
        st.warning("No cleaned dataset found. Please upload and clean a dataset first.")
        if st.button("⬅ Go to Data Cleaner"):
            st.switch_page("pages/1_Data_Cleaner.py")
        return

    df              = st.session_state["cleaned_df"]
    cleaning_report = st.session_state.get("cleaning_report", {})

    if "domain_result" not in st.session_state:
        with st.spinner("Analyzing column patterns, value ranges, and data shape..."):
            detector = DomainDetector()
            st.session_state["domain_result"] = detector.detect(df, cleaning_report)

    render_result_card(st.session_state["domain_result"])

    st.divider()

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("🔄 Re-run Detection"):
            del st.session_state["domain_result"]
            st.rerun()
    with col_b:
        if st.button("Continue to Analysis ➜", type="primary"):
            st.switch_page("pages/3_Chat_Reports.py")


if __name__ == "__main__":
    main()