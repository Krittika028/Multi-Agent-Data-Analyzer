"""
app.py — Navigation entry point for AI Data Cleaner & Analyzer

This project no longer has a sidebar at all. Streamlit's multipage
"pages/" folder auto-detection is what generates the sidebar page-list
and its collapse arrow — that mechanism has been removed entirely by
routing through st.navigation(..., position="hidden") instead. This is
not a CSS hack (like `[data-testid="stSidebar"] { display: none }`
used to be) — with position="hidden" Streamlit never builds the
sidebar navigation DOM in the first place, so there is nothing to
hide, collapse, or accidentally leak through on any page (including
the login screen).

All in-app navigation continues to work exactly as before via
st.switch_page(...) calls inside each page.
"""

import streamlit as st

pages = [
    st.Page("home_page.py",          title="Home",             icon="🧑‍💼", default=True),
    st.Page("pages/1_data_cleaner.py",   title="Data Cleaner",     icon="🧹"),
    st.Page("pages/2_domain_detector.py", title="Domain Detector",  icon="🧭"),
    st.Page("pages/3_chat_reports.py",    title="Chat with Reports", icon="💬"),
]

nav = st.navigation(pages, position="hidden")
nav.run()