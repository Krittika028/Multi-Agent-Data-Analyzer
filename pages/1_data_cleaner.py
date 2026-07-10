"""
pages/1_Data_Cleaner.py
"""

import streamlit as st
import pandas as pd
import time
import traceback
from data_cleaner import DataCleaner
from auth import render_login_page, check_authentication, render_logout

# ── set_page_config MUST be the very first st call ────────────────────────────
st.set_page_config(page_title="Data Cleaner", page_icon="🧹", layout="wide")

# ── Auth — hide sidebar until logged in ───────────────────────────────────────
authenticator, config = render_login_page()
is_authenticated, name, username = check_authentication(authenticator, config)
if not is_authenticated:
    st.stop()

render_logout(authenticator, name)


# ── Styles ────────────────────────────────────────────────────────────────────
def inject_styles():
    st.markdown(
        """
        <style>
        .stApp { background-color: #0a0a0f; color: #e2e8f0; }
        h1 {
            background: linear-gradient(90deg, #00f5ff, #7c3aed);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.2rem !important; font-weight: 800 !important;
        }
        h2, h3 { color: #00f5ff !important; }
        .hero { padding: 1.5rem 0 0.5rem 0; }
        .hero p { color: #94a3b8; font-size: 1.05rem; margin-top: 0.25rem; }
        .section-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 1rem;
        }
        .stat-pill {
            display: inline-block;
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 999px;
            padding: 6px 16px;
            margin-right: 8px;
            font-size: 0.85rem;
            color: #cbd5e1;
        }
        .stat-pill b { color: white; }
        div[data-testid="stExpander"] {
            border: 1px solid #334155;
            border-radius: 12px;
        }
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero">
            <h1>🧹 Data Cleaner</h1>
            <p>Upload your dataset, choose what to keep, and let the cleaning agent handle the rest —
            missing values, duplicates, currency formats, outliers, and AI-powered imputation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_pills(df):
    n_missing = int(df.isnull().sum().sum())
    n_dupes   = int(df.duplicated().sum())
    st.markdown(
        f"""
        <div style="margin: 0.5rem 0 1rem 0;">
            <span class="stat-pill">Rows: <b>{len(df):,}</b></span>
            <span class="stat-pill">Columns: <b>{len(df.columns)}</b></span>
            <span class="stat-pill">Missing cells: <b>{n_missing:,}</b></span>
            <span class="stat-pill">Duplicate rows: <b>{n_dupes:,}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_styles()
    render_header()
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel files are supported.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file above to get started.")
        return

    if (
        "raw_df" not in st.session_state
        or st.session_state.get("uploaded_filename") != uploaded_file.name
    ):
        try:
            st.session_state["raw_df"]           = load_uploaded_file(uploaded_file)
            st.session_state["uploaded_filename"] = uploaded_file.name
            for key in ["cleaned_df", "cleaning_report", "domain_result", "columns_to_drop"]:
                st.session_state.pop(key, None)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

    raw_df = st.session_state["raw_df"]

    st.subheader("📋 Raw Dataset Preview")
    render_stat_pills(raw_df)
    st.dataframe(raw_df.head(15), use_container_width=True, height=320)

    st.divider()

    # ── Column removal ────────────────────────────────────────────────────────
    st.subheader("🗑️ Select Columns to Remove")
    st.caption("Tick any columns you don't want included in the analysis (e.g. IDs, PII, irrelevant fields).")

    if "columns_to_drop" not in st.session_state:
        st.session_state["columns_to_drop"] = set()

    with st.container():
        cols_per_row  = 4
        columns_list  = list(raw_df.columns)
        checkbox_states = {}

        for i in range(0, len(columns_list), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col_name in enumerate(columns_list[i:i + cols_per_row]):
                with row_cols[j]:
                    checked = st.checkbox(
                        col_name,
                        value=col_name in st.session_state["columns_to_drop"],
                        key=f"chk_{col_name}",
                    )
                    checkbox_states[col_name] = checked

    selected_to_drop = {c for c, checked in checkbox_states.items() if checked}
    st.session_state["columns_to_drop"] = selected_to_drop

    col_remove, col_count = st.columns([1, 3])
    with col_remove:
        remove_clicked = st.button("🗑️ Remove Selected Columns", use_container_width=True)
    with col_count:
        if selected_to_drop:
            st.caption(f"Selected for removal: {', '.join(sorted(selected_to_drop))}")
        else:
            st.caption("No columns selected for removal.")

    if remove_clicked:
        if selected_to_drop:
            st.session_state["raw_df"]           = raw_df.drop(columns=list(selected_to_drop))
            st.session_state["columns_to_drop"]  = set()
            st.success(f"Removed {len(selected_to_drop)} column(s).")
            st.rerun()
        else:
            st.warning("No columns selected.")

    st.divider()

    # ── Clean button ──────────────────────────────────────────────────────────
    st.subheader("🤖 Run Cleaning Agent")
    st.caption(
        "Standardizes formats, fixes data types, removes duplicates, strips currency symbols, "
        "predicts missing values with Random Forest imputation, and caps outliers — automatically."
    )

    clean_clicked = st.button("✨ Clean Dataset", type="primary", use_container_width=True)

    if len(raw_df) > 50_000:
        st.caption(
            f"⏳ {len(raw_df):,} rows detected — large-dataset optimizations "
            "(capped LLM calls, sampled model training, bounded row-level parsing) "
            "are active. This may still take a minute or two."
        )

    if clean_clicked:
        current_df = st.session_state["raw_df"]
        start_time = time.time()
        try:
            with st.spinner(
                "Running adaptive cleaning pipeline (profiling → adaptive imputation → "
                "ensemble outlier voting)... large files can take a minute or two."
            ):
                cleaner    = DataCleaner(current_df)
                cleaned_df, report = cleaner.clean()

            st.session_state["cleaned_df"]      = cleaned_df
            # column_profile / column_classification are now populated for real
            # (previously an empty dict), so domain_detector.py gets actual
            # evidence to reason over instead of guessing from raw columns alone.
            st.session_state["cleaning_report"] = {
                "log":                   report,
                "column_profile":        cleaner.get_profile(),
                "column_classification": cleaner.get_classification(),
            }
            st.session_state["cleaning_scorecard"] = cleaner.get_scorecard()
            st.session_state["near_duplicates"]    = cleaner.get_near_duplicates()
            elapsed = time.time() - start_time
            st.success(f"Dataset cleaned successfully in {elapsed:.1f}s.")
        except Exception as e:
            # A cleaning failure previously had no safety net here — an
            # uncaught exception mid-pipeline could leave the app in a
            # broken state with no explanation shown to the user (the
            # generic Streamlit Cloud "no response from server" screen).
            # Surface it explicitly instead, and never leave stale/partial
            # session state behind.
            for key in ["cleaned_df", "cleaning_report", "cleaning_scorecard", "near_duplicates"]:
                st.session_state.pop(key, None)
            st.error(f"❌ Cleaning failed: {e}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            st.info(
                "This is usually caused by an unusual column format. Try removing "
                "the flagged column above (in '🗑️ Select Columns to Remove') and "
                "re-running, or reduce the file size and try again."
            )

    # ── Results ───────────────────────────────────────────────────────────────
    if "cleaned_df" in st.session_state:
        st.divider()
        st.subheader("✅ Cleaned Dataset Preview")
        render_stat_pills(st.session_state["cleaned_df"])
        st.dataframe(st.session_state["cleaned_df"].head(15), use_container_width=True, height=320)

        # ── Cleaning Impact Scorecard ────────────────────────────────────────
        scorecard = st.session_state.get("cleaning_scorecard")
        if scorecard:
            st.subheader("📊 Data Health Scorecard")
            before, after = scorecard["before"], scorecard["after"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Overall Grade", f"{before['grade']} → {after['grade']}",
                       f"+{scorecard['delta_points']} pts" if scorecard['delta_points'] >= 0 else f"{scorecard['delta_points']} pts")
            c2.metric("Completeness", f"{after['completeness_pct']}%", f"{after['completeness_pct']-before['completeness_pct']:+.1f}")
            c3.metric("Uniqueness", f"{after['uniqueness_pct']}%", f"{after['uniqueness_pct']-before['uniqueness_pct']:+.1f}")
            c4.metric("Validity", f"{after['validity_pct']}%", f"{after['validity_pct']-before['validity_pct']:+.1f}")
            c5.metric("Consistency", f"{after['consistency_pct']}%", f"{after['consistency_pct']-before['consistency_pct']:+.1f}")

            near_dupes = st.session_state.get("near_duplicates", [])
            if near_dupes:
                with st.expander(f"🔍 {len(near_dupes)} possible near-duplicate row pair(s) — review only, nothing was auto-removed", expanded=False):
                    st.caption(
                        "These rows are highly similar but not identical (e.g. a typo'd name or a re-keyed "
                        "field). They were intentionally NOT auto-deleted — collapsing two genuinely distinct "
                        "records is worse than leaving a near-duplicate in place. Review manually if needed."
                    )
                    st.dataframe(pd.DataFrame(near_dupes[:50]), use_container_width=True, height=200)

        with st.expander("📜 Full Cleaning Report", expanded=False):
            for line in st.session_state["cleaning_report"]["log"]:
                st.text(line)

        st.divider()
        if st.button("Continue to Domain Detection ➜", type="primary"):
            st.switch_page("pages/2_domain_detector.py")


if __name__ == "__main__":
    main()