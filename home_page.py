"""
home_page.py — Home/upload screen for AI Data Cleaner & Analyzer

Rendered as the default page via st.navigation() in app.py. The
project has no sidebar at all (removed at the source, not hidden via
CSS), so there's no initial_sidebar_state to configure here.
"""

from dashboard_agent import DashboardAgent
from dashboard_renderer import render_dashboard
from auth import render_login_page, check_authentication, render_logout
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
from crew import run_crew
import re
from vector_store import store_report
import polars as pl

# ── Page config — must be first st call ───────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Data Analyzer",
    page_icon="👨‍💻",
    layout="wide",
)

# ── AUTH — blocks everything below until signed in ────────────────────────────
authenticator, config = render_login_page()
is_authenticated, name, username = check_authentication(authenticator, config)

if not is_authenticated:
    st.stop()

render_logout(authenticator, name)


# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0a0f; color: #e2e8f0; }
h1 {
    background: linear-gradient(90deg, #00f5ff, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.4rem !important; font-weight: 800 !important;
}
h2, h3 { color: #00f5ff !important; }
h4 { color: #a78bfa !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #00f5ff33; border-radius: 12px;
    padding: 16px; box-shadow: 0 0 20px #00f5ff22;
}
[data-testid="metric-container"] label {
    color: #00f5ff !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00f5ff22, #7c3aed22) !important;
    border: 1px solid #00f5ff !important; color: #00f5ff !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#00f5ff44,#7c3aed44) !important;
    box-shadow: 0 0 20px #00f5ff44 !important;
}
[data-testid="baseButton-primary"] > button {
    background: linear-gradient(135deg,#00f5ff,#7c3aed) !important;
    color: #000 !important; border: none !important; font-weight: 700 !important;
}
[data-testid="stFileUploader"] {
    background: #1a1a2e; border: 2px dashed #00f5ff44; border-radius: 12px; padding: 20px;
}
.stDataFrame { border: 1px solid #00f5ff22; border-radius: 8px; }
.stCheckbox label { color: #e2e8f0 !important; }
hr { border-color: #00f5ff22 !important; }

/* ── FAB wrapper — fixed bottom-right ───────────────── */
div[data-testid="stButton"].ask-ai-fab {
    position: fixed !important;
    bottom: 32px !important;
    right: 32px !important;
    z-index: 9999 !important;
    margin: 0 !important;
}
div[data-testid="stButton"].ask-ai-fab > div > button {
    background: linear-gradient(135deg, #00f5ff, #7c3aed) !important;
    color: #0a0a0f !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 14px 22px !important;
    border-radius: 50px !important;
    border: none !important;
    box-shadow: 0 4px 24px #00f5ff55, 0 2px 8px #0005 !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"].ask-ai-fab > div > button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 6px 32px #00f5ffaa !important;
}
</style>
""", unsafe_allow_html=True)

# ── Ask AI FAB button (switch_page — always works regardless of URL routing) ──
st.markdown('<div class="ask-ai-fab" data-testid="stButton">', unsafe_allow_html=True)
if st.button("💬Ask AI", key="_fab_ask_ai"):
    st.switch_page("pages/3_chat_reports.py")
st.markdown('</div>', unsafe_allow_html=True)

# Also inject JS to float the button — belt-and-suspenders for the CSS above
st.markdown("""
<script>
(function floatAskAI() {
    function apply() {
        var btns = window.parent.document.querySelectorAll("button");
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].innerText && btns[i].innerText.trim().startsWith("💬 Ask AI")) {
                var w = btns[i].closest('[data-testid="stButton"]') || btns[i].parentElement.parentElement;
                w.style.cssText = "position:fixed!important;bottom:32px!important;right:32px!important;z-index:9999!important;margin:0!important;";
                btns[i].style.cssText = "background:linear-gradient(135deg,#00f5ff,#7c3aed)!important;color:#0a0a0f!important;font-weight:800!important;font-size:0.95rem!important;padding:14px 22px!important;border-radius:50px!important;border:none!important;box-shadow:0 4px 24px #00f5ff55!important;cursor:pointer!important;";
                break;
            }
        }
    }
    setTimeout(apply, 400);
    setTimeout(apply, 1200);
    setTimeout(apply, 2500);
})();
</script>
""", unsafe_allow_html=True)


# ── Session State — init defaults only if not already set (preserves progress) ─
for k, v in {
    "cleaned_df": None, "report_text": None, "cleaning_report": None,
    "show_dashboard": False, "dataset_name": None,
    "cols_to_remove": [], "df_original": None, "verified_stats": {},
    "business_summary": "", "domain_config": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Restore progress banner — shown when user returns after browser close ──────
if st.session_state.get("cleaned_df") is not None and st.session_state.get("dataset_name"):
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#0f2027,#203a43);
                    border:1px solid #00f5ff44;border-left:4px solid #00f5ff;
                    border-radius:10px;padding:12px 18px;margin-bottom:16px;
                    display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.3rem;">🔄</span>
            <div>
                <div style="color:#00f5ff;font-weight:700;font-size:0.9rem;">Previous session restored</div>
                <div style="color:#94a3b8;font-size:0.8rem;">
                    Dataset: <b style="color:#e2e8f0">{st.session_state.dataset_name}</b>
                    &nbsp;·&nbsp; Your analysis results are still available below.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧑‍💼 Multi-Agent Data Analyzer")
st.markdown("Upload any messy dataset — AI will **clean**, **analyze**, and generate **business insights** automatically.")
st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload Dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > 300:
        st.error(f"File is {size_mb:.0f}MB — this deployment supports up to 300MB. Try filtering or sampling the data first.")
        st.stop()

    st.session_state.dataset_name = uploaded_file.name.rsplit(".", 1)[0]

    if uploaded_file.name.endswith(".csv"):
        lf = pl.scan_csv(uploaded_file)
        preview_df = lf.head(20).collect().to_pandas()
        df = lf.collect(streaming=True).to_pandas()
    else:
        # polars excel read is eager only, no lazy scan for xlsx
        df = pl.read_excel(uploaded_file).to_pandas()
        preview_df = df.head(20)

    st.session_state.df_original = df

    none_count = sum((df == s).sum().sum() for s in ["None", "none", "NULL", "null", "nan", "NaN", "NA", "N/A"])
    actual_missing = df.isnull().sum().sum() + none_count

    st.subheader("📋 Raw Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, height=380)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{df.shape[0]:,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", f"{actual_missing:,}")
    c4.metric("Duplicate Rows", df.duplicated().sum())

    st.divider()

    if st.button("🚀 Clean & Analyze Dataset", type="primary", use_container_width=True):
        with st.spinner("⚙️ AI agents cleaning and analyzing your data..."):
            try:
                cleaned_df, cleaning_report, report_text, verified_stats, business_summary, domain_config = run_crew(
                    df,
                    st.session_state.dataset_name,
                )
                st.session_state.cleaned_df       = cleaned_df
                st.session_state.cleaning_report  = cleaning_report
                st.session_state.report_text      = report_text
                st.session_state.verified_stats   = verified_stats
                st.session_state.business_summary = business_summary
                st.session_state.domain_config     = domain_config   # NEW
                st.session_state.cols_to_remove   = []
                st.session_state.show_dashboard   = False
                st.session_state.pop("chart_specs", None)
                st.session_state.pop("chart_specs_for", None)

                # ── NEW: free the raw upload once cleaning succeeded ──────
                if "df_original" in st.session_state:
                    del st.session_state["df_original"]
                    import gc; gc.collect()
                # ────────────────────────────────────────────────────────

                st.success("✅ Done! Dataset cleaned and analyzed.")

                chunks = store_report(
                    report_text=report_text,
                    dataset_name=st.session_state.dataset_name,
                    username=st.session_state["auth_username"],
                )
                st.success(f"💾 Report saved to memory — {chunks} chunks indexed.")
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
                st.exception(e)

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.cleaned_df is not None:
    cleaned_df      = st.session_state.cleaned_df
    report_text     = st.session_state.report_text
    cleaning_report = st.session_state.cleaning_report
    dataset_name    = st.session_state.dataset_name or "dataset"

    st.divider()
    st.subheader("✅ Cleaned Data Preview")

    if st.session_state.cols_to_remove:
        st.info(f"🗑️ Removed after cleaning: `{'`, `'.join(st.session_state.cols_to_remove)}`")

    st.dataframe(cleaned_df.head(20), use_container_width=True)

    with st.expander("🗑️ Remove columns from this dataset", expanded=False):
        st.caption(
            "Removing a column here updates the cleaned data, KPIs, and "
            "trends used by the dashboard. It does not re-run the AI "
            "cleaning/analysis pipeline."
        )
        cols_to_remove = st.multiselect(
            "Columns to remove",
            options=cleaned_df.columns.tolist(),
            default=[],
            key="post_clean_col_select",
        )
        if st.button("🗑️ Remove Selected Columns", type="primary", disabled=not cols_to_remove):
            new_cleaned_df = cleaned_df.drop(columns=cols_to_remove, errors="ignore")
            st.session_state.cleaned_df    = new_cleaned_df
            st.session_state.cols_to_remove = list(
                set(st.session_state.cols_to_remove) | set(cols_to_remove)
            )
            from stats_engine import StatsEngine
            st.session_state.verified_stats = StatsEngine(new_cleaned_df).generate_full_report()
            st.session_state.show_dashboard = False
            st.session_state.pop("chart_specs", None)
            st.session_state.pop("chart_specs_for", None)
            st.success(f"Removed {len(cols_to_remove)} column(s). KPIs and trends recalculated.")
            st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows After Cleaning", f"{cleaned_df.shape[0]:,}")
    c2.metric("Columns", cleaned_df.shape[1])
    c3.metric("Missing Values", cleaned_df.isnull().sum().sum())
    c4.metric("None Strings Left", int((cleaned_df == "None").sum().sum()))

    st.divider()

    with st.expander("🔍 View Cleaning Log", expanded=False):
        for line in cleaning_report:
            if "[VERDICT]" in line:
                msg = line.replace("[VERDICT]", "").strip()
                if "no cleaning" in msg.lower():
                    st.success(msg)
                elif "🔧" in msg:
                    st.warning(msg)
                else:
                    st.info(msg)
            elif "[START]" in line or "[END]" in line or "─" in line:
                st.markdown(f"**{line}**")
            elif "✅" in line:
                st.markdown(f"🟢 {line}")
            elif "⚠" in line:
                st.markdown(f"🟡 {line}")
            elif "✔" in line:
                st.markdown(f"⚪ {line}")
            elif "[SUMMARY]" in line:
                st.markdown(f"📊 {line}")
            else:
                st.markdown(f"• {line}")

    st.divider()

    st.subheader("📝 AI Analysis Report")
    st.markdown(report_text)
    st.divider()

    c1, c2, c3 = st.columns(3)

    # ── PDF Download ───────────────────────────────────────────────────────────
    
    with c1:
        if st.button("📄 Download PDF Report", use_container_width=True):
            try:
                from pdf_generator import generate_pdf_bytes
                pdf_bytes = generate_pdf_bytes(report_text, dataset_name)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"{dataset_name}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                import traceback
                st.error(f"PDF generation failed: {e}")
                st.code(traceback.format_exc())

    # ── Dashboard ──────────────────────────────────────────────────────────────
    with c2:
        if st.button("📊 Generate Smart Dashboard", type="primary", use_container_width=True):
            st.session_state.show_dashboard = True

    # ── CSV Download ───────────────────────────────────────────────────────────
    with c3:
        st.download_button(
            "⬇️ Download Cleaned CSV",
            cleaned_df.to_csv(index=False),
            f"{dataset_name}_cleaned.csv", "text/csv",
            use_container_width=True
        )

# ── Smart Dashboard ───────────────────────────────────────────────────────────
if st.session_state.show_dashboard and st.session_state.cleaned_df is not None:
    cleaned_df       = st.session_state.cleaned_df
    report_text      = st.session_state.report_text or ""
    dataset_name     = st.session_state.dataset_name or "dataset"
    verified_stats   = st.session_state.get("verified_stats", {})
    business_summary = st.session_state.get("business_summary", "")

    st.divider()
    st.subheader("📊 Smart Business Dashboard")
    st.caption(
        "Charts are selected by an AI agent based on verified statistics, "
        "machine learning findings, and the business report — only "
        "findings that actually matter get a chart."
    )

    if "chart_specs" not in st.session_state or st.session_state.get("chart_specs_for") != dataset_name:
        with st.spinner("🤖 Dashboard agent is deciding which charts matter..."):
            try:
                dashboard_agent = DashboardAgent()
                chart_specs = dashboard_agent.generate_chart_specs(
                    verified_stats=verified_stats,
                    report_text=report_text,
                    available_columns=cleaned_df.columns.tolist(),
                )
                st.session_state.chart_specs     = chart_specs
                st.session_state.chart_specs_for = dataset_name
            except Exception as e:
                st.error(f"Dashboard agent failed: {e}")
                st.session_state.chart_specs = []

    chart_specs = st.session_state.get("chart_specs", [])

    if chart_specs:
        with st.expander(f"🧠 Why these {len(chart_specs)} charts were chosen", expanded=False):
            for spec in chart_specs:
                st.markdown(
                    f"**{spec.get('title', 'Untitled')}** ({spec.get('chart_type')}) — "
                    f"{spec.get('reasoning', 'No reasoning provided.')}"
                )

    render_dashboard(
        cleaned_df, chart_specs, verified_stats,
        business_summary=business_summary,
        domain_config=st.session_state.get("domain_config", {}),
    )

    if verified_stats.get("quality_score"):
        st.divider()
        st.markdown("### 🧹 Verified Data Quality Score")
        qs = verified_stats["quality_score"]
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Overall Score", f"{qs['overall_score']} / 100", qs["grade"])
        qc2.metric("Completeness", f"{qs['completeness_pct']}%")
        qc3.metric("Uniqueness",   f"{qs['uniqueness_pct']}%")
        qc4.metric("Validity",     f"{qs['validity_pct']}%")

    st.divider()
    if st.button("🔄 Regenerate Dashboard", use_container_width=True):
        st.session_state.pop("chart_specs", None)
        st.session_state.pop("chart_specs_for", None)
        st.rerun()

    st.divider()
    st.download_button(
        "⬇️ Download Cleaned Dataset (CSV)",
        cleaned_df.to_csv(index=False),
        f"{dataset_name}_cleaned.csv", "text/csv",
        use_container_width=True
    )