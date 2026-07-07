"""
crew.py

Updated: computes an explicit, independent row-retention check
(original rows vs. cleaned rows) and threads it through to both
get_analysis_task and get_business_summary_task, so the reporting
layer has its own tripwire for silent row loss — independent of
whatever data_cleaner.py's internal safety gates already caught.
"""

from crewai import Crew, Process
from tasks import (
    get_cleaning_task,
    get_analysis_task,
    get_report_task,
    get_business_summary_task,
)
from agents import get_cleaner_agent, get_analyst_agent, get_report_agent
from data_cleaner import DataCleaner
from stats_engine import StatsEngine
import pandas as pd
from domain_detector import DomainDetector
from domain_context import get_domain_config, classify_status_values


def run_crew(df: pd.DataFrame, dataset_name: str, columns_to_drop=None, model=None):
    """
    Returns exactly 5 values:
        cleaned_df, cleaning_report, report_text, verified_stats, business_summary

    columns_to_drop : iterable of column names to drop before cleaning
                       (e.g. st.session_state.cols_to_remove)
    model           : optional LLM identifier to override the default
                       agent.llm (e.g. st.session_state.selected_model)
    """

    # Step 1 — Clean the data (DataCleaner.clean accepts a list of columns to drop)
    cleaner = DataCleaner(df)
    cleaned_df, cleaning_report = cleaner.clean(columns_to_drop or [])

    # ── Independent row-retention check — computed here, NOT trusted from
    # the cleaner's internal log, so a future cleaning bug can't bypass
    # this tripwire the same way it bypassed the previous dedup logic. ────
    original_rows    = df.shape[0]
    cleaned_rows     = cleaned_df.shape[0]
    retention_pct    = round((cleaned_rows / original_rows * 100), 1) if original_rows else 100.0
    rows_dropped_pct = round(100 - retention_pct, 1)
    
    # ── Domain detection — resolved ONCE per run, feeds every downstream
    # consumer (stats engine's status classification, dashboard's KPI
    # card 4 selection, and every report/task section naming). ──────────
    try:
        detector = DomainDetector()
        domain_result = detector.detect(cleaned_df, {"log": cleaning_report})
    except Exception:
        domain_result = {}
    domain_config = get_domain_config(domain_result)

    # ── Status classification — LLM semantic pass over the ACTUAL status
    # column values in THIS dataset, replacing the old commerce-keyword
    # match so it works correctly on IT/HR/healthcare status columns too.
    stats_engine_preview = StatsEngine(cleaned_df)
    status_col = stats_engine_preview._find_status_column()
    status_classification = None
    if status_col:
        unique_status_vals = cleaned_df[status_col].dropna().astype(str).unique().tolist()
        status_classification = classify_status_values(status_col, unique_status_vals)

    # Step 2 — Compute VERIFIED statistics deterministically, now with
    # domain-aware status classification passed through.
    stats_engine = StatsEngine(cleaned_df)
    verified_stats = stats_engine.generate_full_report(status_classification)

    dataset_info = f"""
    Dataset Name   : {dataset_name}
    Original Shape : {df.shape[0]:,} rows x {df.shape[1]} columns
    Columns        : {', '.join(df.columns.tolist())}
    Detected Domain: {domain_config['domain']} (primary entity: {domain_config['primary_entity']})
    """

    cleaning_verdict = next(
        (r for r in cleaning_report if '[VERDICT]' in r), 'No verdict available'
    )

    retention_flag_line = (
        f"⚠️ {rows_dropped_pct}% of rows were removed during cleaning — "
        f"this exceeds the 10% reliability threshold and MUST be flagged "
        f"prominently in the analysis and report."
        if rows_dropped_pct > 10 else
        f"✅ Row retention is healthy ({retention_pct}% of original rows kept)."
    )

    cleaning_context = f"""
    Cleaned Shape  : {cleaned_df.shape[0]:,} rows x {cleaned_df.shape[1]} columns
    Columns        : {', '.join(cleaned_df.columns.tolist())}

    ROW RETENTION CHECK (independent of the cleaning pipeline's own logs):
    Original rows : {original_rows:,}
    Cleaned rows  : {cleaned_rows:,}
    Retention     : {retention_pct}%
    Dropped       : {rows_dropped_pct}%
    {retention_flag_line}

    Cleaning Verdict:
    {cleaning_verdict}

    Data Quality Score (verified):
    {verified_stats.get('quality_score', 'Not computed')}
    """

    cleaning_task = get_cleaning_task(cleaning_report, dataset_info)
    analysis_task = get_analysis_task(verified_stats, cleaning_context, rows_dropped_pct, domain_config)
    report_task   = get_report_task(
        analysis_context="[See context from previous tasks]",
        dataset_name=dataset_name,
        domain_config=domain_config,
    )
    summary_task  = get_business_summary_task(verified_stats, dataset_name, rows_dropped_pct, domain_config)

    report_task.context = [cleaning_task, analysis_task]
    summary_task.context = [analysis_task]

    agents = [get_cleaner_agent(), get_analyst_agent(), get_report_agent()]
    if model:
        for agent in agents:
            agent.llm = model

    crew = Crew(
        agents=agents,
        tasks=[cleaning_task, analysis_task, report_task, summary_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    report_text      = str(report_task.output) if report_task.output else str(result)
    business_summary = str(summary_task.output) if summary_task.output else ""

    # Step 6 — Return 6 values now (added domain_config) so home_page.py
    # can pass it to render_dashboard for domain-aware KPI cards.
    return cleaned_df, cleaning_report, report_text, verified_stats, business_summary, domain_config