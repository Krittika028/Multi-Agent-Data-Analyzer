"""
crew.py
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

    # Step 2 — Compute VERIFIED statistics deterministically
    stats_engine = StatsEngine(cleaned_df)
    verified_stats = stats_engine.generate_full_report()

    # Step 3 — Build context strings
    dataset_info = f"""
    Dataset Name   : {dataset_name}
    Original Shape : {df.shape[0]:,} rows x {df.shape[1]} columns
    Columns        : {', '.join(df.columns.tolist())}
    """

    cleaning_verdict = next(
        (r for r in cleaning_report if '[VERDICT]' in r), 'No verdict available'
    )

    cleaning_context = f"""
    Cleaned Shape  : {cleaned_df.shape[0]:,} rows x {cleaned_df.shape[1]} columns
    Columns        : {', '.join(cleaned_df.columns.tolist())}

    Cleaning Verdict:
    {cleaning_verdict}

    Data Quality Score (verified):
    {verified_stats.get('quality_score', 'Not computed')}
    """

    # Step 4 — Create tasks
    cleaning_task = get_cleaning_task(cleaning_report, dataset_info)
    analysis_task = get_analysis_task(verified_stats, cleaning_context)
    report_task   = get_report_task(
        analysis_context="[See context from previous tasks]",
        dataset_name=dataset_name,
    )
    summary_task  = get_business_summary_task(verified_stats, dataset_name)

    # Report agent sees both cleaning + analysis context.
    # Summary task only needs the analysis context — it stays short and
    # independent of the full 8-section report.
    report_task.context = [cleaning_task, analysis_task]
    summary_task.context = [analysis_task]

    # Step 5 — Assemble agents, optionally overriding the LLM per the
    # sidebar's "AI Model" dropdown (st.session_state.selected_model)
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


    # Pull each task's own output directly rather than relying solely on
    # the crew's final result, since report_task and summary_task are
    # independent leaves (summary_task isn't "after" report_task).
    report_text      = str(report_task.output) if report_task.output else str(result)
    business_summary = str(summary_task.output) if summary_task.output else ""

    # Step 6 — Return exactly 5 values
    return cleaned_df, cleaning_report, report_text, verified_stats, business_summary