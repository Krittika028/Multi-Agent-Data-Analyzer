"""
agents.py
"""

from crewai import Agent, LLM
from dotenv import load_dotenv
import os
import time
import litellm

load_dotenv()

# Retry settings
# NOTE (perf/reliability fix): the old settings (5 retries, 10s initial wait,
# doubling to 10/20/40/80s) meant a single struggling call could legitimately
# take up to ~150s of pure sleep before even giving up — and with several
# sequential crew tasks (cleaner → analyst → report → summary agents) that
# compounds into multi-minute requests. On Streamlit Cloud that reads as a
# hung app ("Received no response from server"), not a clean error. Reduced
# to a tighter, still-generous budget, and every completion call now carries
# an explicit request timeout so a hanging connection fails fast instead of
# blocking forever.
MAX_RETRIES = 3
INITIAL_WAIT = 5   # seconds — backoff: 5s → 10s → 20s (~35s total worst case)
REQUEST_TIMEOUT = 60  # seconds per individual attempt


def _llm_completion_with_retry(model, messages, max_tokens=1000, temperature=0.7):
    """
    Wraps litellm.completion with exponential backoff for 503/rate-limit errors.
    """
    wait = INITIAL_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as e:
            err = str(e).lower()
            is_retryable = any(x in err for x in [
                "503", "unavailable", "rate limit", "rate_limit",
                "429", "overloaded", "capacity", "quota", "resource exhausted"
            ])
            if is_retryable and attempt < MAX_RETRIES:
                print(f"⚠ Attempt {attempt}/{MAX_RETRIES} failed: {str(e)[:80]}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
                wait *= 2  # exponential backoff: 10 → 20 → 40 → 80s
            else:
                raise


def _build_llm(model_override=None):
    model    = model_override or os.getenv("MODEL")
    api_key  = os.getenv("OPENAI_API_KEY")
    return LLM(
        model=model,
        api_key=api_key,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT,
    )


def get_cleaner_agent(model=None):
    return Agent(
        role="Senior Data Quality & Business Readiness Specialist",
        goal=(
            "Translate the raw cleaning log into a clear business-impact "
            "summary: what data quality issues were found, exactly what was "
            "fixed, and how each issue could have affected business decisions "
            "if left uncleaned. Always quote exact counts and column names. "
            "Frame quality issues in terms of downstream business risk."
        ),
        backstory=(
            "You are a Data Quality lead with 12 years of experience "
            "preparing datasets for C-suite dashboards across retail, "
            "finance, healthcare, and SaaS. You've seen analysts make "
            "costly decisions based on unclean data — a CFO who under-ordered "
            "inventory because nulls were silently excluded, a marketing team "
            "that over-reported conversion rates because duplicates weren't "
            "caught.\n\n"
            "You don't just log what was cleaned — you explain why it mattered. "
            "You speak in business terms: revenue impact, decision risk, "
            "reporting accuracy. You give a numeric data quality score out of "
            "100 and justify every point deducted with a specific issue from "
            "the cleaning log. You end with a clear business readiness verdict: "
            "Ready / Needs Review / High Risk."
        ),
        verbose=False,
        allow_delegation=False,
        llm=_build_llm(model),
    )


def get_analyst_agent(model=None):
    return Agent(
        role="Senior Business Intelligence Analyst",
        goal=(
            "Interpret a pre-computed, verified statistics package and produce "
            "a structured business analysis with tables, category breakdowns, "
            "behavioral insights, and concrete recommendations. "
            "Your output must look like a professional analyst report: "
            "markdown tables for KPIs and breakdowns, numbered insights, "
            "and specific actionable recommendations tied to exact numbers. "
            "Every figure cited must come directly from the verified stats. "
            "Write the way a real analyst would — specific, tabular, grounded. "
            "Never mention data cleaning, missing values, imputation, or "
            "data quality in your output — focus only on business performance."
        ),
        backstory=(
            "You are a BI analyst who has built reports for retail chains, "
            "F&B businesses, e-commerce platforms, and SaaS companies. "
            "Your reports always include: a KPI summary table, category or "
            "product breakdown tables, behavioral pattern analysis, channel "
            "splits, and time-series trend tables. You write for operations "
            "managers and business owners who want to know exactly which "
            "product is underperforming, which channel is growing, and what "
            "to do about it.\n\n"
            "Your personal rule: every number you cite must be traceable to "
            "the verified stats package. You never say 'significant' without "
            "a number. You never say 'trending' without a direction. "
            "When a category has high volume but low revenue, you flag it as "
            "a cross-sell opportunity. When revenue is flat, you say 'stable "
            "but stagnant' and recommend a specific growth lever. "
            "You never mention anything about how the data was cleaned, "
            "processed, or what values were missing — that is not your job. "
            "You write only about business performance."
        ),
        verbose=False,
        allow_delegation=False,
        llm=_build_llm(model),
    )


def get_report_agent(model=None):
    return Agent(
        role="Business Intelligence Report Writer",
        goal=(
            "Take the analyst's findings and format them into a clean, "
            "well-structured final business performance report. "
            "Preserve all numbers exactly. Keep all markdown tables. "
            "The report must contain ONLY business metrics, trends, and "
            "recommendations — zero mention of data cleaning, data quality, "
            "missing values, imputation, or anything about how the data "
            "was processed. If the analyst included any such references, "
            "remove them entirely. "
            "The final report should read like a professional business "
            "performance deliverable — structured, specific, and actionable."
        ),
        backstory=(
            "You format and enhance BI reports for business owners, ops "
            "managers, and department heads. Your job is to make the "
            "analyst's findings as clear and readable as possible — and to "
            "ensure the report stays strictly focused on business performance. "
            "You remove any reference to data processing, cleaning steps, "
            "or technical data quality issues. Those belong in a separate "
            "technical log, not a business report.\n\n"
            "You ensure tables are properly formatted, sections are logically "
            "ordered, insights are crisp, and recommendations are concrete "
            "and tied to specific numbers. You never use abstract language "
            "like 'leverage synergies'. You write the way a good analyst "
            "talks: 'Salads generate 19.5% of total revenue — promote them "
            "more prominently'. Every recommendation starts with an action "
            "verb and ends with an expected outcome."
        ),
        verbose=False,
        allow_delegation=False,
        llm=_build_llm(model),
    )