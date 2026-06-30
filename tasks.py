"""
tasks.py
"""

import json
from crewai import Task
from agents import get_cleaner_agent, get_analyst_agent, get_report_agent


def get_cleaning_task(cleaning_report: list, dataset_info: str):
    return Task(
        description=f"""
You have received a data cleaning report from an automated pipeline.
Your job is to interpret what was found and fixed, and explain what 
each issue means for business decisions.

Dataset Info:
{dataset_info}

Cleaning Report (automated pipeline output):
{chr(10).join(cleaning_report)}

Your output must include:

**1. Business Readiness Verdict**
One of: Ready for Analysis | Proceed with Caution | High Risk
Justify in one sentence using specific counts from the report.

**2. Data Quality Issues — Business Risk Translation**
For every issue found, answer: "If this had NOT been fixed, which business
decision would have been wrong, and how wrong?"
Be specific — e.g. "45 missing revenue values would have understated average
order value by ~12%, causing the pricing team to set margins too low."

**3. What the Cleaning Reveals**
What do these patterns suggest about the business process that generated this data?
- High null rate → data was never collected, or a system change occurred
- Currency format errors → data sourced from multiple regions/systems
- Clustered outliers → possible fraud, VIP segment, or a one-time event

**4. Actions Taken (exact)**
Bullet list with exact counts — "RF-imputed 45 missing values in 'revenue'"
not "handled missing data."

**5. Data Quality Score**
Score out of 100. Show deductions for each specific issue.
Grade: A (90+) / B (80+) / C (70+) / D (60+) / F (<60).

**6. Analyst Warnings**
Flag ML-imputed columns, heavily cleaned columns, and any columns that
still carry uncertainty. Tell the analyst how to treat each one.
        """,
        agent=get_cleaner_agent(),
        expected_output=(
            "A business-risk data quality report: readiness verdict, "
            "per-issue business impact translation, interpretation of "
            "what the data patterns reveal, exact action counts, "
            "quality score with deductions, and analyst warnings."
        ),
    )


def get_analysis_task(verified_stats: dict, cleaning_context: str):
    stats_json = json.dumps(verified_stats, indent=2, default=str)

    return Task(
        description=f"""
You are a Senior BI Analyst. Below is a VERIFIED STATISTICS PACKAGE
computed deterministically from the dataset. Every number is exact.
DO NOT recompute, estimate, or invent any figures.

VERIFIED STATISTICS (ground truth — use only these numbers):
{stats_json}

Cleaning Context:
{cleaning_context}

Your output must follow this structure EXACTLY:

**Executive Summary**
- Key numeric totals (e.g. Total Revenue, Total Transactions)
- Average Order Value or equivalent primary metric
- 3–4 bullet points summarizing the most important patterns
- Keep it factual and concise

**Core Business KPIs**
Present as a markdown table:
| KPI | Value |
Include: totals, averages, highest/lowest performing category, best/worst period.

**Category / Product / Segment Performance** (use whatever dimension fits the data)
- Revenue or volume breakdown per category as a markdown table
- Revenue per unit or per transaction as a separate markdown table
- 1–2 sentence insight below each table explaining what it means

**Customer / Behavioral Patterns** (if applicable)
- Volume leaders vs revenue leaders — are they the same?
- What does high volume but low revenue suggest?
- What opportunities does this create?

**Channel / Segment Split** (if applicable)
- Breakdown table by channel, region, or segment
- Is revenue balanced or concentrated?
- What does the distribution suggest?

**Time / Period Analysis**
- Monthly, quarterly, or yearly breakdown table
- Identify best and worst periods with exact numbers
- Is the trend growing, declining, or flat?

**Top 5 Business Insights**
Numbered list. Each insight must:
- State an exact number from the verified stats
- Explain what it means for the business in plain English
- Name one specific opportunity or risk it creates

**Strategic Recommendations**
Exactly 5 recommendations. Each must:
- Be grounded in a specific number from the analysis
- Name a concrete action (e.g. "Bundle Coffee + Cake to increase basket size")
- State the expected business outcome
- Be written in plain operational language — no jargon

**Management Takeaway**
2–3 sentences summarizing the overall picture and the single most 
important action the business should take right now.
        """,
        agent=get_analyst_agent(),
        expected_output=(
            "A structured BI analysis with executive summary, KPI table, "
            "category/segment breakdown tables, behavioral insights, "
            "channel split, time analysis, top 5 insights, 5 recommendations, "
            "and a management takeaway — all grounded in exact verified numbers."
        ),
    )


def get_report_task(analysis_context: str, dataset_name: str):
    return Task(
        description=f"""
You are writing a business performance report for operations managers
and business owners.

Dataset Name: {dataset_name}

Analysis Findings:
{analysis_context}

STRICT RULES:
- Do NOT mention the dataset, data cleaning, missing values, data quality,
  imputation, null values, or anything related to how the data was processed.
- Do NOT include any "Data Quality" section or data caveats of any kind.
- Write ONLY about business performance, metrics, trends, and recommendations.
- Every number must be preserved exactly as given in the analysis.
- Keep all markdown tables intact and properly formatted.
- Write in plain operational English — no jargon, no abstract strategy language.
- Every recommendation must reference a specific number.
- Every insight must be grounded in a metric from the analysis.

Output the report in this exact structure:

# {dataset_name} — Business Performance Report

## Executive Summary
3–4 bullet points covering:
- Key revenue or volume totals with exact numbers
- Top performing category/product/segment with its exact revenue or volume
- Most important trend observed
- Single most important opportunity

## Core Business KPIs
Markdown table:
| KPI | Value |
Include: all key totals, averages, top performer, bottom performer,
best period, worst period.

## [Primary Dimension] Performance
Replace [Primary Dimension] with whatever fits — Product, Category,
Region, Customer Segment, etc.

Sub-section 1: Revenue or Volume Breakdown
Markdown table sorted by revenue or volume descending.
Follow with 1–2 sentence insight.

Sub-section 2: Revenue per Unit / per Transaction (if applicable)
Markdown table.
Follow with 1–2 sentence insight explaining which items are
premium vs high-volume-low-value.

## Behavioral Patterns
Answer these with specific numbers:
- Which categories have high volume but low revenue — and what does that mean?
- What cross-sell or upsell opportunities exist based on the data?
- What does the purchase or usage pattern suggest about customer behavior?

## Channel & Payment Breakdown
Markdown table of revenue or volume by channel and/or payment method.
1 sentence per table: is it balanced or concentrated, and what does that mean?

## Monthly Performance Trend
Markdown table: Month | Revenue (or volume).
Follow with:
- Best month and worst month with exact numbers
- Is the overall trend growing, flat, or declining?
- What does the pattern suggest the business should do?

## Top 5 Business Insights
Numbered. Each insight:
1. States an exact number
2. Explains the business meaning in one sentence
3. Names the specific opportunity or risk it creates

## Strategic Recommendations
Exactly 5. Each in this format:

**[Number]. [Action verb + what to do]**
Based on: [exact metric]
Action: [specific thing to do — e.g. "introduce coffee + cake bundle at checkout"]
Expected outcome: [what business result this drives]

## Management Takeaway
2–3 sentences only:
- What is the overall business health?
- What is the single most important action right now?
- What will happen if it is acted on?
        """,
        agent=get_report_agent(),
        expected_output=(
            "A clean business performance report with no mention of data, "
            "cleaning, or quality issues. Contains: executive summary bullets, "
            "KPI table, primary dimension breakdown tables with insights, "
            "behavioral patterns grounded in numbers, channel/payment tables, "
            "monthly trend table, 5 numbered insights, 5 concrete recommendations "
            "each with action and expected outcome, and a management takeaway."
        ),
    )


def get_business_summary_task(verified_stats: dict, dataset_name: str):
    stats_json = json.dumps(verified_stats, indent=2, default=str)

    return Task(
        description=f"""
Write a SHORT business performance paragraph for an executive dashboard —
3 to 5 sentences in plain business English.

This is the first thing a business reader sees. It must:
1. State what kind of business activity this data covers (1 sentence)
2. Name the single most important performance finding with its exact number (1 sentence)
3. Give ONE specific opportunity or action the business should take (1 sentence)
4. Optionally name the biggest risk or watch item (1 sentence)

Rules:
- No mention of data cleaning, missing values, data quality, or imputation
- No section headers, no bullet points
- No statistical jargon ('mean', 'r_squared', 'std', 'coefficient')
- Do not describe the dataset shape or column names
- Every sentence must be useful to a business reader

Dataset name: {dataset_name}

Verified statistics:
{stats_json}

Output ONLY the paragraph. No title, no "Summary:" prefix, no markdown.
        """,
        agent=get_analyst_agent(),
        expected_output=(
            "A 3-5 sentence business performance paragraph: what the business "
            "does, the top performance finding with its exact number, one "
            "specific opportunity or action, and optionally one risk. "
            "No data/cleaning references. Plain prose only."
        ),
    )