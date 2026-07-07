"""
tasks.py

Updated:
  1. DATA RELIABILITY RULES — every task that consumes verified_stats now
     explicitly checks the sample-size guard fields added in stats_engine.py
     (insufficient_edge_data, excluded_years, start_sample_size, end_sample_size,
     total_sample_size, min_sample_size_required) before it's allowed to state
     a percentage change as fact. A null/flagged metric gets narrated as
     "not enough data to compare" — never silently skipped, never invented.
  2. ANALYST-GRADE KPI FRAMING — the analyst and report agents are now told
     to behave like a human analyst, not a numbers dictation service: compute
     and cite ratios (revenue per transaction, mean-vs-median gap as a skew
     signal), flag concentration risk, and connect every number to a decision
     — not just repeat totals/means/medians verbatim.
  3. ROW RETENTION TRIPWIRE (this version) — a second, independent safety
     check layered on top of data_cleaner.py's own dedup safety gates. Even
     if a future cleaning bug causes silent row loss some other way, the
     analyst agent is now contractually required to check the explicit
     "ROW RETENTION CHECK" numbers passed in via cleaning_context (see
     crew.py) and lead the report with a loud warning if more than 10% of
     rows were dropped — rather than presenting KPIs as if they represent
     the full dataset.
"""

import json
from crewai import Task
from agents import get_cleaner_agent, get_analyst_agent, get_report_agent


# ── Shared reliability contract injected into every stats-consuming task ─────
DATA_RELIABILITY_RULES = """
DATA RELIABILITY RULES (apply before citing ANY trend or period comparison):

The verified statistics package includes explicit sample-size guard fields.
You MUST check these before stating a percentage change, trend, or "X vs Y"
comparison as fact:

- In "trends": if "insufficient_edge_data" is true, the start_value/end_value/
  pct_change_start_to_end for that metric are NOT reliable — do not report
  the percentage. Instead say something like: "There isn't enough data at
  the start or end of the period to measure a reliable change for [metric]."
  Always check "total_sample_size", "start_sample_size", "end_sample_size"
  before treating a trend as a real business signal, and if you do use a
  trend, silently reflect that reliability rather than naming numeric fields.

- In "period_trends": if "yoy_pct_change" is null, do NOT invent a YoY number.
  Check "excluded_years" — if a year was excluded, explain briefly that a
  year had too few orders/records to be a fair comparison (state the record
  count if given), rather than comparing it to a full year anyway. Never
  describe a period with very few records as a "collapse", "crash", or
  "plummet" — that language implies a real business event, not a data gap.

- Never present a percentage change, before/after comparison, or trend
  direction that is null, flagged insufficient, or built from an excluded
  period. If the only available comparison is unreliable, say so honestly
  and pivot to what IS reliable (totals, category breakdowns, correlations).

- When in doubt about whether a number is trustworthy, prefer describing
  what is definitely true (totals, counts, top/bottom performers) over a
  fragile percentage swing.

ROW RETENTION TRIPWIRE (check this FIRST, before anything else in your output):

The cleaning context you are given includes an explicit "ROW RETENTION CHECK"
section showing the ORIGINAL row count vs. the CLEANED row count and the
resulting retention percentage. This check is INDEPENDENT of anything the
cleaning log itself claims — you must verify it yourself from the numbers
given, even if the cleaning log shows no explicit errors or warnings.

- If retention is below 90% (i.e. more than 10% of rows were removed during
  cleaning), you MUST open your Executive Summary with a clear, prominent
  warning stating the exact original row count, cleaned row count, and
  percentage dropped — BEFORE any KPI, trend, or insight.
- State plainly that all totals, KPIs, and trends below may understate true
  business volume as a result, since they are computed only on the surviving
  rows.
- Do NOT bury this warning inside a caveat sentence at the end of the
  summary, and do NOT proceed to present KPIs as if they represent the full
  original dataset without this warning appearing first.
- If retention is 90% or above, no special warning is needed — proceed
  normally, but you may still briefly note the retention percentage in
  passing for transparency.
"""

# ── Shared instruction for analyst-grade KPI framing (not vanity numbers) ────
ANALYST_FRAMING_RULES = """
ANALYST-GRADE FRAMING (write like a human analyst, not a numbers dictation service):

For every KPI or metric you cite, don't just restate mean/median/total —
add the one layer of judgment a real analyst would:

- Ratios over raw totals where possible: revenue per transaction, revenue
  per customer, discount as % of revenue, cost per unit, etc. These are
  usually more decision-relevant than a raw total alone.
- Mean vs median gap = a skew signal, not just two numbers. If mean is
  meaningfully higher than median, say what that implies (a small number
  of large orders/outliers are pulling the average up — the "typical"
  order is smaller than the headline average suggests). If mean and
  median are close, say the metric is consistent/predictable.
- Concentration risk: if one category/channel/segment accounts for a
  disproportionate share of revenue or volume relative to its count,
  name that explicitly as a dependency risk, not just "top performer."
- Context over isolated numbers: don't just say "Average Order Value is
  $1,706.74" — say what that means relative to the median, relative to
  other categories, or relative to what it implies about customer
  behavior (e.g. are most orders small with a few large outliers, or is
  spending fairly uniform?).
- Every recommendation must trace to a SPECIFIC number and explain the
  mechanism — not "improve marketing" but "X channel drives only Y% of
  revenue despite Z% of transaction volume — investigate why conversion
  value is low there."
"""


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
If the report contains a "ROW RETENTION WARNING" or shows any dedup step
was aborted due to a safety threshold, this MUST be reflected in your
verdict — a large unexplained row-count drop or an aborted dedup safety
gate means "Proceed with Caution" at minimum, never "Ready for Analysis".

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


def get_analysis_task(verified_stats: dict, cleaning_context: str, rows_dropped_pct: float = 0.0):
    stats_json = json.dumps(verified_stats, indent=2, default=str)

    retention_warning_block = ""
    if rows_dropped_pct and rows_dropped_pct > 10:
        retention_warning_block = f"""
⚠️ CRITICAL ROW RETENTION FLAG: {rows_dropped_pct}% of rows were removed
during cleaning (see the ROW RETENTION CHECK in the cleaning context below).
You MUST open your Executive Summary with an explicit warning about this
before presenting any KPI, trend, or insight — per the ROW RETENTION
TRIPWIRE rule below.
"""

    return Task(
        description=f"""
You are a Senior BI Analyst. Below is a VERIFIED STATISTICS PACKAGE
computed deterministically from the dataset. Every number is exact.
DO NOT recompute, estimate, or invent any figures.
{retention_warning_block}
VERIFIED STATISTICS (ground truth — use only these numbers):
{stats_json}

Cleaning Context:
{cleaning_context}

{DATA_RELIABILITY_RULES}

{ANALYST_FRAMING_RULES}

Your output must follow this structure EXACTLY:

**Executive Summary**
- If the ROW RETENTION TRIPWIRE applies (see rules above), this section
  MUST open with that warning before anything else.
- Key numeric totals (e.g. Total Revenue, Total Transactions)
- Average Order Value or equivalent primary metric, with mean-vs-median
  context (see ANALYST-GRADE FRAMING above)
- 3–4 bullet points summarizing the most important RELIABLE patterns
- Keep it factual and concise — no unreliable percentage claims

**Core Business KPIs**
Present as a markdown table:
| KPI | Value | What it means |
Include: totals, averages (with mean-vs-median skew note where relevant),
highest/lowest performing category, best/worst period. The third column
should add analyst judgment, not repeat the number.

**Category / Product / Segment Performance** (use whatever dimension fits the data)
- Revenue or volume breakdown per category as a markdown table
- Revenue per unit or per transaction as a separate markdown table
- 1–2 sentence insight below each table naming any concentration risk or
  cross-sell opportunity — not just describing the numbers

**Customer / Behavioral Patterns** (if applicable)
- Volume leaders vs revenue leaders — are they the same?
- What does high volume but low revenue suggest?
- What opportunities does this create?

**Channel / Segment Split** (if applicable)
- Breakdown table by channel, region, or segment
- Is revenue balanced or concentrated? Name the concentration risk if one exists.
- What does the distribution suggest?

**Time / Period Analysis**
- Monthly, quarterly, or yearly breakdown table
- Identify best and worst periods with exact numbers — but ONLY periods
  that clear the sample-size guard (check excluded_years / insufficient_edge_data)
- Is the trend growing, declining, or flat? If not enough reliable data
  exists to say, state that plainly instead of guessing

**Top 5 Business Insights**
Numbered list. Each insight must:
- State an exact, RELIABLE number from the verified stats (never a flagged
  or excluded one)
- Explain what it means for the business in plain English, with analyst
  judgment (ratio, concentration, skew — not just the raw figure)
- Name one specific opportunity or risk it creates

**Strategic Recommendations**
Exactly 5 recommendations. Each must:
- Be grounded in a specific, reliable number from the analysis
- Name a concrete action (e.g. "Bundle Coffee + Cake to increase basket size")
- State the expected business outcome
- Be written in plain operational language — no jargon

**Management Takeaway**
2–3 sentences summarizing the overall picture and the single most 
important action the business should take right now.
        """,
        agent=get_analyst_agent(),
        expected_output=(
            "A structured BI analysis with executive summary (leading with a "
            "row-retention warning if more than 10% of rows were dropped during "
            "cleaning), KPI table, category/segment breakdown tables, behavioral "
            "insights, channel split, time analysis, top 5 insights, 5 "
            "recommendations, and a management takeaway — all grounded in exact, "
            "reliable verified numbers with analyst-grade context (ratios, skew, "
            "concentration risk), never a flagged or insufficient-sample metric."
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
- EXCEPTION: if the analysis findings above contain a row-retention warning
  (i.e. a significant percentage of rows were removed during cleaning),
  you MUST preserve that warning at the top of the Executive Summary in
  plain business language (e.g. "This report reflects a subset of records
  after data validation — see note below" is NOT acceptable phrasing;
  state plainly that a portion of records could not be included and that
  totals may understate true volume). This is the one data-related fact
  that must survive into the business report, because omitting it would
  make every number below misleading.
- Write ONLY about business performance, metrics, trends, and recommendations
  otherwise.
- Every number must be preserved exactly as given in the analysis.
- If the analysis explicitly notes a metric had insufficient data to reliably
  compare, preserve that honesty in the report rather than smoothing it into
  a confident percentage — say "not enough data to compare" in plain business
  language instead of a caveat-flavored data disclaimer.
- Keep all markdown tables intact and properly formatted.
- Write in plain operational English — no jargon, no abstract strategy language.
- Every recommendation must reference a specific number.
- Every insight must be grounded in a metric from the analysis.

{ANALYST_FRAMING_RULES}

Output the report in this exact structure:

# {dataset_name} — Business Performance Report

## Executive Summary
3–4 bullet points covering:
- (If applicable) A leading row-retention warning per the rule above
- Key revenue or volume totals with exact numbers
- Top performing category/product/segment with its exact revenue or volume
- Most important RELIABLE trend observed (skip if none clear the bar)
- Single most important opportunity

## Core Business KPIs
Markdown table:
| KPI | Value | What it means |
Include: all key totals, averages, top performer, bottom performer,
best period, worst period. Third column carries analyst judgment
(ratio, skew, or context) — never just a restated number.

## [Primary Dimension] Performance
Replace [Primary Dimension] with whatever fits — Product, Category,
Region, Customer Segment, etc.

Sub-section 1: Revenue or Volume Breakdown
Markdown table sorted by revenue or volume descending.
Follow with 1–2 sentence insight naming any concentration risk.

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
- Is the overall trend growing, flat, or declining — based only on periods
  with enough data to be trustworthy? If not enough reliable history exists,
  say so plainly instead of forcing a trend claim.
- What does the pattern suggest the business should do?

## Top 5 Business Insights
Numbered. Each insight:
1. States an exact, reliable number
2. Explains the business meaning in one sentence, with analyst judgment
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
            "cleaning, or quality issues (except a preserved row-retention "
            "warning if one applies), and no unreliable/flagged percentage "
            "claims. Contains: executive summary bullets, KPI table with analyst "
            "judgment column, primary dimension breakdown tables with insights, "
            "behavioral patterns grounded in numbers, channel/payment tables, "
            "monthly trend table, 5 numbered insights, 5 concrete recommendations "
            "each with action and expected outcome, and a management takeaway."
        ),
    )


def get_business_summary_task(verified_stats: dict, dataset_name: str, rows_dropped_pct: float = 0.0):
    stats_json = json.dumps(verified_stats, indent=2, default=str)

    retention_instruction = ""
    if rows_dropped_pct and rows_dropped_pct > 10:
        retention_instruction = f"""
IMPORTANT: {rows_dropped_pct}% of rows were removed during data cleaning.
Your first sentence MUST briefly and plainly note that this summary reflects
a subset of records and totals may understate true business volume, before
moving to the performance finding.
"""

    return Task(
        description=f"""
Write a SHORT business performance paragraph for an executive dashboard —
3 to 5 sentences in plain business English.

This is the first thing a business reader sees. It must:
1. State what kind of business activity this data covers (1 sentence)
2. Name the single most important RELIABLE performance finding with its
   exact number (1 sentence) — check the sample-size guard fields below
   before citing any percentage change
3. Give ONE specific opportunity or action the business should take (1 sentence)
4. Optionally name the biggest risk or watch item (1 sentence)
{retention_instruction}
{DATA_RELIABILITY_RULES}

Rules:
- No mention of data cleaning, missing values, data quality, or imputation
  UNLESS the row-retention instruction above applies — that one fact must
  be stated plainly if it applies, everything else about cleaning stays out
- No section headers, no bullet points
- No statistical jargon ('mean', 'r_squared', 'std', 'coefficient')
- Do not describe the dataset shape or column names
- Never state a percentage change, trend, or before/after comparison that
  is null or flagged as insufficient/excluded in the verified statistics
- Every sentence must be useful to a business reader

Dataset name: {dataset_name}

Verified statistics:
{stats_json}

Output ONLY the paragraph. No title, no "Summary:" prefix, no markdown.
        """,
        agent=get_analyst_agent(),
        expected_output=(
            "A 3-5 sentence business performance paragraph: (if applicable) a "
            "brief row-retention note, what the business does, the top "
            "RELIABLE performance finding with its exact number, one specific "
            "opportunity or action, and optionally one risk. No unreliable "
            "percentage claims. Plain prose only."
        ),
    )