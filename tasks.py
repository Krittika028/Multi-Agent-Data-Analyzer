"""
tasks.py — rewritten for insight density.

Key changes vs. previous version:
  1. BLUF (Bottom Line Up Front) is now MANDATORY, not optional — every
     report/summary opens with a 2-3 sentence answer to "what happened and
     what do I do about it," before any table or KPI appears.
  2. Ratios, skew commentary, and concentration risk are no longer "if
     applicable" — they are REQUIRED wherever the underlying columns exist,
     with an explicit instruction that skipping them is a failure condition.
  3. COMPARATIVE ANCHORING RULE — no number may be presented in isolation.
     Every KPI must be stated against at least one benchmark: category
     average, prior reliable period, or peer segment.
  4. Recommendations now require a quantified "stake" — not just an action,
     but the estimated $ or % impact of acting vs. not acting, computed only
     from verified numbers (never invented).
  5. report_task explicitly treats injected task.context as primary source
     of truth and is forbidden from diluting the analyst's BLUF, ratios, or
     comparisons during reformatting.
  6. All other behavior (reliability rules, row-retention tripwire, domain
     adaptivity) is preserved unchanged from the previous version.
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

# ── NEW: mandatory insight-density contract, applied to analyst + report ─────
INSIGHT_DENSITY_RULES = """
INSIGHT DENSITY RULES — these are REQUIREMENTS, not suggestions. A report
that skips any of these because "it wasn't applicable" is only acceptable if
you can point to which specific verified-stats field made it genuinely
impossible (e.g. no categorical column existed to compute a ratio against).
Absence of effort is not the same as absence of data.

1. BOTTOM LINE UP FRONT (BLUF) — MANDATORY, FIRST THING IN THE OUTPUT:
   Before any table, KPI list, or section header, write 2-3 sentences that
   answer: "What is the single most important thing happening in this
   business right now, and what should the reader do about it in the next
   week?" This is not a summary of sections below — it is the ANSWER, stated
   plainly, with the one number that matters most. A reader who reads only
   this paragraph and nothing else should still know what to do.

2. NO ISOLATED NUMBERS — every KPI, total, or metric you state must be
   anchored against at least ONE comparison point:
     - vs. the category/segment average (e.g. "23% above the average category")
     - vs. a prior reliable period (only if not flagged insufficient/excluded)
     - vs. another segment/channel in the same dataset
   "Revenue was $45,000" is a data dump. "Salads generated $45,000 — 2.3x the
   average category's $19,600" is an insight. If truly no comparison point
   exists for a given number (rare), say so explicitly rather than silently
   omitting the comparison.

3. RATIOS ARE REQUIRED WHENEVER THE COLUMNS EXIST — not optional:
   revenue/volume per unit, per transaction, per segment member, discount as
   % of revenue, cost per unit — compute and state at least 2 such ratios
   if the underlying numeric + categorical columns exist in verified_stats.
   If none of the required column combinations exist in this dataset, say
   explicitly "no per-unit ratio could be computed because [specific reason]"
   rather than silently skipping the section.

4. MEAN-VS-MEDIAN SKEW — for every primary KPI, compare mean to median and
   state what the gap implies (a few large outliers vs. a consistent
   pattern). This is required for every KPI table, not just the first one.

5. CONCENTRATION / DEPENDENCY RISK — for every category breakdown, identify
   whether the top entry holds a disproportionate share (top entry's % of
   total vs. its % of row count). If the top category holds >40% of the
   primary metric, this MUST be named explicitly as a dependency risk with
   a sentence on what happens if that segment underperforms.

6. QUANTIFIED RECOMMENDATION STAKES — every recommendation must state not
   just the action, but the estimated stake, computed from verified numbers:
   "If category X's share moved from A% to B% (matching the top performer),
   that implies roughly $Y in additional [metric]." Use only real numbers
   from verified_stats for this math — never invent a projection number
   that isn't derivable from what's given. If a clean quantified estimate
   truly isn't derivable, state the qualitative stakes explicitly instead
   of silently reverting to vague language like "could improve results."

7. NAME THE MECHANISM, NOT JUST THE PATTERN — never write "X is
   underperforming." Write what specifically is happening: "X has 3x the
   order volume of Y but only 1.1x the revenue — average order value in X is
   roughly a third of Y's, suggesting X attracts high-frequency, low-basket
   customers while Y attracts fewer, higher-spend customers."
"""

# ── Shared instruction for analyst-grade KPI framing (not vanity numbers) ────
def get_analyst_framing_rules(domain_config: dict = None) -> str:
    domain_config = domain_config or {}
    entity = domain_config.get("primary_entity", "Record")
    dimension = domain_config.get("dimension_label", "Category/Segment")
    entity_plural = domain_config.get("entity_plural", "Records")

    return f"""
ANALYST-GRADE FRAMING (write like a human analyst, not a numbers dictation service):

For every KPI or metric you cite, don't just restate mean/median/total —
add the one layer of judgment a real analyst would:

- Ratios over raw totals where possible: metric per {entity.lower()},
  metric per {dimension.split('/')[0].lower()}, or any other per-unit
  ratio the actual columns support. These are usually more decision-
  relevant than a raw total alone.
- Mean vs median gap = a skew signal, not just two numbers. If mean is
  meaningfully higher than median, say what that implies (a small number
  of large/outlier {entity_plural.lower()} are pulling the average up —
  the "typical" {entity.lower()} is smaller than the headline average
  suggests). If mean and median are close, say the metric is consistent/
  predictable.
- Concentration risk: if one {dimension.lower()} accounts for a
  disproportionate share of the primary metric or volume relative to its
  count, name that explicitly as a dependency risk, not just "top
  performer."
- Context over isolated numbers: don't just restate a headline average —
  say what that means relative to the median, relative to other
  {dimension.lower()} values, or relative to what it implies about the
  underlying process or behavior.
- Every recommendation must trace to a SPECIFIC number and explain the
  mechanism — not "improve X" but "Y {dimension.split('/')[0].lower()}
  drives only Z% of the primary metric despite W% of volume — investigate
  why value is low there."

{INSIGHT_DENSITY_RULES}
"""


ANALYST_FRAMING_RULES = get_analyst_framing_rules(None)


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


def get_analysis_task(verified_stats: dict, cleaning_context: str, rows_dropped_pct: float = 0.0, domain_config: dict = None):
    stats_json = json.dumps(verified_stats, indent=2, default=str)
    domain_config = domain_config or {}
    dimension_label = domain_config.get("dimension_label", "Category/Segment")
    entity_plural = domain_config.get("entity_plural", "Records")
    domain_name = domain_config.get("domain", "")

    retention_warning = ""
    if rows_dropped_pct > 10:
        retention_warning = f"""
⚠️ CRITICAL: {rows_dropped_pct}% of rows were removed during cleaning.
You MUST lead your Executive Summary with this warning before any other content.
"""

    domain_note = (
        f"\nDATASET DOMAIN: this data has been identified as **{domain_name}** data, "
        f"where each row represents a **{domain_config.get('primary_entity', 'record')}**. "
        f"Use domain-appropriate section names and vocabulary throughout your analysis — "
        f"e.g. use \"{dimension_label} Performance\" instead of a generic or commerce-specific "
        f"section name, and refer to rows as \"{entity_plural}\" rather than defaulting to "
        f"\"transactions\" or \"orders\" if those terms don't fit this domain.\n"
        if domain_name else ""
    )

    return Task(
        description=f"""
You are a Senior BI Analyst. Below is a VERIFIED STATISTICS PACKAGE
computed deterministically from the dataset. Every number is exact.
DO NOT recompute, estimate, or invent any figures — but you ARE required
to derive ratios, comparisons, and percentages FROM these exact numbers
wherever the underlying fields exist (see INSIGHT DENSITY RULES below).
{retention_warning}{domain_note}
VERIFIED STATISTICS (ground truth — use only these numbers):
{stats_json}

Cleaning Context:
{cleaning_context}

{DATA_RELIABILITY_RULES}

{get_analyst_framing_rules(domain_config)}

Your output must follow this structure — adapt section TITLES to fit the
domain noted above, but keep the same underlying content requirements:

**Bottom Line Up Front**
(MANDATORY — see INSIGHT DENSITY RULES, rule 1. This comes before anything
else, including the row-retention warning if one applies — the retention
warning is its own sentence within or immediately after this paragraph,
not a replacement for it.)
2-3 sentences: what matters most right now, the one number that proves it,
and what to do about it this week.

**Executive Summary**
- Key numeric totals for this dataset's primary metrics — each anchored
  against a comparison point (see rule 2)
- Primary average metric, with mean-vs-median context (rule 4)
- 3–4 bullet points summarizing the most important RELIABLE patterns
- Keep it factual and concise — no unreliable percentage claims

**Core {domain_config.get('primary_entity', 'Business')} KPIs**
Present as a markdown table:
| KPI | Value | vs. Benchmark | What it means |
Include: totals, averages (with mean-vs-median skew note per rule 4),
highest/lowest performing {dimension_label.split('/')[0].lower()}, best/worst period.
The "vs. Benchmark" column is required (rule 2) — average, prior period, or
peer segment. The last column carries analyst judgment, not a repeated number.

**{dimension_label} Performance**
- Breakdown per {dimension_label.split('/')[0].lower()} as a markdown table
- Per-unit ratio breakdown as a REQUIRED second table (rule 3) unless you
  state explicitly why no ratio is computable
- 2-3 sentence insight below each table naming concentration risk (rule 5)
  and the underlying mechanism (rule 7), not just describing the numbers

**Behavioral / Process Patterns** (if applicable to this domain)
- Volume leaders vs value leaders — are they the same {dimension_label.split('/')[0].lower()}?
- What does high volume but low value/outcome suggest? Name the mechanism.
- What opportunities does this create?

**Segment / Channel Split** (if a genuine secondary dimension exists in the data)
- Breakdown table by whatever secondary dimension fits (region, team, source, etc.)
- Is the primary metric balanced or concentrated? Name the concentration risk if one exists.

**Time / Period Analysis**
- Monthly, quarterly, or yearly breakdown table
- Identify best and worst periods with exact numbers — but ONLY periods
  that clear the sample-size guard (check excluded_years / insufficient_edge_data)
- Is the trend growing, declining, or flat? If not enough reliable data
  exists to say, state that plainly instead of guessing

**Top 5 Insights**
Numbered list. Each insight must:
- State an exact, RELIABLE number from the verified stats, anchored against
  a comparison point (never a flagged or excluded one)
- Name the underlying mechanism (rule 7), not just restate the pattern
- Name one specific opportunity or risk it creates, with a quantified stake
  where derivable (rule 6)

**Strategic Recommendations**
Exactly 5 recommendations. Each must:
- Be grounded in a specific, reliable number from the analysis
- Name a concrete action
- State the QUANTIFIED expected outcome per rule 6 (derived from verified
  numbers — never invented) — if truly not derivable, state the qualitative
  stake explicitly instead of vague language
- Be written in plain operational language — no jargon

**Management Takeaway**
2–3 sentences summarizing the overall picture and the single most 
important action to take right now. This should echo — not contradict —
the Bottom Line Up Front paragraph.
        """,
        agent=get_analyst_agent(),
        expected_output=(
            "A structured analysis opening with a mandatory Bottom Line Up "
            "Front paragraph, followed by executive summary, KPI table with "
            "benchmark column, {0} breakdown tables including a per-unit "
            "ratio table, behavioral insights naming mechanisms not just "
            "patterns, segment split, time analysis, top 5 insights each "
            "with a quantified stake, 5 recommendations each with a "
            "quantified expected outcome, and a management takeaway — all "
            "grounded in exact, reliable verified numbers, every number "
            "anchored against a comparison point, using domain-appropriate "
            "vocabulary, and never a flagged or insufficient-sample metric."
        ).format(dimension_label),
    )


def get_report_task(analysis_context: str, dataset_name: str, domain_config: dict = None):
    return Task(
        description=f"""
You are writing a business performance report for operations managers
and business owners.

Dataset Name: {dataset_name}

Analysis Findings (this is your PRIMARY SOURCE OF TRUTH — it was produced
by a senior analyst who already did the ratio math, comparison anchoring,
and mechanism analysis; your job is to format and polish it, NOT to
re-summarize it into something vaguer):
{analysis_context}

CRITICAL — DO NOT DILUTE THE SOURCE MATERIAL:
- The analysis already contains a "Bottom Line Up Front" paragraph, ratio
  tables, benchmark comparisons, concentration-risk callouts, and quantified
  recommendation stakes. PRESERVE these exactly — do not compress a
  quantified stake ("$X potential uplift") down to a vague verb phrase
  ("could improve results"). If you shorten prose, shorten padding, never
  the numbers or the mechanism explanation.
- If a specific ratio, comparison, or quantified stake appears in the
  analysis findings, it MUST appear somewhere in your final report. Losing
  it during formatting is a failure condition.

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
- Keep all markdown tables intact and properly formatted, including any
  "vs. Benchmark" columns and per-unit ratio tables from the analysis.
- Write in plain operational English — no jargon, no abstract strategy language.
- Every recommendation must reference a specific number AND its quantified
  stake, exactly as computed in the analysis findings.
- Every insight must be grounded in a metric from the analysis, anchored
  against the same comparison point the analyst used.

{get_analyst_framing_rules(domain_config)}

Output the report in this exact structure:

# {dataset_name} — Business Performance Report

## Bottom Line
(MANDATORY, FIRST SECTION — carry the analyst's Bottom Line Up Front
paragraph through essentially unchanged. This is the single most important
part of the report. A reader who stops here should already know what
matters and what to do.)

## Executive Summary
3–4 bullet points covering:
- (If applicable) A leading row-retention warning per the rule above
- Key revenue or volume totals with exact numbers, anchored vs. benchmark
- Top performing category/product/segment with its exact revenue or volume
  AND how it compares to the average/other segments
- Most important RELIABLE trend observed (skip if none clear the bar)
- Single most important opportunity, with its quantified stake if derivable

## Core Business KPIs
Markdown table:
| KPI | Value | vs. Benchmark | What it means |
Include: all key totals, averages, top performer, bottom performer,
best period, worst period. The benchmark column and the judgment column
are both required — never just a restated number.

## [Primary Dimension] Performance
Replace [Primary Dimension] with whatever fits — Product, Category,
Region, Customer Segment, etc.

Sub-section 1: Revenue or Volume Breakdown
Markdown table sorted by revenue or volume descending.
Follow with 2-3 sentences naming any concentration risk AND the mechanism
behind it (e.g. why is this segment concentrated — pricing, demand, mix).

Sub-section 2: Revenue per Unit / per Transaction
Markdown table (required unless the analysis explicitly states no ratio
was computable).
Follow with 1–2 sentence insight explaining which items are
premium vs high-volume-low-value, and the implied customer behavior.

## Behavioral Patterns
Answer these with specific numbers, anchored against comparisons:
- Which categories have high volume but low revenue — and what does the
  ratio between them (not just the raw numbers) actually mean?
- What cross-sell or upsell opportunities exist based on the data, with a
  quantified stake if derivable?
- What does the purchase or usage pattern suggest about customer behavior?

## Channel & Payment Breakdown
Markdown table of revenue or volume by channel and/or payment method.
1-2 sentences: is it balanced or concentrated, what's the mechanism, and
what should the business do about it?

## Monthly Performance Trend
Markdown table: Month | Revenue (or volume).
Follow with:
- Best month and worst month with exact numbers
- Is the overall trend growing, flat, or declining — based only on periods
  with enough data to be trustworthy? If not enough reliable history exists,
  say so plainly instead of forcing a trend claim.
- What does the pattern suggest the business should do this month?

## Top 5 Business Insights
Numbered. Each insight:
1. States an exact, reliable number, anchored against a comparison point
2. Names the underlying mechanism in one sentence — not just the pattern
3. Names the specific opportunity or risk, with a quantified stake where
   the analysis provides one

## Strategic Recommendations
Exactly 5. Each in this format:

**[Number]. [Action verb + what to do]**
Based on: [exact metric, with its benchmark comparison]
Action: [specific thing to do — e.g. "introduce coffee + cake bundle at checkout"]
Expected outcome: [QUANTIFIED — the $ or % stake, taken directly from the
analysis findings; if the analysis only gave a qualitative stake, state
that qualitative stake explicitly rather than inventing a number]

## Management Takeaway
2–3 sentences only:
- What is the overall business health?
- What is the single most important action right now?
- What will happen if it is acted on? (echo the quantified stake if one exists)
        """,
        agent=get_report_agent(),
        expected_output=(
            "A clean business performance report opening with a mandatory "
            "'Bottom Line' section carried through from the analyst's BLUF, "
            "followed by executive summary, KPI table with a benchmark "
            "column, primary dimension breakdown tables (including a "
            "per-unit ratio table) with mechanism-level insights, "
            "behavioral patterns grounded in ratios not raw numbers, "
            "channel/payment tables, monthly trend table, 5 numbered "
            "insights each naming a mechanism, 5 concrete recommendations "
            "each with a quantified expected outcome carried over from the "
            "analysis (never diluted to vague language), and a management "
            "takeaway — with no mention of data cleaning/quality issues "
            "except a preserved row-retention warning if one applies, and "
            "no unreliable/flagged percentage claims."
        ),
    )


def get_business_summary_task(verified_stats: dict, dataset_name: str, rows_dropped_pct: float = 0.0, domain_config: dict = None):
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
   exact number, ANCHORED against a comparison point — average, prior
   reliable period, or peer segment (1 sentence) — check the sample-size
   guard fields below before citing any percentage change
3. Give ONE specific opportunity or action the business should take, with
   its quantified stake if derivable from the verified stats (1 sentence)
4. Optionally name the biggest risk or watch item, including concentration
   risk if one category dominates disproportionately (1 sentence)
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
- Every sentence must be useful to a business reader AND anchored — no
  isolated numbers with nothing to compare them to

Dataset name: {dataset_name}

Verified statistics:
{stats_json}

Output ONLY the paragraph. No title, no "Summary:" prefix, no markdown.
        """,
        agent=get_analyst_agent(),
        expected_output=(
            "A 3-5 sentence business performance paragraph: (if applicable) a "
            "brief row-retention note, what the business does, the top "
            "RELIABLE performance finding with its exact number anchored "
            "against a comparison point, one specific opportunity or action "
            "with a quantified stake where derivable, and optionally one "
            "risk. No unreliable percentage claims. Plain prose only."
        ),
    )