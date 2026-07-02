"""
dashboard_agent.py

Fixes vs previous version:
  1. Business-only KPI filtering — derived date columns (_year, _month,
     _month_name), dataset-structural columns (row counts, IDs), and
     any column that was removed from the cleaned dataset are all excluded
     from chart generation. Charts only reflect business metrics.
  2. available_columns is filtered through _business_columns() before
     being passed to the LLM and to fallback spec generation, so a
     removed column can never sneak into a chart even if it still exists
     in verified_stats (which is recomputed after column removal).
  3. _DERIVED_SUFFIXES list makes it easy to extend in future.
  4. Chart type selection now follows a rule-based decision tree matching
     the nature of the comparison (trend/relationship/distribution/
     composition/ranking/part-to-whole), including scatter and histogram,
     driven directly by verified_stats (correlations, anomalies, trends).

Updated (this version):
  5. DATA RELIABILITY AWARENESS — the agent now reads the sample-size
     guard fields stats_engine.py added ("insufficient_edge_data",
     "excluded_years", "start_sample_size"/"end_sample_size",
     "total_sample_size"). A trend/period comparison flagged insufficient
     is never charted as if it were a real business signal — the agent is
     told to either skip it or pick a chart type that shows what IS
     reliable (e.g. a distribution/category chart instead of a fragile
     before/after line).
  6. ANALYST-GRADE REASONING — chart "reasoning" and "title" fields must
     now read like a human analyst's insight (ratio, concentration,
     context) rather than a bare restatement of a number, matching the
     framing standard used in tasks.py for the report agents.
"""

import os
import json
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

ALLOWED_CHART_TYPES = {
    "pie", "donut", "bar", "column", "stacked_bar", "line", "scatter", "histogram",
}

# Suffixes added by DataCleaner's date-split step — never useful as chart axes
_DERIVED_SUFFIXES = ("_year", "_month", "_month_name")

# Column name keywords that are structural/metadata, not business metrics
_NON_BUSINESS_KEYWORDS = [
    "_id", "row_number", "index", "record_id", "seq", "serial",
    "uuid", "guid", "hash",
]

CHART_USAGE_GUIDE = """
CHART TYPE SELECTION GUIDE (rule-based — match the chart to the comparison, not the data alone):

| chart_type  | Use when...                                             | x_column        | y_column        |
|-------------|--------------------------------------------------------|-----------------|-----------------|
| pie         | Composition / share — only if <= 8 unique values        | category column | null or numeric |
| donut       | Same as pie but with a centre hole — preferred over pie | category column | null or numeric |
| bar         | Ranked category comparison, many/long labels            | category column | numeric (opt)   |
| column      | Vertical category or time-bucketed comparison           | category column | numeric (opt)   |
| stacked_bar | Part-to-whole across categories (needs color_col)       | category column | numeric column  |
| line        | Metric changing over TIME only — requires a date column | date column     | numeric column  |
| scatter     | Relationship/correlation between TWO numeric variables   | numeric column  | numeric column  |
| histogram   | Distribution/spread of ONE numeric variable              | numeric column  | null            |

DECISION LOGIC (apply in this order):
1. Is there a date/time column and a metric changing over it, AND is that
   trend reliable (see DATA RELIABILITY below)? → line
2. Are there two numeric columns with a meaningful correlation (see "correlations" in verified stats)? → scatter
3. Does verified stats flag a column with notable skew/outliers worth showing its shape? → histogram
4. Is one column composition/share of a whole with <= 8 categories? → pie or donut (prefer donut)
5. Is it a category-vs-metric ranking or comparison? → bar (many/long labels) or column (fewer, short labels)
6. Is it part-to-whole across MULTIPLE categories at once? → stacked_bar

Only use scatter for two columns with a real correlation entry in verified_stats — never invent a pairing.
Only use histogram for a column that appears in verified_stats "anomalies" or "kpis" — never a random numeric column.
"""

DATA_RELIABILITY_GUIDE = """
DATA RELIABILITY AWARENESS (check before selecting any line/trend chart):

The "trends" and "period_trends" entries in verified_stats carry explicit
sample-size guard fields:

- "trends[].insufficient_edge_data" — if true, the start/end values for that
  metric are NOT trustworthy (too few rows at one or both edges of the date
  range). Do NOT select a line chart built on that trend's pct_change. If
  r_squared is still reasonably high and total_sample_size is adequate, a
  line chart can still show the shape over time — just don't title/reason
  around a "before vs after" percentage.
- "period_trends[].excluded_years" — if non-empty, at least one year in that
  metric's history had too few records to be a fair YoY comparison. Do not
  build a chart or reasoning line implying a YoY collapse/spike from that
  metric. Prefer a category or distribution chart for that metric instead,
  or a line chart scoped only to the years with sufficient data.
- If you're unsure whether a trend is reliable, prefer a chart type that
  shows composition, ranking, or distribution (bar/column/donut/histogram)
  over a fragile time-based percentage claim.

Never write a chart "reasoning" that states a percentage change sourced
from a flagged/insufficient/excluded period.
"""

ANALYST_REASONING_GUIDE = """
ANALYST-GRADE CHART REASONING (apply to every "reasoning" and "title" field):

Don't just restate a number — explain WHY it's worth a chart slot, the way
a human analyst would justify a chart to a business owner:

- Prefer framing that shows a ratio, share, concentration, or comparison
  over an isolated total (e.g. "Curry Corner drives 18% of revenue from
  11% of orders — check whether that's premium pricing or upsell" beats
  "Curry Corner has the highest order amount").
- If a KPI's mean is notably higher than its median, that's worth noting
  in reasoning when relevant (a few large orders are skewing the average).
- If picking a category breakdown, prefer noting concentration risk when
  the top category holds a disproportionate share, not just "top performer."
- Every chart earns its slot because it supports a decision — the reasoning
  should make that decision explicit (what should the business DO
  differently because of this chart?).
"""


class DashboardAgent:

    def __init__(self, llm=None):
        self.llm = llm or LLM(
            model=os.getenv("MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.agent = Agent(
            role="Expert Business Intelligence Analyst — Dashboard Specialist",
            goal=(
                "Design a focused executive dashboard by selecting exactly 3 charts "
                "from verified statistics. Every x_column and y_column must be "
                "an exact match from AVAILABLE COLUMNS. Never invent column names. "
                "Only use columns that represent real business metrics — never "
                "use derived date parts (_year, _month, _month_name) as chart axes. "
                "Prefer actionable, decision-relevant metrics over vanity numbers. "
                "Never chart a trend or period comparison flagged as built on "
                "insufficient sample data — treat data reliability as a hard "
                "constraint, not a suggestion."
            ),
            backstory=(
                "You are a BI analyst who designs C-suite dashboards. You know "
                "exactly which chart type fits which business question. Your golden "
                "rule: never put a column on a chart unless it appears exactly in "
                "the available columns list AND it represents a real business metric. "
                "Date-derived helper columns like order_date_year or order_date_month "
                "are never chart axes — use the original date column for line charts. "
                "You always ask 'what decision does this chart support?' before "
                "picking a chart type — trend over time gets a line, a relationship "
                "between two numbers gets a scatter, a single variable's spread gets "
                "a histogram, and category comparisons get a bar or column chart. "
                "You've been burned before by charting a metric that looked dramatic "
                "but was actually just a data artifact from a near-empty period — so "
                "now you always check sample-size reliability flags before trusting "
                "a percentage swing, and you always frame chart reasoning the way a "
                "sharp analyst would in a review meeting: ratios, concentration, and "
                "what decision the number should drive — not just the number itself."
            ),
            llm=self.llm,
            verbose=False,
        )

    # ── Filter available_columns to business-only columns ─────────────────────
    def _business_columns(self, available_columns: list) -> list:
        """
        Remove columns that are:
          1. Derived date parts (_year, _month, _month_name)
          2. Structural/metadata identifiers (row_number, uuid, etc.)
        These are never meaningful chart axes; they clutter the prompt and
        can produce misleading charts (e.g. bar chart of 'year' values).
        """
        result = []
        for col in available_columns:
            col_lower = col.lower()
            # Skip derived date columns
            if any(col_lower.endswith(sfx) for sfx in _DERIVED_SUFFIXES):
                continue
            # Skip pure structural metadata
            if any(kw in col_lower for kw in _NON_BUSINESS_KEYWORDS):
                continue
            result.append(col)
        return result

    # ── Summarise which trends/periods are reliable vs flagged ────────────────
    def _reliability_note(self, verified_stats: dict) -> str:
        """
        Builds an explicit, human-readable note listing which trend/period
        entries are flagged unreliable, so the LLM doesn't have to infer it
        from nested JSON — it's called out directly in the prompt.
        """
        flagged = []

        for t in verified_stats.get("trends", []):
            if t.get("insufficient_edge_data"):
                flagged.append(
                    f"- Trend '{t.get('value_column')}' over '{t.get('date_column')}': "
                    f"insufficient edge data (start_n={t.get('start_sample_size')}, "
                    f"end_n={t.get('end_sample_size')}, total_n={t.get('total_sample_size')}). "
                    f"Do not chart or reason about its pct_change_start_to_end."
                )

        for pt in verified_stats.get("period_trends", []):
            excluded = pt.get("excluded_years") or []
            if excluded:
                years_str = ", ".join(
                    f"{e.get('year')} ({e.get('row_count')} rows, needed {e.get('min_required')})"
                    for e in excluded
                )
                flagged.append(
                    f"- Period metric '{pt.get('metric_column')}': excluded year(s) {years_str}. "
                    f"Do not chart or reason about a YoY figure implying a collapse/spike "
                    f"driven by these excluded periods."
                )

        if not flagged:
            return "No trend or period metrics are flagged as unreliable — all sample sizes are sufficient."

        return "FLAGGED / UNRELIABLE METRICS (do not chart percentage claims from these):\n" + "\n".join(flagged)

    def _build_task(self, verified_stats, report_text, available_columns):
        # Filter to business columns BEFORE building the prompt
        biz_columns = self._business_columns(available_columns)

        stats_json     = json.dumps(verified_stats, indent=2, default=str)
        report_excerpt = (report_text or "")[:2000]
        reliability_note = self._reliability_note(verified_stats)

        # Build exclusion list — high cardinality categoricals
        cat_summary = verified_stats.get("category_summary", [])
        high_card_cols = [
            c["column"] for c in cat_summary
            if (c.get("unique_values") or 0) > 12
        ]
        near_id_cols = [
            c["column"] for c in cat_summary
            if (c.get("top_values") or [{}])[0].get("pct", 0) < 5
            and c["column"] not in high_card_cols
        ]
        # Also exclude derived columns from the exclusion note
        derived_in_stats = [
            c["column"] for c in cat_summary
            if any(c["column"].lower().endswith(sfx) for sfx in _DERIVED_SUFFIXES)
        ]
        excluded_cols = list(set(high_card_cols + near_id_cols + derived_in_stats))
        excluded_note = (
            f"\nEXCLUDED COLUMNS (do NOT use as x_column — too many unique values, "
            f"near-identifier, or derived date helpers):\n"
            + json.dumps(excluded_cols, indent=2)
            if excluded_cols else ""
        )

        return Task(
            description=f"""
Design an executive business dashboard from the verified statistics below.

VERIFIED STATISTICS (StatsEngine output — exact keys shown):
{stats_json}

NOTE on verified_stats structure:
- "kpis": list of {{column, mean, median, std, min, max}}
- "trends": list of {{date_column, value_column, direction, r_squared,
  pct_change_start_to_end, start_sample_size, end_sample_size,
  total_sample_size, insufficient_edge_data}}
- "period_trends": list of {{metric_column, yoy_pct_change, mom_pct_change,
  excluded_years, min_sample_size_required, summary}}
- "correlations": list of {{column_a, column_b, correlation, direction}}
- "category_summary": list of {{column, unique_values, top_values}}
- "anomalies": list of {{column, z_score_outliers, iqr_outliers, skew}}

{reliability_note}

BUSINESS REPORT EXCERPT:
{report_excerpt}

AVAILABLE COLUMNS — ONLY use these exact names (derived date columns already removed):
{json.dumps(biz_columns, indent=2)}
{excluded_note}

{CHART_USAGE_GUIDE}

{DATA_RELIABILITY_GUIDE}

{ANALYST_REASONING_GUIDE}

ALLOWED CHART TYPES: {sorted(ALLOWED_CHART_TYPES)}

RULES:
1. x_column and y_column MUST be copied exactly from AVAILABLE COLUMNS or be null
2. NEVER use any column listed in EXCLUDED COLUMNS as x_column or color_column
3. NEVER use columns ending in _year, _month, or _month_name — these are date helpers, not chart axes
4. For "line": x_column MUST be a date/time column — use the original date column (e.g. order_date).
   Check the FLAGGED / UNRELIABLE METRICS note above first — if this metric's trend is flagged,
   either skip the line chart or keep it but never reason about the flagged percentage.
5. For "pie"/"donut": x_column must have <= 8 unique values (check category_summary)
6. For "stacked_bar": x_column=category, y_column=numeric, color_column=category (all must exist)
7. For "bar"/"column": y_column should be numeric; x_column should be categorical with <= 12 unique values
8. For "scatter": both x_column and y_column MUST be numeric and MUST appear together in "correlations"
9. For "histogram": x_column MUST be numeric; y_column must be null
10. NEVER use funnel, area, box, or map — not supported
11. Select exactly 3 charts that together tell the complete, RELIABLE business story — favor
    decision-relevant metrics over vanity numbers (a metric nobody would act on doesn't earn
    a chart slot, and neither does a flagged/unreliable one)
12. Every chart must be backed by a number from verified statistics that is NOT flagged unreliable
13. Prefer "donut" over "pie" for a cleaner look
14. Only pick columns that represent real business activity (revenue, orders, ratings, etc.)
15. "reasoning" and "title" must read like a human analyst's insight (ratio, concentration,
    context, or the decision it supports) — not a bare restatement of a number

OUTPUT: valid JSON list only — no markdown, no explanation outside the JSON.

[
  {{
    "chart_type": "...",
    "title": "Specific business title with a number from verified stats",
    "x_column": "exact_column_name",
    "y_column": "exact_column_name_or_null",
    "color_column": null,
    "reasoning": "One sentence citing the specific verified number, framed like an analyst — what it means and what decision it supports.",
    "priority": 1
  }}
]
""",
            expected_output=(
                "Valid JSON list of exactly 3 chart specs. All column names exact matches "
                "from AVAILABLE COLUMNS. No derived date columns. No high-cardinality "
                "categorical columns used. No chart reasons on flagged/insufficient "
                "sample data. Reasoning reads like analyst insight, not a bare number."
            ),
            agent=self.agent,
        )

    def generate_chart_specs(self, verified_stats, report_text, available_columns, min_charts=3, max_charts=3):
        # Always filter to business columns first
        biz_columns = self._business_columns(available_columns)

        task = self._build_task(verified_stats, report_text, biz_columns)

        try:
            crew       = Crew(agents=[self.agent], tasks=[task], verbose=False)
            raw_result = crew.kickoff()
            specs      = self._parse_specs(str(raw_result))
        except Exception:
            specs = []

        valid_specs = self._validate_specs(specs, biz_columns, verified_stats)

        if len(valid_specs) < min_charts:
            existing_keys = {
                (s.get("chart_type"), s.get("x_column"), s.get("y_column"))
                for s in valid_specs
            }
            for fb in self._build_fallback_specs(verified_stats, biz_columns):
                key = (fb.get("chart_type"), fb.get("x_column"), fb.get("y_column"))
                if key not in existing_keys:
                    valid_specs.append(fb)
                    existing_keys.add(key)
                if len(valid_specs) >= min_charts:
                    break

        valid_specs.sort(key=lambda s: s.get("priority", 99))
        return valid_specs[:max_charts]

    def _build_fallback_specs(self, verified_stats, available_columns):
        """
        Deterministic fallbacks using the 7 approved chart types.
        Now skips any trend flagged "insufficient_edge_data" when picking
        the line-chart fallback, and writes analyst-style reasoning instead
        of a bare stat restatement.
        """
        col_set  = set(available_columns)
        fallback = []
        priority = 10

        # 1. Line from trends (date + value required, must be reliable)
        trends = sorted(
            [t for t in verified_stats.get("trends", []) if not t.get("insufficient_edge_data")],
            key=lambda t: t.get("r_squared", 0), reverse=True,
        )
        for t in trends[:1]:
            dc = t.get("date_column")
            vc = t.get("value_column")
            if dc in col_set and vc in col_set:
                fallback.append({
                    "chart_type": "line",
                    "title": f"{vc.replace('_', ' ').title()} Over Time",
                    "x_column": dc, "y_column": vc, "color_column": None,
                    "reasoning": (
                        f"r²={t.get('r_squared'):.2f}, {t.get('direction')} trend across "
                        f"{t.get('total_sample_size', 'all')} records — enough history to "
                        f"treat this direction as a real signal, not noise."
                    ),
                    "priority": priority,
                })
                priority += 1

        # 2. Donut / column from category_summary — flag concentration where relevant
        for cat in verified_stats.get("category_summary", [])[:3]:
            col    = cat.get("column")
            unique = cat.get("unique_values", 99)
            if col not in col_set:
                continue
            # Skip derived columns in fallback too
            if any(col.lower().endswith(sfx) for sfx in _DERIVED_SUFFIXES):
                continue
            chart_type = "donut" if unique <= 8 else "column"
            y_col = None
            if chart_type == "column":
                for k in verified_stats.get("kpis", []):
                    if k.get("column") in col_set and k.get("column") != col:
                        y_col = k["column"]
                        break

            top_values = cat.get("top_values") or []
            top_pct = top_values[0].get("pct") if top_values else None
            if top_pct and top_pct >= 40:
                reasoning = (
                    f"{unique} unique values in {col}; the top value alone accounts for "
                    f"{top_pct}% — worth checking whether that's a healthy concentration "
                    f"or a dependency risk."
                )
            else:
                reasoning = f"{unique} unique values in {col}, distribution is reasonably spread."

            fallback.append({
                "chart_type": chart_type,
                "title": f"{col.replace('_', ' ').title()} Distribution",
                "x_column": col, "y_column": y_col, "color_column": None,
                "reasoning": reasoning,
                "priority": priority,
            })
            priority += 1

        # 3. Bar for the top KPI by category — note mean-vs-median skew if present
        kpis        = verified_stats.get("kpis", [])
        cat_summary = verified_stats.get("category_summary", [])
        if kpis and cat_summary:
            primary_kpi_entry = kpis[0]
            primary_kpi = primary_kpi_entry.get("column")
            mean_v, median_v = primary_kpi_entry.get("mean"), primary_kpi_entry.get("median")
            skew_note = ""
            if mean_v and median_v and median_v != 0 and abs(mean_v - median_v) / abs(median_v) > 0.15:
                skew_note = (
                    f" Mean ({mean_v}) is notably above median ({median_v}), suggesting a "
                    f"handful of large values are pulling the average up."
                )
            for cat in cat_summary:
                cat_col = cat.get("column")
                if any(cat_col.lower().endswith(sfx) for sfx in _DERIVED_SUFFIXES):
                    continue
                if (cat_col in col_set and primary_kpi in col_set
                        and cat_col != primary_kpi
                        and 2 <= (cat.get("unique_values") or 0) <= 15):
                    fallback.append({
                        "chart_type": "bar",
                        "title": f"{primary_kpi.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
                        "x_column": cat_col, "y_column": primary_kpi, "color_column": None,
                        "reasoning": f"Top KPI breakdown by {cat_col}.{skew_note}",
                        "priority": priority,
                    })
                    priority += 1
                    break

        # 4. Scatter from strongest correlation (relationship between two numerics)
        correlations = sorted(
            verified_stats.get("correlations", []),
            key=lambda c: abs(c.get("correlation", 0) or 0), reverse=True,
        )
        for c in correlations[:1]:
            ca, cb = c.get("column_a"), c.get("column_b")
            if ca in col_set and cb in col_set and abs(c.get("correlation", 0) or 0) >= 0.3:
                fallback.append({
                    "chart_type": "scatter",
                    "title": f"{ca.replace('_',' ').title()} vs {cb.replace('_',' ').title()}",
                    "x_column": ca, "y_column": cb, "color_column": None,
                    "reasoning": (
                        f"Correlation r={c.get('correlation'):.2f} ({c.get('direction','')}) — "
                        f"worth investigating whether this relationship can be used predictively."
                    ),
                    "priority": priority,
                })
                priority += 1

        # 5. Histogram from the most anomalous/skewed numeric column
        anomalies_sorted = sorted(
            verified_stats.get("anomalies", []),
            key=lambda a: abs(a.get("skew", 0) or 0), reverse=True,
        )
        for a in anomalies_sorted[:1]:
            col = a.get("column")
            if col in col_set:
                fallback.append({
                    "chart_type": "histogram",
                    "title": f"{col.replace('_',' ').title()} Distribution",
                    "x_column": col, "y_column": None, "color_column": None,
                    "reasoning": (
                        f"Skew={a.get('skew', 0):.2f}, {a.get('z_score_outliers', 0)} outliers — "
                        f"the shape here matters more than the average for pricing/ops decisions."
                    ),
                    "priority": priority,
                })
                priority += 1

        return fallback

    def _parse_specs(self, raw_text):
        text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
        bracket = text.find("[")
        if bracket > 0:
            text = text[bracket:]
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed = parsed.get("charts", [])
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, AttributeError):
            return []

    def _validate_specs(self, specs, available_columns, verified_stats=None):
        col_set    = set(available_columns)
        valid      = []
        NEEDS_Y    = {"bar", "column", "stacked_bar", "line"}
        OPTIONAL_Y = {"pie", "donut"}

        # Build exclusion set
        excluded = set()
        if verified_stats:
            for c in verified_stats.get("category_summary", []):
                unique  = c.get("unique_values") or 0
                top_pct = (c.get("top_values") or [{}])[0].get("pct", 0)
                if unique > 12 or top_pct < 5:
                    excluded.add(c["column"])

        # Columns whose trend is flagged insufficient — block line charts on them
        # unless the spec avoids implying a percentage claim (we can't inspect
        # the LLM's prose intent here, so we conservatively still allow the
        # chart but this set is available for stricter future enforcement).
        unreliable_trend_cols = set()
        if verified_stats:
            for t in verified_stats.get("trends", []):
                if t.get("insufficient_edge_data"):
                    unreliable_trend_cols.add(t.get("value_column"))

        for spec in specs:
            if not isinstance(spec, dict):
                continue

            chart_type = str(spec.get("chart_type", "")).lower().strip()
            x_col      = spec.get("x_column")
            y_col      = spec.get("y_column")
            color_col  = spec.get("color_column")

            if chart_type not in ALLOWED_CHART_TYPES:
                continue
            if not x_col or x_col not in col_set:
                continue
            # Reject derived date columns as axes
            if any(x_col.lower().endswith(sfx) for sfx in _DERIVED_SUFFIXES):
                continue
            if x_col in excluded:
                continue
            if chart_type == "scatter" and (not y_col or y_col not in col_set):
                continue
            if chart_type == "histogram":
                spec["y_column"] = None
            elif chart_type in NEEDS_Y and (not y_col or y_col not in col_set):
                continue
            elif chart_type in OPTIONAL_Y and y_col and y_col not in col_set:
                spec["y_column"] = None
            if color_col and color_col not in col_set:
                spec["color_column"] = None

            spec["chart_type"] = chart_type
            valid.append(spec)

        return valid