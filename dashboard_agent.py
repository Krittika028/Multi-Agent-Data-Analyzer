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
"""

import os
import json
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

ALLOWED_CHART_TYPES = {
    "pie", "donut", "bar", "column", "stacked_bar", "line",
}

# Suffixes added by DataCleaner's date-split step — never useful as chart axes
_DERIVED_SUFFIXES = ("_year", "_month", "_month_name")

# Column name keywords that are structural/metadata, not business metrics
_NON_BUSINESS_KEYWORDS = [
    "_id", "row_number", "index", "record_id", "seq", "serial",
    "uuid", "guid", "hash",
]

CHART_USAGE_GUIDE = """
CHART TYPE SELECTION GUIDE (only these 5 types are allowed):

| chart_type  | Use when...                                             | x_column        | y_column        |
|-------------|--------------------------------------------------------|-----------------|-----------------|
| pie         | Composition / share — only if <= 8 unique values        | category column | null or numeric |
| donut       | Same as pie but with a centre hole — preferred over pie | category column | null or numeric |
| bar         | Horizontal category comparison (ranked list)            | category column | numeric (opt)   |
| column      | Vertical category or time-bucketed comparison           | category column | numeric (opt)   |
| stacked_bar | Part-to-whole across categories (needs color_col)       | category column | numeric column  |
| line        | Metric changing over TIME only — requires a date column | date column     | numeric column  |

IMPORTANT: Do NOT use scatter, histogram, area, funnel, or box charts.
Use "line" ONLY when x_column is a date/time column. For all other numeric comparisons, use bar or column.
"""


class DashboardAgent:

    def __init__(self, llm=None):
        self.llm = llm or LLM(
            model=os.getenv("GEMINI_MODEL"),
            api_key=os.getenv("GEMINI_API_KEY"),
        )

        self.agent = Agent(
            role="Expert Business Intelligence Analyst — Dashboard Specialist",
            goal=(
                "Design a focused executive dashboard by selecting 4–6 charts "
                "from verified statistics. Every x_column and y_column must be "
                "an exact match from AVAILABLE COLUMNS. Never invent column names. "
                "Only use columns that represent real business metrics — never "
                "use derived date parts (_year, _month, _month_name) as chart axes."
            ),
            backstory=(
                "You are a BI analyst who designs C-suite dashboards. You know "
                "exactly which chart type fits which business question. Your golden "
                "rule: never put a column on a chart unless it appears exactly in "
                "the available columns list AND it represents a real business metric. "
                "Date-derived helper columns like order_date_year or order_date_month "
                "are never chart axes — use the original date column for line charts."
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

    def _build_task(self, verified_stats, report_text, available_columns):
        # Filter to business columns BEFORE building the prompt
        biz_columns = self._business_columns(available_columns)

        stats_json     = json.dumps(verified_stats, indent=2, default=str)
        report_excerpt = (report_text or "")[:2000]

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
- "trends": list of {{date_column, value_column, direction, r_squared, pct_change_start_to_end}}
- "correlations": list of {{column_a, column_b, correlation, direction}}
- "category_summary": list of {{column, unique_values, top_values}}
- "anomalies": list of {{column, z_score_outliers, iqr_outliers, skew}}

BUSINESS REPORT EXCERPT:
{report_excerpt}

AVAILABLE COLUMNS — ONLY use these exact names (derived date columns already removed):
{json.dumps(biz_columns, indent=2)}
{excluded_note}

{CHART_USAGE_GUIDE}

ALLOWED CHART TYPES: {sorted(ALLOWED_CHART_TYPES)}

RULES:
1. x_column and y_column MUST be copied exactly from AVAILABLE COLUMNS or be null
2. NEVER use any column listed in EXCLUDED COLUMNS as x_column or color_column
3. NEVER use columns ending in _year, _month, or _month_name — these are date helpers, not chart axes
4. For "line": x_column MUST be a date/time column — use the original date column (e.g. order_date)
5. For "pie"/"donut": x_column must have <= 8 unique values (check category_summary)
6. For "stacked_bar": x_column=category, y_column=numeric, color_column=category (all must exist)
7. For "bar"/"column": y_column should be numeric; x_column should be categorical with <= 12 unique values
8. NEVER use scatter, histogram, area, funnel, or box — not supported
9. Select 4–6 charts that together tell the complete business story
10. Every chart must be backed by a number from verified statistics
11. Prefer "donut" over "pie" for a cleaner look
12. Only pick columns that represent real business activity (revenue, orders, ratings, etc.)

OUTPUT: valid JSON list only — no markdown, no explanation outside the JSON.

[
  {{
    "chart_type": "...",
    "title": "Specific business title with a number from verified stats",
    "x_column": "exact_column_name",
    "y_column": "exact_column_name_or_null",
    "color_column": null,
    "reasoning": "One sentence citing the specific verified number.",
    "priority": 1
  }}
]
""",
            expected_output=(
                "Valid JSON list of 4–6 chart specs. All column names exact matches "
                "from AVAILABLE COLUMNS. No derived date columns. No high-cardinality "
                "categorical columns used."
            ),
            agent=self.agent,
        )

    def generate_chart_specs(self, verified_stats, report_text, available_columns, min_charts=3):
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
        return valid_specs

    def _build_fallback_specs(self, verified_stats, available_columns):
        """Deterministic fallbacks using only the 5 approved chart types."""
        col_set  = set(available_columns)
        fallback = []
        priority = 10

        # 1. Line from trends (date + value required)
        trends = sorted(verified_stats.get("trends", []),
                        key=lambda t: t.get("r_squared", 0), reverse=True)
        for t in trends[:1]:
            dc = t.get("date_column")
            vc = t.get("value_column")
            if dc in col_set and vc in col_set:
                fallback.append({
                    "chart_type": "line",
                    "title": f"{vc.replace('_', ' ').title()} Over Time",
                    "x_column": dc, "y_column": vc, "color_column": None,
                    "reasoning": f"r²={t.get('r_squared'):.2f}, {t.get('direction')} trend",
                    "priority": priority,
                })
                priority += 1

        # 2. Donut / column from category_summary
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
            fallback.append({
                "chart_type": chart_type,
                "title": f"{col.replace('_', ' ').title()} Distribution",
                "x_column": col, "y_column": y_col, "color_column": None,
                "reasoning": f"{unique} unique values in {col}",
                "priority": priority,
            })
            priority += 1

        # 3. Bar for the top KPI by category
        kpis        = verified_stats.get("kpis", [])
        cat_summary = verified_stats.get("category_summary", [])
        if kpis and cat_summary:
            primary_kpi = kpis[0].get("column")
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
                        "reasoning": f"Top KPI breakdown by {cat_col}",
                        "priority": priority,
                    })
                    priority += 1
                    break

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
            if chart_type in NEEDS_Y and (not y_col or y_col not in col_set):
                continue
            if chart_type in OPTIONAL_Y and y_col and y_col not in col_set:
                spec["y_column"] = None
            if color_col and color_col not in col_set:
                spec["color_column"] = None

            spec["chart_type"] = chart_type
            valid.append(spec)

        return valid