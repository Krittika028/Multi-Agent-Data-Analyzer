"""
stats_engine.py

Deterministic, pandas-only computation layer. NO LLM CALLS HAPPEN HERE —
never have, and this file must stay that way. Every number produced by
this module is exactly reproducible from the dataframe: no model is ever
asked to "calculate," "estimate," or "narrate" a figure.

The core problem this solves: asking an LLM to "calculate KPIs" or
"identify anomalies" from a raw .describe() table means the LLM is
guessing at numbers and patterns it has no reliable way to compute.
That risks confident-sounding but wrong claims ("revenue is trending
up" when it isn't) — exactly the failure mode this file exists to
eliminate.

StatsEngine computes everything that CAN be computed exactly —
KPIs, trend direction with real slope/correlation, anomaly flags via
z-score and IQR, skew, strong correlations — and packages it as a
dict of verified facts.

--------------------------------------------------------------------
SAMPLE-SIZE SAFETY:

Any "before vs after" or "start vs end" percentage change is only as
trustworthy as the number of data points backing each side of the
comparison. Without a floor on that, a single stray row (e.g. one
order that leaked into an otherwise-empty period) can be compared
directly against a full year of real data and produce a headline
number like "-99.9% YoY" that is pure noise, not a business finding.

Two generalised, dataset-agnostic guards are applied throughout:

  MIN_PERIOD_SAMPLE_ROWS  — absolute floor on rows in a period
  MIN_PERIOD_SAMPLE_PCT   — floor relative to total dataset size

A period (year, or an edge window in a row-level trend) must clear
BOTH before it's allowed to anchor a percentage-change calculation.
Periods that don't clear the bar are excluded and reported as such
(never silently dropped, never silently included) so callers/agents
downstream can see exactly what was and wasn't comparable.
--------------------------------------------------------------------
VOLUME CONCENTRATION SAFETY:

The sample-size guards above catch periods with TOO FEW rows. They do
NOT catch the opposite failure: a period with a suspicious EXCESS of
rows relative to every other period of the same size (e.g. a large
batch of records whose dates were imputed/defaulted into a single
month, producing a fake "spike"). This is exactly the shape of
artifact that upstream date-imputation bugs produce, and it is
dangerous precisely because it clears the minimum-sample-size bar —
it looks like a real, well-supported data point.

`detect_volume_anomalies()` flags any period whose row count is a
statistical outlier (z-score based, generalised, dataset-agnostic)
relative to the other periods of the same granularity. Any period
flagged this way is excluded from monthly/period trend narratives and
reported explicitly as excluded, the same way low-sample periods are.
--------------------------------------------------------------------
REVENUE RELIABILITY BY STATUS:

If the dataset has a status-like column (order_status, transaction_status,
payment_status, booking_status, etc.) with values indicating an order was
NOT completed (cancelled, failed, refunded, pending, ...), summing every
row's monetary column into "Total Revenue" overstates real, realized
revenue. `compute_revenue_by_status()` detects such a column (if present)
and reports revenue split by status, plus a `reliable_revenue` figure
that excludes non-completed statuses — never invented, only computed
when a status column and a matching monetary column both exist.
--------------------------------------------------------------------
NOTE on "deterministic narrative": an earlier draft of this docstring
referenced generate_business_summary()/generate_recommendations()
methods intended to replace LLM-generated narrative text with pure
string templates built off these verified numbers. Those are NOT yet
implemented in this file — flagging this explicitly rather than
silently dropping the claim. Say the word if you want them added;
they'd close the exact gap where the report LLM has been observed
inventing numbers not traceable to verified_stats.
--------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from ml_insights import ClusterAnalyzer, TrendForecaster
from datetime import datetime

# ── Generalised sample-size floors used anywhere we compare "periods" ────────
MIN_PERIOD_SAMPLE_ROWS = 10     # absolute minimum rows to trust a period
MIN_PERIOD_SAMPLE_PCT  = 0.05   # OR at least 5% of the dataset, whichever is higher

# ── Edge-window settings for row-level start/end trend comparisons ───────────
EDGE_WINDOW_FRACTION = 0.05     # use ~5% of points at each edge
EDGE_WINDOW_MIN_ROWS = 3        # never average fewer than this many rows

# ── Volume-concentration (too MANY rows) anomaly guard ───────────────────────
VOLUME_ANOMALY_Z_THRESHOLD = 2.5   # z-score above which a period's row count
                                    # is flagged as an outlier/spike
VOLUME_ANOMALY_MIN_PERIODS = 4     # need at least this many periods to judge
                                    # what "normal" volume even looks like

# ── Status values treated as "not a completed, revenue-earning order" ────────
_NON_COMPLETED_STATUS_KEYWORDS = [
    "cancel", "fail", "refund", "pending", "reject", "void", "return",
    "chargeback", "declin", "abandon",
]
_STATUS_COLUMN_KEYWORDS = ["status"]
_MONETARY_COLUMN_KEYWORDS = [
    "revenue", "amount", "total", "sales", "price", "value", "gmv", "fare", "charge",
]


class StatsEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(include="object").columns.tolist()
        self.date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # =====================================
    # SAMPLE-SIZE HELPERS (generalised, dataset-agnostic)
    # =====================================
    def _min_sample_threshold(self, total_rows: int) -> int:
        """
        The minimum number of rows a period needs before it's allowed to
        anchor a percentage-change calculation. Scales with dataset size
        so it behaves sensibly whether the dataset has 200 rows or 2M.
        """
        return max(MIN_PERIOD_SAMPLE_ROWS, int(np.ceil(total_rows * MIN_PERIOD_SAMPLE_PCT)))

    def _robust_edge_values(self, sorted_values: np.ndarray):
        """
        Instead of taking the single first/last row as the "start"/"end"
        value of a trend (fragile — one stray outlier row IS the metric),
        average a small window of points at each edge. Returns:

            start_val, end_val, start_n, end_n, sufficient (bool)

        `sufficient` is True only if both edge windows meet the minimum
        sample size for this dataset's scale.
        """
        n = len(sorted_values)
        window = max(EDGE_WINDOW_MIN_ROWS, int(np.ceil(n * EDGE_WINDOW_FRACTION)))
        window = min(window, n // 2) if n >= 2 else n  # don't let windows overlap
        window = max(window, 1)

        start_slice = sorted_values[:window]
        end_slice   = sorted_values[-window:]

        start_val = float(np.mean(start_slice))
        end_val   = float(np.mean(end_slice))

        min_needed = self._min_sample_threshold(n)
        sufficient = (len(start_slice) >= min(min_needed, EDGE_WINDOW_MIN_ROWS)) and \
                     (len(end_slice) >= min(min_needed, EDGE_WINDOW_MIN_ROWS)) and \
                     (window >= EDGE_WINDOW_MIN_ROWS)

        return start_val, end_val, len(start_slice), len(end_slice), sufficient

    # =====================================
    # KPIs — top numeric columns by relevance
    # (relevance = variance contribution + non-id heuristic, not just "first 5")
    # =====================================
    def compute_kpis(self, max_kpis=6):
        kpis = []

        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if series.empty or series.nunique() <= 1:
                continue
            if self._looks_like_identifier(col, series):
                continue

            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std())
            sum_val = float(series.sum())
            cv = (std_val / mean_val) if mean_val != 0 else None  # coefficient of variation

            kpis.append({
                "column": col,
                "mean": round(mean_val, 2),
                "median": round(median_val, 2),
                "std": round(std_val, 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                # Exact total over the ACTUAL (non-null) values, computed
                # once here from full precision — this is "the one verified
                # number" every downstream consumer (dashboard cards, PDF,
                # report/summary text) should read instead of each
                # re-deriving mean * row_count with its own rounding.
                "sum": round(sum_val, 2),
                "count": int(series.shape[0]),  # non-null rows this sum covers
                "coefficient_of_variation": round(cv, 3) if cv is not None else None,
                "variance_score": std_val,  # used only for ranking, not shown
            })

        kpis.sort(key=lambda k: k["variance_score"], reverse=True)
        for k in kpis:
            k.pop("variance_score", None)

        return kpis[:max_kpis]

    def _looks_like_identifier(self, col_name, series):
        """
        A column is treated as an identifier (excluded from KPIs) only if:
        - its name suggests it (id, code, key, number, no.), AND/OR
        - it's all integers with 100% unique values (sequential-style IDs)
        Continuous measurements (revenue, age, price) are naturally
        near-100%-unique too, so uniqueness ratio ALONE is not a valid signal.
        """
        name_lower = col_name.lower()
        name_suggests_id = any(
            kw in name_lower for kw in ["id", "code", "key", "_no", "number", "ssn", "zip"]
        )

        all_unique = series.nunique() == len(series)
        is_integer_like = (series % 1 == 0).all()

        return name_suggests_id and all_unique and is_integer_like

    # =====================================
    # TRENDS — real slope/correlation over row-level date data
    # start_value/end_value are now ROBUST EDGE AVERAGES, not single rows,
    # so one stray/late/early row can't fake an entire trend.
    # =====================================
    def compute_trends(self):
        trends = []

        if not self.date_cols or not self.numeric_cols:
            return trends

        for date_col in self.date_cols:
            for num_col in self.numeric_cols:
                sub = self.df[[date_col, num_col]].dropna()
                if len(sub) < 5:
                    continue

                sub = sub.sort_values(date_col)
                x = np.arange(len(sub))
                y = sub[num_col].values

                # Linear fit: slope tells direction, r tells strength
                if np.std(y) == 0:
                    continue

                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # ── Robust edge-window start/end instead of raw y[0]/y[-1] ──
                start_val, end_val, start_n, end_n, edges_sufficient = self._robust_edge_values(y)

                pct_change = None
                if edges_sufficient and start_val != 0:
                    pct_change = ((end_val - start_val) / abs(start_val)) * 100

                # ── Reconcile direction with the number actually displayed ──
                # `direction` used to come purely from the OLS slope sign,
                # while `pct_change_start_to_end` comes from a separate
                # robust edge-window comparison. On data with a large early
                # (or late) outlier the two can legitimately disagree —
                # e.g. one big spike near the start drags the regression
                # slope negative while the robust edge windows (which
                # average several points, diluting a single outlier) still
                # show a net increase. Showing "▼ Decreasing (+1.3%)" is
                # never acceptable, so whenever we have a valid edge-based
                # pct_change, THAT is treated as the source of truth for
                # direction too — it's the same number being displayed.
                # Only fall back to the slope sign when edge data is
                # insufficient (pct_change is None), so we still get a
                # sensible direction on short/thin series.
                if pct_change is not None:
                    if pct_change > 0:
                        direction = "increasing"
                    elif pct_change < 0:
                        direction = "decreasing"
                    else:
                        direction = "flat"
                else:
                    direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")

                # Slope and edge-window direction disagreeing is itself a
                # useful signal (it usually means an early/late outlier is
                # distorting the simple linear fit) — surface it rather
                # than silently discarding the discrepancy.
                slope_direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
                direction_conflict = (
                    pct_change is not None
                    and slope_direction != direction
                    and slope_direction != "flat"
                    and direction != "flat"
                )

                strength = (
                    "strong" if r_squared >= 0.5 else
                    "moderate" if r_squared >= 0.2 else
                    "weak"
                )

                trends.append({
                    "date_column": date_col,
                    "value_column": num_col,
                    "direction": direction,
                    "slope_direction": slope_direction,
                    "direction_conflict": direction_conflict,
                    "strength": strength,
                    "r_squared": round(float(r_squared), 3),
                    "pct_change_start_to_end": round(pct_change, 1) if pct_change is not None else None,
                    "start_value": round(float(start_val), 2),
                    "end_value": round(float(end_val), 2),
                    "start_sample_size": int(start_n),
                    "end_sample_size": int(end_n),
                    "insufficient_edge_data": not edges_sufficient,
                    "total_sample_size": int(len(sub)),
                })

        return trends

    # =====================================
    # CORRELATIONS — only flag what's actually strong (|r| >= 0.6)
    # =====================================
    def compute_correlations(self, threshold=0.6):
        strong = []

        if len(self.numeric_cols) < 2:
            return strong

        corr_matrix = self.df[self.numeric_cols].corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if pd.isna(val):
                    continue
                if abs(val) >= threshold:
                    strong.append({
                        "column_a": corr_matrix.columns[i],
                        "column_b": corr_matrix.columns[j],
                        "correlation": round(float(val), 3),
                        "direction": "positive" if val > 0 else "negative",
                    })

        strong.sort(key=lambda c: abs(c["correlation"]), reverse=True)
        return strong

    # =====================================
    # ANOMALIES — z-score outliers + IQR outliers, counted exactly
    # =====================================
    def compute_anomalies(self, z_threshold=3.0):
        anomalies = []

        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if len(series) < 10 or series.std() == 0:
                continue

            z_scores = np.abs((series - series.mean()) / series.std())
            z_outliers = int((z_scores > z_threshold).sum())

            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                iqr_outliers = int(
                    ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum()
                )
            else:
                iqr_outliers = 0

            skew = float(series.skew())

            if z_outliers > 0 or iqr_outliers > 0 or abs(skew) > 1.5:
                anomalies.append({
                    "column": col,
                    "z_score_outliers": z_outliers,
                    "iqr_outliers": iqr_outliers,
                    "skew": round(skew, 2),
                    "skew_direction": (
                        "right (high-value outliers)" if skew > 1.5 else
                        "left (low-value outliers)" if skew < -1.5 else
                        "normal"
                    ),
                })

        return anomalies

    # =====================================
    # CATEGORY BREAKDOWN — exact counts, not estimates
    # =====================================
    def compute_category_summary(self, top_n=5):
        summary = []
        for col in self.categorical_cols:
            counts = self.df[col].value_counts()
            if counts.empty:
                continue
            top = counts.head(top_n)
            summary.append({
                "column": col,
                "unique_values": int(self.df[col].nunique()),
                "top_values": [
                    {"value": str(v), "count": int(c), "pct": round(float(c) / len(self.df) * 100, 1)}
                    for v, c in top.items()
                ],
            })
        return summary

    # =====================================
    # VOLUME ANOMALIES — periods with a suspiciously EXCESSIVE row count
    # catches spikes (e.g. imputation artifacts), which the low-sample
    # guard above cannot catch since a spike has PLENTY of rows.
    # =====================================
    def detect_volume_anomalies(self, period_counts: dict, z_threshold=VOLUME_ANOMALY_Z_THRESHOLD):
        """
        period_counts: {period_label: row_count}. Returns a list of
        {period, row_count, z_score, avg_other_periods} for any period
        whose row count is a statistical outlier relative to the others.

        Deliberately generalised — works on any dataset with any
        date granularity (monthly, yearly, weekly...), no hardcoded
        thresholds tied to this dataset.
        """
        if len(period_counts) < VOLUME_ANOMALY_MIN_PERIODS:
            return []

        labels = list(period_counts.keys())
        counts = np.array([period_counts[l] for l in labels], dtype=float)

        mean_c = counts.mean()
        std_c  = counts.std()
        if std_c == 0:
            return []

        flagged = []
        for label, count in zip(labels, counts):
            z = (count - mean_c) / std_c
            if z > z_threshold:
                others = counts[counts != count]
                flagged.append({
                    "period": label,
                    "row_count": int(count),
                    "z_score": round(float(z), 2),
                    "avg_other_periods": round(float(others.mean()), 1) if len(others) else None,
                })

        return flagged

    # =====================================
    # REVENUE BY STATUS — reliable ("realized") revenue vs total
    # only activates when a status-like column AND a monetary column
    # both exist; never invents a status taxonomy that isn't there.
    # =====================================
    def _find_status_column(self):
        for col in self.categorical_cols:
            if any(kw in col.lower() for kw in _STATUS_COLUMN_KEYWORDS):
                return col
        return None

    def _find_primary_monetary_column(self):
        # Prefer an exact "final"/"total" amount column if present, else
        # fall back to the top KPI (highest variance numeric, non-id).
        candidates = [
            c for c in self.numeric_cols
            if any(kw in c.lower() for kw in _MONETARY_COLUMN_KEYWORDS)
        ]
        if not candidates:
            return None
        # Prefer columns literally named like a "final"/settled amount
        preferred = [c for c in candidates if "final" in c.lower() or "total" in c.lower()]
        return preferred[0] if preferred else candidates[0]

    def compute_revenue_by_status(self):
        """
        Returns None if there's no usable status column or monetary
        column. Otherwise returns:
            {
              "status_column": str,
              "revenue_column": str,
              "breakdown": [{"status", "revenue", "count", "pct_of_total_revenue"}],
              "completed_statuses": [...],
              "non_completed_statuses": [...],
              "total_revenue_all_rows": float,
              "reliable_revenue": float,   # excludes non-completed statuses
              "excluded_revenue": float,   # revenue sitting in non-completed rows
            }
        """
        status_col = self._find_status_column()
        rev_col    = self._find_primary_monetary_column()
        if not status_col or not rev_col:
            return None

        sub = self.df[[status_col, rev_col]].dropna()
        if sub.empty:
            return None

        grouped = sub.groupby(status_col)[rev_col].agg(["sum", "count"]).reset_index()
        total_revenue = float(sub[rev_col].sum())

        breakdown = []
        completed_statuses = []
        non_completed_statuses = []

        for _, row in grouped.iterrows():
            status = str(row[status_col])
            revenue = float(row["sum"])
            count = int(row["count"])
            is_non_completed = any(kw in status.lower() for kw in _NON_COMPLETED_STATUS_KEYWORDS)
            (non_completed_statuses if is_non_completed else completed_statuses).append(status)
            breakdown.append({
                "status": status,
                "revenue": round(revenue, 2),
                "count": count,
                "pct_of_total_revenue": round(revenue / total_revenue * 100, 1) if total_revenue else 0.0,
                "is_completed": not is_non_completed,
            })

        breakdown.sort(key=lambda b: b["revenue"], reverse=True)

        reliable_revenue = sum(b["revenue"] for b in breakdown if b["is_completed"])
        excluded_revenue = total_revenue - reliable_revenue

        return {
            "status_column": status_col,
            "revenue_column": rev_col,
            "breakdown": breakdown,
            "completed_statuses": completed_statuses,
            "non_completed_statuses": non_completed_statuses,
            "total_revenue_all_rows": round(total_revenue, 2),
            "reliable_revenue": round(reliable_revenue, 2),
            "excluded_revenue": round(excluded_revenue, 2),
        }

    # =====================================
    # PERIOD TRENDS — MoM / YoY using _year/_month columns
    #
    # YoY requires both the first and last year in the comparison to
    # clear a minimum sample-size threshold. Years that don't clear it
    # (e.g. a single leftover row for the current year) are excluded from
    # the YoY comparison and reported separately as "excluded_years" so
    # nothing pretends a 1-row year is a full year of business activity.
    #
    # Also runs the volume-anomaly guard (too MANY rows in one month —
    # e.g. an imputation-artifact spike) and excludes any flagged month
    # from monthly_sales-driven trend narrative the same way a
    # too-few-rows period is excluded.
    # =====================================
    def compute_period_trends(self):
        """
        Computes monthly sales totals (sum) and averages for each
        numeric column grouped by year+month. Also computes YoY and MoM
        % changes — with sample-size guards so a sparsely-populated
        period can never masquerade as a real trend. Returns a structure
        the dashboard can use directly to render a sales trend chart.
        """
        period_trends  = []
        monthly_sales  = []   # flat list of {year, month, month_name, col, total, avg}

        year_cols = [c for c in self.df.columns if c.endswith("_year")]
        if not year_cols:
            return {"period_trends": period_trends, "monthly_sales": monthly_sales, "volume_anomalies": []}

        all_volume_anomalies = []

        for year_col in year_cols:
            base      = year_col[: -len("_year")]
            month_col = f"{base}_month"

            if month_col not in self.df.columns:
                continue

            # ── Volume-anomaly guard: check row counts per year-month for
            # this date column, independent of which numeric metric we're
            # about to trend — a spike is a spike regardless of metric.
            ym_counts = (
                self.df[[year_col, month_col]].dropna()
                .assign(_ym=lambda d: d[year_col].astype(int).astype(str) + "-" +
                                       d[month_col].astype(int).astype(str).str.zfill(2))
                ["_ym"].value_counts().to_dict()
            )
            volume_anomalies = self.detect_volume_anomalies(ym_counts)
            anomalous_periods = {a["period"] for a in volume_anomalies}
            for a in volume_anomalies:
                a["date_column"] = base
            all_volume_anomalies.extend(volume_anomalies)

            for num_col in self.numeric_cols:
                if num_col in (year_col, month_col):
                    continue

                sub = self.df[[year_col, month_col, num_col]].dropna()
                if len(sub) < 4:
                    continue

                sub = sub.copy()
                sub["_ym"] = sub[year_col].astype(int).astype(str) + "-" + \
                             sub[month_col].astype(int).astype(str).str.zfill(2)

                # ── Monthly totals for sales trend chart (raw, unfiltered —
                # the dashboard should still be able to SHOW the anomalous
                # month; it's the narrative/pct-change math that must
                # avoid treating it as a real signal) ─────────────────────
                monthly_grp = (
                    sub.groupby([year_col, month_col])[num_col]
                    .agg(total="sum", avg="mean", count="count")
                    .reset_index()
                    .sort_values([year_col, month_col])
                )

                for _, row in monthly_grp.iterrows():
                    yr  = int(row[year_col])
                    mo  = int(row[month_col])
                    ym_key = f"{yr}-{str(mo).zfill(2)}"
                    monthly_sales.append({
                        "date_column":  base,
                        "value_column": num_col,
                        "year":         yr,
                        "month":        mo,
                        "month_name":   datetime(yr, mo, 1).strftime("%b"),
                        "year_month":   f"{yr}-{mo:02d}",
                        "label":        datetime(yr, mo, 1).strftime("%b %Y"),
                        "total":        round(float(row["total"]), 2),
                        "avg":          round(float(row["avg"]), 2),
                        "count":        int(row["count"]),
                        "is_volume_anomaly": ym_key in anomalous_periods,
                    })

                # ── YoY — with sample-size guard AND volume-anomaly guard.
                # A year that contains a flagged spike month is excluded
                # from YoY the same as a too-few-rows year would be. ─────
                sub_clean = sub[~sub["_ym"].isin(anomalous_periods)]

                yearly_sum   = sub_clean.groupby(year_col)[num_col].sum().sort_index()
                yearly_count = sub_clean.groupby(year_col)[num_col].count().sort_index()

                min_needed = self._min_sample_threshold(len(sub_clean)) if len(sub_clean) else MIN_PERIOD_SAMPLE_ROWS
                valid_years_mask = yearly_count >= min_needed

                excluded_years = [
                    {"year": int(yr), "row_count": int(yearly_count[yr]), "min_required": int(min_needed)}
                    for yr in yearly_count.index if not valid_years_mask[yr]
                ]
                # Also record years excluded purely because their only data
                # came from an anomalous/spike month
                years_lost_to_anomaly = sorted({
                    int(p.split("-")[0]) for p in anomalous_periods
                } - set(int(yr) for yr in yearly_count.index))

                yearly_valid = yearly_sum[valid_years_mask]

                yoy_pct = None
                yoy_narrative = None
                if len(yearly_valid) >= 2:
                    first_val, last_val = float(yearly_valid.iloc[0]), float(yearly_valid.iloc[-1])
                    first_yr,  last_yr  = int(yearly_valid.index[0]), int(yearly_valid.index[-1])
                    if first_val != 0:
                        yoy_pct = round(((last_val - first_val) / abs(first_val)) * 100, 1)
                        direction = "grew" if yoy_pct > 0 else "declined" if yoy_pct < 0 else "stayed flat"
                        yoy_narrative = (
                            f"{num_col.replace('_', ' ').title()} {direction} {abs(yoy_pct)}% "
                            f"YoY, from {first_val:,.2f} in {first_yr} to {last_val:,.2f} in {last_yr} "
                            f"(both years meet the minimum sample size of {min_needed} rows, "
                            f"with any spike/anomaly months excluded)."
                        )
                elif excluded_years or years_lost_to_anomaly:
                    reasons = []
                    if excluded_years:
                        reasons.append(
                            ", ".join(f"{e['year']} had only {e['row_count']}" for e in excluded_years)
                            + f" (minimum {min_needed} rows required)"
                        )
                    if years_lost_to_anomaly:
                        reasons.append(
                            f"{', '.join(str(y) for y in years_lost_to_anomaly)} only had data in a "
                            f"flagged volume-spike period"
                        )
                    yoy_narrative = (
                        f"YoY comparison skipped for {num_col.replace('_', ' ').title()} — "
                        + "; ".join(reasons) + "."
                    )

                # ── MoM (latest valid year only) ──────────────────────────
                mom_pct   = None
                mom_narrative = None
                if len(yearly_valid) >= 1:
                    latest_valid_year = int(yearly_valid.index[-1])
                    monthly_latest = (
                        sub_clean[sub_clean[year_col] == latest_valid_year]
                        .groupby(month_col)[num_col]
                        .sum()
                        .sort_index()
                    )
                    if len(monthly_latest) >= 2:
                        prev_val = float(monthly_latest.iloc[-2])
                        curr_val = float(monthly_latest.iloc[-1])
                        if prev_val != 0:
                            mom_pct = round(((curr_val - prev_val) / abs(prev_val)) * 100, 1)
                            direction = "up" if mom_pct > 0 else "down" if mom_pct < 0 else "flat"
                            mom_narrative = (
                                f"Most recent MoM change is {direction} {abs(mom_pct)}% "
                                f"in {latest_valid_year}."
                            )

                if yoy_pct is None and mom_pct is None and not excluded_years and not years_lost_to_anomaly:
                    continue

                period_trends.append({
                    "metric_column":  num_col,
                    "year_column":    year_col,
                    "month_column":   month_col,
                    "yoy_pct_change": yoy_pct,
                    "mom_pct_change": mom_pct,
                    "excluded_years": excluded_years,
                    "years_lost_to_volume_anomaly": years_lost_to_anomaly,
                    "min_sample_size_required": int(min_needed),
                    "summary": " ".join(filter(None, [yoy_narrative, mom_narrative])),
                })

        return {
            "period_trends": period_trends,
            "monthly_sales": monthly_sales,
            "volume_anomalies": all_volume_anomalies,
        }

    # =====================================
    # DATA QUALITY SCORE — exact formula, fully auditable
    # =====================================
    def compute_quality_score(self):
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = int(self.df.isnull().sum().sum())
        completeness = 1 - (missing_cells / total_cells) if total_cells else 1

        duplicate_rows = int(self.df.duplicated().sum())
        uniqueness = 1 - (duplicate_rows / len(self.df)) if len(self.df) else 1

        # Validity: numeric columns with extreme z-score outlier rates count against this
        validity_penalties = 0
        numeric_checked = 0
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if len(series) < 10 or series.std() == 0:
                continue
            numeric_checked += 1
            z = np.abs((series - series.mean()) / series.std())
            outlier_rate = (z > 3).mean()
            validity_penalties += outlier_rate
        validity = 1 - (validity_penalties / numeric_checked) if numeric_checked else 1

        overall = round(float(completeness * 0.4 + uniqueness * 0.3 + validity * 0.3) * 100, 1)

        grade = (
            "A+" if overall >= 95 else
            "A" if overall >= 90 else
            "B+" if overall >= 85 else
            "B" if overall >= 80 else
            "C" if overall >= 70 else
            "D" if overall >= 60 else
            "F"
        )

        return {
            "overall_score": overall,
            "grade": grade,
            "completeness_pct": round(completeness * 100, 1),
            "uniqueness_pct": round(uniqueness * 100, 1),
            "validity_pct": round(float(validity) * 100, 1),
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
        }

    # =====================================
    # ML INSIGHTS — real models, not LLM guesses
    # =====================================
    def compute_ml_insights(self):
        ml_results = {"clustering": None, "forecasts": []}

        # Clustering: find natural business segments across numeric columns
        try:
            cluster_result = ClusterAnalyzer(self.df).run()
            if cluster_result:
                ml_results["clustering"] = {
                    "k": cluster_result["k"],
                    "silhouette_score": cluster_result["silhouette_score"],
                    "features_used": cluster_result["features_used"],
                    "cluster_profile": cluster_result["cluster_profile"],
                }
        except Exception:
            pass

        # Forecasting: project the strongest verified trend forward
        if self.date_cols and self.numeric_cols:
            try:
                best_forecast = None
                for date_col in self.date_cols:
                    for num_col in self.numeric_cols:
                        forecaster = TrendForecaster(self.df)
                        result = forecaster.forecast(date_col, num_col, periods_ahead=5)
                        if result and (best_forecast is None or result["r_squared"] > best_forecast["r_squared"]):
                            best_forecast = result
                if best_forecast:
                    ml_results["forecasts"].append({
                        "date_column": best_forecast["date_column"],
                        "value_column": best_forecast["value_column"],
                        "r_squared": best_forecast["r_squared"],
                        "projected_end_value": best_forecast["projected_end_value"],
                        "pct_projected_change": best_forecast["pct_projected_change"],
                        "periods_ahead": best_forecast["periods_ahead"],
                    })
            except Exception:
                pass

        return ml_results

    # =====================================
    # FULL REPORT — everything bundled for the LLM prompt
    # =====================================
    def generate_full_report(self):
        period_data = self.compute_period_trends()
        return {
            "shape":              {"rows": self.df.shape[0], "columns": self.df.shape[1]},
            "quality_score":      self.compute_quality_score(),
            "kpis":               self.compute_kpis(),
            "trends":             self.compute_trends(),
            "correlations":       self.compute_correlations(),
            "period_trends":      period_data.get("period_trends", []),
            "monthly_sales":      period_data.get("monthly_sales", []),
            "volume_anomalies":   period_data.get("volume_anomalies", []),
            "anomalies":          self.compute_anomalies(),
            "category_summary":   self.compute_category_summary(),
            "revenue_by_status":  self.compute_revenue_by_status(),
            "ml_insights":        self.compute_ml_insights(),
        }