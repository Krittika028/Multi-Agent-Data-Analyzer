"""
stats_engine.py

Deterministic, pandas-only computation layer. No LLM calls happen here.

The core problem this solves: asking an LLM to "calculate KPIs" or
"identify anomalies" from a raw .describe() table means the LLM is
guessing at numbers and patterns it has no reliable way to compute.
That risks confident-sounding but wrong claims ("revenue is trending
up" when it isn't).

StatsEngine computes everything that CAN be computed exactly —
KPIs, trend direction with real slope/correlation, anomaly flags via
z-score and IQR, skew, strong correlations — and packages it as a
dict of verified facts. The LLM's job becomes narrating and
recommending based on these facts, not inventing them.
"""

import pandas as pd
import numpy as np
from ml_insights import ClusterAnalyzer, TrendForecaster
from datetime import datetime

class StatsEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(include="object").columns.tolist()
        self.date_cols = df.select_dtypes(include="datetime").columns.tolist()

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
            cv = (std_val / mean_val) if mean_val != 0 else None  # coefficient of variation

            kpis.append({
                "column": col,
                "mean": round(mean_val, 2),
                "median": round(median_val, 2),
                "std": round(std_val, 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
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


    # not just "looks like it's going up"
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

                pct_change = (
                    ((y[-1] - y[0]) / abs(y[0])) * 100 if y[0] != 0 else None
                )

                direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
                strength = (
                    "strong" if r_squared >= 0.5 else
                    "moderate" if r_squared >= 0.2 else
                    "weak"
                )

                trends.append({
                    "date_column": date_col,
                    "value_column": num_col,
                    "direction": direction,
                    "strength": strength,
                    "r_squared": round(float(r_squared), 3),
                    "pct_change_start_to_end": round(pct_change, 1) if pct_change is not None else None,
                    "start_value": round(float(y[0]), 2),
                    "end_value": round(float(y[-1]), 2),
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
    # PERIOD TRENDS — MoM / YoY using _year/_month columns
    # =====================================
    def compute_period_trends(self):
        """
        Computes monthly sales totals (sum) and averages for each
        numeric column grouped by year+month. Also computes YoY and MoM
        % changes. Returns a structure the dashboard can use directly
        to render a sales trend chart.
        """
        period_trends  = []
        monthly_sales  = []   # flat list of {year, month, month_name, col, total, avg}

        year_cols = [c for c in self.df.columns if c.endswith("_year")]
        if not year_cols:
            return {"period_trends": period_trends, "monthly_sales": monthly_sales}

        for year_col in year_cols:
            base      = year_col[: -len("_year")]
            month_col = f"{base}_month"
            month_name_col = f"{base}_month_name"

            if month_col not in self.df.columns:
                continue

            for num_col in self.numeric_cols:
                if num_col in (year_col, month_col):
                    continue

                sub = self.df[[year_col, month_col, num_col]].dropna()
                if len(sub) < 4:
                    continue

                # ── Monthly totals for sales trend chart ─────────────────────
                monthly_grp = (
                    sub.groupby([year_col, month_col])[num_col]
                    .agg(total="sum", avg="mean", count="count")
                    .reset_index()
                    .sort_values([year_col, month_col])
                )

                for _, row in monthly_grp.iterrows():
                    yr  = int(row[year_col])
                    mo  = int(row[month_col])
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
                    })

                # ── YoY ──────────────────────────────────────────────────────
                yearly    = sub.groupby(year_col)[num_col].sum().sort_index()
                yoy_pct   = None
                yoy_narrative = None
                if len(yearly) >= 2:
                    first_val, last_val = float(yearly.iloc[0]), float(yearly.iloc[-1])
                    first_yr,  last_yr  = int(yearly.index[0]), int(yearly.index[-1])
                    if first_val != 0:
                        yoy_pct = round(((last_val - first_val) / abs(first_val)) * 100, 1)
                        direction = "grew" if yoy_pct > 0 else "declined" if yoy_pct < 0 else "stayed flat"
                        yoy_narrative = (
                            f"{num_col.replace('_', ' ').title()} {direction} {abs(yoy_pct)}% "
                            f"YoY, from {first_val:,.2f} in {first_yr} to {last_val:,.2f} in {last_yr}."
                        )

                # ── MoM (latest year only) ────────────────────────────────────
                mom_pct   = None
                mom_narrative = None
                latest_year = int(sub[year_col].max())
                monthly_latest = (
                    sub[sub[year_col] == latest_year]
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
                            f"in {latest_year}."
                        )

                if yoy_pct is None and mom_pct is None:
                    continue

                period_trends.append({
                    "metric_column":  num_col,
                    "year_column":    year_col,
                    "month_column":   month_col,
                    "yoy_pct_change": yoy_pct,
                    "mom_pct_change": mom_pct,
                    "summary": " ".join(filter(None, [yoy_narrative, mom_narrative])),
                })

        return {"period_trends": period_trends, "monthly_sales": monthly_sales}

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
            "shape":           {"rows": self.df.shape[0], "columns": self.df.shape[1]},
            "quality_score":   self.compute_quality_score(),
            "kpis":            self.compute_kpis(),
            "trends":          self.compute_trends(),
            "correlations":    self.compute_correlations(),
            "period_trends":   period_data.get("period_trends", []),
            "monthly_sales":   period_data.get("monthly_sales", []),
            "anomalies":       self.compute_anomalies(),
            "category_summary": self.compute_category_summary(),
            "ml_insights":     self.compute_ml_insights(),
        }