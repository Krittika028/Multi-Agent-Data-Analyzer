"""
ml_insights.py

Real machine learning, computed deterministically — not LLM-guessed.

1. ClusterAnalyzer: KMeans clustering on numeric columns to find natural
   business segments (e.g. customer groups, transaction types). Picks
   k automatically via silhouette score rather than a hardcoded guess.

2. TrendForecaster: simple linear regression forecast on a date+value
   pair, projecting N periods forward with a confidence band based on
   residual standard error. This is intentionally simple (not ARIMA/
   Prophet) so it's transparent and fast — the point is to show "what's
   likely next" grounded in the actual historical slope, not a guess.

Both classes return plain dicts/dataframes so dashboard_agent.py and
dashboard_renderer.py can consume them without any ML knowledge.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class ClusterAnalyzer:

    def __init__(self, df: pd.DataFrame, max_k=6, min_rows=20):
        self.df = df
        self.max_k = max_k
        self.min_rows = min_rows

    def _select_features(self):
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        # Drop near-identifier columns (all unique integers) — same logic
        # rationale as StatsEngine: identifiers aren't meaningful features.
        features = []
        for col in numeric_cols:
            series = self.df[col].dropna()
            if series.empty or series.nunique() <= 1:
                continue
            is_integer_like = (series % 1 == 0).all()
            all_unique = series.nunique() == len(series)
            name_suggests_id = any(kw in col.lower() for kw in ["id", "code", "key", "number"])
            if is_integer_like and all_unique and name_suggests_id:
                continue
            features.append(col)
        return features

    def run(self):
        """
        Returns None if clustering isn't meaningful for this dataset
        (too few rows, too few usable numeric features). Otherwise
        returns a dict with cluster labels, chosen k, feature columns
        used, and a per-cluster profile (mean of each feature).
        """
        features = self._select_features()

        if len(self.df) < self.min_rows or len(features) < 2:
            return None

        data = self.df[features].dropna()
        if len(data) < self.min_rows:
            return None

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        best_k, best_score, best_labels = None, -1, None
        max_k = min(self.max_k, len(data) // 10)  # avoid over-clustering small data

        for k in range(2, max(3, max_k + 1)):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(scaled)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(scaled, labels)
                if score > best_score:
                    best_k, best_score, best_labels = k, score, labels
            except Exception:
                continue

        if best_labels is None:
            return None

        # Weak clustering structure isn't worth presenting as a finding
        if best_score < 0.15:
            return None

        data = data.copy()
        data["cluster"] = best_labels

        profile = (
            data.groupby("cluster")[features]
            .mean()
            .round(2)
            .reset_index()
        )
        cluster_sizes = data["cluster"].value_counts().sort_index()
        profile["size"] = cluster_sizes.values
        profile["pct_of_total"] = round(profile["size"] / len(data) * 100, 1)

        return {
            "k": int(best_k),
            "silhouette_score": round(float(best_score), 3),
            "features_used": features,
            "cluster_profile": profile.to_dict(orient="records"),
            "row_cluster_labels": data["cluster"],  # aligned to `data.index`, for plotting
            "data_index": data.index,
        }


class TrendForecaster:

    def __init__(self, df: pd.DataFrame, min_points=8):
        self.df = df
        self.min_points = min_points

    def forecast(self, date_col, value_col, periods_ahead=5):
        """
        Fits a simple linear trend to (date_col, value_col) and projects
        `periods_ahead` future points. Returns None if there isn't enough
        clean data or the trend has essentially no explanatory power.

        This is intentionally a transparent linear model, not a complex
        time-series model — the goal is a defensible, simple "if this
        trend continues" projection with an honest confidence band, not
        a black-box forecast.
        """
        sub = self.df[[date_col, value_col]].dropna().sort_values(date_col)
        if len(sub) < self.min_points:
            return None

        sub = sub.copy()
        sub["_t"] = pd.to_datetime(sub[date_col], errors="coerce")
        sub = sub.dropna(subset=["_t"])
        if len(sub) < self.min_points:
            return None

        # Aggregate to one value per time unit if there are duplicates
        sub = sub.groupby("_t")[value_col].mean().reset_index()
        if len(sub) < self.min_points:
            return None

        x = np.arange(len(sub))
        y = sub[value_col].values

        if np.std(y) == 0:
            return None

        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        if r_squared < 0.1:
            # Trend has essentially no explanatory power — forecasting
            # would be misleading, so don't produce one.
            return None

        residual_std = float(np.std(residuals))

        # Infer typical time step to project forward realistically
        time_diffs = sub["_t"].diff().dropna()
        typical_step = time_diffs.median() if not time_diffs.empty else pd.Timedelta(days=1)

        future_x = np.arange(len(sub), len(sub) + periods_ahead)
        future_y = slope * future_x + intercept
        future_dates = [sub["_t"].iloc[-1] + typical_step * (i + 1) for i in range(periods_ahead)]

        forecast_df = pd.DataFrame({
            date_col: future_dates,
            value_col: future_y,
            "lower_bound": future_y - 1.96 * residual_std,
            "upper_bound": future_y + 1.96 * residual_std,
            "type": "forecast",
        })

        historical_df = sub.rename(columns={"_t": date_col}).copy()
        historical_df["type"] = "historical"
        historical_df["lower_bound"] = historical_df[value_col]
        historical_df["upper_bound"] = historical_df[value_col]

        combined = pd.concat([historical_df, forecast_df], ignore_index=True)

        pct_projected_change = (
            ((future_y[-1] - y[-1]) / abs(y[-1])) * 100 if y[-1] != 0 else None
        )

        return {
            "date_column": date_col,
            "value_column": value_col,
            "r_squared": round(float(r_squared), 3),
            "slope_per_period": round(float(slope), 4),
            "periods_ahead": periods_ahead,
            "projected_end_value": round(float(future_y[-1]), 2),
            "pct_projected_change": round(pct_projected_change, 1) if pct_projected_change is not None else None,
            "combined_data": combined,  # ready to plot: historical + forecast with bounds
        }