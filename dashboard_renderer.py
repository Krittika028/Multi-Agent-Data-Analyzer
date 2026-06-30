"""
dashboard_renderer.py

Allowed chart types:
  - pie / donut      (composition, ≤ 8 categories)
  - bar              (horizontal category comparison)
  - column           (vertical category comparison)
  - stacked_bar      (part-to-whole across categories)
  - line             (trend analysis over time ONLY)

No scatter, histogram, area, funnel, box, or cluster charts.
Layout: one chart per row for wide charts; 2-column grid only for
small categorical charts so nothing looks clumsy.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

# ── Palette ────────────────────────────────────────────────────────────────────
DARK   = "#0a0a0f"
CARD   = "#1a1a2e"
BORDER = "#1e293b"

# Distinct, accessible colour sequence
ACCENT = [
    "#00f5ff", "#7c3aed", "#f59e0b", "#10b981",
    "#ef4444", "#3b82f6", "#ec4899", "#f97316",
]

# ── Business context lookup ────────────────────────────────────────────────────
BUSINESS_CONTEXT = [
    (["revenue", "sales", "income", "turnover", "gmv"],          "💰", "Revenue",     "$"),
    (["profit", "margin", "earnings", "ebit", "ebitda", "net"],  "📈", "Profit",       "$"),
    (["cost", "expense", "spend", "cogs", "opex"],               "💸", "Cost",         "$"),
    (["price", "rate", "fee", "tariff", "charge", "amount"],     "🏷️", "Price",        "$"),
    (["quantity", "qty", "units", "volume", "sold", "orders"],   "📦", "Volume",       ""),
    (["customer", "client", "user", "member", "subscriber"],     "👥", "Customers",    ""),
    (["transaction", "txn", "purchase", "order", "invoice"],     "🧾", "Transactions", ""),
    (["discount", "promo", "rebate", "coupon", "offer"],         "🎯", "Discount",     "$"),
    (["rating", "score", "nps", "csat", "review", "stars"],      "⭐", "Score",        ""),
    (["age", "tenure", "duration", "days", "months", "years"],   "📅", "Duration",     ""),
    (["weight", "kg", "lb", "gram", "ton"],                      "⚖️", "Weight",       ""),
    (["count", "num", "total", "sum", "freq", "visits", "hits"], "🔢", "Count",        ""),
    (["growth", "change", "delta", "variance", "diff"],          "📊", "Growth",       ""),
    (["tax", "vat", "gst", "levy", "duty"],                      "🧾", "Tax",          "$"),
    (["salary", "wage", "pay", "compensation", "bonus"],         "💼", "Salary",       "$"),
    (["inventory", "stock", "units_on_hand", "available"],       "🏭", "Inventory",    ""),
    (["clicks", "impressions", "ctr", "conversion", "bounce"],   "🖱️", "Engagement",   ""),
]


def _get_business_context(col_name: str):
    col_lower = col_name.lower()
    for keywords, emoji, label, prefix in BUSINESS_CONTEXT:
        if any(kw in col_lower for kw in keywords):
            return emoji, label, prefix
    clean = col_name.replace("_", " ").replace("-", " ").title()
    return "📊", clean, ""


def _format_value(val: float, prefix: str) -> str:
    if val is None:
        return "N/A"
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"{prefix}{val / 1_000_000_000:,.2f}B"
    elif abs_val >= 1_000_000:
        return f"{prefix}{val / 1_000_000:,.2f}M"
    elif abs_val >= 1_000:
        return f"{prefix}{val:,.1f}"
    else:
        return f"{prefix}{val:,.2f}"


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _is_numeric(df: pd.DataFrame, col: str) -> bool:
    if pd.api.types.is_numeric_dtype(df[col]):
        return True
    return _to_numeric_series(df, col).notna().mean() >= 0.8


def _ensure_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df = df.copy()
        df[col] = _to_numeric_series(df, col)
    return df


# ── Base style applied to every figure ────────────────────────────────────────

def _style(fig, height: int = 400):
    fig.update_layout(
        paper_bgcolor=DARK,
        plot_bgcolor=CARD,
        font_color="#e2e8f0",
        font_family="Inter, sans-serif",
        font_size=13,
        title_font_color="#00f5ff",
        title_font_size=15,
        title_font_family="Inter, sans-serif",
        margin=dict(t=60, b=50, l=50, r=30),
        height=height,
        legend=dict(
            bgcolor="#1a1a2e",
            bordercolor="#334155",
            borderwidth=1,
            font_color="#cbd5e1",
            font_size=12,
            itemsizing="constant",
        ),
    )
    fig.update_xaxes(
        gridcolor="#1e293b",
        zerolinecolor="#334155",
        tickfont=dict(size=12, color="#94a3b8"),
        title_font=dict(size=12, color="#94a3b8"),
    )
    fig.update_yaxes(
        gridcolor="#1e293b",
        zerolinecolor="#334155",
        tickfont=dict(size=12, color="#94a3b8"),
        title_font=dict(size=12, color="#94a3b8"),
    )
    return fig


# ── Section divider helper ─────────────────────────────────────────────────────

def _section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
                    border-bottom: 1px solid #1e293b;">
            <span style="font-size: 1.1rem; font-weight: 700; color: #00f5ff;">
                {title}
            </span>
            {"<span style='color:#64748b;font-size:0.85rem;margin-left:10px;'>" + subtitle + "</span>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHART RENDERERS  (pie/donut · bar · column · stacked_bar · line)
# ══════════════════════════════════════════════════════════════════════════════

def _pie_chart(df: pd.DataFrame, spec: dict, donut: bool = True):
    """Pie or donut — composition view, max 8 slices."""
    x, y = spec.get("x_column"), spec.get("y_column")
    if not x or x not in df.columns:
        return None

    if y and y in df.columns and _is_numeric(df, y):
        df = _ensure_numeric(df, y)
        data = (
            df.groupby(x)[y]
            .sum()
            .reset_index()
            .sort_values(y, ascending=False)
            .head(8)
        )
        values, names = data[y], data[x]
    else:
        counts = df[x].value_counts().head(8)
        values, names = counts.values, counts.index

    fig = px.pie(
        values=values,
        names=names,
        title=spec.get("title", ""),
        hole=0.48 if donut else 0,
        template="plotly_dark",
        color_discrete_sequence=ACCENT,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        textfont_size=12,
        marker=dict(line=dict(color=DARK, width=2)),
        pull=[0.03] * len(names),
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            bgcolor="#1a1a2e",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    return _style(fig, height=420)


def _bar_chart(df: pd.DataFrame, spec: dict):
    """Horizontal bar — category comparison, sorted descending."""
    x, y = spec.get("x_column"), spec.get("y_column")
    title = spec.get("title", "")
    if not x or x not in df.columns:
        return None

    if y and y in df.columns and _is_numeric(df, y):
        df = _ensure_numeric(df, y)
        grouped = (
            df.groupby(x)[y]
            .mean()
            .reset_index()
            .sort_values(y, ascending=True)   # ascending for horizontal bars
            .tail(12)
        )
        fig = px.bar(
            grouped, x=y, y=x,
            orientation="h",
            title=title,
            template="plotly_dark",
            color=y,
            color_continuous_scale=["#1e293b", "#00f5ff"],
            text=y,
        )
        fig.update_traces(
            texttemplate="%{text:,.1f}",
            textposition="outside",
            marker_line_width=0,
        )
    else:
        counts = df[x].value_counts().head(12).sort_values(ascending=True)
        fig = px.bar(
            x=counts.values, y=counts.index,
            orientation="h",
            title=title,
            template="plotly_dark",
            color=counts.values,
            color_continuous_scale=["#1e293b", "#7c3aed"],
            labels={"x": "Count", "y": x},
            text=counts.values,
        )
        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            marker_line_width=0,
        )

    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    # Give labels room to breathe
    fig.update_xaxes(range=[0, fig.data[0].x.max() * 1.18] if fig.data else None)
    return _style(fig, height=max(380, min(12, len(df[x].unique())) * 36 + 120))


def _column_chart(df: pd.DataFrame, spec: dict):
    """Vertical bar — good for time-bucketed or ranked categories."""
    x, y = spec.get("x_column"), spec.get("y_column")
    title = spec.get("title", "")
    if not x or x not in df.columns:
        return None

    if y and y in df.columns and _is_numeric(df, y):
        df = _ensure_numeric(df, y)
        grouped = (
            df.groupby(x)[y]
            .mean()
            .reset_index()
            .sort_values(y, ascending=False)
            .head(15)
        )
        fig = px.bar(
            grouped, x=x, y=y,
            title=title,
            template="plotly_dark",
            color=y,
            color_continuous_scale=["#1e293b", "#7c3aed"],
            text=y,
        )
        fig.update_traces(
            texttemplate="%{text:,.1f}",
            textposition="outside",
            marker_line_width=0,
        )
    else:
        counts = df[x].value_counts().head(15)
        fig = px.bar(
            x=counts.index, y=counts.values,
            title=title,
            template="plotly_dark",
            color=counts.values,
            color_continuous_scale=["#1e293b", "#00f5ff"],
            labels={"x": x, "y": "Count"},
            text=counts.values,
        )
        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            marker_line_width=0,
        )

    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    fig.update_yaxes(range=[0, None])
    return _style(fig, height=400)


def _stacked_bar_chart(df: pd.DataFrame, spec: dict):
    """Stacked bar — part-to-whole composition across categories."""
    x, y, color = spec.get("x_column"), spec.get("y_column"), spec.get("color_column")
    if not x or x not in df.columns:
        return None

    # Fall back to simple column if no colour split available
    if not color or color not in df.columns:
        return _column_chart(df, spec)

    cols_needed = [c for c in [x, y, color] if c and c in df.columns]
    sub = df[cols_needed].dropna()

    if y and y in df.columns and _is_numeric(df, y):
        sub = _ensure_numeric(sub, y)
        grouped = sub.groupby([x, color])[y].sum().reset_index()
        fig = px.bar(
            grouped, x=x, y=y, color=color,
            title=spec.get("title", ""),
            template="plotly_dark",
            barmode="stack",
            color_discrete_sequence=ACCENT,
            text=y,
        )
        fig.update_traces(texttemplate="%{text:,.1f}", textposition="inside",
                          marker_line_width=0, insidetextanchor="middle")
    else:
        grouped = sub.groupby([x, color]).size().reset_index(name="Count")
        fig = px.bar(
            grouped, x=x, y="Count", color=color,
            title=spec.get("title", ""),
            template="plotly_dark",
            barmode="stack",
            color_discrete_sequence=ACCENT,
            text="Count",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="inside",
                          marker_line_width=0, insidetextanchor="middle")

    fig.update_xaxes(tickangle=-30)
    return _style(fig, height=420)


def _line_chart(df: pd.DataFrame, spec: dict):
    """
    Line chart — ONLY for time-series / trend data.
    Groups by the x column (date/period) and aggregates y by mean.
    """
    x, y, color = spec.get("x_column"), spec.get("y_column"), spec.get("color_column")
    if not x or not y or x not in df.columns or y not in df.columns:
        return None
    if not _is_numeric(df, y):
        return None

    df = _ensure_numeric(df, y)
    cols = [c for c in [x, y, color] if c and c in df.columns]
    sub = df[cols].dropna()

    # Aggregate if duplicates exist on x (common for date columns)
    if color and color in df.columns:
        agg = sub.groupby([x, color])[y].mean().reset_index().sort_values(x)
    else:
        agg = sub.groupby(x)[y].mean().reset_index().sort_values(x)

    fig = px.line(
        agg, x=x, y=y,
        color=color if color and color in df.columns else None,
        title=spec.get("title", ""),
        template="plotly_dark",
        color_discrete_sequence=ACCENT,
        markers=True,
    )
    fig.update_traces(
        line_width=2.5,
        marker=dict(size=6, symbol="circle"),
    )
    fig.update_xaxes(tickangle=-30)
    return _style(fig, height=420)


# ── Chart registry ─────────────────────────────────────────────────────────────
CHART_RENDERERS = {
    "pie":         _pie_chart,
    "donut":       lambda df, spec: _pie_chart(df, spec, donut=True),
    "bar":         _bar_chart,
    "column":      _column_chart,
    "stacked_bar": _stacked_bar_chart,
    "line":        _line_chart,
    # Legacy aliases kept so old chart_specs from session state still work
    "area":        _line_chart,   # redirect to line
}


def render_chart(df: pd.DataFrame, spec: dict, verified_stats: dict = None):
    chart_type = spec.get("chart_type", "column")
    renderer = CHART_RENDERERS.get(chart_type)
    if renderer is None:
        # Unknown type — try column as a safe fallback
        renderer = CHART_RENDERERS["column"]
    try:
        return renderer(df, spec)
    except Exception as e:
        st.warning(f"Could not render **{spec.get('title', 'chart')}**: `{e}`")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY CARDS
# ══════════════════════════════════════════════════════════════════════════════

def render_summary_cards(df: pd.DataFrame, verified_stats: dict):
    kpis        = verified_stats.get("kpis", [])
    primary_kpi = kpis[0] if kpis else None

    total_cells  = df.shape[0] * df.shape[1]
    missing_pct  = round(df.isnull().sum().sum() / total_cells * 100, 2) if total_cells else 0
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    cols = st.columns(6)
    cols[0].metric("ROWS",            f"{df.shape[0]:,}")
    cols[1].metric("COLUMNS",         f"{df.shape[1]}")
    cols[2].metric("NUMERIC COLS",    f"{len(numeric_cols)}")
    cols[3].metric("MISSING %",       f"{missing_pct:.2f}%")

    if primary_kpi:
        emoji, label, prefix = _get_business_context(primary_kpi["column"])
        cols[4].metric(f"AVG {label.upper()}", _format_value(primary_kpi["mean"], prefix))
        total_sum = primary_kpi["mean"] * df.shape[0]
        cols[5].metric(f"SUM {label.upper()}", _format_value(total_sum, prefix))
    else:
        cols[4].metric("AVG", "N/A")
        cols[5].metric("SUM", "N/A")


def render_business_context_summary(business_summary: str):
    if not business_summary or not business_summary.strip():
        return
    st.markdown(
        f"""
        <div style="background:#1a1a2e;border:1px solid #334155;
                    border-left:4px solid #00f5ff;border-radius:10px;
                    padding:20px 24px;margin:1rem 0 1.5rem 0;
                    line-height:1.75;color:#cbd5e1;font-size:0.95rem;">
            {business_summary.strip().replace(chr(10), "<br>")}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════

def render_kpi_cards(verified_stats: dict):
    kpis = verified_stats.get("kpis", [])
    if not kpis:
        return

    _section_header("💼 Key Business KPIs")

    # Max 8 KPI cards; 4 per row
    for row_start in range(0, min(len(kpis), 8), 4):
        row_kpis = kpis[row_start : row_start + 4]
        cols = st.columns(len(row_kpis))

        for col, kpi in zip(cols, row_kpis):
            raw_col  = kpi.get("column", "")
            mean_val = kpi.get("mean",   0) or 0
            med_val  = kpi.get("median", 0) or 0
            min_val  = kpi.get("min",    0)
            max_val  = kpi.get("max",    0)

            emoji, biz_label, prefix = _get_business_context(raw_col)
            display_value = _format_value(mean_val, prefix)
            sub_line      = f"Median {_format_value(med_val, prefix)}"

            with col:
                st.markdown(
                    f"""
                    <div style="background:#1a1a2e;border:1px solid #334155;
                                border-radius:12px;padding:18px 16px;margin-bottom:10px;">
                        <div style="color:#64748b;font-size:0.7rem;
                                    text-transform:uppercase;letter-spacing:1px;">
                            {emoji} {biz_label}
                        </div>
                        <div style="color:#ffffff;font-size:1.6rem;
                                    font-weight:700;margin:6px 0 2px 0;">
                            {display_value}
                        </div>
                        <div style="color:#64748b;font-size:0.75rem;">
                            {sub_line}
                        </div>
                        <div style="color:#475569;font-size:0.72rem;margin-top:6px;">
                            Min {_format_value(min_val, prefix)} &nbsp;·&nbsp;
                            Max {_format_value(max_val, prefix)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TREND ANALYSIS  (line charts — one per detected trend, full width)
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_trend_smart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> tuple:
    """
    Decide the right aggregation granularity for a trend chart.

    Rules (matching the Caterpillar-style reference image):
      • Parse the date column to datetime.
      • Count distinct calendar years.
      • If distinct years > 2  → aggregate to ANNUAL totals (one dot per year).
        Each year's label is the 4-digit year string.
      • Otherwise             → aggregate to MONTHLY totals (one dot per month).
        Each month's label is "YYYY-MM".

    Returns:
        (labels, totals, granularity)
        where granularity is "annual" or "monthly".
    """
    sub = df[[date_col, value_col]].copy().dropna()
    sub[value_col] = pd.to_numeric(
        sub[value_col].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce",
    )
    sub = sub.dropna(subset=[value_col])

    parsed = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub[parsed.notna()].copy()
    sub["_parsed"] = parsed[parsed.notna()].values

    if sub.empty:
        return [], [], "monthly"

    years = sub["_parsed"].dt.year.nunique()

    if years > 2:
        # Annual aggregation — one data point per year
        sub["_key"] = sub["_parsed"].dt.year
        agg = (
            sub.groupby("_key")[value_col]
            .sum()
            .reset_index()
            .sort_values("_key")
        )
        labels = agg["_key"].astype(str).tolist()
        totals = agg[value_col].tolist()
        return labels, totals, "annual"
    else:
        # Monthly aggregation
        sub["_key"] = sub["_parsed"].dt.to_period("M")
        agg = (
            sub.groupby("_key")[value_col]
            .sum()
            .reset_index()
            .sort_values("_key")
        )
        labels = agg["_key"].astype(str).tolist()
        totals = agg[value_col].tolist()
        return labels, totals, "monthly"


def _build_caterpillar_trend_fig(
    labels: list,
    totals: list,
    biz_label: str,
    title: str,
    prefix: str = "",
    granularity: str = "annual",
) -> go.Figure:
    """
    Build a clean Caterpillar-style trend chart:
      - Thin blue line connecting the dots.
      - Filled circle markers at each data point.
      - Value labels displayed above every marker (formatted, e.g. $22,763).
      - OLS trend line (dashed amber) overlaid.
      - No clutter: gridlines are minimal, no bar chart overlay.
    This matches the reference image aesthetic exactly.
    """
    y_arr = np.array(totals, dtype=float)
    x_idx = np.arange(len(y_arr), dtype=float)

    # Format value labels
    def _fmt(v):
        abs_v = abs(v)
        if abs_v >= 1_000_000_000:
            return f"{prefix}{v/1_000_000_000:,.2f}B"
        if abs_v >= 1_000_000:
            return f"{prefix}{v/1_000_000:,.2f}M"
        if abs_v >= 1_000:
            return f"{prefix}{v:,.0f}"
        return f"{prefix}{v:,.2f}"

    text_labels = [_fmt(v) for v in totals]

    fig = go.Figure()

    # Main line + markers (Caterpillar style)
    fig.add_trace(go.Scatter(
        x=labels,
        y=totals,
        mode="lines+markers+text",
        name=biz_label,
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=9, color="#3b82f6", symbol="circle",
                    line=dict(color="white", width=1.5)),
        text=text_labels,
        textposition="top center",
        textfont=dict(size=11, color="#e2e8f0"),
    ))

    # OLS trend line overlay
    if len(y_arr) >= 3:
        try:
            import statsmodels.api as sm
            X = sm.add_constant(x_idx)
            ols_y = sm.OLS(y_arr, X).fit().predict(X)
        except Exception:
            slope, intercept = np.polyfit(x_idx, y_arr, 1)
            ols_y = slope * x_idx + intercept

        fig.add_trace(go.Scatter(
            x=labels,
            y=ols_y.tolist(),
            mode="lines",
            name="Trend",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            opacity=0.85,
        ))

    x_axis_title = "Year" if granularity == "annual" else "Month"
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=biz_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    # Extra top margin so value labels above first/last points don't clip
    fig.update_layout(margin=dict(t=80, b=60, l=60, r=30))
    fig.update_xaxes(tickangle=0 if granularity == "annual" else -40)
    # Y-axis: give 15% breathing room above max so labels don't clip
    y_max = max(totals) if totals else 1
    fig.update_yaxes(range=[0, y_max * 1.20])

    return _style(fig, height=440)


def render_trend_charts(verified_stats: dict, df: pd.DataFrame):
    """
    Trend Analysis.

    Aggregation strategy (Caterpillar-style reference image):
      • If the date column spans > 2 distinct years  → one dot per YEAR (annual total).
      • Otherwise                                    → one dot per MONTH (monthly total).

    This avoids cluttered dot-per-row charts and produces a clean, readable
    trend line with value labels at every data point — exactly like the
    reference chart.

    High-cardinality date columns (e.g. raw timestamps with hundreds of
    unique values) are also handled: they get bucketed to the right period
    automatically via _aggregate_trend_smart().
    """
    monthly_sales = verified_stats.get("monthly_sales", [])
    rendered_period = _render_period_trends(df, verified_stats, monthly_sales)

    if rendered_period:
        return  # period-based charts already rendered; skip fallback

    # ── Fallback: raw datetime column trends ──────────────────────────────────
    trends = verified_stats.get("trends", [])
    valid_trends = [
        t for t in trends
        if t.get("date_column") in df.columns and t.get("value_column") in df.columns
    ]
    if not valid_trends:
        return

    _section_header("📈 Trend Analysis", "Annual or monthly performance — value at every data point")

    for trend in valid_trends[:4]:
        date_col  = trend["date_column"]
        value_col = trend["value_column"]
        r2        = trend.get("r_squared", 0) or 0
        direction = trend.get("direction", "")
        pct_chg   = trend.get("pct_change_start_to_end")

        _, biz_label, prefix = _get_business_context(value_col)

        if direction == "increasing":
            badge_color, badge_icon = "#10b981", "↑"
        elif direction == "decreasing":
            badge_color, badge_icon = "#ef4444", "↓"
        else:
            badge_color, badge_icon = "#94a3b8", "→"

        pct_text = (
            f" &nbsp;<span style='color:{badge_color};font-size:0.8rem;'>"
            f"{badge_icon} {pct_chg:+.1f}%</span>"
            if pct_chg is not None else ""
        )
        st.markdown(
            f"""
            <div style="margin:0.5rem 0 0.3rem 0;font-size:0.85rem;color:#64748b;">
                {biz_label} &nbsp;·&nbsp; R² = {r2:.2f}{pct_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        labels, totals, granularity = _aggregate_trend_smart(df, date_col, value_col)
        if not totals:
            continue

        gran_label = "Annual Total" if granularity == "annual" else "Monthly Total"
        fig = _build_caterpillar_trend_fig(
            labels, totals, biz_label,
            title=f"{biz_label} — {gran_label}",
            prefix=prefix,
            granularity=granularity,
        )
        st.plotly_chart(fig, use_container_width=True)

        if r2 >= 0.25:
            _render_forecast_inline(df, date_col, value_col, r2, biz_label, prefix)


def _render_period_trends(df: pd.DataFrame, verified_stats: dict, monthly_sales: list) -> bool:
    """
    Renders period SUM trend charts using pre-computed monthly_sales rows.

    Aggregation granularity (Caterpillar-style):
      • Count distinct years in the monthly_sales rows.
      • If distinct years > 2  → collapse to ANNUAL totals (one dot per year).
      • Otherwise              → keep MONTHLY totals (one dot per year-month).

    Each data point is rendered as a filled circle with a value label above it,
    connected by a thin line — matching the reference Caterpillar chart style.
    A dashed OLS trend line is overlaid in amber.

    Returns True if at least one chart was rendered.
    """
    if not monthly_sales:
        return False

    # Group monthly_sales rows by (date_column, value_column)
    pairs: dict = {}
    for row in monthly_sales:
        key = (row["date_column"], row["value_column"])
        pairs.setdefault(key, []).append(row)

    if not pairs:
        return False

    _section_header(
        "📈 Trend Analysis",
        "Annual totals (>2 yrs) or monthly totals · value labels at every point",
    )

    rendered = False
    for (date_col, value_col), rows in list(pairs.items())[:4]:
        _, biz_label, prefix = _get_business_context(value_col)

        period_df = pd.DataFrame(rows).sort_values(["year", "month"])
        if len(period_df) < 3:
            continue

        # ── Decide: annual or monthly ─────────────────────────────────────────
        distinct_years = period_df["year"].nunique()
        if distinct_years > 2:
            # Collapse to annual totals
            agg = (
                period_df.groupby("year")["total"]
                .sum()
                .reset_index()
                .sort_values("year")
            )
            labels    = agg["year"].astype(str).tolist()
            totals    = agg["total"].tolist()
            granularity = "annual"
        else:
            labels    = period_df["label"].tolist()
            totals    = period_df["total"].tolist()
            granularity = "monthly"

        y_arr = np.array(totals, dtype=float)

        # ── OLS ───────────────────────────────────────────────────────────────
        ols_trend = None
        r2_val    = None
        try:
            import statsmodels.api as sm
            x_idx  = np.arange(len(y_arr), dtype=float)
            X      = sm.add_constant(x_idx)
            model  = sm.OLS(y_arr, X).fit()
            ols_trend = model.predict(X)
            r2_val    = round(float(model.rsquared), 3)
        except Exception:
            pass

        # ── Direction badge ───────────────────────────────────────────────────
        slope = (ols_trend[-1] - ols_trend[0]) if ols_trend is not None and len(ols_trend) >= 2 else 0
        badge_color = "#10b981" if slope > 0 else "#ef4444" if slope < 0 else "#94a3b8"
        badge_icon  = "↑" if slope > 0 else "↓" if slope < 0 else "→"
        pct_chg     = round(((y_arr[-1] - y_arr[0]) / abs(y_arr[0])) * 100, 1) if y_arr[0] != 0 else None

        pct_text = (
            f" &nbsp;<span style='color:{badge_color};font-size:0.8rem;'>"
            f"{badge_icon} {pct_chg:+.1f}%</span>"
            if pct_chg is not None else ""
        )
        gran_label = "annual totals" if granularity == "annual" else "monthly totals"
        r2_text    = f" &nbsp;·&nbsp; OLS R² = {r2_val:.3f}" if r2_val is not None else ""
        st.markdown(
            f"""
            <div style="margin:0.5rem 0 0.25rem 0;font-size:0.85rem;color:#64748b;">
                {biz_label} {gran_label}{r2_text}{pct_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Caterpillar-style chart ───────────────────────────────────────────
        gran_title = "Annual Total" if granularity == "annual" else "Monthly Total"
        fig = _build_caterpillar_trend_fig(
            labels, totals, biz_label,
            title=f"{biz_label} — {gran_title} (SUM)",
            prefix=prefix,
            granularity=granularity,
        )
        st.plotly_chart(fig, use_container_width=True)

        # YoY/MoM summary caption
        for pt in verified_stats.get("period_trends", []):
            if pt.get("metric_column") == value_col and pt.get("summary"):
                st.caption(f"📊 {pt['summary']}")
                break

        rendered = True

    return rendered


def _add_ols_trend_line(fig: go.Figure, x_labels, y_vals: np.ndarray):
    """Add a dashed OLS trend line (statsmodels) to an existing figure."""
    try:
        import statsmodels.api as sm
        x_idx = np.arange(len(y_vals), dtype=float)
        X = sm.add_constant(x_idx)
        model = sm.OLS(y_vals.astype(float), X).fit()
        trend_y = model.predict(X)
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=trend_y,
            mode="lines",
            name="OLS Trend",
            line=dict(color="#f59e0b", width=1.8, dash="dash"),
            opacity=0.8,
        ))
    except Exception:
        # numpy polyfit fallback if statsmodels unavailable
        try:
            x_idx = np.arange(len(y_vals), dtype=float)
            slope, intercept = np.polyfit(x_idx, y_vals.astype(float), 1)
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=(slope * x_idx + intercept),
                mode="lines",
                name="Trend",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
                opacity=0.7,
            ))
        except Exception:
            pass


def _render_forecast_inline(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    r2: float,
    biz_label: str,
    prefix: str,
):
    """Render a 5-period forecast beneath the historical trend chart."""
    try:
        from ml_insights import TrendForecaster
        result = TrendForecaster(df).forecast(date_col, value_col, periods_ahead=5)
        if not result:
            return
    except ImportError:
        return

    combined  = result["combined_data"]
    pct_proj  = result.get("pct_projected_change")
    end_val   = result.get("projected_end_value")

    historical = combined[combined["type"] == "historical"]
    forecast   = combined[combined["type"] == "forecast"]

    st.caption(
        f"📡 **5-Period Forecast** — projected end value: "
        f"**{_format_value(end_val, prefix)}**"
        + (f" ({pct_proj:+.1f}% vs last observed)" if pct_proj is not None else "")
        + f" · Confidence band = 95%  ·  R² = {r2:.2f}"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical[date_col], y=historical[value_col],
        mode="lines+markers", name="Historical",
        line=dict(color="#00f5ff", width=2.5),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=forecast[date_col], y=forecast["upper_bound"],
        mode="lines", name="Upper bound",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=forecast[date_col], y=forecast["lower_bound"],
        mode="lines", name="Confidence Band",
        fill="tonexty", fillcolor="rgba(124, 58, 237, 0.15)",
        line=dict(width=0),
    ))
    fig.add_trace(go.Scatter(
        x=forecast[date_col], y=forecast[value_col],
        mode="lines+markers", name="Forecast",
        line=dict(color="#7c3aed", width=2.5, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
    ))
    fig.update_layout(
        title=f"{biz_label} — Historical + 5-Period Forecast",
        xaxis_title=date_col.replace("_", " ").title(),
        yaxis_title=biz_label,
    )
    st.plotly_chart(_style(fig, height=400), use_container_width=True)
    st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AI-SELECTED CHARTS  (pie · donut · bar · column · stacked_bar only)
# ══════════════════════════════════════════════════════════════════════════════

def _high_cardinality_cols(verified_stats: dict) -> set:
    """
    Returns the set of categorical columns that should NEVER appear in charts
    because they have too many distinct values (>12) or are near-identifiers
    (top value represents <5% of rows). These produce unreadable, clumsy
    charts and are filtered out everywhere in the dashboard.
    """
    excluded = set()
    for c in verified_stats.get("category_summary", []):
        unique  = c.get("unique_values") or 0
        top_pct = (c.get("top_values") or [{}])[0].get("pct", 0)
        if unique > 12 or top_pct < 5:
            excluded.add(c["column"])
    return excluded


def render_ai_charts(df: pd.DataFrame, chart_specs: list, verified_stats: dict):
    """
    Renders AI-selected charts (LLM-curated via DashboardAgent).
    - Line charts → skipped here (rendered in render_trend_charts instead).
    - Pie / donut → centred, not full-bleed (they look bad full width).
    - Bar / column / stacked_bar → two-column grid.

    High-cardinality columns (>12 unique values or top value <5% share)
    are silently dropped even if the LLM spec included them — they produce
    unreadable charts.
    """
    if not chart_specs:
        return

    # Filter to only allowed non-line types
    allowed   = {"pie", "donut", "bar", "column", "stacked_bar"}
    excluded  = _high_cardinality_cols(verified_stats)

    specs = [
        s for s in chart_specs
        if s.get("chart_type", "") in allowed
        # Drop specs whose x_column is high-cardinality
        and s.get("x_column") not in excluded
    ]
    if not specs:
        return

    _section_header("📊 Business Intelligence Charts", f"{len(specs)} AI-selected charts")

    pie_specs   = [s for s in specs if s.get("chart_type") in ("pie", "donut")]
    other_specs = [s for s in specs if s.get("chart_type") not in ("pie", "donut")]

    # Pie / donut charts: centred, not full-bleed (they look bad full width)
    for spec in pie_specs:
        left, centre, right = st.columns([1, 3, 1])
        with centre:
            fig = render_chart(df, spec, verified_stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                if spec.get("reasoning"):
                    st.caption(f"📌 {spec['reasoning']}")
        st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # Bar / column / stacked_bar: 2-column grid
    for i in range(0, len(other_specs), 2):
        pair = other_specs[i : i + 2]
        grid = st.columns(len(pair))
        for col, spec in zip(grid, pair):
            with col:
                fig = render_chart(df, spec, verified_stats)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    if spec.get("reasoning"):
                        st.caption(f"📌 {spec['reasoning']}")
        st.markdown("<div style='margin-bottom:0.25rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY BREAKDOWN  (auto-generated bar charts per categorical column)
# ══════════════════════════════════════════════════════════════════════════════

def render_category_breakdown(verified_stats: dict, df: pd.DataFrame):
    """
    For each categorical column with 2–12 unique values AND a top-value
    share ≥ 5%, renders a bar chart of the mean primary KPI per category.
    Columns with more distinct values are skipped — they produce clumsy,
    unreadable charts.  Shows max 3 breakdowns in a 3-column grid.
    """
    kpis        = verified_stats.get("kpis", [])
    cat_summary = verified_stats.get("category_summary", [])
    if not kpis or not cat_summary:
        return

    primary_kpi = kpis[0]["column"] if kpis else None
    if not primary_kpi or primary_kpi not in df.columns:
        return

    good_cats = [
        c["column"] for c in cat_summary
        if 2 <= (c.get("unique_values") or 0) <= 12
        and c["column"] in df.columns
        and c["column"] != primary_kpi
        # Exclude near-identifier columns: top value must represent ≥5% of rows
        and (c.get("top_values") or [{}])[0].get("pct", 0) >= 5
    ]
    if not good_cats:
        return

    _, biz_label, prefix = _get_business_context(primary_kpi)
    display_cats = good_cats[:3]

    _section_header("📂 Category Breakdown", f"Avg {biz_label} by segment")

    cols = st.columns(len(display_cats))
    for col, cat_col in zip(cols, display_cats):
        with col:
            df_num = _ensure_numeric(df, primary_kpi)
            grouped = (
                df_num.groupby(cat_col)[primary_kpi]
                .mean()
                .reset_index()
                .sort_values(primary_kpi, ascending=False)
                .head(10)
            )
            grouped.columns = [cat_col, "Mean"]

            fig = px.bar(
                grouped, x=cat_col, y="Mean",
                title=f"Avg {biz_label} by {cat_col.replace('_', ' ').title()}",
                template="plotly_dark",
                color="Mean",
                color_continuous_scale=["#1e293b", "#00f5ff"],
                text="Mean",
            )
            fig.update_traces(
                texttemplate="%{text:,.1f}",
                textposition="outside",
                marker_line_width=0,
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(_style(fig, height=360), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard(
    df: pd.DataFrame,
    chart_specs: list,
    verified_stats: dict = None,
    business_summary: str = "",
):
    verified_stats = verified_stats or {}

    # ── 1. Dataset overview metrics ───────────────────────────────────────────
    render_summary_cards(df, verified_stats)

    # ── 2. Business context paragraph ────────────────────────────────────────
    if business_summary:
        render_business_context_summary(business_summary)

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # ── 3. KPI cards ──────────────────────────────────────────────────────────
    render_kpi_cards(verified_stats)

    st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

    # ── 4. Trend analysis (line charts + optional forecast) ───────────────────
    render_trend_charts(verified_stats, df)

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # ── 5. AI-selected charts (pie · bar · column · stacked_bar) ──────────────
    render_ai_charts(df, chart_specs, verified_stats)

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # ── 6. Auto category breakdown ────────────────────────────────────────────
    render_category_breakdown(verified_stats, df)