"""
dashboard_renderer.py

Allowed chart types:
  - pie / donut      (composition, ≤ 8 categories)
  - bar              (horizontal category comparison)
  - column           (vertical category comparison)
  - stacked_bar      (part-to-whole across categories)
  - line             (trend analysis over time ONLY)
  - scatter          (relationship/correlation between two numeric variables)
  - histogram        (distribution of a single numeric variable)

No funnel, area, or box charts.
Layout: one chart per row for wide charts; 2-column grid only for
small categorical charts so nothing looks clumsy.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from domain_context import get_domain_config, DEFAULT_LABELS
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
# BUSINESS_CONTEXT stays as-is (unchanged) — it's the commerce-domain
# vocabulary and still applies whenever those keywords genuinely appear
# (many domains share words like "cost", "count", "score"). What changes
# is _get_business_context now also checks a domain-provided extra list
# FIRST, so a domain-specific label (e.g. "Diagnosis" for healthcare) wins
# over a generic fallback when the column doesn't match commerce keywords.
BUSINESS_CONTEXT = [
    (["revenue", "sales", "income", "turnover", "gmv"],          "💰", "Revenue",     "$"),
    (["profit", "margin", "earnings", "ebit", "ebitda", "net"],  "📈", "Profit",       "$"),
    (["cost", "expense", "spend", "cogs", "opex"],               "💸", "Cost",         "$"),
    (["price", "rate", "fee", "tariff", "charge", "amount", "spent", "spending"], "🏷️", "Price", "$"),
    (["quantity", "qty", "units", "volume", "sold", "orders"],   "📦", "Volume",       ""),
    (["customer", "client", "user", "member", "subscriber"],     "👥", "Customers",    ""),
    (["transaction", "txn", "purchase", "order", "invoice"],     "🧾", "Transactions", ""),
    (["discount", "promo", "rebate", "coupon", "offer"],         "🎯", "Discount",     "$"),
    (["rating", "score", "nps", "csat", "review", "stars"],      "⭐", "Score",        ""),
    (["age", "tenure", "duration", "days", "months", "years"],   "📅", "Duration",     ""),
    (["weight", "kg", "lb", "gram", "ton"],                      "⚖️", "Weight",       ""),
    (["growth", "change", "delta", "variance", "diff"],          "📊", "Growth",       ""),
    (["tax", "vat", "gst", "levy", "duty"],                      "🧾", "Tax",          "$"),
    (["salary", "wage", "pay", "compensation", "bonus"],         "💼", "Salary",       "$"),
    (["inventory", "stock", "units_on_hand", "available"],       "🏭", "Inventory",    ""),
    (["clicks", "impressions", "ctr", "conversion", "bounce"],   "🖱️", "Engagement",   ""),
    (["count", "num", "freq", "visits", "hits"],                 "🔢", "Count",        ""),
    (["total", "sum"],                                            "🔢", "Total",        ""),
]


def _get_business_context(col_name: str, domain_extra: list = None):
    """
    domain_extra: optional list of (keywords, emoji, label, prefix) tuples
    supplied by the calling domain — checked with the SAME specificity
    scoring as BUSINESS_CONTEXT, and merged into the same competition
    rather than short-circuiting it, so a truly better generic commerce
    match (e.g. an HR dataset that also has a genuine "salary" column)
    isn't incorrectly overridden by a weaker domain-specific guess.
    """
    col_lower = col_name.lower()
    all_groups = list(BUSINESS_CONTEXT) + list(domain_extra or [])

    best = None
    for idx, (keywords, emoji, label, prefix) in enumerate(all_groups):
        matched_kw = None
        for kw in keywords:
            if kw in col_lower:
                if matched_kw is None or len(kw) > len(matched_kw):
                    matched_kw = kw
        if matched_kw is None:
            continue

        specificity = len(matched_kw)
        prefix_bonus = 1 if prefix else 0
        rank = (specificity, prefix_bonus, -idx)

        if best is None or rank > best[0]:
            best = (rank, (emoji, label, prefix))

    if best is not None:
        return best[1]

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
# CHART RENDERERS  (pie/donut · bar · column · stacked_bar · line · scatter · histogram)
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


def _scatter_chart(df: pd.DataFrame, spec: dict):
    """Scatter — relationship/correlation between two numeric variables."""
    x, y, color = spec.get("x_column"), spec.get("y_column"), spec.get("color_column")
    if not x or not y or x not in df.columns or y not in df.columns:
        return None
    if not (_is_numeric(df, x) and _is_numeric(df, y)):
        return None

    df = _ensure_numeric(df, x)
    df = _ensure_numeric(df, y)
    cols = [c for c in [x, y, color] if c and c in df.columns]
    sub = df[cols].dropna()
    if sub.empty:
        return None
    if len(sub) > 3000:
        sub = sub.sample(3000, random_state=42)

    fig = px.scatter(
        sub, x=x, y=y,
        color=color if color and color in df.columns else None,
        title=spec.get("title", ""),
        template="plotly_dark",
        color_discrete_sequence=ACCENT,
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color=DARK)))

    if len(sub) >= 3:
        try:
            slope, intercept = np.polyfit(sub[x], sub[y], 1)
            x_range = np.linspace(sub[x].min(), sub[x].max(), 50)
            fig.add_trace(go.Scatter(
                x=x_range, y=slope * x_range + intercept,
                mode="lines", name="Trend",
                line=dict(color="#f59e0b", width=2, dash="dash"),
            ))
        except Exception:
            pass

    return _style(fig, height=420)


def _histogram_chart(df: pd.DataFrame, spec: dict):
    """Histogram — distribution shape of a single numeric variable."""
    x = spec.get("x_column")
    if not x or x not in df.columns or not _is_numeric(df, x):
        return None

    df = _ensure_numeric(df, x)
    series = df[x].dropna()
    if series.empty:
        return None

    fig = px.histogram(
        df, x=x,
        title=spec.get("title", ""),
        template="plotly_dark",
        color_discrete_sequence=["#00f5ff"],
        nbins=30,
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color=DARK)

    mean_val = series.mean()
    fig.add_vline(
        x=mean_val, line_width=2, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Mean {mean_val:,.1f}", annotation_font_color="#f59e0b",
    )
    fig.update_layout(showlegend=False, bargap=0.05)
    return _style(fig, height=380)


# ── Chart registry ─────────────────────────────────────────────────────────────
CHART_RENDERERS = {
    "pie":         _pie_chart,
    "donut":       lambda df, spec: _pie_chart(df, spec, donut=True),
    "bar":         _bar_chart,
    "column":      _column_chart,
    "stacked_bar": _stacked_bar_chart,
    "line":        _line_chart,
    "scatter":     _scatter_chart,
    "histogram":   _histogram_chart,
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
# BUSINESS CONTEXT SUMMARY  (was called but never defined — that's the
# NameError in your Problems panel: "render_business_context_summary" is
# not defined. It's referenced once, at the top of render_dashboard,
# right after the KPI cards. Nothing else in this file defines it, and
# it wasn't defined in the file you originally gave me either — this
# isn't something my edit introduced.)
# ══════════════════════════════════════════════════════════════════════════════

def render_business_context_summary(business_summary: str):
    """
    Renders the short (3–5 sentence) narrative paragraph produced by
    get_business_summary_task() in tasks.py — plain prose, no markdown,
    no stats jargon. Shown as a single highlighted card directly under
    the KPI row, before any charts.
    """
    if not business_summary or not business_summary.strip():
        return
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                    border:1px solid #334155;border-left:4px solid #00f5ff;
                    border-radius:12px;padding:18px 24px;margin:0.5rem 0 1rem 0;">
            <div style="color:#00f5ff;font-size:0.7rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                📋 Business Snapshot
            </div>
            <div style="color:#cbd5e1;font-size:0.95rem;line-height:1.6;">
                {business_summary.strip()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS  (single, unified row — replaces the previous dual
# render_summary_cards() + render_kpi_cards() stack, which rendered two
# separate 4-card rows back to back and duplicated the same trend/KPI
# numbers between them. That was the actual bug behind the cluttered
# dashboard — now there is one function, one row, four cards, each one
# a distinct signal.)
# ══════════════════════════════════════════════════════════════════════════════

def render_kpi_cards(df: pd.DataFrame, verified_stats: dict, domain_config: dict = None):
    """
    Four cards — business metrics only. No risk, flag, outlier, or anomaly
    framing anywhere in this function; that content belongs in the written
    report, not the KPI strip.

      1. TOTAL REVENUE — a genuine revenue/sales/amount-type column, summed.
         If no such column exists in this dataset, this card is NOT faked;
         it falls back to the domain's actual primary metric with an
         honest label (e.g. "TOTAL SALARY" for HR data) instead of
         pretending revenue exists where it doesn't.
      2. ADAPTIVE SECOND METRIC — the best *distinct* business signal this
         dataset actually has: average order/transaction value if there's
         a revenue column, otherwise the next-ranked KPI's average. Not a
         repeat of card 1 in a different shape.
      3. VOLUME — total records, labeled with the domain's actual entity
         name (Orders / Patients / Employees / etc.), not a generic count.
      4. TOP PERFORMER — the leading business segment's actual contribution
         (e.g. top category's revenue share), or a growth-rate signal if no
         good segment column exists. Never a risk/at-risk framing.

    Card composition is driven by what's actually present in THIS dataset —
    a dataset with no category column and no second KPI will legitimately
    look different from one that has both, rather than forcing 4 fixed
    shapes onto every dataset regardless of what it contains.
    """
    kpis        = verified_stats.get("kpis", [])
    trends      = verified_stats.get("trends", [])
    cat_summary = verified_stats.get("category_summary", [])

    if not kpis:
        return

    trend_by_col = {t.get("value_column"): t for t in trends}
    entity_plural = (domain_config or {}).get("entity_plural", "Records")

    # ── Revenue-specific detection — deliberately narrower than the general
    # KPI priority list. "Volume"/"Transactions"-type columns rank high for
    # general KPI purposes but are NOT revenue, and showing them under a
    # "TOTAL REVENUE" label would be a fabricated business claim. ──────────
    _REVENUE_KEYWORDS = ["revenue", "sales", "gmv", "amount", "price", "earnings", "income", "turnover"]

    def _is_revenue_col(kpi):
        col_lower = kpi.get("column", "").lower()
        return any(k in col_lower for k in _REVENUE_KEYWORDS)

    revenue_kpi = next((k for k in kpis if _is_revenue_col(k)), None)

    _HIGH_PRIORITY = [
        "revenue", "sales", "profit", "cost", "price", "amount", "total",
        "quantity", "volume", "orders", "transactions",
    ]

    def _priority(kpi):
        col_lower = kpi.get("column", "").lower()
        return 0 if any(k in col_lower for k in _HIGH_PRIORITY) else 1

    ranked = sorted(kpis, key=_priority)
    primary_kpi = revenue_kpi or ranked[0]
    remaining_kpis = [k for k in ranked if k["column"] != primary_kpi["column"]]

    def _trend_html(raw_col):
        """Colored inline trend badge for a KPI, or None if no reliable trend exists."""
        trend = trend_by_col.get(raw_col)
        if not trend:
            return None
        pct = trend.get("pct_change_start_to_end")
        if pct is None or trend.get("insufficient_edge_data"):
            return None
        direction = (trend.get("direction") or "").lower()
        arrow = "▲" if "increas" in direction else ("▼" if "decreas" in direction else "→")
        color = "#10b981" if "increas" in direction else ("#ef4444" if "decreas" in direction else "#64748b")
        return f'<span style="color:{color};font-weight:700;">{arrow} {pct:+.1f}%</span>'

    def _card(container, big_value, label, trend_html=None, caption=None):
        with container:
            st.markdown(
                f"""
                <div style="background:#1a1a2e;border:1px solid #334155;
                            border-radius:12px;padding:20px 18px;margin-bottom:10px;
                            min-height:118px;">
                    <div style="color:#ffffff;font-size:1.75rem;
                                font-weight:700;line-height:1.15;">
                        {big_value}
                    </div>
                    <div style="color:#64748b;font-size:0.78rem;margin-top:8px;">
                        {label}{(" &nbsp; " + trend_html) if trend_html else ""}
                    </div>
                    {f'<div style="color:#475569;font-size:0.72rem;margin-top:3px;">{caption}</div>' if caption else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

    _section_header("💼 Key Business KPIs")
    cols = st.columns(4)

    # ── 1 — TOTAL REVENUE (guaranteed real, never fabricated) ────────────────
    emoji, label, prefix = _get_business_context(primary_kpi["column"])
    total = (primary_kpi.get("mean") or 0) * len(df)
    card1_label = f"💰 TOTAL REVENUE" if revenue_kpi else f"{emoji} TOTAL {label.upper()}"
    _card(
        cols[0],
        _format_value(total, prefix),
        card1_label,
        _trend_html(primary_kpi["column"]),
    )

    # ── 2 — Adaptive second metric: avg order/transaction value when we
    # have real revenue, otherwise the next best distinct KPI. Never just
    # re-shows card 1's number as an average. ─────────────────────────────
    if revenue_kpi:
        avg_val = revenue_kpi.get("mean") or 0
        med_val = revenue_kpi.get("median") or 0
        _card(
            cols[1],
            _format_value(avg_val, prefix),
            f"🧮 AVG ORDER VALUE",
            caption=f"Median {_format_value(med_val, prefix)}",
        )
    elif remaining_kpis:
        second_kpi = remaining_kpis[0]
        emoji2, label2, prefix2 = _get_business_context(second_kpi["column"])
        _card(
            cols[1],
            _format_value(second_kpi.get("mean") or 0, prefix2),
            f"{emoji2} AVG {label2.upper()}",
            _trend_html(second_kpi["column"]),
        )
    else:
        avg_val = primary_kpi.get("mean") or 0
        med_val = primary_kpi.get("median") or 0
        _card(cols[1], _format_value(avg_val, prefix), f"{emoji} AVG {label.upper()}",
              caption=f"Median {_format_value(med_val, prefix)}")

    # ── 3 — VOLUME, labeled with the actual entity for this dataset ──────────
    _card(cols[2], f"{len(df):,}", f"🧾 TOTAL {entity_plural.upper()}", caption="Records analyzed")

    # ── 4 — TOP PERFORMER: leading segment's real revenue/KPI contribution,
    # or a growth-rate signal as fallback. No risk/at-risk framing — that
    # kind of flag belongs in the written report, not the KPI strip. ────────
    excluded_cols = _high_cardinality_cols(verified_stats)
    good_segment_cols = [
        c for c in cat_summary
        if c.get("column") not in excluded_cols and (c.get("top_values") or [])
    ]

    def _try_top_segment():
        if good_segment_cols:
            seg_entry = good_segment_cols[0]
            seg_col   = seg_entry["column"]
            top_val_entry = seg_entry["top_values"][0]
            seg_name  = str(top_val_entry.get("value", "N/A"))
            if primary_kpi["column"] in df.columns and seg_col in df.columns:
                kpi_col = primary_kpi["column"]
                df_num  = _ensure_numeric(df, kpi_col)
                seg_total   = df_num.loc[df[seg_col].astype(str) == seg_name, kpi_col].sum()
                grand_total = df_num[kpi_col].sum()
                seg_pct_of_kpi = (seg_total / grand_total * 100) if grand_total else 0
                _card(
                    cols[3], seg_name[:18], f"🏆 TOP {seg_col.replace('_',' ').upper()}",
                    caption=f"{_format_value(seg_total, prefix)} · {seg_pct_of_kpi:.1f}% of {label.lower()}",
                )
            else:
                _card(cols[3], seg_name[:18], f"🏆 TOP {seg_col.replace('_',' ').upper()}")
            return True
        return False

    def _try_growth_rate():
        trend = trend_by_col.get(primary_kpi["column"])
        if trend and trend.get("pct_change_start_to_end") is not None and not trend.get("insufficient_edge_data"):
            pct = trend["pct_change_start_to_end"]
            direction = (trend.get("direction") or "").lower()
            arrow = "▲" if "increas" in direction else ("▼" if "decreas" in direction else "→")
            _card(
                cols[3], f"{arrow} {pct:+.1f}%", "📈 GROWTH RATE",
                caption=f"{label} change over period",
            )
            return True
        return False

    def _try_secondary_kpi():
        pool = [k for k in remaining_kpis if k["column"] != (revenue_kpi or {}).get("column")]
        if pool:
            secondary_kpi = pool[0]
            emoji2, label2, prefix2 = _get_business_context(secondary_kpi["column"])
            _card(
                cols[3],
                _format_value(secondary_kpi.get("mean") or 0, prefix2),
                f"{emoji2} AVG {label2.upper()}",
                _trend_html(secondary_kpi["column"]),
            )
            return True
        return False

    if not any(strategy() for strategy in (_try_top_segment, _try_growth_rate, _try_secondary_kpi)):
        _card(cols[3], f"{len(df):,}", f"🧾 TOTAL {entity_plural.upper()}", caption="No additional segment signal in this dataset")


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
    Renders one Caterpillar-style trend chart per detected trend
    (full width), plus a forecast chart if forecast data is available
    in verified_stats.
    """
    trends = verified_stats.get("trends", [])
    if not trends:
        return

    _section_header("📈 Trend Analysis", f"{len(trends)} trend(s) detected")

    for t in trends:
        date_col  = t.get("date_column")
        value_col = t.get("value_column")
        if not date_col or not value_col or date_col not in df.columns or value_col not in df.columns:
            continue

        labels, totals, granularity = _aggregate_trend_smart(df, date_col, value_col)
        if not labels:
            continue

        emoji, biz_label, prefix = _get_business_context(value_col)
        # `direction` is already reconciled with pct_change_start_to_end in
        # stats_engine.py (same source of truth), so the arrow word and the
        # sign of the percentage shown next to it can never contradict —
        # they're derived from the same number.
        direction = t.get("direction", "")
        pct = t.get("pct_change_start_to_end")
        pct_txt = f" ({pct:+.1f}%)" if pct is not None else ""
        title = f"{biz_label} Trend — {direction.title()}{pct_txt}"
        if t.get("direction_conflict"):
            title += "  ⚠ early/late outlier distorts the simple trend line"

        fig = _build_caterpillar_trend_fig(
            labels, totals, biz_label, title, prefix, granularity
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

    _render_forecast_if_available(verified_stats, df)


def _render_forecast_if_available(verified_stats: dict, df: pd.DataFrame):
    forecast_data = verified_stats.get("forecast")
    if not forecast_data:
        return

    combined  = forecast_data.get("combined_data")
    date_col  = forecast_data.get("date_column")
    value_col = forecast_data.get("value_column")
    if combined is None or not date_col or not value_col:
        return

    if isinstance(combined, dict):
        combined = pd.DataFrame(combined)
    if not isinstance(combined, pd.DataFrame) or combined.empty:
        return

    historical = combined[combined["type"] == "historical"]
    forecast   = combined[combined["type"] == "forecast"]
    if forecast.empty:
        return

    emoji, biz_label, prefix = _get_business_context(value_col)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical[date_col], y=historical[value_col],
        mode="lines+markers", name="Historical",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=6),
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
# AI-SELECTED CHARTS  (pie · donut · bar · column · stacked_bar · scatter · histogram)
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
    - Bar / column / stacked_bar / scatter / histogram → two-column grid.

    High-cardinality columns (>12 unique values or top value <5% share)
    are silently dropped even if the LLM spec included them — they produce
    unreadable charts.
    """
    if not chart_specs:
        return

    # Filter to only allowed non-line types
    allowed   = {"pie", "donut", "bar", "column", "stacked_bar", "scatter", "histogram"}
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

    # Bar / column / stacked_bar / scatter / histogram: 2-column grid
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
# ESSENTIAL CHARTS  (single capped section — replaces the old uncapped
# render_trend_charts + forecast + render_ai_charts + render_category_breakdown
# stack. Max 1 trend line + max 3 curated charts = 4 charts total.)
# ══════════════════════════════════════════════════════════════════════════════

MAX_TREND_CHARTS = 1
MAX_CURATED_CHARTS = 3


def render_essential_charts(df: pd.DataFrame, chart_specs: list, verified_stats: dict):
    trends = verified_stats.get("trends", [])

    # ── At most one trend line: the strongest RELIABLE one. This is the
    # same trend the business_summary paragraph's "top performance finding"
    # sentence is grounded in (get_business_summary_task in tasks.py is
    # told to cite the single most important reliable finding) — so the
    # chart people see backs up the sentence they just read, instead of
    # a wall of one-line-chart-per-column noise.
    reliable_trends = [t for t in trends if not t.get("insufficient_edge_data")]
    pool = reliable_trends or trends
    best_trend = (
        sorted(pool, key=lambda t: t.get("r_squared", 0) or 0, reverse=True)[0]
        if pool else None
    )

    any_chart_rendered = False

    if best_trend:
        date_col, value_col = best_trend.get("date_column"), best_trend.get("value_column")
        if date_col in df.columns and value_col in df.columns:
            labels, totals, granularity = _aggregate_trend_smart(df, date_col, value_col)
            if labels:
                _section_header("📈 Trend That Matters")
                emoji, biz_label, prefix = _get_business_context(value_col)
                direction = best_trend.get("direction", "")
                pct = best_trend.get("pct_change_start_to_end")
                pct_txt = f" ({pct:+.1f}%)" if pct is not None else ""
                title = f"{biz_label} Trend — {direction.title()}{pct_txt}"
                if best_trend.get("direction_conflict"):
                    title += "  ⚠ early/late outlier distorts the simple trend line"
                fig = _build_caterpillar_trend_fig(labels, totals, biz_label, title, prefix, granularity)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
                any_chart_rendered = True

    # ── Up to 3 AI-curated charts, already grounded in verified_stats with
    # per-chart reasoning tied to a real number (see dashboard_agent.py).
    excluded = _high_cardinality_cols(verified_stats)
    curated = [
        s for s in (chart_specs or [])
        if s.get("chart_type") in {"pie", "donut", "bar", "column", "stacked_bar", "scatter", "histogram"}
        and s.get("x_column") not in excluded
    ]
    curated = sorted(curated, key=lambda s: s.get("priority", 99))[:MAX_CURATED_CHARTS]

    if curated:
        if any_chart_rendered:
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        _section_header("📊 Essential Business Charts", f"{len(curated)} chart(s) selected")

        pie_specs   = [s for s in curated if s.get("chart_type") in ("pie", "donut")]
        other_specs = [s for s in curated if s.get("chart_type") not in ("pie", "donut")]

        for spec in pie_specs:
            left, centre, right = st.columns([1, 3, 1])
            with centre:
                fig = render_chart(df, spec, verified_stats)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    if spec.get("reasoning"):
                        st.caption(f"📌 {spec['reasoning']}")
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

        for i in range(0, len(other_specs), 2):
            pair = other_specs[i: i + 2]
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
# MAIN DASHBOARD ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard(
    df: pd.DataFrame,
    chart_specs: list,
    verified_stats: dict = None,
    business_summary: str = "",
    domain_config: dict = None,
):
    verified_stats = verified_stats or {}

    render_kpi_cards(df, verified_stats, domain_config)

    if business_summary:
        render_business_context_summary(business_summary)

    st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

    render_essential_charts(df, chart_specs, verified_stats)