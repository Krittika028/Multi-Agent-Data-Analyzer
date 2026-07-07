"""
domain_context.py

Central domain-adaptation layer. Resolves ONE domain_config dict per
pipeline run from DomainDetector's output, consumed by:
  - stats_engine.py   (status classification for revenue/completion logic)
  - dashboard_renderer.py (KPI card labels, business context vocabulary)
  - tasks.py           (report section names, analyst framing rules)

Previously, "Revenue", "Customers", "Transactions" business-context
keywords, the order-fulfillment status keyword list, and the report's
fixed section names ("Channel & Payment Breakdown", "Category
Performance") were all hardcoded for commerce/retail data. On an IT
ticketing, HR, or healthcare dataset this produced generic fallback
labels at best (harmless but unhelpful) and actively wrong framing at
worst (e.g. a legitimate terminal status like "Rejected" for an
insurance claim swept into "non-completed revenue" using commerce
keywords that don't apply to that domain).

This module does NOT hardcode a second commerce-shaped template per
domain. Instead:
  1. Domain-specific label overrides are configuration (small, explicit,
     easy to extend) — not business logic.
  2. Status-value classification (which categories mean "terminal/done"
     vs "in-progress/failed") is now an LLM semantic call over the
     ACTUAL values found in the ACTUAL status column — not a keyword
     match against a commerce-specific word list. This mirrors the
     existing _llm_canonical_map pattern in data_cleaner.py.
"""

import os
import json
import re

# ── Domain label overrides ──────────────────────────────────────────────────
# Small, explicit config — NOT a full re-hardcoded template per domain.
# Anything not listed here falls back to sensible generic labels, so an
# unlisted/unknown domain degrades gracefully rather than breaking.
DOMAIN_LABELS = {
    "Retail":                    {"dimension": "Product/Category", "entity_plural": "Orders",   "kpi4": "revenue_at_risk"},
    "E-commerce":                {"dimension": "Product/Category", "entity_plural": "Orders",   "kpi4": "revenue_at_risk"},
    "Banking & Finance":         {"dimension": "Account/Segment",  "entity_plural": "Transactions", "kpi4": "revenue_at_risk"},
    "Insurance":                 {"dimension": "Policy/Claim Type","entity_plural": "Claims",    "kpi4": "top_segment"},
    "Healthcare":                {"dimension": "Diagnosis/Department", "entity_plural": "Patient Visits", "kpi4": "top_segment"},
    "Human Resources":           {"dimension": "Department/Role",  "entity_plural": "Employees", "kpi4": "secondary_kpi"},
    "Manufacturing":              {"dimension": "Product Line/Plant","entity_plural": "Production Runs", "kpi4": "top_segment"},
    "Education":                  {"dimension": "Course/Department","entity_plural": "Enrollments", "kpi4": "secondary_kpi"},
    "Logistics":                  {"dimension": "Route/Carrier",   "entity_plural": "Shipments", "kpi4": "top_segment"},
    "Telecom":                     {"dimension": "Plan/Region",     "entity_plural": "Subscribers","kpi4": "secondary_kpi"},
    "Real Estate":                 {"dimension": "Property Type/Region", "entity_plural": "Listings", "kpi4": "top_segment"},
    "Agriculture":                 {"dimension": "Crop/Region",     "entity_plural": "Yields",    "kpi4": "top_segment"},
    "Energy & Utilities":         {"dimension": "Region/Meter Type","entity_plural": "Readings",  "kpi4": "secondary_kpi"},
    "Government / Public Sector": {"dimension": "Department/Program","entity_plural": "Records", "kpi4": "secondary_kpi"},
    # Generic fallback for "Other" or any domain not listed above
    "Other":                      {"dimension": "Category/Segment","entity_plural": "Records",   "kpi4": "secondary_kpi"},
}

DEFAULT_LABELS = DOMAIN_LABELS["Other"]


def get_domain_config(domain_result: dict) -> dict:
    """
    domain_result: the dict returned by DomainDetector.detect() —
    {"domain", "primary_entity", "confidence", "evidence", ...}

    Returns a config dict used across stats_engine/dashboard_renderer/tasks:
        {
          "domain": str,
          "primary_entity": str,
          "dimension_label": str,   # e.g. "Department/Role" for HR
          "entity_plural": str,     # e.g. "Employees" for HR
          "kpi4_type": str,         # "revenue_at_risk" | "top_segment" | "secondary_kpi"
        }
    """
    domain_result = domain_result or {}
    domain = domain_result.get("domain", "Other")
    labels = DOMAIN_LABELS.get(domain, DEFAULT_LABELS)

    return {
        "domain": domain,
        "primary_entity": domain_result.get("primary_entity", "Record"),
        "dimension_label": labels["dimension"],
        "entity_plural": labels["entity_plural"],
        "kpi4_type": labels["kpi4"],
    }


# ── LLM-driven status-value semantic classification ─────────────────────────
# Replaces the hardcoded _NON_COMPLETED_STATUS_KEYWORDS keyword match in
# stats_engine.py, which only recognized commerce fulfillment language
# (cancel/refund/pending/...). This asks the LLM to classify the ACTUAL
# values found in the ACTUAL status column for THIS dataset, the same
# pattern data_cleaner.py already uses for categorical canonicalization.

def classify_status_values(column_name: str, unique_values: list) -> dict:
    """
    Returns {"non_completed": [...], "completed": [...]} using only values
    that actually appear in unique_values. Falls back to an empty
    classification (treat everything as completed / no exclusion) if the
    LLM is unavailable or the response can't be parsed — this is a
    conservative fallback: it means revenue-by-status simply won't
    exclude anything, rather than guessing wrong and excluding real
    completed records.
    """
    try:
        model = os.getenv("MODEL", "")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not model or not unique_values:
            return {"non_completed": [], "completed": list(unique_values)}

        import litellm
        prompt = f"""You are a business process analyst. A dataset has a
status-like column named "{column_name}" with these observed values:
{json.dumps(unique_values)}

Classify EVERY value into exactly one of:
- "non_completed": the record has NOT reached a successful, revenue/value
  -realizing terminal state (e.g. cancelled, failed, pending, rejected,
  returned, in-progress, open, abandoned) — the business should NOT count
  this as a fully realized/reliable outcome yet.
- "completed": the record HAS reached a successful terminal state (e.g.
  delivered, resolved, closed-won, approved, paid, shipped, discharged) —
  a normal, successfully finished record for this domain.

Use the DOMAIN CONTEXT implied by the column name and values themselves —
do not assume retail/e-commerce meaning if the values suggest a different
domain (e.g. "Open"/"Resolved" for IT tickets, "Rejected"/"Approved" for
insurance claims, "Admitted"/"Discharged" for healthcare).

Return ONLY valid JSON, no markdown, no explanation:
{{"non_completed": ["value1", ...], "completed": ["value2", ...]}}
Every value in the input list must appear in exactly one of the two lists."""

        resp = litellm.completion(
            model=model, api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        result = json.loads(raw)

        valid_set = set(unique_values)
        non_completed = [v for v in result.get("non_completed", []) if v in valid_set]
        completed = [v for v in result.get("completed", []) if v in valid_set]
        # Anything the LLM missed defaults to "completed" — conservative:
        # under-excluding is safer than over-excluding real revenue/records.
        classified = set(non_completed) | set(completed)
        for v in unique_values:
            if v not in classified:
                completed.append(v)

        return {"non_completed": non_completed, "completed": completed}

    except Exception:
        return {"non_completed": [], "completed": list(unique_values)}