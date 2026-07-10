"""
data_cleaner.py

Senior-grade data cleaning pipeline — generalized for any domain
(Retail, Banking, Healthcare, HR, Logistics, E-commerce, etc.)

Fixes vs previous version:
  1. Post-imputation domain-aware clipping — rating columns clipped to [1,5],
     all physically-impossible negatives floored to 0 (amounts, time, age, etc.)
  2. Outlier capping has a domain-aware floor — negative values removed from
     columns where they are impossible (amounts, durations, counts, ages)
  3. Date split: month_name uses full "January" not "Jan"; month is an integer 1-12
     (both more precise and unambiguous for charting/sorting)
  4. Fuzzy threshold uses relative edit-distance so short words are never over-merged
     and longer typo-words (Italian/Italain) always merge correctly
  5. step4_remove_duplicates no longer collapses rows with a MISSING id into
     "duplicates of each other" — rows with a null id are excluded from
     id-based dedup and a separate full-row-duplicate pass always runs too.
  6. NULL_STRINGS restored to the full error/invalid variant set
     (ERR/err/Err, #ERROR!, invalid, corrupt, void, etc.), with a
     module-level precomputed lowercase set (_NULL_STRINGS_LOWER) instead
     of rebuilding it inside step2 on every call.
  7. _sweep_nulls_again() runs after step7 as a backstop against the LLM
     canonical mapping introducing a placeholder-style label post-step2.

--------------------------------------------------------------------
CRITICAL FIX (this version) — ROW-LOSS BUG:

step4_remove_duplicates previously trusted ANY column ending in "_id"
(or literally named "id") as a safe deduplication key, without ever
verifying that column was actually close to unique. On a dataset where
an id-like column repeats across genuinely distinct rows (e.g. a
replicated/synthetic file, or an id that isn't a true primary key),
this caused MASSIVE, SILENT data loss — a real production run went
from 150,000 rows to 1,500 rows (99% of rows discarded) because
`order_id` was not unique in that file, and the pipeline treated every
repeat as "the same order."

Fix — a two-stage safety gate, both independent of each other:
  1. UNIQUENESS GATE: an _id column is only trusted as a dedup key if
     it was >= 95% unique BEFORE any cleaning touched it. Below that,
     id-based dedup is skipped entirely and only full-row dedup runs.
  2. REMOVAL-SIZE GATE: even if the column passes gate 1, if id-based
     dedup would still remove more than 30% of rows, that is treated
     as a red flag (ID collision, not real duplication) and the
     id-based removal is aborted — full-row dedup still runs on top.
Every skip/abort is logged explicitly and loudly (not folded into a
routine cleaning log line) so it can never again pass silently into
downstream KPIs, dashboards, or the business report.
--------------------------------------------------------------------
PERFORMANCE FIXES (this version):

  - step2_convert_nulls_to_nan and _sweep_nulls_again were rewritten
    from per-cell Python `for val in series:` loops to vectorized
    pandas string/boolean operations. At 150K rows x 17 columns this
    was ~2.5M Python-level iterations before any real work happened —
    now it's vectorized C-level pandas ops.
  - step7's fuzzy near-duplicate merge (_build_canonical_map) is
    O(n^2) in the number of UNIQUE categorical values. A cap
    (_MAX_FUZZY_UNIQUE_VALUES) now skips fuzzy merging (Title Case
    still applied) above that threshold, so one noisy high-cardinality
    column can't blow up total cleaning time.
  - step9 RF imputation now caps training rows per model fit
    (MAX_RF_TRAIN_ROWS) via random sampling, so a 150-tree RF fit
    on a 150K-row column costs roughly the same as on a capped sample,
    without materially hurting imputation quality.
--------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from dateutil import parser as _dateutil_parser
import os
import json
import re
import unicodedata

NULL_STRINGS = {
    # Standard null representations
    "None", "none", "NONE", "NULL", "null", "nan", "NaN", "NAN",
    "NA", "N/A", "n/a", "na", "N/a", "", " ", "-", "--", "?",
    # Unknown / undefined
    "unknown", "Unknown", "UNKNOWN", "undefined", "Undefined", "nil", "Nil",
    # Missing
    "missing", "Missing", "MISSING", "not available", "Not Available",
    # Abbreviation variants
    "n.a", "N.A", "n.a.", "N.A.", "#N/A", "#NA", "TBD", "tbd", "TBC", "tbc",
    "N/A.", "n/a.", "NOT APPLICABLE", "not applicable", "NA.", "n/a -",
    # Not provided
    "not provided", "Not Provided", "NOT PROVIDED", "N/P", "n/p",
    # ERROR variants — real-world data-entry / system errors
    "ERROR", "error", "Error", "ERR", "err", "Err",
    "#ERROR", "#ERROR!", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NUM!", "#NULL!",
    # Other invalid markers
    "invalid", "Invalid", "INVALID",
    "corrupt", "Corrupt", "CORRUPT",
    "bad", "BAD",
    "void", "Void", "VOID",
}

# Precomputed once at module load — avoid rebuilding this set on every
# step2/sweep call across every column.
_NULL_STRINGS_LOWER = {s.lower().strip() for s in NULL_STRINGS}

MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
    'oct': 10, 'nov': 11, 'dec': 12,
}

# Date columns where we skip year/month split (metadata / DOB)
_DATE_NOSPLIT_KEYWORDS = ['dob', 'birth', 'created_at', 'updated_at', 'modified', 'timestamp']

# ── Domain-aware floor rules ───────────────────────────────────────────────────
# Any numeric column whose name contains one of these keywords must be >= 0.
# NOTE: this list intentionally EXCLUDES 'balance', 'credit', 'debit',
# 'profit', 'loss', 'margin', 'refund', 'adjustment' — those can be
# legitimately negative in banking/finance and retail contexts (an
# overdrawn account, a refund/chargeback, a loss-making SKU, a credit
# memo). Blanket-flooring those to 0 silently destroys real signal for
# SMB finance and retail P&L use cases. See _SIGNED_ALLOWED_KEYWORDS.
_NON_NEGATIVE_KEYWORDS = [
    'amount', 'price', 'cost', 'revenue', 'sales', 'fee', 'charge',
    'payment', 'salary', 'wage', 'income', 'loan', 'budget', 'tax',
    'invoice', 'total', 'mrp', 'unit_price', 'subtotal', 'shipping',
    'time', 'duration', 'minutes', 'hours', 'days', 'age',
    'count', 'quantity', 'qty', 'volume', 'units', 'stock', 'inventory',
    'distance', 'weight', 'height',
]

# Columns where negative values are legitimate business signal and must
# NEVER be floored to 0 — e.g. an overdrawn bank balance, a refund /
# chargeback amount, a loss-making product margin, a credit adjustment.
# Checked FIRST in step9b/step10, before the keyword lists above/below.
_SIGNED_ALLOWED_KEYWORDS = [
    'balance', 'credit', 'debit', 'profit', 'loss', 'margin',
    'refund', 'chargeback', 'adjustment', 'variance', 'delta',
    'change', 'net',
]

# Rating/score columns: clip to [1, 5] unless the column name suggests a different range
_RATING_KEYWORDS  = ['rating', 'score', 'stars', 'review_score', 'satisfaction', 'nps']
_RATING_MIN, _RATING_MAX = 1.0, 5.0

# Percentage columns: clip to [0, 100]. 'margin' and 'rate' deliberately
# excluded — a profit margin can be negative (loss-making) or exceed
# 100% (markup math), and a generic "*_rate" column is too ambiguous
# (interest rate, growth rate, error rate can all be negative or >100).
# Handled as a signed/unbounded metric instead of a bounded percentage.
_PERCENT_KEYWORDS = ['percent', 'pct', 'discount', 'utilization', 'conversion_rate', 'churn_rate']
_PERCENT_MIN, _PERCENT_MAX = 0.0, 100.0

# ── SMB domain keyword extensions — retail / e-commerce / banking ─────────────
# Extra "identifier-like" keywords so corrupted-ID scrubbing (step8) also
# catches the identifiers most common in these three domains for SMBs.
_ID_LIKE_EXTRA_KEYWORDS = [
    'sku', 'upc', 'ean', 'barcode', 'iban', 'ifsc', 'swift',
    'account_number', 'card_number', 'routing_number', 'gstin', 'pan',
    'order_number', 'invoice_number', 'tracking_number', 'awb',
]

# ── Row-loss safety thresholds for id-based dedup ──────────────────────────────
_MIN_ID_UNIQUENESS_FOR_DEDUP = 0.95   # id column must be this unique pre-clean to be trusted
_MAX_SAFE_ID_REMOVAL_PCT     = 0.30   # abort id-based dedup if it would remove more than this

# ── Fuzzy dedup performance cap ─────────────────────────────────────────────────
_MAX_FUZZY_UNIQUE_VALUES = 500   # above this, skip pairwise edit-distance merge (too slow to be worth it)

# ── RF imputation performance cap ───────────────────────────────────────────────
_MAX_RF_TRAIN_ROWS = 20_000      # sample down to this many rows per column fit on large datasets

# ── LLM call safety caps (this version) ─────────────────────────────────────────
# Every LLM call in this file previously had no request timeout and could be
# retried by nothing but a bare try/except — a single slow/hanging endpoint
# could stall the ENTIRE cleaning pipeline indefinitely, which on Streamlit
# Cloud shows up as the platform killing the app with a generic "Received no
# response from server" error (no Python traceback is produced, because the
# process is still "running", just stuck).
#
# Fixes:
#   1. Every litellm.completion() call in this file now passes an explicit
#      `timeout=` so a hung request fails fast instead of hanging forever.
#   2. step7's per-COLUMN canonical-mapping LLM call (previously unbounded —
#      one call per categorical column, however many that is) is now capped
#      at _MAX_LLM_CANONICAL_CALLS total calls per clean() run. Columns
#      beyond the cap fall back to the deterministic Title Case + fuzzy
#      edit-distance merge — the same graceful path already used whenever
#      the LLM call fails, just bounded proactively instead of only
#      reactively (after already burning the time on the call).
#   3. A column's unique-value list is only sent to the canonical-mapping
#      LLM call if it's small enough to answer quickly/cheaply
#      (_MAX_LLM_CANONICAL_UNIQUE_VALS) — a very high-cardinality column
#      goes straight to the deterministic fuzzy merge instead.
_LLM_TIMEOUT_SECONDS            = 20   # hard per-call timeout — fail fast, don't hang the pipeline
_MAX_LLM_CANONICAL_CALLS        = 12   # total step7 LLM canonicalization calls allowed per clean() run
_MAX_LLM_CANONICAL_UNIQUE_VALS  = 150  # skip the LLM call for columns with more uniques than this

# ── Date-parsing performance cap ────────────────────────────────────────────────
# The final fallback in _try_parse_date_column() parses every value
# one-at-a-time with dateutil — fine for a few thousand rows, but on a
# 100K+ row column where none of the fast vectorized/format attempts hit
# the 75% threshold, this row-by-row pass can take minutes by itself. Above
# this row count, skip the row-by-row fallback entirely (the column is left
# as text rather than the app burning minutes on it) — the vectorized
# attempts above already give this every reasonable shot first.
_MAX_ROWS_FOR_ROWWISE_DATE_PARSE = 20_000

# ── RF imputation total-time-budget cap ─────────────────────────────────────────
# n_estimators=150 per RF fit is fine for one or two columns, but a wide
# dataset with many missing columns fits one full forest PER column,
# sequentially. Above this many columns needing RF imputation, later
# columns automatically use a cheaper forest (fewer trees) so total
# clean() time stays roughly bounded regardless of dataset width.
_RF_CHEAP_MODE_AFTER_N_COLUMNS = 6
_RF_ESTIMATORS_NORMAL = 150
_RF_ESTIMATORS_CHEAP  = 60


# ── Fuzzy near-duplicate merging ───────────────────────────────────────────────

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev[j-1] + cost)
    return dp[n]


def _build_canonical_map(values: list, max_edit_dist: int = 2) -> dict:
    """
    Groups near-duplicate strings → most-frequent canonical form.
    Uses BOTH absolute edit distance AND relative distance to avoid
    merging short distinct words (e.g. 'Cat' vs 'Car') while still
    catching longer typos ('Italain' → 'Italian', 'Bangalroe' → 'Bangalore').

    NOTE: O(n^2) in len(values) — callers should gate this behind
    _MAX_FUZZY_UNIQUE_VALUES for large datasets (see step7).
    """
    freq = pd.Series(values).value_counts()
    sorted_vals = freq.index.tolist()

    normalized_to_canonical: dict = {}
    raw_to_canonical: dict = {}

    for raw in sorted_vals:
        norm = _normalize_text(raw)
        if not norm:
            continue

        if norm in normalized_to_canonical:
            raw_to_canonical[raw] = normalized_to_canonical[norm]
            continue

        matched = False
        for known_norm, canonical_raw in normalized_to_canonical.items():
            dist = _edit_distance(norm, known_norm)
            max_len = max(len(norm), len(known_norm))
            # Relative threshold: 20% for short words (stricter), 30% for longer
            relative_threshold = 0.20 if max_len <= 6 else 0.30
            relative_dist = dist / max_len if max_len > 0 else 1
            if dist <= max_edit_dist and relative_dist <= relative_threshold:
                raw_to_canonical[raw] = canonical_raw
                matched = True
                break

        if not matched:
            normalized_to_canonical[norm] = raw
            raw_to_canonical[raw] = raw

    return raw_to_canonical


# ── LLM dtype detection ────────────────────────────────────────────────────────

def _llm_detect_dtypes(column_names: list, sample_rows: list) -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        model = os.getenv("MODEL", "")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not model:
            return {}
        import litellm
        prompt = f"""
You are a data type detection expert. Given column names and sample values,
identify the correct data type for each column.

Column names: {json.dumps(column_names)}

Sample rows (first 5):
{json.dumps(sample_rows, default=str)}

For each column, respond with one of these types:
- "numeric"     : numbers, amounts, prices, quantities, scores, percentages
- "datetime"    : dates, times, timestamps (any format)
- "categorical" : low-cardinality labels like status, type, category, gender, region, city, country
- "text"        : free-form text, names, descriptions, comments (high cardinality)
- "id"          : identifiers, codes, transaction IDs (should be ignored in analysis)

Respond ONLY with a valid JSON object mapping each column name to its type.
No explanation, no markdown, no extra text.
"""
        response = litellm.completion(
            model=model,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0,
            timeout=_LLM_TIMEOUT_SECONDS,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {}

def _clean_numeric_string(series: pd.Series) -> pd.Series:
    """
    Generic numeric cleaner — works on any dataset, any language of
    garbage tokens. Doesn't hardcode dataset-specific words; instead:
      1. Known universal null-markers (NULL_STRINGS) -> NaN
      2. Strip universal numeric-formatting noise (currency, commas,
         %, accounting parens, +/-, k/M/B suffixes)
      3. Anything left that still isn't a valid number -> NaN via
         pd.to_numeric(errors='coerce') — this alone catches ANY
         garbage token ("ERROR", "N/A", "xyz", "###", emojis, etc.)
         without needing a hardcoded word list.
    """
    s = series.astype(str).str.strip()

    s = s.where(~s.str.lower().isin(_NULL_STRINGS_LOWER), np.nan)

    s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)         # (123.45) -> -123.45
    s = s.str.replace(r'[₹$€£¥₩,%\s]', '', regex=True)           # currency/commas/%/spaces
    s = s.str.replace(r'^\+', '', regex=True)                     # leading +

    def _expand_unit(val):
        if not isinstance(val, str) or val == '' or val.lower() in ('nan', 'none', '-', 'nat'):
            return val
        m = re.match(r'^(-?\d+\.?\d*)([kKmMbB])$', val)
        if m:
            num, unit = float(m.group(1)), m.group(2).lower()
            return str(num * {'k': 1e3, 'm': 1e6, 'b': 1e9}[unit])
        return val

    s = s.apply(_expand_unit)
    # Anything non-numeric that survives (any garbage in any dataset,
    # any language) becomes NaN here automatically — no word list needed.
    return pd.to_numeric(s, errors='coerce')


def _detect_dayfirst(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    pattern = re.compile(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$')
    day_votes = month_votes = 0
    for val in sample:
        m = pattern.match(val.strip())
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 and b <= 12: day_votes += 1
        elif b > 12 and a <= 12: month_votes += 1
    if day_votes != month_votes:
        return day_votes > month_votes
    try:
        model, api_key = os.getenv("MODEL", ""), os.getenv("OPENAI_API_KEY", "")
        if model:
            import litellm
            prompt = (f"All values below come from ONE column, same date format. "
                      f"Is it day-first (DD-MM-YYYY) or month-first (MM-DD-YYYY)?\n"
                      f"Sample: {json.dumps(sample.head(15).tolist())}\n"
                      f'Respond with ONLY: "dayfirst" or "monthfirst".')
            resp = litellm.completion(model=model, api_key=api_key,
                messages=[{"role": "user", "content": prompt}], max_tokens=10, temperature=0,
                timeout=_LLM_TIMEOUT_SECONDS)
            return "dayfirst" in resp.choices[0].message.content.strip().lower()
    except Exception:
        pass
    return False

# ── Date parsing helpers ───────────────────────────────────────────────────────

def _parse_month_year_string(val: str):
    if not isinstance(val, str):
        return None
    val = val.strip()
    try:
        return pd.to_datetime(val, infer_datetime_format=True)
    except Exception:
        pass
    match = re.search(r'(?P<month>[a-zA-Z]+)[-\s/](?P<year>\d{2,4})', val, re.IGNORECASE)
    if match:
        m = match.group('month').lower()
        y = match.group('year')
        if m in MONTH_MAP:
            year = int(y) + (2000 if len(y) == 2 and int(y) < 50 else
                             1900 if len(y) == 2 else 0)
            try:
                return pd.Timestamp(year=year, month=MONTH_MAP[m], day=1)
            except Exception:
                pass
    match = re.search(r'(?P<year>\d{4})[-\s/](?P<month>[a-zA-Z]+)', val, re.IGNORECASE)
    if match:
        m = match.group('month').lower()
        y = int(match.group('year'))
        if m in MONTH_MAP:
            try:
                return pd.Timestamp(year=y, month=MONTH_MAP[m], day=1)
            except Exception:
                pass
    match = re.match(r'^(?P<month>\d{1,2})[-/](?P<year>\d{4})$', val)
    if match:
        try:
            return pd.Timestamp(year=int(match.group('year')), month=int(match.group('month')), day=1)
        except Exception:
            pass
    return None


def _column_has_month_names(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(20)
    pattern = re.compile(
        r'\b(january|february|march|april|may|june|july|august|'
        r'september|october|november|december|'
        r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
        re.IGNORECASE
    )
    hits = sample.apply(lambda v: bool(pattern.search(v))).sum()
    return hits >= min(3, len(sample) // 2 + 1)


def _try_parse_date_column(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    try:
        numeric = pd.to_numeric(series, errors='coerce')
        unix_mask = numeric.notna() & (numeric > 1e9) & (numeric < 2e10)
        if unix_mask.sum() / max(len(non_null), 1) >= 0.5:
            converted = pd.to_datetime(numeric, unit='s', errors='coerce')
            if converted.notnull().sum() / len(non_null) >= 0.5:
                return converted
    except Exception:
        pass

    dayfirst = _detect_dayfirst(non_null)

    try:
        converted = pd.to_datetime(series, dayfirst=dayfirst, errors='coerce')
        if converted.notnull().sum() / len(non_null) >= 0.75:
            return converted
    except Exception:
        pass

    fmt_candidates = (
        ["%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y"]
        if dayfirst else
        ["%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M", "%m-%d-%Y", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"]
    ) + [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M", "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
        "%b-%Y", "%B-%Y", "%b %Y", "%B %Y", "%m/%Y", "%Y%m%d",
    ]

    for fmt in fmt_candidates:
        try:
            converted = pd.to_datetime(series, format=fmt, errors='coerce')
            if converted.notnull().sum() / len(non_null) >= 0.75:
                return converted
        except Exception:
            continue

    # PERFORMANCE GUARD: the row-by-row dateutil fallback below is a Python-
    # level loop over every value. On a 100K+ row column that didn't parse
    # cleanly with any of the fast vectorized/format attempts above, this
    # can take minutes on its own and is the kind of thing that makes an
    # app just hang with no error — exactly what shows up on Streamlit
    # Cloud as "Received no response from server". Above the row cap, skip
    # it: the column is left unparsed (stays text) rather than the pipeline
    # silently burning minutes trying to save it value-by-value.
    if len(non_null) > _MAX_ROWS_FOR_ROWWISE_DATE_PARSE:
        return None

    def _parse_single(val):
        if pd.isnull(val):
            return pd.NaT
        s = str(val).strip()
        try:
            n = float(s)
            if 1e9 < n < 2e10:
                return pd.Timestamp(n, unit='s')
        except Exception:
            pass
        try:
            return _dateutil_parser.parse(s, dayfirst=dayfirst)
        except Exception:
            pass
        for fmt in fmt_candidates:
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        result = _parse_month_year_string(s)
        return result if result is not None else pd.NaT

    try:
        parsed = series.apply(_parse_single)
        if parsed.notnull().sum() / len(non_null) >= 0.75:
            return pd.to_datetime(parsed, errors='coerce')
    except Exception:
        pass
    return None


# ── DataCleaner ────────────────────────────────────────────────────────────────

class DataCleaner:

    def __init__(self, df: pd.DataFrame):
        self.df             = df.copy()
        self.original_shape = df.shape
        # Raw, untouched snapshot — kept purely for get_scorecard()'s
        # "before" metrics, so the scorecard reflects the ACTUAL input
        # data quality rather than a re-derived estimate.
        self._raw_df        = df.copy()
        self.report         = []
        self.issues_found   = []
        self.fixes_applied  = []
        self._encoders      = {}
        self._llm_dtypes    = {}
        # Track which columns came from imputation (for post-clip)
        self._imputed_cols  = set()
        # Row-retention bookkeeping — surfaced to crew.py for the
        # independent reporting-layer tripwire (see tasks.py).
        self.rows_before_dedup = None
        self.rows_after_dedup  = None
        # Near-duplicate ROW pairs found during cleaning (populated by
        # _detect_near_duplicate_rows(), called from clean()).
        self._near_duplicate_rows = []
        # Budget counter for step7's per-column LLM canonicalization calls —
        # see _MAX_LLM_CANONICAL_CALLS. Reset per clean() run.
        self._llm_canonical_calls_used = 0

    # ── STEP 0: Drop user-selected columns ────────────────────────────────────
    def step0_drop_columns(self, columns_to_drop):
        if not columns_to_drop:
            self.report.append("[DROP COLUMNS] ✔ No columns removed by user")
            return
        existing = [c for c in columns_to_drop if c in self.df.columns]
        missing  = [c for c in columns_to_drop if c not in self.df.columns]
        if existing:
            self.df = self.df.drop(columns=existing)
            self.report.append(f"[DROP COLUMNS] ✅ Removed {len(existing)} column(s): {', '.join(existing)}")
            self.fixes_applied.append(f"Removed {len(existing)} user-selected column(s)")
        if missing:
            self.report.append(f"[DROP COLUMNS] ⚠ {len(missing)} requested column(s) not found: {', '.join(missing)}")

    # ── STEP 1: Standardize column names ──────────────────────────────────────
    def step1_standardize_columns(self):
        original = list(self.df.columns)
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(r'\s+', '_', regex=True)
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )
        changed = [f"'{o}'->'{n}'" for o, n in zip(original, self.df.columns) if o != n]
        if changed:
            self.report.append(f"[COLUMNS] ✅ Renamed {len(changed)} columns: {', '.join(changed)}")
            self.fixes_applied.append(f"Renamed {len(changed)} column(s)")
        else:
            self.report.append(f"[COLUMNS] ✔ All {len(self.df.columns)} column names are clean")

    # ── STEP 2: Convert null-like strings → NaN (VECTORIZED) ──────────────────
    def step2_convert_nulls_to_nan(self):
        """
        Vectorized replacement of the previous per-cell Python loop.
        At 150K rows x 17 columns the old `for val in series:` approach
        meant ~2.5M Python-level iterations before any real work started.
        This does the equivalent check as a single vectorized pandas
        string/boolean operation per column.
        """
        total = 0
        for col in self.df.columns:
            s = self.df[col]
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                continue  # can't contain null-marker strings
            str_s = s.astype(str).str.strip()
            mask = str_s.str.lower().isin(_NULL_STRINGS_LOWER) & s.notna()
            count = int(mask.sum())
            if count:
                self.df.loc[mask, col] = np.nan
            total += count
        if total > 0:
            self.report.append(f"[NULL STRINGS] ✅ {total} null-string cells converted to NaN")
            self.issues_found.append(f"{total} null-string values")
            self.fixes_applied.append(f"Converted {total} null strings to NaN")
        else:
            self.report.append(f"[NULL STRINGS] ✔ No null strings found")

    # ── STEP 3: Report missing values ─────────────────────────────────────────
    def step3_check_missing(self):
        missing_per_col = self.df.isnull().sum()
        total_missing   = missing_per_col.sum()
        if total_missing > 0:
            self.issues_found.append(
                f"{total_missing} missing values across {(missing_per_col > 0).sum()} column(s)"
            )
            for col, cnt in missing_per_col[missing_per_col > 0].items():
                pct = round(cnt / len(self.df) * 100, 1)
                self.report.append(f"[MISSING] ⚠ '{col}': {cnt} missing ({pct}%)")
        else:
            self.report.append(f"[MISSING] ✔ No missing values found")

    # ── STEP 4: Remove duplicates (CRITICAL FIX: uniqueness + removal-size gates) ──
    def step4_remove_duplicates(self):
        """
        FIXED — two independent safety gates before any id-based dedup runs:

          GATE 1 (uniqueness): an *_id column is only trusted as a dedup
          key if it was >= _MIN_ID_UNIQUENESS_FOR_DEDUP unique BEFORE any
          cleaning touched it. A repeating id (e.g. a replicated/synthetic
          file where order_id is not a true primary key) previously caused
          rows to be silently collapsed as "duplicates" when they were
          genuinely distinct records — this is the exact bug that took a
          150,000-row dataset down to 1,500 rows.

          GATE 2 (removal size): even if gate 1 passes, if id-based dedup
          would still remove more than _MAX_SAFE_ID_REMOVAL_PCT of rows,
          that is treated as an ID-collision red flag rather than real
          duplication, and the id-based removal is aborted.

        A separate, always-run, definition-consistent full-row-duplicate
        pass runs afterward regardless — using the exact same logic a raw
        "duplicate rows" stat display would use, so the two numbers can
        never silently disagree.
        """
        before = len(self.df)
        self.rows_before_dedup = before

        id_col = next((c for c in self.df.columns if c.endswith('_id') or c == 'id'), None)
        id_dupes_removed = 0

        if id_col:
            non_null_ids = self.df[id_col].dropna()
            uniqueness_ratio = (
                non_null_ids.nunique() / len(non_null_ids) if len(non_null_ids) else 0
            )

            if uniqueness_ratio < _MIN_ID_UNIQUENESS_FOR_DEDUP:
                self.report.append(
                    f"[DUPLICATES] ⚠ '{id_col}' is only {uniqueness_ratio:.1%} unique "
                    f"pre-clean — NOT trusted as a dedup key (would silently discard "
                    f"legitimate rows). Skipping id-based dedup; running full-row "
                    f"dedup only."
                )
            else:
                null_id_mask = self.df[id_col].isna()
                n_null_ids   = int(null_id_mask.sum())

                non_null_part = self.df[~null_id_mask]
                null_part     = self.df[null_id_mask]

                deduped_non_null = non_null_part.drop_duplicates(subset=[id_col])
                id_dupes_removed  = len(non_null_part) - len(deduped_non_null)
                removal_pct = (
                    id_dupes_removed / len(non_null_part) if len(non_null_part) else 0
                )

                if removal_pct > _MAX_SAFE_ID_REMOVAL_PCT:
                    self.report.append(
                        f"[DUPLICATES] 🛑 id-based dedup on '{id_col}' would remove "
                        f"{removal_pct:.1%} of rows ({id_dupes_removed:,}) — this "
                        f"exceeds the {_MAX_SAFE_ID_REMOVAL_PCT:.0%} safety threshold "
                        f"and looks like an ID collision, not real duplicates. "
                        f"ABORTED — falling back to full-row dedup only."
                    )
                    id_dupes_removed = 0
                else:
                    # Recombine — rows with a missing id are NEVER dropped here.
                    # Preserve original row order.
                    self.df = pd.concat([deduped_non_null, null_part]).sort_index()

                    if id_dupes_removed > 0:
                        self.report.append(
                            f"[DUPLICATES] ✅ {id_dupes_removed} duplicate '{id_col}' rows "
                            f"removed ({uniqueness_ratio:.1%} pre-clean uniqueness — "
                            f"trusted as dedup key)"
                        )
                        self.issues_found.append(f"{id_dupes_removed} id-based duplicates")
                        self.fixes_applied.append(f"Removed {id_dupes_removed} duplicate '{id_col}' rows")
                    else:
                        self.report.append(
                            f"[DUPLICATES] ✔ No duplicates found on non-null '{id_col}' values"
                        )

                    if n_null_ids > 0:
                        self.report.append(
                            f"[DUPLICATES] ℹ {n_null_ids} row(s) have a missing '{id_col}' — "
                            f"these are NOT treated as duplicates of each other (a shared "
                            f"missing ID isn't evidence of a real duplicate); they'll be "
                            f"imputed later"
                        )

        # ── Always run a definition-consistent, full-row duplicate check on
        # top — this is the same "duplicate rows" definition any stat-pill/
        # preview display should use, so it can never silently disagree with
        # the id-based count above. ────────────────────────────────────────
        full_row_dupes_before = int(self.df.duplicated().sum())
        if full_row_dupes_before > 0:
            self.df = self.df.drop_duplicates()
            self.report.append(
                f"[DUPLICATES] ✅ {full_row_dupes_before} additional exact full-row "
                f"duplicate(s) removed (every column identical)"
            )
            self.issues_found.append(f"{full_row_dupes_before} full-row duplicates")
            self.fixes_applied.append(f"Removed {full_row_dupes_before} full-row duplicate(s)")
        elif id_dupes_removed == 0 and (not id_col):
            self.report.append(f"[DUPLICATES] ✔ No duplicate rows found")

        total_removed = before - len(self.df)
        self.rows_after_dedup = len(self.df)

        retention_pct = round((len(self.df) / before * 100), 1) if before else 100.0
        self.report.append(
            f"[DUPLICATES] 📊 Total rows removed: {total_removed} "
            f"({id_dupes_removed} by id, {full_row_dupes_before} by full-row match) — "
            f"retention {retention_pct}% ({len(self.df):,} of {before:,} rows kept)"
        )

        # Loud, unmissable flag if overall retention still looks suspicious
        # after both gates — this should be rare now, but if it happens it
        # must not be buried in a routine log line.
        if retention_pct < 90.0:
            self.report.append(
                f"[DUPLICATES] ⚠⚠ ROW RETENTION WARNING: only {retention_pct}% of rows "
                f"survived deduplication. Verify this reflects genuine duplicate records "
                f"and not an ID/key assumption issue before trusting downstream KPIs."
            )

    # ── STEP 5: Fix currency symbols ──────────────────────────────────────────
    def step5_fix_currency(self):
        fixed = 0
        for col in self.df.columns:
            as_str = self.df[col].astype(str)
            if as_str.str.contains(r'[₹\$€£¥₩]', regex=True).any():
                null_mask = self.df[col].isnull()
                cleaned   = as_str.str.replace(r'[₹\$€£¥₩,\s]', '', regex=True).str.strip()
                cleaned   = cleaned.replace(list(NULL_STRINGS) + ['nan', 'None', 'NaT'], np.nan)
                self.df[col] = pd.to_numeric(cleaned, errors='coerce')
                self.df.loc[null_mask, col] = np.nan
                self.report.append(f"[CURRENCY] ✅ '{col}': currency symbols stripped → float")
                self.fixes_applied.append(f"Stripped currency from '{col}'")
                fixed += 1
        if fixed == 0:
            self.report.append(f"[CURRENCY] ✔ No currency symbols found")

    # ── STEP 6: Fix data types ─────────────────────────────────────────────────
    def step6_fix_data_types(self):
        obj_cols = [c for c in self.df.columns if self.df[c].dtype == object]
        if not obj_cols:
            self.report.append(f"[DTYPE] ✔ All column types are already correct")
            return

        sample_rows = self.df[obj_cols].head(5).to_dict(orient="records")
        llm_result  = _llm_detect_dtypes(obj_cols, sample_rows)

        if llm_result:
            self.report.append(f"[DTYPE] 🤖 LLM detected dtypes for {len(llm_result)} column(s)")
            self._llm_dtypes = llm_result
        else:
            self.report.append(f"[DTYPE] ⚠ LLM dtype detection unavailable — using heuristics")

        fixed = 0
        for col in obj_cols:
            if self.df[col].dtype != object:
                continue

            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                continue

            llm_type = llm_result.get(col, "").lower() if llm_result else ""

            if llm_type == "id":
                self.report.append(f"[DTYPE] ✔ '{col}': classified as ID — no conversion")
                continue

            cleaned_numeric = _clean_numeric_string(non_null)
            numeric_rate = cleaned_numeric.notna().sum() / len(non_null)
            if numeric_rate > 0.50:
                self.df[col] = _clean_numeric_string(self.df[col])
                self.report.append(f"[DTYPE] ✅ '{col}': string → numeric ({numeric_rate:.0%} parsed; handled commas/currency/%/parentheses/units)")
                self.fixes_applied.append(f"Converted '{col}' to numeric (robust cleaning)")
                fixed += 1
                continue

            date_keywords = [
                'date', 'time', 'dt', 'created', 'updated', 'at', '_on',
                'day', 'month', 'year', 'dob', 'birth', 'timestamp',
                'period', 'transaction_date', 'invoice_date', 'order_date',
            ]
            is_datetime_candidate = (
                llm_type == "datetime"
                or any(k in col.lower() for k in date_keywords)
                or _column_has_month_names(non_null)
            )

            if is_datetime_candidate:
                parsed = _try_parse_date_column(self.df[col])
                if parsed is not None:
                    self.df[col] = parsed
                    self.report.append(f"[DTYPE] ✅ '{col}': string → datetime")
                    self.fixes_applied.append(f"Converted '{col}' to datetime")
                    fixed += 1
                    continue

            
        if fixed == 0:
            self.report.append(f"[DTYPE] ✔ All column types are already correct")

    # ── STEP 6b: Standardize dates → YYYY-MM-DD + precise split ───────────────
    def step6b_split_dates(self):
        """
        Date split improvements:
        - month_name: full name "January", "February" ... (unambiguous, sortable)
        - month: integer 1–12 (precise, directly usable in math/sorting)
        - Only splits when multiple years or months exist in data
        - Skips DOB / metadata timestamps
        """
        # Pass 1: rescue remaining object columns that look like dates
        _DERIVED_SUFFIXES = ("_year", "_month", "_month_name")
        for col in list(self.df.select_dtypes(include="object").columns):
            if col.endswith(_DERIVED_SUFFIXES):
                continue
            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                continue
            date_keywords = [
                'date', 'time', 'dt', 'created', 'updated', 'at', '_on',
                'day', 'month', 'year', 'dob', 'birth', 'timestamp',
                'period', 'transaction_date', 'invoice_date', 'order_date',
            ]
            sample_str = non_null.astype(str).head(20)
            looks_like_date = (
                any(k in col.lower() for k in date_keywords)
                or _column_has_month_names(non_null)
                or sample_str.str.match(
                    r'^\d{4}-\d{2}-\d{2}'
                    r'|^\d{2}-\d{2}-\d{4}'
                    r'|^\d{2}/\d{2}/\d{4}'
                    r'|^\d{4}/\d{2}/\d{2}'
                    r'|^\d{8}$'
                ).sum() >= 3
            )
            if looks_like_date:
                parsed = _try_parse_date_column(self.df[col])
                if parsed is not None:
                    self.df[col] = parsed
                    self.report.append(f"[DATE RESCUE] ✅ '{col}': rescued as datetime")
                    self.fixes_applied.append(f"Rescued '{col}' as datetime")

        # Pass 2: smart split
        date_cols = self.df.select_dtypes(include="datetime").columns.tolist()
        if not date_cols:
            self.report.append("[DATE SPLIT] ✔ No date columns found")
            return

        for col in date_cols:
            self.df[col] = self.df[col].dt.normalize()

            series  = self.df[col].dropna()
            n_unique = series.nunique()

            if n_unique < 2:
                self.report.append(f"[DATE SPLIT] ✔ '{col}': only {n_unique} unique value(s) — skipped")
                continue

            col_lower = col.lower()
            if any(k in col_lower for k in _DATE_NOSPLIT_KEYWORDS):
                self.report.append(f"[DATE SPLIT] ✔ '{col}': metadata/DOB column — skipped analytical split")
                continue

            split_parts = []
            year_col       = f"{col}_year"
            month_col      = f"{col}_month"
            month_name_col = f"{col}_month_name"

            # Idempotency guard: wipe any stale derived columns from a
            # previous run before recomputing, so nothing lingers mismatched.
            for stale in (year_col, month_col, month_name_col):
                if stale in self.df.columns:
                    self.df = self.df.drop(columns=[stale])

            n_years  = series.dt.year.nunique()
            n_months = series.dt.month.nunique()

            if n_years > 1:
                self.df[year_col] = self.df[col].dt.year.astype("Int64")
                split_parts.append(year_col)

            if n_months > 1 or n_years > 1:
                import calendar
                month_ints = self.df[col].dt.month
                self.df[month_col] = month_ints.astype("Int64")
                # Single source of truth: name is DERIVED from the integer,
                # never computed independently — they can no longer disagree.
                self.df[month_name_col] = month_ints.map(
                    lambda m: calendar.month_name[int(m)] if pd.notna(m) else np.nan
                )
                split_parts.extend([month_col, month_name_col])

            if split_parts:
                self.report.append(
                    f"[DATE SPLIT] ✅ '{col}': standardized YYYY-MM-DD, "
                    f"split → {', '.join(repr(p) for p in split_parts)}"
                )
                self.fixes_applied.append(
                    f"Standardized '{col}' + split {len(split_parts)} derived columns"
                )
            else:
                self.report.append(f"[DATE SPLIT] ✔ '{col}': standardized (no split needed)")

    # ── STEP 7: SMART standardization for object/text columns ─────────────────
    def step7_standardize_categoricals(self):
        fixed = 0

        ENCODING_FIXES = {
            '\u2019': "",    '\u2018': "",    '\u201c': "",    '\u201d': "",
            '\u2013': " ",   '\u2014': " ",   '\u2026': "",    '\u00a0': ' ',
            '\u200b': '',    '\u200e': '',    '\u200f': '',    '\ufeff': '',
            '\u00ad': '',    '\r\n': ' ',     '\r': ' ',       '\n': ' ',
            '\t': ' ',
        }

        _SPECIAL_CHARS_RE = re.compile(r'[:\.,\-_!;"\'?/*#@|\\+=(){}\[\]<>^~`%]')

        _BOOL_TRUE  = {"yes", "y", "true", "t", "1", "on", "enabled",
                       "positive", "correct", "right", "affirmative"}
        _BOOL_FALSE = {"no", "n", "false", "f", "0", "off", "disabled",
                       "negative", "incorrect", "wrong"}

        def _is_boolean_column(series: pd.Series) -> bool:
            vals = {str(v).strip().lower() for v in series.dropna()}
            return bool(vals) and vals.issubset(_BOOL_TRUE | _BOOL_FALSE)

        def _clean_string(s: str) -> str:
            if not isinstance(s, str):
                return s
            for bad, good in ENCODING_FIXES.items():
                s = s.replace(bad, good)
            s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
            s = _SPECIAL_CHARS_RE.sub(' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        def _llm_canonical_map(col_name: str, unique_vals: list) -> dict:
            try:
                model = os.getenv("MODEL", "")
                api_key = os.getenv("OPENAI_API_KEY", "")
                if not model:
                    return {}
                import litellm
                vals_json = json.dumps(unique_vals)
                prompt = f"""You are a data standardisation expert.

Column name: "{col_name}"
Unique values found in this column (may contain typos, abbreviations, inconsistent casing):
{vals_json}

Your task:
1. Identify the TRUE set of canonical categories.
2. Map EVERY value to one canonical category.
3. Resolve: abbreviations (e.g. "Dlvrd" -> "Delivered", "Cncld" -> "Cancelled"),
   obvious typos (e.g. "Itlaian" -> "Italian", "Italain" -> "Italian",
   "Mexcian" -> "Mexican", "Chineese" -> "Chinese", "chenai" -> "Chennai",
   "Bangalroe" -> "Bangalore", "Ahemadabad" -> "Ahmedabad"),
   spacing variants, case variants, boolean variants.
4. Every canonical value MUST be in Title Case.
5. If NOT confident, map value to itself (Title Cased).
6. Do NOT map any value to a null/placeholder-style label such as
   "N/A", "Unknown", "None", "Missing", "Error", "Invalid" or similar —
   if a value is genuinely ambiguous, map it to itself (Title Cased)
   instead of inventing a placeholder category.
7. Return ONLY a valid JSON object. No explanation, no markdown.
   Format: {{"dirty_value": "Canonical Value", ...}}
   Every key must be an exact match for a value in the input list."""

                resp = litellm.completion(
                    model=model,
                    api_key=api_key,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0,
                    timeout=_LLM_TIMEOUT_SECONDS,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
                brace = raw.find("{")
                if brace > 0:
                    raw = raw[brace:]
                result = json.loads(raw)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
            return {}

        for col in self.df.columns:
            if col.endswith("_month_name"):
                continue
            is_text_col = (
                self.df[col].dtype == object
                or pd.api.types.is_string_dtype(self.df[col])
            )
            if not is_text_col:
                continue
            if (
                pd.api.types.is_numeric_dtype(self.df[col])
                or pd.api.types.is_datetime64_any_dtype(self.df[col])
            ):
                continue

            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                continue

            # ── SAFETY NET: catches any numeric column Step 6 missed
            # (e.g. LLM sees only 5 sample rows and guesses "categorical"
            # for a low-cardinality numeric column like Quantity 1-5).
            # Converts it to numeric HERE, before any string-cleaning
            # regex can corrupt decimal points into spaces. Dataset-agnostic.
            numeric_probe = _clean_numeric_string(non_null)
            if numeric_probe.notna().sum() / len(non_null) > 0.5:
                full = _clean_numeric_string(self.df[col])
                vals = full.dropna()
                is_whole = bool((vals % 1 == 0).all()) if len(vals) else True
                self.df[col] = full.round().astype("Int64") if is_whole else full.astype("float64")
                self.report.append(
                    f"[DTYPE] ✅ '{col}': string → {'int' if is_whole else 'float'} "
                    f"(caught by categorical-step safety net)"
                )
                self.fixes_applied.append(f"Converted '{col}' to numeric (safety net)")
                fixed += 1
                continue

            nunique           = non_null.nunique()
            n_rows            = len(non_null)
            cardinality_ratio = nunique / n_rows
            is_categorical    = nunique <= 60 or cardinality_ratio < 0.05
            is_name_like      = any(k in col.lower() for k in [
                'name', 'city', 'town', 'country', 'region', 'state',
                'province', 'product', 'brand', 'category', 'item',
            ])

            # Phase 1: Universal string cleaning
            cleaned = self.df[col].apply(
                lambda v: _clean_string(v) if isinstance(v, str) else v
            )
            cleaned = cleaned.replace('', np.nan)
            changed_count = int((cleaned.fillna("__NA__") != self.df[col].fillna("__NA__")).sum())
            self.df[col] = cleaned

            if not is_categorical and not is_name_like:
                if changed_count > 0:
                    self.report.append(
                        f"[TEXT] ✅ '{col}': special chars + whitespace cleaned ({changed_count} values normalised)"
                    )
                    self.fixes_applied.append(f"Cleaned text column '{col}'")
                    fixed += 1
                else:
                    self.report.append(f"[TEXT] ✔ '{col}': already clean")
                continue

            # Phase 2: Boolean normalisation
            if _is_boolean_column(self.df[col]):
                self.df[col] = self.df[col].apply(
                    lambda v: (
                        "Yes" if str(v).strip().lower() in _BOOL_TRUE
                        else ("No" if str(v).strip().lower() in _BOOL_FALSE else v)
                    ) if pd.notna(v) else v
                )
                n_bool = self.df[col].dropna().nunique()
                self.report.append(
                    f"[BOOLEAN] ✅ '{col}': normalised mixed boolean variants → {n_bool} canonical value(s) (Yes/No)"
                )
                self.fixes_applied.append(f"Normalised boolean column '{col}'")
                fixed += 1
                continue

            # Phase 3: LLM canonical mapping — bounded so a wide dataset
            # (many categorical columns) can't turn into dozens of
            # sequential network calls that stall the whole pipeline.
            current_non_null = self.df[col].dropna()
            raw_unique       = current_non_null.unique().tolist()

            llm_map  = {}
            llm_used = False
            can_use_llm = (
                self._llm_canonical_calls_used < _MAX_LLM_CANONICAL_CALLS
                and len(raw_unique) <= _MAX_LLM_CANONICAL_UNIQUE_VALS
            )
            if can_use_llm:
                self._llm_canonical_calls_used += 1
                llm_map = _llm_canonical_map(col, raw_unique)
            elif self._llm_canonical_calls_used >= _MAX_LLM_CANONICAL_CALLS:
                self.report.append(
                    f"[LLM-CLEAN] ⚠ '{col}': skipped LLM canonicalization — "
                    f"reached the {_MAX_LLM_CANONICAL_CALLS}-call budget for this run "
                    f"(deterministic Title Case + fuzzy merge still applied below)"
                )

            if llm_map:
                valid_llm_map = {
                    k: v for k, v in llm_map.items()
                    if k in set(raw_unique) and isinstance(v, str) and v.strip()
                }
                if valid_llm_map:
                    merges_llm = {k: v for k, v in valid_llm_map.items() if k != v}
                    self.df[col] = self.df[col].apply(
                        lambda v: valid_llm_map.get(v, v) if pd.notna(v) else v
                    )
                    after_unique = self.df[col].dropna().nunique()
                    self.report.append(
                        f"[LLM-CLEAN] ✅ '{col}': LLM resolved {len(merges_llm)} variant(s) "
                        f"({len(raw_unique)} → {after_unique} canonical categories)"
                    )
                    self.fixes_applied.append(
                        f"LLM-standardised '{col}': {len(merges_llm)} variant(s) resolved"
                    )
                    llm_used = True
                    fixed += 1

            # Phase 4: Title Case + fuzzy edit-distance dedup
            # PERFORMANCE CAP: fuzzy merge is O(n^2) in unique value count.
            # Above _MAX_FUZZY_UNIQUE_VALUES, skip it — Title Case still applies.
            post_non_null = self.df[col].dropna()
            title_cased   = post_non_null.apply(
                lambda v: str(v).title() if isinstance(v, str) else v
            )
            unique_tc = title_cased.unique().tolist()

            if len(unique_tc) > _MAX_FUZZY_UNIQUE_VALUES:
                self.report.append(
                    f"[CATEGORICAL] ⚠ '{col}': {len(unique_tc)} unique values exceeds "
                    f"fuzzy-merge performance threshold ({_MAX_FUZZY_UNIQUE_VALUES}) — "
                    f"Title Case applied, fuzzy near-duplicate merge skipped."
                )
                self.df[col] = self.df[col].apply(
                    lambda v: str(v).title() if isinstance(v, str) and pd.notna(v) else v
                )
                if changed_count > 0 or llm_used:
                    if not llm_used:
                        self.fixes_applied.append(f"Title Cased high-cardinality '{col}'")
                        fixed += 1
                continue

            canon_map    = _build_canonical_map(unique_tc, max_edit_dist=2)
            merges_fuzzy = {k: v for k, v in canon_map.items() if k != v}

            if merges_fuzzy:
                self.df[col] = self.df[col].apply(
                    lambda v: canon_map.get(
                        str(v).title() if isinstance(v, str) else v,
                        str(v).title() if isinstance(v, str) else v,
                    ) if pd.notna(v) else v
                )
                final_unique = self.df[col].dropna().nunique()
                self.report.append(
                    f"[CATEGORICAL] ✅ '{col}': Title Cased + fuzzy-merged "
                    f"{len(merges_fuzzy)} near-duplicate(s) → {final_unique} categories"
                )
                if not llm_used:
                    self.fixes_applied.append(f"Standardised categorical '{col}'")
                fixed += 1
            else:
                self.df[col] = self.df[col].apply(
                    lambda v: str(v).title() if isinstance(v, str) and pd.notna(v) else v
                )
                if changed_count > 0 or llm_used:
                    if not llm_used:
                        self.report.append(f"[CATEGORICAL] ✅ '{col}': Title Cased ({nunique} categories)")
                        self.fixes_applied.append(f"Standardised categorical '{col}'")
                        fixed += 1
                else:
                    self.report.append(f"[CATEGORICAL] ✔ '{col}': already clean")

        if fixed == 0:
            self.report.append(f"[CATEGORICAL] ✔ All text/categorical columns already clean")

    # ── NULL SWEEP: re-run after step7 canonicalization (VECTORIZED) ───────────
    def _sweep_nulls_again(self):
        """
        Second null-string -> NaN pass, run after step7.

        This is NOT guarding against Title-Case turning a case-variant into
        something that escapes NULL_STRINGS — step2 already does a
        case-insensitive check, so that specific failure mode does not
        exist in this pipeline.

        What this DOES guard against: step7 Phase 3's LLM canonical mapping
        can, in principle, map some dirty value to a placeholder-style
        label (e.g. "Unknown", "N/A") as its chosen "canonical" category.
        Because that string is introduced AFTER step2 already ran, it would
        otherwise sail into step9 as a real category never flagged as
        missing. The Phase 3 prompt now explicitly forbids the LLM from
        doing this, so ideally this sweep finds nothing — but relying on a
        prompt instruction alone is not a guarantee, hence this pass as a
        backstop.

        Rewritten to use vectorized pandas ops instead of a per-cell Python
        loop — same rationale as step2.
        """
        pre_missing = int(self.df.isnull().sum().sum())
        total = 0
        for col in self.df.columns:
            s = self.df[col]
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                continue  # numeric/datetime columns already get this treatment in step12
            str_s = s.astype(str).str.strip()
            mask = str_s.str.lower().isin(_NULL_STRINGS_LOWER) & s.notna()
            count = int(mask.sum())
            if count:
                self.df.loc[mask, col] = np.nan
            total += count
        if total > 0:
            post_missing = int(self.df.isnull().sum().sum())
            self.report.append(
                f"[NULL SWEEP 2] ✅ {total} additional null-string value(s) caught after "
                f"categorical standardisation → NaN (total missing: {pre_missing} → {post_missing})"
            )
            self.fixes_applied.append(f"Post-standardisation null sweep: {total} additional value(s) → NaN")
        else:
            self.report.append("[NULL SWEEP 2] ✔ No additional null strings found after categorical standardisation")

    # ── STEP 8: Fix corrupted IDs ──────────────────────────────────────────────
    def step8_fix_corrupted_ids(self):
        fixed = 0
        id_like_keywords = (
            ['account', 'phone', 'mobile', 'zip', 'pin', 'contact']
            + _ID_LIKE_EXTRA_KEYWORDS
        )
        for col in self.df.columns:
            if not any(k in col.lower() for k in id_like_keywords):
                continue
            # SKU/barcode/tracking-number style identifiers legitimately
            # mix letters and digits (e.g. "SKU-A1023", "AWB1Z9992"), so the
            # "strip anything non-numeric" rule below only applies to
            # purely-numeric identifiers (accounts, phones, zips, cards).
            if any(k in col.lower() for k in ['sku', 'upc', 'ean', 'barcode',
                                               'iban', 'gstin', 'pan',
                                               'order_number', 'invoice_number',
                                               'tracking_number', 'awb']):
                continue
            if self.df[col].dtype != object:
                continue
            mask  = self.df[col].astype(str).str.contains(r'[^0-9]', regex=True, na=False)
            count = mask.sum()
            if count > 0:
                self.df.loc[mask, col] = np.nan
                self.report.append(f"[CORRUPTED] ✅ '{col}': {count} corrupted values → NaN")
                self.issues_found.append(f"{count} corrupted values in '{col}'")
                self.fixes_applied.append(f"Nullified {count} corrupted in '{col}'")
                fixed += 1
        if fixed == 0:
            self.report.append(f"[CORRUPTED] ✔ No corrupted ID/numeric fields found")

    # ── STEP 9: RF Imputation (PERFORMANCE CAP on training rows) ──────────────
    def step9_rf_impute_missing(self):
        missing_cols = [c for c in self.df.columns if self.df[c].isnull().sum() > 0]
        if not missing_cols:
            self.report.append(f"[IMPUTE] ✔ No missing values to impute")
            return

        total_missing = self.df.isnull().sum().sum()
        self.report.append(
            f"[IMPUTE] 🤖 RF Imputation starting — {total_missing} missing values "
            f"across {len(missing_cols)} column(s)"
        )

        # Encode ONCE — this call also populates self._encoders, which is now
        # the authoritative record of which columns are categorical.
        df_enc = self._encode_for_rf(self.df)

        for col in missing_cols:
            missing_count = self.df[col].isnull().sum()
            missing_pct   = round(missing_count / len(self.df) * 100, 1)

            if missing_pct > 70:
                self._simple_fill(col, missing_count, missing_pct, reason=">70% missing")
                self._imputed_cols.add(col)
                continue

            feature_cols = [
                c for c in df_enc.columns
                if c != col and df_enc[c].isnull().sum() / len(df_enc) < 0.5
            ]
            if not feature_cols:
                self._simple_fill(col, missing_count, missing_pct, reason="no usable features")
                self._imputed_cols.add(col)
                continue

            known_mask   = df_enc[col].notna()
            unknown_mask = df_enc[col].isna()

            if known_mask.sum() < 5:
                self._simple_fill(col, missing_count, missing_pct, reason="insufficient training rows")
                self._imputed_cols.add(col)
                continue

            X_train = df_enc.loc[known_mask,   feature_cols].fillna(0).values
            y_train = df_enc.loc[known_mask,   col].values
            X_pred  = df_enc.loc[unknown_mask, feature_cols].fillna(0).values

            # PERFORMANCE CAP — sample training rows on large datasets.
            sampled_note = ""
            if X_train.shape[0] > _MAX_RF_TRAIN_ROWS:
                rng = np.random.RandomState(42)
                idx = rng.choice(X_train.shape[0], _MAX_RF_TRAIN_ROWS, replace=False)
                X_train, y_train = X_train[idx], y_train[idx]
                sampled_note = f" (sampled {_MAX_RF_TRAIN_ROWS:,} of {known_mask.sum():,} rows for speed)"

            # ── FIX: categorical-ness now comes from the encoding step itself
            # (col in self._encoders), not a second, independently-computed
            # dtype check that could silently disagree with it. A numeric
            # low-cardinality column (e.g. a 1-5 star rating) is still
            # correctly treated as categorical too. ─────────────────────────
            is_cat = (
                col in self._encoders
                or (pd.api.types.is_numeric_dtype(self.df[col]) and self.df[col].nunique() <= 10)
            )

            # WIDE-DATASET GUARD: n_estimators=150 per column is fine for a
            # couple of columns, but a dataset with many missing columns
            # fits one full forest PER column, sequentially — that stacks
            # up fast and is a common cause of the pipeline just running
            # for minutes with nothing visibly wrong. After the first
            # _RF_CHEAP_MODE_AFTER_N_COLUMNS columns, later columns use a
            # smaller forest so total imputation time stays bounded
            # regardless of how many columns need it.
            n_estimators = (
                _RF_ESTIMATORS_NORMAL
                if len(self._imputed_cols) < _RF_CHEAP_MODE_AFTER_N_COLUMNS
                else _RF_ESTIMATORS_CHEAP
            )
            try:
                model = (
                    RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42, n_jobs=-1)
                    if is_cat else
                    RandomForestRegressor(n_estimators=n_estimators, max_depth=10, random_state=42, n_jobs=-1)
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_pred)

                if col in self._encoders:
                    le      = self._encoders[col]
                    indices = np.clip(np.round(preds).astype(int), 0, len(le.classes_) - 1)
                    preds   = le.classes_[indices]

                self.df.loc[unknown_mask, col] = preds
                mtype = "Classifier" if is_cat else "Regressor"
                self.report.append(
                    f"[IMPUTE] ✅ '{col}': {missing_count} missing ({missing_pct}%) → "
                    f"RF {mtype} predicted (trained on {X_train.shape[0]:,} rows{sampled_note})"
                )
                self.fixes_applied.append(f"RF predicted {missing_count} missing in '{col}'")
                self._imputed_cols.add(col)

            except Exception as e:
                self._simple_fill(col, missing_count, missing_pct, reason=f"RF error: {str(e)[:40]}")
                self._imputed_cols.add(col)

        remaining = self.df.isnull().sum().sum()
        self.report.append(f"[IMPUTE] 📊 Before: {total_missing} missing → After: {remaining} missing")
        if remaining == 0:
            self.report.append(f"[IMPUTE] ✅ All missing values imputed successfully!")

    def _encode_for_rf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes non-numeric columns via LabelEncoder for RF feature use.
        self._encoders now doubles as the SINGLE SOURCE OF TRUTH for
        "this column is categorical" — step9 reads membership in this
        dict instead of re-checking dtype independently, which is what
        previously let a Regressor get fit on label-encoded categorical
        targets (e.g. order_status, payment_method) whenever a column's
        dtype wasn't exactly numpy `object` (pandas nullable "string"
        dtype, or any dtype introduced by an intermediate astype call,
        would silently fail a `dtype == object` check even though it was
        genuinely categorical and got label-encoded here).
        """
        df_num = pd.DataFrame(index=df.index)
        self._encoders = {}  # reset per call — avoids stale entries across runs
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_num[col] = df[col].astype(float)
            else:
                non_null = df[col].dropna().astype(str)
                if len(non_null) == 0:
                    df_num[col] = np.nan
                    continue
                le = LabelEncoder()
                le.fit(non_null)
                self._encoders[col] = le
                encoded = np.full(len(df), np.nan, dtype=float)
                mask = df[col].notna()
                if mask.sum() > 0:
                    encoded[mask.values] = le.transform(df[col][mask].astype(str)).astype(float)
                df_num[col] = encoded
        return df_num

    def _simple_fill(self, col: str, missing_count: int, missing_pct: float, reason: str = ""):
        if pd.api.types.is_numeric_dtype(self.df[col]):
            fill_val = self.df[col].median()
            # Nullable pandas integer dtypes (Int64/Int32/...) reject a
            # non-integer fill value outright (e.g. a median of 1.5 for a
            # ratings column) — this previously crashed the whole cleaning
            # pipeline. Widen to float64 whenever the median isn't a whole
            # number, so the fill always succeeds regardless of dtype.
            if pd.api.types.is_extension_array_dtype(self.df[col].dtype) and fill_val % 1 != 0:
                self.df[col] = self.df[col].astype("float64")
            self.df[col] = self.df[col].fillna(fill_val)
            self.report.append(
                f"[IMPUTE] ⚠ '{col}': {missing_count} ({missing_pct}%) → median ({fill_val:,.2f}) [{reason}]"
            )
        elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
            self.df[col] = self.df[col].ffill().bfill()
            self.report.append(
                f"[IMPUTE] ⚠ '{col}': {missing_count} ({missing_pct}%) → forward/back filled [{reason}]"
            )
        else:
            mode_vals = self.df[col].mode()
            fill_val  = mode_vals[0] if not mode_vals.empty else "UNKNOWN"
            self.df[col] = self.df[col].fillna(fill_val)
            self.report.append(
                f"[IMPUTE] ⚠ '{col}': {missing_count} ({missing_pct}%) → '{fill_val}' [{reason}]"
            )

    # ── STEP 9b: Post-imputation domain-aware clipping ─────────────────────────
    def step9b_post_imputation_clip(self):
        """
        After RF imputation, clamp columns whose values must stay within
        known physical/business bounds. This prevents RF producing
        impossible values like negative order amounts or ratings of 8.5.

        Rules (generalized for any dataset):
          - Rating/score columns   → clip to [1.0, 5.0]
          - Percentage columns     → clip to [0.0, 100.0]
          - Non-negative keywords  → floor at 0.0
        """
        clipped_any = False
        for col in self.df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            clipped   = False

            # Rating columns
            if any(k in col_lower for k in _RATING_KEYWORDS):
                n_bad = int(((self.df[col] < _RATING_MIN) | (self.df[col] > _RATING_MAX)).sum())
                if n_bad > 0:
                    self.df[col] = self.df[col].clip(_RATING_MIN, _RATING_MAX)
                    self.report.append(
                        f"[CLIP] ✅ '{col}': {n_bad} out-of-range value(s) clipped to [{_RATING_MIN}, {_RATING_MAX}]"
                    )
                    self.fixes_applied.append(f"Clipped '{col}' to [{_RATING_MIN}, {_RATING_MAX}]")
                    clipped = True

            # Percentage columns
            elif any(k in col_lower for k in _PERCENT_KEYWORDS):
                n_bad = int(((self.df[col] < _PERCENT_MIN) | (self.df[col] > _PERCENT_MAX)).sum())
                if n_bad > 0:
                    self.df[col] = self.df[col].clip(_PERCENT_MIN, _PERCENT_MAX)
                    self.report.append(
                        f"[CLIP] ✅ '{col}': {n_bad} out-of-range value(s) clipped to [{_PERCENT_MIN}%, {_PERCENT_MAX}%]"
                    )
                    self.fixes_applied.append(f"Clipped '{col}' to [0%, 100%]")
                    clipped = True

            # Non-negative columns — skip entirely if the column name matches
            # a signed-allowed keyword (balance, margin, refund, etc.), since
            # negative values there are legitimate business signal, not
            # data-quality errors (e.g. an overdrawn account, a loss-making
            # product, a refund line item).
            elif any(k in col_lower for k in _SIGNED_ALLOWED_KEYWORDS):
                pass

            elif any(k in col_lower for k in _NON_NEGATIVE_KEYWORDS):
                n_neg = int((self.df[col] < 0).sum())
                if n_neg > 0:
                    self.df[col] = self.df[col].clip(lower=0)
                    self.report.append(
                        f"[CLIP] ✅ '{col}': {n_neg} negative value(s) floored to 0 "
                        f"(physically impossible for this column)"
                    )
                    self.fixes_applied.append(f"Floored {n_neg} negatives in '{col}' to 0")
                    clipped = True

            if clipped:
                clipped_any = True

        if not clipped_any:
            self.report.append("[CLIP] ✔ No domain-bound violations found after imputation")

    # ── STEP 10: Cap outliers ──────────────────────────────────────────────────
    def step10_cap_outliers(self):
        """
        IQR-based outlier capping. For columns with domain-known minimums
        (non-negative), the lower cap is max(IQR_lower, 0) so we never
        introduce new negative values through the outlier cap.
        """
        capped_any = False
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if len(self.df[col].dropna().unique()) <= 5:
                continue

            col_lower = col.lower()
            Q1  = self.df[col].quantile(0.25)
            Q3  = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue

            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR

            # For non-negative columns, never let the lower cap go below 0 —
            # UNLESS the column is a signed-allowed metric (balance, margin,
            # refund, credit, etc.), where a negative value is legitimate
            # business signal, not an outlier artifact.
            is_signed_allowed = any(k in col_lower for k in _SIGNED_ALLOWED_KEYWORDS)
            is_non_negative = (not is_signed_allowed) and (
                any(k in col_lower for k in _NON_NEGATIVE_KEYWORDS) or
                any(k in col_lower for k in _RATING_KEYWORDS) or
                any(k in col_lower for k in _PERCENT_KEYWORDS)
            )
            if is_non_negative:
                lower = max(lower, 0.0)

            n_capped = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())
            if n_capped > 0:
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                self.report.append(
                    f"[OUTLIERS] ✅ '{col}': {n_capped} outliers capped to [{lower:,.2f} — {upper:,.2f}]"
                )
                self.issues_found.append(f"{n_capped} outliers in '{col}'")
                self.fixes_applied.append(f"Capped {n_capped} outliers in '{col}'")
                capped_any = True
            else:
                self.report.append(f"[OUTLIERS] ✔ '{col}': no outliers detected")

        if not capped_any:
            self.report.append(f"[OUTLIERS] ✔ All numeric columns within normal bounds")

    # ── STEP 11: Numeric summary ───────────────────────────────────────────────
    def step11_numeric_summary(self):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            s = self.df[col]
            self.report.append(
                f"[SUMMARY] '{col}': mean={s.mean():,.2f} | std={s.std():,.2f} | "
                f"min={s.min():,.2f} | median={s.median():,.2f} | max={s.max():,.2f}"
            )

    # ── STEP 12: Final dtype self-audit ─────────────────────────────────────────
    def step12_final_dtype_audit(self):
        """
        Last-line safety pass: scans every remaining object column and
        force-converts any that are still actually numeric under the hood.
        Makes the pipeline self-correcting regardless of which earlier
        step should have caught it — works identically on any dataset.
        """
        caught = 0
        for col in self.df.columns:
            if self.df[col].dtype != object:
                continue
            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                continue
            probe = _clean_numeric_string(non_null)
            if probe.notna().sum() / len(non_null) > 0.5:
                full = _clean_numeric_string(self.df[col])
                vals = full.dropna()
                is_whole = bool((vals % 1 == 0).all()) if len(vals) else True
                self.df[col] = full.round().astype("Int64") if is_whole else full.astype("float64")
                self.report.append(
                    f"[DTYPE-AUDIT] ✅ '{col}': force-converted to "
                    f"{'int' if is_whole else 'float'} on final pass"
                )
                self.fixes_applied.append(f"Final-pass numeric conversion for '{col}'")
                caught += 1
        if caught == 0:
            self.report.append("[DTYPE-AUDIT] ✔ Final audit: no numeric-as-text columns remain")

    # ── Final verdict ──────────────────────────────────────────────────────────
    def _generate_verdict(self):
        if not self.issues_found and not self.fixes_applied:
            verdict = (
                "✅ This dataset needs no cleaning and is good to go for analysis! "
                "All columns are properly named, no null values, no duplicates, "
                "no outliers, and all data types are correct."
            )
        else:
            verdict = (
                f"🔧 Found {len(self.issues_found)} issue(s) — all fixed automatically:\n" +
                "\n".join(f"  • {f}" for f in self.fixes_applied)
            )
        self.report.append(f"[VERDICT] {verdict}")
        return verdict

    # ── Near-duplicate ROW detection (distinct from step7's fuzzy CELL merge) ──
    def _detect_near_duplicate_rows(self, max_report=50):
        """
        Finds rows that are highly similar but NOT identical — e.g. a
        re-keyed order with one typo'd field, or a customer entered twice
        with a slightly different spelling. These are intentionally never
        auto-removed (collapsing two genuinely distinct records is worse
        than leaving a near-duplicate in place) — this is purely a
        surfaced-for-review signal.

        Approach: build a normalized text signature per row from its
        text/categorical columns, then reuse the same bounded edit-distance
        clustering as step7's fuzzy merge (capped by
        _MAX_FUZZY_UNIQUE_VALUES for performance on large/high-cardinality
        datasets).
        """
        self._near_duplicate_rows = []
        if len(self.df) < 2:
            return

        text_cols = [
            c for c in self.df.columns
            if self.df[c].dtype == object and not c.endswith("_month_name")
        ][:6]
        if not text_cols:
            return

        work_df = self.df
        if len(work_df) > 5000:
            work_df = work_df.sample(5000, random_state=42)

        signatures = (
            work_df[text_cols].fillna("").astype(str)
            .agg(" ".join, axis=1)
            .apply(_normalize_text)
        )

        unique_sigs = signatures.unique().tolist()
        if len(unique_sigs) > _MAX_FUZZY_UNIQUE_VALUES:
            return  # too expensive to compare pairwise — skip, not worth blocking cleaning

        # Cluster similar (but non-identical) signatures using the same
        # relative edit-distance rule as _build_canonical_map.
        clusters: dict = {}
        seen_norms = []
        for sig in unique_sigs:
            if not sig:
                continue
            matched_to = None
            for known in seen_norms:
                if known == sig:
                    continue
                dist = _edit_distance(sig, known)
                max_len = max(len(sig), len(known))
                if max_len == 0:
                    continue
                relative_threshold = 0.20 if max_len <= 6 else 0.15
                if dist > 0 and dist <= 3 and (dist / max_len) <= relative_threshold:
                    matched_to = known
                    break
            if matched_to:
                clusters.setdefault(matched_to, [matched_to]).append(sig)
            else:
                seen_norms.append(sig)

        preview_cols = text_cols + [
            c for c in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[c])
        ][:3]

        for canonical_sig, group_sigs in clusters.items():
            matching_rows = work_df[signatures.isin(group_sigs)]
            if len(matching_rows) < 2:
                continue
            for _, row in matching_rows.head(max_report).iterrows():
                record = row[preview_cols].to_dict()
                record["_row_index"] = row.name
                self._near_duplicate_rows.append(record)
            if len(self._near_duplicate_rows) >= max_report:
                break

    def get_near_duplicates(self) -> list:
        """Returns the near-duplicate row records found during clean()."""
        return self._near_duplicate_rows

    # ── Column profile / classification accessors (for domain_detector.py) ────
    def get_profile(self) -> dict:
        """
        Per-column profile of the CLEANED dataframe — consumed by
        DomainDetector as evidence for its domain-guess reasoning.
        Dataset-agnostic: works identically on retail, e-commerce,
        banking, or any other domain's columns.
        """
        profile = {}
        for col in self.df.columns:
            s = self.df[col]
            non_null = s.dropna()
            profile[col] = {
                "dtype": str(s.dtype),
                "n_unique": int(non_null.nunique()) if len(non_null) else 0,
                "missing_pct": round(float(s.isnull().mean() * 100), 1),
                "sample_values": non_null.astype(str).unique()[:5].tolist(),
            }
        return profile

    def get_classification(self) -> dict:
        """
        Per-column role classification — "id" / "numeric" / "datetime" /
        "categorical" / "text". Prefers the LLM's dtype call from step6
        (self._llm_dtypes) when available, falls back to dtype-based
        heuristics otherwise, so this never returns empty even if the LLM
        was unavailable during cleaning.
        """
        classification = {}
        for col in self.df.columns:
            s = self.df[col]
            llm_type = self._llm_dtypes.get(col, "").lower() if self._llm_dtypes else ""
            if llm_type in ("id", "numeric", "datetime", "categorical", "text"):
                classification[col] = llm_type
                continue

            col_lower = col.lower()
            if col_lower.endswith("_id") or col_lower == "id" or \
               any(k in col_lower for k in _ID_LIKE_EXTRA_KEYWORDS):
                classification[col] = "id"
            elif pd.api.types.is_datetime64_any_dtype(s):
                classification[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(s):
                classification[col] = "numeric"
            elif s.dropna().nunique() <= 60:
                classification[col] = "categorical"
            else:
                classification[col] = "text"
        return classification

    # ── Data-health scorecard (for the Data Cleaner page's UI metrics) ─────────
    def get_scorecard(self) -> dict:
        """
        Computes 4 dimensions of data quality, BEFORE (raw upload) vs
        AFTER (cleaned) — completeness, uniqueness, validity, consistency —
        plus an overall letter grade, so SMB users get an at-a-glance
        health readout without needing to read the full text log.
        """

        def _completeness(df):
            total_cells = df.shape[0] * df.shape[1] if df.shape[1] else 1
            return round(float(100 - (df.isnull().sum().sum() / total_cells * 100)), 1)

        def _uniqueness(df):
            if len(df) == 0:
                return 100.0
            dupes = int(df.duplicated().sum())
            return round(100 - (dupes / len(df) * 100), 1)

        def _validity(df):
            # % of object-column cells that are NOT a recognized null-marker
            # string (garbage tokens like "ERROR", "N/A", "#DIV/0!", etc.) —
            # a proxy for how much of the raw text data was well-formed.
            obj_cols = df.select_dtypes(include="object").columns
            if len(obj_cols) == 0:
                return 100.0
            total = invalid = 0
            for col in obj_cols:
                s = df[col].dropna().astype(str).str.strip()
                total += len(s)
                invalid += int(s.str.lower().isin(_NULL_STRINGS_LOWER).sum())
            if total == 0:
                return 100.0
            return round(100 - (invalid / total * 100), 1)

        def _consistency(df):
            # % of categorical columns (<=60 uniques) whose values are
            # already Title Case + free of stray whitespace/special chars —
            # a proxy for formatting consistency across categories.
            cat_cols = [
                c for c in df.select_dtypes(include="object").columns
                if df[c].dropna().nunique() <= 60
            ]
            if not cat_cols:
                return 100.0
            consistent = 0
            for col in cat_cols:
                vals = df[col].dropna().astype(str)
                if len(vals) == 0:
                    consistent += 1
                    continue
                well_formed = vals.apply(
                    lambda v: v == v.title() and v == v.strip() and not re.search(r'\s{2,}', v)
                )
                if well_formed.mean() >= 0.9:
                    consistent += 1
            return round(consistent / len(cat_cols) * 100, 1)

        def _grade(avg_score):
            if avg_score >= 95: return "A"
            if avg_score >= 85: return "B"
            if avg_score >= 70: return "C"
            if avg_score >= 50: return "D"
            return "F"

        def _score_block(df):
            completeness = _completeness(df)
            uniqueness   = _uniqueness(df)
            validity     = _validity(df)
            consistency  = _consistency(df)
            avg = (completeness + uniqueness + validity + consistency) / 4
            return {
                "completeness_pct": completeness,
                "uniqueness_pct":   uniqueness,
                "validity_pct":     validity,
                "consistency_pct":  consistency,
                "grade":            _grade(avg),
                "_avg":             avg,
            }

        before = _score_block(self._raw_df)
        after  = _score_block(self.df)
        delta_points = round(float(after["_avg"] - before["_avg"]), 1)

        before.pop("_avg"); after.pop("_avg")

        return {
            "before": before,
            "after": after,
            "delta_points": delta_points,
        }

    # ── Row-retention accessor (for crew.py's independent tripwire) ────────────
    def get_row_retention(self) -> dict:
        """
        Returns the row counts around the dedup step specifically (not just
        overall before/after clean), so callers can distinguish "rows lost
        to dedup" from "rows lost to other steps" if ever needed.
        """
        return {
            "rows_before_dedup": self.rows_before_dedup,
            "rows_after_dedup": self.rows_after_dedup,
            "original_shape": self.original_shape,
        }

    # ── Main clean() ───────────────────────────────────────────────────────────
    def clean(self, columns_to_drop=None):
        self.report.append(
            f"[START] Analyzing: {self.original_shape[0]:,} rows x {self.original_shape[1]} columns"
        )
        self.report.append(f"[START] Columns: {', '.join(self.df.columns.tolist())}")
        self.report.append("─" * 60)

        self.step0_drop_columns(columns_to_drop or [])
        self.step1_standardize_columns()
        self.step2_convert_nulls_to_nan()
        self.step3_check_missing()
        self.step4_remove_duplicates()
        self._detect_near_duplicate_rows()
        self.step5_fix_currency()
        self.step6_fix_data_types()
        self.step6b_split_dates()
        self.step7_standardize_categoricals()
        self._sweep_nulls_again()
        self.step12_final_dtype_audit()
        self.step8_fix_corrupted_ids()
        self.step9_rf_impute_missing()
        self.step9b_post_imputation_clip()   
        self.step10_cap_outliers()
        self.step11_numeric_summary()
        self.report.append("─" * 60)
        self._generate_verdict()

        self.report.append(
            f"[END] Final shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns"
        )
        self.report.append(f"[END] Missing values remaining: {self.df.isnull().sum().sum()}")
        self.report.append(
            f"[END] Cleaned at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        return self.df, self.report