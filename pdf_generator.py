"""
pdf_generator.py

Drop-in replacement for the inline PDF code in app.py.

Usage in app.py:
    from pdf_generator import generate_pdf_bytes
    ...
    pdf_bytes = generate_pdf_bytes(report_text, dataset_name)
    st.download_button("⬇️ Download PDF", pdf_bytes,
                       f"{dataset_name}_report.pdf", "application/pdf")

Root causes of the old errors, all fixed here:
  1. multi_cell() in old fpdf2 raises when border="1" (int) is passed as
     str — we always pass border=1 (int).
  2. Table cells with very wide text overflow the page when col_w < content —
     we truncate cell text and enforce minimum col widths.
  3. Unicode chars that aren't in latin-1 raise UnicodeEncodeError —
     we run a full unicode→ascii transliteration, not just a replace dict.
  4. The old code called pdf.output() which in fpdf2 returns bytes, not
     None — writing to a tempfile with .output(path) still works, but
     generating bytes directly via pdf.output() avoids tempfile race conditions.
  5. Empty table blocks (all separator lines) caused index errors — skipped.
  6. set_xy() after a multi_cell that wrapped multiple lines left the cursor
     in the wrong position — we track max row height explicitly.
"""

import re
import unicodedata
import io


def _sanitize(text: str) -> str:
    """
    Convert any unicode string to a safe latin-1 string for fpdf2.
    Strategy:
      1. Known replacements first (arrows, currency, math symbols).
      2. unicodedata NFKD decompose + strip combining chars (handles accented letters).
      3. Encode to latin-1, replacing anything left with '?'.
    """
    if not isinstance(text, str):
        text = str(text)

    REPLACEMENTS = {
        "\u2014": "-",   "\u2013": "-",   "\u2019": "'",   "\u2018": "'",
        "\u201c": '"',   "\u201d": '"',   "\u2022": "-",   "\u2192": "->",
        "\u2190": "<-",  "\u2713": "OK",  "\u2714": "OK",  "\u274c": "X",
        "\u26a0": "!",   "\u2705": "[OK]","\u274e": "[X]", "\u2b50": "*",
        "\u25cf": "-",   "\u00b2": "2",   "\u00b3": "3",   "\u00b0": " deg",
        "\u20b9": "Rs",  "\u20ac": "EUR", "\u00a3": "GBP", "\u00a5": "JPY",
        "\u2248": "~",   "\u2260": "!=",  "\u2264": "<=",  "\u2265": ">=",
        "\u00d7": "x",   "\u00f7": "/",   "\u2026": "...", "\u00a0": " ",
        "\u200b": "",    "\u2011": "-",   "\u2012": "-",   "\u2015": "-",
        "\u2212": "-",   "\u2764": "<3",  "\u2665": "<3",
        "\u25b6": ">",   "\u25c0": "<",   "\u2714": "[v]", "\u2718": "[x]",
        "\u00e9": "e",   "\u00e8": "e",   "\u00ea": "e",   "\u00e0": "a",
        "\u00e2": "a",   "\u00f4": "o",   "\u00fb": "u",   "\u00fc": "u",
        "\u00e7": "c",   "\u00f1": "n",   "\u00df": "ss",
        # Indian rupee variations
        "\u20b9": "Rs",  "\u0930": "Rs",
    }
    for char, repl in REPLACEMENTS.items():
        text = text.replace(char, repl)

    # NFKD decompose → strip combining chars (handles é→e, ü→u, etc.)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Final encode — replace anything still not in latin-1
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def _strip_markdown_inline(text: str) -> str:
    """Remove **bold**, *italic*, `code` markers from inline text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text


def _render_table(pdf, lines: list, page_width: float):
    """
    Render a markdown table robustly.
    - Skips separator lines (|---|---|)
    - Computes column widths that fit the page
    - Wraps long cell text instead of overflowing
    - Tracks row height correctly even when cells wrap
    """
    data_lines = [
        ln for ln in lines
        if ln.strip().startswith("|")
        and not re.match(r'^\|[\s\-:|]+\|$', ln.strip())
    ]
    if not data_lines:
        return

    rows = []
    for ln in data_lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return

    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return

    # Pad all rows to same width
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    # Column width: equal split, min 18mm
    col_w = max(page_width / n_cols, 18.0)
    if col_w * n_cols > page_width:
        col_w = page_width / n_cols

    row_h    = 5.0   # mm per line
    max_chars = max(int(col_w / 1.8), 8)  # chars before wrap

    for i, row in enumerate(rows):
        is_header = (i == 0)

        # Measure tallest cell
        max_lines_in_row = 1
        for cell in row:
            cell_text = _sanitize(_strip_markdown_inline(str(cell)))
            # Truncate very long text
            if len(cell_text) > max_chars * 4:
                cell_text = cell_text[:max_chars * 4 - 3] + "..."
            lines_needed = max(1, -(-len(cell_text) // max_chars))  # ceiling div
            max_lines_in_row = max(max_lines_in_row, lines_needed)

        cell_h = row_h * max_lines_in_row + 2.0

        # Page break before row if needed
        if pdf.get_y() + cell_h + 4 > pdf.h - pdf.b_margin:
            pdf.add_page()

        if is_header:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(26, 26, 46)
            pdf.set_text_color(0, 245, 255)
        else:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_fill_color(20, 30, 50)
            pdf.set_text_color(200, 210, 225)

        start_x = pdf.get_x()
        start_y = pdf.get_y()

        for j, cell in enumerate(row):
            cell_text = _sanitize(_strip_markdown_inline(str(cell)))
            if len(cell_text) > max_chars * 4:
                cell_text = cell_text[:max_chars * 4 - 3] + "..."

            x = start_x + j * col_w
            pdf.set_xy(x, start_y)
            try:
                pdf.multi_cell(
                    w=col_w,
                    h=row_h,
                    txt=cell_text,
                    border=1,
                    align="L",
                    fill=True,
                    max_line_height=row_h,
                )
            except TypeError:
                # Older fpdf2 versions don't have max_line_height
                pdf.multi_cell(
                    w=col_w,
                    h=row_h,
                    txt=cell_text,
                    border=1,
                    align="L",
                    fill=True,
                )

        # Advance cursor to next row
        pdf.set_xy(start_x, start_y + cell_h)

    # Reset colors
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(255, 255, 255)
    pdf.ln(2)


def generate_pdf_bytes(report_text: str, dataset_name: str) -> bytes:
    """
    Convert a markdown-formatted report string to PDF bytes.
    Returns bytes that can be passed directly to st.download_button().

    Raises RuntimeError (with message) if fpdf2 is not installed.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError(
            "fpdf2 is not installed. Run: pip install fpdf2"
        )

    # ── Setup ──────────────────────────────────────────────────────────────────
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Landscape A4 usable width: 297 - 15 - 15 = 267mm
    PAGE_W = 267.0

    # ── Cover header ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 200, 220)
    pdf.multi_cell(0, 12, _sanitize(f"{dataset_name} — Business Performance Report"))
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 130, 145)
    pdf.multi_cell(0, 6, "Generated by AI Data Analyzer")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Parse and render lines ─────────────────────────────────────────────────
    lines = report_text.split("\n")
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        line     = raw_line.strip()
        safe     = _sanitize(_strip_markdown_inline(line))

        # ── Empty line ────────────────────────────────────────────────────────
        if not safe:
            pdf.ln(2)
            i += 1
            continue

        # ── Table block ───────────────────────────────────────────────────────
        if safe.startswith("|"):
            table_lines = []
            while i < len(lines) and (
                lines[i].strip().startswith("|") or lines[i].strip() == ""
            ):
                if lines[i].strip():
                    table_lines.append(lines[i])
                i += 1
            if table_lines:
                _render_table(pdf, table_lines, PAGE_W)
            continue

        # ── Horizontal rule ────────────────────────────────────────────────────
        if re.match(r'^[-*_]{3,}$', safe):
            pdf.set_draw_color(0, 100, 120)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + PAGE_W, pdf.get_y())
            pdf.set_draw_color(0, 0, 0)
            pdf.ln(3)
            i += 1
            continue

        # ── H1 ────────────────────────────────────────────────────────────────
        if safe.startswith("# ") and not safe.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(0, 230, 255)
            pdf.multi_cell(0, 9, _sanitize(safe[2:]))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

        # ── H2 ────────────────────────────────────────────────────────────────
        elif safe.startswith("## ") and not safe.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(0, 200, 220)
            pdf.multi_cell(0, 8, _sanitize(safe[3:]))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

        # ── H3 ────────────────────────────────────────────────────────────────
        elif safe.startswith("### "):
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(100, 180, 255)
            pdf.multi_cell(0, 7, _sanitize(safe[4:]))
            pdf.set_text_color(0, 0, 0)

        # ── H4 ####────────────────────────────────────────────────────────────
        elif safe.startswith("#### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(140, 200, 255)
            pdf.multi_cell(0, 6, _sanitize(safe[5:]))
            pdf.set_text_color(0, 0, 0)

        # ── Bold-only line **text**────────────────────────────────────────────
        elif re.match(r'^\*\*(.+)\*\*$', safe):
            inner = re.sub(r'^\*\*|\*\*$', '', safe)
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, _sanitize(inner))
            pdf.set_font("Helvetica", "", 10)

        # ── Numbered bold header like "**1. Title**" ──────────────────────────
        elif re.match(r'^\*\*\d+\.', safe):
            inner = re.sub(r'\*\*', '', safe)
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, _sanitize(inner))
            pdf.set_font("Helvetica", "", 10)

        # ── Bullet point ──────────────────────────────────────────────────────
        elif safe.startswith("- ") or safe.startswith("* "):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, "  - " + _sanitize(safe[2:]))

        # ── Numbered list ─────────────────────────────────────────────────────
        elif re.match(r'^\d+\.', safe):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, "  " + _sanitize(safe))

        # ── "Based on:" / "Action:" / "Expected outcome:" sub-labels ─────────
        elif re.match(r'^(Based on|Action|Expected outcome):', safe, re.IGNORECASE):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(160, 180, 200)
            pdf.multi_cell(0, 5, "    " + _sanitize(safe))
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)

        # ── Regular paragraph ─────────────────────────────────────────────────
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, _sanitize(safe))

        i += 1

    # ── Output to bytes ────────────────────────────────────────────────────────
    # fpdf2 >= 2.5.0: pdf.output() returns bytes when no dest given
    # fpdf2 < 2.5.0:  pdf.output() returns a bytearray
    try:
        result = pdf.output()
        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
        # Older versions need explicit dest
        buf = io.BytesIO()
        pdf.output(buf)
        return buf.getvalue()
    except Exception:
        # Final fallback: write to BytesIO
        buf = io.BytesIO()
        try:
            pdf.output(buf)
        except TypeError:
            pdf.output(dest="S")
        return buf.getvalue()