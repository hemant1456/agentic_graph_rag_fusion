"""
CSV parser for Step 04 — format-aware chunking.

Produces two kinds of SmartChunk per CSV file:
  1. "row"       — one chunk per data row (same as baseline)
  2. "aggregate" — one summary chunk with computed totals, breakdowns,
                   date-period aggregations, per-row ratios, etc.

The aggregate chunk is built by a general algorithm that inspects each
column's data type at runtime — no values are hardcoded.
"""

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from step_04_chunking.implementation.types import SmartChunk


# ── Type detection helpers ────────────────────────────────────────────────────

def _to_numeric(val: str) -> float | None:
    """Try to parse a value as a number, stripping currency symbols and commas."""
    cleaned = val.strip().replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def _to_date(val: str) -> datetime | None:
    """Try to parse a value as YYYY-MM or YYYY-MM-DD ISO date."""
    val = val.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            pass
    return None


def _classify_columns(headers: list[str], rows: list[dict[str, str]]) -> dict[str, str]:
    """
    Returns a dict mapping column_name → type:
      "numeric"     — all non-empty values parse as numbers
      "date"        — all non-empty values parse as ISO dates
      "categorical" — string column with cardinality 2–15
      "name"        — single column, all-unique values, <30 rows
      "string"      — other string column
    """
    col_types: dict[str, str] = {}
    n = len(rows)

    for col in headers:
        values = [r[col].strip() for r in rows if r.get(col, "").strip()]
        if not values:
            col_types[col] = "string"
            continue

        # Check numeric
        numeric_vals = [_to_numeric(v) for v in values]
        if all(x is not None for x in numeric_vals):
            col_types[col] = "numeric"
            continue

        # Check date
        date_vals = [_to_date(v) for v in values]
        if all(x is not None for x in date_vals):
            col_types[col] = "date"
            continue

        # Categorical: cardinality 2–15
        unique_vals = set(values)
        if 2 <= len(unique_vals) <= 15:
            col_types[col] = "categorical"
            continue

        # Name column: single column with all-unique values and <30 rows
        if len(unique_vals) == len(values) and n < 30:
            col_types[col] = "name"
            continue

        col_types[col] = "string"

    return col_types


# ── Number formatting ─────────────────────────────────────────────────────────

def _fmt_number(val: float) -> str:
    """Format as integer if whole, else 2 decimal places. Add thousand separators."""
    if val == int(val):
        return f"{int(val):,}"
    return f"{val:,.2f}"


# ── Aggregate text builder ────────────────────────────────────────────────────

def _build_aggregate_text(
    source: str,
    headers: list[str],
    rows: list[dict[str, str]],
    col_types: dict[str, str],
) -> str:
    """
    Build a rich text block summarising the entire CSV table.
    General algorithm — no hardcoded values.
    """
    n = len(rows)
    lines: list[str] = []

    lines.append(f"Table: {source} | {n} rows")
    lines.append(f"Schema: {', '.join(headers)}")
    lines.append("")
    lines.append("AGGREGATE SUMMARY")
    lines.append(f"Row count: {n}")
    lines.append("")

    # ── 1. Numeric column totals ──────────────────────────────────────────────
    numeric_cols = [c for c in headers if col_types.get(c) == "numeric"]
    numeric_totals: dict[str, float] = {}
    if numeric_cols:
        lines.append("Numeric column totals:")
        for col in numeric_cols:
            vals = [_to_numeric(r.get(col, "")) for r in rows]
            total = sum(v for v in vals if v is not None)
            numeric_totals[col] = total
            lines.append(f"  {col}: {_fmt_number(total)}")
        lines.append("")

    # ── 2. Categorical breakdowns ─────────────────────────────────────────────
    cat_cols = [c for c in headers if col_types.get(c) == "categorical"]
    for cat_col in cat_cols:
        group_counts: dict[str, int] = defaultdict(int)
        group_numeric: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for row in rows:
            cat_val = row.get(cat_col, "").strip()
            if not cat_val:
                continue
            group_counts[cat_val] += 1
            for num_col in numeric_cols:
                v = _to_numeric(row.get(num_col, ""))
                if v is not None:
                    group_numeric[cat_val][num_col] += v

        lines.append(f"{cat_col} breakdown:")
        for val, count in sorted(group_counts.items(), key=lambda x: -x[1]):
            if numeric_cols:
                # Show totals for the first numeric column (primary metric)
                primary_num = numeric_cols[0]
                grp_total = group_numeric[val].get(primary_num, 0.0)
                overall_total = numeric_totals.get(primary_num, 0.0)
                pct = (grp_total / overall_total * 100) if overall_total else 0.0
                lines.append(
                    f"  {val}: {count} rows, {primary_num}={_fmt_number(grp_total)} ({pct:.1f}%)"
                )
            else:
                lines.append(f"  {val}: {count} rows")
        lines.append("")

    # ── 3. Date-based aggregations ────────────────────────────────────────────
    date_cols = [c for c in headers if col_types.get(c) == "date"]
    for date_col in date_cols:
        for num_col in numeric_cols:
            # Parse dates + numerics, skip missing
            pairs: list[tuple[datetime, float]] = []
            for row in rows:
                d = _to_date(row.get(date_col, ""))
                v = _to_numeric(row.get(num_col, ""))
                if d is not None and v is not None:
                    pairs.append((d, v))
            if not pairs:
                continue

            lines.append(f"Date-based aggregations ({date_col} x {num_col}):")

            # By year
            year_map: dict[int, list[tuple[datetime, float]]] = defaultdict(list)
            for d, v in pairs:
                year_map[d.year].append((d, v))
            for yr in sorted(year_map):
                yr_pairs = year_map[yr]
                yr_total = sum(v for _, v in yr_pairs)
                lines.append(f"  {yr}: {len(yr_pairs)} rows, {_fmt_number(yr_total)}")

                # By half-year within this year
                h1 = [(d, v) for d, v in yr_pairs if d.month <= 6]
                h2 = [(d, v) for d, v in yr_pairs if d.month >= 7]
                if h1:
                    lines.append(
                        f"  H1 {yr} (Jan-Jun): {len(h1)} rows, {_fmt_number(sum(v for _, v in h1))}"
                    )
                if h2:
                    lines.append(
                        f"  H2 {yr} (Jul-Dec): {len(h2)} rows, {_fmt_number(sum(v for _, v in h2))}"
                    )

                # By quarter within this year
                quarters = {
                    "Q1": (1, 3),
                    "Q2": (4, 6),
                    "Q3": (7, 9),
                    "Q4": (10, 12),
                }
                for qname, (qstart, qend) in quarters.items():
                    q_pairs = [(d, v) for d, v in yr_pairs if qstart <= d.month <= qend]
                    if q_pairs:
                        lines.append(
                            f"  {qname} {yr} ({_month_range(qstart, qend)}): "
                            f"{len(q_pairs)} rows, {_fmt_number(sum(v for _, v in q_pairs))}"
                        )
            lines.append("")

    # ── 4. Per-row ratios (only meaningful pairs: ratio > 1.0 on average) ────────
    if len(numeric_cols) == 2:
        # With exactly 2 numeric columns, compute ratio both ways; keep only the
        # pair where the average ratio is >= 1.0 (e.g. budget/headcount ≫ 1).
        num_col, den_col = numeric_cols[0], numeric_cols[1]
        den_vals = [_to_numeric(r.get(den_col, "")) or 0.0 for r in rows]
        if not all(v == 0.0 for v in den_vals):
            key_col = _find_key_column(headers, col_types)
            ratio_rows: list[tuple[str, float, float, float]] = []
            for row in rows:
                num_v = _to_numeric(row.get(num_col, ""))
                den_v = _to_numeric(row.get(den_col, ""))
                key_v = row.get(key_col, "?").strip()
                if num_v is not None and den_v is not None and den_v != 0:
                    ratio_rows.append((key_v, num_v, den_v, num_v / den_v))
            if ratio_rows:
                avg_ratio = sum(r[3] for r in ratio_rows) / len(ratio_rows)
                if avg_ratio >= 1.0:
                    best = max(ratio_rows, key=lambda x: x[3])
                    lines.append(f"Per-row {num_col}/{den_col} ratios:")
                    for key_v, num_v, den_v, ratio in ratio_rows:
                        lines.append(
                            f"  {key_v}: {_fmt_number(num_v)} / {_fmt_number(den_v)} = {ratio:.0f}/unit"
                        )
                    lines.append(f"  Highest: {best[0]} at {best[3]:.0f}/unit")
                    lines.append("")

    # ── 5. List all unique values for name columns ────────────────────────────
    name_cols = [c for c in headers if col_types.get(c) == "name"]
    for col in name_cols:
        vals = [r.get(col, "").strip() for r in rows if r.get(col, "").strip()]
        lines.append(f"All {col} values: {', '.join(vals)}")
        lines.append("")

    # ── 6. Location column special handling ───────────────────────────────────
    loc_cols = [c for c in headers if c.lower() == "location"]
    for loc_col in loc_cols:
        loc_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            loc = row.get(loc_col, "").strip()
            if loc:
                loc_counts[loc] += 1
        lines.append("Location breakdown:")
        for loc, cnt in sorted(loc_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {loc}: {cnt} employees")
        lines.append("")

    return "\n".join(lines).rstrip()


def _month_range(start_month: int, end_month: int) -> str:
    """Return abbreviated month names for a range, e.g. 'Jan-Mar'."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{months[start_month - 1]}-{months[end_month - 1]}"


def _find_key_column(headers: list[str], col_types: dict[str, str]) -> str:
    """Return the best column to use as a row label."""
    # Prefer name columns
    for col in headers:
        if col_types.get(col) == "name":
            return col
    # Prefer first string column that isn't numeric/date/categorical
    for col in headers:
        if col_types.get(col) in ("string", "categorical"):
            return col
    return headers[0] if headers else "row"


# ── Public parser function ────────────────────────────────────────────────────

def parse_csv(path: Path, source: str, department: str) -> list[SmartChunk]:
    """
    Parse a CSV file into SmartChunks:
      - One "row" chunk per data row (same info as baseline but with chunk_type)
      - One "aggregate" chunk containing the computed aggregate summary

    Returns row chunks first, aggregate chunk last.
    """
    chunks: list[SmartChunk] = []

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        headers: list[str] = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = [dict(row) for row in reader]

    if not rows:
        return chunks

    col_types = _classify_columns(headers, rows)

    # ── Row chunks ────────────────────────────────────────────────────────────
    for idx, row in enumerate(rows):
        row_text = f"[{source}]\n" + " | ".join(f"{k}: {v}" for k, v in row.items())
        chunks.append(SmartChunk(
            text=row_text,
            source=source,
            department=department,
            format="csv",
            chunk_type="row",
            chunk_index=idx,
            extra={"row_number": idx},
        ))

    # ── Aggregate chunk ───────────────────────────────────────────────────────
    agg_text = _build_aggregate_text(source, headers, rows, col_types)
    # aggregate gets index = len(rows) so it never collides with row chunks
    chunks.append(SmartChunk(
        text=agg_text,
        source=source,
        department=department,
        format="csv",
        chunk_type="aggregate",
        chunk_index=len(rows),
        extra={"row_count": len(rows), "headers": ", ".join(headers)},
    ))

    return chunks
