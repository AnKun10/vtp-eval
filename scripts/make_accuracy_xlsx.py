#!/usr/bin/env python
"""Build an Accuracy comparison .xlsx from docs/summary.csv.

Layout (matches the requested frame):
    method | gqa | mme | ocrbench | pope | scienceqa_img | textvqa | vizwiz_vqa | avg (%)
    baseline ...                                                              100%
    --- Retain 32 ---
    divprune / fastv / sparsevlm / visionzip / proposed
    --- Retain 64 ---  ...
    --- Retain 128 --- ...

Every cell is the score RELATIVE to baseline (method / baseline), so the baseline
row is 100% on every benchmark and a cell reads "X% of baseline" (e.g. divprune
gqa 95.9% = 95.9% of the baseline's gqa accuracy). The "avg (%)" column is the
mean of those per-benchmark ratios. All cells use Excel percentage formatting.

    python scripts/make_accuracy_xlsx.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs/summary.csv"
OUT = ROOT / "docs/accuracy_summary.xlsx"

# Display column -> (csv task, csv metric, max value for %-normalisation).
BENCH = [
    ("gqa",           "gqa",            "exact_match",       1.0),
    ("mme",           "mme",            "mme_total",         2800.0),
    ("ocrbench",      "ocrbench",       "ocrbench_accuracy", 1.0),
    ("pope",          "pope",           "pope_f1_score",     1.0),
    ("scienceqa_img", "scienceqa_img",  "exact_match",       1.0),
    ("textvqa",       "textvqa_val",    "exact_match",       1.0),
    ("vizwiz_vqa",    "vizwiz_vqa_val", "exact_match",       1.0),
]
BASELINE = ("baseline", "baseline")
SECTIONS = [
    ("Retain 32", [("divprune", "divprune_retain32"), ("fastv", "fastv_retain32"),
                   ("sparsevlm", "sparsevlm_retain32"), ("visionzip", "visionzip_retain32"),
                   ("proposed", "proposed_retain32")]),
    ("Retain 64", [("divprune", "divprune_retain64"), ("fastv", "fastv_retain64"),
                   ("sparsevlm", "sparsevlm_retain64"), ("visionzip", "visionzip_retain64"),
                   ("proposed", "proposed_retain64")]),
    ("Retain 128", [("divprune", "divprune_retain128"), ("fastv", "fastv_retain128"),
                    ("sparsevlm", "sparsevlm_retain128"), ("visionzip", "visionzip_retain128"),
                    ("proposed", "proposed_retain128")]),
]

N_COLS = 1 + len(BENCH) + 1                       # method + benchmarks + avg
PCT2, PCT1 = "0.00%", "0.0%"
HDR_FILL = PatternFill("solid", fgColor="DDEBF7")
SEC_FILL = PatternFill("solid", fgColor="FCE4D6")
BASE_FILL = PatternFill("solid", fgColor="E2EFDA")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")


def load() -> dict:
    out = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["method"], r["task"], r["metric"])] = float(r["value"])
    return out


def fracs(data, method_key):
    """0-1 fraction per benchmark (MME normalised by its 2800 max)."""
    return {col: data[(method_key, task, metric)] / mx
            for col, task, metric, mx in BENCH}


def main():
    data = load()
    base = fracs(data, BASELINE[1])

    def avg_ratio(fr):
        return sum(fr[c] / base[c] for c, *_ in BENCH) / len(BENCH)

    wb = Workbook()
    ws = wb.active
    ws.title = "Accuracy"

    # header
    ws.append(["method"] + [b[0] for b in BENCH] + ["avg (%)"])
    for c in range(1, N_COLS + 1):
        cell = ws.cell(1, c)
        cell.font = Font(bold=True)
        cell.fill = HDR_FILL
        cell.alignment = CENTER
        cell.border = BORDER

    def write_method(display, method_key, *, baseline=False, best=None):
        fr = fracs(data, method_key)
        # Every cell is RELATIVE to baseline (method / baseline) so the baseline
        # row is 100% on every benchmark and a cell reads "X% of baseline".
        ws.append([display] + [fr[c] / base[c] for c, *_ in BENCH] + [avg_ratio(fr)])
        row = ws.max_row
        for c in range(1, N_COLS + 1):
            cell = ws.cell(row, c)
            cell.border = BORDER
            if c == 1:
                cell.font = Font(bold=baseline or display == "proposed")
            else:
                cell.alignment = CENTER
                cell.number_format = PCT1 if c == N_COLS else PCT2
                col_name = (BENCH[c - 2][0] if c <= N_COLS - 1 else "avg")
                is_best = best is not None and best.get(col_name) == method_key
                cell.font = Font(bold=baseline or is_best)
            if baseline:
                cell.fill = BASE_FILL
        return fr

    write_method(*BASELINE, baseline=True)

    for title, methods in SECTIONS:
        # section banner row (merged across all columns)
        ws.append([title] + [""] * (N_COLS - 1))
        srow = ws.max_row
        ws.merge_cells(start_row=srow, start_column=1, end_row=srow, end_column=N_COLS)
        sc = ws.cell(srow, 1)
        sc.font = Font(bold=True)
        sc.fill = SEC_FILL
        sc.alignment = CENTER
        for c in range(1, N_COLS + 1):
            ws.cell(srow, c).border = BORDER
        # best (max) per column within this section -> for bolding
        section_fr = {mk: fracs(data, mk) for _, mk in methods}
        best = {}
        for col, *_ in BENCH:
            best[col] = max(section_fr, key=lambda m: section_fr[m][col])
        best["avg"] = max(section_fr, key=lambda m: avg_ratio(section_fr[m]))
        for display, mk in methods:
            write_method(display, mk, best=best)

    # column widths + freeze header
    ws.column_dimensions["A"].width = 16
    for c in range(2, N_COLS + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13
    ws.freeze_panes = "B2"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}  ({ws.max_row} rows)")


if __name__ == "__main__":
    main()
