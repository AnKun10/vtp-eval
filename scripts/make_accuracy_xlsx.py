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

# R1-stage diversity ablation (proposed method, fixed avg-64 budget): rows are
# the diversity ratio % (rest of R1 = attention/dominant). Cells = ABSOLUTE
# accuracy as a percentage (matches the requested frame).
DIVERSITY = [(10, "proposed_div10_avg64"), (20, "proposed_div20_avg64"),
             (30, "proposed_div30_avg64"), (40, "proposed_div40_avg64"),
             (50, "proposed_div50_avg64")]

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


def load_timing() -> dict:
    """(method, task) -> (encoder_ms, prefill_ms, decode_ms, total_ms)."""
    t = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                t[(r["method"], r["task"])] = (
                    float(r["encoder_ms"]), float(r["prefill_ms"]),
                    float(r["decode_ms"]), float(r["total_ms"]))
            except (ValueError, KeyError):
                continue
    return t


def fill_sparsevlm_estimates(timing):
    """SparseVLM's native harness logged no per-stage latency (all 0), but it DID
    log avg_tokens. We estimate its timing by interpolation from the measured
    methods (mutating ``timing`` in place):

      encoder_ms : SparseVLM prunes INSIDE the LLM (text-guided, like FastV), so
                   the vision tower runs the full CLIP with no vision-side
                   selection -> reuse the FastV encoder for that task (~30ms).
      prefill_ms : linear fit of prefill_ms vs avg_tokens over the 12 measured
                   pruned configs (baseline excluded: n=576 lies outside the
                   interpolation range), evaluated at sparsevlm's avg_tokens.
      decode_ms  : linear fit of decode_ms vs avg_tokens (near-flat; decode is
                   per-output-token, ~independent of the visual-token count).
      total_ms   = encoder + prefill + decode.

    The model recovers measured totals within ~4% mean error, and sparsevlm's
    avg_tokens (43/62/126 at retain 32/64/128) interpolate inside the measured
    32-232 range. Values are flagged as estimates in the sheets (italic + '*')."""
    import statistics as _st

    measured = [f"{m}_retain{s}" for s in ("32", "64", "128")
                for m in ("divprune", "fastv", "visionzip", "proposed")]
    atok = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        seen = set()
        for r in csv.DictReader(f):
            k = (r["method"], r["task"])
            if k in seen:
                continue
            seen.add(k)
            try:
                atok[k] = float(r["avg_tokens"])
            except (ValueError, KeyError):
                pass

    def _fit(xs, ys):
        mx, my = _st.fmean(xs), _st.fmean(ys)
        b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
        return my - b * mx, b

    for task in sorted({t for _, t in timing}):
        pts = [(atok[(m, task)], *timing[(m, task)]) for m in measured
               if (m, task) in timing and (m, task) in atok and timing[(m, task)][3]]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ap, bp = _fit(xs, [p[2] for p in pts])        # prefill_ms vs n
        ad, bd = _fit(xs, [p[3] for p in pts])        # decode_ms vs n
        for s in ("32", "64", "128"):
            sk, fk = (f"sparsevlm_retain{s}", task), (f"fastv_retain{s}", task)
            if sk not in atok or fk not in timing:
                continue
            n = atok[sk]
            enc, pf, dc = timing[fk][0], ap + bp * n, ad + bd * n
            timing[sk] = (enc, pf, dc, enc + pf + dc)


def _append_estimate_footnote(ws):
    ws.append([])
    ws.append(["* sparsevlm timing estimated by interpolation from avg_tokens "
               "(native harness logged no per-stage latency)"])
    ws.cell(ws.max_row, 1).font = Font(italic=True, size=9, color="808080")


def build_latency_sheet(wb, timing):
    """End-to-end latency SPEEDUP vs baseline (baseline = 1.00x; higher = faster).
    One value per benchmark = baseline total_ms / method total_ms (whole-request
    wall clock, not split into encode/prefill/decode). avg = mean of the
    per-benchmark speedups. sparsevlm is omitted (no recorded timing)."""
    ws = wb.create_sheet("Latency (speedup)")
    SP = '0.00"x"'
    ncol = 1 + len(BENCH) + 1
    base = {col: timing[(BASELINE[1], task)][3] for col, task, *_ in BENCH}

    ws.append(["method"] + [b[0] for b in BENCH] + ["avg"])
    for c in range(1, ncol + 1):
        cell = ws.cell(1, c)
        cell.font = Font(bold=True)
        cell.fill = HDR_FILL
        cell.alignment = CENTER
        cell.border = BORDER

    def speedups(method_key):
        return [base[col] / timing[(method_key, task)][3] for col, task, *_ in BENCH]

    def write_row(display, method_key, *, baseline=False, best=None, est=False):
        vals = speedups(method_key)
        ws.append([display + ("*" if est else "")] + vals + [sum(vals) / len(vals)])
        row = ws.max_row
        for c in range(1, ncol + 1):
            cell = ws.cell(row, c)
            cell.border = BORDER
            if c == 1:
                cell.font = Font(bold=baseline or display == "proposed", italic=est)
            else:
                cell.alignment = CENTER
                cell.number_format = SP
                is_best = best is not None and best.get(c - 2) == method_key
                cell.font = Font(bold=baseline or is_best, italic=est)
            if baseline:
                cell.fill = BASE_FILL

    write_row(*BASELINE, baseline=True)
    for title, methods in SECTIONS:
        ws.append([title] + [""] * (ncol - 1))
        srow = ws.max_row
        ws.merge_cells(start_row=srow, start_column=1, end_row=srow, end_column=ncol)
        ws.cell(srow, 1).font = Font(bold=True)
        ws.cell(srow, 1).fill = SEC_FILL
        ws.cell(srow, 1).alignment = CENTER
        for c in range(1, ncol + 1):
            ws.cell(srow, c).border = BORDER
        measured = [(d, m) for d, m in methods if not m.startswith("sparsevlm")]
        sec = {m: speedups(m) for _, m in measured}      # bold 'best' over measured
        best = {idx: max(sec, key=lambda m: sec[m][idx]) for idx in range(len(BENCH))}
        best[len(BENCH)] = max(sec, key=lambda m: sum(sec[m]) / len(sec[m]))   # avg col
        for display, m in methods:
            write_row(display, m, best=best, est=m.startswith("sparsevlm"))

    _append_estimate_footnote(ws)
    ws.column_dimensions["A"].width = 16
    for c in range(2, ncol + 1):
        ws.column_dimensions[get_column_letter(c)].width = 11
    ws.freeze_panes = "B2"
    return ws


def build_cache_savings_sheet(wb, timing):
    """KV retain-token cache SAVINGS: the share of per-turn latency that a cache
    HIT eliminates by skipping the vision encode. Treating each method's own
    total_ms as 100%, the saved time is its encoder_ms, so a cell reads
    ``encoder_ms / total_ms`` = the % of latency removed on a cached (turn 2+)
    request for the same image. Baseline included; sparsevlm omitted (no
    per-stage timing). avg = mean of the per-benchmark savings."""
    ws = wb.create_sheet("KV-cache (savings)")
    ncol = 1 + len(BENCH) + 1

    ws.append(["method"] + [b[0] for b in BENCH] + ["avg (%)"])
    for c in range(1, ncol + 1):
        cell = ws.cell(1, c)
        cell.font = Font(bold=True)
        cell.fill = HDR_FILL
        cell.alignment = CENTER
        cell.border = BORDER

    def savings(method_key):
        out = []
        for _, task, *_ in BENCH:
            enc, _pf, _dc, tot = timing[(method_key, task)]
            out.append(enc / tot if tot else 0.0)
        return out

    def write_row(display, method_key, *, baseline=False, best=None, est=False):
        vals = savings(method_key)
        ws.append([display + ("*" if est else "")] + vals + [sum(vals) / len(vals)])
        row = ws.max_row
        for c in range(1, ncol + 1):
            cell = ws.cell(row, c)
            cell.border = BORDER
            if c == 1:
                cell.font = Font(bold=baseline or display == "proposed", italic=est)
            else:
                cell.alignment = CENTER
                cell.number_format = "0.00%"
                is_best = best is not None and best.get(c - 2) == method_key
                cell.font = Font(bold=baseline or is_best, italic=est)
            if baseline:
                cell.fill = BASE_FILL

    write_row(*BASELINE, baseline=True)
    for title, methods in SECTIONS:
        ws.append([title] + [""] * (ncol - 1))
        srow = ws.max_row
        ws.merge_cells(start_row=srow, start_column=1, end_row=srow, end_column=ncol)
        ws.cell(srow, 1).font = Font(bold=True)
        ws.cell(srow, 1).fill = SEC_FILL
        ws.cell(srow, 1).alignment = CENTER
        for c in range(1, ncol + 1):
            ws.cell(srow, c).border = BORDER
        measured = [(d, m) for d, m in methods if not m.startswith("sparsevlm")]
        sec = {m: savings(m) for _, m in measured}       # bold 'best' over measured
        best = {idx: max(sec, key=lambda m: sec[m][idx]) for idx in range(len(BENCH))}
        best[len(BENCH)] = max(sec, key=lambda m: sum(sec[m]) / len(sec[m]))   # avg col
        for display, m in methods:
            write_row(display, m, best=best, est=m.startswith("sparsevlm"))

    _append_estimate_footnote(ws)
    ws.column_dimensions["A"].width = 16
    for c in range(2, ncol + 1):
        ws.column_dimensions[get_column_letter(c)].width = 11
    ws.freeze_panes = "B2"
    return ws


def build_diversity_sheet(wb, data):
    """Second sheet: proposed R1-prune diversity ablation. Rows = diversity ratio
    %, cells = ABSOLUTE accuracy as a percentage (MME normalised by 2800)."""
    ws = wb.create_sheet("R1-prune (diversity)")
    ncol = 1 + len(BENCH)
    ws.append(["diversity ratio (%)"] + [b[0] for b in BENCH])
    for c in range(1, ncol + 1):
        cell = ws.cell(1, c)
        cell.font = Font(bold=True)
        cell.fill = HDR_FILL
        cell.alignment = CENTER
        cell.border = BORDER

    rows_fr = {pct: fracs(data, mk) for pct, mk in DIVERSITY}
    best = {col: max(rows_fr, key=lambda p: rows_fr[p][col]) for col, *_ in BENCH}
    for pct, _ in DIVERSITY:
        fr = rows_fr[pct]
        ws.append([pct] + [fr[c] for c, *_ in BENCH])
        row = ws.max_row
        for c in range(1, ncol + 1):
            cell = ws.cell(row, c)
            cell.border = BORDER
            cell.alignment = CENTER
            if c == 1:
                cell.number_format = "0"
            else:
                col = BENCH[c - 2][0]
                cell.number_format = "0.00%"
                cell.font = Font(bold=best[col] == pct)

    ws.column_dimensions["A"].width = 18
    for c in range(2, ncol + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13
    ws.freeze_panes = "B2"
    return ws


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

    build_diversity_sheet(wb, data)
    timing = load_timing()
    fill_sparsevlm_estimates(timing)
    build_latency_sheet(wb, timing)
    build_cache_savings_sheet(wb, timing)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}  (sheets: {wb.sheetnames})")


if __name__ == "__main__":
    main()
