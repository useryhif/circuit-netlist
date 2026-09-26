"""Assemble the result charts and case figures into a single PDF report.

Usage:
  python make_report_pdf.py --run output/final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
PAGE = (11.69, 8.27)          # A4 landscape


def register_cjk_font() -> str | None:
    """Make the report's Chinese labels render instead of showing empty boxes."""
    candidates = {
        "Microsoft YaHei": "msyh.ttc",
        "SimHei": "simhei.ttf",
        "SimSun": "simsun.ttc",
        "Noto Sans CJK SC": "NotoSansCJK-Regular.ttc",
    }
    for name, filename in candidates.items():
        for directory in (Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts")):
            path = directory / filename
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name
    return None


def text_page(pdf: PdfPages, title: str, lines: list[str], size: int = 11) -> None:
    figure = plt.figure(figsize=PAGE)
    figure.suptitle(title, fontsize=17, y=0.94)
    axis = figure.add_axes([0.06, 0.06, 0.88, 0.82])
    axis.axis("off")
    axis.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=size,
              family="DejaVu Sans Mono" if any(line.startswith("|") for line in lines) else None)
    pdf.savefig(figure)
    plt.close(figure)


def image_page(pdf: PdfPages, path: Path, title: str) -> None:
    figure = plt.figure(figsize=PAGE)
    figure.suptitle(title, fontsize=15, y=0.95)
    axis = figure.add_axes([0.03, 0.06, 0.94, 0.82])
    axis.axis("off")
    axis.imshow(mpimg.imread(path))
    pdf.savefig(figure)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "output" / "final")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cases", default="038,039", help="example cases to include")
    args = parser.parse_args()

    font = register_cjk_font()
    print(f"CJK font: {font or 'not found (labels will show as boxes)'}")
    run = args.run.resolve()
    charts = run / "charts"
    output = (args.out or run / "result_report.pdf").resolve()
    report = json.loads((run / "batch_report.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((run / "connection_diagnostic.json").read_text(encoding="utf-8"))
    bounds = json.loads((run / "ged_bounds.json").read_text(encoding="utf-8"))
    levels = {row["case"]: row.get("trust_level") for row in report["cases"]}
    counts = {level: sum(1 for value in levels.values() if value == level)
              for level in ("high", "caution", "review")}

    with PdfPages(output) as pdf:
        text_page(pdf, "Schematic to Netlist - Results Overview", [
            "Data: 40 public schematics (data/circuit-dataset), compared per drawing with the reference netlists.",
            "",
            "Structural diagnostic (node-port signature matching)",
            f"  mean node-port F1        : {diagnostic['mean_proxy_f1']:.4f}",
            f"  fully isomorphic         : {diagnostic['exact_isomorphic_count']}/40",
            f"  strict (Pos/Neg kept)    : {diagnostic['mean_strict_proxy_f1']:.4f} "
            f"/ {diagnostic['strict_exact_isomorphic_count']}/40",
            "",
            "Graph edit distance (heterogeneous graph + NetworkX approximate search)",
            f"  sum GED                  : {bounds['sum_lower']} - {bounds['sum_upper']} (lower/upper bound)",
            f"  netlist coefficient K    : {bounds['k_lower']:.4f} - {bounds['k_upper']:.4f}",
            f"  GED = 0 cases            : {sum(1 for row in bounds['cases'] if row['upper'] == 0)}/40",
            "  note                     : approximate metric, used for error localisation only",
            f"  function classification  : {report['type_agreement_on_same_40_used_for_training']:.2%} agreement on the same 40, "
            f"{report['gcn_generated_netlist_heldout_accuracy']:.2%} held-out (generated netlists)",
            "",
            "Delivery quality",
            f"  format validation        : {report['schema_valid_count']}/{report['succeeded']} passed",
            f"  component rows           : "
            f"{sum(row['components_emitted'] for row in report['cases'])}/"
            f"{sum(row['components_true'] for row in report['cases'])} rows",
            f"  port entries             : "
            f"{sum(row['ports_emitted'] for row in report['cases'])}/"
            f"{sum(row['ports_true'] for row in report['cases'])} ports "
            f"({report['mean_emitted_port_count_to_true_count_ratio']:.1%} by count)",
            f"  trust levels             : high={counts['high']}  caution={counts['caution']}  "
            f"review={counts['review']}",
            "",
            "Per-drawing artifacts: three-stage figures (detection / nodes / polarity), debug/topology.json,",
            "trust summary qa_summary.md; scripts and usage are documented in the repository README.",
        ], size=12)
        for name, title in (("per_case_f1.png", "Per-case structural F1 and trust level"),
                            ("ged_bounds.png", "Per-case graph edit distance bounds")):
            if (charts / name).is_file():
                image_page(pdf, charts / name, title)
        for case in args.cases.split(","):
            case = case.strip()
            if not case:
                continue
            reasons = next((row.get("trust_reasons", []) for row in report["cases"]
                            if row["case"] == case.zfill(3)), [])
            ged = next((row for row in bounds["cases"] if row["case"] == case.zfill(3)), {})
            lines = [f"structural F1 = {next(r['proxy_f1'] for r in diagnostic['cases'] if r['case'] == case.zfill(3)):.4f}",
                     f"graph edit distance bounds in [{ged.get('lower', '-')}, {ged.get('upper', '-')}] (approximate)",
                     f"trust level = {levels.get(case.zfill(3))}"]
            if reasons:
                lines += ["", "risk reasons:"] + [f"  - {reason}" for reason in reasons]
            text_page(pdf, f"Case {case}", lines, size=12)
            for figure_name, figure_title in (("01_detection.png", f"{case} stage 1: component detection"),
                                              ("02_connected_nodes.png", f"{case} stage 2: electrical nodes"),
                                              ("03_polarity.png", f"{case} stage 3: polarity labels")):
                path = run / case.zfill(3) / figure_name
                if path.is_file():
                    image_page(pdf, path, figure_title)

    print(f"report written: {output} ({output.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
