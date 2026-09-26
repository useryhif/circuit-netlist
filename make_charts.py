"""Draw the result charts: per-case structural F1 and graph edit distance.

Usage:
  python make_charts.py --run output/final --out output/final/charts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
LEVEL_COLOUR = {"high": "#2e7d32", "caution": "#f9a825", "review": "#c62828", "unknown": "#757575"}


def register_cjk_font() -> str | None:
    """Let the charts carry Chinese labels instead of empty boxes."""
    candidates = {"Microsoft YaHei": "msyh.ttc", "SimHei": "simhei.ttf", "SimSun": "simsun.ttc"}
    for name, filename in candidates.items():
        for directory in (Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts")):
            path = directory / filename
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "output" / "final")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run = args.run.resolve()
    out = (args.out or run / "charts").resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"CJK font: {register_cjk_font() or 'not found (Chinese labels will show as boxes)'}")
    report = json.loads((run / "batch_report.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((run / "connection_diagnostic.json").read_text(encoding="utf-8"))
    f1 = {row["case"]: row["proxy_f1"] for row in diagnostic["cases"]}
    exact = {row["case"]: row["exact_typed_pin_graph_isomorphic"] for row in diagnostic["cases"]}
    level = {row["case"]: row.get("trust_level", "unknown") for row in report["cases"]}
    bounds_path = run / "ged_bounds.json"
    ged_upper = {}
    ged_lower = {}
    if bounds_path.is_file():
        bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
        ged_upper = {row["case"]: row["upper"] for row in bounds["cases"]}
        ged_lower = {row["case"]: row["lower"] for row in bounds["cases"]}

    # 1. per-case status
    cases = sorted(f1)
    figure, axis = plt.subplots(figsize=(11, 3.8), dpi=160)
    values = [f1[case] for case in cases]
    colours = [LEVEL_COLOUR.get(level.get(case, "unknown"), "#757575") for case in cases]
    axis.bar(cases, values, color=colours)
    for index, case in enumerate(cases):
        if exact.get(case):
            axis.annotate("*", (index, values[index]), textcoords="offset points",
                          xytext=(0, 2), ha="center", fontsize=10, color="#2e7d32")
    axis.set_ylim(0.4, 1.02)
    axis.set_ylabel("node-port F1")
    axis.set_title("Per-case structural F1 (bar colour = trust level, * = fully isomorphic)")
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in
               ("#2e7d32", "#f9a825", "#c62828")]
    axis.legend(handles, ("high", "caution", "review"), fontsize=8, ncol=3, loc="lower right")
    figure.tight_layout()
    figure.savefig(out / "per_case_f1.png")
    plt.close(figure)

    # 2. GED bounds
    if ged_upper:
        figure, axis = plt.subplots(figsize=(11, 3.6), dpi=160)
        cases = sorted(ged_upper)
        axis.bar(cases, [ged_upper[c] for c in cases], color="#90a4ae", label="upper bound")
        axis.bar(cases, [ged_lower[c] for c in cases], color="#1565c0", label="lower bound")
        axis.set_ylabel("graph edit distance")
        axis.set_title("Per-case graph edit distance bounds (approximate)")
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8, loc="upper left")
        axis.set_ylim(0, max(ged_upper.values()) * 1.45)
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
        axis.text(0.98, 0.96,
                  "Graph edit distance is solved approximately here:\n"
                  "exact search times out on large graphs, so reachable bounds are reported.",
                  transform=axis.transAxes, ha="right", va="top", fontsize=7.5, color="#546e7a")
        figure.tight_layout()
        figure.savefig(out / "ged_bounds.png")
        plt.close(figure)

    print(f"charts written to {out}")
    for path in sorted(out.glob("*.png")):
        print("  ", path.name, f"{path.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
