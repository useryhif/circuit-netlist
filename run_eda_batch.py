"""Run the image-to-netlist pipeline and write the standard netlist outputs."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from validate_eda_outputs import validate

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "circuit-dataset"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "final")
    ap.add_argument("--report-existing", action="store_true",
                    help="Rebuild the report from already generated case files without rerunning inference.")
    ap.add_argument("--labels-cache", type=Path,
                    help="Reuse per-case YOLO labels from a previous batch output.")
    ap.add_argument("--aux-labels-cache", type=Path,
                    help="Reuse auxiliary per-case labels for NPN/PNP conflict resolution.")
    ap.add_argument("--aux-label-data", type=Path,
                    default=ROOT.parent / "data" / "circuit-dataset" / "aux-detector" / "data" / "circuit.yaml")
    args = ap.parse_args()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for number in range(args.start, args.end + 1):
        case = f"{number:03d}"
        image = DATA / "images" / f"{case}.png"
        case_dir = out_root / case
        generated_file = out_root / f"{case}.txt"
        if args.report_existing:
            if not generated_file.exists() or not (case_dir / "debug" / "topology.json").exists():
                summary.append({"case": case, "status": "failed", "error": "missing generated netlist or topology"})
                print(f"{case}: MISSING", flush=True)
                continue
        else:
            command = [sys.executable, str(ROOT / "pipeline.py"), str(image), "--out", str(case_dir)]
            if args.labels_cache:
                label = (args.labels_cache.resolve() / case / "debug" / "original_yolo" /
                         "predict" / "labels" / f"{case}.txt")
                command.extend(["--labels", str(label)])
            if args.aux_labels_cache:
                cache = args.aux_labels_cache.resolve()
                nested = cache / case / "debug" / "auxiliary_yolo" / "predict" / "labels" / f"{case}.txt"
                flat = cache / f"{case}.txt"
                auxiliary_label = nested if nested.exists() else flat
                command.extend(["--aux-labels", str(auxiliary_label),
                                "--aux-label-data", str(args.aux_label_data.resolve())])
            process = subprocess.run(command,
                                     cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            (case_dir / "batch.log").write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
            if process.returncode:
                summary.append({"case": case, "status": "failed", "error": process.stderr[-1200:]})
                print(f"{case}: FAILED", flush=True)
                continue
            generated_file.write_text((case_dir / "netlist.txt").read_text(encoding="utf-8"), encoding="utf-8")
        obj = ast.literal_eval(generated_file.read_text(encoding="utf-8"))
        schema_errors = validate(generated_file)
        truth = ast.literal_eval((DATA / "true" / f"{case}.txt").read_text(encoding="utf-8"))
        topology = json.loads((case_dir / "debug" / "topology.json").read_text(encoding="utf-8"))
        trust = topology.get("trust", {})
        expected = sum(len(row["port_connection"]) for row in truth["ckt_netlist"])
        emitted = sum(len(row["port_connection"]) for row in obj["ckt_netlist"])
        top_warnings = list(topology.get("warnings", []))
        summary.append({"case": case, "status": "ok", "predicted_type": obj["ckt_type"],
                        "true_type": str(truth["ckt_type"]).strip(), "type_correct": obj["ckt_type"] == str(truth["ckt_type"]).strip(),
                        "components_emitted": len(obj["ckt_netlist"]), "components_true": len(truth["ckt_netlist"]),
                        "ports_emitted": emitted, "ports_true": expected,
                        "port_coverage": round(emitted / max(1, expected), 4),
                        "schema_valid": not schema_errors, "schema_errors": schema_errors,
                        "nets": len(topology["nets"]), "junctions": len(topology["junctions"]),
                        "warnings": len(top_warnings), "warning_messages": top_warnings,
                        "trust_level": trust.get("level", "unknown"),
                        "trust_reasons": trust.get("reasons", [])})
        print(f"{case}: type={obj['ckt_type']} ({'correct' if obj['ckt_type'] == str(truth['ckt_type']).strip() else 'wrong'}), "
              f"components={len(obj['ckt_netlist'])}/{len(truth['ckt_netlist'])}, ports={emitted}/{expected}", flush=True)
    ok = [row for row in summary if row["status"] == "ok"]
    gcn_report = json.loads((ROOT / "models" / "eda_gcn_report.json").read_text(encoding="utf-8"))
    report = {"processed": len(summary), "succeeded": len(ok),
              "schema_valid_count": sum(row["schema_valid"] for row in ok),
              "type_agreement_on_same_40_used_for_training": sum(row["type_correct"] for row in ok) / max(1, len(ok)),
              "gcn_stratified_5fold_accuracy": gcn_report.get("stratified_cv_accuracy"),
              "gcn_generated_netlist_heldout_accuracy": gcn_report.get("generated_netlist_accuracy"),
              "mean_emitted_port_count_to_true_count_ratio": sum(row["port_coverage"] for row in ok) / max(1, len(ok)),
              "metrics_note": "Port counts are a quantity ratio, not port or topology accuracy. Same-40 type agreement uses a classifier trained on these labels and is not held-out evaluation; gcn_generated_netlist_heldout_accuracy uses fold models that did not see each evaluated circuit.",
              "cases": summary}
    (out_root / "batch_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    levels = {"high": [], "caution": [], "review": [], "unknown": []}
    for row in ok:
        levels.setdefault(row.get("trust_level", "unknown"), []).append(row["case"])
    lines = ["# Batch review summary", "",
             f"{report['processed']} drawings processed, {report['succeeded']} succeeded, "
             f"{report['schema_valid_count']} passed format validation.", "",
             "Trust levels come from the components' and nets' own features (no reference "
             "netlist is used) and decide what needs manual review:", "",
             "| level | meaning | drawings |", "|---|---|---|",
             f"| high | no known risk signal | {' '.join(levels['high']) or '-'} |",
             f"| caution | depends on per-drawing conventions | {' '.join(levels['caution']) or '-'} |",
             f"| review | contains a known unreliable stage | {' '.join(levels['review']) or '-'} |",
             "", "## Drawings that need manual review (review / caution)", ""]
    for level in ("review", "caution"):
        for row in ok:
            if row.get("trust_level") != level:
                continue
            lines.append(f"- **{row['case']}**（{level}）")
            for reason in row.get("trust_reasons", []):
                lines.append(f"  - {reason}")
    (out_root / "qa_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {out_root / 'batch_report.json'}", flush=True)
    return 0 if len(ok) == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
