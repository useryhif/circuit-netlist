"""Re-apply the current GCN classifier to already generated netlists.

The pixel pipeline is the expensive part, so when only the classifier changes,
relabel the stored netlists instead of re-running detector inference.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from eda_gcn import DATA, predict


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel stored netlists with the current GCN")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "final")
    args = parser.parse_args()
    root = args.out.resolve()
    rows = []
    for case_dir in sorted(path for path in root.glob("[0-9][0-9][0-9]") if path.is_dir()):
        netlist_path = case_dir / "netlist.txt"
        if not netlist_path.is_file():
            continue
        netlist = ast.literal_eval(netlist_path.read_text(encoding="utf-8"))
        label, scores = predict(netlist)
        netlist["ckt_type"] = label
        netlist_path.write_text(str(netlist), encoding="utf-8")
        (root / f"{case_dir.name}.txt").write_text(str(netlist), encoding="utf-8")
        (case_dir / "ckt_type_scores.json").write_text(
            json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        truth = ast.literal_eval((DATA / "true" / f"{case_dir.name}.txt").read_text(encoding="utf-8"))
        true_type = str(truth["ckt_type"]).strip()
        rows.append({"case": case_dir.name, "predicted_type": label, "true_type": true_type})
    accuracy = sum(row["predicted_type"] == row["true_type"] for row in rows) / max(1, len(rows))
    print(f"relabelled {len(rows)} cases; agreement with truth {accuracy:.4f} "
          f"(final model saw all 40 labels, so this is not a generalization score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
