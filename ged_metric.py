"""Graph edit distance metric: heterogeneous graph + NetworkX approximate search.

Metric definition: nodes are components and nets, edges are the
component-port-to-net connections, all edit costs are 1, and the usual
equivalences are normalised first (passive polarity ignored, MOS Source/Drain
interchangeable, net names irrelevant).

Usage:
  python ged_metric.py --mode bounds --generated output/final
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent
if not (ROOT / "pipeline.py").is_file():
    # When run from output/_scratch/ the project root is three levels up.
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT.parent / "data" / "circuit-dataset"
PASSIVE = {"Res", "Cap", "Ind", "Switch"}
MOS = {"NMOS", "PMOS"}


def edge_label(component_type: str, port: str) -> str:
    if component_type in PASSIVE:
        return "Terminal"
    if component_type in MOS and port in ("Drain", "Source"):
        return "S/D"
    return port


def hetero_graph(obj: dict) -> nx.Graph:
    graph = nx.Graph()
    for index, row in enumerate(obj["ckt_netlist"]):
        kind = row["component_type"]
        owner = ("comp", index)
        graph.add_node(owner, label=f"comp:{kind}")
        nets: dict[str, list[str]] = {}
        for port, net in row["port_connection"].items():
            nets.setdefault(str(net), []).append(edge_label(kind, port))
        for net, labels in nets.items():
            # A net is one node shared by every component that touches it; the
            # name itself carries no meaning (only the topology is compared).
            node = ("net", net)
            graph.add_node(node, label="net")
            for label in labels:
                graph.add_edge(owner, node, label=f"port:{label}")
    return graph


def ged(truth: dict, predicted: dict, timeout: float) -> tuple[int, bool]:
    first, second = hetero_graph(truth), hetero_graph(predicted)
    same = lambda a, b: 0 if a["label"] == b["label"] else 1
    started = time.time()
    value = nx.graph_edit_distance(
        first, second,
        node_subst_cost=same, node_del_cost=lambda node: 1, node_ins_cost=lambda node: 1,
        edge_subst_cost=same, edge_del_cost=lambda edge: 1, edge_ins_cost=lambda edge: 1,
        timeout=timeout)
    return int(value), time.time() - started < timeout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=ROOT / "output" / "final")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--cases", default="",
                        help="case list, e.g. 1-40 or 3,7; default = every <case>.txt in --generated")
    parser.add_argument("--mode", choices=("exact", "bounds"), default="exact",
                        help="bounds: fast lower/upper bound instead of the exact search")
    args = parser.parse_args()

    numbers = cases_from_dir(args.cases, args.generated)

    if args.mode == "bounds":
        return run_bounds(args)

    rows, function_hits = [], 0
    for number in numbers:
        case = f"{number:03d}"
        truth = ast.literal_eval((DATA / "true" / f"{case}.txt").read_text(encoding="utf-8"))
        predicted = ast.literal_eval((args.generated / f"{case}.txt").read_text(encoding="utf-8"))
        value, finished = ged(truth, predicted, args.timeout)
        correct = predicted["ckt_type"] == str(truth["ckt_type"]).strip()
        function_hits += correct
        rows.append({"case": case, "ged": value, "timeout_hit": not finished,
                     "type_correct": correct,
                     "true_nodes": sum(1 for row_ in truth["ckt_netlist"] for _ in [0]),
                     "predicted_nodes": len(predicted["ckt_netlist"])})
        print(f"{case}: GED={value:3d} type={'ok ' if correct else 'wrong'} "
              f"{'' if finished else '(timeout, best-so-far)'}", flush=True)
    total = sum(row["ged"] for row in rows)
    coefficient = 1 / math.log10(10 + total)
    print()
    print(f"cases={len(rows)} sum_GED={total} mean_GED={total/len(rows):.3f} "
          f"K=1/lg(10+sum)={coefficient:.4f}")
    print(f"function accuracy F={function_hits}/{len(rows)} = {function_hits/len(rows):.3f}")
    print(f"perfect cases (GED=0): {sum(1 for row in rows if row['ged'] == 0)}")
    (args.generated / "ged.json").write_text(
        json.dumps({"cases": rows, "sum_ged": total, "coefficient_k": round(coefficient, 4),
                    "function_accuracy": round(function_hits / len(rows), 4)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def run_bounds(args) -> int:
    """Fast GED bounds: the exact search needs minutes per drawing.

    Lower bound: node/edge count differences plus the component-type histogram
    distance (each missing part costs at least one insertion, each extra part at
    least one deletion, and pairs of them at least one substitution).
    Upper bound: the node-signature matching used by the diagnostic, expressed as
    edits -- every unmatched port is one edge edit, every leftover component one
    node edit.
    """
    sys.path.insert(0, str(ROOT / "output" / "_scratch"))
    from evaluate_connections import compare  # noqa: E402

    def histogram(obj: dict) -> Counter[str]:
        return Counter(row["component_type"] for row in obj["ckt_netlist"])

    rows, function_hits = [], 0
    for number in cases_from_dir(args.cases, args.generated):
        case = f"{number:03d}"
        truth = ast.literal_eval((DATA / "true" / f"{case}.txt").read_text(encoding="utf-8"))
        predicted = ast.literal_eval((args.generated / f"{case}.txt").read_text(encoding="utf-8"))
        score = compare(truth, predicted)
        true_hist, pred_hist = histogram(truth), histogram(predicted)
        extra = sum(max(0, pred_hist[k] - true_hist[k]) for k in set(true_hist) | set(pred_hist))
        missing = sum(max(0, true_hist[k] - pred_hist[k]) for k in set(true_hist) | set(pred_hist))
        edges_truth = sum(len(row["port_connection"]) for row in truth["ckt_netlist"])
        edges_pred = sum(len(row["port_connection"]) for row in predicted["ckt_netlist"])
        lower = max(abs(len(truth["ckt_netlist"]) - len(predicted["ckt_netlist"])),
                    abs(edges_truth - edges_pred),
                    abs(score["true_nets"] - score["predicted_nets"]),
                    max(extra, missing))
        upper = (edges_truth + edges_pred - 2 * score["matched_typed_pin_incidence"]
                 + extra + missing)
        correct = predicted["ckt_type"] == str(truth["ckt_type"]).strip()
        function_hits += correct
        rows.append({"case": case, "lower": int(lower), "upper": int(upper),
                     "type_correct": correct})
        print(f"{case}: GED in [{lower:3d}, {upper:3d}]  (proxy matched ports "
              f"{score['matched_typed_pin_incidence']}/{edges_truth})", flush=True)
    low, high = sum(row["lower"] for row in rows), sum(row["upper"] for row in rows)
    print()
    print(f"cases={len(rows)} sum_GED in [{low}, {high}]  "
          f"K in [{1/math.log10(10+high):.4f}, {1/math.log10(10+low):.4f}]")
    print(f"function accuracy F={function_hits}/{len(rows)} = {function_hits/len(rows):.3f}")
    (args.generated / "ged_bounds.json").write_text(
        json.dumps({"cases": rows, "sum_lower": low, "sum_upper": high,
                    "k_lower": round(1 / math.log10(10 + high), 4),
                    "k_upper": round(1 / math.log10(10 + low), 4),
                    "function_accuracy": round(function_hits / len(rows), 4)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def numbers_from(spec: str) -> list[int]:
    numbers: list[int] = []
    for chunk in spec.split(","):
        if "-" in chunk:
            start, end = chunk.split("-")
            numbers.extend(range(int(start), int(end) + 1))
        else:
            numbers.append(int(chunk))
    return numbers


def cases_from_dir(spec: str, generated: Path) -> list[int]:
    """Case numbers from an explicit spec, or every netlist file on disk."""
    if spec.strip():
        return numbers_from(spec)
    numbers = []
    for path in sorted(generated.glob("*.txt")):
        if path.stem.isdigit():
            numbers.append(int(path.stem))
    if not numbers:
        raise SystemExit(f"no <case>.txt netlists found in {generated}")
    return numbers


if __name__ == "__main__":
    raise SystemExit(main())
