"""Independent topology diagnostic that needs no extra graph-library runtime.

It compares the multisets of typed component pins attached to each electrical
node after optimally pairing predicted nodes with true nodes. This is an upper
bound proxy, not an exact graph edit distance: component identities are not
paired, so repeated components can make the proxy optimistic.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "circuit-dataset"
SYMMETRIC_PORT_COMPONENTS = {"Res", "Cap", "Ind", "Switch"}


def _port_label(component_type: str, port: str, symmetric_ports: bool) -> str:
    """Use the EDA equivalence rule for interchangeable terminals."""
    if symmetric_ports and component_type in SYMMETRIC_PORT_COMPONENTS:
        return "Terminal"
    return port


def node_signatures(obj: dict, symmetric_ports: bool = True) -> dict[str, Counter[str]]:
    nodes: dict[str, Counter[str]] = defaultdict(Counter)
    for component in obj["ckt_netlist"]:
        kind = component["component_type"]
        for port, net in component["port_connection"].items():
            label = _port_label(kind, port, symmetric_ports)
            nodes[str(net)][f"{kind}.{label}"] += 1
    return dict(nodes)


def typed_pin_graph(obj: dict, symmetric_ports: bool = True) -> nx.Graph:
    graph = nx.Graph()
    for i, component in enumerate(obj["ckt_netlist"]):
        owner = ("component", i)
        graph.add_node(owner, label=component["component_type"])
        for port, net in component["port_connection"].items():
            pin = ("pin", i, port)
            wire = ("net", str(net))
            label = _port_label(component["component_type"], port, symmetric_ports)
            graph.add_node(pin, label=f"pin:{label}")
            graph.add_node(wire, label="NET")
            graph.add_edge(owner, pin)
            graph.add_edge(pin, wire)
    return graph


def compare(truth: dict, predicted: dict) -> dict:
    def compare_mode(symmetric_ports: bool) -> dict:
        a, b = node_signatures(truth, symmetric_ports), node_signatures(predicted, symmetric_ports)
        true_nets, pred_nets = list(a), list(b)
        cost = np.zeros((len(true_nets), len(pred_nets)), dtype=np.int32)
        for i, tn in enumerate(true_nets):
            for j, pn in enumerate(pred_nets):
                cost[i, j] = sum((a[tn] & b[pn]).values())
        if cost.size:
            ri, ci = linear_sum_assignment(cost, maximize=True)
            matched = int(cost[ri, ci].sum())
            pairs = [(true_nets[i], pred_nets[j], int(cost[i, j])) for i, j in zip(ri, ci)]
        else:
            matched, pairs = 0, []
        truth_ports = sum(sum(c.values()) for c in a.values())
        pred_ports = sum(sum(c.values()) for c in b.values())
        true_graph, pred_graph = typed_pin_graph(truth, symmetric_ports), typed_pin_graph(predicted, symmetric_ports)
        exact = nx.algorithms.isomorphism.GraphMatcher(
            true_graph, pred_graph, node_match=lambda left, right: left["label"] == right["label"]
        ).is_isomorphic()
        return {"matched": matched, "truth_ports": truth_ports, "predicted_ports": pred_ports,
                "pairs": pairs, "exact": exact,
                "f1": round(2 * matched / (truth_ports + pred_ports), 4) if truth_ports + pred_ports else 0}

    symmetric = compare_mode(True)
    strict = compare_mode(False)
    return {"matched_typed_pin_incidence": symmetric["matched"], "truth_pins": symmetric["truth_ports"],
            "predicted_pins": symmetric["predicted_ports"],
            "proxy_recall": round(symmetric["matched"] / symmetric["truth_ports"], 4) if symmetric["truth_ports"] else 0,
            "proxy_precision": round(symmetric["matched"] / symmetric["predicted_ports"], 4) if symmetric["predicted_ports"] else 0,
            "proxy_f1": symmetric["f1"], "true_nets": len(node_signatures(truth)),
            "predicted_nets": len(node_signatures(predicted)), "paired_nets": symmetric["pairs"],
            "exact_typed_pin_graph_isomorphic": symmetric["exact"],
            "strict_proxy_f1": strict["f1"],
            "strict_exact_typed_pin_graph_isomorphic": strict["exact"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=ROOT / "output" / "final")
    parser.add_argument("--case", type=str)
    args = parser.parse_args()
    paths = sorted(args.generated.glob("[0-9][0-9][0-9].txt"))
    if args.case:
        paths = [p for p in paths if p.stem == args.case]
    cases = []
    for path in paths:
        truth = ast.literal_eval((DATA / "true" / path.name).read_text(encoding="utf-8"))
        predicted = ast.literal_eval(path.read_text(encoding="utf-8"))
        cases.append({"case": path.stem, **compare(truth, predicted)})
    report = {"cases": cases,
              "mean_proxy_f1": round(sum(c["proxy_f1"] for c in cases) / len(cases), 4) if cases else 0,
              "exact_isomorphic_count": sum(c["exact_typed_pin_graph_isomorphic"] for c in cases),
              "mean_strict_proxy_f1": round(sum(c["strict_proxy_f1"] for c in cases) / len(cases), 4) if cases else 0,
              "strict_exact_isomorphic_count": sum(c["strict_exact_typed_pin_graph_isomorphic"] for c in cases),
              "metric_note": "proxy_f1 and exact use EDA's symmetric relation for Res/Cap/Ind/Switch; strict_* preserves Pos/Neg labels."}
    out = args.generated / (f"{args.case}_connection_diagnostic.json" if args.case else "connection_diagnostic.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"cases={len(cases)} mean_proxy_f1={report['mean_proxy_f1']:.4f} "
          f"exact={report['exact_isomorphic_count']}/{len(cases)}; saved {out}")
    if args.case and cases:
        for true_net, pred_net, overlap in cases[0]["paired_nets"]:
            print(f"{true_net:>12} ~ {pred_net:<8} shared typed pins={overlap}")


if __name__ == "__main__":
    main()
