"""Small graph-convolution classifier for circuit-function classes.

Training examples are the supplied reference netlists. A circuit becomes a graph
whose nodes are the components and the electrical nets; component-to-net edges
carry the port role, and supply nets are flagged because VDD/GND are recognised
targets of this task rather than free-form net names.

Two numbers are reported. The first is repeated stratified cross-validation on
the reference netlists, because 40 circuits are far too few for a single 5-fold
split to be stable. The second feeds the pipeline's own generated netlists to
the same held-out models, which is the honest end-to-end recognition accuracy.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "circuit-dataset"
CLASSES = ["Bandgap", "Comparator", "DIDO-Amplifier", "DISO-Amplifier", "LDO", "SISO-Amplifier"]
COMPONENTS = ["PMOS", "NMOS", "PNP", "NPN", "Res", "Cap", "Ind", "Diode", "Switch", "Current", "Voltage", "Diso_amp", "Dido_amp", "Siso_amp"]
PORTS = ["Drain", "Source", "Gate", "Body", "Collector", "Base", "Emitter", "Pos", "Neg", "In", "Out", "Positive", "Negative", "InN", "InP", "OutN", "OutP", "VDD", "VSS"]
SUPPLY_NETS = {"VDD", "VSS", "GND"}

IS_COMPONENT = len(COMPONENTS)
PORT_BASE = IS_COMPONENT + 1
RATIO = PORT_BASE + len(PORTS)
IS_VDD = RATIO + 1
IS_VSS = RATIO + 2
FEATURES = IS_VSS + 1
GLOBAL_FEATURES = len(COMPONENTS) + 4


def graph_from_netlist(netlist: dict) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Return node features, normalized adjacency (self loops), component count, graph features."""
    rows = netlist.get("ckt_netlist", [])
    names = sorted({str(net) for row in rows for net in row.get("port_connection", {}).values()})
    net_index = {name: len(rows) + i for i, name in enumerate(names)}
    x = torch.zeros((len(rows) + len(names), FEATURES), dtype=torch.float32)
    edges: list[tuple[int, int]] = []
    graph = torch.zeros(GLOBAL_FEATURES, dtype=torch.float32)
    for i, row in enumerate(rows):
        kind = row.get("component_type", "")
        if kind in COMPONENTS:
            x[i, COMPONENTS.index(kind)] = 1
            graph[COMPONENTS.index(kind)] += 1
        x[i, IS_COMPONENT] = 1
        ports = row.get("port_connection", {})
        for port, net in ports.items():
            j = net_index[str(net)]
            if port in PORTS:
                x[j, PORT_BASE + PORTS.index(port)] = 1
            edges.extend(((i, j), (j, i)))
        x[i, RATIO] = len(ports) / 4
    for name, j in net_index.items():
        degree = sum(1 for source, _ in edges if source == j)
        x[j, RATIO] = degree / max(1, len(rows))
        x[j, IS_VDD] = 1 if name == "VDD" else 0
        x[j, IS_VSS] = 1 if name in SUPPLY_NETS and name != "VDD" else 0
    graph[len(COMPONENTS):] = torch.tensor([
        len(rows) / 20,
        len(names) / 20,
        sum(len(row.get("port_connection", {})) for row in rows) / max(1, 4 * len(rows)),
        float(any(name in SUPPLY_NETS for name in net_index)),
    ])
    n = len(x)
    adj = torch.eye(n, dtype=torch.float32)
    for a, b in edges:
        adj[a, b] = 1
    degree = adj.sum(1).clamp_min(1).rsqrt()
    adj = degree[:, None] * adj * degree[None, :]
    return x, adj, len(rows), graph


class GCN(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(FEATURES, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + GLOBAL_FEATURES, hidden), nn.ReLU(),
                                  nn.Dropout(.2), nn.Linear(hidden, len(CLASSES)))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, component_count: int,
                graph: torch.Tensor) -> torch.Tensor:
        h = F.relu(adj @ self.lin1(x))
        h = F.relu(adj @ self.lin2(h))
        comp = h[:component_count]
        nets = h[component_count:]
        comp_pool = comp.mean(0) if len(comp) else h.new_zeros(h.shape[-1])
        net_pool = nets.mean(0) if len(nets) else h.new_zeros(h.shape[-1])
        return self.head(torch.cat([comp_pool, net_pool, graph]))


def read_dataset() -> list[tuple[dict, int, str]]:
    samples = []
    for path in sorted((DATA / "true").glob("*.txt")):
        obj = ast.literal_eval(path.read_text(encoding="utf-8"))
        label = str(obj["ckt_type"]).strip()
        if label in CLASSES:
            samples.append((obj, CLASSES.index(label), path.stem))
    return samples


def read_generated(directory: Path | None) -> dict[str, dict]:
    generated: dict[str, dict] = {}
    if directory and directory.is_dir():
        for path in sorted(directory.glob("[0-9][0-9][0-9].txt")):
            generated[path.stem] = ast.literal_eval(path.read_text(encoding="utf-8"))
    return generated


def class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=len(CLASSES)).astype(np.float32)
    weights = np.where(counts > 0, len(labels) / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def fit(graphs: list[tuple], ids: list[int], epochs: int, seed: int | None = None,
        weighted: bool = True) -> GCN:
    if seed is not None:
        torch.manual_seed(seed)
    model = GCN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.003)
    weights = class_weights([graphs[i][1] for i in ids]) if weighted else None
    for _ in range(epochs):
        random.shuffle(ids)
        model.train()
        for i in ids:
            (x, adj, count, graph), y, _ = graphs[i]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x, adj, count, graph).unsqueeze(0),
                                   torch.tensor([y]), weight=weights)
            loss.backward(); optimizer.step()
    return model


def apply_transform(graphs: list[tuple], transform) -> list[tuple]:
    return graphs if transform is None else [(transform(graph), label, name) for graph, label, name in graphs]


def stratified_folds(samples: list, repeats: int, folds: int, seed: int) -> list[list[list[int]]]:
    """Repeat-wise stratified fold assignment over the labelled circuits."""
    rounds = []
    for repeat in range(repeats):
        assignment: list[list[int]] = [[] for _ in range(folds)]
        for cls in range(len(CLASSES)):
            indices = [i for i, (_, y, _) in enumerate(samples) if y == cls]
            random.Random(seed + 1000 * repeat + cls).shuffle(indices)
            for position, index in enumerate(indices):
                assignment[position % folds].append(index)
        rounds.append(assignment)
    return rounds


def cross_validate(graphs: list[tuple], generated_graphs: dict, epochs: int, repeats: int,
                   folds: int, seed: int, weighted: bool = True, transform=None) -> dict:
    """Held-out scores over repeated stratified folds.

    Every evaluated circuit is scored by a model that never saw its label.
    """
    graphs = apply_transform(graphs, transform)
    if transform is not None:
        generated_graphs = {name: transform(graph) for name, graph in generated_graphs.items()}
    cv_scores: list[float] = []
    end_to_end_scores: list[float] = []
    fold_predictions: list[dict] = []
    for repeat, assignment in enumerate(stratified_folds(graphs, repeats, folds, seed)):
        pred = [-1] * len(graphs)
        generated_pred: dict[str, int] = {}
        for fold, val_ids in enumerate(assignment):
            held_out = set(val_ids)
            train_ids = [i for i in range(len(graphs)) if i not in held_out]
            model = fit(graphs, train_ids, epochs, seed=seed + 100 * repeat + fold, weighted=weighted)
            model.eval()
            with torch.no_grad():
                for i in val_ids:
                    pred[i] = int(model(*graphs[i][0]).argmax())
                    name = graphs[i][2]
                    if name in generated_graphs:
                        generated_pred[name] = int(model(*generated_graphs[name]).argmax())
        cv_scores.append(sum(int(p == graphs[i][1]) for i, p in enumerate(pred)) / len(pred))
        matched = [i for i, (_, _, name) in enumerate(graphs) if name in generated_pred]
        if matched:
            end_to_end_scores.append(sum(int(generated_pred[graphs[i][2]] == graphs[i][1])
                                         for i in matched) / len(matched))
        fold_predictions.extend({"image": graphs[i][2], "true": CLASSES[graphs[i][1]],
                                 "predicted": CLASSES[pred[i]], "repeat": repeat + 1}
                                for i in range(len(graphs)))
    return {"cv_scores": cv_scores, "end_to_end_scores": end_to_end_scores,
            "fold_predictions": fold_predictions}


def train(epochs: int = 140, seed: int = 17, repeats: int = 5, folds: int = 5,
          generated_dir: Path | None = None, weighted: bool = True, transform=None) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    samples = read_dataset()
    if len(samples) < 6:
        raise RuntimeError("Need labelled reference netlists to train the GCN")
    graphs = [(graph_from_netlist(obj), label, name) for obj, label, name in samples]
    generated = read_generated(generated_dir)
    generated_graphs = {name: graph_from_netlist(obj) for name, obj in generated.items()}
    scores = cross_validate(graphs, generated_graphs, epochs, repeats, folds, seed,
                            weighted=weighted, transform=transform)
    cv_scores = scores["cv_scores"]
    end_to_end_scores = scores["end_to_end_scores"]
    fold_predictions = scores["fold_predictions"]

    final_graphs = apply_transform(graphs, transform)
    model = fit(final_graphs, list(range(len(final_graphs))), epochs, seed=seed, weighted=weighted)
    artifact = ROOT / "models" / "eda_gcn.pt"
    artifact.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "components": COMPONENTS,
                "ports": PORTS, "features": FEATURES, "global_features": GLOBAL_FEATURES,
                "supply_nets": sorted(SUPPLY_NETS), "seed": seed, "epochs": epochs,
                "train_count": len(graphs)}, artifact)

    mean_cv, std_cv = round(float(np.mean(cv_scores)), 4), round(float(np.std(cv_scores)), 4)
    mean_e2e, std_e2e = (round(float(np.mean(end_to_end_scores)), 4),
                         round(float(np.std(end_to_end_scores)), 4)) if end_to_end_scores else (None, None)
    report = {
        "samples": len(graphs),
        "class_counts": {name: sum(y == i for _, y, _ in graphs) for i, name in enumerate(CLASSES)},
        "repeats": repeats,
        "folds": folds,
        "class_weighted": weighted,
        "ablation": getattr(transform, "__name__", "full"),
        "stratified_cv_accuracy": mean_cv,
        "stratified_cv_accuracy_std": std_cv,
        "stratified_cv_per_repeat": [round(score, 4) for score in cv_scores],
        "generated_netlist_accuracy": mean_e2e,
        "generated_netlist_accuracy_std": std_e2e,
        "generated_netlist_cases": len(generated),
        "fold_predictions": fold_predictions,
        "note": ("Only 40 public labelled circuits. Cross-validated models never see the evaluated "
                 "circuit; the generated-netlist score reuses those held-out models on the "
                 "pipeline's own output."),
        "model": artifact.relative_to(ROOT).as_posix(),
    }
    (ROOT / "models" / "eda_gcn_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def predict(obj: dict, model_path: Path | None = None) -> tuple[str, dict[str, float]]:
    path = model_path or ROOT / "models" / "eda_gcn.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GCN(); model.load_state_dict(payload["state_dict"]); model.eval()
    x, adj, count, graph = graph_from_netlist(obj)
    with torch.no_grad():
        prob = model(x, adj, count, graph).softmax(0).numpy()
    scores = {name: float(prob[i]) for i, name in enumerate(CLASSES)}
    label = max(scores, key=scores.get)
    return label, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--generated", type=Path, default=ROOT / "output" / "final")
    args = parser.parse_args()
    print(json.dumps(train(args.epochs, repeats=args.repeats, generated_dir=args.generated),
                     indent=2, ensure_ascii=False))
