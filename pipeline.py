"""Circuit image -> detected components -> wire graph -> DFS topology netlist.

The two YOLOv5 detector directories stay unmodified; this module loads them at
runtime and fuses their output before tracing the wiring.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from diagram_versions import draw_detection, draw_polarity, infer_polarity
from pixel_topology import trace as trace_pixel_topology, trust_summary


ROOT = Path(__file__).resolve().parent
MAIN_DETECTOR_ROOT = ROOT.parent / "data" / "circuit-dataset" / "main-detector"
MAIN_WEIGHTS = MAIN_DETECTOR_ROOT / "best-241022.pt"
MAIN_DATA_YAML = MAIN_DETECTOR_ROOT / "data" / "circuit.yaml"
AUX_DETECTOR_ROOT = ROOT.parent / "data" / "circuit-dataset" / "aux-detector"
AUX_WEIGHTS = AUX_DETECTOR_ROOT / "runs" / "train" / "exp17" / "weights" / "best.pt"
AUX_DATA_YAML = AUX_DETECTOR_ROOT / "data" / "circuit.yaml"
CLASS_TO_COMPONENT = {
    "resistor": ("R", "Res", ("Pos", "Neg")),
    "resistor2": ("R", "Res", ("Pos", "Neg")),
    "resistor2_3": ("R", "Res", ("Pos", "Neg")),
    "capacitor": ("C", "Cap", ("Pos", "Neg")),
    "capacitor-3": ("C", "Cap", ("Pos", "Neg")),
    "inductor": ("L", "Ind", ("Pos", "Neg")),
    "inductor-3": ("L", "Ind", ("Pos", "Neg")),
    "voltage": ("V", "Voltage", ("Positive", "Negative")),
    # The user's detector separates the battery-plate glyph from compact
    # circular voltage sources.  Both represent the same two-terminal device
    # in the target netlist format.
    "voltage-lines": ("V", "Voltage", ("Positive", "Negative")),
    "current": ("I", "Current", ("In", "Out")),
    "diode": ("D", "Diode", ("In", "Out")),
    "switch": ("S", "Switch", ("Pos", "Neg")),
    "nmos": ("M", "NMOS", ("Drain", "Gate", "Source")),
    "nmos-cross": ("M", "NMOS", ("Drain", "Gate", "Source")),
    "nmos-bulk": ("M", "NMOS", ("Drain", "Gate", "Source", "Body")),
    "pmos": ("M", "PMOS", ("Drain", "Gate", "Source")),
    "pmos-cross": ("M", "PMOS", ("Drain", "Gate", "Source")),
    "pmos-bulk": ("M", "PMOS", ("Drain", "Gate", "Source", "Body")),
    "npn": ("Q", "NPN", ("Collector", "Base", "Emitter")),
    "npn-cross": ("Q", "NPN", ("Collector", "Base", "Emitter")),
    "pnp": ("Q", "PNP", ("Collector", "Base", "Emitter")),
    "pnp-cross": ("Q", "PNP", ("Collector", "Base", "Emitter")),
    "single-end-amp": ("X", "Diso_amp", ("InN", "InP", "Out")),
    "diff-amp": ("X", "Dido_amp", ("InN", "InP", "OutN", "OutP")),
    "single-input-single-end-amp": ("X", "Siso_amp", ("In", "Out")),
}
NON_COMPONENT_CLASSES = {"port", "cross-line-curved", "vdd", "gnd", "antenna"}
Point = tuple[float, float]


def display_path(path: Path | str) -> str:
    """Repository-relative path for debug output, so artifacts stay portable."""
    resolved = Path(path).resolve()
    for base in (ROOT, ROOT.parent):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return Path(path).name


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def detect_with_original_yolo(image: np.ndarray, image_path: Path, out: Path) -> list[dict]:
    """Run the user's original 31-class YOLOv5 checkpoint and decode its labels."""
    if not MAIN_WEIGHTS.is_file() or not MAIN_DATA_YAML.is_file():
        raise FileNotFoundError(f"Original YOLO files missing: {MAIN_WEIGHTS} / {MAIN_DATA_YAML}")
    (ROOT / ".config").mkdir(exist_ok=True)
    os.environ["YOLOV5_CONFIG_DIR"] = str(ROOT / ".config")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(MAIN_DETECTOR_ROOT))
    from detect import run as run_original_yolo  # type: ignore

    save_dir = out / "debug" / "original_yolo"
    opt = SimpleNamespace(
        weights=[str(MAIN_WEIGHTS)], source=str(image_path), data=str(MAIN_DATA_YAML), imgsz=[640, 640],
        conf_thres=0.25, iou_thres=0.45, max_det=1000, device="cpu", view_img=False,
        save_txt=True, save_cls=False, save_conf=True, save_crop=False, nosave=True,
        classes=None, agnostic_nms=False, augment=False, visualize=False, update=False,
        project=str(save_dir), name="predict", exist_ok=True, line_thickness=2,
        hide_labels=True, hide_conf=True, half=False, dnn=False, vid_stride=1,
    )
    stale_label = save_dir / "predict" / "labels" / f"{image_path.stem}.txt"
    stale_label.unlink(missing_ok=True)
    prediction_dir = Path(run_original_yolo(**vars(opt)))
    return decode_yolo_labels(image, prediction_dir / "labels" / f"{image_path.stem}.txt")


def detect_with_auxiliary_yolo(image: np.ndarray, image_path: Path, out: Path) -> list[dict]:
    """Run the older checkpoint used only to resolve NPN/PNP disagreements."""
    if not AUX_WEIGHTS.is_file() or not AUX_DATA_YAML.is_file():
        return []
    save_dir = out / "debug" / "auxiliary_yolo"
    label = save_dir / "predict" / "labels" / f"{image_path.stem}.txt"
    label.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["YOLOV5_CONFIG_DIR"] = str(ROOT / ".config")
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    command = [
        sys.executable, str(AUX_DETECTOR_ROOT / "detect.py"), "--weights", str(AUX_WEIGHTS),
        "--source", str(image_path), "--data", str(AUX_DATA_YAML), "--imgsz", "640",
        "--conf-thres", "0.25", "--iou-thres", "0.45", "--save-txt", "--save-conf",
        "--nosave", "--project", str(save_dir), "--name", "predict", "--exist-ok",
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=environment)
    (save_dir / "detect.log").parent.mkdir(parents=True, exist_ok=True)
    (save_dir / "detect.log").write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Auxiliary YOLO failed; see {save_dir / 'detect.log'}")
    return decode_yolo_labels(image, label, AUX_DATA_YAML)


def fuse_bjt_polarity(primary: list[dict], auxiliary: list[dict]) -> list[dict]:
    """Use the auxiliary checkpoint only for overlapping NPN/PNP conflicts."""
    bjt = {"npn", "pnp", "npn-cross", "pnp-cross"}
    def iou(left: list[float], right: list[float]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0
    for component in primary:
        if component["class_name"] not in bjt:
            continue
        matches = [(iou(component["bbox"], candidate["bbox"]), candidate)
                   for candidate in auxiliary if candidate["class_name"] in bjt]
        if not matches:
            continue
        overlap, candidate = max(matches, key=lambda item: item[0])
        if overlap <= 0.75:
            continue
        primary_family = "npn" if component["class_name"].startswith("npn") else "pnp"
        auxiliary_family = "npn" if candidate["class_name"].startswith("npn") else "pnp"
        if primary_family == auxiliary_family:
            continue
        prefix, component_type, ports = CLASS_TO_COMPONENT[candidate["class_name"]]
        component.update(class_name=candidate["class_name"], kind=prefix,
                         component_type=component_type, port_names=list(ports), pin_count=len(ports))
    return primary


def fuse_mos_bulk_variants(primary: list[dict], auxiliary: list[dict]) -> list[dict]:
    """Promote plain MOS detections when the auxiliary model sees a bulk glyph."""
    def iou(left: list[float], right: list[float]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0

    for component in primary:
        if component["class_name"] not in {"nmos", "pmos"}:
            continue
        family = component["class_name"]
        matches = [(iou(component["bbox"], candidate["bbox"]), candidate)
                   for candidate in auxiliary
                   if candidate["class_name"] in {"nmos-bulk", "pmos-bulk"}
                   and candidate["class_name"].startswith(family)]
        if not matches:
            continue
        overlap, candidate = max(matches, key=lambda item: item[0])
        if overlap <= 0.75:
            continue
        prefix, component_type, ports = CLASS_TO_COMPONENT[candidate["class_name"]]
        component.update(class_name=candidate["class_name"], kind=prefix,
                         component_type=component_type, port_names=list(ports), pin_count=len(ports))
    return primary


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


# Both checkpoints read the same glyphs, so a box reported as NMOS by one model
# and as PMOS by the other means one of them mis-read the symbol arrow.  The
# auxiliary checkpoint is the tie-breaker, exactly like the BJT polarity rule.
MOS_POLARITY_OVERLAP = 0.75
# Only override a *low-confidence* primary box: when the primary is sure of its
# polarity it is right far more often than the auxiliary model is.
MOS_POLARITY_MAX_PRIMARY_CONFIDENCE = 0.6
# A glyph occasionally gets two boxes with different classes (a MOS reported as
# MOS *and* capacitor).  The richer one is the real part; a genuine small part
# sitting inside a bigger box never reaches this overlap.
DUPLICATE_BODY_OVERLAP = 0.8


def fuse_mos_polarity(primary: list[dict], auxiliary: list[dict]) -> list[dict]:
    """Resolve NMOS/PMOS conflicts between the two checkpoints."""
    for component in primary:
        name = component["class_name"]
        if not name.startswith(("nmos", "pmos")):
            continue
        if component.get("confidence", 1.0) > MOS_POLARITY_MAX_PRIMARY_CONFIDENCE:
            continue
        wanted = "pmos" if name.startswith("nmos") else "nmos"
        matches = [(_iou(component["bbox"], candidate["bbox"]), candidate)
                   for candidate in auxiliary
                   if candidate["class_name"].startswith(wanted)]
        if not matches:
            continue
        overlap, candidate = max(matches, key=lambda item: item[0])
        if overlap < MOS_POLARITY_OVERLAP:
            continue
        suffix = ""
        if "bulk" in name or "bulk" in candidate["class_name"]:
            suffix = "-bulk"
        elif "cross" in name or "cross" in candidate["class_name"]:
            suffix = "-cross"
        resolved = wanted + suffix
        prefix, component_type, ports = CLASS_TO_COMPONENT[resolved]
        component.update(class_name=resolved, kind=prefix, component_type=component_type,
                         port_names=list(ports), pin_count=len(ports))
    return primary


def suppress_duplicate_bodies(components: list[dict]) -> list[dict]:
    """Drop a part whose box coincides with a richer part of another class."""
    kept: list[dict] = []
    for component in components:
        if component.get("marker"):
            kept.append(component)
            continue
        duplicate = any(
            not other.get("marker")
            and other["component_type"] != component["component_type"]
            and other["pin_count"] > component["pin_count"]
            and _iou(other["bbox"], component["bbox"]) >= DUPLICATE_BODY_OVERLAP
            for other in components)
        if not duplicate:
            kept.append(component)
    return kept


def fuse_detections(primary: list[dict], auxiliary: list[dict]) -> list[dict]:
    """Apply every cross-checkpoint rule in the order the pipeline expects."""
    primary = fuse_bjt_polarity(primary, auxiliary)
    primary = fuse_mos_polarity(primary, auxiliary)
    primary = fuse_mos_bulk_variants(primary, auxiliary)
    return suppress_duplicate_bodies(primary)


def decode_yolo_labels(image: np.ndarray, label_path: Path, data_path: Path = MAIN_DATA_YAML) -> list[dict]:
    """Turn an original 31-class YOLOv5 label file into component records.

    Kept separate from the detector call so cached label files can rebuild the
    exact component list without paying for inference again.
    """
    if not label_path.exists():
        return []

    with data_path.open("r", encoding="utf-8") as stream:
        names = __import__("yaml").safe_load(stream)["names"]
    h, w = image.shape[:2]
    components: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) < 6:
            continue
        class_id, cx, cy, bw, bh, confidence = values[:6]
        class_name = names[int(class_id)]
        mapped = CLASS_TO_COMPONENT.get(class_name)
        marker = class_name in {"vdd", "gnd", "port", "cross-line-curved"}
        if mapped is None and not marker:
            continue
        prefix, component_type, ports = mapped if mapped else ("P", "", ())
        counts[prefix] += 1
        box_w, box_h = bw * w, bh * h
        components.append(
            {
                "ref": f"{prefix}{counts[prefix]}",
                "kind": prefix,
                "component_type": component_type,
                "class_name": class_name,
                "port_names": list(ports),
                "pin_count": len(ports),
                "bbox": [cx * w - box_w / 2, cy * h - box_h / 2,
                         cx * w + box_w / 2, cy * h + box_h / 2],
                "confidence": confidence,
                "marker": class_name if marker else None,
            }
        )

    # The detector can emit two mutually exclusive marker classes for the same
    # small symbol (observed for a -5 V terminal detected as both ``port`` and
    # ``gnd``).  Keep the higher-confidence marker when boxes almost coincide;
    # otherwise the false GND marker merges an independent supply with ground.
    def overlap_iou(left: list[float], right: list[float]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0

    marker_indices = sorted((index for index, item in enumerate(components) if item.get("marker")),
                            key=lambda index: components[index]["confidence"], reverse=True)
    suppressed: set[int] = set()
    for position, index in enumerate(marker_indices):
        if index in suppressed:
            continue
        for other in marker_indices[position + 1:]:
            if other in suppressed:
                continue
            if overlap_iou(components[index]["bbox"], components[other]["bbox"]) >= 0.85:
                suppressed.add(other)
    return [item for index, item in enumerate(components) if index not in suppressed]


def detect_objects(image: np.ndarray, image_path: Path, out: Path) -> tuple[list[dict], list[Point]]:
    """Use the user's trained 31-class YOLO model for component symbols."""
    components = detect_with_original_yolo(image, image_path, out)
    auxiliary = detect_with_auxiliary_yolo(image, image_path, out)
    return fuse_detections(components, auxiliary), []


def key(point: Point) -> Point:
    return (round(point[0], 2), round(point[1], 2))


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segment_intersection(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    """Finite straight-segment intersection; parallel overlaps handled separately."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    denominator = rx * sy - ry * sx
    if abs(denominator) < 1e-8:
        return None
    qx, qy = c[0] - a[0], c[1] - a[1]
    t = (qx * sy - qy * sx) / denominator
    u = (qx * ry - qy * rx) / denominator
    if -1e-6 <= t <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6:
        return key((a[0] + t * rx, a[1] + t * ry))
    return None


def point_on_segment(p: Point, a: Point, b: Point, tolerance: float = 1.5) -> bool:
    length = distance(a, b)
    if length < 1e-8:
        return distance(p, a) <= tolerance
    cross = abs((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])) / length
    dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    return cross <= tolerance and -tolerance * length <= dot <= length * length + tolerance * length


def rectangle_intersections(a: Point, b: Point, bbox: list[float]) -> list[Point]:
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    result = []
    for c, d in zip(corners, corners[1:] + corners[:1]):
        p = segment_intersection(a, b, c, d)
        if p is not None and all(distance(p, other) > 1 for other in result):
            result.append(p)
    return result


def inside_box(point: Point, bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 + 0.1 < point[0] < x2 - 0.1 and y1 + 0.1 < point[1] < y2 - 0.1


def build_topology(components: list[dict], wire_lines: list[tuple[Point, Point]]) -> dict:
    """Split at intersections and boxes, then DFS each electrical wire net."""
    splits = [{key(a), key(b)} for a, b in wire_lines]
    crossings = set()
    terminals: dict[str, list[Point]] = defaultdict(list)

    for i, j in itertools.combinations(range(len(wire_lines)), 2):
        a, b = wire_lines[i]
        c, d = wire_lines[j]
        p = segment_intersection(a, b, c, d)
        if p is not None:
            splits[i].add(p)
            splits[j].add(p)
            if all(not inside_box(p, comp["bbox"]) for comp in components):
                crossings.add(p)
        # Collinear candidate lines can overlap without sharing an endpoint.
        for p in (a, b):
            if point_on_segment(p, c, d):
                splits[j].add(key(p))
        for p in (c, d):
            if point_on_segment(p, a, b):
                splits[i].add(key(p))

    for i, (a, b) in enumerate(wire_lines):
        for comp in components:
            if comp.get("marker"):
                continue
            for p in rectangle_intersections(a, b, comp["bbox"]):
                splits[i].add(p)
                if all(distance(p, q) > 3 for q in terminals[comp["ref"]]):
                    terminals[comp["ref"]].append(p)

    graph: dict[Point, set[Point]] = defaultdict(set)
    for (a, b), points in zip(wire_lines, splits):
        dx, dy = b[0] - a[0], b[1] - a[1]
        points_sorted = sorted(points, key=lambda p: (p[0] - a[0]) * dx + (p[1] - a[1]) * dy)
        for p, q in zip(points_sorted, points_sorted[1:]):
            if distance(p, q) < 0.5:
                continue
            midpoint = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            if any(inside_box(midpoint, comp["bbox"]) for comp in components):
                continue
            graph[p].add(q)
            graph[q].add(p)

    warnings = []
    for comp in components:
        anchors = terminals[comp["ref"]]
        if comp.get("marker"):
            continue
        required = comp.get("pin_count", 2)
        if len(anchors) >= required:
            # Keep this legacy geometry helper usable with the compact test
            # fixtures, which intentionally omit detector-only metadata.
            cls = comp.get("class_name", "")
            remaining = list(anchors)
            x1, y1, x2, y2 = comp["bbox"]
            if cls.startswith(("nmos", "pmos")):
                gate = min(remaining, key=lambda p: p[0]); remaining.remove(gate)
                vertical = sorted(remaining, key=lambda p: p[1])
                pin_map = ({"Gate": gate, "Source": vertical[0], "Drain": vertical[-1]} if cls.startswith("pmos")
                           else {"Gate": gate, "Drain": vertical[0], "Source": vertical[-1]})
                if "bulk" in cls and len(vertical) >= 3:
                    pin_map["Body"] = max(vertical, key=lambda p: p[0])
            elif cls.startswith(("npn", "pnp")):
                base = min(remaining, key=lambda p: p[0]); remaining.remove(base)
                vertical = sorted(remaining, key=lambda p: p[1])
                pin_map = {"Base": base, "Collector": vertical[0], "Emitter": vertical[-1]}
            elif cls == "single-end-amp":
                left = sorted((p for p in remaining if p[0] < (x1+x2)/2), key=lambda p:p[1])
                right = [p for p in remaining if p[0] >= (x1+x2)/2]
                pin_map = {"InN": left[0], "InP": left[-1], "Out": right[0]}
            elif cls == "diff-amp":
                left = sorted((p for p in remaining if p[0] < (x1+x2)/2), key=lambda p:p[1])
                right = sorted((p for p in remaining if p[0] >= (x1+x2)/2), key=lambda p:p[1])
                pin_map = {"InN": left[0], "InP": left[-1], "OutN": right[0], "OutP": right[-1]}
            else:
                port_names = comp.get("port_names", [f"pin{i}" for i in range(required)])
                pin_map = dict(zip(port_names, sorted(remaining, key=lambda p:(p[1],p[0]))))
            port_names = comp.get("port_names", [f"pin{i}" for i in range(required)])
            assigned = [pin_map[name] for name in port_names if name in pin_map]
            terminals[comp["ref"]] = assigned if len(assigned) == required else anchors
            if not terminals[comp["ref"]]:
                warnings.append(f"{comp['ref']}: unable to map {required} detected contacts to named ports")
            elif len(anchors) > required:
                warnings.append(f"{comp['ref']}: selected {required} of {len(anchors)} candidate contacts")
        else:
            warnings.append(f"{comp['ref']}: only {len(anchors)} of {required} contacts found")
            # Preserve contacts for debugging and the legacy geometry API even
            # when the component cannot be emitted to the netlist.
            terminals[comp["ref"]] = anchors
        for anchor in terminals[comp["ref"]]:
            graph.setdefault(anchor, set())

    # DFS: every point reachable through wire edges belongs to the same net.
    node_to_net: dict[Point, str] = {}
    nets: dict[str, list[list[float]]] = {}
    for start in sorted(graph):
        if start in node_to_net:
            continue
        net = f"N{len(nets) + 1:03d}"
        stack = [start]
        points = []
        node_to_net[start] = net
        while stack:
            current = stack.pop()
            points.append(current)
            for neighbor in sorted(graph[current]):
                if neighbor not in node_to_net:
                    node_to_net[neighbor] = net
                    stack.append(neighbor)
        nets[net] = [[float(x), float(y)] for x, y in sorted(points)]

    # EDA rules merge every VDD marker and every GND marker into a shared net.
    for marker_name, canonical in (("vdd", "VDD"), ("gnd", "VSS")):
        marker_nets = set()
        graph_points = list(node_to_net)
        for marker in (c for c in components if c.get("marker") == marker_name):
            x1, y1, x2, y2 = marker["bbox"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            nearest = min(graph_points, key=lambda p: distance(p, center), default=None)
            if nearest is not None and distance(nearest, center) <= 45:
                marker_nets.add(node_to_net[nearest])
        if marker_nets:
            for point, net in list(node_to_net.items()):
                if net in marker_nets:
                    node_to_net[point] = canonical
            merged_points = [p for net in marker_nets for p in nets.pop(net, [])]
            nets[canonical] = merged_points

    component_rows = []
    net_to_components: dict[str, set[str]] = defaultdict(set)
    for comp in components:
        anchors = terminals[comp["ref"]]
        node_names = [node_to_net[p] for p in anchors] if not comp.get("marker") else []
        for node in node_names:
            net_to_components[node].add(comp["ref"])
        component_rows.append({**comp, "contacts": [[*p] for p in anchors], "nets": node_names})

    junction_rows = []
    for p in sorted(crossings):
        if p not in node_to_net or len(graph[p]) < 3:
            continue
        net = node_to_net[p]
        junction_rows.append({"ref": f"J{len(junction_rows) + 1}", "point": [*p], "net": net,
                              "connected_components": sorted(net_to_components[net])})

    component_connections = sorted({tuple(sorted(pair)) for refs in net_to_components.values()
                                    for pair in itertools.combinations(refs, 2)})
    return {"components": component_rows, "nets": nets, "junctions": junction_rows,
            "connected_component_pairs": [list(pair) for pair in component_connections],
            "warnings": warnings}


def write_debug(image: np.ndarray, result: dict, wire_lines: list[tuple[Point, Point]], path: Path,
                skeleton: np.ndarray | None = None) -> None:
    canvas = image.copy()
    if skeleton is not None:
        canvas[skeleton > 0] = (255, 80, 0)
    for a, b in wire_lines:
        cv2.line(canvas, tuple(map(round, a)), tuple(map(round, b)), (255, 0, 0), 2)
    for comp in result["components"]:
        x1, y1, x2, y2 = map(round, comp["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, comp["ref"], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        for p, node in zip(comp["contacts"], comp["nets"]):
            cv2.circle(canvas, tuple(map(round, p)), 4, (0, 160, 0), -1)
            cv2.putText(canvas, node, tuple(map(round, p)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 100, 0), 1)
    for junction in result["junctions"]:
        p = tuple(map(round, junction["point"]))
        cv2.circle(canvas, p, 6, (0, 165, 255), 2)
    cv2.imencode(".png", canvas)[1].tofile(str(path))


def netlist_rows(result: dict) -> list[dict]:
    """Netlist rows: only components whose every named port got a net."""
    rows = []
    for comp in result["components"]:
        if comp.get("marker") or len(comp.get("port_nets", {})) != comp.get("pin_count", 2):
            continue
        rows.append({"component_type": comp["component_type"],
                     "port_connection": comp["port_nets"]})
    return rows


def apply_component_aliases(result: dict, aliases: dict[str, str]) -> None:
    """Replace detector-order references with verified schematic labels.

    EDA netlist rows deliberately contain only component type and ports, but
    the readable topology artefacts contain references.  An optional mapping
    lets a user preserve labels printed on a source schematic after verifying
    their spatial correspondence.
    """
    existing = {comp["ref"] for comp in result["components"] if not comp.get("marker")}
    unknown = sorted(set(aliases) - existing)
    duplicate_targets = sorted(target for target, count in Counter(aliases.values()).items() if count > 1)
    if unknown or duplicate_targets:
        detail = []
        if unknown:
            detail.append(f"unknown detector refs: {', '.join(unknown)}")
        if duplicate_targets:
            detail.append(f"duplicate aliases: {', '.join(duplicate_targets)}")
        raise ValueError("Invalid component aliases; " + "; ".join(detail))
    for comp in result["components"]:
        if not comp.get("marker"):
            comp["ref"] = aliases.get(comp["ref"], comp["ref"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a circuit image to a topology netlist")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--loss-threshold", type=float, default=430000)
    parser.add_argument("--labels", type=Path,
                        help="Reuse a cached YOLO label file instead of running detector inference")
    parser.add_argument("--aux-labels", type=Path,
                        help="Cached auxiliary YOLO labels used for NPN/PNP conflict resolution")
    parser.add_argument("--aux-label-data", type=Path, default=AUX_DATA_YAML,
                        help="Class YAML corresponding to --aux-labels")
    parser.add_argument("--component-aliases", type=Path,
                        help="JSON mapping from detector refs to verified source labels, for readable artefacts")
    args = parser.parse_args()
    image_path = args.image.resolve()
    out = (args.out or ROOT / "output" / image_path.stem).resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = read_image(image_path)
    scale = 640 / source.shape[1]
    image = cv2.resize(source, (640, round(source.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if args.labels:
        components = decode_yolo_labels(image, args.labels.resolve())
        if args.aux_labels:
            auxiliary = decode_yolo_labels(image, args.aux_labels.resolve(), args.aux_label_data.resolve())
            # Same fusion chain as the live detector path, otherwise a cached
            # batch run would silently miss rules such as the MOS polarity
            # arbitration and the duplicate-glyph suppression.
            components = fuse_detections(components, auxiliary)
        junctions = []
        cached_label = out / "debug" / "original_yolo" / "predict" / "labels" / f"{image_path.stem}.txt"
        cached_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.labels.resolve(), cached_label)
        if args.aux_labels:
            auxiliary_label = out / "debug" / "auxiliary_yolo" / "predict" / "labels" / f"{image_path.stem}.txt"
            auxiliary_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.aux_labels.resolve(), auxiliary_label)
    else:
        components, junctions = detect_objects(image, image_path, out)
    if not components:
        raise RuntimeError("YOLOv5 found no components; cannot produce a trustworthy netlist")
    result, skeleton, lines = trace_pixel_topology(image, components)
    if args.component_aliases:
        aliases = json.loads(args.component_aliases.read_text(encoding="utf-8"))
        if not isinstance(aliases, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                                    for key, value in aliases.items()):
            raise ValueError("--component-aliases must be a JSON object of string-to-string mappings")
        apply_component_aliases(result, aliases)
    line_stats = {"method": "threshold+skeleton+8-connected-components", "skeleton_pixels": result["skeleton_pixels"],
                  "wire_components": result["wire_components"], "accepted": result["wire_components"]}
    result.update({"source_image": display_path(image_path), "line_stats": line_stats,
                   "junction_seed_count": len(junctions), "wire_segments": [[list(a), list(b)] for a, b in lines]})

    draw_detection(image, components, junctions, lines, out / "01_detection.png", skeleton)
    debug_dir = out / "debug"
    debug_dir.mkdir(exist_ok=True)
    wire_canvas = image.copy()
    wire_canvas[skeleton > 0] = (255, 80, 0)
    cv2.imencode(".png", wire_canvas)[1].tofile(str(debug_dir / "skeleton_overlay.png"))

    # The full net pixel lists dominate this file (a single case reaches 165 KB)
    # and are only needed when re-checking the geometry, so the review copy keeps
    # the readable fields and the coordinates move next to it.
    pixels = result.get("nets", {})
    compact = {key: value for key, value in result.items() if key != "nets"}
    compact["trust"] = trust_summary(result)
    compact["nets"] = {
        name: {"pixel_count": len(points),
               "bbox": [round(min(p[0] for p in points), 1), round(min(p[1] for p in points), 1),
                        round(max(p[0] for p in points), 1), round(max(p[1] for p in points), 1)]
               if points else None,
               "terminals": sorted({f"{comp['ref']}.{port}"
                                    for comp in result["components"]
                                    if not comp.get("marker")
                                    for port, net in (comp.get("port_nets") or {}).items()
                                    if net == name})}
        for name, points in pixels.items()
    }
    (debug_dir / "topology.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    (debug_dir / "topology_pixels.json").write_text(
        json.dumps({"nets": pixels}, ensure_ascii=False), encoding="utf-8")
    (out / "topology.json").unlink(missing_ok=True)
    lines_out = ["* EDA named-port topology (heuristic image extraction)"]
    for comp in result["components"]:
        if comp.get("marker"):
            continue
        if len(comp.get("port_nets", {})) == comp.get("pin_count", 2):
            lines_out.append(f"{comp['ref']} {comp['port_nets']}")
        elif comp.get("pin_count", 2) != 2:
            lines_out.append(f"* {comp['ref']} omitted: named ports could not be mapped")
        else:
            lines_out.append(f"* {comp['ref']} omitted: two contacts were not found")
    for junction in result["junctions"]:
        lines_out.append(f"* {junction['ref']} at {junction['point']} on {junction['net']}")
    (out / "topology.net").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    write_debug(image, result, lines, out / "overlay.png", skeleton)
    shutil.copyfile(out / "overlay.png", out / "02_connected_nodes.png")
    polarity = infer_polarity(image, result)
    (out / "polarity.json").write_text(json.dumps(polarity, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_polarity(image, result, lines, polarity, out / "03_polarity.png", skeleton)
    # Output format: an ast.literal_eval-compatible Python dict string.
    netlist = {"ckt_netlist": netlist_rows(result), "ckt_type": "DISO-Amplifier"}
    try:
        from eda_gcn import predict as classify_circuit
        circuit_type, scores = classify_circuit(netlist)
        netlist["ckt_type"] = circuit_type
        (out / "ckt_type_scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    except (FileNotFoundError, RuntimeError):
        netlist["ckt_type"] = "DISO-Amplifier"
        (out / "ckt_type_scores.json").write_text(json.dumps({"error": "train eda_gcn.py first"}), encoding="utf-8")
    (out / "netlist.txt").write_text(str(netlist), encoding="utf-8")
    print(f"Components: {len(components)}; skeleton pixels: {line_stats['skeleton_pixels']}; wire components: {line_stats['wire_components']}")
    print(f"Nets: {len(result['nets'])}; crossings: {len(result['junctions'])}")
    for line in lines_out:
        print(line)
    for warning in result["warnings"]:
        print("WARNING:", warning)
    for node in polarity["nodes"]:
        print(f"NODE: {node['id']} {node['polarity']}")
    print("Saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
