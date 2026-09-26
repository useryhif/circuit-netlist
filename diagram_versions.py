"""Create three reviewable circuit-image stages and conservative polarity labels."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def _xy(point: list[float] | tuple[float, float], y_offset: int = 0) -> tuple[int, int]:
    return round(point[0]), round(point[1]) + y_offset


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0


def _put_label(canvas: np.ndarray, text: str, colour: tuple[int, int, int],
               anchors: list[tuple], placed: list[tuple[int, int, int, int]],
               scale: float = .4) -> None:
    """Draw a label at the first anchor that keeps it clear of the other labels."""
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    height, width = canvas.shape[:2]
    best, best_score = None, None
    for anchor in anchors:
        x, y = int(anchor[0]), int(anchor[1])
        if len(anchor) > 2 and anchor[2] == "right":
            x -= text_width
        x = min(max(2, x), max(2, width - text_width - 2))
        y = min(max(text_height + 2, y), height - 3)
        box = (x - 1, y - text_height - 1, x + text_width + 1, y + 1)
        score = sum(_overlap_area(box, other) for other in placed)
        if score == 0:
            best, best_score = (x, y, box), 0
            break
        if best_score is None or score < best_score:
            best, best_score = (x, y, box), score
    x, y, box = best
    placed.append(box)
    # A white halo keeps the label readable on top of dense line art.
    cv2.putText(canvas, text, (x, y), font, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def draw_detection(image: np.ndarray, components: list[dict], seeds: list[tuple[float, float]],
                   wire_lines: list[tuple[tuple[float, float], tuple[float, float]]], path: Path,
                   wire_mask: np.ndarray | None = None) -> None:
    """Version 1: detected objects and accepted line segments, before DFS labels."""
    canvas = image.copy()
    if wire_mask is not None:
        canvas[wire_mask > 0] = (255, 80, 0)
    for a, b in wire_lines:
        cv2.line(canvas, _xy(a), _xy(b), (255, 0, 0), 2)
    for point in seeds:
        cv2.circle(canvas, _xy(point), 3, (0, 180, 255), -1)
    placed_labels: list[tuple[int, int, int, int]] = []
    for comp in components:
        x1, y1, x2, y2 = map(round, comp["bbox"])
        # Markers (supply/port glyphs) are drawn in a different colour together
        # with their confidence, so the detection stage shows what the model
        # actually saw next to every box.
        colour = (255, 0, 200) if comp.get("marker") else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label = f"{comp['ref']} {comp['class_name']} {comp.get('confidence', 0.0):.2f}"
        (_, line_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)
        _put_label(canvas, label, colour, [
            (x1, y1 - 4),                        # above the box, left aligned
            (x2, y1 - 4, "right"),               # above the box, right aligned
            (x1, y2 + line_height + 5),          # below the box
            (x2, y2 + line_height + 5, "right"),
            (x2 + 4, y1 + line_height + 2),      # right of the box
            (x1 - 4, y1 + line_height + 2, "right"),
            (x1, y1 - line_height - 8),          # one line higher
            (x1, y2 + 2 * line_height + 10),     # two lines lower
        ], placed_labels)
    _save_png(path, canvas)


def _stroke_score(binary: np.ndarray, low: float, high: float) -> tuple[int, int, int]:
    h, w = binary.shape
    y1, y2 = round(low * h), round(high * h)
    x1, x2 = round(.25 * w), round(.75 * w)
    region = binary[y1:y2, x1:x2]
    if region.size == 0:
        return 0, 0, 0
    vertical = int(region.sum(axis=0).max())
    horizontal = int(region.sum(axis=1).max())
    return vertical, horizontal, region.shape[0]


def detect_voltage_sign(image: np.ndarray, comp: dict) -> dict:
    """Find the half of a source symbol containing '+', using its vertical stroke.

    This is deliberately conservative: low contrast or similar scores on both
    sides leave the sign unknown.  The result is a visual heuristic, not OCR.
    """
    contacts = comp.get("contacts", [])
    if len(contacts) != 2:
        return {"plus_index": None, "method": "unknown", "reason": "two contacts not found"}
    x1, y1, x2, y2 = map(round, comp["bbox"])
    h, w = image.shape[:2]
    crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return {"plus_index": None, "method": "unknown", "reason": "empty source crop"}
    vertical_source = abs(contacts[0][1] - contacts[1][1]) >= abs(contacts[0][0] - contacts[1][0])
    if not vertical_source:
        # After clockwise rotation, the upper half corresponds to the original left half.
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    binary = (gray < 220).astype(np.uint8)
    top_v, top_h, top_size = _stroke_score(binary, .20, .48)
    bottom_v, bottom_h, bottom_size = _stroke_score(binary, .52, .80)
    difference = abs(top_v - bottom_v)
    enough_horizontal = min(top_h, bottom_h) >= max(4, round(binary.shape[1] * .08))
    threshold = max(2, .12 * max(top_size, bottom_size))
    if not enough_horizontal or difference < threshold:
        return {"plus_index": None, "method": "unknown",
                "reason": "plus/minus strokes not distinct enough",
                "stroke_scores": {"upper": top_v, "lower": bottom_v}}
    plus_in_upper_half = top_v > bottom_v
    if vertical_source:
        plus_index = min(range(2), key=lambda i: contacts[i][1]) if plus_in_upper_half else max(
            range(2), key=lambda i: contacts[i][1])
    else:
        plus_index = min(range(2), key=lambda i: contacts[i][0]) if plus_in_upper_half else max(
            range(2), key=lambda i: contacts[i][0])
    return {"plus_index": plus_index, "method": "visual_plus_minus_strokes",
            "reason": "vertical stroke is stronger on the plus-sign side",
            "stroke_scores": {"upper": top_v, "lower": bottom_v}}


def infer_polarity(image: np.ndarray, result: dict) -> dict:
    """Assign each electrical net one label, then inherit it at every terminal."""
    components = result["components"]
    source_count = sum(comp["kind"] == "V" for comp in components)
    source_signs: dict[str, list[str]] = {net: [] for net in result["nets"]}
    for comp in components:
        if comp["kind"] != "V" or len(comp.get("nets", [])) != 2:
            continue
        plus_index = detect_voltage_sign(image, comp)["plus_index"]
        if plus_index is None:
            continue
        source_signs[comp["nets"][plus_index]].append(f"{comp['ref']}+")
        source_signs[comp["nets"][1 - plus_index]].append(f"{comp['ref']}-")

    node_labels = {}
    for net in sorted(result["nets"]):
        signs = sorted(set(source_signs[net]))
        if net == "VDD":
            label = "+"
        elif net == "VSS":
            label = "-"
        elif not signs:
            label = "?"
        elif source_count == 1:
            polarities = {sign[-1] for sign in signs}
            label = next(iter(polarities)) if len(polarities) == 1 else "conflict"
        else:
            label = "/".join(signs)
        node_labels[net] = label

    return {
        "nodes": [{"id": net, "polarity": label} for net, label in node_labels.items()],
        "components": [
            {"id": comp["ref"], "type": comp["kind"],
             "terminals": [{"node": net, "polarity": node_labels[net]} for net in comp["nets"]]}
            for comp in components
        ],
    }


def draw_polarity(image: np.ndarray, result: dict, wire_lines: list, polarity: dict, path: Path,
                  wire_mask: np.ndarray | None = None) -> None:
    """Version 3: readable electrical-node and polarity overlay."""
    top = 42
    h, w = image.shape[:2]
    all_node_labels = {node["id"]: node["polarity"] for node in polarity["nodes"]}
    active_nets = {net for comp in result["components"]
                   for net in comp.get("port_nets", {}).values()}
    node_labels = {net: all_node_labels[net] for net in sorted(active_nets) if net in all_node_labels}
    # The right-hand node list must show every active net: with many nets a single
    # column runs past the bottom of the canvas, so lay it out in columns that fit.
    list_left, list_top = w + 15, top + 50
    list_pitch = 120
    available = max(0, (h + top - 8) - list_top)
    max_rows = max(1, available // 22 + 1)
    columns = max(1, -(-len(node_labels) // max_rows)) if node_labels else 1
    rows = max(1, -(-len(node_labels) // columns)) if node_labels else 1
    step = 27 if rows <= 1 else max(11, min(27, available // (rows - 1)))
    label_scale = .48 if step >= 24 else (.40 if step >= 18 else (.34 if step >= 14 else .28))
    canvas = np.full((h + top, w + 235 + (columns - 1) * list_pitch, 3), 255, np.uint8)
    canvas[top:top + h, :w] = image
    if wire_mask is not None:
        region = canvas[top:top + h, :w]
        region[wire_mask > 0] = (255, 80, 0)
    cv2.putText(canvas, "Electrical nodes; VDD/VSS carry the supply polarity",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (30, 30, 30), 1)
    cv2.putText(canvas, "Each active node label is placed once to keep the drawing readable",
                (8, 35), cv2.FONT_HERSHEY_SIMPLEX, .38, (30, 30, 30), 1)
    for a, b in wire_lines:
        cv2.line(canvas, _xy(a, top), _xy(b, top), (255, 0, 0), 2)
    for junction in result["junctions"]:
        cv2.circle(canvas, _xy(junction["point"], top), 6, (0, 165, 255), 2)
    def label_color(label: str) -> tuple[int, int, int]:
        if label == "+":
            return (0, 0, 215)
        if label == "-":
            return (200, 80, 0)
        return (100, 100, 100) if label == "?" else (0, 120, 220)

    labelled_nets: set[str] = set()
    for comp in result["components"]:
        # Keep the source diagram's own M1/M2... labels visible.  Detector
        # sequence numbers are implementation details and may use a different
        # ordering from labels printed on the circuit, so they do not belong in
        # the final topology view.
        for point, net in zip(comp["contacts"], comp["nets"]):
            x, y = _xy(point, top)
            label = all_node_labels.get(net, "?")
            color = label_color(label)
            cv2.circle(canvas, (x, y), 3, color, -1)
            if net not in labelled_nets:
                labelled_nets.add(net)
                suffix = label if label in ("+", "-") else ""
                text = f"{net}{suffix}"
                lx = min(max(0, x + 5), max(0, w - 58))
                ly = min(max(14, y - 6), h + top - 3)
                cv2.putText(canvas, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, .36, color, 1)
    cv2.line(canvas, (w + 5, top), (w + 5, h + top), (210, 210, 210), 1)
    for column in range(1, columns):
        divider = list_left + column * list_pitch - 10
        cv2.line(canvas, (divider, top), (divider, h + top), (235, 235, 235), 1)
    cv2.putText(canvas, "Electrical nodes", (w + 15, top + 22), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (30, 30, 30), 1)
    for index, (net, label) in enumerate(node_labels.items()):
        column, row = divmod(index, rows)
        cv2.putText(canvas, f"{net}: {label}",
                    (list_left + column * list_pitch, list_top + row * step),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_color(label), 1)
    _save_png(path, canvas)
