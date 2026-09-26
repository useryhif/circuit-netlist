"""Pixel skeleton wire tracing and terminal-to-net extraction for EDA drawings."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


# A symmetric two-terminal symbol carries no pin-1 marker, so the drawing alone
# cannot say which end is "Pos" and which is "Neg". Treat the first named port
# as the left/top end, which is what calibrates best on the labelled circuits;
# ablate_connections.py re-fits this constant when more figures are labelled.
PORT_ORDER = {
    "two_pin": {"h": "left", "v": "top"},
}
CONTACT_RADIUS = 5
FALLBACK_CONTACT_RADIUS = 10
TWO_PIN_FALLBACK_CONTACT_RADIUS = 6
SOURCE_FALLBACK_CONTACT_RADIUS = 10
WIRE_CLOSE_SIZE = 5
HOLLOW_PORT_PROTECTION_REACH = 30
BORDER_CORNER_MARGIN = 20
BORDER_CORNER_GAP = 4
MIN_UNJOINED_CROSSING_DEGREE = 4
CROSSING_PATCH_RADIUS = 6
CROSSING_DARK_PIXEL_VALUE = 140
DOT_DARK_THRESHOLD = 26
# Some drawings use solid black wires, so counting "dark" pixels cannot tell a
# printed junction dot from a plain crossing: both are dark.  A dot is a filled
# disk, so it is locally much fatter than the two wires it joins; measure that
# with a distance transform instead.
DOT_MIN_RADIUS = 4.0
DOT_THICKNESS_GAIN = 3.0
# A printed junction dot also shows up as a bulge of the wire right next to the
# crossing: the stroke is a couple of pixels wider than it is farther away.
# This is measured on the raster (not on the skeleton) and is the most reliable
# of the three signals, because the drawing's line width cancels out.
DOT_MIN_BULGE = 3
# Radius of the hole punched at a crossing before re-labelling.  It has to
# exceed the wire width, otherwise the two lines stay 8-connected around the
# hole and the split silently re-joins every arm.
CROSSING_HOLE_RADIUS = 2
# Some drawings mark the drain with the MOS arrow, others the source; the
# pipeline default keeps the geometric rule and this switch enables the arrow
# when a corpus is drawn that way.
MOS_ARROW_MARKS_DRAIN = False
# A detected PNP whose emitter arrow is measurably heavier than its collector
# lead is labelled NPN by the drawings of case023/039.  The rule is **disabled**:
# measured on the corpus it flips six devices where the truth flips four
# (case018 0.9294 -> 0.8353, case039 0.7602 -> 0.7251, case023 +0.0154, net
# -0.0028), and the margins overlap (truth-NPN 0.6-1.8 vs truth-PNP 0.2-0.0).
# Set to e.g. 0.5 to re-enable for experiments.
BJT_ARROW_FLIP_MARGIN = 0.0
# Largest white gap (in pixels) that a free wire end may jump to reach another
# wire, or 0 to disable the rule.  ``_reconnect_free_ends`` was measured on the
# 40-case corpus and *rejected*: it merges nets that the drawings keep apart
# (case012 1.0000 -> 0.8776, case018 -0.0235, case002 -0.0093) while not
# repairing the gaps it was written for.  It stays available for experiments.
# Only bridge when the end stops just short of *crossing* another wire (a T
# junction); collinear gaps stay untouched because the drawing uses them to keep
# two nets apart.
# ``_reconnect_free_ends`` itself stays disabled: even restricted to T junctions
# it merged nets that the drawings keep apart (case012 1.0000 -> 0.9388), and it
# did not repair the gap it was written for.  Larger structural gaps (a wire cut
# by a component box) need the box-aware handling described in the logs.
FREE_END_MAX_GAP = 0
FREE_END_STRAIGHT_ONLY = True
# A real terminal starts at the cut edge of its component box, while the part's
# printed text sits a few pixels away.  Contacts in the normal sampling pass must
# reach the box within this many pixels.
CONTACT_TOUCH_REACH = 2.0
# A supply symbol (VDD/GND glyph) only merges the nets whose wire reaches its
# own box, within this many pixels.
SUPPLY_MARKER_TOUCH_REACH = 3.0
# A bottom rail only counts as the page ground when it spans this fraction of
# the drawing width and the page already has a ground symbol.
BOTTOM_RAIL_FULL_WIDTH_RATIO = 0.85
CONTACT_CLUSTER_GAP = 10
# Diagonal feedback paths leave a cut crossover through its corners, where the
# four side strips used for horizontal/vertical cross-lines see nothing.  The
# arm pairing below groups the surviving wire labels by their bearing from the
# gap and only joins two arms that run opposite to each other.
ARM_ANGULAR_GAP_DEGREES = 50.0
ARM_OPPOSITE_MIN_STRAIGHTNESS = 0.906  # cos(25 deg): near-collinear arms only
ARM_OPPOSITE_MAX_OFFSET_RATIO = 0.32   # centre must lie on the arm-to-arm line
ARM_PROBE_REACH = 14
ARM_MIN_SUPPORT = 3
ARM_MAX_CANDIDATES = 6
# A terminal wire can be much shorter than any real net: two parts drawn in
# series (a resistor above a capacitor) are joined by a stub of a few pixels,
# and that stub is erased by the line-opening kernel.  Such a fragment is still
# a genuine wire when it is a thin bar that reaches a component box, so the
# validity filter below accepts it on geometry instead of on length.
STUB_MAX_THICKNESS = 6
STUB_MAX_LENGTH = 28
STUB_MIN_PIXELS = 4
STUB_TOUCH_REACH = 2
# Series parts are sometimes drawn with their boxes touching, so the connecting
# wire is entirely inside the two symbol boxes and no wire pixel survives.  Two
# stacked boxes that overlap along the shared edge are joined by a synthetic
# net; a duplicate detection of one glyph fails the gap test below because its
# boxes overlap in both directions.
# Series parts are often separated by their own lead glyphs (a capacitor's plate
# sits between the two boxes), so the accepted gap is a bit wider than the boxes
# themselves; the "no valid contact on either facing side" requirement keeps
# parallel parts from being joined.
BODY_TOUCH_GAP = 8
BODY_TOUCH_MIN_OVERLAP = 0.6
# Extra line orientations opened in addition to horizontal/vertical.  The EDA
# corpus draws feedback paths at 30/45 degrees; leaving them out drops whole
# nets, but opening too many orientations starts to pick up italic text.
DIAGONAL_LINE_ANGLES = (30, 45, 135, 150)
# Diagonal runs are measured with their own kernel length: real feedback wires
# are 70 px and longer, while italic component text produces much shorter
# strokes, so a longer kernel keeps wires and rejects the text.
DIAGONAL_LINE_LENGTH = 30


def _port_order(cls: str) -> dict[str, str]:
    for prefix, rule in PORT_ORDER.items():
        if prefix != "two_pin" and cls.startswith(prefix):
            return rule
    return PORT_ORDER["two_pin"]


def _side_pair(ends: dict[str, list[tuple[int, tuple[int, int]]]], first_side: str, second_side: str):
    """Widest label pair bridging two opposite sides, or None."""
    candidates = []
    for first, point_a in ends.get(first_side, []):
        for second, point_b in ends.get(second_side, []):
            separation = abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])
            candidates.append((first, second, separation))
    return max(candidates, key=lambda item: item[2]) if candidates else None


def _order_two_pin_contacts(comp: dict, found: dict) -> tuple[int, int] | None:
    """Return (first_label, second_label) for a two-terminal symbol.

    A real end contact touches the body close to its mid line, while the part's
    own text label also bleeds into the sampling strips and must not be mistaken
    for a terminal. The wider of the two opposite-side pairs is the drawn axis;
    the per-family rule then says which end carries the first named port.
    """
    x1, y1, x2, y2 = comp["bbox"]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    height_slack, width_slack = max(6.0, (y2 - y1) * 0.35), max(6.0, (x2 - x1) * 0.35)
    ends: dict[str, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for (side, label, _), (x, y) in found.items():
        slack = height_slack if side in ("left", "right") else width_slack
        offset = abs(y - center_y) if side in ("left", "right") else abs(x - center_x)
        if offset <= slack:
            ends[side].append((label, (x, y)))
    rule = _port_order(comp["class_name"])
    horizontal, vertical = _side_pair(ends, "left", "right"), _side_pair(ends, "top", "bottom")
    if horizontal and (vertical is None or horizontal[2] >= vertical[2]):
        return (horizontal[0], horizontal[1]) if rule["h"] == "left" else (horizontal[1], horizontal[0])
    if vertical:
        return (vertical[0], vertical[1]) if rule["v"] == "top" else (vertical[1], vertical[0])
    return None


def _contact_labels(labels: np.ndarray, bbox: list[float], radius: int = 5,
                    split_same_label_contacts: bool = False):
    h, w = labels.shape
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    sides: dict[str, list[tuple[int, int, int]]] = {s: [] for s in ("left", "right", "top", "bottom")}
    # Sample a thin strip immediately outside each YOLO box. The body itself is
    # removed before connected-component labeling, so the component cannot short pins.
    for y in range(max(0, y1 - 2), min(h, y2 + 3)):
        for x in range(max(0, x1 - radius), min(w, x1 + 1)):
            lab = int(labels[y, x])
            if lab: sides["left"].append((x, y, lab))
        for x in range(max(0, x2), min(w, x2 + radius + 1)):
            lab = int(labels[y, x])
            if lab: sides["right"].append((x, y, lab))
    for x in range(max(0, x1 - 2), min(w, x2 + 3)):
        for y in range(max(0, y1 - radius), min(h, y1 + 1)):
            lab = int(labels[y, x])
            if lab: sides["top"].append((x, y, lab))
        for y in range(max(0, y2), min(h, y2 + radius + 1)):
            lab = int(labels[y, x])
            if lab: sides["bottom"].append((x, y, lab))
    # Per side, keep one median anchor per connected wire component.
    found = {}
    for side, hits in sides.items():
        by_label: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for x, y, label in hits:
            by_label[label].append((x, y))
        for label, pts in by_label.items():
            axis = 1 if side in ("left", "right") else 0
            ordered = sorted(pts, key=lambda point: point[axis])
            groups: list[list[tuple[int, int]]] = [[]]
            previous = None
            for point in ordered:
                if (split_same_label_contacts and previous is not None
                        and point[axis] - previous > CONTACT_CLUSTER_GAP):
                    groups.append([])
                groups[-1].append(point)
                previous = point[axis]
            for index, group in enumerate(groups):
                xx = int(np.median([p[0] for p in group])); yy = int(np.median([p[1] for p in group]))
                found[(side, label, index)] = (xx, yy)
    return found


def _lead_thickness(distance: np.ndarray, bbox: list[float],
                    point: tuple[float, float], reach: int = 16) -> float:
    """Thickest ink found on a lead running from ``point`` towards the body.

    A MOS glyph draws its arrow on one of the two channel leads, so the arrow
    side is a couple of pixels thicker than the plain lead.
    """
    x1, y1, x2, y2 = bbox
    centre = ((x1 + x2) / 2, (y1 + y2) / 2)
    direction = np.array([centre[0] - point[0], centre[1] - point[1]], dtype=float)
    norm = float(np.hypot(*direction))
    if norm < 1e-6:
        return 0.0
    direction /= norm
    height, width = distance.shape
    best = 0.0
    for step in range(3, reach + 1):
        px = int(round(point[0] + direction[0] * step))
        py = int(round(point[1] + direction[1] * step))
        window = distance[max(0, py - 5):min(height, py + 6),
                          max(0, px - 5):min(width, px + 6)]
        if window.size:
            best = max(best, float(window.max()))
    return best


def _amp_input_signs(ink: np.ndarray, bbox: list[float],
                     points: list[tuple[float, float]]) -> list[str]:
    """Read the printed ``+`` / ``-`` next to each amplifier input.

    Both marks are small ink blobs sitting inside the triangle, separate from
    its outline: a cross is roughly as tall as it is wide, a minus is a flat bar.
    """
    if ink is None or len(points) < 2:
        return []
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    height, width = ink.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2 + 1), min(height, y2 + 1)
    if x2 <= x1 or y2 <= y1:
        return []
    patch = ink[y1:y2, x1:x2].astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(patch, 8)
    marks = []
    for index in range(1, count):
        box_w, box_h, area = int(stats[index, 2]), int(stats[index, 3]), int(stats[index, 4])
        if not 6 <= area <= 90 or max(box_w, box_h) > 18:
            continue
        marks.append((float(centroids[index][0]) + x1, float(centroids[index][1]) + y1,
                      box_w, box_h))
    signs = []
    for point in points:
        nearest = None
        for mark in marks:
            distance = float(np.hypot(mark[0] - point[0], mark[1] - point[1]))
            if distance <= 34 and (nearest is None or distance < nearest[0]):
                nearest = (distance, mark)
        if nearest is None:
            signs.append("?")
            continue
        box_w, box_h = nearest[1][2], nearest[1][3]
        if box_h >= 5 and box_w >= 5:
            signs.append("+")
        elif box_h <= 3:
            signs.append("-")
        else:
            signs.append("?")
    return signs


def _assign_contacts(comp: dict, found: dict, ink: np.ndarray | None = None,
                     distance: np.ndarray | None = None) -> dict[str, int]:
    cls = comp["class_name"]
    x1, y1, x2, y2 = comp["bbox"]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    contacts: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for (side, label, _), (x, y) in found.items():
        pos = y if side in ("left", "right") else x
        contacts[side].append((pos, label))
    def pick(side, from_start=True):
        vals = sorted(contacts[side], reverse=not from_start)
        return [label for _, label in vals]
    def pick_centre(side):
        """Candidate closest to the middle of the box, where a lead really ends.

        A wire that only passes by a corner also shows up in the strip (case011
        had the VDD rail touching a PMOS gate side), while the drawn gate lead
        sits at mid height.
        """
        if not contacts[side]:
            return None
        centre = center_y if side in ("left", "right") else center_x
        return min(contacts[side], key=lambda item: abs(item[0] - centre))[1]
    if cls.startswith(("nmos", "pmos")):
        gate_side = "left" if pick("left") else "right"
        gate = pick(gate_side)
        ends = pick("top") + pick("bottom")
        ends = list(dict.fromkeys(ends))
        mapped = {}
        rotated = False
        if gate:
            mapped["Gate"] = pick_centre(gate_side)
        elif len(pick("top")) >= 2 or len(pick("bottom")) >= 2:
            # Rotated symbol: the gate lead leaves through the single contact on
            # the top/bottom edge and both channel ends share the opposite edge.
            # (case009 draws a PMOS sideways with its gate lead going down.)
            for gate_candidate, ends_side in (("bottom", "top"), ("top", "bottom")):
                if len(pick(gate_candidate)) == 1 and len(pick(ends_side)) >= 2:
                    channel_ends = pick(ends_side)[:2]
                    mapped["Gate"] = pick(gate_candidate)[0]
                    if cls.startswith("pmos"):
                        mapped["Source"], mapped["Drain"] = channel_ends[0], channel_ends[1]
                    else:
                        mapped["Drain"], mapped["Source"] = channel_ends[0], channel_ends[1]
                    rotated = True
                    break
        def channel_end(side: str):
            candidates = contacts[side]
            if not candidates:
                return None
            # The channel is on the side opposite the gate.  Text strokes and
            # the gate lead can also enter the top/bottom sampling strip, so
            # select the candidate aligned with the channel edge.
            return (max(candidates) if gate_side == "left" else min(candidates))[1]
        if not rotated:
            top_label = channel_end("top") or (ends[0] if ends else None)
            bottom_label = channel_end("bottom") or (ends[-1] if len(ends) > 1 else None)
            if cls.startswith("pmos"):
                if top_label: mapped["Source"] = top_label
                if bottom_label: mapped["Drain"] = bottom_label
            else:
                if top_label: mapped["Drain"] = top_label
                if bottom_label: mapped["Source"] = bottom_label
        if "bulk" in cls:
            # These EDA bulk glyphs expose three external wires: the body
            # arrow is tied internally to Source. Sampling a lateral side a
            # second time aliases Body to Gate and invents the wrong net.
            if "Source" in mapped:
                mapped["Body"] = mapped["Source"]
        if MOS_ARROW_MARKS_DRAIN and distance is not None and "Gate" in mapped:
            # The arrow sits on one of the two channel leads; when the drawing
            # uses it to mark the drain, that lead decides the naming instead of
            # the fixed "top is drain" rule.
            channel = [(side, point) for (side, label, _), point in found.items()
                       if side in ("top", "bottom", "left", "right")
                       and label in (mapped.get("Drain"), mapped.get("Source"))]
            thickness = [( _lead_thickness(distance, comp["bbox"], point), side)
                         for side, point in channel]
            if len(thickness) >= 2:
                thickness.sort(reverse=True)
                if thickness[0][0] > thickness[1][0] + 0.4:
                    arrow_side = thickness[0][1]
                    drain_label = mapped.get("Drain")
                    source_label = mapped.get("Source")
                    if drain_label is not None and source_label is not None:
                        arrow_keys = [key for key in found if key[0] == arrow_side
                                      and key[1] in (drain_label, source_label)]
                        if arrow_keys and arrow_keys[0][1] == source_label:
                            mapped["Drain"], mapped["Source"] = source_label, drain_label
        return mapped
    if cls.startswith(("npn", "pnp")):
        left = pick("left") or pick("right"); top = pick("top"); bottom = pick("bottom")
        # A lead can leave the symbol through a corner: the side strips then pick
        # it up near the top or bottom edge of the box instead of the top/bottom
        # strips (case039's Q15 collector).  Only accept a corner contact that is
        # clearly away from the base lead, which sits at mid height.
        base_label = left[0] if left else None
        def corner(favour_top: bool) -> list[int]:
            best: tuple[float, int] | None = None
            for side in ("left", "right"):
                for pos, label in contacts[side]:
                    if label == base_label or abs(pos - center_y) < 0.3 * (y2 - y1):
                        continue
                    if favour_top and pos <= y1 + 0.25 * (y2 - y1):
                        if best is None or pos < best[0]:
                            best = (pos, label)
                    if not favour_top and pos >= y2 - 0.25 * (y2 - y1):
                        if best is None or pos > best[0]:
                            best = (pos, label)
            return [best[1]] if best else []
        top = top or corner(True)
        bottom = bottom or corner(False)
        # The public EDA symbols draw NPN and PNP with opposite vertical
        # collector/emitter directions.  Treating both like NPN systematically
        # swaps the PNP terminals even when the three wire contacts are found.
        collector = bottom[0] if cls.startswith("pnp") and bottom else (top[0] if top else None)
        emitter = top[0] if cls.startswith("pnp") and top else (bottom[0] if bottom else None)
        return {k:v for k,v in (("Base", left[0] if left else None),
                                ("Collector", collector), ("Emitter", emitter)) if v}
    if cls in ("single-end-amp", "diff-amp"):
        # The amplifier triangle may point right, up, left or down. Its two
        # input terminals share one side; outputs occupy the opposite side.
        # ``pick`` already orders vertical pairs top-to-bottom and horizontal
        # pairs left-to-right, which preserves the printed -/+ and N/P order.
        opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        sides = ("left", "right", "top", "bottom")
        # The two inputs are two different nets.  A single wire that touches the
        # sampling strip twice (case009 has one running past the triangle) must
        # not be mistaken for an input pair.
        input_sides = [side for side in sides
                       if len(pick(side)) >= 2 and len(set(pick(side))) >= 2]
        if not input_sides:
            input_sides = [side for side in sides if len(pick(side)) >= 2]
        if input_sides:
            input_side = max(input_sides, key=lambda side: len(pick(side)))
            inputs, outputs = pick(input_side), pick(opposite[input_side])
            # A triangle rotated 90 degrees puts its two outputs on the corners
            # beside the apex, so they show up as one contact each on the two
            # sides adjacent to the flat input edge.
            corner_sides = ("left", "right") if input_side in ("top", "bottom") else ("top", "bottom")
            corners = [pick(side)[0] for side in corner_sides if pick(side)]
            # The printed +/- marks decide which input is the inverting one; the
            # geometric order is only a fallback when they cannot be read.
            ordered_inputs = (inputs[0], inputs[-1])
            input_points = []
            for label in ordered_inputs:
                point = next((value for key, value in found.items() if key[1] == label
                              and key[0] == input_side), None)
                input_points.append(point)
            if all(point is not None for point in input_points):
                signs = _amp_input_signs(ink, comp["bbox"], input_points)
                if len(signs) == 2 and signs[0] != signs[1] and "?" not in signs:
                    # ``ordered_inputs`` starts as (InN, InP); the ``+`` mark
                    # belongs to the non-inverting input, so a plus on the first
                    # entry means the two names have to swap.
                    ordered_inputs = ((ordered_inputs[1], ordered_inputs[0])
                                      if signs[0] == "+" else ordered_inputs)
            if len(inputs) >= 2 and len(outputs) < 2 and len(corners) >= 2:
                return {"InN": ordered_inputs[0], "InP": ordered_inputs[1],
                        "OutN": corners[0], "OutP": corners[1]}
            if cls == "single-end-amp" and len(inputs) >= 2 and outputs:
                return {"InN": ordered_inputs[0], "InP": ordered_inputs[1], "Out": outputs[0]}
            if cls == "diff-amp" and len(inputs) >= 2 and len(outputs) >= 2:
                return {"InN": ordered_inputs[0], "InP": ordered_inputs[1],
                        "OutN": outputs[0], "OutP": outputs[-1]}
        left = pick("left") or pick("right"); right = pick("right") or pick("left")
        left = [v for v in left if v not in right[:1]] if len(left) > 1 else left
        if cls == "single-end-amp":
            return {k:v for k,v in (("InN", left[0] if left else None), ("InP", left[-1] if len(left)>1 else None), ("Out", right[0] if right else None)) if v}
        return {k:v for k,v in (("InN", left[0] if left else None), ("InP", left[-1] if len(left)>1 else None), ("OutN", right[0] if right else None), ("OutP", right[-1] if len(right)>1 else None)) if v}
    if comp["pin_count"] == 2:
        ordered = _order_two_pin_contacts(comp, found)
        if ordered is not None:
            return {comp["port_names"][0]: ordered[0], comp["port_names"][1]: ordered[1]}
        # A single wire label on both strips means the body sat on one net.
        a = pick("left") or pick("top")
        b = pick("right") or pick("bottom")
        if a and b:
            return {comp["port_names"][0]: a[0], comp["port_names"][1]: b[0]}
    return {}


def _strip_radii(comp: dict, image_scale: float) -> tuple[int, int]:
    """Sampling radii (normal, widened) used to look for a part's terminals."""
    if comp["component_type"] in {"Current", "Voltage"}:
        fallback = SOURCE_FALLBACK_CONTACT_RADIUS
    elif comp["pin_count"] >= 3:
        fallback = FALLBACK_CONTACT_RADIUS
    else:
        fallback = TWO_PIN_FALLBACK_CONTACT_RADIUS
    base = max(1, round(CONTACT_RADIUS * image_scale))
    return base, max(base, max(1, round(fallback * image_scale)))


def _promote_amp_arity(comp: dict, mapping: dict) -> dict:
    """A triangle amp with four terminals is differential, whatever YOLO said.

    The same triangle glyph is used for the single-ended and the differential
    part; the only difference is how many leads leave it.  When four corners
    carry wires the component is a ``Dido_amp`` with both output pins.
    """
    if comp["component_type"] != "Diso_amp" or len(mapping) < 4:
        return mapping
    comp.update(class_name="diff-amp", kind="X", component_type="Dido_amp",
                port_names=["InN", "InP", "OutN", "OutP"], pin_count=4)
    return mapping


def _box_distance(points: np.ndarray, bbox: list[float]) -> float:
    """Chebyshev distance from the nearest wire pixel to the component box."""
    x1, y1, x2, y2 = bbox
    ys, xs = points[:, 0], points[:, 1]
    dx = np.maximum(np.maximum(x1 - xs, xs - x2), 0)
    dy = np.maximum(np.maximum(y1 - ys, ys - y2), 0)
    return float(np.min(np.maximum(dx, dy)))


def _touches_box(points: np.ndarray, bbox: list[float], reach: float) -> bool:
    """True when some wire pixel sits outside but immediately beside the box."""
    return _box_distance(points, bbox) <= reach


def _bjt_arrow_polarity(comp: dict, mapping: dict, found: dict,
                        distance: np.ndarray | None) -> str | None:
    """Polarity implied by the emitter arrow, or None when it cannot be read.

    The arrow makes one lead measurably thicker.  For an NPN the detector and the
    arrow agree (arrow on the lower lead), but the drawings also contain devices
    the detector calls PNP while the arrow sits on the upper lead with a clear
    margin -- those are the ones the truth labels NPN (case023/039).  When both
    leads measure the same the arrow is too small to read, so the detector's
    polarity is kept (case037).
    """
    if distance is None or BJT_ARROW_FLIP_MARGIN <= 0:
        return None
    if comp["component_type"] not in {"NPN", "PNP"}:
        return None
    points = {}
    for port in ("Collector", "Emitter"):
        label = mapping.get(port)
        if label is None:
            return None
        point = next((value for key, value in found.items() if key[1] == label), None)
        if point is None:
            return None
        points[port] = point
    collector = _lead_thickness(distance, comp["bbox"], points["Collector"])
    emitter = _lead_thickness(distance, comp["bbox"], points["Emitter"])
    if emitter - collector >= BJT_ARROW_FLIP_MARGIN:
        return "NPN" if comp["component_type"] == "PNP" else None
    return None


def _stub_label_ids(strips: dict, components_by_label: dict, image_scale: float) -> set[int]:
    """Wire labels that are too short to be any net but are real terminal stubs.

    Only fragments that were sampled by a component contact strip (so they start
    at a box edge) and that are drawn as a thin bar leaving the box are
    accepted.  A part's own glyph can also sit just outside a loose YOLO box --
    a MOS gate bar, for example -- but that stroke runs parallel to the edge,
    while a terminal wire has to leave the edge outwards.  Text glyphs next to a
    part are wider than ``STUB_MAX_THICKNESS``.
    """
    max_thickness = max(2, round(STUB_MAX_THICKNESS * image_scale))
    max_length = max(4, round(STUB_MAX_LENGTH * image_scale))
    min_pixels = max(3, round(STUB_MIN_PIXELS * image_scale))
    reach = max(1.0, STUB_TOUCH_REACH * image_scale)
    accepted: set[int] = set()
    for _, (comp, _, _, found, expanded) in strips.items():
        candidates = dict(found)
        candidates.update({key: point for key, point in expanded.items() if key not in candidates})
        for (side, label, _), _ in candidates.items():
            if label in accepted:
                continue
            points = components_by_label.get(label)
            if points is None or len(points) < min_pixels:
                continue
            if not _touches_box(points, comp["bbox"], reach):
                continue
            height = int(np.ptp(points[:, 0])) + 1
            width = int(np.ptp(points[:, 1])) + 1
            if min(height, width) > max_thickness or max(height, width) > max_length:
                continue
            outward, along = (width, height) if side in ("left", "right") else (height, width)
            if outward < along:
                continue
            accepted.add(label)
    return accepted


def _touching_body_pairs(components: list[dict], image_scale: float) -> list[dict]:
    """Component pairs stacked so closely that no wire pixel can survive.

    Returns one entry per qualifying pair with the facing sides, the shared
    contact points and the label id that should join the two terminals.
    """
    reach = max(1.0, BODY_TOUCH_GAP * image_scale)
    bodies = [comp for comp in components if not comp.get("marker")]
    pairs: list[dict] = []
    for index, first in enumerate(bodies):
        for second in bodies[index + 1:]:
            fx1, fy1, fx2, fy2 = first["bbox"]
            sx1, sy1, sx2, sy2 = second["bbox"]
            for first_side, second_side, gap, span, along in (
                    ("bottom", "top", sy1 - fy2, (max(fx1, sx1), min(fx2, sx2)), "x"),
                    ("top", "bottom", fy1 - sy2, (max(fx1, sx1), min(fx2, sx2)), "x"),
                    ("right", "left", sx1 - fx2, (max(fy1, sy1), min(fy2, sy2)), "y"),
                    ("left", "right", fx1 - sx2, (max(fy1, sy1), min(fy2, sy2)), "y")):
                if not (-reach <= gap <= reach):
                    continue
                overlap = span[1] - span[0]
                if along == "x":
                    smaller = min(fx2 - fx1, sx2 - sx1)
                else:
                    smaller = min(fy2 - fy1, sy2 - sy1)
                if overlap <= 0 or overlap < BODY_TOUCH_MIN_OVERLAP * max(1.0, smaller):
                    continue
                middle = (span[0] + span[1]) / 2.0
                if along == "x":
                    first_point = (middle, fy2 + 1)
                    second_point = (middle, sy1 - 1)
                else:
                    first_point = (fx2 + 1, middle)
                    second_point = (sx1 - 1, middle)
                pairs.append({"first": first, "second": second,
                              "first_side": first_side, "second_side": second_side,
                              "first_point": first_point, "second_point": second_point})
    return pairs


def _label_by_dfs(skeleton: np.ndarray) -> tuple[int, np.ndarray]:
    """Label 8-neighbor wire pixels with an explicit depth-first traversal."""
    h, w = skeleton.shape
    labels = np.zeros((h, w), np.int32)
    next_id = 0
    ys, xs = np.where(skeleton > 0)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if labels[sy, sx]:
            continue
        next_id += 1
        labels[sy, sx] = next_id
        stack = [(sy, sx)]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = next_id
                        stack.append((ny, nx))
    return next_id + 1, labels


def _stroke_width(ink: np.ndarray, x: int, y: int, vertical: bool, half: int = 8) -> int:
    """Width of the stroke through (x, y) measured across its own direction."""
    h, w = ink.shape
    if vertical:
        line = ink[y, max(0, x - half):min(w, x + half + 1)]
        centre = min(half, x)
    else:
        line = ink[max(0, y - half):min(h, y + half + 1), x]
        centre = min(half, y)
    if not line.size or not line[centre]:
        return 0
    low = centre
    while low > 0 and line[low - 1]:
        low -= 1
    high = centre
    while high + 1 < len(line) and line[high + 1]:
        high += 1
    return int(high - low + 1)


def _junction_bulge(ink: np.ndarray, x: int, y: int, reach: int = 10) -> int:
    """How much wider the wires get right next to a junction.

    A printed junction dot shows up as a bulge of the stroke width next to the
    crossing, while a plain crossing keeps the wire's own width, so this signal
    does not depend on how thick the drawing's lines are.
    """
    def profile(vertical: bool) -> int:
        height, width = ink.shape
        offsets = list(range(-reach, -reach + 4)) + list(range(reach - 3, reach + 1))
        if vertical:
            baseline = [_stroke_width(ink, x, y + dy, True) for dy in offsets
                        if 0 <= y + dy < height]
        else:
            baseline = [_stroke_width(ink, x + dx, y, False) for dx in offsets
                        if 0 <= x + dx < width]
        baseline = [value for value in baseline if value]
        if not baseline:
            return 0
        typical = int(np.median(baseline))
        limit = max(3, typical * 2 + 2)
        peak = 0
        for offset in range(-5, 6):
            if vertical:
                if not 0 <= y + offset < height:
                    continue
                width = _stroke_width(ink, x, y + offset, True)
            else:
                if not 0 <= x + offset < width:
                    continue
                width = _stroke_width(ink, x + offset, y, False)
            if 0 < width <= limit:
                peak = max(peak, width)
        return peak - typical

    return max(profile(True), profile(False))


def _reclassify_arrow_sources(components: list[dict], ink: np.ndarray,
                              thickness: np.ndarray) -> list[dict]:
    """A circle carrying a filled arrow is a current source, not a voltage one.

    Both symbols are drawn as a small circle, so the detector confuses them.  The
    current source has a thick arrow inside (the probe measures half-width and
    the longest vertical run of the interior ink), while the voltage glyph is
    built from thin plates or ``+``/``-`` marks.
    """
    height, width = ink.shape
    taken = {int(match.group(1)) for match in
             (re.match(r"^I(\d+)$", item["ref"]) for item in components) if match}
    next_index = max(taken) + 1 if taken else 1
    for comp in components:
        if comp.get("marker") or comp["class_name"] != "voltage":
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in comp["bbox"])
        inset_x, inset_y = int((x2 - x1) * 0.28), int((y2 - y1) * 0.28)
        x1, y1 = max(0, x1 + inset_x), max(0, y1 + inset_y)
        x2, y2 = min(width, x2 - inset_x), min(height, y2 - inset_y)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = ink[y1:y2, x1:x2]
        longest = 0
        for column in range(patch.shape[1]):
            run = 0
            for row in range(patch.shape[0]):
                run = run + 1 if patch[row, column] else 0
                longest = max(longest, run)
        peak = float(thickness[y1:y2, x1:x2].max())
        if longest >= 8 and peak >= 2.5:
            while next_index in taken:
                next_index += 1
            taken.add(next_index)
            comp.update(class_name="current", kind="I", component_type="Current",
                        port_names=["In", "Out"], pin_count=2,
                        ref=f"I{next_index}")
            next_index += 1
    return components


def _crossing_centers(skeleton: np.ndarray, gray: np.ndarray, components: list[dict],
                      dot_dark_threshold: int | None = None, image_scale: float = 1.0) -> list[dict]:
    """Find four-way line crossings that do not contain a printed junction dot.

    The line extractor intentionally keeps every horizontal/vertical stroke.  A
    plain crossing therefore becomes one 8-connected DFS component even when
    the drawing means the wires to pass over one another.  The labelled EDA
    drawings use a filled circle for a real four-way junction, so a crossing is
    treated as joined only when the ink is locally much fatter than the wires
    around it (see ``DOT_MIN_RADIUS`` / ``DOT_THICKNESS_GAIN``).
    """
    def arm_count(x: int, y: int) -> int:
        """Number of separate wire arms leaving a junction (4 = true crossing).

        A wide L corner or a T junction can also show four skeleton neighbours,
        but erasing its middle leaves two (or three) arms, so splitting one of
        those would cut a wire that the drawing means to keep.
        """
        radius = max(4, round(6 * image_scale))
        hole = max(1, round(CROSSING_HOLE_RADIUS * image_scale))
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        window = skeleton[y1:y2, x1:x2].copy()
        cy, cx = y - y1, x - x1
        window[max(0, cy - hole):cy + hole + 1, max(0, cx - hole):cx + hole + 1] = 0
        seen = np.zeros_like(window, dtype=bool)
        arms = 0
        ys, xs = np.nonzero(window)
        for sy, sx in zip(ys.tolist(), xs.tolist()):
            if seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            touches_border = False
            while stack:
                py, px = stack.pop()
                if py in (0, window.shape[0] - 1) or px in (0, window.shape[1] - 1):
                    touches_border = True
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = py + dy, px + dx
                        if (0 <= ny < window.shape[0] and 0 <= nx < window.shape[1]
                                and window[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if touches_border:
                arms += 1
        return arms

    """Find four-way line crossings that do not contain a printed junction dot.

    The line extractor intentionally keeps every horizontal/vertical stroke.  A
    plain crossing therefore becomes one 8-connected DFS component even when
    the drawing means the wires to pass over one another.  The labelled EDA
    drawings use a filled circle for a real four-way junction, so a crossing is
    treated as joined only when the ink is locally much fatter than the wires
    around it (see ``DOT_MIN_RADIUS`` / ``DOT_THICKNESS_GAIN``).
    """
    h, w = skeleton.shape
    dot_dark_threshold = DOT_DARK_THRESHOLD if dot_dark_threshold is None else dot_dark_threshold
    patch_radius = max(2, round(CROSSING_PATCH_RADIUS * image_scale))
    # A filled dot covers area, so its darkness count scales with the image.
    dot_dark_threshold = max(1, round(dot_dark_threshold * image_scale * image_scale))
    ink_mask = (gray < 240).astype(np.uint8)
    thickness = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 5)
    dot_radius = DOT_MIN_RADIUS * image_scale
    thickness_gain = DOT_THICKNESS_GAIN * image_scale
    local_radius = max(2, round(4 * image_scale))
    ring_radius = max(6, round(14 * image_scale))
    blocked = np.zeros_like(skeleton, dtype=np.uint8)
    for comp in components:
        if comp.get("marker"):
            x1, y1, x2, y2 = (round(v) for v in comp["bbox"])
            blocked[max(0, y1 - 2):min(h, y2 + 3), max(0, x1 - 2):min(w, x2 + 3)] = 1
    ys, xs = np.where(skeleton > 0)
    candidates: list[tuple[int, int, int, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if blocked[y, x]:
            continue
        degree = int(skeleton[max(0, y - 1):min(h, y + 2),
                              max(0, x - 1):min(w, x + 2)].sum() - 1)
        if degree < MIN_UNJOINED_CROSSING_DEGREE:
            continue
        if arm_count(x, y) < 4:
            continue
        patch = gray[max(0, y - patch_radius):min(h, y + patch_radius + 1),
                     max(0, x - patch_radius):min(w, x + patch_radius + 1)]
        dark = int((patch < CROSSING_DARK_PIXEL_VALUE).sum())
        ink = int((patch < 240).sum())
        local = float(thickness[max(0, y - local_radius):min(h, y + local_radius + 1),
                                max(0, x - local_radius):min(w, x + local_radius + 1)].max())
        ring = thickness[max(0, y - ring_radius):min(h, y + ring_radius + 1),
                         max(0, x - ring_radius):min(w, x + ring_radius + 1)]
        ring_values = ring[ring > 0]
        baseline = float(np.median(ring_values)) if ring_values.size else 0.0
        # Split only when the junction looks like a plain crossing: a printed
        # dot widens the wires next to the crossing.
        bulge = _junction_bulge(ink_mask, x, y)
        printed_dot = (bulge >= DOT_MIN_BULGE * image_scale
                       or local >= dot_radius
                       or (local - baseline) >= thickness_gain)
        if not printed_dot:
            candidates.append((x, y, degree, ink))

    # One crossing can produce several adjacent skeleton pixels. Collapse those
    # pixels to one center and retain the strongest candidate evidence.
    centers: list[dict] = []
    for x, y, degree, ink in candidates:
        owner = next((item for item in centers
                      if (item["x"] - x) ** 2 + (item["y"] - y) ** 2 <= 36), None)
        if owner is None:
            centers.append({"x": x, "y": y, "degree": degree, "ink": ink, "count": 1})
        else:
            count = owner["count"] + 1
            owner["x"] = round((owner["x"] * owner["count"] + x) / count)
            owner["y"] = round((owner["y"] * owner["count"] + y) / count)
            owner["degree"] = max(owner["degree"], degree)
            owner["ink"] = max(owner["ink"], ink)
            owner["count"] = count
    return centers


def _outward_arms(labels: np.ndarray, center: tuple[float, float], inner_radius: float,
                  outer_radius: float, min_support: int,
                  excluded_bbox: tuple[int, int, int, int] | None = None) -> list[dict]:
    """Group wire pixels around a gap by label and bearing from the gap centre.

    A cut crossover keeps its wire arms outside the cut region, but the arms can
    leave through the corners (diagonal nets) instead of through the middle of a
    side.  Grouping by bearing recovers the arms wherever they exit.
    """
    h, w = labels.shape
    x0, y0 = int(round(center[0])), int(round(center[1]))
    outer = max(1, int(round(outer_radius)))
    window = labels[max(0, y0 - outer):min(h, y0 + outer + 1),
                    max(0, x0 - outer):min(w, x0 + outer + 1)]
    ys, xs = np.nonzero(window)
    if not xs.size:
        return []
    values = window[ys, xs]
    x = xs + max(0, x0 - outer)
    y = ys + max(0, y0 - outer)
    dx, dy = x - x0, y - y0
    distance = np.hypot(dx, dy)
    keep = distance >= max(0.0, float(inner_radius))
    if excluded_bbox is not None:
        bx1, by1, bx2, by2 = excluded_bbox
        keep &= ~((x >= bx1) & (x < bx2) & (y >= by1) & (y < by2))
    if not keep.any():
        return []
    x, y, dx, dy = x[keep], y[keep], dx[keep], dy[keep]
    distance, values = distance[keep], values[keep]
    angles = np.degrees(np.arctan2(dy, dx))
    arms: list[dict] = []
    for label in np.unique(values):
        selected = np.nonzero(values == label)[0]
        if selected.size < min_support:
            continue
        order = selected[np.argsort(angles[selected])]
        sorted_angles = angles[order]
        groups: list[np.ndarray] = []
        start = 0
        for index in range(1, len(sorted_angles)):
            if sorted_angles[index] - sorted_angles[index - 1] > ARM_ANGULAR_GAP_DEGREES:
                groups.append(order[start:index])
                start = index
        groups.append(order[start:])
        if len(groups) > 1 and sorted_angles[0] + 360 - sorted_angles[-1] <= ARM_ANGULAR_GAP_DEGREES:
            groups[0] = np.concatenate([groups.pop(), groups[0]])
        for group in groups:
            if group.size < min_support:
                continue
            unit = np.array([float(np.mean(dx[group] / distance[group])),
                             float(np.mean(dy[group] / distance[group]))])
            norm = float(np.hypot(unit[0], unit[1]))
            if norm < 1e-6:
                continue
            unit /= norm
            closest = int(np.argmin(distance[group]))
            arms.append({"label": int(label), "support": int(group.size),
                         "angle": round(float(np.degrees(np.arctan2(unit[1], unit[0]))), 1),
                         "unit": [float(unit[0]), float(unit[1])],
                         "point": [int(x[group][closest]), int(y[group][closest])],
                         "distance": round(float(distance[group][closest]), 1)})
    arms.sort(key=lambda arm: arm["support"], reverse=True)
    return arms[:ARM_MAX_CANDIDATES]


def _opposite_arm_pairs(labels: np.ndarray, center: tuple[float, float], inner_radius: float,
                        outer_radius: float, min_support: int,
                        excluded_bbox: tuple[int, int, int, int] | None = None
                        ) -> tuple[list[tuple[int, int]], list[dict], list[dict]]:
    """Pair two arms leaving a gap along one straight line.

    Only arms whose bearings differ by at least 155 degrees and whose
    centre-to-arm line passes through the gap centre are accepted, so an X
    crossover becomes NW-SE plus NE-SW instead of one merged node.
    """
    arms = _outward_arms(labels, center, inner_radius, outer_radius, min_support, excluded_bbox)
    candidates = []
    for index in range(len(arms)):
        for other in range(index + 1, len(arms)):
            first, second = arms[index], arms[other]
            if first["label"] == second["label"]:
                continue
            straightness = -(first["unit"][0] * second["unit"][0]
                             + first["unit"][1] * second["unit"][1])
            if straightness < ARM_OPPOSITE_MIN_STRAIGHTNESS:
                continue
            span = float(np.hypot(first["point"][0] - second["point"][0],
                                  first["point"][1] - second["point"][1]))
            if span <= 1:
                continue
            offset = abs((first["point"][0] - second["point"][0]) * (center[1] - second["point"][1])
                         - (first["point"][1] - second["point"][1]) * (center[0] - second["point"][0])) / span
            if offset > ARM_OPPOSITE_MAX_OFFSET_RATIO * span:
                continue
            candidates.append((round(straightness - offset / span, 4), index, other,
                               round(float(offset), 1)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    detail: list[dict] = []
    for score, index, other, offset in candidates:
        if index in used or other in used:
            continue
        used.update((index, other))
        pairs.append((arms[index]["label"], arms[other]["label"]))
        detail.append({"labels": [arms[index]["label"], arms[other]["label"]],
                       "points": [arms[index]["point"], arms[other]["point"]],
                       "bearings": [arms[index]["angle"], arms[other]["angle"]],
                       "straightness": score, "offset_px": offset})
        if len(pairs) == 2:
            break
    return pairs, detail, arms


def _split_unjoined_crossings(skeleton: np.ndarray, gray: np.ndarray, components: list[dict],
                              image_scale: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Split no-dot crossings, then reconnect each of their two straight arms."""
    centers = _crossing_centers(skeleton, gray, components, image_scale=image_scale)
    if not centers:
        count, labels = _label_by_dfs(skeleton)
        return labels, []

    work = skeleton.copy()
    h, w = work.shape
    # The erased hole has to be wider than the wires themselves, otherwise the
    # two lines stay 8-connected around it and the "split" re-joins everything.
    hole = max(1, round(CROSSING_HOLE_RADIUS * image_scale))
    for center in centers:
        x, y = center["x"], center["y"]
        work[max(0, y - hole):min(h, y + hole + 1),
             max(0, x - hole):min(w, x + hole + 1)] = 0
    count, labels = _label_by_dfs(work)
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def join(first: int | None, second: int | None) -> bool:
        if first is None or second is None or first <= 0 or second <= 0:
            return False
        first, second = find(first), find(second)
        if first == second:
            return True
        parent[second] = first
        return True

    def arm_label(x: int, y: int, dx: int, dy: int) -> int | None:
        values: list[int] = []
        for step in range(2, 9):
            cx, cy = x + dx * step, y + dy * step
            if not (0 <= cx < w and 0 <= cy < h):
                break
            patch = labels[max(0, cy - 1):min(h, cy + 2),
                           max(0, cx - 1):min(w, cx + 2)]
            values.extend(int(value) for value in patch.ravel() if value > 0)
            if values:
                break
        if not values:
            return None
        unique, counts = np.unique(values, return_counts=True)
        return int(unique[np.argmax(counts)])

    events = []
    for center in centers:
        x, y = center["x"], center["y"]
        sides = {
            "left": arm_label(x, y, -1, 0), "right": arm_label(x, y, 1, 0),
            "top": arm_label(x, y, 0, -1), "bottom": arm_label(x, y, 0, 1),
        }
        horizontal = join(sides["left"], sides["right"])
        vertical = join(sides["top"], sides["bottom"])
        opposite: list[dict] = []
        if not horizontal and not vertical:
            reach = max(4.0, round(ARM_PROBE_REACH * image_scale))
            pairs, opposite, _ = _opposite_arm_pairs(
                labels, (x, y), 2.0, reach, max(2, round(ARM_MIN_SUPPORT * image_scale)))
            for first, second in pairs:
                join(first, second)
        events.append({"point": [x, y], "sides": sides,
                       "horizontal_reconnected": horizontal,
                       "vertical_reconnected": vertical,
                       "opposite_arm_pairs": opposite,
                       "ink": center["ink"]})
    lookup = np.array([find(i) for i in range(count)], dtype=np.int32)
    labels = lookup[labels]
    return labels, events


def _bridge_side_label(labels: np.ndarray, bbox: list[float], side: str,
                       image_scale: float = 1.0) -> int | None:
    h, w = labels.shape
    x1, y1, x2, y2 = (round(v) for v in bbox)
    cx, cy = round((x1 + x2) / 2), round((y1 + y2) / 2)
    reach, half_width = max(2, round(10 * image_scale)), max(1, round(3 * image_scale))
    if side == "left":
        patch = labels[max(0, cy-half_width):min(h, cy+half_width+1), max(0, x1-reach):max(0, x1)]
    elif side == "right":
        patch = labels[max(0, cy-half_width):min(h, cy+half_width+1), min(w, x2):min(w, x2+reach)]
    elif side == "top":
        patch = labels[max(0, y1-reach):max(0, y1), max(0, cx-half_width):min(w, cx+half_width+1)]
    else:
        patch = labels[min(h, y2):min(h, y2+reach), max(0, cx-half_width):min(w, cx+half_width+1)]
    values = patch[patch > 0]
    if values.size == 0:
        return None
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def _reconnect_bridges(labels: np.ndarray, count: int, components: list[dict],
                       image_scale: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Reconnect straight arms of a curved crossover without joining its two wires."""
    parent = list(range(count))
    events = []
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def join(a: int | None, b: int | None) -> bool:
        if a is None or b is None:
            return False
        a, b = find(a), find(b)
        if a == b:
            return True
        parent[b] = a
        return True
    for bridge in (c for c in components if c.get("marker") == "cross-line-curved"):
        sides = {side: _bridge_side_label(labels, bridge["bbox"], side, image_scale)
                 for side in ("left", "right", "top", "bottom")}
        horizontal = join(sides["left"], sides["right"])
        vertical = join(sides["top"], sides["bottom"])
        opposite: list[dict] = []
        if not horizontal and not vertical:
            x1, y1, x2, y2 = bridge["bbox"]
            reach = max(4.0, round(ARM_PROBE_REACH * image_scale))
            outer = max(x2 - x1, y2 - y1) / 2 + reach
            pairs, opposite, _ = _opposite_arm_pairs(
                labels, ((x1 + x2) / 2, (y1 + y2) / 2), 0.0, outer,
                max(2, round(ARM_MIN_SUPPORT * image_scale)),
                (round(x1), round(y1), round(x2), round(y2)))
            for first, second in pairs:
                join(first, second)
        events.append({"ref": bridge["ref"], "bbox": bridge["bbox"],
                       "sides": sides, "horizontal_reconnected": horizontal,
                       "vertical_reconnected": vertical,
                       "opposite_arm_pairs": opposite})
    if events:
        lookup = np.array([find(i) for i in range(count)], dtype=np.int32)
        labels = lookup[labels]
    return labels, events


def _reconnect_mos_horizontal_buses(labels: np.ndarray, components: list[dict],
                                    image_scale: float = 1.0,
                                    gray: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """Join a horizontal wire entering both sides of a transistor symbol.

    This captures the printed diode-connected MOS style where a bias bus touches
    the central gate contact on one side and continues from the other side.
    Bulk MOS symbols are excluded because the right-side contact can be Body.
    """
    present = np.unique(labels)
    parent = {int(v): int(v) for v in present if v > 0}
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    hollow_ports: list[tuple[list[int], set[int]]] = []
    if gray is not None:
        h, w = labels.shape
        margin = max(1, round(3 * image_scale))
        for marker in (item for item in components if item.get("marker") == "port"):
            x1, y1, x2, y2 = [int(round(value)) for value in marker["bbox"]]
            cx, cy = min(w - 1, max(0, (x1 + x2) // 2)), min(h - 1, max(0, (y1 + y2) // 2))
            # Hollow terminal circles have a light centre; filled junction dots
            # have a dark centre and are allowed to participate in bus joins.
            if gray[cy, cx] <= 180:
                continue
            patch = labels[max(0, y1 - margin):min(h, y2 + margin + 1),
                           max(0, x1 - margin):min(w, x2 + margin + 1)]
            hollow_ports.append(([x1, y1, x2, y2],
                                 {int(value) for value in np.unique(patch) if value > 0}))
    def touches_near_hollow_port(comp: dict, side: str, label: int) -> bool:
        x1, y1, x2, y2 = [int(round(value)) for value in comp["bbox"]]
        cy = (y1 + y2) // 2
        reach = round(HOLLOW_PORT_PROTECTION_REACH * image_scale)
        for (px1, py1, px2, py2), port_labels in hollow_ports:
            if label not in port_labels or not (py1 - 4 <= cy <= py2 + 4):
                continue
            if side == "left" and px2 <= x1 and x1 - px2 <= reach:
                return True
            if side == "right" and px1 >= x2 and px1 - x2 <= reach:
                return True
        return False
    events = []
    for comp in components:
        if comp["class_name"] not in {"nmos", "pmos", "nmos-cross", "pmos-cross",
                                       "npn-cross", "pnp-cross"}:
            continue
        left_label = _bridge_side_label(labels, comp["bbox"], "left", image_scale)
        right_label = _bridge_side_label(labels, comp["bbox"], "right", image_scale)
        if left_label is None or right_label is None:
            continue
        if left_label == right_label:
            continue
        if (touches_near_hollow_port(comp, "left", left_label)
                or touches_near_hollow_port(comp, "right", right_label)):
            continue
        parent[find(right_label)] = find(left_label)
        events.append({"ref": comp["ref"], "left_label": left_label, "right_label": right_label})
    if events:
        lookup = np.arange(int(labels.max()) + 1, dtype=np.int32)
        for lab in parent:
            lookup[lab] = find(lab)
        labels = lookup[labels]
    return labels, events


def _reconnect_border_corners(labels: np.ndarray, image_scale: float = 1.0
                              ) -> tuple[np.ndarray, list[dict]]:
    """Close tiny L-corner gaps near the top or bottom drawing boundary.

    Long feedback rails often turn at the page edge, where rasterisation leaves
    a few white pixels between the horizontal and vertical strokes.  Limiting
    this repair to long perpendicular components and border endpoints avoids
    the false joins produced by a global endpoint-neighbour rule.
    """
    h, _ = labels.shape
    margin = round(BORDER_CORNER_MARGIN * image_scale)
    max_gap = max(2, round(BORDER_CORNER_GAP * image_scale))
    present = [int(value) for value in np.unique(labels) if value > 0]
    parent = {value: value for value in present}
    items: list[tuple[int, str, list[tuple[int, int]]]] = []
    kernel = np.ones((3, 3), np.uint8)
    for label in present:
        mask = (labels == label).astype(np.uint8)
        ys, xs = np.where(mask)
        if not xs.size:
            continue
        span_x, span_y = int(xs.max() - xs.min()), int(ys.max() - ys.min())
        orientation = ("h" if span_x >= 2 * max(1, span_y) and span_x >= 30 * image_scale
                       else "v" if span_y >= 2 * max(1, span_x) and span_y >= 30 * image_scale
                       else None)
        if orientation is None:
            continue
        neighbours = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT) - mask
        endpoint_y, endpoint_x = np.where((mask > 0) & (neighbours <= 1))
        endpoints = [(int(x), int(y)) for x, y in zip(endpoint_x, endpoint_y)
                     if y <= margin or y >= h - 1 - margin]
        if endpoints:
            items.append((label, orientation, endpoints))
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    events = []
    for index, (first_label, first_orientation, first_endpoints) in enumerate(items):
        for second_label, second_orientation, second_endpoints in items[index + 1:]:
            if first_orientation == second_orientation:
                continue
            pair = next((((x1, y1), (x2, y2))
                         for x1, y1 in first_endpoints for x2, y2 in second_endpoints
                         if ((abs(x1 - x2) <= 1 and 1 < abs(y1 - y2) <= max_gap)
                             or (abs(y1 - y2) <= 1 and 1 < abs(x1 - x2) <= max_gap))), None)
            if pair is None:
                continue
            first_root, second_root = find(first_label), find(second_label)
            if first_root != second_root:
                parent[second_root] = first_root
                events.append({"labels": [first_label, second_label], "points": pair})
    if events:
        lookup = np.arange(int(labels.max()) + 1, dtype=np.int32)
        for label in parent:
            lookup[label] = find(label)
        labels = lookup[labels]
    return labels, events


def _reconnect_free_ends(labels: np.ndarray, count: int, image_scale: float = 1.0
                         ) -> tuple[np.ndarray, list[dict]]:
    """Join a wire whose free end stops a few pixels short of the next wire.

    Some drawings leave a small white gap where a lead meets a bus (case028's
    gate bus is 8px away from one MOS gate bar).  Only a free end that probes
    *along its own direction* is bridged: the two plates of a capacitor point
    sideways rather than across their gap, so they stay separate.
    """
    if FREE_END_MAX_GAP <= 0:
        return labels, []
    max_gap = max(3, round(FREE_END_MAX_GAP * image_scale))
    min_label_pixels = max(6, round(8 * image_scale))
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    label_pixels: dict[int, set[tuple[int, int]]] = {}
    for label in (int(v) for v in np.unique(labels) if v > 0):
        points = np.argwhere(labels == label)
        if len(points) >= min_label_pixels:
            label_pixels[label] = {(int(y), int(x)) for y, x in points}
    height, width = labels.shape
    events: list[dict] = []
    for label, pixels in label_pixels.items():
        for y, x in pixels:
            neighbours = [(yy, xx) for yy in (y - 1, y, y + 1) for xx in (x - 1, x, x + 1)
                          if (yy, xx) != (y, x) and (yy, xx) in pixels]
            if len(neighbours) != 1:
                continue
            ny, nx = neighbours[0]
            norm = float(np.hypot(ny - y, nx - x)) or 1.0
            step_y, step_x = (y - ny) / norm, (x - nx) / norm
            target = None
            for step in range(2, max_gap + 1):
                py = int(round(y + step_y * step))
                px = int(round(x + step_x * step))
                if not (0 <= py < height and 0 <= px < width):
                    break
                other = int(labels[py, px])
                if other == label:
                    break
                if other:
                    if other in label_pixels:
                        target = (py, px, other, step)
                    break
            if target is None:
                continue
            py, px, other, step = target
            if FREE_END_STRAIGHT_ONLY:
                # Only bridge a wire end that stops just before *crossing* another
                # wire (a T junction).  Bridging collinear ends instead would join
                # two wires that the drawing keeps apart (measured on case012).
                offsets = [(dy, dx) for dy in range(-6, 7) for dx in range(-6, 7)
                           if (dy or dx) and 0 <= py + dy < height and 0 <= px + dx < width
                           and int(labels[py + dy, px + dx]) == other]
                if len(offsets) >= 3:
                    ys = np.array([offset[0] for offset in offsets], dtype=float)
                    xs = np.array([offset[1] for offset in offsets], dtype=float)
                    spread = np.cov(np.vstack([xs, ys])) if xs.size > 2 else np.eye(2)
                    values, vectors = np.linalg.eigh(spread)
                    axis = vectors[:, -1]
                    alignment = abs(axis[0] * step_x + axis[1] * step_y)
                    if alignment > 0.6:
                        continue
            root, other_root = find(label), find(other)
            if root == other_root:
                continue
            parent[other_root] = root
            events.append({"first": label, "second": other, "gap": int(step),
                           "point": [int(x), int(y)], "target": [int(px), int(py)]})
    if events:
        lookup = np.arange(int(labels.max()) + 1, dtype=np.int32)
        for label in range(len(parent)):
            lookup[label] = find(label)
        labels = lookup[labels]
    return labels, events


def trust_summary(result: dict) -> dict:
    """Truth-free risk notes for one case, so a user knows what to double-check.

    The flags are calibrated on the 40 public drawings (see trust_probe.py):
    ``omitted``, ``small_two_pin`` and ``small_bjt`` mark the drawings where the
    detector is known to be unreliable (Res/Cap confusion, BJT polarity below
    ~45px), while ``bulk`` and ``supply_with_gate`` mark cases whose reading
    depends on a convention that varies between drawings.  A case without any
    flag is not guaranteed correct, only free of the known risk patterns.
    """
    components = [comp for comp in result.get("components", []) if not comp.get("marker")]
    omitted = [comp["ref"] for comp in components
               if len(comp.get("port_nets") or {}) != comp.get("pin_count", 2)]
    small_two_pin = [comp["ref"] for comp in components
                     if comp["component_type"] in {"Res", "Cap"}
                     and max(comp["bbox"][2] - comp["bbox"][0],
                             comp["bbox"][3] - comp["bbox"][1]) < 22]
    small_bjt = [comp["ref"] for comp in components
                 if comp["component_type"] in {"NPN", "PNP"}
                 and (comp["bbox"][3] - comp["bbox"][1]) < 45]
    bulk = [comp["ref"] for comp in components if "bulk" in comp["class_name"]]
    supply_with_gate = sorted({f"{comp['ref']}.{port}->{net}"
                               for comp in components
                               for port, net in (comp.get("port_nets") or {}).items()
                               if net in ("VDD", "VSS") and port in ("Gate", "Base")})
    reasons = []
    if omitted:
        reasons.append(f"{len(omitted)} component(s) missing a full port mapping: {', '.join(omitted)}")
    if small_two_pin:
        reasons.append(f"small two-terminal devices (R/C easily confused): {', '.join(small_two_pin)}")
    if small_bjt:
        reasons.append(f"small BJTs (polarity rule unreliable, box height <45px): {', '.join(small_bjt)}")
    if bulk:
        reasons.append(f"bulk-style devices (reference records Body for some drawings only): {', '.join(bulk)}")
    if supply_with_gate:
        reasons.append("gate/base attached to a supply net (possible marker over-merge): "
                       + ", ".join(supply_with_gate))
    if omitted or small_two_pin or small_bjt:
        level = "review"
    elif bulk or supply_with_gate:
        level = "caution"
    else:
        level = "high"
    return {"level": level, "reasons": reasons, "omitted": omitted,
            "small_two_pin": small_two_pin, "small_bjt": small_bjt,
            "bulk": bulk, "supply_with_gate": supply_with_gate}


def trace(image: np.ndarray, components: list[dict], image_scale: float = 1.0) -> tuple[dict, np.ndarray, list[tuple[tuple[float,float],tuple[float,float]]]]:
    """Build electrical nets directly from skeleton connected components."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Printed EDA wires can be light gray while text and symbols stay dark.
    ink = (gray < 240).astype(np.uint8)
    # Both supply symbols are small circles; the one with a filled arrow inside
    # is a current source even when the detector called it a voltage source.
    components = _reclassify_arrow_sources(
        components, ink, cv2.distanceTransform(ink, cv2.DIST_L2, 5))
    ink_distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    line_length = max(3, round(18 * image_scale))
    close_size = max(3, round(WIRE_CLOSE_SIZE * image_scale))
    if close_size % 2 == 0:
        close_size += 1
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length)))
    # Feedback paths in the EDA corpus can be drawn diagonally (case 030 has
    # two crossing 45-degree nets).  Horizontal/vertical openings erase those
    # paths entirely, leaving otherwise valid MOS drains on isolated labels.
    diagonal = np.zeros_like(ink)
    diagonal_length = max(3, round(DIAGONAL_LINE_LENGTH * image_scale))
    if diagonal_length % 2 == 0:
        diagonal_length += 1
    for angle_degrees in DIAGONAL_LINE_ANGLES:
        kernel = np.zeros((diagonal_length, diagonal_length), dtype=np.uint8)
        radius = (diagonal_length - 1) / 2
        angle = np.deg2rad(angle_degrees)
        dx, dy = radius * np.cos(angle), radius * np.sin(angle)
        center = radius
        start = (round(center - dx), round(center - dy))
        end = (round(center + dx), round(center + dy))
        cv2.line(kernel, start, end, 1, 1)
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        # A crossing is often drawn with a small white break so readers know
        # the two diagonal nets do not join. Close each orientation separately:
        # opposite collinear halves reconnect, while the other diagonal stays
        # on its own layer and cannot be shorted at the crossing.
        opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        diagonal = cv2.bitwise_or(diagonal, opened)
    linear_ink = cv2.bitwise_or(cv2.bitwise_or(horizontal, vertical), diagonal)
    binary = cv2.morphologyEx(linear_ink, cv2.MORPH_CLOSE,
                              np.ones((close_size, close_size), np.uint8)) > 0
    skel = skeletonize(binary).astype(np.uint8)
    h, w = skel.shape
    # Cut detected symbol bodies from the skeleton; wire fragments remain separate.
    for comp in components:
        # A port circle can also be a genuine branch point (for example an
        # amplifier output with a feedback wire).  Removing the whole marker
        # breaks every branch at that node, while leaving its ring in the
        # skeleton preserves the intended electrical connection.
        if comp.get("marker") == "port":
            continue
        x1,y1,x2,y2 = [int(round(v)) for v in comp["bbox"]]
        x1,x2=max(0,x1),min(w,x2); y1,y2=max(0,y1),min(h,y2)
        if x2>x1 and y2>y1: skel[y1:y2,x1:x2]=0
    labels, unjoined_crossing_events = _split_unjoined_crossings(skel, gray, components, image_scale)
    labels, bridge_events = _reconnect_bridges(labels, int(labels.max()) + 1, components, image_scale)
    labels, bus_events = _reconnect_mos_horizontal_buses(labels, components, image_scale, gray)
    labels, border_corner_events = _reconnect_border_corners(labels, image_scale)
    labels, free_end_events = _reconnect_free_ends(labels, int(labels.max()) + 1, image_scale)
    components_by_label = {int(lab): np.argwhere(labels == lab) for lab in np.unique(labels) if lab > 0}
    # Ignore small text strokes and isolated marks; actual wire nets span a useful distance.
    min_pixels = max(8, round(8 * image_scale))
    min_span = max(12, round(12 * image_scale))
    valid = {lab for lab, pts in components_by_label.items() if len(pts) >= min_pixels and max(np.ptp(pts[:,0]),np.ptp(pts[:,1])) >= min_span}
    # Components in series can be joined by a stub shorter than any real net.
    # Those fragments still carry a terminal, so sample every contact strip
    # once here and let the stub geometry test promote them to real labels.
    strips = {}
    for comp in components:
        if comp.get("marker"): continue
        split_contacts = comp["class_name"] in {"single-end-amp", "diff-amp"}
        base_radius, wide_radius = _strip_radii(comp, image_scale)
        found = _contact_labels(labels, comp["bbox"], radius=base_radius,
                                split_same_label_contacts=split_contacts)
        expanded = found if wide_radius == base_radius else _contact_labels(
            labels, comp["bbox"], radius=wide_radius,
            split_same_label_contacts=split_contacts)
        strips[comp["ref"]] = (comp, base_radius, wide_radius, found, expanded)
    valid |= _stub_label_ids(strips, components_by_label, image_scale)
    # Parts drawn with touching boxes have no wire between them at all.  Join
    # the two facing terminals on a fresh label, but only when neither side
    # already found a real net, so existing connections are never rerouted.
    body_bridge_events = []
    next_label = int(labels.max()) + 1
    for pair in _touching_body_pairs(components, image_scale):
        first_strip = strips[pair["first"]["ref"]]
        second_strip = strips[pair["second"]["ref"]]
        first_has_net = any(key[0] == pair["first_side"] and key[1] in valid
                            for key in first_strip[3])
        second_has_net = any(key[0] == pair["second_side"] and key[1] in valid
                             for key in second_strip[3])
        if first_has_net or second_has_net:
            continue
        new_label = next_label
        next_label += 1
        for point in (pair["first_point"], pair["second_point"]):
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= y < h and 0 <= x < w:
                labels[y, x] = new_label
        components_by_label[new_label] = np.argwhere(labels == new_label)
        valid.add(new_label)
        first_strip[3][(pair["first_side"], new_label, 0)] = tuple(
            int(round(v)) for v in pair["first_point"])
        second_strip[3][(pair["second_side"], new_label, 0)] = tuple(
            int(round(v)) for v in pair["second_point"])
        body_bridge_events.append({
            "first": pair["first"]["ref"], "second": pair["second"]["ref"],
            "first_side": pair["first_side"], "second_side": pair["second_side"],
            "net": new_label})
    port_maps = {}
    contact_points = {}
    for comp in components:
        if comp.get("marker"): continue
        split_contacts = comp["class_name"] in {"single-end-amp", "diff-amp"}
        _, radius, fallback_radius, raw_found, raw_expanded = strips[comp["ref"]]
        # A part's printed text also sits next to its box, and a letter stroke is
        # long and thin enough to pass the net-validity filter.  Real terminals
        # start at the cut edge of the box, so the normal pass only accepts
        # labels that actually reach it; the widened pass below stays permissive
        # for boxes that stop a few pixels short of the wire.
        touch_reach = max(1.0, CONTACT_TOUCH_REACH * image_scale)
        distances = {key: _box_distance(components_by_label[key[1]], comp["bbox"])
                     for key in raw_found if key[1] in valid}
        touching: dict[str, int] = {}
        counts: dict[str, int] = {}
        for key, distance in distances.items():
            side = key[0]
            counts[side] = counts.get(side, 0) + 1
            if distance <= touch_reach:
                touching[side] = touching.get(side, 0) + 1
        found = {key: point for key, point in raw_found.items()
                 if key[1] in valid
                 and (distances[key] <= touch_reach
                      or (touching.get(key[0], 0) == 0 and counts[key[0]] == 1))}
        mapping = _assign_contacts(comp, found, ink, ink_distance)
        mapping = _promote_amp_arity(comp, mapping)
        implied = _bjt_arrow_polarity(comp, mapping, found, ink_distance)
        if implied is not None:
            # The arrow is on the emitter lead and the detector called it PNP;
            # the drawings label this glyph NPN, and the port mapping (collector
            # on the lower lead, emitter on the upper one) already matches.
            comp.update(class_name="npn", kind="Q", component_type="NPN",
                        port_names=["Collector", "Base", "Emitter"], pin_count=3)
        # A YOLO box can end a few pixels before a real terminal.  Do not make
        # every component more permissive: only expand the strip when its
        # normal-radius pass did not recover every named terminal.
        if len(mapping) < comp["pin_count"] and fallback_radius > radius:
            expanded = {key:point for key,point in raw_expanded.items() if key[1] in valid}
            expanded_mapping = _assign_contacts(comp, expanded, ink, ink_distance)
            expanded_mapping = _promote_amp_arity(comp, expanded_mapping)
            if len(expanded_mapping) > len(mapping):
                found, mapping = expanded, expanded_mapping
        port_maps[comp["ref"]] = mapping
        contact_points[comp["ref"]] = {
            name: next((list(found[key]) for key in found if key[1] == label), [0, 0])
            for name, label in port_maps[comp["ref"]].items()
        }
    label_to_net = {lab:f"N{i:03d}" for i,lab in enumerate(sorted(valid),1)}
    # Canonical supply markers merge same-rail components, as required by EDA.
    for marker_name, canonical in (("vdd","VDD"),("gnd","VSS")):
        matched=set()
        for marker in (c for c in components if c.get("marker")==marker_name):
            # A supply symbol only merges the wires that actually reach it.  An
            # earlier version took every label inside a 22px circle around the
            # marker centre, which also swallowed wires that merely passed by
            # (case011/013 pulled a MOS gate onto VDD that way).
            reach = max(1.0, SUPPLY_MARKER_TOUCH_REACH * image_scale)
            for label, points in components_by_label.items():
                if label in valid and _box_distance(points, marker["bbox"]) <= reach:
                    matched.add(label)
        if matched:
            for lab in matched: label_to_net[lab]=canonical
    # In this EDA dataset VSS is drawn as the long bottom rail (the label is
    # intentionally not used as an OCR signal).  The rule also runs when a GND
    # symbol was detected, because such drawings can carry both: case038 has a
    # GND symbol on a signal group *and* the bottom rail as the real ground, and
    # the truth merges them into one canonical node.
    bottom = [(lab, pts) for lab, pts in components_by_label.items() if lab in valid
              and np.max(pts[:, 0]) >= h - max(5, round(40 * image_scale))
              and np.ptp(pts[:, 1]) >= max(5, round(18 * image_scale))]
    if any(c.get("marker") == "gnd" for c in components):
        # When the page already carries a ground symbol, only a rail that runs
        # almost across the whole drawing is treated as the same ground (case038
        # spans 90% of the width).  Shorter bottom rails are often a separate
        # node in these drawings (case012 keeps its 80% rail apart from the
        # symbol's ground, case003's is a signal rail).
        limit = BOTTOM_RAIL_FULL_WIDTH_RATIO * w
        bottom = [(lab, pts) for lab, pts in bottom if np.ptp(pts[:, 1]) >= limit]
    if bottom:
        for lab, _ in bottom:
            label_to_net[lab] = "VSS"
    # Compact net aliases after supply merging.
    unique = {}
    for lab in sorted(valid): unique.setdefault(label_to_net[lab], label_to_net[lab])
    next_idx=1
    for name in list(unique):
        if name not in ("VDD","VSS"):
            unique[name]=f"N{next_idx:03d}"; next_idx+=1
    label_to_net={lab:unique[name] for lab,name in label_to_net.items()}
    rows=[]; warnings=[]
    for comp in components:
        if comp.get("marker"): continue
        mapping=port_maps.get(comp["ref"],{})
        ordered=[mapping.get(p) for p in comp["port_names"]]
        net_ids=[label_to_net.get(v) if v is not None else None for v in ordered]
        if any(v is None for v in net_ids): warnings.append(f"{comp['ref']}: mapped {sum(v is not None for v in net_ids)}/{comp['pin_count']} pins")
        rows.append({**comp,"contacts":[contact_points[comp["ref"]].get(p,[0,0]) for p in comp["port_names"] if p in mapping],"nets":[v for v in net_ids if v is not None],"port_nets":{p:label_to_net[v] for p,v in mapping.items() if v in label_to_net}})
    nets={}
    for lab in valid:
        name=label_to_net[lab]
        nets.setdefault(name,[]).extend([[float(x),float(y)] for y,x in components_by_label[lab]])
    junctions=[]
    for lab in valid:
        pts=components_by_label[lab]
        mask=(labels==lab).astype(np.uint8)
        ys,xs=np.where(mask)
        for y,x in zip(ys,xs):
            patch=labels[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)]
            if np.count_nonzero(patch==lab)>=4:
                junctions.append({"ref":f"J{len(junctions)+1}","point":[float(x),float(y)],"net":label_to_net[lab]})
                break
    result={"components":rows,"nets":nets,"junctions":junctions,"connected_component_pairs":[],"warnings":warnings,
            "pixel_topology":True,"skeleton_pixels":int(skel.sum()),"wire_components":len(valid),
            "bridge_events":bridge_events,"unjoined_crossing_events":unjoined_crossing_events,
            "mos_bus_events":bus_events, "border_corner_events": border_corner_events,
            "body_bridge_events": body_bridge_events, "free_end_events": free_end_events}
    return result, (skel*255), []
