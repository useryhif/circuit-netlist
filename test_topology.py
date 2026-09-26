"""Small geometry cases for the DFS net assignment."""

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from diagram_versions import detect_voltage_sign, infer_polarity
from pipeline import build_topology, decode_yolo_labels, fuse_bjt_polarity, fuse_mos_bulk_variants
from pixel_topology import (_assign_contacts, _opposite_arm_pairs, _reconnect_border_corners,
                            _reconnect_bridges)


def part(ref, bbox):
    return {"ref": ref, "kind": ref[0], "bbox": bbox, "confidence": 1.0}


def diagonal_gap_labels():
    """Four diagonal arms around an empty 20x20 cut box centred at (30, 30)."""
    labels = np.zeros((60, 60), dtype=np.int32)
    for step in range(9):
        labels[18 - step, 18 - step] = 1      # NW arm
        labels[18 - step, 42 + step] = 2      # NE arm
        labels[42 + step, 42 + step] = 3      # SE arm
        labels[42 + step, 18 - step] = 4      # SW arm
    return labels


class TopologyTests(unittest.TestCase):
    def test_small_border_corner_gap_is_reconnected(self):
        labels = np.zeros((80, 100), dtype=np.int32)
        labels[10, 10:70] = 1
        labels[14:70, 69] = 2
        repaired, events = _reconnect_border_corners(labels)
        self.assertTrue(events)
        self.assertEqual(repaired[10, 69], repaired[14, 69])

    def test_auxiliary_detector_only_resolves_overlapping_bjt_polarity(self):
        primary = [{"class_name": "npn", "bbox": [10, 10, 30, 40], "kind": "Q",
                    "component_type": "NPN", "port_names": ["Collector", "Base", "Emitter"],
                    "pin_count": 3}]
        auxiliary = [{"class_name": "pnp", "bbox": [11, 10, 31, 40]}]
        fused = fuse_bjt_polarity(primary, auxiliary)
        self.assertEqual(fused[0]["class_name"], "pnp")
        self.assertEqual(fused[0]["component_type"], "PNP")

    def test_auxiliary_detector_promotes_overlapping_same_family_bulk_mos(self):
        primary = [{"class_name": "pmos", "bbox": [10, 10, 30, 40], "kind": "M",
                    "component_type": "PMOS", "port_names": ["Drain", "Gate", "Source"],
                    "pin_count": 3}]
        auxiliary = [{"class_name": "pmos-bulk", "bbox": [11, 10, 31, 40]}]
        fused = fuse_mos_bulk_variants(primary, auxiliary)
        self.assertEqual(fused[0]["class_name"], "pmos-bulk")
        self.assertEqual(fused[0]["port_names"], ["Drain", "Gate", "Source", "Body"])

    def test_overlapping_marker_classes_keep_higher_confidence(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.yaml"
            labels = root / "sample.txt"
            data.write_text("names: [gnd, port]\n", encoding="utf-8")
            labels.write_text("0 0.5 0.5 0.2 0.2 0.35\n1 0.5 0.5 0.2 0.2 0.80\n",
                              encoding="utf-8")
            decoded = decode_yolo_labels(image, labels, data)
        self.assertEqual([item["marker"] for item in decoded], ["port"])

    def test_voltage_lines_detector_class_emits_voltage_component(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.yaml"
            labels = root / "sample.txt"
            data.write_text("names: [voltage-lines]\n", encoding="utf-8")
            labels.write_text("0 0.5 0.5 0.2 0.3 0.95\n", encoding="utf-8")
            decoded = decode_yolo_labels(image, labels, data)
        self.assertEqual(decoded[0]["component_type"], "Voltage")
        self.assertEqual(decoded[0]["port_names"], ["Positive", "Negative"])

    def test_mos_channel_contacts_are_opposite_the_gate(self):
        comp = {"bbox": [10, 10, 30, 30], "class_name": "nmos"}
        found = {("left", 1, 0): (9, 20),
                 ("top", 2, 0): (11, 9), ("top", 3, 0): (29, 9),
                 ("bottom", 4, 0): (11, 31), ("bottom", 5, 0): (29, 31)}
        self.assertEqual(_assign_contacts(comp, found),
                         {"Gate": 1, "Drain": 3, "Source": 5})

    def test_bulk_mos_body_is_internally_tied_to_source(self):
        comp = {"bbox": [10, 10, 30, 30], "class_name": "pmos-bulk"}
        found = {("left", 1, 0): (9, 20),
                 ("top", 2, 0): (29, 9),
                 ("bottom", 3, 0): (29, 31)}
        self.assertEqual(_assign_contacts(comp, found),
                         {"Gate": 1, "Source": 2, "Drain": 3, "Body": 2})

    def test_bjt_vertical_terminals_follow_symbol_polarity(self):
        found = {("left", 1, 0): (9, 20),
                 ("top", 2, 0): (20, 9),
                 ("bottom", 3, 0): (20, 31)}
        common = {"bbox": [10, 10, 30, 30]}
        npn = _assign_contacts({**common, "class_name": "npn"}, found)
        pnp = _assign_contacts({**common, "class_name": "pnp"}, found)
        self.assertEqual(npn, {"Base": 1, "Collector": 2, "Emitter": 3})
        self.assertEqual(pnp, {"Base": 1, "Collector": 3, "Emitter": 2})

    def test_diagonal_arms_pair_along_each_crossing_line(self):
        labels = diagonal_gap_labels()
        pairs, detail, _ = _opposite_arm_pairs(labels, (30, 30), 0.0, 24.0, 2,
                                               (20, 20, 40, 40))
        self.assertEqual(sorted(tuple(sorted(pair)) for pair in pairs), [(1, 3), (2, 4)])
        self.assertEqual(len(detail), 2)

    def test_right_angle_arms_are_not_paired(self):
        labels = diagonal_gap_labels()
        labels[labels == 2] = 0   # keep NW + SE + SW: no straight line through the gap
        labels[labels == 3] = 0
        pairs, _, _ = _opposite_arm_pairs(labels, (30, 30), 0.0, 24.0, 2, (20, 20, 40, 40))
        self.assertEqual(pairs, [])

    def test_curved_crossover_marker_pairs_diagonal_arms(self):
        labels = diagonal_gap_labels()
        marker = {"ref": "P1", "kind": "P", "class_name": "cross-line-curved",
                  "marker": "cross-line-curved", "bbox": [20.0, 20.0, 40.0, 40.0],
                  "confidence": 1.0}
        reconnected, events = _reconnect_bridges(labels.copy(), int(labels.max()) + 1, [marker])
        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0]["opposite_arm_pairs"]), 2)
        self.assertEqual(reconnected[12, 12], reconnected[48, 48])
        self.assertEqual(reconnected[12, 48], reconnected[48, 12])
        self.assertNotEqual(reconnected[12, 12], reconnected[12, 48])

    def test_curved_crossover_marker_keeps_horizontal_and_vertical_nets(self):
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[29:32, 5:19] = 1
        labels[29:32, 41:55] = 2
        labels[5:19, 29:32] = 3
        labels[41:55, 29:32] = 4
        marker = {"ref": "P1", "kind": "P", "class_name": "cross-line-curved",
                  "marker": "cross-line-curved", "bbox": [20.0, 20.0, 40.0, 40.0],
                  "confidence": 1.0}
        reconnected, events = _reconnect_bridges(labels.copy(), int(labels.max()) + 1, [marker])
        self.assertTrue(events[0]["horizontal_reconnected"])
        self.assertTrue(events[0]["vertical_reconnected"])
        self.assertEqual(events[0]["opposite_arm_pairs"], [])
        self.assertEqual(reconnected[30, 10], reconnected[30, 50])
        self.assertEqual(reconnected[10, 30], reconnected[50, 30])
        self.assertNotEqual(reconnected[30, 10], reconnected[10, 30])

    def test_crossing_joins_all_four_branches(self):
        components = [
            part("R1", [0, 40, 20, 60]),
            part("R2", [80, 40, 100, 60]),
            part("C1", [40, 0, 60, 20]),
            part("C2", [40, 80, 60, 100]),
        ]
        wires = [
            ((20, 50), (80, 50)),
            ((50, 20), (50, 80)),
            ((0, 50), (-10, 50)),
            ((100, 50), (110, 50)),
            ((50, 0), (50, -10)),
            ((50, 100), (50, 110)),
        ]
        result = build_topology(components, wires)
        center = next(j for j in result["junctions"] if j["point"] == [50.0, 50.0])
        self.assertEqual(center["connected_components"], ["C1", "C2", "R1", "R2"])
        self.assertEqual(len(result["connected_component_pairs"]), 6)

    def test_device_body_separates_two_nets(self):
        components = [part("R1", [40, 40, 60, 60])]
        wires = [((0, 50), (100, 50))]
        result = build_topology(components, wires)
        self.assertEqual(len(result["components"][0]["nets"]), 2)
        self.assertNotEqual(*result["components"][0]["nets"])

    def test_missing_contact_is_reported(self):
        components = [part("R1", [40, 40, 60, 60])]
        wires = [((0, 50), (50, 50))]
        result = build_topology(components, wires)
        self.assertEqual(len(result["components"][0]["nets"]), 1)
        self.assertTrue(result["warnings"])


def sample_source_image():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.line(image, (43, 39), (57, 39), (0, 0, 0), 2)
    cv2.line(image, (50, 32), (50, 46), (0, 0, 0), 2)
    cv2.line(image, (43, 62), (57, 62), (0, 0, 0), 2)
    return image


def source(ref, positive, negative):
    return {"ref": ref, "kind": "V", "bbox": [20, 20, 80, 80],
            "contacts": [[50, 20], [50, 80]], "nets": [positive, negative]}


def circuit(components):
    return {"components": components,
            "nets": {net: [] for comp in components for net in comp["nets"]}}


class PolarityTests(unittest.TestCase):
    def test_source_plus_is_read_from_symbol(self):
        detected = detect_voltage_sign(sample_source_image(), source("V1", "N1", "N3"))
        self.assertEqual(detected["plus_index"], 0)

    def test_horizontal_source_side_is_not_reversed(self):
        image = cv2.rotate(sample_source_image(), cv2.ROTATE_90_COUNTERCLOCKWISE)
        comp = {"ref": "V1", "kind": "V", "bbox": [20, 20, 80, 80],
                "contacts": [[20, 50], [80, 50]], "nets": ["N1", "N3"]}
        detected = detect_voltage_sign(image, comp)
        self.assertEqual(detected["plus_index"], 0)

    def test_every_terminal_inherits_its_node_polarity(self):
        components = [source("V1", "N1", "N3"),
                      {"ref": "R1", "kind": "R", "contacts": [[0, 0], [1, 1]], "nets": ["N1", "N2"]},
                      {"ref": "L1", "kind": "L", "contacts": [[0, 0], [1, 1]], "nets": ["N2", "N3"]}]
        result = infer_polarity(sample_source_image(), circuit(components))
        nodes = {n["id"]: n["polarity"] for n in result["nodes"]}
        entries = {e["id"]: e for e in result["components"]}
        self.assertEqual(nodes, {"N1": "+", "N2": "?", "N3": "-"})
        self.assertEqual(entries["V1"]["terminals"],
                         [{"node": "N1", "polarity": "+"}, {"node": "N3", "polarity": "-"}])
        self.assertEqual(entries["R1"]["terminals"][1]["polarity"], "?")
        self.assertEqual(entries["L1"]["terminals"][0]["polarity"], "?")

    def test_two_sources_use_source_specific_node_labels(self):
        components = [source("V1", "N1", "N3"), source("V2", "N2", "N3"),
                      {"ref": "R1", "kind": "R", "contacts": [[0, 0], [1, 1]], "nets": ["N1", "N2"]}]
        result = infer_polarity(sample_source_image(), circuit(components))
        nodes = {n["id"]: n["polarity"] for n in result["nodes"]}
        self.assertEqual(nodes, {"N1": "V1+", "N2": "V2+", "N3": "V1-/V2-"})
        resistor = next(e for e in result["components"] if e["id"] == "R1")
        self.assertEqual([t["polarity"] for t in resistor["terminals"]], ["V1+", "V2+"])

    def test_diode_does_not_make_internal_node_a_source_terminal(self):
        components = [source("V1", "N1", "N3"),
                      {"ref": "D1", "kind": "D", "contacts": [[0, 0], [1, 1]], "nets": ["N1", "N2"]},
                      {"ref": "R1", "kind": "R", "contacts": [[0, 0], [1, 1]], "nets": ["N2", "N3"]}]
        result = infer_polarity(sample_source_image(), circuit(components))
        nodes = {n["id"]: n["polarity"] for n in result["nodes"]}
        self.assertEqual(nodes["N2"], "?")


if __name__ == "__main__":
    unittest.main()
