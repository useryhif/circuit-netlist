"""Project interface: one schematic image in, one netlist dict out.

The public interface is a single Python function: an image path in, a netlist dict out.

    from api import predict
    result = predict("images/001.png")      # {'ckt_type': ..., 'ckt_netlist': [...]}

Command line usage (prints the same dictionary, for direct comparison with a reference netlist):

    python api.py images/001.png
    python api.py images/001.png --out output/run_001

Optional `--labels/--aux-labels` reuse cached YOLO labels to skip inference while debugging.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

import pipeline as P
from pixel_topology import trace as trace_pixel_topology

ROOT = Path(__file__).resolve().parent
FALLBACK_TYPE = "DISO-Amplifier"


def predict(image_path: str | Path, *, out: str | Path | None = None,
            labels: str | Path | None = None,
            aux_labels: str | Path | None = None,
            aux_label_data: str | Path | None = None,
            classify: bool = True) -> dict:
    """Convert one schematic image into the target netlist dictionary.

    ``ckt_netlist`` only contains components whose every named port reached a
    net; components that could not be mapped are listed in the pipeline's
    warnings (and in the readable topology when ``out`` is given).
    """
    image_path = Path(image_path).resolve()
    out_dir = Path(out).resolve() if out else ROOT / "output" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    source = P.read_image(image_path)
    scale = 640 / source.shape[1]
    image = cv2.resize(source, (640, round(source.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    if labels:
        components = P.decode_yolo_labels(image, Path(labels).resolve())
        if aux_labels:
            auxiliary = P.decode_yolo_labels(
                image, Path(aux_labels).resolve(),
                Path(aux_label_data).resolve() if aux_label_data else P.AUX_DATA_YAML)
            components = P.fuse_detections(components, auxiliary)
    else:
        components, _ = P.detect_objects(image, image_path, out_dir)
    if not components:
        raise RuntimeError("YOLOv5 found no components; cannot produce a trustworthy netlist")

    result, skeleton, lines = trace_pixel_topology(image, components)
    rows = P.netlist_rows(result)
    netlist = {"ckt_netlist": rows, "ckt_type": FALLBACK_TYPE}
    if classify:
        try:
            from eda_gcn import predict as classify_circuit
            netlist["ckt_type"] = classify_circuit(netlist)[0]
        except (FileNotFoundError, RuntimeError):
            pass

    if out:
        from diagram_versions import draw_detection, draw_polarity, infer_polarity
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
        P.write_debug(image, result, lines, out_dir / "02_connected_nodes.png", skeleton)
        polarity = infer_polarity(image, result)
        draw_polarity(image, result, lines, polarity, out_dir / "03_polarity.png", skeleton)
        draw_detection(image, components, [], lines, out_dir / "01_detection.png")
        (out_dir / "netlist.txt").write_text(str(netlist), encoding="utf-8")
        (out_dir / "debug" / "topology.json").write_text(
            json.dumps({"components": result["components"], "nets": result["nets"],
                        "junctions": result["junctions"], "warnings": result["warnings"],
                        "trust": P.trust_summary(result)},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    return netlist


def main() -> int:
    parser = argparse.ArgumentParser(description="Schmatic image -> netlist dictionary")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, help="also write figures and debug JSON here")
    parser.add_argument("--labels", type=Path, help="cached primary YOLO label file")
    parser.add_argument("--aux-labels", type=Path, help="cached auxiliary YOLO label file")
    parser.add_argument("--aux-label-data", type=Path, help="class yaml for the auxiliary labels")
    parser.add_argument("--no-classify", action="store_true", help="skip the GCN circuit-type head")
    args = parser.parse_args()

    netlist = predict(args.image, out=args.out, labels=args.labels,
                         aux_labels=args.aux_labels, aux_label_data=args.aux_label_data,
                         classify=not args.no_classify)
    print(f"ckt_type: {netlist['ckt_type']}")
    print(f"components: {len(netlist['ckt_netlist'])}")
    print(netlist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
