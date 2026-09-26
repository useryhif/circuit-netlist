"""Validate generated Python-dict netlists against the project netlist contract."""
from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "final"
TYPES = {"DISO-Amplifier", "DIDO-Amplifier", "SISO-Amplifier", "Bandgap", "LDO", "Comparator"}
PORTS = {
    "PMOS": {"Drain", "Gate", "Source", "Body"}, "NMOS": {"Drain", "Gate", "Source", "Body"},
    "PNP": {"Collector", "Base", "Emitter"}, "NPN": {"Collector", "Base", "Emitter"},
    "Res": {"Pos", "Neg"}, "Cap": {"Pos", "Neg"}, "Ind": {"Pos", "Neg"},
    "Current": {"In", "Out"}, "Voltage": {"Positive", "Negative"}, "Diode": {"In", "Out"},
    "Switch": {"Pos", "Neg"}, "Diso_amp": {"InN", "InP", "Out"},
    "Dido_amp": {"InN", "InP", "OutN", "OutP"}, "Siso_amp": {"In", "Out"},
}


def validate(path: Path) -> list[str]:
    errors = []
    try:
        obj = ast.literal_eval(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid Python dict literal: {exc}"]
    if not isinstance(obj, dict) or set(obj) != {"ckt_netlist", "ckt_type"}:
        return ["top-level keys must be exactly ckt_netlist and ckt_type"]
    if obj["ckt_type"] not in TYPES:
        errors.append(f"unknown ckt_type: {obj['ckt_type']!r}")
    if not isinstance(obj["ckt_netlist"], list):
        return errors + ["ckt_netlist must be a list"]
    for i, row in enumerate(obj["ckt_netlist"]):
        if not isinstance(row, dict) or set(row) != {"component_type", "port_connection"}:
            errors.append(f"row {i}: expected component_type and port_connection")
            continue
        kind = row["component_type"]
        if kind not in PORTS:
            errors.append(f"row {i}: unsupported component_type {kind!r}")
            continue
        mapping = row["port_connection"]
        if not isinstance(mapping, dict) or not mapping:
            errors.append(f"row {i}: empty or invalid port_connection")
            continue
        if not set(mapping).issubset(PORTS[kind]):
            errors.append(f"row {i}: invalid ports {sorted(set(mapping)-PORTS[kind])}")
        if not all(isinstance(net, str) and net for net in mapping.values()):
            errors.append(f"row {i}: net names must be nonempty strings")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated netlist files")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    files = sorted(p for p in output.glob("[0-9][0-9][0-9].txt"))
    bad = []
    for path in files:
        errors = validate(path)
        if errors: bad.append({"file": path.name, "errors": errors})
    report = {"files_checked": len(files), "valid": len(files)-len(bad), "invalid": bad}
    (output / "format_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
