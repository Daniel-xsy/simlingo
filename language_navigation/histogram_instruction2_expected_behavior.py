#!/usr/bin/env python3
"""
Compute histogram for instruction id=2 expected_behavior in XML files.

Default input directory is ./temp (recursive).
Example:
    python language_navigation/histogram_instruction2_expected_behavior.py
    python language_navigation/histogram_instruction2_expected_behavior.py --input-dir ./tmp
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MISSING_KEY = "__MISSING_EXPECTED_BEHAVIOR__"
EMPTY_KEY = "__EMPTY_EXPECTED_BEHAVIOR__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute histogram of <expected_behavior> for instruction id=2 "
            "across XML files in a directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./tmp"),
        help="Directory containing XML files (searched recursively). Default: ./temp",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print histogram as JSON instead of table format.",
    )
    return parser.parse_args()


def canonical_expected_behavior(expected_behavior_elem: ET.Element) -> str:
    if not expected_behavior_elem.attrib:
        return EMPTY_KEY
    items: List[Tuple[str, str]] = sorted(expected_behavior_elem.attrib.items())
    return ", ".join(f"{key}={value}" for key, value in items)


def iter_xml_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.rglob("*.xml"))


def compute_histogram(xml_files: Iterable[Path]) -> Tuple[Counter, int, int, int]:
    histogram: Counter = Counter()
    files_scanned = 0
    parse_failures = 0
    instruction_hits = 0

    for xml_path in xml_files:
        files_scanned += 1
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            parse_failures += 1
            continue

        for instruction in root.findall(".//instruction[@id='2']"):
            instruction_hits += 1
            expected_behavior = instruction.find("expected_behavior")
            if expected_behavior is None:
                histogram[MISSING_KEY] += 1
                continue
            histogram[canonical_expected_behavior(expected_behavior)] += 1

    return histogram, files_scanned, parse_failures, instruction_hits


def print_table(histogram: Dict[str, int], total: int) -> None:
    if total == 0:
        print("No instruction id=2 entries found.")
        return

    print("Histogram for instruction id=2 expected_behavior")
    print("-" * 72)
    print(f"{'Count':>8}  {'Percent':>8}  Value")
    print("-" * 72)
    for key, count in sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0])):
        percent = 100.0 * count / total
        print(f"{count:8d}  {percent:7.2f}%  {key}")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser()

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        return 1

    xml_files = list(iter_xml_files(input_dir))
    histogram, files_scanned, parse_failures, instruction_hits = compute_histogram(xml_files)

    print(
        f"Scanned {files_scanned} XML files from {input_dir} "
        f"(parse failures: {parse_failures})."
    )
    print(f"Found {instruction_hits} instruction id=2 entries.")

    if args.output_json:
        print(json.dumps(dict(sorted(histogram.items())), indent=2))
    else:
        print_table(dict(histogram), instruction_hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
