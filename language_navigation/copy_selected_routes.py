#!/usr/bin/env python3
"""Copy generated language XMLs for selected Bench2Drive route ids."""

import argparse
import shutil
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy generated language XMLs for route ids listed in a text file."
    )
    parser.add_argument(
        "--select-file",
        type=Path,
        default=Path("language_navigation/route.txt"),
        help="Text file containing one Bench2Drive route id per line.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("leaderboard/data/language_benchmark/instruction_following_v0.5"),
        help="Directory containing generated language XML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("leaderboard/data/language_benchmark/instruction_following_v0.5_selected"),
        help="Directory to copy the selected XML files into.",
    )
    return parser.parse_args()


def _read_selected_route_ids(select_file: Path) -> List[str]:
    route_ids: List[str] = []
    for line in select_file.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        route_ids.append(f"{int(token):02d}" if token.isdigit() else token)
    return route_ids


def main() -> None:
    args = parse_args()
    select_file = args.select_file.resolve()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not select_file.is_file():
        raise FileNotFoundError(f"Selection file not found: {select_file}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_route_ids = _read_selected_route_ids(select_file)
    copied = 0

    for route_id in selected_route_ids:
        pattern = f"bench2drive_{route_id}_language_*.xml"
        matches = sorted(source_dir.glob(pattern))
        if not matches:
            print(f"[WARN] No generated XML files found for route {route_id}")
            continue

        for src_path in matches:
            dst_path = output_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            copied += 1
            print(f"Copied: {src_path.name}")

    print(f"Copied {copied} files into {output_dir}")


if __name__ == "__main__":
    main()
