#!/usr/bin/env python3
"""Copy a balanced subset of generated language XMLs by filename category."""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set


CATEGORY_TOKEN = "_language_rebuilt_"
DEFAULT_EXCLUDED_CATEGORIES = ("lane_change_left", "lane_change_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a balanced subset of generated language XML files, grouped by "
            "the category suffix in each filename."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("leaderboard/data/language_benchmark/instruction_following_full"),
        help="Directory containing the full generated language XML dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "leaderboard/data/language_benchmark/instruction_following_full_category50"
        ),
        help="Directory to copy the balanced XML subset into.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=50,
        help="Target number of XML files to copy per included category.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling within a category.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional list of categories to balance. Defaults to all non-excluded categories found.",
    )
    parser.add_argument(
        "--seed-route-file",
        type=Path,
        default=Path("language_navigation/route.txt"),
        help=(
            "Route-id list to include first. All matching XMLs are copied before "
            "per-category top-up."
        ),
    )
    parser.add_argument(
        "--exclude-categories",
        nargs="+",
        default=list(DEFAULT_EXCLUDED_CATEGORIES),
        help="Categories to exclude from balancing and shortage checks.",
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help=(
            "Copy all available XML files for underrepresented categories instead "
            "of failing."
        ),
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove existing XML files from the output directory before copying.",
    )
    return parser.parse_args()


def _extract_category(xml_path: Path) -> str:
    stem = xml_path.stem
    if CATEGORY_TOKEN not in stem:
        raise ValueError(
            f"Could not extract category from filename {xml_path.name!r}; "
            f"expected token {CATEGORY_TOKEN!r}."
        )
    return stem.split(CATEGORY_TOKEN, 1)[1]


def _group_xmls_by_category(source_dir: Path) -> Dict[str, List[Path]]:
    grouped_paths: DefaultDict[str, List[Path]] = defaultdict(list)
    for xml_path in sorted(source_dir.glob("*.xml")):
        grouped_paths[_extract_category(xml_path)].append(xml_path)
    return dict(sorted(grouped_paths.items()))


def _read_selected_route_ids(select_file: Path) -> List[str]:
    route_ids: List[str] = []
    for line in select_file.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        route_ids.append(f"{int(token):02d}" if token.isdigit() else token)
    return route_ids


def _collect_seed_paths(source_dir: Path, seed_route_file: Path) -> List[Path]:
    if not seed_route_file.is_file():
        raise FileNotFoundError(f"Seed route file not found: {seed_route_file}")

    seed_paths: List[Path] = []
    for route_id in _read_selected_route_ids(seed_route_file):
        seed_paths.extend(sorted(source_dir.glob(f"bench2drive_{route_id}_language_*.xml")))
    return seed_paths


def _validate_output_dir(output_dir: Path, clear_output: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    existing_xmls = sorted(output_dir.glob("*.xml"))
    if not existing_xmls:
        return

    if clear_output:
        for xml_path in existing_xmls:
            xml_path.unlink()
        return

    raise FileExistsError(
        f"Output directory {output_dir} already contains {len(existing_xmls)} XML files. "
        "Use --clear-output to replace them or choose a different --output-dir."
    )


def _select_categories(
    grouped_paths: Dict[str, List[Path]],
    requested_categories: Optional[List[str]],
    excluded_categories: Set[str],
) -> List[str]:
    if requested_categories is None:
        return sorted(
            category
            for category in grouped_paths
            if category not in excluded_categories
        )

    missing_categories = [
        category for category in requested_categories if category not in grouped_paths
    ]
    if missing_categories:
        raise ValueError(
            "Requested categories were not found in the source directory: "
            + ", ".join(sorted(missing_categories))
        )
    selected = sorted(
        category
        for category in dict.fromkeys(requested_categories)
        if category not in excluded_categories
    )
    if not selected:
        raise ValueError("No categories remain after applying --exclude-categories.")
    return selected


def _summarize_shortages(
    grouped_paths: Dict[str, List[Path]],
    categories: List[str],
    samples_per_category: int,
) -> Dict[str, int]:
    return {
        category: len(grouped_paths[category])
        for category in categories
        if len(grouped_paths[category]) < samples_per_category
    }


def _summarize_all_counts(
    grouped_paths: Dict[str, List[Path]], categories: List[str]
) -> str:
    return ", ".join(
        f"{category}={len(grouped_paths[category])}" for category in categories
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    seed_route_file = args.seed_route_file.resolve()
    excluded_categories = set(args.exclude_categories)

    if args.samples_per_category <= 0:
        raise ValueError("--samples-per-category must be positive.")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    grouped_paths = _group_xmls_by_category(source_dir)
    if not grouped_paths:
        raise FileNotFoundError(f"No XML files found in source directory: {source_dir}")

    categories = _select_categories(grouped_paths, args.categories, excluded_categories)
    seed_paths = _collect_seed_paths(source_dir, seed_route_file)
    selected_paths: Dict[str, Path] = {path.name: path for path in seed_paths}

    shortages = _summarize_shortages(
        grouped_paths=grouped_paths,
        categories=categories,
        samples_per_category=args.samples_per_category,
    )
    if shortages and not args.allow_fewer:
        shortage_summary = ", ".join(
            f"{category}={count}" for category, count in sorted(shortages.items())
        )
        full_count_summary = _summarize_all_counts(
            grouped_paths=grouped_paths, categories=categories
        )
        raise ValueError(
            "Source directory does not contain enough XML files to satisfy "
            f"--samples-per-category={args.samples_per_category} for every category. "
            f"Undersized categories: {shortage_summary}. "
            f"All category counts: {full_count_summary}. "
            "Use --allow-fewer to copy all available files for those categories."
        )

    _validate_output_dir(output_dir, clear_output=args.clear_output)

    rng = random.Random(args.seed)

    for category in categories:
        candidates = list(grouped_paths[category])
        already_selected = [
            path for path in candidates if path.name in selected_paths
        ]
        remaining_candidates = [
            path for path in candidates if path.name not in selected_paths
        ]
        needed = max(0, args.samples_per_category - len(already_selected))
        top_up_paths = (
            sorted(rng.sample(remaining_candidates, needed))
            if len(remaining_candidates) > needed
            else remaining_candidates
        )
        for path in top_up_paths:
            selected_paths[path.name] = path

        print(
            f"{category}: selected {len(already_selected) + len(top_up_paths)} files "
            f"(seed={len(already_selected)}, added={len(top_up_paths)}, available={len(candidates)})"
        )

    copied = 0
    for src_path in sorted(selected_paths.values()):
        dst_path = output_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(
        f"Copied {copied} XML files into {output_dir} "
        f"(including {len(seed_paths)} seed files from {seed_route_file})"
    )


if __name__ == "__main__":
    main()
