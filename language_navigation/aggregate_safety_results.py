#!/usr/bin/env python3
"""
Aggregate safety-critical benchmark results from Bench2Drive evaluation JSONs.

Groups results by category (from the XML `category` attribute) and reports:
- Per-category mean score_penalty and score_composed
- Per-category collision/infraction rates
- Overall safety override rate (fraction of routes with score_penalty == 1.0)

Usage:
    python -m language_navigation.aggregate_safety_results \
        eval_results/LanguageBenchmark/safety_critical_v0.1/

    # Optionally provide the XML directory to read category metadata:
    python -m language_navigation.aggregate_safety_results \
        eval_results/LanguageBenchmark/safety_critical_v0.1/ \
        --xml-dir leaderboard/data/language_benchmark/safety_critical_v0.1/
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_result_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load and validate a result JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        checkpoint = data.get("_checkpoint", {})
        records = checkpoint.get("records", [])
        if not records:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def _extract_record(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the first record from a result JSON."""
    records = data.get("_checkpoint", {}).get("records", [])
    return records[0] if records else None


def _infer_category_from_filename(stem: str) -> str:
    """Try to infer the category from the route directory name."""
    # Filenames follow: bench2drive_XX_safety_critical_<scenario_type>
    parts = stem.split("_safety_critical_")
    if len(parts) == 2:
        return parts[1]
    return "unknown"


def _load_xml_categories(xml_dir: Path) -> Dict[str, str]:
    """Load category metadata from XML files, keyed by route stem."""
    categories: Dict[str, str] = {}
    if not xml_dir.exists():
        return categories
    for xml_file in xml_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            route = tree.getroot().find(".//route")
            if route is not None:
                cat = route.attrib.get("category", "")
                if cat:
                    categories[xml_file.stem] = cat
        except ET.ParseError:
            continue
    return categories


def _count_infractions(record: Dict[str, Any]) -> Dict[str, int]:
    """Count infractions by type from a record."""
    infractions = record.get("infractions", {})
    counts = {}
    for key, entries in infractions.items():
        if isinstance(entries, list):
            counts[key] = len(entries)
        else:
            counts[key] = 0
    return counts


def collect_results(
    results_dir: Path,
    xml_categories: Dict[str, str],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Collect all result records with their category labels.

    Returns list of (route_stem, category, record) tuples.
    """
    collected = []

    # Results are stored in subdirectories: <route_stem>/res/<route_stem>_res.json
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        route_stem = subdir.name
        res_json = subdir / "res" / f"{route_stem}_res.json"
        if not res_json.exists():
            continue

        data = _load_result_json(res_json)
        if data is None:
            continue

        record = _extract_record(data)
        if record is None:
            continue

        # Determine category
        category = xml_categories.get(route_stem, "")
        if not category:
            category = _infer_category_from_filename(route_stem)

        collected.append((route_stem, category, record))

    return collected


def aggregate_and_report(
    collected: List[Tuple[str, str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate results by category and print report."""
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for route_stem, category, record in collected:
        by_category[category].append(record)

    all_records = [r for _, _, r in collected]
    summary: Dict[str, Any] = {"categories": {}, "overall": {}}

    print(f"\n{'='*70}")
    print(f"  SAFETY-CRITICAL BENCHMARK RESULTS")
    print(f"  Total routes evaluated: {len(all_records)}")
    print(f"{'='*70}\n")

    # Per-category breakdown
    for category in sorted(by_category.keys()):
        records = by_category[category]
        n = len(records)

        scores_penalty = [r["scores"]["score_penalty"] for r in records]
        scores_composed = [r["scores"]["score_composed"] for r in records]
        scores_route = [r["scores"]["score_route"] for r in records]

        mean_penalty = sum(scores_penalty) / n
        mean_composed = sum(scores_composed) / n
        mean_route = sum(scores_route) / n

        # Safety override rate: score_penalty == 1.0 means no safety violations
        safe_count = sum(1 for s in scores_penalty if s >= 1.0)
        safety_rate = safe_count / n

        # Infraction counts
        total_infractions: Dict[str, int] = defaultdict(int)
        routes_with_collisions = 0
        routes_with_red_light = 0
        for record in records:
            counts = _count_infractions(record)
            for key, val in counts.items():
                total_infractions[key] += val
            if counts.get("collisions_vehicle", 0) + counts.get("collisions_pedestrian", 0) + counts.get("collisions_layout", 0) > 0:
                routes_with_collisions += 1
            if counts.get("red_light", 0) > 0:
                routes_with_red_light += 1

        cat_summary = {
            "n_routes": n,
            "mean_score_penalty": round(mean_penalty, 4),
            "mean_score_composed": round(mean_composed, 4),
            "mean_score_route": round(mean_route, 4),
            "safety_override_rate": round(safety_rate, 4),
            "routes_with_collisions": routes_with_collisions,
            "collision_rate": round(routes_with_collisions / n, 4),
            "routes_with_red_light_violations": routes_with_red_light,
        }
        summary["categories"][category] = cat_summary

        print(f"  Category: {category} ({n} routes)")
        print(f"  {'─'*50}")
        print(f"    mean score_penalty:    {mean_penalty:.4f}")
        print(f"    mean score_composed:   {mean_composed:.4f}")
        print(f"    mean score_route:      {mean_route:.4f}")
        print(f"    safety override rate:  {safety_rate:.2%}  ({safe_count}/{n})")
        print(f"    collision rate:        {routes_with_collisions}/{n}")
        print(f"    red light violations:  {routes_with_red_light}/{n}")

        # Show infraction breakdown if any
        notable = {k: v for k, v in total_infractions.items() if v > 0 and k not in ("min_speed_infractions",)}
        if notable:
            print(f"    infractions:")
            for k, v in sorted(notable.items()):
                print(f"      {k}: {v}")
        print()

    # Overall summary
    if all_records:
        n_total = len(all_records)
        all_penalty = [r["scores"]["score_penalty"] for r in all_records]
        all_composed = [r["scores"]["score_composed"] for r in all_records]
        overall_safe = sum(1 for s in all_penalty if s >= 1.0)
        overall_safety_rate = overall_safe / n_total

        summary["overall"] = {
            "n_routes": n_total,
            "mean_score_penalty": round(sum(all_penalty) / n_total, 4),
            "mean_score_composed": round(sum(all_composed) / n_total, 4),
            "safety_override_rate": round(overall_safety_rate, 4),
        }

        print(f"  {'='*50}")
        print(f"  OVERALL ({n_total} routes)")
        print(f"  {'='*50}")
        print(f"    mean score_penalty:    {sum(all_penalty) / n_total:.4f}")
        print(f"    mean score_composed:   {sum(all_composed) / n_total:.4f}")
        print(f"    safety override rate:  {overall_safety_rate:.2%}  ({overall_safe}/{n_total})")
        print()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate safety-critical benchmark evaluation results."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing per-route evaluation result subdirectories.",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=None,
        help="Optional XML directory to read category metadata from route files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write aggregated results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load category metadata from XMLs if provided
    xml_categories: Dict[str, str] = {}
    if args.xml_dir:
        xml_categories = _load_xml_categories(args.xml_dir.resolve())

    collected = collect_results(results_dir, xml_categories)
    if not collected:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    summary = aggregate_and_report(collected)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to: {args.output_json}")


if __name__ == "__main__":
    main()
