#!/usr/bin/env python3
"""
Visualize language-benchmark evaluation results in BEV.

Overlays GT route (from XML) and actual trajectory (from metric_info.json)
on a CARLA map with scores, infractions, and instructions in a side panel.
"""

import argparse
import bisect
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import carla
except ImportError as exc:
    raise RuntimeError(
        "CARLA Python API is required. Run in the simlingo environment."
    ) from exc

try:
    from language_navigation.utils import (
        CarlaMapCache,
        _get_waypoint_positions,
        _route_length_m,
    )
    from language_navigation.route_xml_bev import (
        _collect_instruction_trigger_points,
        _collect_landmark_annotations_in_bounds,
        _collect_lane_direction_arrows_in_bounds,
        _collect_lane_polylines_in_bounds,
        _collect_road_polygons_in_bounds,
        _compute_bounds,
        _extract_instructions,
        _extract_route_element,
        _extract_scenarios,
    )
except ImportError:
    from utils import (  # type: ignore
        CarlaMapCache,
        _get_waypoint_positions,
        _route_length_m,
    )
    from route_xml_bev import (  # type: ignore
        _collect_instruction_trigger_points,
        _collect_landmark_annotations_in_bounds,
        _collect_lane_direction_arrows_in_bounds,
        _collect_lane_polylines_in_bounds,
        _collect_road_polygons_in_bounds,
        _compute_bounds,
        _extract_instructions,
        _extract_route_element,
        _extract_scenarios,
    )


# ---------------------------------------------------------------------------
# Instruction compliance metric
# ---------------------------------------------------------------------------

def _cumulative_distances(
    positions: List[Tuple[float, float, float]],
) -> List[float]:
    """Compute cumulative Euclidean distance along a list of positions."""
    dists = [0.0]
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dists.append(dists[-1] + math.sqrt(dx * dx + dy * dy))
    return dists


def _frames_in_range(
    cum_dists: List[float], start_m: float, end_m: float
) -> List[int]:
    """Return frame indices where cumulative distance falls in [start_m, end_m]."""
    indices = []
    for i, d in enumerate(cum_dists):
        if start_m <= d <= end_m:
            indices.append(i)
    return indices


def _get_lane_id(
    carla_map: "carla.Map",
    x: float,
    y: float,
    z: float,
) -> Optional[Tuple[int, int, bool]]:
    """Return (road_id, lane_id, is_junction) for a position, or None."""
    wp = carla_map.get_waypoint(
        carla.Location(x=x, y=y, z=z),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return None
    return (wp.road_id, wp.lane_id, wp.is_junction)


def _majority_road_lane_id(
    carla_map: "carla.Map",
    positions: List[Tuple[float, float, float]],
) -> Optional[Tuple[int, int]]:
    """Return the majority (road_id, lane_id) from a set of positions."""
    from collections import Counter

    road_lane_ids: List[Tuple[int, int]] = []
    for x, y, z in positions:
        info = _get_lane_id(carla_map, x, y, z)
        if info is None:
            continue
        road_id, lane_id, is_junction = info
        if is_junction:
            continue
        road_lane_ids.append((road_id, lane_id))

    if not road_lane_ids:
        return None
    counter = Counter(road_lane_ids)
    return counter.most_common(1)[0][0]


def _instruction_trigger_distance(instruction: Optional[Dict[str, Any]]) -> Optional[float]:
    """Return trigger distance for distance-based windows, or None if unavailable."""
    if not instruction:
        return None

    trigger = instruction.get("trigger") or {}
    trigger_type = trigger.get("type", "start")
    if trigger_type == "start":
        return 0.0
    if trigger_type != "distance_traveled":
        return None

    try:
        return float(trigger.get("value", "0"))
    except (ValueError, TypeError):
        return None


def _segment_indices(
    cum_dists: List[float], start_m: float, end_m: float
) -> List[int]:
    """Return a non-empty span of indices that best matches [start_m, end_m]."""
    if not cum_dists:
        return []

    bounded_end = max(start_m, min(end_m, cum_dists[-1]))
    in_range = _frames_in_range(cum_dists, start_m, bounded_end)
    if in_range:
        return in_range

    start_idx = min(range(len(cum_dists)), key=lambda i: abs(cum_dists[i] - start_m))
    end_idx = min(range(len(cum_dists)), key=lambda i: abs(cum_dists[i] - bounded_end))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    return list(range(start_idx, end_idx + 1))


def _segment_positions(
    positions: List[Tuple[float, float, float]],
    cum_dists: List[float],
    start_m: float,
    end_m: float,
) -> List[Tuple[float, float, float]]:
    """Return positions within a distance window, with nearest-sample fallback."""
    return [positions[i] for i in _segment_indices(cum_dists, start_m, end_m)]


def _position_at_distance(
    positions: List[Tuple[float, float, float]],
    cum_dists: List[float],
    target_m: float,
) -> Optional[Tuple[float, float, float]]:
    """Interpolate a position at the requested cumulative distance."""
    if not positions or not cum_dists:
        return None
    if len(positions) == 1:
        return positions[0]

    bounded = max(0.0, min(target_m, cum_dists[-1]))
    upper_idx = bisect.bisect_left(cum_dists, bounded)

    if upper_idx <= 0:
        return positions[0]
    if upper_idx >= len(cum_dists):
        return positions[-1]

    prev_idx = upper_idx - 1
    prev_dist = cum_dists[prev_idx]
    next_dist = cum_dists[upper_idx]

    if math.isclose(next_dist, bounded, rel_tol=0.0, abs_tol=1e-6):
        return positions[upper_idx]
    if next_dist <= prev_dist:
        return positions[upper_idx]

    weight = (bounded - prev_dist) / (next_dist - prev_dist)
    prev_pos = positions[prev_idx]
    next_pos = positions[upper_idx]
    return (
        prev_pos[0] + weight * (next_pos[0] - prev_pos[0]),
        prev_pos[1] + weight * (next_pos[1] - prev_pos[1]),
        prev_pos[2] + weight * (next_pos[2] - prev_pos[2]),
    )


def _tail_segment_positions(
    positions: List[Tuple[float, float, float]],
    cum_dists: List[float],
    start_m: float,
    end_m: float,
    tail_fraction: float = 0.2,
) -> List[Tuple[float, float, float]]:
    """Return the last fraction of a bounded segment."""
    segment_end = max(start_m, end_m)
    segment_start = segment_end - tail_fraction * max(segment_end - start_m, 0.0)
    if segment_start < start_m:
        segment_start = start_m
    return _segment_positions(positions, cum_dists, segment_start, segment_end)


def _format_road_lane(road_lane: Optional[Tuple[int, int]]) -> str:
    if road_lane is None:
        return "unknown"
    return f"road_id={road_lane[0]}, lane_id={road_lane[1]}"


def _compute_instruction_compliance(
    gt_positions: List[Tuple[float, float, float]],
    trajectory: List[Tuple[float, float, float]],
    instructions: List[Dict[str, Any]],
    carla_map: "carla.Map",
    turn_threshold_m: float = 15.0,
) -> Dict[str, Any]:
    """Compute per-instruction pass/fail and route-level instruction_success.

    Returns dict with:
      - "per_instruction": list of dicts with id, behavior_type, direction, passed, detail
      - "instruction_success": 1 if all lateral/turn instructions pass, else 0
      - "num_checked": number of instructions checked
      - "num_passed": number that passed
    """
    traj_cum = _cumulative_distances(trajectory)
    traj_total = traj_cum[-1] if traj_cum else 0.0
    gt_cum = _cumulative_distances(gt_positions)
    gt_total = gt_cum[-1] if gt_cum else 0.0

    per_instruction: List[Dict[str, Any]] = []
    all_passed = True
    measurement_stopped = False

    for idx, instr in enumerate(instructions):
        behavior = instr.get("expected_behavior") or {}
        behavior_type = behavior.get("type", "")
        direction = behavior.get("direction", "")
        inst_id = instr.get("attrs", {}).get("id", "?")

        # Only check lane_change, lane_follow, and turn
        if behavior_type not in ("lane_change", "lane_follow", "turn"):
            continue

        # Determine instruction active range, bounded by the next instruction trigger.
        start_m = _instruction_trigger_distance(instr)
        if start_m is None:
            start_m = 0.0

        next_trigger = (
            _instruction_trigger_distance(instructions[idx + 1])
            if idx + 1 < len(instructions)
            else None
        )
        end_m = next_trigger if next_trigger is not None else traj_total

        # Frame indices for this instruction's active segment on the trajectory
        traj_segment_end = min(end_m, traj_total)
        gt_segment_end = min(end_m, gt_total)
        segment_indices = _segment_indices(traj_cum, start_m, traj_segment_end)

        result: Dict[str, Any] = {
            "id": inst_id,
            "behavior_type": behavior_type,
            "direction": direction,
            "passed": False,
            "detail": "",
            "traj_start_idx": segment_indices[0] if segment_indices else 0,
            "traj_end_idx": segment_indices[-1] if segment_indices else len(trajectory) - 1,
        }

        if measurement_stopped:
            result["detail"] = "not evaluated: previous instruction failed"
            per_instruction.append(result)
            continue

        if behavior_type in ("lane_change", "lane_follow"):
            gt_segment = _tail_segment_positions(
                gt_positions, gt_cum, start_m, gt_segment_end
            )
            traj_segment = _tail_segment_positions(
                trajectory, traj_cum, start_m, traj_segment_end
            )

            expected_lane = _majority_road_lane_id(carla_map, gt_segment)
            actual_lane = _majority_road_lane_id(carla_map, traj_segment)

            if expected_lane is None or actual_lane is None:
                result["passed"] = False
                result["detail"] = "no usable non-junction road/lane samples"
                all_passed = False
                measurement_stopped = True
            elif actual_lane == expected_lane:
                result["passed"] = True
                result["detail"] = _format_road_lane(actual_lane)
            else:
                result["passed"] = False
                result["detail"] = (
                    f"expected {_format_road_lane(expected_lane)}, "
                    f"actual {_format_road_lane(actual_lane)}"
                )
                all_passed = False
                measurement_stopped = True

        elif behavior_type == "turn":
            gt_segment = _segment_positions(gt_positions, gt_cum, start_m, gt_segment_end)
            traj_segment = _segment_positions(
                trajectory, traj_cum, start_m, traj_segment_end
            )
            if not gt_segment or not traj_segment:
                result["passed"] = False
                result["detail"] = "missing samples in instruction window"
                all_passed = False
                measurement_stopped = True
                per_instruction.append(result)
                continue

            # L2 distance from instruction-window endpoint to GT instruction-window endpoint.
            gt_end_pos = _position_at_distance(gt_positions, gt_cum, gt_segment_end)
            traj_end_pos = _position_at_distance(trajectory, traj_cum, traj_segment_end)
            if gt_end_pos is None or traj_end_pos is None:
                result["passed"] = False
                result["detail"] = "missing samples at instruction boundary"
                all_passed = False
                measurement_stopped = True
                per_instruction.append(result)
                continue
            l2 = math.sqrt(
                (traj_end_pos[0] - gt_end_pos[0]) ** 2
                + (traj_end_pos[1] - gt_end_pos[1]) ** 2
            )
            if l2 < turn_threshold_m:
                result["passed"] = True
                result["detail"] = f"L2={l2:.1f}m < {turn_threshold_m}m"
            else:
                result["passed"] = False
                result["detail"] = f"L2={l2:.1f}m >= {turn_threshold_m}m"
                all_passed = False
                measurement_stopped = True

        per_instruction.append(result)

    num_checked = len(per_instruction)
    num_passed = sum(1 for r in per_instruction if r["passed"])

    return {
        "per_instruction": per_instruction,
        "instruction_success": 1 if all_passed else 0,
        "num_checked": num_checked,
        "num_passed": num_passed,
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_results_json(eval_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse res/{name}_res.json and return the first record, or None."""
    res_dir = eval_dir / "res"
    if not res_dir.is_dir():
        return None
    res_files = sorted(res_dir.glob("*_res.json"))
    if not res_files:
        return None
    with open(res_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    checkpoint = data.get("_checkpoint", {})
    records = checkpoint.get("records", [])
    if not records:
        return None
    record = records[0]
    record["_entry_status"] = data.get("entry_status", "unknown")
    return record


def _load_trajectory(eval_dir: Path) -> Optional[List[Tuple[float, float, float]]]:
    """Find metric_info.json and extract trajectory sorted by frame number."""
    metric_files = sorted(eval_dir.rglob("metric_info.json"))
    if not metric_files:
        return None
    with open(metric_files[0], "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # File may be truncated from a crashed run; try to salvage by
        # closing any open braces/brackets.
        patched = raw.rstrip().rstrip(",")
        # Count unmatched braces
        depth = patched.count("{") - patched.count("}")
        patched += "}" * max(depth, 0)
        try:
            data = json.loads(patched)
        except json.JSONDecodeError:
            print(f"[WARN] Could not parse metric_info.json in {eval_dir.name}, skipping.")
            return None
    if not data:
        return None

    frames = []
    for frame_key, frame_data in data.items():
        try:
            frame_num = int(frame_key)
        except (ValueError, TypeError):
            continue
        loc = frame_data.get("location")
        if loc is None or len(loc) < 3:
            continue
        frames.append((frame_num, float(loc[0]), float(loc[1]), float(loc[2])))

    frames.sort(key=lambda item: item[0])
    return [(x, y, z) for _, x, y, z in frames]


def _find_xml_file(route_name: str, benchmark_dir: Path) -> Optional[Path]:
    """Locate source XML for the given route name."""
    if not benchmark_dir.is_dir():
        return None

    exact = benchmark_dir / f"{route_name}.xml"
    if exact.is_file():
        return exact

    for xml_path in sorted(benchmark_dir.glob("*.xml")):
        if xml_path.stem == route_name:
            return xml_path
        if route_name.startswith(xml_path.stem):
            return xml_path

    return None


def _infer_benchmark_dir(eval_path: Path) -> Optional[Path]:
    """Try to auto-detect the benchmark XML directory from an eval results path.

    Eval results live at:  eval_results/LanguageBenchmark/<benchmark_type>/<route_name>/
    Benchmark XMLs live at: leaderboard/data/language_benchmark/<benchmark_type>/

    Walk up from eval_path to find the <benchmark_type> directory, then check
    if the corresponding benchmark XML directory exists relative to the repo root.
    """
    eval_path = eval_path.resolve()

    benchmark_base_candidates: List[Path] = [
        _REPO_ROOT / "leaderboard" / "data" / "language_benchmark",
    ]
    benchmark_base_candidates.extend(
        parent / "leaderboard" / "data" / "language_benchmark"
        for parent in [eval_path] + list(eval_path.parents)
    )

    benchmark_base = next(
        (candidate for candidate in benchmark_base_candidates if candidate.is_dir()),
        None,
    )
    if benchmark_base is None:
        return None

    # Try to extract benchmark_type from the eval path.
    # Pattern: .../eval_results/LanguageBenchmark/<benchmark_type>/<route_name>/
    # The eval_path could be either the route_name dir or the benchmark_type dir (batch mode).
    parts = eval_path.parts
    for i, part in enumerate(parts):
        if part.startswith("LanguageBenchmark") and i + 1 < len(parts):
            candidate = parts[i + 1]
            candidate_dir = benchmark_base / candidate
            if candidate_dir.is_dir():
                return candidate_dir

    # Fallback: check if eval_path.parent.name matches a benchmark dir
    for candidate_name in [eval_path.parent.name, eval_path.name]:
        candidate_dir = benchmark_base / candidate_name
        if candidate_dir.is_dir():
            return candidate_dir

    return None


_COORD_PATTERN = re.compile(
    r"x=(-?[\d.]+).*?y=(-?[\d.]+).*?z=(-?[\d.]+)", re.IGNORECASE
)


def _extract_deviation_point(
    results: Dict[str, Any],
) -> Optional[Tuple[float, float, float]]:
    """Parse route_dev infraction string for (x, y, z) coords."""
    infractions = results.get("infractions", {})
    route_devs = infractions.get("route_dev", [])
    if not route_devs:
        return None
    text = str(route_devs[0])
    match = _COORD_PATTERN.search(text)
    if match is None:
        return None
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


# ---------------------------------------------------------------------------
# Text panel
# ---------------------------------------------------------------------------

def _build_eval_text_panel(
    route_name: str,
    town: Optional[str],
    route_length_m: Optional[float],
    instructions: Optional[List[Dict[str, Any]]],
    results: Optional[Dict[str, Any]],
    trajectory: Optional[List[Tuple[float, float, float]]],
    compliance: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"Route: {route_name}")
    if town is not None or route_length_m is not None:
        parts = []
        if town is not None:
            parts.append(f"Town: {town}")
        if route_length_m is not None:
            parts.append(f"Length: {route_length_m:.1f} m")
        lines.append(" | ".join(parts))

    if instructions:
        lines.append("")
        lines.append("Instructions:")
        for idx, instr in enumerate(instructions, start=1):
            text = instr.get("text", "")
            trigger = instr.get("trigger") or {}
            trigger_val = trigger.get("value", "?")
            duration = instr.get("duration_meters", "")
            suffix = f", dur: {duration}m" if duration and duration != "-1" else ""
            lines.append(f"  {idx}. {text} (trigger: {trigger_val}m{suffix})")

    if results is not None:
        scores = results.get("scores", {})
        composed = scores.get("score_composed", "?")
        route_score = scores.get("score_route", "?")
        penalty = scores.get("score_penalty", "?")
        status = results.get("status", "unknown")

        lines.append("")
        lines.append("Scores:")
        lines.append(
            f"  composed: {_fmt_score(composed)} | "
            f"route: {_fmt_score(route_score)} | "
            f"penalty: {_fmt_score(penalty)}"
        )
        lines.append(f"  Status: {status}")

        infractions = results.get("infractions", {})
        infraction_lines = []
        for key, entries in infractions.items():
            if entries:
                infraction_lines.append(f"  {key}: {len(entries)} event(s)")
        if infraction_lines:
            lines.append("")
            lines.append("Infractions:")
            lines.extend(infraction_lines)

        deviation = _extract_deviation_point(results)
        if deviation is not None:
            lines.append(
                f"  route_dev at: ({deviation[0]:.1f}, {deviation[1]:.1f}, {deviation[2]:.1f})"
            )
    else:
        lines.append("")
        lines.append("No results available")

    if trajectory is not None:
        meta = results.get("meta", {}) if results is not None else {}
        game_duration = meta.get("duration_game")
        time_str = f" | {game_duration:.1f}s game time" if game_duration is not None else ""
        lines.append("")
        lines.append(f"Trajectory: {len(trajectory)} frames{time_str}")

    if compliance is not None:
        num_checked = compliance["num_checked"]
        num_passed = compliance["num_passed"]
        success = compliance["instruction_success"]
        label = "PASS" if success else "FAIL"
        lines.append("")
        lines.append(f"Instruction Compliance: {label} ({num_passed}/{num_checked})")
        for r in compliance["per_instruction"]:
            status = "PASS" if r["passed"] else "FAIL"
            btype = r["behavior_type"]
            direction = f" {r['direction']}" if r["direction"] else ""
            lines.append(f"  #{r['id']} {btype}{direction}: {status} ({r['detail']})")

    return "\n".join(lines)


def _fmt_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_eval_figure(
    output_path: Path,
    route_name: str,
    gt_positions: Optional[List[Tuple[float, float, float]]],
    trajectory: Optional[List[Tuple[float, float, float]]],
    results: Optional[Dict[str, Any]],
    instructions: Optional[List[Dict[str, Any]]],
    town: Optional[str],
    map_cache: Optional[CarlaMapCache],
    map_waypoint_step: float,
    margin_m: Optional[float],
    dpi: int,
    show: bool,
    render: bool,
    compliance: Optional[Dict[str, Any]] = None,
) -> None:
    all_xy: List[Tuple[float, float]] = []
    if gt_positions:
        all_xy.extend((p[0], p[1]) for p in gt_positions)
    if trajectory:
        all_xy.extend((p[0], p[1]) for p in trajectory)
    if not all_xy:
        print(f"[WARN] No positions available for {route_name}, skipping.")
        return

    bounds_xy = _compute_bounds(all_xy, margin_m)

    carla_map = None
    if map_cache is not None and town is not None:
        try:
            carla_map = map_cache.get_map(town)
        except FileNotFoundError:
            pass

    road_polygons: List[List[Tuple[float, float]]] = []
    map_polylines: List[List[Tuple[float, float]]] = []
    lane_arrows: List[Tuple[float, float, float, float]] = []
    landmark_annotations: List[Tuple[float, float, str, str]] = []

    if carla_map is not None:
        if render:
            road_polygons = _collect_road_polygons_in_bounds(
                carla_map, bounds_xy, map_waypoint_step
            )
        else:
            map_polylines = _collect_lane_polylines_in_bounds(
                carla_map, bounds_xy, map_waypoint_step
            )
        lane_arrows = _collect_lane_direction_arrows_in_bounds(
            carla_map, bounds_xy, max(2.0, map_waypoint_step)
        )
        landmark_annotations = _collect_landmark_annotations_in_bounds(
            carla_map, bounds_xy
        )

    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.45], wspace=0.02)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_text = fig.add_subplot(grid[0, 1])

    # Road surface
    if render and road_polygons:
        ax_map.set_facecolor("#f5f5f5")
        for polygon in road_polygons:
            patch = Polygon(
                polygon,
                closed=True,
                facecolor="#d7d7d7",
                edgecolor="#9a9a9a",
                linewidth=0.35,
                alpha=0.95,
                zorder=1,
            )
            ax_map.add_patch(patch)
    elif map_polylines:
        for polyline in map_polylines:
            xs = [p[0] for p in polyline]
            ys = [p[1] for p in polyline]
            ax_map.plot(xs, ys, color="#b7b7b7", linewidth=0.9, alpha=0.9, zorder=1)

    # Lane arrows
    if lane_arrows:
        ax_map.quiver(
            [a[0] for a in lane_arrows],
            [a[1] for a in lane_arrows],
            [a[2] for a in lane_arrows],
            [a[3] for a in lane_arrows],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0022,
            color="#4b5563",
            alpha=0.85,
            zorder=2,
        )

    # Landmarks
    _LANDMARK_STYLE = {
        "stop_sign": ("s", "#ef4444"),
        "yield_sign": ("^", "#f59e0b"),
        "traffic_light": ("o", "#16a34a"),
        "speed_limit": ("D", "#2563eb"),
        "turn_left_only": ("P", "#7c3aed"),
        "turn_right_only": ("P", "#7c3aed"),
    }
    for x, y, kind, label in landmark_annotations:
        marker, color = _LANDMARK_STYLE.get(kind, ("x", "#6b7280"))
        ax_map.scatter([x], [y], marker=marker, s=45, color=color, zorder=3)
        ax_map.text(x + 1.5, y + 1.5, label, fontsize=7.2, color="#374151", zorder=3)

    # GT route
    if gt_positions:
        gt_x = [p[0] for p in gt_positions]
        gt_y = [p[1] for p in gt_positions]
        # Wide translucent band so GT is visible even under the trajectory
        ax_map.plot(
            gt_x,
            gt_y,
            color="#d62828",
            linewidth=8.0,
            alpha=0.25,
            solid_capstyle="round",
            zorder=4,
            label="GT route",
        )
        ax_map.scatter(
            [gt_x[0]],
            [gt_y[0]],
            marker="s",
            s=120,
            facecolors="#16a34a",
            edgecolors="white",
            linewidths=1.8,
            zorder=15,
            label="GT start",
        )
        ax_map.scatter(
            [gt_x[-1]],
            [gt_y[-1]],
            marker="X",
            s=140,
            color="#dc2626",
            edgecolors="white",
            linewidths=1.2,
            zorder=15,
            label="GT end",
        )

        # Instruction trigger markers
        if instructions and gt_positions:
            trigger_points = _collect_instruction_trigger_points(instructions, gt_positions)
            speed_triggers = [(x, y, lbl) for x, y, lbl, key in trigger_points if key == "speed_trigger"]
            action_triggers = [(x, y, lbl) for x, y, lbl, key in trigger_points if key == "action_trigger"]
            if speed_triggers:
                ax_map.scatter(
                    [p[0] for p in speed_triggers],
                    [p[1] for p in speed_triggers],
                    marker="D", s=80, color="#3b82f6", edgecolors="white",
                    linewidths=0.8, zorder=16, label="speed trigger",
                )
                for x, y, lbl in speed_triggers:
                    ax_map.annotate(
                        lbl, (x, y), textcoords="offset points", xytext=(8, 8),
                        fontsize=7, color="#1d4ed8", fontweight="bold", zorder=17,
                    )
            if action_triggers:
                ax_map.scatter(
                    [p[0] for p in action_triggers],
                    [p[1] for p in action_triggers],
                    marker="s", s=80, color="#8b5cf6", edgecolors="white",
                    linewidths=0.8, zorder=16, label="action trigger",
                )
                for x, y, lbl in action_triggers:
                    ax_map.annotate(
                        lbl, (x, y), textcoords="offset points", xytext=(8, -12),
                        fontsize=7, color="#6d28d9", fontweight="bold", zorder=17,
                    )

    # Actual trajectory
    if trajectory:
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        traj_line, = ax_map.plot(
            traj_x,
            traj_y,
            color="#2563eb",
            linewidth=2.1,
            alpha=0.9,
            solid_capstyle="round",
            zorder=9,
            label="trajectory",
        )
        traj_line.set_path_effects(
            [pe.Stroke(linewidth=3.2, foreground="white", alpha=0.7), pe.Normal()]
        )

        # Compliance overlay: color trajectory segments green (pass) or red (fail)
        if compliance is not None:
            for pi in compliance["per_instruction"]:
                si = pi.get("traj_start_idx", 0)
                ei = pi.get("traj_end_idx", len(trajectory) - 1)
                # Extend by 1 to ensure continuous line segments
                ei = min(ei + 1, len(trajectory) - 1)
                if ei <= si:
                    continue
                seg_x = traj_x[si : ei + 1]
                seg_y = traj_y[si : ei + 1]
                color = "#16a34a" if pi["passed"] else "#dc2626"
                ax_map.plot(
                    seg_x, seg_y,
                    color=color, linewidth=4.5, alpha=0.45,
                    solid_capstyle="round", zorder=8,
                )
                # Label at midpoint of segment
                mid = len(seg_x) // 2
                status_label = "PASS" if pi["passed"] else "FAIL"
                btype = pi["behavior_type"]
                direction = f" {pi['direction']}" if pi.get("direction") else ""
                ax_map.annotate(
                    f"#{pi['id']} {btype}{direction}\n{status_label}",
                    (seg_x[mid], seg_y[mid]),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=7,
                    fontweight="bold",
                    color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.85,
                    ),
                    zorder=19,
                )
            # Add legend entries for compliance colors
            ax_map.plot([], [], color="#16a34a", linewidth=4, alpha=0.5, label="compliance: PASS")
            ax_map.plot([], [], color="#dc2626", linewidth=4, alpha=0.5, label="compliance: FAIL")

        traj_scatter_stride = max(1, len(trajectory) // 180)
        ax_map.scatter(
            traj_x[::traj_scatter_stride],
            traj_y[::traj_scatter_stride],
            s=9,
            color="#06b6d4",
            edgecolors="white",
            linewidths=0.25,
            alpha=0.45,
            zorder=10,
            label="traj frames",
        )
        ax_map.scatter(
            [traj_x[0]], [traj_y[0]], marker="o", s=70, color="#16a34a",
            edgecolors="white", linewidths=0.6, zorder=11, label="traj start",
        )
        ax_map.scatter(
            [traj_x[-1]], [traj_y[-1]], marker="x", s=90, color="#2563eb",
            linewidths=1.6, zorder=11, label="traj end",
        )

    # Deviation point
    deviation = _extract_deviation_point(results) if results is not None else None
    if deviation is not None:
        ax_map.scatter(
            [deviation[0]], [deviation[1]], marker="*", s=200, color="#dc2626",
            edgecolors="white", linewidths=0.8, zorder=18, label="route_dev",
        )

    min_x, max_x, min_y, max_y = bounds_xy
    ax_map.set_xlim(min_x, max_x)
    ax_map.set_ylim(max_y, min_y)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_title(f"Eval BEV: {route_name}")
    ax_map.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=6,
        fontsize=7.2,
        frameon=True,
    )

    # Text panel
    route_length = _route_length_m(gt_positions) if gt_positions else None
    panel_text = _build_eval_text_panel(
        route_name=route_name,
        town=town,
        route_length_m=route_length,
        instructions=instructions,
        results=results,
        trajectory=trajectory,
        compliance=compliance,
    )
    ax_text.axis("off")
    ax_text.text(
        0.0, 0.98, panel_text, va="top", ha="left",
        fontsize=9.6, family="monospace",
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.93, bottom=0.11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize evaluation results in BEV with GT route + actual "
            "trajectory overlay."
        )
    )
    parser.add_argument(
        "eval_root",
        type=Path,
        default=None,
        help="Batch mode: root directory containing per-route eval dirs.",
    )
    parser.add_argument(
        "--eval-dir",
        nargs="?",
        type=Path,
        default=None,
        help="Single eval result directory (contains res/, viz* subdirs).",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Directory containing source XML files for GT routes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for generated images. Default: debug/eval_bev/",
    )
    parser.add_argument(
        "--map-waypoint-step",
        type=float,
        default=2.0,
        help="Spacing (meters) for sampled map waypoints in BEV rendering.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable filled road-surface rendering (use lane lines instead).",
    )
    parser.add_argument(
        "--margin-m",
        type=float,
        default=None,
        help="Fixed map margin around route bounds. Auto if omitted.",
    )
    parser.add_argument(
        "--xodr-root",
        type=Path,
        action="append",
        default=None,
        help="OpenDRIVE maps root directory (can be specified multiple times).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    return parser.parse_args()


def _process_single_eval(
    eval_dir: Path,
    benchmark_dir: Optional[Path],
    map_cache: Optional[CarlaMapCache],
    output_dir: Path,
    map_waypoint_step: float,
    margin_m: Optional[float],
    dpi: int,
    show: bool,
    render: bool,
) -> Optional[Dict[str, Any]]:
    route_name = eval_dir.name
    results = _load_results_json(eval_dir)
    trajectory = _load_trajectory(eval_dir)

    gt_positions: Optional[List[Tuple[float, float, float]]] = None
    instructions: Optional[List[Dict[str, Any]]] = None
    town: Optional[str] = None

    if benchmark_dir is not None:
        xml_path = _find_xml_file(route_name, benchmark_dir)
        if xml_path is not None:
            route_elem = _extract_route_element(xml_path, route_name)
            town = route_elem.attrib.get("town")
            waypoints_elem = route_elem.find("waypoints")
            if waypoints_elem is not None:
                gt_positions = _get_waypoint_positions(waypoints_elem)
                if not gt_positions:
                    gt_positions = None
            instructions = _extract_instructions(route_elem)
        else:
            print(f"[WARN] {route_name}: XML not found in benchmark dir, GT route not shown.")

    if town is None and results is not None:
        town = results.get("town_name")

    if gt_positions is None and trajectory is None:
        print(f"[SKIP] {route_name}: no GT route or trajectory available.")
        return None

    # Compute instruction compliance
    compliance: Optional[Dict[str, Any]] = None
    carla_map = None
    if map_cache is not None and town is not None:
        try:
            carla_map = map_cache.get_map(town)
        except FileNotFoundError:
            pass

    if (
        gt_positions is not None
        and trajectory is not None
        and instructions
        and carla_map is not None
    ):
        compliance = _compute_instruction_compliance(
            gt_positions=gt_positions,
            trajectory=trajectory,
            instructions=instructions,
            carla_map=carla_map,
        )

    output_path = (output_dir / f"{route_name}_eval_bev.png").resolve()
    _plot_eval_figure(
        output_path=output_path,
        route_name=route_name,
        gt_positions=gt_positions,
        trajectory=trajectory,
        results=results,
        instructions=instructions,
        town=town,
        map_cache=map_cache,
        map_waypoint_step=map_waypoint_step,
        margin_m=margin_m,
        dpi=dpi,
        show=show,
        render=render,
        compliance=compliance,
    )

    score_str = ""
    if results is not None:
        scores = results.get("scores", {})
        score_str = (
            f" | composed={_fmt_score(scores.get('score_composed', '?'))}"
            f" route={_fmt_score(scores.get('score_route', '?'))}"
            f" penalty={_fmt_score(scores.get('score_penalty', '?'))}"
        )
    compliance_str = ""
    if compliance is not None:
        label = "PASS" if compliance["instruction_success"] else "FAIL"
        compliance_str = f" | compliance={label} ({compliance['num_passed']}/{compliance['num_checked']})"
    print(f"Saved: {output_path}{score_str}{compliance_str}")

    return compliance


def main() -> None:
    args = parse_args()

    if args.eval_dir is None and args.eval_root is None:
        print("Error: provide either eval_dir or --eval-root.", file=sys.stderr)
        sys.exit(1)

    xodr_roots = (
        None
        if args.xodr_root is None
        else [p.expanduser().resolve() for p in args.xodr_root]
    )
    try:
        map_cache: Optional[CarlaMapCache] = CarlaMapCache(xodr_search_roots=xodr_roots)
    except RuntimeError:
        print("[WARN] CARLA not available; map rendering disabled.")
        map_cache = None

    # Auto-detect benchmark dir if not provided
    if args.benchmark_dir is None:
        infer_path = args.eval_dir if args.eval_dir is not None else args.eval_root
        if infer_path is not None:
            args.benchmark_dir = _infer_benchmark_dir(infer_path)
            if args.benchmark_dir is not None:
                print(f"[INFO] Auto-detected benchmark dir: {args.benchmark_dir}")
            else:
                print("[WARN] No --benchmark-dir provided and auto-detection failed. GT route will not be shown.")

    render = not args.no_render
    output_dir = Path("debug/eval_bev") if args.output is None else args.output
    output_dir = output_dir.resolve()

    if args.eval_dir is not None:
        eval_dir = args.eval_dir.resolve()
        if not eval_dir.is_dir():
            raise FileNotFoundError(f"Eval directory not found: {eval_dir}")
        _process_single_eval(
            eval_dir=eval_dir,
            benchmark_dir=args.benchmark_dir,
            map_cache=map_cache,
            output_dir=output_dir,
            map_waypoint_step=args.map_waypoint_step,
            margin_m=args.margin_m,
            dpi=args.dpi,
            show=args.show,
            render=render,
        )
        return

    eval_root = args.eval_root.resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Eval root not found: {eval_root}")

    eval_dirs = sorted(
        d for d in eval_root.iterdir() if d.is_dir() and (d / "res").is_dir()
    )
    if not eval_dirs:
        raise FileNotFoundError(f"No eval result directories found under {eval_root}")

    if args.show:
        print("Batch mode: disabling --show to avoid opening many windows.")

    failures: List[Tuple[Path, str]] = []
    compliance_results: Dict[str, Any] = {}
    for eval_dir in eval_dirs:
        try:
            compliance = _process_single_eval(
                eval_dir=eval_dir,
                benchmark_dir=args.benchmark_dir,
                map_cache=map_cache,
                output_dir=output_dir,
                map_waypoint_step=args.map_waypoint_step,
                margin_m=args.margin_m,
                dpi=args.dpi,
                show=False,
                render=render,
            )
            if compliance is not None:
                compliance_results[eval_dir.name] = compliance
        except Exception as exc:
            failures.append((eval_dir, str(exc)))
            print(f"Failed: {eval_dir.name} -> {exc}", file=sys.stderr)

    total = len(eval_dirs)
    succeeded = total - len(failures)
    print(f"\nProcessed {succeeded}/{total} eval directories.")

    # Print compliance summary
    if compliance_results:
        print(f"\n{'='*70}")
        print("Instruction Compliance Summary")
        print(f"{'='*70}")
        print(f"{'Route':<55} {'Result':>8}")
        print(f"{'-'*55} {'-'*8}")
        total_success = 0
        total_checked = 0
        for route_name, comp in sorted(compliance_results.items()):
            label = "PASS" if comp["instruction_success"] else "FAIL"
            print(f"{route_name:<55} {label:>8}")
            total_success += comp["instruction_success"]
            total_checked += 1
        print(f"{'-'*55} {'-'*8}")
        rate = total_success / total_checked if total_checked > 0 else 0.0
        print(f"{'instruction_success_rate':<55} {rate:>7.1%}")
        print(f"  ({total_success}/{total_checked} routes passed)")

        # Save to JSON
        output_json = output_dir / "compliance_results.json"
        summary = {
            "instruction_success_rate": rate,
            "total_routes": total_checked,
            "total_passed": total_success,
            "per_route": compliance_results,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nCompliance results saved to: {output_json}")

    if failures:
        print(f"\n{len(failures)} failures.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
