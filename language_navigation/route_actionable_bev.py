#!/usr/bin/env python3
"""
Debug visualization for Bench2Drive route actionability.

Features:
1) Input by Bench2Drive route file id (e.g., 22, 022, bench2drive_22).
2) Plot a zoomed-in BEV map around the route area.
3) Overlay route waypoints parsed from the original XML.
4) List map-aware actionable navigation items in a side text panel.
"""

import argparse
import math
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

try:
    import carla
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "CARLA Python API is required. Run in the simlingo environment."
    ) from exc

# Support running both as:
#   python language_navigation/debug_route_actionable_bev.py ...
# and as a module import.
try:
    from language_navigation.generate_language_xml_distance import (
        CarlaMapCache,
        RouteSpecialCase,
        _can_change_lane,
        _get_waypoint_positions,
        _route_length_m,
        _scan_turn_actions,
        detect_route_special_case,
        _sample_navigation_instruction_for_action,
        select_navigation_trigger,
    )
except ImportError:
    from generate_language_xml_distance import (  # type: ignore
        CarlaMapCache,
        RouteSpecialCase,
        _can_change_lane,
        _get_waypoint_positions,
        _route_length_m,
        _scan_turn_actions,
        detect_route_special_case,
        _sample_navigation_instruction_for_action,
        select_navigation_trigger,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one Bench2Drive split route (or all routes in a folder) in BEV and display "
            "map-aware actionable navigation items."
        )
    )
    parser.add_argument(
        "route_id",
        nargs="?",
        default=None,
        help=(
            "Optional route id to select input XML. Examples: 22, 022, "
            "bench2drive_22, 1711 (original route id inside XML). "
            "If omitted, process all bench2drive_*.xml files in --input-dir."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("leaderboard/data/bench2drive_split"),
        help="Directory containing bench2drive_*.xml files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Default: "
            "./debug/<xml_stem>_actionable_debug.png"
        ),
    )
    parser.add_argument(
        "--trigger-step-m",
        type=float,
        default=50.0,
        help="Navigation trigger distance used for actionability check (meters).",
    )
    parser.add_argument(
        "--map-waypoint-step",
        type=float,
        default=2.0,
        help="Spacing (meters) for sampled map waypoints used in BEV map drawing.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help=(
            "Render a filled CARLA BEV road surface backdrop instead of only "
            "lane topology lines."
        ),
    )
    parser.add_argument(
        "--margin-m",
        type=float,
        default=None,
        help="Optional fixed map margin around route bounds. Auto if omitted.",
    )
    parser.add_argument(
        "--xodr-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional OpenDRIVE maps root directory. Can be provided multiple "
            "times. Falls back to default search roots if unset."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive matplotlib window.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    return parser.parse_args()


def _candidate_xml_paths(route_id: str, input_dir: Path) -> List[Path]:
    token = route_id.strip()
    stems: List[str] = []

    def add_stem(stem: str) -> None:
        if stem and stem not in stems:
            stems.append(stem)

    if token.endswith(".xml"):
        add_stem(Path(token).stem)
    add_stem(token)
    if token.startswith("bench2drive_"):
        add_stem(token[len("bench2drive_") :])
    if token.isdigit():
        number = int(token)
        add_stem(str(number))
        add_stem(f"{number:02d}")
        add_stem(f"{number:03d}")

    candidates: List[Path] = []
    for stem in stems:
        for prefix in ("", "bench2drive_"):
            name = stem if stem.endswith(".xml") else f"{prefix}{stem}.xml"
            path = input_dir / name
            if path not in candidates:
                candidates.append(path)
    return candidates


def resolve_route_xml(route_id: str, input_dir: Path) -> Path:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for candidate in _candidate_xml_paths(route_id, input_dir):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    # Fallback: treat route_id as original route attribute id inside XML.
    matches: List[Path] = []
    for xml_path in sorted(input_dir.glob("bench2drive_*.xml")):
        root = ET.parse(xml_path).getroot()
        for route_elem in root.findall("route"):
            if route_elem.attrib.get("id") == route_id:
                matches.append(xml_path.resolve())
                break

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = ", ".join(str(path.name) for path in matches)
        raise ValueError(
            f"Route id {route_id} matched multiple XML files in {input_dir}: {joined}"
        )

    raise FileNotFoundError(
        f"Could not resolve route id '{route_id}' under {input_dir}."
    )


def _extract_route_element(xml_path: Path, route_id: str) -> ET.Element:
    root = ET.parse(xml_path).getroot()
    routes = root.findall("route")
    if not routes:
        raise ValueError(f"No <route> element found in {xml_path}")

    if len(routes) == 1:
        return routes[0]

    # If multiple routes exist, prefer exact original route id match.
    for route in routes:
        if route.attrib.get("id") == route_id:
            return route

    return routes[0]


def _iter_input_xml_paths(input_dir: Path) -> List[Path]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted(path.resolve() for path in input_dir.glob("bench2drive_*.xml"))


def _compute_bounds(
    points_xy: Sequence[Tuple[float, float]], margin_m: Optional[float]
) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max_x - min_x
    span_y = max_y - min_y
    auto_margin = max(20.0, 0.15 * max(span_x, span_y))
    margin = auto_margin if margin_m is None else max(0.0, margin_m)
    return (
        min_x - margin,
        max_x + margin,
        min_y - margin,
        max_y + margin,
    )


def _collect_lane_polylines_in_bounds(
    carla_map: "carla.Map",
    bounds_xy: Tuple[float, float, float, float],
    waypoint_step_m: float,
) -> List[List[Tuple[float, float]]]:
    min_x, max_x, min_y, max_y = bounds_xy
    grouped: Dict[Tuple[int, int, int], List[Tuple[float, float, float]]] = defaultdict(list)

    for waypoint in carla_map.generate_waypoints(waypoint_step_m):
        loc = waypoint.transform.location
        if min_x <= loc.x <= max_x and min_y <= loc.y <= max_y:
            grouped[(waypoint.road_id, waypoint.section_id, waypoint.lane_id)].append(
                (waypoint.s, loc.x, loc.y)
            )

    polylines: List[List[Tuple[float, float]]] = []
    for lane_points in grouped.values():
        lane_points.sort(key=lambda item: item[0])
        polyline = [(x, y) for _, x, y in lane_points]
        if len(polyline) >= 2:
            polylines.append(polyline)
    return polylines


def _collect_lane_direction_arrows_in_bounds(
    carla_map: "carla.Map",
    bounds_xy: Tuple[float, float, float, float],
    waypoint_step_m: float,
) -> List[Tuple[float, float, float, float]]:
    min_x, max_x, min_y, max_y = bounds_xy
    # Store (s, x, y, yaw_deg) so we can use the waypoint's own heading for
    # re-orientation instead of projecting to a potentially wrong nearby lane.
    grouped: Dict[Tuple[int, int, int], List[Tuple[float, float, float, float]]] = defaultdict(list)

    for waypoint in carla_map.generate_waypoints(waypoint_step_m):
        location = waypoint.transform.location
        if min_x <= location.x <= max_x and min_y <= location.y <= max_y:
            grouped[(waypoint.road_id, waypoint.section_id, waypoint.lane_id)].append(
                (waypoint.s, location.x, location.y, waypoint.transform.rotation.yaw)
            )

    arrows: List[Tuple[float, float, float, float]] = []
    for lane_points in grouped.values():
        lane_points.sort(key=lambda item: item[0])
        if len(lane_points) < 2:
            continue
        mid = len(lane_points) // 2
        if mid >= len(lane_points) - 1:
            mid = len(lane_points) - 2
        _, x0, y0, yaw0 = lane_points[mid]
        _, x1, y1, _ = lane_points[mid + 1]
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < 1e-5 and abs(dy) < 1e-5:
            continue

        # `s` ordering can be opposite to driving direction on negative-lane-id
        # roads.  Re-orient using the waypoint's own heading (stored above) to
        # avoid snapping to an adjacent counter-direction lane via get_waypoint().
        yaw_rad = math.radians(yaw0)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        if dx * forward_x + dy * forward_y < 0.0:
            dx = -dx
            dy = -dy
        arrows.append((x0, y0, dx, dy))
    return arrows


def _landmark_kind(landmark: "carla.Landmark") -> Optional[str]:
    land_type = str(landmark.type)
    text = f"{landmark.name} {landmark.sub_type} {land_type}".lower()

    if "stopline" in text:
        return None
    if "stop" in text or land_type == "206":
        return "stop_sign"
    if "yield" in text or land_type == "205":
        return "yield_sign"
    if "signal" in text or land_type == "1000001":
        return "traffic_light"
    if "speed" in text or land_type == "274":
        return "speed_limit"
    if "left" in text and "right" not in text:
        return "turn_left_only"
    if "right" in text and "left" not in text:
        return "turn_right_only"
    if land_type.isdigit():
        return "traffic_sign"
    return None


def _collect_landmark_annotations_in_bounds(
    carla_map: "carla.Map",
    bounds_xy: Tuple[float, float, float, float],
) -> List[Tuple[float, float, str, str]]:
    min_x, max_x, min_y, max_y = bounds_xy
    annotations: List[Tuple[float, float, str, str]] = []
    seen_ids: Set[str] = set()

    for landmark in carla_map.get_all_landmarks():
        landmark_id = str(landmark.id)
        if landmark_id in seen_ids:
            continue
        seen_ids.add(landmark_id)

        location = landmark.transform.location
        if not (min_x <= location.x <= max_x and min_y <= location.y <= max_y):
            continue

        kind = _landmark_kind(landmark)
        if kind is None:
            continue

        if kind == "speed_limit":
            speed = str(landmark.sub_type).strip()
            label = f"speed {speed}" if speed and speed != "-1" else "speed"
        elif kind == "stop_sign":
            label = "stop"
        elif kind == "yield_sign":
            label = "yield"
        elif kind == "traffic_light":
            label = "traffic_light"
        elif kind == "turn_left_only":
            label = "left_only"
        elif kind == "turn_right_only":
            label = "right_only"
        else:
            label = "sign"

        annotations.append((location.x, location.y, kind, label))

    return annotations


def _collect_road_polygons_in_bounds(
    carla_map: "carla.Map",
    bounds_xy: Tuple[float, float, float, float],
    waypoint_step_m: float,
) -> List[List[Tuple[float, float]]]:
    min_x, max_x, min_y, max_y = bounds_xy
    polygons: List[List[Tuple[float, float]]] = []

    topology = [waypoint for waypoint, _ in carla_map.get_topology()]
    topology = sorted(topology, key=lambda w: (w.road_id, w.section_id, w.lane_id, w.s))
    seen: set = set()

    for start_waypoint in topology:
        key = (start_waypoint.road_id, start_waypoint.section_id, start_waypoint.lane_id)
        if key in seen:
            continue
        seen.add(key)

        sampled_waypoints: List["carla.Waypoint"] = [start_waypoint]
        waypoint = start_waypoint
        while True:
            next_waypoints = waypoint.next(waypoint_step_m)
            if not next_waypoints:
                break
            next_waypoint = next_waypoints[0]
            if (
                next_waypoint.road_id != start_waypoint.road_id
                or next_waypoint.section_id != start_waypoint.section_id
                or next_waypoint.lane_id != start_waypoint.lane_id
            ):
                break
            sampled_waypoints.append(next_waypoint)
            waypoint = next_waypoint

        left_border: List[Tuple[float, float]] = []
        right_border: List[Tuple[float, float]] = []
        for waypoint in sampled_waypoints:
            location = waypoint.transform.location
            yaw = math.radians(waypoint.transform.rotation.yaw)
            normal_x = -math.sin(yaw)
            normal_y = math.cos(yaw)
            half_width = 0.5 * waypoint.lane_width
            left_border.append(
                (location.x + normal_x * half_width, location.y + normal_y * half_width)
            )
            right_border.append(
                (location.x - normal_x * half_width, location.y - normal_y * half_width)
            )

        if len(left_border) < 2:
            continue

        polygon = left_border + list(reversed(right_border))
        if any(min_x <= x <= max_x and min_y <= y <= max_y for x, y in polygon):
            polygons.append(polygon)

    return polygons


def _extract_bench2drive_scenarios(route_elem: ET.Element) -> List[Dict[str, Any]]:
    scenarios_elem = route_elem.find("scenarios")
    if scenarios_elem is None:
        return []

    scenarios: List[Dict[str, Any]] = []
    for scenario_elem in scenarios_elem.findall("scenario"):
        trigger_elem = scenario_elem.find("trigger_point")
        trigger_point = None
        if trigger_elem is not None:
            trigger_point = (
                float(trigger_elem.attrib.get("x", "0")),
                float(trigger_elem.attrib.get("y", "0")),
                float(trigger_elem.attrib.get("z", "0")),
                float(trigger_elem.attrib.get("yaw", "0")),
            )

        attributes: List[str] = []
        for child in list(scenario_elem):
            if child.tag == "trigger_point":
                continue
            attr_text = ", ".join(f"{k}={v}" for k, v in sorted(child.attrib.items()))
            attributes.append(f"{child.tag}({attr_text})" if attr_text else child.tag)

        scenarios.append(
            {
                "name": scenario_elem.attrib.get("name", "unknown"),
                "type": scenario_elem.attrib.get("type", "unknown"),
                "trigger_point": trigger_point,
                "attributes": attributes,
            }
        )
    return scenarios


def _special_case_prompt_text(
    special_case: RouteSpecialCase,
    rng: random.Random,
) -> Tuple[str, Optional[str]]:
    primary = _sample_navigation_instruction_for_action(
        rng, action=special_case.primary_action, phrasing_mode=special_case.primary_phrasing_mode
    )["text"]
    secondary = None
    if special_case.secondary_action is not None:
        secondary = _sample_navigation_instruction_for_action(
            rng,
            action=special_case.secondary_action,
            phrasing_mode=special_case.secondary_phrasing_mode or "approach",
        )["text"]
    return str(primary), None if secondary is None else str(secondary)


def _build_text_panel_lines(
    route_id_input: str,
    xml_path: Path,
    route_elem: ET.Element,
    trigger_distance_m: float,
    route_length_m: float,
    navigation_position: Tuple[float, float, float],
    actionable_actions: List[str],
    ego_waypoint: Optional["carla.Waypoint"],
    selected_action: str,
    selected_prompt: str,
    trigger_score: int,
    trigger_source_kind: str,
    trigger_phrasing_mode: str,
    bench2drive_scenarios: List[Dict[str, Any]],
    sampled_actions: List[str],
    sampled_trigger_distance_m: float,
    sampled_trigger_position: Tuple[float, float, float],
    sampled_selected_action: str,
    sampled_selected_prompt: str,
    sampled_trigger_score: int,
    special_case: Optional[RouteSpecialCase],
    special_case_primary_prompt: Optional[str],
    special_case_secondary_prompt: Optional[str],
) -> str:
    lines: List[str] = []
    lines.append("Route Debug Summary")
    lines.append("")
    lines.append(f"input id: {route_id_input}")
    lines.append(f"xml: {xml_path.name}")
    lines.append(f"route id: {route_elem.attrib.get('id', 'unknown')}")
    lines.append(f"town: {route_elem.attrib.get('town', 'unknown')}")
    lines.append(f"route length: {route_length_m:.1f} m")
    lines.append(f"algo trigger distance: {trigger_distance_m:.1f} m")
    lines.append(
        f"algo trigger xyz: ({navigation_position[0]:.1f}, "
        f"{navigation_position[1]:.1f}, {navigation_position[2]:.1f})"
    )
    lines.append(f"selected action: {selected_action}")
    lines.append(f"selected prompt: {selected_prompt}")
    lines.append(f"actionability score: {trigger_score}")
    lines.append(f"trigger source: {trigger_source_kind}")
    lines.append(f"phrasing mode: {trigger_phrasing_mode}")
    if special_case is not None:
        lines.append(f"route special case: {special_case.kind}")
        lines.append(
            f"special action: {special_case.action_name} @ {special_case.primary_trigger_distance_m:.1f} m"
        )
        lines.append(f"special primary prompt: {special_case_primary_prompt}")
        if special_case.secondary_action is not None and special_case.secondary_trigger_distance_m is not None:
            lines.append(
                f"special secondary action: {special_case.secondary_action} @ "
                f"{special_case.secondary_trigger_distance_m:.1f} m"
            )
            lines.append(f"special secondary prompt: {special_case_secondary_prompt}")
    lines.append("")
    lines.append("Bench2Drive scenarios:")
    if bench2drive_scenarios:
        for index, scenario in enumerate(bench2drive_scenarios, start=1):
            trigger_point = scenario["trigger_point"]
            trigger_text = (
                f"trigger=({trigger_point[0]:.1f}, {trigger_point[1]:.1f}, "
                f"{trigger_point[2]:.1f}, yaw={trigger_point[3]:.1f})"
                if trigger_point is not None
                else "trigger=none"
            )
            attrs = "; ".join(scenario["attributes"]) if scenario["attributes"] else "attrs=none"
            lines.append(
                f"{index}. {scenario['name']} [{scenario['type']}] {trigger_text} {attrs}"
            )
    else:
        lines.append("none")
    lines.append("")
    lines.append("Displayed actionable items:")
    for index, action in enumerate(actionable_actions, start=1):
        lines.append(f"{index}. {action}")
    lines.append("")
    lines.append("Raw sampled actionability:")
    lines.append(f"sampled trigger distance: {sampled_trigger_distance_m:.1f} m")
    lines.append(
        f"sampled trigger xyz: ({sampled_trigger_position[0]:.1f}, "
        f"{sampled_trigger_position[1]:.1f}, {sampled_trigger_position[2]:.1f})"
    )
    lines.append(f"sampled selected action: {sampled_selected_action}")
    lines.append(f"sampled selected prompt: {sampled_selected_prompt}")
    lines.append(f"sampled actionability score: {sampled_trigger_score}")
    for index, action in enumerate(sampled_actions, start=1):
        lines.append(f"sampled {index}. {action}")

    if ego_waypoint is not None:
        lines.append("")
        lines.append("Ego waypoint @ trigger:")
        lines.append(
            f"road/section/lane: {ego_waypoint.road_id}/"
            f"{ego_waypoint.section_id}/{ego_waypoint.lane_id}"
        )
        lines.append(f"is_junction: {ego_waypoint.is_junction}")
        lines.append(f"lane_change flag: {ego_waypoint.lane_change}")
        lines.append(f"can_change_left: {_can_change_lane(ego_waypoint, 'left')}")
        lines.append(f"can_change_right: {_can_change_lane(ego_waypoint, 'right')}")
        turn_actions = sorted(_scan_turn_actions(ego_waypoint))
        lines.append(
            "detected_turn_options: "
            + (", ".join(turn_actions) if turn_actions else "none")
        )

    return "\n".join(lines)


def _plot_debug_figure(
    output_path: Path,
    route_id_input: str,
    xml_path: Path,
    route_elem: ET.Element,
    route_positions: List[Tuple[float, float, float]],
    map_polylines: List[List[Tuple[float, float]]],
    lane_arrows: List[Tuple[float, float, float, float]],
    landmark_annotations: List[Tuple[float, float, str, str]],
    road_polygons: List[List[Tuple[float, float]]],
    bounds_xy: Tuple[float, float, float, float],
    trigger_distance_m: float,
    navigation_position: Tuple[float, float, float],
    sampled_trigger_distance_m: float,
    sampled_navigation_position: Tuple[float, float, float],
    original_trigger_points: List[Tuple[float, float, float, float]],
    actionable_actions: List[str],
    ego_waypoint: Optional["carla.Waypoint"],
    ego_turn_actions: List[str],
    selected_action: str,
    selected_prompt: str,
    trigger_score: int,
    trigger_source_kind: str,
    trigger_phrasing_mode: str,
    bench2drive_scenarios: List[Dict[str, Any]],
    sampled_actions: List[str],
    sampled_selected_action: str,
    sampled_selected_prompt: str,
    sampled_trigger_score: int,
    special_case: Optional[RouteSpecialCase],
    special_case_primary_prompt: Optional[str],
    special_case_secondary_prompt: Optional[str],
    dpi: int,
    show: bool,
    render: bool,
) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.9, 1.25], wspace=0.02)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_text = fig.add_subplot(grid[0, 1])

    # Draw BEV backdrop.
    if render:
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
    else:
        for polyline in map_polylines:
            xs = [point[0] for point in polyline]
            ys = [point[1] for point in polyline]
            ax_map.plot(xs, ys, color="#b7b7b7", linewidth=0.9, alpha=0.9, zorder=1)

    # Draw lane direction arrows.
    if lane_arrows:
        arrow_x = [item[0] for item in lane_arrows]
        arrow_y = [item[1] for item in lane_arrows]
        arrow_u = [item[2] for item in lane_arrows]
        arrow_v = [item[3] for item in lane_arrows]
        ax_map.quiver(
            arrow_x,
            arrow_y,
            arrow_u,
            arrow_v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0022,
            color="#4b5563",
            alpha=0.85,
            zorder=2,
        )

    # Draw nearby traffic sign annotations.
    for x, y, kind, label in landmark_annotations:
        if kind == "stop_sign":
            marker, color = "s", "#ef4444"
        elif kind == "yield_sign":
            marker, color = "^", "#f59e0b"
        elif kind == "traffic_light":
            marker, color = "o", "#16a34a"
        elif kind == "speed_limit":
            marker, color = "D", "#2563eb"
        elif kind in ("turn_left_only", "turn_right_only"):
            marker, color = "P", "#7c3aed"
        else:
            marker, color = "x", "#6b7280"
        ax_map.scatter([x], [y], marker=marker, s=45, color=color, zorder=3)
        ax_map.text(x + 1.5, y + 1.5, label, fontsize=7.2, color="#374151", zorder=3)

    route_x = [position[0] for position in route_positions]
    route_y = [position[1] for position in route_positions]

    # Route polyline + explicit XML waypoint markers.
    ax_map.plot(route_x, route_y, color="#d62828", linewidth=2.4, zorder=4, label="route path")
    ax_map.scatter(
        route_x,
        route_y,
        s=12,
        color="#ff7f11",
        edgecolors="none",
        alpha=0.9,
        zorder=5,
        label="xml waypoints",
    )

    # Start / end markers.
    ax_map.scatter(
        [route_x[0]],
        [route_y[0]],
        marker="o",
        s=90,
        color="#16a34a",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label="start",
    )
    ax_map.scatter(
        [route_x[-1]],
        [route_y[-1]],
        marker="x",
        s=110,
        color="#dc2626",
        linewidths=1.6,
        zorder=6,
        label="end",
    )

    if original_trigger_points:
        trigger_x = [point[0] for point in original_trigger_points]
        trigger_y = [point[1] for point in original_trigger_points]
        ax_map.scatter(
            trigger_x,
            trigger_y,
            marker="^",
            s=70,
            color="#f59e0b",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="Bench2Drive trigger",
        )

    # Raw sampled trigger point.
    ax_map.scatter(
        [sampled_navigation_position[0]],
        [sampled_navigation_position[1]],
        marker="D",
        s=95,
        color="#264653",
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
        label=f"sampled trigger @ {sampled_trigger_distance_m:.1f}m",
    )

    if special_case is not None:
        ax_map.scatter(
            [navigation_position[0]],
            [navigation_position[1]],
            marker="P",
            s=110,
            color="#7c3aed",
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
            label=f"{special_case.kind} trigger @ {trigger_distance_m:.1f}m",
        )
        if special_case.secondary_position is not None and special_case.secondary_trigger_distance_m is not None:
            ax_map.scatter(
                [special_case.secondary_position[0]],
                [special_case.secondary_position[1]],
                marker="X",
                s=105,
                color="#0f766e",
                edgecolors="white",
                linewidths=0.8,
                zorder=8,
                label=f"{special_case.kind} secondary @ {special_case.secondary_trigger_distance_m:.1f}m",
            )

    lane_type_label = (
        str(ego_waypoint.lane_type).split(".")[-1] if ego_waypoint is not None else "unknown"
    )
    turn_label = ", ".join(ego_turn_actions) if ego_turn_actions else "none"
    ax_map.text(
        navigation_position[0] + 4.0,
        navigation_position[1] + 4.0,
        (
            f"lane_type={lane_type_label}\nturn_options={turn_label}\n"
            f"action={selected_action}\nscore={trigger_score}"
        ),
        fontsize=8.2,
        color="#111827",
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9ca3af", alpha=0.85),
    )

    min_x, max_x, min_y, max_y = bounds_xy
    ax_map.set_xlim(min_x, max_x)
    # CARLA's Y axis increases southward, so invert the display axis to put
    # north at the top (standard geographic convention).  All arrow dy values
    # are in CARLA coords, so this also makes lane direction arrows visually
    # correct (southbound arrows point down, northbound arrows point up).
    ax_map.set_ylim(max_y, min_y)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_title(
        f"BEV Route Debug: {xml_path.name} | route_id={route_elem.attrib.get('id', 'unknown')}"
    )
    ax_map.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        fontsize=8.6,
        frameon=True,
    )

    route_length_m = _route_length_m(route_positions)
    panel_text = _build_text_panel_lines(
        route_id_input=route_id_input,
        xml_path=xml_path,
        route_elem=route_elem,
        trigger_distance_m=trigger_distance_m,
        route_length_m=route_length_m,
        navigation_position=navigation_position,
        actionable_actions=actionable_actions,
        ego_waypoint=ego_waypoint,
        selected_action=selected_action,
        selected_prompt=selected_prompt,
        trigger_score=trigger_score,
        trigger_source_kind=trigger_source_kind,
        trigger_phrasing_mode=trigger_phrasing_mode,
        bench2drive_scenarios=bench2drive_scenarios,
        sampled_actions=sampled_actions,
        sampled_trigger_distance_m=sampled_trigger_distance_m,
        sampled_trigger_position=sampled_navigation_position,
        sampled_selected_action=sampled_selected_action,
        sampled_selected_prompt=sampled_selected_prompt,
        sampled_trigger_score=sampled_trigger_score,
        special_case=special_case,
        special_case_primary_prompt=special_case_primary_prompt,
        special_case_secondary_prompt=special_case_secondary_prompt,
    )
    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.98,
        panel_text,
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.93, bottom=0.11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _resolve_output_path(
    xml_path: Path,
    route_id: Optional[str],
    output_arg: Optional[Path],
) -> Path:
    if route_id is None:
        if output_arg is not None and output_arg.suffix:
            raise ValueError(
                "When route_id is omitted (batch mode), --output must be a directory path."
            )
        output_dir = Path("debug") if output_arg is None else output_arg
        return (output_dir / f"{xml_path.stem}_actionable_debug.png").resolve()

    if output_arg is None:
        return (Path("debug") / f"{xml_path.stem}_actionable_debug.png").resolve()
    return output_arg.resolve()


def _process_single_xml(
    *,
    xml_path: Path,
    route_id_input: str,
    map_cache: CarlaMapCache,
    trigger_step_m: float,
    map_waypoint_step: float,
    margin_m: Optional[float],
    dpi: int,
    show: bool,
    output_path: Path,
    render: bool,
) -> None:
    route_elem = _extract_route_element(xml_path, route_id_input)
    town = route_elem.attrib.get("town")
    if town is None:
        raise ValueError(f"Route in {xml_path} has no 'town' attribute.")

    waypoints_elem = route_elem.find("waypoints")
    if waypoints_elem is None:
        raise ValueError(f"Route in {xml_path} has no <waypoints> element.")
    route_positions = _get_waypoint_positions(waypoints_elem)
    if not route_positions:
        raise ValueError(f"Route in {xml_path} has empty <waypoints>.")

    route_length_m = _route_length_m(route_positions)
    carla_map = map_cache.get_map(town)
    bench2drive_scenarios = _extract_bench2drive_scenarios(route_elem)
    trigger_rng = random.Random(f"{xml_path.name}:{route_elem.attrib.get('id', route_id_input)}")
    selected_trigger, samples = select_navigation_trigger(
        route_positions=route_positions,
        trigger_step_m=trigger_step_m,
        carla_map=carla_map,
        rng=trigger_rng,
    )
    sampled_trigger_distance_m = selected_trigger.distance_m
    sampled_navigation_position = selected_trigger.position
    sampled_actions = list(samples[selected_trigger.sample_index].actions)
    special_case = detect_route_special_case(
        route_positions=route_positions,
        carla_map=carla_map,
        samples=samples,
    )
    if special_case is not None:
        special_rng = random.Random(
            f"special:{xml_path.name}:{route_elem.attrib.get('id', route_id_input)}"
        )
        special_case_primary_prompt, special_case_secondary_prompt = _special_case_prompt_text(
            special_case, special_rng
        )
        trigger_distance_m = special_case.primary_trigger_distance_m
        navigation_position = special_case.primary_position
        actionable_actions = [special_case.primary_action]
        if special_case.secondary_action is not None:
            actionable_actions.append(special_case.secondary_action)
        selected_action = special_case.primary_action
        selected_prompt = special_case_primary_prompt
        trigger_source_kind = f"special_case_{special_case.kind}"
        trigger_phrasing_mode = special_case.primary_phrasing_mode
        trigger_score = len(actionable_actions)
    else:
        special_case_primary_prompt = None
        special_case_secondary_prompt = None
        trigger_distance_m = sampled_trigger_distance_m
        navigation_position = sampled_navigation_position
        actionable_actions = sampled_actions
        selected_action = selected_trigger.selected_action
        selected_prompt = selected_trigger.sampled_text
        trigger_source_kind = selected_trigger.source_kind
        trigger_phrasing_mode = selected_trigger.phrasing_mode
        trigger_score = selected_trigger.score

    ego_waypoint = carla_map.get_waypoint(
        carla.Location(
            x=navigation_position[0],
            y=navigation_position[1],
            z=navigation_position[2],
        ),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    route_xy = [(position[0], position[1]) for position in route_positions]
    bounds_xy = _compute_bounds(route_xy, margin_m)
    map_polylines = _collect_lane_polylines_in_bounds(
        carla_map=carla_map,
        bounds_xy=bounds_xy,
        waypoint_step_m=map_waypoint_step,
    )
    lane_arrows = _collect_lane_direction_arrows_in_bounds(
        carla_map=carla_map,
        bounds_xy=bounds_xy,
        waypoint_step_m=max(2.0, map_waypoint_step),
    )
    landmark_annotations = _collect_landmark_annotations_in_bounds(
        carla_map=carla_map,
        bounds_xy=bounds_xy,
    )
    ego_turn_actions = sorted(_scan_turn_actions(ego_waypoint)) if ego_waypoint else []
    road_polygons = (
        _collect_road_polygons_in_bounds(
            carla_map=carla_map,
            bounds_xy=bounds_xy,
            waypoint_step_m=map_waypoint_step,
        )
        if render
        else []
    )

    _plot_debug_figure(
        output_path=output_path,
        route_id_input=route_id_input,
        xml_path=xml_path,
        route_elem=route_elem,
        route_positions=route_positions,
        map_polylines=map_polylines,
        lane_arrows=lane_arrows,
        landmark_annotations=landmark_annotations,
        road_polygons=road_polygons,
        bounds_xy=bounds_xy,
        trigger_distance_m=trigger_distance_m,
        navigation_position=navigation_position,
        sampled_trigger_distance_m=sampled_trigger_distance_m,
        sampled_navigation_position=sampled_navigation_position,
        original_trigger_points=[
            scenario["trigger_point"]
            for scenario in bench2drive_scenarios
            if scenario["trigger_point"] is not None
        ],
        actionable_actions=actionable_actions,
        ego_waypoint=ego_waypoint,
        ego_turn_actions=ego_turn_actions,
        selected_action=selected_action,
        selected_prompt=selected_prompt,
        trigger_score=trigger_score,
        trigger_source_kind=trigger_source_kind,
        trigger_phrasing_mode=trigger_phrasing_mode,
        bench2drive_scenarios=bench2drive_scenarios,
        sampled_actions=sampled_actions,
        sampled_selected_action=selected_trigger.selected_action,
        sampled_selected_prompt=selected_trigger.sampled_text,
        sampled_trigger_score=selected_trigger.score,
        special_case=special_case,
        special_case_primary_prompt=special_case_primary_prompt,
        special_case_secondary_prompt=special_case_secondary_prompt,
        dpi=dpi,
        show=show,
        render=render,
    )

    print(f"Input XML: {xml_path}")
    print(f"Town: {town}")
    print(f"Route length: {route_length_m:.1f} m")
    print(f"Bench2Drive scenarios: {len(bench2drive_scenarios)}")
    print(f"Sampled trigger distance: {sampled_trigger_distance_m:.1f} m")
    print(f"Sampled trigger source: {selected_trigger.source_kind}")
    print(f"Sampled trigger phrasing: {selected_trigger.phrasing_mode}")
    print(f"Sampled prompt: {selected_trigger.sampled_text}")
    if special_case is not None:
        print(f"Route special case: {special_case.kind}")
        print(f"Special primary trigger distance: {trigger_distance_m:.1f} m")
        print(f"Special primary prompt: {special_case_primary_prompt}")
        if special_case.secondary_action is not None and special_case.secondary_trigger_distance_m is not None:
            print(f"Special secondary trigger distance: {special_case.secondary_trigger_distance_m:.1f} m")
            print(f"Special secondary prompt: {special_case_secondary_prompt}")
    print(f"Actionable items: {actionable_actions}")
    print(f"Saved debug figure: {output_path}")


def main() -> None:
    args = parse_args()

    if args.trigger_step_m <= 0:
        raise ValueError("--trigger-step-m must be > 0.")
    if args.map_waypoint_step <= 0:
        raise ValueError("--map-waypoint-step must be > 0.")

    xodr_roots = (
        None if args.xodr_root is None else [path.expanduser().resolve() for path in args.xodr_root]
    )
    map_cache = CarlaMapCache(xodr_search_roots=xodr_roots)

    if args.route_id is None:
        xml_paths = _iter_input_xml_paths(args.input_dir)
        if not xml_paths:
            raise FileNotFoundError(
                f"No bench2drive_*.xml files found under {args.input_dir.resolve()}"
            )
        if args.show:
            print("Batch mode detected; disabling --show to avoid opening many windows.")

        failures: List[Tuple[Path, str]] = []
        for xml_path in xml_paths:
            output_path = _resolve_output_path(xml_path, args.route_id, args.output)
            try:
                _process_single_xml(
                    xml_path=xml_path,
                    route_id_input=xml_path.stem,
                    map_cache=map_cache,
                    trigger_step_m=args.trigger_step_m,
                    map_waypoint_step=args.map_waypoint_step,
                    margin_m=args.margin_m,
                    dpi=args.dpi,
                    show=False,
                    output_path=output_path,
                    render=args.render,
                )
            except Exception as exc:
                failures.append((xml_path, str(exc)))
                print(f"Failed: {xml_path} -> {exc}", file=sys.stderr)
            print("-" * 80)

        print(f"Processed {len(xml_paths) - len(failures)}/{len(xml_paths)} route files.")
        if failures:
            raise RuntimeError(f"{len(failures)} files failed during batch visualization.")
        return

    xml_path = resolve_route_xml(args.route_id, args.input_dir)
    output_path = _resolve_output_path(xml_path, args.route_id, args.output)
    _process_single_xml(
        xml_path=xml_path,
        route_id_input=args.route_id,
        map_cache=map_cache,
        trigger_step_m=args.trigger_step_m,
        map_waypoint_step=args.map_waypoint_step,
        margin_m=args.margin_m,
        dpi=args.dpi,
        show=args.show,
        output_path=output_path,
        render=args.render,
    )


if __name__ == "__main__":
    main()
