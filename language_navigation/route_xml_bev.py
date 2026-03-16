#!/usr/bin/env python3
"""
Visualize language-benchmark XML routes in BEV with XML metadata side panel.
"""

import argparse
import math
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

try:
    from language_navigation.generate_language_xml_distance import (
        CarlaMapCache,
        _get_waypoint_positions,
        _route_length_m,
    )
except ImportError:
    from generate_language_xml_distance import (  # type: ignore
        CarlaMapCache,
        _get_waypoint_positions,
        _route_length_m,
    )


DEFAULT_INPUT_DIR = Path(
    "leaderboard/data/language_benchmark/instruction_following_v0.1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one language benchmark XML route (or all routes in a folder) "
            "in BEV and show XML-derived metadata on the right."
        )
    )
    parser.add_argument(
        "route_id",
        nargs="?",
        default=None,
        help=(
            "Optional route selector. Supports XML path, route stem, numeric id, "
            "or original route id inside XML. If omitted, processes all "
            "bench2drive_*.xml files in --input-dir."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing language benchmark XML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Default: ./debug/<xml_stem>_xml_bev.png. "
            "In batch mode, must be a directory path."
        ),
    )
    parser.add_argument(
        "--map-waypoint-step",
        type=float,
        default=2.0,
        help="Spacing (meters) for sampled map waypoints used in BEV map drawing.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable filled CARLA BEV road-surface rendering.",
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

    direct_path = Path(token).expanduser()
    if direct_path.suffix == ".xml":
        add_stem(direct_path.stem)
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
    if direct_path.exists() and direct_path.is_file():
        candidates.append(direct_path.resolve())

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
            speed = str(int(landmark.value)) if landmark.value > 0 else str(landmark.sub_type).strip()
            label = f"speed {speed} km/h" if speed and speed != "-1" else "speed"
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
    seen: Set[Tuple[int, int, int]] = set()

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


def _extract_scenarios(route_elem: ET.Element) -> List[Dict[str, Any]]:
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

        extras: List[str] = []
        for child in list(scenario_elem):
            if child.tag == "trigger_point":
                continue
            attr_text = ", ".join(f"{k}={v}" for k, v in sorted(child.attrib.items()))
            extras.append(f"{child.tag}({attr_text})" if attr_text else child.tag)

        scenarios.append(
            {
                "name": scenario_elem.attrib.get("name", "unknown"),
                "type": scenario_elem.attrib.get("type", "unknown"),
                "attrs": dict(sorted(scenario_elem.attrib.items())),
                "trigger_point": trigger_point,
                "extras": extras,
            }
        )
    return scenarios


def _extract_instructions(route_elem: ET.Element) -> List[Dict[str, Any]]:
    instructions_elem = route_elem.find("instructions")
    if instructions_elem is None:
        return []

    instructions: List[Dict[str, Any]] = []
    for instruction_elem in instructions_elem.findall("instruction"):
        trigger_elem = instruction_elem.find("trigger")
        expected_behavior_elem = instruction_elem.find("expected_behavior")
        instructions.append(
            {
                "attrs": dict(sorted(instruction_elem.attrib.items())),
                "trigger": None if trigger_elem is None else dict(sorted(trigger_elem.attrib.items())),
                "text": (instruction_elem.findtext("text") or "").strip(),
                "command_id": (instruction_elem.findtext("command_id") or "").strip(),
                "expected_behavior": None
                if expected_behavior_elem is None
                else dict(sorted(expected_behavior_elem.attrib.items())),
                "duration_meters": (instruction_elem.findtext("duration_meters") or "").strip(),
            }
        )
    return instructions


def _extract_evaluation(route_elem: ET.Element) -> List[Dict[str, Any]]:
    evaluation_elem = route_elem.find("evaluation")
    if evaluation_elem is None:
        return []

    metrics: List[Dict[str, Any]] = []
    for metric_elem in evaluation_elem.findall("metric"):
        params = []
        for param_elem in metric_elem.findall("param"):
            params.append(dict(sorted(param_elem.attrib.items())))
        metrics.append(
            {
                "attrs": dict(sorted(metric_elem.attrib.items())),
                "params": params,
            }
        )
    return metrics


def _extract_weathers(route_elem: ET.Element) -> List[Dict[str, str]]:
    weathers_elem = route_elem.find("weathers")
    if weathers_elem is None:
        return []
    return [dict(sorted(weather_elem.attrib.items())) for weather_elem in weathers_elem.findall("weather")]


def _position_at_distance(
    route_positions: List[Tuple[float, float, float]], target_distance: float
) -> Optional[Tuple[float, float]]:
    """Return (x, y) position along the route at the given cumulative distance.

    If target_distance exceeds the route length, returns the last position.
    Returns None if route_positions is empty.
    """
    if not route_positions:
        return None

    if target_distance <= 0.0:
        return (route_positions[0][0], route_positions[0][1])

    cumulative = 0.0
    for i in range(len(route_positions) - 1):
        x0, y0, _ = route_positions[i]
        x1, y1, _ = route_positions[i + 1]
        segment_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        if cumulative + segment_len >= target_distance:
            remaining = target_distance - cumulative
            ratio = remaining / segment_len if segment_len > 1e-9 else 0.0
            interp_x = x0 + ratio * (x1 - x0)
            interp_y = y0 + ratio * (y1 - y0)
            return (interp_x, interp_y)

        cumulative += segment_len

    return (route_positions[-1][0], route_positions[-1][1])


def _format_attrs(attrs: Dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in attrs.items()) if attrs else "none"


def _build_text_panel_lines(
    xml_path: Path,
    route_elem: ET.Element,
    route_positions: List[Tuple[float, float, float]],
    route_length_m: float,
    scenarios: List[Dict[str, Any]],
    instructions: List[Dict[str, Any]],
    evaluation_metrics: List[Dict[str, Any]],
    weathers: List[Dict[str, str]],
) -> str:
    lines: List[str] = []
    route_attrs = dict(sorted(route_elem.attrib.items()))

    lines.append("XML Route Summary")
    lines.append("")
    lines.append(f"xml: {xml_path.name}")
    lines.append(f"route id: {route_attrs.get('id', 'unknown')}")
    lines.append(f"town: {route_attrs.get('town', 'unknown')}")
    lines.append(f"benchmark_type: {route_attrs.get('benchmark_type', 'unknown')}")
    lines.append(f"category: {route_attrs.get('category', 'unknown')}")
    lines.append(f"disable_bg_vehicle: {route_attrs.get('disable_bg_vehicle', 'unknown')}")
    lines.append(f"waypoints: {len(route_positions)}")
    lines.append(f"route length: {route_length_m:.1f} m")
    lines.append("")
    lines.append("Route attributes:")
    for key, value in route_attrs.items():
        if key == "id":
            continue
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Instructions:")
    if instructions:
        for index, instruction in enumerate(instructions, start=1):
            attrs = instruction["attrs"]
            lines.append(
                f"{index}. id={attrs.get('id', '?')} priority={attrs.get('priority', '?')}"
            )
            lines.append(f"   trigger: {_format_attrs(instruction['trigger'] or {})}")
            lines.append(f"   text: {instruction['text'] or 'none'}")
            lines.append(f"   command_id: {instruction['command_id'] or 'none'}")
            lines.append(
                f"   expected_behavior: {_format_attrs(instruction['expected_behavior'] or {})}"
            )
            lines.append(f"   duration_meters: {instruction['duration_meters'] or 'none'}")
    else:
        lines.append("none")

    lines.append("")
    lines.append("Evaluation:")
    if evaluation_metrics:
        for index, metric in enumerate(evaluation_metrics, start=1):
            lines.append(f"{index}. {_format_attrs(metric['attrs'])}")
            if metric["params"]:
                for param in metric["params"]:
                    lines.append(f"   param: {_format_attrs(param)}")
            else:
                lines.append("   param: none")
    else:
        lines.append("none")

    lines.append("")
    lines.append("Scenarios:")
    if scenarios:
        for index, scenario in enumerate(scenarios, start=1):
            trigger = scenario["trigger_point"]
            if trigger is None:
                trigger_text = "none"
            else:
                trigger_text = (
                    f"x={trigger[0]:.1f}, y={trigger[1]:.1f}, z={trigger[2]:.1f}, yaw={trigger[3]:.1f}"
                )
            lines.append(
                f"{index}. {scenario['name']} type={scenario['type']} trigger={trigger_text}"
            )
            extra_attrs = {
                key: value
                for key, value in scenario["attrs"].items()
                if key not in {"name", "type"}
            }
            lines.append(f"   attrs: {_format_attrs(extra_attrs)}")
            lines.append(
                f"   extras: {', '.join(scenario['extras']) if scenario['extras'] else 'none'}"
            )
    else:
        lines.append("none")

    lines.append("")
    lines.append("Weathers:")
    if weathers:
        for index, weather in enumerate(weathers, start=1):
            route_percentage = weather.get("route_percentage", "?")
            summary_keys = [
                "cloudiness",
                "precipitation",
                "wetness",
                "fog_density",
                "wind_intensity",
                "sun_altitude_angle",
            ]
            summary = ", ".join(
                f"{key}={weather[key]}" for key in summary_keys if key in weather
            )
            lines.append(f"{index}. route_percentage={route_percentage} {summary}")
    else:
        lines.append("none")

    return "\n".join(lines)


def _collect_instruction_trigger_points(
    instructions: List[Dict[str, Any]],
    route_positions: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, str, str]]:
    """Collect instruction trigger points as (x, y, label, color_key).

    Distinguishes speed triggers (first instruction, typically id=1) from
    action triggers (subsequent instructions).
    """
    trigger_points: List[Tuple[float, float, str, str]] = []

    for instruction in instructions:
        trigger = instruction.get("trigger")
        if trigger is None:
            continue

        trigger_type = trigger.get("type", "")
        if trigger_type != "distance_traveled":
            continue

        try:
            distance_value = float(trigger.get("value", "0"))
        except (ValueError, TypeError):
            continue

        position = _position_at_distance(route_positions, distance_value)
        if position is None:
            continue

        inst_id = instruction.get("attrs", {}).get("id", "?")
        command_id = instruction.get("command_id", "")
        text_snippet = instruction.get("text", "")[:20]

        if inst_id == "1" or distance_value == 0.0:
            label = f"T1: {text_snippet}"
            color_key = "speed_trigger"
        else:
            label = f"T{inst_id}: {text_snippet}"
            color_key = "action_trigger"

        trigger_points.append((position[0], position[1], label, color_key))

    return trigger_points


def _plot_figure(
    output_path: Path,
    xml_path: Path,
    route_elem: ET.Element,
    route_positions: List[Tuple[float, float, float]],
    map_polylines: List[List[Tuple[float, float]]],
    lane_arrows: List[Tuple[float, float, float, float]],
    landmark_annotations: List[Tuple[float, float, str, str]],
    road_polygons: List[List[Tuple[float, float]]],
    bounds_xy: Tuple[float, float, float, float],
    scenarios: List[Dict[str, Any]],
    instructions: List[Dict[str, Any]],
    evaluation_metrics: List[Dict[str, Any]],
    weathers: List[Dict[str, str]],
    dpi: int,
    show: bool,
    render: bool,
) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.45], wspace=0.02)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_text = fig.add_subplot(grid[0, 1])

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

    trigger_points = [scenario["trigger_point"] for scenario in scenarios if scenario["trigger_point"]]
    if trigger_points:
        trigger_x = [point[0] for point in trigger_points]
        trigger_y = [point[1] for point in trigger_points]
        ax_map.scatter(
            trigger_x,
            trigger_y,
            marker="^",
            s=70,
            color="#f59e0b",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="scenario trigger",
        )

    instruction_triggers = _collect_instruction_trigger_points(instructions, route_positions)
    speed_triggers = [(x, y, label) for x, y, label, key in instruction_triggers if key == "speed_trigger"]
    action_triggers = [(x, y, label) for x, y, label, key in instruction_triggers if key == "action_trigger"]

    if speed_triggers:
        speed_x = [pt[0] for pt in speed_triggers]
        speed_y = [pt[1] for pt in speed_triggers]
        ax_map.scatter(
            speed_x,
            speed_y,
            marker="D",
            s=80,
            color="#3b82f6",
            edgecolors="white",
            linewidths=0.8,
            zorder=7,
            label="speed trigger",
        )
        for x, y, label in speed_triggers:
            ax_map.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=7,
                color="#1d4ed8",
                fontweight="bold",
                zorder=8,
            )

    if action_triggers:
        action_x = [pt[0] for pt in action_triggers]
        action_y = [pt[1] for pt in action_triggers]
        ax_map.scatter(
            action_x,
            action_y,
            marker="s",
            s=80,
            color="#8b5cf6",
            edgecolors="white",
            linewidths=0.8,
            zorder=7,
            label="action trigger",
        )
        for x, y, label in action_triggers:
            ax_map.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(8, -12),
                fontsize=7,
                color="#6d28d9",
                fontweight="bold",
                zorder=8,
            )

    min_x, max_x, min_y, max_y = bounds_xy
    ax_map.set_xlim(min_x, max_x)
    ax_map.set_ylim(max_y, min_y)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_title(
        f"BEV XML Route: {xml_path.name} | route_id={route_elem.attrib.get('id', 'unknown')}"
    )
    ax_map.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=6,
        fontsize=7.8,
        frameon=True,
    )

    route_length_m = _route_length_m(route_positions)
    panel_text = _build_text_panel_lines(
        xml_path=xml_path,
        route_elem=route_elem,
        route_positions=route_positions,
        route_length_m=route_length_m,
        scenarios=scenarios,
        instructions=instructions,
        evaluation_metrics=evaluation_metrics,
        weathers=weathers,
    )
    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.98,
        panel_text,
        va="top",
        ha="left",
        fontsize=9.6,
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
        is_file_path = (
            output_arg is not None
            and output_arg.suffix
            and not output_arg.is_dir()
            and not str(output_arg).endswith("/")
        )
        if is_file_path:
            raise ValueError(
                "When route_id is omitted (batch mode), --output must be a directory path."
            )
        output_dir = Path("debug") if output_arg is None else output_arg
        return (output_dir / f"{xml_path.stem}_xml_bev.png").resolve()

    if output_arg is None:
        return (Path("debug") / f"{xml_path.stem}_xml_bev.png").resolve()
    return output_arg.resolve()


def _process_single_xml(
    *,
    xml_path: Path,
    route_id_input: str,
    map_cache: CarlaMapCache,
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

    carla_map = map_cache.get_map(town)
    scenarios = _extract_scenarios(route_elem)
    instructions = _extract_instructions(route_elem)
    evaluation_metrics = _extract_evaluation(route_elem)
    weathers = _extract_weathers(route_elem)

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
    road_polygons = (
        _collect_road_polygons_in_bounds(
            carla_map=carla_map,
            bounds_xy=bounds_xy,
            waypoint_step_m=map_waypoint_step,
        )
        if render
        else []
    )

    _plot_figure(
        output_path=output_path,
        xml_path=xml_path,
        route_elem=route_elem,
        route_positions=route_positions,
        map_polylines=map_polylines,
        lane_arrows=lane_arrows,
        landmark_annotations=landmark_annotations,
        road_polygons=road_polygons,
        bounds_xy=bounds_xy,
        scenarios=scenarios,
        instructions=instructions,
        evaluation_metrics=evaluation_metrics,
        weathers=weathers,
        dpi=dpi,
        show=show,
        render=render,
    )

    route_length_m = _route_length_m(route_positions)
    print(f"Input XML: {xml_path}")
    print(f"Town: {town}")
    print(f"Route length: {route_length_m:.1f} m")
    print(f"Instructions: {len(instructions)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Weathers: {len(weathers)}")
    print(f"Saved figure: {output_path}")


def main() -> None:
    args = parse_args()

    if args.map_waypoint_step <= 0:
        raise ValueError("--map-waypoint-step must be > 0.")

    xodr_roots = (
        None if args.xodr_root is None else [path.expanduser().resolve() for path in args.xodr_root]
    )
    map_cache = CarlaMapCache(xodr_search_roots=xodr_roots)
    render = not args.no_render

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
                    map_waypoint_step=args.map_waypoint_step,
                    margin_m=args.margin_m,
                    dpi=args.dpi,
                    show=False,
                    output_path=output_path,
                    render=render,
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
        map_waypoint_step=args.map_waypoint_step,
        margin_m=args.margin_m,
        dpi=args.dpi,
        show=args.show,
        output_path=output_path,
        render=render,
    )


if __name__ == "__main__":
    main()
