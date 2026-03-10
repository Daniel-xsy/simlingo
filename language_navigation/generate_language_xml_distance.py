#!/usr/bin/env python3
"""
Generate naive distance-triggered language benchmark XML from Bench2Drive XML.

Usage:
    python leaderboard/scripts/generate_language_xml_distance.py \
        leaderboard/data/bench2drive_split/bench2drive_00.xml
"""

import argparse
import copy
import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import carla
except ImportError as exc:
    carla = None
    CARLA_IMPORT_ERROR = exc
else:
    CARLA_IMPORT_ERROR = None


# Instruction categories -> paraphrase pools + behavior mapping.
# Add more phrases to each list as you expand your benchmark.
INSTRUCTION_LIBRARY: Dict[str, Dict[str, object]] = {
    "lane_follow": {
        "command_id": 4,
        "expected_behavior": {"type": "lane_follow"},
        "texts": [
            "follow the road",
            "keep the current status",
            "going straight and maintain the speed",
            "stay in your lane and continue",
            "continue driving in your current lane",
            "stay within the lane you are currently in",
            "do not change lanes, maintain your present lane position",
            "keep following the same lane ahead",
            "remain centered in this lane and proceed forward",
            "hold your lane and continue straight",
            "avoid lane changes and track the current lane markings",
            "follow the lane you are in without merging or drifting",
            "drive forward while staying in the same traffic lane",
        ],
    },
    "lane_change_left": {
        "command_id": 5,
        "expected_behavior": {"type": "lane_change", "direction": "left"},
        "texts": [
            "do a lane change to the left",
            "move to the left lane",
            "switch to the left lane now",
            "merge into the lane on the left",
            "shift over to the left lane",
            "change lanes toward the left",
            "slide into the left lane",
            "take the lane to your left",
            "drift over to the left lane",
            "enter the left-hand lane",
        ],
    },
    "lane_change_right": {
        "command_id": 6,
        "expected_behavior": {"type": "lane_change", "direction": "right"},
        "texts": [
            "do a lane change to the right",
            "move to the right lane",
            "switch to the right lane now",
            "merge into the lane on the right",
            "shift over to the right lane",
            "change lanes toward the right",
            "slide into the right lane",
            "take the lane to your right",
            "drift over to the right lane",
            "enter the right-hand lane",
        ],
    },
    "turn_left": {
        "command_id": 1,
        "expected_behavior": {"type": "turn", "direction": "left"},
        "texts": [
            "go left at the next intersection",
            "take the next left turn",
            "turn left at the upcoming junction",
            "make a left turn at the next intersection",
            "take a left at the upcoming corner",
            "hang a left at the next junction",
            "bear left at the intersection ahead",
            "turn to the left at the next crossroad",
            "execute a left turn at the upcoming intersection",
        ],
    },
    "turn_right": {
        "command_id": 2,
        "expected_behavior": {"type": "turn", "direction": "right"},
        "texts": [
            "go right at the next intersection",
            "take the next right turn",
            "turn right at the upcoming junction",
            "make a right turn at the next intersection",
            "take a right at the upcoming corner",
            "hang a right at the next junction",
            "bear right at the intersection ahead",
            "turn to the right at the next crossroad",
            "execute a right turn at the upcoming intersection",
        ],
    },
    "turn_straight": {
        "command_id": 3,
        "expected_behavior": {"type": "turn", "direction": "straight"},
        "texts": [
            "go straight at the next intersection",
            "continue straight through the next junction",
            "keep straight at the intersection",
            "proceed straight ahead at the next intersection",
            "drive straight through the upcoming junction",
            "maintain a straight path at the crossing",
            "do not turn and continue straight",
            "pass straight through the intersection ahead",
            "stay straight at the next crossroad",
            "head straight through the upcoming intersection",
        ],
    },
"accelerate_vague": {
    "command_id": -1,
    "expected_behavior": {"type": "accelerate", "min_acceleration_ms2": "0.8"},
    "texts": [
        "speed up",
        "go faster",
        "increase your speed",
        "pick up the pace",
        "accelerate a bit",
        "gain some speed",
        "step on the gas",
        "move faster",
        "drive faster",
        "speed things up",
    ],
},
"decelerate_vague": {
    "command_id": -1,
    "expected_behavior": {"type": "target_speed", "speed_ms": "6", "tolerance_ms": "2.0"},
    "texts": [
        "slow down",
        "reduce your speed",
        "drive a bit slower",
        "decelerate slightly",
        "take it slower",
        "back off the gas",
        "step on the brakes",
        "drop your speed",
        "slow it a little",
    ],
},
}

OPTIONAL_LANE_FOLLOW_PROBABILITY = 0.2


@dataclass(frozen=True)
class RouteActionabilitySample:
    distance_m: float
    position: Tuple[float, float, float]
    actions: Tuple[str, ...]
    scored_actions: Tuple[str, ...]
    score: int
    is_junction: bool
    lane_id: Optional[int]
    road_id: Optional[int]
    section_id: Optional[int]
    yaw_deg: Optional[float]
    remaining_route_distance_m: float


@dataclass(frozen=True)
class SelectedNavigationTrigger:
    distance_m: float
    position: Tuple[float, float, float]
    actions: Tuple[str, ...]
    selected_action: str
    score: int
    is_junction: bool
    source_kind: str
    phrasing_mode: str
    sample_index: int
    sampled_text: str


class SpeedLimitMapCache:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir = (
            Path("team_code/speed_limits").resolve() if base_dir is None else base_dir.resolve()
        )
        self._maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_speed_limit_kmh(self, town: str, position: Tuple[float, float, float]) -> float:
        locations, speed_limits = self._load_map(town)
        point_xy = np.array([position[0], position[1]], dtype=np.float32)
        deltas = locations[:, :2] - point_xy
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)
        nearest_idx = int(np.argmin(distances_sq))
        return float(speed_limits[nearest_idx])

    def _load_map(self, town: str) -> Tuple[np.ndarray, np.ndarray]:
        if town in self._maps:
            return self._maps[town]

        file_path = self._base_dir / f"{town}_speed_limits.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Speed limit file not found for {town}: {file_path}")

        map_data = np.load(file_path, allow_pickle=True).item()
        locations = np.asarray(map_data["locations"], dtype=np.float32)
        speed_limits = np.asarray(map_data["speed_limits"], dtype=np.float32)
        self._maps[town] = (locations, speed_limits)
        return self._maps[town]


def _default_xodr_search_roots() -> List[Path]:
    roots: List[Path] = []

    carla_xodr_root = os.environ.get("CARLA_XODR_ROOT")
    if carla_xodr_root:
        roots.append(Path(carla_xodr_root).expanduser())

    carla_root = os.environ.get("CARLA_ROOT")
    if carla_root:
        base = Path(carla_root).expanduser()
        roots.extend(
            [
                base / "CarlaUE4" / "Content" / "Carla" / "Maps",
                base / "Carla" / "Maps",
                base / "Maps",
            ]
        )

    home = Path.home()
    roots.extend(
        [
            home / "carlaCache" / "0.9.15" / "Carla" / "Maps",
            home / "software" / "carla0915" / "CarlaUE4" / "Content" / "Carla" / "Maps",
            Path("/opt/carla/CarlaUE4/Content/Carla/Maps"),
        ]
    )

    # Keep first occurrence order and only existing folders.
    deduped: List[Path] = []
    seen: Set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir():
            deduped.append(resolved)
    return deduped


def _resolve_xodr_path(town: str, search_roots: List[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    for root in search_roots:
        candidates.extend(
            [
                root / "OpenDrive" / f"{town}.xodr",
                root / "OpenDrive" / f"{town}_Opt.xodr",
                root / town / "OpenDrive" / f"{town}.xodr",
                root / town / "OpenDrive" / f"{town}_Opt.xodr",
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


class CarlaMapCache:
    def __init__(self, xodr_search_roots: Optional[List[Path]] = None) -> None:
        if carla is None:
            raise RuntimeError(
                "CARLA Python API is required for map-aware action sampling."
            ) from CARLA_IMPORT_ERROR
        self._xodr_search_roots = (
            _default_xodr_search_roots()
            if not xodr_search_roots
            else [p.resolve() for p in xodr_search_roots]
        )
        self._maps: Dict[str, "carla.Map"] = {}

    def get_map(self, town: str) -> "carla.Map":
        if town in self._maps:
            return self._maps[town]

        xodr_path = _resolve_xodr_path(town, self._xodr_search_roots)
        if xodr_path is None:
            roots = ", ".join(str(root) for root in self._xodr_search_roots)
            raise FileNotFoundError(
                f"Could not find OpenDRIVE map for {town}. "
                f"Searched under: {roots}"
            )

        map_obj = carla.Map(town, xodr_path.read_text(encoding="utf-8"))
        self._maps[town] = map_obj
        return map_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a language benchmark XML from one Bench2Drive XML using "
            "distance-based instruction triggers."
        )
    )
    parser.add_argument(
        "input_xml",
        type=Path,
        help="Input Bench2Drive XML file or a folder containing XML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "leaderboard/data/language_benchmark/instruction_following/"
            " with one file per retained action."
        ),
    )
    parser.add_argument(
        "--category",
        default="instruction_following",
        help="language benchmark category to set on route elements.",
    )
    parser.add_argument(
        "--trigger-step-m",
        type=float,
        default=50.0,
        help="Distance interval (meters) between generated instructions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for reproducible instruction sampling. "
            "If omitted, uses non-deterministic randomness."
        ),
    )
    parser.add_argument(
        "--num-instructions",
        type=int,
        default=2,
        help=(
            "Fixed instruction count. Must be 2 for accelerate->action generation."
        ),
    )
    parser.add_argument(
        "--instruction-style",
        choices=["all", "vague", "precise"],
        default="all",
        help=(
            "Speed instruction style. 'vague' uses 'speed up/slow down'; "
            "'precise' uses explicit target speed; 'all' mixes both."
        ),
    )
    parser.add_argument(
        "--xodr-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional OpenDRIVE maps root directory. Can be provided multiple "
            "times. If unset, script searches common CARLA locations."
        ),
    )
    return parser.parse_args()


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _get_waypoint_positions(waypoints_elem: ET.Element) -> List[Tuple[float, float, float]]:
    positions = []
    for pos in waypoints_elem.findall("position"):
        positions.append(
            (
                float(pos.attrib["x"]),
                float(pos.attrib["y"]),
                float(pos.attrib["z"]),
            )
        )
    return positions


def _route_length_m(positions: List[Tuple[float, float, float]]) -> float:
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        total += _distance(positions[i - 1], positions[i])
    return total


def _position_at_distance(
    positions: List[Tuple[float, float, float]], target_distance_m: float
) -> Tuple[float, float, float]:
    if not positions:
        raise ValueError("Cannot sample route position from empty waypoint list.")
    if target_distance_m <= 0.0 or len(positions) == 1:
        return positions[0]

    traversed = 0.0
    for idx in range(1, len(positions)):
        start = positions[idx - 1]
        end = positions[idx]
        seg_len = _distance(start, end)
        if seg_len <= 1e-6:
            continue
        if traversed + seg_len >= target_distance_m:
            t = (target_distance_m - traversed) / seg_len
            return (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                start[2] + t * (end[2] - start[2]),
            )
        traversed += seg_len

    return positions[-1]


def _normalize_yaw_delta_deg(delta_deg: float) -> float:
    while delta_deg > 180.0:
        delta_deg -= 360.0
    while delta_deg < -180.0:
        delta_deg += 360.0
    return delta_deg


def _compute_turn_category(
    current_waypoint: "carla.Waypoint",
    next_waypoint: "carla.Waypoint",
    threshold_deg: float = 35.0,
) -> str:
    yaw_delta = _normalize_yaw_delta_deg(
        next_waypoint.transform.rotation.yaw - current_waypoint.transform.rotation.yaw
    )
    if abs(yaw_delta) < threshold_deg:
        return "turn_straight"
    if yaw_delta < 0.0:
        return "turn_left"
    return "turn_right"


def _is_same_direction_lane(
    source_waypoint: "carla.Waypoint",
    candidate_waypoint: "carla.Waypoint",
    max_yaw_delta_deg: float = 45.0,
) -> bool:
    yaw_delta = _normalize_yaw_delta_deg(
        candidate_waypoint.transform.rotation.yaw - source_waypoint.transform.rotation.yaw
    )
    return abs(yaw_delta) < max_yaw_delta_deg


def _can_change_lane(ego_waypoint: "carla.Waypoint", direction: str) -> bool:
    if direction == "left":
        if ego_waypoint.lane_change not in (carla.LaneChange.Left, carla.LaneChange.Both):
            return False
        adjacent = ego_waypoint.get_left_lane()
    elif direction == "right":
        if ego_waypoint.lane_change not in (carla.LaneChange.Right, carla.LaneChange.Both):
            return False
        adjacent = ego_waypoint.get_right_lane()
    else:
        raise ValueError(f"Unsupported lane-change direction: {direction}")

    if adjacent is None:
        return False
    if adjacent.lane_type != carla.LaneType.Driving:
        return False
    return _is_same_direction_lane(ego_waypoint, adjacent)


def _scan_turn_actions(
    start_waypoint: "carla.Waypoint",
    scan_distance_m: float = 45.0,
    step_m: float = 2.0,
) -> Set[str]:
    """Scan upcoming topology and return feasible turn_* actions."""
    turn_actions: Set[str] = set()
    frontier = [start_waypoint]
    visited: Set[Tuple[int, int, int, int]] = set()
    max_steps = max(1, int(scan_distance_m / step_m))

    for _ in range(max_steps):
        if not frontier:
            break

        next_frontier: List["carla.Waypoint"] = []
        for waypoint in frontier:
            key = (
                waypoint.road_id,
                waypoint.section_id,
                waypoint.lane_id,
                int(round(waypoint.s * 10.0)),
            )
            if key in visited:
                continue
            visited.add(key)

            next_candidates = waypoint.next(step_m)
            if not next_candidates:
                continue

            # Only treat branch points near junctions as turn options.
            is_branch_near_junction = len(next_candidates) > 1 and (
                waypoint.is_junction or any(candidate.is_junction for candidate in next_candidates)
            )
            if is_branch_near_junction:
                for candidate in next_candidates:
                    branch_waypoint = candidate
                    branch_traversed = step_m
                    while branch_traversed < scan_distance_m:
                        next_branch = branch_waypoint.next(step_m)
                        if not next_branch:
                            break
                        next_branch_waypoint = next_branch[0]
                        branch_waypoint = next_branch_waypoint
                        branch_traversed += step_m
                        if not branch_waypoint.is_junction:
                            break
                    turn_actions.add(_compute_turn_category(start_waypoint, branch_waypoint))
                continue

            # Single successor inside/at junction can still imply a forced turn.
            candidate = next_candidates[0]
            if waypoint.is_junction or candidate.is_junction:
                branch_waypoint = candidate
                branch_traversed = step_m
                while branch_traversed < scan_distance_m:
                    next_branch = branch_waypoint.next(step_m)
                    if not next_branch:
                        break
                    next_branch_waypoint = next_branch[0]
                    branch_waypoint = next_branch_waypoint
                    branch_traversed += step_m
                    if not branch_waypoint.is_junction:
                        break
                turn_actions.add(_compute_turn_category(start_waypoint, branch_waypoint))

            next_frontier.append(candidate)

        frontier = next_frontier

    return turn_actions


def _build_actionable_navigation_categories(
    carla_map: "carla.Map",
    ego_position: Tuple[float, float, float],
    max_turn_scan_distance_m: float = 45.0,
) -> List[str]:
    actions: List[str] = ["lane_follow"]
    ego_waypoint = carla_map.get_waypoint(
        carla.Location(x=ego_position[0], y=ego_position[1], z=ego_position[2]),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_waypoint is None:
        return actions

    if _can_change_lane(ego_waypoint, "left"):
        actions.append("lane_change_left")
    if _can_change_lane(ego_waypoint, "right"):
        actions.append("lane_change_right")

    turn_actions = _scan_turn_actions(ego_waypoint, scan_distance_m=max_turn_scan_distance_m)
    for turn_action in ("turn_left", "turn_right", "turn_straight"):
        if turn_action in turn_actions:
            actions.append(turn_action)

    return actions


def _score_actionable_categories(actions: List[str]) -> int:
    return sum(1 for action in actions if action != "lane_follow")


def _sample_route_actionability(
    route_positions: List[Tuple[float, float, float]],
    trigger_step_m: float,
    carla_map: "carla.Map",
) -> List[RouteActionabilitySample]:
    if trigger_step_m <= 0:
        raise ValueError("--trigger-step-m must be > 0.")

    total_length = _route_length_m(route_positions)
    max_distance = max(total_length - 1.0, 0.0)

    distance_values: Set[float] = {0.0, max_distance}

    # Densify the route start so trigger guards can place the trigger at 5 m
    # or at the first junction entry instead of snapping to the main scan step.
    fine_step_m = 1.0
    current = 0.0
    dense_limit = min(max_distance, max(10.0, trigger_step_m))
    while current <= dense_limit + 1e-6:
        distance_values.add(round(min(current, max_distance), 3))
        current += fine_step_m

    current = 0.0
    while current <= max_distance + 1e-6:
        distance_values.add(round(min(current, max_distance), 3))
        current += trigger_step_m

    distances = sorted(distance_values)

    samples: List[RouteActionabilitySample] = []
    for distance_m in distances:
        position = _position_at_distance(route_positions, distance_m)
        remaining_route_distance_m = max_distance - distance_m
        max_turn_scan_distance_m = max(0.0, remaining_route_distance_m)
        actions = _build_actionable_navigation_categories(
            carla_map, position, max_turn_scan_distance_m=max_turn_scan_distance_m
        )
        scored_actions = [action for action in actions if action != "lane_follow"]
        ego_waypoint = carla_map.get_waypoint(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        samples.append(
            RouteActionabilitySample(
                distance_m=distance_m,
                position=position,
                actions=tuple(actions),
                scored_actions=tuple(scored_actions),
                score=len(scored_actions),
                is_junction=bool(ego_waypoint.is_junction) if ego_waypoint is not None else False,
                lane_id=ego_waypoint.lane_id if ego_waypoint is not None else None,
                road_id=ego_waypoint.road_id if ego_waypoint is not None else None,
                section_id=ego_waypoint.section_id if ego_waypoint is not None else None,
                yaw_deg=(
                    ego_waypoint.transform.rotation.yaw if ego_waypoint is not None else None
                ),
                remaining_route_distance_m=remaining_route_distance_m,
            )
        )
    return samples


def _yaw_delta_abs_deg(a_deg: Optional[float], b_deg: Optional[float]) -> float:
    if a_deg is None or b_deg is None:
        return 180.0
    return abs(_normalize_yaw_delta_deg(a_deg - b_deg))


def _is_straight_approach_candidate(
    previous_sample: RouteActionabilitySample,
    junction_sample: RouteActionabilitySample,
    max_yaw_delta_deg: float = 15.0,
) -> bool:
    if previous_sample.is_junction:
        return False
    if junction_sample.road_id is None or previous_sample.road_id is None:
        return False
    return _yaw_delta_abs_deg(previous_sample.yaw_deg, junction_sample.yaw_deg) <= max_yaw_delta_deg


def _collect_straight_approach_indices(
    samples: List[RouteActionabilitySample], junction_index: int
) -> List[int]:
    indices: List[int] = []
    for idx in range(junction_index - 1, -1, -1):
        sample = samples[idx]
        if sample.is_junction:
            break
        if not _is_straight_approach_candidate(sample, samples[junction_index]):
            break
        indices.append(idx)
    indices.reverse()
    return indices


def _first_valid_trigger_index(samples: List[RouteActionabilitySample], min_distance_m: float) -> int:
    for idx, sample in enumerate(samples):
        if sample.distance_m >= min_distance_m:
            return idx
    return len(samples) - 1


def _first_junction_index(samples: List[RouteActionabilitySample]) -> Optional[int]:
    for idx, sample in enumerate(samples):
        if sample.is_junction:
            return idx
    return None


def _navigation_text_variants(category: str, phrasing_mode: str) -> List[str]:
    if category == "turn_left":
        if phrasing_mode == "at_junction":
            return [
                "turn left through the junction ahead",
                "make the left turn now as the junction opens up",
                "take the left path through this junction",
            ]
        return [
            "go left at the next intersection",
            "turn left at the upcoming junction",
            "hang a left at the next junction",
        ]
    if category == "turn_right":
        if phrasing_mode == "at_junction":
            return [
                "turn right through the junction ahead",
                "make the right turn now as the junction opens up",
                "take the right path through this junction",
            ]
        return [
            "go right at the next intersection",
            "turn right at the upcoming junction",
            "hang a right at the next junction",
        ]
    if category == "turn_straight":
        if phrasing_mode == "at_junction":
            return [
                "continue straight through the junction ahead",
                "hold straight as you enter this junction",
                "keep straight through the crossing in front of you",
            ]
        return [
            "go straight at the next intersection",
            "continue straight through the next junction",
            "drive straight through the upcoming junction",
        ]
    return list(INSTRUCTION_LIBRARY[category]["texts"])


def _sample_navigation_instruction(
    rng: random.Random,
    actionable_categories: List[str],
    phrasing_mode: str = "default",
) -> Dict[str, object]:
    """Sample only from map-validated navigation actions for middle slots."""
    if not actionable_categories:
        actionable_categories = ["lane_follow"]
    category = rng.choice(actionable_categories)
    entry = INSTRUCTION_LIBRARY[category]
    return {
        "text": rng.choice(_navigation_text_variants(category, phrasing_mode)),
        "command_id": entry["command_id"],
        "expected_behavior": dict(entry["expected_behavior"]),
    }


def select_navigation_trigger(
    route_positions: List[Tuple[float, float, float]],
    trigger_step_m: float,
    carla_map: "carla.Map",
    rng: random.Random,
) -> Tuple[SelectedNavigationTrigger, List[RouteActionabilitySample]]:
    min_trigger_distance_m = 5.0
    samples = _sample_route_actionability(route_positions, trigger_step_m, carla_map)
    if not samples:
        raise ValueError("Adaptive trigger selection produced no route samples.")

    best_score = max(sample.score for sample in samples)
    candidate_indices = [idx for idx, sample in enumerate(samples) if sample.score == best_score]
    winner_index = candidate_indices[0]
    winner = samples[winner_index]
    source_kind = "max_score_earliest"
    phrasing_mode = "at_junction" if winner.is_junction else "approach"

    if winner.is_junction and winner_index > 0:
        previous_sample = samples[winner_index - 1]
        if _is_straight_approach_candidate(previous_sample, winner):
            straight_indices = _collect_straight_approach_indices(samples, winner_index)
            straight_indices = [
                idx for idx in straight_indices if samples[idx].distance_m >= min_trigger_distance_m
            ]
            if straight_indices and rng.random() < 0.5:
                winner_index = rng.choice(straight_indices)
                winner = samples[winner_index]
                source_kind = "straight_approach_randomized"
                phrasing_mode = "approach"
            else:
                source_kind = "junction_randomized"
                phrasing_mode = "at_junction"

    if winner.distance_m < min_trigger_distance_m:
        candidate_index = _first_valid_trigger_index(samples, min_trigger_distance_m)
        candidate = samples[candidate_index]
        if not winner.is_junction and candidate.is_junction:
            junction_index = _first_junction_index(samples)
            if junction_index is not None:
                winner_index = junction_index
                winner = samples[winner_index]
                source_kind = "junction_entry_min_offset"
                phrasing_mode = "at_junction"
        else:
            winner_index = candidate_index
            winner = candidate
            source_kind = "min_offset_guard"
            phrasing_mode = "at_junction" if winner.is_junction else "approach"

    actionable_categories = list(winner.scored_actions)
    if not actionable_categories:
        actionable_categories = list(winner.actions) if winner.actions else ["lane_follow"]
    template = _sample_navigation_instruction(
        rng, actionable_categories, phrasing_mode=phrasing_mode
    )
    command_id = int(template["command_id"])
    selected_action = next(
        (
            action
            for action in actionable_categories
            if int(INSTRUCTION_LIBRARY[action]["command_id"]) == command_id
        ),
        actionable_categories[0],
    )
    return (
        SelectedNavigationTrigger(
            distance_m=winner.distance_m,
            position=winner.position,
            actions=tuple(actionable_categories),
            selected_action=selected_action,
            score=winner.score,
            is_junction=winner.is_junction,
            source_kind=source_kind,
            phrasing_mode=phrasing_mode,
            sample_index=winner_index,
            sampled_text=str(template["text"]),
        ),
        samples,
    )


def _build_default_scenarios(route_elem: ET.Element) -> ET.Element:
    """
    Build a minimal scenarios block required by RouteParser.
    We intentionally ignore original scenario triggers and keep only FreeRide.
    """
    waypoints = route_elem.find("waypoints")
    if waypoints is None:
        raise ValueError(f"Route {route_elem.attrib.get('id', 'unknown')} has no <waypoints>.")

    positions = _get_waypoint_positions(waypoints)
    if not positions:
        raise ValueError(f"Route {route_elem.attrib.get('id', 'unknown')} has empty <waypoints>.")

    x0, y0, z0 = positions[0]

    # Estimate heading from first segment; fallback to 0.0 if unavailable.
    yaw_deg = 0.0
    if len(positions) >= 2:
        x1, y1, _ = positions[1]
        yaw_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))

    scenarios_elem = ET.Element("scenarios")
    scenario_elem = ET.SubElement(
        scenarios_elem, "scenario", {"name": "FreeRide_1", "type": "FreeRide"}
    )
    ET.SubElement(
        scenario_elem,
        "trigger_point",
        {
            "x": f"{x0:.1f}",
            "y": f"{y0:.1f}",
            "z": f"{z0:.1f}",
            "yaw": f"{yaw_deg:.1f}",
        },
    )
    return scenarios_elem


def _indent_xml_compat(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print XML for Python versions without ET.indent."""
    if hasattr(ET, "indent"):
        ET.indent(elem, space="  ")
        return

    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml_compat(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent + "  "
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _append_instruction(
    instructions_elem: ET.Element,
    instruction_id: int,
    trigger_distance_m: float,
    template: Dict[str, object],
    duration_meters: float,
) -> None:
    instr = ET.SubElement(
        instructions_elem,
        "instruction",
        {"id": str(instruction_id), "priority": "primary"},
    )

    trigger = ET.SubElement(
        instr,
        "trigger",
        {"type": "distance_traveled", "value": f"{trigger_distance_m:.1f}"},
    )
    trigger.text = None

    text_elem = ET.SubElement(instr, "text")
    text_elem.text = str(template["text"])

    command_elem = ET.SubElement(instr, "command_id")
    command_elem.text = str(template["command_id"])

    expected_behavior = dict(template["expected_behavior"])
    ET.SubElement(instr, "expected_behavior", expected_behavior)

    duration_elem = ET.SubElement(instr, "duration_meters")
    if duration_meters < 0:
        duration_elem.text = "-1"
    else:
        duration_elem.text = f"{duration_meters:.1f}"


def _build_precise_accelerate_instruction(
    rng: random.Random,
    target_speed_ms: int,
) -> Dict[str, object]:
    text = rng.choice(
        [
            f"accelerate to {target_speed_ms} m/s",
            f"set your speed to {target_speed_ms} m/s",
            f"reach {target_speed_ms} m/s",
        ]
    )
    return {
        "text": text,
        "command_id": -1,
        "expected_behavior": {
            "type": "target_speed",
            "speed_ms": str(target_speed_ms),
            "tolerance_ms": "1.5",
        },
    }


def _sample_instruction_template(
    rng: random.Random,
    style: str,
) -> Dict[str, object]:
    category = rng.choice(
        [
            "lane_follow",
            "lane_change_left",
            "lane_change_right",
            "turn_left",
            "turn_right",
            "turn_straight",
            "accelerate",
            "decelerate",
        ]
    )

    if category == "accelerate":
        if style == "vague":
            entry = INSTRUCTION_LIBRARY["accelerate_vague"]
            return {
                "text": rng.choice(entry["texts"]),
                "command_id": entry["command_id"],
                "expected_behavior": dict(entry["expected_behavior"]),
            }
        if style == "precise":
            return _build_precise_accelerate_instruction(rng, target_speed_ms=16)
        return (
            _build_precise_accelerate_instruction(rng, target_speed_ms=16)
            if rng.random() < 0.5
            else {
                "text": rng.choice(INSTRUCTION_LIBRARY["accelerate_vague"]["texts"]),
                "command_id": INSTRUCTION_LIBRARY["accelerate_vague"]["command_id"],
                "expected_behavior": dict(
                    INSTRUCTION_LIBRARY["accelerate_vague"]["expected_behavior"]
                ),
            }
        )

    entry = INSTRUCTION_LIBRARY[category]
    return {
        "text": rng.choice(entry["texts"]),
        "command_id": entry["command_id"],
        "expected_behavior": dict(entry["expected_behavior"]),
    }


def _sample_navigation_instruction_for_action(
    rng: random.Random,
    action: str,
    phrasing_mode: str,
) -> Dict[str, object]:
    entry = INSTRUCTION_LIBRARY[action]
    return {
        "text": rng.choice(_navigation_text_variants(action, phrasing_mode)),
        "command_id": entry["command_id"],
        "expected_behavior": dict(entry["expected_behavior"]),
    }


def _build_default_evaluation() -> ET.Element:
    evaluation_elem = ET.Element("evaluation")

    collision_metric = ET.SubElement(evaluation_elem, "metric", {"type": "collision_check"})
    ET.SubElement(collision_metric, "param", {"name": "expect_collision", "value": "false"})

    instruction_metric = ET.SubElement(
        evaluation_elem, "metric", {"type": "instruction_compliance"}
    )
    ET.SubElement(
        instruction_metric,
        "param",
        {"name": "compliance_threshold", "value": "0.8"},
    )

    return evaluation_elem


def _sample_accelerate_speed_ms(
    rng: random.Random,
    town: str,
    position: Tuple[float, float, float],
    speed_limit_cache: SpeedLimitMapCache,
) -> int:
    speed_limit_kmh = speed_limit_cache.get_speed_limit_kmh(town, position)
    low_kmh = max(0.0, speed_limit_kmh - 10.0)
    sampled_kmh = rng.uniform(low_kmh, speed_limit_kmh)
    return int(round(sampled_kmh / 3.6))


def _select_output_actions(
    actions: List[str],
    rng: random.Random,
) -> List[str]:
    unique_actions: List[str] = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)

    non_lane_follow_actions = [action for action in unique_actions if action != "lane_follow"]
    selected_actions = list(non_lane_follow_actions)
    if "lane_follow" in unique_actions:
        if not non_lane_follow_actions or rng.random() < OPTIONAL_LANE_FOLLOW_PROBABILITY:
            selected_actions.append("lane_follow")
    return selected_actions


def _build_two_step_instructions(
    rng: random.Random,
    accelerate_target_speed_ms: int,
    action: str,
    navigation_trigger: SelectedNavigationTrigger,
) -> ET.Element:
    instructions_elem = ET.Element("instructions")
    accelerate_template = _build_precise_accelerate_instruction(
        rng, target_speed_ms=accelerate_target_speed_ms
    )
    _append_instruction(
        instructions_elem,
        instruction_id=1,
        trigger_distance_m=0.0,
        template=accelerate_template,
        duration_meters=navigation_trigger.distance_m,
    )

    navigation_template = _sample_navigation_instruction_for_action(
        rng, action=action, phrasing_mode=navigation_trigger.phrasing_mode
    )
    _append_instruction(
        instructions_elem,
        instruction_id=2,
        trigger_distance_m=navigation_trigger.distance_m,
        template=navigation_template,
        duration_meters=-1.0,
    )
    return instructions_elem


def _build_action_route_tree(
    src_route: ET.Element,
    category: str,
    action: str,
    navigation_trigger: SelectedNavigationTrigger,
    accelerate_target_speed_ms: int,
    rng: random.Random,
) -> ET.Element:
    src_id = src_route.attrib.get("id", "unknown")
    target_root = ET.Element("routes")
    target_route = ET.SubElement(
        target_root,
        "route",
        {
            "id": f"{src_id}_LANG_{action.upper()}",
            "town": src_route.attrib.get("town", "Town12"),
            "benchmark_type": "language_following",
            "category": category,
            "disable_bg_vehicle": "true",
        },
    )

    waypoints_elem = src_route.find("waypoints")
    if waypoints_elem is None:
        raise ValueError(f"Route {src_id} has no <waypoints>.")
    target_route.append(copy.deepcopy(waypoints_elem))
    target_route.append(
        _build_two_step_instructions(
            rng=rng,
            accelerate_target_speed_ms=accelerate_target_speed_ms,
            action=action,
            navigation_trigger=navigation_trigger,
        )
    )
    target_route.append(_build_default_evaluation())
    target_route.append(_build_default_scenarios(src_route))

    weathers_elem = src_route.find("weathers")
    if weathers_elem is not None:
        target_route.append(copy.deepcopy(weathers_elem))

    _indent_xml_compat(target_root)
    return target_root


def convert_file(
    input_xml: Path,
    output_dir: Path,
    category: str,
    trigger_step_m: float,
    seed: Optional[int],
    num_instructions: int,
    instruction_style: str,
    map_cache: CarlaMapCache,
    speed_limit_cache: SpeedLimitMapCache,
) -> List[Path]:
    tree = ET.parse(input_xml)
    source_root = tree.getroot()

    if source_root.tag != "routes":
        raise ValueError(f"Expected root tag <routes>, found <{source_root.tag}>.")

    if num_instructions != 2:
        raise ValueError("--num-instructions must be 2 for accelerate->action generation.")
    _ = instruction_style  # Reserved for future prompt-style extensions.

    output_dir.mkdir(parents=True, exist_ok=True)
    base_rng = random.Random() if seed is None else random.Random(seed)
    written_paths: List[Path] = []
    routes = source_root.findall("route")
    multiple_routes = len(routes) > 1

    for route_index, src_route in enumerate(routes):
        src_id = src_route.attrib.get("id", "unknown")
        waypoints_elem = src_route.find("waypoints")
        if waypoints_elem is None:
            raise ValueError(f"Route {src_id} has no <waypoints>.")
        route_positions = _get_waypoint_positions(waypoints_elem)
        route_town = src_route.attrib.get("town")
        if route_town is None:
            raise ValueError(f"Route {src_id} has no town attribute.")

        route_rng_seed = (
            f"{seed}:{input_xml.stem}:{src_id}:{route_index}"
            if seed is not None
            else f"{base_rng.random()}:{input_xml.stem}:{src_id}:{route_index}"
        )
        route_rng = random.Random(route_rng_seed)
        carla_map = map_cache.get_map(route_town)
        navigation_trigger, samples = select_navigation_trigger(
            route_positions=route_positions,
            trigger_step_m=trigger_step_m,
            carla_map=carla_map,
            rng=route_rng,
        )
        selected_sample = samples[navigation_trigger.sample_index]
        output_actions = _select_output_actions(list(selected_sample.actions), route_rng)
        if not output_actions:
            print(f"[INFO] Route {src_id}: no retained actions, skipping.")
            continue

        accelerate_target_speed_ms = _sample_accelerate_speed_ms(
            route_rng,
            town=route_town,
            position=navigation_trigger.position,
            speed_limit_cache=speed_limit_cache,
        )

        stem_prefix = input_xml.stem if not multiple_routes else f"{input_xml.stem}_{src_id}"
        for action in output_actions:
            route_tree = _build_action_route_tree(
                src_route=src_route,
                category=category,
                action=action,
                navigation_trigger=navigation_trigger,
                accelerate_target_speed_ms=accelerate_target_speed_ms,
                rng=route_rng,
            )
            output_path = output_dir / f"{stem_prefix}_language_{action}.xml"
            ET.ElementTree(route_tree).write(
                output_path, encoding="UTF-8", xml_declaration=True
            )
            written_paths.append(output_path)

    return written_paths


def main() -> None:
    args = parse_args()
    input_path = args.input_xml.resolve()

    xodr_roots = (
        None if args.xodr_root is None else [path.expanduser().resolve() for path in args.xodr_root]
    )
    map_cache = CarlaMapCache(xodr_search_roots=xodr_roots)
    speed_limit_cache = SpeedLimitMapCache()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = (
        repo_root / "leaderboard" / "data" / "language_benchmark" / "instruction_following"
    )

    if input_path.is_file():
        if args.output is None:
            output_dir = default_output_dir
        else:
            output_dir = args.output.resolve()
            if output_dir.suffix:
                raise ValueError("--output must be a directory path for per-action generation.")

        written_paths = convert_file(
            input_xml=input_path,
            output_dir=output_dir,
            category=args.category,
            trigger_step_m=args.trigger_step_m,
            seed=args.seed,
            num_instructions=args.num_instructions,
            instruction_style=args.instruction_style,
            map_cache=map_cache,
            speed_limit_cache=speed_limit_cache,
        )
        if not written_paths:
            print(f"No XML files generated for: {input_path}")
        else:
            for output_xml in written_paths:
                print(f"Generated language benchmark XML: {output_xml}")
        return

    if not input_path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {input_path}")

    input_xml_files = sorted(input_path.glob("*.xml"))
    if not input_xml_files:
        raise ValueError(f"No XML files found in input directory: {input_path}")

    if args.output is None:
        output_dir = default_output_dir
    else:
        output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_xml in input_xml_files:
        written_paths = convert_file(
            input_xml=input_xml,
            output_dir=output_dir,
            category=args.category,
            trigger_step_m=args.trigger_step_m,
            seed=args.seed,
            num_instructions=args.num_instructions,
            instruction_style=args.instruction_style,
            map_cache=map_cache,
            speed_limit_cache=speed_limit_cache,
        )
        if not written_paths:
            print(f"No XML files generated for: {input_xml}")
            continue
        for output_xml in written_paths:
            print(f"Generated language benchmark XML: {output_xml}")


if __name__ == "__main__":
    main()
