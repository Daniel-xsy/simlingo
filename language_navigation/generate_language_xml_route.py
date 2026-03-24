#!/usr/bin/env python3
"""
Generate language-navigation XML by rebuilding the route from the Bench2Drive
start point and branching the GT route after the sampled trigger.
"""

import argparse
import copy
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import carla
except ImportError as exc:
    carla = None
    CARLA_IMPORT_ERROR = exc
else:
    CARLA_IMPORT_ERROR = None

try:
    from language_navigation.utils import (
        ASSUMED_ACCELERATION_MS2,
        OpenDriveSpeedLimitResolver,
        OPTIONAL_LANE_FOLLOW_PROBABILITY,
        CarlaMapCache,
        _append_instruction,
        _build_actionable_navigation_categories,
        _build_default_evaluation,
        _build_lane_follow_instruction,
        _build_precise_accelerate_instruction,
        _can_change_lane,
        _compute_turn_category,
        _fit_accelerate_instruction_to_window,
        _get_waypoint_positions,
        _indent_xml_compat,
        _is_same_direction_lane,
        _normalize_yaw_delta_deg,
        _position_at_distance,
        _route_length_m,
        _sample_accelerate_speed_ms,
        _sample_navigation_instruction_for_action,
    )
except ImportError:
    from utils import (  # type: ignore
        ASSUMED_ACCELERATION_MS2,
        OpenDriveSpeedLimitResolver,
        OPTIONAL_LANE_FOLLOW_PROBABILITY,
        CarlaMapCache,
        _append_instruction,
        _build_actionable_navigation_categories,
        _build_default_evaluation,
        _build_lane_follow_instruction,
        _build_precise_accelerate_instruction,
        _can_change_lane,
        _compute_turn_category,
        _fit_accelerate_instruction_to_window,
        _get_waypoint_positions,
        _indent_xml_compat,
        _is_same_direction_lane,
        _normalize_yaw_delta_deg,
        _position_at_distance,
        _route_length_m,
        _sample_accelerate_speed_ms,
        _sample_navigation_instruction_for_action,
    )


Position3D = Tuple[float, float, float]
LANE_CHANGE_SETTLE_DISTANCE_M = 15.0


@dataclass(frozen=True)
class RebuiltRoute:
    waypoints: Tuple["carla.Waypoint", ...]
    positions: Tuple[Position3D, ...]
    cumulative_distances_m: Tuple[float, ...]
    total_length_m: float


@dataclass(frozen=True)
class InstructionSpec:
    action: str
    trigger_distance_m: float  # cumulative distance from route start
    duration_m: float          # -1 for last instruction
    phrasing_mode: str         # "approach" or "at_junction"


@dataclass
class InstructionStep:
    template: Dict[str, object]
    trigger_distance_m: float
    duration_m: float


@dataclass
class ActionSuffixResult:
    waypoints: List["carla.Waypoint"]
    action_end_index: int  # index in waypoints where action completes


@dataclass(frozen=True)
class RebuiltActionabilitySample:
    index: int
    distance_m: float
    position: Position3D
    actions: Tuple[str, ...]
    scored_actions: Tuple[str, ...]
    score: int
    is_junction: bool


@dataclass(frozen=True)
class RebuiltTrigger:
    index: int
    distance_m: float
    position: Position3D
    actions: Tuple[str, ...]
    scored_actions: Tuple[str, ...]
    is_junction: bool
    phrasing_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create language benchmark XMLs from Bench2Drive starts by "
            "rebuilding a 5 m route and branching GT after the trigger."
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
            "leaderboard/data/language_benchmark/instruction_following_rebuilt/"
        ),
    )
    parser.add_argument(
        "--category",
        default="instruction_following",
        help="Language benchmark category to set on route elements.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible route and text sampling.",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=130.0,
        help="Maximum route length to rebuild from the start point.",
    )
    parser.add_argument(
        "--route-step-m",
        type=float,
        default=3.0,
        help="Sampling step in meters for rebuilt output waypoints.",
    )
    parser.add_argument(
        "--lane-change-prep-m",
        type=float,
        default=10.0,
        help="Forward distance before hopping to the adjacent lane.",
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
    parser.add_argument(
        "--max-chain-depth",
        type=int,
        default=3,
        help="Maximum number of chained navigation instructions per route.",
    )
    parser.add_argument(
        "--min-chain-trigger-gap-m",
        type=float,
        default=30.0,
        help="Minimum distance in meters between consecutive chained instruction triggers.",
    )
    parser.add_argument(
        "--force-all-green-traffic-lights",
        action="store_true",
        help=(
            "If set, add force_all_green_traffic_lights=\"true\" to generated "
            "route elements."
        ),
    )
    return parser.parse_args()


def _require_carla() -> None:
    if carla is None:
        raise RuntimeError(
            "CARLA Python API is required for rebuilt route generation."
        ) from CARLA_IMPORT_ERROR


def _waypoint_position(waypoint: "carla.Waypoint") -> Position3D:
    location = waypoint.transform.location
    return (float(location.x), float(location.y), float(location.z))


def _distance_positions(a: Position3D, b: Position3D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _distance_waypoints(a: "carla.Waypoint", b: "carla.Waypoint") -> float:
    return a.transform.location.distance(b.transform.location)


def _append_waypoint_if_new(route_waypoints: List["carla.Waypoint"], waypoint: "carla.Waypoint") -> None:
    if not route_waypoints:
        route_waypoints.append(waypoint)
        return
    prev = route_waypoints[-1]
    if _distance_waypoints(prev, waypoint) <= 1e-3:
        return
    route_waypoints.append(waypoint)


def _follow_score(current_waypoint: "carla.Waypoint", candidate: "carla.Waypoint") -> Tuple[float, int, int, float]:
    yaw_delta = abs(
        _normalize_yaw_delta_deg(
            candidate.transform.rotation.yaw - current_waypoint.transform.rotation.yaw
        )
    )
    same_road_penalty = 0 if candidate.road_id == current_waypoint.road_id else 1
    lane_delta = abs(abs(candidate.lane_id) - abs(current_waypoint.lane_id))
    return (
        yaw_delta,
        same_road_penalty,
        lane_delta,
        _distance_waypoints(current_waypoint, candidate),
    )


def _select_follow_road_successor(
    current_waypoint: "carla.Waypoint",
    candidates: Sequence["carla.Waypoint"],
) -> "carla.Waypoint":
    if not candidates:
        raise ValueError("Cannot select follow-road successor from empty candidate list.")
    return min(candidates, key=lambda waypoint: _follow_score(current_waypoint, waypoint))


def _trace_candidate_turn(
    approach_waypoint: "carla.Waypoint",
    candidate: "carla.Waypoint",
    step_m: float,
    max_scan_distance_m: float = 40.0,
) -> str:
    traversed = _distance_waypoints(approach_waypoint, candidate)
    branch_waypoint = candidate

    while traversed < max_scan_distance_m:
        next_candidates = branch_waypoint.next(step_m)
        if not next_candidates:
            break
        next_waypoint = _select_follow_road_successor(branch_waypoint, next_candidates)
        traversed += _distance_waypoints(branch_waypoint, next_waypoint)
        branch_waypoint = next_waypoint
        if not branch_waypoint.is_junction and traversed >= step_m:
            break

    return _compute_turn_category(approach_waypoint, branch_waypoint)


def _select_turn_successor(
    approach_waypoint: "carla.Waypoint",
    current_waypoint: "carla.Waypoint",
    candidates: Sequence["carla.Waypoint"],
    desired_action: str,
    step_m: float,
) -> Optional["carla.Waypoint"]:
    matching = [
        candidate
        for candidate in candidates
        if _trace_candidate_turn(approach_waypoint, candidate, step_m) == desired_action
    ]
    if not matching:
        return None
    return min(matching, key=lambda waypoint: _follow_score(current_waypoint, waypoint))


def _compute_cumulative_distances(positions: Sequence[Position3D]) -> Tuple[float, ...]:
    cumulative = [0.0]
    for idx in range(1, len(positions)):
        cumulative.append(cumulative[-1] + _distance_positions(positions[idx - 1], positions[idx]))
    return tuple(cumulative)


def _resample_positions(positions: Sequence[Position3D], step_m: float) -> List[Position3D]:
    if not positions:
        return []

    deduped = [positions[0]]
    for position in positions[1:]:
        if _distance_positions(deduped[-1], position) > 1e-3:
            deduped.append(position)

    if len(deduped) == 1:
        return deduped

    total_length = _route_length_m(deduped)
    distances = [0.0]
    current = step_m
    while current < total_length - 1e-6:
        distances.append(current)
        current += step_m
    if total_length > 1e-6:
        distances.append(total_length)

    return [_position_at_distance(deduped, distance_m) for distance_m in distances]


def _finalize_route(
    route_waypoints: Sequence["carla.Waypoint"],
    step_m: float,
    resample: bool = False,
) -> RebuiltRoute:
    raw_positions = [_waypoint_position(waypoint) for waypoint in route_waypoints]
    positions = _resample_positions(raw_positions, step_m) if resample else list(raw_positions)
    cumulative_distances = _compute_cumulative_distances(positions)
    total_length = cumulative_distances[-1] if cumulative_distances else 0.0
    return RebuiltRoute(
        waypoints=tuple(route_waypoints),
        positions=tuple(positions),
        cumulative_distances_m=cumulative_distances,
        total_length_m=total_length,
    )


def _rebuild_follow_route(
    start_waypoint: "carla.Waypoint",
    max_distance_m: float,
    step_m: float,
) -> RebuiltRoute:
    route_waypoints = [start_waypoint]
    current_waypoint = start_waypoint
    traveled = 0.0

    while traveled + 1e-6 < max_distance_m:
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates:
            break
        next_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)
        segment_distance = _distance_waypoints(current_waypoint, next_waypoint)
        if segment_distance <= 1e-3:
            break
        traveled += segment_distance
        _append_waypoint_if_new(route_waypoints, next_waypoint)
        current_waypoint = next_waypoint

    return _finalize_route(route_waypoints, step_m)


def _rebuild_turn_suffix(
    start_waypoint: "carla.Waypoint",
    desired_action: str,
    max_distance_m: float,
    step_m: float,
) -> Optional[ActionSuffixResult]:
    route_waypoints = [start_waypoint]
    current_waypoint = start_waypoint
    traveled = 0.0
    approach_waypoint: Optional["carla.Waypoint"] = None
    entered_junction = False
    first_exit_waypoint: Optional["carla.Waypoint"] = None
    action_end_index: Optional[int] = None
    branch_selected = False

    while traveled + 1e-6 < max_distance_m:
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates:
            break

        near_junction = current_waypoint.is_junction or any(
            candidate.is_junction for candidate in next_candidates
        )
        if near_junction and approach_waypoint is None:
            approach_waypoint = current_waypoint

        if near_junction and len(next_candidates) > 1 and not branch_selected:
            selected_waypoint = _select_turn_successor(
                approach_waypoint or current_waypoint,
                current_waypoint,
                next_candidates,
                desired_action,
                step_m,
            )
            if selected_waypoint is None:
                return None
            branch_selected = True
        else:
            selected_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)

        segment_distance = _distance_waypoints(current_waypoint, selected_waypoint)
        if segment_distance <= 1e-3:
            break
        if traveled + segment_distance > max_distance_m + 1e-6:
            break

        traveled += segment_distance
        previous_waypoint = current_waypoint
        current_waypoint = selected_waypoint
        _append_waypoint_if_new(route_waypoints, current_waypoint)

        if current_waypoint.is_junction:
            entered_junction = True
        if entered_junction and not current_waypoint.is_junction and previous_waypoint.is_junction:
            first_exit_waypoint = current_waypoint
            action_end_index = len(route_waypoints) - 1
            break

    if approach_waypoint is None:
        return None

    validation_waypoint = first_exit_waypoint or route_waypoints[-1]
    if _compute_turn_category(approach_waypoint, validation_waypoint) != desired_action:
        return None

    if action_end_index is None:
        action_end_index = len(route_waypoints) - 1

    while traveled + 1e-6 < max_distance_m:
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates:
            break
        selected_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)
        segment_distance = _distance_waypoints(current_waypoint, selected_waypoint)
        if segment_distance <= 1e-3 or traveled + segment_distance > max_distance_m + 1e-6:
            break
        traveled += segment_distance
        current_waypoint = selected_waypoint
        _append_waypoint_if_new(route_waypoints, current_waypoint)

    return ActionSuffixResult(
        waypoints=route_waypoints,
        action_end_index=action_end_index,
    )


def _rebuild_lane_change_suffix(
    start_waypoint: "carla.Waypoint",
    direction: str,
    max_distance_m: float,
    step_m: float,
    prep_distance_m: float,
) -> Optional[ActionSuffixResult]:
    if not _can_change_lane(start_waypoint, direction):
        return None

    route_waypoints = [start_waypoint]
    current_waypoint = start_waypoint
    traveled = 0.0
    prep_target = min(prep_distance_m, max_distance_m)

    while traveled + 1e-6 < prep_target:
        if current_waypoint.is_junction:
            return None
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates or any(candidate.is_junction for candidate in next_candidates):
            return None
        next_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)
        segment_distance = _distance_waypoints(current_waypoint, next_waypoint)
        if segment_distance <= 1e-3 or traveled + segment_distance > max_distance_m + 1e-6:
            break
        traveled += segment_distance
        current_waypoint = next_waypoint
        _append_waypoint_if_new(route_waypoints, current_waypoint)

    if not _can_change_lane(current_waypoint, direction):
        return None

    adjacent_waypoint = (
        current_waypoint.get_left_lane()
        if direction == "left"
        else current_waypoint.get_right_lane()
    )
    if adjacent_waypoint is None:
        return None
    if adjacent_waypoint.lane_type != carla.LaneType.Driving:
        return None
    if not _is_same_direction_lane(current_waypoint, adjacent_waypoint):
        return None

    lane_change_distance = _distance_waypoints(current_waypoint, adjacent_waypoint)
    if traveled + lane_change_distance > max_distance_m + 1e-6:
        return None

    traveled += lane_change_distance
    current_waypoint = adjacent_waypoint
    _append_waypoint_if_new(route_waypoints, current_waypoint)
    post_change_traveled = 0.0
    action_end_index: Optional[int] = None

    while traveled + 1e-6 < max_distance_m:
        next_candidates = current_waypoint.next(step_m)
        if not next_candidates:
            break
        next_waypoint = _select_follow_road_successor(current_waypoint, next_candidates)
        segment_distance = _distance_waypoints(current_waypoint, next_waypoint)
        if segment_distance <= 1e-3 or traveled + segment_distance > max_distance_m + 1e-6:
            break
        traveled += segment_distance
        post_change_traveled += segment_distance
        current_waypoint = next_waypoint
        _append_waypoint_if_new(route_waypoints, current_waypoint)
        if action_end_index is None and post_change_traveled >= LANE_CHANGE_SETTLE_DISTANCE_M - 1e-6:
            action_end_index = len(route_waypoints) - 1

    if action_end_index is None:
        action_end_index = len(route_waypoints) - 1

    return ActionSuffixResult(
        waypoints=route_waypoints,
        action_end_index=action_end_index,
    )


def _rebuild_action_suffix(
    start_waypoint: "carla.Waypoint",
    action: str,
    max_distance_m: float,
    step_m: float,
    lane_change_prep_m: float,
) -> Optional[ActionSuffixResult]:
    if action == "lane_follow":
        follow_route = _rebuild_follow_route(start_waypoint, max_distance_m, step_m)
        return ActionSuffixResult(
            waypoints=list(follow_route.waypoints),
            action_end_index=len(follow_route.waypoints) - 1,
        )
    if action == "lane_change_left":
        return _rebuild_lane_change_suffix(
            start_waypoint, "left", max_distance_m, step_m, lane_change_prep_m
        )
    if action == "lane_change_right":
        return _rebuild_lane_change_suffix(
            start_waypoint, "right", max_distance_m, step_m, lane_change_prep_m
        )
    if action in ("turn_left", "turn_right", "turn_straight"):
        return _rebuild_turn_suffix(start_waypoint, action, max_distance_m, step_m)
    raise ValueError(f"Unsupported action: {action}")


def _build_actionability_samples(
    route: RebuiltRoute,
    carla_map: "carla.Map",
) -> List[RebuiltActionabilitySample]:
    samples: List[RebuiltActionabilitySample] = []
    for index, position in enumerate(route.positions):
        distance_m = route.cumulative_distances_m[index]
        remaining_distance_m = max(route.total_length_m - distance_m, 0.0)
        actions = _build_actionable_navigation_categories(
            carla_map,
            position,
            max_turn_scan_distance_m=remaining_distance_m,
        )
        scored_actions = tuple(action for action in actions if action != "lane_follow")
        waypoint = carla_map.get_waypoint(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        samples.append(
            RebuiltActionabilitySample(
                index=index,
                distance_m=distance_m,
                position=position,
                actions=tuple(actions),
                scored_actions=scored_actions,
                score=len(scored_actions),
                is_junction=bool(waypoint.is_junction) if waypoint is not None else False,
            )
        )
    return samples


def _find_first_junction_distance_m(route: RebuiltRoute) -> Optional[float]:
    """Return cumulative distance to the first junction entrance, or None if no junction."""
    for i, wp in enumerate(route.waypoints):
        if wp.is_junction:
            return route.cumulative_distances_m[i]
    return None


def _select_output_actions(actions: Sequence[str], rng: random.Random) -> List[str]:
    unique_actions: List[str] = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)

    retained = [action for action in unique_actions if action != "lane_follow"]
    if "lane_follow" in unique_actions:
        if not retained or rng.random() < OPTIONAL_LANE_FOLLOW_PROBABILITY:
            retained.append("lane_follow")
    return retained


def _select_trigger(
    samples: Sequence[RebuiltActionabilitySample],
    min_trigger_distance_m: float = 5.0,
) -> Optional[RebuiltTrigger]:
    valid_samples = [sample for sample in samples if sample.distance_m >= min_trigger_distance_m]
    if not valid_samples:
        return None

    best_score = max(sample.score for sample in valid_samples)
    winner = next(sample for sample in valid_samples if sample.score == best_score)

    return RebuiltTrigger(
        index=winner.index,
        distance_m=winner.distance_m,
        position=winner.position,
        actions=winner.actions,
        scored_actions=winner.scored_actions,
        is_junction=winner.is_junction,
        phrasing_mode="at_junction" if winner.is_junction else "approach",
    )


def _build_instruction_chain(
    start_waypoint: "carla.Waypoint",
    carla_map: "carla.Map",
    remaining_distance_m: float,
    step_m: float,
    lane_change_prep_m: float,
    min_chain_trigger_gap_m: float,
    rng: random.Random,
    cumulative_distance_m: float = 0.0,
    previous_trigger_distance_m: Optional[float] = None,
    max_depth: int = 5,
) -> Tuple[List["carla.Waypoint"], List[InstructionSpec]]:
    """
    Recursively build waypoints and instruction specs.

    Returns:
        all_waypoints: chained waypoints from start_waypoint
        specs: list of InstructionSpec for each navigation action
    """
    # Build a follow route from start_waypoint
    follow_route = _rebuild_follow_route(start_waypoint, remaining_distance_m, step_m)
    if len(follow_route.waypoints) < 2:
        return list(follow_route.waypoints), []

    # Compute actionability samples on this follow route
    samples = _build_actionability_samples(follow_route, carla_map)
    if previous_trigger_distance_m is not None:
        effective_min_trigger_distance_m = max(
            0.0,
            previous_trigger_distance_m + min_chain_trigger_gap_m - cumulative_distance_m,
        )
    else:
        effective_min_trigger_distance_m = 0.0
    eligible_samples = [
        sample
        for sample in samples
        if sample.distance_m >= max(0.0, effective_min_trigger_distance_m)
    ]
    if not eligible_samples:
        return list(follow_route.waypoints), []
    trigger = _select_trigger(
        eligible_samples,
        min_trigger_distance_m=0.0,
    )
    if trigger is None:
        return list(follow_route.waypoints), []

    # Check for interesting (non-lane_follow) actions
    interesting_actions = [a for a in trigger.actions if a != "lane_follow"]

    if not interesting_actions or max_depth <= 0:
        return list(follow_route.waypoints), []

    # Pick one non-lane_follow action
    action = rng.choice(interesting_actions)

    # Build action suffix from trigger waypoint
    trigger_waypoint = follow_route.waypoints[trigger.index]
    suffix_remaining_m = max(0.0, remaining_distance_m - trigger.distance_m)
    suffix_result = _rebuild_action_suffix(
        start_waypoint=trigger_waypoint,
        action=action,
        max_distance_m=suffix_remaining_m,
        step_m=step_m,
        lane_change_prep_m=lane_change_prep_m,
    )
    if suffix_result is None:
        # Action failed; terminate the chain instead of inserting a dense fallback prompt.
        return list(follow_route.waypoints), []

    # Compute distances within suffix to find action_end distance
    action_end_waypoint = suffix_result.waypoints[suffix_result.action_end_index]
    # Distance from trigger to action end
    action_segment_positions = [
        _waypoint_position(wp)
        for wp in suffix_result.waypoints[: suffix_result.action_end_index + 1]
    ]
    action_segment_distances = _compute_cumulative_distances(action_segment_positions)
    trigger_to_action_end_m = action_segment_distances[-1] if action_segment_distances else 0.0

    # Build prefix waypoints (start -> trigger) + action segment (trigger -> action_end)
    prefix_waypoints = list(follow_route.waypoints[: trigger.index + 1])
    action_waypoints = list(suffix_result.waypoints[1: suffix_result.action_end_index + 1])

    # Cumulative distances for this instruction
    abs_trigger_distance_m = cumulative_distance_m + trigger.distance_m
    abs_action_end_distance_m = abs_trigger_distance_m + trigger_to_action_end_m

    # Remaining distance after action end
    new_remaining_m = remaining_distance_m - trigger.distance_m - trigger_to_action_end_m

    # Build the current instruction spec (duration set later depending on recursion)
    current_spec = InstructionSpec(
        action=action,
        trigger_distance_m=abs_trigger_distance_m,
        duration_m=trigger_to_action_end_m,  # will be overridden to -1 if last
        phrasing_mode=trigger.phrasing_mode,
    )

    # Try to recurse if enough remaining distance
    if new_remaining_m > 0.0 and max_depth > 1:
        tail_waypoints, tail_specs = _build_instruction_chain(
            start_waypoint=action_end_waypoint,
            carla_map=carla_map,
            remaining_distance_m=new_remaining_m,
            step_m=step_m,
            lane_change_prep_m=lane_change_prep_m,
            min_chain_trigger_gap_m=min_chain_trigger_gap_m,
            rng=rng,
            cumulative_distance_m=abs_action_end_distance_m,
            previous_trigger_distance_m=current_spec.trigger_distance_m,
            max_depth=max_depth - 1,
        )

        # Merge waypoints: prefix + action_segment + tail (skip first of tail, it's action_end)
        merged_waypoints = prefix_waypoints + action_waypoints
        for wp in tail_waypoints[1:]:
            _append_waypoint_if_new(merged_waypoints, wp)

        return merged_waypoints, [current_spec] + tail_specs
    else:
        # Last instruction: set duration to -1
        final_spec = InstructionSpec(
            action=current_spec.action,
            trigger_distance_m=current_spec.trigger_distance_m,
            duration_m=-1.0,
            phrasing_mode=current_spec.phrasing_mode,
        )
        # Use the full suffix waypoints (includes post-action follow)
        merged_waypoints = prefix_waypoints
        for wp in suffix_result.waypoints[1:]:
            _append_waypoint_if_new(merged_waypoints, wp)
        return merged_waypoints, [final_spec]


def _build_waypoints_element(positions: Sequence[Position3D]) -> ET.Element:
    if not positions:
        raise ValueError("Cannot build <waypoints> from an empty route.")

    waypoints_elem = ET.Element("waypoints")
    for x, y, z in positions:
        ET.SubElement(
            waypoints_elem,
            "position",
            {
                "x": f"{x:.1f}",
                "y": f"{y:.1f}",
                "z": f"{z:.1f}",
            },
        )
    return waypoints_elem


def _build_default_scenarios_from_positions(positions: Sequence[Position3D]) -> ET.Element:
    if not positions:
        raise ValueError("Cannot build scenarios from an empty route.")

    x0, y0, z0 = positions[0]
    yaw_deg = 0.0
    if len(positions) >= 2:
        x1, y1, _ = positions[1]
        dy = y1 - y0
        dx = x1 - x0
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            yaw_deg = math.degrees(math.atan2(dy, dx))

    scenarios_elem = ET.Element("scenarios")
    scenario_elem = ET.SubElement(
        scenarios_elem,
        "scenario",
        {"name": "FreeRide_1", "type": "FreeRide"},
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


def _build_instruction_steps(
    rng: random.Random,
    instruction_specs: List[InstructionSpec],
    accel_target_speed_ms: int,
    accel_duration_m: float,
) -> List[InstructionStep]:
    if not instruction_specs:
        return []

    instruction_steps: List[InstructionStep] = []

    # Always place speed instruction at distance 0
    speed_template = _build_precise_accelerate_instruction(
        rng, target_speed_ms=accel_target_speed_ms
    )
    instruction_steps.append(
        InstructionStep(
            template=speed_template,
            trigger_distance_m=0.0,
            duration_m=accel_duration_m,
        )
    )

    # Track where the previous instruction ends
    prev_end_m = accel_duration_m

    for spec in instruction_specs:
        # Insert lane_follow filler if there's a gap before this nav trigger
        gap_m = spec.trigger_distance_m - prev_end_m
        if gap_m > 10.0:
            instruction_steps.append(
                InstructionStep(
                    template=_build_lane_follow_instruction(rng),
                    trigger_distance_m=prev_end_m,
                    duration_m=gap_m,
                )
            )

        # Insert the navigation instruction
        instruction_steps.append(
            InstructionStep(
                template=_sample_navigation_instruction_for_action(
                    rng,
                    action=spec.action,
                    phrasing_mode=spec.phrasing_mode,
                ),
                trigger_distance_m=spec.trigger_distance_m,
                duration_m=spec.duration_m,
            )
        )

        # Update prev_end_m (duration_m=-1 means last instruction, no end)
        if spec.duration_m > 0:
            prev_end_m = spec.trigger_distance_m + spec.duration_m

    return instruction_steps


def _build_instructions_from_steps(
    instruction_steps: Sequence[InstructionStep],
) -> ET.Element:
    instructions_elem = ET.Element("instructions")
    for instruction_id, step in enumerate(instruction_steps, start=1):
        _append_instruction(
            instructions_elem,
            instruction_id=instruction_id,
            trigger_distance_m=step.trigger_distance_m,
            template=step.template,
            duration_meters=step.duration_m,
        )
    return instructions_elem


def _build_route_instructions(
    rng: random.Random,
    accelerate_target_speed_ms: int,
    trigger: RebuiltTrigger,
    action: str,
) -> ET.Element:
    instructions_elem = ET.Element("instructions")
    fitted_speed_ms, accelerate_duration_m = _fit_accelerate_instruction_to_window(
        accelerate_target_speed_ms,
        trigger.distance_m,
        acceleration_ms2=ASSUMED_ACCELERATION_MS2,
    )
    accelerate_template = _build_precise_accelerate_instruction(
        rng, target_speed_ms=fitted_speed_ms
    )
    _append_instruction(
        instructions_elem,
        instruction_id=1,
        trigger_distance_m=0.0,
        template=accelerate_template,
        duration_meters=accelerate_duration_m,
    )

    navigation_template = _sample_navigation_instruction_for_action(
        rng, action=action, phrasing_mode=trigger.phrasing_mode
    )
    _append_instruction(
        instructions_elem,
        instruction_id=2,
        trigger_distance_m=trigger.distance_m,
        template=navigation_template,
        duration_meters=-1.0,
    )
    return instructions_elem


def _build_chained_route_instructions(
    rng: random.Random,
    accelerate_target_speed_ms: int,
    first_trigger_distance_m: float,
    instruction_specs: List[InstructionSpec],
    use_lane_follow_start: bool = False,
    post_action_accelerate_speed_ms: Optional[int] = None,
    post_action_accelerate_trigger_m: Optional[float] = None,
    post_action_accelerate_duration_m: Optional[float] = None,
) -> ET.Element:
    instructions_elem = ET.Element("instructions")
    next_id = 1

    if use_lane_follow_start:
        # Instruction 1: lane_follow until first action
        lane_follow_template = _build_lane_follow_instruction(rng)
        _append_instruction(
            instructions_elem,
            instruction_id=next_id,
            trigger_distance_m=0.0,
            template=lane_follow_template,
            duration_meters=first_trigger_distance_m,
        )
        next_id += 1

        # Instruction 2: first nav action
        first_spec = instruction_specs[0]
        nav_template = _sample_navigation_instruction_for_action(
            rng, action=first_spec.action, phrasing_mode=first_spec.phrasing_mode
        )
        _append_instruction(
            instructions_elem,
            instruction_id=next_id,
            trigger_distance_m=first_spec.trigger_distance_m,
            template=nav_template,
            duration_meters=first_spec.duration_m,
        )
        next_id += 1

        # Instruction 3 (optional): post-action accelerate
        if (
            post_action_accelerate_speed_ms is not None
            and post_action_accelerate_trigger_m is not None
            and post_action_accelerate_duration_m is not None
            and first_spec.duration_m > 0
        ):
            fitted_speed_ms, fitted_duration_m = _fit_accelerate_instruction_to_window(
                post_action_accelerate_speed_ms,
                post_action_accelerate_duration_m,
                acceleration_ms2=ASSUMED_ACCELERATION_MS2,
            )
            accel_template = _build_precise_accelerate_instruction(
                rng, target_speed_ms=fitted_speed_ms
            )
            _append_instruction(
                instructions_elem,
                instruction_id=next_id,
                trigger_distance_m=post_action_accelerate_trigger_m,
                template=accel_template,
                duration_meters=fitted_duration_m,
            )
            next_id += 1

        # Remaining nav actions from instruction_specs[1:]
        for spec in instruction_specs[1:]:
            nav_template = _sample_navigation_instruction_for_action(
                rng, action=spec.action, phrasing_mode=spec.phrasing_mode
            )
            _append_instruction(
                instructions_elem,
                instruction_id=next_id,
                trigger_distance_m=spec.trigger_distance_m,
                template=nav_template,
                duration_meters=spec.duration_m,
            )
            next_id += 1
    else:
        # Normal path: accelerate first
        fitted_speed_ms, accelerate_duration_m = _fit_accelerate_instruction_to_window(
            accelerate_target_speed_ms,
            first_trigger_distance_m,
            acceleration_ms2=ASSUMED_ACCELERATION_MS2,
        )
        accelerate_template = _build_precise_accelerate_instruction(
            rng, target_speed_ms=fitted_speed_ms
        )
        _append_instruction(
            instructions_elem,
            instruction_id=next_id,
            trigger_distance_m=0.0,
            template=accelerate_template,
            duration_meters=accelerate_duration_m,
        )
        next_id += 1

        # Instructions 2..N: navigation actions
        for spec in instruction_specs:
            nav_template = _sample_navigation_instruction_for_action(
                rng, action=spec.action, phrasing_mode=spec.phrasing_mode
            )
            _append_instruction(
                instructions_elem,
                instruction_id=next_id,
                trigger_distance_m=spec.trigger_distance_m,
                template=nav_template,
                duration_meters=spec.duration_m,
            )
            next_id += 1

    return instructions_elem


def _build_action_route_tree(
    src_route: ET.Element,
    category: str,
    action: str,
    final_route: RebuiltRoute,
    accelerate_target_speed_ms: int,
    rng: random.Random,
    force_all_green_traffic_lights: bool,
    instruction_steps: Optional[List[InstructionStep]] = None,
    instruction_specs: Optional[List[InstructionSpec]] = None,
    first_trigger_distance_m: Optional[float] = None,
    trigger: Optional[RebuiltTrigger] = None,
    use_lane_follow_start: bool = False,
    post_action_accelerate_speed_ms: Optional[int] = None,
    post_action_accelerate_trigger_m: Optional[float] = None,
    post_action_accelerate_duration_m: Optional[float] = None,
) -> ET.Element:
    src_id = src_route.attrib.get("id", "unknown")
    root = ET.Element("routes")
    route_attrib = {
        "id": f"{src_id}_LANG_REBUILT_{action.upper()}",
        "town": src_route.attrib.get("town", "Town12"),
        "benchmark_type": "language_following",
        "category": category,
        "disable_bg_vehicle": "true",
    }
    if force_all_green_traffic_lights:
        route_attrib["force_all_green_traffic_lights"] = "true"
    target_route = ET.SubElement(
        root,
        "route",
        route_attrib,
    )

    target_route.append(_build_waypoints_element(final_route.positions))

    if instruction_steps is not None:
        target_route.append(_build_instructions_from_steps(instruction_steps))
    elif instruction_specs is not None and first_trigger_distance_m is not None:
        target_route.append(
            _build_chained_route_instructions(
                rng=rng,
                accelerate_target_speed_ms=accelerate_target_speed_ms,
                first_trigger_distance_m=first_trigger_distance_m,
                instruction_specs=instruction_specs,
                use_lane_follow_start=use_lane_follow_start,
                post_action_accelerate_speed_ms=post_action_accelerate_speed_ms,
                post_action_accelerate_trigger_m=post_action_accelerate_trigger_m,
                post_action_accelerate_duration_m=post_action_accelerate_duration_m,
            )
        )
    elif trigger is not None:
        target_route.append(
            _build_route_instructions(
                rng=rng,
                accelerate_target_speed_ms=accelerate_target_speed_ms,
                trigger=trigger,
                action=action,
            )
        )
    else:
        raise ValueError("Must provide either instruction_specs or trigger.")

    target_route.append(_build_default_evaluation())
    target_route.append(_build_default_scenarios_from_positions(final_route.positions))

    weathers_elem = src_route.find("weathers")
    if weathers_elem is not None:
        target_route.append(copy.deepcopy(weathers_elem))

    _indent_xml_compat(root)
    return root


def _resolve_start_waypoint(
    positions: Sequence[Position3D],
    carla_map: "carla.Map",
) -> "carla.Waypoint":
    if not positions:
        raise ValueError("Bench2Drive route has no start waypoint.")
    start = positions[0]
    waypoint = carla_map.get_waypoint(
        carla.Location(x=start[0], y=start[1], z=start[2]),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise ValueError(f"Could not project start point {start} to a driving waypoint.")
    return waypoint


def convert_file(
    input_xml: Path,
    output_dir: Path,
    category: str,
    seed: Optional[int],
    max_distance_m: float,
    route_step_m: float,
    lane_change_prep_m: float,
    map_cache: CarlaMapCache,
    speed_limit_resolver: OpenDriveSpeedLimitResolver,
    force_all_green_traffic_lights: bool,
    max_chain_depth: int = 5,
    min_chain_trigger_gap_m: float = 25.0,
) -> List[Path]:
    tree = ET.parse(input_xml)
    source_root = tree.getroot()
    if source_root.tag != "routes":
        raise ValueError(f"Expected root tag <routes>, found <{source_root.tag}>.")

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

        source_positions = _get_waypoint_positions(waypoints_elem)
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

        start_waypoint = _resolve_start_waypoint(source_positions, carla_map)

        # Compute acceleration at start.  Cap the speed so that the
        # acceleration distance fits within the route, leaving room for at
        # least one navigation trigger after it.
        sampled_speed_ms = _sample_accelerate_speed_ms(
            route_rng,
            town=route_town,
            waypoint=start_waypoint,
            speed_limit_resolver=speed_limit_resolver,
        )
        accel_target_speed_ms, accel_distance_m = _fit_accelerate_instruction_to_window(
            sampled_speed_ms,
            available_distance_m=max_distance_m,
        )

        base_route = _rebuild_follow_route(start_waypoint, max_distance_m, route_step_m)
        samples = _build_actionability_samples(base_route, carla_map)
        trigger = _select_trigger(samples, min_trigger_distance_m=accel_distance_m)
        if trigger is None:
            print(
                f"[INFO] Route {src_id}: no actionable trigger found after "
                f"acceleration zone ({accel_distance_m:.1f}m), skipping."
            )
            continue

        output_actions = _select_output_actions(trigger.actions, route_rng)
        if not output_actions:
            print(f"[INFO] Route {src_id}: no retained actions, skipping.")
            continue

        prefix_waypoints = list(base_route.waypoints[: trigger.index + 1])
        trigger_waypoint = prefix_waypoints[-1]
        remaining_distance_m = max(0.0, max_distance_m - trigger.distance_m)

        stem_prefix = input_xml.stem if not multiple_routes else f"{input_xml.stem}_{src_id}"
        for action in output_actions:
            # Build the first action suffix
            suffix_result = _rebuild_action_suffix(
                start_waypoint=trigger_waypoint,
                action=action,
                max_distance_m=remaining_distance_m,
                step_m=route_step_m,
                lane_change_prep_m=lane_change_prep_m,
            )
            if suffix_result is None:
                print(f"[INFO] Route {src_id}: action {action} became invalid after rebuild.")
                continue

            # Compute distance from trigger to action end for the first action
            first_action_positions = [
                _waypoint_position(wp)
                for wp in suffix_result.waypoints[: suffix_result.action_end_index + 1]
            ]
            first_action_cumulative = _compute_cumulative_distances(first_action_positions)
            first_trigger_to_action_end_m = (
                first_action_cumulative[-1] if first_action_cumulative else 0.0
            )

            action_end_waypoint = suffix_result.waypoints[suffix_result.action_end_index]
            new_remaining_m = remaining_distance_m - first_trigger_to_action_end_m

            # Build instruction chain starting from action end
            first_spec = InstructionSpec(
                action=action,
                trigger_distance_m=trigger.distance_m,
                duration_m=first_trigger_to_action_end_m,
                phrasing_mode=trigger.phrasing_mode,
            )

            if new_remaining_m > 0.0 and max_chain_depth > 1:
                abs_action_end_m = trigger.distance_m + first_trigger_to_action_end_m
                tail_waypoints, tail_specs = _build_instruction_chain(
                    start_waypoint=action_end_waypoint,
                    carla_map=carla_map,
                    remaining_distance_m=new_remaining_m,
                    step_m=route_step_m,
                    lane_change_prep_m=lane_change_prep_m,
                    min_chain_trigger_gap_m=min_chain_trigger_gap_m,
                    rng=route_rng,
                    cumulative_distance_m=abs_action_end_m,
                    previous_trigger_distance_m=trigger.distance_m,
                    max_depth=max_chain_depth - 1,
                )

                # Merge: prefix + first action segment + tail
                action_segment = list(
                    suffix_result.waypoints[1: suffix_result.action_end_index + 1]
                )
                merged_waypoints = list(prefix_waypoints) + action_segment
                for wp in tail_waypoints[1:]:
                    _append_waypoint_if_new(merged_waypoints, wp)

                all_specs = [first_spec] + tail_specs
            else:
                # Single action, no chaining — use full suffix
                merged_waypoints = list(prefix_waypoints)
                for wp in suffix_result.waypoints[1:]:
                    _append_waypoint_if_new(merged_waypoints, wp)
                # Mark as last instruction
                first_spec = InstructionSpec(
                    action=action,
                    trigger_distance_m=trigger.distance_m,
                    duration_m=-1.0,
                    phrasing_mode=trigger.phrasing_mode,
                )
                all_specs = [first_spec]

            # Ensure the last spec has duration_m=-1
            if all_specs and all_specs[-1].duration_m != -1.0:
                last = all_specs[-1]
                all_specs[-1] = InstructionSpec(
                    action=last.action,
                    trigger_distance_m=last.trigger_distance_m,
                    duration_m=-1.0,
                    phrasing_mode=last.phrasing_mode,
                )

            final_route = _finalize_route(merged_waypoints, route_step_m, resample=True)
            instruction_steps = _build_instruction_steps(
                route_rng,
                instruction_specs=all_specs,
                accel_target_speed_ms=accel_target_speed_ms,
                accel_duration_m=accel_distance_m,
            )

            route_tree = _build_action_route_tree(
                src_route=src_route,
                category=category,
                action=action,
                final_route=final_route,
                accelerate_target_speed_ms=0,
                rng=route_rng,
                force_all_green_traffic_lights=force_all_green_traffic_lights,
                instruction_steps=instruction_steps,
            )
            output_path = output_dir / f"{stem_prefix}_language_rebuilt_{action}.xml"
            ET.ElementTree(route_tree).write(
                output_path, encoding="UTF-8", xml_declaration=True
            )
            written_paths.append(output_path)

    return written_paths


def main() -> None:
    _require_carla()
    args = parse_args()
    input_path = args.input_xml.resolve()

    xodr_roots = (
        None if args.xodr_root is None else [path.expanduser().resolve() for path in args.xodr_root]
    )
    map_cache = CarlaMapCache(xodr_search_roots=xodr_roots)
    speed_limit_resolver = OpenDriveSpeedLimitResolver(xodr_search_roots=xodr_roots)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = (
        repo_root
        / "leaderboard"
        / "data"
        / "language_benchmark"
        / "instruction_following_rebuilt"
    )

    if input_path.is_file():
        output_dir = default_output_dir if args.output is None else args.output.resolve()
        if output_dir.suffix:
            raise ValueError("--output must be a directory path for per-action generation.")

        written_paths = convert_file(
            input_xml=input_path,
            output_dir=output_dir,
            category=args.category,
            seed=args.seed,
            max_distance_m=args.max_distance_m,
            route_step_m=args.route_step_m,
            lane_change_prep_m=args.lane_change_prep_m,
            map_cache=map_cache,
            speed_limit_resolver=speed_limit_resolver,
            force_all_green_traffic_lights=args.force_all_green_traffic_lights,
            max_chain_depth=args.max_chain_depth,
            min_chain_trigger_gap_m=args.min_chain_trigger_gap_m,
        )
        if not written_paths:
            print(f"No XML files generated for: {input_path}")
        else:
            for output_xml in written_paths:
                print(f"Generated rebuilt language benchmark XML: {output_xml}")
        return

    if not input_path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {input_path}")

    input_xml_files = sorted(input_path.glob("*.xml"))
    if not input_xml_files:
        raise ValueError(f"No XML files found in input directory: {input_path}")

    output_dir = default_output_dir if args.output is None else args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_xml in input_xml_files:
        written_paths = convert_file(
            input_xml=input_xml,
            output_dir=output_dir,
            category=args.category,
            seed=args.seed,
            max_distance_m=args.max_distance_m,
            route_step_m=args.route_step_m,
            lane_change_prep_m=args.lane_change_prep_m,
            map_cache=map_cache,
            speed_limit_resolver=speed_limit_resolver,
            force_all_green_traffic_lights=args.force_all_green_traffic_lights,
            max_chain_depth=args.max_chain_depth,
            min_chain_trigger_gap_m=args.min_chain_trigger_gap_m,
        )
        if not written_paths:
            print(f"No XML files generated for: {input_xml}")
            continue
        for output_xml in written_paths:
            print(f"Generated rebuilt language benchmark XML: {output_xml}")


if __name__ == "__main__":
    main()
