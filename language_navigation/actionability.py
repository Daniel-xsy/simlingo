#!/usr/bin/env python3
"""
Turn scanning, actionability sampling, trigger selection, and special-case
detection for language-benchmark routes.

This module answers the question "what navigation actions are feasible at
each point along a route?" and selects the best trigger point for an
instruction.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from language_navigation.geometry import (
    _can_change_lane,
    _compute_turn_category,
    _distance,
    _is_same_direction_lane,
    _normalize_yaw_delta_deg,
    _position_at_distance,
    _route_cumulative_distances,
    _route_length_m,
    _same_direction_adjacent_drive_count,
)
from language_navigation.instructions import (
    INSTRUCTION_LIBRARY,
    _sample_navigation_instruction,
)

try:
    import carla
except ImportError:
    carla = None

__all__ = [
    "RouteActionabilitySample",
    "SelectedNavigationTrigger",
    "RouteSpecialCase",
    "_scan_turn_actions",
    "_build_actionable_navigation_categories",
    "_score_actionable_categories",
    "_sample_route_actionability",
    "_yaw_delta_abs_deg",
    "_is_straight_approach_candidate",
    "_collect_straight_approach_indices",
    "_first_valid_trigger_index",
    "_first_junction_index",
    "_first_sample_index_at_or_after",
    "_infer_exit_side",
    "select_navigation_trigger",
    "detect_route_special_case",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteActionabilitySample:
    """A single sample point along a route with its feasible navigation actions.

    Attributes:
        distance_m: Cumulative distance from route start to this sample.
        position: 3-D world position ``(x, y, z)``.
        actions: All feasible actions (including ``lane_follow``).
        scored_actions: Feasible actions *excluding* ``lane_follow``.
        score: Number of scored (non-lane_follow) actions.
        is_junction: Whether the sample is inside a CARLA junction.
        lane_id / road_id / section_id: CARLA road identifiers (may be ``None``).
        yaw_deg: Waypoint yaw in degrees (may be ``None``).
        remaining_route_distance_m: How much route remains after this sample.
    """
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
    """The trigger point chosen for a navigation instruction.

    Attributes:
        distance_m: Cumulative trigger distance from route start.
        position: World position of the trigger.
        actions: Feasible navigation actions at the trigger.
        selected_action: The action that was actually sampled.
        score: Actionability score at this point.
        is_junction: Whether the trigger is inside a junction.
        source_kind: How the trigger was selected (e.g. ``"max_score_earliest"``).
        phrasing_mode: ``"approach"`` or ``"at_junction"``.
        sample_index: Index into the samples list.
        sampled_text: The instruction text that was sampled.
    """
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


@dataclass(frozen=True)
class RouteSpecialCase:
    """A detected merge or exit transition along the route.

    Attributes:
        kind: ``"merge"`` or ``"exit"``.
        action_name: Human-readable label (e.g. ``"merge_lane_change_left"``).
        primary_action / primary_trigger_distance_m / primary_position:
            The main instruction to issue.
        secondary_*: An optional follow-up instruction (e.g. the actual exit
            after a preparatory lane change).
    """
    kind: str
    action_name: str
    primary_action: str
    primary_trigger_distance_m: float
    primary_position: Tuple[float, float, float]
    primary_phrasing_mode: str
    secondary_action: Optional[str] = None
    secondary_trigger_distance_m: Optional[float] = None
    secondary_position: Optional[Tuple[float, float, float]] = None
    secondary_phrasing_mode: Optional[str] = None


# ---------------------------------------------------------------------------
# Turn scanning (BFS along CARLA topology)
# ---------------------------------------------------------------------------

def _scan_turn_actions(
    start_waypoint: "carla.Waypoint",
    scan_distance_m: float = 45.0,
    step_m: float = 2.0,
) -> Set[str]:
    """Scan the upcoming road topology and return feasible ``turn_*`` actions.

    Uses a BFS-style frontier expansion: at each step, follow road
    successors forward.  When the topology branches near a junction,
    trace each branch to its exit and classify the resulting yaw delta
    as ``turn_left``, ``turn_right``, or ``turn_straight``.
    """
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

            # Detect branching near a junction.
            is_branch_near_junction = len(next_candidates) > 1 and (
                waypoint.is_junction or any(c.is_junction for c in next_candidates)
            )
            if is_branch_near_junction:
                for candidate in next_candidates:
                    branch_waypoint = candidate
                    branch_traversed = step_m
                    while branch_traversed < scan_distance_m:
                        next_branch = branch_waypoint.next(step_m)
                        if not next_branch:
                            break
                        branch_waypoint = next_branch[0]
                        branch_traversed += step_m
                        if not branch_waypoint.is_junction:
                            break
                    turn_actions.add(_compute_turn_category(start_waypoint, branch_waypoint))
                continue

            candidate = next_candidates[0]
            if waypoint.is_junction or candidate.is_junction:
                branch_waypoint = candidate
                branch_traversed = step_m
                while branch_traversed < scan_distance_m:
                    next_branch = branch_waypoint.next(step_m)
                    if not next_branch:
                        break
                    branch_waypoint = next_branch[0]
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
    """Query the CARLA map at *ego_position* and return all feasible navigation actions.

    Always includes ``"lane_follow"``; may also include lane-change and turn
    actions depending on the local road topology.
    """
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
    """Count non-``lane_follow`` actions (the 'interestingness' score)."""
    return sum(1 for action in actions if action != "lane_follow")


# ---------------------------------------------------------------------------
# Route-wide actionability sampling
# ---------------------------------------------------------------------------

def _sample_route_actionability(
    route_positions: List[Tuple[float, float, float]],
    trigger_step_m: float,
    carla_map: "carla.Map",
) -> List[RouteActionabilitySample]:
    """Sample actionability at a dense grid near the start and coarser grid afterward.

    The dense grid uses 1 m steps up to ``max(10, trigger_step_m)``; the
    coarse grid uses *trigger_step_m*.  This ensures short-range triggers
    (e.g. immediate lane changes) are not missed.
    """
    if trigger_step_m <= 0:
        raise ValueError("--trigger-step-m must be > 0.")

    total_length = _route_length_m(route_positions)
    max_distance = max(total_length - 1.0, 0.0)

    distance_values: Set[float] = {0.0, max_distance}

    # Dense sampling near the start.
    fine_step_m = 1.0
    current = 0.0
    dense_limit = min(max_distance, max(10.0, trigger_step_m))
    while current <= dense_limit + 1e-6:
        distance_values.add(round(min(current, max_distance), 3))
        current += fine_step_m

    # Coarse sampling for the rest.
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


# ---------------------------------------------------------------------------
# Trigger selection helpers
# ---------------------------------------------------------------------------

def _yaw_delta_abs_deg(a_deg: Optional[float], b_deg: Optional[float]) -> float:
    """Absolute normalised yaw delta, or 180° if either value is ``None``."""
    if a_deg is None or b_deg is None:
        return 180.0
    return abs(_normalize_yaw_delta_deg(a_deg - b_deg))


def _is_straight_approach_candidate(
    previous_sample: RouteActionabilitySample,
    junction_sample: RouteActionabilitySample,
    max_yaw_delta_deg: float = 15.0,
) -> bool:
    """Check whether *previous_sample* is on a straight approach to *junction_sample*."""
    if previous_sample.is_junction:
        return False
    if junction_sample.road_id is None or previous_sample.road_id is None:
        return False
    return _yaw_delta_abs_deg(previous_sample.yaw_deg, junction_sample.yaw_deg) <= max_yaw_delta_deg


def _collect_straight_approach_indices(
    samples: List[RouteActionabilitySample], junction_index: int
) -> List[int]:
    """Back-track from *junction_index* collecting straight-approach sample indices."""
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


def _first_valid_trigger_index(
    samples: List[RouteActionabilitySample], min_distance_m: float
) -> int:
    """Return the index of the first sample at or past *min_distance_m*."""
    for idx, sample in enumerate(samples):
        if sample.distance_m >= min_distance_m:
            return idx
    return len(samples) - 1


def _first_junction_index(samples: List[RouteActionabilitySample]) -> Optional[int]:
    """Return the index of the first junction sample, or ``None``."""
    for idx, sample in enumerate(samples):
        if sample.is_junction:
            return idx
    return None


def _first_sample_index_at_or_after(
    samples: List[RouteActionabilitySample], distance_m: float
) -> int:
    """Return the index of the first sample at or past *distance_m*."""
    for idx, sample in enumerate(samples):
        if sample.distance_m >= distance_m:
            return idx
    return len(samples) - 1


def _infer_exit_side(
    pre_waypoint: "carla.Waypoint",
    post_waypoint: "carla.Waypoint",
) -> str:
    """Determine whether a road exit goes ``"left"`` or ``"right"``.

    Falls back to whichever lane-change direction is available if the
    turn category is straight.
    """
    turn_direction = _compute_turn_category(pre_waypoint, post_waypoint)
    if turn_direction == "turn_left":
        return "left"
    if turn_direction == "turn_right":
        return "right"
    if _can_change_lane(pre_waypoint, "right"):
        return "right"
    if _can_change_lane(pre_waypoint, "left"):
        return "left"
    return "right"


# ---------------------------------------------------------------------------
# Trigger selection (main entry point)
# ---------------------------------------------------------------------------

def select_navigation_trigger(
    route_positions: List[Tuple[float, float, float]],
    trigger_step_m: float,
    carla_map: "carla.Map",
    rng: random.Random,
) -> Tuple[SelectedNavigationTrigger, List[RouteActionabilitySample]]:
    """Select the best trigger point and sample a navigation instruction.

    Returns a ``(trigger, samples)`` tuple.  The algorithm picks the
    earliest sample with the highest actionability score, with fallback
    logic to enforce a minimum trigger distance and prefer junction-entry
    triggers when the best candidate is too close to the start.
    """
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

    # Guard: enforce minimum trigger distance.
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


# ---------------------------------------------------------------------------
# Special-case detection (merge / exit)
# ---------------------------------------------------------------------------

def detect_route_special_case(
    route_positions: List[Tuple[float, float, float]],
    carla_map: "carla.Map",
    samples: List[RouteActionabilitySample],
) -> Optional[RouteSpecialCase]:
    """Detect merge or exit lane transitions along the route.

    Walks consecutive CARLA waypoints and looks for road/lane-id changes
    outside of junctions.  A transition from 0 adjacent lanes to ≥ 1 is a
    **merge**; from ≥ 1 to 0 is an **exit**.
    """
    if len(route_positions) < 2:
        return None

    cumulative_distances = _route_cumulative_distances(route_positions)
    route_waypoints: List["carla.Waypoint"] = []
    for position in route_positions:
        waypoint = carla_map.get_waypoint(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        route_waypoints.append(waypoint)

    max_trigger_distance = max(cumulative_distances[-1] - 1.0, 0.0)

    for idx in range(1, len(route_waypoints)):
        pre_waypoint = route_waypoints[idx - 1]
        post_waypoint = route_waypoints[idx]
        if pre_waypoint is None or post_waypoint is None:
            continue
        if pre_waypoint.is_junction or post_waypoint.is_junction:
            continue
        if (
            pre_waypoint.road_id == post_waypoint.road_id
            and pre_waypoint.section_id == post_waypoint.section_id
            and pre_waypoint.lane_id == post_waypoint.lane_id
        ):
            continue

        transition_distance = cumulative_distances[idx]
        merge_in_pre_count = _same_direction_adjacent_drive_count(pre_waypoint)
        merge_in_post_count = _same_direction_adjacent_drive_count(post_waypoint)

        # Merge: no adjacent lanes before → adjacent lanes after.
        if merge_in_pre_count == 0 and merge_in_post_count >= 1:
            trigger_distance = min(max(transition_distance + 5.0, 5.0), max_trigger_distance)
            trigger_position = _position_at_distance(route_positions, trigger_distance)
            if _can_change_lane(post_waypoint, "left"):
                primary_action = "lane_change_left"
            elif _can_change_lane(post_waypoint, "right"):
                primary_action = "lane_change_right"
            else:
                primary_action = "lane_follow"
            return RouteSpecialCase(
                kind="merge",
                action_name=f"merge_{primary_action}",
                primary_action=primary_action,
                primary_trigger_distance_m=trigger_distance,
                primary_position=trigger_position,
                primary_phrasing_mode="approach",
            )

        # Exit: adjacent lanes before → no adjacent lanes after.
        if merge_in_pre_count >= 1 and merge_in_post_count == 0:
            exit_side = _infer_exit_side(pre_waypoint, post_waypoint)
            prep_action = (
                f"lane_change_{exit_side}" if _can_change_lane(pre_waypoint, exit_side) else None
            )
            exit_trigger_distance = min(max(transition_distance, 5.0), max_trigger_distance)
            exit_position = _position_at_distance(route_positions, exit_trigger_distance)
            prep_distance = None
            prep_position = None
            if prep_action is not None:
                prep_distance = min(max(5.0, exit_trigger_distance - 10.0), exit_trigger_distance)
                prep_position = _position_at_distance(route_positions, prep_distance)
            return RouteSpecialCase(
                kind="exit",
                action_name=f"exit_{exit_side}",
                primary_action=prep_action or f"exit_{exit_side}",
                primary_trigger_distance_m=prep_distance or exit_trigger_distance,
                primary_position=prep_position or exit_position,
                primary_phrasing_mode="approach",
                secondary_action=f"exit_{exit_side}" if prep_action is not None else None,
                secondary_trigger_distance_m=exit_trigger_distance if prep_action is not None else None,
                secondary_position=exit_position if prep_action is not None else None,
                secondary_phrasing_mode="approach" if prep_action is not None else None,
            )

    return None
