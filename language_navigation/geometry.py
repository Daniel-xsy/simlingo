#!/usr/bin/env python3
"""
Geometry and distance helpers for route analysis.

Pure-computation utilities for 3-D positions, route lengths, cumulative
distances, yaw normalization, turn classification, and CARLA lane queries.
None of these functions perform I/O or modify external state.
"""

import math
from typing import List, Optional, Tuple

try:
    import carla
except ImportError:
    carla = None

__all__ = [
    "_distance",
    "_get_waypoint_positions",
    "_route_length_m",
    "_position_at_distance",
    "_route_cumulative_distances",
    "_normalize_yaw_delta_deg",
    "_compute_turn_category",
    "_is_same_direction_lane",
    "_can_change_lane",
    "_same_direction_adjacent_drive_count",
]


# ---------------------------------------------------------------------------
# Position / distance primitives
# ---------------------------------------------------------------------------

def _distance(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    """Return the Euclidean distance between two 3-D position tuples."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


# ---------------------------------------------------------------------------
# XML waypoint extraction
# ---------------------------------------------------------------------------

def _get_waypoint_positions(waypoints_elem) -> List[Tuple[float, float, float]]:
    """Parse ``<position x=... y=... z=...>`` children of *waypoints_elem*.

    Args:
        waypoints_elem: An ``xml.etree.ElementTree.Element`` whose children
            are ``<position>`` tags with *x*, *y*, *z* attributes.

    Returns:
        A list of ``(x, y, z)`` float tuples in document order.
    """
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


# ---------------------------------------------------------------------------
# Route length / interpolation
# ---------------------------------------------------------------------------

def _route_length_m(
    positions: List[Tuple[float, float, float]],
) -> float:
    """Sum of segment lengths along a polyline of 3-D positions."""
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        total += _distance(positions[i - 1], positions[i])
    return total


def _position_at_distance(
    positions: List[Tuple[float, float, float]],
    target_distance_m: float,
) -> Tuple[float, float, float]:
    """Linearly interpolate along *positions* at the given cumulative distance.

    Returns the first position when *target_distance_m* ≤ 0, or the last
    position when the target exceeds the total polyline length.
    """
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


def _route_cumulative_distances(
    route_positions: List[Tuple[float, float, float]],
) -> List[float]:
    """Build a list of cumulative distances from the first position.

    ``result[0]`` is always ``0.0``; ``result[i]`` is the total distance from
    ``route_positions[0]`` to ``route_positions[i]``.
    """
    cumulative = [0.0]
    for idx in range(1, len(route_positions)):
        cumulative.append(
            cumulative[-1] + _distance(route_positions[idx - 1], route_positions[idx])
        )
    return cumulative


# ---------------------------------------------------------------------------
# Yaw / turn classification
# ---------------------------------------------------------------------------

def _normalize_yaw_delta_deg(delta_deg: float) -> float:
    """Normalize an angle delta to the range ``[-180, 180]`` degrees."""
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
    """Classify the turn between two CARLA waypoints.

    Returns:
        ``"turn_straight"``, ``"turn_left"``, or ``"turn_right"`` depending
        on the yaw delta relative to *threshold_deg*.
    """
    yaw_delta = _normalize_yaw_delta_deg(
        next_waypoint.transform.rotation.yaw - current_waypoint.transform.rotation.yaw
    )
    if abs(yaw_delta) < threshold_deg:
        return "turn_straight"
    if yaw_delta < 0.0:
        return "turn_left"
    return "turn_right"


# ---------------------------------------------------------------------------
# CARLA lane helpers
# ---------------------------------------------------------------------------

def _is_same_direction_lane(
    source_waypoint: "carla.Waypoint",
    candidate_waypoint: "carla.Waypoint",
    max_yaw_delta_deg: float = 45.0,
) -> bool:
    """Check whether two waypoints face roughly the same direction."""
    yaw_delta = _normalize_yaw_delta_deg(
        candidate_waypoint.transform.rotation.yaw - source_waypoint.transform.rotation.yaw
    )
    return abs(yaw_delta) < max_yaw_delta_deg


def _can_change_lane(ego_waypoint: "carla.Waypoint", direction: str) -> bool:
    """Check whether a lane change in *direction* is feasible.

    Verifies that the CARLA lane-change flag permits the manoeuvre, the
    adjacent lane exists, is of type ``Driving``, and faces the same
    direction.

    Args:
        ego_waypoint: Current CARLA waypoint.
        direction: ``"left"`` or ``"right"``.

    Returns:
        ``True`` if the lane change is valid.
    """
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


def _same_direction_adjacent_drive_count(ego_waypoint: "carla.Waypoint") -> int:
    """Count how many adjacent driving lanes face the same direction.

    Checks both left and right neighbours; returns 0, 1, or 2.
    """
    count = 0
    for direction in ("left", "right"):
        adjacent = (
            ego_waypoint.get_left_lane() if direction == "left" else ego_waypoint.get_right_lane()
        )
        if adjacent is None:
            continue
        if adjacent.lane_type != carla.LaneType.Driving:
            continue
        if _is_same_direction_lane(ego_waypoint, adjacent):
            count += 1
    return count
