#!/usr/bin/env python3
"""
Generic XML building blocks for language-benchmark route files.

Contains the canonical ``_indent_xml_compat`` pretty-printer (shared by all
generators), default ``<evaluation>`` and ``<scenarios>`` element builders,
and the ``_build_route_instructions`` / ``_build_action_route_tree`` helpers
used by the *non-rebuilt* (``utils.py``-based) code path.
"""

import copy
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from language_navigation_dev.geometry import _get_waypoint_positions
from language_navigation_dev.instructions import (
    ASSUMED_ACCELERATION_MS2,
    INSTRUCTION_LIBRARY,
    _append_instruction,
    _build_precise_accelerate_instruction,
    _fit_accelerate_instruction_to_window,
    _sample_navigation_instruction_for_action,
)

# Lazy import to avoid heavy circular dependency — only used at function level.
import random

__all__ = [
    "_indent_xml_compat",
    "_build_default_evaluation",
    "_build_default_scenarios",
    "_build_route_instructions",
    "_build_action_route_tree",
]


# ---------------------------------------------------------------------------
# XML pretty-print (Python 3.8+ compatible)
# ---------------------------------------------------------------------------

def _indent_xml_compat(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to an ``ElementTree`` element tree.

    Uses ``ET.indent`` when available (Python ≥ 3.9) and falls back to a
    recursive implementation for older runtimes.
    """
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


# ---------------------------------------------------------------------------
# Default evaluation element
# ---------------------------------------------------------------------------

def _build_default_evaluation() -> ET.Element:
    """Create a standard ``<evaluation>`` element with collision and compliance metrics."""
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


# ---------------------------------------------------------------------------
# Default scenarios element (from ET.Element route)
# ---------------------------------------------------------------------------

def _build_default_scenarios(route_elem: ET.Element) -> ET.Element:
    """Build a minimal ``<scenarios>`` block with a single FreeRide trigger.

    Reads the first ``<position>`` from the route's ``<waypoints>`` child
    and computes the initial yaw from the first two positions.
    """
    waypoints = route_elem.find("waypoints")
    if waypoints is None:
        raise ValueError(f"Route {route_elem.attrib.get('id', 'unknown')} has no <waypoints>.")

    positions = _get_waypoint_positions(waypoints)
    if not positions:
        raise ValueError(f"Route {route_elem.attrib.get('id', 'unknown')} has empty <waypoints>.")

    x0, y0, z0 = positions[0]

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


# ---------------------------------------------------------------------------
# Route-level instruction builder (non-rebuilt path)
# ---------------------------------------------------------------------------

def _build_route_instructions(
    rng: random.Random,
    accelerate_target_speed_ms: int,
    navigation_trigger,  # SelectedNavigationTrigger
    special_case=None,  # Optional[RouteSpecialCase]
    action: Optional[str] = None,
) -> ET.Element:
    """Build ``<instructions>`` for the non-rebuilt (original-waypoint) path.

    Creates an accelerate instruction at distance 0, followed by one or two
    navigation instructions depending on whether a *special_case* (merge/exit)
    is active.
    """
    instructions_elem = ET.Element("instructions")
    second_trigger_distance = (
        special_case.primary_trigger_distance_m
        if special_case is not None
        else navigation_trigger.distance_m
    )
    fitted_speed_ms, accelerate_duration_m = _fit_accelerate_instruction_to_window(
        accelerate_target_speed_ms,
        second_trigger_distance,
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

    if special_case is None:
        if action is None:
            raise ValueError("Normal route instruction building requires an action.")
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

    primary_template = _sample_navigation_instruction_for_action(
        rng,
        action=special_case.primary_action,
        phrasing_mode=special_case.primary_phrasing_mode,
    )
    primary_duration = -1.0
    if special_case.secondary_action is not None and special_case.secondary_trigger_distance_m is not None:
        primary_duration = max(
            0.0, special_case.secondary_trigger_distance_m - special_case.primary_trigger_distance_m
        )
    _append_instruction(
        instructions_elem,
        instruction_id=2,
        trigger_distance_m=special_case.primary_trigger_distance_m,
        template=primary_template,
        duration_meters=primary_duration,
    )
    if special_case.secondary_action is not None and special_case.secondary_trigger_distance_m is not None:
        secondary_template = _sample_navigation_instruction_for_action(
            rng,
            action=special_case.secondary_action,
            phrasing_mode=special_case.secondary_phrasing_mode or "approach",
        )
        _append_instruction(
            instructions_elem,
            instruction_id=3,
            trigger_distance_m=special_case.secondary_trigger_distance_m,
            template=secondary_template,
            duration_meters=-1.0,
        )
    return instructions_elem


# ---------------------------------------------------------------------------
# Full route XML tree builder (non-rebuilt path)
# ---------------------------------------------------------------------------

def _build_action_route_tree(
    src_route: ET.Element,
    category: str,
    action: str,
    navigation_trigger,  # SelectedNavigationTrigger
    accelerate_target_speed_ms: int,
    rng: random.Random,
    special_case=None,  # Optional[RouteSpecialCase]
) -> ET.Element:
    """Assemble a complete ``<routes>`` XML tree for the non-rebuilt code path.

    Combines waypoints (deep-copied from *src_route*), instructions,
    evaluation metrics, scenarios, and weathers into an indented XML tree.
    """
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
        _build_route_instructions(
            rng=rng,
            accelerate_target_speed_ms=accelerate_target_speed_ms,
            navigation_trigger=navigation_trigger,
            special_case=special_case,
            action=action,
        )
    )
    target_route.append(_build_default_evaluation())
    target_route.append(_build_default_scenarios(src_route))

    weathers_elem = src_route.find("weathers")
    if weathers_elem is not None:
        target_route.append(copy.deepcopy(weathers_elem))

    _indent_xml_compat(target_root)
    return target_root
