#!/usr/bin/env python3
"""
Instruction library, text sampling, and speed/acceleration helpers.

This module owns the canonical ``INSTRUCTION_LIBRARY`` mapping (action
categories → paraphrase pools + expected behaviours) and all functions that
build instruction *templates* (dicts with ``text``, ``command_id``, and
``expected_behavior`` keys).  It also contains physics helpers for
acceleration distance budgeting.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from language_navigation.opendrive import OpenDriveSpeedLimitResolver

__all__ = [
    "INSTRUCTION_LIBRARY",
    "OPTIONAL_LANE_FOLLOW_PROBABILITY",
    "ASSUMED_ACCELERATION_MS2",
    "_navigation_text_variants",
    "_sample_navigation_instruction",
    "_sample_navigation_instruction_for_action",
    "_build_precise_accelerate_instruction",
    "_build_lane_follow_instruction",
    "_sample_instruction_template",
    "_sample_accelerate_speed_ms",
    "_required_acceleration_distance_m",
    "_cap_speed_for_distance",
    "_fit_accelerate_instruction_to_window",
    "_select_output_actions",
    "_append_instruction",
]


# ---------------------------------------------------------------------------
# Instruction categories → paraphrase pools + behaviour mapping
# ---------------------------------------------------------------------------

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

# Probability of including a ``lane_follow`` action when other actions exist.
OPTIONAL_LANE_FOLLOW_PROBABILITY = 0.2

# Default acceleration assumption for distance budgeting (m/s²).
ASSUMED_ACCELERATION_MS2 = 3.0


# ---------------------------------------------------------------------------
# Navigation text variants (phrasing-mode-aware)
# ---------------------------------------------------------------------------

def _navigation_text_variants(category: str, phrasing_mode: str) -> List[str]:
    """Return text paraphrases for *category* adapted to *phrasing_mode*.

    *phrasing_mode* is either ``"approach"`` (the trigger fires before the
    junction) or ``"at_junction"`` (the trigger fires inside the junction).
    Exit actions (``exit_left`` / ``exit_right``) have their own fixed set.
    """
    if category == "exit_left":
        return [
            "take the exit on the left",
            "exit from the left lane",
            "leave via the left exit",
        ]
    if category == "exit_right":
        return [
            "take the exit on the right",
            "exit from the right lane",
            "leave via the right exit",
        ]
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
    # Fallback: use the global INSTRUCTION_LIBRARY texts.
    return list(INSTRUCTION_LIBRARY[category]["texts"])


# ---------------------------------------------------------------------------
# Instruction template sampling
# ---------------------------------------------------------------------------

def _sample_navigation_instruction(
    rng: random.Random,
    actionable_categories: List[str],
    phrasing_mode: str = "default",
) -> Dict[str, object]:
    """Sample a navigation instruction from the given *actionable_categories*.

    Picks a random category, then a random text variant for that category.
    Returns a template dict with ``text``, ``command_id``, ``expected_behavior``.
    """
    if not actionable_categories:
        actionable_categories = ["lane_follow"]
    category = rng.choice(actionable_categories)
    entry = INSTRUCTION_LIBRARY[category]
    return {
        "text": rng.choice(_navigation_text_variants(category, phrasing_mode)),
        "command_id": entry["command_id"],
        "expected_behavior": dict(entry["expected_behavior"]),
    }


def _sample_navigation_instruction_for_action(
    rng: random.Random,
    action: str,
    phrasing_mode: str,
) -> Dict[str, object]:
    """Build an instruction template for a *specific* action.

    Handles ``exit_left`` / ``exit_right`` specially (they map to
    ``turn`` expected behaviours with a direction).
    """
    if action in ("exit_left", "exit_right"):
        direction = "left" if action.endswith("left") else "right"
        return {
            "text": rng.choice(_navigation_text_variants(action, phrasing_mode)),
            "command_id": 1 if direction == "left" else 2,
            "expected_behavior": {"type": "turn", "direction": direction},
        }

    entry = INSTRUCTION_LIBRARY[action]
    return {
        "text": rng.choice(_navigation_text_variants(action, phrasing_mode)),
        "command_id": entry["command_id"],
        "expected_behavior": dict(entry["expected_behavior"]),
    }


# ---------------------------------------------------------------------------
# Accelerate / speed instruction builders
# ---------------------------------------------------------------------------

def _build_precise_accelerate_instruction(
    rng: random.Random,
    target_speed_ms: int,
) -> Dict[str, object]:
    """Build a ``target_speed`` instruction template for a precise speed.

    Randomly picks one of several phrasing styles (e.g. "accelerate to …",
    "set your speed to …").
    """
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


def _build_lane_follow_instruction(rng: random.Random) -> Dict[str, object]:
    """Build a ``lane_follow`` instruction template from the library."""
    entry = INSTRUCTION_LIBRARY["lane_follow"]
    return {
        "text": rng.choice(entry["texts"]),
        "command_id": entry["command_id"],
        "expected_behavior": dict(entry["expected_behavior"]),
    }


def _sample_instruction_template(
    rng: random.Random,
    style: str,
) -> Dict[str, object]:
    """Sample a random instruction template from any category.

    The *style* parameter controls accelerate phrasing: ``"vague"`` picks
    from ``accelerate_vague``; ``"precise"`` uses a precise speed target;
    anything else randomly picks between the two.
    """
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


# ---------------------------------------------------------------------------
# Speed sampling & acceleration physics
# ---------------------------------------------------------------------------

def _sample_accelerate_speed_ms(
    rng: random.Random,
    town: str,
    waypoint: "carla.Waypoint",
    speed_limit_resolver: OpenDriveSpeedLimitResolver,
) -> int:
    """Sample a target speed in m/s that respects the road's speed limit.

    Queries the OpenDRIVE speed limit for the waypoint's road and ``s``
    position, then samples uniformly in ``[limit − 10 km/h, limit]``
    converted to m/s.
    """
    speed_limit_kmh = int(
        round(
            speed_limit_resolver.get_speed_limit_kmh(
                town,
                int(waypoint.road_id),
                float(waypoint.s),
            )
        )
    )
    low_kmh = max(0, speed_limit_kmh - 10)
    sampled_kmh = rng.randint(low_kmh, speed_limit_kmh)
    return int(round(sampled_kmh / 3.6))


def _required_acceleration_distance_m(
    target_speed_ms: float,
    acceleration_ms2: float = ASSUMED_ACCELERATION_MS2,
) -> float:
    """Minimum distance to reach *target_speed_ms* from rest: ``d = v²/(2a)``."""
    return (target_speed_ms ** 2) / (2.0 * acceleration_ms2)


def _cap_speed_for_distance(
    target_speed_ms: int,
    available_distance_m: float,
    acceleration_ms2: float = ASSUMED_ACCELERATION_MS2,
) -> int:
    """Cap target speed to what is achievable in *available_distance_m*.

    Uses ``v = sqrt(2 * a * d)``.  Returns at least 3 m/s.
    """
    min_distance = _required_acceleration_distance_m(target_speed_ms, acceleration_ms2)
    if available_distance_m >= min_distance:
        return target_speed_ms
    max_speed = math.sqrt(2.0 * acceleration_ms2 * max(0.0, available_distance_m))
    return max(3, int(max_speed))


def _fit_accelerate_instruction_to_window(
    target_speed_ms: int,
    available_distance_m: float,
    acceleration_ms2: float = ASSUMED_ACCELERATION_MS2,
) -> Tuple[int, float]:
    """Cap an accelerate instruction to the available distance budget.

    Returns:
        A ``(fitted_speed_ms, fitted_duration_m)`` tuple where the speed is
        clamped so the instruction does not extend past the next primary
        trigger window, and the duration is the corresponding acceleration
        distance.
    """
    if available_distance_m <= 0.0:
        return target_speed_ms, 0.0

    fitted_speed_ms = _cap_speed_for_distance(
        target_speed_ms,
        available_distance_m,
        acceleration_ms2,
    )
    fitted_duration_m = min(
        available_distance_m,
        _required_acceleration_distance_m(fitted_speed_ms, acceleration_ms2),
    )
    return fitted_speed_ms, fitted_duration_m


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def _select_output_actions(
    actions: List[str],
    rng: random.Random,
) -> List[str]:
    """De-duplicate *actions*, keep non-lane_follow, optionally add lane_follow.

    ``lane_follow`` is always dropped unless it is the *only* action or a
    random coin-flip (probability ``OPTIONAL_LANE_FOLLOW_PROBABILITY``)
    decides to include it.
    """
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


# ---------------------------------------------------------------------------
# XML instruction element builder
# ---------------------------------------------------------------------------

def _append_instruction(
    instructions_elem,
    instruction_id: int,
    trigger_distance_m: float,
    template: Dict[str, object],
    duration_meters: float,
) -> None:
    """Append an ``<instruction>`` XML child to *instructions_elem*.

    Args:
        instructions_elem: Parent ``<instructions>`` ``Element``.
        instruction_id: Sequential integer id for this instruction.
        trigger_distance_m: Cumulative distance at which the instruction fires.
        template: Dict with ``text``, ``command_id``, ``expected_behavior``.
        duration_meters: Distance the instruction is active (``-1`` = until end).
    """
    import xml.etree.ElementTree as ET

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
