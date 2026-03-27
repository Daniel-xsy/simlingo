#!/usr/bin/env python3
"""
Generate language-navigation XML by rebuilding the route from the Bench2Drive
start point and branching the GT route after the sampled trigger.

This is the main entry point for benchmark XML generation.  Route
reconstruction, actionability sampling, and instruction chaining are
delegated to the ``route_builder``, ``actionability``, and ``instructions``
submodules respectively.
"""

import argparse
import copy
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

from language_navigation.opendrive import (
    CarlaMapCache,
    OpenDriveSpeedLimitResolver,
)
from language_navigation.planner_route_tools import (
    PlannerCache,
    build_planner_safe_export_positions,
)
from language_navigation.geometry import (
    _get_waypoint_positions,
)
from language_navigation.instructions import (
    ASSUMED_ACCELERATION_MS2,
    _append_instruction,
    _build_lane_follow_instruction,
    _build_precise_accelerate_instruction,
    _fit_accelerate_instruction_to_window,
    _sample_accelerate_speed_ms,
    _sample_navigation_instruction_for_action,
)
from language_navigation.xml_builder import (
    _build_default_evaluation,
    _indent_xml_compat,
)
from language_navigation.route_builder import (
    ActionSuffixResult,
    RebuiltActionabilitySample,
    RebuiltRoute,
    RebuiltTrigger,
    _append_waypoint_if_new,
    _build_actionability_samples,
    _build_default_scenarios_from_positions,
    _build_waypoints_element,
    _compute_cumulative_distances,
    _finalize_route,
    _rebuild_action_suffix,
    _rebuild_follow_route,
    _select_output_actions,
    _select_trigger,
    _truncate_waypoints_before_terminal_junction,
    _waypoint_position,
    Position3D,
)


# ---------------------------------------------------------------------------
# Data classes used only by the instruction-building orchestration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstructionSpec:
    """Specification for a single navigation instruction in a chain.

    Attributes:
        action: Navigation action name (e.g. ``"turn_left"``).
        trigger_distance_m: Cumulative distance from route start where
            the instruction fires.
        duration_m: Distance the instruction stays active; ``-1`` means
            "until the end of the route".
        phrasing_mode: ``"approach"`` or ``"at_junction"``.
        trigger_position: World position of the trigger point.  When set,
            the XML uses a ``distance_to_point`` trigger instead of
            ``distance_traveled``.
    """
    action: str
    trigger_distance_m: float
    duration_m: float
    phrasing_mode: str
    trigger_position: Optional[Position3D] = None


@dataclass
class InstructionStep:
    """A concrete instruction ready to be serialised to XML.

    Attributes:
        template: Dict with ``text``, ``command_id``, ``expected_behavior``.
        trigger_distance_m: Cumulative trigger distance.
        duration_m: Active duration in metres (``-1`` = until end).
        trigger_position: World position for ``distance_to_point`` triggers.
    """
    template: Dict[str, object]
    trigger_distance_m: float
    duration_m: float
    trigger_position: Optional[Position3D] = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the route generator."""
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
    """Abort early if the CARLA Python API is not importable."""
    if carla is None:
        raise RuntimeError(
            "CARLA Python API is required for rebuilt route generation."
        ) from CARLA_IMPORT_ERROR


def _resolve_start_waypoint(
    positions: Sequence[Position3D],
    carla_map: "carla.Map",
) -> "carla.Waypoint":
    """Project the first route position onto the nearest CARLA driving waypoint."""
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


# ---------------------------------------------------------------------------
# Recursive instruction-chain builder
# ---------------------------------------------------------------------------

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
    """Recursively build waypoints and instruction specs.

    Starting from *start_waypoint*, rebuild a follow route, find the best
    trigger, execute the selected action, and recurse from the action's end
    point until *max_depth* is exhausted or no interesting trigger remains.

    Returns:
        ``(all_waypoints, specs)`` — the chained waypoints and a list of
        ``InstructionSpec`` for each navigation action.
    """
    # Build a follow route from start_waypoint.
    follow_route = _rebuild_follow_route(start_waypoint, remaining_distance_m, step_m)
    if len(follow_route.waypoints) < 2:
        return list(follow_route.waypoints), []

    # Compute actionability samples on this follow route.
    samples = _build_actionability_samples(follow_route, carla_map)
    if previous_trigger_distance_m is not None:
        effective_min_trigger_distance_m = max(
            0.0,
            previous_trigger_distance_m + min_chain_trigger_gap_m - cumulative_distance_m,
        )
    else:
        effective_min_trigger_distance_m = 0.0
    eligible_samples = [
        s for s in samples
        if s.distance_m >= max(0.0, effective_min_trigger_distance_m)
    ]
    if not eligible_samples:
        return list(follow_route.waypoints), []
    trigger = _select_trigger(eligible_samples, min_trigger_distance_m=0.0)
    if trigger is None:
        return list(follow_route.waypoints), []

    # Check for interesting (non-lane_follow) actions.
    interesting_actions = [a for a in trigger.actions if a != "lane_follow"]
    if not interesting_actions or max_depth <= 0:
        return list(follow_route.waypoints), []

    # Pick one non-lane_follow action.
    action = rng.choice(interesting_actions)

    # Build action suffix from trigger waypoint.
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
        return list(follow_route.waypoints), []

    # Compute distances within suffix to find action_end distance.
    action_end_waypoint = suffix_result.waypoints[suffix_result.action_end_index]
    action_segment_positions = [
        _waypoint_position(wp)
        for wp in suffix_result.waypoints[: suffix_result.action_end_index + 1]
    ]
    action_segment_distances = _compute_cumulative_distances(action_segment_positions)
    trigger_to_action_end_m = action_segment_distances[-1] if action_segment_distances else 0.0

    # Build prefix waypoints (start → trigger) + action segment.
    prefix_waypoints = list(follow_route.waypoints[: trigger.index + 1])
    action_waypoints = list(suffix_result.waypoints[1: suffix_result.action_end_index + 1])

    abs_trigger_distance_m = cumulative_distance_m + trigger.distance_m
    abs_action_end_distance_m = abs_trigger_distance_m + trigger_to_action_end_m
    new_remaining_m = remaining_distance_m - trigger.distance_m - trigger_to_action_end_m

    current_spec = InstructionSpec(
        action=action,
        trigger_distance_m=abs_trigger_distance_m,
        duration_m=trigger_to_action_end_m,
        phrasing_mode=trigger.phrasing_mode,
        trigger_position=trigger.position,
    )

    # Try to recurse if enough remaining distance.
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

        # Merge waypoints: prefix + action_segment + tail.
        merged_waypoints = prefix_waypoints + action_waypoints
        for wp in tail_waypoints[1:]:
            _append_waypoint_if_new(merged_waypoints, wp)

        return merged_waypoints, [current_spec] + tail_specs
    else:
        # Last instruction: set duration to -1.
        final_spec = InstructionSpec(
            action=current_spec.action,
            trigger_distance_m=current_spec.trigger_distance_m,
            duration_m=-1.0,
            phrasing_mode=current_spec.phrasing_mode,
            trigger_position=current_spec.trigger_position,
        )
        merged_waypoints = prefix_waypoints
        for wp in suffix_result.waypoints[1:]:
            _append_waypoint_if_new(merged_waypoints, wp)
        return merged_waypoints, [final_spec]


# ---------------------------------------------------------------------------
# Instruction step / XML builders
# ---------------------------------------------------------------------------

def _build_instruction_steps(
    rng: random.Random,
    instruction_specs: List[InstructionSpec],
    accel_target_speed_ms: int,
    accel_duration_m: float,
) -> List[InstructionStep]:
    """Convert ``InstructionSpec`` objects into concrete ``InstructionStep`` objects.

    Always places a speed instruction at distance 0, then fills gaps between
    navigation instructions with ``lane_follow`` fillers.
    """
    if not instruction_specs:
        return []

    instruction_steps: List[InstructionStep] = []

    # Speed instruction at the start.
    speed_template = _build_precise_accelerate_instruction(
        rng,
        target_speed_ms=accel_target_speed_ms,
        keep_straight=True,
    )
    instruction_steps.append(
        InstructionStep(
            template=speed_template,
            trigger_distance_m=0.0,
            duration_m=accel_duration_m,
        )
    )

    prev_end_m = accel_duration_m

    for spec in instruction_specs:
        # Insert lane_follow filler if there is a significant gap.
        gap_m = spec.trigger_distance_m - prev_end_m
        if gap_m > 10.0:
            instruction_steps.append(
                InstructionStep(
                    template=_build_lane_follow_instruction(rng),
                    trigger_distance_m=prev_end_m,
                    duration_m=gap_m,
                )
            )

        # The actual navigation instruction.
        instruction_steps.append(
            InstructionStep(
                template=_sample_navigation_instruction_for_action(
                    rng,
                    action=spec.action,
                    phrasing_mode=spec.phrasing_mode,
                ),
                trigger_distance_m=spec.trigger_distance_m,
                duration_m=spec.duration_m,
                trigger_position=spec.trigger_position,
            )
        )

        if spec.duration_m > 0:
            prev_end_m = spec.trigger_distance_m + spec.duration_m

    return instruction_steps


def _build_instructions_from_steps(
    instruction_steps: Sequence[InstructionStep],
) -> ET.Element:
    """Serialise ``InstructionStep`` objects into an ``<instructions>`` XML element."""
    instructions_elem = ET.Element("instructions")
    for instruction_id, step in enumerate(instruction_steps, start=1):
        _append_instruction(
            instructions_elem,
            instruction_id=instruction_id,
            trigger_distance_m=step.trigger_distance_m,
            template=step.template,
            duration_meters=step.duration_m,
            trigger_position=step.trigger_position,
        )
    return instructions_elem


def _build_route_instructions(
    rng: random.Random,
    accelerate_target_speed_ms: int,
    trigger: RebuiltTrigger,
    action: str,
) -> ET.Element:
    """Build a simple two-instruction XML block (accelerate + navigate)."""
    instructions_elem = ET.Element("instructions")
    fitted_speed_ms, accelerate_duration_m = _fit_accelerate_instruction_to_window(
        accelerate_target_speed_ms,
        trigger.distance_m,
        acceleration_ms2=ASSUMED_ACCELERATION_MS2,
    )
    accelerate_template = _build_precise_accelerate_instruction(
        rng,
        target_speed_ms=fitted_speed_ms,
        keep_straight=True,
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
        trigger_position=trigger.position,
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
    """Build a multi-instruction XML block for chained navigation.

    Two modes:
        * ``use_lane_follow_start=True`` — starts with a lane-follow filler,
          then the first navigation action, optionally a post-action
          accelerate, then remaining navigation actions.
        * ``use_lane_follow_start=False`` (default) — starts with an
          accelerate instruction, then all navigation actions in sequence.
    """
    instructions_elem = ET.Element("instructions")
    next_id = 1

    if use_lane_follow_start:
        # Lane-follow until first action.
        lane_follow_template = _build_lane_follow_instruction(rng)
        _append_instruction(
            instructions_elem,
            instruction_id=next_id,
            trigger_distance_m=0.0,
            template=lane_follow_template,
            duration_meters=first_trigger_distance_m,
        )
        next_id += 1

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
            trigger_position=first_spec.trigger_position,
        )
        next_id += 1

        # Optional post-action accelerate.
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

        # Remaining nav actions.
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
                trigger_position=spec.trigger_position,
            )
            next_id += 1
    else:
        # Normal path: accelerate first.
        fitted_speed_ms, accelerate_duration_m = _fit_accelerate_instruction_to_window(
            accelerate_target_speed_ms,
            first_trigger_distance_m,
            acceleration_ms2=ASSUMED_ACCELERATION_MS2,
        )
        accelerate_template = _build_precise_accelerate_instruction(
            rng,
            target_speed_ms=fitted_speed_ms,
            keep_straight=True,
        )
        _append_instruction(
            instructions_elem,
            instruction_id=next_id,
            trigger_distance_m=0.0,
            template=accelerate_template,
            duration_meters=accelerate_duration_m,
        )
        next_id += 1

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
                trigger_position=spec.trigger_position,
            )
            next_id += 1

    return instructions_elem


# ---------------------------------------------------------------------------
# Full route XML tree builder (rebuilt path)
# ---------------------------------------------------------------------------

def _build_action_route_tree(
    src_route: ET.Element,
    category: str,
    action: str,
    final_route: RebuiltRoute,
    exported_positions: Sequence[Position3D],
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
    """Assemble the complete ``<routes>`` XML tree for a rebuilt route.

    Supports three instruction modes (checked in order):
        1. Pre-built ``instruction_steps``.
        2. ``instruction_specs`` with ``first_trigger_distance_m``.
        3. A single ``trigger`` for a simple two-instruction layout.
    """
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
    target_route = ET.SubElement(root, "route", route_attrib)

    target_route.append(_build_waypoints_element(exported_positions))

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
    target_route.append(_build_default_scenarios_from_positions(exported_positions))

    weathers_elem = src_route.find("weathers")
    if weathers_elem is not None:
        target_route.append(copy.deepcopy(weathers_elem))

    _indent_xml_compat(root)
    return root


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert_file(
    input_xml: Path,
    output_dir: Path,
    category: str,
    seed: Optional[int],
    max_distance_m: float,
    route_step_m: float,
    lane_change_prep_m: float,
    map_cache: CarlaMapCache,
    planner_cache: PlannerCache,
    speed_limit_resolver: OpenDriveSpeedLimitResolver,
    force_all_green_traffic_lights: bool,
    max_chain_depth: int = 5,
    min_chain_trigger_gap_m: float = 25.0,
) -> List[Path]:
    """Convert one Bench2Drive XML into per-action language benchmark XMLs.

    For each route in the input file:
        1. Rebuild a follow route from the start waypoint.
        2. Sample actionability and select a trigger.
        3. For each feasible action, build an action suffix, chain
           subsequent instructions, and write the output XML.

    Returns the list of written file paths.
    """
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

        # Deterministic per-route RNG.
        route_rng_seed = (
            f"{seed}:{input_xml.stem}:{src_id}:{route_index}"
            if seed is not None
            else f"{base_rng.random()}:{input_xml.stem}:{src_id}:{route_index}"
        )
        route_rng = random.Random(route_rng_seed)
        carla_map = map_cache.get_map(route_town)

        start_waypoint = _resolve_start_waypoint(source_positions, carla_map)

        # Sample and cap the acceleration speed.
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

        # Rebuild base route and find the best trigger.
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
            # Build the first action suffix.
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

            # Compute distance from trigger to action end.
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

            first_spec = InstructionSpec(
                action=action,
                trigger_distance_m=trigger.distance_m,
                duration_m=first_trigger_to_action_end_m,
                phrasing_mode=trigger.phrasing_mode,
                trigger_position=trigger.position,
            )

            # Build instruction chain from action end.
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
                if not tail_specs:
                    tail_waypoints = _truncate_waypoints_before_terminal_junction(
                        tail_waypoints
                    )

                action_segment = list(
                    suffix_result.waypoints[1: suffix_result.action_end_index + 1]
                )
                merged_waypoints = list(prefix_waypoints) + action_segment
                for wp in tail_waypoints[1:]:
                    _append_waypoint_if_new(merged_waypoints, wp)

                all_specs = [first_spec] + tail_specs
            else:
                # Single action, no chaining.
                merged_waypoints = list(prefix_waypoints)
                for wp in suffix_result.waypoints[1:]:
                    _append_waypoint_if_new(merged_waypoints, wp)
                first_spec = InstructionSpec(
                    action=action,
                    trigger_distance_m=trigger.distance_m,
                    duration_m=-1.0,
                    phrasing_mode=trigger.phrasing_mode,
                    trigger_position=trigger.position,
                )
                all_specs = [first_spec]

            # Ensure the last spec has duration_m=-1.
            if all_specs and all_specs[-1].duration_m != -1.0:
                last = all_specs[-1]
                all_specs[-1] = InstructionSpec(
                    action=last.action,
                    trigger_distance_m=last.trigger_distance_m,
                    duration_m=-1.0,
                    phrasing_mode=last.phrasing_mode,
                    trigger_position=last.trigger_position,
                )

            final_route = _finalize_route(merged_waypoints, route_step_m, resample=False)
            exported_positions = build_planner_safe_export_positions(
                src_route.attrib.get("town", "Town12"),
                merged_waypoints,
                planner_cache,
            )

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
                exported_positions=exported_positions,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: parse args, set up caches, convert files."""
    _require_carla()
    args = parse_args()
    input_path = args.input_xml.resolve()

    xodr_roots = (
        None if args.xodr_root is None else [p.expanduser().resolve() for p in args.xodr_root]
    )
    map_cache = CarlaMapCache(xodr_search_roots=xodr_roots)
    planner_cache = PlannerCache(map_cache)
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
            planner_cache=planner_cache,
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
            planner_cache=planner_cache,
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
