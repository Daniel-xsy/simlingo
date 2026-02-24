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
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

# Targets for precise speed instructions.
ACCELERATE_TARGET_SPEEDS = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
DECELERATE_TARGET_SPEEDS = [0, 2, 4, 6, 8]


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
            "Output XML path. Default: "
            "leaderboard/data/language_benchmark/instruction_following/"
            "<input_stem>_language_distance.xml"
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
        default=3,
        help=(
            "Number of combined instructions per route. Example: 3 means "
            "[0-step), [step-2*step), [2*step-end)."
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


def _sample_precise_speed_instruction(rng: random.Random, is_accelerate: bool) -> Dict[str, object]:
    if is_accelerate:
        target = rng.choice(ACCELERATE_TARGET_SPEEDS)
        text = rng.choice(
            [
                f"accelerate to {target} m/s",
                f"set your speed to {target} m/s",
                f"reach {target} m/s",
            ]
        )
    else:
        target = rng.choice(DECELERATE_TARGET_SPEEDS)
        text = rng.choice(
            [
                f"decelerate to {target} m/s",
                f"reduce speed to {target} m/s",
                f"slow down to {target} m/s",
            ]
        )

    return {
        "text": text,
        "command_id": -1,
        "expected_behavior": {
            "type": "target_speed",
            "speed_ms": str(target),
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
            return _sample_precise_speed_instruction(rng, is_accelerate=True)
        return (
            _sample_precise_speed_instruction(rng, is_accelerate=True)
            if rng.random() < 0.5
            else {
                "text": rng.choice(INSTRUCTION_LIBRARY["accelerate_vague"]["texts"]),
                "command_id": INSTRUCTION_LIBRARY["accelerate_vague"]["command_id"],
                "expected_behavior": dict(
                    INSTRUCTION_LIBRARY["accelerate_vague"]["expected_behavior"]
                ),
            }
        )

    if category == "decelerate":
        if style == "vague":
            entry = INSTRUCTION_LIBRARY["decelerate_vague"]
            return {
                "text": rng.choice(entry["texts"]),
                "command_id": entry["command_id"],
                "expected_behavior": dict(entry["expected_behavior"]),
            }
        if style == "precise":
            return _sample_precise_speed_instruction(rng, is_accelerate=False)
        return (
            _sample_precise_speed_instruction(rng, is_accelerate=False)
            if rng.random() < 0.5
            else {
                "text": rng.choice(INSTRUCTION_LIBRARY["decelerate_vague"]["texts"]),
                "command_id": INSTRUCTION_LIBRARY["decelerate_vague"]["command_id"],
                "expected_behavior": dict(
                    INSTRUCTION_LIBRARY["decelerate_vague"]["expected_behavior"]
                ),
            }
        )

    entry = INSTRUCTION_LIBRARY[category]
    return {
        "text": rng.choice(entry["texts"]),
        "command_id": entry["command_id"],
        "expected_behavior": dict(entry["expected_behavior"]),
    }


def _sample_navigation_instruction(rng: random.Random) -> Dict[str, object]:
    """Sample only non-speed navigation actions for middle instruction slots."""
    category = rng.choice(
        [
            "lane_follow",
            "lane_change_left",
            "lane_change_right",
            "turn_left",
            "turn_right",
            "turn_straight",
        ]
    )
    entry = INSTRUCTION_LIBRARY[category]
    return {
        "text": rng.choice(entry["texts"]),
        "command_id": entry["command_id"],
        "expected_behavior": dict(entry["expected_behavior"]),
    }


def _build_instructions(
    route_elem: ET.Element,
    trigger_step_m: float,
    rng: random.Random,
    num_instructions: int,
    instruction_style: str,
) -> ET.Element:
    _ = instruction_style  # Reserved for future extensions.
    waypoints = route_elem.find("waypoints")
    if waypoints is None:
        raise ValueError(f"Route {route_elem.attrib.get('id', 'unknown')} has no <waypoints>.")

    route_positions = _get_waypoint_positions(waypoints)
    total_length = _route_length_m(route_positions)

    instructions_elem = ET.Element("instructions")

    # Create sequential distance triggers: 0, step, 2*step, ...
    if trigger_step_m <= 0:
        raise ValueError("--trigger-step-m must be > 0.")
    if num_instructions < 1:
        raise ValueError("--num-instructions must be >= 1.")

    trigger_distances = [float(i) * trigger_step_m for i in range(num_instructions)]
    max_trigger = max(total_length - 1.0, 0.0)
    trigger_distances = [min(d, max_trigger) for d in trigger_distances]

    # Structured template (fixed):
    # 1) accelerate to target speed
    # 2) one navigation instruction
    # 3) decelerate to target speed
    if num_instructions != 3:
        raise ValueError("--num-instructions must be 3 for accelerate->navigate->decelerate.")

    for idx, dist_m in enumerate(trigger_distances, start=1):
        if idx == 1:
            template = _sample_precise_speed_instruction(rng, is_accelerate=True)
            _append_instruction(
                instructions_elem,
                instruction_id=idx,
                trigger_distance_m=dist_m,
                template=template,
                duration_meters=trigger_step_m,
            )
        elif idx == len(trigger_distances):
            template = _sample_precise_speed_instruction(rng, is_accelerate=False)
            _append_instruction(
                instructions_elem,
                instruction_id=idx,
                trigger_distance_m=dist_m,
                template=template,
                duration_meters=-1.0,
            )
        else:
            template = _sample_navigation_instruction(rng)
            _append_instruction(
                instructions_elem,
                instruction_id=idx,
                trigger_distance_m=dist_m,
                template=template,
                duration_meters=trigger_step_m,
            )

    return instructions_elem


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


def convert_file(
    input_xml: Path,
    output_xml: Path,
    category: str,
    trigger_step_m: float,
    seed: Optional[int],
    num_instructions: int,
    instruction_style: str,
) -> None:
    tree = ET.parse(input_xml)
    source_root = tree.getroot()

    if source_root.tag != "routes":
        raise ValueError(f"Expected root tag <routes>, found <{source_root.tag}>.")

    rng = random.Random() if seed is None else random.Random(seed)
    target_root = ET.Element("routes")

    for src_route in source_root.findall("route"):
        src_id = src_route.attrib.get("id", "unknown")
        target_route = ET.SubElement(
            target_root,
            "route",
            {
                "id": f"{src_id}_LANG_DISTANCE",
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
            _build_instructions(
                src_route,
                trigger_step_m,
                rng,
                num_instructions=num_instructions,
                instruction_style=instruction_style,
            )
        )
        target_route.append(_build_default_evaluation())
        target_route.append(_build_default_scenarios(src_route))

        weathers_elem = src_route.find("weathers")
        if weathers_elem is not None:
            target_route.append(copy.deepcopy(weathers_elem))

    _indent_xml_compat(target_root)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(target_root).write(output_xml, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    input_path = args.input_xml.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = (
        repo_root / "leaderboard" / "data" / "language_benchmark" / "instruction_following"
    )

    if input_path.is_file():
        if args.output is None:
            output_xml = default_output_dir / f"{input_path.stem}_language_distance.xml"
        else:
            output_xml = args.output.resolve()

        convert_file(
            input_xml=input_path,
            output_xml=output_xml,
            category=args.category,
            trigger_step_m=args.trigger_step_m,
            seed=args.seed,
            num_instructions=args.num_instructions,
            instruction_style=args.instruction_style,
        )
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
        output_xml = output_dir / f"{input_xml.stem}_language_distance.xml"
        convert_file(
            input_xml=input_xml,
            output_xml=output_xml,
            category=args.category,
            trigger_step_m=args.trigger_step_m,
            seed=args.seed,
            num_instructions=args.num_instructions,
            instruction_style=args.instruction_style,
        )
        print(f"Generated language benchmark XML: {output_xml}")


if __name__ == "__main__":
    main()
