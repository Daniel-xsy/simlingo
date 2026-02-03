#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Parser for Language-Following Benchmark XML files.

Extends the Bench2Drive route parser to handle instruction-based routes
with dynamic language commands and evaluation criteria.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

import carla

from leaderboard.utils.route_parser import RouteParser, convert_elem_to_transform
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData


class TriggerType(Enum):
    """Types of instruction triggers"""
    START = "start"
    DISTANCE_TRAVELED = "distance_traveled"
    DISTANCE_TO_POINT = "distance_to_point"
    SCENARIO_ACTIVE = "scenario_active"
    TIME_ELAPSED = "time_elapsed"


class BehaviorType(Enum):
    """Types of expected behaviors"""
    LANE_CHANGE = "lane_change"
    LANE_FOLLOW = "lane_follow"
    TURN = "turn"
    TARGET_SPEED = "target_speed"
    STOP = "stop"
    ACCELERATE = "accelerate"
    MAINTAIN_SPEED = "maintain_speed"


@dataclass
class InstructionTrigger:
    """Represents when an instruction should be activated"""
    trigger_type: TriggerType
    value: Optional[float] = None  # Distance in meters, time in seconds
    point: Optional[carla.Location] = None  # For distance_to_point
    scenario_name: Optional[str] = None  # For scenario_active


@dataclass
class ExpectedBehavior:
    """Represents the expected agent behavior for an instruction"""
    behavior_type: BehaviorType
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Instruction:
    """A single language instruction with trigger and expected behavior"""
    id: int
    priority: str
    trigger: InstructionTrigger
    text: str
    command_id: int
    expected_behavior: ExpectedBehavior
    duration_meters: float
    safety_critical: bool = False


@dataclass
class EvaluationMetric:
    """A metric for evaluating instruction following"""
    metric_type: str
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class LanguageRouteConfiguration(RouteScenarioConfiguration):
    """Extended route configuration with language instructions"""
    benchmark_type: str = "language_following"
    category: str = ""  # lateral_control, speed_control, unsafe_command
    instructions: List[Instruction] = field(default_factory=list)
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=list)
    disable_bg_vehicle: bool = False  # If True, disable background traffic


class LanguageBenchmarkParser(RouteParser):
    """
    Parser for Language-Following Benchmark XML files.
    
    Extends RouteParser to handle the additional <instructions> and 
    <evaluation> elements specific to the language benchmark.
    """
    
    @staticmethod
    def parse_routes_file(route_filename: str, routes_subset: str = '') -> List[LanguageRouteConfiguration]:
        """
        Parse a language benchmark XML file.
        
        Args:
            route_filename: Path to the XML file
            routes_subset: Optional subset specification (same as Bench2Drive)
            
        Returns:
            List of LanguageRouteConfiguration objects
        """
        route_configs = []
        tree = ET.parse(route_filename)
        
        for route in tree.iter("route"):
            route_id = route.attrib['id']
            
            # Create extended configuration
            route_config = LanguageRouteConfiguration()
            route_config.town = route.attrib['town']
            route_config.name = f"LanguageRoute_{route_id}"
            
            # Parse benchmark-specific attributes
            route_config.benchmark_type = route.attrib.get('benchmark_type', 'language_following')
            route_config.category = route.attrib.get('category', '')
            route_config.disable_bg_vehicle = route.attrib.get('disable_bg_vehicle', 'false').lower() == 'true'
            
            # Parse weather (using parent class method)
            route_config.weather = LanguageBenchmarkParser.parse_weather(route)
            
            # Parse waypoints
            waypoints_elem = route.find('waypoints')
            if waypoints_elem is not None:
                positions = []
                for position in waypoints_elem.iter('position'):
                    positions.append(carla.Location(
                        x=float(position.attrib['x']),
                        y=float(position.attrib['y']),
                        z=float(position.attrib['z'])
                    ))
                route_config.keypoints = positions
            
            # Parse instructions (new for language benchmark)
            instructions_elem = route.find('instructions')
            if instructions_elem is not None:
                route_config.instructions = LanguageBenchmarkParser._parse_instructions(instructions_elem)
            
            # Parse evaluation metrics (new for language benchmark)
            evaluation_elem = route.find('evaluation')
            if evaluation_elem is not None:
                route_config.evaluation_metrics = LanguageBenchmarkParser._parse_evaluation(evaluation_elem)
            
            # Parse scenarios (same as Bench2Drive)
            scenarios_elem = route.find('scenarios')
            if scenarios_elem is not None:
                scenario_configs = []
                for scenario in scenarios_elem.iter('scenario'):
                    scenario_config = ScenarioConfiguration()
                    scenario_config.name = scenario.attrib.get('name')
                    scenario_config.type = scenario.attrib.get('type')
                    
                    for elem in list(scenario):
                        if elem.tag == 'trigger_point':
                            scenario_config.trigger_points.append(convert_elem_to_transform(elem))
                        elif elem.tag == 'other_actor':
                            scenario_config.other_actors.append(
                                ActorConfigurationData.parse_from_node(elem, 'scenario')
                            )
                        else:
                            scenario_config.other_parameters[elem.tag] = elem.attrib
                    
                    scenario_configs.append(scenario_config)
                route_config.scenario_configs = scenario_configs
            
            route_configs.append(route_config)
        
        return route_configs
    
    @staticmethod
    def _parse_instructions(instructions_elem: ET.Element) -> List[Instruction]:
        """Parse the <instructions> element"""
        instructions = []
        
        for instr_elem in instructions_elem.iter('instruction'):
            # Parse trigger
            trigger_elem = instr_elem.find('trigger')
            trigger = LanguageBenchmarkParser._parse_trigger(trigger_elem)
            
            # Parse expected behavior
            behavior_elem = instr_elem.find('expected_behavior')
            expected_behavior = LanguageBenchmarkParser._parse_expected_behavior(behavior_elem)
            
            # Parse text and other attributes
            text_elem = instr_elem.find('text')
            text = text_elem.text if text_elem is not None else ""
            
            command_id_elem = instr_elem.find('command_id')
            command_id = int(command_id_elem.text) if command_id_elem is not None else -1
            
            duration_elem = instr_elem.find('duration_meters')
            duration = float(duration_elem.text) if duration_elem is not None else -1.0
            
            safety_elem = instr_elem.find('safety_critical')
            safety_critical = safety_elem is not None and safety_elem.text.lower() == 'true'
            
            instruction = Instruction(
                id=int(instr_elem.attrib.get('id', 0)),
                priority=instr_elem.attrib.get('priority', 'primary'),
                trigger=trigger,
                text=text,
                command_id=command_id,
                expected_behavior=expected_behavior,
                duration_meters=duration,
                safety_critical=safety_critical
            )
            instructions.append(instruction)
        
        return instructions
    
    @staticmethod
    def _parse_trigger(trigger_elem: ET.Element) -> InstructionTrigger:
        """Parse an instruction trigger element"""
        if trigger_elem is None:
            return InstructionTrigger(trigger_type=TriggerType.START)
        
        trigger_type_str = trigger_elem.attrib.get('type', 'start')
        trigger_type = TriggerType(trigger_type_str)
        
        trigger = InstructionTrigger(trigger_type=trigger_type)
        
        if trigger_type in [TriggerType.DISTANCE_TRAVELED, TriggerType.TIME_ELAPSED]:
            trigger.value = float(trigger_elem.attrib.get('value', 0))
        
        elif trigger_type == TriggerType.DISTANCE_TO_POINT:
            trigger.value = float(trigger_elem.attrib.get('value', 0))
            trigger.point = carla.Location(
                x=float(trigger_elem.attrib.get('x', 0)),
                y=float(trigger_elem.attrib.get('y', 0)),
                z=float(trigger_elem.attrib.get('z', 0))
            )
        
        elif trigger_type == TriggerType.SCENARIO_ACTIVE:
            trigger.scenario_name = trigger_elem.attrib.get('scenario_name', '')
        
        return trigger
    
    @staticmethod
    def _parse_expected_behavior(behavior_elem: ET.Element) -> ExpectedBehavior:
        """Parse expected behavior element"""
        if behavior_elem is None:
            return ExpectedBehavior(behavior_type=BehaviorType.LANE_FOLLOW)
        
        behavior_type_str = behavior_elem.attrib.get('type', 'lane_follow')
        behavior_type = BehaviorType(behavior_type_str)
        
        # Collect all attributes as parameters
        parameters = {k: v for k, v in behavior_elem.attrib.items() if k != 'type'}
        
        return ExpectedBehavior(
            behavior_type=behavior_type,
            parameters=parameters
        )
    
    @staticmethod
    def _parse_evaluation(evaluation_elem: ET.Element) -> List[EvaluationMetric]:
        """Parse the <evaluation> element"""
        metrics = []
        
        for metric_elem in evaluation_elem.iter('metric'):
            metric_type = metric_elem.attrib.get('type', '')
            
            # Collect parameters
            parameters = {}
            for param_elem in metric_elem.iter('param'):
                param_name = param_elem.attrib.get('name', '')
                param_value = param_elem.attrib.get('value', '')
                parameters[param_name] = param_value
            
            metric = EvaluationMetric(
                metric_type=metric_type,
                parameters=parameters
            )
            metrics.append(metric)
        
        return metrics
    
    @staticmethod
    def get_active_instruction(instructions: List[Instruction], 
                               distance_traveled: float,
                               ego_location: carla.Location,
                               time_elapsed: float,
                               active_scenarios: List[str]) -> Optional[Instruction]:
        """
        Get the currently active instruction based on trigger conditions.
        
        Args:
            instructions: List of instructions from the route config
            distance_traveled: Distance traveled since route start (meters)
            ego_location: Current ego vehicle location
            time_elapsed: Time since route start (seconds)
            active_scenarios: List of currently active scenario names
            
        Returns:
            The currently active instruction, or None
        """
        active_instruction = None
        
        for instruction in instructions:
            trigger = instruction.trigger
            is_triggered = False
            
            if trigger.trigger_type == TriggerType.START:
                is_triggered = True
                
            elif trigger.trigger_type == TriggerType.DISTANCE_TRAVELED:
                is_triggered = distance_traveled >= trigger.value
                
            elif trigger.trigger_type == TriggerType.DISTANCE_TO_POINT:
                if trigger.point is not None:
                    dist = ego_location.distance(trigger.point)
                    is_triggered = dist <= trigger.value
                    
            elif trigger.trigger_type == TriggerType.TIME_ELAPSED:
                is_triggered = time_elapsed >= trigger.value
                
            elif trigger.trigger_type == TriggerType.SCENARIO_ACTIVE:
                is_triggered = trigger.scenario_name in active_scenarios
            
            # Check if within duration (if specified)
            if is_triggered and instruction.duration_meters > 0:
                # For distance-based triggers, check if we're still within duration
                if trigger.trigger_type == TriggerType.DISTANCE_TRAVELED:
                    trigger_distance = trigger.value
                    end_distance = trigger_distance + instruction.duration_meters
                    if distance_traveled > end_distance:
                        is_triggered = False
            
            if is_triggered:
                # Later instructions override earlier ones (sequential priority)
                active_instruction = instruction
        
        return active_instruction


def get_instruction_text_for_agent(instruction: Instruction) -> str:
    """
    Format the instruction text for the agent's prompt.
    
    Args:
        instruction: The active instruction
        
    Returns:
        Formatted instruction text for the agent
    """
    # Map instruction to agent-compatible format
    text = instruction.text
    
    # Add "Command: " prefix to match agent's expected format
    return f"Command: {text}."


# Utility function for converting instructions to agent format
def instruction_to_command_id(instruction: Instruction) -> int:
    """
    Map an instruction to the standard command ID if possible.
    
    The standard command IDs are:
        1: go left at the next intersection
        2: go right at the next intersection
        3: go straight at the next intersection
        4: follow the road
        5: do a lane change to the left
        6: do a lane change to the right
        
    Returns:
        Command ID (1-6) or -1 for custom instructions
    """
    return instruction.command_id


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "leaderboard/data/language_benchmark/lateral_control/lane_change_left_001.xml"
    
    try:
        configs = LanguageBenchmarkParser.parse_routes_file(filename)
        
        for config in configs:
            print(f"Route: {config.name}")
            print(f"  Town: {config.town}")
            print(f"  Category: {config.category}")
            print(f"  Waypoints: {len(config.keypoints)}")
            print(f"  Instructions: {len(config.instructions)}")
            
            for instr in config.instructions:
                print(f"    [{instr.id}] {instr.text}")
                print(f"        Trigger: {instr.trigger.trigger_type.value}")
                print(f"        Expected: {instr.expected_behavior.behavior_type.value}")
            
            print(f"  Evaluation metrics: {len(config.evaluation_metrics)}")
            for metric in config.evaluation_metrics:
                print(f"    - {metric.metric_type}: {metric.parameters}")
                
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        import traceback
        traceback.print_exc()
