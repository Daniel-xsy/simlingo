"""
Language Benchmark Agent for SimLingo

This agent extends the base LingoAgent to support language instruction following
from the Language-Following Benchmark XML files.

Instead of using waypoint-based prompts, this agent reads instruction sequences
from the benchmark XML and provides them to the model based on trigger conditions.
"""

import os
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

import carla
import numpy as np

# Import base agent
from team_code.agent_simlingo import LingoAgent, get_entry_point as base_get_entry_point


def get_entry_point():
    return 'LanguageBenchmarkAgent'


class TriggerType(Enum):
    """Types of instruction triggers"""
    START = "start"
    DISTANCE_TRAVELED = "distance_traveled"
    DISTANCE_TO_POINT = "distance_to_point"
    SCENARIO_ACTIVE = "scenario_active"
    TIME_ELAPSED = "time_elapsed"


@dataclass
class InstructionTrigger:
    """Represents when an instruction should be activated"""
    trigger_type: TriggerType
    value: float = 0.0
    point_x: float = 0.0
    point_y: float = 0.0
    point_z: float = 0.0
    scenario_name: str = ""


@dataclass  
class Instruction:
    """A language instruction with trigger and expected behavior"""
    id: int
    text: str
    command_id: int
    trigger: InstructionTrigger
    duration_meters: float
    safety_critical: bool = False
    
    # Runtime state
    start_distance: float = 0.0
    is_active: bool = False


class LanguageBenchmarkAgent(LingoAgent):
    """
    Agent for Language-Following Benchmark evaluation.
    
    Extends LingoAgent to:
    - Parse language benchmark XML files
    - Track distance traveled and active instruction
    - Override prompts with language instructions instead of waypoints
    """
    
    def setup(self, path_to_conf_file, route_index=None):
        """Setup the agent and parse language benchmark instructions"""
        # Call parent setup
        super().setup(path_to_conf_file, route_index)
        
        # Force command-based evaluation mode
        # Use 'command' mode which generates command-based prompts
        # Our custom_prompt will override the actual command text
        self.config.eval_route_as = 'command'
        
        # Parse language benchmark XML
        self.instructions: List[Instruction] = []
        self.current_instruction: Optional[Instruction] = None
        self.distance_traveled = 0.0
        self.last_location: Optional[carla.Location] = None
        self.start_time: Optional[float] = None
        self._latest_game_time: Optional[float] = None
        self.benchmark_type = ""
        self.benchmark_category = ""
        self.scenario_var_by_name: Dict[str, str] = {}
        self.scenario_trigger_points: Dict[str, carla.Location] = {}
        self._scenario_defs: List[Tuple[str, Optional[carla.Location]]] = []
        
        # Parse the route file for instructions
        self._parse_language_benchmark()
        
        print(f"[LanguageBenchmarkAgent] Loaded {len(self.instructions)} instructions")
        for instr in self.instructions:
            print(f"  [{instr.id}] {instr.text} (trigger: {instr.trigger.trigger_type.value})")
    
    def _parse_language_benchmark(self):
        """Parse the language benchmark XML file"""
        route_path = os.environ.get('ROUTES', '')
        
        if not route_path or not os.path.exists(route_path):
            print(f"[LanguageBenchmarkAgent] Warning: Route file not found: {route_path}")
            return
        
        try:
            tree = ET.parse(route_path)
            root = tree.getroot()
            
            # Find the route element
            route = root.find('.//route')
            if route is None:
                print("[LanguageBenchmarkAgent] Warning: No route element found")
                return
            
            # Get benchmark metadata
            self.benchmark_type = route.attrib.get('benchmark_type', '')
            self.benchmark_category = route.attrib.get('category', '')
            
            # Parse instructions
            instructions_elem = route.find('instructions')
            if instructions_elem is None:
                print("[LanguageBenchmarkAgent] Warning: No instructions element found")
                return
            
            for instr_elem in instructions_elem.findall('instruction'):
                instruction = self._parse_instruction(instr_elem)
                if instruction:
                    self.instructions.append(instruction)
            
            # Sort by ID to ensure correct order
            self.instructions.sort(key=lambda x: x.id)

            # Parse scenarios for scenario-triggered instructions
            scenarios_elem = route.find('scenarios')
            if scenarios_elem is not None:
                self._parse_scenarios(scenarios_elem)
            
        except Exception as e:
            print(f"[LanguageBenchmarkAgent] Error parsing XML: {e}")
            import traceback
            traceback.print_exc()

    def _parse_scenarios(self, scenarios_elem: ET.Element):
        """Parse scenario definitions for scenario_active triggers"""
        self.scenario_var_by_name = {}
        self.scenario_trigger_points = {}
        self._scenario_defs = []

        scenario_index = 0
        for scenario_elem in scenarios_elem.findall('scenario'):
            scenario_name = scenario_elem.attrib.get('name', f"scenario_{scenario_index}")
            trigger_point_elem = scenario_elem.find('trigger_point')

            trigger_location = None
            if trigger_point_elem is not None:
                try:
                    trigger_location = carla.Location(
                        x=float(trigger_point_elem.attrib.get('x', 0)),
                        y=float(trigger_point_elem.attrib.get('y', 0)),
                        z=float(trigger_point_elem.attrib.get('z', 0))
                    )
                except Exception:
                    trigger_location = None

            self._scenario_defs.append((scenario_name, trigger_location))
            self.scenario_var_by_name[scenario_name] = f"ScenarioRouteNumber{scenario_index}"
            if trigger_location is not None:
                self.scenario_trigger_points[scenario_name] = trigger_location

            scenario_index += 1
    
    def _parse_instruction(self, elem: ET.Element) -> Optional[Instruction]:
        """Parse a single instruction element"""
        try:
            instr_id = int(elem.attrib.get('id', 0))
            
            # Parse trigger
            trigger_elem = elem.find('trigger')
            trigger = self._parse_trigger(trigger_elem)
            
            # Parse text
            text_elem = elem.find('text')
            text = text_elem.text if text_elem is not None else "follow the road"
            
            # Parse command ID
            command_id_elem = elem.find('command_id')
            command_id = int(command_id_elem.text) if command_id_elem is not None else 4
            
            # Parse duration
            duration_elem = elem.find('duration_meters')
            duration = float(duration_elem.text) if duration_elem is not None else -1.0
            
            # Parse safety critical flag
            safety_elem = elem.find('safety_critical')
            safety_critical = safety_elem is not None and safety_elem.text.lower() == 'true'
            
            return Instruction(
                id=instr_id,
                text=text,
                command_id=command_id,
                trigger=trigger,
                duration_meters=duration,
                safety_critical=safety_critical
            )
        except Exception as e:
            print(f"[LanguageBenchmarkAgent] Error parsing instruction: {e}")
            return None
    
    def _parse_trigger(self, elem: Optional[ET.Element]) -> InstructionTrigger:
        """Parse trigger element"""
        if elem is None:
            return InstructionTrigger(trigger_type=TriggerType.START)
        
        trigger_type_str = elem.attrib.get('type', 'start')
        try:
            trigger_type = TriggerType(trigger_type_str)
        except ValueError:
            trigger_type = TriggerType.START
        
        trigger = InstructionTrigger(trigger_type=trigger_type)
        
        if 'value' in elem.attrib:
            trigger.value = float(elem.attrib['value'])
        
        if trigger_type == TriggerType.DISTANCE_TO_POINT:
            trigger.point_x = float(elem.attrib.get('x', 0))
            trigger.point_y = float(elem.attrib.get('y', 0))
            trigger.point_z = float(elem.attrib.get('z', 0))
        
        if trigger_type == TriggerType.SCENARIO_ACTIVE:
            trigger.scenario_name = elem.attrib.get('scenario_name', '')
        
        return trigger
    
    def _update_distance_traveled(self, current_location: carla.Location):
        """Update the total distance traveled"""
        if self.last_location is not None:
            delta = math.sqrt(
                (current_location.x - self.last_location.x) ** 2 +
                (current_location.y - self.last_location.y) ** 2
            )
            self.distance_traveled += delta
        self.last_location = current_location
    
    def _get_active_instruction(self, current_location: carla.Location, 
                                 game_time: float) -> Optional[Instruction]:
        """Determine which instruction is currently active"""
        active_instruction = None
        
        for instruction in self.instructions:
            trigger = instruction.trigger
            is_triggered = False
            
            # Check trigger condition
            if trigger.trigger_type == TriggerType.START:
                is_triggered = True
                
            elif trigger.trigger_type == TriggerType.DISTANCE_TRAVELED:
                is_triggered = self.distance_traveled >= trigger.value
                
            elif trigger.trigger_type == TriggerType.DISTANCE_TO_POINT:
                dist = math.sqrt(
                    (current_location.x - trigger.point_x) ** 2 +
                    (current_location.y - trigger.point_y) ** 2
                )
                is_triggered = dist <= trigger.value
                
            elif trigger.trigger_type == TriggerType.TIME_ELAPSED:
                if self.start_time is not None:
                    elapsed = game_time - self.start_time
                    is_triggered = elapsed >= trigger.value

            elif trigger.trigger_type == TriggerType.SCENARIO_ACTIVE:
                is_triggered = self._is_scenario_active(trigger, current_location)
            
            # Check duration constraints
            if is_triggered:
                if not instruction.is_active:
                    # First time this instruction is triggered
                    instruction.is_active = True
                    instruction.start_distance = self.distance_traveled
                
                # Check if still within duration
                if instruction.duration_meters > 0:
                    end_distance = instruction.start_distance + instruction.duration_meters
                    if self.distance_traveled > end_distance:
                        is_triggered = False
            
            if is_triggered:
                active_instruction = instruction
        
        return active_instruction

    def _is_scenario_active(self, trigger: InstructionTrigger, ego_location: carla.Location) -> bool:
        """Check if a scenario is currently active (via blackboard or actor presence)."""
        scenario_name = trigger.scenario_name
        if not scenario_name:
            return False

        # Prefer the blackboard variable set by ScenarioTriggerer
        var_name = None
        if scenario_name.startswith("ScenarioRouteNumber"):
            var_name = scenario_name
        else:
            var_name = self.scenario_var_by_name.get(scenario_name)

        if var_name:
            try:
                import py_trees
                blackboard = py_trees.blackboard.Blackboard()
                value = blackboard.get(var_name)
                if bool(value):
                    return True
            except Exception:
                pass

        # Fallback: detect scenario actor activation (best effort)
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            world = CarlaDataProvider.get_world()
        except Exception:
            world = None

        if world is None:
            return False

        actors = world.get_actors().filter("vehicle.*")
        if not actors:
            return False

        trigger_location = self.scenario_trigger_points.get(scenario_name)
        max_distance = trigger.value if trigger.value > 0 else 30.0

        for actor in actors:
            role_name = actor.attributes.get("role_name", "")
            if role_name != "scenario":
                continue

            actor_location = actor.get_location()
            if actor_location is None:
                continue

            # Ignore actors still hidden below ground
            if actor_location.z < -10.0:
                continue

            if trigger_location is not None:
                if actor_location.distance(trigger_location) <= max_distance:
                    return True

            if ego_location is not None and actor_location.distance(ego_location) <= max_distance:
                return True

        return False
    
    def _get_prompt_for_instruction(self, instruction: Optional[Instruction], 
                                     speed: float) -> str:
        """Generate the prompt for the current instruction"""
        if instruction is None:
            # Default instruction
            instruction_text = "follow the road"
        else:
            instruction_text = instruction.text
        
        # Format as command prompt
        prompt_tp = f"Command: {instruction_text}."
        
        if self.config.use_cot:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
        else:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} Predict the waypoints."
        
        # Add instruction following tag for safety-critical instructions
        if instruction is not None and instruction.safety_critical:
            prompt = f"<INSTRUCTION_FOLLOWING> {prompt}"
        
        return prompt
    
    def tick(self, input_data):
        """
        Override tick to track distance and update instructions.
        
        We need to update distance and instructions BEFORE the parent's tick()
        generates the prompt, so we extract GPS early.
        """
        # Extract GPS position early (before parent's tick)
        # This replicates the GPS processing from parent's tick method
        gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
        current_location = carla.Location(x=float(gps_pos[0]), y=float(gps_pos[1]), z=float(gps_pos[2]))
        
        # Update distance traveled
        self._update_distance_traveled(current_location)
        
        # Get active instruction based on current distance, position, and real game time
        if self._latest_game_time is not None:
            game_time = self._latest_game_time
        else:
            game_time = self.step * self.config.carla_frame_rate if self.step > 0 else 0.0

        if self.start_time is None:
            self.start_time = game_time
        self.current_instruction = self._get_active_instruction(current_location, game_time)
        
        # Set custom_prompt and user_flag for this instruction
        # This must happen BEFORE parent's tick() generates the prompt
        instruction = self.current_instruction
        if instruction is not None:
            self.custom_prompt = f"{instruction.text}."
            self.user_flag = 3  # Use custom prompt without special tag
        else:
            self.custom_prompt = "Command: follow the road."
            self.user_flag = 3
        
        # Log instruction changes and distance
        if self.step % 20 == 0:  # Log every 20 steps
            instr_text = instruction.text if instruction else "None"
            print(f"[LanguageBenchmarkAgent] Step {self.step}, Distance: {self.distance_traveled:.1f}m, "
                  f"Location: ({current_location.x:.1f}, {current_location.y:.1f}), "
                  f"Instruction: '{instr_text}'")
        
        # Now call parent tick - it will use the custom_prompt we just set
        result = super().tick(input_data)
        
        return result
    
    def run_step(self, input_data, timestamp, sensors=None):
        """
        Execute one step with language benchmark instructions.
        
        The instruction update and prompt setting happens in tick().
        """
        # Record latest game time for time-based triggers
        self._latest_game_time = timestamp

        # Call parent run_step (which calls tick internally)
        return super().run_step(input_data, timestamp, sensors)


# Add torch import that parent uses
import torch
