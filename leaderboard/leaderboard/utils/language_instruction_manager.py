#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Language Instruction Manager for the Language-Following Benchmark.

This module provides a manager class that tracks instruction state during 
evaluation and provides the appropriate instruction text to the agent.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import carla

from leaderboard.utils.language_benchmark_parser import (
    LanguageRouteConfiguration,
    Instruction,
    InstructionTrigger,
    ExpectedBehavior,
    TriggerType,
    BehaviorType,
    EvaluationMetric,
)


@dataclass
class InstructionState:
    """Tracks the state of instruction execution"""
    instruction: Instruction
    start_distance: float = 0.0
    start_time: float = 0.0
    start_location: Optional[carla.Location] = None
    end_distance: float = float('inf')
    is_active: bool = False
    compliance_samples: List[bool] = field(default_factory=list)


@dataclass
class EvaluationState:
    """Tracks evaluation metrics during the run"""
    distance_traveled: float = 0.0
    time_elapsed: float = 0.0
    speed_samples: List[float] = field(default_factory=list)
    lateral_position_samples: List[float] = field(default_factory=list)
    acceleration_samples: List[float] = field(default_factory=list)
    collisions: List[Dict[str, Any]] = field(default_factory=list)
    instruction_history: List[Tuple[float, int, str]] = field(default_factory=list)
    instruction_states: Dict[int, InstructionState] = field(default_factory=dict)


class LanguageInstructionManager:
    """
    Manages language instructions during Language-Following Benchmark evaluation.
    
    This class:
    - Tracks which instruction is currently active based on triggers
    - Provides the current instruction text for the agent's prompt
    - Records data for evaluation metrics
    """
    
    def __init__(self, route_config: LanguageRouteConfiguration):
        """
        Initialize the instruction manager.
        
        Args:
            route_config: Parsed language benchmark route configuration
        """
        self.route_config = route_config
        self.instructions = route_config.instructions
        self.evaluation_metrics = route_config.evaluation_metrics
        
        # State tracking
        self.eval_state = EvaluationState()
        self.current_instruction: Optional[Instruction] = None
        self.previous_instruction: Optional[Instruction] = None
        
        # Scenario tracking (for scenario-triggered instructions)
        self.active_scenarios: List[str] = []
        
        # Initialize instruction states
        for instr in self.instructions:
            self.eval_state.instruction_states[instr.id] = InstructionState(
                instruction=instr
            )
        
        # Reference positions for distance calculations
        self.start_location: Optional[carla.Location] = None
        self.previous_location: Optional[carla.Location] = None
        
        # Lane tracking for lateral control evaluation
        self.initial_lateral_offset: Optional[float] = None
        
    def reset(self):
        """Reset the manager state for a new run"""
        self.eval_state = EvaluationState()
        self.current_instruction = None
        self.previous_instruction = None
        self.active_scenarios = []
        self.start_location = None
        self.previous_location = None
        self.initial_lateral_offset = None
        
        for instr in self.instructions:
            self.eval_state.instruction_states[instr.id] = InstructionState(
                instruction=instr
            )
    
    def update(self, 
               ego_location: carla.Location,
               ego_velocity: carla.Vector3D,
               ego_transform: carla.Transform,
               timestamp: float,
               waypoint: Optional[carla.Waypoint] = None) -> str:
        """
        Update the manager state and return the current instruction text.
        
        Args:
            ego_location: Current ego vehicle location
            ego_velocity: Current ego vehicle velocity
            ego_transform: Current ego vehicle transform
            timestamp: Current simulation timestamp (seconds)
            waypoint: Current waypoint from CARLA map (optional, for lane tracking)
            
        Returns:
            Current instruction text formatted for the agent
        """
        # Initialize start location on first update
        if self.start_location is None:
            self.start_location = ego_location
            self.previous_location = ego_location
        
        # Calculate distance traveled
        if self.previous_location is not None:
            delta_distance = ego_location.distance(self.previous_location)
            self.eval_state.distance_traveled += delta_distance
        self.previous_location = ego_location
        
        # Update time
        self.eval_state.time_elapsed = timestamp
        
        # Record speed
        speed = math.sqrt(
            ego_velocity.x ** 2 + 
            ego_velocity.y ** 2 + 
            ego_velocity.z ** 2
        )
        self.eval_state.speed_samples.append(speed)
        
        # Record lateral position if waypoint available
        if waypoint is not None:
            lateral_offset = self._calculate_lateral_offset(ego_location, waypoint)
            if self.initial_lateral_offset is None:
                self.initial_lateral_offset = lateral_offset
            self.eval_state.lateral_position_samples.append(lateral_offset)
        
        # Determine active instruction
        self.previous_instruction = self.current_instruction
        self.current_instruction = self._get_active_instruction(ego_location)
        
        # Record instruction change
        if self.current_instruction != self.previous_instruction and self.current_instruction is not None:
            self.eval_state.instruction_history.append((
                self.eval_state.distance_traveled,
                self.current_instruction.id,
                self.current_instruction.text
            ))
            
            # Mark instruction as active and record start state
            state = self.eval_state.instruction_states[self.current_instruction.id]
            state.is_active = True
            state.start_distance = self.eval_state.distance_traveled
            state.start_time = timestamp
            state.start_location = ego_location
            
            if self.current_instruction.duration_meters > 0:
                state.end_distance = (
                    state.start_distance + self.current_instruction.duration_meters
                )
        
        # Get instruction text
        if self.current_instruction is not None:
            return self._format_instruction_text(self.current_instruction)
        else:
            # Default instruction if none active
            return "Command: follow the road."
    
    def _get_active_instruction(self, ego_location: carla.Location) -> Optional[Instruction]:
        """Determine which instruction is currently active"""
        active_instruction = None
        
        for instruction in self.instructions:
            trigger = instruction.trigger
            is_triggered = False
            
            # Check trigger condition
            if trigger.trigger_type == TriggerType.START:
                is_triggered = True
                
            elif trigger.trigger_type == TriggerType.DISTANCE_TRAVELED:
                is_triggered = self.eval_state.distance_traveled >= trigger.value
                
            elif trigger.trigger_type == TriggerType.DISTANCE_TO_POINT:
                if trigger.point is not None:
                    dist = ego_location.distance(trigger.point)
                    is_triggered = dist <= trigger.value
                    
            elif trigger.trigger_type == TriggerType.TIME_ELAPSED:
                is_triggered = self.eval_state.time_elapsed >= trigger.value
                
            elif trigger.trigger_type == TriggerType.SCENARIO_ACTIVE:
                is_triggered = trigger.scenario_name in self.active_scenarios
            
            # Check duration constraints
            if is_triggered and instruction.duration_meters > 0:
                state = self.eval_state.instruction_states[instruction.id]
                if state.is_active:
                    if self.eval_state.distance_traveled > state.end_distance:
                        is_triggered = False
            
            if is_triggered:
                active_instruction = instruction
        
        return active_instruction
    
    def _format_instruction_text(self, instruction: Instruction) -> str:
        """Format instruction for agent prompt"""
        return f"Command: {instruction.text}."
    
    def _calculate_lateral_offset(self, 
                                   ego_location: carla.Location,
                                   waypoint: carla.Waypoint) -> float:
        """Calculate lateral offset from lane center"""
        # Get waypoint transform
        wp_location = waypoint.transform.location
        wp_rotation = waypoint.transform.rotation
        
        # Calculate vector from waypoint to ego
        dx = ego_location.x - wp_location.x
        dy = ego_location.y - wp_location.y
        
        # Get right vector of the lane
        yaw_rad = math.radians(wp_rotation.yaw)
        right_x = math.cos(yaw_rad + math.pi / 2)
        right_y = math.sin(yaw_rad + math.pi / 2)
        
        # Project onto right vector
        lateral_offset = dx * right_x + dy * right_y
        
        return lateral_offset
    
    def set_scenario_active(self, scenario_name: str):
        """Mark a scenario as active (called by scenario manager)"""
        if scenario_name not in self.active_scenarios:
            self.active_scenarios.append(scenario_name)
    
    def set_scenario_inactive(self, scenario_name: str):
        """Mark a scenario as inactive"""
        if scenario_name in self.active_scenarios:
            self.active_scenarios.remove(scenario_name)
    
    def record_collision(self, collision_info: Dict[str, Any]):
        """Record a collision event"""
        collision_info['distance'] = self.eval_state.distance_traveled
        collision_info['time'] = self.eval_state.time_elapsed
        collision_info['instruction_id'] = (
            self.current_instruction.id if self.current_instruction else None
        )
        self.eval_state.collisions.append(collision_info)
    
    def get_current_command_id(self) -> int:
        """Get the command ID for the current instruction"""
        if self.current_instruction is not None:
            return self.current_instruction.command_id
        return 4  # Default: follow the road
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics based on collected data.
        
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        for metric in self.evaluation_metrics:
            if metric.metric_type == "lane_change_success":
                metrics["lane_change_success"] = self._compute_lane_change_success(metric)
            
            elif metric.metric_type == "speed_compliance":
                metrics["speed_compliance"] = self._compute_speed_compliance(metric)
            
            elif metric.metric_type == "instruction_compliance":
                metrics["instruction_compliance"] = self._compute_instruction_compliance(metric)
            
            elif metric.metric_type == "collision_check":
                metrics["collision_occurred"] = len(self.eval_state.collisions) > 0
                metrics["collision_count"] = len(self.eval_state.collisions)
            
            elif metric.metric_type == "lane_follow_stability":
                metrics["lane_follow_stability"] = self._compute_lane_stability(metric)
            
            elif metric.metric_type == "safety_compliance_tradeoff":
                metrics["safety_compliance_tradeoff"] = self._compute_safety_tradeoff()
        
        # Add general metrics
        metrics["total_distance"] = self.eval_state.distance_traveled
        metrics["total_time"] = self.eval_state.time_elapsed
        metrics["instruction_changes"] = len(self.eval_state.instruction_history)
        
        return metrics
    
    def _compute_lane_change_success(self, metric: EvaluationMetric) -> Dict[str, Any]:
        """Compute lane change success metrics"""
        expected_direction = metric.parameters.get("expected_direction", "left")
        min_lateral_change = float(metric.parameters.get("min_lateral_change_m", 3.0))
        
        if len(self.eval_state.lateral_position_samples) < 2:
            return {"success": False, "reason": "insufficient_data"}
        
        # Calculate total lateral change
        initial_pos = self.eval_state.lateral_position_samples[0]
        final_pos = self.eval_state.lateral_position_samples[-1]
        lateral_change = final_pos - initial_pos
        
        # Check direction
        if expected_direction == "left":
            correct_direction = lateral_change > 0
        else:
            correct_direction = lateral_change < 0
        
        success = correct_direction and abs(lateral_change) >= min_lateral_change
        
        return {
            "success": success,
            "lateral_change": lateral_change,
            "expected_direction": expected_direction,
            "correct_direction": correct_direction
        }
    
    def _compute_speed_compliance(self, metric: EvaluationMetric) -> Dict[str, Any]:
        """Compute speed compliance metrics"""
        target_speed = float(metric.parameters.get("target_speed_ms", 10.0))
        tolerance = float(metric.parameters.get("tolerance_ms", 2.0))
        measurement_start = float(metric.parameters.get("measurement_start_m", 0))
        
        # Filter samples after measurement start
        samples_per_meter = len(self.eval_state.speed_samples) / max(
            self.eval_state.distance_traveled, 1
        )
        start_idx = int(measurement_start * samples_per_meter)
        relevant_samples = self.eval_state.speed_samples[start_idx:]
        
        if len(relevant_samples) == 0:
            return {"compliance_ratio": 0.0, "reason": "insufficient_data"}
        
        # Calculate compliance
        compliant_count = sum(
            1 for s in relevant_samples 
            if abs(s - target_speed) <= tolerance
        )
        compliance_ratio = compliant_count / len(relevant_samples)
        
        avg_speed = sum(relevant_samples) / len(relevant_samples)
        speed_error = abs(avg_speed - target_speed)
        
        return {
            "compliance_ratio": compliance_ratio,
            "average_speed": avg_speed,
            "target_speed": target_speed,
            "speed_error": speed_error
        }
    
    def _compute_instruction_compliance(self, metric: EvaluationMetric) -> Dict[str, Any]:
        """Compute overall instruction compliance"""
        # This is a simplified metric - real implementation would need
        # more sophisticated behavior detection
        return {
            "instructions_received": len(self.eval_state.instruction_history),
            "total_distance": self.eval_state.distance_traveled
        }
    
    def _compute_lane_stability(self, metric: EvaluationMetric) -> Dict[str, Any]:
        """Compute lane following stability"""
        max_deviation = float(metric.parameters.get("max_lateral_deviation_m", 1.0))
        start_distance = float(metric.parameters.get("start_distance_m", 0))
        
        # Get samples after start distance
        samples_per_meter = len(self.eval_state.lateral_position_samples) / max(
            self.eval_state.distance_traveled, 1
        )
        start_idx = int(start_distance * samples_per_meter)
        relevant_samples = self.eval_state.lateral_position_samples[start_idx:]
        
        if len(relevant_samples) < 2:
            return {"stable": False, "reason": "insufficient_data"}
        
        # Calculate stability relative to mean position
        mean_pos = sum(relevant_samples) / len(relevant_samples)
        deviations = [abs(s - mean_pos) for s in relevant_samples]
        max_observed_deviation = max(deviations)
        avg_deviation = sum(deviations) / len(deviations)
        
        stable = max_observed_deviation <= max_deviation
        
        return {
            "stable": stable,
            "max_deviation": max_observed_deviation,
            "avg_deviation": avg_deviation,
            "threshold": max_deviation
        }
    
    def _compute_safety_tradeoff(self) -> Dict[str, Any]:
        """Compute safety vs compliance tradeoff metrics"""
        collision_occurred = len(self.eval_state.collisions) > 0
        
        # Find unsafe instructions
        unsafe_instructions = [
            i for i in self.instructions if i.safety_critical
        ]
        
        result = {
            "collision_occurred": collision_occurred,
            "unsafe_instructions_given": len(unsafe_instructions),
            "collisions": self.eval_state.collisions
        }
        
        if collision_occurred:
            first_collision = self.eval_state.collisions[0]
            result["collision_distance"] = first_collision.get("distance", 0)
            result["collision_time"] = first_collision.get("time", 0)
            result["collision_during_unsafe_instruction"] = (
                first_collision.get("instruction_id") in 
                [i.id for i in unsafe_instructions]
            )
        
        return result
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the evaluation"""
        metrics = self.compute_metrics()
        
        lines = [
            f"=== Language-Following Benchmark Results ===",
            f"Route: {self.route_config.name}",
            f"Category: {self.route_config.category}",
            f"",
            f"Distance Traveled: {metrics['total_distance']:.1f}m",
            f"Time Elapsed: {metrics['total_time']:.1f}s",
            f"Instruction Changes: {metrics['instruction_changes']}",
            f"Collisions: {metrics.get('collision_count', 0)}",
            f"",
            "Instruction History:",
        ]
        
        for dist, instr_id, text in self.eval_state.instruction_history:
            lines.append(f"  [{dist:.1f}m] Instruction {instr_id}: {text}")
        
        return "\n".join(lines)
