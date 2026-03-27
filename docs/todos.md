# Language Benchmark Design and TODOs

This document covers two things: (1) how the current benchmark XML generation and trigger mechanism work, and (2) known issues and planned improvements.

For script-level details on the generation toolkit, see `language_navigation/README.md`.

---

## Current Design

### How Benchmark XMLs Are Generated

The generation pipeline lives in `language_navigation/generate_language_xml_route.py`. It takes Bench2Drive route XMLs as input and produces language-conditioned benchmark XMLs.

**High-level flow:**

```
For each Bench2Drive source route:
    1. Load the CARLA map for the route's town
    2. Project the source start position onto a CARLA driving waypoint
    3. Rebuild a follow route from the start waypoint (up to max_distance_m)
    4. Sample a speed instruction window using OpenDRIVE speed limits
    5. Compute actionability samples along the rebuilt route
       (which waypoints have feasible turns, lane changes, etc.)
    6. Select a trigger point after the acceleration window
    7. Choose one or more navigation actions at that trigger
    8. For each chosen action:
       a. Rebuild the route suffix executing that action
       b. Optionally chain further instructions from the action endpoint
       c. Merge prefix + action segment + chained tail into one waypoint chain
       d. Build planner-safe XML waypoint anchors
       e. Validate with trace_route() (same logic the evaluator uses)
       f. Write the final benchmark XML
```

Each output XML contains:

- A rebuilt waypoint route (not the original Bench2Drive route)
- One or more `<instruction>` elements with trigger, text, command_id, expected behavior, and duration
- Disabled background vehicles and green traffic lights (for instruction-following benchmarks)

**Instruction chaining:** The generator can recursively chain multiple instructions. After completing one action (e.g., a left turn), it rebuilds a new follow route from the action endpoint and samples the next trigger. This produces multi-instruction routes (up to 3 instructions by default) with a minimum gap between triggers.

**Route safety:** The generator exports sparse planner-safe anchors rather than dense interior-junction waypoints. Every adjacent XML waypoint pair is validated offline with `trace_route()` to match the evaluator's runtime route densification. See `language_navigation/README.md` for the rationale.

### How the Benchmark Agent Works

The benchmark agent (`team_code/agent_simlingo_language_benchmark.py`) inherits from the base SimLingo agent and adds an instruction overlay.

**Initialization:**

1. Parses the benchmark XML to extract `Instruction` objects (each with trigger, text, command_id, duration)
2. Forces the base agent into command mode (`eval_route_as = 'command'`)
3. Initializes distance tracking and instruction state

**Per-frame execution:**

1. Get the ego's true CARLA location
2. Update cumulative `distance_traveled` (Euclidean per-tick movement)
3. Determine the active instruction via `_get_active_instruction()`
4. Set `self.custom_prompt` to the instruction text (or a fallback follow-road command if no instruction is active)
5. Set `self.user_flag = 3` so the base agent replaces the entire command prompt
6. Call the parent `tick()`, which builds the final prompt as:
   - active instruction: `"Current speed: {speed} m/s. {instruction_text}."`
   - no active instruction: `"Current speed: {speed} m/s. Command: follow the road."`
7. Model forward pass uses this instruction-guided prompt
8. PID control converts predictions to vehicle control (unchanged from base agent)

The benchmark agent does **not** modify vision, LLM, trajectory decoding, or PID control. It only overrides the prompt.

### Trigger Mechanism

Each instruction has a trigger that determines when it activates. The agent evaluates triggers every frame.

| Trigger Type | Condition | Typical Use |
|---|---|---|
| `start` | Always active from frame 0 | Initial speed instruction |
| `distance_traveled` | Cumulative ego distance >= threshold | Sequential instructions |
| `distance_to_point` | Euclidean distance to (x,y) <= threshold | Location-anchored instructions |
| `time_elapsed` | Simulation time since start >= threshold | Time-based sequences |
| `scenario_active` | Scenario actor detected (blackboard or proximity) | Event-driven instructions |

**Distance tracking** uses true CARLA actor locations (not GPS-converted coordinates) for accuracy.

**Instruction lifecycle:**

```
[Not yet triggered]
    | trigger condition met
    v
[Active] -- start_distance recorded
    | distance_traveled > start_distance + duration_meters
    v
[Expired]
```

- `duration_meters > 0`: instruction expires after traveling that many meters from trigger
- `duration_meters = -1`: instruction stays active until the route ends (used for the last instruction)

Only one instruction is active at a time. Earlier instructions expire via their duration, and the next instruction's trigger takes over.

### XML Schema

```xml
<route id="..." town="..." benchmark_type="language_following"
       category="instruction_following"
       disable_bg_vehicle="true"
       force_all_green_traffic_lights="true">
  <waypoints>
    <position x="..." y="..." z="..."/>
    ...
  </waypoints>
  <instructions>
    <instruction id="1" priority="primary">
      <trigger type="distance_traveled" value="0.0"/>
      <text>accelerate to reach 10 m/s</text>
      <command_id>-1</command_id>
      <expected_behavior type="target_speed" speed_ms="10" tolerance_ms="1.5"/>
      <duration_meters>20.5</duration_meters>
    </instruction>
    <instruction id="2" priority="primary">
      <trigger type="distance_to_point" value="5.0" x="123.45" y="-67.89" z="0.30"/>
      <text>turn left at the next intersection</text>
      <command_id>1</command_id>
      <expected_behavior type="turn" direction="left"/>
      <duration_meters>30</duration_meters>
    </instruction>
  </instructions>
  <evaluation>
    <metric type="collision_check">
      <param name="expect_collision" value="false"/>
    </metric>
    <metric type="instruction_compliance">
      <param name="compliance_threshold" value="0.8"/>
    </metric>
  </evaluation>
  <scenarios>
    <scenario name="FreeRide_1" type="FreeRide">
      <trigger_point x="..." y="..." z="..." yaw="..."/>
    </scenario>
  </scenarios>
  <weathers>...</weathers>
</route>
```

Notes:

- In the shared XML schema, speed-only instructions use `command_id = -1`.
- SimLingo follows these speed instructions through the instruction text injected into the prompt.
- Orion treats `command_id = -1` as "do not override the native route-planner command", because Orion has no dedicated accelerate/decelerate navigation mode.

---

## Known Issues

## 1. (Fixed) Shared Issue: `distance_traveled` Triggering Is Not Spatially Robust

### Problem (original)

- `distance_traveled` is not equivalent to "ego reached the intended trigger location on the rebuilt route".
- If the ego deviates from the route, takes a wider turn, oscillates, slows, or overshoots, the trigger can fire at the wrong place.
- This is especially problematic near intersections and lane changes, where a few meters of trigger error can change the maneuver outcome.

### Fix

- Navigation instructions (turns, lane changes) now use `distance_to_point` triggers anchored to the trigger waypoint's world position.
- The XML generator emits `<trigger type="distance_to_point" value="5.0" x="..." y="..." z="..."/>` for navigation instructions, where `value` is the tolerance radius in metres.
- Speed-only instructions (`command_id = -1`) and lane-follow fillers still use `distance_traveled` triggers, since they are not spatially anchored.
- Both benchmark agents already supported `distance_to_point` evaluation. A latching fix was added so that once a `distance_to_point` trigger fires and the instruction becomes active, it stays active until its duration expires (previously, the instruction would deactivate if the ego moved past the tolerance radius).

### Implementation details

- `InstructionSpec` and `InstructionStep` gained an optional `trigger_position` field (`Position3D`).
- `_append_instruction()` accepts `trigger_position` and `trigger_tolerance_m` (default 5.0 m). When `trigger_position` is set, it emits a `distance_to_point` trigger; otherwise it falls back to `distance_traveled`.
- The trigger position is sourced from `RebuiltTrigger.position`, which is the world-space location of the selected trigger waypoint on the rebuilt route.

### Status

- Fixed.

## 2. Orion-Specific Issue: Instruction Expiry Is Too Weak

### Current Implementation

- `OrionLanguageBenchmarkAgent` parses XML instructions and selects an active instruction based on the configured trigger.
- When an instruction is active and `command_id != -1`, the agent overrides `command_curr`.
- The parent Orion agent then converts that command into `ego_fut_cmd`, and Orion uses it for post-hoc trajectory mode selection.
- Standard Orion's internal route planner still advances its own queue normally, but the benchmark agent can overwrite the route planner's current command afterward.

### Problem

- If an instruction has `duration_meters = -1`, it can stay active until another instruction replaces it.
- In the current Orion adaptation, this means the benchmark agent can keep forcing an old `command_curr` even after Orion's route planner has already advanced to a later route command.
- This can keep Orion selecting a stale maneuver mode after the maneuver is already completed.
- More generally, the benchmark instruction layer can outlive the intended maneuver because expiry is based only on trigger/duration logic, not action completion.

### Potential Fix

- Prefer finite `duration_meters` for turn and lane-change instructions.
- Add an instruction completion condition so the override expires when the intended maneuver is likely finished.
- Possible completion signals:
  - route planner command has moved past the intended maneuver
  - ego has reached a post-action waypoint / point window
  - lane-change completion detected from lane identity or lateral displacement
- Keep `command_id = -1` speed-only instructions as non-overriding commands.

### Status

- Current implementation is active.
- Known benchmark-design risk.
- Not fixed yet.

## 3. Orion-Specific Issue: Benchmark Instruction Layer Is Not Synchronized With Native Navigation

### Current Implementation

- Standard Orion uses a deque of pre-labeled route points and advances commands by local distance to the route.
- The language benchmark adds a separate instruction layer that is driven by XML trigger logic.
- The benchmark layer does not currently synchronize instruction activation/deactivation with the route planner queue.

### Problem

- The benchmark instruction layer can drift apart from the model's native navigation mechanism.
- In the current Orion adaptation, the instruction layer is not synchronized with Orion's native route-command queue, so command override timing can differ from the original Bench2Drive schedule.
- This can create evaluation artifacts that do not reflect how Orion behaves under the original route-command progression.

### Potential Fix

- Make instruction activation and expiry aware of the route planner state.
- For navigation instructions, consider:
  - trigger near the intended route point using `distance_to_point`
  - expire when the planner queue has advanced past the maneuver
- Treat the route planner as the reference timeline for discrete command progression.

### Status

- Identified conceptually.
- No implementation yet.
