# Benchmark Adaptations

This document describes how each VLA model is adapted for the instruction-following benchmark. The adaptations are purely at the agent level — no model weights or training code are modified.

For how the original models work, see `simlingo.md` and `orion.md`. For the benchmark design (XML generation, trigger mechanism, known issues), see `todos.md`.

---

## SimLingo Adaptation

### What Changes

The benchmark agent (`team_code/agent_simlingo_language_benchmark.py`) inherits from `LingoAgent` and overrides only the prompt that the model receives.

| Aspect | Base Agent | Benchmark Agent |
|--------|-----------|-----------------|
| Prompt source | Auto-generated from route planner (target point or command) | Parsed from XML instruction text |
| `eval_route_as` | `'target_point'` (default) | Forced to `'command'` |
| `custom_prompt` | `None` (unused) | Set to instruction text every frame |
| `user_flag` | `None` | Always `3` (plain replacement, no special tags) |
| Special tags | `<INSTRUCTION_FOLLOWING>` / `<SAFETY>` available | Not used |
| Trajectory decoding | Unchanged | Unchanged |
| PID control | Unchanged | Unchanged |

### How Prompt Override Works

The base agent has a built-in override mechanism via `custom_prompt` and `user_flag`:

```python
# Base agent prompt construction (agent_simlingo.py, lines 539-549)
if self.custom_prompt is not None:
    if self.user_flag == 2 or self.user_flag == 3:
        # Replace entire prompt_tp with custom_prompt
        prompt = f"Current speed: {speed} m/s. {self.custom_prompt}"
    else:
        # Append custom_prompt after auto-generated prompt_tp
        prompt = f"Current speed: {speed} m/s. {prompt_tp} {self.custom_prompt}"

if self.user_flag == 1 or self.user_flag == 2:
    prompt = f"<INSTRUCTION_FOLLOWING> {prompt}"
elif self.user_flag == 0:
    prompt = f"<SAFETY> {prompt}"
```

The `user_flag` values control two things: whether `custom_prompt` replaces or supplements the auto-generated prompt, and which special tag (if any) is prepended.

| user_flag | Prompt behavior | Special tag |
|-----------|----------------|-------------|
| `0` | Supplement | `<SAFETY>` |
| `1` | Supplement | `<INSTRUCTION_FOLLOWING>` |
| `2` | Replace | `<INSTRUCTION_FOLLOWING>` |
| `3` | Replace | None |

The benchmark agent always uses `user_flag = 3`: full replacement, no special tag.

### Resulting Prompt

**Base agent (target point mode):**
```
Current speed: 5.2 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.
```

**Base agent (command mode):**
```
Current speed: 5.2 m/s. Command: go left in 32 meters. Predict the waypoints.
```

**Benchmark agent:**
```
Current speed: 5.2 m/s. Turn left at the next intersection.
```

The model sees the instruction text as direct language input. Since SimLingo's trajectory decoder attends to the full language+vision input through the LLM, the instruction text directly influences trajectory prediction.

### Why This Works

SimLingo has a genuine language-conditioned planning path. The trajectory is decoded from query embeddings that attend to the entire input sequence. There is no discrete mode selection — the model produces a single trajectory that is conditioned on whatever text appears in the prompt. This means language instructions can influence the trajectory without any changes to the model or decoding logic.

### What Is Not Used

The model was trained with `<INSTRUCTION_FOLLOWING>` and `<SAFETY>` prompt tags. These tags signal the model to adjust its behavior (e.g., override to a safe trajectory when `<SAFETY>` is present and the instruction is unsafe). The benchmark agent currently does not use these tags (`user_flag = 3`), which means:

- The model treats benchmark instructions the same as normal navigation prompts
- There is no explicit signal telling the model it should follow the instruction strictly
- Safety-critical evaluation could use `user_flag = 0` to prepend `<SAFETY>`, but this is handled separately

---

## Orion Adaptation

### What Changes

The benchmark agent (`Orion/team_code/orion_language_benchmark_agent.py`) inherits from `OrionAgent` and overrides `command_curr` — the discrete navigation command that selects which of Orion's 6 trajectory modes is used.

| Aspect | Base Agent | Benchmark Agent |
|--------|-----------|-----------------|
| `command_curr` source | Route planner queue | Instruction's `command_id` (when active) |
| `ego_fut_cmd` | From route planner command | From overridden `command_curr` |
| Language prompt | Not used for planning | Instruction text parsed but not fed to model |
| Trajectory modes | 6 modes, selected by route command | 6 modes, selected by instruction command |
| Trajectory decoding | Unchanged | Unchanged |
| PID control | Unchanged | Unchanged |

### How Command Override Works

Orion predicts 6 trajectory modes unconditionally. The active mode is selected post-hoc using `ego_fut_cmd`, a 6-dim one-hot vector derived from `command_curr`. The benchmark agent overrides `command_curr` in `tick()`:

```python
# Benchmark agent tick() override (orion_language_benchmark_agent.py)
if self.current_instruction is not None and self.current_instruction.command_id != -1:
    result['command_curr'] = self.current_instruction.command_id
```

The parent agent then converts this to `ego_fut_cmd`:

```python
# Base agent (orion_b2d_agent.py)
results['ego_fut_cmd'] = command2hot(tick_data['command_curr'])
```

### Command ID Mapping

```
1 -> LEFT          (turn left at intersection)
2 -> RIGHT         (turn right at intersection)
3 -> STRAIGHT      (go straight at intersection)
4 -> LANE FOLLOW   (continue in current lane)
5 -> CHANGE LEFT   (lane change left)
6 -> CHANGE RIGHT  (lane change right)
-1 -> (no override, keep route planner command — used for speed-only instructions)
```

### Why This Works (and Its Limitations)

Orion's planning is not language-conditioned at the model level. The LLM processes text but the trajectory decoder operates on `ego_fut_cmd` mode selection, not on language features. This means:

- **It works** because we can directly control which trajectory mode is selected by setting the command
- **It is limited** because the model never actually "understands" the instruction — it just receives a remapped mode selector
- The instruction text is parsed by the agent but never reaches the model's planning path
- Speed instructions (`command_id = -1`) cannot be enforced through this mechanism since there is no speed-specific mode

### Contrast with SimLingo

The fundamental difference:

- **SimLingo:** Language instructions enter the model through the prompt and influence trajectory via LLM attention. The model can, in principle, understand and follow arbitrary language instructions.
- **Orion:** Language instructions are converted to a discrete command ID outside the model. The model only sees a mode selector. It cannot follow instructions that don't map to one of the 6 predefined trajectory modes.

---

## Shared Infrastructure

Both benchmark agents share the same:

- **XML format:** Same `<instructions>` schema with triggers, text, command_id, duration
- **Trigger mechanism:** Same `TriggerType` enum and evaluation logic (start, distance_traveled, distance_to_point, time_elapsed, scenario_active)
- **Distance tracking:** True CARLA actor locations, not GPS-converted coordinates
- **Instruction lifecycle:** Trigger → active (start_distance recorded) → expired (after duration_meters)
- **Evaluation scripts:** `dist_eval.sh` / `dist_eval_orion.sh` and `debug_language_benchmark.sh` / `debug_language_benchmark_orion.sh`

The XML generation pipeline (`language_navigation/generate_language_xml_route.py`) produces XMLs that work with both models. The `command_id` field serves double duty: SimLingo ignores it (uses instruction text), Orion uses it for mode selection.

---

## Files

| File | Purpose |
|------|---------|
| `team_code/agent_simlingo_language_benchmark.py` | SimLingo benchmark agent |
| `Orion/team_code/orion_language_benchmark_agent.py` | Orion benchmark agent |
| `debug_language_benchmark.sh` | SimLingo single-route debug |
| `debug_language_benchmark_orion.sh` | Orion single-route debug |
| `dist_eval.sh` | SimLingo distributed evaluation |
| `dist_eval_orion.sh` | Orion distributed evaluation |
