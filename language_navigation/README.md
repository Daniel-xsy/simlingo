# Language Navigation Toolkit

Tools for generating, visualizing, and evaluating language-navigation benchmark XML files built on top of Bench2Drive route data and the CARLA simulator.

## Scripts

### `utils.py`

Shared utility module imported by all other scripts. Contains:

- **`CarlaMapCache`** / **`SpeedLimitMapCache`** — cached OpenDRIVE map and speed-limit lookups.
- **`INSTRUCTION_LIBRARY`** — paraphrase pools for lane follow, lane change, turn, accelerate, and decelerate instructions.
- **Geometry helpers** — `_distance`, `_route_length_m`, `_position_at_distance`, `_normalize_yaw_delta_deg`, `_compute_turn_category`, etc.
- **Route actionability analysis** — `_scan_turn_actions`, `_build_actionable_navigation_categories`, `_sample_route_actionability`, `select_navigation_trigger`, `detect_route_special_case`.
- **XML building blocks** — `_append_instruction`, `_build_default_evaluation`, `_build_default_scenarios`, `_build_action_route_tree`, `_indent_xml_compat`.

### `generate_language_xml_route.py`

Generates language-benchmark XMLs by **rebuilding routes from scratch** using CARLA waypoint queries. Starting from a Bench2Drive spawn point, it traces a fresh route with configurable step size (default 3 m), identifies the best navigation trigger via actionability scoring, then branches the route for each feasible action (turn left/right/straight, lane change, lane follow). Each action variant is written as a separate XML file.

```bash
python -m language_navigation.generate_language_xml_route \
  leaderboard/data/bench2drive_split/bench2drive_00.xml \
  --output leaderboard/data/language_benchmark/instruction_following_rebuilt/
```

### `generate_safety_critical_xml.py`

Generates safety-critical benchmark XMLs from Bench2Drive routes. Copies the original route geometry verbatim (keeping background vehicles and default traffic lights) and injects dangerous language instructions that conflict with the active scenario — e.g., telling the agent to run a red light or ignore a stop sign.

```bash
python -m language_navigation.generate_safety_critical_xml \
  leaderboard/data/bench2drive_split/ \
  --output leaderboard/data/language_benchmark/safety_critical_v0.1/
```

### `route_xml_bev.py`

Visualizes a generated language-benchmark XML route in bird's-eye view. Renders the CARLA road surface, lane direction arrows, landmarks, the GT route path with waypoint markers, and instruction trigger points. A side panel shows all XML metadata (route attributes, instructions, evaluation metrics, scenarios, weathers).

```bash
python -m language_navigation.route_xml_bev \
  bench2drive_02_language_rebuilt_turn_left \
  --input-dir leaderboard/data/language_benchmark/instruction_following_v0.3_subset
```

### `route_actionable_bev.py`

Debug visualization for raw Bench2Drive route actionability. Loads an original Bench2Drive split XML, runs the full trigger-selection algorithm, and plots the route with the selected and sampled trigger positions, available turn options, lane-change feasibility, and special-case detection (merge/exit). Useful for diagnosing why a particular route gets a specific action assignment.

```bash
python -m language_navigation.route_actionable_bev 22 \
  --input-dir leaderboard/data/bench2drive_split
```

### `eval_results_bev.py`

Visualizes evaluation results in BEV by overlaying the GT route (from XML, red) and the actual agent trajectory (from `metric_info.json`, blue) on the CARLA map. Marks the route-deviation point if present. The side panel shows scores (`composed`, `route`, `penalty`), status, infractions, instruction text, and trajectory frame count. Supports single-route and batch modes.

```bash
# Single route
python -m language_navigation.eval_results_bev \
  eval_results/LanguageBenchmark/.../bench2drive_02_language_rebuilt_turn_right \
  --benchmark-dir leaderboard/data/language_benchmark/instruction_following_v0.3_subset

# Batch mode
python -m language_navigation.eval_results_bev \
  --eval-root eval_results/LanguageBenchmark/instruction_following_v0.3_subset \
  --benchmark-dir leaderboard/data/language_benchmark/instruction_following_v0.3_subset
```

### `aggregate_safety_results.py`

Aggregates safety-critical benchmark results from evaluation JSONs. Groups by category and reports per-category mean scores, collision/infraction rates, and overall safety override rate.

```bash
python -m language_navigation.aggregate_safety_results \
  eval_results/LanguageBenchmark/safety_critical_v0.1/
```

### `copy_selected_routes.py`

Copies generated language XMLs for a subset of Bench2Drive route ids listed in a text file (e.g., `route_subset.txt`). Useful for creating smaller benchmark splits from a full generation run.

## Data Files

- **`route_subset.txt`** — list of Bench2Drive route ids used for subset selection.

## Instruction Categories

Instructions are sampled from `INSTRUCTION_LIBRARY` in `utils.py`:

- Lane follow, lane change left/right, turn left/right/straight
- Accelerate (vague: "speed up" / precise: "reach 11 m/s")
- Decelerate (vague: "slow down")
- Exit left/right (for merge/exit special cases)

To extend instruction diversity, edit the text pools in `INSTRUCTION_LIBRARY`.
