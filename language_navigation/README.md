# Language Navigation Toolkit

This directory contains the route-generation, verification, visualization, and analysis utilities for the language-following driving benchmark.

The main goal is to turn Bench2Drive routes into language-conditioned benchmark XMLs that:
- keep the route topologically valid for the official evaluator,
- expose actionable navigation instructions such as turn, lane change, lane follow, and speed control,
- remain easy to inspect offline before running expensive CARLA evaluation.

## Quick Start

Recommended workflow:

1. Generate the latest instruction-following benchmark:
```bash
bash language_navigation/create_data.sh
```
This wrapper is cwd-independent: you can run it from the repo root or from
inside `language_navigation/`. It also defaults `CARLA_ROOT` to
`/home/nvidia/software/carla0915` and falls back to `conda run -n simlingo`
when the current Python cannot import `carla`.

2. Verify the generated XMLs with the same `trace_route()` logic used by evaluation:
```bash
export CARLA_ROOT=/home/nvidia/software/carla0915
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$PYTHONPATH
python language_navigation/verify_planner_routes.py \
  leaderboard/data/language_benchmark/instruction_following_full
```

3. Compare two benchmark directories offline:
```bash
python language_navigation/verify_planner_routes.py \
  leaderboard/data/language_benchmark/instruction_following \
  --compare-dir leaderboard/data/language_benchmark/instruction_following_full \
  --output-json /tmp/instruction_following_compare.json
```

## Main Scripts

### Benchmark generation

- [`generate_language_xml_route.py`](/home/nvidia/vla-project/simlingo/language_navigation/generate_language_xml_route.py)
  Main generator for instruction-following benchmark XMLs. It rebuilds a CARLA waypoint route from the Bench2Drive start, samples a trigger, executes one or more actions, constructs instructions, and writes the final XML.

- [`create_data.sh`](/home/nvidia/vla-project/simlingo/language_navigation/create_data.sh)
  Convenience wrapper for benchmark generation. It resolves repository/CARLA paths relative to the script so it can be run from any working directory. The wrapper still uses an internal `VERSION` variable for generation output naming, but the current official instruction-following datasets are maintained in the unversioned directories `instruction_following`, `instruction_following_full`, and `instruction_following_mini`.

- [`generate_safety_critical_xml.py`](/home/nvidia/vla-project/simlingo/language_navigation/generate_safety_critical_xml.py)
  Generator for safety-critical benchmark XMLs. Unlike instruction-following generation, it keeps the original Bench2Drive route and injects dangerous instructions that intentionally conflict with the scenario.

### Verification and planner safety

- [`verify_planner_routes.py`](/home/nvidia/vla-project/simlingo/language_navigation/verify_planner_routes.py)
  Offline verifier that runs `GlobalRoutePlanner.trace_route()` on every adjacent XML waypoint pair. This is the fastest way to catch malformed routes without running full evaluation.

- [`planner_route_tools.py`](/home/nvidia/vla-project/simlingo/language_navigation/planner_route_tools.py)
  Shared planner-based utilities used by both generation and verification. This module builds planner-safe export anchors and detects pathological segment expansion.

### Route building and instruction logic

- [`route_builder.py`](/home/nvidia/vla-project/simlingo/language_navigation/route_builder.py)
  CARLA waypoint reconstruction logic. Handles follow-road traversal, turn execution, lane changes, trigger/action sampling on rebuilt routes, and XML waypoint/scenario element helpers.

- [`instructions.py`](/home/nvidia/vla-project/simlingo/language_navigation/instructions.py)
  Instruction text and XML helpers. Samples phrasing, builds expected-behavior tags, and fits accelerate instructions into available route windows.

- [`actionability.py`](/home/nvidia/vla-project/simlingo/language_navigation/actionability.py)
  Detects which navigation actions are feasible at a route point.

- [`geometry.py`](/home/nvidia/vla-project/simlingo/language_navigation/geometry.py)
  Low-level geometry helpers used by route reconstruction and sampling.

- [`opendrive.py`](/home/nvidia/vla-project/simlingo/language_navigation/opendrive.py)
  OpenDRIVE map loading and speed-limit lookup. This is the source of truth for speed-limit-aware generation.

- [`xml_builder.py`](/home/nvidia/vla-project/simlingo/language_navigation/xml_builder.py)
  Small XML assembly helpers reused across generators.

### Visualization and analysis

- [`route_xml_bev.py`](/home/nvidia/vla-project/simlingo/language_navigation/route_xml_bev.py)
  Visualizes raw XML routes in bird’s-eye view with route, instructions, triggers, scenarios, and map context.

- [`eval_results_bev.py`](/home/nvidia/vla-project/simlingo/language_navigation/eval_results_bev.py)
  Visualizes evaluation outputs by overlaying the XML GT route and the recorded trajectory, with infractions and instruction compliance in a side panel.

- [`benchmark_statistics.py`](/home/nvidia/vla-project/simlingo/language_navigation/benchmark_statistics.py)
  Computes dataset-level statistics from generated benchmark XMLs.

### Dataset management

- [`copy_selected_routes.py`](/home/nvidia/vla-project/simlingo/language_navigation/copy_selected_routes.py)
  Copies a subset of generated XMLs from a route list file into a new directory.

- [`route.txt`](/home/nvidia/vla-project/simlingo/language_navigation/route.txt)
  The curated list used for the `selected` benchmark split.

- [`route_subset.txt`](/home/nvidia/vla-project/simlingo/language_navigation/route_subset.txt)
  The smaller curated list used for the `subset` split.

### Helper modules

- [`utils.py`](/home/nvidia/vla-project/simlingo/language_navigation/utils.py)
  Shared helper functions that remain available for visualization and compatibility paths. The current planner-safe generation logic is centered in `route_builder.py`, `opendrive.py`, and `planner_route_tools.py`.

## XML Generation Logic

High-level pseudocode for [`generate_language_xml_route.py`](/home/nvidia/vla-project/simlingo/language_navigation/generate_language_xml_route.py):

```text
for each Bench2Drive source route:
    load CARLA map for the town
    project the source start position onto a driving waypoint

    rebuild a follow route from the start waypoint up to max_distance_m
    sample speed instruction parameters from OpenDRIVE speed limits

    compute actionability samples along the rebuilt route
    select a trigger after the acceleration window
    choose output actions at that trigger

    for each chosen action:
        rebuild the action suffix from the trigger waypoint
        if the action is invalid after rebuild:
            skip it

        optionally chain further instructions from the action end
        merge prefix + action segment + chained tail into one waypoint chain

        if there is no further chained instruction and the tail ends inside a junction:
            truncate the terminal tail before the junction entrance

        build planner-safe exported XML positions from the CARLA waypoint chain
        validate exported pairs with the same trace_route() logic used by evaluation

        build instruction XML
        build scenario/evaluation/weather XML
        write the final benchmark XML
```

## Why The Export Logic Is Designed This Way

### 1. Do not export dense interior-junction waypoints

The official evaluator does not directly use the raw XML polyline. It re-interpolates every adjacent XML pair with:

- `GlobalRoutePlanner.trace_route(start, end)`

That means XML positions are not just geometry; they are planner anchors.

Dense points inside a junction are dangerous because nearby connector lanes can be geometrically close but topologically different. Even if the original CARLA waypoint chain is valid, the evaluator only sees coordinates, not the original lane identity. A short `2-3 m` XML pair can therefore expand into a long detour when `trace_route()` snaps its endpoints onto different junction branches.

So the generator now prefers sparse, planner-safe anchors instead of exporting all raw `3 m` waypoint samples through the junction interior.

### 2. Validate with the same planner path as evaluation

The generator now uses planner-based validation before export. This mirrors evaluation intentionally:

- generation safety check: `trace_route()` offline,
- evaluation densification: `trace_route()` at runtime.

This removes the previous mismatch where a route looked correct in BEV but failed during actual evaluation.

### 3. Truncate terminal junction tails

Some routes naturally end while still inside a junction. In that case there may be no clean post-junction anchor to export. Forcing the route to continue to the full nominal length can create bad terminal pairs that the planner cannot reconstruct safely.

The design choice is:

- if there is no further chained instruction,
- and the remaining tail would end inside a junction,
- truncate that terminal tail before entering the junction.

This is intentional. It is better to end slightly early with a planner-safe route than to export a longer route that becomes malformed under the evaluator.

### 4. Use OpenDRIVE as the speed-limit source of truth

Benchmark speed instructions should reflect the road definition actually used by CARLA. For this reason the generator uses OpenDRIVE speed data and does not rely on precomputed local lookup tables.

## Verification Philosophy

Use [`verify_planner_routes.py`](/home/nvidia/vla-project/simlingo/language_navigation/verify_planner_routes.py) before running evaluation whenever:

- a new benchmark version is generated,
- route export logic changes,
- a suspicious `route_dev` appears in evaluation,
- a route looks correct in BEV but fails in the official evaluator.

The verifier reports:

- raw XML route length,
- planner-interpolated route length,
- failing segment count,
- worst expanded segment,
- optional version-to-version comparison.

If the verifier says a route is pathological, evaluation is likely to fail even if the raw XML BEV looks reasonable.

## Recommended Reuse Guidance For Later Agents

- Start with this README, then inspect only the relevant script.
- For route-generation bugs, begin with:
  - `generate_language_xml_route.py`
  - `route_builder.py`
  - `planner_route_tools.py`
- For route-validity questions, run `verify_planner_routes.py` before touching evaluation.
- For visualization mismatches, remember:
  - `route_xml_bev.py` shows the raw XML route,
  - evaluation uses the planner-densified route,
  - `eval_results_bev.py` overlays XML GT plus trajectory, not the hidden internal evaluator route.
