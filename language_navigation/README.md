# Language Navigation XML Generator

This folder contains tools for generating language-navigation benchmark XML files from original Bench2Drive route XML files.

Current script:

- `generate_language_xml_distance.py`: converts one Bench2Drive XML file to language-benchmark format using **distance-based** instruction triggers.

## What The Script Does

Given an input route XML from `leaderboard/data/bench2drive_split`:

- keeps original route geometry and environment:
  - `<waypoints>`
  - `<scenarios>`
  - `<weathers>`
- writes language-benchmark route attributes:
  - `benchmark_type="language_following"`
  - `category=<your value>`
  - `disable_bg_vehicle="true"` (default)
- creates `<instructions>` with trigger type:
  - `distance_traveled`
- ignores original `<scenarios>` trigger definitions from Bench2Drive XML
- writes a minimal `<scenarios>` block required by evaluator:
  - `<scenario name="FreeRide_1" type="FreeRide">`
  - `trigger_point` set to first waypoint (start of route)
- adds a default `<evaluation>` block:
  - `collision_check`
  - `instruction_compliance`

## Instruction Sampling

Instructions are sampled from category-specific paraphrase pools in `INSTRUCTION_LIBRARY`.

Implemented categories include:

- lane follow
- lane change left/right
- turn left/right/straight
- accelerate (vague + precise)
- decelerate (vague + precise)

### Speed Instruction Styles

Use `--instruction-style` to control speed-command wording:

- `vague`: "speed up", "slow down", etc.
- `precise`: "accelerate to 20 m/s", "decelerate to 6 m/s", etc.
- `all`: mixture of vague and precise

## Multi-Instruction Composition

Use `--num-instructions` and `--trigger-step-m` to compose multiple instructions in one route.

Example with:

- `--num-instructions 3`
- `--trigger-step-m 40`

Result:

- instruction 1 starts at `0m`, lasts `40m`
- instruction 2 starts at `40m`, lasts `40m`
- instruction 3 starts at `80m`, lasts to end (`duration_meters = -1`)

## Usage

From repo root:

```bash
python language_navigation/generate_language_xml_distance.py \
  leaderboard/data/bench2drive_split/bench2drive_00.xml
```

Output default:

- `leaderboard/data/language_benchmark/instruction_following/bench2drive_00_language_distance.xml`

Generate for a whole folder:

```bash
python language_navigation/generate_language_xml_distance.py \
  leaderboard/data/bench2drive_split
```

This iterates all `*.xml` files in that folder and writes outputs to:

- `leaderboard/data/language_benchmark/instruction_following/`

Specify output path:

```bash
python language_navigation/generate_language_xml_distance.py \
  leaderboard/data/bench2drive_split/bench2drive_00.xml \
  --output leaderboard/data/language_benchmark/tmp/bench2drive_00_distance.xml
```

Generate 4-step composed instructions with mixed speed style:

```bash
python language_navigation/generate_language_xml_distance.py \
  leaderboard/data/bench2drive_split/bench2drive_00.xml \
  --num-instructions 4 \
  --trigger-step-m 40 \
  --instruction-style all \
  --seed 7
```

## Arguments

- `input_xml` (required): input Bench2Drive XML path
- `input_xml` (required): input Bench2Drive XML file or folder
- `--output`: output XML path (default: `leaderboard/data/language_benchmark/instruction_following/<input_stem>_language_distance.xml`)
- `--category`: route `category` field in output XML
- `--trigger-step-m`: distance interval between instruction triggers
- `--seed`: random seed for reproducibility (default: unset/non-deterministic)
- `--num-instructions`: number of instruction segments per route
- `--instruction-style`: `all`, `vague`, or `precise`

## Customization Notes

To extend instruction diversity, edit:

- `INSTRUCTION_LIBRARY` text pools
- `ACCELERATE_TARGET_SPEEDS`
- `DECELERATE_TARGET_SPEEDS`

The script is intentionally simple and designed for fast iteration of benchmark data templates.
