# Orion Codebase Notes

This document summarizes the Orion implementation in this repository. It focuses only on Orion itself: model structure, training setup, inference path, route-command handling, and how trajectory selection works in code.

## 1. High-Level Architecture

Orion is a vision-language-action model built on the local `mmcv/mmdet3d` stack.

- Vision backbone: `EVAViT`
- Language backbone: `LlavaLlamaForCausalLM`
- Perception branches:
  - 3D object branch: `pts_bbox_head`
  - map branch: `map_head`
- Planning branch:
  - hidden state at `<waypoint_ego>` is extracted from the LLM
  - that feature is passed into a trajectory decoder
  - the default decoder predicts 6 trajectory modes

Main implementation files:

- `Orion/mmcv/models/detectors/orion.py`
- `Orion/mmcv/utils/llava_llama.py`
- `Orion/mmcv/datasets/pipelines/transforms_3d.py`
- `Orion/team_code/orion_b2d_agent.py`
- `Orion/team_code/planner.py`

## 2. Training Stages

The repository defines three main Orion training stages.

### Stage 1

Config: `Orion/adzoo/orion/configs/orion_stage1_train.py`

- `use_gen_token = False`
- `pretrain = True`
- `LoadAnnoatationVQA` loads external `Chat-B2D` conversations from `./data/chat-B2D/train`

This stage trains the LLM-style QA behavior and perception losses, but does not use the `<waypoint_ego>` planning-token path.

### Stage 2

Config: `Orion/adzoo/orion/configs/orion_stage2_train.py`

- `use_gen_token = True`
- `planning_qa_only = True`
- `planning_qa_last = True`
- `base_desc_path = None`

This stage is planning-focused. The training prompt is reduced to the planning QA containing `<waypoint_ego>`, and the planning branch is trained from the hidden state at that token.

### Stage 3

Config: `Orion/adzoo/orion/configs/orion_stage3_train.py`

- `use_gen_token = True`
- `mix_qa_training = True`
- `planning_qa_ratio = 0.8`
- `base_desc_path = './data/chat-B2D/train'`

This stage mixes planning samples and QA/chat samples. The code shows the mixing logic, but the actual external `Chat-B2D` conversation files are not included in this repository.

## 3. Language Input Construction

Language inputs are built in the dataset pipeline, not inside the agent.

Relevant file:

- `Orion/mmcv/datasets/pipelines/transforms_3d.py`

### Training Prompt Construction

`LoadAnnoatationVQA` builds the training conversations.

When `use_gen_token=True`, the planning QA is:

```text
Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car.
ASSISTANT: Here is the planning trajectory <waypoint_ego>
```

When `use_gen_token=False`, the model is trained on normal QA/chat turns without `<waypoint_ego>`.

### Inference Prompt Construction

`LoadAnnoatationCriticalVQATest` builds inference-time input.

In the default close-loop Orion agent config (`orion_stage3_agent.py`), inference uses:

```text
<image>
You are driving a car. Please provide the planning trajectory for the ego car without reasons.
ASSISTANT: Here is the planning trajectory <waypoint_ego>
```

This is the standard planning-only inference path.

### CoT-Style Inference

There is also a CoT-oriented config: `Orion/adzoo/orion/configs/orion_stage3_cot.py`

In that path:

- `desc_qa=True`
- the pipeline adds several earlier QA rounds
- the planning-token round is appended at the end

So the codebase contains both:

- standard planning-only inference
- multi-round QA followed by planning-token inference

## 4. LLM and Planning Token

The planning token is `<waypoint_ego>`.

In `orion.py`, Orion adds this token and records its token id:

- `add_special_token([EGO_WAYPOINT_TOKEN], ...)`
- `self.lm_head.config.waypoint_token_idx = ...`

In `llava_llama.py`, the model extracts the hidden state at the planning token position:

- `return_ego_feature=True`
- locate `<waypoint_ego>` in tokenized input
- return the hidden state at that position

So the planning branch is conditioned on the LLM hidden state at a special token, not on a separate planner input tensor.

## 5. Planning Decoder and 6-Mode Trajectory Output

In the default Orion path, the planning decoder is a VAE-style trajectory branch in `Orion/mmcv/models/detectors/orion.py`.

Key points:

- `self.ego_fut_mode = 6`
- `ego_fut_decoder` is an MLP
- the decoder predicts all 6 modes at each future step

The output tensor shape is:

- `[B, 6_modes, 6_timesteps, 2]`

The default decoder does not take the navigation command as an explicit input argument. It decodes from the latent / hidden-state path built from the LLM feature.

## 6. How Command Supervision Works

The route command is represented as a 6-dim one-hot vector called `ego_fut_cmd`.

During training:

- all 6 modes are predicted
- the planning regression loss is weighted by `ego_fut_cmd`
- only the active command mode receives planning supervision

So the model learns a fixed association between trajectory slots and command categories through supervision, not because the decoder is explicitly fed the command as an input feature.

During inference:

- all 6 modes are predicted
- the active mode is selected afterward using `ego_fut_cmd`

This is post-hoc mode selection.

## 7. Agent Runtime Path

The main close-loop agent is:

- `Orion/team_code/orion_b2d_agent.py`

At runtime, the agent:

1. reads 6 camera images, GPS, IMU, speed
2. converts GPS to local map position
3. queries the local route planner for the current and next route entries
4. builds the input dict for the model
5. runs the inference pipeline to produce `input_ids`
6. runs the Orion model
7. converts predicted trajectory to control using the PID controller

The agent writes:

- `results['command'] = command2nohot(curr_command)`
- `results['ego_fut_cmd'] = command2hot(curr_command)`

So the route command directly determines which trajectory mode will be selected.

## 8. Route Planner Behavior

Relevant file:

- `Orion/team_code/planner.py`

Orion's route planner is a lightweight queue over the global route provided by leaderboard / Bench2Drive.

Important detail:

- the discrete route command is already precomputed upstream as `RoadOption`
- Orion does not infer `LEFT/RIGHT/STRAIGHT` from raw offset geometry at runtime

### Stored Route Representation

`RoutePlanner.set_route()` stores an ordered deque of:

- `(position, command)` when using GPS plan

The `command` is already attached to each route point.

### Runtime Progression

At each tick, `run_step(gps)`:

1. computes 2D Euclidean distance from ego position to upcoming route points
2. finds the furthest route point still within the local threshold
3. pops earlier route entries
4. returns the current and next route entries

Core logic:

```python
distance = np.linalg.norm(self.route[i][0] - gps)
if distance <= self.min_distance and distance > farthest_in_range:
    farthest_in_range = distance
    to_pop = i
```

This means the route queue advances as the ego gets close enough to later route points.

After popping:

- `self.route[0]` is the current route entry
- `self.route[1]` is the next route entry

And the agent reads:

```python
(_, curr_command), (near_node, near_command) = self._route_planner.run_step(pos)
```

So Orion's current navigation command is the command attached to the head of the route queue.

## 9. Command Encoding

In `orion_b2d_agent.py`, command encoding is:

- `1 -> LEFT`
- `2 -> RIGHT`
- `3 -> STRAIGHT`
- `4 -> LANE FOLLOW`
- `5 -> CHANGE LANE LEFT`
- `6 -> CHANGE LANE RIGHT`

Then:

- `command2hot()` converts to 6-dim one-hot
- `command2nohot()` converts to zero-based integer index for logging / visualization

So the planner-to-model command path is:

```text
RoutePlanner -> curr_command -> command2hot() -> ego_fut_cmd -> select trajectory mode
```

## 10. What the Code Shows About Language Conditioning

Based on this codebase:

- the standard inference prompt does not contain route-command text
- the default planning decoder predicts 6 modes regardless of command
- command supervision/selection is carried by `ego_fut_cmd`
- the planning branch is conditioned on the hidden state at `<waypoint_ego>`
- CoT-style inference can prepend earlier QA turns before the planning-token round

The code therefore supports this interpretation:

- Orion has a real LLM-planning-token path
- but the route command is still handled as a discrete mode-selection signal in the default planning pipeline

## 11. Data Flow Summary

```text
6 Camera Images
    | (image preprocessing + multi-view feature extraction)
    v
EVAViT Vision Backbone
    |
    v
Image Features
    |
    |   Inference Prompt / QA Context
    |        |
    |   Tokenizer + conversation builder
    |        |
    v        v
    Multi-modal token sequence with <image> and <waypoint_ego>
    |
    v
LlavaLlamaForCausalLM
    |
    +---> Hidden state at <waypoint_ego>
    |         |
    |         v
    |     Planning feature
    |         |
    |         v
    |     Trajectory decoder
    |         |
    |         v
    |     6 predicted trajectory modes [6, 6, 2]
    |
    v
RoutePlanner -> curr_command
    |
    v
command2hot(curr_command) -> ego_fut_cmd
    |
    v
Post-hoc mode selection from the 6 predicted trajectories
    |
    v
Selected ego trajectory
    |
    v
PID controller
    |
    v
carla.VehicleControl
```

Notes:

- In the default Orion path, the model predicts all 6 trajectory modes regardless of the current route command.
- The route command affects which mode is selected through `ego_fut_cmd`, not by being injected into the decoder as a direct continuous planning input.
- In CoT-style inference, earlier QA turns can be prepended before the final planning-token round, but the planning trajectory is still extracted from the hidden state at `<waypoint_ego>`.

## 12. Files Worth Reading

If you need to trace the implementation directly, start with:

- `Orion/mmcv/models/detectors/orion.py`
- `Orion/mmcv/utils/llava_llama.py`
- `Orion/mmcv/datasets/pipelines/transforms_3d.py`
- `Orion/team_code/orion_b2d_agent.py`
- `Orion/team_code/planner.py`
- `Orion/team_code/pid_controller.py`
- `Orion/adzoo/orion/configs/orion_stage1_train.py`
- `Orion/adzoo/orion/configs/orion_stage2_train.py`
- `Orion/adzoo/orion/configs/orion_stage3_train.py`
- `Orion/adzoo/orion/configs/orion_stage3_agent.py`
- `Orion/adzoo/orion/configs/orion_stage3_cot.py`
