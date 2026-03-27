# SimLingo Codebase Notes

This document summarizes the original SimLingo implementation in this repository. It focuses only on SimLingo itself: model structure, training setup, inference path, prompt construction, and how trajectory prediction works in code. It does not cover the language benchmark customization layer.

## 1. High-Level Architecture

SimLingo is a vision-language-action model built on InternVL2.

- Vision backbone: InternVL2 (1B variant by default, configurable to 2B/4B)
- Language model: InternVL2's language model (Qwen-based) with optional LoRA fine-tuning
- Planning branches:
  - route head: query-based MLP that predicts 20 future waypoints
  - speed head: query-based MLP that predicts 10 future speed waypoints
- Language generation: greedy autoregressive sampling from the LLM

Main implementation files:

- `simlingo_training/models/driving.py`
- `simlingo_training/models/encoder/internvl2_model.py`
- `simlingo_training/models/language_model/llm.py`
- `simlingo_training/models/adaptors/adaptors.py`
- `team_code/agent_simlingo.py`
- `team_code/nav_planner.py`
- `team_code/privileged_route_planner.py`

## 2. Vision Backbone

SimLingo uses InternVL2 as the vision encoder.

Configuration (`simlingo_training/config.py`):

- Variant: `OpenGVLab/InternVL2-1B` (default)
- Embedding dim: 512
- Optional weight freezing

Image processing (`team_code/agent_simlingo.py`):

- Input: the current default inference config uses 1 front RGB camera (`num_cameras = [0]`) at 1024x512, with the lower part of the image cropped before preprocessing
- The agent code is written to iterate over `num_cameras`, so multi-camera input is structurally possible, but the checked-in SimLingo config enables only camera 0 by default
- Dynamic preprocessing: `dynamic_preprocess()` creates patch-based representation
- Each frame produces up to 2 patches of 448x448
- Output shape: `[B, T, N_patches, 3, 448, 448]`

Image features are extracted through `self.model.extract_feature()` and merged into the text embedding sequence by replacing `<IMG_CONTEXT>` placeholder tokens.

## 3. Language Input Construction

Language inputs are built inside the agent at inference time, not in the dataset pipeline.

Relevant file: `team_code/agent_simlingo.py`

### Special Tokens

```python
['<WAYPOINTS>', '<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>',
 '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']
```

### Prompt Modes

SimLingo supports two navigation input modes controlled by `eval_route_as`:

**Target Point mode:**

```text
Current speed: {speed} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.
```

The `<TARGET_POINT>` tokens are replaced with learned embeddings of the actual waypoint coordinates via `WaypointInputAdaptor`.

**Command mode:**

```text
Current speed: {speed} m/s. Command: {command} in {dist} meters{next_command}. Predict the waypoints.
```

Command mapping: `{1: 'left', 2: 'right', 3: 'straight', 4: 'follow', 5: 'left lane change', 6: 'right lane change'}`.

### Optional Modifiers

- Chain-of-thought: prepends "What should the ego do next?"
- Safety flag: wraps with `<SAFETY> {prompt}`
- Instruction following flag: wraps with `<INSTRUCTION_FOLLOWING> {prompt}`

### Tokenization

The prompt is wrapped with InternVL2's conversation template, including image context tokens `<img><IMG_CONTEXT>*n</img>`. The tokenizer produces `LanguageLabel` containing token IDs, validity masks, loss masks, and placeholder coordinate values keyed by token ID.

## 4. Language Model

Architecture (`simlingo_training/models/language_model/llm.py`):

- Base: InternVL2's extracted language model (Qwen-based)
- Hidden size: 2048 (for InternVL2-1B)
- LoRA: rank=32, alpha=64, dropout=0.1

Forward pass:

```python
def forward(embeddings, attention_mask, position_ids=None):
    outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask,
                        output_hidden_states=True, position_ids=position_ids)
    features = outputs.hidden_states[-1]  # Last hidden state
    logits = outputs[0]
    return features, logits
```

Language generation uses greedy sampling (up to 100 new tokens) during inference. In the current inference path, the model first generates assistant text from the prompt and image-conditioned embeddings, and then uses the resulting extended embedding sequence as context for a second pass that predicts route and speed waypoints. When the prompt asks "What should the ego do next?", this generated text functions as CoT-style reasoning before planning.

## 5. Planning Decoder and Trajectory Output

The planning heads are defined in `simlingo_training/models/adaptors/adaptors.py` as part of `DrivingAdaptor`.

### Route Head

- Predicts **20 future waypoints** in ego-centric frame
- Query embeddings: learnable `nn.Parameter` of shape `[1, 20, hidden_size]`
- MLP: `hidden_size -> mlp_dim*2 -> ReLU -> mlp_dim -> ReLU -> 2`
- Output: cumulative sum of predicted offsets -> `[B, 20, 2]` (x, y)

### Speed Head

- Predicts **10 future speed waypoints**
- Query embeddings: learnable `nn.Parameter` of shape `[1, 10, hidden_size]`
- MLP: `hidden_size -> mlp_dim -> ReLU -> dim`
- Default mode: 2D (x, y trajectory for speed) -> `[B, 10, 2]`
- Output: cumulative sum of predicted offsets

### How Queries Work

In the current inference path, the driving query embeddings (20 route + 10 speed = 30 total) are appended after greedy language generation, not at the initial prompt-encoding stage:

1. Build image-conditioned prompt embeddings from the question/prompt.
2. Run greedy sampling to generate assistant text.
3. Take the returned `input_embeds`, which now contain the original prompt plus the generated language tokens.
4. Concatenate the driving query embeddings to this extended sequence.
5. Run the LLM again.
6. Extract the final 30 query-token features and decode them with the route/speed heads.

SimLingo does not use a special planning token or discrete mode-selection mechanism. Instead, the trajectory is directly decoded from query embeddings that attend to the full input sequence through the LLM.

## 6. Waypoint Encoder

`WaypointInputAdaptor` (`simlingo_training/models/adaptors/adaptors.py`):

- Input: waypoint coordinates `[B, N, 2]`
- MLP: `2 -> 256 -> ReLU -> 512 -> ReLU -> token_size`
- Used to embed target points into the prompt by replacing `<TARGET_POINT>` placeholder tokens

## 7. Training

Configuration: `simlingo_training/config/experiment/simlingo_seed1.yaml`

### Data

- Source: CARLA simulated driving data from `database/simlingo`
- Prediction length: 11 future timesteps (including current)
- History: 1 frame
- Augmentations: image shift (50%), commentary augmentation, QA augmentation
- Data composition: base driving + commentary + QA pairs + navigation instructions
- Template augmentation: 60% augmented, 40% default

### Training Setup

- Framework: PyTorch Lightning
- Strategy: DeepSpeed Stage 2 or DDP
- Learning rate: 3e-5
- Optimizer: AdamW (betas=0.9, 0.999)
- Precision: fp16 mixed
- Batch size: 6
- Max epochs: 15
- Validation: every 2 epochs
- Gradient clipping: 0.3

SimLingo trains end-to-end on a mixed dataset of driving, commentary, and QA samples.

## 8. Agent Runtime Path

The main close-loop agent is `team_code/agent_simlingo.py` (`LingoAgent`).

At runtime, the agent:

1. Captures the configured RGB camera set; in the current default config this is a single front camera
2. Applies dynamic preprocessing to produce 448x448 patches
3. Reads GPS + IMU, optionally filters with Unscented Kalman Filter
4. Queries the route planner for current position, target points, and navigation command
5. Constructs the language prompt with speed, command/target point, and optional modifiers
6. Tokenizes with special tokens and placeholder handling
7. Runs model inference in two stages:
   - greedy-generate assistant text from the prompt/image-conditioned embeddings
   - append driving queries and run a second forward pass to get `pred_route` and `pred_speed_wps`
8. Converts predictions to vehicle control via PID controllers

### Control Generation (`control_pid()`)

**Steering:**
- Interpolates the 20 predicted waypoints to 0.1m spacing
- Uses `LateralPIDController` with speed-adaptive lookahead
- Lookahead: `clip(0.9755 * speed_kmh + 1.9152, 24, 105) / 10`
- PID gains: kp=3.118, ki=0.6406, kd=1.378

**Throttle:**
- Extracts desired speed from the speed waypoint predictions
- Uses speed PID controller: kp=1.75, ki=1.0, kd=2.0
- Error = desired_speed - current_speed
- Brake if speed > 1.1x target speed

Output: `carla.VehicleControl(steer, throttle, brake)`

## 9. Route Planner

SimLingo uses `PrivilegedRoutePlanner` (`team_code/privileged_route_planner.py`), which wraps CARLA's global route with preprocessing.

It provides:

- Route interpolation
- Stop sign and traffic light detection
- Lane change detection
- Leading/trailing vehicle detection
- Speed limit awareness

At each step, the planner is called with the current GPS position and returns route points, waypoints, commands, traffic info, and speed limits. The agent extracts target points from this for waypoint prediction.

## 10. What the Code Shows About Language Conditioning

Based on this codebase:

- SimLingo's prompt directly contains the navigation directive (target point coordinates or command text)
- The model first generates assistant text, then decodes trajectory from query embeddings appended to the prompt-plus-generated-language context
- There is no discrete mode selection or `ego_fut_cmd` mechanism
- The model produces a single trajectory, not multiple modes
- Language generation happens before trajectory prediction in the current inference path and can provide reasoning/context for the planning pass

The code therefore supports this interpretation:

- SimLingo has a genuine language-conditioned planning path
- Navigation information enters through the prompt and influences trajectory via LLM attention
- The planning path is continuous and prompt-conditioned rather than selected from a discrete mode bank

## 11. Data Flow Summary

```
Camera Images [Nx1024x512, default N=1]
    | (crop + dynamic preprocess)
    v
Image Patches [up to 6 patches, 448x448]
    |
    v
InternVL2 Vision Encoder
    |
    v
Image Embeddings [N_patches, 512]
    |
    |   Target Points [2, 2]      Language Prompt
    |        |                         |
    |   WaypointInputAdaptor      Tokenizer
    |        |                         |
    |   Waypoint Embeddings       Token Embeddings
    |        |                         |
    v        v                         v
    Replace <IMG_CONTEXT>    Replace <TARGET_POINT>
    |                              |
    v                              v
    Prompt/Image Embedding Sequence
    |
    v
InternVL2 Language Model
    |
    +---> Greedy Sampling ---> Assistant Text / CoT-style Reasoning
    |
    v
Extended Embedding Sequence [prompt + generated language]
    |
    +---> Append Driving Queries [30]
    |
    v
InternVL2 Language Model (second pass)
    |
    v
Output Features [seq_len + 30, 2048]
    |
    +---> Extract driving query features [30, 2048]
    |         |
    |         +---> Route Head MLP ---> [1, 20, 2] route waypoints
    |         |
    |         +---> Speed Head MLP ---> [1, 10, 2] speed waypoints
    |
    v
Route Waypoints          Speed Waypoints
    |                         |
    | (interpolate)           | (extract target speed)
    v                         v
LateralPIDController     Speed PID
    |                         |
    v                         v
Steering                 Throttle/Brake
    |                         |
    v                         v
    carla.VehicleControl
```

## 12. Files Worth Reading

If you need to trace the implementation directly, start with:

- `simlingo_training/models/driving.py` — main model class
- `simlingo_training/models/encoder/internvl2_model.py` — vision encoder and token merging
- `simlingo_training/models/language_model/llm.py` — LLM forward and greedy sampling
- `simlingo_training/models/adaptors/adaptors.py` — driving adaptors (route/speed heads, waypoint encoder)
- `simlingo_training/config.py` — model configuration dataclasses
- `simlingo_training/config/experiment/simlingo_seed1.yaml` — training config
- `simlingo_training/train.py` — training loop
- `simlingo_training/dataloader/dataset_driving.py` — dataset
- `team_code/agent_simlingo.py` — inference agent
- `team_code/nav_planner.py` — lateral PID controller and throttle control
- `team_code/privileged_route_planner.py` — route planner
- `team_code/config_simlingo.py` — agent config
