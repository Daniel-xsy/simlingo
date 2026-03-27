# Documentation

This directory contains documentation that decouples the original model implementations from our language benchmark customization.

## Structure

| Document | What It Covers |
|----------|---------------|
| [simlingo.md](simlingo.md) | Original SimLingo model: architecture, training, inference agent, prompt construction, trajectory decoding, and PID control. Read this to understand how the base model works before any benchmark modifications. |
| [orion.md](orion.md) | Original Orion model: architecture, 3-stage training, 6-mode trajectory decoder, route-command handling, and post-hoc mode selection. Read this to understand the alternative VLA model we also evaluate. |
| [benchmark_adaptations.md](benchmark_adaptations.md) | How each model is adapted for the instruction-following benchmark. Covers the prompt override mechanism (SimLingo) and command override mechanism (Orion), what changes and what stays the same, and the contrast between the two approaches. |
| [todos.md](todos.md) | Current benchmark design: how we generate instruction-following XMLs, the trigger mechanism, and known issues with the current approach. |

## How to Use These Docs

**If you are new to the codebase**, start with `simlingo.md` or `orion.md` to understand the base model you will be working with. Then read `benchmark_adaptations.md` to see how the model is adapted for benchmarking, and `todos.md` for the benchmark design and known issues.

**If you are debugging evaluation behavior**, `benchmark_adaptations.md` explains exactly what the benchmark agent changes (prompt override for SimLingo, command override for Orion). `todos.md` describes the trigger mechanism, instruction expiry, and known issues.

**If you are modifying XML generation**, `todos.md` covers the generation pipeline design. The `language_navigation/README.md` has script-level details.

**If you are adapting a new model**, read `benchmark_adaptations.md` for the pattern: inherit from the base agent, override the minimal surface area (prompt or command), and reuse the shared trigger/instruction infrastructure.

## Relationship Between Layers

```
Original Model (simlingo.md / orion.md)
    Defines: vision encoder, LLM, trajectory decoder, PID control, route planner
    Entry point: team_code/agent_simlingo.py or Orion/team_code/orion_b2d_agent.py

Benchmark Adaptation (benchmark_adaptations.md)
    SimLingo: overrides prompt with instruction text
    Orion: overrides command_curr with instruction command_id
    Entry points: team_code/agent_simlingo_language_benchmark.py
                  Orion/team_code/orion_language_benchmark_agent.py

Benchmark Design & Generation (todos.md, language_navigation/README.md)
    XML generation: language_navigation/generate_language_xml_route.py
    Trigger mechanism, instruction lifecycle, known issues

Evaluation
    Runs: debug_language_benchmark.sh / dist_eval.sh (SimLingo)
          debug_language_benchmark_orion.sh / dist_eval_orion.sh (Orion)
    Output: eval_results/LanguageBenchmark/
```

The benchmark agents inherit from their respective base agents and override only the minimal surface area needed. Everything else (vision, LLM, trajectory decoding, PID control) runs unchanged.
