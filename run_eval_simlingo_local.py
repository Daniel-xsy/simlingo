#!/usr/bin/env python3
"""
Local Bench2Drive evaluation launcher for SimLingo.

This script mirrors the logic of start_eval_simlingo.py but executes the jobs
sequentially on a single machine without SLURM. Update the CONFIGS list at the
bottom with your paths before running:

    python run_eval_simlingo_local.py

Prerequisites:
  * A CARLA server (0.9.15) must be running and listening on the configured
    ports (carla_port / carla_tm_port).
  * The Conda/venv that contains SimLingo dependencies should be activated.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import ujson
from tqdm.autonotebook import tqdm

FAIL_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}


def expand_path(path: str) -> str:
    """Expand ~ and convert to an absolute path."""
    return os.path.abspath(os.path.expanduser(path))


def preprocess_config(cfg: Dict) -> Dict:
    """Resolve paths and fill defaults."""
    cfg = cfg.copy()
    path_fields = [
        "checkpoint",
        "route_path",
        "out_root",
        "carla_root",
        "repo_root",
        "agent_file",
    ]
    for key in path_fields:
        if key in cfg:
            cfg[key] = expand_path(cfg[key])

    cfg.setdefault("carla_port", 2000)
    cfg.setdefault("carla_tm_port", 2500)
    cfg.setdefault("timeout", 600)
    cfg.setdefault("tries", 1)

    required_paths = {
        "route_path": Path(cfg["route_path"]),
        "carla_root": Path(cfg["carla_root"]),
        "repo_root": Path(cfg["repo_root"]),
        "agent_file": Path(cfg["agent_file"]),
        "checkpoint": Path(cfg["checkpoint"]),
    }
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    return cfg


def discover_routes(route_dir: Path) -> List[Path]:
    """Return sorted list of .xml route files."""
    return sorted([p for p in route_dir.iterdir() if p.suffix == ".xml"])


def build_job_queue(configs: List[Dict]) -> List[Dict]:
    """Create job descriptions for all configs / seeds / routes."""
    job_queue: List[Dict] = []
    for cfg_idx, raw_cfg in enumerate(configs):
        cfg = preprocess_config(raw_cfg)
        route_dir = Path(cfg["route_path"])
        routes = discover_routes(route_dir)

        if cfg["benchmark"].lower() == "bench2drive":
            fill_zeros = 3
        else:
            fill_zeros = 2

        seeds = cfg.get("seeds", [1])
        for seed in seeds:
            seed_int = int(seed)
            seed_str = str(seed_int)

            base_dir = Path(cfg["out_root"]) / cfg["agent"] / cfg["benchmark"] / seed_str
            for sub in ("run", "res", "out", "err"):
                (base_dir / sub).mkdir(parents=True, exist_ok=True)

            for route in routes:
                route_suffix = route.stem.split("_")[-1]
                route_id = route_suffix.zfill(fill_zeros)

                viz_path = base_dir / "viz" / route_id
                viz_path.parent.mkdir(parents=True, exist_ok=True)

                job = {
                    "cfg": cfg,
                    "route": str(route),
                    "route_id": route_id,
                    "seed": seed_int,
                    "viz_path": viz_path,
                    "result_file": base_dir / "res" / f"{route_id}_res.json",
                    "log_file": base_dir / "out" / f"{route_id}_out.log",
                    "err_file": base_dir / "err" / f"{route_id}_err.log",
                    "remaining_tries": cfg["tries"],
                }
                job_queue.append(job)

    return job_queue


def needs_rerun(job: Dict) -> bool:
    """Determine if a job still needs to run based on its result JSON."""
    result_file: Path = job["result_file"]
    if not result_file.exists():
        return True

    try:
        with result_file.open("r", encoding="utf-8") as f:
            evaluation_data = ujson.load(f)
    except Exception:
        return True

    try:
        checkpoint = evaluation_data["_checkpoint"]
        progress = checkpoint["progress"]
    except KeyError:
        return True

    need_to_resubmit = False
    if len(progress) < 2 or progress[0] < progress[1]:
        need_to_resubmit = True
    else:
        for record in checkpoint.get("records", []):
            if record.get("status") in FAIL_STATUSES:
                need_to_resubmit = True
                break

    return need_to_resubmit


def prepare_environment(cfg: Dict, viz_path: Path) -> Dict[str, str]:
    """Create env vars for the CARLA evaluation run."""
    env = os.environ.copy()
    carla_root = cfg["carla_root"]
    repo_root = cfg["repo_root"]
    leaderboard_root = os.path.join(repo_root, "Bench2Drive", "leaderboard")
    scenario_runner_root = os.path.join(repo_root, "Bench2Drive", "scenario_runner")

    env["CARLA_ROOT"] = carla_root
    env["SCENARIO_RUNNER_ROOT"] = scenario_runner_root
    env["LEADERBOARD_ROOT"] = leaderboard_root
    env["SAVE_PATH"] = str(viz_path)

    extra_py_paths = [
        os.path.join(carla_root, "PythonAPI", "carla"),
        os.path.join(
            carla_root,
            "PythonAPI",
            "carla",
            "dist",
            "carla-0.9.15-py3.7-linux-x86_64.egg",
        ),
        repo_root,
        leaderboard_root,
        scenario_runner_root,
    ]
    existing_py = env.get("PYTHONPATH", "")
    combined_paths = extra_py_paths + ([existing_py] if existing_py else [])
    env["PYTHONPATH"] = os.pathsep.join(combined_paths)

    return env


def launch_job(job: Dict) -> bool:
    """Run a single evaluation job locally."""
    cfg = job["cfg"]
    viz_path: Path = job["viz_path"]
    if viz_path.exists():
        shutil.rmtree(viz_path)
    viz_path.mkdir(parents=True, exist_ok=True)

    env = prepare_environment(cfg, viz_path)
    leaderboard_entry = os.path.join(
        cfg["repo_root"],
        "Bench2Drive",
        "leaderboard",
        "leaderboard",
        "leaderboard_evaluator.py",
    )

    command = [
        "python",
        "-u",
        leaderboard_entry,
        f"--routes={job['route']}",
        "--repetitions=1",
        "--track=SENSORS",
        f"--checkpoint={job['result_file']}",
        f"--timeout={cfg['timeout']}",
        f"--agent={cfg['agent_file']}",
        f"--agent-config={cfg['checkpoint']}",
        f"--traffic-manager-seed={job['seed']}",
        f"--port={cfg['carla_port']}",
        f"--traffic-manager-port={cfg['carla_tm_port']}",
    ]

    log_path: Path = job["log_file"]
    err_path: Path = job["err_file"]
    with log_path.open("w", encoding="utf-8") as log_file, err_path.open(
        "w", encoding="utf-8"
    ) as err_file:
        log_file.write(f"COMMAND: {' '.join(command)}\n")
        log_file.flush()
        result = subprocess.run(
            command,
            stdout=log_file,
            stderr=err_file,
            env=env,
            cwd=cfg["repo_root"],
        )

    return result.returncode == 0


def process_job(job: Dict) -> bool:
    """Run a job until it succeeds or exhausts retries."""
    total_tries = job["remaining_tries"]
    while job["remaining_tries"] > 0:
        if not needs_rerun(job):
            print(
                f"[SKIP] Route {job['route_id']} seed {job['seed']} already completed."
            )
            return True

        attempt_idx = total_tries - job["remaining_tries"] + 1
        print(f"[RUN ] Route {job['route_id']} seed {job['seed']} attempt {attempt_idx}")
        success = launch_job(job)
        job["remaining_tries"] -= 1

        if success and not needs_rerun(job):
            print(
                f"[DONE] Route {job['route_id']} seed {job['seed']} finished successfully."
            )
            return True

        if job["remaining_tries"] > 0:
            print(
                f"[RETRY] Route {job['route_id']} seed {job['seed']} "
                f"retrying ({job['remaining_tries']} tries left)."
            )

    print(
        f"[FAIL] Route {job['route_id']} seed {job['seed']} exhausted all retries."
    )
    return False


def main():
    job_queue = build_job_queue(CONFIGS)
    if not job_queue:
        print("No jobs discovered. Check your CONFIGS.")
        return

    failures = []
    progress = tqdm(total=len(job_queue), desc="Bench2Drive routes")
    for job in job_queue:
        try:
            succeeded = process_job(job)
            if not succeeded:
                failures.append((job["route_id"], job["seed"]))
        finally:
            progress.update(1)

    progress.close()
    if failures:
        print("The following jobs failed:")
        for route_id, seed in failures:
            print(f"  - Route {route_id}, seed {seed}")
    else:
        print("All jobs completed successfully.")


CONFIGS = [
    {
        "agent": "simlingo",
        "checkpoint": "ckpts/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
        "benchmark": "bench2drive",
        "route_path": "leaderboard/data/bench2drive_split",
        "seeds": [1],
        "tries": 1,
        "out_root": "./eval_results/Bench2Drive",
        "carla_root": "~/software/carla0915",
        "repo_root": "/home/shaoyux/models/simlingo",
        "agent_file": "./team_code/agent_simlingo.py",
        "carla_port": 2000,  # Adjust to match your running CARLA server.
        "carla_tm_port": 2500,
        "timeout": 600,
    }
    # Add more config entries here if needed.
]


if __name__ == "__main__":
    # import debugpy  # type: ignore
    # debugpy.listen(("0.0.0.0", 5678))
    # print(f"[DEBUG] Waiting for debugger on 0.0.0.0:5678", flush=True)
    # debugpy.wait_for_client()
    # print("[DEBUG] Debugger attached; resuming execution.", flush=True)
    
    main()

