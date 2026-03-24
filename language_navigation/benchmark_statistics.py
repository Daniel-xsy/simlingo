#!/usr/bin/env python3
"""
Compute and visualize statistics for generated language benchmark XML files.

Statistics:
1. Distribution of action categories (turn_left, turn_right, lane_change_left, etc.)
2. Distribution of speed instructions (target_speed values, accelerate, decelerate)
3. Distribution of duration_meters per instruction
4. Waypoint distribution in ego-centric frame (heatmap)

Usage:
    python language_navigation/benchmark_statistics.py <xml_dir_or_file> [--output-dir <dir>]
    python language_navigation/benchmark_statistics.py leaderboard/data/language_benchmark/instruction_following_v0.12_selected/
"""

import argparse
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# XML Parsing
# ---------------------------------------------------------------------------

def parse_xml_files(input_path: Path) -> list:
    """Parse all XML routes from a file or directory. Returns list of route dicts."""
    xml_files = []
    if input_path.is_file() and input_path.suffix == ".xml":
        xml_files = [input_path]
    elif input_path.is_dir():
        xml_files = sorted(input_path.glob("*.xml"))
    else:
        raise ValueError(f"Invalid input: {input_path}")

    routes = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for route_elem in root.findall("route"):
            route = parse_route(route_elem, xml_file)
            routes.append(route)
    return routes


def parse_route(route_elem, xml_file: Path) -> dict:
    """Parse a single <route> element into a dict."""
    # Waypoints
    waypoints = []
    for pos in route_elem.findall("waypoints/position"):
        waypoints.append((
            float(pos.get("x")),
            float(pos.get("y")),
            float(pos.get("z", "0")),
        ))

    # Instructions
    instructions = []
    for inst in route_elem.findall("instructions/instruction"):
        eb = inst.find("expected_behavior")
        eb_type = eb.get("type") if eb is not None else None
        eb_attrs = dict(eb.attrib) if eb is not None else {}

        trigger = inst.find("trigger")
        trigger_type = trigger.get("type") if trigger is not None else None
        trigger_value = trigger.get("value") if trigger is not None else None

        dur_elem = inst.find("duration_meters")
        duration = float(dur_elem.text) if dur_elem is not None else -1

        cmd_elem = inst.find("command_id")
        command_id = int(cmd_elem.text) if cmd_elem is not None else None

        instructions.append({
            "id": inst.get("id"),
            "text": inst.findtext("text", ""),
            "command_id": command_id,
            "expected_behavior_type": eb_type,
            "expected_behavior_attrs": eb_attrs,
            "trigger_type": trigger_type,
            "trigger_value": float(trigger_value) if trigger_value else None,
            "duration_meters": duration,
        })

    return {
        "route_id": route_elem.get("id"),
        "town": route_elem.get("town"),
        "category": route_elem.get("category"),
        "xml_file": str(xml_file),
        "waypoints": waypoints,
        "instructions": instructions,
    }


# ---------------------------------------------------------------------------
# 1. Action Category Distribution
# ---------------------------------------------------------------------------

def get_action_category(inst: dict) -> str:
    """Map an instruction to its high-level action category."""
    eb_type = inst["expected_behavior_type"]
    attrs = inst["expected_behavior_attrs"]

    if eb_type == "turn":
        direction = attrs.get("direction", "unknown")
        return f"turn_{direction}"
    elif eb_type == "lane_change":
        direction = attrs.get("direction", "unknown")
        return f"lane_change_{direction}"
    elif eb_type == "lane_follow":
        return "lane_follow"
    elif eb_type == "target_speed":
        return "target_speed"
    elif eb_type == "accelerate":
        return "accelerate"
    elif eb_type == "decelerate":
        return "decelerate"
    elif eb_type == "dangerous":
        return "dangerous"
    else:
        return eb_type or "unknown"


def plot_action_distribution(routes: list, output_dir: Path):
    """Plot distribution of action categories."""
    counter = Counter()
    for route in routes:
        for inst in route["instructions"]:
            cat = get_action_category(inst)
            counter[cat] += 1

    # Sort by count descending
    categories, counts = zip(*counter.most_common()) if counter else ([], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(categories)), counts, color="#4C72B0", edgecolor="white")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Action Category Distribution", fontsize=14)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(count), ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)
    fig.tight_layout()
    out_path = output_dir / "action_category_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 2. Speed Instruction Distribution
# ---------------------------------------------------------------------------

def plot_speed_distribution(routes: list, output_dir: Path):
    """Plot distribution of target speed values (m/s)."""
    speeds = []
    for route in routes:
        for inst in route["instructions"]:
            if inst["expected_behavior_type"] == "target_speed":
                speed = inst["expected_behavior_attrs"].get("speed_ms")
                if speed is not None:
                    speeds.append(float(speed))

    if not speeds:
        print("  No target_speed instructions found. Skipping speed distribution.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use integer bins since speeds are typically integers
    speed_counter = Counter(speeds)
    speed_vals = sorted(speed_counter.keys())
    speed_counts = [speed_counter[v] for v in speed_vals]

    bars = ax.bar(
        [str(int(v)) if v == int(v) else str(v) for v in speed_vals],
        speed_counts, color="#DD8452", edgecolor="white",
    )
    ax.set_xlabel("Target Speed (m/s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Speed Instruction Distribution", fontsize=14)

    for bar, count in zip(bars, speed_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(count), ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, max(speed_counts) * 1.15)
    fig.tight_layout()
    out_path = output_dir / "speed_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 3. Duration Meters Distribution
# ---------------------------------------------------------------------------

def plot_duration_distribution(routes: list, output_dir: Path):
    """Plot distribution of duration_meters values."""
    durations = []
    for route in routes:
        for inst in route["instructions"]:
            d = inst["duration_meters"]
            durations.append(d)

    if not durations:
        print("  No instructions found. Skipping duration distribution.")
        return

    # Separate open-ended (-1) from finite durations
    finite = [d for d in durations if d > 0]
    open_ended_count = sum(1 for d in durations if d == -1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    # Left: histogram of finite durations
    ax = axes[0]
    if finite:
        bins = np.arange(0, max(finite) + 5, 5)
        ax.hist(finite, bins=bins, color="#55A868", edgecolor="white", alpha=0.9)
        ax.set_xlabel("Duration (meters)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Duration Meters Distribution (finite)", fontsize=14)
    else:
        ax.text(0.5, 0.5, "No finite durations", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Duration Meters Distribution (finite)", fontsize=14)

    # Right: pie chart for finite vs open-ended
    ax2 = axes[1]
    labels = ["Finite", "Open-ended (-1)"]
    sizes = [len(finite), open_ended_count]
    colors = ["#55A868", "#C44E52"]
    ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 11})
    ax2.set_title("Finite vs Open-ended", fontsize=14)

    fig.tight_layout()
    out_path = output_dir / "duration_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 4. Waypoint Distribution (ego-centric heatmap)
# ---------------------------------------------------------------------------

def compute_ego_relative_waypoints(routes: list, window_size: int = 20) -> np.ndarray:
    """
    For each route, at every position t in [0, N-window_size), compute where
    waypoints t+1 through t+window_size are relative to t in ego-centric coords.

    Returns Nx2 array of (lateral, longitudinal) offsets.
    Convention: forward = +longitudinal, left = +lateral.
    """
    all_relative = []

    for route in routes:
        wps = route["waypoints"]
        n = len(wps)
        if n < window_size + 1:
            continue

        for t in range(n - window_size):
            x0, y0, _ = wps[t]
            x1, y1, _ = wps[t + 1]

            # Heading from t to t+1
            dx_fwd = x1 - x0
            dy_fwd = y1 - y0
            heading = math.atan2(dy_fwd, dx_fwd)

            cos_h = math.cos(-heading)
            sin_h = math.sin(-heading)

            for j in range(1, window_size + 1):
                wx, wy, _ = wps[t + j]
                dx = wx - x0
                dy = wy - y0
                # Rotate so forward aligns with +X in rotated frame
                rx = dx * cos_h - dy * sin_h  # forward
                ry = dx * sin_h + dy * cos_h  # leftward

                # (lateral, longitudinal)
                all_relative.append((ry, rx))

    return np.array(all_relative) if all_relative else np.empty((0, 2))


def plot_waypoint_distribution(routes: list, output_dir: Path, window_size: int = 20):
    """
    Plot ego-centric waypoint distribution as a 2D heatmap.

    Ego is at top center, forward direction points down,
    left direction points right, right direction points left.
    """
    pts = compute_ego_relative_waypoints(routes, window_size=window_size)

    if len(pts) == 0:
        print("  No waypoints to plot. Skipping waypoint distribution.")
        return

    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    lateral = pts[:, 0]   # left = positive (right in plot)
    longitudinal = pts[:, 1]  # forward = positive (down in plot)

    fig, ax = plt.subplots(figsize=(8, 10))

    # Estimate waypoint spacing from data to pick bin size
    wp_spacings = []
    for route in routes:
        wps = route["waypoints"]
        for i in range(min(5, len(wps) - 1)):
            dx = wps[i + 1][0] - wps[i][0]
            dy = wps[i + 1][1] - wps[i][1]
            wp_spacings.append(math.sqrt(dx * dx + dy * dy))
    avg_spacing = np.median(wp_spacings) if wp_spacings else 2.0

    # Bin ~1.5x waypoint spacing to avoid aliasing from discrete steps
    bin_size = max(1.0, round(avg_spacing * 1.2))

    lat_range = max(abs(lateral.min()), abs(lateral.max()), 25)
    lon_max = max(longitudinal.max(), 50)
    lon_min = min(longitudinal.min(), 0)

    bins_x = np.arange(-lat_range, lat_range + bin_size, bin_size)
    bins_y = np.arange(lon_min, lon_max + bin_size, bin_size)

    h, xedges, yedges = np.histogram2d(lateral, longitudinal, bins=[bins_x, bins_y])

    # Mask zeros so background stays white
    h_masked = np.ma.masked_where(h < 0.5, h)

    img = ax.pcolormesh(
        xedges, yedges, h_masked.T,
        cmap="Reds",
        norm=LogNorm(vmin=1),
        shading="flat",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label("Count", fontsize=12)

    # Draw ego vehicle marker at origin (top center in this orientation)
    ax.plot(0, 0, marker="v", color="red", markersize=14, zorder=10)
    ax.annotate("EGO", (0, 0), textcoords="offset points", xytext=(12, -15),
                fontsize=10, fontweight="bold", color="red")

    ax.set_xlabel("Lateral (left +, right -)", fontsize=12)
    ax.set_ylabel("Longitudinal (forward +)", fontsize=12)
    ax.set_title(f"Waypoint Distribution (ego-centric, window={window_size})", fontsize=14)

    # Clamp view to the data extent (with small margin)
    max_lon = window_size * avg_spacing * 1.2
    max_lat = max_lon * 0.6
    ax.set_xlim(-max_lat, max_lat)
    ax.set_ylim(0, max_lon)

    # Invert axes: x so left=right in plot; y so ego is at top, forward points down
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "waypoint_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

def print_summary(routes: list):
    """Print a text summary of the dataset."""
    n_routes = len(routes)
    n_instructions = sum(len(r["instructions"]) for r in routes)
    n_waypoints = sum(len(r["waypoints"]) for r in routes)

    action_counter = Counter()
    for route in routes:
        for inst in route["instructions"]:
            action_counter[get_action_category(inst)] += 1

    towns = Counter(r["town"] for r in routes)

    print(f"\n{'='*60}")
    print(f"  Benchmark Statistics Summary")
    print(f"{'='*60}")
    print(f"  Routes:       {n_routes}")
    print(f"  Instructions: {n_instructions}")
    print(f"  Waypoints:    {n_waypoints}")
    print(f"\n  Action categories:")
    for cat, count in action_counter.most_common():
        print(f"    {cat:25s} {count:5d}  ({100*count/n_instructions:.1f}%)")
    print(f"\n  Towns:")
    for town, count in towns.most_common():
        print(f"    {town:25s} {count:5d}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute statistics for language benchmark XML files."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to an XML file or directory of XML files.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to <input_dir>/statistics/.",
    )
    parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=20,
        help="Waypoint window size for ego-centric distribution (default: 20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (input_path if input_path.is_dir() else input_path.parent) / "statistics"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing XML files from: {input_path}")
    routes = parse_xml_files(input_path)
    print(f"Found {len(routes)} routes.")

    if not routes:
        print("No routes found. Exiting.")
        return

    print_summary(routes)

    print("Generating plots...")
    plot_action_distribution(routes, output_dir)
    plot_speed_distribution(routes, output_dir)
    plot_duration_distribution(routes, output_dir)
    plot_waypoint_distribution(routes, output_dir, window_size=args.window_size)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
