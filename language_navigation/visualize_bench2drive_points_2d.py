#!/usr/bin/env python3
"""Visualize Bench2Drive route points in 2D (ignore z-axis)."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Bench2Drive XML points using x/y only and ignore z."
        )
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("leaderboard/data/bench2drive_split/bench2drive_00.xml"),
        help="Path to Bench2Drive XML route file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output SVG path. Default: "
            "./debug/<xml_stem>_xy.svg"
        ),
    )
    return parser.parse_args()


def _extract_xy_points(xml_path: Path) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    root = ET.parse(xml_path).getroot()
    route = root.find("route")
    if route is None:
        raise ValueError(f"No <route> found in {xml_path}")

    waypoint_points: List[Tuple[float, float]] = []
    for pos in route.findall(".//waypoints/position"):
        x = float(pos.attrib["x"])
        y = float(pos.attrib["y"])
        waypoint_points.append((x, y))

    trigger_points: List[Tuple[float, float]] = []
    for trig in route.findall(".//scenarios/scenario/trigger_point"):
        x = float(trig.attrib["x"])
        y = float(trig.attrib["y"])
        trigger_points.append((x, y))

    if not waypoint_points:
        raise ValueError(f"No waypoint points found in {xml_path}")
    return waypoint_points, trigger_points


def _to_canvas(
    x: float,
    y: float,
    min_x: float,
    max_y: float,
    scale: float,
    pad: float,
) -> Tuple[float, float]:
    return (pad + (x - min_x) * scale, pad + (max_y - y) * scale)


def _build_svg(
    waypoint_points: List[Tuple[float, float]],
    trigger_points: List[Tuple[float, float]],
    xml_name: str,
) -> str:
    xs = [p[0] for p in waypoint_points]
    ys = [p[1] for p in waypoint_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)

    canvas_w = 1000.0
    canvas_h = 1000.0
    pad = 60.0
    scale = min((canvas_w - 2 * pad) / span_x, (canvas_h - 2 * pad) / span_y)

    route_xy = [
        _to_canvas(x, y, min_x=min_x, max_y=max_y, scale=scale, pad=pad)
        for x, y in waypoint_points
    ]
    trig_xy = [
        _to_canvas(x, y, min_x=min_x, max_y=max_y, scale=scale, pad=pad)
        for x, y in trigger_points
    ]

    polyline_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in route_xy)
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.3" fill="#2563eb" />'
        for x, y in route_xy
    )
    trig_marks = "\n".join(
        (
            f'<line x1="{x-6:.2f}" y1="{y-6:.2f}" x2="{x+6:.2f}" y2="{y+6:.2f}" '
            'stroke="#dc2626" stroke-width="2" />\n'
            f'<line x1="{x-6:.2f}" y1="{y+6:.2f}" x2="{x+6:.2f}" y2="{y-6:.2f}" '
            'stroke="#dc2626" stroke-width="2" />'
        )
        for x, y in trig_xy
    )

    start_x, start_y = route_xy[0]
    end_x, end_y = route_xy[-1]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{int(canvas_w)}" height="{int(canvas_h)}" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">
  <rect x="0" y="0" width="{int(canvas_w)}" height="{int(canvas_h)}" fill="white" />
  <text x="20" y="28" font-family="Arial, sans-serif" font-size="18" fill="#111827">2D Route (z ignored): {xml_name}</text>
  <polyline points="{polyline_points}" fill="none" stroke="#0f172a" stroke-width="1.5" />
  {circles}
  <rect x="{start_x-5:.2f}" y="{start_y-5:.2f}" width="10" height="10" fill="#16a34a" />
  <polygon points="{end_x:.2f},{end_y-7:.2f} {end_x-6:.2f},{end_y+6:.2f} {end_x+6:.2f},{end_y+6:.2f}" fill="#ca8a04" />
  {trig_marks}
  <text x="20" y="{int(canvas_h-20)}" font-family="Arial, sans-serif" font-size="14" fill="#374151">legend: blue=waypoints, green square=start, yellow triangle=end, red x=trigger_point</text>
</svg>
"""


def _write_svg(svg: str, output: Path) -> None:
    if output.suffix.lower() != ".svg":
        raise ValueError("Output must be an .svg path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(f"Saved plot to: {output}")


def main() -> None:
    args = parse_args()
    xml_path = args.xml.resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file does not exist: {xml_path}")

    waypoint_points, trigger_points = _extract_xy_points(xml_path)
    svg = _build_svg(
        waypoint_points=waypoint_points,
        trigger_points=trigger_points,
        xml_name=xml_path.name,
    )
    if args.output is None:
        output_path = (Path("debug") / f"{xml_path.stem}_xy.svg").resolve()
    else:
        output_path = args.output.resolve()
    _write_svg(svg, output_path)


if __name__ == "__main__":
    main()
